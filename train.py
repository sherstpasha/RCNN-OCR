import json
import os
import logging
import csv

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import (
    OCRDatasetAttn,
    MultiDataset,
    ProportionalBatchSampler,
    decode_tokens,
    get_train_transform,
    get_val_transform,
    load_charset,
)
from metrics import character_error_rate, compute_accuracy, word_error_rate
from model import RCNN
from utils import load_checkpoint, save_checkpoint, save_weights, set_seed


# -------------------------
# logging
# -------------------------
def setup_logger(exp_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # формат
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # консоль
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # файл
    os.makedirs(exp_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(exp_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class Config:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for k, v in data.items():
            setattr(self, k, v)

        if not hasattr(self, "exp_dir") or self.exp_dir is None:
            exp_idx = 1
            while os.path.exists(f"exp{exp_idx}"):
                exp_idx += 1
            self.exp_dir = f"exp{exp_idx}"

    def save(self, out_path: str | None = None):
        if out_path is None:
            out_path = os.path.join(self.exp_dir, "config.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def __getitem__(self, key):
        return getattr(self, key)


def split_train_val(
    csvs,
    roots,
    stoi,
    img_h,
    img_w,
    train_transform,
    val_transform,
    encoding="utf-8",
    val_size=3000,
):
    train_sets, val_sets = [], []
    for c, r in zip(csvs, roots):
        full_ds = OCRDatasetAttn(
            c,
            r,
            stoi,
            img_height=img_h,
            img_max_width=img_w,
            transform=None,
            encoding=encoding,
        )
        n_val = min(val_size, len(full_ds))
        n_train = len(full_ds) - n_val
        if n_train <= 0:
            raise ValueError(
                f"В датасете {c} всего {len(full_ds)} примеров, меньше чем {val_size}"
            )

        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

        train_ds.dataset.transform = train_transform
        val_ds.dataset.transform = val_transform

        train_sets.append(train_ds)
        val_sets.append(val_ds)
    return train_sets, val_sets


def run_training(cfg: Config, device: str = "cuda"):
    seed = getattr(cfg, "seed", 42)
    set_seed(seed)

    # --- базовые настройки и пути ---
    exp_dir = getattr(cfg, "exp_dir", None)
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logger(exp_dir)

    logger.info("Start training")
    logger.info(f"Experiment dir: {exp_dir}")
    logger.info(f"Seed: {seed}")

    try:
        cfg.save()
        logger.info("Saved config to exp_dir/config.json")
    except Exception as e:
        logger.info(f"Config save skipped: {e}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # пути/данные
    train_csvs = cfg.train_csvs
    train_roots = cfg.train_roots
    val_csvs = getattr(cfg, "val_csvs", None)
    val_roots = getattr(cfg, "val_roots", None)
    charset_path = cfg.charset_path
    encoding = getattr(cfg, "encoding", "utf-8")

    # модель/данные
    img_h = getattr(cfg, "img_h", 64)
    img_w = getattr(cfg, "img_w", 256)
    max_len = getattr(cfg, "max_len", 25)
    hidden_size = getattr(cfg, "hidden_size", 256)

    # оптимизация
    batch_size = getattr(cfg, "batch_size", 32)
    epochs = getattr(cfg, "epochs", 20)
    lr = getattr(cfg, "lr", 1e-3)
    optimizer_name = getattr(cfg, "optimizer", "Adam")
    scheduler_name = getattr(cfg, "scheduler", "ReduceLROnPlateau")
    weight_decay = getattr(cfg, "weight_decay", 0.0)
    momentum = getattr(cfg, "momentum", 0.9)

    # прочее
    resume_path = getattr(cfg, "resume_path", None)
    save_every = getattr(cfg, "save_every", 1)
    train_proportions = getattr(cfg, "train_proportions", None)
    val_size = getattr(cfg, "val_size", 3000)
    num_workers = getattr(cfg, "num_workers", 0)

    # --- директории и TensorBoard ---
    if resume_path:
        exp_dir = os.path.dirname(resume_path)
        os.makedirs(exp_dir, exist_ok=True)
        logger = setup_logger(exp_dir)

    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    metrics_csv_path = os.path.join(exp_dir, "metrics_epoch.csv")
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_acc",
                    "val_cer",
                    "val_wer",
                    "lr",
                ]
            )

    best_loss_path = os.path.join(exp_dir, "best_loss_ckpt.pth")
    best_acc_path = os.path.join(exp_dir, "best_acc_ckpt.pth")
    last_path = os.path.join(exp_dir, "last_ckpt.pth")
    best_loss_weights_path = os.path.join(exp_dir, "best_loss_weights.pth")
    best_acc_weights_path = os.path.join(exp_dir, "best_acc_weights.pth")
    last_weights_path = os.path.join(exp_dir, "last_weights.pth")

    # --- charset ---
    itos, stoi = load_charset(charset_path)
    PAD = stoi["<PAD>"]
    SOS = stoi["<SOS>"]
    EOS = stoi["<EOS>"]
    BLANK = stoi.get("<BLANK>", None)
    num_classes = len(itos)
    logger.info(f"Charset loaded: {num_classes} tokens")

    # --- модель ---
    model = RCNN(
        num_classes=num_classes,
        hidden_size=hidden_size,
        sos_id=SOS,
        eos_id=EOS,
        pad_id=PAD,
        blank_id=BLANK,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # --- optimizer ---
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # --- scheduler ---
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=False, min_lr=1e-7
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name in ("None", None):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    scaler = amp.GradScaler()

    # --- трансформации ---
    train_transform = get_train_transform(cfg.__dict__, img_h=img_h, img_w=img_w)
    val_transform = get_val_transform(img_h, img_w)

    # --- датасеты и лоадеры ---
    if val_csvs and val_roots:
        train_sets = [
            OCRDatasetAttn(
                c,
                r,
                stoi,
                img_height=img_h,
                img_max_width=img_w,
                transform=train_transform,
                encoding=encoding,
                max_len=max_len,
                strict_max_len=True
            )
            for c, r in zip(train_csvs, train_roots)
        ]
        val_sets = [
            OCRDatasetAttn(
                c,
                r,
                stoi,
                img_height=img_h,
                img_max_width=img_w,
                transform=val_transform,
                encoding=encoding,
                max_len=max_len,
                strict_max_len=True
            )
            for c, r in zip(val_csvs, val_roots)
        ]
    else:
        train_sets, val_sets = split_train_val(
            train_csvs,
            train_roots,
            stoi,
            img_h,
            img_w,
            train_transform,
            val_transform,
            encoding=encoding,
            val_size=val_size,
        )

    collate_train = OCRDatasetAttn.make_collate_attn(
        stoi, max_len=max_len, drop_blank=True
    )
    collate_val = OCRDatasetAttn.make_collate_attn(
        stoi, max_len=max_len, drop_blank=True
    )

    if train_proportions is not None:
        total = sum(train_proportions)
        proportions = [p / total for p in train_proportions]
        assert len(proportions) == len(
            train_sets
        ), "train_proportions != num train_sets"
        train_dataset = MultiDataset(train_sets)
        batch_sampler = ProportionalBatchSampler(train_sets, batch_size, proportions)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_train,
        )
    else:
        train_loader = DataLoader(
            ConcatDataset(train_sets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_train,
        )

    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_val,
    )

    # --- stats about dataset sizes ---
    def _total_len(ds_list):
        total = 0
        for ds in ds_list:
            try:
                total += len(ds)   # работает и для Subset, и для обычных датасетов
            except Exception:
                pass
        return total

    n_train_samples = _total_len(train_sets)
    n_val_samples   = _total_len(val_sets)

    msg_ds = (
        f"Datasets: train={n_train_samples} samples across {len(train_sets)} set(s); "
        f"val={n_val_samples} samples across {len(val_sets)} set(s)"
    )
    msg_ld = (
        f"Loaders: train_batches/epoch={len(train_loader)}; "
        f"val_batches={len(val_loader)}; batch_size={batch_size}"
    )

    print(msg_ds);  logger.info(msg_ds)
    print(msg_ld);  logger.info(msg_ld)

    # --- resume ---
    start_epoch = 1
    global_step = 0
    best_val_loss, best_val_acc = float("inf"), -1.0

    if resume_path and os.path.isfile(resume_path):
        ckpt = load_checkpoint(
            resume_path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
        logger.info(
            f"Resumed from: {resume_path} (epoch={start_epoch-1}, step={global_step})"
        )

    # --- training loop ---
    for epoch in range(start_epoch, epochs + 1):
        # train
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for imgs, text_in, target_y, lengths in pbar:
            imgs = imgs.to(device)
            text_in = text_in.to(device)
            target_y = target_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits = model(
                    imgs, text=text_in, is_train=True, batch_max_length=max_len
                )  # [B,T,V]
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_y.reshape(-1)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.item())
            total_train_loss += loss_val
            writer.add_scalar("Loss/train_step", loss_val, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # validate
        model.eval()
        total_val_loss = 0.0
        refs, hyps = [], []
        pbar_val = tqdm(val_loader, desc=f"Valid {epoch}/{epochs}", leave=False)
        with torch.no_grad():
            for imgs, text_in, target_y, lengths in pbar_val:
                imgs = imgs.to(device)
                text_in = text_in.to(device)
                target_y = target_y.to(device)

                with amp.autocast():
                    logits_tf = model(
                        imgs, text=text_in, is_train=True, batch_max_length=max_len
                    )
                    val_loss = criterion(
                        logits_tf.reshape(-1, logits_tf.size(-1)), target_y.reshape(-1)
                    )
                total_val_loss += float(val_loss.item())

                logits = model(
                    imgs, is_train=False, batch_max_length=max_len
                )  # [B,T,V]
                pred_ids = logits.argmax(-1).cpu()
                tgt_ids = target_y.cpu()

                for p_row, t_row in zip(pred_ids, tgt_ids):
                    hyp = decode_tokens(
                        p_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK
                    )
                    ref = decode_tokens(
                        t_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK
                    )
                    hyps.append(hyp)
                    refs.append(ref)

                pbar_val.set_postfix(val_loss=f"{float(val_loss.item()):.4f}")

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_acc = compute_accuracy(refs, hyps)
        val_cer = sum(character_error_rate(r, h) for r, h in zip(refs, hyps)) / max(
            1, len(refs)
        )
        val_wer = sum(word_error_rate(r, h) for r, h in zip(refs, hyps)) / max(
            1, len(refs)
        )

        # TensorBoard
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("CER/val", val_cer, epoch)
        writer.add_scalar("WER/val", val_wer, epoch)

        # CSV лог
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    f"{avg_train_loss:.6f}",
                    f"{avg_val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_cer:.6f}",
                    f"{val_wer:.6f}",
                    f"{optimizer.param_groups[0]['lr']:.6e}",
                ]
            )

        # печать/лог
        msg = (
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
            f"acc={val_acc:.4f} | CER={val_cer:.4f} | WER={val_wer:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        print(msg)
        logger.info(msg)

        # save "last"
        if (epoch % save_every) == 0:
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                avg_val_loss,
                val_acc,
                itos,
                stoi,
                {
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "img_h": img_h,
                    "img_w": img_w,
                    "encoding": encoding,
                    "max_len": max_len,
                    "charset_path": charset_path,
                    "train_csvs": train_csvs,
                    "train_roots": train_roots,
                    "val_csvs": val_csvs,
                    "val_roots": val_roots,
                },
                log_dir,
            )
            save_weights(last_weights_path, model)

        # track bests
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                best_loss_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_val_loss,
                val_acc,
                itos,
                stoi,
                {
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "img_h": img_h,
                    "img_w": img_w,
                    "encoding": encoding,
                    "max_len": max_len,
                    "charset_path": charset_path,
                    "train_csvs": train_csvs,
                    "train_roots": train_roots,
                    "val_csvs": val_csvs,
                    "val_roots": val_roots,
                },
                log_dir,
            )
            save_weights(best_loss_weights_path, model)
            logger.info(f"New best val_loss: {best_val_loss:.4f} (epoch {epoch})")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_acc_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_val_loss,
                best_val_acc,
                itos,
                stoi,
                {
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "img_h": img_h,
                    "img_w": img_w,
                    "encoding": encoding,
                    "max_len": max_len,
                    "charset_path": charset_path,
                    "train_csvs": train_csvs,
                    "train_roots": train_roots,
                    "val_csvs": val_csvs,
                    "val_roots": val_roots,
                },
                log_dir,
            )
            save_weights(best_acc_weights_path, model)
            logger.info(f"New best acc: {best_val_acc:.4f} (epoch {epoch})")

        # scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    writer.close()
    logger.info("Training finished.")
    return {"val_acc": best_val_acc, "val_loss": best_val_loss, "exp_dir": exp_dir}
