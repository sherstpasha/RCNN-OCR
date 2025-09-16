import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import (
    OCRDatasetAttn,
    load_charset,
    ResizeAndPadA,
)
from model import RCNN
from metrics import compute_accuracy, word_error_rate, character_error_rate


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_train_transform(params, img_h, img_w):
    def suggest(name, default):
        if isinstance(default, tuple):
            return (
                params.get(name, default[0]) if isinstance(params, dict) else default[0]
            )
        else:
            return params.get(name, default) if isinstance(params, dict) else default

    return A.Compose(
        [
            ResizeAndPadA(img_h=img_h, img_w=img_w),
            A.ShiftScaleRotate(
                shift_limit=suggest("shift_limit", (0.0, 0.05)),
                scale_limit=suggest("scale_limit", (0.0, 0.1)),
                rotate_limit=suggest("rotate_limit", (0, 5)),
                border_mode=0,
                value=(255, 255, 255),
                p=suggest("p_ShiftScaleRotate", (0.0, 0.5)),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=suggest("brightness_limit", (0.1, 0.3)),
                contrast_limit=suggest("contrast_limit", (0.1, 0.3)),
                p=suggest("p_BrightnessContrast", (0.0, 0.5)),
            ),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_h, img_w):
    return A.Compose(
        [
            ResizeAndPadA(img_h=img_h, img_w=img_w),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def save_checkpoint(
    path,
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
    config,
    log_dir,
):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "itos": itos,
        "stoi": stoi,
        "config": config,
        "log_dir": log_dir,
    }
    torch.save(ckpt, path)


def save_weights(path, model):
    torch.save(model.state_dict(), path)


def load_checkpoint(
    path, model, optimizer=None, scheduler=None, scaler=None, map_location="auto"
):
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt


def decode_tokens(ids, itos, pad_id, eos_id, blank_id=None):
    out = []
    for t in ids:
        t = int(t)
        if t == eos_id:
            break
        if t == pad_id or (blank_id is not None and t == blank_id):
            continue
        out.append(itos[t])
    return "".join(out)


def run_training(
    train_csvs,
    train_roots,
    val_csvs,
    val_roots,
    charset_path,
    img_h=64,
    img_w=256,
    batch_size=32,
    epochs=20,
    lr=1e-3,
    optimizer_name="Adam",
    scheduler_name="ReduceLROnPlateau",
    weight_decay=0.0,
    momentum=0.9,
    device="cuda",
    encoding="utf-8",
    max_len=25,
    train_transform=None,
    exp_dir=None,
    resume_path=None,
    save_every=1,
):
    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if resume_path is not None:
        exp_dir = os.path.dirname(resume_path)
        os.makedirs(exp_dir, exist_ok=True)
    else:
        if exp_dir is None:
            exp_idx = 1
            while os.path.exists(f"exp{exp_idx}"):
                exp_idx += 1
            exp_dir = f"exp{exp_idx}"
        os.makedirs(exp_dir, exist_ok=True)

    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

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

    model = RCNN(
        num_classes=num_classes,
        hidden_size=256,
        sos_id=SOS,
        eos_id=EOS,
        pad_id=PAD,
        blank_id=BLANK,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    # optimizer
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

    # scheduler
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=False, min_lr=1e-7
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "None":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    scaler = amp.GradScaler()

    # dataloaders
    val_transform = get_val_transform(img_h, img_w)

    train_sets = [
        OCRDatasetAttn(
            c,
            r,
            stoi,
            img_height=img_h,
            img_max_width=img_w,
            transform=train_transform,
            encoding=encoding,
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
        )
        for c, r in zip(val_csvs, val_roots)
    ]

    collate_train = OCRDatasetAttn.make_collate_attn(
        stoi, max_len=max_len, drop_blank=True
    )
    collate_val = OCRDatasetAttn.make_collate_attn(
        stoi, max_len=max_len, drop_blank=True
    )

    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_val,
    )

    # resume logic
    start_epoch = 1
    global_step = 0
    best_val_loss, best_val_acc = float("inf"), -1.0
    writer = SummaryWriter(log_dir=log_dir)

    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = load_checkpoint(
            resume_path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))

    # training loop
    for epoch in range(start_epoch, epochs + 1):
        # --- training ---
        model.train()
        total_train_loss = 0.0
        for imgs, text_in, target_y, lengths in train_loader:
            imgs = imgs.to(device)
            text_in = text_in.to(device)
            target_y = target_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits = model(
                    imgs, text=text_in, is_train=True, batch_max_length=max_len
                )  # [B, T, V]
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_y.reshape(-1)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += float(loss.item())
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # --- validation ---
        model.eval()
        total_val_loss = 0.0
        refs, hyps = [], []
        with torch.no_grad():
            for imgs, text_in, target_y, lengths in val_loader:
                imgs = imgs.to(device)
                text_in = text_in.to(device)
                target_y = target_y.to(device)

                with amp.autocast():
                    # лосс на teacher forcing
                    logits_tf = model(
                        imgs, text=text_in, is_train=True, batch_max_length=max_len
                    )  # [B,T,V]
                    val_loss = criterion(
                        logits_tf.reshape(-1, logits_tf.size(-1)), target_y.reshape(-1)
                    )
                total_val_loss += float(val_loss.item())

                # гриди-декод для метрик
                logits = model(
                    imgs, is_train=False, batch_max_length=max_len
                )  # [B, T, V]
                pred_ids = logits.argmax(-1).cpu()  # [B, T]
                tgt_ids = target_y.cpu()  # [B, T]

                for p_row, t_row in zip(pred_ids, tgt_ids):
                    hyp = decode_tokens(
                        p_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK
                    )
                    ref = decode_tokens(
                        t_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK
                    )
                    hyps.append(hyp)
                    refs.append(ref)

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_acc = compute_accuracy(refs, hyps)
        # CER/WER
        val_cer = sum(character_error_rate(r, h) for r, h in zip(refs, hyps)) / max(
            1, len(refs)
        )
        val_wer = sum(word_error_rate(r, h) for r, h in zip(refs, hyps)) / max(
            1, len(refs)
        )

        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("CER/val", val_cer, epoch)
        writer.add_scalar("WER/val", val_wer, epoch)

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

        # scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    writer.close()
    return {"val_acc": best_val_acc, "val_loss": best_val_loss, "exp_dir": exp_dir}


if __name__ == "__main__":
    IMG_H = 64
    IMG_W = 256

    base_config = dict(
        train_csvs=[r"C:\shared\orig_cyrillic\train.tsv"],
        train_roots=[r"C:\shared\orig_cyrillic\train"],
        val_csvs=[r"C:\shared\orig_cyrillic\test.tsv"],
        val_roots=[r"C:\shared\orig_cyrillic\test"],
        charset_path="charset.txt",
        img_h=IMG_H,
        img_w=IMG_W,
        device="cuda",
        encoding="utf-8",
        max_len=25,
    )

    default_params = {
        "batch_size": 128,
        "epochs": 100,
        "lr": 1e-4,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "weight_decay": 0.0,
        "momentum": 0.9,
        "shift_limit": 0.03,
        "scale_limit": 0.08,
        "rotate_limit": 3,
        "p_ShiftScaleRotate": 0.3,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "p_BrightnessContrast": 0.3,
    }

    try:
        with open("best_params.json", "r", encoding="utf-8") as f:
            best_params = json.load(f)
            print("Загружены best_params.json")
    except FileNotFoundError:
        best_params = default_params
        print("best_params.json не найден — использую дефолтные параметры.")

    train_transform = get_train_transform(best_params, img_h=IMG_H, img_w=IMG_W)

    RESUME_PATH = None

    run_training(
        **base_config,
        batch_size=best_params.get("batch_size", default_params["batch_size"]),
        epochs=best_params.get("epochs", default_params["epochs"]),
        lr=best_params.get("lr", default_params["lr"]),
        optimizer_name=best_params.get("optimizer", default_params["optimizer"]),
        scheduler_name=best_params.get("scheduler", default_params["scheduler"]),
        weight_decay=best_params.get("weight_decay", default_params["weight_decay"]),
        momentum=best_params.get("momentum", default_params["momentum"]),
        train_transform=train_transform,
        resume_path=RESUME_PATH,
        save_every=1,
    )