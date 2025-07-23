import os
import csv
import random
import warnings
import textwrap

# suppress TF & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.models._utils"
)

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import torch.profiler as profiler

import matplotlib.pyplot as plt

from dataset import OCRDataset
from model import TRBA
from utils import ctc_greedy_decoder
from metrics import character_error_rate, word_error_rate, compute_accuracy


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # allow TF32 for faster matmuls on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # deterministic vs benchmark
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_dataloaders(
    train_csvs,
    train_roots,
    val_csvs,
    val_roots,
    alphabet,
    img_h,
    img_w,
    batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
):
    train_sets = [
        OCRDataset(c, r, alphabet, img_h, img_w, augment=True, ignore_case=True)
        for c, r in zip(train_csvs, train_roots)
    ]
    val_sets = [
        OCRDataset(c, r, alphabet, img_h, img_w, augment=False, ignore_case=True)
        for c, r in zip(val_csvs, val_roots)
    ]
    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=OCRDataset.collate_fn,
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=OCRDataset.collate_fn,
    )
    return train_loader, val_loader


def log_samples(
    epoch, imgs, lab_lens, preds, raws, truths, writer, n=5, tag="Examples"
):
    """
    Показывает случайные n примеров из батча в TensorBoard.
    tag — название вкладки.
    """
    count = len(truths)
    n = min(n, count)
    idxs = random.sample(range(count), n)
    fig, axs = plt.subplots(n, 1, figsize=(6, 2 * n))
    for i, idx in enumerate(idxs):
        img_np = imgs[idx].cpu().squeeze(0).numpy()
        axs[i].imshow(img_np, cmap="gray")
        title = f"GT: {truths[idx]} | Pred: {preds[idx]}"
        wrapped = "\n".join(textwrap.wrap(title, width=40))
        axs[i].set_title(f"{wrapped}\nRaw: {raws[idx]}", fontsize=8)
        axs[i].axis("off")
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def train_epoch(
    model, loader, criterion, optimizer, device, scaler, writer, epoch, alphabet
):
    model.train()
    loss_sum, refs, hyps = 0.0, [], []
    step = (epoch - 1) * len(loader)
    for imgs, labs, _, lab_lens in tqdm(loader, desc=f"Train {epoch}"):
        step += 1
        imgs, labs = imgs.to(device), labs.to(device)
        lab_lens = lab_lens.to(device)
        optimizer.zero_grad()

        if model.use_attention:
            # 1) делаем text_input: [B, max_L+1], text_input[:,0]=GO=0
            B = imgs.size(0)
            seqs, offset = [], 0
            for L in lab_lens.tolist():
                seqs.append(labs[offset:offset+L])
                offset += L
            max_L = max(s.size(0) for s in seqs)
            text_input = torch.zeros(B, max_L+1, dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                text_input[i, 1:1+s.size(0)] = s

            # 2) сдвигаем: inputs=[:, :-1], targets=[:, 1:]
            inputs  = text_input[:, :-1]  # [B, T]
            targets = text_input[:, 1:]   # [B, T]

            # 3) forward + CE-loss
            with autocast("cuda"):
                out = model(imgs, text=inputs, is_train=True)  # [T, B, C]
                T, B2, C = out.size()
                logits = out.permute(1,0,2).reshape(B2*T, C)
                loss = criterion(logits, targets.reshape(B2*T))

        else:
            # CTC path без изменений
            with autocast("cuda"):
                out = model(imgs)
                T, B2, _ = out.size()
                inp_lens = torch.full((B2,), T, dtype=torch.long, device=device)
                loss = criterion(out.float(), labs, inp_lens, lab_lens)

        # 4) backward + clip + step
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        writer.add_scalar("train/loss_step", loss.item(), step)

        # 5) жадный декодер
        if model.use_attention:
            pred_idxs = out.argmax(dim=2).transpose(0,1)  # [B, T]
            preds, raws = [], []
            for seq in pred_idxs:
                s, prev = [], 0
                for idx in seq.tolist():
                    if idx != 0 and idx != prev:
                        s.append(alphabet[idx-1])
                    prev = idx
                preds.append("".join(s))
                raws.append("".join(alphabet[i-1] if i>0 else "_" for i in seq.tolist()))
        else:
            preds, raws = ctc_greedy_decoder(out, alphabet)

        # 6) собираем метрики
        offset = 0
        for L in lab_lens.tolist():
            seq = labs[offset:offset+L].tolist()
            offset += L
            refs.append("".join(alphabet[i-1] for i in seq if i>0))
        hyps.extend(preds)

    # финальные счёты
    avg_loss = loss_sum / len(loader)
    acc = compute_accuracy(refs, hyps)
    cer = sum(character_error_rate(r,h) for r,h in zip(refs,hyps)) / len(refs)
    wer = sum(word_error_rate(r,h)       for r,h in zip(refs,hyps)) / len(refs)
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", acc,      epoch)
    writer.add_scalar("train/cer",      cer,      epoch)
    writer.add_scalar("train/wer",      wer,      epoch)
    return avg_loss, acc, cer, wer


def validate_epoch(model, loader, criterion, device, writer, epoch, alphabet):
    model.eval()
    loss_sum, refs, hyps = 0.0, [], []
    with torch.no_grad():
        for imgs, labs, _, lab_lens in tqdm(loader, desc=f"Val {epoch}"):
            imgs, labs = imgs.to(device), labs.to(device)
            lab_lens = lab_lens.to(device)

            if model.use_attention:
                # готовим text_input и inputs/targets точно как в train
                B = imgs.size(0)
                seqs, offset = [], 0
                for L in lab_lens.tolist():
                    seqs.append(labs[offset:offset+L])
                    offset += L
                max_L = max(s.size(0) for s in seqs)
                text_input = torch.zeros(B, max_L+1, dtype=torch.long, device=device)
                for i, s in enumerate(seqs):
                    text_input[i,1:1+s.size(0)] = s

                inputs  = text_input[:, :-1]
                targets = text_input[:,  1:]

                with autocast("cuda"):
                    out = model(imgs, text=inputs, is_train=False)
                    T, B2, C = out.size()
                    logits = out.permute(1,0,2).reshape(B2*T, C)
                    loss   = criterion(logits, targets.reshape(B2*T))

            else:
                with autocast("cuda"):
                    out = model(imgs)
                    T, B2, _ = out.size()
                    inp_lens = torch.full((B2,), T, dtype=torch.long, device=device)
                    loss = criterion(out.float(), labs, inp_lens, lab_lens)

            loss_sum += loss.item()

            # inference‑декодер
            if model.use_attention:
                pred_idxs = out.argmax(dim=2).transpose(0,1)
                preds, raws = [], []
                for seq in pred_idxs:
                    s, prev = [], 0
                    for idx in seq.tolist():
                        if idx != 0 and idx != prev:
                            s.append(alphabet[idx-1])
                        prev = idx
                    preds.append("".join(s))
                    raws.append("".join(alphabet[i-1] if i>0 else "_" for i in seq.tolist()))
            else:
                preds, raws = ctc_greedy_decoder(out, alphabet)

            offset = 0
            for L in lab_lens.tolist():
                seq = labs[offset:offset+L].tolist()
                offset += L
                refs.append("".join(alphabet[i-1] for i in seq if i>0))
            hyps.extend(preds)

    avg_loss = loss_sum / len(loader)
    acc = compute_accuracy(refs, hyps)
    cer = sum(character_error_rate(r,h) for r,h in zip(refs,hyps)) / len(refs)
    wer = sum(word_error_rate(r,h)       for r,h in zip(refs,hyps)) / len(refs)
    writer.add_scalar("val/loss",     avg_loss, epoch)
    writer.add_scalar("val/accuracy", acc,      epoch)
    writer.add_scalar("val/cer",      cer,      epoch)
    writer.add_scalar("val/wer",      wer,      epoch)
    return avg_loss, acc, cer, wer


def main(use_profiler=False):
    set_seed(42)
    # dataset paths
    train_csvs = [
#        r"C:\shared\Archive_19_04\data_archive\gt_train.txt",
#                   r"C:\shared\Archive_19_04\data_cyrillic\gt_train.txt",
                     r"C:\shared\Archive_19_04\data_hkr\gt_train.txt",
#                     r"C:\shared\Archive_19_04\data_school\gt_train.txt",
#                     r"C:\shared\Archive_19_04\foreverschool_notebooks_RU\train.csv"
                     ]
    train_roots = [
#        r"C:\shared\Archive_19_04\data_archive",
#                    r"C:\shared\Archive_19_04\data_cyrillic\train",
                      r"C:\shared\Archive_19_04\data_hkr\train",
#                      r"C:\shared\Archive_19_04\data_school",
#                      r"C:\shared\Archive_19_04\foreverschool_notebooks_RU\train"
                      ]
    val_csvs = [
#        r"C:\shared\Archive_19_04\data_archive\gt_test.txt",
#                 r"C:\shared\Archive_19_04\data_cyrillic\gt_test.txt",
                 r"C:\shared\Archive_19_04\data_hkr\gt_test.txt",
#                 r"C:\shared\Archive_19_04\data_school\gt_test.txt",
#                 r"C:\shared\Archive_19_04\foreverschool_notebooks_RU\val.csv"
                 ]
    val_roots = [
#        r"C:\shared\Archive_19_04\data_archive",
#                  r"C:\shared\Archive_19_04\data_cyrillic\test",
                    r"C:\shared\Archive_19_04\data_hkr\test",
#                    r"C:\shared\Archive_19_04\data_school",
#                    r"C:\shared\Archive_19_04\foreverschool_notebooks_RU\val"
                    ]
    img_h, img_w = 60, 240
    batch_size, epochs, lr = 128, 40, 1e-3

    alphabet = OCRDataset.build_alphabet(train_csvs + val_csvs, min_char_freq=30, ignore_case=True)
    num_classes = len(alphabet) + 1

    train_loader, val_loader = create_dataloaders(
        train_csvs, train_roots, val_csvs, val_roots, alphabet, img_h, img_w, batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRBA(
        img_h,
        img_w,
        num_classes,
        transform=None,
        use_attention=True
    ).to(device)

    if model.use_attention:
        pad_idx = num_classes - 1
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    else:
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = GradScaler()

    writer = SummaryWriter(log_dir="runs/expatt")
    best_loss, best_acc = float("inf"), 0.0
    os.makedirs("checkpointsatt", exist_ok=True)

    for e in range(1, epochs + 1):
        t_loss, t_acc, t_cer, t_wer = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            writer,
            e,
            alphabet,
        )
        # log random train samples
        imgs, labs, _, lab_lens = next(iter(train_loader))
        with torch.no_grad():
            if model.use_attention:
                # подготовка text_input
                B = imgs.size(0)
                offset = 0
                seqs = []
                for L in lab_lens.tolist():
                    seqs.append(labs[offset : offset + L])
                    offset += L
                max_L = max(s.size(0) for s in seqs)
                text_input = torch.zeros(B, max_L + 1, dtype=torch.long, device=device)
                for i, s in enumerate(seqs):
                    text_input[i, 1 : s.size(0) + 1] = s
                out = model(imgs.to(device), text=text_input, is_train=False)
                pred_idxs = out.argmax(dim=2).transpose(0,1)
                preds, raws = [], []
                for seq in pred_idxs:
                    s, prev = [], 0
                    for idx in seq.tolist():
                        if idx != 0 and idx != prev:
                            s.append(alphabet[idx-1])
                        prev = idx
                    preds.append("".join(s))
                    raws.append("".join(alphabet[i-1] if i>0 else "_" for i in seq.tolist()))
            else:
                out = model(imgs.to(device))
                preds, raws = ctc_greedy_decoder(out, alphabet)

        truths = []
        off = 0
        for L in lab_lens.tolist():
            seq = labs[off : off + L].tolist()
            off += L
            truths.append("".join(alphabet[i - 1] for i in seq if i > 0))
        log_samples(
            e, imgs, lab_lens, preds, raws, truths, writer, n=5, tag="Train/Examples"
        )

        v_loss, v_acc, v_cer, v_wer = validate_epoch(
            model, val_loader, criterion, device, writer, e, alphabet
        )
        # log random val samples
        imgs, labs, _, lab_lens = next(iter(val_loader))
        with torch.no_grad():
            if model.use_attention:
                # та же подготовка text_input
                B = imgs.size(0)
                offset = 0
                seqs = []
                for L in lab_lens.tolist():
                    seqs.append(labs[offset : offset + L])
                    offset += L
                max_L = max(s.size(0) for s in seqs)
                text_input = torch.zeros(B, max_L + 1, dtype=torch.long, device=device)
                for i, s in enumerate(seqs):
                    text_input[i, 1 : s.size(0) + 1] = s
                out = model(imgs.to(device), text=text_input, is_train=False)
                pred_idxs = out.argmax(dim=2).transpose(0,1)
                preds, raws = [], []
                for seq in pred_idxs:
                    s, prev = [], 0
                    for idx in seq.tolist():
                        if idx != 0 and idx != prev:
                            s.append(alphabet[idx-1])
                        prev = idx
                    preds.append("".join(s))
                    raws.append("".join(alphabet[i-1] if i>0 else "_" for i in seq.tolist()))
            else:
                out = model(imgs.to(device))
                preds, raws = ctc_greedy_decoder(out, alphabet)

        log_samples(
            e, imgs, lab_lens, preds, raws, truths, writer, n=5,
            tag="Train/Examples"  # или "Val/Examples" для validation
        )

        print(
            f"Epoch {e}/{epochs} | "
            f"Train L={t_loss:.4f} Acc={t_acc:.4f} CER={t_cer:.4f} WER={t_wer:.4f} | "
            f"Val L={v_loss:.4f} Acc={v_acc:.4f} CER={v_cer:.4f} WER={v_wer:.4f}"
        )

        # save best
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), "checkpointsatt/best_by_loss.pth")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), "checkpointsatt/best_by_acc.pth")

        scheduler.step(v_loss)

    writer.close()


if __name__ == "__main__":
    main(use_profiler=False)
