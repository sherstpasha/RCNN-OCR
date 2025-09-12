import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import torch.cuda.amp as amp

from dataset import OCRDataset, build_alphabet, ResizeAndPad
from model import RCNN
from utils import decode
from metrics import compute_accuracy, character_error_rate, word_error_rate
import torchvision.transforms as T


# reproducibility + ускорения
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# загрузка датасетов
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
):
    transform = T.Compose(
        [
            ResizeAndPad(img_h=img_h, img_w=img_w),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0..1] → [-1,1]
        ]
    )

    train_sets = [
        OCRDataset(c, r, alphabet, img_h, img_w, transform=transform)
        for c, r in zip(train_csvs, train_roots)
    ]
    val_sets = [
        OCRDataset(c, r, alphabet, img_h, img_w, transform=transform)
        for c, r in zip(val_csvs, val_roots)
    ]

    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=OCRDataset.collate_fn,
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=OCRDataset.collate_fn,
    )
    return train_loader, val_loader


# одна эпоха обучения
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for imgs, targets, target_lengths in tqdm(loader, desc="Train"):
        imgs, targets, target_lengths = (
            imgs.to(device),
            targets.to(device),
            target_lengths.to(device),
        )
        optimizer.zero_grad()
        with amp.autocast():
            logits = model(imgs)  # (T, B, C)
            T, B, C = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss = criterion(logits.float(), targets, input_lengths, target_lengths)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


# валидация
def validate(model, loader, criterion, device, alphabet, scaler):
    model.eval()
    total_loss, refs, hyps = 0.0, [], []
    with torch.no_grad():
        for imgs, targets, target_lengths in tqdm(loader, desc="Val"):
            imgs, targets, target_lengths = (
                imgs.to(device),
                targets.to(device),
                target_lengths.to(device),
            )
            with amp.autocast():
                logits = model(imgs)
                T, B, C = logits.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                loss = criterion(logits.float(), targets, input_lengths, target_lengths)
            total_loss += loss.item()

            preds, _ = decode(logits, alphabet, method="greedy")

            offset = 0
            for L in target_lengths.tolist():
                seq = targets[offset : offset + L].tolist()
                offset += L
                refs.append("".join(alphabet[c - 1] for c in seq if c > 0))
            hyps.extend(preds)

    avg_loss = total_loss / len(loader)
    return (
        avg_loss,
        compute_accuracy(refs, hyps),
        sum(character_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs),
        sum(word_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs),
    )


def run_training(
    train_csvs,
    train_roots,
    val_csvs,
    val_roots,
    alphabet=None,
    img_h=64,
    img_w=256,
    num_classes=None,
    batch_size=32,
    epochs=20,
    lr=1e-3,
    device="cuda",
    min_char_freq=30,
):
    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # алфавит
    paths = train_csvs + val_csvs
    if alphabet is None:
        alphabet = build_alphabet(paths, min_char_freq)
    if num_classes is None:
        num_classes = 1 + len(alphabet)
    print(alphabet)
    # эксперимент
    exp_idx = 1
    while os.path.exists(f"exp{exp_idx}"):
        exp_idx += 1
    os.makedirs(f"exp{exp_idx}")

    best_loss_path = os.path.join(f"exp{exp_idx}", "best_loss.pth")
    best_acc_path = os.path.join(f"exp{exp_idx}", "best_acc.pth")

    # модель
    model = RCNN(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler()
    train_loader, val_loader = create_dataloaders(
        train_csvs, train_roots, val_csvs, val_roots, alphabet, img_h, img_w, batch_size
    )

    best_val_loss, best_val_acc = float("inf"), -1.0

    # цикл обучения
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, val_cer, val_wer = validate(
            model, val_loader, criterion, device, alphabet, scaler
        )
        print(
            f"Epoch {epoch}/{epochs} | Train {train_loss:.4f}"
            f" | Val {val_loss:.4f} Acc={val_acc:.4f}"
            f" CER={val_cer:.4f} WER={val_wer:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_loss_path)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_acc_path)

    print(f"Models saved in exp{exp_idx}/best_loss.pth, best_acc.pth")
    return model


if __name__ == "__main__":
    from argparse import Namespace

    cfg = Namespace(
        train_csvs=[r"C:\shared\orig_cyrillic\train.tsv"],
        train_roots=[r"C:\shared\orig_cyrillic\train"],
        val_csvs=[r"C:\shared\orig_cyrillic\test.tsv"],
        val_roots=[r"C:\shared\orig_cyrillic\test"],
        alphabet=None,
        img_h=64,
        img_w=256,
        num_classes=None,
        batch_size=16,
        epochs=60,
        lr=1e-4,
        device="cuda",
        min_char_freq=30,
    )
    run_training(**vars(cfg))
