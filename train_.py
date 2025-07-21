import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import torch.cuda.amp as amp
import torch.profiler as profiler

from model_ import OCRCTC
from dataset import OCRDataset
from utils import decode
from metrics import compute_accuracy, character_error_rate, word_error_rate


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    # Enable fast algorithms when possible
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Allow TF32 on Ampere+ GPUs for faster matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_alphabet(
    paths,
    min_char_freq: int = 30,
    encoding: str = "utf-8",
    case_insensitive: bool = False,
) -> str:
    ctr = {}
    for p in paths:
        with open(p, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            for _, lbl in reader:
                lbl = lbl.lower() if case_insensitive else lbl
                for ch in lbl:
                    ctr[ch] = ctr.get(ch, 0) + 1
    return "".join(sorted(ch for ch, freq in ctr.items() if freq >= min_char_freq))


def create_dataloaders(
    train_csvs,
    train_roots,
    val_csvs,
    val_roots,
    alphabet,
    img_h,
    img_w,
    batch_size,
    augment=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
):
    train_sets = [
        OCRDataset(c, r, alphabet, img_h, img_w, augment=augment)
        for c, r in zip(train_csvs, train_roots)
    ]
    val_sets = [
        OCRDataset(c, r, alphabet, img_h, img_w, augment=False)
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
        collate_fn=OCRDataset.collate_fn_hybrid,
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=OCRDataset.collate_fn_hybrid,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for imgs, ctc_labs, inp_lens, lab_lens, *_ in tqdm(loader, desc="Train"):
        imgs, ctc_labs, lab_lens = (
            imgs.to(device),
            ctc_labs.to(device),
            lab_lens.to(device),
        )
        optimizer.zero_grad()
        with amp.autocast():
            logits = model(imgs)  # (T, B, C)
            T, B, C = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss = criterion(logits.float(), ctc_labs, input_lengths, lab_lens)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device, alphabet, scaler):
    model.eval()
    total_loss, refs, hyps = 0.0, [], []
    with torch.no_grad():
        for imgs, ctc_labs, inp_lens, lab_lens, *_ in tqdm(loader, desc="Val"):
            imgs, ctc_labs, lab_lens = (
                imgs.to(device),
                ctc_labs.to(device),
                lab_lens.to(device),
            )
            with amp.autocast():
                logits = model(imgs)
                T, B, C = logits.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                loss = criterion(logits.float(), ctc_labs, input_lengths, lab_lens)
            total_loss += loss.item()
            preds, _ = decode(logits, alphabet, method="greedy")
            offset = 0
            for L in lab_lens.tolist():
                seq = ctc_labs[offset : offset + L].tolist()
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
    backbone="res18",
    pretrained=True,
    msf=True,
    batch_size=32,
    epochs=20,
    lr=1e-3,
    device="cuda",
    min_char_freq=30,
    case_insensitive=False,
    use_profiler: bool = False,
    max_seq_len: int = 30,
):
    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    paths = train_csvs + val_csvs
    if alphabet is None:
        alphabet = build_alphabet(paths, min_char_freq, "utf-8", case_insensitive)
    if num_classes is None:
        num_classes = 1 + len(alphabet)

    exp_idx = 1
    while os.path.exists(f"exp{exp_idx}"):
        exp_idx += 1
    os.makedirs(f"exp{exp_idx}")

    best_loss_path = os.path.join(f"exp{exp_idx}", "best_loss.pth")
    best_acc_path = os.path.join(f"exp{exp_idx}", "best_acc.pth")
    tb_dir = os.path.join(f"exp{exp_idx}", "tb")

    model = OCRCTC(
        imgH=img_h,
        imgW=img_w,
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        msf=msf,
        max_seq_len=max_seq_len,
    ).to(device)
    try:
        model = torch.compile(model)
    except:
        pass

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler()
    train_loader, val_loader = create_dataloaders(
        train_csvs, train_roots, val_csvs, val_roots, alphabet, img_h, img_w, batch_size
    )

    best_val_loss, best_val_acc = float("inf"), -1.0
    if use_profiler:
        os.makedirs(tb_dir, exist_ok=True)
        with profiler.profile(
            schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=profiler.tensorboard_trace_handler(tb_dir),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for epoch in range(1, epochs + 1):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, scaler
                )
                prof.step()
                val_loss, val_acc, val_cer, val_wer = validate(
                    model, val_loader, criterion, device, alphabet, scaler
                )
                print(
                    f"Epoch {epoch}/{epochs} | Train L={train_loss:.4f}"
                    f" | Val L={val_loss:.4f} Acc={val_acc:.4f}"
                    f" CER={val_cer:.4f} WER={val_wer:.4f}"
                )
                # save best loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_loss_path)
                # save best acc (>= to catch epoch1)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_acc_path)
        print(f"Models & traces: exp{exp_idx}/best_loss.pth, best_acc.pth, tb/")
    else:
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_loss, val_acc, val_cer, val_wer = validate(
                model, val_loader, criterion, device, alphabet, scaler
            )
            print(
                f"Epoch {epoch}/{epochs} | Train L={train_loss:.4f}"
                f" | Val L={val_loss:.4f} Acc={val_acc:.4f}"
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
        train_csvs=[r"C:\shared\Archive_19_04\data_archive\gt_train.txt"],
        train_roots=[r"C:\shared\Archive_19_04\data_archive"],
        val_csvs=[r"C:\shared\Archive_19_04\data_archive\gt_test.txt"],
        val_roots=[r"C:\shared\Archive_19_04\data_archive"],
        alphabet=None,
        img_h=64,
        img_w=256,
        num_classes=None,
        backbone="convnextv2",
        pretrained=True,
        msf=True,
        batch_size=128,
        epochs=60,
        lr=1e-4,
        device="cuda",
        min_char_freq=30,
        case_insensitive=True,
        use_profiler=False,
    )
    run_training(**vars(cfg))
