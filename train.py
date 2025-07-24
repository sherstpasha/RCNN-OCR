import os
import warnings

# suppress TF & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.models._utils"
)

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from dataset import OCRDataset, create_dataloaders
from model import TRBA
from utils import ctc_greedy_decoder, set_seed, log_samples
from metrics import character_error_rate, word_error_rate, compute_accuracy


def infer_batch(model, imgs, labs, lab_lens, alphabet, device, is_train):
    """
    Выполняет прямой проход и жадный декодинг для батча.
    Возвращает (out, preds, raws, loss, refs).
    """
    imgs, labs = imgs.to(device), labs.to(device)
    lab_lens = lab_lens.to(device)

    if model.use_attention:
        B = imgs.size(0)
        seqs, offset = [], 0
        for L in lab_lens.tolist():
            seqs.append(labs[offset : offset + L])
            offset += L
        max_L = max(s.size(0) for s in seqs)
        text_input = torch.zeros(B, max_L + 1, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            text_input[i, 1 : 1 + s.size(0)] = s
        inputs, targets = text_input[:, :-1], text_input[:, 1:]
        with autocast(device.type):
            out = model(imgs, text=inputs, is_train=is_train)
            T, B2, C = out.size()
            logits = out.permute(1, 0, 2).reshape(B2 * T, C)
            loss = model.criterion(logits, targets.reshape(B2 * T))
        pred_idxs = out.argmax(dim=2).transpose(0, 1)
        preds, raws = [], []
        for seq in pred_idxs:
            s, prev = [], 0
            for idx in seq.tolist():
                if idx != 0 and idx != prev:
                    s.append(alphabet[idx - 1])
                prev = idx
            preds.append("".join(s))
            raws.append(
                "".join(alphabet[i - 1] if i > 0 else "_" for i in seq.tolist())
            )
    else:
        with autocast(device.type):
            out = model(imgs)
            T, B2, _ = out.size()
            inp_lens = torch.full((B2,), T, dtype=torch.long, device=device)
            loss = model.criterion(out.float(), labs, inp_lens, lab_lens)
        preds, raws = ctc_greedy_decoder(out, alphabet)

    refs = []
    offset = 0
    for L in lab_lens.tolist():
        seq = labs[offset : offset + L].tolist()
        offset += L
        refs.append("".join(alphabet[i - 1] for i in seq if i > 0))
    return out, preds, raws, loss, refs


def train_epoch(
    model, loader, criterion, optimizer, device, scaler, writer, epoch, alphabet
):
    model.train()
    model.criterion = criterion
    loss_sum, refs, hyps = 0.0, [], []
    step = (epoch - 1) * len(loader)
    for imgs, labs, _, lab_lens in tqdm(loader, desc=f"Train {epoch}"):
        step += 1
        optimizer.zero_grad()
        _, preds, _, loss, batch_refs = infer_batch(
            model, imgs, labs, lab_lens, alphabet, device, is_train=True
        )
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        writer.add_scalar("train/loss_step", loss.item(), step)
        refs.extend(batch_refs)
        hyps.extend(preds)

    avg_loss = loss_sum / len(loader)
    acc = compute_accuracy(refs, hyps)
    cer = sum(character_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
    wer = sum(word_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", acc, epoch)
    writer.add_scalar("train/cer", cer, epoch)
    writer.add_scalar("train/wer", wer, epoch)
    return avg_loss, acc, cer, wer


def validate_epoch(model, loader, criterion, device, writer, epoch, alphabet):
    model.eval()
    model.criterion = criterion
    loss_sum, refs, hyps = 0.0, [], []
    with torch.no_grad():
        for imgs, labs, _, lab_lens in tqdm(loader, desc=f"Val {epoch}"):
            _, preds, raws, loss, batch_refs = infer_batch(
                model, imgs, labs, lab_lens, alphabet, device, is_train=False
            )
            loss_sum += loss.item()
            refs.extend(batch_refs)
            hyps.extend(preds)
    avg_loss = loss_sum / len(loader)
    acc = compute_accuracy(refs, hyps)
    cer = sum(character_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
    wer = sum(word_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", acc, epoch)
    writer.add_scalar("val/cer", cer, epoch)
    writer.add_scalar("val/wer", wer, epoch)
    return avg_loss, acc, cer, wer


def main():
    set_seed(42)
    exp_name = "exp1"  # задайте имя эксперимента здесь
    train_csvs = [r"C:\shared\Archive_19_04\data_cyrillic\gt_train.txt"]
    train_roots = [r"C:\shared\Archive_19_04\data_cyrillic\train"]
    val_csvs = [r"C:\shared\Archive_19_04\data_cyrillic\gt_test.txt"]
    val_roots = [r"C:\shared\Archive_19_04\data_cyrillic\test"]
    img_h, img_w = 60, 240
    batch_size, epochs, lr = 64, 40, 1e-3

    alphabet = OCRDataset.build_alphabet(
        train_csvs + val_csvs, min_char_freq=30, ignore_case=True
    )
    num_classes = len(alphabet) + 1
    train_loader, val_loader = create_dataloaders(
        train_csvs, train_roots, val_csvs, val_roots, alphabet, img_h, img_w, batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRBA(img_h, img_w, num_classes, transform=None, use_attention=True).to(
        device
    )

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

    # Setup experiment directories
    base_dir = exp_name
    log_dir = os.path.join(base_dir, "logs")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    best_loss, best_acc = float("inf"), 0.0

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

        v_loss, v_acc, v_cer, v_wer = validate_epoch(
            model, val_loader, criterion, device, writer, e, alphabet
        )
        imgs, labs, _, lab_lens = next(iter(val_loader))
        with torch.no_grad():
            _, preds, raws, _, truths = infer_batch(
                model, imgs, labs, lab_lens, alphabet, device, is_train=False
            )
        log_samples(
            e, imgs, lab_lens, preds, raws, truths, writer, n=10, tag="Val/Examples"
        )

        print(
            f"Epoch {e}/{epochs} | "
            f"Train L={t_loss:.4f} Acc={t_acc:.4f} CER={t_cer:.4f} WER={t_wer:.4f} | "
            f"Val L={v_loss:.4f} Acc={v_acc:.4f} CER={v_cer:.4f} WER={v_wer:.4f}"
        )
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_by_loss.pth"))
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_by_acc.pth"))
        scheduler.step(v_loss)

    writer.close()


if __name__ == "__main__":
    main()
