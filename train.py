# train.py

import os
import csv
import random
import warnings

# suppress TF & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
deprecated_mod = "torchvision.models._utils"
warnings.filterwarnings("ignore", category=UserWarning, module=deprecated_mod)

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from dataset import OCRDataset
from model import CRNN
from utils import ctc_greedy_decoder
from metrics import character_error_rate, word_error_rate, compute_accuracy

# 1) Fix random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2) Experiment settings
experiment_name = "exp_1_archive"
log_dir        = os.path.join("runs", experiment_name)
checkpoint_dir = os.path.join("checkpoints", experiment_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 3) Hyperparameters & paths
img_height, img_max_width = 60, 200
batch_size, epochs       = 128*3, 40
learning_rate            = 1e-3

# 4) Data annotation files (can list multiple) and their image roots
train_csvs = [
    r"C:\shared\Archive_19_04\data_archive\gt_train.txt",
#    r"C:\shared\Archive_19_04\data_cyrillic\gt_train.txt",
#    r"C:\shared\Archive_19_04\data_cyrillic\gt_test.txt",
#    r"C:\shared\Archive_19_04\data_hkr\gt_train.txt",
#    r"C:\shared\Archive_19_04\data_hkr\gt_test.txt",
#    r"C:\shared\Archive_19_04\data_school\gt_train.txt",
#    r"C:\shared\Archive_19_04\data_school\gt_test.txt",
]

train_image_roots = [
    r"C:\shared\Archive_19_04\data_archive",
#    r"C:\shared\Archive_19_04\data_cyrillic\train",
#    r"C:\shared\Archive_19_04\data_cyrillic\test",
#    r"C:\shared\Archive_19_04\data_hkr\train",
#    r"C:\shared\Archive_19_04\data_hkr\test",
#    r"C:\shared\Archive_19_04\data_school",
#    r"C:\shared\Archive_19_04\data_school",
]

val_csvs   = [
    r"C:\shared\Archive_19_04\data_archive\gt_test.txt",
]
val_image_roots = [
    r"C:\shared\Archive_19_04\data_archive",
    
]

# 5) Build alphabet from all CSVs, filtering rare chars
alphabet = OCRDataset.build_alphabet(
    train_csvs + val_csvs,
    min_char_freq=30,
    encoding="utf-8"
)
print(f"Using alphabet ({len(alphabet)}): {alphabet}")
num_classes = len(alphabet) + 1  # +1 for CTC blank

# 6) Create datasets and dataloaders
train_datasets = []
for csv_path, img_root in zip(train_csvs, train_image_roots):
    train_datasets.append(
        OCRDataset(
            csv_path=csv_path,
            images_dir=img_root,
            alphabet=alphabet,
            img_height=img_height,
            img_max_width=img_max_width,
            min_char_freq=1
        )
    )

val_datasets = []
for csv_path, img_root in zip(val_csvs, val_image_roots):
    val_datasets.append(
        OCRDataset(
            csv_path=csv_path,
            images_dir=img_root,
            alphabet=alphabet,
            img_height=img_height,
            img_max_width=img_max_width,
            min_char_freq=1
        )
    )

train_ds = ConcatDataset(train_datasets)
val_ds   = ConcatDataset(val_datasets)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    collate_fn=OCRDataset.collate_fn
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    collate_fn=OCRDataset.collate_fn
)

# 7) Model, loss, optimizer, scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = CRNN(
    img_height, img_max_width, num_classes,
    pretrained=True, transform="none", backbone="resnet50"
).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# 8) TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

# 9) Prepare metrics history CSV and best-metrics TXT
metrics_csv = os.path.join(checkpoint_dir, "metrics_history.csv")
with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "epoch",
        "train_loss","train_acc","train_cer","train_wer",
        "val_loss","val_acc","val_cer","val_wer","lr"
    ])

best_txt = os.path.join(checkpoint_dir, "best_metrics.txt")
with open(best_txt, "w", encoding="utf-8") as f:
    f.write(f"Experiment: {experiment_name}\n\n")

best_val_loss = float("inf")
best_val_acc  = 0.0

# How many samples to log each epoch
num_logged_samples = 10

def log_samples(epoch, imgs, lab_lens, preds, raws, truths, n=num_logged_samples):
    n = min(n, len(truths))
    fig, axs = plt.subplots(n, 1, figsize=(6, 2 * n))
    for i in range(n):
        img_np = imgs[i].cpu().squeeze(0).numpy()
        axs[i].imshow(img_np, cmap="gray")
        axs[i].set_title(f"GT: {truths[i]} | Pred: {preds[i]} | Raw: {raws[i]}")
        axs[i].axis("off")
    writer.add_figure("Examples", fig, epoch)
    plt.close(fig)

# 10) Training loop
global_step = 0
for epoch in range(1, epochs + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    train_refs, train_hyps = [], []
    for imgs, labs, _, lab_lens in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        lab_lens = lab_lens.to(device)

        optimizer.zero_grad()
        out = model(imgs)  # [T, B, C]
        T, B, _ = out.size()
        inp_lens = torch.full((B,), T, dtype=torch.long, device=device)
        loss = criterion(out, labs, inp_lens, lab_lens)
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), global_step)
        train_loss += loss.item()
        global_step += 1

        preds, _ = ctc_greedy_decoder(out, alphabet)
        offset = 0
        for L in lab_lens.tolist():
            seq = labs[offset:offset+L].tolist()
            offset += L
            train_refs.append("".join(alphabet[i-1] for i in seq if i>0))
        train_hyps.extend(preds)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = compute_accuracy(train_refs, train_hyps)
    train_cer = sum(character_error_rate(r, h) for r, h in zip(train_refs, train_hyps)) / len(train_refs)
    train_wer = sum(word_error_rate(r, h)       for r, h in zip(train_refs, train_hyps)) / len(train_refs)

    writer.add_scalar("train/accuracy", train_acc, epoch)
    writer.add_scalar("train/CER", train_cer, epoch)
    writer.add_scalar("train/WER", train_wer, epoch)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_refs, val_hyps = [], []
    with torch.no_grad():
        for imgs, labs, _, lab_lens in val_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            lab_lens = lab_lens.to(device)

            out = model(imgs)
            T, B, _ = out.size()
            inp_lens = torch.full((B,), T, dtype=torch.long, device=device)
            val_loss += criterion(out, labs, inp_lens, lab_lens).item()

            preds, _ = ctc_greedy_decoder(out, alphabet)
            offset = 0
            for L in lab_lens.tolist():
                seq = labs[offset:offset+L].tolist()
                offset += L
                val_refs.append("".join(alphabet[i-1] for i in seq if i>0))
            val_hyps.extend(preds)

    avg_val_loss = val_loss / len(val_loader)
    val_acc      = compute_accuracy(val_refs, val_hyps)
    val_cer      = sum(character_error_rate(r, h) for r, h in zip(val_refs, val_hyps)) / len(val_refs)
    val_wer      = sum(word_error_rate(r, h)       for r, h in zip(val_refs, val_hyps)) / len(val_refs)

    writer.add_scalar("val/loss", avg_val_loss, epoch)
    writer.add_scalar("val/accuracy", val_acc, epoch)
    writer.add_scalar("val/CER", val_cer, epoch)
    writer.add_scalar("val/WER", val_wer, epoch)

    lr = optimizer.param_groups[0]["lr"]

    # --- Record metrics to CSV ---
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            epoch,
            f"{avg_train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{train_cer:.6f}",
            f"{train_wer:.6f}",
            f"{avg_val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{val_cer:.6f}",
            f"{val_wer:.6f}",
            f"{lr:.6f}",
        ])

    # --- Checkpoint by loss ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        path = os.path.join(checkpoint_dir, "best_by_loss.pth")
        torch.save(model.state_dict(), path)
        with open(best_txt, "a", encoding="utf-8") as f:
            f.write(f"[Epoch {epoch}] best val_loss = {best_val_loss:.6f}\n")

    # --- Checkpoint by accuracy ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        path = os.path.join(checkpoint_dir, "best_by_acc.pth")
        torch.save(model.state_dict(), path)
        with open(best_txt, "a", encoding="utf-8") as f:
            f.write(f"[Epoch {epoch}] best val_acc  = {best_val_acc:.6f}\n")

    # --- LR scheduler step ---
    scheduler.step(avg_val_loss)
    writer.add_scalar("learning_rate", lr, epoch)

    print(
        f"Epoch {epoch}/{epochs} | "
        f"TrainLoss={avg_train_loss:.4f} Acc={train_acc:.4f} CER={train_cer:.4f} WER={train_wer:.4f} | "
        f"ValLoss={avg_val_loss:.4f} Acc={val_acc:.4f} CER={val_cer:.4f} WER={val_wer:.4f} | "
        f"LR={lr:.6f}"
    )

    # --- Log sample predictions ---
    imgs, labs, _, lab_lens = next(iter(val_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        out = model(imgs)
    preds, raws = ctc_greedy_decoder(out, alphabet)
    truths = []
    offset = 0
    for L in lab_lens.tolist():
        seq = labs[offset:offset+L].tolist()
        offset += L
        truths.append("".join(alphabet[i-1] for i in seq if i>0))
    log_samples(epoch, imgs, lab_lens, preds, raws, truths, n=num_logged_samples)

writer.close()
