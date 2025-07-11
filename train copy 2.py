import os
import csv
import random
import warnings
import textwrap

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from dataset import OCRDataset
from model import TransformerOCRModel
from vocab import build_vocab, SPECIAL_TOKENS

# --- Settings ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_seq2seq = True
experiment_name = "transformer_ocr"
log_dir = os.path.join("runs", experiment_name)
checkpoint_dir = os.path.join("checkpoints", experiment_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

img_height, img_max_width = 60, 240
batch_size, epochs = 64, 40
learning_rate = 1e-3

train_csvs = [r"C:\\shared\\Archive_19_04\\data_cyrillic\\gt_train.txt"]
train_image_roots = [r"C:\\shared\\Archive_19_04\\data_cyrillic\\train"]
val_csvs = [r"C:\\shared\\Archive_19_04\\data_cyrillic\\gt_test.txt"]
val_image_roots = [r"C:\\shared\\Archive_19_04\\data_cyrillic\\test"]

alphabet = OCRDataset.build_alphabet(train_csvs + val_csvs, min_char_freq=30, encoding="utf-8")
print(f"Using alphabet ({len(alphabet)}): {alphabet}")
vocab, inv_vocab = build_vocab(alphabet)
vocab_size = len(vocab)

# --- Data ---
train_datasets = [
    OCRDataset(csv_path=p, images_dir=img_root, alphabet=alphabet, img_height=img_height,
               img_max_width=img_max_width, augment=True, use_seq2seq=use_seq2seq)
    for p, img_root in zip(train_csvs, train_image_roots)
]
val_datasets = [
    OCRDataset(csv_path=p, images_dir=img_root, alphabet=alphabet, img_height=img_height,
               img_max_width=img_max_width, augment=False, use_seq2seq=use_seq2seq)
    for p, img_root in zip(val_csvs, val_image_roots)
]
train_ds = ConcatDataset(train_datasets)
val_ds = ConcatDataset(val_datasets)
collate_fn = OCRDataset.collate_fn_seq2seq if use_seq2seq else OCRDataset.collate_fn
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = TransformerOCRModel(vocab_size=vocab_size, backbone="resnet").to(device)
criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

writer = SummaryWriter(log_dir=log_dir)
metrics_csv = os.path.join(checkpoint_dir, "metrics_history.csv")
with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "val_loss", "lr"])

best_val_loss = float("inf")
global_step = 0

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for imgs, decoder_input, targets in train_loader:
        imgs, decoder_input, targets = imgs.to(device), decoder_input.to(device), targets.to(device)
        optimizer.zero_grad()
        out = model(imgs, decoder_input)  # [T, B, V]
        out = out.permute(1, 0, 2)        # [B, T, V]
        loss = criterion(out.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), global_step)
        train_loss += loss.item()
        global_step += 1

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, decoder_input, targets in val_loader:
            imgs, decoder_input, targets = imgs.to(device), decoder_input.to(device), targets.to(device)
            out = model(imgs, decoder_input)
            out = out.permute(1, 0, 2)
            val_loss += criterion(out.reshape(-1, vocab_size), targets.reshape(-1)).item()
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    # Save checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))

    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{optimizer.param_groups[0]['lr']:.6f}"])

    print(
        f"Epoch {epoch}/{epochs} | TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
        f"LR={optimizer.param_groups[0]['lr']:.6f}"
    )

writer.close()
