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
import torchvision.utils as vutils

from dataset import OCRDataset
from model import TransformerOCRModel
from vocab import build_vocab, SPECIAL_TOKENS
from metrics import character_error_rate, word_error_rate, compute_accuracy

# -------------------------------------------------------------------
# Функция для логирования примеров в TensorBoard
def log_samples_seq2seq(epoch, imgs, preds, truths, writer, n=10, tag="Examples"):
    n = min(n, len(truths))
    fig, axs = plt.subplots(n, 1, figsize=(6, 2 * n))
    for i in range(n):
        img_np = imgs[i].cpu().squeeze(0).numpy()
        axs[i].imshow(img_np, cmap="gray")
        title = f"GT: {truths[i]} | Pred: {preds[i]}"
        wrapped = "\n".join(textwrap.wrap(title, width=40))
        axs[i].set_title(wrapped, fontsize=8)
        axs[i].axis("off")
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)

# -------------------------------------------------------------------
# Жадный декодер (greedy) для Transformer
def greedy_decode_batch(model, imgs, inv_vocab, sos_id, eos_id, max_len=32, device="cpu"):
    """
    imgs: [B,1,H,W]
    Возвращает preds: List[str], raws: List[str]
    """
    model.eval()
    B = imgs.size(0)
    preds, raws = [""] * B, [""] * B
    with torch.no_grad():
        # начнём со все <sos>
        cur_input = torch.full((B, 1), sos_id, dtype=torch.long, device=device)  # [B,1]
        for _step in range(max_len):
            out = model(imgs, cur_input)           # [T, B, V]
            logits = out[-1, :, :]                  # последний шаг [B,V]
            next_tokens = logits.argmax(dim=-1)     # [B]
            cur_input = torch.cat([cur_input, next_tokens.unsqueeze(1)], dim=1)  # append
        # теперь cur_input: [B, L_pred]
        for b in range(B):
            seq = cur_input[b].tolist()[1:]  # пропустили <sos>
            string = ""
            raw = []
            for t in seq:
                if t == eos_id:
                    break
                ch = inv_vocab.get(t, "")
                string += ch
                raw.append(ch)
            preds[b] = string
            raws[b] = "".join(raw)
    return preds, raws

# -------------------------------------------------------------------
# 1) Fix random seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2) Settings
use_seq2seq = True
experiment_name = "transformer_ocr"
log_dir = os.path.join("runs", experiment_name)
checkpoint_dir = os.path.join("checkpoints", experiment_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

img_height, img_max_width = 60, 240
batch_size, epochs = 64, 40
learning_rate = 1e-3

train_csvs = [r"C:\shared\Archive_19_04\data_cyrillic\gt_train.txt"]
train_image_roots = [r"C:\shared\Archive_19_04\data_cyrillic\train"]
val_csvs = [r"C:\shared\Archive_19_04\data_cyrillic\gt_test.txt"]
val_image_roots = [r"C:\shared\Archive_19_04\data_cyrillic\test"]

# 3) Alphabet & Vocab
alphabet = OCRDataset.build_alphabet(train_csvs + val_csvs, min_char_freq=30, encoding="utf-8")
print(f"Using alphabet ({len(alphabet)}): {alphabet}")
vocab, inv_vocab = build_vocab(alphabet)
vocab_size = len(vocab)
sos_id, eos_id, pad_id = SPECIAL_TOKENS["<sos>"], SPECIAL_TOKENS["<eos>"], SPECIAL_TOKENS["<pad>"]

# 4) Data
train_ds = ConcatDataset([
    OCRDataset(csv_path=p, images_dir=r, alphabet=alphabet,
               img_height=img_height, img_max_width=img_max_width,
               augment=True, use_seq2seq=use_seq2seq)
    for p,r in zip(train_csvs, train_image_roots)
])
val_ds = ConcatDataset([
    OCRDataset(csv_path=p, images_dir=r, alphabet=alphabet,
               img_height=img_height, img_max_width=img_max_width,
               augment=False, use_seq2seq=use_seq2seq)
    for p,r in zip(val_csvs, val_image_roots)
])
collate_fn = OCRDataset.collate_fn_seq2seq if use_seq2seq else OCRDataset.collate_fn
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 5) Model, Loss, Optimizer, Scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = TransformerOCRModel(vocab_size=vocab_size, backbone="resnet").to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# 6) TensorBoard & metrics files
writer = SummaryWriter(log_dir=log_dir)
metrics_csv = os.path.join(checkpoint_dir, "metrics_history.csv")
with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_cer", "val_wer", "lr"])
best_val_loss = float("inf")
global_step = 0

# 7) Training loop
for epoch in range(1, epochs+1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for imgs, decoder_input, targets in train_loader:
        imgs, decoder_input, targets = imgs.to(device), decoder_input.to(device), targets.to(device)
        optimizer.zero_grad()
        out = model(imgs, decoder_input)        # [T, B, V]
        out = out.permute(1, 0, 2)              # [B, T, V]
        loss = criterion(out.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), global_step)
        train_loss += loss.item()
        global_step += 1

    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_preds, all_truths = [], []
    with torch.no_grad():
        for imgs, decoder_input, targets in val_loader:
            imgs, decoder_input, targets = imgs.to(device), decoder_input.to(device), targets.to(device)
            out = model(imgs, decoder_input).permute(1,0,2)
            val_loss += criterion(out.reshape(-1, vocab_size), targets.reshape(-1)).item()
            # собираем ground truth
            for seq in targets.cpu().tolist():
                # пропускаем <pad>, считаем до <eos>
                s = "".join(inv_vocab[t] for t in seq if t!=pad_id and t!=eos_id and t!=sos_id)
                all_truths.append(s)

    avg_val_loss = val_loss / len(val_loader)

    # —— greedy decode для метрик и логов
    sample_imgs, _, _ = next(iter(val_loader))
    sample_imgs = sample_imgs.to(device)
    preds, raws = greedy_decode_batch(model, sample_imgs, inv_vocab, sos_id, eos_id, device=device)
    truths = []
    # ground truths для этого батча
    for seq in sample_imgs.new_zeros(1): pass  # placeholder
    # на самом деле берем первые B из all_truths
    truths = all_truths[: len(preds) ]

    # Метрики
    val_acc = compute_accuracy(all_truths, 
                               greedy_decode_batch(model, 
                                                   torch.cat([sample_imgs for _ in range(1)],0),
                                                   inv_vocab, sos_id, eos_id, device=device)[0])
    val_cer = np.mean([
        character_error_rate(r, h)
        for r, h in zip(
            all_truths,
            greedy_decode_batch(
                model,
                sample_imgs,
                inv_vocab,       # ← тут было invocab
                sos_id,
                eos_id,
                device=device
            )[0]
        )
    ])
    val_wer = np.mean([
        word_error_rate(r, h)
        for r, h in zip(
            all_truths,
            greedy_decode_batch(
                model,
                sample_imgs,
                inv_vocab,       # ← и тут
                sos_id,
                eos_id,
                device=device
            )[0]
        )
    ])

    writer.add_scalar("val/loss", avg_val_loss, epoch)
    writer.add_scalar("val/accuracy", val_acc, epoch)
    writer.add_scalar("val/CER", val_cer, epoch)
    writer.add_scalar("val/WER", val_wer, epoch)

    scheduler.step(avg_val_loss)

    # Сохраняем лучший чекпоинт
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))

    # Запись в CSV
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            epoch,
            f"{avg_train_loss:.6f}",
            f"{avg_val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{val_cer:.6f}",
            f"{val_wer:.6f}",
            f"{optimizer.param_groups[0]['lr']:.6f}",
        ])

    # Логи в консоль
    print(
        f"Epoch {epoch}/{epochs} | "
        f"TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
        f"Acc={val_acc:.4f} CER={val_cer:.4f} WER={val_wer:.4f} | "
        f"LR={optimizer.param_groups[0]['lr']:.6f}"
    )

    # Логирование примеров
    log_samples_seq2seq(epoch, sample_imgs.cpu(), preds, truths, writer, n=10, tag="Val/Examples")

writer.close()
