import os

# suppress TensorFlow & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings

# suppress torchvision pretrained warnings
deprecated_mod = "torchvision.models._utils"
warnings.filterwarnings("ignore", category=UserWarning, module=deprecated_mod)

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from dataset import OCRDataset
from model import CRNN
from utils import ctc_greedy_decoder
from metrics import character_error_rate, word_error_rate, compute_accuracy

# Fix random seeds for reproducibility
seed = 42
import random as _random

_random.seed(seed)
import numpy as _np

_np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters & paths
csv_path = "C:/translit/labels.csv"
images_dir = "C:/translit/images"
log_dir = "runs/exp1"
img_height, img_max_width = 60, 200
batch_size, epochs = 128, 40
learning_rate, val_split = 1e-3, 0.1
alphabet = "бвгджклмнпрст2456789"
num_classes = len(alphabet) + 1  # +1 for blank

# Prepare data tensorboard --logdir=runs/exp1

dataset = OCRDataset(csv_path, images_dir, alphabet, img_height, img_max_width)
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=OCRDataset.collate_fn
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, collate_fn=OCRDataset.collate_fn
)

# Model, loss, optimizer, scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transform can be 'none', 'affine' or 'tps'
model = CRNN(
    img_height,
    img_max_width,
    num_classes,
    pretrained=True,
    transform="none",
    backbone="resnet34",
).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

best_val_loss = float("inf")


def log_samples(epoch, imgs, labs, lab_lens, preds, raws, truths):
    # show predictions
    fig, axs = plt.subplots(3, 1, figsize=(6, 6))
    for i in range(3):
        img_np = imgs[i].cpu().squeeze(0).numpy()
        axs[i].imshow(img_np, cmap="gray")
        axs[i].set_title(f"GT: {truths[i]} | Pred: {preds[i]} | Raw: {raws[i]}")
        axs[i].axis("off")
    writer.add_figure("Examples", fig, epoch)
    plt.close(fig)


# Training loop
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
        out = model(imgs)
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
        refs = []
        for L in lab_lens.tolist():
            seq = labs[offset : offset + L].tolist()
            offset += L
            refs.append("".join([alphabet[i - 1] for i in seq]))
        train_refs.extend(refs)
        train_hyps.extend(preds)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = compute_accuracy(train_refs, train_hyps)
    train_cer = sum(
        character_error_rate(r, h) for r, h in zip(train_refs, train_hyps)
    ) / len(train_refs)
    train_wer = sum(
        word_error_rate(r, h) for r, h in zip(train_refs, train_hyps)
    ) / len(train_refs)
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
            refs = []
            for L in lab_lens.tolist():
                seq = labs[offset : offset + L].tolist()
                offset += L
                refs.append("".join([alphabet[i - 1] for i in seq]))
            val_refs.extend(refs)
            val_hyps.extend(preds)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = compute_accuracy(val_refs, val_hyps)
    val_cer = sum(character_error_rate(r, h) for r, h in zip(val_refs, val_hyps)) / len(
        val_refs
    )
    val_wer = sum(word_error_rate(r, h) for r, h in zip(val_refs, val_hyps)) / len(
        val_refs
    )
    writer.add_scalar("val/loss", avg_val_loss, epoch)
    writer.add_scalar("val/accuracy", val_acc, epoch)
    writer.add_scalar("val/CER", val_cer, epoch)
    writer.add_scalar("val/WER", val_wer, epoch)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f"Epoch {epoch}: Best val_loss={avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)
    lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("learning_rate", lr, epoch)
    print(
        f"Epoch {epoch}/{epochs} TL={avg_train_loss:.4f} TA={train_acc:.4f} VC={val_cer:.4f} VL={avg_val_loss:.4f} VA={val_acc:.4f} LR={lr:.6f}"
    )

    # --- Log Transform Visualization ---
    imgs, labs, _, lab_lens = next(iter(val_loader))
    imgs = imgs.to(device)
    if model.stn is not None:
        with torch.no_grad():
            corr = model.stn(imgs)
        writer.add_images("Transform/Input", imgs[:4], epoch)
        writer.add_images("Transform/Corrected", corr[:4], epoch)

    # --- Log Predictions ---
    preds, raws = ctc_greedy_decoder(model(imgs), alphabet)
    truths = []
    offset = 0
    for L in lab_lens.tolist():
        seq = labs[offset : offset + L].tolist()
        offset += L
        truths.append("".join([alphabet[i - 1] for i in seq]))
    log_samples(epoch, imgs, labs, lab_lens, preds, raws, truths)

writer.close()
