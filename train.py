import os
import csv
import random
import warnings
import textwrap
from tqdm import tqdm

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
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from dataset import OCRDataset
from model import CRNN
from utils import decode
from metrics import character_error_rate, word_error_rate, compute_accuracy
import torch.nn.functional as F
from vocab import SPECIAL_TOKENS
from collections import Counter as _Counter


def build_alphabet(paths, min_char_freq=30, encoding="utf-8"):
    ctr = _Counter()
    for p in paths:
        with open(p, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="	")
            for fn, lbl in reader:
                ctr.update(lbl)
    return "".join(sorted(ch for ch, freq in ctr.items() if freq >= min_char_freq))


def log_samples(
    epoch, imgs, lab_lens, preds, raws, truths, writer, n=10, tag="Examples"
):
    """
    Показывает первые n примеров из батча в TensorBoard.
    tag — название вкладки (например, 'Examples' или 'Train/Examples').
    """
    n = min(n, len(truths))
    fig, axs = plt.subplots(n, 1, figsize=(6, 2 * n))
    for i in range(n):
        img_np = imgs[i].cpu().squeeze(0).numpy()
        axs[i].imshow(img_np, cmap="gray")
        title = f"GT: {truths[i]} | Pred: {preds[i]}"
        wrapped = "\n".join(textwrap.wrap(title, width=40))
        raw_str = raws[i]
        axs[i].set_title(f"{wrapped}\nRaw: {raw_str}", fontsize=8)
        axs[i].axis("off")
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def get_worst_examples(
    imgs_list, preds_list, truths_list, raws_list, lens_list, top_k=20
):
    cer_list = [
        character_error_rate(gt, pred) for gt, pred in zip(truths_list, preds_list)
    ]
    # Каждая запись: (CER, img, pred, truth, raw, lens)
    sorted_data = sorted(
        zip(cer_list, imgs_list, preds_list, truths_list, raws_list, lens_list),
        key=lambda x: x[0],
        reverse=True,
    )
    top = sorted_data[:top_k]
    # Распаковываем топ по полям
    _, imgs, preds, truths, raws, lens = zip(*top)
    return imgs, preds, truths, raws, lens


# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# settings
experiment_name = "exp_1_archive1"
log_dir = os.path.join("runs", experiment_name)
checkpoint_dir = os.path.join("checkpoints", experiment_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# hyperparams
img_height, img_max_width = 60, 240
batch_size, epochs = 16 * 3, 60
learning_rate = 1e-3
decode_type = "greedy"

# data
train_csvs = [
    r"C:\data_19_04\Archive_19_04\data_archive\gt_train.txt",
    # r"C:\data_19_04\Archive_19_04\data_school\gt_train.txt",
]
train_roots = [
    r"C:\data_19_04\Archive_19_04\data_archive",
    # r"C:\data_19_04\Archive_19_04\\data_school",
]
val_csvs = [
    r"C:\data_19_04\Archive_19_04\data_archive\gt_test.txt",
    # r"C:\data_19_04\Archive_19_04\data_school\gt_test.txt",
]
val_roots = [
    r"C:\data_19_04\Archive_19_04\data_archive",
    # r"C:\data_19_04\Archive_19_04\\data_school",
]

alphabet = build_alphabet(train_csvs + val_csvs, min_char_freq=30, encoding="utf-8")
print(f"Using alphabet ({len(alphabet)}): {alphabet}")
num_ctc_classes = 1 + len(alphabet)  # blank + буквы
num_attn_classes = len(SPECIAL_TOKENS) + len(alphabet)  # pad, sos, eos, unk + буквы

# datasets + loaders
train_ds, val_ds = [], []
for csv_path, root in zip(train_csvs, train_roots):
    train_ds.append(
        OCRDataset(
            csv_path,
            root,
            alphabet=alphabet,
            img_height=img_height,
            img_max_width=img_max_width,
            augment=True,
        )
    )
for csv_path, root in zip(val_csvs, val_roots):
    val_ds.append(
        OCRDataset(
            csv_path,
            root,
            alphabet=alphabet,
            img_height=img_height,
            img_max_width=img_max_width,
            augment=False,
        )
    )
train_ds = ConcatDataset(train_ds)
val_ds = ConcatDataset(val_ds)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=OCRDataset.collate_fn_hybrid,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=OCRDataset.collate_fn_hybrid,
)

# model + losses + optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = CRNN(
    img_height,
    img_max_width,
    num_ctc_classes=num_ctc_classes,
    num_attn_classes=num_attn_classes,
    pretrained=False,
    transform="none",
    backbone="resnet",
).to(device)
criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
alpha = 0.3  # weight for CTC vs Attention
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# tensorboard
writer = SummaryWriter(log_dir=log_dir)

# metrics files
metrics_csv = os.path.join(checkpoint_dir, "metrics_history.csv")
with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        [
            "epoch",
            "train_loss",
            "train_acc",
            "train_cer",
            "train_wer",
            "val_loss",
            "val_acc",
            "val_cer",
            "val_wer",
            "lr",
        ]
    )
best_txt = os.path.join(checkpoint_dir, "best_metrics.txt")
with open(best_txt, "w", encoding="utf-8") as f:
    f.write(f"Experiment: {experiment_name}\n\n")
best_val_loss, best_val_acc = float("inf"), 0.0

global_step = 0
for epoch in range(1, epochs + 1):
    (
        imgs_all_train,
        preds_all_train,
        truths_all_train,
        raws_all_train,
        lens_all_train,
    ) = ([], [], [], [], [])
    imgs_all_val, preds_all_val, truths_all_val, raws_all_val, lens_all_val = (
        [],
        [],
        [],
        [],
        [],
    )
    # === Train ===
    model.train()

    ctc_train_loss = 0.0
    attn_train_loss = 0.0
    train_refs, train_hyps = [], []

    for imgs, labs_cat, inp_lens, lab_lens, dec_inputs, targets in tqdm(
        train_loader, desc=f"Train Epoch {epoch}"
    ):

        imgs, labs_cat = imgs.to(device), labs_cat.to(device)
        inp_lens, lab_lens = inp_lens.to(device), lab_lens.to(device)
        dec_inputs, targets = dec_inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        ctc_out, attn_out = model(imgs, decoder_inputs=dec_inputs)

        # 1) CTC loss
        T_len, B_len, _ = ctc_out.size()
        input_lengths = torch.full((B_len,), T_len, dtype=torch.long, device=device)
        ctc_loss = criterion_ctc(ctc_out, labs_cat, input_lengths, lab_lens)

        # 2) Attention loss
        B, L, V = attn_out.size()
        attn_logits = attn_out.view(-1, V)
        attn_loss = F.cross_entropy(attn_logits, targets.view(-1), ignore_index=0)

        # 3) Joint loss and backward
        loss = alpha * ctc_loss + (1 - alpha) * attn_loss
        loss.backward()
        optimizer.step()

        # Accumulate for epoch‐level logging
        ctc_train_loss += ctc_loss.item()
        attn_train_loss += attn_loss.item()

        # (опционально) логируем потери на каждом шаге
        writer.add_scalar("train/CTC_loss_iter", ctc_loss.item(), global_step)
        writer.add_scalar("train/ATTN_loss_iter", attn_loss.item(), global_step)
        writer.add_scalar("train/loss_iter", loss.item(), global_step)
        global_step += 1

        # decode CTC и собираем refs/hyps как раньше …
        preds, raws = decode(ctc_out, alphabet, method=decode_type, beam_width=5)
        offset = 0
        for length in lab_lens.tolist():
            seq = labs_cat[offset : offset + length].tolist()
            offset += length
            train_refs.append("".join(alphabet[i - 1] for i in seq if i > 0))
        train_hyps.extend(preds)

        # Накопим всё для худших примеров
        imgs_all_train = [] if epoch == 1 else imgs_all_train
        truths_all_train = [] if epoch == 1 else truths_all_train
        preds_all_train = [] if epoch == 1 else preds_all_train
        raws_all_train = [] if epoch == 1 else raws_all_train
        lens_all_train = [] if epoch == 1 else lens_all_train

        imgs_all_train.extend(imgs.cpu())
        truths_all_train.extend(train_refs[-len(imgs) :])
        preds_all_train.extend(train_hyps[-len(imgs) :])
        raws_all_train.extend(raws)
        lens_all_train.extend(lab_lens.cpu())

    # === Epoch‐level logging ===
    avg_ctc_train = ctc_train_loss / len(train_loader)
    avg_attn_train = attn_train_loss / len(train_loader)
    avg_train_loss = alpha * avg_ctc_train + (1 - alpha) * avg_attn_train

    # Лосс
    writer.add_scalar("train/CTC_loss", avg_ctc_train, epoch)
    writer.add_scalar("train/ATTN_loss", avg_attn_train, epoch)
    writer.add_scalar("train/loss", avg_train_loss, epoch)

    # ------ Новое: вычисляем метрики по батчам ------
    # train_refs и train_hyps собирались в цикле выше
    train_acc = compute_accuracy(train_refs, train_hyps)
    train_cer = sum(
        character_error_rate(r, h) for r, h in zip(train_refs, train_hyps)
    ) / len(train_refs)
    train_wer = sum(
        word_error_rate(r, h) for r, h in zip(train_refs, train_hyps)
    ) / len(train_refs)

    # Логируем метрики
    writer.add_scalar("train/accuracy", train_acc, epoch)
    writer.add_scalar("train/CER", train_cer, epoch)
    writer.add_scalar("train/WER", train_wer, epoch)

    # Log STN outputs for 5 random train examples
    if model.stn is not None:
        imgs_tr, *_ = next(iter(train_loader))
        imgs_tr = imgs_tr.to(device)
        with torch.no_grad():
            stn_tr_out = model.stn(imgs_tr)
        N = min(5, stn_tr_out.size(0))
        grid_tr = vutils.make_grid(
            stn_tr_out[:N], nrow=N, normalize=True, scale_each=True
        )
        writer.add_image("STN/Train_Examples", grid_tr, epoch)

    # === Validation ===
    model.eval()

    ctc_val_loss = 0.0
    attn_val_loss = 0.0
    val_refs, val_hyps = [], []

    with torch.no_grad():
        for imgs, labs_cat, _, lab_lens, dec_inputs, targets in tqdm(
            val_loader, desc=f"Val Epoch {epoch}"
        ):

            imgs, labs_cat = imgs.to(device), labs_cat.to(device)
            lab_lens = lab_lens.to(device)
            dec_inputs, targets = dec_inputs.to(device), targets.to(device)

            # прогоняем через модель
            ctc_out, attn_out = model(imgs, decoder_inputs=dec_inputs)

            # пересчитываем длину временной оси по факту
            T_len, B_len, _ = ctc_out.size()
            input_lengths = torch.full((B_len,), T_len, dtype=torch.long, device=device)

            # теперь CTC-loss с корректными input_lengths
            ctc_val_loss += criterion_ctc(
                ctc_out, labs_cat, input_lengths, lab_lens
            ).item()

            # 2) Attention loss
            B, L, V = attn_out.size()
            attn_logits = attn_out.view(-1, V)
            attn_val_loss += F.cross_entropy(
                attn_logits, targets.view(-1), ignore_index=0
            ).item()

            # 3) CTC-decoding для расчёта метрик
            preds, raws = decode(ctc_out, alphabet, method=decode_type, beam_width=5)
            offset = 0
            for length in lab_lens.tolist():
                seq = labs_cat[offset : offset + length].tolist()
                offset += length
                val_refs.append("".join(alphabet[i - 1] for i in seq if i > 0))
            val_hyps.extend(preds)

            # === Сбор данных для логирования худших примеров (валидация)
            imgs_all_val.extend(imgs.cpu())
            offset = 0
            for length in lab_lens.tolist():
                seq = labs_cat[offset : offset + length].tolist()
                offset += length
                truths_all_val.append("".join(alphabet[i - 1] for i in seq if i > 0))
            preds_all_val.extend(preds)
            raws_all_val.extend(raws)
            lens_all_val.extend(lab_lens.cpu())

    # Усреднённые лоссы
    avg_ctc_val = ctc_val_loss / len(val_loader)
    avg_attn_val = attn_val_loss / len(val_loader)
    avg_val_loss = alpha * avg_ctc_val + (1 - alpha) * avg_attn_val

    # Считаем метрики
    val_acc = compute_accuracy(val_refs, val_hyps)
    val_cer = sum(character_error_rate(r, h) for r, h in zip(val_refs, val_hyps)) / len(
        val_refs
    )
    val_wer = sum(word_error_rate(r, h) for r, h in zip(val_refs, val_hyps)) / len(
        val_refs
    )

    # Логируем в TensorBoard
    writer.add_scalar("val/CTC_loss", avg_ctc_val, epoch)
    writer.add_scalar("val/ATTN_loss", avg_attn_val, epoch)
    writer.add_scalar("val/loss", avg_val_loss, epoch)
    writer.add_scalar("val/accuracy", val_acc, epoch)
    writer.add_scalar("val/CER", val_cer, epoch)
    writer.add_scalar("val/WER", val_wer, epoch)

    # Log STN outputs for 5 random val examples
    if model.stn is not None:
        imgs_val, *_ = next(iter(val_loader))
        imgs_val = imgs_val.to(device)
        with torch.no_grad():
            stn_val_out = model.stn(imgs_val)
        N = min(5, stn_val_out.size(0))
        grid_val = vutils.make_grid(
            stn_val_out[:N], nrow=N, normalize=True, scale_each=True
        )
        writer.add_image("STN/Val_Examples", grid_val, epoch)

    # --- Record metrics & checkpoints ---
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                epoch,
                f"{avg_train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{train_cer:.6f}",
                f"{train_wer:.6f}",
                f"{avg_val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{val_cer:.6f}",
                f"{val_wer:.6f}",
                f"{optimizer.param_groups[0]['lr']:.6f}",
            ]
        )
    # Checkpoint by val loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_by_loss.pth"))
        with open(best_txt, "a", encoding="utf-8") as f:
            f.write(f"[Epoch {epoch}] best val_loss = {best_val_loss:.6f}\n")
    # Checkpoint by val acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_by_acc.pth"))
        with open(best_txt, "a", encoding="utf-8") as f:
            f.write(f"[Epoch {epoch}] best val_acc  = {best_val_acc:.6f}\n")

    scheduler.step(avg_val_loss)
    print(
        f"Epoch {epoch}/{epochs} | "
        f"TrainLoss={avg_train_loss:.4f} Acc={train_acc:.4f} CER={train_cer:.4f} WER={train_wer:.4f} | "
        f"ValLoss={avg_val_loss:.4f} Acc={val_acc:.4f} CER={val_cer:.4f} WER={val_wer:.4f} | "
        f"LR={optimizer.param_groups[0]['lr']:.6f}"
    )

    # --- Log sample predictions ---
    # === Log WORST VALIDATION ===
    worst_imgs, worst_preds, worst_truths, worst_raws, worst_lens = get_worst_examples(
        imgs_all_val,
        preds_all_val,
        truths_all_val,
        raws_all_val,
        lens_all_val,
        top_k=20,
    )
    log_samples(
        epoch,
        torch.stack(worst_imgs),
        worst_lens,
        worst_preds,
        worst_raws,
        worst_truths,
        writer,
        n=20,
        tag="Val/Worst",
    )

    # === Log WORST TRAINING ===
    worst_imgs, worst_preds, worst_truths, worst_raws, worst_lens = get_worst_examples(
        imgs_all_train,
        preds_all_train,
        truths_all_train,
        raws_all_train,
        lens_all_train,
        top_k=20,
    )
    log_samples(
        epoch,
        torch.stack(worst_imgs),
        worst_lens,
        worst_preds,
        worst_raws,
        worst_truths,
        writer,
        n=20,
        tag="Train/Worst",
    )
    # === Log RANDOM TRAINING EXAMPLES from full epoch ===
    idxs = random.sample(range(len(preds_all_train)), min(10, len(preds_all_train)))
    rand_imgs = [imgs_all_train[i] for i in idxs]
    rand_preds = [preds_all_train[i] for i in idxs]
    rand_truths = [truths_all_train[i] for i in idxs]
    rand_raws = [raws_all_train[i] for i in idxs]
    rand_lens = [lens_all_train[i] for i in idxs]

    log_samples(
        epoch,
        torch.stack(rand_imgs),
        rand_lens,
        rand_preds,
        rand_raws,
        rand_truths,
        writer,
        n=10,
        tag="Train/Examples",
    )
writer.close()
