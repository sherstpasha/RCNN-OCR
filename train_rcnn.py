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
import optuna
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import OCRDataset, build_alphabet, ResizeAndPadA
from model import RCNN
from utils import decode
from metrics import compute_accuracy


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# Albumentations transforms
# ============================================================
def get_train_transform(params, img_h, img_w, from_trial=True):
    def suggest(name, default):
        if from_trial:
            # params = trial
            if isinstance(default, tuple) and len(default) == 2:
                lo, hi = default
                if isinstance(lo, int) and isinstance(hi, int):
                    return params.suggest_int(name, lo, hi)
                elif isinstance(lo, float) or isinstance(hi, float):
                    return params.suggest_uniform(name, lo, hi)
            else:
                return params.suggest_categorical(name, default)
        else:
            # params = dict
            return params.get(
                name, default[0] if isinstance(default, tuple) else default
            )

    return A.Compose(
        [
            ResizeAndPadA(img_h=img_h, img_w=img_w),
            A.ShiftScaleRotate(
                shift_limit=suggest("shift_limit", (0.0, 0.05)),
                scale_limit=suggest("scale_limit", (0.0, 0.2)),
                rotate_limit=suggest("rotate_limit", (0, 10)),
                border_mode=0,
                value=(255, 255, 255),
                p=suggest("p_ShiftScaleRotate", (0.0, 0.7)),
            ),
            A.OpticalDistortion(
                distort_limit=suggest("optical_distort_limit", (0.0, 0.05)),
                shift_limit=suggest("optical_shift_limit", (0.0, 0.05)),
                p=suggest("p_OpticalDistortion", (0.0, 0.5)),
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=suggest("grid_distort_limit", (0.0, 0.05)),
                p=suggest("p_GridDistortion", (0.0, 0.5)),
            ),
            A.MotionBlur(
                blur_limit=params.suggest_int(
                    "motion_blur_limit", 3, 7, step=2
                ),  # 3, 5, 7
                p=suggest("p_MotionBlur", (0.0, 0.5)),
            ),
            A.GaussNoise(
                var_limit=(
                    suggest("noise_var_min", (5, 20)),
                    suggest("noise_var_max", (30, 80)),
                ),
                p=suggest("p_GaussNoise", (0.0, 0.5)),
            ),
            A.ImageCompression(
                quality_lower=suggest("jpeg_qmin", (30, 60)),
                quality_upper=suggest("jpeg_qmax", (70, 100)),
                p=suggest("p_ImageCompression", (0.0, 0.5)),
            ),
            A.CoarseDropout(
                max_holes=suggest("dropout_holes", (1, 5)),
                max_height=suggest("dropout_h", (5, 20)),
                max_width=suggest("dropout_w", (5, 20)),
                p=suggest("p_CoarseDropout", (0.0, 0.5)),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=suggest("brightness_limit", (0.1, 0.5)),
                contrast_limit=suggest("contrast_limit", (0.1, 0.5)),
                p=suggest("p_BrightnessContrast", (0.0, 0.5)),
            ),
            A.ToGray(p=suggest("p_ToGray", (0.0, 0.3))),
            A.ElasticTransform(
                alpha=suggest("elastic_alpha", (0.5, 2.0)),
                sigma=suggest("elastic_sigma", (20, 80)),
                alpha_affine=suggest("elastic_affine", (5, 15)),
                p=suggest("p_ElasticTransform", (0.0, 0.3)),
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


# ============================================================
# Training
# ============================================================
def run_training(
    train_csvs,
    train_roots,
    val_csvs,
    val_roots,
    symbols: str = None,
    img_h=64,
    img_w=256,
    num_classes=None,
    batch_size=32,
    epochs=20,
    lr=1e-3,
    optimizer_name="Adam",
    scheduler_name="ReduceLROnPlateau",
    weight_decay=0.0,
    momentum=0.9,
    device="cuda",
    min_char_freq=3,
    encoding="utf-8",
    train_transform=None,
):
    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # alphabet
    paths = train_csvs + val_csvs
    if symbols is None:
        symbols, char2idx, idx2char = build_alphabet(
            paths, min_char_freq, encoding=encoding, case_insensitive=False
        )
    else:
        char2idx = {c: i + 1 for i, c in enumerate(symbols)}  # 0 = blank
        idx2char = {i: c for c, i in char2idx.items()}
        idx2char[0] = "␣"

    if num_classes is None:
        num_classes = 1 + len(char2idx)

    # experiment dirs
    exp_idx = 1
    while os.path.exists(f"exp{exp_idx}"):
        exp_idx += 1
    os.makedirs(f"exp{exp_idx}")
    log_dir = os.path.join(f"exp{exp_idx}", "logs")
    writer = SummaryWriter(log_dir=log_dir)

    best_loss_path = os.path.join(f"exp{exp_idx}", "best_loss.pth")
    best_acc_path = os.path.join(f"exp{exp_idx}", "best_acc.pth")

    # model
    model = RCNN(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

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
        OCRDataset(
            c, r, char2idx, img_h, img_w, transform=train_transform, encoding=encoding
        )
        for c, r in zip(train_csvs, train_roots)
    ]
    val_sets = [
        OCRDataset(
            c, r, char2idx, img_h, img_w, transform=val_transform, encoding=encoding
        )
        for c, r in zip(val_csvs, val_roots)
    ]

    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=OCRDataset.collate_fn,
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=OCRDataset.collate_fn,
    )

    best_val_loss, best_val_acc = float("inf"), -1.0
    global_step = 0

    for epoch in range(1, epochs + 1):
        # --- training ---
        model.train()
        total_train_loss = 0.0
        for imgs, targets, target_lengths in train_loader:
            imgs, targets, target_lengths = (
                imgs.to(device),
                targets.to(device),
                target_lengths.to(device),
            )
            optimizer.zero_grad()
            with amp.autocast():
                logits = model(imgs)
                T_, B, C = logits.shape
                input_lengths = torch.full((B,), T_, dtype=torch.long, device=device)
                loss = criterion(logits.float(), targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)

        # --- validation ---
        model.eval()
        total_val_loss, refs, hyps = 0.0, [], []
        with torch.no_grad():
            for imgs, targets, target_lengths in val_loader:
                imgs, targets, target_lengths = (
                    imgs.to(device),
                    targets.to(device),
                    target_lengths.to(device),
                )
                with amp.autocast():
                    logits = model(imgs)
                    T_, B, C = logits.shape
                    input_lengths = torch.full(
                        (B,), T_, dtype=torch.long, device=device
                    )
                    loss = criterion(
                        logits.float(), targets, input_lengths, target_lengths
                    )
                total_val_loss += loss.item()

                preds, _ = decode(logits, symbols, method="greedy")
                offset = 0
                for L in target_lengths.tolist():
                    seq = targets[offset : offset + L].tolist()
                    offset += L
                    refs.append("".join(symbols[c - 1] for c in seq if c > 0))
                hyps.extend(preds)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = compute_accuracy(refs, hyps)

        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_loss_path)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_acc_path)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    writer.close()
    return {"val_acc": best_val_acc, "val_loss": best_val_loss}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    mode = "optuna"

    # === фиксированные параметры проекта ===
    IMG_H = 64
    IMG_W = 256

    base_config = dict(
        train_csvs=[r"C:\shared\orig_cyrillic\train.tsv"],
        train_roots=[r"C:\shared\orig_cyrillic\train"],
        val_csvs=[r"C:\shared\orig_cyrillic\test.tsv"],
        val_roots=[r"C:\shared\orig_cyrillic\test"],
        symbols=None,
        img_h=IMG_H,
        img_w=IMG_W,
        num_classes=None,
        device="cuda",
        min_char_freq=30,
        encoding="utf-8",
    )

    storage_url = "sqlite:///optuna_ocr.db"

    if mode == "optuna":

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            optimizer_name = trial.suggest_categorical(
                "optimizer", ["Adam", "AdamW", "SGD"]
            )
            scheduler_name = trial.suggest_categorical(
                "scheduler", ["ReduceLROnPlateau", "CosineAnnealingLR", "None"]
            )
            weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
            momentum = trial.suggest_float("momentum", 0.7, 0.99)

            train_transform = get_train_transform(
                trial, img_h=IMG_H, img_w=IMG_W, from_trial=True
            )

            metrics = run_training(
                **base_config,
                batch_size=batch_size,
                epochs=5,
                lr=lr,
                optimizer_name=optimizer_name,
                scheduler_name=scheduler_name,
                weight_decay=weight_decay,
                momentum=momentum,
                train_transform=train_transform,
            )
            return -metrics["val_loss"]

        study = optuna.create_study(
            study_name="ocr_tuning",
            direction="maximize",
            storage=storage_url,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=15)

        print("Лучшие параметры:", study.best_params)
        print("Лучший результат:", study.best_value)

        with open("best_params.json", "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=4, ensure_ascii=False)

        print("\n📊 Чтобы открыть дашборд, выполни в консоли:")
        print(f"optuna-dashboard {storage_url}\n")

    elif mode == "train":
        with open("best_params.json", "r", encoding="utf-8") as f:
            best_params = json.load(f)

        train_transform = get_train_transform(
            best_params, img_h=IMG_H, img_w=IMG_W, from_trial=False
        )

        run_training(
            **base_config,
            batch_size=best_params.get("batch_size", 128),
            epochs=50,
            lr=best_params.get("lr", 1e-4),
            optimizer_name=best_params.get("optimizer", "Adam"),
            scheduler_name=best_params.get("scheduler", "ReduceLROnPlateau"),
            weight_decay=best_params.get("weight_decay", 0.0),
            momentum=best_params.get("momentum", 0.9),
            train_transform=train_transform,
        )
