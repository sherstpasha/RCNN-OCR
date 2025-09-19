import optuna
import json
import os
from albumentations.pytorch import ToTensorV2
import albumentations as A
from dataset import ResizeAndPadA
from train2 import run_training  # твой run_training из текущего скрипта


# ============================================================
# Albumentations transforms
# ============================================================
def get_train_transform(params, img_h, img_w, from_trial=True):
    def suggest(name, default):
        if from_trial:
            if isinstance(default, tuple) and len(default) == 2:
                lo, hi = default
                if isinstance(lo, int) and isinstance(hi, int):
                    return params.suggest_int(name, lo, hi)
                elif isinstance(lo, float) or isinstance(hi, float):
                    return params.suggest_float(name, lo, hi)
            else:
                return params.suggest_categorical(name, default)
        else:
            if isinstance(default, tuple):
                return params.get(name, default[0])
            else:
                return params.get(name, default)

    return A.Compose(
        [
            ResizeAndPadA(img_h=img_h, img_w=img_w),
            A.ShiftScaleRotate(
                shift_limit=suggest("shift_limit", (0.0, 0.05)),
                scale_limit=suggest("scale_limit", (0.0, 0.1)),
                rotate_limit=suggest("rotate_limit", (0, 5)),
                border_mode=0,
                value=(255, 255, 255),
                p=suggest("p_ShiftScaleRotate", (0.0, 0.5)),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=suggest("brightness_limit", (0.1, 0.3)),
                contrast_limit=suggest("contrast_limit", (0.1, 0.3)),
                p=suggest("p_BrightnessContrast", (0.0, 0.5)),
            ),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


# ============================================================
# Optuna objective
# ============================================================
def objective(trial):
    IMG_H, IMG_W = 64, 256

    # === фиксированные гиперпараметры (из best_params.json) ===
    with open("best_params.json", "r", encoding="utf-8") as f:
        base_params = json.load(f)

    lr = base_params["lr"]
    batch_size = base_params["batch_size"]
    optimizer_name = base_params["optimizer"]
    scheduler_name = base_params["scheduler"]
    weight_decay = base_params["weight_decay"]
    momentum = base_params["momentum"]

    # === варьируем только аугментации ===
    train_transform = get_train_transform(
        trial, img_h=IMG_H, img_w=IMG_W, from_trial=True
    )

    metrics = run_training(
        train_csvs=[r"C:\shared\orig_cyrillic\train.tsv"],
        train_roots=[r"C:\shared\orig_cyrillic\train"],
        val_csvs=[r"C:\shared\orig_cyrillic\test.tsv"],
        val_roots=[r"C:\shared\orig_cyrillic\test"],
        charset_path="charset.txt",
        img_h=IMG_H,
        img_w=IMG_W,
        device="cuda",
        encoding="utf-8",
        max_len=25,
        batch_size=batch_size,
        epochs=15,  # для Optuna обычно делаем меньше
        lr=lr,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        weight_decay=weight_decay,
        momentum=momentum,
        train_transform=train_transform,
        exp_dir=None,
        resume_path=None,
    )

    return metrics["val_acc"]  # оптимизируем по accuracy


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    storage_url = "sqlite:///optuna_aug.db"

    from optuna.samplers import TPESampler

    study = optuna.create_study(
        study_name="ocr_aug_tuning",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        sampler=TPESampler(multivariate=True, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(objective, n_trials=30)

    print("Лучшие параметры:", study.best_params)
    print("Лучший результат (Accuracy):", study.best_value)

    # сохраняем только аугментации + фиксированные гиперы
    with open("best_params.json", "r", encoding="utf-8") as f:
        base_params = json.load(f)

    best_all = {**base_params, **study.best_params}

    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_all, f, indent=4, ensure_ascii=False)

    print("\n📊 Чтобы открыть дашборд, выполни в консоли:")
    print(f"optuna-dashboard {storage_url}\n")
