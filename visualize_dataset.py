# visualize_dataset.py

import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset import OCRDataset


def decode_labels(
    labs_cat: torch.Tensor, lab_lens: torch.Tensor, char2idx: dict
) -> list[str]:
    """Разбить единый тензор labs_cat по длинам lab_lens и перевести в строки."""
    idx2char = {v: k for k, v in char2idx.items()}
    strs = []
    offset = 0
    for L in lab_lens.tolist():
        seq = labs_cat[offset : offset + L].tolist()
        offset += L
        s = "".join(idx2char[i] for i in seq if i > 0)
        strs.append(s)
    return strs


def visualize_batches(
    csv_paths: list[str],
    img_roots: list[str],
    img_height: int,
    img_max_width: int,
    batch_size: int = 4,
    n_batches: int = 3,
    min_char_freq: int = 1,
    augment: bool = False,
):
    # 1) Построить алфавит по всем CSV
    alphabet = OCRDataset.build_alphabet(csv_paths, min_char_freq=min_char_freq)
    print(f"→ Alphabet ({len(alphabet)}): {alphabet}")

    # 2) Собрать ConcatDataset
    datasets = []
    for csv_path, root in zip(csv_paths, img_roots):
        ds = OCRDataset(
            csv_path=csv_path,
            images_dir=root,
            alphabet=alphabet,
            img_height=img_height,
            img_max_width=img_max_width,
            min_char_freq=min_char_freq,
            augment=augment,
        )
        datasets.append(ds)
    ds = ConcatDataset(datasets)

    # 3) DataLoader
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=OCRDataset.collate_fn,
    )

    # 4) Визуализация первых n_batches батчей
    for batch_idx, (imgs, labs_cat, inp_lens, lab_lens) in enumerate(loader):
        if batch_idx >= n_batches:
            break

        # декодируем GT
        char2idx = datasets[0].char2idx
        truths = decode_labels(labs_cat, lab_lens, char2idx)

        # рисуем батч
        B = imgs.size(0)
        fig, axes = plt.subplots(1, B, figsize=(B * 3, 3))
        if B == 1:
            axes = [axes]
        for i in range(B):
            ax = axes[i]
            img_np = imgs[i].cpu().squeeze(0).numpy()
            ax.imshow(img_np, cmap="gray")
            ax.set_title(truths[i], fontsize=10)
            ax.axis("off")
        plt.suptitle(f"Batch {batch_idx+1}/{n_batches}", fontsize=12)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # === 3) Hyperparameters & paths ===
    img_height, img_max_width = 60, 240

    train_csvs = [
        r"C:\data_cyrillic\gt_train.txt",
    ]
    train_image_roots = [
        r"C:\data_cyrillic\train",
    ]

    val_csvs = [
        r"C:\data_cyrillic\gt_test.txt",
    ]
    val_image_roots = [
        r"C:\data_cyrillic\test",
    ]

    # Запустить визуализацию TRAIN
    visualize_batches(
        csv_paths=train_csvs,
        img_roots=train_image_roots,
        img_height=img_height,
        img_max_width=img_max_width,
        batch_size=16,
        n_batches=10,
        min_char_freq=1,
        augment=True,
    )

    # (по желанию) визуализация VAL
    # visualize_batches(
    #     csv_paths=val_csvs,
    #     img_roots=val_image_roots,
    #     img_height=img_height,
    #     img_max_width=img_max_width,
    #     batch_size=16,
    #     n_batches=3,
    #     min_char_freq=1,
    # )
