import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import (
    OCRDatasetAttn,
    load_charset,
    ResizeAndPadA,
)

# --- пути ---
train_csv = r"C:\shared\orig_cyrillic\train.tsv"
train_dir = r"C:\shared\orig_cyrillic\train"

# --- размеры ---
img_height = 32
img_max_width = 256
max_len = 10

# --- алфавит ---
itos, stoi = load_charset("charset.txt")

# --- трансформации с аугментациями ---
transform = A.Compose(
    [
        ResizeAndPadA(img_h=img_height, img_w=img_max_width),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=5,
            border_mode=0,
            value=(255, 255, 255),
            p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.InvertImg(p=0.05),  # маленький шанс инверсии
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)

# --- датасет и лоадер ---
train_ds = OCRDatasetAttn(
    csv_path=train_csv,
    images_dir=train_dir,
    stoi=stoi,
    img_height=img_height,
    img_max_width=img_max_width,
    transform=transform,
)

collate_attn = OCRDatasetAttn.make_collate_attn(stoi, max_len=max_len, drop_blank=True)

train_loader = DataLoader(
    train_ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_attn
)


# --- helpers ---
def ids_to_tokens_str(ids_row, itos):
    return " ".join(
        itos[int(t)] if 0 <= int(t) < len(itos) else f"<UNK:{int(t)}>" for t in ids_row
    )


# берём батч
imgs, text_in, target_y, lengths = next(iter(train_loader))
B = imgs.size(0)

# рисуем
plt.figure(figsize=(14, 6))
for i in range(B):
    img = imgs[i].permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)  # денормализация

    ti_str = ids_to_tokens_str(text_in[i].tolist(), itos)
    ty_str = ids_to_tokens_str(target_y[i].tolist(), itos)
    L = int(lengths[i])

    plt.subplot(2, B, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"len={L}")

    plt.subplot(2, B, B + i + 1)
    plt.text(0.0, 0.8, f"text_in:\n{ti_str}", fontsize=9, va="top", family="monospace")
    plt.text(0.0, 0.3, f"target_y:\n{ty_str}", fontsize=9, va="top", family="monospace")
    plt.axis("off")

plt.tight_layout()
plt.show()
