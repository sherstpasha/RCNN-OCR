import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import OCRDataset, build_alphabet, ResizeAndPad

import torchvision.transforms as T

transform = T.Compose(
    [
        ResizeAndPad(img_h=32, img_w=256),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# пути к данным
train_csv = r"C:\shared\orig_cyrillic\train.tsv"
train_dir = r"C:\shared\orig_cyrillic\train"
val_csv = r"C:\shared\orig_cyrillic\test.tsv"
val_dir = r"C:\shared\orig_cyrillic\test"
img_height = 32
img_max_width = 256

# алфавит
_, char2idx, idx2char = build_alphabet([train_csv, val_csv], case_insensitive=False)

# датасет и лоадер
train_ds = OCRDataset(
    train_csv, train_dir, char2idx, img_height, img_max_width, transform=transform
)
train_loader = DataLoader(
    train_ds, batch_size=4, shuffle=True, collate_fn=OCRDataset.collate_fn
)

# берём одну пачку
imgs, targets, lengths = next(iter(train_loader))

# показать первые картинки
plt.figure(figsize=(12, 4))
start = 0
for i in range(len(lengths)):
    text_len = lengths[i].item()
    label = "".join(idx2char[idx.item()] for idx in targets[start : start + text_len])
    start += text_len

    img = imgs[i].permute(1, 2, 0).numpy()
    plt.subplot(1, len(lengths), i + 1)
    plt.imshow((img + 1) / 2)
    plt.title(label)
    plt.axis("off")

plt.show()
