import os
import csv
from collections import Counter
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
import albumentations as A
import cv2


def build_alphabet(paths, min_char_freq=1, encoding="utf-8", case_insensitive=False):
    ctr = Counter()
    for p in paths:
        with open(p, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            for fname, lbl in reader:
                if case_insensitive:
                    lbl = lbl.lower()
                ctr.update(lbl)
    symbols = "".join(sorted(ch for ch, freq in ctr.items() if freq >= min_char_freq))
    char2idx = {c: i + 1 for i, c in enumerate(symbols)}  # 0 = blank
    idx2char = {i: c for c, i in char2idx.items()}
    idx2char[0] = "␣"
    return symbols, char2idx, idx2char


class ResizeAndPadA(A.ImageOnlyTransform):
    def __init__(self, img_h=32, img_w=256, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.img_h = img_h
        self.img_w = img_w

    def apply(self, img, **params):
        h, w = img.shape[:2]
        new_w = min(self.img_w, max(1, int(w * self.img_h / h)))
        img_resized = cv2.resize(
            img, (new_w, self.img_h), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.ones((self.img_h, self.img_w, 3), dtype=img.dtype) * 255
        canvas[:, :new_w] = img_resized
        return canvas


class OCRDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        char2idx: dict,
        img_height: int = 32,
        img_max_width: int = 128,
        encoding: str = "utf-8",
        transform: Optional[callable] = None,
    ):
        self.images_dir = images_dir
        self.img_h = img_height
        self.img_w = img_max_width
        self.char2idx = char2idx
        self.transform = transform

        self.samples: List[Tuple[str, str]] = []
        with open(csv_path, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            for fname, label in reader:
                if all(c in self.char2idx for c in label):
                    self.samples.append((fname, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")

        if self.transform:
            augmented = self.transform(image=np.array(img))
            tensor = augmented["image"]
        else:
            tensor = (
                torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1)
                / 255.0
            )

        labels = torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)
        return tensor, labels

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs)
        targets = torch.cat(labels)
        target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
        return imgs, targets, target_lengths
