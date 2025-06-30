# dataset.py

import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OCRDataset(Dataset):
    def __init__(
        self, csv_path, images_dir, alphabet, img_height=32, img_max_width=128
    ):
        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            for label, fname in reader:
                self.samples.append((label, fname))
        self.images_dir = images_dir
        self.alphabet = alphabet
        self.char2idx = {c: i + 1 for i, c in enumerate(alphabet)}
        self.img_h = img_height
        self.img_w = img_max_width
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(1),
                transforms.Resize((img_height, img_max_width)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        label, fname = self.samples[i]
        img = Image.open(os.path.join(self.images_dir, fname)).convert("L")
        img = self.transform(img)
        lab = torch.tensor([self.char2idx.get(c, 0) for c in label], dtype=torch.long)
        return img, lab

    @staticmethod
    def collate_fn(batch):
        imgs, labs = zip(*batch)
        imgs = torch.stack(imgs)
        lab_lens = torch.tensor([len(l) for l in labs], dtype=torch.long)
        labs_cat = torch.cat(labs)
        B, _, H, W = imgs.shape
        inp_lens = torch.full((B,), W // 4, dtype=torch.long)
        return imgs, labs_cat, inp_lens, lab_lens
