# dataset.py

import os
import csv
import random
from collections import Counter
from typing import List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from vocab import build_vocab, SPECIAL_TOKENS


class AddGaussianNoise:
    """Аугментация: гауссов шум с динамическим std."""

    def __init__(self, mean: float = 0.0, std_range: tuple = (0.0, 0.02)):
        self.mean = mean
        self.std_min, self.std_max = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(self.std_min, self.std_max)
        return tensor + torch.randn_like(tensor) * std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std_range=({self.std_min},{self.std_max}))"


class SaltPepperNoise:
    """Аугментация: 'соль-перец' — случайные белые/чёрные пиксели."""

    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x in [–1..1], но мы работаем с padding=1→белый, padding=–1→чёрный
        mask = torch.rand_like(x)
        x = torch.where(mask < self.p / 2, -1.0, x)  # "чёрные"
        x = torch.where(mask > 1 - self.p / 2, +1.0, x)  # "белые"
        return x


class RandomInvert:
    """Аугментация: инвертировать полярность (текст/фон)."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return -x if random.random() < self.p else x


class RandomRescale:
    """Аугментация: случайный down–upsampling, имитируем низкое DPI."""

    def __init__(self, scale_range=(0.8, 1.0), p: float = 0.2):
        self.scale_min, self.scale_max = scale_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            _, H, W = x.shape
            s = random.uniform(self.scale_min, self.scale_max)
            h, w = max(1, int(H * s)), max(1, int(W * s))
            x = TF.resize(x, (h, w), interpolation=TF.InterpolationMode.BILINEAR)
            x = TF.resize(x, (H, W), interpolation=TF.InterpolationMode.BILINEAR)
        return x


class OCRDataset(Dataset):
    """
    Датасет для OCR c CTC и аугментациями:
      1) Aspect-ratio resize по высоте + right-pad белым
      2) Random zoom-crop (имитация плохой детекции)
      3) PIL-аугментации: affine, perspective, color jitter
      4) Tensor-аугментации: rescale, blur, шум, salt&pepper, invert, random-erase
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        alphabet: Optional[str] = None,
        img_height: int = 32,
        img_max_width: int = 128,
        min_char_freq: int = 1,
        encoding: str = "utf-8",
        augment: bool = False,
        zoom_prob: float = 0.2,
        zoom_ratio: float = 0.2,
        use_seq2seq: bool = False,
    ):
        self.use_seq2seq = use_seq2seq

        # 1) читаем CSV
        samples = []
        with open(csv_path, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            for fname, label in reader:
                samples.append((label, fname))

        # 2) строим or принимаем алфавит
        if alphabet is None:
            counter = Counter(ch for lbl, _ in samples for ch in lbl)
            self.alphabet = "".join(
                sorted(ch for ch, freq in counter.items() if freq >= min_char_freq)
            )
        else:
            self.alphabet = alphabet

        # 3) map char→idx (1..N), 0 – blank/unk
        self.char2idx = {c: i + 1 for i, c in enumerate(self.alphabet)}

        # 4) фильтруем все с непонятными символами
        self.samples = [
            (lbl, fn) for lbl, fn in samples if all(c in self.char2idx for c in lbl)
        ]

        # сохраним параметры
        self.images_dir = images_dir
        self.img_h = img_height
        self.img_w = img_max_width
        self.augment = augment
        self.zoom_prob = zoom_prob
        self.zoom_ratio = zoom_ratio

        # 5) PIL-аугментации
        self.aug_pil = T.Compose(
            [
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=3,
                            translate=(0.02, 0.02),
                            scale=(0.9, 1.1),
                            shear=2,
                            fill=255,
                        )
                    ],
                    p=0.5,
                ),
                T.RandomApply(
                    [T.RandomPerspective(distortion_scale=0.1, p=1.0, fill=255)], p=0.2
                ),
                T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
            ]
        )

        # 6) Tensor-аугментации
        self.aug_tensor = T.Compose(
            [
                RandomRescale(scale_range=(0.8, 1.0), p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))], p=0.2),
                AddGaussianNoise(mean=0.0, std_range=(0.0, 0.02)),
                SaltPepperNoise(p=0.05),
                RandomInvert(p=0.1),
                T.RandomErasing(
                    p=0.05, scale=(0.001, 0.02), ratio=(0.5, 2.0), value=1.0
                ),
            ]
        )

        self.vocab, self.inv_vocab = build_vocab(self.alphabet)
        self.char2idx = {c: i + 1 for i, c in enumerate(self.alphabet)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, fname = self.samples[idx]
        path = os.path.join(self.images_dir, fname)
        img = Image.open(path).convert("L")

        # — 1) Aspect-ratio resize по высоте
        w, h = img.size
        new_w = min(self.img_w, max(1, int(w * self.img_h / h)))
        img = img.resize((new_w, self.img_h), Image.BILINEAR)

        # — 2) Random zoom-crop
        if self.augment and random.random() < self.zoom_prob:
            max_cut = int(new_w * self.zoom_ratio)
            cut = random.randint(0, max_cut)
            left = cut // 2
            right = cut - left
            img = img.crop((left, 0, new_w - right, self.img_h))
            new_w = img.size[0]

        # — 3) PIL-аугментации
        if self.augment:
            img = self.aug_pil(img)

        # — 4) to_tensor + normalize → [–1..+1]
        tensor = TF.to_tensor(img)  # [0..1]
        tensor = TF.normalize(tensor, [0.5], [0.5])  # [–1..+1]

        # — 5) right-pad белым до img_w
        pad_w = self.img_w - new_w
        if pad_w > 0:
            # TF.pad принимает (left, top, right, bottom)
            tensor = TF.pad(tensor, (0, 0, pad_w, 0), fill=1.0)

        # — 6) Tensor-аугментации
        if self.augment:
            tensor = self.aug_tensor(tensor)

        # — 7) encode label → indices
        lab = torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)

        if self.use_seq2seq:
            tokens = [self.vocab["<sos>"]] + [self.vocab[c] for c in label] + [self.vocab["<eos>"]]
            lab = torch.tensor(tokens, dtype=torch.long)
        else:
            lab = torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)

        return tensor, lab

    @staticmethod
    def collate_fn(batch):
        imgs, labs = zip(*batch)
        imgs = torch.stack(imgs)
        lab_lens = torch.tensor([len(l) for l in labs], dtype=torch.long)
        labs_cat = torch.cat(labs)
        B, _, H, W = imgs.shape
        inp_lens = torch.full((B,), W // 4, dtype=torch.long)
        return imgs, labs_cat, inp_lens, lab_lens

    @staticmethod
    def collate_fn_seq2seq(batch):
        from torch.nn.utils.rnn import pad_sequence
        imgs, labs = zip(*batch)
        imgs = torch.stack(imgs)
        max_len = max(len(l) for l in labs)

        # decoder_input = [<sos>, ..., last_token]
        # target = [..., <eos>]
        decoder_inputs = [l[:-1] for l in labs]
        targets = [l[1:] for l in labs]

        decoder_inputs_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=0)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

        return imgs, decoder_inputs_padded, targets_padded
    
    @staticmethod
    def build_alphabet(
        csv_paths: List[str], min_char_freq: int = 1, encoding: str = "utf-8"
    ) -> str:
        counter = Counter()
        for p in csv_paths:
            with open(p, newline="", encoding=encoding) as f:
                reader = csv.reader(f, delimiter="\t")
                for _, lbl in reader:
                    counter.update(lbl)
        return "".join(
            sorted(ch for ch, freq in counter.items() if freq >= min_char_freq)
        )
