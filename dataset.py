import os
import csv
import random
from collections import Counter
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from vocab import build_vocab, SPECIAL_TOKENS


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std_range: tuple = (0.0, 0.02)):
        self.mean = mean
        self.std_min, self.std_max = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(self.std_min, self.std_max)
        return tensor + torch.randn_like(tensor) * std + self.mean


class SaltPepperNoise:
    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(x)
        x = torch.where(mask < self.p / 2, -1.0, x)
        x = torch.where(mask > 1 - self.p / 2, +1.0, x)
        return x


class RandomInvert:
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return -x if random.random() < self.p else x


class RandomRescale:
    def __init__(self, scale_range=(0.8, 1.0), p: float = 0.2):
        self.scale_min, self.scale_max = scale_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            _, H, W = x.shape
            s = random.uniform(self.scale_min, self.scale_max)
            h, w = max(1, int(H * s)), max(1, int(W * s))
            x = TF.resize(
                x, (h, w), interpolation=TF.InterpolationMode.BILINEAR, antialias=True
            )
            x = TF.resize(
                x, (H, W), interpolation=TF.InterpolationMode.BILINEAR, antialias=True
            )
        return x


class OCRDataset(Dataset):
    """
    Датасет для гибридного обучения CTC + Attention.
    __getitem__ возвращает:
      - img_tensor: torch.FloatTensor [1,H,W]
      - ctc_lab:    torch.LongTensor [L]      (для CTC)
      - seq_lab:    torch.LongTensor [L+2]    (для Attention: <sos>…<eos>)
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
        use_seq2seq: bool = False,
    ):
        self.use_seq2seq = use_seq2seq
        # 1) read CSV
        samples = []
        with open(csv_path, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            for fname, label in reader:
                samples.append((label, fname))
        # 2) build or accept alphabet
        if alphabet is None:
            ctr = Counter(ch for lbl, _ in samples for ch in lbl)
            self.alphabet = "".join(
                sorted(ch for ch, freq in ctr.items() if freq >= min_char_freq)
            )
        else:
            self.alphabet = alphabet
        # 3) build vocab for Seq2Seq
        self.vocab, self.inv_vocab = build_vocab(self.alphabet)
        # 4) char2idx for CTC: map alphabet to 1..N
        self.char2idx = {c: i + 1 for i, c in enumerate(self.alphabet)}
        # 5) filter samples
        self.samples = [
            (lbl, fn) for lbl, fn in samples if all(c in self.char2idx for c in lbl)
        ]
        # 6) save params
        self.images_dir = images_dir
        self.img_h = img_height
        self.img_w = img_max_width
        self.augment = augment
        # 7) PIL aug
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
        # 8) tensor aug
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, fname = self.samples[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert("L")
        # preprocess
        w, h = img.size
        new_w = min(self.img_w, max(1, int(w * self.img_h / h)))
        img = img.resize((new_w, self.img_h), Image.BILINEAR)
        if self.augment and random.random() < 0.2:
            max_cut = int(new_w * 0.05)
            cut = random.randint(0, max_cut)
            l, r = cut // 2, cut - cut // 2
            img = img.crop((l, 0, new_w - r, self.img_h))
        if self.augment:
            img = self.aug_pil(img)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, [0.5], [0.5])
        pad_w = self.img_w - tensor.shape[-1]
        if pad_w > 0:
            tensor = TF.pad(tensor, (0, 0, pad_w, 0), fill=1.0)
        if self.augment:
            tensor = self.aug_tensor(tensor)
        # labels
        ctc_lab = torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)
        seq = (
            [self.vocab["<sos>"]]
            + [self.vocab[c] for c in label]
            + [self.vocab["<eos>"]]
        )
        seq_lab = torch.tensor(seq, dtype=torch.long)
        return tensor, ctc_lab, seq_lab

    @staticmethod
    def collate_fn_hybrid(batch):
        from torch.nn.utils.rnn import pad_sequence

        imgs, ctc_l, seq_l = zip(*batch)
        imgs = torch.stack(imgs)
        lab_lens = torch.tensor([len(l) for l in ctc_l], dtype=torch.long)
        labs_cat = torch.cat(ctc_l)
        B, _, H, W = imgs.shape
        inp_lens = torch.full((B,), W // 4, dtype=torch.long)
        dec_in = [l[:-1] for l in seq_l]
        tgt = [l[1:] for l in seq_l]
        dec_in_p = pad_sequence(
            dec_in, batch_first=True, padding_value=SPECIAL_TOKENS["<pad>"]
        )
        tgt_p = pad_sequence(
            tgt, batch_first=True, padding_value=SPECIAL_TOKENS["<pad>"]
        )
        return imgs, labs_cat, inp_lens, lab_lens, dec_in_p, tgt_p
