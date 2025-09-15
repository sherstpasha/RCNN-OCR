import os
import csv
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A


def imread_cv2(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 


def load_charset(charset_path: str):
    """
    Файл формата:
        <PAD>
        <SOS>
        <EOS>
        <BLANK>
        a
        b
        ...
    Возвращает (itos, stoi).
    """
    itos = []
    with open(charset_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok == "":
                continue
            itos.append(tok)
    stoi = {s: i for i, s in enumerate(itos)}
    return itos, stoi

class ResizeAndPadA(A.ImageOnlyTransform):
    def __init__(self, img_h=32, img_w=256, align_h="left", align_v="center",
                 always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.align_h = align_h
        self.align_v = align_v

    def _interp(self, src_h, src_w, dst_h, dst_w):
        if dst_h < src_h or dst_w < src_w:
            return cv2.INTER_AREA
        return cv2.INTER_LINEAR

    def apply(self, img, **params):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        h, w = img.shape[:2]

        scale = min(self.img_h / max(h, 1), self.img_w / max(w, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        interp = self._interp(h, w, new_h, new_w)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        canvas = np.full((self.img_h, self.img_w, 3), 255, dtype=img.dtype)

        if self.align_h == "left":
            x0 = 0
        elif self.align_h == "right":
            x0 = self.img_w - new_w
        else:
            x0 = (self.img_w - new_w) // 2

        if self.align_v == "top":
            y0 = 0
        elif self.align_v == "bottom":
            y0 = self.img_h - new_h
        else: 
            y0 = (self.img_h - new_h) // 2
        x0 = max(0, min(x0, self.img_w - new_w))
        y0 = max(0, min(y0, self.img_h - new_h))

        canvas[y0:y0+new_h, x0:x0+new_w] = img_resized
        return canvas
    
def pack_attention_targets(texts, stoi, max_len, drop_blank=True):
    PAD = stoi["<PAD>"]
    SOS = stoi["<SOS>"]
    EOS = stoi["<EOS>"]
    BLANK = stoi.get("<BLANK>", None)

    B = len(texts)
    T = max_len + 1
    
    text_in  = torch.full((B, T), PAD, dtype=torch.long)
    text_in[:, 0] = SOS

    target_y = torch.full((B, T), PAD, dtype=torch.long)
    lengths  = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(texts):
        ids = []
        for ch in s:
            if ch not in stoi:
                continue
            idx = stoi[ch]
            if drop_blank and BLANK is not None and idx == BLANK:
                continue
            ids.append(idx)

        L = min(len(ids), max_len) 
        if L > 0:
            text_in[i, 1:1+L] = torch.tensor(ids[:L], dtype=torch.long)
            target_y[i, :L] = torch.tensor(ids[:L], dtype=torch.long) 

        target_y[i, L] = EOS 
        lengths[i] = L + 1

    return text_in, target_y, lengths

class OCRDatasetAttn(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        stoi: dict,                        
        img_height: int = 32,
        img_max_width: int = 128,
        encoding: str = "utf-8",
        transform: Optional[callable] = None,
    ):
        self.images_dir = images_dir
        self.img_h = img_height
        self.img_w = img_max_width
        self.stoi = stoi
        self.transform = transform

        self.samples: List[Tuple[str, str]] = []
        with open(csv_path, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            for fname, label in reader:
                if all(c in self.stoi for c in label):
                    self.samples.append((fname, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.images_dir, fname)
        img = imread_cv2(path)

        if self.transform:
            augmented = self.transform(image=img)
            tensor = augmented["image"]
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return tensor, label

    @staticmethod   
    def make_collate_attn(stoi, max_len: int, drop_blank: bool = True):
        def collate(batch):
            imgs, labels_text = zip(*batch)
            imgs = torch.stack(imgs)
            text_in, target_y, lengths = pack_attention_targets(
                labels_text, stoi=stoi, max_len=max_len, drop_blank=drop_blank
            )
            return imgs, text_in, target_y, lengths
        return collate