# dataset.py

import os
import csv
from collections import Counter
from typing import List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OCRDataset(Dataset):
    """
    Датасет для OCR с CTC:
      - CSV-файл: каждая строка "имя_файла<TAB>текст"
      - images_dir: папка с этими файлами
      - alphabet: строка допустимых символов (если None, строится автоматически)
      - min_char_freq: минимальная частота символа для попадания в алфавит
      - img_height, img_max_width: форм-фактор входных изображений
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
    ):
        # 1) Считываем все пары (label, fname)
        samples = []
        with open(csv_path, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            for fname, label in reader:
                samples.append((label, fname))
        # 2) Если алфавит не задан, строим его по всем выборкам:
        if alphabet is None:
            counter = Counter()
            for label, _ in samples:
                counter.update(label)
            # Оставляем только символы с достаточной частотой
            self.alphabet = "".join(sorted(
                ch for ch, freq in counter.items() if freq >= min_char_freq
            ))
        else:
            self.alphabet = alphabet

        # 3) Строим словарь char→idx (1…|alphabet|), 0 зарезервирован под blank/unk
        self.char2idx = {c: i + 1 for i, c in enumerate(self.alphabet)}

        # 4) Фильтруем выборку: убираем все образцы, где встретился незнакомый символ
        filtered = []
        for label, fname in samples:
            if all((c in self.char2idx) for c in label):
                filtered.append((label, fname))
        self.samples = filtered

        # 5) Сохраняем остальное и настраиваем трансформации
        self.images_dir = images_dir
        self.img_h = img_height
        self.img_w = img_max_width
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((img_height, img_max_width)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        label, fname = self.samples[i]
        # Загружаем и преобразуем изображение
        path = os.path.join(self.images_dir, fname)
        img = Image.open(path).convert("L")
        img = self.transform(img)
        # Кодируем текст
        lab = torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)
        return img, lab

    @staticmethod
    def collate_fn(batch):
        imgs, labs = zip(*batch)
        imgs = torch.stack(imgs)
        lab_lens = torch.tensor([len(l) for l in labs], dtype=torch.long)
        labs_cat = torch.cat(labs)
        B, _, H, W = imgs.shape
        # Пример: если два MaxPool2d с kernel=2, то W//4
        inp_lens = torch.full((B,), W // 4, dtype=torch.long)
        return imgs, labs_cat, inp_lens, lab_lens

    @staticmethod
    def build_alphabet(
        csv_paths: List[str],
        min_char_freq: int = 1,
        encoding: str = "utf-8"
    ) -> str:
        """
        Построить алфавит по одному или нескольким CSV-файлам,
        отфильтровав символы, встретившиеся реже min_char_freq.
        """
        counter = Counter()
        for p in csv_paths:
            with open(p, newline="", encoding=encoding) as f:
                reader = csv.reader(f, delimiter="\t")
                for _, label in reader:
                    counter.update(label)
        alphabet = "".join(sorted(
            ch for ch, freq in counter.items() if freq >= min_char_freq
        ))
        return alphabet
