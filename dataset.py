import csv
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from collections import Counter, defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm


def build_file_index(roots, exts={".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}):
    if isinstance(roots, str):
        roots = [roots]
    index = defaultdict(list)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if exts and ext not in exts:
                    continue
                index[fn.lower()].append(os.path.join(dirpath, fn))
    return index

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
            tok = line.rstrip("\n")
            if tok == "":
                continue
            itos.append(tok)
    stoi = {s: i for i, s in enumerate(itos)}
    return itos, stoi


class ResizeAndPadA(A.ImageOnlyTransform):
    def __init__(
        self,
        img_h=32,
        img_w=256,
        align_h="left",
        align_v="center",
        always_apply=True,
        p=1.0,
    ):
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

        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = img_resized
        return canvas


def pack_attention_targets(texts, stoi, max_len, drop_blank=True):
    PAD = stoi["<PAD>"]
    SOS = stoi["<SOS>"]
    EOS = stoi["<EOS>"]
    BLANK = stoi.get("<BLANK>", None)

    B = len(texts)
    T = max_len + 1

    text_in = torch.full((B, T), PAD, dtype=torch.long)
    text_in[:, 0] = SOS

    target_y = torch.full((B, T), PAD, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)

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
            text_in[i, 1 : 1 + L] = torch.tensor(ids[:L], dtype=torch.long)
            target_y[i, :L] = torch.tensor(ids[:L], dtype=torch.long)

        target_y[i, L] = EOS
        lengths[i] = L + 1

    return text_in, target_y, lengths


def build_file_index(roots, exts={".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}):
    if isinstance(roots, str):
        roots = [roots]
    index = defaultdict(list)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if exts and ext not in exts:
                    continue
                index[fn.lower()].append(os.path.join(dirpath, fn))
    return index


class OCRDatasetAttn(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: str | list,
        stoi: dict,
        img_height: int = 32,
        img_max_width: int = 128,
        encoding: str = "utf-8",
        transform: Optional[callable] = None,
        num_workers: int = -1,
        delimiter: str | None = None,
        has_header: bool | None = None,
        strict_charset: bool = True,
        validate_image: bool = True,
        max_len: Optional[int] = None,
        strict_max_len: bool = True,
    ):
        self.images_dir = images_dir
        self.img_h = img_height
        self.img_w = img_max_width
        self.stoi = stoi
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []
        self._file_index = build_file_index(images_dir)
        self._encoding = encoding
        self._delimiter = delimiter if delimiter is not None else ("\t" if csv_path.lower().endswith(".tsv") else ",")
        self._has_header = has_header
        self._strict_charset = strict_charset
        self._validate_image = validate_image
        self._max_len = max_len
        self._strict_max_len = strict_max_len

        self._reasons = {
            "bad_row": 0, "empty_fname": 0, "empty_label": 0,
            "charset": 0, "too_long": 0,
            "missing_path": 0, "ambiguous": 0, "readfail": 0,
        }
        self._examples = {k: [] for k in self._reasons}
        self._EX_MAX = 8
        self._missing_chars = Counter()

        rows = self._read_rows(csv_path)
        self._maybe_detect_header(rows)
        self._build_samples(rows, num_workers)
        self._print_summary(csv_path)

        if not self.samples:
            raise RuntimeError(f"В датасете {csv_path} не осталось валидных примеров!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        abs_path, label = self.samples[idx]
        try:
            img = imread_cv2(abs_path)
        except Exception as e:
            raise IndexError(f"Ошибка чтения изображения {abs_path}: {e}")

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

    def _read_rows(self, csv_path: str):
        with open(csv_path, newline="", encoding=self._encoding) as f:
            reader = csv.reader(f, delimiter=self._delimiter)
            rows = list(reader)
        return rows

    def _maybe_detect_header(self, rows: list[list[str]]):
        if self._has_header is not None or not rows:
            return
        head0 = str(rows[0][0]).strip().lower()
        self._has_header = head0 in {"file", "filename", "image", "path", "img", "name"}
        if self._has_header:
            rows.pop(0)
        self._rows = rows

        if not hasattr(self, "_rows"):
            self._rows = rows

    @staticmethod
    def _norm_label(s: str) -> str:

        return s.replace("\u00A0", " ").strip().replace("\ufeff", "")

    @staticmethod
    def _norm_fname(s: str) -> str:
        return s.strip().replace("\ufeff", "").replace("\\", "/")

    def _resolve_path(self, fname: str) -> Optional[str]:
        if os.path.isabs(fname) and os.path.exists(fname):
            return fname

        if isinstance(self.images_dir, str):
            p = os.path.join(self.images_dir, fname)
            if os.path.exists(p):
                return p
        else:
            for root in self.images_dir:
                p = os.path.join(root, fname)
                if os.path.exists(p):
                    return p

        base = os.path.basename(fname).lower()
        candidates = self._file_index.get(base, [])
        if not candidates:
            return None
        if len(candidates) > 1:
            self._reasons["ambiguous"] += 1
            if len(self._examples["ambiguous"]) < self._EX_MAX:
                self._examples["ambiguous"].append((fname, candidates[:3]))
        return candidates[0]

    def _effective_len(self, label: str) -> int:
        if not self._strict_charset:
            return len(label)
        return sum(1 for c in label if c in self.stoi)

    def _validate_row(self, row: list[str]) -> Optional[tuple[str, str]]:
        if len(row) < 2:
            self._reasons["bad_row"] += 1
            if len(self._examples["bad_row"]) < self._EX_MAX:
                self._examples["bad_row"].append(row)
            return None

        fname = self._norm_fname(row[0])
        label = self._norm_label(row[1])

        if not fname:
            self._reasons["empty_fname"] += 1
            if len(self._examples["empty_fname"]) < self._EX_MAX:
                self._examples["empty_fname"].append(row)
            return None

        if label == "":
            self._reasons["empty_label"] += 1
            if len(self._examples["empty_label"]) < self._EX_MAX:
                self._examples["empty_label"].append(fname)
            return None

        if self._strict_charset:
            missing = [c for c in label if c not in self.stoi]
            if missing:
                self._reasons["charset"] += 1
                self._missing_chars.update(missing)
                if len(self._examples["charset"]) < self._EX_MAX:
                    uniq = "".join(sorted(set(missing)))[:20]
                    self._examples["charset"].append((fname, label[:50], uniq))
                return None

        if self._strict_max_len and self._max_len is not None:
            if self._effective_len(label) > self._max_len:
                self._reasons["too_long"] += 1
                if len(self._examples["too_long"]) < self._EX_MAX:
                    self._examples["too_long"].append((fname, len(label), f"eff>{self._max_len}"))
                return None

        abs_path = self._resolve_path(fname)
        if not abs_path or not os.path.exists(abs_path):
            self._reasons["missing_path"] += 1
            if len(self._examples["missing_path"]) < self._EX_MAX:
                self._examples["missing_path"].append(fname)
            return None

        if self._validate_image:
            try:
                _ = imread_cv2(abs_path)
            except Exception as e:
                self._reasons["readfail"] += 1
                if len(self._examples["readfail"]) < self._EX_MAX:
                    self._examples["readfail"].append(f"{fname} :: {type(e).__name__}")
                return None

        return abs_path, label 

    def _build_samples(self, rows: list[list[str]], num_workers: int):
        if num_workers == -1:
            workers = os.cpu_count() or 4
        elif num_workers is None:
            workers = 8
        else:
            workers = max(1, num_workers)

        results, skipped = [], 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._validate_row, row) for row in self._rows]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="Проверка датасета", leave=False):
                res = fut.result()
                if res is not None:
                    results.append(res)
                else:
                    skipped += 1
        self.samples = results
        self._skipped = skipped

    def _print_summary(self, csv_path: str):
        if self._skipped > 0:
            print(f"[OCRDatasetAttn] {csv_path}: пропущено {self._skipped} записей.")
            order = ["bad_row","empty_fname","empty_label","charset","too_long","missing_path","ambiguous","readfail"]
            for k in order:
                cnt = self._reasons[k]
                if cnt > 0:
                    print(f"  - {k}: {cnt}")
                    ex = self._examples[k]
                    if ex:
                        print(f"    примеры: {ex[:self._EX_MAX]}")
            if self._reasons["charset"] > 0 and self._missing_chars:
                print("  Отсутствующие символы (TOP 30):")
                for ch, cnt in self._missing_chars.most_common(30):
                    print(f"    '{ch}' (U+{ord(ch):04X}, repr={repr(ch)}): {cnt} раз(а)")

class ProportionalBatchSampler:
    def __init__(self, datasets, batch_size, proportions):
        assert abs(sum(proportions) - 1.0) < 1e-6, "Пропорции должны давать сумму = 1"
        self.datasets = datasets
        self.batch_size = batch_size
        self.proportions = proportions
        self.idxs = [list(range(len(ds))) for ds in datasets]
        for idxs in self.idxs:
            random.shuffle(idxs)

    def __iter__(self):
        n_batches = len(self)
        for _ in range(n_batches):
            batch = []
            for ds_idx, prop in enumerate(self.proportions):
                n = int(round(self.batch_size * prop))
                if n == 0:
                    continue

                if len(self.idxs[ds_idx]) < n:
                    self.idxs[ds_idx] = list(range(len(self.datasets[ds_idx])))
                    random.shuffle(self.idxs[ds_idx])

                chosen = [self.idxs[ds_idx].pop() for _ in range(n)]
                batch.extend([(ds_idx, c) for c in chosen])

            random.shuffle(batch)
            yield batch

    def __len__(self):
        min_batches = min(
            len(ds) // max(1, int(round(self.batch_size * prop)))
            for ds, prop in zip(self.datasets, self.proportions)
            if prop > 0
        )
        return min_batches


class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        ds_idx, sample_idx = index
        return self.datasets[ds_idx][sample_idx]

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)


def get_train_transform(params, img_h, img_w):
    return A.Compose(
        [
            ResizeAndPadA(img_h=img_h, img_w=img_w),
            A.ShiftScaleRotate(
                shift_limit=round(params.get("shift_limit", 0.03), 4),
                scale_limit=round(params.get("scale_limit", 0.08), 4),
                rotate_limit=int(params.get("rotate_limit", 3)),
                border_mode=0,
                value=(255, 255, 255),
                p=round(params.get("p_ShiftScaleRotate", 0.3), 4),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=round(params.get("brightness_limit", 0.2), 4),
                contrast_limit=round(params.get("contrast_limit", 0.2), 4),
                p=round(params.get("p_BrightnessContrast", 0.3), 4),
            ),
            A.InvertImg(p=round(params.get("invert_p", 0.0), 4)),  # 🔑 теперь как все
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_h, img_w):
    return A.Compose(
        [
            ResizeAndPadA(img_h=img_h, img_w=img_w),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def decode_tokens(ids, itos, pad_id, eos_id, blank_id=None):
    out = []
    for t in ids:
        t = int(t)
        if t == eos_id:
            break
        if t == pad_id or (blank_id is not None and t == blank_id):
            continue
        out.append(itos[t])
    return "".join(out)
