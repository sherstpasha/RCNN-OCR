import os
import csv
import random
from typing import List, Tuple
from tqdm.auto import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torchvision import transforms as T
from utils import log_samples
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    get_linear_schedule_with_warmup,
)
from metrics import character_error_rate, word_error_rate, compute_accuracy
import random
import textwrap
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OCRDataset(Dataset):
    """
    Датасет для TrOCR: короткие обрезки текста (1-2 слова).
    Возвращает изображения, токены и оригинальные строки.
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        processor: TrOCRProcessor,
        img_size: Tuple[int, int] = (240, 60),
        max_target_length: int = 32,
        ignore_case: bool = True,
        encoding: str = "utf-8",
    ):
        self.processor = processor
        self.images_dir = images_dir
        self.img_size = img_size
        self.max_target_length = max_target_length
        samples: List[Tuple[str, str]] = []
        with open(csv_path, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            for fname, label in reader:
                if ignore_case:
                    label = label.lower()
                samples.append((fname, label))
        self.samples = samples

        self.transform = T.Compose(
            [
                T.Resize(self.img_size, interpolation=Image.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.images_dir, fname)
        image = Image.open(path).convert("RGB")
        pixel_values = self.transform(image)

        tok = self.processor.tokenizer(
            label,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = tok.input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return pixel_values, labels, label

    @staticmethod
    def collate_fn(batch):
        pixels = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        raw = [b[2] for b in batch]
        return {
            "pixel_values": pixels,
            "labels": labels,
            "raw_labels": raw,
        }


def create_dataloaders(
    train_csvs: List[str],
    train_dirs: List[str],
    val_csvs: List[str],
    val_dirs: List[str],
    processor: TrOCRProcessor,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = [OCRDataset(c, d, processor) for c, d in zip(train_csvs, train_dirs)]
    val_ds = [OCRDataset(c, d, processor) for c, d in zip(val_csvs, val_dirs)]
    train_loader = DataLoader(
        ConcatDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=OCRDataset.collate_fn,
    )
    val_loader = DataLoader(
        ConcatDataset(val_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=OCRDataset.collate_fn,
    )
    return train_loader, val_loader


def run_experiment(
    exp_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    device: torch.device,
    epochs: int = 10,
    lr: float = 5e-5,
):
    """
    Запуск эксперимента: обучение и валидация модели с логированием в TensorBoard,
    прогресс-барами и примерами предсказаний.
    """
    set_seed(42)
    os.makedirs(exp_name, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(exp_name, "logs"))

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_loss = float("inf")
    best_acc = 0.0
    global_step = 0

    for epoch in range(1, epochs + 1):
        # === TRAINING ===
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch} [train]", unit="batch")
        for batch in train_loop:
            optimizer.zero_grad()
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pv, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1
            train_loop.set_postfix(batch_loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)

        # === VALIDATION ===
        torch.cuda.empty_cache()
        model.eval()
        val_loss = 0.0
        hyps, refs = [], []

        # Prepare a sample batch for logging
        sample_batch = next(iter(val_loader))
        sample_pixels = sample_batch["pixel_values"].to(device)
        sample_raws = sample_batch["raw_labels"]

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch} [val]", unit="batch")
        with torch.no_grad():
            for batch in val_loop:
                pv = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                raw = batch["raw_labels"]

                out = model(pixel_values=pv, labels=labels)
                val_loss += out.loss.item()

                gen_ids = model.generate(
                    pv,
                    max_length=32,
                    use_cache=False,
                )
                preds = processor.batch_decode(gen_ids, skip_special_tokens=True)
                hyps.extend(preds)
                refs.extend(raw)
                val_loop.set_postfix(batch_loss=out.loss.item())

        avg_val_loss = val_loss / len(val_loader)
        acc = compute_accuracy(refs, hyps)
        cer = sum(character_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
        wer = sum(word_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)

        writer.add_scalar("val/loss_epoch", avg_val_loss, epoch)
        writer.add_scalar("val/accuracy", acc, epoch)
        writer.add_scalar("val/cer", cer, epoch)
        writer.add_scalar("val/wer", wer, epoch)

        # Log sample predictions to TensorBoard
        sample_gen_ids = model.generate(
            sample_pixels,
            max_length=32,
            use_cache=False,
        )
        sample_preds = processor.batch_decode(sample_gen_ids, skip_special_tokens=True)
        # compute lengths of the ground‐truth strings
        # получаем тензор с id, восстанавливаем pad-токен там, где стоит -100
        labels = sample_batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        # декодируем в список строк
        truths = processor.batch_decode(labels, skip_special_tokens=True)
        imgs_cpu = sample_pixels.cpu().mean(dim=1, keepdim=True)
        log_samples(
            epoch=epoch,
            imgs=imgs_cpu,
            lab_lens=[len(s) for s in truths],
            preds=sample_preds,
            raws=sample_raws,  # raw_labels по-прежнему для визуализации
            truths=truths,  # а здесь распакованные из токенов
            writer=writer,
            n=5,
            tag="Val Examples",
        )
        # Print metrics and save best checkpoints
        print(
            f"[{exp_name}] Epoch {epoch}/{epochs} "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} "
            f"acc={acc:.4f} cer={cer:.4f} wer={wer:.4f}"
        )
        ckpt_dir = os.path.join(exp_name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model.save_pretrained(os.path.join(ckpt_dir, "best_by_loss"))
            processor.save_pretrained(os.path.join(ckpt_dir, "best_by_loss"))
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(os.path.join(ckpt_dir, "best_by_acc"))
            processor.save_pretrained(os.path.join(ckpt_dir, "best_by_acc"))

    writer.close()


if __name__ == "__main__":
    exp_name = "trocr_experiment"
    train_csvs = [r"C:\shared\orig_cyrillic\test.tsv"]
    train_dirs = [r"C:\shared\orig_cyrillic\test"]
    val_csvs = [r"C:\shared\orig_cyrillic\test.tsv"]
    val_dirs = [r"C:\shared\orig_cyrillic\test"]

    # Включаем use_fast=True
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-small-handwritten", use_fast=True
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-small-handwritten"
    )

    # Задаём специальные токены в конфиге модели
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # (Опционально) Установим размер словаря декодера, чтобы увязать его с токенизатором
    model.config.vocab_size = model.config.decoder.vocab_size

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        train_csvs,
        train_dirs,
        val_csvs,
        val_dirs,
        processor,
        batch_size=8,
        num_workers=4,
    )

    run_experiment(
        exp_name,
        train_loader,
        val_loader,
        model,
        processor,
        device,
        epochs=10,
        lr=5e-5,
    )
