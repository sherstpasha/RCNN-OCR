import os
import warnings

# suppress TF & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.models._utils"
)

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from dataset import OCRDataset, create_dataloaders
from model import TRBA, decode_attention, build_attention_inputs
from utils import ctc_greedy_decoder, set_seed, log_samples
from metrics import character_error_rate, word_error_rate, compute_accuracy
from typing import Optional, List, Tuple


def infer_batch(model, imgs, labs, lab_lens, alphabet, device):
    imgs = imgs.to(device)
    lab_lens = lab_lens.to(device)

    if model.use_attention:
        B = imgs.size(0)
        seqs = []
        offset = 0
        for L in lab_lens.tolist():
            seqs.append(labs[offset : offset + L])
            offset += L
        max_L = max(s.size(0) for s in seqs)

        text_input = torch.full(
            (B, max_L + 2), model.PAD_IDX, dtype=torch.long, device=device
        )
        text_input[:, 0] = model.SOS_IDX
        for i, s in enumerate(seqs):
            L = s.size(0)
            text_input[i, 1 : 1 + L] = s.to(device)
            text_input[i, 1 + L] = model.EOS_IDX

        with torch.no_grad():
            out = model(imgs, text=text_input, is_train=False)

        if out.dim() == 3:
            pred_idxs = out.argmax(dim=2)
        else:
            pred_idxs = out

        preds, raws, refs = [], [], []
        for i, seq in enumerate(pred_idxs):
            s = []
            for idx in seq.tolist():
                # как только встретили EOS — выходим из цикла
                if idx == model.EOS_IDX:
                    break
                # пропускаем только специальные токены
                if idx in (model.SOS_IDX, model.PAD_IDX):
                    continue
                # добавляем ВСЕ символы, включая дубли
                s.append(alphabet[idx - 1])
            preds.append("".join(s))

            raws.append(
                "".join(
                    alphabet[i - 1] if 1 <= i <= len(alphabet) else "_"
                    for i in seq.tolist()
                )
            )

            lab_seq = seqs[i].tolist()
            refs.append("".join(alphabet[j - 1] for j in lab_seq if j > 0))

        return preds, raws, refs
    else:
        with torch.no_grad():
            out = model(imgs)
        preds, raws = ctc_greedy_decoder(out, alphabet)

        refs = []
        offset = 0
        for L in lab_lens.tolist():
            lab_seq = labs[offset : offset + L].tolist()
            offset += L
            refs.append("".join(alphabet[j - 1] for j in lab_seq if j > 0))

        return preds, raws, refs


def train_epoch(
    model: TRBA,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    writer: SummaryWriter,
    epoch: int,
    alphabet: List[str],
) -> Tuple[float, float, float, float]:
    """
    Runs one training epoch and returns (avg_loss, accuracy, CER, WER).
    """
    model.train()
    loss_sum = 0.0
    all_refs, all_hyps = [], []
    global_step = (epoch - 1) * len(loader)

    for imgs, labs, _, lab_lens in tqdm(loader, desc=f"Train {epoch}"):
        global_step += 1
        imgs, labs = imgs.to(device), labs.to(device)
        lab_lens = lab_lens.to(device)
        optimizer.zero_grad()

        if model.use_attention:
            # split labels into list of tensors
            seqs = []
            offset = 0
            for L in lab_lens.tolist():
                seqs.append(labs[offset : offset + L])
                offset += L
            text_input, targets = build_attention_inputs(
                seqs, model.SOS_IDX, model.EOS_IDX, model.PAD_IDX, device
            )
            with autocast(device.type):
                outputs, _ = model(imgs, text=text_input, is_train=True)
                # outputs: [B, T-1, C]
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
                )
                pred_idxs = outputs.argmax(dim=2)
            # decode
            hyps_batch, refs_batch = decode_attention(
                seqs, pred_idxs, alphabet, model.SOS_IDX, model.PAD_IDX
            )
        else:
            with autocast(device.type):
                out = model(imgs)
                T, B2, C = out.size()
                inp_lens = torch.full((B2,), T, device=device, dtype=torch.long)
                loss = criterion(out.log_softmax(2), labs, inp_lens, lab_lens)
            pred_idxs, _ = ctc_greedy_decoder(out, alphabet)
            # references
            refs_batch = []
            offset = 0
            for L in lab_lens.tolist():
                seq = labs[offset : offset + L].tolist()
                offset += L
                refs_batch.append("".join(alphabet[j - 1] for j in seq if j > 0))
            hyps_batch = pred_idxs

        # backward
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        writer.add_scalar("train/loss_step", loss.item(), global_step)

        all_refs.extend(refs_batch)
        all_hyps.extend(hyps_batch)

    # compute metrics
    avg_loss = loss_sum / len(loader)
    acc = compute_accuracy(all_refs, all_hyps)
    cer = sum(character_error_rate(r, h) for r, h in zip(all_refs, all_hyps)) / len(
        all_refs
    )
    wer = sum(word_error_rate(r, h) for r, h in zip(all_refs, all_hyps)) / len(all_refs)

    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", acc, epoch)
    writer.add_scalar("train/cer", cer, epoch)
    writer.add_scalar("train/wer", wer, epoch)
    return avg_loss, acc, cer, wer


def validate_epoch(
    model: TRBA,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    alphabet: List[str],
) -> Tuple[float, float, float, float]:
    """
    Runs one validation epoch and returns (avg_loss, accuracy, CER, WER).
    """
    model.eval()
    loss_sum = 0.0
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for imgs, labs, _, lab_lens in tqdm(loader, desc=f"Val {epoch}"):
            imgs, labs = imgs.to(device), labs.to(device)
            lab_lens = lab_lens.to(device)

            if model.use_attention:
                seqs = []
                offset = 0
                for L in lab_lens.tolist():
                    seqs.append(labs[offset : offset + L])
                    offset += L
                text_input, targets = build_attention_inputs(
                    seqs, model.SOS_IDX, model.EOS_IDX, model.PAD_IDX, device
                )
                out, _ = model(imgs, text=text_input, is_train=True)
                loss = criterion(out.reshape(-1, out.size(-1)), targets.reshape(-1))
                pred_idxs = out.argmax(dim=2)
                hyps_batch, refs_batch = decode_attention(
                    seqs, pred_idxs, alphabet, model.SOS_IDX, model.PAD_IDX
                )
            else:
                out = model(imgs)
                T, B2, C = out.size()
                inp_lens = torch.full((B2,), T, device=device, dtype=torch.long)
                loss = criterion(out.log_softmax(2), labs, inp_lens, lab_lens)
                pred_idxs, _ = ctc_greedy_decoder(out, alphabet)
                refs_batch = []
                offset = 0
                for L in lab_lens.tolist():
                    seq = labs[offset : offset + L].tolist()
                    offset += L
                    refs_batch.append("".join(alphabet[j - 1] for j in seq if j > 0))
                hyps_batch = pred_idxs

            loss_sum += loss.item()
            writer.add_scalar("val/loss_step", loss.item(), epoch)

            all_refs.extend(refs_batch)
            all_hyps.extend(hyps_batch)

    avg_loss = loss_sum / len(loader)
    acc = compute_accuracy(all_refs, all_hyps)
    cer = sum(character_error_rate(r, h) for r, h in zip(all_refs, all_hyps)) / len(
        all_refs
    )
    wer = sum(word_error_rate(r, h) for r, h in zip(all_refs, all_hyps)) / len(all_refs)

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", acc, epoch)
    writer.add_scalar("val/cer", cer, epoch)
    writer.add_scalar("val/wer", wer, epoch)
    return avg_loss, acc, cer, wer


def run_experiment(
    exp_name: str,
    train_csvs: List[str],
    train_roots: List[str],
    val_csvs: List[str],
    val_roots: List[str],
    img_h: int = 60,
    img_w: int = 240,
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    transform: Optional[str] = None,
    use_attention: bool = False,
):
    set_seed(42)
    alphabet = OCRDataset.build_alphabet(
        train_csvs + val_csvs, min_char_freq=1, ignore_case=True
    )
    train_loader, val_loader = create_dataloaders(
        train_csvs,
        train_roots,
        val_csvs,
        val_roots,
        alphabet,
        img_h,
        img_w,
        batch_size,
        num_workers=8,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRBA(
        alphabet=alphabet,
        img_height=img_h,
        img_width=img_w,
        transform=transform,
        use_attention=use_attention,
        att_max_length=img_w // 4,
    ).to(device)

    if use_attention:
        criterion = nn.CrossEntropyLoss(ignore_index=model.PAD_IDX)
    else:
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = GradScaler()

    os.makedirs(exp_name, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(exp_name, "logs"))

    best_loss, best_acc = float("inf"), 0.0
    for e in range(1, epochs + 1):
        t_loss, t_acc, t_cer, t_wer = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            writer,
            e,
            alphabet,
        )
        v_loss, v_acc, v_cer, v_wer = validate_epoch(
            model, val_loader, criterion, device, writer, e, alphabet
        )

        # лог примеров
        imgs, labs, _, lab_lens = next(iter(val_loader))
        preds, raws, truths = infer_batch(model, imgs, labs, lab_lens, alphabet, device)
        log_samples(
            e, imgs, lab_lens, preds, raws, truths, writer, n=10, tag="Val/Examples"
        )

        print(
            f"[{exp_name}] Epoch {e}/{epochs} "
            f"Train L={t_loss:.4f} Acc={t_acc:.4f} CER={t_cer:.4f} WER={t_wer:.4f} | "
            f"Val L={v_loss:.4f} Acc={v_acc:.4f} CER={v_cer:.4f} WER={v_wer:.4f}"
        )

        ckpt_dir = os.path.join(exp_name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_by_loss.pth"))
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_by_acc.pth"))
        scheduler.step(v_loss)

    writer.close()


if __name__ == "__main__":
    experiments = [
        {
            "exp_name": "exp_attn",
            "train_csvs": [r"C:\shared\orig_cyrillic\train.tsv"],
            "train_roots": [r"C:\shared\orig_cyrillic\train"],
            "val_csvs": [r"C:\shared\orig_cyrillic\test.tsv"],
            "val_roots": [r"C:\shared\orig_cyrillic\test"],
            "use_attention": True,
            "transform": None,
        },
    ]

    for cfg in experiments:
        run_experiment(
            exp_name=cfg["exp_name"],
            train_csvs=cfg["train_csvs"],
            train_roots=cfg["train_roots"],
            val_csvs=cfg["val_csvs"],
            val_roots=cfg["val_roots"],
            transform=cfg.get("transform"),
            use_attention=cfg.get("use_attention", False),
            batch_size=64,
            epochs=40,
        )
