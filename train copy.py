import os
import csv
import random
import warnings
import textwrap
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.utils as vutils

from dataset import OCRDataset
from model import CRNN
from utils import decode, CharNGramLM, beam_search_ctc_with_lm
from metrics import character_error_rate, word_error_rate, compute_accuracy
from vocab import SPECIAL_TOKENS

# suppress TF & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.models._utils"
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_alphabet(
    paths, min_char_freq=30, encoding="utf-8", case_insensitive: bool = False
):
    ctr = {}
    for p in paths:
        with open(p, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            for _, lbl in reader:
                if case_insensitive:
                    lbl = lbl.lower()
                for ch in lbl:
                    ctr[ch] = ctr.get(ch, 0) + 1
    return "".join(sorted(ch for ch, freq in ctr.items() if freq >= min_char_freq))


def create_dataloaders(
    train_csvs,
    train_roots,
    val_csvs,
    val_roots,
    alphabet,
    img_height,
    img_max_width,
    batch_size,
):
    train_ds = [
        OCRDataset(csv, root, alphabet, img_height, img_max_width, augment=True)
        for csv, root in zip(train_csvs, train_roots)
    ]
    val_ds = [
        OCRDataset(csv, root, alphabet, img_height, img_max_width, augment=False)
        for csv, root in zip(val_csvs, val_roots)
    ]
    train_loader = DataLoader(
        ConcatDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=OCRDataset.collate_fn_hybrid,
    )
    val_loader = DataLoader(
        ConcatDataset(val_ds),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=OCRDataset.collate_fn_hybrid,
    )
    return train_loader, val_loader


def init_model(
    img_height,
    img_max_width,
    alphabet,
    device,
    pretrained=False,
    freeze_cnn_layers=None,
):
    num_ctc = 1 + len(alphabet)
    num_attn = len(SPECIAL_TOKENS) + len(alphabet)
    model = CRNN(
        img_height,
        img_max_width,
        num_ctc_classes=num_ctc,
        num_attn_classes=num_attn,
        pretrained=pretrained,
        backbone="msfresnet",
        freeze_cnn_layers=freeze_cnn_layers,
    ).to(device)
    criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    criterion_attn = nn.CrossEntropyLoss(ignore_index=0)
    return model, criterion_ctc, criterion_attn


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion_ctc,
    criterion_attn,
    alphabet,
    decode_type,
    device,
    writer,
    epoch,
    alpha,
):
    model.train()
    total_ctc, total_attn = 0.0, 0.0
    refs, hyps = [], []
    step = 0

    for imgs, labs_cat, inp_lens, lab_lens, dec_in, targets in tqdm(
        loader, desc=f"Train Epoch {epoch}"
    ):
        imgs, labs_cat = imgs.to(device), labs_cat.to(device)
        lab_lens = lab_lens.to(device)
        dec_in, targets = dec_in.to(device), targets.to(device)

        optimizer.zero_grad()
        ctc_out, attn_out = model(imgs, decoder_inputs=dec_in)

        B = attn_out.size(0)
        input_lengths = torch.full(
            (B,), ctc_out.size(0), dtype=torch.long, device=device
        )

        loss_ctc = criterion_ctc(ctc_out, labs_cat, input_lengths, lab_lens)
        loss_attn = criterion_attn(
            attn_out.view(-1, attn_out.size(-1)), targets.view(-1)
        )
        loss = alpha * loss_ctc + (1 - alpha) * loss_attn
        loss.backward()
        optimizer.step()

        total_ctc += loss_ctc.item()
        total_attn += loss_attn.item()

        preds, _ = decode(ctc_out, alphabet, method=decode_type)
        offset = 0
        for L in lab_lens.tolist():
            seq = labs_cat[offset : offset + L]
            offset += L
            refs.append("".join(alphabet[i - 1] for i in seq if i > 0))
        hyps.extend(preds)

        import random

        n = min(10, len(refs))
        sample_idxs = random.sample(range(len(refs)), n)
        print("\n=== Sample Greedy Validation Results ===")
        print(f"{'#':<3} {'Reference':<30} {'Prediction':<30} {'OK'}")
        for i in sample_idxs:
            ref = refs[i]
            hyp = hyps[i]
            ok = "✔" if ref == hyp else "✘"
            print(f"{i:<3} {ref:<30} {hyp:<30} {ok}")
        print("=== end samples ===\n")

        writer.add_scalar("train/loss_iter", loss.item(), global_step=step)
        step += 1

    avg_ctc = total_ctc / len(loader)
    avg_attn = total_attn / len(loader)
    train_loss = alpha * avg_ctc + (1 - alpha) * avg_attn

    acc = compute_accuracy(refs, hyps)
    cer = sum(character_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
    wer = sum(word_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)

    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/accuracy", acc, epoch)
    writer.add_scalar("train/CER", cer, epoch)
    writer.add_scalar("train/WER", wer, epoch)

    return train_loss, acc, cer, wer


def validate_beam(
    model,
    loader,
    criterion_ctc,
    criterion_attn,
    alphabet,
    lm_model,
    alpha,
    lm_alpha,
    device,
    writer,
    epoch,
):
    model.eval()
    total_ctc, total_attn = 0.0, 0.0
    refs, hyps = [], []
    with torch.no_grad():
        for imgs, labs_cat, _, lab_lens, dec_in, targets in tqdm(
            loader, desc=f"Val Beam Epoch {epoch}"
        ):
            imgs, labs_cat = imgs.to(device), labs_cat.to(device)
            lab_lens = lab_lens.to(device)
            dec_in, targets = dec_in.to(device), targets.to(device)

            # Use decoder inputs for attention head
            ctc_out, attn_out = model(imgs, decoder_inputs=dec_in)
            # ctc_out: [T, B, C_ctc]
            T, B, _ = ctc_out.size()
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            # Compute losses
            total_ctc += criterion_ctc(
                ctc_out, labs_cat, input_lengths, lab_lens
            ).item()
            total_attn += criterion_attn(
                attn_out.view(-1, attn_out.size(-1)), targets.view(-1)
            ).item()

            # Beam + LM decoding
            preds, _ = beam_search_ctc_with_lm(
                ctc_out, alphabet, lm_model, alpha=lm_alpha, beam_width=5
            )
            offset = 0
            for L in lab_lens.tolist():
                seq = labs_cat[offset : offset + L]
                offset += L
                refs.append("".join(alphabet[i - 1] for i in seq if i > 0))
            hyps.extend(preds)

            import random

            n = min(10, len(refs))
            sample_idxs = random.sample(range(len(refs)), n)
            print("\n=== Sample Greedy Validation Results ===")
            print(f"{'#':<3} {'Reference':<30} {'Prediction':<30} {'OK'}")
            for i in sample_idxs:
                ref = refs[i]
                hyp = hyps[i]
                ok = "✔" if ref == hyp else "✘"
                print(f"{i:<3} {ref:<30} {hyp:<30} {ok}")
            print("=== end samples ===\n")

    avg_loss = alpha * (total_ctc / len(loader)) + (1 - alpha) * (
        total_attn / len(loader)
    )
    acc = compute_accuracy(refs, hyps)
    cer = sum(character_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
    wer = sum(word_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)

    writer.add_scalar("val_beam/loss", avg_loss, epoch)
    writer.add_scalar("val_beam/accuracy", acc, epoch)
    writer.add_scalar("val_beam/CER", cer, epoch)
    writer.add_scalar("val_beam/WER", wer, epoch)

    return avg_loss, acc, cer, wer


def validate_greedy(
    model,
    loader,
    criterion_ctc,
    alphabet,
    alpha,
    device,
    writer,
    epoch,
):
    model.eval()
    total_ctc = 0.0
    refs, hyps = [], []
    with torch.no_grad():
        for imgs, labs_cat, _, lab_lens, _, _ in tqdm(
            loader, desc=f"Val Greedy Epoch {epoch}"
        ):
            imgs, labs_cat = imgs.to(device), labs_cat.to(device)
            lab_lens = lab_lens.to(device)
            ctc_out, _ = model(imgs)
            B = ctc_out.size(1)
            input_lengths = torch.full(
                (B,), ctc_out.size(0), dtype=torch.long, device=device
            )
            total_ctc += criterion_ctc(
                ctc_out, labs_cat, input_lengths, lab_lens
            ).item()
            preds, _ = decode(ctc_out, alphabet, method="greedy")
            offset = 0
            for L in lab_lens.tolist():
                seq = labs_cat[offset : offset + L]
                offset += L
                refs.append("".join(alphabet[i - 1] for i in seq if i > 0))
            hyps.extend(preds)

            import random

            n = min(10, len(refs))
            sample_idxs = random.sample(range(len(refs)), n)
            print("\n=== Sample Greedy Validation Results ===")
            print(f"{'#':<3} {'Reference':<30} {'Prediction':<30} {'OK'}")
            for i in sample_idxs:
                ref = refs[i]
                hyp = hyps[i]
                ok = "✔" if ref == hyp else "✘"
                print(f"{i:<3} {ref:<30} {hyp:<30} {ok}")
            print("=== end samples ===\n")
    avg_loss = alpha * (total_ctc / len(loader))
    acc = compute_accuracy(refs, hyps)
    cer = sum(character_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
    wer = sum(word_error_rate(r, h) for r, h in zip(refs, hyps)) / len(refs)
    writer.add_scalar("val_greedy/loss", avg_loss, epoch)
    writer.add_scalar("val_greedy/accuracy", acc, epoch)
    writer.add_scalar("val_greedy/CER", cer, epoch)
    writer.add_scalar("val_greedy/WER", wer, epoch)
    return avg_loss, acc, cer, wer


def run_training(config):
    set_seed(config.seed)
    # ensure checkpoint directory exists
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # build alphabet
    paths = config.train_csvs + config.val_csvs
    alphabet = build_alphabet(
        paths,
        min_char_freq=config.min_char_freq,
        case_insensitive=getattr(config, "case_insensitive", False),
    )

    # dataloaders
    train_loader, val_loader = create_dataloaders(
        config.train_csvs,
        config.train_roots,
        config.val_csvs,
        config.val_roots,
        alphabet,
        config.img_height,
        config.img_max_width,
        config.batch_size,
    )

    # device and model
    device = torch.device(config.device)
    model, crit_ctc, crit_attn = init_model(
        config.img_height,
        config.img_max_width,
        alphabet,
        device,
        pretrained=True,
        freeze_cnn_layers=getattr(config, "freeze_cnn_layers", None),
    )
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {n_total:,}")
    print(f"Trainable parameters: {n_trainable:,}")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
    )
    # scheduler based on default LM alpha
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # tensorboard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    # load LM
    lm_model = CharNGramLM.load(config.lm_path)

    # Prepare multiple LM alphas for beam validation
    lm_alphas = getattr(config, "lm_alphas", [config.lm_alpha])
    # Best losses and metrics paths per alpha
    best_beam_losses = {a: float("inf") for a in lm_alphas}
    metrics_beam_files = {
        a: os.path.join(config.checkpoint_dir, f"metrics_beam_alpha{a}.csv")
        for a in lm_alphas
    }
    for fpath in metrics_beam_files.values():
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            f.write("epoch,loss,acc,CER,WER")

    # Greedy metrics
    best_greedy_loss = float("inf")
    metrics_greedy = os.path.join(config.checkpoint_dir, "metrics_greedy.csv")
    with open(metrics_greedy, "w", newline="", encoding="utf-8") as f:
        f.write("epoch,loss,acc,CER,WER")

    # Training loop
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss, train_acc, train_cer, train_wer = train_one_epoch(
            model,
            train_loader,
            optimizer,
            crit_ctc,
            crit_attn,
            alphabet,
            config.decode_train,
            device,
            writer,
            epoch,
            config.alpha,
        )

        # Beam+LM validations for each alpha
        beam_results = {}
        for a in lm_alphas:
            loss_b, acc_b, cer_b, wer_b = validate_beam(
                model,
                val_loader,
                crit_ctc,
                crit_attn,
                alphabet,
                lm_model,
                config.alpha,
                a,
                device,
                writer,
                epoch,
            )
            # record
            beam_results[a] = (loss_b, acc_b)
            with open(metrics_beam_files[a], "a", newline="", encoding="utf-8") as f:
                f.write(f"{epoch},{loss_b:.6f},{acc_b:.6f},{cer_b:.6f},{wer_b:.6f}")
            # save best
            if loss_b < best_beam_losses[a]:
                best_beam_losses[a] = loss_b
                torch.save(
                    model.state_dict(),
                    os.path.join(config.checkpoint_dir, f"best_beam_alpha{a}.pth"),
                )

        # Greedy validation
        greedy_loss, greedy_acc, greedy_cer, greedy_wer = validate_greedy(
            model,
            val_loader,
            crit_ctc,
            alphabet,
            config.alpha,
            device,
            writer,
            epoch,
        )
        with open(metrics_greedy, "a", newline="", encoding="utf-8") as f:
            f.write(
                f"{epoch},{greedy_loss:.6f},{greedy_acc:.6f},{greedy_cer:.6f},{greedy_wer:.6f}"
            )
        if greedy_loss < best_greedy_loss:
            best_greedy_loss = greedy_loss
            torch.save(
                model.state_dict(),
                os.path.join(config.checkpoint_dir, "best_greedy.pth"),
            )

        # Scheduler step on default LM alpha (first)
        scheduler.step(beam_results[lm_alphas[0]][0])

        # Console log
        stats = [f"Train L={train_loss:.4f} Acc={train_acc:.4f}"]
        for a, (lb, ab) in beam_results.items():
            stats.append(f"Beam@{a} L={lb:.4f} Acc={ab:.4f}")
        stats.append(f"Greedy L={greedy_loss:.4f} Acc={greedy_acc:.4f}")
        print(f"Epoch {epoch}/{config.epochs} | " + " | ".join(stats))

    writer.close()


if __name__ == "__main__":
    from argparse import Namespace

    config = Namespace(
        seed=42,
        train_csvs=[r"C:\shared\Archive_19_04\data_archive\gt_train.txt"],
        train_roots=[r"C:\shared\Archive_19_04\data_archive"],
        val_csvs=[r"C:\shared\Archive_19_04\data_archive\gt_test.txt"],
        val_roots=[r"C:\shared\Archive_19_04\data_archive"],
        img_height=60,
        img_max_width=240,
        batch_size=48,
        epochs=60,
        lr=1e-3,
        alpha=0.3,
        decode_train="greedy",
        lm_alpha=0.3,
        lm_path="char6gram.pkl",
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir="runs/exp",
        checkpoint_dir="checkpoints/exp_1",
        min_char_freq=30,
        case_insensitive=True,
        freeze_cnn_layers=[
            "scaleA.0",  # conv1
            "scaleA.4",  # layer1
            "scaleA.5",  # layer2
            "scaleB.0",  # conv1
            "scaleB.4",  # layer1
            "scaleB.5",  # layer2
        ],
    )
    run_training(config)
