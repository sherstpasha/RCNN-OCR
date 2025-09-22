import torch
from model import RCNN
import torch.nn.functional as F
import random


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_val_loss,
    best_val_acc,
    itos,
    stoi,
    config,
    log_dir,
):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "itos": itos,
        "stoi": stoi,
        "config": config,
        "log_dir": log_dir,
    }
    torch.save(ckpt, path)


def save_weights(path, model):
    torch.save(model.state_dict(), path)


def load_checkpoint(
    path, model, optimizer=None, scheduler=None, scaler=None, map_location="auto"
):
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_crnn(
    checkpoint_path: str,
    img_height: int,
    img_width: int,
    num_classes: int,
    device: torch.device = None,
    pretrained: bool = False,
    transform: str = "none",
) -> RCNN:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNN(
        img_height, img_width, num_classes, pretrained=pretrained, transform=transform
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def ctc_greedy_decoder(logits: torch.Tensor, alphabet: str, blank: int = 0):
    """
    Greedy decoder для CTC.
    Args:
        logits: (T, B, C) или (B, T, C) — выход модели
        alphabet: строка символов
        blank: индекс blank (обычно 0)
    Returns:
        список предсказанных строк, список индексов
    """
    if logits.dim() == 3 and logits.shape[0] < logits.shape[1]:
        logits = logits.permute(1, 0, 2)

    B, T, C = logits.shape
    preds = logits.argmax(dim=2)

    texts, seqs = [], []
    for b in range(B):
        prev = blank
        seq, chars = [], []
        for t in range(T):
            p = preds[b, t].item()
            if p != blank and p != prev:
                seq.append(p)
                chars.append(alphabet[p - 1]) 
            prev = p
        texts.append("".join(chars))
        seqs.append(seq)
    return texts, seqs


def decode(ctc_out, alphabet: str, method: str = "greedy"):
    if isinstance(ctc_out, tuple):
        ctc_out = ctc_out[0]

    log_probs = F.log_softmax(ctc_out, dim=-1)

    if method == "greedy":
        return ctc_greedy_decoder(log_probs, alphabet)
    else:
        raise ValueError(f"Unsupported decode method: {method}")
