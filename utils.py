import torch
from model import RCNN
import torch.nn.functional as F


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
        # (T, B, C) → (B, T, C)
        logits = logits.permute(1, 0, 2)

    B, T, C = logits.shape
    preds = logits.argmax(dim=2)  # (B, T)

    texts, seqs = [], []
    for b in range(B):
        prev = blank
        seq, chars = [], []
        for t in range(T):
            p = preds[b, t].item()
            if p != blank and p != prev:
                seq.append(p)
                chars.append(alphabet[p - 1])  # сдвиг, т.к. blank=0
            prev = p
        texts.append("".join(chars))
        seqs.append(seq)
    return texts, seqs


def decode(ctc_out, alphabet: str, method: str = "greedy"):
    """
    Декодирование выхода CTC.
    Args:
        ctc_out: выход модели (T, B, C) или (B, T, C)
        alphabet: строка символов
        method: 'greedy' (по умолчанию)
    """
    if isinstance(ctc_out, tuple):
        ctc_out = ctc_out[0]

    log_probs = F.log_softmax(ctc_out, dim=-1)

    if method == "greedy":
        return ctc_greedy_decoder(log_probs, alphabet)
    else:
        raise ValueError(f"Unsupported decode method: {method}")
