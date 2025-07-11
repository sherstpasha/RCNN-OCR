# utils.py

import torch
from model import CRNN


def load_crnn(
    checkpoint_path: str,
    img_height: int,
    num_classes: int,
    device: torch.device = None,
    pretrained: bool = False,
    transform: str = "none",
) -> CRNN:
    """
    Создаёт CRNN с опциональным STN/TPS, загружает веса из checkpoint и переводит сеть в eval-режим.

    Args:
        checkpoint_path: путь к файлу state_dict
        img_height: высота входного изображения
        num_classes: число классов (включая blank)
        device: torch.device, куда загрузить модель
        pretrained: использовать ImageNet-веса для бэкбона
        transform: 'none', 'affine' или 'tps'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(
        img_height, num_classes, pretrained=pretrained, transform=transform
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Загружаем строго, ключи должны соответствовать, поэтому transform должен совпадать
    model.load_state_dict(state)
    model.eval()
    return model


def ctc_greedy_decoder(
    out_probs: torch.Tensor, alphabet: str
) -> (list[str], list[str]):
    """
    Greedy decoder for CTC outputs.
    out_probs: Tensor of shape [T, B, C]
    Returns two lists of length B: (predictions, raw_sequences).
    """
    indices = out_probs.argmax(dim=2)
    seqs = indices.permute(1, 0).tolist()

    preds, raws = [], []
    for s in seqs:
        raw = " ".join([alphabet[i - 1] if i > 0 else "-" for i in s])
        raws.append(raw)
        coll, prev = [], None
        for i in s:
            if i != prev and i > 0:
                coll.append(alphabet[i - 1])
            prev = i
        preds.append("".join(coll))
    return preds, raws
