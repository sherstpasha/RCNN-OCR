import torch
from model import CRNN
import torch.nn.functional as F


def load_crnn(
    checkpoint_path: str,
    img_height: int,
    img_width: int,
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
        img_width: ширина входного изображения (максимальная ширина для pooling/CTC)
        num_classes: число классов (включая blank для CTC и спецтокены)
        device: torch.device, куда загрузить модель
        pretrained: использовать ImageNet-веса для бэкбона
        transform: 'none', 'affine' или 'tps'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(
        img_height, img_width, num_classes, pretrained=pretrained, transform=transform
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def ctc_greedy_decoder(out_probs, alphabet: str) -> (list[str], list[str]):
    """
    Greedy decoder for CTC outputs.
    Принимает Tensor [T, B, C] или кортеж (ctc_out, attn_out).
    Возвращает два списка длины B: (predictions, raw_sequences).
    """
    # Если модель вернула tuple, достаём CTC-логиты
    if isinstance(out_probs, tuple):
        out_probs = out_probs[0]

    indices = out_probs.argmax(dim=2)  # [T, B]
    seqs = indices.permute(1, 0).tolist()

    preds, raws = [], []
    max_idx = len(alphabet)
    for s in seqs:
        # Raw sequence: '-' for blanks and out-of-range indices
        raw = " ".join(alphabet[i - 1] if (i > 0 and i <= max_idx) else "-" for i in s)
        raws.append(raw)
        # Collapse repeats and filter blanks/out-of-range
        coll, prev = [], None
        for i in s:
            if i > 0 and i <= max_idx and i != prev:
                coll.append(alphabet[i - 1])
            prev = i
        preds.append("".join(coll))
    return preds, raws


def beam_search_decode(
    model,
    img,
    inv_vocab,
    sos_idx,
    eos_idx,
    beam_width: int = 5,
    max_len: int = 80,
    device: str = "cuda",
) -> str:
    """
    Beam-search decoding with attention decoder.

    Args:
        model: ваша CRNN+Attention модель
        img: тензор [1,1,H,W]
        inv_vocab: dict idx->token
        sos_idx: индекс <sos>
        eos_idx: индекс <eos>
        beam_width: ширина beam
        max_len: макс. длина последовательности
        device: устройство

    Returns:
        Предсказанная строка (без <sos>, <eos>).
    """
    model.eval()
    img = img.to(device)
    beams = [(0.0, [sos_idx])]
    completed = []
    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            for score, seq in beams:
                if seq[-1] == eos_idx:
                    completed.append((score, seq))
                    continue
                decoder_input = torch.tensor([seq], device=device)
                _, attn_out = model(img, decoder_inputs=decoder_input)
                logp = F.log_softmax(attn_out[0, -1], dim=-1)
                topk = torch.topk(logp, beam_width)
                for lp, idx in zip(topk.values, topk.indices):
                    new_beams.append((score + lp.item(), seq + [idx.item()]))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
            if not beams:
                break
    best_seq = max(completed or beams, key=lambda x: x[0])[1]
    return "".join(inv_vocab[i] for i in best_seq if i not in (sos_idx, eos_idx))
