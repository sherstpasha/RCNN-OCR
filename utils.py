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
    if isinstance(out_probs, tuple):
        out_probs = out_probs[0]
    indices = out_probs.argmax(dim=2)  # [T, B]
    seqs = indices.permute(1, 0).tolist()
    preds, raws = [], []
    max_idx = len(alphabet)
    for s in seqs:
        raw = " ".join(alphabet[i - 1] if (i > 0 and i <= max_idx) else "-" for i in s)
        raws.append(raw)
        coll, prev = [], None
        for i in s:
            if i > 0 and i <= max_idx and i != prev:
                coll.append(alphabet[i - 1])
            prev = i
        preds.append("".join(coll))
    return preds, raws


def beam_search_ctc(out_probs, alphabet: str, beam_width: int = 5):
    """
    Beam Search decoding for CTC output.

    Args:
        out_probs: Tensor [T, B, C] — логиты CTC
        alphabet: строка с символами
        beam_width: ширина поиска

    Returns:
        Списки строковых предсказаний и "сырых" последовательностей
    """
    if isinstance(out_probs, tuple):
        out_probs = out_probs[0]
    log_probs = F.log_softmax(out_probs, dim=2)  # [T, B, C]
    T, B, C = log_probs.shape
    preds, raws = [], []

    for b in range(B):
        beam = [([], 0.0)]  # (sequence, log_prob)

        for t in range(T):
            new_beam = []
            for seq, score in beam:
                for c in range(C):
                    new_seq = seq + [c]
                    new_score = score + log_probs[t, b, c].item()
                    new_beam.append((new_seq, new_score))
            # Сохраняем top-K гипотез
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]

        # Берём лучший результат
        best_seq, _ = beam[0]

        # raw output
        raw = " ".join(
            alphabet[i - 1] if i > 0 and i <= len(alphabet) else "-" for i in best_seq
        )
        raws.append(raw)

        # collapse repeats + remove blank (0)
        final = []
        prev = None
        for i in best_seq:
            if i > 0 and i <= len(alphabet) and i != prev:
                final.append(alphabet[i - 1])
            prev = i
        preds.append("".join(final))

    return preds, raws


def beam_search_attention(
    model,
    img,
    inv_vocab,
    sos_idx,
    eos_idx,
    beam_width: int = 5,
    max_len: int = 80,
    device: str = "cuda",
) -> str:
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


def decode(ctc_out, alphabet, method="beam", beam_width=5):
    if method == "greedy":
        return ctc_greedy_decoder(ctc_out, alphabet)
    elif method == "beam":
        return beam_search_ctc(ctc_out, alphabet, beam_width)
    else:
        raise ValueError(f"Unknown decoding method: {method}")
