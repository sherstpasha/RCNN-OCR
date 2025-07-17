import torch
from model import CRNN
import torch.nn.functional as F

from collections import Counter, defaultdict
import math
import pickle


class CharNGramLM:
    def __init__(self, N: int, alpha: float = 1.0):
        """
        N     — порядок модели (например, 6 для 6‑грамм)
        alpha — параметр сглаживания (1.0 = add‑1)
        """
        self.N = N
        self.alpha = alpha
        # counts[n-1][gram] = число вхождений n‑граммы
        self.counts = [Counter() for _ in range(N)]
        # context_counts[n-1][prefix] = число вхождений (n‑1)‑грамм, т.е. «контекста»
        self.context_counts = [Counter() for _ in range(N)]
        self.vocab = set()

    def train(self, words: list[str]):
        """Считает все n‑граммы по списку словоформ."""
        for w in words:
            chars = list(w.strip())
            for i, c in enumerate(chars):
                self.vocab.add(c)
                # от 1‑грамм до N‑грамм, которые поместятся
                for n in range(1, self.N + 1):
                    if i - n + 1 < 0:
                        break
                    gram = tuple(chars[i - n + 1 : i + 1])
                    ctx = gram[:-1]
                    self.counts[n - 1][gram] += 1
                    self.context_counts[n - 1][ctx] += 1

    def log_prob(self, word: str) -> float:
        """Возвращает log‑вероятность целого слова как сумму условных вероятностей."""
        chars = list(word.strip())
        V = len(self.vocab)
        logp = 0.0
        for i, c in enumerate(chars):
            # используем самый большой контекст ≤ N‑1
            for n in range(self.N, 0, -1):
                if i - n + 1 < 0:
                    continue
                gram = tuple(chars[i - n + 1 : i + 1])
                ctx = gram[:-1]
                num = self.counts[n - 1][gram] + self.alpha
                den = self.context_counts[n - 1][ctx] + self.alpha * V
                logp += math.log(num) - math.log(den)
                break
        return logp

    def cond_log_prob(self, context: list[str], next_char: str) -> float:
        """
        Возвращает лог‑вероятность next_char при данном контексте.
        context — список предыдущих символов (может быть длиннее N‑1, используется только последние N‑1).
        """
        # Берём актуальный контекст длиной не более N-1
        ctx_list = context[-(self.N - 1) :] if self.N > 1 else []
        V = len(self.vocab)
        # Пробуем самый длинный доступный контекст
        for n in range(len(ctx_list) + 1, 0, -1):
            gram = tuple(ctx_list[-(n - 1) :] + [next_char]) if n > 1 else (next_char,)
            ctx = gram[:-1]
            num = self.counts[n - 1].get(gram, 0) + self.alpha
            den = self.context_counts[n - 1].get(ctx, 0) + self.alpha * V
            # Если ни одна n‑грамма не встречалась (num == alpha), всё равно возвращаем сглаженную
            return math.log(num) - math.log(den)
        # Фолбэк — равномерная по словарю
        return math.log(1.0 / V)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CharNGramLM":
        with open(path, "rb") as f:
            return pickle.load(f)


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


def ctc_greedy_decoder(
    out_probs, alphabet: str, beam_width: int = 5
) -> (list[str], list[str]):
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


from collections import defaultdict


import torch
import torch.nn.functional as F


def fast_simplify(seq, blank=0):
    """
    Упрощает CTC-последовательность: убирает повторы и blank
    """
    simplified = []
    prev = None
    for i in seq:
        if i != prev:
            if i != blank:
                simplified.append(i)
            prev = i
    return tuple(simplified)


import torch
import torch.nn.functional as F


def beam_search_ctc(
    ctc_out: torch.Tensor,
    alphabet: str,
    beam_width: int = 5,
    top_k_per_t: int = 10,
) -> list[str]:
    """
    Vectorized beam search CTC decoder over a batch.

    Args:
        ctc_out: лог‑пробы после модели, shape [T, B, C]
        alphabet: строка из допустимых символов (blank=0 не включается)
        beam_width: сколько гипотез держать
        top_k_per_t: сколько символов рассматривать на каждом шаге

    Returns:
        preds: список распознанных строк длины B
    """
    # 1) Переводим в log_softmax
    log_probs = F.log_softmax(ctc_out, dim=2)  # [T, B, C]
    T, B, C = log_probs.shape
    device = log_probs.device

    # 2) Предвыбираем top_k_per_t кандидатов для каждого t и b
    #    topk_probs: [T, B, top_k]; topk_idxs: [T, B, top_k]
    topk_probs, topk_idxs = torch.topk(log_probs, top_k_per_t, dim=2)

    # 3) Инициализируем seqs, scores и lengths
    seqs = torch.zeros((B, beam_width, T), dtype=torch.long, device=device)
    seq_lens = torch.zeros((B, beam_width), dtype=torch.long, device=device)
    scores = torch.full((B, beam_width), float("-inf"), device=device)
    scores[:, 0] = 0.0  # у первой гипотезы нулевая лог‑вероятность

    # 4) Основной цикл по времени
    for t in range(T):
        # расширяем каждый beam на top_k_per_t кандидатов
        # prev_scores: [B, beam_width, 1]
        prev_scores = scores.unsqueeze(-1)
        # probs на шаге t: [B, 1, top_k]
        probs_t = topk_probs[t].unsqueeze(1)
        # новые баллы: [B, beam_width, top_k]
        new_scores = prev_scores + probs_t
        # сглаживаем гипотезы в размер beam_width * top_k
        flat_scores = new_scores.view(B, -1)  # [B, beam_width * top_k]
        # берем лучшие beam_width
        top_scores, top_idx = flat_scores.topk(beam_width, dim=1)

        # из flat-индекса восстанавливаем, из какой гипотезы и какой символ
        prev_beam = top_idx // top_k_per_t  # [B, beam_width]
        char_slot = top_idx % top_k_per_t  # [B, beam_width]
        # конкретные индексы символов в исходном алфавите
        chars = topk_idxs[t].gather(1, char_slot)  # [B, beam_width]

        # 5) Обновляем seqs и seq_lens:
        # собираем предыдущие seqs из выбранных beam’ов
        # prev_seqs: [B, beam_width, T]
        prev_seqs = seqs.gather(1, prev_beam.unsqueeze(-1).expand(-1, -1, T))
        # клонируем и пишем на позицию t новые символы
        new_seqs = prev_seqs.clone()
        new_seqs[:, :, t] = chars

        # обновляем длины (просто считаем non-blank)
        prev_lens = seq_lens.gather(1, prev_beam)
        new_lens = prev_lens + (chars != 0).long()

        # сохраняем для следующей итерации
        seqs = new_seqs
        seq_lens = new_lens
        scores = top_scores

    # 6) Финальный подбор: первая (лучшая) гипотеза в каждом батче
    best_seqs = seqs[:, 0, :]  # [B, T]
    preds = []
    for b in range(B):
        seq = best_seqs[b].tolist()
        # сворачиваем blank=0 и повторения
        pred = []
        prev = None
        for idx in seq:
            if idx != prev and idx != 0:
                if 1 <= idx <= len(alphabet):
                    pred.append(alphabet[idx - 1])
            prev = idx
        preds.append("".join(pred))

    return preds


def decode(ctc_out, alphabet, method="beam", beam_width=5):
    if isinstance(ctc_out, tuple):
        ctc_out = ctc_out[0]
    log_probs = F.log_softmax(ctc_out, dim=2)

    if method == "greedy":
        return ctc_greedy_decoder(ctc_out, alphabet)
    elif method == "beam":
        preds = beam_search_ctc(log_probs, alphabet, beam_width=beam_width)
        raws = [""] * len(preds)
        return preds, raws
    else:
        raise ValueError(f"Unknown decoding method: {method}")


def fast_beam_search_topk(
    ctc_out: torch.Tensor, alphabet: str, beam_width: int = 5, top_k_per_t: int = 10
) -> list[list[tuple[float, tuple]]]:
    """
    Vectorized beam search that returns top_k_per_t hypotheses and their CTC scores for each batch element.
    Returns:
        List of length B, each is list of (score, seq_tuple) sorted by descending score.
    """
    log_probs = F.log_softmax(ctc_out, dim=2)
    T, B, C = log_probs.shape
    device = log_probs.device

    topk_probs, topk_idxs = torch.topk(log_probs, top_k_per_t, dim=2)

    # Initialize sequences and scores
    seqs = torch.zeros((B, top_k_per_t, T), dtype=torch.long, device=device)
    scores = torch.full((B, top_k_per_t), float("-inf"), device=device)
    scores[:, 0] = 0.0

    for t in range(T):
        prev_scores = scores.unsqueeze(-1)  # [B, k, 1]
        probs_t = topk_probs[t].unsqueeze(1)  # [B, 1, k]
        new_scores = prev_scores + probs_t  # [B, k, k]
        flat = new_scores.view(B, -1)  # [B, k*k]
        top_scores, top_idx = flat.topk(top_k_per_t, dim=1)

        prev_beam = top_idx // top_k_per_t
        char_slot = top_idx % top_k_per_t
        chars = topk_idxs[t].gather(1, char_slot)

        prev_seqs = seqs.gather(1, prev_beam.unsqueeze(-1).expand(-1, -1, T))
        new_seqs = prev_seqs.clone()
        new_seqs[:, :, t] = chars

        seqs = new_seqs
        scores = top_scores

    # Collect top-k for each batch
    results = []
    for b in range(B):
        # sort within batch
        sc, idxs = scores[b].sort(descending=True)
        hyps = []
        for i in range(top_k_per_t):
            hyp = seqs[b, idxs[i]].tolist()
            hyps.append((float(sc[i].item()), tuple(hyp)))
        results.append(hyps)
    return results


def beam_search_ctc_with_lm(
    ctc_out: torch.Tensor,
    alphabet: str,
    lm_model: CharNGramLM,
    alpha: float = 0.5,
    beam_width: int = 5,
    top_k_per_t: int = 10,
) -> (list[str], list[str]):
    """
    Hybrid decoding: fast vectorized beam-search (top_k_per_t) followed by LM-based reranking to beam_width.
    """
    # 1) fast beam for top candidates per batch
    candidates = fast_beam_search_topk(
        ctc_out, alphabet, beam_width=top_k_per_t, top_k_per_t=top_k_per_t
    )

    B = len(candidates)
    final_preds, final_raws = [], []

    for b in range(B):
        best_score = -float("inf")
        best_pred, best_raw = "", ""
        for ctc_score, seq in candidates[b]:
            pred, raw, prev = [], [], None
            for idx in seq:
                if idx == 0:
                    raw.append("<blank>")
                elif 1 <= idx <= len(alphabet):
                    raw.append(alphabet[idx - 1])
                else:
                    raw.append("?")
                if idx != prev and idx != 0 and 1 <= idx <= len(alphabet):
                    pred.append(alphabet[idx - 1])
                prev = idx
            pred_str = "".join(pred)
            raw_str = " ".join(raw)
            lm_score = lm_model.log_prob(pred_str)
            total_score = ctc_score + alpha * lm_score
            if total_score > best_score:
                best_score = total_score
                best_pred, best_raw = pred_str, raw_str
        final_preds.append(best_pred)
        final_raws.append(best_raw)

    return final_preds, final_raws
