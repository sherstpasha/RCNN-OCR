# metrics.py

import Levenshtein as lev
from jiwer import wer as jiwer_wer


def character_error_rate(reference: str, hypothesis: str) -> float:
    if len(reference) == 0:
        return float("inf") if len(hypothesis) > 0 else 0.0
    dist = lev.distance(reference, hypothesis)
    cer = dist / len(reference)
    return min(cer, 1.0)       

def word_error_rate(reference: str, hypothesis: str) -> float:
    wer = jiwer_wer(reference, hypothesis)
    return min(wer, 1.0) 


def compute_accuracy(references: list[str], hypotheses: list[str]) -> float:
    """
    Простая точность: доля точных совпадений (prediction == reference).
    """
    total = len(references)
    if total == 0:
        return 0.0
    hits = sum(1 for r, h in zip(references, hypotheses) if r == h)
    return hits / total
