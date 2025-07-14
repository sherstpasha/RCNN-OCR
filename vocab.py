"""
vocab.py

Определяет специальные токены и функции для построения словаря (token ↔ index).
"""

from typing import Tuple, Dict

# Специальные токены и их индексы
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


def build_vocab(alphabet: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Строит словарь из специального токена и символов алфавита.

    Args:
        alphabet: строка символов, например 'абвгд...'.

    Returns:
        vocab: словарь token -> index.
        inv_vocab: словарь index -> token.
    """
    # Инициализация с специальными токенами
    vocab: Dict[str, int] = dict(SPECIAL_TOKENS)
    # Начинаем нумерацию символов сразу после спецтокенов
    next_index = max(SPECIAL_TOKENS.values()) + 1

    # Добавляем каждый символ алфавита
    for ch in alphabet:
        if ch not in vocab:
            vocab[ch] = next_index
            next_index += 1

    # Обратный словарь
    inv_vocab: Dict[int, str] = {idx: tok for tok, idx in vocab.items()}
    return vocab, inv_vocab
