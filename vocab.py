# vocab.py

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
}

def build_vocab(alphabet: str):
    vocab = {ch: i + 3 for i, ch in enumerate(alphabet)}
    inv_vocab = {i: ch for ch, i in vocab.items()}
    vocab = {**SPECIAL_TOKENS, **vocab}
    inv_vocab.update({v: k for k, v in SPECIAL_TOKENS.items()})
    return vocab, inv_vocab
