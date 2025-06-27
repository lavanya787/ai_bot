import json

def __init__(self, vocab=None):
    self.vocab = vocab or {}
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
def build_vocab(self, corpus):
    tokens = set()
    for line in corpus:
        tokens.update(line.strip().split())
    self.vocab = {token: idx + 3 for idx, token in enumerate(sorted(tokens))}
    self.vocab["<pad>"] = 0
    self.vocab["<unk>"] = 1
    self.vocab["<eos>"] = 2
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
def encode(self, text):
    return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in text.strip().split()]
def decode(self, ids):
    return " ".join([self.inv_vocab.get(i, "<unk>") for i in ids])
def save_vocab(self, path):
    with open(path, "w") as f:
        json.dump(self.vocab, f)
def load_vocab(self, path):
    with open(path, "r") as f:
        self.vocab = json.load(f)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}