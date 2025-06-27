import json

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1
MAX_LEN = 50


def tokenize_text(text, vocab, max_len=MAX_LEN):
    tokens = text.lower().split()
    token_ids = []

    for token in tokens:
        if token in vocab:
            token_ids.append(vocab[token])
        else:
            # Fallback to byte-level encoding
            for b in token.encode("utf-8"):
                byte_token = f"byte_{b}"
                token_ids.append(vocab.get(byte_token, UNK_ID))

    token_ids = token_ids[:max_len]
    token_ids += [PAD_ID] * (max_len - len(token_ids))
    return token_ids


def detokenize(token_ids, reverse_vocab):
    tokens = []
    byte_accumulator = []

    for tid in token_ids:
        if tid == PAD_ID:
            continue
        token = reverse_vocab.get(str(tid), UNK_TOKEN)

        if token.startswith("byte_"):
            byte_value = int(token[5:])
            byte_accumulator.append(byte_value)
        else:
            if byte_accumulator:
                tokens.append(bytes(byte_accumulator).decode("utf-8", errors="ignore"))
                byte_accumulator = []
            tokens.append(token)

    if byte_accumulator:
        tokens.append(bytes(byte_accumulator).decode("utf-8", errors="ignore"))

    return " ".join(tokens)


def load_vocab(vocab_path="tokenizer/vocab.json"):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab


def build_reverse_vocab(vocab):
    return {str(v): k for k, v in vocab.items()}


# --- Optional: extend vocab with byte tokens ---
def extend_vocab_with_bytes(vocab):
    byte_tokens = {f"byte_{i}": len(vocab) + i for i in range(256) if f"byte_{i}" not in vocab}
    vocab.update(byte_tokens)
    return vocab


# --- Demo ---
if __name__ == "__main__":
    # Initial vocab
    vocab = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
        "hello": 2,
        "world": 3
    }

    # Extend for byte fallback
    vocab = extend_vocab_with_bytes(vocab)
    reverse_vocab = build_reverse_vocab(vocab)

    text = "Hello qwerty 🤖"
    ids = tokenize_text(text, vocab)
    print("Token IDs:", ids)

    decoded = detokenize(ids, reverse_vocab)
    print("Decoded:", decoded)
