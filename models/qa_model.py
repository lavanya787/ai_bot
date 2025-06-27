import os
import json
import torch
from models.transformer_generator import TransformerModel
from tokenizer.vocab_builder import build_vocab

# ✅ Path to vocab file
VOCAB_FILE = "tokenizer/vocab.json"

# ✅ Fallback file to build vocab if missing
FALLBACK_CORPUS = "training_data/qa_corpus.csv"  # or "data/sample_corpus.txt"

# Load or build vocab
def load_vocab():
    if not os.path.exists(VOCAB_FILE):
        print("⚠️ tokenizer/vocab.json not found. Building vocab from fallback corpus...")

        # Fallback: try to use QA corpus or your own sample file
        if FALLBACK_CORPUS.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(FALLBACK_CORPUS)
            texts = df["question"].astype(str).tolist() + df["answer"].astype(str).tolist()
        else:
            with open(FALLBACK_CORPUS, "r", encoding="utf-8") as f:
                texts = f.readlines()

        vocab = build_vocab(texts)
        os.makedirs("tokenizer", exist_ok=True)
        with open(VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print("✅ Vocab rebuilt and saved to tokenizer/vocab.json.")
    else:
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    return vocab

# ✅ Example: Get embedding from transformer
def get_embedding(text):
    vocab = load_vocab()
    config = {
        "device": "cpu",
        "d_model": 128,
        "n_layers": 2,
        "num_heads": 4,
        "ff_dim": 256,
        "max_len": 64,
    }
    model = TransformerModel(vocab, config)
    tokens = model.encode_prompt(text)
    with torch.no_grad():
        outputs = model.model(tokens)
    # Get average of last layer (or use [CLS] logic if you design it that way)
    embedding = outputs.mean(dim=1).squeeze().cpu().numpy()
    return embedding
