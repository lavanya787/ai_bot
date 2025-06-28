import os
import json
import torch
import re
import numpy as np
import pickle
from models.transformer_generator import TransformerModel
from vector_db.faiss_indexer import VectorDB
from file_processing.processor import chunk_text
import pandas as pd

# -------- CONFIG --------
VOCAB_FILE = "tokenizer/vocab.json"
FALLBACK_CORPUS = "training_data/qa_corpus.csv"
EMBEDDING_PATH = "embedding_store/embeddings.npy"
CHUNK_PATH = "embedding_store/chunks.pkl"

# -------- BUILD VOCAB LOCALLY --------
def build_vocab(corpus, max_vocab_size=80000, min_freq=1):
    word_freq = {}
    for text in corpus:
        text = re.sub(r"[^\w\s]", "", text.lower())
        for token in text.split():
            word_freq[token] = word_freq.get(token, 0) + 1

    sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in sorted_vocab:
        if freq >= min_freq and len(vocab) < max_vocab_size:
            vocab[word] = len(vocab)
    return vocab

# -------- LOAD OR BUILD VOCAB --------
def load_vocab():
    if not os.path.exists(VOCAB_FILE):
        print("⚠️ tokenizer/vocab.json not found. Building vocab from fallback corpus...")
        df = pd.read_csv(FALLBACK_CORPUS)

        aliases = {
            "question": ["question", "query", "prompt", "input"],
            "answer": ["answer", "response", "output"]
        }

        def find_column(possible_names, columns):
            for name in possible_names:
                for col in columns:
                    if name.lower() == col.lower():
                        return col
            return None

        question_col = find_column(aliases["question"], df.columns)
        answer_col = find_column(aliases["answer"], df.columns)

        if not question_col or not answer_col:
            raise ValueError("CSV must have 'question' and 'answer' columns (case-insensitive).")

        texts = df[question_col].astype(str).tolist() + df[answer_col].astype(str).tolist()

        vocab = build_vocab(texts)
        os.makedirs("tokenizer", exist_ok=True)
        with open(VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print("✅ Vocab rebuilt and saved to tokenizer/vocab.json.")
    else:
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    return vocab

# -------- EMBEDDING --------
def get_embedding(text, vocab, config):
    model = TransformerModel(vocab, config)
    tokens = model.encode_prompt(text)
    with torch.no_grad():
        outputs = model.model(tokens)
    return outputs.mean(dim=1).squeeze().cpu().numpy()

# -------- QA HANDLER --------
class QAHandler:
    def __init__(self):
        self.vocab = load_vocab()
        self.config = {
            "device": "cpu",
            "d_model": 128,
            "n_layers": 2,
            "num_heads": 4,
            "ff_dim": 256,
            "max_len": 64,
        }
        self.model = TransformerModel(self.vocab, self.config)
        self.retriever = self._load_vector_db()

    def _load_vector_db(self):
        if not os.path.exists(EMBEDDING_PATH) or not os.path.exists(CHUNK_PATH):
            raise FileNotFoundError("Missing embeddings/chunks — run training first.")
        embeddings = np.load(EMBEDDING_PATH).astype("float32")
        with open(CHUNK_PATH, "rb") as f:
            chunks = pickle.load(f)

        db = VectorDB(dim=embeddings.shape[1])
        db.add(embeddings, chunks)
        return db

    def answer(self, question: str, top_k: int = 3) -> str:
        query_vector = get_embedding(question, self.vocab, self.config)
        results = self.retriever.search(query_vector, k=top_k)
        if not results:
            return "🔍 No relevant information found in documents."

        context = "\n".join([r["text"] for r in results])
        prompt = f"Q: {question}\n\n{context}"

        tokens = self.model.encode_prompt(prompt)
        with torch.no_grad():
            output = self.model.model(tokens)

        return f"🧠 Based on the documents:\n\n{context}\n\n(Transformer response simulated)"
