# train_general_qa.py

import sys, os
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn
import pandas as pd
import pickle
import json
from torch.utils.data import Dataset, DataLoader
from models.qa_model import SimpleQAModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
from utils.domain_detector import detect_domain
# === CONFIG ===
MAX_LEN = 20
DATA_DIR = "training_data"
MODEL_DIR = "saved_models/general"
CSV_PATH = os.path.join(DATA_DIR, "qa_corpus.csv")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
RESULT_PATH = os.path.join(MODEL_DIR, "results.json")

# === TOKENIZE / VOCAB ===
def tokenize(text):
    return text.lower().split()

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
    return ids[:MAX_LEN] + [0] * (MAX_LEN - len(ids))

# === DATASET ===
class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.data = df
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = encode(row['context'], self.vocab)
        y = int(row['is_answer'])
        return torch.tensor(x), torch.tensor(y)

# === CREATE SAMPLE CSV IF MISSING ===
def create_sample_csv():
    print("📁 Creating sample training_data/qa_corpus.csv ...")
    sample_data = [
        ["What is Newton's first law?", "An object in motion stays in motion unless acted upon.", 1],
        ["What is Newton's first law?", "Water boils at 100 degrees Celsius.", 0],
        ["What is gravity?", "Gravity is the force that attracts objects toward each other.", 1],
        ["What is gravity?", "Mitosis is a process of cell division.", 0],
        ["What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight.", 1],
        ["What is photosynthesis?", "Albert Einstein developed the theory of relativity.", 0],
        ["What is an atom?", "An atom is the smallest unit of ordinary matter.", 1],
        ["What is an atom?", "Evolution occurs through natural selection.", 0]
    ]
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame(sample_data, columns=["question", "context", "is_answer"])
    df.to_csv(CSV_PATH, index=False)

# === TRAINING PIPELINE ===
def train_general_model(df, file_path):
    domain = detect_domain(' '.join(df['context'].tolist()))
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    model_dir = os.path.join("saved_models", domain, model_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"📂 Saving model under: {model_dir}")

    vocab = build_vocab(list(df['question']) + list(df['context']))
    pickle.dump(vocab, open(os.path.join(model_dir, "vocab.pkl"), "wb"))

    train_df, val_df = train_test_split(df, test_size=0.4, stratify=df['is_answer'])

    train_ds = QADataset(train_df, vocab)
    val_ds = QADataset(val_df, vocab)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = SimpleQAModel(len(vocab))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            output = model(x).squeeze()
            loss = criterion(output, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"✅ Epoch {epoch + 1} complete")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            pred = torch.round(torch.sigmoid(model(x).squeeze()))
            all_preds.extend(pred.view(-1).tolist())
            all_labels.extend(y.view(-1).tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump({'accuracy': acc, 'f1_score': f1}, f, indent=4)

    print(f"✅ Model saved to {model_dir} | Accuracy: {acc:.3f}, F1: {f1:.3f}")
