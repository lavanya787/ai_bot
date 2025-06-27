import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import json
import os
import sys
from models.sentiment_model import SentimentClassifier
from tokenizer.vocab_builder import build_vocab

MAX_LEN = 50
BATCH_SIZE = 16
EPOCHS = 5
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

def tokenize(text, vocab):
    return [vocab.get(w.lower(), vocab["<UNK>"]) for w in text.split()[:MAX_LEN]]

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.data = [tokenize(t, vocab) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x + [0] * (MAX_LEN - len(x))  # pad
        return torch.tensor(x), torch.tensor(self.labels[idx])

def train_model(csv_path):
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()

    # Train/val split
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

    vocab = build_vocab(texts)
    train_dataset = SentimentDataset(texts_train, labels_train, vocab)
    val_dataset = SentimentDataset(texts_val, labels_val, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = SentimentClassifier(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📚 Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"\n🔍 Validation Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    # Save
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/sentiment_model.pt")
    os.makedirs("tokenizer", exist_ok=True)
    with open("tokenizer/vocab.json", "w") as f:
        json.dump(vocab, f)

    print("✅ Sentiment model saved to saved_models/sentiment_model.pt")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/train_sentiment.py path/to/your.csv")
    else:
        train_model(sys.argv[1])
