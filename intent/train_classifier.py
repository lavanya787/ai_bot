# intent/train_classifier.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from intent.classifier import TransformerIntentClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datetime import datetime

# ----------- CONFIG -----------
MAX_VOCAB_SIZE = 80000
EMBEDDING_DIM = 512
HIDDEN_SIZE = 768
NUM_LAYERS = 6
NUM_HEADS = 6
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-4
DROPOUT = 0.2
GRAD_CLIP = 1.0

INTENT_MODEL_PATH = 'intent/intent_model.pth'
VOCAB_PATH = 'intent/intent_vocab.json'
TOKENIZER_PATH = 'intent/vocab.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- UTILITIES -----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def tokenize(text):
    return clean_text(text).split()

# ----------- BUILD VOCAB -----------
def build_vocab(texts):
    word_freq = {}
    for text in texts:
        for token in tokenize(text):
            word_freq[token] = word_freq.get(token, 0) + 1
    sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:MAX_VOCAB_SIZE - 2]
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(sorted_vocab, start=2):
        word2idx[word] = idx
    with open(TOKENIZER_PATH, 'w') as f:
        json.dump(word2idx, f)
    return word2idx

def encode(text, word2idx):
    return [word2idx.get(token, word2idx["<UNK>"]) for token in tokenize(text)][:MAX_SEQ_LENGTH]

# ----------- DATASET -----------
class IntentDataset(Dataset):
    def __init__(self, texts, labels, word2idx, label2id):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode(self.texts[idx], self.word2idx)
        x += [0] * (MAX_SEQ_LENGTH - len(x))  # Pad
        y = self.label2id[self.labels[idx]]
        return torch.tensor(x), torch.tensor(y)

# ----------- TRAINING -----------
def train_loop(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, label_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds).tolist()

    # Log results
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    os.makedirs("logs/intent_metrics", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"logs/intent_metrics/intent_eval_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return accuracy
# ----------- MAIN FINE-TUNE ENTRY -----------
def train_model(df: pd.DataFrame):
    assert "sentence" in df.columns and "intent" in df.columns, "DataFrame must contain 'sentence' and 'intent' columns."

    texts, labels = df["sentence"].tolist(), df["intent"].tolist()
    label2id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    with open(VOCAB_PATH, 'w') as f:
        json.dump(label2id, f)

    word2idx = build_vocab(texts)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_ds = IntentDataset(X_train, y_train, word2idx, label2id)
    val_ds = IntentDataset(X_val, y_val, word2idx, label2id)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = TransformerIntentClassifier(
        vocab_size=len(word2idx),
        embed_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_SIZE,
        num_classes=len(label2id),
        num_layers=NUM_LAYERS,
        max_len=MAX_SEQ_LENGTH
    ).to(device)

    # 🔁 Load previous weights if available
    if os.path.exists(INTENT_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(INTENT_MODEL_PATH))
            print("🔁 Loaded existing model weights for fine-tuning.")
        except Exception as e:
            print(f"⚠️ Could not load previous model weights: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        loss = train_loop(model, train_loader, optimizer, criterion)
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), INTENT_MODEL_PATH)
    print(f"✅ Model fine-tuned and saved to {INTENT_MODEL_PATH}")

if __name__ == "__main__":
    train_model()
