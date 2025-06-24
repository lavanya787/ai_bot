import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple, Dict


# ----- Dataset -----
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [torch.tensor([vocab.get(tok, vocab["<unk>"]) for tok in text.split()]) for text in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# ----- Positional Encoding -----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# ----- Transformer Classifier -----
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=2, dim_feedforward=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        mask = (x == 0)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        pooled = encoded.mean(dim=0)
        return self.fc(pooled)


# ----- Vocab Builder -----
def build_vocab(texts, min_freq=1) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


# ----- Padding Function -----
def pad_batch(batch):
    inputs, labels = zip(*batch)
    max_len = max(len(x) for x in inputs)
    padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
    for i, x in enumerate(inputs):
        padded[i, :len(x)] = x
    return padded, torch.tensor(labels)


# ----- Training Function -----
def train_transformer(train_df, model_name="scratch_model", domain="general", save_model=True, save_path=None) -> Tuple[nn.Module, Dict[str, int], Dict[str, int], Dict]:
    texts = train_df['text'].tolist()
    labels_raw = train_df['label'].tolist()

    label_set = sorted(set(labels_raw))
    label_to_idx = {label: i for i, label in enumerate(label_set)}
    labels = [label_to_idx[label] for label in labels_raw]

    vocab = build_vocab(texts)
    dataset = TextDataset(texts, labels, vocab)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=pad_batch)

    model = TransformerClassifier(vocab_size=len(vocab), num_classes=len(label_set))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = [b.to(device) for b in batch]
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Train Loss = {total_loss:.4f}")

    # Evaluation
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = [b.to(device) for b in batch]
            out = model(x)
            pred = out.argmax(1)
            preds.extend(pred.cpu().numpy())
            truths.extend(y.cpu().numpy())

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='weighted')
    print(f"✅ Accuracy: {acc:.2f} | F1: {f1:.2f}")

    if save_model:
        save_dir = f"trained_models/{domain}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pt"))
        with open(os.path.join(save_dir, f"{model_name}_vocab.json"), "w") as f:
            json.dump(vocab, f)
        with open(os.path.join(save_dir, f"{model_name}_labels.json"), "w") as f:
            json.dump(label_to_idx, f)

    return model, vocab, label_to_idx, {"accuracy": acc, "f1_score": f1}
def predict_with_transformer(text: str, model_path: str, model_name: str = "scratch_model") -> str:
    """
    Predict the label for a given input text using a trained Transformer model.

    Args:
        text (str): Input text for classification.
        model_path (str): Path to trained model folder (e.g., "trained_models/general")
        model_name (str): Model file name without extension (e.g., "scratch_model")

    Returns:
        str: Predicted label.
    """
    # Load vocab and labels
    with open(os.path.join(model_path, f"{model_name}_vocab.json"), "r") as f:
        vocab = json.load(f)
    with open(os.path.join(model_path, f"{model_name}_labels.json"), "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Build model
    model = TransformerClassifier(vocab_size=len(vocab), num_classes=len(label_to_idx))
    model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}.pt"), map_location="cpu"))
    model.eval()

    # Preprocess
    tokens = text.lower().split()
    token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    return idx_to_label[pred_idx]
