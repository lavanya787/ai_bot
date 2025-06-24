import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from intent.classifier import BiLSTMClassifier
import json
import re
class Memory:
    def __init__(self):
        self.history = []
    def append(self, role, message):
        self.history.append({"role": role, "message": message})
    def get_formatted(self):
        return "\n".join(f"{entry['role']}: {entry['message']}" for entry in self.history)
    def clear(self):
        self.history = []
    
# Training data
data = [
    ("What are the marks of student A?", "file", "neutral"),
    ("Tell me about the history of India.", "general", "neutral"),
    ("This is so confusing!", "general", "negative"),
    ("Summarize chapter 3 from the uploaded file.", "file", "neutral"),
    ("You are doing great!", "general", "positive")
]

intent_labels = {"file": 0, "general": 1}
sentiment_labels = {"positive": 0, "neutral": 1, "negative": 2}

# Tokenization
tokens = set()
for q, _, _ in data:
    tokens.update(re.findall(r"\b\w+\b", q.lower()))
tokens = sorted(tokens | {"<unk>"})
stoi = {w: i for i, w in enumerate(tokens)}

# Save vocab
with open("intent/intent_vocab.json", "w") as f:
    json.dump({
        "stoi": stoi,
        "intent_labels": {v: k for k, v in intent_labels.items()},
        "sentiment_labels": {v: k for k, v in sentiment_labels.items()}
    }, f)

# Dataset
class IntentDataset(Dataset):
    def __init__(self, data, task="intent"):
        self.data = data
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, intent, sentiment = self.data[idx]
        x = torch.tensor([stoi.get(tok, stoi["<unk>"]) for tok in re.findall(r"\b\w+\b", q.lower())])
        y = torch.tensor(intent_labels[intent] if self.task == "intent" else sentiment_labels[sentiment])
        return x, y

# Pad batch
def collate(batch):
    inputs, labels = zip(*batch)
    lengths = [len(x) for x in inputs]
    max_len = max(lengths)
    padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
    for i, seq in enumerate(inputs):
        padded[i, :lengths[i]] = seq
    return padded, torch.tensor(labels)

# Train intent model
intent_dataset = IntentDataset(data, task="intent")
intent_loader = DataLoader(intent_dataset, batch_size=2, shuffle=True, collate_fn=collate)
intent_model = BiLSTMClassifier(len(stoi), output_dim=2)
optimizer = torch.optim.Adam(intent_model.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    for x, y in intent_loader:
        optimizer.zero_grad()
        pred = intent_model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

torch.save(intent_model.state_dict(), "intent/intent_model.pth")

# Train sentiment model
sentiment_dataset = IntentDataset(data, task="sentiment")
sentiment_loader = DataLoader(sentiment_dataset, batch_size=2, shuffle=True, collate_fn=collate)
sentiment_model = BiLSTMClassifier(len(stoi), output_dim=3)
optimizer = torch.optim.Adam(sentiment_model.parameters())

for epoch in range(5):
    for x, y in sentiment_loader:
        optimizer.zero_grad()
        pred = sentiment_model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

torch.save(sentiment_model.state_dict(), "intent/sentiment_model.pth")
