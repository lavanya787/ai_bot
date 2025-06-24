# intent/train_classifier.py
import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from intent.classifier import BiLSTMClassifier
from utils.text_utils import tokenize_text
from evaluation.evaluate import smart_evaluate

class IntentDataset(Dataset):
    def __init__(self, data, stoi, label_map, task="intent"):
        self.data = data
        self.stoi = stoi
        self.label_map = label_map
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = tokenize_text(row["query"])
        x = torch.tensor([self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens])
        y = torch.tensor(self.label_map[row[self.task]])
        return x, y

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels)

def train_from_df(df, epochs=5, batch_size=4):
    if not all(col in df.columns for col in ["query", "intent", "sentiment"]):
        raise ValueError("DataFrame must contain 'query', 'intent', and 'sentiment' columns")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if df["intent"].nunique() < 1 or df["sentiment"].nunique() < 1:
        raise ValueError("At least one unique intent and sentiment required")

    tokens = set(["<unk>"])
    for q in df["query"]:
        tokens.update(tokenize_text(q))
    stoi = {w: i for i, w in enumerate(sorted(tokens))}

    unique_intents = sorted(df["intent"].unique().tolist())
    unique_sentiments = sorted(df["sentiment"].unique().tolist())

    intent_labels = {label: i for i, label in enumerate(unique_intents)}
    sentiment_labels = {label: i for i, label in enumerate(unique_sentiments)}
    intent_map = {str(i): label for i, label in enumerate(unique_intents)}
    sentiment_map = {str(i): label for i, label in enumerate(unique_sentiments)}

    embedding_dim = 64
    hidden_dim = 128

    intent_model = BiLSTMClassifier(input_dim=len(stoi), embedding_dim=embedding_dim,
                                    hidden_dim=hidden_dim, output_dim=len(intent_labels))
    sentiment_model = BiLSTMClassifier(input_dim=len(stoi), embedding_dim=embedding_dim,
                                       hidden_dim=hidden_dim, output_dim=len(sentiment_labels))

    criterion = nn.CrossEntropyLoss()
    intent_opt = torch.optim.Adam(intent_model.parameters(), lr=0.001)
    sentiment_opt = torch.optim.Adam(sentiment_model.parameters(), lr=0.001)

    intent_ds = IntentDataset(df, stoi, intent_labels, task="intent")
    sentiment_ds = IntentDataset(df, stoi, sentiment_labels, task="sentiment")
    intent_loader = DataLoader(intent_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    sentiment_loader = DataLoader(sentiment_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    i_losses, s_losses = [], []

    for epoch in range(epochs):
        intent_model.train()
        sentiment_model.train()
        i_total, s_total = 0, 0

        for x, y in intent_loader:
            intent_opt.zero_grad()
            out = intent_model(x)
            loss = criterion(out, y)
            loss.backward()
            intent_opt.step()
            i_total += loss.item()

        for x, y in sentiment_loader:
            sentiment_opt.zero_grad()
            out = sentiment_model(x)
            loss = criterion(out, y)
            loss.backward()
            sentiment_opt.step()
            s_total += loss.item()

        i_losses.append(i_total / len(intent_loader))
        s_losses.append(s_total / len(sentiment_loader))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = f"intent/{timestamp}"
    os.makedirs(base_path, exist_ok=True)

    torch.save(intent_model.state_dict(), f"{base_path}/intent_model.pth")
    torch.save(sentiment_model.state_dict(), f"{base_path}/sentiment_model.pth")

    with open(f"{base_path}/intent_vocab.json", "w") as f:
        json.dump({
            "stoi": stoi,
            "intent_labels": intent_map,
            "sentiment_labels": sentiment_map
        }, f)

    with open("intent/latest_model.json", "w") as f:
        json.dump({"latest_path": base_path}, f)

    model_config = {
        "intent": {
            "input_dim": len(stoi),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "output_dim": len(intent_labels)
        },
        "sentiment": {
            "input_dim": len(stoi),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "output_dim": len(sentiment_labels)
        }
    }

    with open(f"{base_path}/model_config.json", "w") as f:
        json.dump(model_config, f)

    # Save loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(i_losses, label="Intent Loss")
    plt.plot(s_losses, label="Sentiment Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_path}/loss_plot.png")
    plt.close()

    # Evaluate and save metrics
    intent_eval = smart_evaluate("lstm_classification", model=intent_model,
                                 dataloader=intent_loader, label_names=list(intent_labels.keys()))
    sentiment_eval = smart_evaluate("lstm_classification", model=sentiment_model,
                                    dataloader=sentiment_loader, label_names=list(sentiment_labels.keys()))

    metrics = {
        "intent": intent_eval,
        "sentiment": sentiment_eval,
        "timestamp": timestamp
    }

    with open(f"{base_path}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return intent_model, sentiment_model, stoi, intent_map, sentiment_map, i_losses, s_losses
