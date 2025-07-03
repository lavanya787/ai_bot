# train_llm.py (Plugged into Streamlit)

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import logging

MODELS_DIR = "models"
METADATA_FILE = "metadata/model_versions.json"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)

logger = logging.getLogger("llm_handler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

def generate_dummy_data(num_samples=1000):
    X = torch.randn(num_samples, 50)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(X, y)

def train_model(model, train_loader, val_loader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        y_true_train, y_pred_train = [], []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            y_true_train.extend(y_batch.tolist())
            y_pred_train.extend(torch.argmax(outputs, dim=1).tolist())

        val_loss = 0.0
        y_true_val, y_pred_val = [], []
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                y_true_val.extend(y_batch.tolist())
                y_pred_val.extend(torch.argmax(outputs, dim=1).tolist())

        acc = accuracy_score(y_true_val, y_pred_val)
        f1 = f1_score(y_true_val, y_pred_val)

        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "accuracy": acc,
            "f1_score": f1
        })

    return history

def get_next_version():
    if not os.path.exists(METADATA_FILE):
        return "v1"
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    return f"v{len(metadata)+1}"

def save_model_with_metadata(model, metrics):
    version = get_next_version()
    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    model_path = os.path.join(MODELS_DIR, f"model_{version}.pt")
    torch.save(model.state_dict(), model_path)

    entry = {
        "name": "SimpleClassifier",
        "version": version,
        "accuracy": metrics[-1]["accuracy"],
        "loss": metrics[-1]["val_loss"],
        "f1_score": metrics[-1]["f1_score"],
        "timestamp": timestamp,
        "path": model_path
    }

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = []

    metadata.append(entry)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\U0001F4CA Model Parameters: Total = {sum(p.numel() for p in model.parameters())}, Trainable = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"✅ Model saved: {model_path}")
    logger.info(f"✅ Metadata updated with version {version}")

    return version

def main(dummy=False):
    dataset = generate_dummy_data() if dummy else generate_dummy_data(2000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    model = SimpleClassifier()
    logger.info("\U0001F680 Model training started on 1 batches")
    logger.info(f"🖥️ Before Training | 🖥️ | CPU: {os.cpu_count()} cores | RAM: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 'N/A'}")

    metrics = train_model(model, train_loader, val_loader)
    return save_model_with_metadata(model, metrics)

if __name__ == "__main__":
    main()