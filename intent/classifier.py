import os
import re
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import tempfile
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "intent/intent_model.pt"
LABEL_MAP_PATH = "intent/label_map.pkl"
VOCAB_PATH = "intent/intent_vocab.json"
TOKENIZER_PATH = "intent/vocab.json"

MAX_SEQ_LENGTH = 256
VOCAB_SIZE = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Tokenizer Class ----------
class SimpleTokenizer:
    def __init__(self, word2idx, max_length=MAX_SEQ_LENGTH):
        self.word2idx = word2idx
        self.max_length = max_length
        self.trained = False  # Default value

    def encode(self, text):
        words = word_tokenize(text.lower())
        input_ids = [self.word2idx.get(word, 1) for word in words][:self.max_length]
        input_ids += [0] * (self.max_length - len(input_ids))
        return torch.tensor(input_ids, dtype=torch.long)


# ---------- Dataset ----------
class IntentDataset(Dataset):
    def __init__(self, data: pd.DataFrame, word2idx: dict, label2id: dict):
        if not isinstance(data, pd.DataFrame) or not all(col in data for col in ["sentence", "intent"]):
            logger.error("Invalid dataset format: must be DataFrame with 'sentence' and 'intent' columns")
            raise ValueError("Invalid dataset format")
        self.sentences = data["sentence"].tolist()
        self.labels = [label2id.get(intent, 0) for intent in data["intent"]]
        self.word2idx = word2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = word_tokenize(sentence.lower())
        input_ids = [self.word2idx.get(word, 1) for word in words][:MAX_SEQ_LENGTH]
        input_ids += [0] * (MAX_SEQ_LENGTH - len(input_ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ---------- Classifier ----------
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256, num_labels=1):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.word2idx = {"[PAD]": 0, "[UNK]": 1}
        self.label2id = {}
        self.num_labels = num_labels
        self.id2label = {}
        self.tokenizer = None
        self.is_trained = False
        self.device = device
        self.to(self.device)

    def _build_vocab_from_data(self, sentences):
        vocab = set(["[PAD]", "[UNK]"])
        for sentence in sentences:
            if not isinstance(sentence, str) or not sentence.strip():
                continue
            words = word_tokenize(sentence.lower())
            vocab.update(words)
        vocab = list(vocab)[:VOCAB_SIZE]
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        logger.info(f"Vocabulary size: {len(self.word2idx)}")
        self.tokenizer = SimpleTokenizer(self.word2idx)
        self.tokenizer.trained = True

        # ✅ Save vocab to disk
        os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
        import json
        with open(VOCAB_PATH, "w") as f:
            json.dump(self.word2idx, f)

    def _save_label_map(self):
        os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
        try:
            with open(LABEL_MAP_PATH, "wb") as f:
                pickle.dump({"label2id": self.label2id, "id2label": self.id2label}, f)
            logger.info(f"Saved label map to {LABEL_MAP_PATH}")
        except Exception as e:
            logger.error(f"Failed to save label map: {e}")

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        logits = self.fc(hidden[-1])
        return logits

    def train_model(self, data: pd.DataFrame, domain: str = "default") -> tuple:
        if not isinstance(data, pd.DataFrame) or not all(col in data for col in ["sentence", "intent"]):
            logger.error("Invalid training data format")
            return False, 0.0, None, []

        if data.empty or data["sentence"].str.strip().eq("").any():
            logger.error("Empty or invalid sentences in training data")
            return False, 0.0, None, []

        self._build_vocab_from_data(data["sentence"])

        unique_intents = data["intent"].unique().tolist()
        self.label2id = {label: idx for idx, label in enumerate(unique_intents)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self._save_label_map()

        self.fc = nn.Linear(256, len(self.label2id)).to(self.device)
        self.num_labels = len(self.label2id)

        from sklearn.model_selection import train_test_split
        train_data, val_data = data, pd.DataFrame()
        if len(data) >= 5:
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        else:
            logger.warning("Dataset too small for validation split")

        try:
            train_dataset = IntentDataset(train_data, self.word2idx, self.label2id)
            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
            self.train()

            losses = []
            for epoch in range(3):
                total_loss = 0
                for batch in train_dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    logits = self(input_ids)
                    loss = F.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_dataloader)
                losses.append(avg_loss)
                logger.info(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

            accuracy = 0.0
            if not val_data.empty:
                val_dataset = IntentDataset(val_data, self.word2idx, self.label2id)
                val_dataloader = DataLoader(val_dataset, batch_size=8)
                self.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids = batch["input_ids"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        logits = self(input_ids)
                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                accuracy = correct / total if total > 0 else 0.0
                logger.info(f"Validation Accuracy: {accuracy:.4f}")

            # Save model
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                torch.save(self.state_dict(), tmp.name)
                shutil.move(tmp.name, MODEL_PATH)
                logger.info(f"✅ Saved model to {MODEL_PATH}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy(MODEL_PATH, f"intent/intent_model_{timestamp}.pt")

            self.is_trained = True
            return True, accuracy, self, losses

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False, 0.0, None, []

    def predict(self, text: str) -> str:
        if not self.is_trained or not self.tokenizer:
            logger.warning("Model not trained or tokenizer missing")
            return "default"
        try:
            input_ids = self.tokenizer.encode(text).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self(input_ids)
                pred_id = torch.argmax(logits, dim=1).item()
            return self.id2label.get(pred_id, "default")
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "default"

    def load_model(self, path=MODEL_PATH):
        if not os.path.exists(path):
            logger.info(f"Model not found at {path}, skipping load")
            return False
        try:
            state_dict = torch.load(path, map_location=self.device)

            if os.path.exists(LABEL_MAP_PATH):
                with open(LABEL_MAP_PATH, "rb") as f:
                    label_map = pickle.load(f)
                    self.label2id = label_map["label2id"]
                    self.id2label = label_map["id2label"]
                    self.fc = nn.Linear(256, len(self.label2id)).to(self.device)

            if os.path.exists(VOCAB_PATH):
                import json
                with open(VOCAB_PATH, "r") as f:
                    self.word2idx = json.load(f)
                self.tokenizer = SimpleTokenizer(self.word2idx)
                self.tokenizer.trained = True
                logger.info(f"Loaded tokenizer with vocab size: {len(self.word2idx)}")
            else:
                logger.warning("VOCAB_PATH not found; using fallback vocab")
                self.word2idx = {"[PAD]": 0, "[UNK]": 1}
                self.tokenizer = SimpleTokenizer(self.word2idx)
                self.tokenizer.trained = False

            self.load_state_dict(state_dict)
            self.is_trained = True
            logger.info(f"✅ Loaded model from {path}")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Failed to load model from {path}: {e}. Deleting file.")
            if os.path.exists(path):
                os.remove(path)
            return False
