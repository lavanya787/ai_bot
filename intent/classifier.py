import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import pandas as pd
import logging
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Logging Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
# Ensure NLTK resources with fallback
try:
    for res in ["punkt", "wordnet", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)
except ImportError:
    logger.error("NLTK not installed. Please install with 'pip install nltk'")
    raise ImportError("NLTK is required for IntentClassifier")
# CONFIG
MODEL_PATH = 'intent/intent_model.pt'
VOCAB_PATH = 'intent/intent_vocab.json'
TOKENIZER_PATH = 'intent/vocab.json'
MAX_SEQ_LENGTH = 256
VOCAB_SIZE = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UTILITIES
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def tokenize(text):
    return clean_text(text).split()

def encode(text, word2idx):
    try:
        tokens = word_tokenize(text.lower())
    except Exception as e:
        logger.warning(f"Tokenization failed: {e}. Using basic split as fallback.")
        tokens = text.lower().split()
    return [word2idx.get(token, word2idx["[UNK]"]) for token in tokens][:MAX_SEQ_LENGTH]

# DATASET
class IntentDataset(Dataset):
    def __init__(self, dataframe, word2idx):
        self.data = dataframe
        self.word2idx = word2idx
        self.label2id = {
            "question": 0, "generate": 1, "sentiment": 2, "rag_query": 3, "default": 4,
            "summarize_document": 5, "get_insights": 6, "ask_question": 7, "train_model": 8,
            "book_flight": 9, "set_reminder": 10
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            sentence = self.data.iloc[idx]["sentence"]
            intent = self.data.iloc[idx]["intent"]
            input_ids = encode(sentence, self.word2idx)
            input_ids += [self.word2idx["[PAD]"]] * (MAX_SEQ_LENGTH - len(input_ids))
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(self.label2id.get(intent, self.label2id["default"]), dtype=torch.long)  # Changed to "labels"
            }

# MODEL
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).float())  # ✅ safe buffer

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].clone().detach()  # ✅ clone/detach
        return x

class TransformerIntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=6, max_len=MAX_SEQ_LENGTH):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)              # [B, T, D]
        x = self.pos_encoder(x)            # [B, T, D]
        x = self.transformer(x)            # [B, T, D]
        x = x.mean(dim=1)                  # [B, D] - Global Average Pooling
        return self.fc(x)                  # [B, num_classes]

# PREDICTOR
class IntentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.word2idx = {"[PAD]": 0, "[UNK]": 1}
        self.embedding = nn.Embedding(VOCAB_SIZE, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, len(IntentDataset(None, self.word2idx).label2id))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self._build_vocab()

    def _build_vocab(self):
        try:
            nltk_words = set(nltk.corpus.words.words())
            for word in list(nltk_words)[:VOCAB_SIZE - 2]:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
        except Exception as e:
            logger.warning(f"NLTK corpus unavailable: {e}. Using minimal vocab")
            for i in range(2, VOCAB_SIZE):
                self.word2idx[f"word_{i}"] = i

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        output = output.contiguous()[:, -1, :] # Take the last hidden state
        return self.fc(output)

    def train_model(self, data):
        if not isinstance(data, pd.DataFrame) or not all(col in data for col in ["sentence", "intent"]):
            logger.error("DataFrame must contain 'sentence' and 'intent' columns")
            return False

        dataset = IntentDataset(data, self.word2idx)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        super().train(True)
        
        try:
            for epoch in range(3):
                total_loss = 0
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    outputs = self(input_ids)
                    loss = F.cross_entropy(outputs, labels)
                    total_loss += loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/3, Average Loss: {avg_loss:.4f}")
            
            torch.save(self.state_dict(), MODEL_PATH)
            logger.info(f"IntentClassifier model saved to {MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def predict(self, text):
        self.eval()
        with torch.no_grad():
            input_ids = encode(text, self.word2idx)
            input_ids += [self.word2idx["[PAD]"]] * (MAX_SEQ_LENGTH - len(input_ids))
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            outputs = self(input_tensor)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            for intent, idx in IntentDataset(None, self.word2idx).label2id.items():
                if idx == predicted_idx:
                    return intent
        return "default"