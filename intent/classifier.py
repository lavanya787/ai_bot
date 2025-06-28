import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# -------- CONFIG --------
MODEL_PATH = 'intent/intent_model.pth'
VOCAB_PATH = 'intent/intent_vocab.json'
TOKENIZER_PATH = 'intent/vocab.json'
MAX_SEQ_LENGTH = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- UTILITIES --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def tokenize(text):
    return clean_text(text).split()

def encode(text, word2idx):
    return [word2idx.get(token, word2idx.get("<UNK>", 1)) for token in tokenize(text)][:MAX_SEQ_LENGTH]

# -------- MODEL --------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerIntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=6, max_len=MAX_SEQ_LENGTH):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)              # [B, T, D]
        x = self.pos_encoder(x)            # [B, T, D]
        x = self.transformer(x)            # [B, T, D]
        x = x.mean(dim=1)                  # [B, D] - Global Average Pooling
        return self.fc(x)                  # [B, num_classes]

# -------- PREDICTOR --------
class IntentClassifier:
    def __init__(self):
        if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(VOCAB_PATH) or not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("🛑 Required model or vocab files are missing.")

        with open(TOKENIZER_PATH) as f:
            self.word2idx = json.load(f)

        with open(VOCAB_PATH) as f:
            self.label2id = json.load(f)

        self.id2label = {v: k for k, v in self.label2id.items()}

        self.model = TransformerIntentClassifier(
            vocab_size=len(self.word2idx),
            embed_dim=512,
            num_heads=6,
            hidden_dim=768,
            num_classes=len(self.label2id),
            num_layers=6,
            max_len=MAX_SEQ_LENGTH
        ).to(device)

        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        self.model.eval()

# Singleton classifier instance
_classifier_instance = None

def predict_intent(text: str) -> str:
    """
    Utility function to predict intent from a text input.
    Reuses a singleton instance of the IntentClassifier for performance.
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance.predict(text)
