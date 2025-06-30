import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import logging

# -------- Logging Setup --------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True  # Added to fix UserWarning
        )
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
        logger.info(f"Initializing IntentClassifier with paths: {TOKENIZER_PATH}, {VOCAB_PATH}, {MODEL_PATH}")
        
        # Create directories
        try:
            os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise

        # Default intents
        self.label2id = {
            "question": 0,
            "generate": 1,
            "sentiment": 2,
            "rag_query": 3,
            "default": 4,
            "summarize_document": 5,
            "get_insights": 6,
            "ask_question": 7,
            "train_model": 8,
            "book_flight": 9,
            "set_reminder": 10
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Initialize default vocab and tokenizer if files are missing
        try:
            if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(VOCAB_PATH):
                logger.warning(f"Model or vocab files missing. Creating defaults at {TOKENIZER_PATH}, {VOCAB_PATH}")
                self.word2idx = {"[PAD]": 0, "[UNK]": 1}
                vocab = list(self.word2idx.keys()) + [f"word{i}" for i in range(1000)]
                with open(TOKENIZER_PATH, 'w') as f:
                    json.dump(self.word2idx, f)
                with open(VOCAB_PATH, 'w') as f:
                    json.dump(vocab, f)
            else:
                with open(TOKENIZER_PATH, 'r') as f:
                    self.word2idx = json.load(f)
                with open(VOCAB_PATH, 'r') as f:
                    vocab = json.load(f)
                    self.word2idx.update({word: idx + len(self.word2idx) for idx, word in enumerate(vocab) if word not in self.word2idx})
        except Exception as e:
            logger.error(f"Failed to handle vocab/tokenizer files: {e}")
            raise

        self.model = TransformerIntentClassifier(
            vocab_size=len(self.word2idx),  #5,000-10,000
            embed_dim=768,
            num_heads=8,
            hidden_dim=2048,   #feed-forward dim
           num_classes=len(self.label2id),
            num_layers=8,
            max_len=MAX_SEQ_LENGTH
        ).to(device)

    # Save default model weights if missing
        try:
            if not os.path.exists(MODEL_PATH):
                logger.warning(f"Model weights missing. Saving untrained model to {MODEL_PATH}")
                torch.save(self.model.state_dict(), MODEL_PATH)
            else:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to handle model weights: {e}")
            raise

def predict(self, text: str) -> str:
        try:
            input_ids = encode(text, self.word2idx)
            input_ids += [self.word2idx["[PAD]"]] * (MAX_SEQ_LENGTH - len(input_ids))
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                intent_idx = torch.argmax(logits, dim=-1).item()
            return self.id2label[intent_idx]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "default"

# Singleton classifier instance
_classifier_instance = None

def predict_intent(text: str) -> str:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance.predict(text)