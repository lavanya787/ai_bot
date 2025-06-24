import torch
import torch.nn as nn
import re
from utils.text_utils import tokenize_text

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(BiLSTMClassifier, self).__init__()
        if output_dim < 1:
            raise ValueError(f"output_dim must be at least 1, got {output_dim}")
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

def classify_query_intent(text, model_i, model_s, stoi, intent_map, sentiment_map):
    if not isinstance(text, str) or not text.strip():
        return "general", "neutral"

    tokens = [stoi.get(tok, stoi.get("<unk>", 0)) for tok in tokenize_text(text)][:512]

    if not tokens or any(not isinstance(t, (int, float)) for t in tokens):
        return "general", "neutral"

    try:
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            intent_out = model_i(x)
            sent_out = model_s(x)

            intent_idx = torch.argmax(intent_out, dim=1).item()
            sent_idx = torch.argmax(sent_out, dim=1).item()

            intent = intent_map.get(str(intent_idx), "general")
            sentiment = sentiment_map.get(str(sent_idx), "neutral")

            # Optional debug
            print(f"[DEBUG] Intent probs: {torch.softmax(intent_out, dim=1)}")
            print(f"[DEBUG] Sentiment probs: {torch.softmax(sent_out, dim=1)}")

            return intent, sentiment

    except Exception as e:
        print(f"Error in classify_query_intent: {e}")
        return "general", "neutral"
