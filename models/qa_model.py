# models/qa_model.py
import torch
import torch.nn as nn

class SimpleQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embed(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out  # Scalar score for relevance
