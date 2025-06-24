# models/sentiment_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.text_utils import tokenize_text
from sklearn.preprocessing import LabelEncoder

class SentimentAnalyzer:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.vocab = {"<unk>": 0}
        self.model = None
        self.trained = False

    def build_vocab(self, texts):
        idx = 1
        for text in texts:
            for tok in tokenize_text(text):
                if tok not in self.vocab:
                    self.vocab[tok] = idx
                    idx += 1

    def encode_text(self, text):
        return [self.vocab.get(tok, 0) for tok in tokenize_text(text)]

    def train(self, texts, labels, epochs=10):
        self.encoder.fit(labels)
        y = torch.tensor(self.encoder.transform(labels))
        self.build_vocab(texts)
        X = [torch.tensor(self.encode_text(t)) for t in texts]
        padded = nn.utils.rnn.pad_sequence(X, batch_first=True)
        
        self.model = LSTMSentiment(len(self.vocab), len(self.encoder.classes_))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(padded)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        self.trained = True

    def predict(self, text):
        if not self.trained:
            return "neutral"
        x = torch.tensor(self.encode_text(text)).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
            idx = torch.argmax(out, dim=1).item()
            return self.encoder.inverse_transform([idx])[0]


class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
