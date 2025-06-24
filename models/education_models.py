import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class EducationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=64, num_classes=3, bidirectional=False):
        super(EducationModel, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.lstm1 = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=bidirectional)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(lstm_units * self.num_directions, lstm_units, batch_first=True, bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(lstm_units * self.num_directions, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=50):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.tokenizer(self.texts[idx])
        x = x[:self.max_len] + [0] * (self.max_len - len(x))
        return torch.tensor(x, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def simple_tokenizer(text):
    # map characters to integers
    return [ord(c) for c in text if ord(c) < 128]


def train_model(df: pd.DataFrame, target_column: str, epochs=5, batch_size=32, learning_rate=0.001):
    texts = df.drop(columns=[target_column]).astype(str).agg(' '.join, axis=1).tolist()
    labels = df[target_column].astype(str).tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(texts, y, test_size=0.2, random_state=42)

    vocab_size = 128  # ASCII
    num_classes = len(set(y))
    tokenizer = simple_tokenizer

    train_dataset = TextDataset(X_train, y_train, tokenizer)
    val_dataset = TextDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = EducationModel(vocab_size=vocab_size, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model, label_encoder
