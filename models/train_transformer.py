# train_transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.transformer_generator import TransformerGenerator
from models.reward_model import SimpleRewardModel
import os
import json

# ----------- Tokenizer and Vocab Builder -----------
class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def build_vocab(self, corpus):
        tokens = set()
        for line in corpus:
            tokens.update(line.strip().split())
        self.vocab = {token: idx+3 for idx, token in enumerate(sorted(tokens))}
        self.vocab["<pad>"] = 0
        self.vocab["<unk>"] = 1
        self.vocab["<eos>"] = 2
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in text.strip().split()]

    def decode(self, ids):
        return " ".join([self.inv_vocab.get(i, "<unk>") for i in ids])

# ----------- Dataset Class -----------
class TextDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.data[idx])
        input_ids = tokens
        target_ids = tokens[1:] + [self.tokenizer.vocab["<eos>"]]
        return torch.tensor(input_ids), torch.tensor(target_ids)

# ----------- Training Function -----------
def train(model, dataloader, optimizer, criterion, reward_model=None):
    model.model.train()
    total_loss = 0

    for input_ids, target_ids in dataloader:
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(model.device)
        target_ids = nn.utils.rnn.pad_sequence(target_ids, batch_first=True).to(model.device)

        logits = model.model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        if reward_model:
            embeddings = logits.mean(dim=1)
            rewards = reward_model(embeddings).squeeze()
            loss = (loss * rewards).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ----------- Save and Load -----------
def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state': model.model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

# ----------- Main -----------
if __name__ == "__main__":
    with open("training_data/qa_corpus.csv") as f:
        corpus = [line.strip() for line in f.readlines()]

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(corpus)

    config = {
        "d_model": 128,
        "n_layers": 2,
        "num_heads": 4,
        "ff_dim": 256,
        "max_len": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    dataset = TextDataset(corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    generator = TransformerGenerator(tokenizer.vocab, config)
    reward_model = SimpleRewardModel(hidden_size=config["d_model"]).to(config["device"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["<pad>"], reduction="none")
    optimizer = optim.Adam(generator.model.parameters(), lr=1e-4)

    ckpt_path = "saved_models/general/transformer_rlhf.pt"
    if os.path.exists(ckpt_path):
        load_checkpoint(generator, optimizer, ckpt_path)

    for epoch in range(10):
        loss = train(generator, dataloader, optimizer, criterion, reward_model)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        save_checkpoint(generator, optimizer, ckpt_path)

    with open("models/vocab.json", "w") as f:
        json.dump(tokenizer.vocab, f)

    print("Training completed and model saved.")
