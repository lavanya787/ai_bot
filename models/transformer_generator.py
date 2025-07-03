import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerModel(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.vocab = vocab
        self.word2idx = vocab
        self.idx2word = {v: k for k, v in vocab.items()}
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

        self.vocab_size = len(vocab)
        self.d_model = config.get("d_model", 64) #chnaged from 128 to 64
        self.max_len = config.get("max_len", 64)

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = self._build_positional_encoding(self.max_len, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.get("num_heads", 2),#chnaged from 4 to 2
            dim_feedforward=config.get("ff_dim", 128),#changed from 256 to 128
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.get("n_layers", 1)) #chnaged from 2 to 1

        self.output_layer = nn.Linear(self.d_model, self.vocab_size)  # Optional: For generation

        self.to(self.device)

    def _build_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).to(self.device)

    def encode_prompt(self, text: str):
        tokens = text.lower().split()
        indices = [self.word2idx.get(tok, self.word2idx.get("<UNK>", 1)) for tok in tokens]
        return torch.tensor(indices[:self.max_len], dtype=torch.long, device=self.device).unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.encoder(x)
        return x

    def classify(self, x):
        x = self.forward(x)
        return x.mean(dim=1)

    def generate(self, prompt, max_new_tokens=30, temperature=1.0, top_k=0, top_p=0.0):
        self.eval()
        input_ids = self.encode_prompt(prompt)
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            logits = self.output_layer(outputs[:, -1, :]) / temperature
            probs = F.softmax(logits, dim=-1)

            if top_k > 0:
                topk_probs, topk_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(1, topk_indices, topk_probs)
                probs = probs / probs.sum()

            elif top_p > 0.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_probs[sorted_indices_to_remove] = 0
                probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.word2idx.get("<EOS>", -1):
                break

        output_tokens = generated.squeeze().tolist()
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in output_tokens])
    
class TransformerGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):#reduced from 128 to 64 ;256 to 128
        super(TransformerGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=2,#changed from 8 to 2
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=1 #changed frm 2 to 1
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output = self.transformer(embedded)
        output = self.fc(output)
        return output
