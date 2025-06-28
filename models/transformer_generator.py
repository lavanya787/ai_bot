import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self,config,vocab, d_model, max_len=5000):
        super().__init__()
        self.config = config
        self.device = config.get("device", "cpu")  # ✅ Add this line

        self.embedding = nn.Embedding(len(vocab), config["d_model"])
        self.pos_encoding = self._positional_encoding(config["max_len"], config["d_model"])

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = (attn_weights @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, num_heads, ff_dim, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_embed(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

class TransformerGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = self.transformer(x)
        return self.decoder(x)

    def generate(self, start_token_id, max_len=50):
        generated = [start_token_id]
        input = torch.tensor([generated], dtype=torch.long)

        for _ in range(max_len - 1):
            logits = self.forward(input)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_token)
            input = torch.tensor([generated], dtype=torch.long)
        return generated

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.vocab = vocab
        self.idx2word = {v: k for k, v in vocab.items()}
        self.word2idx = vocab
        self.config = config
        self.vocab_size = len(vocab)

        self.device = torch.device(config["device"])  # ✅ define early

        self.embedding = nn.Embedding(self.vocab_size, config["d_model"])
        self.pos_encoding = self._positional_encoding(config["max_len"], config["d_model"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["num_heads"],
            dim_feedforward=config["ff_dim"]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["n_layers"])
        self.output_layer = nn.Linear(config["d_model"], self.vocab_size)

        self.to(self.device)

    def _positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).to(self.device)

    def encode_prompt(self, text):
        tokens = [self.word2idx.get(t, self.word2idx.get("<UNK>", 1)) for t in text.lower().split()]
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        return self.output_layer(x)

    def generate(self, prompt, max_new_tokens=30, temperature=1.0, top_k=0, top_p=0.0):
        self.eval()
        input_ids = self.encode_prompt(prompt)
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            logits = outputs[:, -1, :] / temperature
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
