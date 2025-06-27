import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
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


class TransformerModel:
    def __init__(self, vocab: dict, config: dict):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.device = config.get("device", "cpu")

        self.model = TransformerDecoder(
            vocab_size=len(vocab),
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            max_len=config["max_len"]
        ).to(self.device)

    def encode_prompt(self, text: str) -> torch.Tensor:
        tokens = [self.vocab.get(tok, self.vocab.get("<UNK>", 1)) for tok in text.split()]
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

    def decode_output(self, tokens: List[int]) -> str:
        return " ".join([self.inv_vocab.get(t, "<UNK>") for t in tokens])

    def generate(
        self,
        prompt: str,
        max_new_tokens=20,
        strategy="greedy",
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        beam_width=3,
        tone_prefix: Optional[str] = None,
    ) -> str:
        if tone_prefix:
            prompt = f"[{tone_prefix}] {prompt}"

        input_ids = self.encode_prompt(prompt)
        generated = input_ids.tolist()[0]

        self.model.eval()
        with torch.no_grad():
            if strategy == "beam":
                return self._beam_search(input_ids, beam_width, max_new_tokens)
            for _ in range(max_new_tokens):
                x = torch.tensor([generated], dtype=torch.long, device=self.device)
                logits = self.model(x)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                if strategy == "greedy":
                    next_token = torch.argmax(probs, dim=-1).item()
                elif strategy == "top_k":
                    top_k_probs, top_k_idx = torch.topk(probs, top_k)
                    next_token = top_k_idx[0][torch.multinomial(top_k_probs[0], 1)].item()
                elif strategy == "top_p":
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative_probs <= top_p
                    mask[..., 0] = 1
                    filtered_idx = sorted_idx[0][mask[0]]
                    filtered_probs = sorted_probs[0][mask[0]]
                    next_token = filtered_idx[torch.multinomial(filtered_probs, 1)].item()
                elif strategy == "top_k_top_p":
                    top_k_probs, top_k_idx = torch.topk(probs, top_k)
                    sorted_probs, sorted_idx = torch.sort(top_k_probs[0], descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative_probs <= top_p
                    mask[0] = 1
                    filtered_idx = top_k_idx[0][mask]
                    filtered_probs = top_k_probs[0][mask]
                    next_token = filtered_idx[torch.multinomial(filtered_probs, 1)].item()
                else:
                    raise ValueError("Unsupported strategy")

                generated.append(next_token)

        return self.decode_output(generated)

    def _beam_search(self, input_ids: torch.Tensor, beam_width: int, max_len: int) -> str:
        beams = [(input_ids.tolist()[0], 0)]  # (tokens, log_prob)
        for _ in range(max_len):
            candidates = []
            for tokens, score in beams:
                x = torch.tensor([tokens], dtype=torch.long, device=self.device)
                logits = self.model(x)[:, -1, :]
                probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, beam_width)

                for i in range(beam_width):
                    new_tokens = tokens + [topk_idx[0][i].item()]
                    new_score = score + topk_probs[0][i].item()
                    candidates.append((new_tokens, new_score))

            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        best_tokens = beams[0][0]
        return self.decode_output(best_tokens)
