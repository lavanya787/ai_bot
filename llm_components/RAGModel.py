import torch
import torch.nn as nn
import torch.nn.functional as F

def top_k_sampling(logits, top_k=50, temperature=1.0):
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))
    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices.gather(-1, next_token_idx)

class RAGModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super(RAGModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim * 2, num_layers=num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=8, batch_first=True)
        self.context_projection = nn.Linear(hidden_dim * 2, embed_dim)  # Project to embed_dim
        self.output_layer = nn.Linear(hidden_dim * 2, vocab_size)

    def _merge_bidir(self, h):
        # h: [num_layers*2, batch, hidden_dim]
        # return: [num_layers, batch, hidden_dim*2]
        h = h.view(self.num_layers, 2, h.size(1), h.size(2))  # [num_layers, 2, batch, hidden_dim]
        return torch.cat([h[:, 0], h[:, 1]], dim=-1)

    def forward(self, input_ids, target_ids):
        embedded_input = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        encoder_output, (hidden, cell) = self.encoder(embedded_input)  # encoder_output: [batch_size, seq_len, hidden_dim*2]
        hidden = self._merge_bidir(hidden)  # [num_layers, batch, hidden_dim*2]
        cell = self._merge_bidir(cell)  # [num_layers, batch, hidden_dim*2]

        embedded_target = self.embedding(target_ids)  # [batch_size, seq_len, embed_dim]
        decoder_output, _ = self.decoder(embedded_target, (hidden, cell))  # [batch_size, seq_len, hidden_dim*2]
        attn_output, _ = self.attention(decoder_output, encoder_output, encoder_output)  # [batch_size, seq_len, hidden_dim*2]
        context = self.context_projection(attn_output)  # [batch_size, seq_len, embed_dim]
        logits = self.output_layer(attn_output)  # Use attn_output directly: [batch_size, seq_len, vocab_size]
        return logits

    def encode_document(self, input_ids):
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        output, _ = self.encoder(embedded)  # [batch_size, seq_len, hidden_dim*2]
        pooled = torch.mean(output, dim=1)  # [batch_size, hidden_dim*2]
        return pooled

    def infer(self, prompt: str, tokenizer, device=None) -> str:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
        self.eval()
        device = device or next(self.parameters()).device

        input_ids = tokenizer.encode_ids(prompt)
        if not input_ids:
            return "⚠️ Prompt is empty or cannot be tokenized."

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  # [1, seq_len]
        with torch.no_grad():
            embedded_input = self.embedding(input_tensor)  # [1, seq_len, embed_dim]
            encoder_output, (hidden, cell) = self.encoder(embedded_input)  # encoder_output: [1, seq_len, hidden_dim*2]
            hidden = self._merge_bidir(hidden)  # [num_layers, 1, hidden_dim*2]
            cell = self._merge_bidir(cell)  # [num_layers, 1, hidden_dim*2]

            start_token_id = getattr(tokenizer, "sos_token_id", 2)
            end_token_id = getattr(tokenizer, "eos_token_id", 3)
            pad_token_id = getattr(tokenizer, "pad_token_id", 0)

            cur_input = torch.tensor([[start_token_id]], dtype=torch.long).to(device)  # [1, 1]
            generated = []

            for _ in range(200):
                embedded = self.embedding(cur_input)  # [1, 1, embed_dim]
                output, (hidden, cell) = self.decoder(embedded, (hidden, cell))  # output: [1, 1, hidden_dim*2]
                attn_output, _ = self.attention(output, encoder_output, encoder_output)  # [1, 1, hidden_dim*2]
                context = self.context_projection(attn_output)  # [1, 1, embed_dim]
                logits = self.output_layer(attn_output.squeeze(1))  # [1, vocab_size]
                next_token_id = top_k_sampling(logits, top_k=50, temperature=0.8)
                token = next_token_id.item()

                if token in (end_token_id, pad_token_id):
                    break
                generated.append(token)
                cur_input = torch.tensor([[token]], dtype=torch.long).to(device)  # [1, 1]

        return tokenizer.decode(generated)

    def generate(self, input_ids, context_embeddings=None, max_length=200, temperature=0.8, top_k=50):
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        with torch.no_grad():
            embedded_input = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
            encoder_output, (hidden, cell) = self.encoder(embedded_input)  # encoder_output: [batch_size, seq_len, hidden_dim*2]
            hidden = self._merge_bidir(hidden)  # [num_layers, batch_size, hidden_dim*2]
            cell = self._merge_bidir(cell)  # [num_layers, batch_size, hidden_dim*2]

            start_token_id = 2
            end_token_id = 3
            pad_token_id = 0

            current_input = torch.tensor([[start_token_id] for _ in range(batch_size)], dtype=torch.long, device=device)  # [batch_size, 1]
            generated = [[] for _ in range(batch_size)]

            for _ in range(max_length):
                embedded = self.embedding(current_input)  # [batch_size, 1, embed_dim]
                
                if context_embeddings is not None:
                    # Ensure context_embeddings is [batch_size, seq_len, embed_dim]
                    if context_embeddings.dim() == 2:
                        context_embeddings = context_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim*2]
                    context_proj = self.context_projection(context_embeddings)  # [batch_size, 1, embed_dim]
                    attn_output, _ = self.attention(embedded, context_proj, context_proj)  # [batch_size, 1, hidden_dim*2]
                else:
                    attn_output = embedded  # [batch_size, 1, embed_dim]

                output, (hidden, cell) = self.decoder(attn_output, (hidden, cell))  # output: [batch_size, 1, hidden_dim*2]
                logits = self.output_layer(output.squeeze(1))  # [batch_size, vocab_size]

                next_tokens = top_k_sampling(logits, top_k=top_k, temperature=temperature)  # [batch_size, 1]
                
                for i, token in enumerate(next_tokens.squeeze(-1).tolist()):
                    if token not in (end_token_id, pad_token_id):
                        generated[i].append(token)
                    else:
                        generated[i].append(end_token_id)
                        break
                
                current_input = next_tokens  # [batch_size, 1]

                if all(end_token_id in seq or pad_token_id in seq for seq in generated):
                    break

            max_len = max(len(seq) for seq in generated)
            padded_generated = [seq + [pad_token_id] * (max_len - len(seq)) for seq in generated]
            return torch.tensor(padded_generated, dtype=torch.long, device=device)  # [batch_size, max_len]