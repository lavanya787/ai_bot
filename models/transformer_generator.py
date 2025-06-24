# models/transformer_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline

class TransformerChatModel(nn.Module):
    def __init__(self, vocab_size, emb_size=128, nhead=4, hidden=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, emb_size))  # max 512 tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=hidden)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x) + self.pos_encoder[:, :x.size(1)]
        out = self.transformer(embed)
        return self.decoder(out)

# -------- Generation Function --------
def beam_search(model, input_tensor, beam_width=3, max_len=30, stoi=None, itos=None):
    model.eval()
    sequences = [[list(input_tensor.squeeze().tolist()), 0.0]]  # [tokens, score]

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            input_seq = torch.tensor(seq).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_seq)
                probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze()

            topk = torch.topk(probs, beam_width)
            for i in range(beam_width):
                word_idx = topk.indices[i].item()
                candidate = seq + [word_idx]
                candidate_score = score + topk.values[i].item()
                all_candidates.append((candidate, candidate_score))

        # Select top beam_width sequences
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]

    best_seq = sequences[0][0]
    return " ".join([itos.get(i, "") for i in best_seq])

def generate_response_transformer(query, model, stoi, itos,context=None):
    if context:
        query = f"{context}\n\n{query}"
    tokens = query.lower().split()
    indices = [stoi.get(tok, stoi["<unk>"]) for tok in tokens]
    input_tensor = torch.tensor(indices).unsqueeze(0)
    if not query or not isinstance(query, str):
        return ""
    
    # Use a pre-trained text generation model (e.g., GPT-2)
    generator = pipeline("text-generation", model="gpt2")
    response = generator(query, max_length=50, num_return_sequences=1)[0]["generated_text"]
    
    # Clean up the response (remove the prompt if it's included)
    if response.startswith(query):
        response = response[len(query):].strip()
    
