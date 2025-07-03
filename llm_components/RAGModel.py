import torch
import torch.nn as nn
import torch.nn.functional as F

class RAGModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super(RAGModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: Bidirectional LSTM with 2 layers
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Decoder: LSTM with doubled hidden size to account for bidirectional encoder
        self.decoder = nn.LSTM(embed_dim, hidden_dim * 2, num_layers=num_layers, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=8, batch_first=True)
        
        # Context projection layer
        self.context_projection = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_ids, target_ids):
        embedded_input = self.embedding(input_ids)
        encoder_output, (hidden, cell) = self.encoder(embedded_input)

        def merge_bidir(h):
            h = h.view(self.num_layers, 2, h.size(1), h.size(2))
            return torch.cat([h[:, 0], h[:, 1]], dim=-1)

        hidden = merge_bidir(hidden)
        cell = merge_bidir(cell)

        embedded_target = self.embedding(target_ids)
        decoder_output, (dec_hidden, dec_cell) = self.decoder(embedded_target, (hidden, cell))
        
        # Apply attention
        attn_output, _ = self.attention(decoder_output, encoder_output, encoder_output)
        
        # Project context
        context = self.context_projection(attn_output)
        
        logits = self.output_layer(context)
        return logits

    def encode_document(self, input_ids):
        embedded = self.embedding(input_ids)
        output, _ = self.encoder(embedded)
        pooled = torch.mean(output, dim=1)
        return pooled

    def infer(self, prompt: str, tokenizer, device=None) -> str:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
    
        self.eval()
        device = device or next(self.parameters()).device
    
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            return "⚠️ Prompt is empty or cannot be tokenized."
    
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
        with torch.no_grad():
            embedded_input = self.embedding(input_tensor)
            encoder_output, (hidden, cell) = self.encoder(embedded_input)
    
            def merge_bidir(h):
                h = h.view(self.num_layers, 2, h.size(1), h.size(2))
                return torch.cat([h[:, 0], h[:, 1]], dim=-1)
    
            hidden = merge_bidir(hidden)
            cell = merge_bidir(cell)
    
            start_token_id = getattr(tokenizer, "start_token_id", tokenizer.pad_token_id)
            end_token_id = getattr(tokenizer, "end_token_id", tokenizer.pad_token_id)
            pad_token_id = tokenizer.pad_token_id
    
            cur_input = torch.tensor([[start_token_id]], dtype=torch.long).to(device)
            generated = []
    
            for _ in range(len(input_ids) * 3):
                embedded = self.embedding(cur_input)
                output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
                attn_output, _ = self.attention(output, encoder_output, encoder_output)
                context = self.context_projection(attn_output[:, -1, :])
                logits = self.output_layer(context)
                next_token_id = logits.argmax(dim=-1)
                token = next_token_id.item()
    
                if token in (end_token_id, pad_token_id):
                    break
                
                generated.append(token)
                cur_input = next_token_id.unsqueeze(0).unsqueeze(0)
    
        return tokenizer.decode(generated)