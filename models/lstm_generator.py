import torch
import torch.nn as nn

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2):
        super(LSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_token_id, max_length=50):
        input = torch.tensor([[start_token_id]], dtype=torch.long)
        hidden = None
        generated = [start_token_id]

        for _ in range(max_length):
            logits, hidden = self.forward(input, hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(next_token.item())
            input = next_token.unsqueeze(0)

        return generated

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out

def generate_response_lstm(prompt, model, stoi=None, itos=None):
    """
    Generate a response using an LSTM model.
    Args:
        prompt (str): The input prompt/query.
        model (LSTMModel): The trained LSTM model.
        stoi (dict, optional): String-to-index vocabulary mapping.
        itos (dict, optional): Index-to-string vocabulary mapping.
    Returns:
        str: The generated response.
    """
    if not prompt or not isinstance(prompt, str):
        return ""
    if model is None or stoi is None or itos is None:
        return "Error: Model, stoi, or itos is not provided."
    
    tokens = [stoi.get(w.lower(), stoi.get("<unk>", 0)) for w in prompt.split()]
    if not tokens:
        return ""
    
    x = torch.tensor([tokens], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs[0, -1, :], dim=-1).item()  # Take the last timestep
    return itos.get(next_token, "<unk>")