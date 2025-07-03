import torch
from models.reward_model import SimpleRewardModel
from tokenizer.tokenizer import CustomTokenizer

tokenizer = CustomTokenizer.load("tokenizer/vocab.json")
model = SimpleRewardModel(input_dim=50)
model.load_state_dict(torch.load("saved_models/reward_model.pt"))
model.eval()

def rank_suggestions(prompt, suggestions, top_k=2):
    scored = []
    for s in suggestions:
        combo = f"{prompt} [SEP] {s}"
        ids = tokenizer.encode(combo)[:50]
        x = torch.tensor(ids + [0] * (50 - len(ids))).float().unsqueeze(0)
        with torch.no_grad():
            score = model(x).item()
        scored.append((s, score))

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_k]]
