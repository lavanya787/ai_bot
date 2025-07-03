import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from models.reward_model import SimpleRewardModel
from tokenizer.tokenizer import CustomTokenizer

# -------- Dataset --------
class RewardDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=50):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("feedback") not in ["positive", "negative"]:
                    continue
                reward = 1.0 if entry["feedback"] == "positive" else 0.0
                combined = f"{entry['prompt']} [SEP] {entry['response']}"
                token_ids = tokenizer.encode(combined)[:max_len]
                token_tensor = torch.tensor(token_ids + [0] * (max_len - len(token_ids)))
                self.samples.append((token_tensor.float(), torch.tensor(reward)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# -------- Training Loop --------
def train(model, dataloader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "saved_models/reward_model.pt")
    print("✅ Reward model saved.")

if __name__ == "__main__":
    tokenizer = CustomTokenizer.load("tokenizer/vocab.json")
    dataset = RewardDataset("logs/cli_feedback_log.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SimpleRewardModel(input_dim=50)  # assuming max token length 50
    train(model, dataloader)
