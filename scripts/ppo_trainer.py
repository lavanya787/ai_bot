# scripts/ppo_trainer.py
import torch
import random
from models.transformer_generator import TransformerGenerator
from models.sentiment_model import SentimentClassifier
from tokenizer.tokenizer import CustomTokenizer
import json

# Load reward model
reward_model = SentimentClassifier(vocab_size=len(tokenizer.vocab))
reward_model.load_state_dict(torch.load("saved_models/sentiment_model.pt"))
reward_model.eval()

# Load reward model
def load_sentiment_reward_model():
    with open("tokenizer/vocab.json", "r") as f:
        vocab = json.load(f)
    model = SentimentClassifier(vocab_size=len(vocab))
    model.load_state_dict(torch.load("saved_models/sentiment_model.pt", map_location="cpu"))
    model.eval()
    return model, vocab

# Reward function using sentiment score
def reward_function(text):
    model, vocab = load_sentiment_reward_model()
    tokens = tokenize_text(text, vocab, max_len=50)
    input_tensor = torch.tensor([tokens])
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        reward = probs[0][1].item()  # Probability of "positive"
    return reward
    
def compute_loss(reward_preferred, reward_rejected):
    # Policy prefers higher reward
    return torch.clamp(reward_rejected - reward_preferred, min=0).mean()

def load_pairs(path="training_data/rlhf_pairs.json"):
    with open(path, "r") as f:
        return json.load(f)

def fine_tune_with_ppo(model, tokenizer, pairs, epochs=3, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0

        for pair in pairs:
            prompt = pair["prompt"]
            input_ids = torch.tensor([tokenizer.encode(prompt)])

            # Forward pass with preferred response
            preferred_output = pair["preferred"]
            rejected_output = pair["rejected"]

            r_pref = reward_function(preferred_output)
            r_rej = reward_function(rejected_output)

            loss = compute_loss(torch.tensor(r_pref), torch.tensor(r_rej))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss / len(pairs):.4f}")

    model.save_checkpoint("saved_models/rlhf_finetuned.pt")
    print("✅ RLHF fine-tuning complete.")

if __name__ == "__main__":
    model = TransformerGenerator.load_from_checkpoint("saved_models/general/model.pt")
    tokenizer = CustomTokenizer.load("tokenizer/vocab.json")
    pairs = load_pairs()
    fine_tune_with_ppo(model, tokenizer, pairs)
