import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer_generator import TransformerGenerator
from models.reward_model import SimpleRewardModel
import json

class PPOTrainer:
    def __init__(self, model, reward_model, tokenizer, lr=1e-5):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(self.model.model.parameters(), lr=lr)
        self.kl_beta = 0.01

    def compute_rewards(self, generated_outputs):
        embeddings = torch.stack(generated_outputs)
        rewards = self.reward_model(embeddings).squeeze()
        return rewards

    def step(self, prompts):
        self.model.model.train()
        all_losses = []
        for prompt in prompts:
            input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.model.device)
            logits = self.model.model(input_ids)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            output_embedding = logits.mean(dim=1)

            reward = self.reward_model(output_embedding).squeeze()
            loss = -reward.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_losses.append(loss.item())

        return sum(all_losses) / len(all_losses)

if __name__ == "__main__":
    with open("models/vocab.json") as f:
        vocab = json.load(f)

    from tokenizer.vocab_builder import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab)

    config = {
        "d_model": 128,
        "n_layers": 2,
        "num_heads": 4,
        "ff_dim": 256,
        "max_len": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    model = TransformerGenerator(vocab, config)
    model.model.load_state_dict(torch.load("saved_models/general/transformer_rlhf.pt")["model_state"])

    reward_model = SimpleRewardModel(input_dim=128).to(config["device"])
    trainer = PPOTrainer(model, reward_model, tokenizer)

    prompts = ["how does ppo work", "explain reinforcement", "define attention mechanism"]
    trainer.step(prompts)
