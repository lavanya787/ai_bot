import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRewardModel(nn.Module):
    def __init__(self, input_dim=128, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x) .squeeze(-1) # Returns scalar reward
