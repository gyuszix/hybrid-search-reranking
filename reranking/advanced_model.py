import torch
import torch.nn as nn



class AdvancedDeepReranker(nn.Module):
    def __init__(self, input_dim=17):
        super(AdvancedDeepReranker, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), # Narrowed width to reduce memory capacity
            nn.LayerNorm(64),         # Switched to LayerNorm
            nn.LeakyReLU(0.1),        # LeakyReLU helps prevent dead neurons
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # We use a raw score here; the MarginRankingLoss handles the comparison
        return self.mlp(x)