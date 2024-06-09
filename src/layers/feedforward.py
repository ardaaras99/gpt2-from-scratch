import torch.nn as nn

from src import GPT2Config
from src.layers.activations import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)
