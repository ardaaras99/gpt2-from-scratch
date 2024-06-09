import torch.nn as nn

from src import GPT2Config
from src.layers.attention import MultiHeadAttention
from src.layers.feedforward import FeedForward
from src.layers.layernorm import LayerNorm


class DecoderBlock(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.emb_dim)
        self.masked_attn = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.ln_2 = LayerNorm(cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.ln_1(x)
        x = self.masked_attn(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.ln_2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
