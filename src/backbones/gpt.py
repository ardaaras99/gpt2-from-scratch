import torch
import torch.nn as nn

from src import GPT2Config
from src.blocks.decoder import DecoderBlock
from src.layers.embedding import EmbeddingLayer
from src.layers.layernorm import LayerNorm


class GPT(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.emb_layer = EmbeddingLayer(cfg.vocab_size, cfg.emb_dim, cfg.context_length)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.decoder = nn.Sequential(*[DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape

        x = self.emb_layer(input_ids)
        x = self.drop_emb(x)
        x = self.decoder(x)
        x = self.final_norm(x)
        return x
