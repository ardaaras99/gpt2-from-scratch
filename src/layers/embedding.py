import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_length):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(context_length, emb_dim)

    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)

        pos_vec = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.pos_embedding(pos_vec)
        return token_emb + pos_emb
