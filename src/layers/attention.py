import torch
import torch.nn as nn

from src import GPT2Config


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        k = self.W_K(x)
        q = self.W_Q(x)
        v = self.W_V(x)
        attn_scores = q @ k.T
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ v
        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        ones = torch.ones(context_length, context_length)
        self.register_buffer("mask", torch.triu(ones, diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        k = self.W_K(x)
        q = self.W_Q(x)
        v = self.W_V(x)

        attn_scores = q @ k.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ v
        return context_vec


class MultiHeadAttentionv1(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        ones = torch.ones(context_length, context_length)
        self.register_buffer("mask", torch.triu(ones, diagonal=1))

    def forward(self, x: torch.Tensor):
        # x has shape B,L,D
        b, num_tokens, d_in = x.shape

        # Q,K,V have shape B,L,D_out
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # D_out = num_heads * head_dim
        # Q,K,V have shape B,L,num_heads,head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Q,K,V have shape B,num_heads,L,head_dim
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # to multiply Q and K we need to transpose K
        # B,num_heads,L,head_dim @ B,num_heads,head_dim,L -> B,num_heads,L,L
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_K = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_V = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)

        self.W_O = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        ones = torch.ones(cfg.context_length, cfg.context_length)
        mask = torch.triu(ones, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        head_dim = self.cfg.emb_dim // self.cfg.n_heads

        Q = self.W_Q(x).view(B, L, self.cfg.n_heads, head_dim)
        K = self.W_K(x).view(B, L, self.cfg.n_heads, head_dim)
        V = self.W_V(x).view(B, L, self.cfg.n_heads, head_dim)

        Q = Q.transpose(1, 2)  # B, n_heads, L, head_dim
        K = K.transpose(1, 2)  # B, n_heads, L, head_dim
        V = V.transpose(1, 2)  # B, n_heads, L, head_dim

        attn_scores = Q @ K.transpose(2, 3)  # B, n_heads, L, L
        mask_bool = self.mask.bool()[:L, :L]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / head_dim**0.5, dim=-1)

        # (B, n_heads, L, L) @ (B, n_heads, L, head_dim) -> (B, n_heads, L, head_dim)
        context_vec = attn_weights @ V
        context_vec = context_vec.transpose(1, 2)  # B, L, n_heads, head_dim

        if self.cfg.emb_dim != self.cfg.n_heads * head_dim:
            raise ValueError("emb_dim must be divisible by n_heads")

        # context_vec1 = context_vec.reshape(B, L, self.cfg.emb_dim)
        context_vec = context_vec.contiguous().view(B, L, self.cfg.emb_dim)

        # if not torch.allclose(context_vec1, context_vec2, atol=1e-10):
        #     raise ValueError("context_vec1 and context_vec2 are not equal")
        # else:
        #     print("context_vec1 and context_vec2 are equal")
        return self.W_O(context_vec)
