from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    dtype: torch.dtype = torch.float32
    vocab_size: int = 50240
    lr: float = 1e-6
    n_layers: int = 2
    d_model: int = 512
    norm_eps: float = 1e-5
    num_heads: int = 8
    embedding_dim: int = 512
    dropout: float = 0.01
    gradient_accumulation_steps: int = 8
    # batch_size: int = 16

    gradient_accumulation_steps: int = 8
    epochs: int = 1
    train_steps: int = 5000
    block_size: int = 10000


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x):
        return (
            x
            * self.weight
            * (1 / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + self.eps))
        )


def apply_rope(x, freqs):
    pass


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x):
        return (
            x
            * self.weight
            * (1 / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + self.eps))
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        # Learned Q, K, V projections — essential for the model to learn different subspaces
        self.w_q = nn.Linear(
            config.d_model, config.d_model, bias=False, dtype=config.dtype
        )
        self.w_k = nn.Linear(
            config.d_model, config.d_model, bias=False, dtype=config.dtype
        )
        self.w_v = nn.Linear(
            config.d_model, config.d_model, bias=False, dtype=config.dtype
        )
        self.out_proj = nn.Linear(
            config.d_model, config.d_model, bias=False, dtype=config.dtype
        )
        self.dropout_val = config.dropout

        # Zero-init out_proj so each attention block starts as an identity residual.
        # Without this, 8 stacked random projections compound variance and the
        # loss explodes to 8-10 at init (GPT-2 / Megatron-LM trick).
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        B, T, C = x.shape

        # Project into Q, K, V then split into heads
        Q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=self.dropout_val if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, layer_id, config: Config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm = LayerNorm(config.d_model, eps=config.norm_eps, dtype=config.dtype)

        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4, dtype=config.dtype),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model, dtype=config.dtype),
            nn.Dropout(config.dropout),
        )
        self.ln_mlp = nn.LayerNorm(config.d_model, dtype=config.dtype)
        # self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        # self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, x, freqs=None):
        x = self.attention(x)
        x = self.norm(x)
        x = self.mlp(x)
        x = self.ln_mlp(x)
        return x


class QuasLLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.d_model = config.d_model
        self.norm_eps = config.norm_eps

        self.token_embedding = nn.Embedding(
            self.vocab_size, self.d_model, dtype=config.dtype
        )
        self.dropout = nn.Dropout()

        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(i, config) for i in range(self.n_layers)]
        )
        self.norm = nn.LayerNorm(self.d_model, eps=self.norm_eps, dtype=config.dtype)
        # self.norm = RMSNorm(self.d_model, eps=self.norm_eps, dtype=config.dtype)
        self.output = nn.Linear(self.d_model, self.vocab_size, dtype=config.dtype)

        self.logits = nn.Linear(self.d_model, self.vocab_size, dtype=config.dtype)

    def forward(self, x):
        x = self.token_embedding(x)
        # print("embd_shape:", x)
        for block in self.blocks:
            x = block(x, freqs=None)
        x = self.norm(x)
        x = self.logits(x)
        return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = Config(dtype=torch.bfloat16)
# vocab_size = config.vocab_size
# block_size = 1024

# # token indices must be int64 — nn.Embedding handles the dtype cast internally
# dataset = torch.randint(0, vocab_size, (8, block_size)).to(device)

# model = QuasLLM(config).to(device)
# result = model(dataset)


# print(result.shape)
# print(result.dtype)
