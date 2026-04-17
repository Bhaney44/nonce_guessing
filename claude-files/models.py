"""
models.py
---------
Three model architectures for nonce prediction from real block history:

1. MLP       — Feedforward net on engineered lag/rolling features
2. LSTM      — Sequence model over the last N blocks
3. Transformer — Attention-based sequence model (most expressive)

Each model outputs a single value (predicted nonce, scaled).
"""

import torch
import torch.nn as nn
import math


# ─── MLP ──────────────────────────────────────────────────────────────────────

class NonceMLP(nn.Module):
    """
    Feedforward network on the full feature vector.
    Best for: fast training, interpretability, good baseline.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int] = [512, 256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(0.15),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── LSTM ─────────────────────────────────────────────────────────────────────

class NonceLSTM(nn.Module):
    """
    Bidirectional LSTM over a window of past block features.
    Best for: capturing sequential dependencies in miner behavior.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)                            # (batch, seq_len, hidden*2)
        attn_out, _ = self.attention(out, out, out)      # self-attention over sequence
        pooled = attn_out.mean(dim=1)                    # global average pool
        return self.head(pooled).squeeze(-1)


# ─── Transformer ──────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class NonceTransformer(nn.Module):
    """
    Transformer encoder over a window of past block features.
    Best for: long-range pattern detection, most expressive architecture.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)       # global average pool over sequence
        return self.head(x).squeeze(-1)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(model_type: str, input_dim: int, seq_len: int = 50) -> nn.Module:
    if model_type == "mlp":
        return NonceMLP(input_dim=input_dim)
    elif model_type == "lstm":
        return NonceLSTM(input_dim=input_dim)
    elif model_type == "transformer":
        return NonceTransformer(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
