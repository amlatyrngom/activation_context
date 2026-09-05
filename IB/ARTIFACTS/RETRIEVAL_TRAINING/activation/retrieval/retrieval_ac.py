"""
Retrieval activation-context (AC) model.

An AC model places new latent input in the base model: a small byte-level side encoder reads the
same text the base model reads and produces V "view" rows in the base model's input space. Those
rows are appended after the text tokens, and the retrieval model reads its embedding off them. If
the base also carries a LoRA, the rows can live in a subspace with new properties.
"""
import math
import typing as t

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..dataset.dataset_utils import safe_truncate_embedding_chunk

LAYER_DROPOUT = 0.1
"""Attention, feed-forward and residual dropout inside the byte pooling, the encoder layers and the view
cross-attention: the standard rate for a small transformer trained from scratch. The view queries and the
output projection carry none, since noise there lands directly in the base model's input space."""

if t.TYPE_CHECKING:
    from ..harness import HarnessRuntime


BYTE_VOCAB = 256
BYTE_WINDOW = 8
BYTE_STRIDE = 4


def sinusoidal_positions(length: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """[length, d] sinusoidal positional encoding; no length cap."""
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    encoding = torch.zeros(length, d, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)[:, : d // 2]
    return encoding.to(dtype)


class FeedForward(nn.Module):
    """Pre-norm FFN with mult 4 and GELU."""
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.down(F.gelu(self.up(self.norm(x)))))


class WindowedBytePooling(nn.Module):
    """
    Self-attention on BYTE_WINDOW-wide windows with BYTE_STRIDE stride, pooled to one vector per
    window, then a standard FFN. Reduces the byte sequence to about a quarter of its length.
    """
    def __init__(self, d: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        # No attention dropout here: the windows are flattened into a batch of B * W rows (hundreds of
        # thousands for a document batch), and SDPA's dropout path refuses batches beyond 65535 rows.
        # Attention over 8 bytes gains nothing from dropout anyway; the FFN keeps its dropout.
        self.attention = nn.MultiheadAttention(d, num_heads, dropout=0.0, batch_first=True)
        self.ffn = FeedForward(d, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, L, d], mask [B, L] bool -> pooled [B, W, d], window mask [B, W] bool."""
        batch_size, length, d = x.shape
        pad = (-(length - BYTE_WINDOW)) % BYTE_STRIDE if length > BYTE_WINDOW else BYTE_WINDOW - length
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        windows = x.unfold(1, BYTE_WINDOW, BYTE_STRIDE).permute(0, 1, 3, 2)          # [B, W, 8, d]
        window_masks = mask.unfold(1, BYTE_WINDOW, BYTE_STRIDE)                        # [B, W, 8]
        num_windows = windows.shape[1]
        flat = windows.reshape(batch_size * num_windows, BYTE_WINDOW, d)
        flat_masks = window_masks.reshape(batch_size * num_windows, BYTE_WINDOW)
        window_valid = flat_masks.any(dim=1)                                           # [B*W]
        key_padding = ~flat_masks
        key_padding[~window_valid] = False  # Fully padded windows attend to themselves harmlessly and are masked downstream.
        normed = self.norm(flat)
        attended, _ = self.attention(normed, normed, normed, key_padding_mask=key_padding, need_weights=False)
        flat = flat + attended
        weights = flat_masks.to(flat.dtype)[..., None]
        pooled = (flat * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)      # [B*W, d]
        pooled = self.ffn(pooled)
        return pooled.view(batch_size, num_windows, d), window_valid.view(batch_size, num_windows)


class StandardRetrievalACModel(nn.Module):
    """
    Byte-level side encoder producing V view rows in the base model's input space.
    Byte table 256 x d_ac -> positional encoding -> windowed attention (window 8, stride 4) pooled to
    one vector per window -> num_ac_layers bidirectional pre-norm layers (FFN mult 4) -> V learned
    queries cross-attend the sequence -> linear d_ac -> d_model.
    """
    def __init__(
        self,
        harness: "HarnessRuntime",
        base_model_name: str,
        d_ac_model: int | None = None,      # None: d_model // 4
        num_view_tokens: int = 8,
        num_ac_layers: int | None = None,   # None: base layer count // 4
    ):
        super().__init__()
        description = harness.loaded_models[base_model_name].model_config.model_description
        self.harness = harness
        self.base_model_name = base_model_name
        self.d_model = description.d_model
        self.d_ac_model = d_ac_model or self.d_model // 4
        self.num_view_tokens = num_view_tokens
        self.num_ac_layers = num_ac_layers or max(1, len(description.layer_descriptions) // 4)
        self.input_limit_chars = harness.harness_config.doc_embedding_input_limit_chars
        self.gradient_checkpointing = False
        num_heads = max(1, self.d_ac_model // 64)
        assert self.d_ac_model % num_heads == 0, f"d_ac_model {self.d_ac_model} must be divisible by {num_heads} heads."

        self.byte_embedding = nn.Embedding(BYTE_VOCAB, self.d_ac_model)
        dropout = LAYER_DROPOUT
        self.byte_pooling = WindowedBytePooling(self.d_ac_model, num_heads, dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                self.d_ac_model, num_heads, dim_feedforward=4 * self.d_ac_model, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(self.num_ac_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_ac_model)
        self.view_queries = nn.Parameter(torch.randn(num_view_tokens, self.d_ac_model) * 0.02)
        self.view_attention = nn.MultiheadAttention(self.d_ac_model, num_heads, dropout=dropout, batch_first=True)
        self.view_norm = nn.LayerNorm(self.d_ac_model)
        self.output_projection = nn.Linear(self.d_ac_model, self.d_model)
        self.output_scale = nn.Parameter(torch.ones(()))                                # rows leave at unit RMS * scale

    @property
    def device(self) -> torch.device:
        return self.byte_embedding.weight.device

    def encode_bytes(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """UTF-8 byte ids [B, L] (right-padded) and mask [B, L] of the char-limited texts."""
        encoded = [
            list(safe_truncate_embedding_chunk(text, self.input_limit_chars).encode("utf-8")) or [0]
            for text in texts
        ]
        length = max(len(row) for row in encoded)
        ids = torch.zeros(len(encoded), length, dtype=torch.long)
        mask = torch.zeros(len(encoded), length, dtype=torch.bool)
        for index, row in enumerate(encoded):
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[index, : len(row)] = True
        return ids.to(self.device), mask.to(self.device)

    def forward(self, texts: list[str]) -> torch.Tensor:
        """[B, V, d_model] view rows, one group per text."""
        ids, mask = self.encode_bytes(texts)
        x = self.byte_embedding(ids)
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device, x.dtype)[None]
        x, valid = self.byte_pooling(x, mask)                                                  # [B, W, d_ac]
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device, x.dtype)[None]
        key_padding = ~valid
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, None, key_padding, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=key_padding)
        x = self.final_norm(x)
        queries = self.view_queries[None].expand(x.shape[0], -1, -1)
        views, _ = self.view_attention(queries, x, x, key_padding_mask=key_padding, need_weights=False)
        rows = self.output_projection(self.view_norm(queries + views))                       # [B, V, d_model]
        rows = rows * torch.rsqrt(rows.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(rows.dtype)
        return rows * self.output_scale

    @staticmethod
    def checkpoint(self):
        raise NotImplementedError # Do not implement yet.

    @staticmethod
    def load_from_checkpoint(self):
        raise NotImplementedError # Do not implement yet.
