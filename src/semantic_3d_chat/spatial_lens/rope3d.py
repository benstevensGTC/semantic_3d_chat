"""Rotary position encoding in three dimensions.

A transformer with RoPE gives every token a semantic vector and encodes its
position as a *rotation* of that vector, so an attention score between two
tokens depends on the displacement between them rather than on either absolute
index. Nothing about that construction is specific to a sequence: the only
requirement is a coordinate to rotate by.

Here the coordinate is a point's position in the room. The head dimension is
split into three bands and each band is rotated by one axis, exactly the way
Qwen2-VL's M-RoPE splits position into time, height and width. Two points that
sit half a metre apart therefore attend to each other the same way wherever in
the room they are, which is the property a grid of learned absolute cell codes
cannot express.

Nothing here is trained. The rotation is a fixed geometric prior, as it is in
every RoPE transformer.
"""

from __future__ import annotations

import torch
from torch import nn


def _bands(head_dim: int) -> tuple[int, int, int]:
    """Split a head into three rotary bands, one per axis.

    Each band needs an even width because rotary works on pairs. Any remainder
    goes to the axes the room actually varies in most: x and y before z.
    """

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary encoding")
    pairs = head_dim // 2
    base = pairs // 3
    extra = pairs - base * 3
    x_pairs = base + (1 if extra > 0 else 0)
    y_pairs = base + (1 if extra > 1 else 0)
    z_pairs = base
    return x_pairs * 2, y_pairs * 2, z_pairs * 2


class Rope3D(nn.Module):
    """Rotate query/key vectors by a point's (x, y, z) position.

    ``metres_per_cycle`` sets the coarsest wavelength: the lowest frequency band
    completes one rotation over roughly that distance, so it should be on the
    order of the room. Higher bands resolve finer offsets, giving the same
    multi-scale behaviour sequence RoPE gets over token distance.
    """

    def __init__(self, head_dim: int, *, metres_per_cycle: float = 8.0) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.widths = _bands(self.head_dim)
        if sum(self.widths) != self.head_dim:
            raise ValueError("rotary bands must cover the head dimension")
        for axis, width in zip("xyz", self.widths, strict=True):
            pairs = width // 2
            if pairs == 0:
                self.register_buffer(f"freq_{axis}", torch.zeros(0), persistent=False)
                continue
            # Geometric spread of spatial frequencies, low to high.
            exponent = torch.arange(pairs, dtype=torch.float32) / max(pairs, 1)
            self.register_buffer(
                f"freq_{axis}",
                (2.0 * torch.pi / metres_per_cycle) * (1000.0 ** exponent),
                persistent=False,
            )

    def forward(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """values [B, H, N, head_dim], positions [B, N, 3] in metres."""

        if values.shape[-1] != self.head_dim:
            raise ValueError("value head dimension differs from the encoder")
        if positions.shape[-1] != 3 or positions.shape[:2] != values.shape[0::2][:2]:
            raise ValueError("positions must be [B, N, 3] matching the values")

        chunks = torch.split(values, list(self.widths), dim=-1)
        rotated = []
        for index, (axis, chunk) in enumerate(zip("xyz", chunks, strict=True)):
            if chunk.shape[-1] == 0:
                rotated.append(chunk)
                continue
            frequency = getattr(self, f"freq_{axis}").to(values.dtype)
            angle = positions[..., index].unsqueeze(1).unsqueeze(-1) * frequency
            cos, sin = torch.cos(angle), torch.sin(angle)
            even, odd = chunk[..., 0::2], chunk[..., 1::2]
            turned = torch.stack(
                (even * cos - odd * sin, even * sin + odd * cos), dim=-1
            )
            rotated.append(turned.flatten(-2))
        return torch.cat(rotated, dim=-1)


class Rope3DAttention(nn.Module):
    """Multi-head self-attention whose geometry is 3D relative position."""

    def __init__(self, model_dim: int, heads: int, *, metres_per_cycle: float = 8.0) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError("model_dim must divide evenly into heads")
        self.heads = int(heads)
        self.head_dim = model_dim // heads
        self.rope = Rope3D(self.head_dim, metres_per_cycle=metres_per_cycle)
        self.to_qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.project = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, count, model_dim = tokens.shape
        qkv = self.to_qkv(tokens).reshape(batch, count, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        # Only queries and keys are rotated: that is what makes the score depend
        # on the displacement between two points rather than their coordinates.
        query = self.rope(query, positions)
        key = self.rope(key, positions)
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return self.project(
            attended.transpose(1, 2).reshape(batch, count, model_dim)
        )


class Rope3DBlock(nn.Module):
    """Pre-norm transformer block over points positioned in a room."""

    def __init__(
        self, model_dim: int, heads: int, *, metres_per_cycle: float = 8.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = Rope3DAttention(model_dim, heads, metres_per_cycle=metres_per_cycle)
        self.feed_norm = nn.LayerNorm(model_dim)
        self.feed = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.dropout(
            self.attention(self.attention_norm(tokens), positions)
        )
        return tokens + self.dropout(self.feed(self.feed_norm(tokens)))


__all__ = ["Rope3D", "Rope3DAttention", "Rope3DBlock"]
