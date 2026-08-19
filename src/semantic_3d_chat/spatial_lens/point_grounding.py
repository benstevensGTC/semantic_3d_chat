"""Ground language in the point cloud itself, with 3D rotary position.

The grid model pooled points into cells and let a learned code per cell carry
position. That works, but it is absolute, quantised, and specific to one room
shape. This model takes the points as they are: each is a token carrying its
Gemma embedding, and its place in the room enters only as a rotation of that
embedding, so attention between two points depends on the displacement between
them. It is the same construction a RoPE transformer uses along a sequence,
with a three-dimensional offset instead of an index.

Two consequences matter. Predictions are continuous, because the answer is a
weighted position over real points rather than a cell centre. And nothing in the
model is tied to a particular room size, because only relative geometry is
encoded -- which is what a claim about unseen rooms should rest on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from semantic_3d_chat.spatial_lens.rope3d import Rope3DBlock

POINT_SCHEMA = "semantic_3d_chat.spatial_lens.point_grounding.v1"


@dataclass(frozen=True)
class PointExample:
    """One room sampled as points, one phrase, and which points it covers."""

    room: str
    phrase: str
    points: np.ndarray        # [N, 3] metres, the tokens the model sees
    features: np.ndarray      # [N, F] Gemma embeddings
    target: np.ndarray        # [N] float32 over those tokens, sums to 1
    room_size_m: tuple[float, float, float]
    # Every voxel of the object at full resolution, not just the sampled ones.
    # Scoring against the sample would charge the model for gaps the downsample
    # opened up, and would not be comparable to the grid head's footprint gap.
    footprint: np.ndarray | None = None  # [K, 3] metres


class PointGroundingModel(nn.Module):
    """Score every point in a room for how well it matches a phrase."""

    def __init__(
        self,
        feature_dim: int = 1536,
        model_dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        metres_per_cycle: float = 8.0,
        dropout: float = 0.0,
        position_mode: str = "rope3d",
    ) -> None:
        super().__init__()
        if position_mode not in {"rope3d", "learned_absolute", "none"}:
            raise ValueError(f"unknown position_mode: {position_mode}")
        self.position_mode = position_mode
        self.point_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.query_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        # The two controls. 'learned_absolute' is the grid model's scheme, given
        # raw coordinates instead of a cell index; 'none' removes position
        # entirely, leaving a bag of semantic points.
        if position_mode == "learned_absolute":
            self.absolute = nn.Sequential(
                nn.Linear(3, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
            )
        self.blocks = nn.ModuleList(
            Rope3DBlock(
                model_dim, heads, metres_per_cycle=metres_per_cycle, dropout=dropout
            )
            for _ in range(layers)
        )
        self.score = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """features [B,N,F], positions [B,N,3], query [B,F] -> logits [B,N]."""

        tokens = self.point_projection(features)
        tokens = tokens + self.query_projection(query).unsqueeze(1)
        if self.position_mode == "learned_absolute":
            tokens = tokens + self.absolute(positions)
        geometry = positions if self.position_mode == "rope3d" else torch.zeros_like(positions)
        for block in self.blocks:
            tokens = block(tokens, geometry)
        logits = self.score(tokens).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits

    def predict_position(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        temperature: float = 0.25,
    ) -> torch.Tensor:
        """A continuous answer: the attention-weighted centroid of the match."""

        logits = self.forward(features, positions, query, mask)
        weights = torch.softmax(logits / temperature, dim=-1)
        return torch.einsum("bn,bnd->bd", weights, positions)


def save_model(path: str | Path, model: PointGroundingModel, metadata: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "point_grounding.pt")
    (destination / "metadata.json").write_text(
        json.dumps({"schema": POINT_SCHEMA, **metadata}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_model(path: str | Path) -> tuple[PointGroundingModel, dict[str, Any]]:
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != POINT_SCHEMA:
        raise ValueError("Unexpected point-grounding schema")
    model = PointGroundingModel(
        feature_dim=int(metadata["feature_dim"]),
        model_dim=int(metadata["model_dim"]),
        heads=int(metadata["heads"]),
        layers=int(metadata["layers"]),
        metres_per_cycle=float(metadata["metres_per_cycle"]),
        position_mode=str(metadata["position_mode"]),
    )
    model.load_state_dict(torch.load(source / "point_grounding.pt", map_location="cpu"))
    model.eval()
    return model, metadata


__all__ = ["POINT_SCHEMA", "PointExample", "PointGroundingModel", "load_model", "save_model"]
