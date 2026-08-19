"""A learned readout that lets language address the 3D semantic field.

Two independent measurements said the same thing about this stack. A linear
probe showed position is present in the scene tokens (R² = 0.99), yet the
control head trained on them ignored it; and a frozen decoder handed the same
tokens can describe a room accurately but cannot say which cell anything is in.
The geometry is in the representation. What is missing is a readout trained to
*address* it.

This module is that readout. It takes the per-cell semantic embeddings of a
room and a phrase, and produces a distribution over the room's floor: "where is
the lamp". It is small, it is the only thing trained on this machine, and the
decoder it reads from stays frozen.

The supervision costs nothing to collect. Perception already discovers objects
geometrically and names them by asking Gemma what it sees, so every scanned room
yields (phrase, occupied cells) pairs on its own -- no oracle, no annotation, no
human in the loop. That is what makes training on a handful of rooms and working
in unseen ones a reasonable thing to expect.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

GROUNDING_SCHEMA = "semantic_3d_chat.spatial_lens.grounding_head.v1"


@dataclass(frozen=True)
class GroundingExample:
    """One room, one phrase, and the cells that phrase actually covers."""

    room: str
    phrase: str
    scene: np.ndarray          # [cells, feature_dim] float32
    target: np.ndarray         # [cells] float32, sums to 1
    grid: int
    room_size_m: tuple[float, float, float]


class SpatialGroundingHead(nn.Module):
    """Cross-attend a phrase into the room's cells and score every one.

    Deliberately small. The scene embeddings and the text embedding both come
    from Gemma and are frozen; all this learns is the correspondence between
    them and the spatial layout, which is exactly the piece that was missing.
    """

    def __init__(
        self,
        feature_dim: int = 1536,
        model_dim: int = 256,
        heads: int = 8,
        layers: int = 2,
        grid: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = int(grid)
        self.scene_projection = nn.Sequential(
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
        # A learned position code per cell: the head must be free to use
        # *where* a cell is, not only what it contains.
        self.cell_position = nn.Parameter(torch.randn(grid * grid, model_dim) * 0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=heads,
                dim_feedforward=model_dim * 4,
                batch_first=True,
                norm_first=True,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.score = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        # Cells quantise position to a third of a metre, and a scored cell says
        # nothing about how far off a miss was. A soft-argmax over the same
        # scores gives a continuous position for free, which is what a rover
        # needs, and it costs no extra parameters.
        self.register_buffer(
            "cell_grid",
            torch.stack(
                [
                    torch.arange(grid * grid) % grid,   # column
                    torch.arange(grid * grid) // grid,  # row
                ],
                dim=1,
            ).float(),
            persistent=False,
        )

    def forward(self, scene: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """scene [B, cells, F], query [B, F] -> logits [B, cells]."""

        cells = self.scene_projection(scene) + self.cell_position.unsqueeze(0)
        # The phrase is injected additively so every layer keeps seeing it.
        cells = cells + self.query_projection(query).unsqueeze(1)
        for block in self.blocks:
            cells = block(cells)
        return self.score(cells).squeeze(-1)

    def expected_cell(self, logits: torch.Tensor) -> torch.Tensor:
        """Soft-argmax column/row, so the answer is not stuck on a cell centre."""

        weights = torch.softmax(logits, dim=-1)
        return weights @ self.cell_grid.to(weights.dtype)


def dihedral(
    scene: np.ndarray, target: np.ndarray, grid: int, variant: int
) -> tuple[np.ndarray, np.ndarray]:
    """One of the eight rotations/reflections of a room, applied consistently.

    A room turned ninety degrees is still a room, and the phrase that described
    it still describes it. With only a handful of scanned rooms this is the
    cheapest honest way to stop the head memorising absolute cell indices --
    which is exactly the failure mode that would not survive an unseen room.
    """

    field = scene.reshape(grid, grid, -1)
    mass = target.reshape(grid, grid)
    turns = variant % 4
    if turns:
        field = np.rot90(field, turns, axes=(0, 1))
        mass = np.rot90(mass, turns, axes=(0, 1))
    if variant >= 4:
        field = field[:, ::-1]
        mass = mass[:, ::-1]
    return (
        np.ascontiguousarray(field).reshape(grid * grid, -1),
        np.ascontiguousarray(mass).reshape(grid * grid),
    )


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match a distribution over cells, not a single argmax cell.

    An object covers several cells, and which one is "the" cell is arbitrary.
    Supervising the whole footprint is both better signal and better defined.
    """

    return -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def locate_error_m(
    logits: torch.Tensor,
    target: torch.Tensor,
    grid: int,
    room_size_m: tuple[float, float, float],
) -> list[float]:
    """Metres between the predicted cell and the target's centre of mass."""

    width, depth, _ = room_size_m
    errors: list[float] = []
    for row_logits, row_target in zip(logits, target, strict=True):
        index = int(row_logits.argmax())
        # The target covers several cells, so compare against its centre of
        # mass in grid coordinates rather than an arbitrary single cell.
        cells = torch.arange(grid * grid, device=row_target.device, dtype=row_target.dtype)
        centre = float((row_target * cells).sum())
        pred_row, pred_col = divmod(index, grid)
        true_row, true_col = divmod(round(centre), grid)
        errors.append(
            math.hypot(
                (pred_col - true_col) * width / grid,
                (pred_row - true_row) * depth / grid,
            )
        )
    return errors


def footprint_distance_m(
    logits: torch.Tensor,
    target: torch.Tensor,
    grid: int,
    room_size_m: tuple[float, float, float],
    *,
    soft: torch.Tensor | None = None,
) -> list[float]:
    """Metres from the prediction to the nearest cell the object occupies.

    Zero when the prediction lands on the object. This is what matters for
    driving somewhere: the centre-of-mass error punishes a correct hit on the
    far corner of a two-metre bed, which is not a real failure.
    """

    width, depth, _ = room_size_m
    cell_w, cell_d = width / grid, depth / grid
    results: list[float] = []
    for index in range(logits.shape[0]):
        occupied = (target[index] > 0).nonzero(as_tuple=False).flatten().tolist()
        if not occupied:
            results.append(float("nan"))
            continue
        if soft is None:
            best = int(logits[index].argmax())
            point = (best % grid, best // grid)
        else:
            point = (float(soft[index, 0]), float(soft[index, 1]))
        results.append(
            min(
                math.hypot(
                    (point[0] - cell % grid) * cell_w,
                    (point[1] - cell // grid) * cell_d,
                )
                for cell in occupied
            )
        )
    return results


def save_head(path: str | Path, head: SpatialGroundingHead, metadata: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), destination / "grounding.pt")
    (destination / "metadata.json").write_text(
        json.dumps({"schema": GROUNDING_SCHEMA, **metadata}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_head(path: str | Path) -> tuple[SpatialGroundingHead, dict[str, Any]]:
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != GROUNDING_SCHEMA:
        raise ValueError("Unexpected grounding-head schema")
    head = SpatialGroundingHead(
        feature_dim=int(metadata["feature_dim"]),
        model_dim=int(metadata["model_dim"]),
        heads=int(metadata["heads"]),
        layers=int(metadata["layers"]),
        grid=int(metadata["grid"]),
    )
    head.load_state_dict(torch.load(source / "grounding.pt", map_location="cpu"))
    head.eval()
    return head, metadata


__all__ = [
    "GROUNDING_SCHEMA",
    "GroundingExample",
    "SpatialGroundingHead",
    "load_head",
    "locate_error_m",
    "save_head",
    "soft_cross_entropy",
]
