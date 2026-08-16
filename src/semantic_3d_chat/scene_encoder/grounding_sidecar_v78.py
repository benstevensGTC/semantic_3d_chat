"""Numeric, zero-preserving full-scene grounding architecture for V78."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from semantic_3d_chat.scene_encoder.perceiver import spatial_anchors

ARTIFACT = "continuous_full_scene_grounding_sidecar_v78"
ARCHITECTURE = "zero_preserving_all_scene_token_query_attention_v1"
WEIGHTS_FILENAME = "grounding.safetensors"
METADATA_FILENAME = "metadata.json"
EXPECTED_CHECKPOINT_FILES = frozenset({WEIGHTS_FILENAME, METADATA_FILENAME})
DEFAULT_SCENE_DIM = 1536
DEFAULT_LATENT_COUNT = 256


class GroundingSidecarV78(nn.Module):
    """Question-conditioned numeric grounding over every full-scene token.

    The residual path is bias-free and receives only scene-derived values or
    question/scene products. Multiplication by ``scene_present`` makes the
    zero-scene result exactly zero in normalized room coordinates, proving that
    there is no question-only coordinate decoder.
    """

    def __init__(
        self,
        *,
        scene_dim: int = DEFAULT_SCENE_DIM,
        latent_count: int = DEFAULT_LATENT_COUNT,
        rank: int = 64,
        hidden_dim: int = 256,
        maximum_residual: float = 0.5,
    ) -> None:
        super().__init__()
        if scene_dim < 1 or latent_count < 1 or rank < 1 or hidden_dim < 1:
            raise ValueError("Grounding dimensions must be positive")
        if not 0.0 <= maximum_residual <= 1.0:
            raise ValueError("maximum_residual must be in [0, 1]")
        self.scene_dim = int(scene_dim)
        self.latent_count = int(latent_count)
        self.rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.maximum_residual = float(maximum_residual)
        self.question_down = nn.Linear(scene_dim, rank, bias=False)
        self.question_up = nn.Linear(rank, scene_dim, bias=False)
        self.query_residual_gate = nn.Parameter(torch.tensor(-4.0))
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.coordinate_residual = nn.Sequential(
            nn.Linear(scene_dim * 2 + 3, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, 3, bias=False),
            nn.Tanh(),
        )
        anchors = spatial_anchors(latent_count).float()
        self.register_buffer("anchors_normalized", anchors, persistent=False)
        self.register_buffer(
            "centered_anchors_normalized",
            anchors - anchors.mean(dim=0, keepdim=True),
            persistent=False,
        )
        nn.init.normal_(self.question_down.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.question_up.weight)

    def forward(
        self,
        question_embeddings: torch.Tensor,
        scene_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if question_embeddings.ndim != 2 or question_embeddings.shape[-1] != self.scene_dim:
            raise ValueError(f"question_embeddings must have shape [B,{self.scene_dim}]")
        expected_tail = (self.latent_count, self.scene_dim)
        if scene_tokens.ndim != 3 or tuple(scene_tokens.shape[1:]) != expected_tail:
            raise ValueError(
                f"scene_tokens must have shape [B,{self.latent_count},{self.scene_dim}]"
            )
        if scene_tokens.shape[0] != question_embeddings.shape[0]:
            raise ValueError("Question and scene batches must have equal size")
        if not torch.isfinite(question_embeddings).all() or not torch.isfinite(scene_tokens).all():
            raise ValueError("Grounding inputs contain NaN or infinity")
        question = question_embeddings.float()
        scene = scene_tokens.float()
        adapted = question + torch.sigmoid(self.query_residual_gate) * self.question_up(
            self.question_down(question)
        )
        query = F.normalize(adapted, dim=-1, eps=1e-6)
        keys = F.normalize(scene, dim=-1, eps=1e-6)
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        logits = torch.einsum("bd,bld->bl", query, keys) * scale
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bl,bld->bd", weights, keys)
        anchors = self.centered_anchors_normalized.to(
            device=weights.device, dtype=weights.dtype
        )
        anchor_xyz = torch.matmul(weights, anchors)
        residual_input = torch.cat((attended, attended * query, anchor_xyz), dim=-1)
        residual = self.coordinate_residual(residual_input) * self.maximum_residual
        scene_present = (scene.abs().sum(dim=(1, 2), keepdim=False) > 0).to(
            dtype=anchor_xyz.dtype
        )
        predicted = (anchor_xyz + residual).clamp(-1.0, 1.0)
        predicted = predicted * scene_present.unsqueeze(-1)
        return predicted, logits, weights

    def nearest_anchor_targets(self, normalized_xyz: torch.Tensor) -> torch.Tensor:
        if normalized_xyz.ndim != 2 or normalized_xyz.shape[-1] != 3:
            raise ValueError("normalized_xyz must have shape [B,3]")
        anchors = self.anchors_normalized.to(
            device=normalized_xyz.device, dtype=normalized_xyz.dtype
        )
        distances = (normalized_xyz.unsqueeze(1) - anchors.unsqueeze(0)).square().sum(-1)
        return distances.argmin(dim=-1)


def normalize_xyz(
    xyz: torch.Tensor, room_min: torch.Tensor, room_max: torch.Tensor
) -> torch.Tensor:
    return ((xyz - room_min) / (room_max - room_min).clamp_min(1e-6)).mul(2).sub(1)


def denormalize_xyz(
    xyz: torch.Tensor, room_min: torch.Tensor, room_max: torch.Tensor
) -> torch.Tensor:
    return (xyz.add(1).mul(0.5) * (room_max - room_min)) + room_min


__all__ = [
    "ARCHITECTURE",
    "ARTIFACT",
    "DEFAULT_LATENT_COUNT",
    "DEFAULT_SCENE_DIM",
    "EXPECTED_CHECKPOINT_FILES",
    "METADATA_FILENAME",
    "WEIGHTS_FILENAME",
    "GroundingSidecarV78",
    "denormalize_xyz",
    "normalize_xyz",
]
