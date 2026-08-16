from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .perceiver import (
    SCENE_ENCODER_ARCHITECTURE_VERSION,
    GlobalSceneResampler,
    SignalPreservingProjection,
    spatial_anchors,
    spatial_coverage_weights,
)
from .point_tokens import PointTokenProjection
from .spatial_blocks import SpatialBlockEncoder


@dataclass
class SceneTokenizerOutput:
    scene_tokens: torch.Tensor
    native_latents: torch.Tensor
    block_tokens: torch.Tensor
    audit: dict[str, torch.Tensor]
    # Optional all-voxel coverage field kept separate from the established
    # scene-token path.  A post-stack adapter may consume it after every frozen
    # base residual has run; it is never selected by the user's question.
    aligned_sidecar_tokens: torch.Tensor | None = None


class NativeAlignedVoxelCoverage(nn.Module):
    """Pool every voxel's native LM-aligned tail into fixed spatial anchors."""

    def __init__(self, num_latents: int, coverage_temperature: float) -> None:
        super().__init__()
        if coverage_temperature <= 0:
            raise ValueError("coverage_temperature must be positive")
        self.num_latents = int(num_latents)
        self.coverage_temperature = float(coverage_temperature)
        self.register_buffer("latent_anchors", spatial_anchors(num_latents), persistent=False)

    def forward(
        self,
        aligned_features: torch.Tensor,
        xyz: torch.Tensor,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if aligned_features.ndim != 2 or aligned_features.shape[0] == 0:
            raise ValueError("aligned_features must be nonempty [N,H]")
        if xyz.shape != (aligned_features.shape[0], 3):
            raise ValueError("xyz must have shape [N,3] matching aligned_features")
        if room_min.shape != (3,) or room_max.shape != (3,):
            raise ValueError("room bounds must have shape [3]")
        extent = room_max.float() - room_min.float()
        if not torch.all(extent > 0):
            raise ValueError("room bounds must have positive extent")
        normalized_xyz = ((xyz.float() - room_min.float()) / extent).mul(2.0).sub(1.0)
        weights = spatial_coverage_weights(
            normalized_xyz,
            self.latent_anchors,
            self.coverage_temperature,
        )
        pooled = torch.matmul(weights, aligned_features.float().unsqueeze(0))
        return pooled.to(aligned_features.dtype), weights


class SceneTokenizer(nn.Module):
    """Hierarchically transform the entire persistent map into continuous LM tokens."""

    def __init__(
        self,
        semantic_dim: int,
        model_dim: int,
        language_hidden_dim: int,
        block_size_m: float = 0.25,
        tokens_per_block: int = 2,
        global_latents: int = 256,
        heads: int = 8,
        global_layers: int = 2,
        fourier_bands: int = 8,
        coverage_temperature: float = 0.20,
        coverage_scale: float = 4.0,
        query_identity_scale: float = 0.05,
        projection_skip_scale: float = 1.0,
        semantic_skip_scale: float = 1.0,
        geometry_skip_scale: float = 0.5,
        block_content_residual_scale: float = 1.0,
        language_aligned_tail_dim: int = 0,
        native_aligned_coverage_scale: float = 0.0,
        learned_scene_token_scale: float = 1.0,
        learned_scene_token_rms_target: float | None = None,
        architecture_version: str = SCENE_ENCODER_ARCHITECTURE_VERSION,
    ) -> None:
        super().__init__()
        if architecture_version != SCENE_ENCODER_ARCHITECTURE_VERSION:
            raise ValueError(
                "Scene encoder implementation/config mismatch: "
                f"{architecture_version!r} != {SCENE_ENCODER_ARCHITECTURE_VERSION!r}"
            )
        self.architecture_version = architecture_version
        self.language_aligned_tail_dim = int(language_aligned_tail_dim)
        self.native_aligned_coverage_scale = float(native_aligned_coverage_scale)
        self.learned_scene_token_scale = float(learned_scene_token_scale)
        self.learned_scene_token_rms_target = (
            None
            if learned_scene_token_rms_target is None
            else float(learned_scene_token_rms_target)
        )
        if not 0 <= self.language_aligned_tail_dim <= semantic_dim:
            raise ValueError("language_aligned_tail_dim must be within semantic_dim")
        if (
            self.language_aligned_tail_dim > 0
            and self.language_aligned_tail_dim != language_hidden_dim
        ):
            raise ValueError(
                "language_aligned_tail_dim must equal language_hidden_dim when enabled"
            )
        if self.native_aligned_coverage_scale < 0:
            raise ValueError("native_aligned_coverage_scale cannot be negative")
        if self.learned_scene_token_scale < 0:
            raise ValueError("learned_scene_token_scale cannot be negative")
        if self.language_aligned_tail_dim == 0 and self.native_aligned_coverage_scale != 0:
            raise ValueError("native_aligned_coverage_scale requires language_aligned_tail_dim")
        if (
            self.learned_scene_token_rms_target is not None
            and self.learned_scene_token_rms_target <= 0
        ):
            raise ValueError("learned_scene_token_rms_target must be positive")
        if self.learned_scene_token_scale == 0 and self.native_aligned_coverage_scale == 0:
            raise ValueError("At least one scene-token path must have nonzero scale")
        self.point_projection = PointTokenProjection(
            semantic_dim,
            model_dim,
            fourier_bands,
            semantic_skip_scale=semantic_skip_scale,
            geometry_skip_scale=geometry_skip_scale,
        )
        self.block_encoder = SpatialBlockEncoder(
            model_dim,
            block_size_m,
            tokens_per_block,
            heads,
            fourier_bands,
            content_residual_scale=block_content_residual_scale,
        )
        self.resampler = GlobalSceneResampler(
            model_dim,
            global_latents,
            heads,
            global_layers,
            coverage_temperature=coverage_temperature,
            coverage_scale=coverage_scale,
            query_identity_scale=query_identity_scale,
        )
        self.language_projection = SignalPreservingProjection(
            model_dim, language_hidden_dim, projection_skip_scale
        )
        self.native_aligned_coverage = (
            NativeAlignedVoxelCoverage(global_latents, coverage_temperature)
            if self.language_aligned_tail_dim > 0
            else None
        )

    def forward(
        self,
        semantic: torch.Tensor,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        normal: torch.Tensor,
        confidence: torch.Tensor,
        observation_count: torch.Tensor,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
        *,
        aligned_sidecar: torch.Tensor | None = None,
        aligned_sidecar_scale: float = 0.0,
    ) -> SceneTokenizerOutput:
        if not isinstance(aligned_sidecar_scale, (int, float)) or isinstance(
            aligned_sidecar_scale, bool
        ):
            raise TypeError("aligned_sidecar_scale must be a finite number")
        if not torch.isfinite(torch.tensor(float(aligned_sidecar_scale))):
            raise ValueError("aligned_sidecar_scale must be finite")
        if float(aligned_sidecar_scale) < 0.0:
            raise ValueError("aligned_sidecar_scale cannot be negative")
        if aligned_sidecar is None and float(aligned_sidecar_scale) != 0.0:
            raise ValueError("aligned_sidecar_scale requires aligned_sidecar")
        points, _ = self.point_projection(
            semantic, xyz, rgb, normal, confidence, observation_count, room_min, room_max
        )
        blocks, audit = self.block_encoder(points, xyz, room_min, room_max)
        latents = self.resampler(blocks, audit["block_token_positions_normalized"])
        learned_scene_tokens = self.language_projection(latents)
        if self.learned_scene_token_rms_target is not None:
            learned_scene_tokens = (
                nn.functional.layer_norm(
                    learned_scene_tokens.float(),
                    (learned_scene_tokens.shape[-1],),
                )
                .mul(self.learned_scene_token_rms_target)
                .to(learned_scene_tokens.dtype)
            )
        scene_tokens = self.learned_scene_token_scale * learned_scene_tokens
        aligned_sidecar_tokens = None
        audit["learned_scene_token_rms"] = (
            learned_scene_tokens.detach().float().square().mean().sqrt()
        )
        if self.native_aligned_coverage is not None:
            aligned_features = semantic[:, -self.language_aligned_tail_dim :]
            native_aligned_tokens, native_weights = self.native_aligned_coverage(
                aligned_features, xyz, room_min, room_max
            )
            scene_tokens = scene_tokens + self.native_aligned_coverage_scale * native_aligned_tokens
            voxel_total_contribution = native_weights.sum(dim=(0, 1))
            audit["native_aligned_processed_voxels"] = torch.tensor(
                aligned_features.shape[0], device=semantic.device, dtype=torch.long
            )
            audit["native_aligned_min_weight"] = native_weights.detach().min()
            audit["native_aligned_min_voxel_contribution"] = voxel_total_contribution.detach().min()
            audit["native_aligned_token_rms"] = (
                native_aligned_tokens.detach().float().square().mean().sqrt()
            )
            if aligned_sidecar is not None:
                expected_shape = (semantic.shape[0], self.language_aligned_tail_dim)
                if aligned_sidecar.shape != expected_shape:
                    raise ValueError(
                        "aligned_sidecar must have shape "
                        f"{expected_shape}; got {tuple(aligned_sidecar.shape)}"
                    )
                if not torch.is_floating_point(aligned_sidecar) or not torch.isfinite(
                    aligned_sidecar
                ).all():
                    raise ValueError("aligned_sidecar must contain finite floating-point values")
                sidecar_tokens, sidecar_weights = self.native_aligned_coverage(
                    aligned_sidecar, xyz, room_min, room_max
                )
                # A zero scale is an exact routing mode for the post-stack
                # adapter.  Skipping the arithmetic (instead of adding
                # ``0 * sidecar``) guarantees bit-identical base tokens.
                if float(aligned_sidecar_scale) > 0.0:
                    scene_tokens = scene_tokens + float(aligned_sidecar_scale) * sidecar_tokens
                aligned_sidecar_tokens = sidecar_tokens
                sidecar_total_contribution = sidecar_weights.sum(dim=(0, 1))
                audit["aligned_sidecar_processed_voxels"] = torch.tensor(
                    aligned_sidecar.shape[0], device=semantic.device, dtype=torch.long
                )
                audit["aligned_sidecar_min_weight"] = sidecar_weights.detach().min()
                audit["aligned_sidecar_min_voxel_contribution"] = (
                    sidecar_total_contribution.detach().min()
                )
                audit["aligned_sidecar_token_rms"] = (
                    sidecar_tokens.detach().float().square().mean().sqrt()
                )
                audit["aligned_sidecar_scale"] = torch.tensor(
                    float(aligned_sidecar_scale),
                    device=semantic.device,
                    dtype=torch.float32,
                )
        elif aligned_sidecar is not None:
            raise ValueError("aligned_sidecar requires native aligned voxel coverage")
        if not torch.isfinite(scene_tokens).all():
            raise RuntimeError("Scene tokenizer produced NaN or infinity")
        return SceneTokenizerOutput(
            scene_tokens,
            latents,
            blocks,
            audit,
            aligned_sidecar_tokens=aligned_sidecar_tokens,
        )
