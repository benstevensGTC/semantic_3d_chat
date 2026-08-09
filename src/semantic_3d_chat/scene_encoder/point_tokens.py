from __future__ import annotations

import math

import torch
from torch import nn


def _fixed_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    """Build a deterministic approximately isometric feature projection.

    The matrix is a non-persistent architectural buffer.  It gives raw scene
    content a gradient-free route around the learned point MLP, so optimizing a
    useful soft prompt cannot erase all between-scene differences.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if output_dim >= input_dim:
        sample = torch.randn(output_dim, input_dim, generator=generator)
        return torch.linalg.qr(sample, mode="reduced").Q.transpose(0, 1).contiguous()
    sample = torch.randn(input_dim, output_dim, generator=generator)
    return torch.linalg.qr(sample, mode="reduced").Q.contiguous()


class FourierXYZ(nn.Module):
    """Deterministic Fourier encoding for normalized XYZ coordinates."""

    def __init__(self, bands: int = 8) -> None:
        super().__init__()
        frequencies = (2.0 ** torch.arange(bands, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        return int(3 * self.frequencies.numel() * 2)

    def forward(self, xyz_normalized: torch.Tensor) -> torch.Tensor:
        angles = xyz_normalized.unsqueeze(-1) * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)


class PointTokenProjection(nn.Module):
    """Project high-dimensional voxel payloads without compressing the stored map."""

    def __init__(
        self,
        semantic_dim: int,
        model_dim: int,
        fourier_bands: int = 8,
        *,
        semantic_skip_scale: float = 1.0,
        geometry_skip_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if semantic_skip_scale < 0 or geometry_skip_scale < 0:
            raise ValueError("Point-token skip scales cannot be negative")
        self.semantic_dim = semantic_dim
        self.semantic_skip_scale = float(semantic_skip_scale)
        self.geometry_skip_scale = float(geometry_skip_scale)
        self.fourier = FourierXYZ(fourier_bands)
        # normalized xyz + Fourier xyz + rgb + normal + confidence + log count
        geometry_dim = 3 + self.fourier.output_dim + 3 + 3 + 1 + 1
        input_dim = semantic_dim + geometry_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.register_buffer(
            "fixed_semantic_projection",
            _fixed_projection(semantic_dim, model_dim, 0x53EA),
            persistent=False,
        )
        self.register_buffer(
            "fixed_geometry_projection",
            _fixed_projection(geometry_dim, model_dim, 0x630),
            persistent=False,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if semantic.ndim != 2 or semantic.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"Expected semantic [N,{self.semantic_dim}], got {tuple(semantic.shape)}"
            )
        count = semantic.shape[0]
        if any(value.shape[0] != count for value in (xyz, rgb, normal, confidence, observation_count)):
            raise ValueError("All voxel fields must have the same leading dimension")
        extent = (room_max - room_min).clamp_min(1e-6)
        xyz_normalized = ((xyz - room_min) / extent).mul(2).sub(1)
        rgb_normalized = rgb.float() / 255.0 if rgb.max().item() > 1.0 else rgb.float()
        geometry = torch.cat(
            (
                xyz_normalized.float(),
                self.fourier(xyz_normalized.float()),
                rgb_normalized,
                normal.float(),
                confidence.float().reshape(count, 1),
                torch.log1p(observation_count.float()).reshape(count, 1),
            ),
            dim=-1,
        )
        semantic_float = semantic.float()
        payload = torch.cat((semantic_float, geometry), dim=-1)
        if not torch.isfinite(payload).all():
            raise ValueError("Point-token input contains NaN or infinity")
        learned = self.network(payload)
        semantic_normalized = nn.functional.layer_norm(
            semantic_float, (self.semantic_dim,)
        )
        geometry_normalized = nn.functional.layer_norm(geometry, (geometry.shape[-1],))
        semantic_bypass = torch.matmul(
            semantic_normalized, self.fixed_semantic_projection.to(semantic_normalized)
        )
        geometry_bypass = torch.matmul(
            geometry_normalized, self.fixed_geometry_projection.to(geometry_normalized)
        )
        output = (
            learned
            + self.semantic_skip_scale * semantic_bypass
            + self.geometry_skip_scale * geometry_bypass
        )
        return output, xyz_normalized
