from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from torch import nn

from .perceiver import spatial_anchors


@dataclass(frozen=True)
class GlobalSceneResidualSettings:
    enabled: bool = False
    width: int = 128
    fourier_bands: int = 8
    initialization_seed: int = 16015
    expected_initial_state_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("global_scene_residual.enabled must be a boolean")
        for name, value in {
            "width": self.width,
            "fourier_bands": self.fourier_bands,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"global_scene_residual.{name} must be a positive integer")
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, int
        ):
            raise TypeError("global_scene_residual.initialization_seed must be an integer")
        expected = self.expected_initial_state_sha256
        if expected is not None and re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(
                "global_scene_residual.expected_initial_state_sha256 must be lowercase SHA-256"
            )
        if self.enabled and expected is None:
            raise ValueError(
                "Enabled global_scene_residual requires expected_initial_state_sha256"
            )

    def contract(self) -> dict[str, Any]:
        if not self.enabled:
            return {"schema_version": 1, "enabled": False}
        return {
            "schema_version": 1,
            "enabled": True,
            "width": self.width,
            "fourier_bands": self.fourier_bands,
            "initialization_seed": self.initialization_seed,
            "expected_initial_state_sha256": self.expected_initial_state_sha256,
        }


def global_scene_residual_settings(config: Mapping[str, Any]) -> GlobalSceneResidualSettings:
    scene_encoder = config.get("scene_encoder")
    if not isinstance(scene_encoder, Mapping):
        raise TypeError("scene_encoder config must be a mapping")
    raw = scene_encoder.get("global_scene_residual", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("scene_encoder.global_scene_residual must be a mapping")
    allowed = {
        "enabled",
        "width",
        "fourier_bands",
        "initialization_seed",
        "expected_initial_state_sha256",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown global_scene_residual settings: {unknown}")
    return GlobalSceneResidualSettings(
        enabled=raw.get("enabled", False),
        width=raw.get("width", 128),
        fourier_bands=raw.get("fourier_bands", 8),
        initialization_seed=raw.get("initialization_seed", 16015),
        expected_initial_state_sha256=raw.get("expected_initial_state_sha256"),
    )


def construct_global_scene_residual(
    config: Mapping[str, Any], *, scene_dim: int, latent_count: int
) -> GlobalSceneResidual | None:
    settings = global_scene_residual_settings(config)
    if not settings.enabled:
        return None
    return GlobalSceneResidual(
        scene_dim=scene_dim,
        latent_count=latent_count,
        width=settings.width,
        fourier_bands=settings.fourier_bands,
        initialization_seed=settings.initialization_seed,
    )


def apply_global_scene_residual(output: Any, module: GlobalSceneResidual | None) -> Any:
    """Return a tokenizer output whose LM tokens include the static residual.

    The tokenizer's native latents, block tokens, and full-map accounting stay
    untouched.  A shallow audit copy records the exact pre/post magnitude.
    """

    if module is None:
        return output
    scene_tokens = output.scene_tokens
    adapted = module(scene_tokens)
    audit = dict(output.audit)
    audit["global_scene_residual_input_rms"] = (
        scene_tokens.detach().float().square().mean().sqrt()
    )
    audit["global_scene_residual_delta_rms"] = (
        (adapted.detach().float() - scene_tokens.detach().float()).square().mean().sqrt()
    )
    return replace(output, scene_tokens=adapted, audit=audit)


class GlobalSceneResidual(nn.Module):
    """Question-independent, position-aware residual over every scene slot.

    The identity path preserves every input token exactly.  The learned path
    combines each slot with the mean projected content of *all* slots and a
    persistent spatial-position encoding.  Its final projection is initialized
    to exact zero, so adding this module to an older checkpoint does not alter
    that checkpoint's prefix before the first optimizer update.

    This module deliberately accepts no question, answer, retrieval query, or
    oracle coordinate.  It is part of static full-scene tokenization and must be
    run before user text is supplied to the language model.
    """

    def __init__(
        self,
        scene_dim: int,
        latent_count: int,
        width: int,
        fourier_bands: int,
        *,
        initialization_seed: int = 16015,
    ) -> None:
        super().__init__()
        for name, value in {
            "scene_dim": scene_dim,
            "latent_count": latent_count,
            "width": width,
            "fourier_bands": fourier_bands,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int):
            raise TypeError("initialization_seed must be an integer")

        self.scene_dim = int(scene_dim)
        self.latent_count = int(latent_count)
        self.width = int(width)
        self.fourier_bands = int(fourier_bands)
        self.initialization_seed = int(initialization_seed)

        anchors = spatial_anchors(self.latent_count)
        frequencies = (
            2.0 ** torch.arange(self.fourier_bands, dtype=torch.float32)
        ) * math.pi
        angles = anchors.unsqueeze(-1) * frequencies
        fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)
        position_features = torch.cat((anchors, fourier), dim=-1).contiguous()
        # These values define slot identity and therefore belong in checkpoint
        # state and state hashes.  They are intentionally persistent.
        self.register_buffer("position_features", position_features, persistent=True)

        # nn.Linear constructors initialize parameters.  fork_rng makes the
        # initialization deterministic without perturbing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.scene_norm = nn.LayerNorm(self.scene_dim)
            self.scene_projection = nn.Linear(self.scene_dim, self.width)
            self.position_projection = nn.Linear(position_features.shape[-1], self.width)
            self.output_projection = nn.Linear(self.width, self.scene_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        if scene_tokens.ndim != 3:
            raise ValueError("scene_tokens must have shape [B,L,H]")
        expected = (self.latent_count, self.scene_dim)
        if tuple(scene_tokens.shape[1:]) != expected:
            raise ValueError(
                "scene_tokens shape mismatch: "
                f"expected [B,{self.latent_count},{self.scene_dim}], "
                f"observed {list(scene_tokens.shape)}"
            )
        if not torch.isfinite(scene_tokens).all():
            raise ValueError("scene_tokens must contain only finite values")

        normalized = self.scene_norm(scene_tokens)
        local_content = self.scene_projection(normalized)
        # Reusing the local projection keeps the bridge compact while making
        # every residual slot differentiably depend on every scene slot.
        global_content = local_content.float().mean(dim=1, keepdim=True).to(local_content.dtype)
        positions = self.position_projection(
            self.position_features.to(device=scene_tokens.device, dtype=scene_tokens.dtype)
        ).unsqueeze(0)
        hidden = torch.tanh(local_content + global_content + positions)
        output = scene_tokens + self.output_projection(hidden)
        if not torch.isfinite(output).all():
            raise RuntimeError("Global scene residual produced NaN or infinity")
        return output


__all__ = [
    "GlobalSceneResidual",
    "GlobalSceneResidualSettings",
    "apply_global_scene_residual",
    "construct_global_scene_residual",
    "global_scene_residual_settings",
]
