"""Zero-output post-stack fusion for calibrated dense-scene evidence.

The adapter consumes the already-established full-scene tokens and a
question-independent, all-voxel sidecar pooled into the same spatial slots.
It is deliberately applied *after* the frozen scene-token stack.  At
construction both output routes are exact zero, so installing the module is a
bit-identical no-op while still exposing useful first-step gradients.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from semantic_3d_chat.language.lora import tensor_state_sha256

from .perceiver import spatial_anchors

DENSE_SIDECAR_ADAPTER_ARCHITECTURE_VERSION = "post_stack_dense_sidecar_zero_output_v1"
DENSE_SIDECAR_ADAPTER_ARCHITECTURE_MARKER = 0x44534131


def _position_features(latent_count: int, fourier_bands: int) -> torch.Tensor:
    """Return deterministic slot positions and Fourier features in FP32."""

    anchors = spatial_anchors(latent_count)
    frequencies = (2.0 ** torch.arange(fourier_bands, dtype=torch.float32)) * math.pi
    angles = anchors.unsqueeze(-1) * frequencies
    fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)
    return torch.cat((anchors, fourier), dim=-1).contiguous()


@dataclass(frozen=True)
class DenseSidecarAdapterSettings:
    """Configuration and provenance contract for post-stack sidecar fusion."""

    enabled: bool = False
    width: int = 256
    fourier_bands: int = 8
    max_direct_scale: float = 0.25
    initialization_seed: int = 28028
    expected_initial_state_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("dense_sidecar_adapter.enabled must be a boolean")
        for name, value in {
            "width": self.width,
            "fourier_bands": self.fourier_bands,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"dense_sidecar_adapter.{name} must be a positive integer"
                )
        if isinstance(self.max_direct_scale, bool) or not isinstance(
            self.max_direct_scale, (int, float)
        ):
            raise TypeError("dense_sidecar_adapter.max_direct_scale must be numeric")
        if not math.isfinite(float(self.max_direct_scale)) or self.max_direct_scale <= 0:
            raise ValueError(
                "dense_sidecar_adapter.max_direct_scale must be finite and positive"
            )
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, int
        ):
            raise TypeError("dense_sidecar_adapter.initialization_seed must be an integer")
        expected = self.expected_initial_state_sha256
        if expected is not None and re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(
                "dense_sidecar_adapter.expected_initial_state_sha256 must be "
                "lowercase SHA-256"
            )
        if self.enabled and expected is None:
            raise ValueError(
                "Enabled dense_sidecar_adapter requires expected_initial_state_sha256"
            )

    def contract(self) -> dict[str, Any]:
        if not self.enabled:
            return {"schema_version": 1, "enabled": False}
        return {
            "schema_version": 1,
            "enabled": True,
            "architecture_version": DENSE_SIDECAR_ADAPTER_ARCHITECTURE_VERSION,
            "width": self.width,
            "fourier_bands": self.fourier_bands,
            "max_direct_scale": float(self.max_direct_scale),
            "initialization_seed": self.initialization_seed,
            "expected_initial_state_sha256": self.expected_initial_state_sha256,
            "application_point": "post_frozen_scene_stack",
            "output_projection_initialization": "exact_zero",
            "channel_gain_initialization": "exact_zero",
            "normalization": "separate_affine_layer_norm",
            "direct_route": "full_dimensional_tanh_bounded",
            "base_identity_path": True,
            "question_dependent_inputs": False,
            "environmental_metadata_inputs": False,
        }


def dense_sidecar_adapter_settings(
    config: Mapping[str, Any],
) -> DenseSidecarAdapterSettings:
    """Parse ``scene_encoder.dense_sidecar_adapter`` without permissive extras."""

    scene_encoder = config.get("scene_encoder")
    if not isinstance(scene_encoder, Mapping):
        raise TypeError("scene_encoder config must be a mapping")
    raw = scene_encoder.get("dense_sidecar_adapter", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("scene_encoder.dense_sidecar_adapter must be a mapping")
    allowed = {
        "enabled",
        "width",
        "fourier_bands",
        "max_direct_scale",
        "initialization_seed",
        "expected_initial_state_sha256",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown dense_sidecar_adapter settings: {unknown}")
    return DenseSidecarAdapterSettings(
        enabled=raw.get("enabled", False),
        width=raw.get("width", 256),
        fourier_bands=raw.get("fourier_bands", 8),
        max_direct_scale=raw.get("max_direct_scale", 0.25),
        initialization_seed=raw.get("initialization_seed", 28028),
        expected_initial_state_sha256=raw.get("expected_initial_state_sha256"),
    )


def construct_dense_sidecar_adapter(
    config: Mapping[str, Any], *, scene_dim: int, latent_count: int
) -> DenseSidecarAdapter | None:
    """Construct the configured post-stack adapter, or ``None`` if disabled."""

    settings = dense_sidecar_adapter_settings(config)
    if not settings.enabled:
        return None
    return DenseSidecarAdapter(
        scene_dim=scene_dim,
        latent_count=latent_count,
        width=settings.width,
        fourier_bands=settings.fourier_bands,
        max_direct_scale=settings.max_direct_scale,
        initialization_seed=settings.initialization_seed,
    )


def validate_dense_sidecar_adapter_state(
    module: DenseSidecarAdapter,
    *,
    expected_parameter_count: int | None = None,
    expected_state_sha256: str | None = None,
    context: str = "dense_sidecar_adapter",
) -> dict[str, Any]:
    """Validate structure plus optional exact count/hash provenance."""

    if not isinstance(module, DenseSidecarAdapter):
        raise TypeError("module must be a DenseSidecarAdapter")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")
    if expected_parameter_count is not None and (
        isinstance(expected_parameter_count, bool)
        or not isinstance(expected_parameter_count, int)
        or expected_parameter_count < 1
    ):
        raise ValueError("expected_parameter_count must be a positive integer or None")
    if expected_state_sha256 is not None and re.fullmatch(
        r"[0-9a-f]{64}", expected_state_sha256
    ) is None:
        raise ValueError("expected_state_sha256 must be lowercase SHA-256 or None")
    try:
        audit = module.validate_structural_state()
    except ValueError as error:
        raise ValueError(f"{context}: {error}") from error
    if (
        expected_parameter_count is not None
        and audit["parameter_count"] != expected_parameter_count
    ):
        raise ValueError(
            f"{context}: parameter count mismatch: expected={expected_parameter_count} "
            f"observed={audit['parameter_count']}"
        )
    if expected_state_sha256 is not None and audit["state_sha256"] != expected_state_sha256:
        raise ValueError(
            f"{context}: state SHA-256 mismatch: expected={expected_state_sha256} "
            f"observed={audit['state_sha256']}"
        )
    return audit


def apply_dense_sidecar_adapter(
    output: Any, module: DenseSidecarAdapter | None
) -> Any:
    """Apply sidecar fusion after the complete frozen scene stack.

    When enabled, the tokenizer output must expose its separately routed
    ``aligned_sidecar_tokens``.  Only ``scene_tokens`` and a shallow audit copy
    are replaced; native latents, block tokens, and coverage accounting remain
    intact.
    """

    if module is None:
        return output
    if not hasattr(output, "scene_tokens") or not hasattr(output, "audit"):
        raise TypeError("output must expose scene_tokens and audit")
    if not hasattr(output, "aligned_sidecar_tokens"):
        raise ValueError("output must expose aligned_sidecar_tokens")
    sidecar_tokens = output.aligned_sidecar_tokens
    if sidecar_tokens is None:
        raise ValueError(
            "dense sidecar adapter requires nonempty output.aligned_sidecar_tokens"
        )
    base_tokens = output.scene_tokens
    adapted = module(base_tokens, sidecar_tokens)
    delta = adapted.detach().float() - base_tokens.detach().float()
    audit = dict(output.audit)
    audit.update(
        {
            "dense_sidecar_adapter_input_rms": (
                base_tokens.detach().float().square().mean().sqrt()
            ),
            "dense_sidecar_adapter_sidecar_rms": (
                sidecar_tokens.detach().float().square().mean().sqrt()
            ),
            "dense_sidecar_adapter_delta_rms": delta.square().mean().sqrt(),
            "dense_sidecar_adapter_output_rms": (
                adapted.detach().float().square().mean().sqrt()
            ),
            "dense_sidecar_adapter_direct_gain_abs_max": (
                module.bounded_channel_gain().detach().float().abs().max()
            ),
        }
    )
    return replace(output, scene_tokens=adapted, audit=audit)


class DenseSidecarAdapter(nn.Module):
    """Fuse base and all-voxel sidecar scene slots through two safe routes.

    The learned hidden route jointly projects normalized base content,
    normalized sidecar content, and deterministic position features.  Its
    output projection starts at exact zero.  In parallel, a full-dimensional
    direct sidecar route is bounded per channel by ``max_direct_scale`` and its
    gain also starts at exact zero.  Consequently construction is an exact
    identity while both output surfaces receive gradients on the first step.
    """

    def __init__(
        self,
        *,
        scene_dim: int,
        latent_count: int,
        width: int = 256,
        fourier_bands: int = 8,
        max_direct_scale: float = 0.25,
        initialization_seed: int = 28028,
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
        if isinstance(max_direct_scale, bool) or not isinstance(
            max_direct_scale, (int, float)
        ):
            raise TypeError("max_direct_scale must be numeric")
        parsed_direct_scale = float(max_direct_scale)
        if not math.isfinite(parsed_direct_scale) or parsed_direct_scale <= 0.0:
            raise ValueError("max_direct_scale must be finite and positive")
        if isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int):
            raise TypeError("initialization_seed must be an integer")

        self.scene_dim = int(scene_dim)
        self.latent_count = int(latent_count)
        self.width = int(width)
        self.fourier_bands = int(fourier_bands)
        self.configured_max_direct_scale = parsed_direct_scale
        self.initialization_seed = int(initialization_seed)
        self.architecture_version = DENSE_SIDECAR_ADAPTER_ARCHITECTURE_VERSION

        positions = _position_features(self.latent_count, self.fourier_bands)
        self.register_buffer("position_features", positions, persistent=True)
        self.register_buffer(
            "architecture_marker",
            torch.tensor(DENSE_SIDECAR_ADAPTER_ARCHITECTURE_MARKER, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "max_direct_scale",
            torch.tensor(parsed_direct_scale, dtype=torch.float32),
            persistent=True,
        )

        # Isolate deterministic construction from the caller's global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.base_norm = nn.LayerNorm(self.scene_dim)
            self.sidecar_norm = nn.LayerNorm(self.scene_dim)
            self.base_projection = nn.Linear(self.scene_dim, self.width)
            self.sidecar_projection = nn.Linear(self.scene_dim, self.width)
            self.position_projection = nn.Linear(positions.shape[-1], self.width)
            self.output_projection = nn.Linear(self.width, self.scene_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self.channel_gain = nn.Parameter(torch.zeros(self.scene_dim, dtype=torch.float32))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def state_sha256(self) -> str:
        """Hash every persistent parameter and structural buffer."""

        return tensor_state_sha256(self.state_dict())

    def validate_structural_state(self) -> dict[str, Any]:
        """Fail closed on architecture, shape, deterministic state, or finiteness drift."""

        if self.architecture_version != DENSE_SIDECAR_ADAPTER_ARCHITECTURE_VERSION:
            raise ValueError(
                "Dense-sidecar-adapter architecture version does not match implementation"
            )
        expected_marker = torch.tensor(
            DENSE_SIDECAR_ADAPTER_ARCHITECTURE_MARKER,
            dtype=self.architecture_marker.dtype,
            device=self.architecture_marker.device,
        )
        if self.architecture_marker.dtype != torch.int64:
            raise ValueError("Dense-sidecar-adapter architecture marker must remain int64")
        if self.architecture_marker.ndim != 0 or not torch.equal(
            self.architecture_marker, expected_marker
        ):
            raise ValueError("Dense-sidecar-adapter architecture marker mismatch")
        expected_scale = torch.tensor(
            self.configured_max_direct_scale,
            dtype=self.max_direct_scale.dtype,
            device=self.max_direct_scale.device,
        )
        if self.max_direct_scale.ndim != 0 or not torch.equal(
            self.max_direct_scale, expected_scale
        ):
            raise ValueError(
                "Persistent max direct scale does not match active configuration"
            )
        expected_positions = _position_features(
            self.latent_count, self.fourier_bands
        ).to(device=self.position_features.device, dtype=self.position_features.dtype)
        if self.position_features.shape != expected_positions.shape or not torch.equal(
            self.position_features, expected_positions
        ):
            raise ValueError(
                "Persistent position features do not match deterministic spatial anchors"
            )

        position_dim = expected_positions.shape[-1]
        expected_shapes = {
            "base_norm.weight": (self.scene_dim,),
            "base_norm.bias": (self.scene_dim,),
            "sidecar_norm.weight": (self.scene_dim,),
            "sidecar_norm.bias": (self.scene_dim,),
            "base_projection.weight": (self.width, self.scene_dim),
            "base_projection.bias": (self.width,),
            "sidecar_projection.weight": (self.width, self.scene_dim),
            "sidecar_projection.bias": (self.width,),
            "position_projection.weight": (self.width, position_dim),
            "position_projection.bias": (self.width,),
            "output_projection.weight": (self.scene_dim, self.width),
            "channel_gain": (self.scene_dim,),
        }
        state = self.state_dict()
        for name, expected_shape in expected_shapes.items():
            if name not in state:
                raise ValueError(f"Dense-sidecar-adapter state is missing {name}")
            value = state[name]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Dense-sidecar-adapter {name} shape mismatch: "
                    f"expected={expected_shape} observed={tuple(value.shape)}"
                )
            if not torch.is_floating_point(value):
                raise ValueError(f"Dense-sidecar-adapter {name} must be floating point")
            if not torch.isfinite(value).all():
                raise ValueError(f"Dense-sidecar-adapter {name} contains NaN or infinity")
        for name, value in state.items():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise ValueError(f"Dense-sidecar-adapter {name} contains NaN or infinity")

        return {
            "architecture_version": self.architecture_version,
            "architecture_marker": DENSE_SIDECAR_ADAPTER_ARCHITECTURE_MARKER,
            "scene_dim": self.scene_dim,
            "latent_count": self.latent_count,
            "width": self.width,
            "fourier_bands": self.fourier_bands,
            "position_dim": int(position_dim),
            "max_direct_scale": self.configured_max_direct_scale,
            "parameter_count": self.parameter_count,
            "application_point": "post_frozen_scene_stack",
            "normalization": "separate_affine_layer_norm",
            "output_projection_exact_zero": bool(
                torch.count_nonzero(self.output_projection.weight).item() == 0
            ),
            "channel_gain_exact_zero": bool(
                torch.count_nonzero(self.channel_gain).item() == 0
            ),
            "direct_route": "full_dimensional_tanh_bounded",
            "base_identity_path": True,
            "question_dependent_inputs": False,
            "environmental_metadata_inputs": False,
            "state_sha256": self.state_sha256(),
        }

    def _validate_inputs(
        self, base_scene_tokens: torch.Tensor, aligned_sidecar_tokens: torch.Tensor
    ) -> None:
        for name, value in {
            "base_scene_tokens": base_scene_tokens,
            "aligned_sidecar_tokens": aligned_sidecar_tokens,
        }.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != 3 or tuple(value.shape[1:]) != (
                self.latent_count,
                self.scene_dim,
            ):
                raise ValueError(
                    f"{name} must have shape [B,{self.latent_count},{self.scene_dim}], "
                    f"observed={list(value.shape)}"
                )
            if value.shape[0] < 1:
                raise ValueError(f"{name} must contain at least one scene")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must have a floating-point dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
        if base_scene_tokens.shape != aligned_sidecar_tokens.shape:
            raise ValueError("base and sidecar scene-token shapes must match exactly")
        if base_scene_tokens.dtype != aligned_sidecar_tokens.dtype:
            raise ValueError("base and sidecar scene tokens must have the same dtype")
        if base_scene_tokens.device != aligned_sidecar_tokens.device:
            raise ValueError("base and sidecar scene tokens must be on the same device")
        if base_scene_tokens.device != self.output_projection.weight.device:
            raise ValueError("scene tokens and dense-sidecar-adapter must be on the same device")
        if base_scene_tokens.dtype != self.output_projection.weight.dtype:
            raise ValueError("scene tokens and dense-sidecar-adapter must have the same dtype")

    def bounded_channel_gain(self) -> torch.Tensor:
        """Return the per-channel direct gain, strictly bounded by configuration."""

        self.validate_structural_state()
        return self._bounded_channel_gain_unchecked()

    def _bounded_channel_gain_unchecked(self) -> torch.Tensor:
        """Return the gain after a caller has completed structural preflight."""

        return self.max_direct_scale.to(dtype=self.channel_gain.dtype) * torch.tanh(
            self.channel_gain
        )

    def residual_delta(
        self, base_scene_tokens: torch.Tensor, aligned_sidecar_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Return the learned plus full-dimensional direct residual."""

        self._validate_inputs(base_scene_tokens, aligned_sidecar_tokens)
        normalized_base = self.base_norm(base_scene_tokens)
        normalized_sidecar = self.sidecar_norm(aligned_sidecar_tokens)
        positions = self.position_projection(
            self.position_features.to(
                device=base_scene_tokens.device, dtype=base_scene_tokens.dtype
            )
        ).unsqueeze(0)
        hidden = torch.tanh(
            self.base_projection(normalized_base)
            + self.sidecar_projection(normalized_sidecar)
            + positions
        )
        learned_delta = self.output_projection(hidden)
        direct_delta = self._bounded_channel_gain_unchecked().view(1, 1, -1) * torch.tanh(
            normalized_sidecar
        )
        delta = learned_delta + direct_delta
        if delta.shape != base_scene_tokens.shape:
            raise RuntimeError("Dense-sidecar adapter produced an invalid residual shape")
        if not torch.isfinite(delta).all():
            raise RuntimeError("Dense-sidecar adapter produced NaN or infinity")
        return delta

    def forward(
        self, base_scene_tokens: torch.Tensor, aligned_sidecar_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Return adapted scene tokens while retaining an exact base identity path."""

        delta = self.residual_delta(base_scene_tokens, aligned_sidecar_tokens)
        output = base_scene_tokens + delta.to(dtype=base_scene_tokens.dtype)
        if output.shape != base_scene_tokens.shape:
            raise RuntimeError("Dense-sidecar adapter changed the scene-token shape")
        if not torch.isfinite(output).all():
            raise RuntimeError("Dense-sidecar adapter produced NaN or infinity")
        return output


__all__ = [
    "DENSE_SIDECAR_ADAPTER_ARCHITECTURE_MARKER",
    "DENSE_SIDECAR_ADAPTER_ARCHITECTURE_VERSION",
    "DenseSidecarAdapter",
    "DenseSidecarAdapterSettings",
    "apply_dense_sidecar_adapter",
    "construct_dense_sidecar_adapter",
    "dense_sidecar_adapter_settings",
    "validate_dense_sidecar_adapter_state",
]
