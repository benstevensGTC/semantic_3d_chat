"""Question-independent, all-voxel alignment of dense visual features.

The production Gemma-4 semantic payload is ordered as a 1,536-dimensional
native middle/late stream followed by a 1,536-dimensional language-aligned
tail.  This module learns only a low-rank residual from the native stream into
that aligned tail.  Its API accepts no question, label, object identifier,
coordinate, or other environmental metadata, and it preserves every input
voxel in its original order.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from semantic_3d_chat.language.lora import tensor_state_sha256

DENSE_ALIGNMENT_ARCHITECTURE_VERSION = "middle_late_to_aligned_tail_low_rank_v1"
DENSE_ALIGNMENT_ARCHITECTURE_MARKER = 0x44414C31
DENSE_ALIGNMENT_REPLACE_TAIL = "replace_tail"
DENSE_ALIGNMENT_COVERAGE_SIDECAR = "coverage_sidecar"
_DENSE_ALIGNMENT_APPLICATION_MODES = frozenset(
    {DENSE_ALIGNMENT_REPLACE_TAIL, DENSE_ALIGNMENT_COVERAGE_SIDECAR}
)


@dataclass(frozen=True)
class DenseAlignmentSettings:
    """Config contract for the all-voxel dense-alignment residual."""

    enabled: bool = False
    dense_dim: int = 1536
    aligned_dim: int = 1536
    rank: int = 8
    alpha: float = 16.0
    initialization_seed: int = 25025
    expected_initial_state_sha256: str | None = None
    application_mode: str = DENSE_ALIGNMENT_REPLACE_TAIL
    sidecar_scale: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("dense_alignment.enabled must be a boolean")
        for name, value in {
            "dense_dim": self.dense_dim,
            "aligned_dim": self.aligned_dim,
            "rank": self.rank,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"dense_alignment.{name} must be a positive integer")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError("dense_alignment.alpha must be a finite positive number")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0.0:
            raise ValueError("dense_alignment.alpha must be a finite positive number")
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, int
        ):
            raise TypeError("dense_alignment.initialization_seed must be an integer")
        expected = self.expected_initial_state_sha256
        if expected is not None and re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(
                "dense_alignment.expected_initial_state_sha256 must be lowercase SHA-256"
            )
        if self.enabled and expected is None:
            raise ValueError("Enabled dense_alignment requires expected_initial_state_sha256")
        if self.application_mode not in _DENSE_ALIGNMENT_APPLICATION_MODES:
            raise ValueError(
                "dense_alignment.application_mode must be replace_tail or coverage_sidecar"
            )
        if isinstance(self.sidecar_scale, bool) or not isinstance(
            self.sidecar_scale, (int, float)
        ):
            raise TypeError("dense_alignment.sidecar_scale must be a finite number")
        if not math.isfinite(float(self.sidecar_scale)) or float(self.sidecar_scale) < 0.0:
            raise ValueError("dense_alignment.sidecar_scale must be finite and nonnegative")
        if self.application_mode == DENSE_ALIGNMENT_REPLACE_TAIL and self.sidecar_scale != 0.0:
            raise ValueError("replace_tail mode requires dense_alignment.sidecar_scale=0")

    def contract(self) -> dict[str, Any]:
        if not self.enabled:
            return {"schema_version": 1, "enabled": False}
        contract = {
            "schema_version": 1,
            "enabled": True,
            "architecture_version": DENSE_ALIGNMENT_ARCHITECTURE_VERSION,
            "dense_dim": self.dense_dim,
            "aligned_dim": self.aligned_dim,
            "rank": self.rank,
            "alpha": float(self.alpha),
            "scale": float(self.alpha) / self.rank,
            "initialization_seed": self.initialization_seed,
            "expected_initial_state_sha256": self.expected_initial_state_sha256,
            "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
            "layer_norm_elementwise_affine": False,
            "question_dependent_inputs": False,
            "environmental_metadata_inputs": False,
            "spatial_reduction": "none",
            "every_voxel_retained": True,
        }
        # Preserve the exact historical V25/V26 contract for the default
        # replacement mode.  The sidecar integration is an explicit new
        # contract and can therefore never be loaded accidentally by an older
        # runtime/checkpoint pair.
        if self.application_mode == DENSE_ALIGNMENT_COVERAGE_SIDECAR:
            contract.update(
                {
                    "application_mode": self.application_mode,
                    "sidecar_scale": float(self.sidecar_scale),
                    "base_semantic_path_modified": False,
                }
            )
        return contract


def dense_alignment_settings(config: Mapping[str, Any]) -> DenseAlignmentSettings:
    """Parse ``scene_encoder.dense_alignment`` without permissive extras."""

    scene_encoder = config.get("scene_encoder")
    if not isinstance(scene_encoder, Mapping):
        raise TypeError("scene_encoder config must be a mapping")
    raw = scene_encoder.get("dense_alignment", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("scene_encoder.dense_alignment must be a mapping")
    allowed = {
        "enabled",
        "dense_dim",
        "aligned_dim",
        "rank",
        "alpha",
        "initialization_seed",
        "expected_initial_state_sha256",
        "application_mode",
        "sidecar_scale",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown dense_alignment settings: {unknown}")
    return DenseAlignmentSettings(
        enabled=raw.get("enabled", False),
        dense_dim=raw.get("dense_dim", 1536),
        aligned_dim=raw.get("aligned_dim", 1536),
        rank=raw.get("rank", 8),
        alpha=raw.get("alpha", 16.0),
        initialization_seed=raw.get("initialization_seed", 25025),
        expected_initial_state_sha256=raw.get("expected_initial_state_sha256"),
        application_mode=raw.get("application_mode", DENSE_ALIGNMENT_REPLACE_TAIL),
        sidecar_scale=raw.get("sidecar_scale", 0.0),
    )


def construct_dense_alignment(
    config: Mapping[str, Any], *, semantic_dim: int
) -> DenseAlignmentResidual | None:
    """Construct the configured residual, or ``None`` when disabled."""

    settings = dense_alignment_settings(config)
    if not settings.enabled:
        return None
    return DenseAlignmentResidual(
        semantic_dim=semantic_dim,
        dense_dim=settings.dense_dim,
        aligned_dim=settings.aligned_dim,
        rank=settings.rank,
        alpha=settings.alpha,
        initialization_seed=settings.initialization_seed,
        application_mode=settings.application_mode,
        sidecar_scale=settings.sidecar_scale,
    )


def validate_dense_alignment_state(
    module: DenseAlignmentResidual,
    *,
    expected_parameter_count: int | None = None,
    context: str = "dense_alignment",
) -> dict[str, Any]:
    """Validate a residual with context suitable for checkpoint diagnostics."""

    if not isinstance(module, DenseAlignmentResidual):
        raise TypeError("module must be a DenseAlignmentResidual")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")
    if expected_parameter_count is not None and (
        isinstance(expected_parameter_count, bool)
        or not isinstance(expected_parameter_count, int)
        or expected_parameter_count < 1
    ):
        raise ValueError("expected_parameter_count must be a positive integer or None")
    try:
        audit = module.validate_structural_state()
    except ValueError as error:
        raise ValueError(f"{context}: {error}") from error
    if (
        expected_parameter_count is not None
        and audit["parameter_count"] != expected_parameter_count
    ):
        raise ValueError(
            f"{context}: parameter count mismatch: "
            f"expected={expected_parameter_count} observed={audit['parameter_count']}"
        )
    return audit


class DenseAlignmentResidual(nn.Module):
    """Apply a voxel-local low-rank residual to the aligned semantic tail.

    For ``semantic = concat(dense, tail)`` this computes, in float32,

    ``delta = (alpha / rank) * B(A(LayerNorm(dense)))``
    ``tail' = tail + delta``

    and returns ``concat(dense, tail')``.  ``B`` starts at exact zero, making
    construction an exact functional no-op.  Layer normalization has no
    affine parameters, so A and B are the complete trainable surface.
    """

    def __init__(
        self,
        *,
        semantic_dim: int = 3072,
        dense_dim: int = 1536,
        aligned_dim: int = 1536,
        rank: int = 8,
        alpha: float = 16.0,
        initialization_seed: int = 25025,
        application_mode: str = DENSE_ALIGNMENT_REPLACE_TAIL,
        sidecar_scale: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in {
            "semantic_dim": semantic_dim,
            "dense_dim": dense_dim,
            "aligned_dim": aligned_dim,
            "rank": rank,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if semantic_dim != dense_dim + aligned_dim:
            raise ValueError("semantic_dim must equal dense_dim + aligned_dim")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a finite positive number")
        parsed_alpha = float(alpha)
        if not math.isfinite(parsed_alpha) or parsed_alpha <= 0.0:
            raise ValueError("alpha must be a finite positive number")
        if isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int):
            raise TypeError("initialization_seed must be an integer")
        if application_mode not in _DENSE_ALIGNMENT_APPLICATION_MODES:
            raise ValueError("application_mode must be replace_tail or coverage_sidecar")
        if isinstance(sidecar_scale, bool) or not isinstance(sidecar_scale, (int, float)):
            raise TypeError("sidecar_scale must be a finite number")
        parsed_sidecar_scale = float(sidecar_scale)
        if not math.isfinite(parsed_sidecar_scale) or parsed_sidecar_scale < 0.0:
            raise ValueError("sidecar_scale must be finite and nonnegative")
        if application_mode == DENSE_ALIGNMENT_REPLACE_TAIL and parsed_sidecar_scale != 0.0:
            raise ValueError("replace_tail mode requires sidecar_scale=0")

        self.semantic_dim = int(semantic_dim)
        self.dense_dim = int(dense_dim)
        self.aligned_dim = int(aligned_dim)
        self.rank = int(rank)
        self.alpha = parsed_alpha
        self.initialization_seed = int(initialization_seed)
        self.architecture_version = DENSE_ALIGNMENT_ARCHITECTURE_VERSION
        self.layer_norm_eps = 1e-5
        self.application_mode = application_mode
        self.sidecar_scale = parsed_sidecar_scale

        self.register_buffer(
            "architecture_marker",
            torch.tensor(DENSE_ALIGNMENT_ARCHITECTURE_MARKER, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "scaling",
            torch.tensor(self.alpha / self.rank, dtype=torch.float32),
            persistent=True,
        )

        self.alignment_a = nn.Parameter(torch.empty(self.rank, self.dense_dim, dtype=torch.float32))
        self.alignment_b = nn.Parameter(
            torch.zeros(self.aligned_dim, self.rank, dtype=torch.float32)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.initialization_seed)
        nn.init.kaiming_uniform_(
            self.alignment_a,
            a=math.sqrt(5),
            generator=generator,
        )

    @property
    def parameter_count(self) -> int:
        """Return the complete trainable A/B parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def scale(self) -> float:
        """Return the configured LoRA-style scale ``alpha / rank``."""

        return self.alpha / self.rank

    def state_sha256(self) -> str:
        """Hash all persistent parameters and structural buffers."""

        return tensor_state_sha256(self.state_dict())

    def validate_structural_state(self) -> dict[str, Any]:
        """Fail closed on shape, dtype, marker, scale, or finiteness drift."""

        if self.architecture_version != DENSE_ALIGNMENT_ARCHITECTURE_VERSION:
            raise ValueError("Dense-alignment architecture version does not match implementation")
        expected_marker = torch.tensor(
            DENSE_ALIGNMENT_ARCHITECTURE_MARKER,
            dtype=self.architecture_marker.dtype,
            device=self.architecture_marker.device,
        )
        if not torch.equal(self.architecture_marker, expected_marker):
            raise ValueError("Dense-alignment architecture marker does not match implementation")
        expected_scale = torch.tensor(
            self.scale,
            dtype=self.scaling.dtype,
            device=self.scaling.device,
        )
        if not torch.equal(self.scaling, expected_scale):
            raise ValueError("Dense-alignment persistent scale does not match alpha / rank")
        expected_shapes = {
            "alignment_a": (self.rank, self.dense_dim),
            "alignment_b": (self.aligned_dim, self.rank),
        }
        for name, expected_shape in expected_shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Dense-alignment {name} shape mismatch: "
                    f"expected={expected_shape} observed={tuple(value.shape)}"
                )
            if value.dtype != torch.float32:
                raise ValueError(f"Dense-alignment {name} must remain float32")
            if not torch.isfinite(value).all():
                raise ValueError(f"Dense-alignment {name} contains NaN or infinity")
        if self.scaling.dtype != torch.float32 or not torch.isfinite(self.scaling):
            raise ValueError("Dense-alignment scaling must remain finite float32")
        audit = {
            "architecture_version": self.architecture_version,
            "semantic_dim": self.semantic_dim,
            "dense_dim": self.dense_dim,
            "aligned_dim": self.aligned_dim,
            "rank": self.rank,
            "alpha": self.alpha,
            "scale": self.scale,
            "parameter_count": self.parameter_count,
            "dense_slice": [0, self.dense_dim],
            "aligned_tail_slice": [self.dense_dim, self.semantic_dim],
            "layer_norm_elementwise_affine": False,
            "layer_norm_eps": self.layer_norm_eps,
            "question_dependent_inputs": False,
            "environmental_metadata_inputs": False,
            "spatial_reduction": "none",
            "every_voxel_retained": True,
            "b_exact_zero": bool(torch.count_nonzero(self.alignment_b).item() == 0),
            "state_sha256": self.state_sha256(),
        }
        if self.application_mode == DENSE_ALIGNMENT_COVERAGE_SIDECAR:
            audit.update(
                {
                    "application_mode": self.application_mode,
                    "sidecar_scale": self.sidecar_scale,
                    "base_semantic_path_modified": False,
                }
            )
        return audit

    def scene_inputs(
        self, semantic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, float]:
        """Return the base semantic path plus an optional aligned sidecar.

        ``replace_tail`` retains the V25/V26 behavior.  ``coverage_sidecar``
        leaves the complete semantic tensor untouched and supplies the
        calibrated residual separately for all-voxel spatial coverage.  This
        isolates new semantic evidence from a previously validated scene path.
        """

        if self.application_mode == DENSE_ALIGNMENT_REPLACE_TAIL:
            return self(semantic), None, 0.0
        delta = self.residual_delta(semantic).to(dtype=semantic.dtype)
        return semantic, delta, self.sidecar_scale

    def _validate_semantic(self, semantic: torch.Tensor) -> None:
        if not isinstance(semantic, torch.Tensor):
            raise TypeError("semantic must be a torch.Tensor")
        if semantic.ndim != 2 or semantic.shape[1] != self.semantic_dim:
            raise ValueError(
                f"semantic must have shape [N,{self.semantic_dim}], observed={list(semantic.shape)}"
            )
        if semantic.shape[0] < 1:
            raise ValueError("semantic must contain at least one voxel")
        if not torch.is_floating_point(semantic):
            raise TypeError("semantic must have a floating-point dtype")
        if not torch.isfinite(semantic).all():
            raise ValueError("semantic contains NaN or infinity")
        if semantic.device != self.alignment_a.device:
            raise ValueError("semantic and dense-alignment parameters must be on the same device")

    def residual_delta(self, semantic: torch.Tensor) -> torch.Tensor:
        """Return the FP32 ``[N, aligned_dim]`` residual before tail addition."""

        self._validate_semantic(semantic)
        self.validate_structural_state()
        dense = semantic[:, : self.dense_dim].float()
        normalized = F.layer_norm(
            dense,
            (self.dense_dim,),
            weight=None,
            bias=None,
            eps=self.layer_norm_eps,
        )
        hidden = F.linear(normalized, self.alignment_a)
        delta = F.linear(hidden, self.alignment_b) * self.scaling
        if delta.shape != (semantic.shape[0], self.aligned_dim):
            raise RuntimeError("Dense-alignment residual produced an invalid shape")
        if not torch.isfinite(delta).all():
            raise RuntimeError("Dense-alignment residual produced NaN or infinity")
        return delta

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        """Return adapted ``[N, semantic_dim]`` features without dropping voxels."""

        delta = self.residual_delta(semantic)
        dense = semantic[:, : self.dense_dim]
        aligned_tail = semantic[:, self.dense_dim :]
        adapted_tail = aligned_tail + delta.to(dtype=aligned_tail.dtype)
        output = torch.cat((dense, adapted_tail), dim=-1)
        if output.shape != semantic.shape:
            raise RuntimeError("Dense-alignment residual changed the semantic tensor shape")
        if not torch.isfinite(output).all():
            raise RuntimeError("Dense-alignment residual produced NaN or infinity")
        return output

    def _apply(self, fn: Any, recurse: bool = True) -> DenseAlignmentResidual:
        """Honor device moves while retaining FP32 low-rank state."""

        super()._apply(fn, recurse=recurse)
        for parameter in (self.alignment_a, self.alignment_b):
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
            if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                parameter.grad.data = parameter.grad.data.float()
        if self.scaling.dtype != torch.float32:
            self.scaling.data = self.scaling.data.float()
        return self


__all__ = [
    "DENSE_ALIGNMENT_ARCHITECTURE_MARKER",
    "DENSE_ALIGNMENT_ARCHITECTURE_VERSION",
    "DENSE_ALIGNMENT_COVERAGE_SIDECAR",
    "DENSE_ALIGNMENT_REPLACE_TAIL",
    "DenseAlignmentResidual",
    "DenseAlignmentSettings",
    "construct_dense_alignment",
    "dense_alignment_settings",
    "validate_dense_alignment_state",
]
