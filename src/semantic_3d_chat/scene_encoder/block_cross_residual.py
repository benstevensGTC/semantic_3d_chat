"""Question-independent all-block cross-attention residual for V35.

The module reads every occupied-block token emitted by the scene tokenizer and
adds a bounded residual to the already-established scene prefix.  Its output
projection starts at exact zero, making installation a bit-identical no-op.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from semantic_3d_chat.language.lora import tensor_state_sha256

from .perceiver import spatial_anchors

BLOCK_CROSS_RESIDUAL_ARCHITECTURE_VERSION = "all_block_fp32_cross_residual_zero_output_v1"
BLOCK_CROSS_RESIDUAL_ARCHITECTURE_MARKER = 0x42435231
BLOCK_CROSS_RESIDUAL_SCENE_DIM = 1536
BLOCK_CROSS_RESIDUAL_BLOCK_DIM = 384
BLOCK_CROSS_RESIDUAL_LATENT_COUNT = 256
BLOCK_CROSS_RESIDUAL_ATTENTION_DIM = 256
BLOCK_CROSS_RESIDUAL_HEADS = 4
BLOCK_CROSS_RESIDUAL_HEAD_DIM = 64
BLOCK_CROSS_RESIDUAL_PARAMETER_COUNT = 983_040
BLOCK_CROSS_RESIDUAL_SPATIAL_TEMPERATURE = 0.20
BLOCK_CROSS_RESIDUAL_UNIFORM_FLOOR = 0.01
BLOCK_CROSS_RESIDUAL_RESIDUAL_SCALE = 0.25
BLOCK_CROSS_RESIDUAL_INITIALIZATION_SEED = 35_035


def _fixed_number(name: str, value: object, expected: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"block_cross_residual.{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed != expected:
        raise ValueError(f"block_cross_residual.{name} must equal {expected}")
    return parsed


def _fixed_integer(name: str, value: object, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"block_cross_residual.{name} must be an integer")
    if value != expected:
        raise ValueError(f"block_cross_residual.{name} must equal {expected}")
    return value


@dataclass(frozen=True)
class BlockCrossResidualSettings:
    enabled: bool = False
    attention_dim: int = BLOCK_CROSS_RESIDUAL_ATTENTION_DIM
    heads: int = BLOCK_CROSS_RESIDUAL_HEADS
    spatial_temperature: float = BLOCK_CROSS_RESIDUAL_SPATIAL_TEMPERATURE
    uniform_floor: float = BLOCK_CROSS_RESIDUAL_UNIFORM_FLOOR
    residual_scale: float = BLOCK_CROSS_RESIDUAL_RESIDUAL_SCALE
    initialization_seed: int = BLOCK_CROSS_RESIDUAL_INITIALIZATION_SEED
    expected_initial_state_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("block_cross_residual.enabled must be a boolean")
        _fixed_integer("attention_dim", self.attention_dim, BLOCK_CROSS_RESIDUAL_ATTENTION_DIM)
        _fixed_integer("heads", self.heads, BLOCK_CROSS_RESIDUAL_HEADS)
        _fixed_number(
            "spatial_temperature",
            self.spatial_temperature,
            BLOCK_CROSS_RESIDUAL_SPATIAL_TEMPERATURE,
        )
        _fixed_number("uniform_floor", self.uniform_floor, BLOCK_CROSS_RESIDUAL_UNIFORM_FLOOR)
        _fixed_number("residual_scale", self.residual_scale, BLOCK_CROSS_RESIDUAL_RESIDUAL_SCALE)
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, int
        ):
            raise TypeError("block_cross_residual.initialization_seed must be an integer")
        if self.initialization_seed < 0:
            raise ValueError("block_cross_residual.initialization_seed cannot be negative")
        expected = self.expected_initial_state_sha256
        if expected is not None and re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(
                "block_cross_residual.expected_initial_state_sha256 must be lowercase SHA-256"
            )
        if self.enabled and expected is None:
            raise ValueError(
                "Enabled block_cross_residual requires expected_initial_state_sha256"
            )

    def contract(self) -> dict[str, Any]:
        if not self.enabled:
            return {"schema_version": 1, "enabled": False}
        return {
            "schema_version": 1,
            "enabled": True,
            "architecture_version": BLOCK_CROSS_RESIDUAL_ARCHITECTURE_VERSION,
            "scene_dim": BLOCK_CROSS_RESIDUAL_SCENE_DIM,
            "block_dim": BLOCK_CROSS_RESIDUAL_BLOCK_DIM,
            "latent_count": BLOCK_CROSS_RESIDUAL_LATENT_COUNT,
            "attention_dim": self.attention_dim,
            "heads": self.heads,
            "head_dim": BLOCK_CROSS_RESIDUAL_HEAD_DIM,
            "spatial_temperature": float(self.spatial_temperature),
            "uniform_floor": float(self.uniform_floor),
            "residual_scale": float(self.residual_scale),
            "initialization_seed": self.initialization_seed,
            "expected_initial_state_sha256": self.expected_initial_state_sha256,
            "parameter_count": BLOCK_CROSS_RESIDUAL_PARAMETER_COUNT,
            "attention": "custom_fp32_all_block_cross_attention",
            "base_normalization": "non_affine_layer_norm",
            "block_normalization": "mean_center_then_non_affine_layer_norm",
            "output_projection_initialization": "exact_zero",
            "question_dependent_inputs": False,
            "environmental_metadata_inputs": False,
        }


def block_cross_residual_settings(config: Mapping[str, Any]) -> BlockCrossResidualSettings:
    scene_encoder = config.get("scene_encoder")
    if not isinstance(scene_encoder, Mapping):
        raise TypeError("scene_encoder config must be a mapping")
    raw = scene_encoder.get("block_cross_residual", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("scene_encoder.block_cross_residual must be a mapping")
    allowed = {
        "enabled",
        "attention_dim",
        "heads",
        "spatial_temperature",
        "uniform_floor",
        "residual_scale",
        "initialization_seed",
        "expected_initial_state_sha256",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown block_cross_residual settings: {unknown}")
    return BlockCrossResidualSettings(
        enabled=raw.get("enabled", False),
        attention_dim=raw.get("attention_dim", BLOCK_CROSS_RESIDUAL_ATTENTION_DIM),
        heads=raw.get("heads", BLOCK_CROSS_RESIDUAL_HEADS),
        spatial_temperature=raw.get(
            "spatial_temperature", BLOCK_CROSS_RESIDUAL_SPATIAL_TEMPERATURE
        ),
        uniform_floor=raw.get("uniform_floor", BLOCK_CROSS_RESIDUAL_UNIFORM_FLOOR),
        residual_scale=raw.get("residual_scale", BLOCK_CROSS_RESIDUAL_RESIDUAL_SCALE),
        initialization_seed=raw.get(
            "initialization_seed", BLOCK_CROSS_RESIDUAL_INITIALIZATION_SEED
        ),
        expected_initial_state_sha256=raw.get("expected_initial_state_sha256"),
    )


def construct_block_cross_residual(
    config: Mapping[str, Any], *, scene_dim: int, block_dim: int, latent_count: int
) -> BlockCrossResidual | None:
    settings = block_cross_residual_settings(config)
    if not settings.enabled:
        return None
    return BlockCrossResidual(
        scene_dim=scene_dim,
        block_dim=block_dim,
        latent_count=latent_count,
        attention_dim=settings.attention_dim,
        heads=settings.heads,
        spatial_temperature=settings.spatial_temperature,
        uniform_floor=settings.uniform_floor,
        residual_scale=settings.residual_scale,
        initialization_seed=settings.initialization_seed,
    )


def validate_block_cross_residual_state(
    module: BlockCrossResidual,
    *,
    expected_parameter_count: int | None = None,
    expected_state_sha256: str | None = None,
    context: str = "block_cross_residual",
) -> dict[str, Any]:
    if not isinstance(module, BlockCrossResidual):
        raise TypeError("module must be a BlockCrossResidual")
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
    if expected_parameter_count is not None and audit["parameter_count"] != expected_parameter_count:
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


def apply_block_cross_residual(output: Any, module: BlockCrossResidual | None) -> Any:
    if module is None:
        return output
    for name in ("scene_tokens", "block_tokens", "audit"):
        if not hasattr(output, name):
            raise TypeError(f"output must expose {name}")
    if not isinstance(output.audit, Mapping):
        raise TypeError("output.audit must be a mapping")
    positions = output.audit.get("block_token_positions_normalized")
    if positions is None:
        raise ValueError("output.audit must contain block_token_positions_normalized")
    adapted, residual_audit = module.forward_with_audit(
        output.scene_tokens, output.block_tokens, positions
    )
    audit = dict(output.audit)
    audit.update(residual_audit)
    return replace(output, scene_tokens=adapted, audit=audit)


class BlockCrossResidual(nn.Module):
    """Four-head FP32 cross-attention from all block tokens to scene slots."""

    def __init__(
        self,
        *,
        scene_dim: int = BLOCK_CROSS_RESIDUAL_SCENE_DIM,
        block_dim: int = BLOCK_CROSS_RESIDUAL_BLOCK_DIM,
        latent_count: int = BLOCK_CROSS_RESIDUAL_LATENT_COUNT,
        attention_dim: int = BLOCK_CROSS_RESIDUAL_ATTENTION_DIM,
        heads: int = BLOCK_CROSS_RESIDUAL_HEADS,
        spatial_temperature: float = BLOCK_CROSS_RESIDUAL_SPATIAL_TEMPERATURE,
        uniform_floor: float = BLOCK_CROSS_RESIDUAL_UNIFORM_FLOOR,
        residual_scale: float = BLOCK_CROSS_RESIDUAL_RESIDUAL_SCALE,
        initialization_seed: int = BLOCK_CROSS_RESIDUAL_INITIALIZATION_SEED,
    ) -> None:
        super().__init__()
        for name, value, expected in (
            ("scene_dim", scene_dim, BLOCK_CROSS_RESIDUAL_SCENE_DIM),
            ("block_dim", block_dim, BLOCK_CROSS_RESIDUAL_BLOCK_DIM),
            ("latent_count", latent_count, BLOCK_CROSS_RESIDUAL_LATENT_COUNT),
            ("attention_dim", attention_dim, BLOCK_CROSS_RESIDUAL_ATTENTION_DIM),
            ("heads", heads, BLOCK_CROSS_RESIDUAL_HEADS),
        ):
            _fixed_integer(name, value, expected)
        _fixed_number(
            "spatial_temperature", spatial_temperature, BLOCK_CROSS_RESIDUAL_SPATIAL_TEMPERATURE
        )
        _fixed_number("uniform_floor", uniform_floor, BLOCK_CROSS_RESIDUAL_UNIFORM_FLOOR)
        _fixed_number("residual_scale", residual_scale, BLOCK_CROSS_RESIDUAL_RESIDUAL_SCALE)
        if isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int):
            raise TypeError("block_cross_residual.initialization_seed must be an integer")
        if initialization_seed < 0:
            raise ValueError("block_cross_residual.initialization_seed cannot be negative")

        self.scene_dim = scene_dim
        self.block_dim = block_dim
        self.latent_count = latent_count
        self.attention_dim = attention_dim
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.configured_spatial_temperature = float(spatial_temperature)
        self.configured_uniform_floor = float(uniform_floor)
        self.configured_residual_scale = float(residual_scale)
        self.initialization_seed = initialization_seed
        self.architecture_version = BLOCK_CROSS_RESIDUAL_ARCHITECTURE_VERSION

        self.w_q = nn.Parameter(torch.empty(scene_dim, attention_dim, dtype=torch.float32))
        self.w_k = nn.Parameter(torch.empty(block_dim, attention_dim, dtype=torch.float32))
        self.w_v = nn.Parameter(torch.empty(block_dim, attention_dim, dtype=torch.float32))
        self.w_o = nn.Parameter(torch.zeros(attention_dim, scene_dim, dtype=torch.float32))
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        with torch.no_grad():
            for weight in (self.w_q, self.w_k, self.w_v):
                bound = math.sqrt(6.0 / (weight.shape[0] + weight.shape[1]))
                weight.uniform_(-bound, bound, generator=generator)

        self.register_buffer(
            "architecture_marker",
            torch.tensor(BLOCK_CROSS_RESIDUAL_ARCHITECTURE_MARKER, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "architecture_dimensions",
            torch.tensor(
                [scene_dim, block_dim, latent_count, attention_dim, heads, self.head_dim],
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.register_buffer(
            "initialization_seed_state",
            torch.tensor(initialization_seed, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer("latent_anchors", spatial_anchors(latent_count), persistent=True)
        self.register_buffer(
            "spatial_temperature",
            torch.tensor(spatial_temperature, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "uniform_floor", torch.tensor(uniform_floor, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "residual_scale", torch.tensor(residual_scale, dtype=torch.float32), persistent=True
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def state_sha256(self) -> str:
        return tensor_state_sha256(self.state_dict())

    def validate_structural_state(self) -> dict[str, Any]:
        if self.architecture_version != BLOCK_CROSS_RESIDUAL_ARCHITECTURE_VERSION:
            raise ValueError("Block-cross-residual architecture version mismatch")
        expected_shapes = {
            "w_q": (BLOCK_CROSS_RESIDUAL_SCENE_DIM, BLOCK_CROSS_RESIDUAL_ATTENTION_DIM),
            "w_k": (BLOCK_CROSS_RESIDUAL_BLOCK_DIM, BLOCK_CROSS_RESIDUAL_ATTENTION_DIM),
            "w_v": (BLOCK_CROSS_RESIDUAL_BLOCK_DIM, BLOCK_CROSS_RESIDUAL_ATTENTION_DIM),
            "w_o": (BLOCK_CROSS_RESIDUAL_ATTENTION_DIM, BLOCK_CROSS_RESIDUAL_SCENE_DIM),
        }
        for name, shape in expected_shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"Block-cross-residual {name} shape mismatch: expected={shape} observed={tuple(value.shape)}")
            if value.dtype != torch.float32:
                raise ValueError(f"Block-cross-residual {name} must remain float32")
            if not torch.isfinite(value).all():
                raise ValueError(f"Block-cross-residual {name} contains NaN or infinity")
        expected_dimensions = torch.tensor(
            [1536, 384, 256, 256, 4, 64],
            dtype=torch.int64,
            device=self.architecture_dimensions.device,
        )
        if self.architecture_dimensions.dtype != torch.int64 or not torch.equal(
            self.architecture_dimensions, expected_dimensions
        ):
            raise ValueError("Block-cross-residual architecture dimensions mismatch")
        expected_marker = torch.tensor(
            BLOCK_CROSS_RESIDUAL_ARCHITECTURE_MARKER,
            dtype=torch.int64,
            device=self.architecture_marker.device,
        )
        if self.architecture_marker.ndim != 0 or not torch.equal(
            self.architecture_marker, expected_marker
        ):
            raise ValueError("Block-cross-residual architecture marker mismatch")
        expected_seed = torch.tensor(
            self.initialization_seed,
            dtype=torch.int64,
            device=self.initialization_seed_state.device,
        )
        if self.initialization_seed_state.ndim != 0 or not torch.equal(
            self.initialization_seed_state, expected_seed
        ):
            raise ValueError("Block-cross-residual initialization seed mismatch")
        expected_anchors = spatial_anchors(self.latent_count).to(self.latent_anchors)
        if self.latent_anchors.dtype != torch.float32 or not torch.equal(
            self.latent_anchors, expected_anchors
        ):
            raise ValueError("Block-cross-residual latent anchors mismatch")
        for name, configured in (
            ("spatial_temperature", self.configured_spatial_temperature),
            ("uniform_floor", self.configured_uniform_floor),
            ("residual_scale", self.configured_residual_scale),
        ):
            value = getattr(self, name)
            expected = torch.tensor(configured, dtype=torch.float32, device=value.device)
            if value.dtype != torch.float32 or value.ndim != 0 or not torch.equal(value, expected):
                raise ValueError(f"Block-cross-residual persistent {name} mismatch")
        if self.parameter_count != BLOCK_CROSS_RESIDUAL_PARAMETER_COUNT:
            raise ValueError(
                "Block-cross-residual parameter count mismatch: "
                f"expected={BLOCK_CROSS_RESIDUAL_PARAMETER_COUNT} observed={self.parameter_count}"
            )
        return {
            "architecture_version": self.architecture_version,
            "architecture_marker": BLOCK_CROSS_RESIDUAL_ARCHITECTURE_MARKER,
            "scene_dim": self.scene_dim,
            "block_dim": self.block_dim,
            "latent_count": self.latent_count,
            "attention_dim": self.attention_dim,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "parameter_count": self.parameter_count,
            "spatial_temperature": self.configured_spatial_temperature,
            "uniform_floor": self.configured_uniform_floor,
            "residual_scale": self.configured_residual_scale,
            "output_projection_exact_zero": bool(torch.count_nonzero(self.w_o).item() == 0),
            "all_blocks_processed": True,
            "question_dependent_inputs": False,
            "environmental_metadata_inputs": False,
            "state_sha256": self.state_sha256(),
        }

    def _validated_inputs(
        self,
        base_scene_tokens: torch.Tensor,
        block_tokens: torch.Tensor,
        block_positions_normalized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for name, value in {
            "base_scene_tokens": base_scene_tokens,
            "block_tokens": block_tokens,
            "block_positions_normalized": block_positions_normalized,
        }.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must have a floating-point dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
        if base_scene_tokens.ndim != 3 or tuple(base_scene_tokens.shape[1:]) != (
            self.latent_count,
            self.scene_dim,
        ) or base_scene_tokens.shape[0] < 1:
            raise ValueError(
                f"base_scene_tokens must have shape [B,{self.latent_count},{self.scene_dim}]"
            )
        batch = base_scene_tokens.shape[0]
        if block_tokens.ndim == 2:
            if block_tokens.shape[0] < 1 or block_tokens.shape[1] != self.block_dim:
                raise ValueError(f"block_tokens must be nonempty [T,{self.block_dim}]")
            blocks = block_tokens.unsqueeze(0).expand(batch, -1, -1)
        elif block_tokens.ndim == 3:
            if block_tokens.shape[0] != batch or block_tokens.shape[1] < 1 or block_tokens.shape[2] != self.block_dim:
                raise ValueError(f"block_tokens must have shape [B,T,{self.block_dim}]")
            blocks = block_tokens
        else:
            raise ValueError(f"block_tokens must have shape [T,{self.block_dim}] or [B,T,{self.block_dim}]")
        token_count = blocks.shape[1]
        if block_positions_normalized.ndim == 2:
            if tuple(block_positions_normalized.shape) != (token_count, 3):
                raise ValueError("block_positions_normalized must match block token count")
            positions = block_positions_normalized.unsqueeze(0).expand(batch, -1, -1)
        elif block_positions_normalized.ndim == 3:
            if tuple(block_positions_normalized.shape) != (batch, token_count, 3):
                raise ValueError("block_positions_normalized must have shape [B,T,3]")
            positions = block_positions_normalized
        else:
            raise ValueError("block_positions_normalized must have shape [T,3] or [B,T,3]")
        if not (base_scene_tokens.device == blocks.device == positions.device == self.w_q.device):
            raise ValueError("block-cross-residual inputs and module must be on the same device")
        if base_scene_tokens.dtype != blocks.dtype:
            raise ValueError("base_scene_tokens and block_tokens must have the same dtype")
        return base_scene_tokens, blocks, positions

    def _attention_context(
        self,
        base_scene_tokens: torch.Tensor,
        block_tokens: torch.Tensor,
        block_positions_normalized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scene, blocks, positions = self._validated_inputs(
            base_scene_tokens, block_tokens, block_positions_normalized
        )
        scene_fp32 = F.layer_norm(scene.float(), (self.scene_dim,))
        blocks_fp32 = blocks.float()
        centered_blocks = blocks_fp32 - blocks_fp32.mean(dim=1, keepdim=True)
        normalized_blocks = F.layer_norm(centered_blocks, (self.block_dim,))
        query = torch.matmul(scene_fp32, self.w_q)
        key = torch.matmul(normalized_blocks, self.w_k)
        value = torch.matmul(normalized_blocks, self.w_v)
        batch, token_count = blocks.shape[:2]
        query = query.view(batch, self.latent_count, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, token_count, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, token_count, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        squared_distance = (
            self.latent_anchors.view(1, self.latent_count, 1, 3)
            - positions.float().unsqueeze(1)
        ).square().sum(dim=-1)
        logits = logits + (-squared_distance / self.spatial_temperature).unsqueeze(1)
        softmax_weights = torch.softmax(logits, dim=-1)
        weights = (1.0 - self.uniform_floor) * softmax_weights + self.uniform_floor / token_count
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).contiguous().view(
            batch, self.latent_count, self.attention_dim
        )
        if not torch.isfinite(weights).all() or not torch.isfinite(context).all():
            raise RuntimeError("Block-cross-residual attention produced NaN or infinity")
        return context, weights

    def attention_weights(
        self,
        base_scene_tokens: torch.Tensor,
        block_tokens: torch.Tensor,
        block_positions_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Return auditable FP32 weights shaped ``[B,4,256,T]``."""

        return self._attention_context(
            base_scene_tokens, block_tokens, block_positions_normalized
        )[1]

    def residual_delta(
        self,
        base_scene_tokens: torch.Tensor,
        block_tokens: torch.Tensor,
        block_positions_normalized: torch.Tensor,
    ) -> torch.Tensor:
        context, _ = self._attention_context(
            base_scene_tokens, block_tokens, block_positions_normalized
        )
        delta = self.residual_scale * torch.tanh(torch.matmul(context, self.w_o))
        if delta.shape != base_scene_tokens.shape or not torch.isfinite(delta).all():
            raise RuntimeError("Block-cross-residual produced an invalid residual")
        return delta

    def forward_with_audit(
        self,
        base_scene_tokens: torch.Tensor,
        block_tokens: torch.Tensor,
        block_positions_normalized: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context, weights = self._attention_context(
            base_scene_tokens, block_tokens, block_positions_normalized
        )
        delta = self.residual_scale * torch.tanh(torch.matmul(context, self.w_o))
        output = base_scene_tokens + delta.to(base_scene_tokens.dtype)
        if output.shape != base_scene_tokens.shape or not torch.isfinite(output).all():
            raise RuntimeError("Block-cross-residual produced NaN, infinity, or invalid shape")
        row_error = (weights.sum(dim=-1) - 1.0).abs().max()
        block_contribution = weights.sum(dim=(0, 1, 2))
        token_count = weights.shape[-1]
        audit = {
            "block_cross_residual_processed_block_tokens": torch.tensor(
                token_count, device=output.device, dtype=torch.long
            ),
            "block_cross_residual_attention_min_weight": weights.detach().min(),
            "block_cross_residual_uniform_floor_per_block": (
                self.uniform_floor.detach() / token_count
            ),
            "block_cross_residual_attention_row_sum_max_error": row_error.detach(),
            "block_cross_residual_min_block_contribution": block_contribution.detach().min(),
            "block_cross_residual_delta_rms": delta.detach().square().mean().sqrt(),
            "block_cross_residual_output_rms": output.detach().float().square().mean().sqrt(),
        }
        if row_error.item() > 2e-6:
            raise RuntimeError("Block-cross-residual attention rows do not sum to one")
        floor = float(self.uniform_floor.detach().item()) / token_count
        if float(weights.detach().min().item()) + 1e-9 < floor:
            raise RuntimeError("Block-cross-residual attention violated its all-block floor")
        if not torch.all(block_contribution > 0):
            raise RuntimeError("Block-cross-residual omitted an occupied block token")
        return output, audit

    def forward(
        self,
        base_scene_tokens: torch.Tensor,
        block_tokens: torch.Tensor,
        block_positions_normalized: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_audit(
            base_scene_tokens, block_tokens, block_positions_normalized
        )[0]


__all__ = [
    "BLOCK_CROSS_RESIDUAL_ARCHITECTURE_MARKER",
    "BLOCK_CROSS_RESIDUAL_ARCHITECTURE_VERSION",
    "BLOCK_CROSS_RESIDUAL_PARAMETER_COUNT",
    "BlockCrossResidual",
    "BlockCrossResidualSettings",
    "apply_block_cross_residual",
    "block_cross_residual_settings",
    "construct_block_cross_residual",
    "validate_block_cross_residual_state",
]
