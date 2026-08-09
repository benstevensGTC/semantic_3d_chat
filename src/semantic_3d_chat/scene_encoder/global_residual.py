from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from .perceiver import spatial_anchors

GLOBAL_MEAN_V1 = "global_mean_v1"
ZERO_SPATIAL_MEAN_CONTENT_GATE_V1 = "zero_spatial_mean_content_gate_v1"
_SUPPORTED_ARCHITECTURES = frozenset({GLOBAL_MEAN_V1, ZERO_SPATIAL_MEAN_CONTENT_GATE_V1})


def _validate_architecture_version(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("global_scene_residual.architecture_version must be a string")
    if value not in _SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "Unsupported global_scene_residual.architecture_version: "
            f"{value!r}; supported={sorted(_SUPPORTED_ARCHITECTURES)}"
        )
    return value


def _validate_gate_temperature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("global_scene_residual.gate_temperature must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(
            "global_scene_residual.gate_temperature must be finite and strictly positive"
        )
    return parsed


def _spatial_position_features(latent_count: int, fourier_bands: int) -> torch.Tensor:
    anchors = spatial_anchors(latent_count)
    frequencies = (2.0 ** torch.arange(fourier_bands, dtype=torch.float32)) * math.pi
    angles = anchors.unsqueeze(-1) * frequencies
    fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)
    return torch.cat((anchors, fourier), dim=-1).contiguous()


@dataclass(frozen=True)
class GlobalSceneResidualSettings:
    enabled: bool = False
    width: int = 128
    fourier_bands: int = 8
    initialization_seed: int = 16015
    expected_initial_state_sha256: str | None = None
    architecture_version: str = GLOBAL_MEAN_V1
    gate_temperature: float = 1.0

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
        _validate_architecture_version(self.architecture_version)
        _validate_gate_temperature(self.gate_temperature)
        expected = self.expected_initial_state_sha256
        if expected is not None and re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(
                "global_scene_residual.expected_initial_state_sha256 must be lowercase SHA-256"
            )
        if self.enabled and expected is None:
            raise ValueError("Enabled global_scene_residual requires expected_initial_state_sha256")

    def contract(self) -> dict[str, Any]:
        if not self.enabled:
            return {"schema_version": 1, "enabled": False}
        legacy_contract = {
            "schema_version": 1,
            "enabled": True,
            "width": self.width,
            "fourier_bands": self.fourier_bands,
            "initialization_seed": self.initialization_seed,
            "expected_initial_state_sha256": self.expected_initial_state_sha256,
        }
        if self.architecture_version == GLOBAL_MEAN_V1:
            return legacy_contract
        return {
            **legacy_contract,
            "schema_version": 2,
            "architecture_version": self.architecture_version,
            "gate_temperature": float(self.gate_temperature),
            "spatial_centering": "all_slots_fp32",
            "content_gate": "bias_free_scalar_sigmoid_centered_content",
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
        "architecture_version",
        "gate_temperature",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown global_scene_residual settings: {unknown}")
    architecture_version = raw.get("architecture_version", GLOBAL_MEAN_V1)
    if architecture_version == GLOBAL_MEAN_V1 and "gate_temperature" in raw:
        raise ValueError(
            "global_scene_residual.gate_temperature is only valid for "
            f"architecture_version={ZERO_SPATIAL_MEAN_CONTENT_GATE_V1!r}"
        )
    return GlobalSceneResidualSettings(
        enabled=raw.get("enabled", False),
        width=raw.get("width", 128),
        fourier_bands=raw.get("fourier_bands", 8),
        initialization_seed=raw.get("initialization_seed", 16015),
        expected_initial_state_sha256=raw.get("expected_initial_state_sha256"),
        architecture_version=architecture_version,
        gate_temperature=raw.get("gate_temperature", 1.0),
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
        architecture_version=settings.architecture_version,
        gate_temperature=settings.gate_temperature,
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
    audit["global_scene_residual_input_rms"] = scene_tokens.detach().float().square().mean().sqrt()
    audit["global_scene_residual_delta_rms"] = (
        (adapted.detach().float() - scene_tokens.detach().float()).square().mean().sqrt()
    )
    return replace(output, scene_tokens=adapted, audit=audit)


class GlobalSceneResidual(nn.Module):
    """Question-independent, position-aware residual over every scene slot.

    The legacy architecture combines each slot with a global projected mean.
    The explicit content-gated architecture instead centers projected content,
    applies a learned scalar gate, and removes the FP32 spatial mean of its
    output.  Both retain an identity path and an exact-zero final projection,
    so adding either module does not alter a source prefix before optimization.

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
        architecture_version: str = GLOBAL_MEAN_V1,
        gate_temperature: float = 1.0,
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
        architecture_version = _validate_architecture_version(architecture_version)
        gate_temperature = _validate_gate_temperature(gate_temperature)

        self.scene_dim = int(scene_dim)
        self.latent_count = int(latent_count)
        self.width = int(width)
        self.fourier_bands = int(fourier_bands)
        self.initialization_seed = int(initialization_seed)
        self.architecture_version = architecture_version
        self.configured_gate_temperature = gate_temperature

        position_features = _spatial_position_features(self.latent_count, self.fourier_bands)
        # These values define slot identity and therefore belong in checkpoint
        # state and state hashes.  They are intentionally persistent.
        self.register_buffer("position_features", position_features, persistent=True)
        if self.architecture_version == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
            self.register_buffer(
                "gate_temperature",
                torch.tensor(gate_temperature, dtype=torch.float32),
                persistent=True,
            )

        # nn.Linear constructors initialize parameters.  fork_rng makes the
        # initialization deterministic without perturbing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.scene_norm = nn.LayerNorm(self.scene_dim)
            self.scene_projection = nn.Linear(self.scene_dim, self.width)
            self.position_projection = nn.Linear(position_features.shape[-1], self.width)
            if self.architecture_version == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
                self.content_gate_projection = nn.Linear(self.width, 1, bias=False)
                nn.init.normal_(
                    self.content_gate_projection.weight,
                    mean=0.0,
                    std=1.0 / math.sqrt(self.width),
                )
            self.output_projection = nn.Linear(self.width, self.scene_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    @property
    def parameter_count(self) -> int:
        """Return the exact parameter surface for provenance checks."""

        return sum(parameter.numel() for parameter in self.parameters())

    def validate_structural_state(self) -> dict[str, Any]:
        """Fail closed on nonfinite or contract-inconsistent persistent state.

        Checkpoint loaders should call this after ``load_state_dict``.  Keeping
        the configured temperature outside tensor state lets this method detect
        a structurally valid checkpoint that overwrote the persistent scalar
        with a value inconsistent with the active configuration.
        """

        nonfinite = sorted(
            name for name, value in self.state_dict().items() if not torch.isfinite(value).all()
        )
        if nonfinite:
            raise ValueError(f"Global scene residual contains nonfinite state: {nonfinite}")
        expected_positions = _spatial_position_features(self.latent_count, self.fourier_bands).to(
            device=self.position_features.device, dtype=self.position_features.dtype
        )
        if not torch.equal(self.position_features, expected_positions):
            raise ValueError(
                "Persistent position features do not match deterministic spatial anchors"
            )
        audit: dict[str, Any] = {
            "architecture_version": self.architecture_version,
            "parameter_count": self.parameter_count,
            "latent_count": self.latent_count,
            "scene_dim": self.scene_dim,
        }
        if self.architecture_version == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
            expected_temperature = torch.tensor(
                self.configured_gate_temperature,
                dtype=self.gate_temperature.dtype,
                device=self.gate_temperature.device,
            )
            if not torch.equal(self.gate_temperature, expected_temperature):
                raise ValueError(
                    "Persistent gate temperature does not match active configuration: "
                    f"checkpoint={self.gate_temperature.detach().float().cpu().item()} "
                    f"config={self.configured_gate_temperature}"
                )
            audit.update(
                {
                    "gate_temperature": self.configured_gate_temperature,
                    "spatial_centering": "all_slots_fp32",
                    "content_gate": "bias_free_scalar_sigmoid_centered_content",
                }
            )
        return audit

    def _validate_scene_tokens(self, scene_tokens: torch.Tensor) -> None:
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

    def _positions(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        return self.position_projection(
            self.position_features.to(device=scene_tokens.device, dtype=scene_tokens.dtype)
        ).unsqueeze(0)

    def _content_gate_hidden(self, scene_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.scene_norm(scene_tokens)
        local_content = self.scene_projection(normalized)
        global_content = local_content.float().mean(dim=1, keepdim=True).to(local_content.dtype)
        centered_content = local_content - global_content
        temperature = self.gate_temperature.float()
        gate_logits = self.content_gate_projection(centered_content).float() / temperature
        gate = (2.0 * torch.sigmoid(gate_logits)).to(local_content.dtype)
        hidden = gate * torch.tanh(centered_content + self._positions(scene_tokens))
        return hidden, gate

    def content_gate_values(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        """Return auditable scalar gates for the explicit content-gated variant."""

        self._validate_scene_tokens(scene_tokens)
        if self.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
            raise RuntimeError("Content gates are unavailable for the legacy architecture")
        self.validate_structural_state()
        return self._content_gate_hidden(scene_tokens)[1]

    def _centered_delta(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        hidden, _gate = self._content_gate_hidden(scene_tokens)
        raw_delta_fp32 = self.output_projection(hidden).float()
        return raw_delta_fp32 - raw_delta_fp32.mean(dim=1, keepdim=True)

    def centered_delta_values(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        """Return the pre-cast FP32 delta used by the centered architecture.

        This audit surface distinguishes the architectural zero-mean guarantee
        from small quantization effects that can appear after casting and
        adding the delta to BF16 scene tokens.
        """

        self._validate_scene_tokens(scene_tokens)
        if self.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
            raise RuntimeError("Centered deltas are unavailable for the legacy architecture")
        self.validate_structural_state()
        return self._centered_delta(scene_tokens)

    def forward(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        self._validate_scene_tokens(scene_tokens)

        if self.architecture_version == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
            centered_delta = self._centered_delta(scene_tokens)
            output = scene_tokens + centered_delta.to(scene_tokens.dtype)
            if not torch.isfinite(output).all():
                raise RuntimeError("Global scene residual produced NaN or infinity")
            return output

        normalized = self.scene_norm(scene_tokens)
        local_content = self.scene_projection(normalized)
        # Reusing the local projection keeps the bridge compact while making
        # every residual slot differentiably depend on every scene slot.
        global_content = local_content.float().mean(dim=1, keepdim=True).to(local_content.dtype)
        positions = self._positions(scene_tokens)
        hidden = torch.tanh(local_content + global_content + positions)
        output = scene_tokens + self.output_projection(hidden)
        if not torch.isfinite(output).all():
            raise RuntimeError("Global scene residual produced NaN or infinity")
        return output


__all__ = [
    "GLOBAL_MEAN_V1",
    "ZERO_SPATIAL_MEAN_CONTENT_GATE_V1",
    "GlobalSceneResidual",
    "GlobalSceneResidualSettings",
    "apply_global_scene_residual",
    "construct_global_scene_residual",
    "global_scene_residual_settings",
]
