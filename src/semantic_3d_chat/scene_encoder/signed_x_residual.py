from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
)
from .perceiver import spatial_anchors

SIGNED_X_MOMENT_V1 = "signed_x_moment_v1"


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _signed_x_anchors(latent_count: int) -> torch.Tensor:
    """Return deterministic centered, unit-RMS signed-X slot weights."""

    latent_count = _validate_positive_int(latent_count, "latent_count")
    if latent_count < 2:
        raise ValueError("latent_count must be at least two for a signed spatial moment")
    signed = spatial_anchors(latent_count)[:, 0].float()
    signed = signed - signed.mean()
    rms = signed.square().mean().sqrt()
    if not torch.isfinite(rms) or float(rms) <= 0.0:
        raise ValueError("Signed-X anchors must have positive finite RMS")
    signed = signed / rms
    # Repeat in FP32 to minimize residual rounding in the persistent contract.
    signed = signed - signed.mean()
    signed = signed / signed.square().mean().sqrt()
    if torch.any(signed == 0):
        raise ValueError(
            "Signed-X anchors contain a zero weight; every slot must contribute to the moment"
        )
    return signed.contiguous()


@dataclass(frozen=True)
class SignedXSceneResidualSettings:
    enabled: bool = False
    architecture_version: str = SIGNED_X_MOMENT_V1
    expected_initial_state_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("signed_x_scene_residual.enabled must be a boolean")
        if self.architecture_version != SIGNED_X_MOMENT_V1:
            raise ValueError(
                "Unsupported signed_x_scene_residual.architecture_version: "
                f"{self.architecture_version!r}"
            )
        expected = self.expected_initial_state_sha256
        if expected is not None and re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(
                "signed_x_scene_residual.expected_initial_state_sha256 must be lowercase SHA-256"
            )
        if self.enabled and expected is None:
            raise ValueError(
                "Enabled signed_x_scene_residual requires expected_initial_state_sha256"
            )

    def contract(self) -> dict[str, Any]:
        if not self.enabled:
            return {"schema_version": 1, "enabled": False}
        return {
            "schema_version": 1,
            "enabled": True,
            "architecture_version": self.architecture_version,
            "expected_initial_state_sha256": self.expected_initial_state_sha256,
            "spatial_statistic": "centered_unit_rms_signed_x_moment",
            "spatial_centering": "all_slots_fp32",
            "trainable_surface": "bias_free_output_projection_only",
        }


def signed_x_scene_residual_settings(
    config: Mapping[str, Any],
) -> SignedXSceneResidualSettings:
    scene_encoder = config.get("scene_encoder")
    if not isinstance(scene_encoder, Mapping):
        raise TypeError("scene_encoder config must be a mapping")
    raw = scene_encoder.get("signed_x_scene_residual", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("scene_encoder.signed_x_scene_residual must be a mapping")
    allowed = {
        "enabled",
        "architecture_version",
        "expected_initial_state_sha256",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown signed_x_scene_residual settings: {unknown}")
    return SignedXSceneResidualSettings(
        enabled=raw.get("enabled", False),
        architecture_version=raw.get("architecture_version", SIGNED_X_MOMENT_V1),
        expected_initial_state_sha256=raw.get("expected_initial_state_sha256"),
    )


def construct_signed_x_scene_residual(
    config: Mapping[str, Any],
    *,
    scene_dim: int,
    latent_count: int,
    content_dim: int,
) -> SignedXSceneResidual | None:
    settings = signed_x_scene_residual_settings(config)
    if not settings.enabled:
        return None
    return SignedXSceneResidual(
        scene_dim=scene_dim,
        latent_count=latent_count,
        content_dim=content_dim,
    )


def apply_signed_x_scene_residual(
    output: Any,
    module: SignedXSceneResidual | None,
    centered_content: torch.Tensor,
) -> Any:
    """Apply the static signed-X branch to an existing scene-token output."""

    if module is None:
        return output
    base_tokens = output.scene_tokens
    adapted = module(base_tokens, centered_content)
    audit = dict(output.audit)
    audit["signed_x_scene_residual_input_rms"] = base_tokens.detach().float().square().mean().sqrt()
    audit["signed_x_scene_residual_delta_rms"] = (
        (adapted.detach().float() - base_tokens.detach().float()).square().mean().sqrt()
    )
    audit["signed_x_scene_residual_moment_rms"] = (
        module.moment_values(centered_content).detach().square().mean().sqrt()
    )
    audit["signed_x_scene_residual_accounted_slots"] = torch.tensor(
        module.accounted_slot_count,
        device=base_tokens.device,
        dtype=torch.long,
    )
    return replace(output, scene_tokens=adapted, audit=audit)


def frozen_v18_centered_content_values(
    source: GlobalSceneResidual,
    scene_tokens: torch.Tensor,
) -> torch.Tensor:
    """Reproduce V18's centered content without modifying its pinned source.

    V18's implementation file is content-hashed research evidence.  Keeping
    this extraction alongside the new V19 branch preserves that historical
    artifact byte-for-byte while reusing the exact trained normalization and
    projection parameters from the frozen source checkpoint.
    """

    if source.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
        raise ValueError("Signed-X content requires the centered-content V18 source")
    if scene_tokens.ndim != 3 or tuple(scene_tokens.shape[1:]) != (
        source.latent_count,
        source.scene_dim,
    ):
        raise ValueError("scene_tokens shape does not match the frozen V18 residual source")
    if not torch.isfinite(scene_tokens).all():
        raise ValueError("scene_tokens must contain only finite values")
    source.validate_structural_state()
    normalized = source.scene_norm(scene_tokens)
    local_content = source.scene_projection(normalized)
    spatial_mean = local_content.float().mean(dim=1, keepdim=True).to(local_content.dtype)
    centered_content = local_content - spatial_mean
    if not torch.isfinite(centered_content).all():
        raise RuntimeError("Frozen V18 centered content produced NaN or infinity")
    return centered_content


class SignedXSceneResidual(nn.Module):
    """A question-independent mirror-odd residual over every scene slot.

    The module consumes centered content produced by a frozen V18 bridge. It
    deliberately does not register or retain that source bridge. The only
    trainable state is a bias-free projection from the fixed 128D signed-X
    moment field into the language-model hidden dimension.
    """

    def __init__(self, *, scene_dim: int, latent_count: int, content_dim: int) -> None:
        super().__init__()
        self.scene_dim = _validate_positive_int(scene_dim, "scene_dim")
        self.latent_count = _validate_positive_int(latent_count, "latent_count")
        self.content_dim = _validate_positive_int(content_dim, "content_dim")
        self.architecture_version = SIGNED_X_MOMENT_V1

        signed = _signed_x_anchors(self.latent_count)
        self.register_buffer("signed_x_anchors", signed, persistent=True)
        self.output_projection = nn.Linear(
            self.content_dim,
            self.scene_dim,
            bias=False,
            dtype=torch.float32,
        )
        nn.init.zeros_(self.output_projection.weight)

    def _apply(
        self,
        fn: Any,
        recurse: bool = True,
    ) -> SignedXSceneResidual:
        """Follow device moves while retaining audited state in FP32."""

        anchors_before = self.signed_x_anchors.detach().clone()
        weight_before = self.output_projection.weight.detach().clone()
        gradient_before = (
            None
            if self.output_projection.weight.grad is None
            else self.output_projection.weight.grad.detach().clone()
        )
        super()._apply(fn, recurse=recurse)
        self.signed_x_anchors.data = anchors_before.to(
            device=self.signed_x_anchors.device,
            dtype=torch.float32,
        )
        weight = self.output_projection.weight
        weight.data = weight_before.to(device=weight.device, dtype=torch.float32)
        if gradient_before is not None:
            weight.grad = gradient_before.to(device=weight.device, dtype=torch.float32)
        return self

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def accounted_slot_count(self) -> int:
        return int(torch.count_nonzero(self.signed_x_anchors).item())

    def validate_structural_state(self) -> dict[str, Any]:
        nonfinite = sorted(
            name for name, value in self.state_dict().items() if not torch.isfinite(value).all()
        )
        if nonfinite:
            raise ValueError(f"Signed-X scene residual contains nonfinite state: {nonfinite}")
        expected = _signed_x_anchors(self.latent_count).to(
            device=self.signed_x_anchors.device,
            dtype=self.signed_x_anchors.dtype,
        )
        if not torch.equal(self.signed_x_anchors, expected):
            raise ValueError("Persistent signed-X anchors do not match deterministic anchors")
        if self.output_projection.bias is not None:
            raise ValueError("Signed-X output projection must remain bias-free")
        parameter_names = [name for name, _ in self.named_parameters()]
        if parameter_names != ["output_projection.weight"]:
            raise ValueError(
                "Signed-X branch must own only output_projection.weight; "
                f"observed={parameter_names}"
            )
        anchor_mean = float(self.signed_x_anchors.mean().cpu())
        anchor_rms = float(self.signed_x_anchors.square().mean().sqrt().cpu())
        if abs(anchor_mean) > 1e-6 or not math.isclose(anchor_rms, 1.0, abs_tol=1e-6):
            raise ValueError("Signed-X anchors must be centered with unit RMS")
        if self.accounted_slot_count != self.latent_count:
            raise ValueError("Signed-X moment does not account for every scene slot")
        return {
            "architecture_version": self.architecture_version,
            "scene_dim": self.scene_dim,
            "latent_count": self.latent_count,
            "content_dim": self.content_dim,
            "parameter_count": self.parameter_count,
            "accounted_slot_count": self.accounted_slot_count,
            "all_slots_accounted": True,
            "signed_x_anchor_mean": anchor_mean,
            "signed_x_anchor_rms": anchor_rms,
            "spatial_centering": "all_slots_fp32",
            "trainable_surface": "bias_free_output_projection_only",
        }

    def _validate_base_tokens(self, base_tokens: torch.Tensor) -> None:
        if base_tokens.ndim != 3 or tuple(base_tokens.shape[1:]) != (
            self.latent_count,
            self.scene_dim,
        ):
            raise ValueError(
                f"base_tokens must have shape [B,{self.latent_count},{self.scene_dim}]"
            )
        if not torch.isfinite(base_tokens).all():
            raise ValueError("base_tokens must contain only finite values")

    def _validated_centered_content(self, centered_content: torch.Tensor) -> torch.Tensor:
        if centered_content.ndim != 3 or tuple(centered_content.shape[1:]) != (
            self.latent_count,
            self.content_dim,
        ):
            raise ValueError(
                f"centered_content must have shape [B,{self.latent_count},{self.content_dim}]"
            )
        if not torch.isfinite(centered_content).all():
            raise ValueError("centered_content must contain only finite values")
        values = centered_content.float()
        spatial_mean = values.mean(dim=1)
        scale = values.detach().square().mean().sqrt().clamp_min(1.0)
        tolerance = 1e-6
        if centered_content.dtype in (torch.float16, torch.bfloat16):
            tolerance = 8.0 * torch.finfo(centered_content.dtype).eps * float(scale.cpu())
        if float(spatial_mean.detach().abs().max().cpu()) > tolerance:
            raise ValueError("centered_content must be centered across all scene slots")
        return values

    def moment_values(self, centered_content: torch.Tensor) -> torch.Tensor:
        """Return the FP32 all-slot signed-X moment with shape ``[B,1,C]``."""

        values = self._validated_centered_content(centered_content)
        signed = self.signed_x_anchors.to(device=values.device).view(1, -1, 1)
        moment = (signed * values).mean(dim=1, keepdim=True)
        if not torch.isfinite(moment).all():
            raise RuntimeError("Signed-X moment produced NaN or infinity")
        return moment

    def hidden_values(self, centered_content: torch.Tensor) -> torch.Tensor:
        """Broadcast the mirror-odd moment back to every signed spatial slot."""

        moment = self.moment_values(centered_content)
        signed = self.signed_x_anchors.to(device=moment.device).view(1, -1, 1)
        hidden = signed * torch.tanh(moment)
        if not torch.isfinite(hidden).all():
            raise RuntimeError("Signed-X hidden field produced NaN or infinity")
        return hidden

    def centered_delta_values(self, centered_content: torch.Tensor) -> torch.Tensor:
        """Return the exact pre-cast FP32 delta centered across all slots."""

        hidden = self.hidden_values(centered_content)
        raw_delta = F.linear(hidden, self.output_projection.weight.float())
        centered_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
        if not torch.isfinite(centered_delta).all():
            raise RuntimeError("Signed-X scene residual produced NaN or infinity")
        return centered_delta

    def forward(
        self,
        base_tokens: torch.Tensor,
        centered_content: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_base_tokens(base_tokens)
        if centered_content.shape[0] != base_tokens.shape[0]:
            raise ValueError("base_tokens and centered_content batch sizes must match")
        centered_delta = self.centered_delta_values(centered_content)
        output = base_tokens + centered_delta.to(dtype=base_tokens.dtype)
        if not torch.isfinite(output).all():
            raise RuntimeError("Signed-X scene residual produced NaN or infinity")
        return output


__all__ = [
    "SIGNED_X_MOMENT_V1",
    "SignedXSceneResidual",
    "SignedXSceneResidualSettings",
    "apply_signed_x_scene_residual",
    "construct_signed_x_scene_residual",
    "frozen_v18_centered_content_values",
    "signed_x_scene_residual_settings",
]
