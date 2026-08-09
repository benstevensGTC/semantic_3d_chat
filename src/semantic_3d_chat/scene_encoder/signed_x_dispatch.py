from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from .signed_x_local_field import (
    SIGNED_X_LOCAL_FIELD_V2,
    SignedXLocalFieldSceneResidual,
)
from .signed_x_residual import (
    SIGNED_X_MOMENT_V1,
    SignedXSceneResidual,
    frozen_v18_centered_content_values,
)
from .signed_x_residual import (
    apply_signed_x_scene_residual as apply_signed_x_moment_residual,
)

SUPPORTED_SIGNED_X_ARCHITECTURES = frozenset({SIGNED_X_MOMENT_V1, SIGNED_X_LOCAL_FIELD_V2})
SignedXResidualModule = SignedXSceneResidual | SignedXLocalFieldSceneResidual


@dataclass(frozen=True)
class SignedXSceneResidualSettings:
    """Versioned settings shared by the historical V19 and local V20 branches."""

    enabled: bool = False
    architecture_version: str = SIGNED_X_MOMENT_V1
    expected_initial_state_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("signed_x_scene_residual.enabled must be a boolean")
        if self.architecture_version not in SUPPORTED_SIGNED_X_ARCHITECTURES:
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
        common = {
            "schema_version": 1,
            "enabled": True,
            "architecture_version": self.architecture_version,
            "expected_initial_state_sha256": self.expected_initial_state_sha256,
        }
        if self.architecture_version == SIGNED_X_MOMENT_V1:
            return {
                **common,
                "spatial_statistic": "centered_unit_rms_signed_x_moment",
                "spatial_centering": "all_slots_fp32",
                "trainable_surface": "bias_free_output_projection_only",
            }
        return {
            **common,
            "spatial_statistic": "centered_local_content_times_unit_rms_signed_x",
            "spatial_reduction": "none",
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
) -> SignedXResidualModule | None:
    settings = signed_x_scene_residual_settings(config)
    if not settings.enabled:
        return None
    module_type = (
        SignedXSceneResidual
        if settings.architecture_version == SIGNED_X_MOMENT_V1
        else SignedXLocalFieldSceneResidual
    )
    return module_type(
        scene_dim=scene_dim,
        latent_count=latent_count,
        content_dim=content_dim,
    )


def apply_signed_x_scene_residual(
    output: Any,
    module: SignedXResidualModule | None,
    centered_content: torch.Tensor,
) -> Any:
    """Apply either version without changing the historical V19 implementation."""

    if module is None:
        return output
    if type(module) is SignedXSceneResidual:
        if module.architecture_version != SIGNED_X_MOMENT_V1:
            raise ValueError(
                "SignedXSceneResidual architecture version does not match its module type"
            )
        return apply_signed_x_moment_residual(output, module, centered_content)
    if type(module) is not SignedXLocalFieldSceneResidual:
        raise TypeError(f"Unsupported signed-X residual module type: {type(module).__name__}")
    if module.architecture_version != SIGNED_X_LOCAL_FIELD_V2:
        raise ValueError(
            "SignedXLocalFieldSceneResidual architecture version does not match its module type"
        )
    base_tokens = output.scene_tokens
    adapted = module(base_tokens, centered_content)
    hidden = module.hidden_values(centered_content)
    audit = dict(output.audit)
    audit["signed_x_scene_residual_input_rms"] = base_tokens.detach().float().square().mean().sqrt()
    audit["signed_x_scene_residual_delta_rms"] = (
        (adapted.detach().float() - base_tokens.detach().float()).square().mean().sqrt()
    )
    audit["signed_x_scene_residual_local_field_rms"] = hidden.detach().square().mean().sqrt()
    audit["signed_x_scene_residual_accounted_slots"] = torch.tensor(
        module.accounted_slot_count,
        device=base_tokens.device,
        dtype=torch.long,
    )
    audit["signed_x_scene_residual_architecture_marker"] = (
        module.architecture_marker.detach().clone().to(device=base_tokens.device)
    )
    return replace(output, scene_tokens=adapted, audit=audit)


__all__ = [
    "SUPPORTED_SIGNED_X_ARCHITECTURES",
    "SignedXResidualModule",
    "SignedXSceneResidual",
    "SignedXSceneResidualSettings",
    "apply_signed_x_scene_residual",
    "construct_signed_x_scene_residual",
    "frozen_v18_centered_content_values",
    "signed_x_scene_residual_settings",
]
