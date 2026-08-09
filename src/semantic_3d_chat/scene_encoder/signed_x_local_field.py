from __future__ import annotations

from typing import Any

import torch

from .signed_x_residual import SignedXSceneResidual

SIGNED_X_LOCAL_FIELD_V2 = "signed_x_local_field_v2"
SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER = 2


class SignedXLocalFieldSceneResidual(SignedXSceneResidual):
    """Coordinate-sensitive residual that preserves every slot's local content.

    V19's signed-X moment intentionally reduced all scene slots to one content
    vector before broadcasting it back over space.  This V20 branch keeps the
    identical anchors, zero-initialized output projection, FP32 centering, and
    parameter count, but applies the signed field directly to each centered
    V18 content slot.  It therefore introduces no question-dependent or
    learned selection step and still accounts for the complete scene.
    """

    def __init__(self, *, scene_dim: int, latent_count: int, content_dim: int) -> None:
        super().__init__(
            scene_dim=scene_dim,
            latent_count=latent_count,
            content_dim=content_dim,
        )
        self.architecture_version = SIGNED_X_LOCAL_FIELD_V2
        self.register_buffer(
            "architecture_marker",
            torch.tensor(SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER, dtype=torch.int64),
            persistent=True,
        )

    def hidden_values(self, centered_content: torch.Tensor) -> torch.Tensor:
        """Return the unreduced FP32 local signed-X field ``[B,L,C]``."""

        values = self._validated_centered_content(centered_content)
        signed = self.signed_x_anchors.to(device=values.device).view(1, -1, 1)
        hidden = signed * torch.tanh(values)
        if not torch.isfinite(hidden).all():
            raise RuntimeError("Signed-X local hidden field produced NaN or infinity")
        return hidden

    def validate_structural_state(self) -> dict[str, Any]:
        if self.architecture_version != SIGNED_X_LOCAL_FIELD_V2:
            raise ValueError(
                "Signed-X local-field architecture version does not match its module type"
            )
        audit = super().validate_structural_state()
        marker = self.architecture_marker
        if (
            marker.ndim != 0
            or marker.dtype != torch.int64
            or int(marker.detach().cpu().item()) != SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER
        ):
            raise ValueError("Signed-X local-field architecture marker is invalid")
        audit.update(
            {
                "architecture_version": self.architecture_version,
                "architecture_marker": SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER,
                "spatial_statistic": "centered_local_content_times_unit_rms_signed_x",
                "spatial_reduction": "none",
            }
        )
        return audit


__all__ = [
    "SIGNED_X_LOCAL_FIELD_ARCHITECTURE_MARKER",
    "SIGNED_X_LOCAL_FIELD_V2",
    "SignedXLocalFieldSceneResidual",
]
