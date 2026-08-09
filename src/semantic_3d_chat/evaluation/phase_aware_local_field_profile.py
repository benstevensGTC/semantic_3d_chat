"""Immutable pins for the V21-family phase-aware local-field controllers.

The controllers are deliberately parameterized only over experiment identity
and the two target margins.  Architecture, data ordering, source checkpoint,
optimizer, precision, and all authorization gates remain fixed in the shared
V21 implementation.  A new experiment therefore cannot silently broaden the
trusted surface by passing arbitrary controller options at the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PhaseAwareLocalFieldProfile:
    """Fail-closed identity pins for one local-field experiment revision."""

    version: str
    config_path: Path
    resolved_config_hash: str
    normalized_contract_sha256: str
    output_namespace: str
    extension_namespace: str
    experiment_role: str
    screen_key: str
    screen_role: str
    preflight_role: str
    update1_verifier_type: str
    selector_type: str
    extension_controller_type: str
    extension_final_selector_type: str
    mirror_candidate_margin: float
    mirror_full_vocab_margin: float
    bind_namespaces_in_screen: bool = False

    def __post_init__(self) -> None:
        if self.version not in {"V21", "V22"}:
            raise ValueError(f"unsupported local-field controller version: {self.version}")
        if len(self.resolved_config_hash) != 12 or any(
            character not in "0123456789abcdef" for character in self.resolved_config_hash
        ):
            raise ValueError("resolved_config_hash must be a lowercase 12-character SHA prefix")
        if len(self.normalized_contract_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.normalized_contract_sha256
        ):
            raise ValueError("normalized_contract_sha256 must be a lowercase SHA-256")
        if not self.screen_key.startswith(self.version.lower()):
            raise ValueError("screen_key must be version-scoped")
        if self.output_namespace == self.extension_namespace:
            raise ValueError("extension namespace must be isolated from the primary namespace")
        if len(self.extension_namespace) > 64:
            raise ValueError("extension namespace exceeds the trainer's 64-character limit")
        for field, value in (
            ("mirror_candidate_margin", self.mirror_candidate_margin),
            ("mirror_full_vocab_margin", self.mirror_full_vocab_margin),
        ):
            if value <= 0.0:
                raise ValueError(f"{field} must be strictly positive")


V21_LOCAL_FIELD_PROFILE = PhaseAwareLocalFieldProfile(
    version="V21",
    config_path=Path(
        "configs/experiments/gemma4_color_mirror_signed_x_local_field_phase_aware_v21.yaml"
    ),
    resolved_config_hash="ae17da8b9a71",
    normalized_contract_sha256=(
        "50e5522a19d4f6a3eb88884cdccfa71ab1301ebe94bf1a512d42505322799b2c"
    ),
    output_namespace="gemma4_color_mirror_signed_x_local_field_phase_aware_v21",
    extension_namespace="gemma4_v21_phase_aware_local_field_extension_u8",
    experiment_role="exploratory_phase_aware_reflection_odd_local_field_screen_v21",
    screen_key="v21_screen",
    screen_role="signed_x_local_field_phase_aware_screen",
    preflight_role="v21_exact_ordered_signed_x_local_field_phase_aware_structural_preflight",
    update1_verifier_type="v21_exact_update1_match_verifier",
    selector_type="strict_v21_signed_x_local_field_phase_aware_epoch_selector",
    extension_controller_type="strict_v21_conditional_extension_controller",
    extension_final_selector_type="strict_v21_conditional_extension_final_selector",
    mirror_candidate_margin=1.0,
    mirror_full_vocab_margin=1.0,
    bind_namespaces_in_screen=False,
)


# The normalized contract digest is independently recomputed by the shared
# validator and pinned here.  Any inherited-config, role, margin, namespace,
# optimizer, data-order, architecture, or gate mutation therefore fails before
# a model or optimizer can be loaded.
V22_LOCAL_FIELD_PROFILE = PhaseAwareLocalFieldProfile(
    version="V22",
    config_path=Path(
        "configs/experiments/gemma4_color_mirror_signed_x_local_field_margin_rebalanced_v22.yaml"
    ),
    resolved_config_hash="b336be25fd68",
    normalized_contract_sha256="a8994abafc02720a96f47fbdab222f487e2ea6310c690dedd4c8b2f5232c3c4b",
    output_namespace="gemma4_v22_margin_rebalanced_local_field",
    extension_namespace="gemma4_v22_margin_rebalanced_local_field_extension_u8",
    experiment_role=(
        "exploratory_phase_aware_margin_rebalanced_reflection_odd_local_field_screen_v22"
    ),
    screen_key="v22_screen",
    screen_role="signed_x_local_field_phase_aware_margin_rebalanced_screen",
    preflight_role=(
        "v22_exact_ordered_signed_x_local_field_phase_aware_margin_rebalanced_"
        "structural_preflight"
    ),
    update1_verifier_type="v22_exact_update1_match_verifier",
    selector_type="strict_v22_margin_rebalanced_local_field_epoch_selector",
    extension_controller_type="strict_v22_conditional_extension_controller",
    extension_final_selector_type="strict_v22_conditional_extension_final_selector",
    mirror_candidate_margin=0.25,
    mirror_full_vocab_margin=0.25,
    bind_namespaces_in_screen=True,
)


__all__ = [
    "V21_LOCAL_FIELD_PROFILE",
    "V22_LOCAL_FIELD_PROFILE",
    "PhaseAwareLocalFieldProfile",
]
