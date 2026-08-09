"""Strict artifact-bound selector for V21's four phase-aware BF16 updates.

The selector binds the exact update-one authorization to all cumulative
checkpoints.  It safely inspects safetensors and one-matrix AdamW state on CPU,
but performs no model inference and reads no QA, map, rendering, runtime, or
oracle artifact.  Its behavioral eligibility, continuation, ranking, and
greedy gates are intentionally identical to V20.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation import v20_epoch_selector as v20_selector
from semantic_3d_chat.evaluation.phase_aware_local_field_profile import (
    V21_LOCAL_FIELD_PROFILE,
    PhaseAwareLocalFieldProfile,
)
from semantic_3d_chat.evaluation.residual_lr_response import EXPECTED_RANKING_FIELDS
from semantic_3d_chat.evaluation.v19_epoch_selector import (
    EXPECTED_FROZEN_BANKS,
    EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
    EXPECTED_GLOBAL_RESIDUAL_PARAMETER_COUNT,
    EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256,
    EXPECTED_PAIR_MEMBERSHIP_SHA256,
    EXPECTED_PAIR_SELECTION_SHA256,
    EXPECTED_SIGNED_X_PARAMETER_COUNT,
    EXPECTED_SOURCE_ADAPTER_SHA256,
    EXPECTED_SOURCE_METADATA_SHA256,
    EXPECTED_TEST_SCENES,
    EXPECTED_TRAIN_SCENES,
    _assert_finite_tree,
    _canonical_sha256,
    _expected_global_residual_contract,
    _extract_epoch_metrics,
    _load_json_strict,
    _mapping,
    _reject_forbidden_input_path,
    _sequence,
    _sha256,
    _validate_lora_contract,
    _validate_source_provenance,
)
from semantic_3d_chat.evaluation.v19_optimizer_state import (
    V19AdamWStateViolation,
    canonical_v19_adamw_state,
    validate_v19_adamw_state_manifest,
)
from semantic_3d_chat.evaluation.v21_structural_preflight import (
    COLOR_PAIR_ID,
    EXPECTED_RESOLVED_CONFIG_HASH,
    EXPECTED_SIGNED_X_INITIAL_STATE_SHA256,
    EXPECTED_SOURCE_SCENE_STATE_SHA256,
    MIRROR_PAIR_ID,
    V21StructuralPreflightViolation,
    validate_v21_config_contract,
)
from semantic_3d_chat.evaluation.v21_update1_verifier import (
    _IMPLEMENTATION_SOURCES,
    MODEL_DTYPE,
    PHASE_ALGORITHM,
    PRECISION_ALGORITHM,
    V21Update1Violation,
    _load_tensor_evidence,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)
from semantic_3d_chat.training.train_adapter import file_sha256

PINNED_CONFIG_PATH = Path(
    "configs/experiments/gemma4_color_mirror_signed_x_local_field_phase_aware_v21.yaml"
)
# Historical public pin retained for audit tests and downstream evidence tools.
EXPECTED_V21_CONTRACT_SHA256 = V21_LOCAL_FIELD_PROFILE.normalized_contract_sha256
PINNED_CONFIG_HASH = EXPECTED_RESOLVED_CONFIG_HASH
OUTPUT_NAMESPACE = "gemma4_color_mirror_signed_x_local_field_phase_aware_v21"
EXPECTED_EPOCHS = (1, 2, 3, 4)
EXPECTED_INITIAL_SIGNED_X_SHA256 = EXPECTED_SIGNED_X_INITIAL_STATE_SHA256
EXPECTED_FROZEN_SCENE_SHA256 = EXPECTED_SOURCE_SCENE_STATE_SHA256

_UPDATE1_REDUCTION_HASH_FIELDS = (
    "implementation_sources_sha256",
    "local_field_structural_state_sha256",
    "local_dependence_sha256",
    "local_hidden_spatial_rank_sha256",
    "centered_content_sha256",
    "raw_fp32_centered_scene_delta_sha256",
    "model_effective_scene_delta_sha256",
    "precision_cast_audit_sha256",
    "raw_fp32_centered_pair_delta_sha256",
    "model_effective_pair_delta_sha256",
    "phase_aware_pair_diagnostics_sha256",
    "predicted_update_functional_audit_sha256",
    "structural_gate_sha256",
)
_UPDATE1_REPORT_KEYS = {
    "schema_version",
    "audit_type",
    "match",
    "stage_2_authorized",
    "report_only",
    "model_loaded",
    "scene_map_loaded",
    "oracle_loaded",
    "model_dtype",
    "optimizer_deserialized",
    "optimizer_deserialization",
    "source_provenance",
    "config_hash",
    "preflight_contract_sha256",
    "preflight_sha256",
    "preflight_implementation_sources",
    "rich_preflight_reduction",
    "checkpoint",
    "checkpoint_artifact_hashes",
    "signed_x_state_sha256",
    "output_projection_sha256",
    "frozen_global_scene_residual_state_sha256",
    "frozen_scene_state_sha256",
    "frozen_lora_bank_state_sha256",
    "optimizer_state_manifest",
    "optimizer_state_sha256",
}


class V21EpochSelectorViolation(ValueError):
    """A fail-closed V21 evidence, provenance, artifact, or policy violation."""


def _fail(message: str) -> None:
    raise V21EpochSelectorViolation(message)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _lexical_absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _safe_checkpoint_file(path: Path, field: str) -> Path:
    try:
        return v20_selector._safe_checkpoint_file(path, field)
    except v20_selector.V20EpochSelectorViolation as error:
        _fail(str(error))


def _file_sha256(path: Path, field: str) -> str:
    return file_sha256(_safe_checkpoint_file(path, field))


def _require_bound_json(
    path: str | Path,
    expected_value: Mapping[str, Any],
    expected_sha256: str,
    field: str,
) -> None:
    safe = _safe_checkpoint_file(_lexical_absolute(path), field)
    try:
        loaded, observed_sha256 = _load_json_strict(safe)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Cannot load {field}: {error}")
    if dict(loaded) != dict(expected_value):
        _fail(f"{field} value differs from the loaded selector input")
    if observed_sha256 != expected_sha256:
        _fail(f"{field} SHA-256 differs from the loaded selector input")


def _validate_implementation_sources(value: Any) -> dict[str, Any]:
    sources = dict(_mapping(value, "update1.preflight_implementation_sources"))
    expected_keys = {
        item for field in _IMPLEMENTATION_SOURCES for item in (field, f"{field}_sha256")
    }
    if set(sources) != expected_keys:
        _fail("update1 preflight implementation-source keys mismatch")
    for field, relative in _IMPLEMENTATION_SOURCES.items():
        if sources.get(field) != relative:
            _fail(f"update1 {field} path mismatch")
        path = _safe_checkpoint_file(PROJECT_ROOT / relative, f"canonical {field}")
        digest = _sha256(sources.get(f"{field}_sha256"), f"update1 {field} hash")
        if file_sha256(path) != digest:
            _fail(f"update1 {field} no longer matches the canonical source bytes")
    return sources


def _validate_rich_preflight_reduction(
    value: Any,
    *,
    implementation_sources: Mapping[str, Any],
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> dict[str, Any]:
    reduction = dict(_mapping(value, "update1.rich_preflight_reduction"))
    expected_keys = {
        "schema_version",
        "verified",
        "model_dtype",
        "precision_algorithm",
        "phase_algorithm_family",
        "phase_algorithm",
        "legacy_effective_total_norm_selectivity_diagnostic_only",
        "preflight_contract_sha256",
        "scene_ids",
        "pair_ids",
        *_UPDATE1_REDUCTION_HASH_FIELDS,
        "canonical_sha256",
    }
    if set(reduction) != expected_keys:
        _fail("update1.rich_preflight_reduction keys mismatch")
    for key, expected in {
        "schema_version": 1,
        "verified": True,
        "model_dtype": MODEL_DTYPE,
        "precision_algorithm": PRECISION_ALGORITHM,
        "phase_algorithm_family": "phase_aware_precision_pair_v1",
        "phase_algorithm": PHASE_ALGORITHM,
        "legacy_effective_total_norm_selectivity_diagnostic_only": True,
        "preflight_contract_sha256": profile.normalized_contract_sha256,
        "scene_ids": list(EXPECTED_TRAIN_SCENES),
        "pair_ids": ["pair_000001", "pair_000003"],
    }.items():
        if reduction.get(key) != expected:
            _fail(f"update1.rich_preflight_reduction.{key} mismatch")
    for key in _UPDATE1_REDUCTION_HASH_FIELDS:
        _sha256(reduction.get(key), f"update1.rich_preflight_reduction.{key}")
    if reduction["implementation_sources_sha256"] != _canonical_sha256(
        dict(implementation_sources)
    ):
        _fail("update1 reduction does not bind its implementation-source evidence")
    expected_digest = _canonical_sha256(
        {key: item for key, item in reduction.items() if key != "canonical_sha256"}
    )
    if _sha256(reduction.get("canonical_sha256"), "update1 reduction canonical hash") != (
        expected_digest
    ):
        _fail("update1 rich preflight reduction canonical hash mismatch")
    return reduction


def _load_update1_authorization(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> dict[str, Any]:
    """Load and validate the exact V21 stage-two authorization report."""

    safe_report = _safe_checkpoint_file(_lexical_absolute(path), "V21 update-one report")
    try:
        raw, report_sha256 = _load_json_strict(safe_report)
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Cannot load required V21 update-one authorization: {error}")
    report = dict(raw)
    _assert_finite_tree(report, "update1")
    if set(report) != _UPDATE1_REPORT_KEYS:
        _fail("update1 report root keys mismatch")
    for key, expected in {
        "schema_version": 1,
        "audit_type": profile.update1_verifier_type,
        "match": True,
        "stage_2_authorized": True,
        "report_only": True,
        "model_loaded": False,
        "scene_map_loaded": False,
        "oracle_loaded": False,
        "model_dtype": MODEL_DTYPE,
        "optimizer_deserialized": True,
    }.items():
        if report.get(key) != expected:
            _fail(f"update1.{key} does not authorize downstream V21 selection")
    if report.get("optimizer_deserialization") != {
        "weights_only": True,
        "map_location": "cpu",
        "canonical_state_validated": True,
    }:
        _fail("update1 optimizer deserialization evidence mismatch")
    full_config_hash = config_hash(dict(config), length=64)
    if report.get("config_hash") != full_config_hash:
        _fail("update1 report config provenance mismatch")
    if report.get("preflight_contract_sha256") != profile.normalized_contract_sha256:
        _fail("update1 normalized preflight-contract hash mismatch")
    source = _validate_source_provenance(report.get("source_provenance"), "update1.source")
    _sha256(report.get("preflight_sha256"), "update1.preflight_sha256")
    implementation_sources = _validate_implementation_sources(
        report.get("preflight_implementation_sources")
    )
    if profile is V21_LOCAL_FIELD_PROFILE:
        reduction = _validate_rich_preflight_reduction(
            report.get("rich_preflight_reduction"),
            implementation_sources=implementation_sources,
        )
    else:
        reduction = _validate_rich_preflight_reduction(
            report.get("rich_preflight_reduction"),
            implementation_sources=implementation_sources,
            profile=profile,
        )
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        _fail("update1.checkpoint must be a nonempty path")
    artifact_hashes = dict(
        _mapping(report.get("checkpoint_artifact_hashes"), "update1 artifact hashes")
    )
    if set(artifact_hashes) != {"adapter_sha256", "metadata_sha256", "optimizer_sha256"}:
        _fail("update1 checkpoint artifact hash keys mismatch")
    for key, digest in artifact_hashes.items():
        _sha256(digest, f"update1 checkpoint artifact {key}")
    signed_state = _sha256(report.get("signed_x_state_sha256"), "update1 signed-X state")
    output_projection = _sha256(report.get("output_projection_sha256"), "update1 output projection")
    frozen_global = _sha256(
        report.get("frozen_global_scene_residual_state_sha256"), "update1 frozen global"
    )
    frozen_scene = _sha256(report.get("frozen_scene_state_sha256"), "update1 frozen scene")
    frozen_lora = dict(_mapping(report.get("frozen_lora_bank_state_sha256"), "update1 LoRA"))
    if frozen_global != EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256:
        _fail("update1 frozen global residual state mismatch")
    if frozen_scene != EXPECTED_FROZEN_SCENE_SHA256:
        _fail("update1 must bind the stable BF16 runtime frozen-scene state")
    if frozen_lora != EXPECTED_FROZEN_BANKS:
        _fail("update1 frozen LoRA state mismatch")
    optimizer_contract = dict(_mapping(config["training"]["optimizer"], "optimizer contract"))
    optimizer_contract["step_index"] = 1
    optimizer_manifest = dict(
        _mapping(report.get("optimizer_state_manifest"), "update1 optimizer manifest")
    )
    try:
        manifest_sha256 = validate_v19_adamw_state_manifest(optimizer_manifest, optimizer_contract)
    except V19AdamWStateViolation as error:
        _fail(f"update1 optimizer manifest violates the one-matrix AdamW contract: {error}")
    if manifest_sha256 != _sha256(
        report.get("optimizer_state_sha256"), "update1 optimizer-state hash"
    ):
        _fail("update1 optimizer manifest hash mismatch")
    return {
        "report_path": _display(safe_report),
        "report_sha256": report_sha256,
        "report_canonical_sha256": _canonical_sha256(report),
        "audit_type": profile.update1_verifier_type,
        "match": True,
        "stage_2_authorized": True,
        "model_dtype": MODEL_DTYPE,
        "source_provenance": source,
        "config_hash": full_config_hash,
        "preflight_contract_sha256": profile.normalized_contract_sha256,
        "preflight_sha256": report["preflight_sha256"],
        "preflight_implementation_sources": implementation_sources,
        "rich_preflight_reduction": reduction,
        "checkpoint": _display(_resolve(checkpoint)),
        "checkpoint_artifact_hashes": artifact_hashes,
        "signed_x_state_sha256": signed_state,
        "output_projection_sha256": output_projection,
        "frozen_global_scene_residual_state_sha256": frozen_global,
        "frozen_scene_state_sha256": frozen_scene,
        "frozen_lora_bank_state_sha256": frozen_lora,
        "optimizer_state_manifest": optimizer_manifest,
        "optimizer_state_sha256": manifest_sha256,
    }


def _inspect_checkpoint_artifacts(
    epoch: int,
    metadata: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    metadata_path: str | Path,
) -> dict[str, Any]:
    raw_metadata_path = Path(metadata_path)
    if not raw_metadata_path.is_absolute():
        raw_metadata_path = PROJECT_ROOT / raw_metadata_path
    if raw_metadata_path.name != "metadata.json":
        _fail(f"epoch_{epoch} metadata path must end in metadata.json")
    _safe_checkpoint_file(raw_metadata_path, f"epoch_{epoch}.metadata")
    checkpoint = raw_metadata_path.parent
    adapter_path = checkpoint / "adapter.safetensors"
    optimizer_path = checkpoint / "optimizer.pt"
    artifact_hashes = {
        "adapter_sha256": _file_sha256(adapter_path, f"epoch_{epoch}.adapter"),
        "metadata_sha256": _file_sha256(raw_metadata_path, f"epoch_{epoch}.metadata"),
        "optimizer_sha256": _file_sha256(optimizer_path, f"epoch_{epoch}.optimizer"),
    }
    try:
        tensors = _load_tensor_evidence(
            adapter_path,
            metadata,
            config=config,
            expected_scene=EXPECTED_FROZEN_SCENE_SHA256,
            expected_global=EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
            expected_lora=EXPECTED_FROZEN_BANKS,
        )
    except V21Update1Violation as error:
        _fail(f"epoch_{epoch} adapter state violates the V21 contract: {error}")
    optimizer_contract = dict(_mapping(config["training"]["optimizer"], "optimizer contract"))
    optimizer_contract["step_index"] = epoch
    try:
        optimizer_state = torch.load(optimizer_path, weights_only=True, map_location="cpu")
        optimizer_manifest, optimizer_sha256 = canonical_v19_adamw_state(
            optimizer_state, optimizer_contract
        )
    except (
        EOFError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
        V19AdamWStateViolation,
    ) as error:
        _fail(f"epoch_{epoch} optimizer violates the one-matrix AdamW contract: {error}")
    return {
        "checkpoint": _display(checkpoint),
        "checkpoint_artifact_hashes": artifact_hashes,
        "tensor_evidence": tensors,
        "optimizer_state_manifest": optimizer_manifest,
        "optimizer_state_sha256": optimizer_sha256,
        "safe_deserialization": {
            "adapter_format": "safetensors",
            "optimizer_weights_only": True,
            "map_location": "cpu",
        },
    }


def _expected_signed_x_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(config["scene_encoder"].get("signed_x_scene_residual"), "signed-X residual")
    expected = {
        "schema_version": 1,
        "enabled": True,
        "architecture_version": "signed_x_local_field_v2",
        "expected_initial_state_sha256": EXPECTED_INITIAL_SIGNED_X_SHA256,
        "spatial_statistic": "centered_local_content_times_unit_rms_signed_x",
        "spatial_reduction": "none",
        "spatial_centering": "all_slots_fp32",
        "trainable_surface": "bias_free_output_projection_only",
    }
    configured = {
        "schema_version": 1,
        **dict(raw),
        "spatial_statistic": "centered_local_content_times_unit_rms_signed_x",
        "spatial_reduction": "none",
        "spatial_centering": "all_slots_fp32",
        "trainable_surface": "bias_free_output_projection_only",
    }
    if configured != expected:
        _fail("Resolved V21 signed-X local-field contract differs from the exact pin")
    return expected


def _validate_config(
    config: Mapping[str, Any],
    *,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> dict[str, Any]:
    try:
        preflight_contract = validate_v21_config_contract(config, profile=profile)
    except (TypeError, ValueError, RuntimeError, V21StructuralPreflightViolation) as error:
        _fail(f"Resolved V21 config is invalid: {error}")
    observed_hash = config_hash(dict(config))
    if observed_hash != profile.resolved_config_hash:
        _fail(
            f"{profile.version} config hash mismatch: "
            f"expected={profile.resolved_config_hash} observed={observed_hash}"
        )
    if preflight_contract.get("contract_sha256") != profile.normalized_contract_sha256:
        _fail(f"{profile.version} normalized preflight contract SHA-256 mismatch")
    language = {
        "model_id": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "backend": "gemma4",
        "dtype": MODEL_DTYPE,
        "scene_prefix_after_bos": True,
        "scene_boundary_mode": "gemma4_native_image",
    }
    for key, expected in language.items():
        if config["language"].get(key) != expected:
            _fail(f"config.language.{key} mismatch")
    objective_policy = copy.deepcopy(preflight_contract["pair_objective_policy"])
    resolved = copy.deepcopy(objective_policy["by_pair"])
    coverage_body = {
        "schema_version": 1,
        "selected_pair_ids": [COLOR_PAIR_ID, MIRROR_PAIR_ID],
        "configured_pair_ids": [COLOR_PAIR_ID, MIRROR_PAIR_ID],
        "unlisted_pair_ids": [],
        "allow_unlisted_pair_ids": False,
        "resolved_by_pair": resolved,
        "complete": True,
    }
    objective_coverage = {
        **coverage_body,
        "coverage_sha256": _canonical_sha256(coverage_body),
    }
    return {
        "config_hash": observed_hash,
        "config_hash_full": config_hash(dict(config), length=64),
        "model_dtype": MODEL_DTYPE,
        "screen": copy.deepcopy(preflight_contract[profile.screen_key]),
        "global_residual": _expected_global_residual_contract(config),
        "signed_x_residual": _expected_signed_x_contract(config),
        "objective_policy": objective_policy,
        "objective_coverage": objective_coverage,
        "language": language,
        "scene_encoder_architecture_version": "signal_preserving_resampler_v3",
        "preflight_contract_sha256": preflight_contract["contract_sha256"],
    }


def _validate_selection(
    selection: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return v20_selector._validate_selection(selection, contract)
    except v20_selector.V20EpochSelectorViolation as error:
        _fail(f"V21 selection contract is invalid: {error}")


def _validate_zero_equivalence(value: Any, field: str) -> dict[str, Any]:
    try:
        return v20_selector._validate_zero_equivalence(value, field)
    except v20_selector.V20EpochSelectorViolation as error:
        _fail(str(error))


def _validate_initialization(value: Any, field: str) -> dict[str, Any]:
    try:
        return v20_selector._validate_initialization(value, field)
    except v20_selector.V20EpochSelectorViolation as error:
        _fail(str(error))


def _require_current_source(
    current_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = dict(
        capture_git_source_provenance(PROJECT_ROOT)
        if current_provenance is None
        else current_provenance
    )
    try:
        require_clean_committed_source(current)
    except RuntimeError as error:
        _fail(f"V21 selector requires clean committed current source: {error}")
    return current


def _validate_epoch_artifact(
    epoch: int,
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    path: str,
    artifact_sha256: str,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> dict[str, Any]:
    field = f"epoch_{epoch}"
    _assert_finite_tree(artifact, field)
    required = {
        "schema_version": 3,
        "epoch": epoch,
        "global_step": epoch * 12,
        "optimizer_step": epoch,
        "config_hash": contract["config_hash"],
        "output_namespace": profile.output_namespace,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": True,
        "question_dependent_scene_processing": False,
        "global_scene_residual_parameter_count": EXPECTED_GLOBAL_RESIDUAL_PARAMETER_COUNT,
        "global_scene_residual": contract["global_residual"],
        "global_scene_residual_initial_state_sha256": EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256,
        "global_scene_residual_state_sha256": EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
        "frozen_global_scene_residual_state_sha256": EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
        "global_scene_residual_zero_output_equivalence": None,
        "signed_x_scene_residual_parameter_count": EXPECTED_SIGNED_X_PARAMETER_COUNT,
        "signed_x_scene_residual": contract["signed_x_residual"],
        "signed_x_scene_residual_initial_state_sha256": EXPECTED_INITIAL_SIGNED_X_SHA256,
        "frozen_scene_state_sha256": EXPECTED_FROZEN_SCENE_SHA256,
        "frozen_lora_bank_state_sha256": EXPECTED_FROZEN_BANKS,
        "lora_bank_state_sha256": EXPECTED_FROZEN_BANKS,
        "lora_trainable_parameter_count": 0,
        "scene_ids": list(EXPECTED_TRAIN_SCENES),
        "train_scene_ids": list(EXPECTED_TRAIN_SCENES),
        "validation_scene_ids": [],
        "test_scene_ids": list(EXPECTED_TEST_SCENES),
        "scene_latents": 256,
        "scene_model_dim": 384,
        "semantic_dim": 3072,
        "language_hidden_dim": 1536,
        "language_model_id": contract["language"]["model_id"],
        "language_revision": contract["language"]["revision"],
        "language_backend": contract["language"]["backend"],
        "scene_encoder_architecture_version": contract["scene_encoder_architecture_version"],
        "scene_prefix_after_bos": True,
        "scene_boundary_mode": "gemma4_native_image",
        "gradient_accumulation": 12,
        "counterfactual_pair_unit_count": 12,
        "counterfactual_pair_unit_selection_sha256": EXPECTED_PAIR_SELECTION_SHA256,
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": EXPECTED_PAIR_MEMBERSHIP_SHA256,
        "initialize_expected_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "initialize_expected_metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
        "initialize_expected_global_scene_residual_state_sha256": (
            EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256
        ),
        "initialize_source_residual_into_frozen_base": True,
    }
    for key, expected in required.items():
        if artifact.get(key) != expected:
            _fail(f"{field}.{key} mismatch")
    for stale in ("v18_stage_execution", "v19_stage_execution", "v19_screen", "v20_screen"):
        if stale in artifact:
            _fail(f"{field} improperly carries stale controller {stale}")
    signed_state = _sha256(
        artifact.get("signed_x_scene_residual_state_sha256"),
        f"{field}.signed_x_scene_residual_state_sha256",
    )
    if signed_state == EXPECTED_INITIAL_SIGNED_X_SHA256:
        _fail(f"{field} signed-X local-field state did not change")
    source = _validate_source_provenance(artifact.get("source_provenance"), f"{field}.source")
    initialization = _validate_initialization(
        artifact.get("initialization_provenance"), f"{field}.initialization"
    )
    try:
        _validate_lora_contract(artifact.get("lora"), f"{field}.lora")
    except ValueError as error:
        _fail(str(error))
    equivalence = _validate_zero_equivalence(
        artifact.get("signed_x_scene_residual_zero_output_equivalence"),
        f"{field}.signed_x_zero_output_equivalence",
    )
    curriculum = _mapping(artifact.get("pair_curriculum"), f"{field}.pair_curriculum")
    expected_curriculum = {
        "enabled": True,
        "pair_only": True,
        "pair_only_scene_ids": list(EXPECTED_TRAIN_SCENES),
        "max_units_per_pair": 6,
        "ranking_weight": 8.0,
        "ranking_margin": 1.0,
        "ranking_mode": "candidate_logit",
        "full_vocab_ranking_weight": 2.0,
        "full_vocab_ranking_margin": 1.0,
        "batch_fraction": 1.0,
        "units_per_batch": 1,
        "steps_per_epoch": 12,
        "gate_enabled": True,
        "objective_policy": contract["objective_policy"],
        "objective_policy_coverage": contract["objective_coverage"],
    }
    if dict(curriculum) != expected_curriculum:
        _fail(f"{field}.pair_curriculum mismatch")
    if artifact.get("pair_gate_policy") != {
        "stop_when_passed": False,
        "first_answer_token_top1_accuracy_threshold": 1.0,
    }:
        _fail(f"{field}.pair_gate_policy mismatch")
    history = list(_sequence(artifact.get("history"), f"{field}.history"))
    if len(history) != epoch:
        _fail(f"{field}.history must contain exactly {epoch} cumulative entries")
    parsed = [
        _extract_epoch_metrics(_mapping(item, f"{field}.history[{index - 1}]"), index)
        for index, item in enumerate(history, start=1)
    ]
    last = _mapping(history[-1], f"{field}.history[-1]")
    if artifact.get("pair_candidate_gate") != last.get("pair_candidate_gate"):
        _fail(f"{field} top-level pair gate is not its final history gate")
    if artifact.get("train_loss") != last.get("train_loss"):
        _fail(f"{field} top-level train loss is not its final history loss")
    return {
        "epoch": epoch,
        "path": path,
        "artifact_sha256": artifact_sha256,
        "source_provenance": source,
        "initialization_provenance": initialization,
        "zero_output_equivalence": equivalence,
        "signed_x_state_sha256": signed_state,
        "history": history,
        "metrics": parsed[-1],
    }


def _color_eligible(candidate: Mapping[str, Any]) -> bool:
    color = _mapping(candidate["color"], "candidate.color")
    return bool(
        color["full_vocab_sides"] == 12
        and color["full_vocab_units"] == 6
        and color["minimum_candidate_margin"] > 0.0
        and color["minimum_full_vocab_margin"] > 0.0
    )


def _continuation_passed(candidate: Mapping[str, Any], screen: Mapping[str, Any]) -> bool:
    mirror = _mapping(candidate["mirror"], "candidate.mirror")
    threshold = _mapping(screen["continuation_requires"], "continuation requirements")
    return bool(
        _color_eligible(candidate)
        and mirror["full_vocab_sides"] >= threshold["mirror_minimum_full_vocab_sides"]
        and mirror["full_vocab_units"] >= threshold["mirror_minimum_full_vocab_units"]
    )


def _full_teacher_passed(candidate: Mapping[str, Any]) -> bool:
    color = _mapping(candidate["color"], "candidate.color")
    mirror = _mapping(candidate["mirror"], "candidate.mirror")
    all_minimums_positive = all(
        pair[key] > 0.0
        for pair in (color, mirror)
        for key in ("minimum_candidate_margin", "minimum_full_vocab_margin")
    )
    return bool(
        color["full_vocab_sides"] == 12
        and color["full_vocab_units"] == 6
        and mirror["full_vocab_sides"] == 12
        and mirror["full_vocab_units"] == 6
        and all_minimums_positive
    )


def _ranking_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    mirror = _mapping(candidate["mirror"], "candidate.mirror")
    values = {
        "mirror_full_vocab_units": mirror["full_vocab_units"],
        "mirror_full_vocab_sides": mirror["full_vocab_sides"],
        "mirror_candidate_units": mirror["candidate_units"],
        "mirror_candidate_sides": mirror["candidate_sides"],
        "mirror_mean_full_vocab_margin": mirror["mean_full_vocab_margin"],
        "mirror_minimum_full_vocab_margin": mirror["minimum_full_vocab_margin"],
    }
    return tuple(
        [-float(values[field]) for field in EXPECTED_RANKING_FIELDS] + [float(candidate["epoch"])]
    )


def summarize_v21_epochs(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    epoch_artifacts: Mapping[int, Mapping[str, Any]],
    *,
    update1_report_path: str | Path,
    selection_path: str | Path,
    selection_sha256: str,
    epoch_paths: Mapping[int, str | Path],
    epoch_sha256: Mapping[int, str],
    current_provenance: Mapping[str, Any] | None = None,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> dict[str, Any]:
    """Validate, artifact-bind, rank, and gate four cumulative V21 updates."""

    expected = set(EXPECTED_EPOCHS)
    if set(epoch_artifacts) != expected:
        _fail("V21 selector requires exactly epoch artifacts 1,2,3,4")
    paths = dict(epoch_paths)
    hashes = dict(epoch_sha256)
    if set(paths) != expected or set(hashes) != expected:
        _fail("Epoch paths and hashes must cover exactly epochs 1,2,3,4")
    for epoch in EXPECTED_EPOCHS:
        _sha256(hashes[epoch], f"epoch_{epoch}.artifact_sha256")
    _sha256(selection_sha256, "selection artifact SHA-256")
    _require_bound_json(selection_path, selection, selection_sha256, "V21 selection artifact")

    current_source = _require_current_source(current_provenance)
    # Keep the historical V21 call shape stable for existing audit tests and
    # downstream users that monkeypatch these narrow seams.  Only a dedicated
    # revision wrapper supplies the immutable non-default profile.
    if profile is V21_LOCAL_FIELD_PROFILE:
        contract = _validate_config(config)
        update1 = _load_update1_authorization(update1_report_path, config=config)
    else:
        contract = _validate_config(config, profile=profile)
        update1 = _load_update1_authorization(
            update1_report_path,
            config=config,
            profile=profile,
        )
    selection_evidence = _validate_selection(selection, contract)
    validated: list[dict[str, Any]] = []
    for epoch in EXPECTED_EPOCHS:
        _require_bound_json(
            paths[epoch],
            epoch_artifacts[epoch],
            hashes[epoch],
            f"V21 epoch_{epoch} metadata",
        )
        inspection = _inspect_checkpoint_artifacts(
            epoch,
            epoch_artifacts[epoch],
            config=config,
            metadata_path=paths[epoch],
        )
        if inspection["checkpoint_artifact_hashes"]["metadata_sha256"] != hashes[epoch]:
            _fail(f"epoch_{epoch} loaded metadata hash differs from its actual file")
        row_arguments = {
            "path": str(paths[epoch]),
            "artifact_sha256": hashes[epoch],
        }
        if profile is V21_LOCAL_FIELD_PROFILE:
            row = _validate_epoch_artifact(
                epoch,
                epoch_artifacts[epoch],
                contract,
                **row_arguments,
            )
        else:
            row = _validate_epoch_artifact(
                epoch,
                epoch_artifacts[epoch],
                contract,
                profile=profile,
                **row_arguments,
            )
        if inspection["tensor_evidence"]["signed_x_state_sha256"] != row["signed_x_state_sha256"]:
            _fail(f"epoch_{epoch} actual signed-X tensor state differs from metadata")
        row["checkpoint_inspection"] = inspection
        validated.append(row)
    source = validated[0]["source_provenance"]
    initialization = validated[0]["initialization_provenance"]
    equivalence = validated[0]["zero_output_equivalence"]
    if selection_evidence["source_provenance"] != source:
        _fail("Selection and epochs do not share exact clean source provenance")
    if update1["source_provenance"] != source:
        _fail("Update-one authorization and epochs do not share exact clean source provenance")
    if current_source != source:
        _fail("Current clean source provenance differs from the exact V21 evidence chain")

    epoch1_inspection = validated[0]["checkpoint_inspection"]
    if _resolve(update1["checkpoint"]) != _resolve(epoch1_inspection["checkpoint"]):
        _fail("Update-one authorization checkpoint path differs from epoch one")
    if update1["checkpoint_artifact_hashes"] != epoch1_inspection["checkpoint_artifact_hashes"]:
        _fail("Update-one authorization artifact hashes differ from epoch one")
    epoch1_tensors = epoch1_inspection["tensor_evidence"]
    tensor_bindings = {
        "signed_x_state_sha256": "signed_x_state_sha256",
        "output_projection_sha256": "output_projection_sha256",
        "frozen_global_scene_residual_state_sha256": "global_scene_residual_state_sha256",
        "frozen_scene_state_sha256": "scene_state_sha256",
        "frozen_lora_bank_state_sha256": "lora_bank_state_sha256",
    }
    for report_key, tensor_key in tensor_bindings.items():
        if update1[report_key] != epoch1_tensors[tensor_key]:
            _fail(f"Update-one authorization {report_key} differs from epoch-one tensors")
    if (
        update1["optimizer_state_manifest"] != epoch1_inspection["optimizer_state_manifest"]
        or update1["optimizer_state_sha256"] != epoch1_inspection["optimizer_state_sha256"]
    ):
        _fail("Update-one authorization optimizer state differs from epoch one")
    for row in validated[1:]:
        if row["source_provenance"] != source:
            _fail("Epoch artifacts do not share exact clean source provenance")
        if row["initialization_provenance"] != initialization:
            _fail("Epoch artifacts do not share exact V18 initialization provenance")
        if row["zero_output_equivalence"] != equivalence:
            _fail("Epoch artifacts do not preserve update-0 local-field equivalence")
    for earlier, later in pairwise(validated):
        if later["history"][: earlier["epoch"]] != earlier["history"]:
            _fail(
                f"Epoch {later['epoch']} does not preserve exact cumulative history "
                f"from epoch {earlier['epoch']}"
            )
    states = [row["signed_x_state_sha256"] for row in validated]
    if len(set(states)) != len(states):
        _fail("Signed-X local-field state repeats or rolls back across updates")

    candidates: list[dict[str, Any]] = []
    for row in validated:
        inspection = row["checkpoint_inspection"]
        candidate = {
            "epoch": row["epoch"],
            "optimizer_step": row["epoch"],
            "cumulative_microsteps": row["epoch"] * 12,
            "checkpoint": inspection["checkpoint"],
            "checkpoint_metadata_path": row["path"],
            "checkpoint_metadata_sha256": row["artifact_sha256"],
            "checkpoint_artifact_hashes": copy.deepcopy(inspection["checkpoint_artifact_hashes"]),
            "signed_x_state_sha256": row["signed_x_state_sha256"],
            "optimizer_state_sha256": inspection["optimizer_state_sha256"],
            "frozen_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
            "frozen_scene_state_sha256": EXPECTED_FROZEN_SCENE_SHA256,
            "model_dtype": MODEL_DTYPE,
            "color": copy.deepcopy(row["metrics"]["color"]),
            "mirror": copy.deepcopy(row["metrics"]["mirror"]),
        }
        candidate["color_eligible"] = _color_eligible(candidate)
        candidate["continuation_gate_passed"] = _continuation_passed(candidate, contract["screen"])
        candidate["full_teacher_gate_passed"] = _full_teacher_passed(candidate)
        candidates.append(candidate)
    ranking = sorted(
        (copy.deepcopy(item) for item in candidates if item["color_eligible"]),
        key=_ranking_key,
    )
    for rank, candidate in enumerate(ranking, start=1):
        candidate["rank"] = rank
    selected = ranking[0] if ranking else None
    continuation = bool(selected and selected["continuation_gate_passed"])
    full_teacher = bool(selected and selected["full_teacher_gate_passed"])
    greedy = bool(full_teacher and contract["screen"]["greedy_audit_only_after_full_teacher_gate"])
    return {
        "schema_version": 1,
        "selector_type": profile.selector_type,
        "report_only": True,
        "model_inference_executed": False,
        "gemma_model_loaded": False,
        "checkpoint_tensor_state_loaded": True,
        "checkpoint_tensor_state_safely_inspected": True,
        "optimizer_deserialized": True,
        "optimizer_deserialization_weights_only": True,
        "question_dependent_scene_processing": False,
        "model_dtype": MODEL_DTYPE,
        "config_path": str(profile.config_path),
        "config_hash": contract["config_hash"],
        "config_hash_full": contract["config_hash_full"],
        "preflight_contract_sha256": contract["preflight_contract_sha256"],
        "update1_authorization": copy.deepcopy(update1),
        "selection_artifact_path": str(selection_path),
        "selection_artifact_sha256": selection_sha256,
        "selection_ids_sha256": selection_evidence["selection_sha256"],
        "pair_unit_selection_sha256": selection_evidence["pair_selection_sha256"],
        "pair_membership_sha256": selection_evidence["pair_membership_sha256"],
        "source_provenance": copy.deepcopy(source),
        "initialization_provenance": copy.deepcopy(initialization),
        "selection_policy": copy.deepcopy(contract["screen"]),
        "selection_policy_sha256": _canonical_sha256(contract["screen"]),
        "ranking_descending": list(EXPECTED_RANKING_FIELDS),
        "epoch_count": len(candidates),
        "epochs": candidates,
        "cumulative_update_evidence": {
            "stage_1_optimizer_updates": 1,
            "stage_1_stop_required": True,
            "screen_optimizer_updates": 4,
            "microsteps_per_optimizer_update": 12,
            "optimizer_steps": [1, 2, 3, 4],
            "cumulative_microsteps": [12, 24, 36, 48],
            "history_prefixes_exact": True,
            "signed_x_states_unique": True,
            "all_checkpoint_artifact_hashes_bound": True,
            "all_optimizer_steps_safely_validated": True,
            "update1_authorization_transitively_bound": True,
            "model_dtype": MODEL_DTYPE,
            "frozen_global_state_sha256": EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
            "frozen_scene_state_sha256": EXPECTED_FROZEN_SCENE_SHA256,
        },
        "eligible_epoch_count": len(ranking),
        "ranking": ranking,
        "selected_epoch": None if selected is None else selected["epoch"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "selected_checkpoint_metadata_path": (
            None if selected is None else selected["checkpoint_metadata_path"]
        ),
        "selected_checkpoint_metadata_sha256": (
            None if selected is None else selected["checkpoint_metadata_sha256"]
        ),
        "selected_checkpoint_artifact_hashes": (
            None if selected is None else copy.deepcopy(selected["checkpoint_artifact_hashes"])
        ),
        "selected_optimizer_state_sha256": (
            None if selected is None else selected["optimizer_state_sha256"]
        ),
        "selected_signed_x_state_sha256": (
            None if selected is None else selected["signed_x_state_sha256"]
        ),
        "continuation_gate_passed": continuation,
        "continuation_authorized": continuation,
        "conditional_max_optimizer_updates": 8,
        "full_teacher_gate_passed": full_teacher,
        "greedy_audit_authorized": greedy,
        "greedy_audit_forbidden": not greedy,
        "decision": (
            "no_color_eligible_epoch_no_extension_no_greedy"
            if selected is None
            else "full_teacher_gate_passed_greedy_audit_allowed"
            if greedy
            else "continue_selected_epoch_no_greedy_audit"
            if continuation
            else "screen_failed_no_extension_no_greedy_audit"
        ),
    }


def _parse_epoch_path(value: str) -> tuple[int, Path]:
    raw_epoch, separator, raw_path = value.partition("=")
    try:
        epoch = int(raw_epoch)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epoch binding must be EPOCH=PATH") from error
    if not separator or epoch not in EXPECTED_EPOCHS or not raw_path:
        raise argparse.ArgumentTypeError("epoch binding must be 1=PATH through 4=PATH")
    return epoch, Path(raw_path)


def write_selector_report(summary: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PINNED_CONFIG_PATH)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--update1-report", type=Path, required=True)
    parser.add_argument("--epoch", action="append", type=_parse_epoch_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    epoch_paths = dict(args.epoch)
    if len(epoch_paths) != len(args.epoch):
        parser.error("duplicate V21 epoch binding")
    _reject_forbidden_input_path(args.config)
    config = load_config(args.config)
    selection, selection_digest = _load_json_strict(args.selection)
    loaded = {epoch: _load_json_strict(path) for epoch, path in epoch_paths.items()}
    summary = summarize_v21_epochs(
        config,
        selection,
        {epoch: value for epoch, (value, _digest) in loaded.items()},
        update1_report_path=args.update1_report,
        selection_path=str(args.selection),
        selection_sha256=selection_digest,
        epoch_paths={epoch: str(path) for epoch, path in epoch_paths.items()},
        epoch_sha256={epoch: digest for epoch, (_value, digest) in loaded.items()},
    )
    destination = write_selector_report(summary, args.output)
    print(
        json.dumps(
            {
                "output": str(destination),
                "selected_epoch": summary["selected_epoch"],
                "continuation_authorized": summary["continuation_authorized"],
                "full_teacher_gate_passed": summary["full_teacher_gate_passed"],
                "greedy_audit_authorized": summary["greedy_audit_authorized"],
                "decision": summary["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
