"""Strict artifact-bound selector for the four predeclared V20 updates.

The selector binds the separately emitted update-one authorization report to
all four cumulative checkpoints.  It safely inspects adapter tensors and the
one-matrix AdamW state on CPU, but never loads Gemma, performs model inference,
or reads a scene map, rendered observation, question, runtime, or oracle file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation.residual_lr_response import EXPECTED_RANKING_FIELDS
from semantic_3d_chat.evaluation.v19_epoch_selector import (
    EXPECTED_FROZEN_BANKS,
    EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
    EXPECTED_FROZEN_SCENE_SHA256,
    EXPECTED_GLOBAL_RESIDUAL_PARAMETER_COUNT,
    EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256,
    EXPECTED_PAIR_MEMBERSHIP_SHA256,
    EXPECTED_PAIR_SELECTION_SHA256,
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SIGNED_X_PARAMETER_COUNT,
    EXPECTED_SOURCE_ADAPTER_SHA256,
    EXPECTED_SOURCE_CHECKPOINT_CONFIG_HASH,
    EXPECTED_SOURCE_CHECKPOINT_EPOCH,
    EXPECTED_SOURCE_CHECKPOINT_NAMESPACE,
    EXPECTED_SOURCE_METADATA_SHA256,
    EXPECTED_TEST_SCENES,
    EXPECTED_TRAIN_SCENES,
    _assert_finite_tree,
    _canonical_sha256,
    _expected_global_residual_contract,
    _expected_objective_coverage,
    _expected_objective_policy,
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
from semantic_3d_chat.evaluation.v20_update1_verifier import (
    UPDATE1_VERIFIER_TYPE,
    V20Update1Violation,
    _load_tensor_evidence,
)

PINNED_CONFIG_PATH = Path("configs/experiments/gemma4_color_mirror_signed_x_local_field_v20.yaml")
PINNED_CONFIG_HASH = "a40636303078"
OUTPUT_NAMESPACE = "gemma4_color_mirror_signed_x_local_field_v20"
EXPECTED_EPOCHS = (1, 2, 3, 4)
EXPECTED_INITIAL_SIGNED_X_SHA256 = (
    "3f249307901df75ba07a758a7dc5b02c7c6ff9bbb969987741a106b8d8977ce1"
)
_UPDATE1_REDUCTION_HASH_FIELDS = (
    "local_field_structural_state_sha256",
    "local_dependence_sha256",
    "local_hidden_spatial_rank_sha256",
    "centered_content_sha256",
    "raw_fp32_centered_scene_delta_sha256",
    "bf16_effective_scene_delta_sha256",
    "bf16_cast_audit_sha256",
    "raw_fp32_centered_pair_delta_sha256",
    "bf16_effective_pair_delta_sha256",
    "structural_gate_sha256",
)


class V20EpochSelectorViolation(ValueError):
    """A fail-closed V20 evidence, provenance, or gate-policy violation."""


def _fail(message: str) -> None:
    raise V20EpochSelectorViolation(message)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _lexical_absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _reject_forbidden_lexical_and_resolved_path(path: Path) -> None:
    for candidate in (path.absolute(), path.resolve()):
        forbidden = sorted(
            component
            for component in (part.casefold() for part in candidate.parts)
            if component in {"oracle", "rendered", "maps", "scene_tokens", "runtime"}
        )
        if forbidden:
            _fail(f"Selector refuses runtime/oracle artifact path components: {forbidden}")


def _safe_checkpoint_file(path: Path, field: str) -> Path:
    """Require an ordinary file with no symlink anywhere in its path."""

    absolute = _lexical_absolute(path)
    _reject_forbidden_lexical_and_resolved_path(absolute)
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        if cursor.is_symlink():
            _fail(f"{field} path must not contain a symbolic link: {cursor}")
    if not absolute.is_file():
        _fail(f"{field} is missing or is not a regular file: {absolute}")
    return absolute


def _file_sha256(path: Path, field: str) -> str:
    safe = _safe_checkpoint_file(path, field)
    digest = hashlib.sha256()
    with safe.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_rich_preflight_reduction(value: Any) -> dict[str, Any]:
    reduction = dict(_mapping(value, "update1.rich_preflight_reduction"))
    expected_keys = {
        "schema_version",
        "verified",
        "bf16_algorithm",
        "scene_ids",
        "pair_ids",
        *_UPDATE1_REDUCTION_HASH_FIELDS,
        "canonical_sha256",
    }
    if set(reduction) != expected_keys:
        _fail("update1.rich_preflight_reduction keys mismatch")
    if reduction.get("schema_version") != 1 or reduction.get("verified") is not True:
        _fail("update1 rich preflight reduction is not verified schema 1 evidence")
    if reduction.get("bf16_algorithm") != "bfloat16_cast_of_fp32_base_plus_fp32_delta":
        _fail("update1 rich preflight reduction has the wrong BF16 algorithm")
    if reduction.get("scene_ids") != list(EXPECTED_TRAIN_SCENES):
        _fail("update1 rich preflight reduction scene IDs mismatch")
    if reduction.get("pair_ids") != ["pair_000001", "pair_000003"]:
        _fail("update1 rich preflight reduction pair IDs mismatch")
    for key in _UPDATE1_REDUCTION_HASH_FIELDS:
        _sha256(reduction.get(key), f"update1.rich_preflight_reduction.{key}")
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
) -> dict[str, Any]:
    """Load and validate the exact stage-two authorization report."""

    safe_report = _safe_checkpoint_file(_lexical_absolute(path), "V20 update-one report")
    resolved = safe_report.resolve()
    try:
        raw, report_sha256 = _load_json_strict(resolved)
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Cannot load required V20 update-one authorization: {error}")
    report = dict(raw)
    _assert_finite_tree(report, "update1")
    for key, expected in {
        "schema_version": 1,
        "audit_type": UPDATE1_VERIFIER_TYPE,
        "match": True,
        "stage_2_authorized": True,
        "report_only": True,
        "model_loaded": False,
        "scene_map_loaded": False,
        "oracle_loaded": False,
        "optimizer_deserialized": True,
    }.items():
        if report.get(key) != expected:
            _fail(f"update1.{key} does not authorize downstream V20 selection")
    if report.get("optimizer_deserialization") != {
        "weights_only": True,
        "map_location": "cpu",
        "canonical_state_validated": True,
    }:
        _fail("update1 optimizer deserialization evidence mismatch")
    full_config_hash = config_hash(dict(config), length=64)
    if report.get("config_hash") != full_config_hash:
        _fail("update1 report config provenance mismatch")
    source = _validate_source_provenance(report.get("source_provenance"), "update1.source")
    _sha256(report.get("preflight_sha256"), "update1.preflight_sha256")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        _fail("update1.checkpoint must be a nonempty path")
    artifact_hashes = dict(
        _mapping(report.get("checkpoint_artifact_hashes"), "update1.checkpoint_artifact_hashes")
    )
    if set(artifact_hashes) != {"adapter_sha256", "metadata_sha256", "optimizer_sha256"}:
        _fail("update1 checkpoint artifact hash keys mismatch")
    for key, digest in artifact_hashes.items():
        _sha256(digest, f"update1.checkpoint_artifact_hashes.{key}")
    signed_state = _sha256(report.get("signed_x_state_sha256"), "update1 signed-X state")
    output_projection = _sha256(report.get("output_projection_sha256"), "update1 output projection")
    frozen_global = _sha256(
        report.get("frozen_global_scene_residual_state_sha256"), "update1 frozen global"
    )
    frozen_scene = _sha256(report.get("frozen_scene_state_sha256"), "update1 frozen scene")
    frozen_lora = dict(_mapping(report.get("frozen_lora_bank_state_sha256"), "update1 LoRA"))
    if frozen_lora != EXPECTED_FROZEN_BANKS:
        _fail("update1 frozen LoRA state mismatch")
    training = _mapping(config.get("training"), "config.training")
    optimizer_contract = dict(_mapping(training.get("optimizer"), "config.training.optimizer"))
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
    reduction = _validate_rich_preflight_reduction(report.get("rich_preflight_reduction"))
    return {
        "report_path": _display(resolved),
        "report_sha256": report_sha256,
        "report_canonical_sha256": _canonical_sha256(report),
        "audit_type": UPDATE1_VERIFIER_TYPE,
        "match": True,
        "stage_2_authorized": True,
        "source_provenance": source,
        "config_hash": full_config_hash,
        "preflight_sha256": report["preflight_sha256"],
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
    """Safely bind actual adapter/metadata/optimizer files to one epoch."""

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
    except V20Update1Violation as error:
        _fail(f"epoch_{epoch} adapter state violates the V20 contract: {error}")
    training = _mapping(config.get("training"), "config.training")
    optimizer_contract = dict(_mapping(training.get("optimizer"), "config.training.optimizer"))
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
    scene = _mapping(config.get("scene_encoder"), "config.scene_encoder")
    raw = _mapping(scene.get("signed_x_scene_residual"), "signed_x_scene_residual")
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
        _fail("Resolved V20 signed-X local-field contract differs from the exact pin")
    return expected


def _expected_screen() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "signed_x_local_field_architecture_screen",
        "source_checkpoint_epoch": 4,
        "screen_optimizer_updates": 4,
        "conditional_max_optimizer_updates": 8,
        "stage_1_optimizer_updates": 1,
        "stage_1_stop_required": True,
        "structural_preflight_requires": {
            "maximum_per_scene_raw_and_effective_delta_to_core_rms_ratio": 0.01,
            "minimum_mirror_effective_residual_to_core_rms_ratio": 0.01,
            "minimum_mirror_to_color_normalized_effective_selectivity": 1.5,
            "minimum_local_hidden_spatial_rank": 2,
        },
        "eligibility_requires": {
            "color_full_vocab_sides": 12,
            "color_full_vocab_units": 6,
            "color_positive_minimum_candidate_margin": True,
            "color_positive_minimum_full_vocab_margin": True,
        },
        "continuation_requires": {
            "mirror_minimum_full_vocab_sides": 8,
            "mirror_minimum_full_vocab_units": 2,
        },
        "full_teacher_gate_requires": {
            "color_full_vocab_sides": 12,
            "color_full_vocab_units": 6,
            "mirror_full_vocab_sides": 12,
            "mirror_full_vocab_units": 6,
            "all_candidate_and_full_vocab_minimum_margins_positive": True,
        },
        "greedy_audit_only_after_full_teacher_gate": True,
    }


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    observed_hash = config_hash(dict(config))
    if observed_hash != PINNED_CONFIG_HASH:
        _fail(f"V20 config hash mismatch: expected={PINNED_CONFIG_HASH} observed={observed_hash}")
    if config.get("structural_preflight") is not None or config.get("v18_screen") is not None:
        _fail("V20 must not inherit a completed V18 preflight or screen controller")
    if config.get("v19_screen") is not None:
        _fail("V20 must not inherit V19's failed screen controller")
    screen = _mapping(config.get("v20_screen"), "config.v20_screen")
    expected_screen = _expected_screen()
    if dict(screen) != expected_screen:
        _fail("Resolved v20_screen differs from the exact predeclared policy")

    training = _mapping(config.get("training"), "config.training")
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": 1.0e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "amsgrad": False,
        "gradient_clip_norm": 1.0,
        "accumulation_divisor": 12,
        "step_index": 1,
    }
    expected_training = {
        "output_namespace": OUTPUT_NAMESPACE,
        "initialize_from": (
            "data_gemma4/checkpoints/gemma4_color_mirror_centered_content_gate_v18/epoch_004"
        ),
        "initialize_expected_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "initialize_expected_metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
        "initialize_expected_global_scene_residual_state_sha256": (
            EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256
        ),
        "initialize_legacy_lora_into_bank": None,
        "initialize_named_lora_freeze_transition": False,
        "initialize_source_residual_into_frozen_base": True,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": True,
        "epochs": 4,
        "pair_steps_per_epoch": 12,
        "gradient_accumulation": 12,
        "pair_gate_every_epochs": 1,
        "pair_gate_stop_when_passed": False,
        "early_stopping_patience": 0,
        "learning_rate": 1.0e-4,
        "weight_decay": 0.0,
        "pair_only_mode": True,
        "pair_only_scene_ids": list(EXPECTED_TRAIN_SCENES),
        "pair_max_units_per_pair": 6,
        "pair_batch_fraction": 1.0,
        "pair_units_per_batch": 1,
        "pair_ranking_mode": "candidate_logit",
        "pair_ranking_weight": 8.0,
        "pair_ranking_margin": 1.0,
        "pair_full_vocab_ranking_weight": 2.0,
        "pair_full_vocab_ranking_margin": 1.0,
        "optimizer": expected_optimizer,
    }
    mismatches = {
        key: {"expected": expected, "observed": training.get(key)}
        for key, expected in expected_training.items()
        if training.get(key) != expected
    }
    if mismatches:
        _fail(f"Resolved V20 training contract mismatch: {mismatches}")
    expected_pair_objectives = {
        "schema_version": 1,
        "allow_unlisted_pair_ids": False,
        "by_pair": deepcopy(_expected_objective_policy()["by_pair"]),
    }
    if training.get("pair_objectives") != expected_pair_objectives:
        _fail("Resolved V20 per-pair objective policy mismatch")
    for field in (
        "latent_diversity_weight",
        "paired_scene_separation_weight",
        "grounding_weight",
        "grounding_anchor_weight",
        "spatial_answer_contrastive_weight",
        "spatial_answer_warmup_steps",
        "spatial_relation_contrastive_weight",
        "spatial_relation_warmup_steps",
    ):
        if training.get(field) not in (0, 0.0):
            _fail(f"config.training.{field} must be disabled for V20")

    scene = _mapping(config.get("scene_encoder"), "config.scene_encoder")
    for key, expected in {
        "architecture_version": "signal_preserving_resampler_v3",
        "input_voxel_size_m": 0.15,
        "model_dim": 384,
        "global_latents": 256,
    }.items():
        if scene.get(key) != expected:
            _fail(f"config.scene_encoder.{key} mismatch")
    language = _mapping(config.get("language"), "config.language")
    expected_language = {
        "model_id": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "backend": "gemma4",
        "scene_prefix_after_bos": True,
        "scene_boundary_mode": "gemma4_native_image",
    }
    for key, expected in expected_language.items():
        if language.get(key) != expected:
            _fail(f"config.language.{key} mismatch")

    expected_experiment = {
        "schema_version": 1,
        "role": "exploratory_reflection_odd_local_field_screen_v20",
        "question_dependent_scene_processing": False,
        "residual_parameter_count": EXPECTED_GLOBAL_RESIDUAL_PARAMETER_COUNT,
        "signed_x_residual_parameter_count": EXPECTED_SIGNED_X_PARAMETER_COUNT,
        "screen_optimizer_updates": 4,
        "conditional_max_optimizer_updates": 8,
        "source_checkpoint_epoch": 4,
        "source_scene_state_sha256": EXPECTED_FROZEN_SCENE_SHA256,
        "source_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
        "source_inherited_bank_sha256": EXPECTED_FROZEN_BANKS["inherited_v12"],
        "source_extension_bank_sha256": EXPECTED_FROZEN_BANKS["extension_v13"],
        "screen_extension_requires": {
            **expected_screen["eligibility_requires"],
            **expected_screen["continuation_requires"],
        },
        "full_teacher_gate_requires": expected_screen["full_teacher_gate_requires"],
        "greedy_audit_only_after_full_teacher_gate": True,
    }
    if dict(_mapping(config.get("experiment"), "config.experiment")) != expected_experiment:
        _fail("Resolved V20 experiment provenance contract mismatch")

    return {
        "config_hash": observed_hash,
        "screen": deepcopy(expected_screen),
        "global_residual": _expected_global_residual_contract(config),
        "signed_x_residual": _expected_signed_x_contract(config),
        "objective_policy": _expected_objective_policy(),
        "objective_coverage": _expected_objective_coverage(),
        "language": expected_language,
        "scene_encoder_architecture_version": "signal_preserving_resampler_v3",
    }


def _validate_selection(
    selection: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    _assert_finite_tree(selection, "selection")
    if selection.get("schema_version") != 1:
        _fail("selection.schema_version must equal 1")
    source = _validate_source_provenance(selection.get("source_provenance"), "selection.source")
    train = _mapping(selection.get("train"), "selection.train")
    for key, expected in {
        "available_count": 24,
        "selected_count": 24,
        "selected_ids_sha256": EXPECTED_SELECTION_SHA256,
        "expected_change_units_selected": 12,
        "expected_change_units_complete": 12,
        "expected_change_units_incomplete": 0,
    }.items():
        if train.get(key) != expected:
            _fail(f"selection.train.{key} mismatch")
    required = {
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": True,
        "gradient_accumulation": 12,
        "initialize_expected_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "initialize_expected_metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
        "initialize_expected_global_scene_residual_state_sha256": (
            EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256
        ),
        "initialize_legacy_lora_into_bank": None,
        "initialize_named_lora_freeze_transition": False,
        "initialize_source_residual_into_frozen_base": True,
        "train_scene_ids": list(EXPECTED_TRAIN_SCENES),
        "validation_scene_ids": [],
        "test_scene_ids": list(EXPECTED_TEST_SCENES),
        "counterfactual_pair_unit_count": 12,
        "counterfactual_pair_unit_selection_sha256": EXPECTED_PAIR_SELECTION_SHA256,
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": EXPECTED_PAIR_MEMBERSHIP_SHA256,
        "global_scene_residual": contract["global_residual"],
        "signed_x_scene_residual": contract["signed_x_residual"],
    }
    for key, expected in required.items():
        if selection.get(key) != expected:
            _fail(f"selection.{key} mismatch")
    curriculum = _mapping(selection.get("pair_curriculum"), "selection.pair_curriculum")
    expected_curriculum = {
        "enabled": True,
        "pair_only": True,
        "pair_only_scene_ids": list(EXPECTED_TRAIN_SCENES),
        "max_units_per_pair": 6,
        "batch_fraction": 1.0,
        "units_per_batch": 1,
        "ranking_mode": "candidate_logit",
        "ranking_margin": 1.0,
        "ranking_weight": 8.0,
        "full_vocab_ranking_margin": 1.0,
        "full_vocab_ranking_weight": 2.0,
        "gate_enabled": True,
        "gate_every_epochs": 1,
        "gate_stop_when_passed": False,
        "gate_first_answer_token_top1_accuracy": 1.0,
        "objective_policy": contract["objective_policy"],
        "objective_policy_coverage": contract["objective_coverage"],
    }
    if dict(curriculum) != expected_curriculum:
        _fail("selection.pair_curriculum mismatch")
    try:
        _validate_lora_contract(selection.get("lora"), "selection.lora")
    except ValueError as error:
        _fail(str(error))
    return {
        "source_provenance": source,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "pair_selection_sha256": EXPECTED_PAIR_SELECTION_SHA256,
        "pair_membership_sha256": EXPECTED_PAIR_MEMBERSHIP_SHA256,
    }


def _validate_zero_equivalence(value: Any, field: str) -> dict[str, Any]:
    equivalence = dict(_mapping(value, field))
    for key, expected in {
        "verified": True,
        "base": "loaded_frozen_global_scene_residual",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
        "scene_count": 4,
    }.items():
        if equivalence.get(key) != expected:
            _fail(f"{field}.{key} mismatch")
    prefixes = _mapping(equivalence.get("scene_prefixes"), f"{field}.scene_prefixes")
    if set(prefixes) != set(EXPECTED_TRAIN_SCENES):
        _fail(f"{field} scene set mismatch")
    for scene_id in EXPECTED_TRAIN_SCENES:
        row = _mapping(prefixes[scene_id], f"{field}.{scene_id}")
        if set(row) != {"v18_base_prefix_sha256", "signed_x_adapted_prefix_sha256"}:
            _fail(f"{field}.{scene_id} prefix evidence keys mismatch")
        base = _sha256(row.get("v18_base_prefix_sha256"), f"{field}.{scene_id}.base")
        adapted = _sha256(row.get("signed_x_adapted_prefix_sha256"), f"{field}.{scene_id}.adapted")
        if base != adapted:
            _fail(f"{field}.{scene_id} update-0 local field is not exact identity")
    return equivalence


def _validate_initialization(value: Any, field: str) -> dict[str, Any]:
    initialization = dict(_mapping(value, field))
    required = {
        "schema_version": 4,
        "mode": "frozen_v18_residual_base_plus_zero_output_signed_x_residual",
        "checkpoint": (
            "data_gemma4/checkpoints/gemma4_color_mirror_centered_content_gate_v18/epoch_004"
        ),
        "adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
        "expected_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "expected_metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
        "checkpoint_epoch": EXPECTED_SOURCE_CHECKPOINT_EPOCH,
        "checkpoint_output_namespace": EXPECTED_SOURCE_CHECKPOINT_NAMESPACE,
        "checkpoint_config_hash": EXPECTED_SOURCE_CHECKPOINT_CONFIG_HASH,
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "source_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
        "expected_source_global_scene_residual_state_sha256": (
            EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256
        ),
        "global_scene_residual_frozen": True,
        "signed_x_scene_residual_initial_state_sha256": EXPECTED_INITIAL_SIGNED_X_SHA256,
        "signed_x_scene_residual_zero_output": True,
    }
    for key, expected in required.items():
        if initialization.get(key) != expected:
            _fail(f"{field}.{key} mismatch")
    _validate_source_provenance(
        initialization.get("checkpoint_source_provenance"),
        f"{field}.checkpoint_source_provenance",
    )
    return initialization


def _validate_epoch_artifact(
    epoch: int,
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    path: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    field = f"epoch_{epoch}"
    _assert_finite_tree(artifact, field)
    required = {
        "schema_version": 3,
        "epoch": epoch,
        "global_step": epoch * 12,
        "optimizer_step": epoch,
        "config_hash": contract["config_hash"],
        "output_namespace": OUTPUT_NAMESPACE,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": True,
        "question_dependent_scene_processing": False,
        "global_scene_residual_parameter_count": EXPECTED_GLOBAL_RESIDUAL_PARAMETER_COUNT,
        "global_scene_residual": contract["global_residual"],
        "global_scene_residual_initial_state_sha256": EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256,
        "global_scene_residual_state_sha256": EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
        "frozen_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
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
    for stale in ("v18_stage_execution", "v19_screen"):
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
    threshold = _mapping(screen["continuation_requires"], "continuation_requires")
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


def summarize_v20_epochs(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    epoch_artifacts: Mapping[int, Mapping[str, Any]],
    *,
    update1_report_path: str | Path,
    selection_path: str | Path,
    selection_sha256: str,
    epoch_paths: Mapping[int, str | Path],
    epoch_sha256: Mapping[int, str],
) -> dict[str, Any]:
    """Validate, artifact-bind, rank, and gate four cumulative V20 updates."""

    expected = set(EXPECTED_EPOCHS)
    if set(epoch_artifacts) != expected:
        _fail("V20 selector requires exactly epoch artifacts 1,2,3,4")
    paths = dict(epoch_paths)
    hashes = dict(epoch_sha256)
    if set(paths) != expected or set(hashes) != expected:
        _fail("Epoch paths and hashes must cover exactly epochs 1,2,3,4")
    for epoch in EXPECTED_EPOCHS:
        _sha256(hashes[epoch], f"epoch_{epoch}.artifact_sha256")
    selection_digest = selection_sha256
    _sha256(selection_digest, "selection_artifact_sha256")
    _require_bound_json(
        selection_path,
        selection,
        selection_digest,
        "V20 selection artifact",
    )

    contract = _validate_config(config)
    update1 = _load_update1_authorization(update1_report_path, config=config)
    selection_evidence = _validate_selection(selection, contract)
    validated: list[dict[str, Any]] = []
    for epoch in EXPECTED_EPOCHS:
        _require_bound_json(
            paths[epoch],
            epoch_artifacts[epoch],
            hashes[epoch],
            f"V20 epoch_{epoch} metadata",
        )
        inspection = _inspect_checkpoint_artifacts(
            epoch,
            epoch_artifacts[epoch],
            config=config,
            metadata_path=paths[epoch],
        )
        if inspection["checkpoint_artifact_hashes"]["metadata_sha256"] != hashes[epoch]:
            _fail(f"epoch_{epoch} loaded metadata hash differs from its actual file")
        row = _validate_epoch_artifact(
            epoch,
            epoch_artifacts[epoch],
            contract,
            path=str(paths[epoch]),
            artifact_sha256=hashes[epoch],
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

    epoch1_inspection = validated[0]["checkpoint_inspection"]
    if _resolve(update1["checkpoint"]) != _resolve(epoch1_inspection["checkpoint"]):
        _fail("Update-one authorization checkpoint path differs from epoch one")
    if update1["checkpoint_artifact_hashes"] != epoch1_inspection["checkpoint_artifact_hashes"]:
        _fail("Update-one authorization artifact hashes differ from epoch one")
    epoch1_tensors = epoch1_inspection["tensor_evidence"]
    tensor_bindings = {
        "signed_x_state_sha256": "signed_x_state_sha256",
        "output_projection_sha256": "output_projection_sha256",
        "frozen_global_scene_residual_state_sha256": ("global_scene_residual_state_sha256"),
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
            "checkpoint_artifact_hashes": deepcopy(inspection["checkpoint_artifact_hashes"]),
            "signed_x_state_sha256": row["signed_x_state_sha256"],
            "optimizer_state_sha256": inspection["optimizer_state_sha256"],
            "frozen_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
            "color": deepcopy(row["metrics"]["color"]),
            "mirror": deepcopy(row["metrics"]["mirror"]),
        }
        candidate["color_eligible"] = _color_eligible(candidate)
        candidate["continuation_gate_passed"] = _continuation_passed(candidate, contract["screen"])
        candidate["full_teacher_gate_passed"] = _full_teacher_passed(candidate)
        candidates.append(candidate)
    ranking = sorted(
        (deepcopy(item) for item in candidates if item["color_eligible"]),
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
        "selector_type": "strict_v20_signed_x_local_field_epoch_selector",
        "report_only": True,
        "model_inference_executed": False,
        "gemma_model_loaded": False,
        "checkpoint_tensor_state_loaded": True,
        "checkpoint_tensor_state_safely_inspected": True,
        "optimizer_deserialized": True,
        "optimizer_deserialization_weights_only": True,
        "question_dependent_scene_processing": False,
        "config_path": str(PINNED_CONFIG_PATH),
        "config_hash": contract["config_hash"],
        "update1_authorization": deepcopy(update1),
        "selection_artifact_path": str(selection_path),
        "selection_artifact_sha256": selection_digest,
        "selection_ids_sha256": selection_evidence["selection_sha256"],
        "pair_unit_selection_sha256": selection_evidence["pair_selection_sha256"],
        "pair_membership_sha256": selection_evidence["pair_membership_sha256"],
        "source_provenance": deepcopy(source),
        "initialization_provenance": deepcopy(initialization),
        "selection_policy": deepcopy(contract["screen"]),
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
            "frozen_global_state_sha256": EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
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
            None if selected is None else deepcopy(selected["checkpoint_artifact_hashes"])
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
        parser.error("duplicate V20 epoch binding")
    _reject_forbidden_input_path(args.config)
    config = load_config(args.config)
    selection, selection_digest = _load_json_strict(args.selection)
    loaded = {epoch: _load_json_strict(path) for epoch, path in epoch_paths.items()}
    summary = summarize_v20_epochs(
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
