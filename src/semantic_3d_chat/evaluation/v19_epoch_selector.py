"""Strict report-only selector for the four predeclared V19 screen epochs.

Only the resolved configuration, deterministic training-selection report, and
four checkpoint ``metadata.json`` files are accepted.  The selector does not
load tensor state, maps, questions, oracle data, or a model.  It fails closed
unless the metadata proves an exact four-update signed-X run from the pinned
V18 epoch-4 source while the scene tokenizer, V18 residual, and LoRA banks stay
frozen.

Color preservation is an eligibility filter.  Eligible epochs are ranked by
the six predeclared mirror teacher-forced fields and then by lower epoch.  A
continuation is authorized only at the configured mirror threshold.  Greedy
generation remains forbidden unless the selected epoch passes the complete
teacher-forced gate for both counterfactual pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation.residual_lr_response import (
    COLOR_PAIR_ID,
    EXPECTED_RANKING_FIELDS,
    MIRROR_PAIR_ID,
)

PINNED_CONFIG_PATH = Path("configs/experiments/gemma4_color_mirror_signed_x_moment_v19.yaml")
PINNED_CONFIG_HASH = "26e2e91e5daf"
OUTPUT_NAMESPACE = "gemma4_color_mirror_signed_x_moment_v19"
EXPECTED_EPOCHS = (1, 2, 3, 4)
EXPECTED_TRAIN_SCENES = (
    "scene_000003",
    "scene_000004",
    "scene_000007",
    "scene_000008",
)
EXPECTED_TEST_SCENES = ("scene_000005", "scene_000006")
EXPECTED_GLOBAL_RESIDUAL_PARAMETER_COUNT = 400_128
EXPECTED_SIGNED_X_PARAMETER_COUNT = 196_608
EXPECTED_SOURCE_CHECKPOINT_EPOCH = 4
EXPECTED_SOURCE_CHECKPOINT_CONFIG_HASH = "38b0fd8e679d"
EXPECTED_SOURCE_CHECKPOINT_NAMESPACE = "gemma4_color_mirror_centered_content_gate_v18"
EXPECTED_SOURCE_ADAPTER_SHA256 = "1a7946d2e40aaf4bf66dc570bff19fa8d6ba4425e4e0d59bd52b809bd23dae7a"
EXPECTED_SOURCE_METADATA_SHA256 = "4853355ef4810f284d9b36eca1f0f1ade71319f4f6f579a5b079ce6178eb2344"
EXPECTED_FROZEN_SCENE_SHA256 = "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256 = (
    "ce3bf864eed6dd4a50f1b67296981e144d6a79e9cf192ad9a9230f2ae18208dc"
)
EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256 = (
    "f7f6353edb6216029bd155e2baab1b5051c85f297a0e6d6b63210354fe0ff0e0"
)
EXPECTED_INITIAL_SIGNED_X_SHA256 = (
    "55b7cb21d0ecbe945cabccfacd5b6aa94693743ceee78443f37a5ca0d1ac68b1"
)
EXPECTED_SELECTION_SHA256 = "7f0714e3151c9ddb57c1da95a457820a833e490c070881a88a9fee4a9168f933"
EXPECTED_PAIR_SELECTION_SHA256 = "d5928cb783339ef62fff5c14a8c7f85f90d3a7a6cb8edad0a784998082740d3e"
EXPECTED_PAIR_MEMBERSHIP_SHA256 = "99ee448c23fb71b7269a353a54b2156ac55701847af170597dcc351af15cbcbe"
EXPECTED_FROZEN_BANKS = {
    "extension_v13": "4eb90fb9b0bea579d14cfcb0f61ebd5b6d566fd600bd3d5e1bfe5177a39e1b34",
    "inherited_v12": "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594",
}
EXPECTED_TEACHER_GATE_TYPE = "teacher_forced_same_distribution_candidate_logit_ranking"

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40,64}")
_FORBIDDEN_INPUT_COMPONENTS = frozenset({"oracle", "rendered", "maps", "scene_tokens", "runtime"})


class V19EpochSelectorViolation(ValueError):
    """A fail-closed V19 evidence, provenance, or gate-policy violation."""


def _fail(message: str) -> None:
    raise V19EpochSelectorViolation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_finite_tree(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{field}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(f"{field} contains NaN or infinity")


def _validate_source_provenance(value: Any, field: str) -> dict[str, Any]:
    source = dict(_mapping(value, field))
    expected_keys = {
        "schema_version",
        "scope",
        "available",
        "head_commit",
        "head_tree",
        "is_clean",
        "tracked_diff_sha256",
    }
    if set(source) != expected_keys:
        _fail(
            f"{field} keys mismatch: missing={sorted(expected_keys - set(source))} "
            f"unknown={sorted(set(source) - expected_keys)}"
        )
    if source.get("schema_version") != 1:
        _fail(f"{field}.schema_version must be 1")
    if source.get("scope") != "repository_excluding_generated_artifacts_v1":
        _fail(f"{field}.scope is not the repository training scope")
    if source.get("available") is not True or source.get("is_clean") is not True:
        _fail(f"{field} must attest clean, available source")
    for key in ("head_commit", "head_tree"):
        item = source.get(key)
        if not isinstance(item, str) or _GIT_OBJECT_ID.fullmatch(item) is None:
            _fail(f"{field}.{key} must be a Git object ID")
    if _sha256(source.get("tracked_diff_sha256"), f"{field}.tracked_diff_sha256") != (
        _EMPTY_SHA256
    ):
        _fail(f"{field} records a non-empty tracked diff")
    return source


def _expected_global_residual_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    scene = _mapping(config.get("scene_encoder"), "config.scene_encoder")
    raw = _mapping(scene.get("global_scene_residual"), "global_scene_residual")
    expected = {
        "schema_version": 2,
        "enabled": True,
        "width": 128,
        "fourier_bands": 4,
        "initialization_seed": 18018,
        "expected_initial_state_sha256": EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256,
        "architecture_version": "zero_spatial_mean_content_gate_v1",
        "gate_temperature": 1.0,
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
    }
    configured = {
        "schema_version": 2,
        **dict(raw),
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
    }
    if configured != expected:
        _fail("Resolved V19 frozen global-residual contract differs from the exact pin")
    return expected


def _expected_signed_x_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    scene = _mapping(config.get("scene_encoder"), "config.scene_encoder")
    raw = _mapping(scene.get("signed_x_scene_residual"), "signed_x_scene_residual")
    expected = {
        "schema_version": 1,
        "enabled": True,
        "architecture_version": "signed_x_moment_v1",
        "expected_initial_state_sha256": EXPECTED_INITIAL_SIGNED_X_SHA256,
        "spatial_statistic": "centered_unit_rms_signed_x_moment",
        "spatial_centering": "all_slots_fp32",
        "trainable_surface": "bias_free_output_projection_only",
    }
    configured = {
        "schema_version": 1,
        **dict(raw),
        "spatial_statistic": "centered_unit_rms_signed_x_moment",
        "spatial_centering": "all_slots_fp32",
        "trainable_surface": "bias_free_output_projection_only",
    }
    if configured != expected:
        _fail("Resolved V19 signed-X contract differs from the exact schema-1 pin")
    return expected


def _expected_objective_policy() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema_version": 1,
        "configured": True,
        "allow_unlisted_pair_ids": False,
        "legacy_default": {
            "role": "legacy_global",
            "language_nll_weight": 1.0,
            "candidate_hinge_weight": 8.0,
            "candidate_margin": 1.0,
            "full_vocab_hinge_weight": 2.0,
            "full_vocab_margin": 1.0,
        },
        "by_pair": {
            COLOR_PAIR_ID: {
                "role": "retention_control",
                "language_nll_weight": 0.0,
                "candidate_hinge_weight": 8.0,
                "candidate_margin": 0.25,
                "full_vocab_hinge_weight": 2.0,
                "full_vocab_margin": 0.25,
            },
            MIRROR_PAIR_ID: {
                "role": "signed_target",
                "language_nll_weight": 0.0,
                "candidate_hinge_weight": 8.0,
                "candidate_margin": 1.0,
                "full_vocab_hinge_weight": 2.0,
                "full_vocab_margin": 1.0,
            },
        },
    }
    return {**contract, "contract_sha256": _canonical_sha256(contract)}


def _expected_objective_coverage() -> dict[str, Any]:
    resolved = deepcopy(_expected_objective_policy()["by_pair"])
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "selected_pair_ids": [COLOR_PAIR_ID, MIRROR_PAIR_ID],
        "configured_pair_ids": [COLOR_PAIR_ID, MIRROR_PAIR_ID],
        "unlisted_pair_ids": [],
        "allow_unlisted_pair_ids": False,
        "resolved_by_pair": resolved,
        "complete": True,
    }
    return {**evidence, "coverage_sha256": _canonical_sha256(evidence)}


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    observed_hash = config_hash(dict(config))
    if observed_hash != PINNED_CONFIG_HASH:
        _fail(f"V19 config hash mismatch: expected={PINNED_CONFIG_HASH} observed={observed_hash}")
    if config.get("structural_preflight") is not None or config.get("v18_screen") is not None:
        _fail("V19 must not inherit the completed V18 preflight or screen controller")
    screen = _mapping(config.get("v19_screen"), "config.v19_screen")
    training = _mapping(config.get("training"), "config.training")
    scene = _mapping(config.get("scene_encoder"), "config.scene_encoder")
    language = _mapping(config.get("language"), "config.language")
    experiment = _mapping(config.get("experiment"), "config.experiment")

    expected_screen = {
        "schema_version": 1,
        "role": "signed_x_moment_architecture_screen",
        "source_checkpoint_epoch": 4,
        "screen_optimizer_updates": 4,
        "conditional_max_optimizer_updates": 12,
        "stage_1_optimizer_updates": 1,
        "stage_1_stop_required": True,
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
    if dict(screen) != expected_screen:
        _fail("Resolved v19_screen differs from the exact predeclared selection policy")

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
    training_expected = {
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
        for key, expected in training_expected.items()
        if training.get(key) != expected
    }
    if mismatches:
        _fail(f"Resolved V19 training contract mismatch: {mismatches}")
    expected_raw_policy = {
        "schema_version": 1,
        "allow_unlisted_pair_ids": False,
        "by_pair": deepcopy(_expected_objective_policy()["by_pair"]),
    }
    if training.get("pair_objectives") != expected_raw_policy:
        _fail("Resolved V19 per-pair objective policy mismatch")
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
            _fail(f"config.training.{field} must be disabled for signed-X-only V19")

    for key, expected in {
        "architecture_version": "signal_preserving_resampler_v3",
        "input_voxel_size_m": 0.15,
        "model_dim": 384,
        "global_latents": 256,
    }.items():
        if scene.get(key) != expected:
            _fail(f"config.scene_encoder.{key} mismatch")
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
        "role": "exploratory_reflection_odd_scene_residual_screen_v19",
        "question_dependent_scene_processing": False,
        "residual_parameter_count": EXPECTED_GLOBAL_RESIDUAL_PARAMETER_COUNT,
        "signed_x_residual_parameter_count": EXPECTED_SIGNED_X_PARAMETER_COUNT,
        "screen_optimizer_updates": 4,
        "conditional_max_optimizer_updates": 12,
        "source_checkpoint_epoch": 4,
        "source_scene_state_sha256": EXPECTED_FROZEN_SCENE_SHA256,
        "source_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
        "source_inherited_bank_sha256": EXPECTED_FROZEN_BANKS["inherited_v12"],
        "source_extension_bank_sha256": EXPECTED_FROZEN_BANKS["extension_v13"],
        # These inherited V16 policy fields remain in the resolved mapping.
        # They are not authoritative for V19, but must agree with v19_screen
        # so inheritance cannot carry a second, weaker selection policy.
        "screen_extension_requires": {
            **expected_screen["eligibility_requires"],
            **expected_screen["continuation_requires"],
        },
        "full_teacher_gate_requires": expected_screen["full_teacher_gate_requires"],
        "greedy_audit_only_after_full_teacher_gate": True,
    }
    if dict(experiment) != expected_experiment:
        _fail("Resolved V19 experiment provenance contract mismatch")

    global_contract = _expected_global_residual_contract(config)
    signed_contract = _expected_signed_x_contract(config)
    return {
        "config_hash": observed_hash,
        "screen": deepcopy(expected_screen),
        "global_residual": global_contract,
        "signed_x_residual": signed_contract,
        "objective_policy": _expected_objective_policy(),
        "objective_coverage": _expected_objective_coverage(),
        "language": expected_language,
        "scene_encoder_architecture_version": "signal_preserving_resampler_v3",
    }


def _validate_lora_contract(value: Any, field: str) -> None:
    lora = _mapping(value, field)
    if lora.get("schema_version") != 2 or lora.get("enabled") is not True:
        _fail(f"{field} must contain the enabled named-bank schema")
    banks = _sequence(lora.get("banks"), f"{field}.banks")
    by_name: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(banks):
        bank = _mapping(raw, f"{field}.banks[{index}]")
        name = bank.get("name")
        if not isinstance(name, str) or name in by_name:
            _fail(f"{field} contains an invalid or duplicate bank name")
        by_name[name] = bank
    if set(by_name) != set(EXPECTED_FROZEN_BANKS):
        _fail(f"{field} bank names differ from the exact frozen banks")
    for name, expected_hash in EXPECTED_FROZEN_BANKS.items():
        bank = by_name[name]
        if bank.get("trainable") is not False:
            _fail(f"{field}.{name} is not frozen")
        if bank.get("initialization_algorithm") != "checkpoint_overwrite":
            _fail(f"{field}.{name} initialization algorithm mismatch")
        if bank.get("expected_initial_state_sha256") != expected_hash:
            _fail(f"{field}.{name} state hash mismatch")


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
    _validate_lora_contract(selection.get("lora"), "selection.lora")
    return {
        "source_provenance": source,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "pair_selection_sha256": EXPECTED_PAIR_SELECTION_SHA256,
        "pair_membership_sha256": EXPECTED_PAIR_MEMBERSHIP_SHA256,
    }


def _count_from_accuracy(value: Any, total: int, field: str) -> int:
    accuracy = _finite(value, field)
    if not 0.0 <= accuracy <= 1.0:
        _fail(f"{field} must be in [0,1]")
    raw = accuracy * total
    count = round(raw)
    if not math.isclose(raw, count, rel_tol=0.0, abs_tol=1.0e-5):
        _fail(f"{field} does not encode an integer count over {total}")
    return int(count)


def _pair_metrics(value: Any, pair_id: str) -> dict[str, float | int]:
    pair = _mapping(value, pair_id)
    expected = {
        "evaluation_type": EXPECTED_TEACHER_GATE_TYPE,
        "ranking_mode": "candidate_logit",
        "same_next_token_distribution": True,
        "shared_candidate_tokens_excluded": True,
        "free_generation_evaluated": False,
        "first_answer_token_full_vocab_evaluated": True,
    }
    for key, expected_value in expected.items():
        if pair.get(key) != expected_value:
            _fail(f"{pair_id}.{key} mismatch")
    units = _positive_int(pair.get("unit_count"), f"{pair_id}.unit_count")
    sides = _positive_int(pair.get("side_count"), f"{pair_id}.side_count")
    if units != 6 or sides != 12:
        _fail(f"{pair_id} must contain exactly 6 units and 12 sides")
    result: dict[str, float | int] = {
        "full_vocab_units": _count_from_accuracy(
            pair.get("first_answer_token_top1_unit_accuracy"),
            units,
            f"{pair_id}.first_answer_token_top1_unit_accuracy",
        ),
        "full_vocab_sides": _count_from_accuracy(
            pair.get("first_answer_token_top1_accuracy"),
            sides,
            f"{pair_id}.first_answer_token_top1_accuracy",
        ),
        "candidate_units": _count_from_accuracy(
            pair.get("changed_unit_accuracy"), units, f"{pair_id}.changed_unit_accuracy"
        ),
        "candidate_sides": _count_from_accuracy(
            pair.get("side_accuracy"), sides, f"{pair_id}.side_accuracy"
        ),
        "mean_full_vocab_margin": _finite(
            pair.get("mean_first_answer_token_target_vs_best_other_logit_margin"),
            f"{pair_id}.mean_first_answer_token_target_vs_best_other_logit_margin",
        ),
        "minimum_full_vocab_margin": _finite(
            pair.get("minimum_first_answer_token_target_vs_best_other_logit_margin"),
            f"{pair_id}.minimum_first_answer_token_target_vs_best_other_logit_margin",
        ),
        "mean_candidate_margin": _finite(
            pair.get("mean_own_vs_alternate_candidate_logit_margin"),
            f"{pair_id}.mean_own_vs_alternate_candidate_logit_margin",
        ),
        "minimum_candidate_margin": _finite(
            pair.get("minimum_own_vs_alternate_candidate_logit_margin"),
            f"{pair_id}.minimum_own_vs_alternate_candidate_logit_margin",
        ),
    }
    for prefix in ("full_vocab", "candidate"):
        if int(result[f"{prefix}_units"]) > int(result[f"{prefix}_sides"]) // 2:
            _fail(f"{pair_id} reports more correct units than its side count permits")
        if float(result[f"minimum_{prefix}_margin"]) > float(result[f"mean_{prefix}_margin"]):
            _fail(f"{pair_id} minimum {prefix} margin exceeds its mean")
        if float(result[f"minimum_{prefix}_margin"]) > 0.0 and (
            int(result[f"{prefix}_sides"]) != sides or int(result[f"{prefix}_units"]) != units
        ):
            _fail(f"{pair_id} positive minimum {prefix} margin contradicts its accuracy")
    return result


def _extract_epoch_metrics(history_item: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    if history_item.get("epoch") != epoch:
        _fail(f"Epoch {epoch} history item has the wrong epoch number")
    if history_item.get("pair_batch_count") != 12:
        _fail(f"Epoch {epoch} history does not attest exactly 12 ordered microsteps")
    if history_item.get("pair_batch_fraction") != 1.0:
        _fail(f"Epoch {epoch} history is not an all-pair update")
    gate = _mapping(history_item.get("pair_candidate_gate"), f"epoch_{epoch}.pair_gate")
    gate_expected = {
        "evaluation_type": EXPECTED_TEACHER_GATE_TYPE,
        "ranking_mode": "candidate_logit",
        "same_next_token_distribution": True,
        "shared_candidate_tokens_excluded": True,
        "free_generation_evaluated": False,
        "first_answer_token_full_vocab_evaluated": True,
        "pair_count": 2,
        "unit_count": 12,
        "side_count": 24,
    }
    for key, expected in gate_expected.items():
        if gate.get(key) != expected:
            _fail(f"epoch_{epoch}.pair_gate.{key} mismatch")
    by_pair = _mapping(gate.get("by_pair"), f"epoch_{epoch}.pair_gate.by_pair")
    if set(by_pair) != {COLOR_PAIR_ID, MIRROR_PAIR_ID}:
        _fail(f"Epoch {epoch} has an unexpected teacher-forced pair set")
    return {
        "epoch": epoch,
        "color": _pair_metrics(by_pair[COLOR_PAIR_ID], COLOR_PAIR_ID),
        "mirror": _pair_metrics(by_pair[MIRROR_PAIR_ID], MIRROR_PAIR_ID),
    }


def _validate_signed_x_equivalence(value: Any, field: str) -> dict[str, Any]:
    equivalence = dict(_mapping(value, field))
    required = {
        "verified": True,
        "base": "loaded_frozen_global_scene_residual",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
        "scene_count": 4,
    }
    for key, expected in required.items():
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
            _fail(f"{field}.{scene_id} update-0 signed-X prefix is not exact identity")
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
    if artifact.get("schema_version") != 3:
        _fail(f"{field}.schema_version must equal 3")
    required = {
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
        "global_scene_residual_initial_state_sha256": (EXPECTED_INITIAL_GLOBAL_RESIDUAL_SHA256),
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
    if "v18_stage_execution" in artifact:
        _fail(f"{field} improperly carries a completed V18 stage controller")
    signed_state = _sha256(
        artifact.get("signed_x_scene_residual_state_sha256"),
        f"{field}.signed_x_scene_residual_state_sha256",
    )
    if signed_state == EXPECTED_INITIAL_SIGNED_X_SHA256:
        _fail(f"{field} signed-X state did not change after its optimizer update")
    source = _validate_source_provenance(artifact.get("source_provenance"), f"{field}.source")
    initialization = _validate_initialization(
        artifact.get("initialization_provenance"), f"{field}.initialization"
    )
    _validate_lora_contract(artifact.get("lora"), f"{field}.lora")
    equivalence = _validate_signed_x_equivalence(
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
    if _finite(artifact.get("train_loss"), f"{field}.train_loss") != _finite(
        last.get("train_loss"), f"{field}.history[-1].train_loss"
    ):
        _fail(f"{field} top-level train loss is not its final cumulative-history loss")
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


def summarize_v19_epochs(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    epoch_artifacts: Mapping[int, Mapping[str, Any]],
    *,
    selection_path: str = "<selection>",
    selection_sha256: str | None = None,
    epoch_paths: Mapping[int, str] | None = None,
    epoch_sha256: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Validate, rank, and gate exactly four V19 teacher-forced epochs."""

    expected_epochs = set(EXPECTED_EPOCHS)
    if set(epoch_artifacts) != expected_epochs:
        _fail(
            "V19 selector requires exactly epoch artifacts 1,2,3,4: "
            f"observed={sorted(epoch_artifacts)}"
        )
    _assert_finite_tree(selection, "selection")
    for epoch in EXPECTED_EPOCHS:
        _assert_finite_tree(epoch_artifacts[epoch], f"epoch_{epoch}")
    paths = dict(epoch_paths or {epoch: f"<epoch_{epoch}>" for epoch in EXPECTED_EPOCHS})
    hashes = dict(
        epoch_sha256
        or {epoch: _canonical_sha256(epoch_artifacts[epoch]) for epoch in EXPECTED_EPOCHS}
    )
    if set(paths) != expected_epochs or set(hashes) != expected_epochs:
        _fail("Epoch paths and hashes must cover exactly epochs 1,2,3,4")
    for epoch in EXPECTED_EPOCHS:
        _sha256(hashes[epoch], f"epoch_{epoch}.artifact_sha256")
    selection_digest = selection_sha256 or _canonical_sha256(selection)
    _sha256(selection_digest, "selection_artifact_sha256")

    contract = _validate_config(config)
    selection_evidence = _validate_selection(selection, contract)
    validated = [
        _validate_epoch_artifact(
            epoch,
            epoch_artifacts[epoch],
            contract,
            path=paths[epoch],
            artifact_sha256=hashes[epoch],
        )
        for epoch in EXPECTED_EPOCHS
    ]
    source = validated[0]["source_provenance"]
    initialization = validated[0]["initialization_provenance"]
    equivalence = validated[0]["zero_output_equivalence"]
    if selection_evidence["source_provenance"] != source:
        _fail("Selection and epoch artifacts do not share exact clean source provenance")
    for row in validated[1:]:
        if row["source_provenance"] != source:
            _fail("Epoch artifacts do not share exact clean source provenance")
        if row["initialization_provenance"] != initialization:
            _fail("Epoch artifacts do not share exact V18 initialization provenance")
        if row["zero_output_equivalence"] != equivalence:
            _fail("Epoch artifacts do not preserve exact update-0 signed-X equivalence")
    for earlier, later in pairwise(validated):
        if later["history"][: earlier["epoch"]] != earlier["history"]:
            _fail(
                f"Epoch {later['epoch']} does not preserve exact cumulative history from "
                f"epoch {earlier['epoch']}"
            )
    signed_states = [row["signed_x_state_sha256"] for row in validated]
    if len(set(signed_states)) != len(signed_states):
        _fail("Signed-X state history repeats or rolls back across optimizer updates")

    candidates: list[dict[str, Any]] = []
    for row in validated:
        candidate = {
            "epoch": row["epoch"],
            "optimizer_step": row["epoch"],
            "cumulative_microsteps": row["epoch"] * 12,
            "checkpoint_metadata_path": row["path"],
            "checkpoint_metadata_sha256": row["artifact_sha256"],
            "signed_x_state_sha256": row["signed_x_state_sha256"],
            "frozen_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
            "color": deepcopy(row["metrics"]["color"]),
            "mirror": deepcopy(row["metrics"]["mirror"]),
        }
        candidate["color_eligible"] = _color_eligible(candidate)
        candidate["continuation_gate_passed"] = _continuation_passed(candidate, contract["screen"])
        candidate["full_teacher_gate_passed"] = _full_teacher_passed(candidate)
        candidates.append(candidate)
    ranking = sorted(
        (deepcopy(candidate) for candidate in candidates if candidate["color_eligible"]),
        key=_ranking_key,
    )
    for rank, candidate in enumerate(ranking, start=1):
        candidate["rank"] = rank
    selected = None if not ranking else ranking[0]
    continuation_authorized = bool(selected is not None and selected["continuation_gate_passed"])
    full_teacher_gate_passed = bool(selected is not None and selected["full_teacher_gate_passed"])
    greedy_authorized = bool(
        full_teacher_gate_passed
        and contract["screen"]["greedy_audit_only_after_full_teacher_gate"] is True
    )
    if greedy_authorized and not full_teacher_gate_passed:  # pragma: no cover
        raise AssertionError("Greedy audit cannot be authorized without the full teacher gate")

    return {
        "schema_version": 1,
        "selector_type": "strict_v19_signed_x_moment_epoch_selector",
        "report_only": True,
        "model_inference_executed": False,
        "checkpoint_tensor_state_loaded": False,
        "question_dependent_scene_processing": False,
        "config_path": str(PINNED_CONFIG_PATH),
        "config_hash": contract["config_hash"],
        "selection_artifact_path": selection_path,
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
            "frozen_global_state_sha256": EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
        },
        "eligible_epoch_count": len(ranking),
        "ranking": ranking,
        "selected_epoch": None if selected is None else selected["epoch"],
        "selected_checkpoint_metadata_path": (
            None if selected is None else selected["checkpoint_metadata_path"]
        ),
        "selected_checkpoint_metadata_sha256": (
            None if selected is None else selected["checkpoint_metadata_sha256"]
        ),
        "selected_signed_x_state_sha256": (
            None if selected is None else selected["signed_x_state_sha256"]
        ),
        "continuation_gate_passed": continuation_authorized,
        "continuation_authorized": continuation_authorized,
        "conditional_max_optimizer_updates": contract["screen"][
            "conditional_max_optimizer_updates"
        ],
        "full_teacher_gate_passed": full_teacher_gate_passed,
        "greedy_audit_authorized": greedy_authorized,
        "greedy_audit_forbidden": not greedy_authorized,
        "decision": (
            "no_color_eligible_epoch_no_extension_no_greedy"
            if selected is None
            else "full_teacher_gate_passed_greedy_audit_allowed"
            if greedy_authorized
            else "continue_selected_epoch_no_greedy_audit"
            if continuation_authorized
            else "screen_failed_no_extension_no_greedy_audit"
        ),
    }


def _reject_forbidden_input_path(path: Path) -> None:
    forbidden = sorted(
        component
        for component in (part.casefold() for part in path.resolve().parts)
        if component in _FORBIDDEN_INPUT_COMPONENTS
    )
    if forbidden:
        _fail(f"Selector refuses runtime/oracle artifact path components: {forbidden}")


def _load_json_strict(path: Path) -> tuple[Mapping[str, Any], str]:
    _reject_forbidden_input_path(path)
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        _fail(f"JSON constant {value} is forbidden")

    value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    return _mapping(value, str(path)), hashlib.sha256(raw).hexdigest()


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
    parser.add_argument("--epoch", action="append", type=_parse_epoch_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    epoch_paths = dict(args.epoch)
    if len(epoch_paths) != len(args.epoch):
        parser.error("duplicate V19 epoch binding")
    _reject_forbidden_input_path(args.config)
    config = load_config(args.config)
    selection, selection_digest = _load_json_strict(args.selection)
    loaded = {epoch: _load_json_strict(path) for epoch, path in epoch_paths.items()}
    summary = summarize_v19_epochs(
        config,
        selection,
        {epoch: value for epoch, (value, _digest) in loaded.items()},
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


if __name__ == "__main__":  # pragma: no cover - exercised through the public CLI
    raise SystemExit(main())
