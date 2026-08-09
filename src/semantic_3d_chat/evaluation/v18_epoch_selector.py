"""Strict report-only selector for the four predeclared V18 screen epochs.

The selector consumes the resolved V18 YAML configuration, the deterministic
training-selection audit, and exactly four checkpoint ``metadata.json``
artifacts.  It never imports a model, opens checkpoint tensor state, reads QA or
oracle data, or runs inference.  Checkpoint metadata must attest the exact V18
architecture, frozen V14 source state, clean source provenance, curriculum,
and cumulative teacher-forced control history before any epoch is eligible.

Color preservation is an eligibility filter.  Eligible epochs are ranked by
the six predeclared mirror fields and then by lower epoch.  Continuation and
full-teacher gates are reported separately; greedy generation is forbidden
unless the selected epoch passes the complete teacher-forced gate.
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

PINNED_CONFIG_PATH = Path("configs/experiments/gemma4_color_mirror_centered_content_gate_v18.yaml")
PINNED_CONFIG_HASH = "38b0fd8e679d"
OUTPUT_NAMESPACE = "gemma4_color_mirror_centered_content_gate_v18"
EXPECTED_EPOCHS = (1, 2, 3, 4)
EXPECTED_TRAIN_SCENES = (
    "scene_000003",
    "scene_000004",
    "scene_000007",
    "scene_000008",
)
EXPECTED_TEST_SCENES = ("scene_000005", "scene_000006")
EXPECTED_RESIDUAL_ARCHITECTURE = "zero_spatial_mean_content_gate_v1"
EXPECTED_RESIDUAL_PARAMETER_COUNT = 400_128
EXPECTED_SOURCE_CHECKPOINT_CONFIG_HASH = "93ff12019b76"
EXPECTED_SOURCE_CHECKPOINT_NAMESPACE = "gemma4_color_mirror_decoder_banks_v14_lr2e3"
EXPECTED_TEACHER_GATE_TYPE = "teacher_forced_same_distribution_candidate_logit_ranking"
EXPECTED_STAGE_EXECUTION = {
    "stage_1_exact_v14_restart_updates": 1,
    "stage_1_stop_required": True,
    "stage_2_resume_from_epoch": 1,
    "stage_2_load_optimizer_state": True,
    "stage_2_load_history": True,
    "stage_2_target_total_optimizer_updates": 4,
}

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40,64}")
_FORBIDDEN_INPUT_COMPONENTS = frozenset({"oracle", "rendered", "maps", "scene_tokens", "runtime"})


class V18EpochSelectorViolation(ValueError):
    """A fail-closed V18 epoch evidence or policy violation."""


def _fail(message: str) -> None:
    raise V18EpochSelectorViolation(message)


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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
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


def _expected_residual_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    residual = _mapping(
        _mapping(config.get("scene_encoder"), "config.scene_encoder").get("global_scene_residual"),
        "config.scene_encoder.global_scene_residual",
    )
    expected = {
        "schema_version": 2,
        "enabled": True,
        "width": 128,
        "fourier_bands": 4,
        "initialization_seed": 18018,
        "expected_initial_state_sha256": (
            "f7f6353edb6216029bd155e2baab1b5051c85f297a0e6d6b63210354fe0ff0e0"
        ),
        "architecture_version": EXPECTED_RESIDUAL_ARCHITECTURE,
        "gate_temperature": 1.0,
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
    }
    configured = {
        "schema_version": 2,
        **dict(residual),
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
    }
    if configured != expected:
        _fail("Resolved V18 residual architecture differs from the pinned schema-2 contract")
    return expected


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    observed_hash = config_hash(dict(config))
    if observed_hash != PINNED_CONFIG_HASH:
        _fail(f"V18 config hash mismatch: expected={PINNED_CONFIG_HASH} observed={observed_hash}")
    preflight = _mapping(config.get("structural_preflight"), "config.structural_preflight")
    hashes = _mapping(preflight.get("expected_hashes"), "structural_preflight.expected_hashes")
    screen = _mapping(config.get("v18_screen"), "config.v18_screen")
    training = _mapping(config.get("training"), "config.training")
    scene = _mapping(config.get("scene_encoder"), "config.scene_encoder")
    language = _mapping(config.get("language"), "config.language")
    experiment = _mapping(config.get("experiment"), "config.experiment")

    expected_screen = {
        "schema_version": 1,
        "role": "v18_slot_centered_residual_screen",
        "learning_rate": 1.0e-3,
        "screen_optimizer_updates": 4,
        "conditional_max_optimizer_updates": 12,
        "epoch_tiebreaker": "lower_epoch",
        "execution_stages": {
            "stage_1_exact_v14_restart_updates": 1,
            "stage_1_stop_required": True,
            "predicted_preflight_state_must_match_epoch_001": True,
            "stage_2_resume_from_epoch": 1,
            "stage_2_load_optimizer_state": True,
            "stage_2_load_history": True,
            "stage_2_target_total_optimizer_updates": 4,
        },
        "eligibility_requires": {
            "color_full_vocab_sides": 12,
            "color_full_vocab_units": 6,
            "color_positive_minimum_candidate_margin": True,
            "color_positive_minimum_full_vocab_margin": True,
        },
        "ranking_descending": list(EXPECTED_RANKING_FIELDS),
        "continuation_requires": {
            "color_full_vocab_sides": 12,
            "color_full_vocab_units": 6,
            "color_positive_minimum_candidate_margin": True,
            "color_positive_minimum_full_vocab_margin": True,
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
        _fail("Resolved v18_screen differs from the exact predeclared selection policy")

    expected_optimizer = dict(_mapping(preflight.get("optimizer"), "preflight.optimizer"))
    if training.get("optimizer") != expected_optimizer:
        _fail("Training and structural-preflight AdamW contracts differ")
    training_expected = {
        "output_namespace": OUTPUT_NAMESPACE,
        "initialize_from": (
            "data_gemma4/checkpoints/gemma4_color_mirror_decoder_banks_v14_lr2e3/epoch_007"
        ),
        "initialize_expected_adapter_sha256": hashes.get("source_adapter_sha256"),
        "initialize_expected_metadata_sha256": hashes.get("source_metadata_sha256"),
        "initialize_legacy_lora_into_bank": None,
        "initialize_named_lora_freeze_transition": True,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": True,
        "epochs": 4,
        "pair_steps_per_epoch": 12,
        "gradient_accumulation": 12,
        "pair_gate_every_epochs": 1,
        "pair_gate_stop_when_passed": False,
        "early_stopping_patience": 0,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "pair_only_mode": True,
        "pair_only_scene_ids": list(EXPECTED_TRAIN_SCENES),
        "pair_max_units_per_pair": 6,
        "pair_batch_fraction": 1.0,
        "pair_units_per_batch": 1,
        "pair_ranking_mode": "candidate_logit",
        "pair_full_vocab_ranking_weight": 2.0,
        "pair_full_vocab_ranking_margin": 1.0,
    }
    mismatches = {
        key: {"expected": value, "observed": training.get(key)}
        for key, value in training_expected.items()
        if training.get(key) != value
    }
    if mismatches:
        _fail(f"Resolved V18 training contract mismatch: {mismatches}")
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
        if training.get(field) != 0 and training.get(field) != 0.0:
            _fail(f"config.training.{field} must be disabled for residual-only V18")

    scene_expected = {
        "architecture_version": "signal_preserving_resampler_v3",
        "input_voxel_size_m": 0.15,
        "model_dim": 384,
        "global_latents": 256,
    }
    for key, expected in scene_expected.items():
        if scene.get(key) != expected:
            _fail(f"config.scene_encoder.{key} mismatch")
    language_expected = {
        "model_id": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "backend": "gemma4",
        "scene_prefix_after_bos": True,
        "scene_boundary_mode": "gemma4_native_image",
    }
    for key, expected in language_expected.items():
        if language.get(key) != expected:
            _fail(f"config.language.{key} mismatch")
    experiment_expected = {
        "schema_version": 1,
        "role": "exploratory_slot_centered_residual_architecture_screen_v18",
        "question_dependent_scene_processing": False,
        "residual_parameter_count": EXPECTED_RESIDUAL_PARAMETER_COUNT,
        "screen_optimizer_updates": 4,
        "conditional_max_optimizer_updates": 12,
        "source_checkpoint_epoch": 7,
        "source_scene_state_sha256": hashes.get("frozen_scene_state_sha256"),
        "source_inherited_bank_sha256": _mapping(
            hashes.get("frozen_lora_bank_state_sha256"), "expected frozen banks"
        ).get("inherited_v12"),
        "source_extension_bank_sha256": _mapping(
            hashes.get("frozen_lora_bank_state_sha256"), "expected frozen banks"
        ).get("extension_v13"),
        # These inherited V16 fields remain in the resolved mapping.  V18's
        # dedicated screen section is authoritative, but they must agree with
        # it exactly so inheritance cannot smuggle in a second gate policy.
        "screen_extension_requires": expected_screen["continuation_requires"],
        "full_teacher_gate_requires": expected_screen["full_teacher_gate_requires"],
        "greedy_audit_only_after_full_teacher_gate": True,
    }
    if dict(experiment) != experiment_expected:
        _fail("Resolved V18 experiment provenance contract mismatch")

    scalar_hash_fields = (
        "source_adapter_sha256",
        "source_metadata_sha256",
        "frozen_scene_state_sha256",
        "initial_residual_state_sha256",
        "position_features_sha256",
        "selection_sha256",
        "pair_membership_sha256",
    )
    normalized_hashes = {
        field: _sha256(hashes.get(field), f"expected_hashes.{field}")
        for field in scalar_hash_fields
    }
    frozen_banks = _mapping(
        hashes.get("frozen_lora_bank_state_sha256"), "expected_hashes.frozen_lora_banks"
    )
    if set(frozen_banks) != {"inherited_v12", "extension_v13"}:
        _fail("Expected frozen LoRA bank names mismatch")
    normalized_hashes["frozen_lora_bank_state_sha256"] = {
        name: _sha256(value, f"expected frozen bank {name}")
        for name, value in sorted(frozen_banks.items())
    }
    prefixes = _mapping(hashes.get("core_prefix_sha256"), "expected core prefixes")
    if set(prefixes) != set(EXPECTED_TRAIN_SCENES):
        _fail("Expected update-0 core-prefix scene set mismatch")
    normalized_hashes["core_prefix_sha256"] = {
        scene_id: _sha256(prefixes[scene_id], f"expected core prefix {scene_id}")
        for scene_id in EXPECTED_TRAIN_SCENES
    }
    residual_contract = _expected_residual_contract(config)
    if (
        residual_contract["expected_initial_state_sha256"]
        != normalized_hashes["initial_residual_state_sha256"]
    ):
        _fail("Residual initial hash disagrees with the preflight hash contract")
    return {
        "config_hash": observed_hash,
        "screen": deepcopy(expected_screen),
        "hashes": normalized_hashes,
        "residual_contract": residual_contract,
        "scene_encoder_architecture_version": scene_expected["architecture_version"],
        "language": language_expected,
    }


def _validate_lora_contract(value: Any, expected_banks: Mapping[str, str], field: str) -> None:
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
    if set(by_name) != set(expected_banks):
        _fail(f"{field} bank names differ from the pinned frozen banks")
    for name, expected_hash in expected_banks.items():
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
    hashes = _mapping(contract["hashes"], "contract.hashes")
    if selection.get("schema_version") != 1:
        _fail("selection.schema_version must equal 1")
    source = _validate_source_provenance(selection.get("source_provenance"), "selection.source")
    train = _mapping(selection.get("train"), "selection.train")
    required_train = {
        "available_count": 24,
        "selected_count": 24,
        "selected_ids_sha256": hashes["selection_sha256"],
        "expected_change_units_selected": 12,
        "expected_change_units_complete": 12,
        "expected_change_units_incomplete": 0,
    }
    for key, expected in required_train.items():
        if train.get(key) != expected:
            _fail(f"selection.train.{key} mismatch")
    required = {
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": True,
        "gradient_accumulation": 12,
        "initialize_expected_adapter_sha256": hashes["source_adapter_sha256"],
        "initialize_expected_metadata_sha256": hashes["source_metadata_sha256"],
        "initialize_legacy_lora_into_bank": None,
        "initialize_named_lora_freeze_transition": True,
        "train_scene_ids": list(EXPECTED_TRAIN_SCENES),
        "validation_scene_ids": [],
        "test_scene_ids": list(EXPECTED_TEST_SCENES),
        "counterfactual_pair_unit_count": 12,
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": hashes["pair_membership_sha256"],
        "global_scene_residual": contract["residual_contract"],
    }
    for key, expected in required.items():
        if selection.get(key) != expected:
            _fail(f"selection.{key} mismatch")
    _validate_lora_contract(
        selection.get("lora"), hashes["frozen_lora_bank_state_sha256"], "selection.lora"
    )
    return {
        "source_provenance": source,
        "selection_sha256": hashes["selection_sha256"],
        "pair_membership_sha256": hashes["pair_membership_sha256"],
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
    return {
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
            pair.get("changed_unit_accuracy"),
            units,
            f"{pair_id}.changed_unit_accuracy",
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


def _extract_epoch_metrics(history_item: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    if history_item.get("epoch") != epoch:
        _fail(f"Epoch {epoch} history item has the wrong epoch number")
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


def _validate_zero_equivalence(value: Any, contract: Mapping[str, Any], field: str) -> None:
    equivalence = _mapping(value, field)
    if equivalence.get("verified") is not True:
        _fail(f"{field} is not verified")
    if equivalence.get("question_dependent_scene_processing") is not False:
        _fail(f"{field} is question-dependent")
    if equivalence.get("scene_count") != 4:
        _fail(f"{field}.scene_count mismatch")
    expected_prefixes = _mapping(contract["hashes"], "contract.hashes")["core_prefix_sha256"]
    prefixes = _mapping(equivalence.get("scene_prefixes"), f"{field}.scene_prefixes")
    if set(prefixes) != set(expected_prefixes):
        _fail(f"{field} scene set mismatch")
    for scene_id, expected_hash in expected_prefixes.items():
        row = _mapping(prefixes[scene_id], f"{field}.{scene_id}")
        if row.get("core_prefix_sha256") != expected_hash:
            _fail(f"{field}.{scene_id} core-prefix hash mismatch")
        if row.get("adapted_prefix_sha256") != expected_hash:
            _fail(f"{field}.{scene_id} update-0 prefix is not exact identity")


def _validate_initialization(value: Any, contract: Mapping[str, Any], field: str) -> dict[str, Any]:
    initialization = dict(_mapping(value, field))
    hashes = _mapping(contract["hashes"], "contract.hashes")
    required = {
        "schema_version": 3,
        "mode": "named_lora_banks_frozen_plus_zero_output_scene_residual",
        "checkpoint": (
            "data_gemma4/checkpoints/gemma4_color_mirror_decoder_banks_v14_lr2e3/epoch_007"
        ),
        "adapter_sha256": hashes["source_adapter_sha256"],
        "metadata_sha256": hashes["source_metadata_sha256"],
        "expected_adapter_sha256": hashes["source_adapter_sha256"],
        "expected_metadata_sha256": hashes["source_metadata_sha256"],
        "checkpoint_epoch": 7,
        "checkpoint_output_namespace": EXPECTED_SOURCE_CHECKPOINT_NAMESPACE,
        "checkpoint_config_hash": EXPECTED_SOURCE_CHECKPOINT_CONFIG_HASH,
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "source_lora_bank_state_sha256": hashes["frozen_lora_bank_state_sha256"],
        "all_source_lora_banks_frozen": True,
        "global_scene_residual_initial_state_sha256": hashes["initial_residual_state_sha256"],
        "global_scene_residual_zero_output": True,
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
    hashes = _mapping(contract["hashes"], "contract.hashes")
    if artifact.get("schema_version") != 3:
        _fail(f"{field}.schema_version must equal 3")
    required = {
        "epoch": epoch,
        "global_step": epoch * 12,
        "optimizer_step": epoch,
        "config_hash": contract["config_hash"],
        "output_namespace": OUTPUT_NAMESPACE,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": True,
        "question_dependent_scene_processing": False,
        "global_scene_residual_parameter_count": EXPECTED_RESIDUAL_PARAMETER_COUNT,
        "global_scene_residual": contract["residual_contract"],
        "global_scene_residual_initial_state_sha256": hashes["initial_residual_state_sha256"],
        "frozen_scene_state_sha256": hashes["frozen_scene_state_sha256"],
        "frozen_lora_bank_state_sha256": hashes["frozen_lora_bank_state_sha256"],
        "lora_bank_state_sha256": hashes["frozen_lora_bank_state_sha256"],
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
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": hashes["pair_membership_sha256"],
        "initialize_expected_adapter_sha256": hashes["source_adapter_sha256"],
        "initialize_expected_metadata_sha256": hashes["source_metadata_sha256"],
        "v18_stage_execution": EXPECTED_STAGE_EXECUTION,
    }
    for key, expected in required.items():
        if artifact.get(key) != expected:
            _fail(f"{field}.{key} mismatch")
    final_residual_hash = _sha256(
        artifact.get("global_scene_residual_state_sha256"),
        f"{field}.global_scene_residual_state_sha256",
    )
    if final_residual_hash == hashes["initial_residual_state_sha256"]:
        _fail(f"{field} residual state did not change after its optimizer step")
    source = _validate_source_provenance(artifact.get("source_provenance"), f"{field}.source")
    initialization = _validate_initialization(
        artifact.get("initialization_provenance"), contract, f"{field}.initialization"
    )
    _validate_lora_contract(
        artifact.get("lora"), hashes["frozen_lora_bank_state_sha256"], f"{field}.lora"
    )
    _validate_zero_equivalence(
        artifact.get("global_scene_residual_zero_output_equivalence"),
        contract,
        f"{field}.zero_output_equivalence",
    )
    curriculum = _mapping(artifact.get("pair_curriculum"), f"{field}.pair_curriculum")
    curriculum_expected = {
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
    }
    if dict(curriculum) != curriculum_expected:
        _fail(f"{field}.pair_curriculum mismatch")
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
    return {
        "epoch": epoch,
        "path": path,
        "artifact_sha256": artifact_sha256,
        "config_hash": contract["config_hash"],
        "source_provenance": source,
        "initialization_provenance": initialization,
        "residual_state_sha256": final_residual_hash,
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


def _continuation_passed(candidate: Mapping[str, Any]) -> bool:
    mirror = _mapping(candidate["mirror"], "candidate.mirror")
    return bool(
        _color_eligible(candidate)
        and mirror["full_vocab_sides"] >= 8
        and mirror["full_vocab_units"] >= 2
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


def summarize_v18_epochs(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    epoch_artifacts: Mapping[int, Mapping[str, Any]],
    *,
    selection_path: str = "<selection>",
    selection_sha256: str | None = None,
    epoch_paths: Mapping[int, str] | None = None,
    epoch_sha256: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Validate, rank, and gate exactly four V18 teacher-forced epochs."""

    expected_epochs = set(EXPECTED_EPOCHS)
    if set(epoch_artifacts) != expected_epochs:
        _fail(
            "V18 selector requires exactly epoch artifacts 1,2,3,4: "
            f"observed={sorted(epoch_artifacts)}"
        )
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
    if selection_evidence["source_provenance"] != source:
        _fail("Selection and epoch artifacts do not share exact clean source provenance")
    for row in validated[1:]:
        if row["source_provenance"] != source:
            _fail("Epoch artifacts do not share exact clean source provenance")
        if row["initialization_provenance"] != initialization:
            _fail("Epoch artifacts do not share exact V14 initialization provenance")
    for earlier, later in pairwise(validated):
        if later["history"][: earlier["epoch"]] != earlier["history"]:
            _fail(
                f"Epoch {later['epoch']} does not preserve exact cumulative history from "
                f"epoch {earlier['epoch']}"
            )

    candidates: list[dict[str, Any]] = []
    for row in validated:
        candidate = {
            "epoch": row["epoch"],
            "checkpoint_metadata_path": row["path"],
            "checkpoint_metadata_sha256": row["artifact_sha256"],
            "residual_state_sha256": row["residual_state_sha256"],
            "color": deepcopy(row["metrics"]["color"]),
            "mirror": deepcopy(row["metrics"]["mirror"]),
        }
        candidate["color_eligible"] = _color_eligible(candidate)
        candidate["continuation_gate_passed"] = _continuation_passed(candidate)
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
        "selector_type": "strict_v18_centered_content_gate_epoch_selector",
        "report_only": True,
        "model_inference_executed": False,
        "checkpoint_tensor_state_loaded": False,
        "question_dependent_scene_processing": False,
        "config_path": str(PINNED_CONFIG_PATH),
        "config_hash": contract["config_hash"],
        "selection_artifact_path": selection_path,
        "selection_artifact_sha256": selection_digest,
        "selection_ids_sha256": selection_evidence["selection_sha256"],
        "pair_membership_sha256": selection_evidence["pair_membership_sha256"],
        "source_provenance": deepcopy(source),
        "initialization_provenance": deepcopy(initialization),
        "selection_policy": deepcopy(contract["screen"]),
        "selection_policy_sha256": _canonical_sha256(contract["screen"]),
        "epoch_count": len(candidates),
        "epochs": candidates,
        "eligible_epoch_count": len(ranking),
        "ranking": ranking,
        "selected_epoch": None if selected is None else selected["epoch"],
        "selected_checkpoint_metadata_path": (
            None if selected is None else selected["checkpoint_metadata_path"]
        ),
        "selected_checkpoint_metadata_sha256": (
            None if selected is None else selected["checkpoint_metadata_sha256"]
        ),
        "selected_residual_state_sha256": (
            None if selected is None else selected["residual_state_sha256"]
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
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        parser.error("duplicate V18 epoch binding")
    _reject_forbidden_input_path(args.config)
    config = load_config(args.config)
    selection, selection_digest = _load_json_strict(args.selection)
    loaded = {epoch: _load_json_strict(path) for epoch, path in epoch_paths.items()}
    summary = summarize_v18_epochs(
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
