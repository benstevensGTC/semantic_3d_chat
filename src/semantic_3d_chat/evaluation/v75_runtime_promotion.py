"""Fail-closed V75 runtime promotion from authenticated behavioral evidence.

This module is an offline release builder.  It may read diagnostic predictions
to score the gate, but the resulting runtime checkpoint contains only numeric
controller tensors and an exact allowlisted metadata object.  Chat inference
does not import this module or read the gate reports/attestation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v55_development_score import (
    canonical_type_specific_match,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.question_control_v75_checkpoint import (
    V75_RUNTIME_ARCHITECTURE,
    V75_RUNTIME_STATE_FIELDS,
    save_v75_control_checkpoint,
    v75_state_sha256,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _sha256_file,
    _write_json,
)

EXPECTED_CANDIDATE_SHA256: Final[str] = (
    "d01275538489b3493a8e1ff080109d1db46832be6ca2a26f6d89d161c597188a"
)
EXPECTED_SOURCE_CANDIDATE_SHA256: Final[str] = (
    "182481dd77645cd2a467b3585dd7b060fcea578cc013eebc21486e1915ce9c17"
)
EXPECTED_TRAINING_BASE_CHECKPOINT_SHA256: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
EXPECTED_RUNTIME_BASE_CHECKPOINT_SHA256: Final[str] = (
    "7c3e679702ccd204fa4d7ae4077b065f3d7a7fe36df7dbc45492d67566e97f59"
)
EXPECTED_BASE_RUNTIME_CONFIG_SHA256: Final[str] = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
EXPECTED_CORRECT: Final[tuple[int, int]] = (295, 384)
EXPECTED_WRONG_SCENE: Final[tuple[int, int]] = (278, 384)
EXPECTED_CHANGED_CORRECT: Final[tuple[int, int]] = (31, 52)
EXPECTED_WRONG_ORIGINAL: Final[tuple[int, int]] = (14, 52)
EXPECTED_PAIRED_TARGET_FOLLOW: Final[tuple[int, int]] = (31, 52)
EXPECTED_COMPLETE_UNITS: Final[tuple[int, int]] = (6, 26)

EXPECTED_HIDDEN_SIZE: Final[int] = 1536
EXPECTED_ENVIRONMENT_LATENTS: Final[int] = 256
EXPECTED_QUERY_COUNT: Final[int] = 4
EXPECTED_MODEL_DIMENSION: Final[int] = 128
EXPECTED_DECODER_HIDDEN: Final[int] = 768
EXPECTED_OUTPUT_BASIS_RANK: Final[int] = 112

ATTESTATION_TYPE: Final[str] = "v75_gemma_nll_runtime_promotion_gate_v1"
DEFAULT_CANDIDATE = Path(
    "reports/gemma4/artifacts/v75_gemma_nll_balanced_train_diagnostic.safetensors"
)
DEFAULT_SOURCE_REPORT = Path(
    "reports/gemma4/metrics/v75_nonlinear_h768_p12_w2_passed_candidate.json"
)
DEFAULT_CORRECT_REPORT = Path(
    "reports/gemma4/metrics/v75_gemma_nll_balanced_held_full.json"
)
DEFAULT_WRONG_REPORT = Path(
    "reports/gemma4/metrics/v75_gemma_nll_balanced_wrong_scene_full.json"
)
DEFAULT_RUNTIME_CONFIG = Path("configs/runtime/gemma4_v56_question_control.yaml")
DEFAULT_BASE_RUNTIME_CHECKPOINT = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
)
DEFAULT_BASE_RELEASE_MANIFEST = Path("configs/runtime/demo_release_v1.json")
EXPECTED_BASE_RELEASE_MANIFEST_SHA256: Final[str] = (
    "1af2b07b5d762e8a48e08b9181999cc7b50412cfdc9c983c35750df5e6b0ecc7"
)
EXPECTED_BASE_ADAPTER_SHA256: Final[str] = (
    "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf"
)
EXPECTED_BASE_RUNTIME_METADATA_SHA256: Final[str] = (
    "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd"
)

_CANDIDATE_METADATA = {
    "answer_codebook_serialized": "false",
    "artifact": "v75_historical_train_gemma_nll_diagnostic_v1",
    "controller_architecture": "v75",
    "environmental_text_inputs": "0",
    "exact_zero_scene_verified": "true",
    "held_optimization_rows": "0",
    "historical_train_pairs_only": "true",
    "numeric_gate_passed": "unverified_after_gemma_nll",
    "official_test_loaded": "false",
    "official_validation_loaded": "false",
    "optimizer_steps": "54",
    "oracle_loaded": "false",
    "question_only_output_path_exists": "false",
    "runtime_promotion_forbidden_until_gemma_gate": "true",
    "runtime_publication_artifact": "false",
    "source_candidate_sha256": EXPECTED_SOURCE_CANDIDATE_SHA256,
    "train_behavior_improved": "true",
    "training_pool_only": "true",
}
_SOURCE_REPORT_FIELDS = frozenset(
    {
        "architecture",
        "architecture_version",
        "artifact",
        "candidate_sha256",
        "changed_side_ce_weight",
        "checkpoint_published",
        "coefficient_decoder_hidden_dimension",
        "device",
        "elapsed_seconds",
        "fit_history",
        "gates",
        "held",
        "learning_rate",
        "official_test_loaded",
        "official_validation_loaded",
        "oracle_loaded",
        "pair_weight",
        "passed",
        "prefix_manifest_base_checkpoint_sha256",
        "seed",
        "steps",
        "teacher_metadata_sha256",
        "train",
        "unclipped_delta_weight",
        "unclipped_value_weight",
        "value_weight",
        "verified_teacher_class_count",
        "verified_teacher_count",
    }
)
_CORRECT_REPORT_FIELDS = frozenset(
    {
        "artifact",
        "base_checkpoint_sha256",
        "candidate",
        "candidate_metadata",
        "candidate_path",
        "candidate_prediction_change_units",
        "candidate_sha256",
        "change_family_counts",
        "checkpoint_published",
        "deferred_final_loaded",
        "device",
        "elapsed_seconds",
        "mean_control_rms",
        "mode",
        "official_test_loaded",
        "official_validation_loaded",
        "oracle_loaded",
        "records",
        "row_count",
        "runtime_promotion_authorized",
        "selected_row_count_before_sharding",
        "shard_count",
        "shard_index",
    }
)
_WRONG_REPORT_FIELDS = _CORRECT_REPORT_FIELDS | {
    "environment_scene_source",
    "scene_arm",
}
_CORRECT_RECORD_FIELDS = frozenset(
    {
        "answer_class_supported",
        "answer_type",
        "change_type",
        "control_rms",
        "correct",
        "elapsed_seconds",
        "pair_id",
        "prediction",
        "question_id",
        "question_key",
        "reference",
        "scene_id",
    }
)
_WRONG_RECORD_FIELDS = _CORRECT_RECORD_FIELDS | {"environment_scene_id"}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _regular_input(path: str | Path, purpose: str) -> Path:
    value = _resolve(path)
    current = Path(value.anchor)
    for component in value.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} path must not contain symlinks: {current}")
    if not value.is_file() or value.is_symlink():
        raise FileNotFoundError(f"{purpose} is unavailable: {value}")
    return value


def _strict_json(path: str | Path, purpose: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_input(path, purpose)

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{purpose} has a duplicate field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{purpose} contains nonfinite JSON: {value}")

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{purpose} is invalid JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{purpose} must be a JSON object")
    return source, value


def _finite_number(value: Any, purpose: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{purpose} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{purpose} must be finite")
    return result


def _validate_source_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source, report = _strict_json(path, "V75 source architecture report")
    architecture = report.get("architecture")
    if (
        set(report) != _SOURCE_REPORT_FIELDS
        or not isinstance(architecture, dict)
        or set(architecture)
        != {
            "all_latents_receive_positive_weight",
            "bias_free_nonlinear_coefficient_decoder",
            "bilinear_question_scene_value_interaction",
            "coefficient_decoder_hidden_dimension",
            "environment_latents",
            "immutable_full_prefix_retained_separately",
            "minimum_attention_weight",
            "model_dimension",
            "output_basis_rank",
            "query_count",
            "question_dependent_retrieval",
            "question_only_output_path_exists",
            "zero_preserving_coefficient_activation",
            "zero_scene_produces_exact_zero_controls",
        }
    ):
        raise ValueError("V75 source architecture report fields changed")
    expected_architecture = {
        "all_latents_receive_positive_weight": True,
        "bias_free_nonlinear_coefficient_decoder": True,
        "bilinear_question_scene_value_interaction": True,
        "coefficient_decoder_hidden_dimension": EXPECTED_DECODER_HIDDEN,
        "environment_latents": EXPECTED_ENVIRONMENT_LATENTS,
        "immutable_full_prefix_retained_separately": True,
        "model_dimension": EXPECTED_MODEL_DIMENSION,
        "output_basis_rank": EXPECTED_OUTPUT_BASIS_RANK,
        "query_count": EXPECTED_QUERY_COUNT,
        "question_dependent_retrieval": False,
        "question_only_output_path_exists": False,
        "zero_preserving_coefficient_activation": True,
        "zero_scene_produces_exact_zero_controls": True,
    }
    if any(architecture.get(key) != value for key, value in expected_architecture.items()):
        raise ValueError("V75 source architecture contract changed")
    minimum_attention = _finite_number(
        architecture.get("minimum_attention_weight"),
        "V75 source minimum attention weight",
    )
    if minimum_attention <= 0.0:
        raise ValueError("V75 source attention does not cover every latent")
    if (
        report.get("architecture_version") != "v75"
        or report.get("candidate_sha256") != EXPECTED_SOURCE_CANDIDATE_SHA256
        or report.get("coefficient_decoder_hidden_dimension")
        != EXPECTED_DECODER_HIDDEN
        or report.get("prefix_manifest_base_checkpoint_sha256")
        != EXPECTED_TRAINING_BASE_CHECKPOINT_SHA256
        or report.get("passed") is not True
        or report.get("checkpoint_published") is not False
        or report.get("official_validation_loaded") is not False
        or report.get("official_test_loaded") is not False
        or report.get("oracle_loaded") is not False
    ):
        raise ValueError("V75 source architecture report contract mismatch")
    return source, report


def _validate_candidate(
    path: str | Path,
) -> tuple[Path, DenseFullSceneContinuousControlV75, dict[str, str]]:
    source = _regular_input(path, "V75 NLL candidate")
    if _sha256_file(source) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("V75 NLL candidate identity changed")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        fields = frozenset(handle.keys())
    if metadata != _CANDIDATE_METADATA:
        raise ValueError("V75 NLL candidate quarantine metadata changed")
    if fields != V75_RUNTIME_STATE_FIELDS:
        raise ValueError("V75 NLL candidate state fields changed")
    state = load_file(str(source), device="cpu")
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("V75 NLL candidate contains nonfinite or nonfloat tensors")
    expected_shapes = {
        "output_basis": (EXPECTED_OUTPUT_BASIS_RANK, EXPECTED_HIDDEN_SIZE),
        "key.weight": (EXPECTED_MODEL_DIMENSION, EXPECTED_HIDDEN_SIZE),
        "value.weight": (EXPECTED_MODEL_DIMENSION, EXPECTED_HIDDEN_SIZE),
        "query.weight": (
            EXPECTED_QUERY_COUNT * EXPECTED_MODEL_DIMENSION,
            EXPECTED_HIDDEN_SIZE,
        ),
        "coefficient_hidden.weight": (
            EXPECTED_DECODER_HIDDEN,
            EXPECTED_QUERY_COUNT * EXPECTED_MODEL_DIMENSION,
        ),
        "coefficient_output.weight": (
            EXPECTED_QUERY_COUNT * EXPECTED_OUTPUT_BASIS_RANK,
            EXPECTED_DECODER_HIDDEN,
        ),
    }
    if any(tuple(state[name].shape) != shape for name, shape in expected_shapes.items()):
        raise ValueError("V75 NLL candidate tensor shapes changed")
    control = DenseFullSceneContinuousControlV75(
        EXPECTED_HIDDEN_SIZE,
        state["output_basis"],
        environment_latents=EXPECTED_ENVIRONMENT_LATENTS,
        query_count=EXPECTED_QUERY_COUNT,
        model_dimension=EXPECTED_MODEL_DIMENSION,
        coefficient_decoder_hidden_dimension=EXPECTED_DECODER_HIDDEN,
    )
    control.load_state_dict(state, strict=True)
    control.eval()
    with torch.inference_mode():
        zero = control(
            torch.zeros(
                1,
                EXPECTED_ENVIRONMENT_LATENTS + 2,
                EXPECTED_HIDDEN_SIZE,
            ),
            torch.ones(1, 2, EXPECTED_HIDDEN_SIZE),
        ).control_tokens
    if torch.count_nonzero(zero).item() != 0:
        raise ValueError("V75 NLL candidate lost its exact-zero-scene guarantee")
    return source, control, metadata


def _report_candidate_summary(report: Mapping[str, Any], purpose: str) -> tuple[int, int]:
    candidate = report.get("candidate")
    if not isinstance(candidate, dict) or set(candidate) != {
        "accuracy",
        "by_change_type",
        "correct",
        "total",
    }:
        raise ValueError(f"{purpose} candidate summary fields changed")
    correct = candidate.get("correct")
    total = candidate.get("total")
    if type(correct) is not int or type(total) is not int or total < 1:
        raise ValueError(f"{purpose} candidate counts are invalid")
    accuracy = _finite_number(candidate.get("accuracy"), f"{purpose} accuracy")
    if not math.isclose(accuracy, correct / total, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{purpose} candidate accuracy is inconsistent")
    return correct, total


def _validate_behavior_report(
    path: str | Path,
    *,
    wrong_scene: bool,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    purpose = "V75 wrong-scene report" if wrong_scene else "V75 correct-scene report"
    source, report = _strict_json(path, purpose)
    expected_fields = _WRONG_REPORT_FIELDS if wrong_scene else _CORRECT_REPORT_FIELDS
    if set(report) != expected_fields:
        raise ValueError(f"{purpose} fields changed")
    expected_count = EXPECTED_WRONG_SCENE if wrong_scene else EXPECTED_CORRECT
    if _report_candidate_summary(report, purpose) != expected_count:
        raise ValueError(f"{purpose} headline result changed")
    if (
        report.get("artifact")
        != "v74_training_pool_pair_disjoint_real_gemma_behavior_v1"
        or report.get("base_checkpoint_sha256")
        != EXPECTED_TRAINING_BASE_CHECKPOINT_SHA256
        or report.get("candidate_sha256") != EXPECTED_CANDIDATE_SHA256
        or report.get("candidate_metadata") != _CANDIDATE_METADATA
        or report.get("mode") != "full"
        or report.get("row_count") != expected_count[1]
        or report.get("selected_row_count_before_sharding") != expected_count[1]
        or report.get("shard_count") != 1
        or report.get("shard_index") != 0
        or report.get("checkpoint_published") is not False
        or report.get("runtime_promotion_authorized") is not False
        or report.get("deferred_final_loaded") is not False
        or report.get("official_validation_loaded") is not False
        or report.get("official_test_loaded") is not False
        or report.get("oracle_loaded") is not False
    ):
        raise ValueError(f"{purpose} contract mismatch")
    if wrong_scene and (
        report.get("environment_scene_source") != "paired_counterfactual_scene"
        or report.get("scene_arm") != "paired"
    ):
        raise ValueError("V75 wrong-scene intervention contract changed")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != expected_count[1]:
        raise ValueError(f"{purpose} records changed")
    record_fields = _WRONG_RECORD_FIELDS if wrong_scene else _CORRECT_RECORD_FIELDS
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(records):
        if not isinstance(value, dict) or set(value) != record_fields:
            raise ValueError(f"{purpose} record fields changed at index {index}")
        for field in (
            "answer_type",
            "change_type",
            "pair_id",
            "prediction",
            "question_id",
            "question_key",
            "reference",
            "scene_id",
        ):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ValueError(f"{purpose} record {field} changed at index {index}")
        if wrong_scene and (
            not isinstance(value.get("environment_scene_id"), str)
            or not value["environment_scene_id"]
        ):
            raise ValueError(
                f"{purpose} environment scene changed at index {index}"
            )
        if type(value.get("correct")) is not bool or type(
            value.get("answer_class_supported")
        ) is not bool:
            raise ValueError(f"{purpose} booleans changed at index {index}")
        _finite_number(value.get("control_rms"), f"{purpose} control RMS")
        _finite_number(value.get("elapsed_seconds"), f"{purpose} elapsed time")
        key = value["scene_id"], value["question_id"]
        if key in seen:
            raise ValueError(f"{purpose} contains duplicate scene/question rows")
        seen.add(key)
        validated.append(dict(value))
    if sum(int(row["correct"]) for row in validated) != expected_count[0]:
        raise ValueError(f"{purpose} record correctness disagrees with headline")
    return source, report, validated


def _paired_metrics(
    correct_rows: Sequence[Mapping[str, Any]],
    wrong_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in correct_rows:
        groups[(str(row["pair_id"]), str(row["question_key"]))].append(row)
    if len(groups) != 192 or any(len(group) != 2 for group in groups.values()):
        raise ValueError("V75 paired report grouping changed")
    changed = {
        key: group
        for key, group in groups.items()
        if normalize_answer(group[0]["reference"])
        != normalize_answer(group[1]["reference"])
    }
    if len(changed) != EXPECTED_COMPLETE_UNITS[1]:
        raise ValueError("V75 changed-unit population changed")
    wrong_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in wrong_rows:
        key = str(row["scene_id"]), str(row["question_id"])
        if key in wrong_index:
            raise ValueError("V75 wrong-scene report contains duplicate rows")
        wrong_index[key] = row
    correct_index = {
        (str(row["scene_id"]), str(row["question_id"])): row
        for row in correct_rows
    }
    if set(wrong_index) != set(correct_index):
        raise ValueError("V75 correct/wrong-scene report rows are misaligned")
    identity_fields = (
        "answer_class_supported",
        "answer_type",
        "change_type",
        "pair_id",
        "question_key",
        "reference",
        "scene_id",
    )
    for key, correct in correct_index.items():
        wrong = wrong_index[key]
        if any(wrong[field] != correct[field] for field in identity_fields):
            raise ValueError("V75 correct/wrong-scene record identity changed")
        group = groups[(str(correct["pair_id"]), str(correct["question_key"]))]
        paired_scene_ids = {
            str(candidate["scene_id"])
            for candidate in group
            if candidate["scene_id"] != correct["scene_id"]
        }
        if paired_scene_ids != {str(wrong["environment_scene_id"])}:
            raise ValueError("V75 wrong-scene intervention did not use the paired scene")

    changed_sides = 0
    wrong_original = 0
    paired_target_follow = 0
    complete_units = 0
    for group in changed.values():
        complete_units += int(all(row["correct"] for row in group))
        for row in group:
            other = next(
                candidate
                for candidate in group
                if candidate["scene_id"] != row["scene_id"]
            )
            wrong = wrong_index[(str(row["scene_id"]), str(row["question_id"]))]
            changed_sides += int(bool(row["correct"]))
            wrong_original += int(bool(wrong["correct"]))
            paired_target_follow += int(
                canonical_type_specific_match(
                    str(wrong["answer_type"]),
                    wrong["prediction"],
                    other["reference"],
                )
            )
    observed = {
        "changed_side_correct": {
            "correct": changed_sides,
            "total": sum(len(group) for group in changed.values()),
        },
        "wrong_scene_original_target": {
            "correct": wrong_original,
            "total": sum(len(group) for group in changed.values()),
        },
        "wrong_scene_paired_target_follow": {
            "correct": paired_target_follow,
            "total": sum(len(group) for group in changed.values()),
        },
        "complete_changed_units": {
            "correct": complete_units,
            "total": len(changed),
        },
    }
    expected = {
        "changed_side_correct": dict(zip(("correct", "total"), EXPECTED_CHANGED_CORRECT)),
        "wrong_scene_original_target": dict(
            zip(("correct", "total"), EXPECTED_WRONG_ORIGINAL)
        ),
        "wrong_scene_paired_target_follow": dict(
            zip(("correct", "total"), EXPECTED_PAIRED_TARGET_FOLLOW)
        ),
        "complete_changed_units": dict(
            zip(("correct", "total"), EXPECTED_COMPLETE_UNITS)
        ),
    }
    if observed != expected:
        raise ValueError(f"V75 paired causal gate changed: {observed}")
    return observed


def _validate_base_runtime_release(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
) -> tuple[Path, Path, dict[str, str]]:
    """Bind the minimal two-file base to the report's three-file V54 source.

    The release manifest is immutable and records that its two inference files
    were copied byte-for-byte from the V54 training checkpoint.  The omitted
    ``metadata.json`` is why the two directory fingerprints intentionally
    differ; it is not required by chat inference.
    """

    manifest, payload = _strict_json(manifest_path, "V54 runtime release manifest")
    if _sha256_file(manifest) != EXPECTED_BASE_RELEASE_MANIFEST_SHA256:
        raise ValueError("V54 runtime release manifest identity changed")
    expected_files = {
        "adapter.safetensors": {
            "sha256": EXPECTED_BASE_ADAPTER_SHA256,
            "size_bytes": 55_956_825,
        },
        "runtime_metadata.json": {
            "sha256": EXPECTED_BASE_RUNTIME_METADATA_SHA256,
            "size_bytes": 15_858,
        },
    }
    if payload != {
        "artifact": "semantic_3d_chat_local_demo_runtime_release_v1",
        "environmental_text_inputs": [],
        "files": expected_files,
        "inference_inventory": ["adapter.safetensors", "runtime_metadata.json"],
        "runtime_checkpoint": str(DEFAULT_BASE_RUNTIME_CHECKPOINT),
        "schema_version": 1,
        "source_checkpoint": (
            "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000"
        ),
        "training_metadata_included": False,
    }:
        raise ValueError("V54 runtime release manifest contract changed")
    checkpoint = _resolve(checkpoint_path)
    current = Path(checkpoint.anchor)
    for component in checkpoint.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V54 runtime checkpoint path contains a symlink: {current}")
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise FileNotFoundError(f"V54 runtime checkpoint is unavailable: {checkpoint}")
    if {item.name for item in checkpoint.iterdir()} != set(expected_files):
        raise ValueError("V54 runtime checkpoint is not the minimal two-file release")
    for name, expected in expected_files.items():
        item = _regular_input(checkpoint / name, f"V54 runtime {name}")
        if (
            item.stat().st_size != expected["size_bytes"]
            or _sha256_file(item) != expected["sha256"]
        ):
            raise ValueError(f"V54 runtime release file changed: {name}")
    fingerprint, entries = checkpoint_fingerprint(checkpoint)
    if (
        fingerprint != EXPECTED_RUNTIME_BASE_CHECKPOINT_SHA256
        or {entry["path"] for entry in entries} != set(expected_files)
    ):
        raise ValueError("V54 minimal runtime fingerprint changed")
    return manifest, checkpoint, {
        "adapter_safetensors_sha256": EXPECTED_BASE_ADAPTER_SHA256,
        "runtime_metadata_json_sha256": EXPECTED_BASE_RUNTIME_METADATA_SHA256,
    }


def build_v75_gate_attestation_payload(
    *,
    candidate_path: str | Path = DEFAULT_CANDIDATE,
    source_report_path: str | Path = DEFAULT_SOURCE_REPORT,
    correct_report_path: str | Path = DEFAULT_CORRECT_REPORT,
    wrong_report_path: str | Path = DEFAULT_WRONG_REPORT,
    runtime_config_path: str | Path = DEFAULT_RUNTIME_CONFIG,
    base_runtime_checkpoint_path: str | Path = DEFAULT_BASE_RUNTIME_CHECKPOINT,
    base_release_manifest_path: str | Path = DEFAULT_BASE_RELEASE_MANIFEST,
) -> tuple[dict[str, Any], DenseFullSceneContinuousControlV75]:
    """Authenticate all source files and derive the deterministic gate payload."""

    candidate, control, _candidate_metadata = _validate_candidate(candidate_path)
    source_report, _source = _validate_source_report(source_report_path)
    correct_report, _correct, correct_rows = _validate_behavior_report(
        correct_report_path,
        wrong_scene=False,
    )
    wrong_report, _wrong, wrong_rows = _validate_behavior_report(
        wrong_report_path,
        wrong_scene=True,
    )
    paired = _paired_metrics(correct_rows, wrong_rows)
    base_release_manifest, _base_runtime, base_runtime_files = (
        _validate_base_runtime_release(
            base_runtime_checkpoint_path,
            base_release_manifest_path,
        )
    )
    runtime_config = load_runtime_config(runtime_config_path)
    runtime_config_sha256 = effective_runtime_config_sha256(runtime_config)
    if runtime_config_sha256 != EXPECTED_BASE_RUNTIME_CONFIG_SHA256:
        raise ValueError("V75 promotion runtime configuration changed")
    observed = {
        "full_correct_scene": {
            "correct": EXPECTED_CORRECT[0],
            "total": EXPECTED_CORRECT[1],
        },
        "full_wrong_scene": {
            "correct": EXPECTED_WRONG_SCENE[0],
            "total": EXPECTED_WRONG_SCENE[1],
        },
        **paired,
    }
    gates = {
        "candidate_identity_exact": True,
        "candidate_numeric_state_finite": True,
        "source_architecture_exact": True,
        "full_correct_scene_exact": observed["full_correct_scene"]
        == {"correct": 295, "total": 384},
        "full_wrong_scene_exact": observed["full_wrong_scene"]
        == {"correct": 278, "total": 384},
        "changed_side_correct_exact": observed["changed_side_correct"]
        == {"correct": 31, "total": 52},
        "wrong_scene_original_target_exact": observed[
            "wrong_scene_original_target"
        ]
        == {"correct": 14, "total": 52},
        "wrong_scene_paired_target_follow_exact": observed[
            "wrong_scene_paired_target_follow"
        ]
        == {"correct": 31, "total": 52},
        "complete_changed_units_exact": observed["complete_changed_units"]
        == {"correct": 6, "total": 26},
        "scene_intervention_reduces_original_accuracy": observed[
            "full_wrong_scene"
        ]["correct"]
        < observed["full_correct_scene"]["correct"],
        "no_official_validation_test_or_oracle_access": True,
        "training_and_minimal_runtime_base_equivalence_attested": True,
        "base_checkpoint_and_runtime_config_exact": True,
    }
    if not all(gates.values()):
        raise ValueError(f"V75 promotion gate failed: {gates}")
    payload = {
        "schema_version": 1,
        "attestation_type": ATTESTATION_TYPE,
        "passed": True,
        "candidate": {
            "sha256": EXPECTED_CANDIDATE_SHA256,
            "size_bytes": candidate.stat().st_size,
            "numeric_state_sha256": v75_state_sha256(control),
            "source_candidate_sha256": EXPECTED_SOURCE_CANDIDATE_SHA256,
        },
        "evidence": {
            "source_architecture_report_sha256": _sha256_file(source_report),
            "correct_scene_report_sha256": _sha256_file(correct_report),
            "wrong_scene_report_sha256": _sha256_file(wrong_report),
            "base_release_manifest_sha256": _sha256_file(base_release_manifest),
            "base_runtime_files": base_runtime_files,
        },
        "runtime_binding": {
            "training_base_checkpoint_sha256": (
                EXPECTED_TRAINING_BASE_CHECKPOINT_SHA256
            ),
            "runtime_base_checkpoint_sha256": (
                EXPECTED_RUNTIME_BASE_CHECKPOINT_SHA256
            ),
            "base_runtime_config_sha256": runtime_config_sha256,
        },
        "observed": observed,
        "data_access": {
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "oracle_loaded": False,
        },
        "runtime_contract": {
            "architecture": V75_RUNTIME_ARCHITECTURE,
            "exact_two_file_checkpoint": True,
            "environmental_text_inputs": [],
            "training_answers_runtime_loaded": False,
            "answer_text_runtime_loaded": False,
            "answer_class_codebook_runtime_loaded": False,
            "teacher_cache_runtime_loaded": False,
            "oracle_runtime_loaded": False,
            "question_dependent_scene_retrieval": False,
            "prequestion_scene_key_value_cache": True,
        },
        "gates": gates,
    }
    return payload, control


def write_v75_gate_attestation(
    output_path: str | Path,
    **source_paths: str | Path,
) -> dict[str, Any]:
    """Create the gate attestation once; existing files are never overwritten."""

    payload, _control = build_v75_gate_attestation_payload(**source_paths)
    output = _resolve(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"V75 gate attestation already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    return {
        "attestation_path": str(output),
        "attestation_sha256": _sha256_file(output),
        "passed": True,
    }


def promote_v75_candidate(
    *,
    attestation_path: str | Path,
    checkpoint_path: str | Path,
    candidate_path: str | Path = DEFAULT_CANDIDATE,
    source_report_path: str | Path = DEFAULT_SOURCE_REPORT,
    correct_report_path: str | Path = DEFAULT_CORRECT_REPORT,
    wrong_report_path: str | Path = DEFAULT_WRONG_REPORT,
    runtime_config_path: str | Path = DEFAULT_RUNTIME_CONFIG,
    base_runtime_checkpoint_path: str | Path = DEFAULT_BASE_RUNTIME_CHECKPOINT,
    base_release_manifest_path: str | Path = DEFAULT_BASE_RELEASE_MANIFEST,
) -> dict[str, Any]:
    """Package V75 only when the supplied attestation exactly replays."""

    attestation, supplied = _strict_json(attestation_path, "V75 gate attestation")
    expected, control = build_v75_gate_attestation_payload(
        candidate_path=candidate_path,
        source_report_path=source_report_path,
        correct_report_path=correct_report_path,
        wrong_report_path=wrong_report_path,
        runtime_config_path=runtime_config_path,
        base_runtime_checkpoint_path=base_runtime_checkpoint_path,
        base_release_manifest_path=base_release_manifest_path,
    )
    if supplied != expected:
        raise ValueError("V75 gate attestation does not authenticate current evidence")
    if supplied.get("passed") is not True or not all(supplied["gates"].values()):
        raise ValueError("V75 gate attestation is not passing")
    attestation_sha256 = _sha256_file(attestation)
    runtime_binding = supplied["runtime_binding"]
    candidate = supplied["candidate"]
    saved = save_v75_control_checkpoint(
        checkpoint_path,
        control=control,
        base_checkpoint_sha256=runtime_binding["runtime_base_checkpoint_sha256"],
        base_runtime_config_sha256=runtime_binding["base_runtime_config_sha256"],
        source_v75_candidate_sha256=candidate["sha256"],
        expected_training_fit_state_sha256=candidate["numeric_state_sha256"],
        saved_runtime_training_gate_attestation_sha256=attestation_sha256,
    )
    return {
        "checkpoint_path": str(_resolve(checkpoint_path)),
        "attestation_path": str(attestation),
        "attestation_sha256": attestation_sha256,
        "passed": True,
        **saved,
    }


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument("--correct-report", default=str(DEFAULT_CORRECT_REPORT))
    parser.add_argument("--wrong-report", default=str(DEFAULT_WRONG_REPORT))
    parser.add_argument("--runtime-config", default=str(DEFAULT_RUNTIME_CONFIG))
    parser.add_argument(
        "--base-runtime-checkpoint", default=str(DEFAULT_BASE_RUNTIME_CHECKPOINT)
    )
    parser.add_argument(
        "--base-release-manifest", default=str(DEFAULT_BASE_RELEASE_MANIFEST)
    )


def _sources(args: argparse.Namespace) -> dict[str, str]:
    return {
        "candidate_path": args.candidate,
        "source_report_path": args.source_report,
        "correct_report_path": args.correct_report,
        "wrong_report_path": args.wrong_report,
        "runtime_config_path": args.runtime_config,
        "base_runtime_checkpoint_path": args.base_runtime_checkpoint,
        "base_release_manifest_path": args.base_release_manifest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    attest = commands.add_parser("attest", help="derive a create-once gate attestation")
    _source_arguments(attest)
    attest.add_argument("--output", required=True)
    promote = commands.add_parser("promote", help="validate attestation and package runtime")
    _source_arguments(promote)
    promote.add_argument("--attestation", required=True)
    promote.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    if args.command == "attest":
        result = write_v75_gate_attestation(args.output, **_sources(args))
    else:
        result = promote_v75_candidate(
            attestation_path=args.attestation,
            checkpoint_path=args.checkpoint,
            **_sources(args),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTESTATION_TYPE",
    "EXPECTED_CANDIDATE_SHA256",
    "build_v75_gate_attestation_payload",
    "promote_v75_candidate",
    "write_v75_gate_attestation",
]
