"""Fail-closed V48 result review and conditional V49 terminal scaffold.

This module authenticates the exact V48 no-step report and proves that exactly
one of its fifteen fixed candidates satisfies the complete teacher-forced
threshold.  V48 itself did not select, authorize, or persist that candidate.

The only possible successor is a deterministic V49 reconstruction and staged
train-only gate.  V49 must first pass every non-greedy teacher, broad,
original-V46-candidate-relative prefix-trust, source, and frozen-state check.
Only then is exhaustive greedy evaluation authorized and mandatory.  V49 may
write its candidate checkpoint only after every final gate check passes.  Until exact
V49 module and test hashes replace the placeholders below, this module can
build only a non-materializable scaffold and ``write_report`` fails closed.
No model, QA, map, optimizer, validation, oracle, final-test, selector, or
promotion access occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT

V47_TERMINAL = Path("reports/gemma4/metrics/v47_book_continuation_terminal_gate.json")
V48_SCRIPT = Path("src/semantic_3d_chat/evaluation/v48_v47_u4_dual_margin_screen.py")
V48_TEST = Path("tests/test_v48_v47_u4_dual_margin_screen.py")
V48_REPORT = Path("reports/gemma4/metrics/v48_v47_u4_dual_margin_no_step_diagnostic.json")
PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v48_v47_u4_dual_margin_terminal_gate.json")

V49_SCRIPT = Path("src/semantic_3d_chat/evaluation/v49_guarded_candidate_greedy_gate.py")
V49_TEST = Path("tests/test_v49_guarded_candidate_greedy_gate.py")
V49_REPORT = Path("reports/gemma4/metrics/v49_guarded_candidate_train_gate.json")
V49_CHECKPOINT = Path("data_gemma4/checkpoints/gemma4_v49_guarded_alpha2_train_gate/update_000")
V49_CONFIG = Path("configs/experiments/gemma4_diverse28_book_continuation_v47.yaml")

V49_SCRIPT_SHA256_PLACEHOLDER = "PENDING_V49_SCRIPT_SHA256"
V49_TEST_SHA256_PLACEHOLDER = "PENDING_V49_TEST_SHA256"
_V49_SCRIPT_SHA256 = "8358e1326eab30292829e0d6978dfbf93d5bc29d6dab96f595e0e80b5eb52854"
_V49_TEST_SHA256 = "46f6cba7354aaa360e8be64a0d0968926fb384031fc54e4c0cb4158bd2960603"

_V49_AUTHORIZATION_ID = "v49_guarded_alpha2_train_candidate_gate"
_V49_ACTION = "one_deterministic_v49_guarded_alpha2_train_candidate_gate"
_V47_TERMINAL_SHA256 = "98b49d60e2d400429c8d34f325885151e9db908c35b42d36a1896eeb5ca1fc06"
_V48_SCRIPT_SHA256 = "c132e66568b626658315dc90c3638e52677d299900d863270ddfced07c580611"
_V48_TEST_SHA256 = "e522471a3ca88ea363bf59c2de0bb5f2a9d1cee628a8e874264fa2d2db52e31a"
_V48_REPORT_SHA256 = "7abd2fa7741f84ea56933383199ec449d47dd99361def15d6a3874b9e154e02c"
_PROTECTED_REPORT_SHA256 = "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
_CONFIG_SHA256 = "6b15813237b217d8daad446c799127215bbb1366f2b442b61e975787efa4f6b7"
_V48_INVENTORY_SHA256 = "263ddedf0086522457bf626cd47ae13c27459a4ba8aaf64b8d5ab94bae3d3f9a"

_SOURCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query/update_004"
)
_SOURCE_FILES = {
    "adapter.safetensors": ("8f903f5d1ba93d37ccd6204e3b58c9a5529ff9ee2b74edca0787ecb5a2c62c66"),
    "metadata.json": ("c6affe7f60c094580e2ea5f5d1330f475bf359e0a3a58bfc3bf3b3ada1de0be1"),
    "optimizer.pt": ("fe66be9cae13951fbfc217e0c512e43366c347181457c9e551230a9d6001db80"),
    "runtime_metadata.json": ("4e3a1af91642c9f2adb0b3e43997455a1aea31f86bf45618459d6005a68d4bbf"),
}
_SOURCE_FULL_SHA256 = "adfc0400d1a3bb49b278cd3012ab571d01465f2380881f986c085a25474276e5"
_SOURCE_AUTHORIZED_SHA256 = "a23de4988774a966c0d7aac378ede5d15a3fa1d96093c5039f181a62b0bb09b0"
_FROZEN_SHA256 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"

_PREFIX_REFERENCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query/update_000"
)
_PREFIX_REFERENCE_FILES = {
    "adapter.safetensors": ("c47bbb9bacbb5bc8178e9a1797ec47b04ee4a3709042c6b30f6935eacc4686f0"),
    "metadata.json": ("e76e8a905af53fb082684000a6a6e16845b79e0795b6dbb047a2703245198574"),
    "runtime_metadata.json": ("01e645b82c5e533dd2319ef8a97171437b149c4e1ef86201f83fdb22de047987"),
}
_PREFIX_REFERENCE_FULL_SHA256 = "1d5adc1fb0d7a895056b77d38c8a12aba95c9997ec8a94edf68673f9c58fb954"
_PREFIX_REFERENCE_AUTHORIZED_SHA256 = (
    "d60b665d9a970433b2ed59e6769b9114468bef608b98eae828268101d39db56c"
)

_CANDIDATE_ID = "guarded_both_sign_alpha_2p0"
_CANDIDATE_DIRECTION = "guarded_both_sign"
_CANDIDATE_ALPHA = 2.0
_CANDIDATE_FULL_SHA256 = "69c5471d141ab56397969e2aac5f1097676f6ea328a3d5577e816c3aef6f3387"
_CANDIDATE_AUTHORIZED_SHA256 = "f43e1b2a84006c8188adeff9f206ba864ae3d51b6567cad679f6d2e5f3610cf2"
_DIRECTION_IDS = ("dual_query_sign", "dual_both_sign", "guarded_both_sign")
_ALPHA_GRID = (0.125, 0.25, 0.5, 1.0, 2.0)
_SCENE_LR = 1.0e-5
_QUERY_LR = 8.0e-6
_BROAD_NLL_MAXIMUM = 2.9213306349515915
_PREFIX_TRUST_RMS_MAXIMUM = 0.002
_ORIGINAL_V41_PRIORITY_DEFICIT = 31.113729119300842
_GRADIENT_SPECS = (
    ("g_book", "pair_000015", "cfq_163eb92339ad35a5", 0),
    ("g_mirror", "pair_000016", "cfq_699675ceeaf65406", 1),
    ("g5_guard", "pair_000006", "cfq_5c84a2c27d2be251", 0),
)
_EXPECTED_SELECTION = {
    "candidate_id": _CANDIDATE_ID,
    "direction_id": _CANDIDATE_DIRECTION,
    "alpha": _CANDIDATE_ALPHA,
    "full_tensor_state_sha256": _CANDIDATE_FULL_SHA256,
    "authorized_surface_state_sha256": _CANDIDATE_AUTHORIZED_SHA256,
    "complete_units": 10,
    "positive_sides": 35,
    "cross_prefix_complete_units": 18,
    "complete_physical_pair_coverage": 5,
    "mirror_complete_units": 2,
    "book_complete_units": 1,
    "book_cross_prefix_complete_units": 2,
    "priority_side_deficit": 29.86921536922455,
    "priority_deficit_improvement": 1.244513750076294,
    "broad_nll": 2.920842170715332,
    "v47_u4_relative_prefix_trust_rms": 0.0008939842227846384,
    "q163_side_margins": [0.17693853378295898, 0.20384597778320312],
    "q699_side_margins": [0.5, 0.3750000596046448],
    "q5_side_margins": [0.0625, 0.25],
}
_NON_GREEDY_PRE_GATE_CHECKS = (
    "source_v47_u4_exact_before_reconstruction",
    "reconstructed_candidate_full_tensor_state_exact",
    "reconstructed_candidate_authorized_surface_state_exact",
    "teacher_complete_units_at_least_10",
    "teacher_positive_sides_at_least_35",
    "teacher_cross_prefix_complete_units_at_least_17",
    "complete_physical_pair_coverage_at_least_5",
    "mirror_complete_units_at_least_2",
    "book_complete_units_at_least_1",
    "book_cross_prefix_complete_units_at_least_1",
    "priority_deficit_improvement_at_least_0_5_vs_original_v41_u0",
    "broad_nll_at_most_v45_maximum",
    "both_lost_sides_strictly_positive",
    "scene_readout_state_changed",
    "query_state_changed",
    "frozen_state_exact",
    "original_v46_candidate_relative_prefix_trust_rms_at_most_0_002",
)
_GREEDY_FINAL_GATE_CHECKS = (
    "train_greedy_complete_units_at_least_5",
    "broad_greedy_exact_correct_at_least_23_of_48",
    "broad_greedy_row_count_exactly_48",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOLERANCE = 1.0e-6


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    combined = value if value.is_absolute() else PROJECT_ROOT / value
    # ``resolve()`` follows the final symlink and defeats the explicit symlink
    # rejection in the authenticated-read and one-shot-write paths.
    return Path(os.path.abspath(combined))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _lower_hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal digits")
    return value


def _locked_file(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{field} bytes changed: expected {expected}, observed {observed}")


def _gradient_specs() -> list[dict[str, Any]]:
    return [
        {
            "gradient_id": gradient_id,
            "pair_id": pair_id,
            "question_key": question_key,
            "side_index": side_index,
            "loss": "negative_selected_side_margin",
        }
        for gradient_id, pair_id, question_key, side_index in _GRADIENT_SPECS
    ]


def _authenticate_static_inputs() -> dict[str, Any]:
    pins = {
        V47_TERMINAL: _V47_TERMINAL_SHA256,
        V48_SCRIPT: _V48_SCRIPT_SHA256,
        V48_TEST: _V48_TEST_SHA256,
        PROTECTED_REPORT: _PROTECTED_REPORT_SHA256,
    }
    for path, expected in pins.items():
        _locked_file(_resolve(path), expected, str(path))
    terminal = _mapping(
        json.loads(_resolve(V47_TERMINAL).read_text(encoding="utf-8")),
        "V47 terminal",
    )
    authorization = _mapping(
        terminal.get("conditional_successor_authorization"),
        "V47 V48 authorization",
    )
    integrity = _mapping(authorization.get("implementation_integrity"), "V48 integrity")
    checks = {
        "artifact": terminal.get("artifact") == "v47_book_continuation_terminal_gate",
        "passed": terminal.get("passed") is True,
        "v47_failed": terminal.get("v47_final_train_only_gate_passed") is False,
        "successor": terminal.get("only_exact_successor_authorized")
        == "v48_v47_u4_dual_margin_no_step_diagnostic",
        "authorization_id": authorization.get("authorization_id")
        == "v48_v47_u4_dual_margin_no_step_diagnostic",
        "authorized": authorization.get("authorized") is True,
        "script": authorization.get("authorized_script") == str(V48_SCRIPT),
        "test": authorization.get("authorized_test") == str(V48_TEST),
        "report": authorization.get("authorized_report") == str(V48_REPORT),
        "script_hash": integrity.get("script_sha256") == _V48_SCRIPT_SHA256,
        "test_hash": integrity.get("test_sha256") == _V48_TEST_SHA256,
        "config_hash": integrity.get("config_sha256") == _CONFIG_SHA256,
    }
    if not all(checks.values()):
        raise ValueError(f"V48 predecessor authorization changed: {checks}")
    return {
        "v47_terminal": {"path": str(V47_TERMINAL), "sha256": _V47_TERMINAL_SHA256},
        "v48_script": {"path": str(V48_SCRIPT), "sha256": _V48_SCRIPT_SHA256},
        "v48_test": {"path": str(V48_TEST), "sha256": _V48_TEST_SHA256},
        "protected_report": {
            "path": str(PROTECTED_REPORT),
            "sha256": _PROTECTED_REPORT_SHA256,
        },
        "authorization_checks": checks,
        "model_loaded": False,
        "qa_loaded": False,
        "maps_loaded": False,
    }


def _priority_deficit(pair_metrics: Mapping[str, Any]) -> float:
    rows = _sequence(pair_metrics.get("units"), "candidate pair units")
    if len(rows) != 25:
        raise ValueError("V48 candidate must contain all 25 pair units")
    counts = {"book_support": 0, "picture_support": 0}
    deficit = 0.0
    for value in rows:
        row = _mapping(value, "candidate pair unit")
        family = str(row.get("family"))
        if family not in counts:
            continue
        margins = _sequence(row.get("side_margins"), "candidate side margins")
        if len(margins) != 2:
            raise ValueError("V48 candidate pair unit lacks two side margins")
        deficit += sum(max(0.0, 0.5 - _finite(margin, "side margin")) for margin in margins)
        counts[family] += 1
    if counts != {"book_support": 4, "picture_support": 4}:
        raise ValueError("V48 priority family inventory changed")
    return deficit


def _lost_margin(pair_metrics: Mapping[str, Any], question_key: str, side_index: int) -> float:
    rows = _sequence(pair_metrics.get("units"), "candidate pair units")
    matches = [
        _mapping(value, "candidate pair unit")
        for value in rows
        if _mapping(value, "candidate pair unit").get("question_key") == question_key
    ]
    if len(matches) != 1:
        raise ValueError(f"V48 focus question is not unique: {question_key}")
    margins = _sequence(matches[0].get("side_margins"), "focus side margins")
    return _finite(margins[side_index], "focus side margin")


def candidate_threshold_checks(
    pair_metrics: Mapping[str, Any], broad_nll: float, prefix_trust_rms: float
) -> dict[str, bool]:
    families = _mapping(pair_metrics.get("complete_units_by_family"), "families")
    cross_families = _mapping(
        pair_metrics.get("cross_prefix_complete_units_by_family"), "cross families"
    )
    deficit = _priority_deficit(pair_metrics)
    return {
        "complete_units_at_least_10": int(pair_metrics["complete_units"]) >= 10,
        "positive_sides_at_least_35": int(pair_metrics["positive_sides"]) >= 35,
        "cross_prefix_complete_units_at_least_17": int(pair_metrics["cross_prefix_complete_units"])
        >= 17,
        "complete_physical_pair_coverage_at_least_5": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= 5,
        "mirror_complete_units_at_least_2": int(families.get("mirror_lr", 0)) >= 2,
        "book_complete_units_at_least_1": int(families.get("book_support", 0)) >= 1,
        "book_cross_prefix_complete_units_at_least_1": int(cross_families.get("book_support", 0))
        >= 1,
        "priority_deficit_improvement_at_least_0_5_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit >= 0.5
        ),
        "broad_nll_at_most_v45_maximum": broad_nll <= _BROAD_NLL_MAXIMUM,
        "both_lost_sides_strictly_positive": _lost_margin(pair_metrics, "cfq_5c84a2c27d2be251", 0)
        > 0.0
        and _lost_margin(pair_metrics, "cfq_699675ceeaf65406", 1) > 0.0,
        "candidate_relative_prefix_trust_rms_at_most_0_002": prefix_trust_rms
        <= _PREFIX_TRUST_RMS_MAXIMUM,
    }


def _validate_gradient_and_direction_audits(report: Mapping[str, Any]) -> dict[str, bool]:
    gradient = _mapping(report.get("gradient_audit"), "V48 gradient audit")
    direction = _mapping(report.get("direction_audit"), "V48 direction audit")
    specifications = _sequence(gradient.get("specifications"), "gradient specifications")
    observed_specs = [
        {
            "gradient_id": row.get("gradient_id"),
            "pair_id": row.get("pair_id"),
            "question_key": row.get("question_key"),
            "side_index": row.get("side_index"),
            "loss": row.get("loss_formula"),
        }
        for value in specifications
        if (row := _mapping(value, "gradient specification"))
    ]
    no_parameter_gradient_accumulation = all(
        _mapping(value, "gradient specification").get("parameter_grad_accumulation") is False
        for value in specifications
    )
    geometry = _mapping(gradient.get("geometry"), "gradient geometry")
    groups = _mapping(geometry.get("groups"), "gradient groups")
    geometry_finite = True
    for group_name in ("scene_readout", "query"):
        group = _mapping(groups.get(group_name), f"{group_name} geometry")
        cosines = _mapping(group.get("pairwise_cosines"), "pairwise cosines")
        if len(cosines) != 3:
            geometry_finite = False
            continue
        for value in cosines.values():
            row = _mapping(value, "gradient cosine")
            for key in ("raw_cosine", "normalized_cosine"):
                cosine = _finite(row.get(key), key)
                geometry_finite = geometry_finite and -1.0 <= cosine <= 1.0
    norms = _mapping(geometry.get("raw_group_l2_norms"), "gradient norms")
    norms_positive = set(norms) == {value[0] for value in _GRADIENT_SPECS} and all(
        _finite(_mapping(value, "gradient norm row").get(group), "gradient norm") > 0.0
        for value in norms.values()
        for group in ("scene_readout", "query")
    )
    return {
        "specifications": observed_specs == _gradient_specs(),
        "no_parameter_gradient_accumulation": no_parameter_gradient_accumulation,
        "source_unchanged": gradient.get("source_state_unchanged") is True,
        "no_optimizer": gradient.get("optimizer_constructed_or_loaded") is False,
        "geometry_finite": geometry_finite,
        "norms_positive": norms_positive,
        "normalization": direction.get("normalization")
        == "each_nonzero_component_unit_l2_within_each_scene_or_query_group",
        "components": direction.get("direction_components")
        == {
            "dual_query_sign": ["g_book", "g_mirror"],
            "dual_both_sign": ["g_book", "g_mirror"],
            "guarded_both_sign": ["g_book", "g_mirror", "g5_guard"],
        },
        "query_mask": direction.get("inactive_scene_group_exact_zero_for_dual_query_sign") is True,
    }


def review_report_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    terminal = _mapping(report.get("terminal"), "V48 terminal attestation")
    source = _mapping(report.get("source_audit"), "V48 source audit")
    replay = _mapping(report.get("source_replay"), "V48 source replay")
    replay_pair = _mapping(replay.get("pair_metrics"), "V48 source replay metrics")
    inventory = _mapping(report.get("candidate_inventory"), "V48 candidate inventory")
    inventory_rows = _sequence(inventory.get("candidates"), "V48 inventory rows")
    results = _sequence(report.get("candidate_results"), "V48 candidate results")
    fixed = {
        "artifact": report.get("artifact") == "v48_v47_u4_dual_margin_no_step_diagnostic",
        "integrity": report.get("screen_integrity_passed") is True,
        "terminal_sha": terminal.get("sha256") == _V47_TERMINAL_SHA256,
        "terminal_id": terminal.get("authorization_id")
        == "v48_v47_u4_dual_margin_no_step_diagnostic",
        "source_checkpoint": source.get("checkpoint") == str(_SOURCE_CHECKPOINT),
        "source_full": source.get("full_tensor_state_sha256") == _SOURCE_FULL_SHA256,
        "source_authorized": source.get("authorized_surface_state_sha256")
        == _SOURCE_AUTHORIZED_SHA256,
        "source_frozen": source.get("frozen_state_sha256") == _FROZEN_SHA256,
        "source_files": source.get("readable_file_sha256")
        == {key: value for key, value in _SOURCE_FILES.items() if key != "optimizer.pt"},
        "source_optimizer_provenance": source.get("optimizer_file_sha256_provenance")
        == _SOURCE_FILES["optimizer.pt"],
        "source_optimizer_unopened": source.get("optimizer_file_opened") is False
        and source.get("optimizer_state_deserialized") is False
        and source.get("optimizer_state_loaded") is False,
        "source_replay": replay.get("passed") is True
        and replay.get("broad_nll") is True
        and replay.get("per_unit_nll") is True
        and replay.get("priority_deficit") is True,
        "source_replay_teacher": replay_pair.get("unit_count") == 25
        and replay_pair.get("complete_units") == 8
        and replay_pair.get("positive_sides") == 33
        and replay_pair.get("cross_prefix_complete_units") == 17
        and replay_pair.get("complete_physical_pair_coverage") == 4,
        "source_replay_families": replay_pair.get("complete_units_by_family")
        == {"book_support": 0, "mirror_lr": 1, "picture_support": 0}
        and replay_pair.get("cross_prefix_complete_units_by_family")
        == {"book_support": 1, "mirror_lr": 4, "picture_support": 2},
        "source_replay_broad": math.isclose(
            _finite(replay.get("broad_nll_value"), "source replay broad NLL"),
            2.9172145972649255,
            rel_tol=0.0,
            abs_tol=_TOLERANCE,
        ),
        "directions": inventory.get("direction_ids") == list(_DIRECTION_IDS),
        "alphas": inventory.get("alpha_grid") == list(_ALPHA_GRID),
        "formula": inventory.get("formula")
        == "float32_P0-alpha*lr_group*sign(normalized_component_sum)",
        "scene_lr": inventory.get("scene_readout_learning_rate") == _SCENE_LR,
        "query_lr": inventory.get("query_learning_rate") == _QUERY_LR,
        "candidate_count": inventory.get("candidate_count") == 15,
        "prehash": inventory.get("candidate_hashes_fixed_before_candidate_forward_evaluation")
        is True,
        "inventory_count": len(inventory_rows) == 15,
        "inventory_hash": inventory.get("candidate_inventory_sha256") == _V48_INVENTORY_SHA256
        and _canonical_sha256(list(inventory_rows)) == _V48_INVENTORY_SHA256,
        "result_count": len(results) == 15,
        "full25": report.get("all_15_candidates_received_full_25_unit_metrics") is True,
        "full48": report.get("all_15_candidates_received_fixed_48_row_broad_nll") is True,
        "full_trust": report.get("all_15_candidates_received_candidate_relative_prefix_trust")
        is True,
        "no_selection": report.get("candidate_selection_performed") is False,
        "nonadaptive": report.get("adaptive_direction_or_scalar_selection") is False,
        "no_authorization": report.get("candidate_authorization_granted") is False,
        "no_checkpoint": report.get("candidate_checkpoint_written") is False,
        "no_optimizer": report.get("optimizer_constructed_or_loaded") is False
        and report.get("optimizer_state_file_opened") is False
        and report.get("optimizer_step_executed") is False,
        "no_persist": report.get("parameter_state_persisted") is False,
        "no_greedy": report.get("greedy_generation_executed") is False,
        "all_maps": report.get("all_16_training_maps_loaded") is True,
        "no_validation": report.get("validation_qa_loaded") is False
        and report.get("validation_environment_maps_loaded") is False,
        "no_oracle": report.get("oracle_loaded") is False,
        "no_final": report.get("final_test_scenes_touched") is False,
        "no_selector": report.get("selector_executed") is False,
        "no_runtime": report.get("runtime_promotion_executed") is False,
        "no_chat": report.get("chat_promotion_executed") is False,
        "no_embodied": report.get("embodied_promotion_executed") is False,
        "protected": report.get("protected_report_sha256_before_and_after")
        == _PROTECTED_REPORT_SHA256,
        "forbidden_empty": report.get("forbidden_file_accesses") == [],
    }
    if not all(fixed.values()):
        raise ValueError(f"V48 fixed report envelope changed: {fixed}")

    expected_order = [(direction, alpha) for direction in _DIRECTION_IDS for alpha in _ALPHA_GRID]
    observed_order: list[tuple[str, float]] = []
    passing: list[dict[str, Any]] = []
    for index, (inventory_value, result_value) in enumerate(
        zip(inventory_rows, results, strict=True)
    ):
        inventory_row = _mapping(inventory_value, "V48 inventory row")
        result = _mapping(result_value, "V48 candidate result")
        identity_fields = (
            "candidate_id",
            "direction_id",
            "alpha",
            "authorized_surface_state_sha256",
            "full_tensor_state_sha256",
        )
        if any(result.get(field) != inventory_row.get(field) for field in identity_fields):
            raise ValueError(f"V48 candidate identity differs from inventory at {index}")
        candidate_id = str(result.get("candidate_id"))
        direction = str(result.get("direction_id"))
        alpha = _finite(result.get("alpha"), "candidate alpha")
        observed_order.append((direction, alpha))
        _lower_hex64(result.get("authorized_surface_state_sha256"), "authorized hash")
        _lower_hex64(result.get("full_tensor_state_sha256"), "full hash")
        state = _mapping(result.get("candidate_state_before_forward"), "candidate state")
        if (
            state.get("passed") is not True
            or state.get("authorized_surface_state_sha256")
            != result.get("authorized_surface_state_sha256")
            or state.get("full_tensor_state_sha256") != result.get("full_tensor_state_sha256")
            or state.get("frozen_state_sha256") != _FROZEN_SHA256
            or state.get("all_parameter_gradients_absent") is not True
            or result.get("candidate_checkpoint_written") is not False
            or result.get("candidate_authorized") is not False
        ):
            raise ValueError(f"V48 candidate state/non-authorization changed: {candidate_id}")
        pair_metrics = _mapping(result.get("pair_metrics"), "candidate pair metrics")
        per_unit = _sequence(result.get("per_unit_nll_diagnostics"), "candidate per-unit NLL")
        if pair_metrics.get("unit_count") != 25 or len(per_unit) != 25:
            raise ValueError(f"V48 candidate lacks full25 diagnostics: {candidate_id}")
        broad = _finite(result.get("broad_nll"), "candidate broad NLL")
        trust = _finite(
            result.get("candidate_relative_prefix_trust_rms"),
            "candidate prefix trust RMS",
        )
        threshold = _mapping(result.get("threshold_diagnostic"), "candidate threshold")
        computed_checks = candidate_threshold_checks(pair_metrics, broad, trust)
        deficit = _priority_deficit(pair_metrics)
        if (
            threshold.get("checks") != computed_checks
            or threshold.get("all_numeric_thresholds_met") is not all(computed_checks.values())
            or threshold.get("diagnostic_only_no_candidate_authorization") is not True
            or not math.isclose(
                _finite(threshold.get("priority_side_deficit"), "priority deficit"),
                deficit,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            )
            or not math.isclose(
                _finite(
                    threshold.get("priority_deficit_improvement_vs_original_v41_u0"),
                    "priority improvement",
                ),
                _ORIGINAL_V41_PRIORITY_DEFICIT - deficit,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            )
            or not math.isclose(
                _finite(threshold.get("broad_nll"), "threshold broad NLL"),
                broad,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            )
            or not math.isclose(
                _finite(
                    threshold.get("candidate_relative_prefix_trust_rms"),
                    "threshold prefix trust",
                ),
                trust,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            )
        ):
            raise ValueError(f"V48 candidate threshold changed: {candidate_id}")
        if all(computed_checks.values()):
            passing.append(dict(result))
    if observed_order != expected_order:
        raise ValueError("V48 candidate result order changed")

    restorations = _sequence(report.get("restoration_audit"), "V48 restorations")
    if len(restorations) != 30:
        raise ValueError("V48 restoration count changed")
    for index, result_value in enumerate(results):
        candidate_id = _mapping(result_value, "candidate").get("candidate_id")
        pair = restorations[index * 2 : index * 2 + 2]
        if [
            (
                _mapping(value, "restoration").get("candidate_id"),
                _mapping(value, "restoration").get("phase"),
            )
            for value in pair
        ] != [(candidate_id, "before"), (candidate_id, "after")]:
            raise ValueError("V48 restoration order changed")
        for value in pair:
            row = _mapping(value, "restoration")
            if (
                row.get("passed") is not True
                or row.get("full_tensor_state_sha256") != _SOURCE_FULL_SHA256
                or row.get("authorized_surface_state_sha256") != _SOURCE_AUTHORIZED_SHA256
                or row.get("frozen_state_sha256") != _FROZEN_SHA256
                or row.get("all_parameter_gradients_absent") is not True
            ):
                raise ValueError("V48 exact source restoration changed")
    final = _mapping(report.get("final_state"), "V48 final state")
    if (
        final.get("passed") is not True
        or final.get("all_15_before_after_restorations_passed") is not True
        or final.get("full_tensor_state_sha256") != _SOURCE_FULL_SHA256
        or final.get("authorized_surface_state_sha256") != _SOURCE_AUTHORIZED_SHA256
        or final.get("frozen_state_sha256") != _FROZEN_SHA256
        or final.get("all_parameter_gradients_absent") is not True
    ):
        raise ValueError("V48 final source restoration changed")
    gradient_checks = _validate_gradient_and_direction_audits(report)
    if not all(gradient_checks.values()):
        raise ValueError(f"V48 gradient/direction audit changed: {gradient_checks}")
    if len(passing) != 1:
        raise ValueError(
            f"V48 must contain exactly one teacher-threshold passing candidate; got {len(passing)}"
        )
    selected = passing[0]
    selection_checks = _authenticate_expected_candidate(selected)
    return {
        "fixed_envelope_checks": fixed,
        "gradient_direction_checks": gradient_checks,
        "candidate_count": len(results),
        "restoration_count": len(restorations),
        "teacher_threshold_passing_candidate_count": 1,
        "unique_candidate_authentication": selection_checks,
        "unique_teacher_threshold_candidate": selected,
        "v48_candidate_selection_performed": False,
        "v48_candidate_authorization_granted": False,
        "v48_candidate_checkpoint_written": False,
        "v49_terminal_required_for_any_reconstruction_or_write": True,
    }


def _authenticate_expected_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    pair = _mapping(candidate.get("pair_metrics"), "selected pair metrics")
    families = _mapping(pair.get("complete_units_by_family"), "selected families")
    cross_families = _mapping(
        pair.get("cross_prefix_complete_units_by_family"), "selected cross families"
    )
    focus = _mapping(candidate.get("focus_units"), "selected focus units")
    observed = {
        "candidate_id": candidate.get("candidate_id"),
        "direction_id": candidate.get("direction_id"),
        "alpha": candidate.get("alpha"),
        "full_tensor_state_sha256": candidate.get("full_tensor_state_sha256"),
        "authorized_surface_state_sha256": candidate.get("authorized_surface_state_sha256"),
        "complete_units": pair.get("complete_units"),
        "positive_sides": pair.get("positive_sides"),
        "cross_prefix_complete_units": pair.get("cross_prefix_complete_units"),
        "complete_physical_pair_coverage": pair.get("complete_physical_pair_coverage"),
        "mirror_complete_units": families.get("mirror_lr"),
        "book_complete_units": families.get("book_support"),
        "book_cross_prefix_complete_units": cross_families.get("book_support"),
        "priority_side_deficit": _priority_deficit(pair),
        "priority_deficit_improvement": _ORIGINAL_V41_PRIORITY_DEFICIT - _priority_deficit(pair),
        "broad_nll": candidate.get("broad_nll"),
        "v47_u4_relative_prefix_trust_rms": candidate.get("candidate_relative_prefix_trust_rms"),
        "q163_side_margins": _mapping(focus.get("cfq_163eb92339ad35a5"), "q163 focus").get(
            "side_margins"
        ),
        "q699_side_margins": _mapping(focus.get("cfq_699675ceeaf65406"), "q699 focus").get(
            "side_margins"
        ),
        "q5_side_margins": _mapping(focus.get("cfq_5c84a2c27d2be251"), "q5 focus").get(
            "side_margins"
        ),
    }
    checks: dict[str, bool] = {}
    for key, expected in _EXPECTED_SELECTION.items():
        value = observed[key]
        if isinstance(expected, float):
            checks[key] = math.isclose(
                _finite(value, key), expected, rel_tol=0.0, abs_tol=_TOLERANCE
            )
        elif isinstance(expected, list):
            checks[key] = len(value) == len(expected) and all(
                math.isclose(_finite(left, key), float(right), rel_tol=0.0, abs_tol=_TOLERANCE)
                for left, right in zip(value, expected, strict=True)
            )
        else:
            checks[key] = value == expected
    if not all(checks.values()):
        raise ValueError(f"V48 unique candidate differs from fixed result: {checks}")
    return {"passed": True, "checks": checks, "observed": observed}


def load_and_review_report(expected_v48_report_sha256: str) -> dict[str, Any]:
    digest = _lower_hex64(expected_v48_report_sha256, "expected V48 report SHA256")
    if digest != _V48_REPORT_SHA256:
        raise ValueError("V48 report SHA256 differs from the fixed reviewed result")
    _authenticate_static_inputs()
    path = _resolve(V48_REPORT)
    _locked_file(path, digest, "V48 report")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "V48 report")
    return {
        "path": str(V48_REPORT),
        "sha256": digest,
        "review": review_report_payload(report),
    }


def _v49_hashes_ready() -> bool:
    return (
        _HEX64.fullmatch(_V49_SCRIPT_SHA256) is not None
        and _HEX64.fullmatch(_V49_TEST_SHA256) is not None
    )


def _v49_implementation_review() -> dict[str, Any]:
    ready = _v49_hashes_ready()
    if ready:
        _locked_file(_resolve(V49_SCRIPT), _V49_SCRIPT_SHA256, "V49 script")
        _locked_file(_resolve(V49_TEST), _V49_TEST_SHA256, "V49 test")
    return {
        "status": (
            "exact_v49_implementation_authenticated"
            if ready
            else "pending_stable_v49_module_and_test_hashes"
        ),
        "ready": ready,
        "script": {"path": str(V49_SCRIPT), "sha256": _V49_SCRIPT_SHA256},
        "test": {"path": str(V49_TEST), "sha256": _V49_TEST_SHA256},
        "report": str(V49_REPORT),
        "conditional_checkpoint": str(V49_CHECKPOINT),
        "no_v49_file_opened_in_placeholder_mode": not ready,
    }


def v49_authorization_template() -> dict[str, Any]:
    ready = _v49_hashes_ready()
    return {
        "schema_version": 1,
        "authorization_id": _V49_AUTHORIZATION_ID,
        "authorized": ready,
        "only_exact_action": _V49_ACTION,
        "authorized_script": str(V49_SCRIPT),
        "authorized_test": str(V49_TEST),
        "authorized_report": str(V49_REPORT),
        "authorized_config": str(V49_CONFIG),
        "conditional_checkpoint_output": str(V49_CHECKPOINT),
        "explicit_terminal_sha256_cli_required": True,
        "invocation_contract": {
            "terminal_path": str(DEFAULT_OUTPUT),
            "required_cli_argument": "--expected-v48-terminal-sha256",
            "v49_must_not_embed_terminal_sha256": True,
            "v49_must_authenticate_terminal_bytes_and_exact_authorization": True,
        },
        "implementation_integrity": {
            "script_sha256": _V49_SCRIPT_SHA256,
            "test_sha256": _V49_TEST_SHA256,
            "config_sha256": _CONFIG_SHA256,
            "hashes_complete": ready,
        },
        "source": {
            "v47_u4": {
                "checkpoint": str(_SOURCE_CHECKPOINT),
                "file_sha256": dict(_SOURCE_FILES),
                "full_tensor_state_sha256": _SOURCE_FULL_SHA256,
                "authorized_surface_state_sha256": _SOURCE_AUTHORIZED_SHA256,
                "frozen_state_sha256": _FROZEN_SHA256,
                "optimizer_file_open_authorized": False,
            },
            "v48_report": {
                "path": str(V48_REPORT),
                "sha256": _V48_REPORT_SHA256,
                "candidate_inventory_sha256": _V48_INVENTORY_SHA256,
                "candidate_selection_performed": False,
                "candidate_authorization_granted": False,
                "candidate_checkpoint_written": False,
            },
            "original_v46_candidate_prefix_reference": {
                "checkpoint": str(_PREFIX_REFERENCE_CHECKPOINT),
                "file_sha256": dict(_PREFIX_REFERENCE_FILES),
                "candidate_id": "g5_both_sign_alpha_1p0",
                "full_tensor_state_sha256": _PREFIX_REFERENCE_FULL_SHA256,
                "authorized_surface_state_sha256": (_PREFIX_REFERENCE_AUTHORIZED_SHA256),
                "frozen_state_sha256": _FROZEN_SHA256,
                "scene_count": 16,
                "question_free_global_scene_prefix": True,
            },
        },
        "candidate_reconstruction": {
            "candidate_id": _CANDIDATE_ID,
            "direction_id": _CANDIDATE_DIRECTION,
            "alpha": _CANDIDATE_ALPHA,
            "isolated_side_gradient_specs": _gradient_specs(),
            "normalize_each_nonzero_component": (
                "unit_l2_within_each_scene_or_query_group_before_combination"
            ),
            "direction_components": ["g_book", "g_mirror", "g5_guard"],
            "candidate_formula": ("float32_P0-alpha*lr_group*sign(normalized_component_sum)"),
            "scene_readout_learning_rate": _SCENE_LR,
            "query_learning_rate": _QUERY_LR,
            "expected_full_tensor_state_sha256": _CANDIDATE_FULL_SHA256,
            "expected_authorized_surface_state_sha256": (_CANDIDATE_AUTHORIZED_SHA256),
            "expected_frozen_state_sha256": _FROZEN_SHA256,
            "candidate_state_authenticated_before_forward": True,
            "all_parameter_gradients_absent_before_forward": True,
        },
        "measurements": {
            "prefix_reference": "exact_original_v46_candidate_v47_update_000",
            "prefix_reference_computed_before_candidate_evaluation": True,
            "prefix_reference_scene_count": 16,
            "complete_global_scene_prefix_question_free": True,
            "full_25_unit_teacher_metrics_and_per_unit_nll": True,
            "full_fixed_48_row_broad_nll": True,
            "replay_exact_v47_final_train_gate": True,
            "non_greedy_pre_gate_evaluated_first": True,
            "full_greedy_mandatory_iff_pre_gate_passes": True,
            "greedy_skipped_due_pre_gate_required_if_failed": True,
            "pre_gate_failure_forces_final_failure_and_no_checkpoint": True,
            "non_greedy_pre_gate_must_run_before_any_greedy_generation": True,
            "greedy_evaluation_authorized_iff_all_non_greedy_pre_gate_checks_pass": True,
            "full_greedy_changed_25_units_mandatory_if_pre_gate_passes": True,
            "full_greedy_broad_48_rows_mandatory_if_pre_gate_passes": True,
            "greedy_skipped_due_pre_gate_must_be_true_if_pre_gate_fails": True,
        },
        "final_train_gate": {
            "execution_order": [
                "authenticate_source_reconstruct_candidate_and_prefix_reference",
                "run_all_non_greedy_teacher_broad_prefix_and_integrity_checks",
                "if_and_only_if_pre_gate_passes_run_full_greedy_25_and_48",
                "combine_pre_gate_and_greedy_checks_for_final_decision",
                "persist_checkpoint_if_and_only_if_final_decision_passes",
            ],
            "non_greedy_pre_gate_check_names": list(_NON_GREEDY_PRE_GATE_CHECKS),
            "greedy_final_gate_check_names": list(_GREEDY_FINAL_GATE_CHECKS),
            "pre_gate_passed_equals_all_non_greedy_checks": True,
            "pre_gate_failure_requires_greedy_skipped_due_pre_gate_true": True,
            "pre_gate_failure_forbids_any_greedy_generation": True,
            "pre_gate_failure_forces_final_gate_failure": True,
            "pre_gate_pass_requires_exhaustive_greedy_evaluation": True,
            "final_gate_passed_equals_pre_gate_passed_and_all_greedy_checks": True,
            "teacher_complete_units_minimum": 10,
            "teacher_positive_sides_minimum": 35,
            "teacher_cross_prefix_complete_units_minimum": 17,
            "complete_physical_pair_coverage_minimum": 5,
            "mirror_complete_units_minimum": 2,
            "book_complete_units_minimum": 1,
            "book_cross_prefix_complete_units_minimum": 1,
            "priority_deficit_improvement_minimum_vs_original_v41_u0": 0.5,
            "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
            "both_lost_side_margins_strictly_positive": True,
            "scene_readout_state_changed": True,
            "query_state_changed": True,
            "frozen_state_exact": True,
            "original_v46_candidate_relative_prefix_trust_rms_maximum": 0.002,
            "train_greedy_complete_units_minimum": 5,
            "broad_greedy_exact_correct_minimum": 23,
            "broad_greedy_row_count_exact": 48,
        },
        "conditional_persistence": {
            "always_write_atomic_report": True,
            "candidate_checkpoint_write_iff_every_final_gate_check_passes": True,
            "checkpoint_write_requires_non_greedy_pre_gate_passed": True,
            "checkpoint_write_requires_exhaustive_greedy_gate_passed": True,
            "failed_gate_writes_no_checkpoint": True,
            "pre_gate_failure_writes_no_checkpoint": True,
            "checkpoint_inventory_if_passed": [
                "adapter.safetensors",
                "metadata.json",
                "runtime_metadata.json",
            ],
            "optimizer_file_in_checkpoint": False,
            "runtime_metadata_exact_sanitization_required": True,
            "checkpoint_provenance_must_include_terminal_report_candidate_source_and_prefix_hashes": True,
        },
        "scope": {
            "train_only": True,
            "deterministic_single_candidate_no_selection": True,
            "exact_three_autograd_grad_probes_authorized": True,
            "backward_or_parameter_gradient_accumulation_authorized": False,
            "optimizer_construction_authorized": False,
            "optimizer_state_file_open_authorized": False,
            "optimizer_state_loading_authorized": False,
            "optimizer_step_authorized": False,
            "arbitrary_candidate_or_checkpoint_write_authorized": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
            "chat_promotion_authorized": False,
            "embodied_promotion_authorized": False,
        },
    }


def build_terminal_scaffold(expected_v48_report_sha256: str) -> dict[str, Any]:
    inputs = _authenticate_static_inputs()
    reviewed = load_and_review_report(expected_v48_report_sha256)
    implementation = _v49_implementation_review()
    ready = implementation["ready"] is True
    return {
        "schema_version": 1,
        "artifact": "v48_v47_u4_dual_margin_terminal_gate_scaffold",
        "passed": True,
        "terminal_materialization_authorized": ready,
        "input_integrity": inputs,
        "v48_report_reference": {
            "path": reviewed["path"],
            "sha256": reviewed["sha256"],
            "authenticated": True,
        },
        "v48_result_review": reviewed["review"],
        "v49_implementation_review": implementation,
        "v49_authorization_template": v49_authorization_template(),
        "only_exact_successor_authorized": _V49_AUTHORIZATION_ID if ready else None,
        "conditional_successor_authorization": (v49_authorization_template() if ready else None),
        "v48_candidate_checkpoint_write_authorized": False,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
        "chat_promotion_authorized": False,
        "embodied_promotion_authorized": False,
        "terminal_process_access_audit": {
            "model_loaded": False,
            "qa_loaded": False,
            "maps_loaded": False,
            "optimizer_state_loaded": False,
            "validation_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
            "selector_executed": False,
            "candidate_checkpoint_written": False,
            "v48_report_read_only": True,
        },
    }


def build_terminal_report(expected_v48_report_sha256: str) -> dict[str, Any]:
    scaffold = build_terminal_scaffold(expected_v48_report_sha256)
    if scaffold.get("terminal_materialization_authorized") is not True:
        raise RuntimeError("V48 terminal is pending stable V49 module and test hashes")
    authorization = _mapping(
        scaffold.get("conditional_successor_authorization"), "V49 authorization"
    )
    if authorization.get("authorized") is not True:
        raise RuntimeError("V49 authorization is not exact and complete")
    return {
        "schema_version": 1,
        "artifact": "v48_v47_u4_dual_margin_terminal_gate",
        "passed": True,
        "terminal_materialization_authorized": True,
        "terminal_conclusion": (
            "exactly_one_teacher_threshold_candidate_authenticated_v49_gate_required"
        ),
        "input_integrity": scaffold["input_integrity"],
        "v48_report_reference": scaffold["v48_report_reference"],
        "v48_result_review": scaffold["v48_result_review"],
        "v49_implementation_review": scaffold["v49_implementation_review"],
        "only_exact_successor_authorized": _V49_AUTHORIZATION_ID,
        "conditional_successor_authorization": dict(authorization),
        "v49_guarded_alpha2_train_candidate_gate_authorized": True,
        "v48_candidate_checkpoint_write_authorized": False,
        "arbitrary_candidate_or_checkpoint_write_authorized": False,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
        "chat_promotion_authorized": False,
        "embodied_promotion_authorized": False,
        "terminal_process_access_audit": scaffold["terminal_process_access_audit"],
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_report(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    expected_v48_report_sha256: str,
) -> dict[str, Any]:
    path = _resolve(output)
    if path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V48 terminal output path is pinned")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V48 terminal is one-shot and will not overwrite {path}")
    report = build_terminal_report(expected_v48_report_sha256)
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-v48-report-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            write_report(
                args.output,
                expected_v48_report_sha256=args.expected_v48_report_sha256,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V49_SCRIPT_SHA256_PLACEHOLDER",
    "V49_TEST_SHA256_PLACEHOLDER",
    "build_terminal_report",
    "build_terminal_scaffold",
    "candidate_threshold_checks",
    "load_and_review_report",
    "review_report_payload",
    "v49_authorization_template",
    "write_report",
]
