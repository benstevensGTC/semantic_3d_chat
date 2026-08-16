"""Fail-closed review of V50 and sole authorization for the V51 query grid.

This module is deliberately offline.  It authenticates the exact V50 report,
proves that the fixed scene-alpha grid completed without a winner, and seals
the scene-alpha-1/query-alpha-2 anchor that missed only one unchanged teacher
threshold.  Its only possible successor is the fixed four-candidate V51
query-only grid.  No model, map, QA, optimizer, validation, oracle, deferred
final, selector, promotion, chat, or embodied access occurs here.
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

V49_TERMINAL = Path(
    "reports/gemma4/metrics/v49_guarded_candidate_terminal_gate.json"
)
V50_SCRIPT = Path("src/semantic_3d_chat/evaluation/v50_scene_query_alpha_grid.py")
V50_TEST = Path("tests/test_v50_scene_query_alpha_grid.py")
V50_REPORT = Path("reports/gemma4/metrics/v50_scene_query_alpha_grid.json")
V50_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v50_scene_query_alpha_grid/update_000"
)
CONFIG = Path("configs/experiments/gemma4_diverse28_book_continuation_v47.yaml")
PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v50_scene_query_alpha_terminal_gate.json"
)

V51_SCRIPT = Path("src/semantic_3d_chat/evaluation/v51_query_alpha_grid.py")
V51_TEST = Path("tests/test_v51_query_alpha_grid.py")
V51_REPORT = Path("reports/gemma4/metrics/v51_query_alpha_grid.json")
V51_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v51_query_alpha_grid/update_000"
)

V51_SCRIPT_SHA256_PLACEHOLDER = "PENDING_V51_SCRIPT_SHA256"
V51_TEST_SHA256_PLACEHOLDER = "PENDING_V51_TEST_SHA256"
_V51_SCRIPT_SHA256 = "03f0fdac1cd253ed663db463d7c404e861cdb8058b36e990f85cd1852387a250"
_V51_TEST_SHA256 = "5c627612001a02da9b3ea78f79ee9489f444e6b00b9a74068d847aa77bb65c8e"

_AUTHORIZATION_ID = "v51_query_alpha_grid"
_ACTION = "one_fixed_v51_train_only_query_alpha_grid"
_V49_TERMINAL_SHA256 = (
    "4f8ac8fadca37499da3dc3e2f672956c78e126ec4f4089d98d2172f1294a0944"
)
_V50_SCRIPT_SHA256 = (
    "965868c3e620147fb23ee77a04ef2e2f51f07f2d47c9cffda0236104ebdc174a"
)
_V50_TEST_SHA256 = (
    "f8ffa0bc37d3a3b9ec4aac1d40b7e50d8d232899ac89de006f2b9d0846ad5166"
)
_V50_REPORT_SHA256 = (
    "158cedd46c73e29fc4cd5e412b6ddd260bb6be187c967c8c4489e5f610cc46f1"
)
_CONFIG_SHA256 = "6b15813237b217d8daad446c799127215bbb1366f2b442b61e975787efa4f6b7"
_PROTECTED_REPORT_SHA256 = (
    "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
)

_SOURCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query/update_004"
)
_SOURCE_FILES = {
    "adapter.safetensors": "8f903f5d1ba93d37ccd6204e3b58c9a5529ff9ee2b74edca0787ecb5a2c62c66",
    "metadata.json": "c6affe7f60c094580e2ea5f5d1330f475bf359e0a3a58bfc3bf3b3ada1de0be1",
    "optimizer.pt": "fe66be9cae13951fbfc217e0c512e43366c347181457c9e551230a9d6001db80",
    "runtime_metadata.json": "4e3a1af91642c9f2adb0b3e43997455a1aea31f86bf45618459d6005a68d4bbf",
}
_SOURCE_FULL_SHA256 = "adfc0400d1a3bb49b278cd3012ab571d01465f2380881f986c085a25474276e5"
_SOURCE_AUTHORIZED_SHA256 = (
    "a23de4988774a966c0d7aac378ede5d15a3fa1d96093c5039f181a62b0bb09b0"
)
_FROZEN_SHA256 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"
_PREFIX_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query/update_000"
)
_PREFIX_FILES = {
    "adapter.safetensors": "c47bbb9bacbb5bc8178e9a1797ec47b04ee4a3709042c6b30f6935eacc4686f0",
    "metadata.json": "e76e8a905af53fb082684000a6a6e16845b79e0795b6dbb047a2703245198574",
    "runtime_metadata.json": "01e645b82c5e533dd2319ef8a97171437b149c4e1ef86201f83fdb22de047987",
}
_PREFIX_FULL_SHA256 = "1d5adc1fb0d7a895056b77d38c8a12aba95c9997ec8a94edf68673f9c58fb954"
_PREFIX_AUTHORIZED_SHA256 = (
    "d60b665d9a970433b2ed59e6769b9114468bef608b98eae828268101d39db56c"
)

_FAILED_CHECK = "teacher_positive_sides_at_least_35"
_ANCHOR_ID = "guarded_scene_alpha_1p0_query_alpha_2p0"
_ANCHOR_PREFIX_RMS = 0.0016845178324729204
_PREFIX_RMS_TOLERANCE = 1.0e-12
_BROAD_NLL_MAXIMUM = 2.9213306349515915
_SCENE_ALPHA = 1.0
_QUERY_ALPHA_GRID = (1.75, 2.25, 1.5, 2.5)
_NONSELECTABLE_QUERY_ALPHA_ANCHOR = 2.0
_SCENE_LR = 1.0e-5
_QUERY_LR = 8.0e-6
_GRADIENT_SPECS = (
    ("g_book", "pair_000015", "cfq_163eb92339ad35a5", 0),
    ("g_mirror", "pair_000016", "cfq_699675ceeaf65406", 1),
    ("g5_guard", "pair_000006", "cfq_5c84a2c27d2be251", 0),
)
_NON_GREEDY_CHECKS = (
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
_GREEDY_CHECKS = (
    "train_greedy_complete_units_at_least_5",
    "broad_greedy_exact_correct_at_least_23_of_48",
    "broad_greedy_row_count_exactly_48",
)
_V50_CANDIDATES = (
    ("guarded_scene_alpha_1p0_query_alpha_2p0", 1.0),
    ("guarded_scene_alpha_0p5_query_alpha_2p0", 0.5),
    ("guarded_scene_alpha_0p25_query_alpha_2p0", 0.25),
)
_EXPECTED_FAILED_CHECKS = {
    _V50_CANDIDATES[0][0]: {_FAILED_CHECK},
    _V50_CANDIDATES[1][0]: {
        "book_cross_prefix_complete_units_at_least_1",
        "both_lost_sides_strictly_positive",
        "broad_nll_at_most_v45_maximum",
        "complete_physical_pair_coverage_at_least_5",
        "teacher_complete_units_at_least_10",
        "teacher_positive_sides_at_least_35",
    },
    _V50_CANDIDATES[2][0]: {
        "book_complete_units_at_least_1",
        "book_cross_prefix_complete_units_at_least_1",
        "both_lost_sides_strictly_positive",
        "broad_nll_at_most_v45_maximum",
        "complete_physical_pair_coverage_at_least_5",
        "mirror_complete_units_at_least_2",
        "teacher_complete_units_at_least_10",
        "teacher_positive_sides_at_least_35",
    },
}
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    combined = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(combined))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_file(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{field} changed: expected {expected}, observed {observed}")


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


def _close(observed: object, expected: float, field: str, tolerance: float = 1e-12) -> None:
    if abs(_finite(observed, field) - expected) > tolerance:
        raise ValueError(f"{field} changed")


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


def _v51_candidates() -> list[dict[str, Any]]:
    def alpha_id(value: float) -> str:
        return str(value).replace(".", "p")

    return [
        {
            "candidate_id": f"guarded_scene_alpha_1p0_query_alpha_{alpha_id(alpha)}",
            "declared_order": index,
            "scene_alpha": _SCENE_ALPHA,
            "query_alpha": alpha,
        }
        for index, alpha in enumerate(_QUERY_ALPHA_GRID)
    ]


def _authenticate_static_inputs() -> dict[str, Any]:
    fixed = {
        str(V49_TERMINAL): _V49_TERMINAL_SHA256,
        str(V50_SCRIPT): _V50_SCRIPT_SHA256,
        str(V50_TEST): _V50_TEST_SHA256,
        str(V50_REPORT): _V50_REPORT_SHA256,
        str(CONFIG): _CONFIG_SHA256,
        str(PROTECTED_REPORT): _PROTECTED_REPORT_SHA256,
    }
    for name, digest in fixed.items():
        _locked_file(_resolve(name), digest, name)
    return {"file_sha256": fixed, "all_exact": True}


def review_report_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    authorization = _mapping(report.get("authorization"), "V50 authorization")
    preparation = _mapping(report.get("preparation"), "V50 preparation")
    grid = _mapping(report.get("candidate_grid"), "V50 grid")
    selection = _mapping(report.get("selection"), "V50 selection")
    final = _mapping(report.get("final_train_gate"), "V50 final gate")
    checkpoint = _mapping(report.get("checkpoint"), "V50 checkpoint")
    restoration = _mapping(report.get("final_source_restoration"), "V50 restoration")
    access = _mapping(report.get("access_audit"), "V50 access")
    rows = _sequence(grid.get("candidates"), "V50 candidates")
    if (
        report.get("schema_version") != 1
        or report.get("artifact") != "v50_scene_query_alpha_grid"
        or report.get("passed") is not False
        or authorization.get("authorization_id") != "v50_scene_query_alpha_grid"
        or authorization.get("terminal_sha256") != _V49_TERMINAL_SHA256
        or not all(_mapping(authorization.get("checks"), "V50 auth checks").values())
        or preparation.get("candidate_count") != 3
        or preparation.get("all_16_training_maps_cached") is not True
        or preparation.get("exact_three_autograd_grad_probes_reused_for_all_candidates")
        is not True
        or preparation.get("optimizer_constructed_or_loaded") is not False
        or grid.get("declared_count") != 3
        or grid.get("evaluated_count") != 3
        or grid.get("complete_fixed_grid_evaluated_before_selection") is not True
        or len(rows) != 3
        or selection.get("performed_after_complete_grid") is not True
        or selection.get("passing_candidate_ids") != []
        or selection.get("winner") is not None
        or final.get("passed") is not False
        or final.get("grid_complete") is not True
        or final.get("evaluation_complete") is not True
        or final.get("winner_exists") is not False
        or final.get("source_restored_exact") is not True
        or final.get("access_audit_passed") is not True
        or final.get("execution_errors") != []
        or checkpoint.get("written") is not False
        or checkpoint.get("inventory") is not None
        or checkpoint.get("optimizer_file_written") is not False
        or checkpoint.get("staged_after_complete_grid") is not False
        or restoration.get("passed") is not True
        or restoration.get("full_tensor_state_sha256") != _SOURCE_FULL_SHA256
        or restoration.get("authorized_surface_state_sha256")
        != _SOURCE_AUTHORIZED_SHA256
        or restoration.get("frozen_state_sha256") != _FROZEN_SHA256
        or access.get("passed") is not True
        or access.get("training_map_count") != 16
        or access.get("optimizer_file_reads") != []
        or access.get("forbidden_file_accesses") != []
        or access.get("validation_qa_loaded") is not False
        or access.get("oracle_loaded") is not False
        or access.get("final_test_loaded") is not False
    ):
        raise ValueError("V50 fixed report envelope changed")
    expected_ids = [candidate_id for candidate_id, _ in _V50_CANDIDATES]
    if grid.get("evaluated_ids") != expected_ids:
        raise ValueError("V50 candidate evaluation order changed")
    reviews: list[dict[str, Any]] = []
    for index, value in enumerate(rows):
        row = _mapping(value, f"V50 candidate {index}")
        candidate = _mapping(row.get("candidate"), f"V50 candidate identity {index}")
        candidate_id, scene_alpha = _V50_CANDIDATES[index]
        non_greedy = _mapping(row.get("non_greedy_pre_gate"), "V50 non-greedy")
        checks = _mapping(non_greedy.get("checks"), "V50 checks")
        evidence = _mapping(non_greedy.get("evidence"), "V50 evidence")
        greedy = _mapping(row.get("greedy_gate"), "V50 greedy")
        row_restore = _mapping(row.get("source_restoration"), "V50 row restoration")
        failed = {name for name, passed in checks.items() if passed is not True}
        if (
            candidate
            != {
                "candidate_id": candidate_id,
                "declared_order": index,
                "scene_alpha": scene_alpha,
                "query_alpha": 2.0,
            }
            or set(checks) != set(_NON_GREEDY_CHECKS)
            or failed != _EXPECTED_FAILED_CHECKS[candidate_id]
            or non_greedy.get("evaluated") is not True
            or non_greedy.get("passed") is not False
            or greedy.get("authorized") is not False
            or greedy.get("executed") is not False
            or greedy.get("skipped_due_pre_gate") is not True
            or greedy.get("checks") != {}
            or greedy.get("evidence") is not None
            or row_restore.get("passed") is not True
            or row.get("evaluation_error") is not None
            or row.get("full_gate_passed") is not False
            or evidence.get("unit_count") != 25
            or evidence.get("per_unit_nll_row_count") != 25
            or evidence.get("broad_row_count") != 48
        ):
            raise ValueError(f"V50 candidate result changed: {candidate_id}")
        reviews.append(
            {
                "candidate_id": candidate_id,
                "failed_checks": sorted(failed),
                "complete_units": int(evidence["complete_units"]),
                "positive_sides": int(evidence["positive_sides"]),
                "cross_prefix_complete_units": int(
                    evidence["cross_prefix_complete_units"]
                ),
                "complete_physical_pair_coverage": int(
                    evidence["complete_physical_pair_coverage"]
                ),
                "broad_nll": _finite(evidence["broad_nll"], "V50 broad NLL"),
                "prefix_rms": _finite(
                    evidence["original_v46_candidate_relative_prefix_trust_rms"],
                    "V50 prefix RMS",
                ),
            }
        )
    anchor = reviews[0]
    if (
        anchor["failed_checks"] != [_FAILED_CHECK]
        or anchor["complete_units"] != 10
        or anchor["positive_sides"] != 34
        or anchor["cross_prefix_complete_units"] != 19
        or anchor["complete_physical_pair_coverage"] != 5
    ):
        raise ValueError("V50 query-alpha-two anchor changed")
    _close(anchor["broad_nll"], 2.9192593644062677, "V50 anchor broad NLL")
    _close(anchor["prefix_rms"], _ANCHOR_PREFIX_RMS, "V50 anchor prefix RMS")
    anchor_row = _mapping(rows[0], "V50 anchor row")
    anchor_evidence = _mapping(
        _mapping(anchor_row["non_greedy_pre_gate"], "V50 anchor pre-gate")["evidence"],
        "V50 anchor evidence",
    )
    if (
        _mapping(anchor_evidence.get("complete_units_by_family"), "V50 anchor families")
        != {"book_support": 1, "mirror_lr": 2, "picture_support": 0}
        or _mapping(
            anchor_evidence.get("cross_prefix_complete_units_by_family"),
            "V50 anchor cross families",
        )
        != {"book_support": 2, "mirror_lr": 4, "picture_support": 3}
    ):
        raise ValueError("V50 anchor family coverage changed")
    for field in (
        "optimizer_constructed_or_loaded",
        "optimizer_state_file_opened",
        "optimizer_step_executed",
        "validation_qa_loaded",
        "validation_environment_maps_loaded",
        "oracle_loaded",
        "final_test_scenes_touched",
        "selector_executed",
        "runtime_promotion_executed",
        "chat_promotion_executed",
        "embodied_promotion_executed",
        "question_dependent_retrieval",
    ):
        if report.get(field) is not False:
            raise ValueError(f"V50 forbidden scope field changed: {field}")
    if _resolve(V50_CHECKPOINT).exists():
        raise ValueError("V50 failed result unexpectedly has a checkpoint")
    return {
        "candidate_count": 3,
        "candidate_reviews": reviews,
        "anchor_candidate_id": _ANCHOR_ID,
        "anchor_only_failed_check": _FAILED_CHECK,
        "anchor_positive_sides": 34,
        "anchor_prefix_rms": _ANCHOR_PREFIX_RMS,
        "greedy_executed": False,
        "checkpoint_written": False,
        "source_restored_exact": True,
        "access_audit_passed": True,
    }


def load_and_review_report(expected_sha256: str) -> dict[str, Any]:
    if expected_sha256 != _V50_REPORT_SHA256:
        raise ValueError("V50 report SHA256 differs from the fixed result")
    _authenticate_static_inputs()
    path = _resolve(V50_REPORT)
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "V50 report")
    return {
        "path": str(V50_REPORT),
        "sha256": expected_sha256,
        "review": review_report_payload(report),
    }


def _hashes_ready() -> bool:
    return (
        _HEX64.fullmatch(_V51_SCRIPT_SHA256) is not None
        and _HEX64.fullmatch(_V51_TEST_SHA256) is not None
    )


def _implementation_review() -> dict[str, Any]:
    ready = _hashes_ready()
    if ready:
        _locked_file(_resolve(V51_SCRIPT), _V51_SCRIPT_SHA256, "V51 script")
        _locked_file(_resolve(V51_TEST), _V51_TEST_SHA256, "V51 test")
    return {
        "ready": ready,
        "status": (
            "exact_v51_implementation_authenticated"
            if ready
            else "pending_stable_v51_module_and_test_hashes"
        ),
        "script": {"path": str(V51_SCRIPT), "sha256": _V51_SCRIPT_SHA256},
        "test": {"path": str(V51_TEST), "sha256": _V51_TEST_SHA256},
        "no_v51_file_opened_in_placeholder_mode": not ready,
    }


def v51_authorization_template() -> dict[str, Any]:
    ready = _hashes_ready()
    return {
        "schema_version": 1,
        "authorization_id": _AUTHORIZATION_ID,
        "authorized": ready,
        "only_exact_action": _ACTION,
        "authorized_script": str(V51_SCRIPT),
        "authorized_test": str(V51_TEST),
        "authorized_report": str(V51_REPORT),
        "authorized_config": str(CONFIG),
        "conditional_checkpoint_output": str(V51_CHECKPOINT),
        "explicit_terminal_sha256_cli_required": True,
        "invocation_contract": {
            "terminal_path": str(DEFAULT_OUTPUT),
            "required_cli_argument": "--expected-v50-terminal-sha256",
            "v51_must_not_embed_terminal_sha256": True,
            "v51_must_authenticate_terminal_bytes_and_exact_authorization": True,
        },
        "implementation_integrity": {
            "script_sha256": _V51_SCRIPT_SHA256,
            "test_sha256": _V51_TEST_SHA256,
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
            "original_v46_candidate_prefix_reference": {
                "checkpoint": str(_PREFIX_CHECKPOINT),
                "file_sha256": dict(_PREFIX_FILES),
                "full_tensor_state_sha256": _PREFIX_FULL_SHA256,
                "authorized_surface_state_sha256": _PREFIX_AUTHORIZED_SHA256,
                "frozen_state_sha256": _FROZEN_SHA256,
                "scene_count": 16,
                "question_free_global_scene_prefix": True,
            },
            "v50_query_alpha_two_anchor": {
                "path": str(V50_REPORT),
                "sha256": _V50_REPORT_SHA256,
                "candidate_id": _ANCHOR_ID,
                "scene_alpha": _SCENE_ALPHA,
                "query_alpha": _NONSELECTABLE_QUERY_ALPHA_ANCHOR,
                "nonselectable": True,
                "only_failed_check": _FAILED_CHECK,
                "positive_sides": 34,
                "complete_units": 10,
                "cross_prefix_complete_units": 19,
                "complete_physical_pair_coverage": 5,
                "broad_nll": 2.9192593644062677,
                "original_prefix_trust_rms": _ANCHOR_PREFIX_RMS,
                "greedy_executed": False,
                "checkpoint_written": False,
                "source_restored_exact": True,
                "access_audit_passed": True,
            },
        },
        "candidate_grid": {
            "fixed_before_any_candidate_forward": True,
            "candidate_count": 4,
            "direction_id": "guarded_both_sign",
            "direction_components": ["g_book", "g_mirror", "g5_guard"],
            "isolated_side_gradient_specs": _gradient_specs(),
            "normalize_each_nonzero_component": (
                "unit_l2_within_each_scene_or_query_group_before_combination"
            ),
            "scene_alpha_fixed": _SCENE_ALPHA,
            "query_alpha_grid_declared_order": list(_QUERY_ALPHA_GRID),
            "query_alpha_two_anchor_nonselectable": True,
            "scene_readout_learning_rate": _SCENE_LR,
            "query_learning_rate": _QUERY_LR,
            "candidate_formula": (
                "float32_P0-alpha_group*lr_group*sign(normalized_component_sum)"
            ),
            "candidates_declared_order": _v51_candidates(),
            "reconstruct_each_candidate_directly_from_v47_u4": True,
            "candidate_hash_inventory_fixed_before_any_candidate_forward": True,
            "all_candidate_scene_tensors_bit_identical": True,
            "all_candidate_scene_prefixes_bit_identical": True,
            "expected_original_prefix_trust_rms": _ANCHOR_PREFIX_RMS,
            "prefix_rms_absolute_tolerance": _PREFIX_RMS_TOLERANCE,
            "adaptive_grid_or_candidate_mutation": False,
            "exact_source_restoration_before_and_after_each_candidate": True,
        },
        "evaluation_and_selection": {
            "all_candidates_receive_full_non_greedy_gate_before_selection": True,
            "all_candidates_receive_full_25_unit_teacher_metrics_and_per_unit_nll": True,
            "all_candidates_receive_full_fixed_48_row_broad_nll": True,
            "greedy_runs_for_every_pre_gate_passing_candidate": True,
            "greedy_runs_only_for_pre_gate_passing_candidates": True,
            "pre_gate_failure_requires_greedy_skipped_due_pre_gate": True,
            "every_pre_gate_passing_candidate_receives_full_changed_25_greedy": True,
            "every_pre_gate_passing_candidate_receives_full_broad_48_greedy": True,
            "winner_eligibility_requires_every_non_greedy_and_greedy_check": True,
            "winner_selection": (
                "first_full_gate_passing_candidate_in_fixed_declared_order_after_all_evaluations"
            ),
            "no_winner_if_no_candidate_passes_full_gate": True,
            "no_early_stop_after_first_full_gate_pass": True,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
        },
        "per_candidate_gate": {
            "non_greedy_check_names": list(_NON_GREEDY_CHECKS),
            "greedy_check_names": list(_GREEDY_CHECKS),
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
            "thresholds_unchanged_from_v49": True,
        },
        "conditional_persistence": {
            "always_write_atomic_report": True,
            "checkpoint_write_iff_full_gate_winner_exists": True,
            "checkpoint_contains_declared_order_winner_only": True,
            "failed_grid_writes_no_checkpoint": True,
            "checkpoint_inventory_if_passed": [
                "adapter.safetensors",
                "metadata.json",
                "runtime_metadata.json",
            ],
            "optimizer_file_in_checkpoint": False,
            "runtime_metadata_exact_sanitization_required": True,
            "checkpoint_provenance_must_bind_terminal_report_source_grid_and_winner": True,
        },
        "scope": {
            "train_only": True,
            "fixed_four_candidate_query_grid": True,
            "exact_three_autograd_grad_probes_reused_for_all_candidates": True,
            "backward_or_parameter_gradient_accumulation_authorized": False,
            "optimizer_construction_authorized": False,
            "optimizer_state_file_open_authorized": False,
            "optimizer_state_loading_authorized": False,
            "optimizer_step_authorized": False,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
            "chat_promotion_authorized": False,
            "embodied_promotion_authorized": False,
        },
    }


def build_terminal_scaffold(expected_v50_report_sha256: str) -> dict[str, Any]:
    inputs = _authenticate_static_inputs()
    reviewed = load_and_review_report(expected_v50_report_sha256)
    implementation = _implementation_review()
    ready = implementation["ready"] is True
    authorization = v51_authorization_template()
    return {
        "schema_version": 1,
        "artifact": "v50_scene_query_alpha_terminal_gate_scaffold",
        "passed": True,
        "terminal_materialization_authorized": ready,
        "input_integrity": inputs,
        "v50_report_reference": {
            "path": reviewed["path"],
            "sha256": reviewed["sha256"],
            "authenticated": True,
        },
        "v50_result_review": reviewed["review"],
        "v51_implementation_review": implementation,
        "v51_authorization_template": authorization,
        "only_exact_successor_authorized": _AUTHORIZATION_ID if ready else None,
        "conditional_successor_authorization": authorization if ready else None,
        "v50_checkpoint_write_authorized": False,
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
            "candidate_checkpoint_written": False,
            "v50_report_read_only": True,
        },
    }


def build_terminal_report(expected_v50_report_sha256: str) -> dict[str, Any]:
    scaffold = build_terminal_scaffold(expected_v50_report_sha256)
    if scaffold["terminal_materialization_authorized"] is not True:
        raise RuntimeError("V50 terminal is pending stable V51 module and test hashes")
    authorization = _mapping(
        scaffold["conditional_successor_authorization"], "V51 authorization"
    )
    if authorization.get("authorized") is not True:
        raise RuntimeError("V51 authorization is incomplete")
    return {
        "schema_version": 1,
        "artifact": "v50_scene_query_alpha_terminal_gate",
        "passed": True,
        "terminal_materialization_authorized": True,
        "terminal_conclusion": "v50_anchor_failed_only_positive_side_v51_query_grid_required",
        "input_integrity": scaffold["input_integrity"],
        "v50_report_reference": scaffold["v50_report_reference"],
        "v50_result_review": scaffold["v50_result_review"],
        "v51_implementation_review": scaffold["v51_implementation_review"],
        "only_exact_successor_authorized": _AUTHORIZATION_ID,
        "conditional_successor_authorization": dict(authorization),
        "v51_query_alpha_grid_authorized": True,
        "v50_checkpoint_write_authorized": False,
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
    output: str | Path = DEFAULT_OUTPUT, *, expected_v50_report_sha256: str
) -> dict[str, Any]:
    path = _resolve(output)
    if path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V50 terminal output path is pinned")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V50 terminal is one-shot and will not overwrite {path}")
    report = build_terminal_report(expected_v50_report_sha256)
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-v50-report-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    scaffold = build_terminal_scaffold(args.expected_v50_report_sha256)
    if scaffold["terminal_materialization_authorized"] is not True:
        print(
            json.dumps(
                {
                    "artifact": scaffold["artifact"],
                    "passed": True,
                    "terminal_materialization_authorized": False,
                    "v51_status": scaffold["v51_implementation_review"]["status"],
                },
                sort_keys=True,
            )
        )
        return 2
    report = write_report(
        args.output, expected_v50_report_sha256=args.expected_v50_report_sha256
    )
    print(
        json.dumps(
            {
                "artifact": report["artifact"],
                "passed": report["passed"],
                "successor": report["only_exact_successor_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V51_SCRIPT_SHA256_PLACEHOLDER",
    "V51_TEST_SHA256_PLACEHOLDER",
    "build_terminal_report",
    "build_terminal_scaffold",
    "load_and_review_report",
    "review_report_payload",
    "v51_authorization_template",
    "write_report",
]
