"""Seal V49's trust-radius failure and bound the sole V50 successor.

This module is an offline, fail-closed review of the exact V49 train-only
report.  V49 reconstructed the one V48-authorized candidate and passed every
non-greedy check except the original-V46-candidate-relative prefix trust
radius: ``0.0020444965921342373 > 0.002``.  Its staged contract therefore
skipped greedy generation, restored the exact V47-u4 source, and wrote no
checkpoint.

The only possible successor is a fixed three-candidate V50 scene/query alpha
grid.  Every candidate receives the complete train-only non-greedy gate;
exhaustive greedy evaluation runs for every (and only every) pre-gate passing
candidate; a declared-order winner is chosen only after all authorized
measurements finish.  No optimizer, validation, oracle, deferred-final,
selector, promotion, chat, or embodied access is authorized.  Until stable
V50 module and test hashes replace the placeholders below, this module can
build only a non-materializable scaffold and never opens either V50 file.
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

V48_TERMINAL = Path("reports/gemma4/metrics/v48_v47_u4_dual_margin_terminal_gate.json")
V49_SCRIPT = Path("src/semantic_3d_chat/evaluation/v49_guarded_candidate_greedy_gate.py")
V49_TEST = Path("tests/test_v49_guarded_candidate_greedy_gate.py")
V49_REPORT = Path("reports/gemma4/metrics/v49_guarded_candidate_train_gate.json")
V49_CONFIG = Path("configs/experiments/gemma4_diverse28_book_continuation_v47.yaml")
V49_CHECKPOINT = Path("data_gemma4/checkpoints/gemma4_v49_guarded_alpha2_train_gate/update_000")
PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v49_guarded_candidate_terminal_gate.json")

V50_SCRIPT = Path("src/semantic_3d_chat/evaluation/v50_scene_query_alpha_grid.py")
V50_TEST = Path("tests/test_v50_scene_query_alpha_grid.py")
V50_REPORT = Path("reports/gemma4/metrics/v50_scene_query_alpha_grid.json")
V50_CHECKPOINT = Path("data_gemma4/checkpoints/gemma4_v50_scene_query_alpha_grid/update_000")
V50_CONFIG = V49_CONFIG

V50_SCRIPT_SHA256_PLACEHOLDER = "PENDING_V50_SCRIPT_SHA256"
V50_TEST_SHA256_PLACEHOLDER = "PENDING_V50_TEST_SHA256"
_V50_SCRIPT_SHA256 = "965868c3e620147fb23ee77a04ef2e2f51f07f2d47c9cffda0236104ebdc174a"
_V50_TEST_SHA256 = "f8ffa0bc37d3a3b9ec4aac1d40b7e50d8d232899ac89de006f2b9d0846ad5166"

_V50_AUTHORIZATION_ID = "v50_scene_query_alpha_grid"
_V50_ACTION = "one_fixed_v50_train_only_scene_query_alpha_grid"
_V48_TERMINAL_SHA256 = "406becf0ec94c7b59ffa0acdef6441dab224d279b2cdccc86645c6048a8ef6f3"
_V49_SCRIPT_SHA256 = "8358e1326eab30292829e0d6978dfbf93d5bc29d6dab96f595e0e80b5eb52854"
_V49_TEST_SHA256 = "46f6cba7354aaa360e8be64a0d0968926fb384031fc54e4c0cb4158bd2960603"
_V49_REPORT_SHA256 = "7d82a503a5402dcfd80816459eea5d653849e89a49fa8ce7b585dc806ff7acc9"
_CONFIG_SHA256 = "6b15813237b217d8daad446c799127215bbb1366f2b442b61e975787efa4f6b7"
_PROTECTED_REPORT_SHA256 = "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"

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
_SOURCE_AUTHORIZED_SHA256 = "a23de4988774a966c0d7aac378ede5d15a3fa1d96093c5039f181a62b0bb09b0"
_FROZEN_SHA256 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"

_PREFIX_REFERENCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query/update_000"
)
_PREFIX_REFERENCE_FILES = {
    "adapter.safetensors": "c47bbb9bacbb5bc8178e9a1797ec47b04ee4a3709042c6b30f6935eacc4686f0",
    "metadata.json": "e76e8a905af53fb082684000a6a6e16845b79e0795b6dbb047a2703245198574",
    "runtime_metadata.json": "01e645b82c5e533dd2319ef8a97171437b149c4e1ef86201f83fdb22de047987",
}
_PREFIX_REFERENCE_FULL_SHA256 = "1d5adc1fb0d7a895056b77d38c8a12aba95c9997ec8a94edf68673f9c58fb954"
_PREFIX_REFERENCE_AUTHORIZED_SHA256 = (
    "d60b665d9a970433b2ed59e6769b9114468bef608b98eae828268101d39db56c"
)

_CANDIDATE_ID = "guarded_both_sign_alpha_2p0"
_CANDIDATE_FULL_SHA256 = "69c5471d141ab56397969e2aac5f1097676f6ea328a3d5577e816c3aef6f3387"
_CANDIDATE_AUTHORIZED_SHA256 = "f43e1b2a84006c8188adeff9f206ba864ae3d51b6567cad679f6d2e5f3610cf2"
_FAILED_CHECK = "original_v46_candidate_relative_prefix_trust_rms_at_most_0_002"
_OBSERVED_PREFIX_TRUST_RMS = 0.0020444965921342373
_PREFIX_TRUST_RMS_MAXIMUM = 0.002
_BROAD_NLL_MAXIMUM = 2.9213306349515915
_ORIGINAL_V41_PRIORITY_DEFICIT = 31.113729119300842
_SCENE_LR = 1.0e-5
_QUERY_LR = 8.0e-6
_SCENE_ALPHA_GRID = (1.0, 0.5, 0.25)
_QUERY_ALPHA = 2.0
_GRADIENT_SPECS = (
    ("g_book", "pair_000015", "cfq_163eb92339ad35a5", 0),
    ("g_mirror", "pair_000016", "cfq_699675ceeaf65406", 1),
    ("g5_guard", "pair_000006", "cfq_5c84a2c27d2be251", 0),
)
_CANDIDATE_IDS = (
    "guarded_scene_alpha_1p0_query_alpha_2p0",
    "guarded_scene_alpha_0p5_query_alpha_2p0",
    "guarded_scene_alpha_0p25_query_alpha_2p0",
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
    _FAILED_CHECK,
)
_GREEDY_CHECKS = (
    "train_greedy_complete_units_at_least_5",
    "broad_greedy_exact_correct_at_least_23_of_48",
    "broad_greedy_row_count_exactly_48",
)
_AUTHORIZATION_CHECKS = frozenset(
    {
        "action",
        "artifact",
        "authorization_id",
        "authorized",
        "candidate",
        "checkpoint",
        "cli",
        "config",
        "config_hash",
        "explicit_sha",
        "greedy_iff",
        "greedy_skip",
        "hashes_complete",
        "no_embedded_sha",
        "passed",
        "persistence",
        "pre_gate_contract",
        "pre_gate_failure",
        "pre_gate_first",
        "prefix_authorized",
        "prefix_checkpoint",
        "prefix_files",
        "prefix_frozen",
        "prefix_full",
        "prefix_scope",
        "report",
        "scope",
        "script",
        "script_hash",
        "source_authorized",
        "source_checkpoint",
        "source_files",
        "source_frozen",
        "source_full",
        "source_optimizer_forbidden",
        "successor",
        "terminal_auth",
        "terminal_path",
        "terminal_ready",
        "test",
        "test_hash",
        "thresholds",
        "v48_report",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOLERANCE = 1.0e-9


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    combined = value if value.is_absolute() else PROJECT_ROOT / value
    # Do not resolve the final component: authenticated reads reject symlinks.
    return Path(os.path.abspath(combined))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _locked_file(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{field} bytes changed: expected {expected}, observed {observed}")


def _isclose(value: object, expected: float, field: str) -> bool:
    return math.isclose(
        _finite(value, field), expected, rel_tol=0.0, abs_tol=_TOLERANCE
    )


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
        V48_TERMINAL: _V48_TERMINAL_SHA256,
        V49_SCRIPT: _V49_SCRIPT_SHA256,
        V49_TEST: _V49_TEST_SHA256,
        V49_CONFIG: _CONFIG_SHA256,
        PROTECTED_REPORT: _PROTECTED_REPORT_SHA256,
    }
    for path, expected in pins.items():
        _locked_file(_resolve(path), expected, str(path))

    terminal = _mapping(
        json.loads(_resolve(V48_TERMINAL).read_text(encoding="utf-8")),
        "V48 terminal",
    )
    authorization = _mapping(
        terminal.get("conditional_successor_authorization"),
        "V49 authorization",
    )
    integrity = _mapping(authorization.get("implementation_integrity"), "V49 integrity")
    scope = _mapping(authorization.get("scope"), "V49 scope")
    checks = {
        "artifact": terminal.get("artifact") == "v48_v47_u4_dual_margin_terminal_gate",
        "passed": terminal.get("passed") is True,
        "successor": terminal.get("only_exact_successor_authorized")
        == "v49_guarded_alpha2_train_candidate_gate",
        "authorization_id": authorization.get("authorization_id")
        == "v49_guarded_alpha2_train_candidate_gate",
        "authorized": authorization.get("authorized") is True,
        "script": authorization.get("authorized_script") == str(V49_SCRIPT),
        "test": authorization.get("authorized_test") == str(V49_TEST),
        "report": authorization.get("authorized_report") == str(V49_REPORT),
        "checkpoint": authorization.get("conditional_checkpoint_output")
        == str(V49_CHECKPOINT),
        "script_hash": integrity.get("script_sha256") == _V49_SCRIPT_SHA256,
        "test_hash": integrity.get("test_sha256") == _V49_TEST_SHA256,
        "config_hash": integrity.get("config_sha256") == _CONFIG_SHA256,
        "scope": scope.get("train_only") is True
        and scope.get("validation_access_authorized") is False
        and scope.get("oracle_access_authorized") is False
        and scope.get("final_test_access_authorized") is False
        and scope.get("optimizer_construction_authorized") is False
        and scope.get("optimizer_state_file_open_authorized") is False
        and scope.get("optimizer_step_authorized") is False
        and scope.get("runtime_promotion_authorized") is False
        and scope.get("chat_promotion_authorized") is False
        and scope.get("embodied_promotion_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"V49 predecessor authorization changed: {checks}")
    if _resolve(V49_CHECKPOINT).exists() or _resolve(V49_CHECKPOINT).is_symlink():
        raise ValueError("V49 failed gate must not have a checkpoint")
    return {
        "v48_terminal": {"path": str(V48_TERMINAL), "sha256": _V48_TERMINAL_SHA256},
        "v49_script": {"path": str(V49_SCRIPT), "sha256": _V49_SCRIPT_SHA256},
        "v49_test": {"path": str(V49_TEST), "sha256": _V49_TEST_SHA256},
        "v49_config": {"path": str(V49_CONFIG), "sha256": _CONFIG_SHA256},
        "protected_report": {
            "path": str(PROTECTED_REPORT),
            "sha256": _PROTECTED_REPORT_SHA256,
        },
        "authorization_checks": checks,
        "v49_checkpoint_absent": True,
        "model_loaded": False,
        "qa_loaded": False,
        "maps_loaded": False,
    }


def _review_reconstruction(value: object) -> dict[str, bool]:
    candidate = _mapping(value, "V49 candidate reconstruction")
    source = _mapping(candidate.get("source_audit"), "V49 source audit")
    prefix = _mapping(candidate.get("prefix_reference"), "V49 prefix reference")
    qa = _mapping(candidate.get("qa_audit"), "V49 QA audit")
    loader = _mapping(candidate.get("loader_transition"), "V49 loader transition")
    readable_source = {key: value for key, value in _SOURCE_FILES.items() if key != "optimizer.pt"}
    train_scenes = [
        *(f"scene_{index:06d}" for index in range(11, 19)),
        *(f"scene_{index:06d}" for index in range(31, 39)),
    ]
    checks = {
        "identity": candidate.get("candidate_id") == _CANDIDATE_ID
        and candidate.get("direction_id") == "guarded_both_sign"
        and candidate.get("alpha") == 2.0,
        "hashes": candidate.get("full_tensor_state_sha256") == _CANDIDATE_FULL_SHA256
        and candidate.get("authorized_surface_state_sha256")
        == _CANDIDATE_AUTHORIZED_SHA256
        and candidate.get("frozen_state_sha256") == _FROZEN_SHA256,
        "single_no_selection": candidate.get("single_candidate_reconstructed") is True
        and candidate.get("candidate_selection_performed") is False,
        "state": candidate.get("source_v47_u4_exact_before_reconstruction") is True
        and candidate.get("scene_readout_state_changed") is True
        and candidate.get("query_state_changed") is True
        and candidate.get("all_parameter_gradients_absent_before_forward") is True,
        "maps": candidate.get("all_16_training_maps_cached") is True,
        "source": source.get("checkpoint") == str(_SOURCE_CHECKPOINT)
        and source.get("directory_inventory")
        == ["adapter.safetensors", "metadata.json", "optimizer.pt", "runtime_metadata.json"]
        and source.get("readable_file_sha256") == readable_source
        and source.get("optimizer_file_sha256_provenance") == _SOURCE_FILES["optimizer.pt"]
        and source.get("optimizer_file_opened") is False
        and source.get("optimizer_state_deserialized") is False
        and source.get("optimizer_state_loaded") is False
        and source.get("full_tensor_state_sha256") == _SOURCE_FULL_SHA256
        and source.get("authorized_surface_state_sha256") == _SOURCE_AUTHORIZED_SHA256
        and source.get("frozen_state_sha256") == _FROZEN_SHA256,
        "prefix": prefix.get("checkpoint") == str(_PREFIX_REFERENCE_CHECKPOINT)
        and prefix.get("directory_inventory")
        == ["adapter.safetensors", "metadata.json", "runtime_metadata.json"]
        and prefix.get("file_sha256") == _PREFIX_REFERENCE_FILES
        and prefix.get("full_tensor_state_sha256") == _PREFIX_REFERENCE_FULL_SHA256
        and prefix.get("authorized_surface_state_sha256")
        == _PREFIX_REFERENCE_AUTHORIZED_SHA256
        and prefix.get("frozen_state_sha256") == _FROZEN_SHA256
        and prefix.get("scene_count") == 16
        and candidate.get("prefix_reference_computed_before_candidate_evaluation") is True,
        "qa": qa.get("train_question_count") == 384
        and qa.get("train_changed_pair_unit_count") == 25
        and qa.get("train_scene_ids") == train_scenes
        and qa.get("validation_qa_loaded") is False
        and qa.get("oracle_environment_files_loaded") is False
        and qa.get("deferred_final_qa_loaded") is False,
        "loader": loader.get("construction_used_v30_compatible_copy") is True
        and loader.get("state_hashes_bit_exact") is True
        and loader.get("target_paths_bit_exact") is True
        and loader.get("bank_names_bit_exact") is True
        and loader.get("v41_trainable_parameter_count") == 16384,
        "audit_hashes": candidate.get("gradient_audit_sha256")
        == "a7bcdbcfb51cf03648abb178157973dcf5d4c8940e1e1704d92467726bd301dd"
        and candidate.get("direction_audit_sha256")
        == "2d325cebd47fe03e3f021f4a2cd150f51d4448b88046e4a1fc370bd1a19d55bf"
        and candidate.get("schedule_audit_sha256")
        == "1aacefbdbf3a562683bc212d3b6a85b820579e03c04f0279613e05b59f38b590"
        and candidate.get("prefix_reference_hash_inventory_sha256")
        == "42960ef8a0dadd58f5b39a7ae6b5f0c8cd996dc0a3770d913689aaa35014ed66",
    }
    if not all(checks.values()):
        raise ValueError(f"V49 candidate reconstruction changed: {checks}")
    return checks


def _review_non_greedy(value: object) -> dict[str, Any]:
    gate = _mapping(value, "V49 non-greedy pre-gate")
    checks = _mapping(gate.get("checks"), "V49 non-greedy checks")
    expected = {name: name != _FAILED_CHECK for name in _NON_GREEDY_CHECKS}
    if dict(checks) != expected:
        raise ValueError("V49 must fail exactly the fixed original-prefix trust check")
    evidence = _mapping(gate.get("evidence"), "V49 non-greedy evidence")
    families = _mapping(evidence.get("complete_units_by_family"), "V49 complete families")
    cross = _mapping(
        evidence.get("cross_prefix_complete_units_by_family"),
        "V49 cross-prefix families",
    )
    evidence_checks = {
        "envelope": gate.get("evaluated_first") is True and gate.get("passed") is False,
        "full25": evidence.get("unit_count") == 25
        and evidence.get("per_unit_nll_row_count") == 25,
        "full48": evidence.get("broad_row_count") == 48,
        "teacher": evidence.get("complete_units") == 10
        and evidence.get("positive_sides") == 35
        and evidence.get("cross_prefix_complete_units") == 18
        and evidence.get("complete_physical_pair_coverage") == 5,
        "families": dict(families)
        == {"book_support": 1, "mirror_lr": 2, "picture_support": 0}
        and dict(cross) == {"book_support": 2, "mirror_lr": 4, "picture_support": 3},
        "deficit": _isclose(evidence.get("priority_side_deficit"), 29.86921536922455, "deficit")
        and _isclose(
            evidence.get("priority_deficit_improvement_vs_original_v41_u0"),
            1.244513750076294,
            "deficit improvement",
        ),
        "broad": _isclose(evidence.get("broad_nll"), 2.920842170715332, "broad NLL"),
        "trust": _isclose(
            evidence.get("original_v46_candidate_relative_prefix_trust_rms"),
            _OBSERVED_PREFIX_TRUST_RMS,
            "prefix trust RMS",
        )
        and _OBSERVED_PREFIX_TRUST_RMS > _PREFIX_TRUST_RMS_MAXIMUM,
        "diagnostic_hashes": evidence.get("pair_metrics_sha256")
        == "1b984d9d776a8c2c905c455ba48896ecbba204aab6c6933dd5027225cd11b912"
        and evidence.get("per_unit_nll_sha256")
        == "452ed2e76b404440cf75f7f6870d7db6c786f2f66f83cdd6d9956ce6a5fc245d",
    }
    if not all(evidence_checks.values()):
        raise ValueError(f"V49 non-greedy evidence changed: {evidence_checks}")
    return {
        "checks": dict(checks),
        "evidence_checks": evidence_checks,
        "failed_checks": [_FAILED_CHECK],
        "observed_prefix_trust_rms": _OBSERVED_PREFIX_TRUST_RMS,
        "prefix_trust_rms_maximum": _PREFIX_TRUST_RMS_MAXIMUM,
        "excess": _OBSERVED_PREFIX_TRUST_RMS - _PREFIX_TRUST_RMS_MAXIMUM,
    }


def review_report_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    authorization = _mapping(report.get("authorization"), "V49 authorization audit")
    authorization_checks = _mapping(authorization.get("checks"), "V49 authorization checks")
    reconstruction_checks = _review_reconstruction(report.get("candidate_reconstruction"))
    non_greedy_review = _review_non_greedy(report.get("non_greedy_pre_gate"))
    greedy = _mapping(report.get("greedy_gate"), "V49 greedy gate")
    final = _mapping(report.get("final_train_gate"), "V49 final train gate")
    checkpoint = _mapping(report.get("checkpoint"), "V49 checkpoint")
    restoration = _mapping(report.get("source_restoration"), "V49 source restoration")
    access = _mapping(report.get("access_audit"), "V49 access audit")

    fixed = {
        "schema": report.get("schema_version") == 1,
        "artifact": report.get("artifact") == "v49_guarded_candidate_train_gate",
        "failed": report.get("passed") is False,
        "terminal_path": authorization.get("terminal_path") == str(V48_TERMINAL),
        "terminal_sha": authorization.get("terminal_sha256") == _V48_TERMINAL_SHA256,
        "authorization_id": authorization.get("authorization_id")
        == "v49_guarded_alpha2_train_candidate_gate",
        "authorization_checks": set(authorization_checks) == _AUTHORIZATION_CHECKS
        and all(value is True for value in authorization_checks.values()),
        "greedy_skipped": greedy
        == {
            "authorized": False,
            "checks": {},
            "evidence": None,
            "executed": False,
            "greedy_skipped_due_pre_gate": True,
            "passed": False,
        },
        "final_failed_cleanly": final.get("passed") is False
        and final.get("pre_gate_passed") is False
        and final.get("greedy_gate_passed") is False
        and final.get("source_restored_exact") is True
        and final.get("access_audit_passed") is True
        and final.get("execution_error") is None,
        "no_checkpoint": checkpoint.get("path") == str(V49_CHECKPOINT)
        and checkpoint.get("staged_after_behavioral_gate") is False
        and checkpoint.get("written") is False
        and checkpoint.get("write_iff_final_gate_passed") is True
        and checkpoint.get("inventory") is None
        and checkpoint.get("optimizer_file_written") is False,
        "restored": restoration.get("attempted") is True
        and restoration.get("passed") is True
        and restoration.get("full_tensor_state_sha256") == _SOURCE_FULL_SHA256
        and restoration.get("authorized_surface_state_sha256") == _SOURCE_AUTHORIZED_SHA256
        and restoration.get("frozen_state_sha256") == _FROZEN_SHA256
        and restoration.get("all_parameter_gradients_absent") is True,
        "access": access.get("passed") is True
        and access.get("training_map_count") == 16
        and access.get("validation_qa_loaded") is False
        and access.get("oracle_loaded") is False
        and access.get("final_test_loaded") is False
        and access.get("optimizer_file_reads") == []
        and access.get("forbidden_file_accesses") == [],
        "top_level_scope": report.get("optimizer_constructed_or_loaded") is False
        and report.get("optimizer_state_file_opened") is False
        and report.get("optimizer_step_executed") is False
        and report.get("candidate_selection_performed") is False
        and report.get("validation_qa_loaded") is False
        and report.get("validation_environment_maps_loaded") is False
        and report.get("oracle_loaded") is False
        and report.get("final_test_scenes_touched") is False
        and report.get("selector_executed") is False
        and report.get("runtime_promotion_executed") is False
        and report.get("chat_promotion_executed") is False
        and report.get("embodied_promotion_executed") is False,
    }
    if not all(fixed.values()):
        raise ValueError(f"V49 fixed result envelope changed: {fixed}")
    return {
        "fixed_envelope_checks": fixed,
        "candidate_reconstruction_checks": reconstruction_checks,
        "non_greedy_review": non_greedy_review,
        "v49_final_train_gate_passed": False,
        "v49_greedy_executed": False,
        "v49_checkpoint_written": False,
        "source_restored_exact": True,
        "access_audit_passed": True,
    }


def load_and_review_report(expected_v49_report_sha256: str) -> dict[str, Any]:
    if not isinstance(expected_v49_report_sha256, str) or _HEX64.fullmatch(
        expected_v49_report_sha256
    ) is None:
        raise ValueError("expected V49 report SHA256 must be 64 lowercase hexadecimal digits")
    if expected_v49_report_sha256 != _V49_REPORT_SHA256:
        raise ValueError("V49 report SHA256 differs from the fixed reviewed result")
    _authenticate_static_inputs()
    path = _resolve(V49_REPORT)
    _locked_file(path, expected_v49_report_sha256, "V49 report")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "V49 report")
    return {
        "path": str(V49_REPORT),
        "sha256": expected_v49_report_sha256,
        "review": review_report_payload(report),
    }


def _v50_hashes_ready() -> bool:
    return (
        _HEX64.fullmatch(_V50_SCRIPT_SHA256) is not None
        and _HEX64.fullmatch(_V50_TEST_SHA256) is not None
    )


def _v50_implementation_review() -> dict[str, Any]:
    ready = _v50_hashes_ready()
    if ready:
        _locked_file(_resolve(V50_SCRIPT), _V50_SCRIPT_SHA256, "V50 script")
        _locked_file(_resolve(V50_TEST), _V50_TEST_SHA256, "V50 test")
    return {
        "status": (
            "exact_v50_implementation_authenticated"
            if ready
            else "pending_stable_v50_module_and_test_hashes"
        ),
        "ready": ready,
        "script": {"path": str(V50_SCRIPT), "sha256": _V50_SCRIPT_SHA256},
        "test": {"path": str(V50_TEST), "sha256": _V50_TEST_SHA256},
        "report": str(V50_REPORT),
        "conditional_checkpoint": str(V50_CHECKPOINT),
        "no_v50_file_opened_in_placeholder_mode": not ready,
    }


def v50_authorization_template() -> dict[str, Any]:
    ready = _v50_hashes_ready()
    candidates = [
        {
            "declared_order": index,
            "candidate_id": candidate_id,
            "scene_alpha": scene_alpha,
            "query_alpha": _QUERY_ALPHA,
        }
        for index, (candidate_id, scene_alpha) in enumerate(
            zip(_CANDIDATE_IDS, _SCENE_ALPHA_GRID, strict=True)
        )
    ]
    return {
        "schema_version": 1,
        "authorization_id": _V50_AUTHORIZATION_ID,
        "authorized": ready,
        "only_exact_action": _V50_ACTION,
        "authorized_script": str(V50_SCRIPT),
        "authorized_test": str(V50_TEST),
        "authorized_report": str(V50_REPORT),
        "authorized_config": str(V50_CONFIG),
        "conditional_checkpoint_output": str(V50_CHECKPOINT),
        "explicit_terminal_sha256_cli_required": True,
        "invocation_contract": {
            "terminal_path": str(DEFAULT_OUTPUT),
            "required_cli_argument": "--expected-v49-terminal-sha256",
            "v50_must_not_embed_terminal_sha256": True,
            "v50_must_authenticate_terminal_bytes_and_exact_authorization": True,
        },
        "implementation_integrity": {
            "script_sha256": _V50_SCRIPT_SHA256,
            "test_sha256": _V50_TEST_SHA256,
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
                "checkpoint": str(_PREFIX_REFERENCE_CHECKPOINT),
                "file_sha256": dict(_PREFIX_REFERENCE_FILES),
                "full_tensor_state_sha256": _PREFIX_REFERENCE_FULL_SHA256,
                "authorized_surface_state_sha256": _PREFIX_REFERENCE_AUTHORIZED_SHA256,
                "frozen_state_sha256": _FROZEN_SHA256,
                "scene_count": 16,
                "question_free_global_scene_prefix": True,
            },
            "v49_failed_result": {
                "path": str(V49_REPORT),
                "sha256": _V49_REPORT_SHA256,
                "only_failed_check": _FAILED_CHECK,
                "observed_prefix_trust_rms": _OBSERVED_PREFIX_TRUST_RMS,
                "prefix_trust_rms_maximum": _PREFIX_TRUST_RMS_MAXIMUM,
                "greedy_executed": False,
                "checkpoint_written": False,
                "source_restored_exact": True,
            },
        },
        "candidate_grid": {
            "fixed_before_any_candidate_forward": True,
            "candidate_count": 3,
            "direction_id": "guarded_both_sign",
            "direction_components": ["g_book", "g_mirror", "g5_guard"],
            "isolated_side_gradient_specs": _gradient_specs(),
            "normalize_each_nonzero_component": (
                "unit_l2_within_each_scene_or_query_group_before_combination"
            ),
            "scene_alpha_grid_declared_order": list(_SCENE_ALPHA_GRID),
            "query_alpha_fixed": _QUERY_ALPHA,
            "scene_readout_learning_rate": _SCENE_LR,
            "query_learning_rate": _QUERY_LR,
            "candidate_formula": (
                "float32_P0-scene_or_query_alpha*lr_group*"
                "sign(normalized_component_sum)"
            ),
            "candidates_declared_order": candidates,
            "candidate_hash_inventory_fixed_before_any_candidate_forward": True,
            "adaptive_grid_or_candidate_mutation": False,
            "exact_source_restoration_before_and_after_each_candidate": True,
        },
        "evaluation_and_selection": {
            "all_candidates_receive_full_non_greedy_gate_before_selection": True,
            "all_candidates_receive_full_25_unit_teacher_metrics_and_per_unit_nll": True,
            "all_candidates_receive_full_fixed_48_row_broad_nll": True,
            "all_candidates_receive_original_v46_candidate_relative_prefix_trust": True,
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
            "original_v46_candidate_relative_prefix_trust_rms_maximum": (
                _PREFIX_TRUST_RMS_MAXIMUM
            ),
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
            "fixed_three_candidate_grid": True,
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


def build_terminal_scaffold(expected_v49_report_sha256: str) -> dict[str, Any]:
    inputs = _authenticate_static_inputs()
    reviewed = load_and_review_report(expected_v49_report_sha256)
    implementation = _v50_implementation_review()
    ready = implementation["ready"] is True
    authorization = v50_authorization_template()
    return {
        "schema_version": 1,
        "artifact": "v49_guarded_candidate_terminal_gate_scaffold",
        "passed": True,
        "terminal_materialization_authorized": ready,
        "input_integrity": inputs,
        "v49_report_reference": {
            "path": reviewed["path"],
            "sha256": reviewed["sha256"],
            "authenticated": True,
        },
        "v49_result_review": reviewed["review"],
        "v50_implementation_review": implementation,
        "v50_authorization_template": authorization,
        "only_exact_successor_authorized": _V50_AUTHORIZATION_ID if ready else None,
        "conditional_successor_authorization": authorization if ready else None,
        "v49_checkpoint_write_authorized": False,
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
            "v49_report_read_only": True,
        },
    }


def build_terminal_report(expected_v49_report_sha256: str) -> dict[str, Any]:
    scaffold = build_terminal_scaffold(expected_v49_report_sha256)
    if scaffold.get("terminal_materialization_authorized") is not True:
        raise RuntimeError("V49 terminal is pending stable V50 module and test hashes")
    authorization = _mapping(
        scaffold.get("conditional_successor_authorization"), "V50 authorization"
    )
    if authorization.get("authorized") is not True:
        raise RuntimeError("V50 authorization is not exact and complete")
    return {
        "schema_version": 1,
        "artifact": "v49_guarded_candidate_terminal_gate",
        "passed": True,
        "terminal_materialization_authorized": True,
        "terminal_conclusion": "v49_failed_only_original_prefix_trust_v50_grid_required",
        "input_integrity": scaffold["input_integrity"],
        "v49_report_reference": scaffold["v49_report_reference"],
        "v49_result_review": scaffold["v49_result_review"],
        "v50_implementation_review": scaffold["v50_implementation_review"],
        "only_exact_successor_authorized": _V50_AUTHORIZATION_ID,
        "conditional_successor_authorization": dict(authorization),
        "v50_scene_query_alpha_grid_authorized": True,
        "v49_checkpoint_write_authorized": False,
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
    expected_v49_report_sha256: str,
) -> dict[str, Any]:
    path = _resolve(output)
    if path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V49 terminal output path is pinned")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V49 terminal is one-shot and will not overwrite {path}")
    report = build_terminal_report(expected_v49_report_sha256)
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-v49-report-sha256", required=True)
    args = parser.parse_args()
    result = build_terminal_scaffold(args.expected_v49_report_sha256)
    if result["terminal_materialization_authorized"] is not True:
        print(
            json.dumps(
                {
                    "artifact": result["artifact"],
                    "passed": result["passed"],
                    "terminal_materialization_authorized": False,
                    "v49_only_failed_check": _FAILED_CHECK,
                    "observed_prefix_trust_rms": _OBSERVED_PREFIX_TRUST_RMS,
                    "v50_status": result["v50_implementation_review"]["status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    report = write_report(
        args.output,
        expected_v49_report_sha256=args.expected_v49_report_sha256,
    )
    print(
        json.dumps(
            {
                "artifact": report["artifact"],
                "passed": report["passed"],
                "only_exact_successor_authorized": report[
                    "only_exact_successor_authorized"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V50_SCRIPT_SHA256_PLACEHOLDER",
    "V50_TEST_SHA256_PLACEHOLDER",
    "build_terminal_report",
    "build_terminal_scaffold",
    "load_and_review_report",
    "review_report_payload",
    "v50_authorization_template",
    "write_report",
]
