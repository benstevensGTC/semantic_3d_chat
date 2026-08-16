"""Exact four-step train-only book-support continuation for V47.

V47 reconstructs the unique fixed V46 numeric-threshold candidate from the
exact V45 update-four checkpoint, refuses to write unless its full and
authorized-surface hashes match the V46 evidence, then performs exactly four
predeclared q163 book-support optimizer updates.  Only the same three V45
tensors may move.  Update two is an integrity/non-catastrophic diagnostic;
the substantive go/no-go decision is the exact original V45 final gate at
update four.  Validation, oracle, final-test, selector, chat, and promotion
access remain forbidden.

The materialized V46 terminal SHA is supplied explicitly on the CLI to avoid
a source/seal hash cycle while the terminal pins this trainer and its tests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.evaluation import v46_v45_u4_lost_side_screen as v46
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.block_cross_residual import BlockCrossResidual
from semantic_3d_chat.training.checkpointing import (
    load_adapter_checkpoint,
    module_collection_state_sha256,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
)
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    build_exact_question_pair_units,
)
from semantic_3d_chat.training.train_block_cross_v35 import (
    broad_answer_nll,
    current_scene_tokens,
    paired_cross_prefix_objective,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    training_broad_nll,
    training_greedy_metrics,
)
from semantic_3d_chat.training.train_joint_pair_v30 import require_approved_v29_source
from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
    _PARAMETER_NAMES,
    _PARAMETER_SHAPES,
    assert_v44_trainable_surface,
    block_source_stack_state_sha256,
    freeze_for_v44,
    frozen_v44_state_sha256,
    source_prefix_trust_penalty,
    v44_contract,
)
from semantic_3d_chat.training.train_projected_gradient_v41 import (
    _prefix_replay_attestation,
    cache_v41_train_scenes,
    load_v41_bundle,
    priority_side_deficit,
    training_pair_gate_diagnostics,
    v41_loader_config,
    validate_per_unit_nll_diagnostics,
)
from semantic_3d_chat.training.train_retention_repair_v45 import (
    _BROAD_NLL_MAXIMUM,
    _FROZEN_SHA256,
    _PROTECTED_REPORT,
    _PROTECTED_REPORT_SHA256,
    _V41_FULL_SHA256,
    V45Settings,
    _backward_v45_retention,
    _mapping,
    _mps_empty_cache,
    _preflight_forbidden_roots,
    _sha256,
    _training_forbidden_roots,
    _unit_index,
    _unit_tokens,
    _v41_source_tensors,
    build_v45_schedule,
    load_v35_train_qa_records,
    v31_contract,
    v45_optimizer,
    v45_optimizer_audit,
    v45_retention_diagnostics,
    v45_settings,
)
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    validate_v37_training_cache_boundary,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_book_continuation_v47.yaml")
DEFAULT_TERMINAL = Path("reports/gemma4/metrics/v46_v45_u4_lost_side_terminal_gate.json")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query")
V47_TRAINER = Path("src/semantic_3d_chat/training/train_book_continuation_v47.py")
V47_TEST = Path("tests/test_train_book_continuation_v47.py")
V47_CONFIG = DEFAULT_CONFIG

_CONFIG_FILE_SHA256 = "6b15813237b217d8daad446c799127215bbb1366f2b442b61e975787efa4f6b7"
_AUTHORIZATION_ID = "v47_exact_book_support_continuation"
_V46_REPORT = v46.DEFAULT_OUTPUT
_V46_REPORT_SHA256 = "ce48a1fd484fa5dab71c76a2dd3e39194dd6964e068d6762925a02fb73f6aee6"
_BASE_CHECKPOINT = v46.DEFAULT_SOURCE
_CANDIDATE_ID = "g5_both_sign_alpha_1p0"
_CANDIDATE_DIRECTION = "g5_both_sign"
_CANDIDATE_ALPHA = 1.0
_CANDIDATE_FULL_SHA256 = "1d5adc1fb0d7a895056b77d38c8a12aba95c9997ec8a94edf68673f9c58fb954"
_CANDIDATE_AUTHORIZED_SHA256 = "d60b665d9a970433b2ed59e6769b9114468bef608b98eae828268101d39db56c"
_CANDIDATE_FROZEN_SHA256 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"
_TARGET_PAIR_ID = "pair_000015"
_TARGET_QUESTION_KEY = "cfq_163eb92339ad35a5"
_BROAD_ROW_NUMBERS = (13, 14, 15, 16)
_BROAD_QUESTION_IDS = ("q_000099", "q_000138", "q_000053", "q_000089")
_SAVED_STEPS = (0, 2, 4)
_CATASTROPHIC_BROAD_MAXIMUM = 3.05
_ORIGINAL_V41_PRIORITY_DEFICIT = v46._ORIGINAL_V41_PRIORITY_DEFICIT
_V45_U4_CONSTRUCTION_FULL_SHA256 = v46._SOURCE_FULL_SHA256
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class V47ScheduleRow:
    optimizer_step: int
    target_unit: CounterfactualPairUnit
    broad_record: Any


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def v47_settings(config: Mapping[str, Any]) -> V45Settings:
    """Validate the exact V47 objective and return the compatible V45 settings."""

    training = _mapping(config.get("training"), "V47 training")
    raw = _mapping(training.get("v47_book_continuation"), "V47 settings")
    expected = {
        "enabled": True,
        "optimizer_steps": 4,
        "checkpoint_steps": list(_SAVED_STEPS),
        "broad_nll_weight": 0.25,
        "pair_correct_nll_weight": 0.5,
        "target_side_hinge_weight": 8.0,
        "target_cross_prefix_weight": 8.0,
        "target_side_hinge_margin": 0.5,
        "target_cross_prefix_margin": 0.1,
        "retention_weight": 8.0,
        "retention_side_floor": 0.125,
        "retention_book_cross_floor": 0.025,
        "source_prefix_trust_weight": 0.001,
        "source_prefix_trust_scale": 0.05,
        "scene_readout_learning_rate": 1.0e-5,
        "query_learning_rate": 8.0e-6,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    if set(raw) != set(expected) or any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("V47 exact optimizer/objective settings changed")
    inherited = v45_settings(config)
    inherited_values = {
        "broad_nll_weight": inherited.broad_nll_weight,
        "pair_correct_nll_weight": inherited.pair_correct_nll_weight,
        "target_side_hinge_weight": inherited.target_side_hinge_weight,
        "target_cross_prefix_weight": inherited.target_cross_prefix_weight,
        "target_side_hinge_margin": inherited.target_side_hinge_margin,
        "target_cross_prefix_margin": inherited.target_cross_prefix_margin,
        "retention_weight": inherited.retention_weight,
        "retention_side_floor": inherited.retention_side_floor,
        "retention_book_cross_floor": inherited.retention_book_cross_floor,
        "source_prefix_trust_weight": inherited.source_prefix_trust_weight,
        "source_prefix_trust_scale": inherited.source_prefix_trust_scale,
        "scene_readout_learning_rate": inherited.scene_readout_learning_rate,
        "query_learning_rate": inherited.query_learning_rate,
        "weight_decay": inherited.weight_decay,
        "gradient_clip_norm": inherited.gradient_clip_norm,
    }
    if any(inherited_values[key] != expected[key] for key in inherited_values):
        raise ValueError("V47 objective differs from inherited exact V45 objective")
    return V45Settings(
        optimizer_steps=4,
        checkpoint_steps=_SAVED_STEPS,
        broad_nll_weight=expected["broad_nll_weight"],
        pair_correct_nll_weight=expected["pair_correct_nll_weight"],
        target_side_hinge_weight=expected["target_side_hinge_weight"],
        target_cross_prefix_weight=expected["target_cross_prefix_weight"],
        target_side_hinge_margin=expected["target_side_hinge_margin"],
        target_cross_prefix_margin=expected["target_cross_prefix_margin"],
        retention_weight=expected["retention_weight"],
        retention_side_floor=expected["retention_side_floor"],
        retention_book_cross_floor=expected["retention_book_cross_floor"],
        source_prefix_trust_weight=expected["source_prefix_trust_weight"],
        source_prefix_trust_scale=expected["source_prefix_trust_scale"],
        scene_readout_learning_rate=expected["scene_readout_learning_rate"],
        query_learning_rate=expected["query_learning_rate"],
        weight_decay=expected["weight_decay"],
        gradient_clip_norm=expected["gradient_clip_norm"],
    )


def v47_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(config.get("v47_book_continuation"), "V47 contract")
    expected_update2 = {
        "role": "diagnostic_integrity_and_noncatastrophic_fail_stop_only",
        "broad_nll_catastrophic_maximum": _CATASTROPHIC_BROAD_MAXIMUM,
        "all_teacher_metrics_must_be_finite": True,
        "both_authorized_parameter_groups_must_change": True,
        "frozen_state_must_remain_exact": True,
        "no_behavioral_count_or_lost_side_fail_stop": True,
    }
    expected_update4 = {
        "require_update2_integrity_gate_passed": True,
        "complete_units_minimum": 10,
        "positive_sides_minimum": 35,
        "cross_prefix_complete_units_minimum": 17,
        "complete_physical_pair_id_coverage_minimum": 5,
        "mirror_complete_units_minimum": 2,
        "book_complete_units_minimum": 1,
        "book_cross_prefix_complete_units_minimum": 1,
        "priority_side_deficit_minimum_improvement_vs_original_v41_u0": 0.5,
        "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
        "train_greedy_complete_units_minimum": 5,
        "broad_greedy_exact_correct_minimum": 23,
        "broad_greedy_row_count_exact": 48,
        "lost_side_margins_must_remain_strictly_positive": True,
        "both_authorized_parameter_groups_must_change": True,
        "frozen_state_must_remain_exact": True,
        "source_prefix_trust_rms_maximum": 0.002,
    }
    checks = {
        "schema": raw.get("schema_version") == 1,
        "role": raw.get("role")
        == "exact_v46_combined_alpha1_candidate_four_step_book_support_continuation",
        "report": raw.get("v46_report") == str(_V46_REPORT),
        "report_sha": raw.get("v46_report_sha256") == _V46_REPORT_SHA256,
        "terminal": raw.get("v46_terminal_report") == str(DEFAULT_TERMINAL),
        "base": raw.get("source_checkpoint") == str(_BASE_CHECKPOINT),
        "candidate_id": raw.get("reconstructed_candidate_id") == _CANDIDATE_ID,
        "direction": raw.get("reconstructed_candidate_direction") == _CANDIDATE_DIRECTION,
        "alpha": raw.get("reconstructed_candidate_alpha") == _CANDIDATE_ALPHA,
        "full": raw.get("reconstructed_candidate_full_tensor_state_sha256")
        == _CANDIDATE_FULL_SHA256,
        "authorized": raw.get("reconstructed_candidate_authorized_surface_state_sha256")
        == _CANDIDATE_AUTHORIZED_SHA256,
        "frozen": raw.get("frozen_excluding_authorized_state_sha256") == _CANDIDATE_FROZEN_SHA256,
        "names": tuple(raw.get("authorized_parameter_names", ())) == _PARAMETER_NAMES,
        "shapes": tuple(tuple(value) for value in raw.get("authorized_parameter_shapes", ()))
        == _PARAMETER_SHAPES,
        "count": raw.get("total_trainable_parameter_count") == 415_744,
        "optimizer": raw.get("optimizer") == "fresh_adamw_two_groups",
        "target_pair": raw.get("target_pair_id") == _TARGET_PAIR_ID,
        "target_schedule": tuple(raw.get("target_question_key_schedule", ()))
        == (_TARGET_QUESTION_KEY,) * 4,
        "broad_rows": tuple(raw.get("fixed_broad_row_numbers", ())) == _BROAD_ROW_NUMBERS,
        "broad_ids": tuple(raw.get("fixed_broad_question_ids", ())) == _BROAD_QUESTION_IDS,
        "gate2": dict(_mapping(raw.get("update2_gate"), "V47 update2 gate")) == expected_update2,
        "gate4": dict(_mapping(raw.get("update4_gate"), "V47 update4 gate")) == expected_update4,
        "no_validation": raw.get("validation_access_authorized") is False,
        "no_oracle": raw.get("oracle_access_authorized") is False,
        "no_final": raw.get("final_test_access_authorized") is False,
        "no_selector": raw.get("selector_execution_authorized") is False,
        "no_promotion": raw.get("runtime_promotion_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"V47 exact contract changed: {checks}")
    return {"checks": checks, "update2_gate": expected_update2, "update4_gate": expected_update4}


def require_v46_report() -> dict[str, Any]:
    path = _resolve(_V46_REPORT)
    if path.is_symlink() or not path.is_file() or _sha256(path) != _V46_REPORT_SHA256:
        raise ValueError("V47 requires the exact immutable V46 report")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "V46 report")
    rows = _sequence(report.get("candidate_results"), "V46 candidate results")
    matching = [
        _mapping(value, "V46 candidate")
        for value in rows
        if _mapping(value, "V46 candidate").get("candidate_id") == _CANDIDATE_ID
    ]
    if len(matching) != 1:
        raise ValueError("V47 exact V46 candidate is not unique")
    candidate = matching[0]
    threshold = _mapping(candidate.get("threshold_diagnostic"), "V46 threshold")
    final_state = _mapping(report.get("final_state"), "V46 final state")
    checks = {
        "artifact": report.get("artifact") == "v46_v45_u4_lost_side_no_step_diagnostic",
        "integrity": report.get("screen_integrity_passed") is True,
        "all_pairs": report.get("all_15_candidates_received_full_25_unit_metrics") is True,
        "all_broad": report.get("all_15_candidates_received_fixed_48_row_broad_nll") is True,
        "all_maps": report.get("all_16_training_maps_loaded") is True,
        "no_selection": report.get("candidate_selection_performed") is False,
        "no_authorization": report.get("candidate_authorization_granted") is False,
        "no_optimizer": report.get("optimizer_constructed_or_loaded") is False,
        "no_checkpoint": report.get("candidate_checkpoint_written") is False,
        "no_validation": report.get("validation_qa_loaded") is False,
        "no_oracle": report.get("oracle_loaded") is False,
        "no_final": report.get("final_test_scenes_touched") is False,
        "no_selector": report.get("selector_executed") is False,
        "no_promotion": report.get("runtime_promotion_executed") is False,
        "forbidden": report.get("forbidden_file_accesses") == [],
        "restored": final_state.get("passed") is True
        and final_state.get("full_tensor_state_sha256") == v46._SOURCE_FULL_SHA256,
        "candidate_direction": candidate.get("direction_id") == _CANDIDATE_DIRECTION,
        "candidate_alpha": candidate.get("alpha") == _CANDIDATE_ALPHA,
        "candidate_full": candidate.get("full_tensor_state_sha256") == _CANDIDATE_FULL_SHA256,
        "candidate_authorized": candidate.get("authorized_surface_state_sha256")
        == _CANDIDATE_AUTHORIZED_SHA256,
        "candidate_numeric": threshold.get("all_numeric_thresholds_met") is True,
        "candidate_not_authorized": candidate.get("candidate_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"V47 exact V46 report evidence changed: {checks}")
    return {
        "path": str(_V46_REPORT),
        "sha256": _V46_REPORT_SHA256,
        "report": dict(report),
        "candidate": dict(candidate),
        "checks": checks,
    }


def _validate_terminal_authorization(
    report: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, bool]:
    integrity = _mapping(authorization.get("implementation_integrity"), "V47 integrity")
    source = _mapping(authorization.get("source"), "V47 source")
    training = _mapping(authorization.get("training"), "V47 training")
    scope = _mapping(authorization.get("scope"), "V47 scope")
    checks = {
        "artifact": report.get("artifact") == "v46_v45_u4_lost_side_terminal_gate",
        "passed": report.get("passed") is True,
        "successor": report.get("only_exact_successor_authorized") == _AUTHORIZATION_ID,
        "id": authorization.get("authorization_id") == _AUTHORIZATION_ID,
        "authorized": authorization.get("authorized") is True,
        "action": authorization.get("only_exact_action")
        == "one_bounded_four_step_v47_book_support_continuation",
        "config": authorization.get("authorized_config") == str(V47_CONFIG),
        "trainer": authorization.get("authorized_trainer") == str(V47_TRAINER),
        "test": authorization.get("authorized_test") == str(V47_TEST),
        "output": authorization.get("authorized_output") == str(DEFAULT_OUTPUT),
        "explicit_cli": authorization.get("explicit_terminal_sha256_cli_required") is True,
        "config_hash": integrity.get("config_sha256") == _sha256(_resolve(V47_CONFIG)),
        "trainer_hash": integrity.get("trainer_sha256") == _sha256(_resolve(V47_TRAINER)),
        "test_hash": integrity.get("test_sha256") == _sha256(_resolve(V47_TEST)),
        "v46_report": source.get("v46_report_sha256") == _V46_REPORT_SHA256,
        "base": source.get("base_checkpoint") == str(_BASE_CHECKPOINT),
        "candidate": source.get("candidate_id") == _CANDIDATE_ID,
        "candidate_full": source.get("candidate_full_tensor_state_sha256")
        == _CANDIDATE_FULL_SHA256,
        "candidate_authorized": source.get("candidate_authorized_surface_sha256")
        == _CANDIDATE_AUTHORIZED_SHA256,
        "candidate_frozen": source.get("candidate_frozen_state_sha256") == _CANDIDATE_FROZEN_SHA256,
        "steps": training.get("optimizer_steps") == 4,
        "checkpoints": list(_sequence(training.get("checkpoint_steps"), "checkpoints"))
        == list(_SAVED_STEPS),
        "target": list(_sequence(training.get("target_question_keys"), "targets"))
        == [_TARGET_QUESTION_KEY] * 4,
        "broad": list(_sequence(training.get("broad_question_ids"), "broad ids"))
        == list(_BROAD_QUESTION_IDS),
        "fresh_optimizer": training.get("fresh_adamw") is True,
        "same_objective": training.get("same_v45_objective") is True,
        "diagnostic_u2": training.get("update2_integrity_only") is True,
        "final_u4": training.get("update4_original_v45_final_gate") is True,
        "train_only": scope.get("train_only") is True,
        "no_validation": scope.get("validation_access_authorized") is False,
        "no_oracle": scope.get("oracle_access_authorized") is False,
        "no_final": scope.get("final_test_access_authorized") is False,
        "no_selector": scope.get("selector_execution_authorized") is False,
        "no_promotion": scope.get("runtime_promotion_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"V47 V46-terminal authorization changed: {checks}")
    return checks


def require_v46_terminal(expected_sha256: str) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V47 expected V46 terminal SHA256 must be lowercase hex")
    path = _resolve(DEFAULT_TERMINAL)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("V47 V46 terminal is unavailable or unsafe")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError("V47 V46 terminal differs from explicit invocation SHA256")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "V46 terminal")
    authorization = _mapping(report.get("conditional_successor_authorization"), "V47 authorization")
    checks = _validate_terminal_authorization(report, authorization)
    return {
        "path": str(DEFAULT_TERMINAL),
        "sha256": observed,
        "authorization": dict(authorization),
        "checks": checks,
    }


def build_v47_schedule(
    records: Sequence[Any],
    units: Sequence[CounterfactualPairUnit],
    *,
    config: Mapping[str, Any],
) -> tuple[list[V47ScheduleRow], dict[str, Any], list[Any]]:
    v45_schedule, _v45_audit, broad_records = build_v45_schedule(records, units, config=config)
    selected = v45_schedule[4:8]
    rows = [
        V47ScheduleRow(
            optimizer_step=index,
            target_unit=value.target_unit,
            broad_record=value.broad_record,
        )
        for index, value in enumerate(selected, start=1)
    ]
    contract_rows = [
        {
            "optimizer_update": row.optimizer_step,
            "target_pair_id": row.target_unit.pair_id,
            "target_question_key": row.target_unit.question_key,
            "broad_row_number": broad_records.index(row.broad_record) + 1,
            "broad_question_id": row.broad_record.question_id,
        }
        for row in rows
    ]
    expected = [
        {
            "optimizer_update": index,
            "target_pair_id": _TARGET_PAIR_ID,
            "target_question_key": _TARGET_QUESTION_KEY,
            "broad_row_number": row_number,
            "broad_question_id": question_id,
        }
        for index, (row_number, question_id) in enumerate(
            zip(_BROAD_ROW_NUMBERS, _BROAD_QUESTION_IDS), start=1
        )
    ]
    if contract_rows != expected:
        raise RuntimeError("V47 fixed q163/broad schedule changed")
    return (
        rows,
        {
            "schema_version": 1,
            "rows": contract_rows,
            "schedule_sha256": _canonical_sha256(contract_rows),
            "fixed_nonadaptive": True,
            "one_true_optimizer_step_per_row": True,
        },
        broad_records,
    )


def _all_teacher_metrics_finite(pair_metrics: Mapping[str, Any], broad_nll: float) -> bool:
    if not math.isfinite(broad_nll):
        return False
    for unit in _sequence(pair_metrics.get("units"), "V47 units"):
        row = _mapping(unit, "V47 unit")
        for field in ("side_margins", "cross_prefix_margins"):
            if not all(math.isfinite(float(value)) for value in _sequence(row.get(field), field)):
                return False
    return math.isfinite(float(priority_side_deficit(pair_metrics)["combined"]))


def v47_update2_gate(
    *,
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    scene_changed: bool,
    query_changed: bool,
    frozen_exact: bool,
    trust_rms: float,
) -> dict[str, Any]:
    hard_checks = {
        "all_teacher_metrics_finite": _all_teacher_metrics_finite(pair_metrics, broad_nll)
        and math.isfinite(trust_rms),
        "broad_nll_below_catastrophic_maximum": broad_nll <= _CATASTROPHIC_BROAD_MAXIMUM,
        "both_authorized_parameter_groups_changed": scene_changed and query_changed,
        "frozen_state_exact": frozen_exact,
    }
    return {
        "role": "diagnostic_integrity_and_noncatastrophic_fail_stop_only",
        "hard_checks": hard_checks,
        "passed": all(hard_checks.values()),
        "behavioral_counts_are_diagnostic_only": True,
        "complete_units": int(pair_metrics["complete_units"]),
        "positive_sides": int(pair_metrics["positive_sides"]),
        "cross_prefix_complete_units": int(pair_metrics["cross_prefix_complete_units"]),
        "complete_physical_pair_coverage": int(pair_metrics["complete_physical_pair_coverage"]),
        "priority_side_deficit": float(priority_side_deficit(pair_metrics)["combined"]),
        "broad_nll": broad_nll,
        "catastrophic_broad_nll_maximum": _CATASTROPHIC_BROAD_MAXIMUM,
        "retention_diagnostics": v45_retention_diagnostics(pair_metrics),
        "source_prefix_trust_rms": trust_rms,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def v47_update4_gate(
    *,
    update2_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    greedy_metrics: Mapping[str, Any],
    scene_changed: bool,
    query_changed: bool,
    frozen_exact: bool,
    trust_rms: float,
) -> dict[str, Any]:
    families = _mapping(pair_metrics.get("complete_units_by_family"), "V47 families")
    cross_families = _mapping(
        pair_metrics.get("cross_prefix_complete_units_by_family"),
        "V47 cross families",
    )
    retention = v45_retention_diagnostics(pair_metrics)
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    checks = {
        "update2_integrity_gate_passed": update2_gate.get("passed") is True,
        "teacher_complete_units_at_least_10": int(pair_metrics["complete_units"]) >= 10,
        "teacher_positive_sides_at_least_35": int(pair_metrics["positive_sides"]) >= 35,
        "teacher_cross_complete_units_at_least_17": int(pair_metrics["cross_prefix_complete_units"])
        >= 17,
        "complete_physical_pair_id_coverage_at_least_5": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= 5,
        "mirror_complete_units_at_least_2": int(families.get("mirror_lr", 0)) >= 2,
        "book_complete_units_at_least_1": int(families.get("book_support", 0)) >= 1,
        "book_cross_prefix_complete_units_at_least_1": int(cross_families.get("book_support", 0))
        >= 1,
        "priority_teacher_deficit_improved_at_least_0_5_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit >= 0.5
        ),
        "broad_nll_at_most_authorized_maximum": broad_nll <= _BROAD_NLL_MAXIMUM,
        "train_greedy_complete_units_at_least_5": int(greedy_metrics["complete_units"]) >= 5,
        "broad_greedy_exact_correct_at_least_23_of_48": int(greedy_metrics["broad_exact_correct"])
        >= 23
        and int(greedy_metrics["broad_row_count"]) == 48,
        "both_lost_side_margins_remain_strictly_positive": retention[
            "both_lost_sides_strictly_positive"
        ],
        "scene_readout_state_changed": scene_changed,
        "query_state_changed": query_changed,
        "both_authorized_parameter_groups_changed": scene_changed and query_changed,
        "frozen_state_exact": frozen_exact,
        "source_prefix_trust_rms_at_most_0_002": trust_rms <= 0.002,
    }
    return {
        "checks": checks,
        **checks,
        "passed": all(checks.values()),
        "priority_teacher_side_deficit": deficit,
        "priority_teacher_side_deficit_improvement_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit
        ),
        "broad_nll": broad_nll,
        "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
        "source_prefix_trust_rms": trust_rms,
        "retention_diagnostics": retention,
        "training_greedy_metrics": dict(greedy_metrics),
        "full_train_pair_unit_count": int(pair_metrics["unit_count"]),
        "full_broad_nll_row_count": 48,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "selector_execution_authorized": False,
    }


def _metadata(
    *,
    source_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    terminal: Mapping[str, Any],
    v46_evidence: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    optimizer_step: int,
    bundle: Any,
    block_core: BlockCrossResidual,
    candidate_prefix_hashes: Mapping[str, str],
    source_audit: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    gate2: Mapping[str, Any] | None,
    gate4: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(source_metadata))
    result.update(
        {
            "schema_version": 1,
            "config_hash": config_hash(dict(config)),
            "optimizer_step": optimizer_step,
            "epoch": optimizer_step,
            "history": [dict(row) for row in history],
            "question_dependent_scene_processing": False,
            **bundle.lora_installation.checkpoint_metadata(),
            "block_cross_residual_state_sha256": block_core.state_sha256(),
            "frozen_block_cross_source_stack_state_sha256": (
                block_source_stack_state_sha256(bundle, block_core)
            ),
        }
    )
    result["v47_book_continuation"] = {
        "schema_version": 1,
        "optimizer_step": optimizer_step,
        "conditional_v46_terminal_gate": {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        },
        "conditional_authorization": dict(terminal["authorization"]),
        "v46_report": {"path": v46_evidence["path"], "sha256": v46_evidence["sha256"]},
        "base_checkpoint": str(_BASE_CHECKPOINT),
        "reconstructed_candidate_id": _CANDIDATE_ID,
        "reconstructed_candidate_full_tensor_state_sha256": _CANDIDATE_FULL_SHA256,
        "reconstructed_candidate_authorized_surface_state_sha256": (_CANDIDATE_AUTHORIZED_SHA256),
        "frozen_excluding_authorized_source_state_sha256": _CANDIDATE_FROZEN_SHA256,
        "frozen_excluding_authorized_state_sha256": frozen_v44_state_sha256(bundle),
        "trainable_surface": assert_v44_trainable_surface(bundle, block_core),
        "candidate_prefix_sha256_by_train_scene": dict(candidate_prefix_hashes),
        "source_prefix_reference": "exact_reconstructed_v46_combined_alpha1_candidate",
        "source_audit": dict(source_audit),
        "schedule_audit": dict(schedule_audit),
        "update2_integrity_gate": None if gate2 is None else dict(gate2),
        "update4_final_train_only_gate": None if gate4 is None else dict(gate4),
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
        "independent_terminal_seal_required": True,
    }
    return result


def _save(
    path: Path,
    *,
    bundle: Any,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V47 checkpoint destination already exists: {path}")
    save_adapter_checkpoint(path, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(path, optimizer)


def _preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path = DEFAULT_OUTPUT,
    v46_terminal_sha256: str,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    output_path = _resolve(output)
    if config_file != _resolve(DEFAULT_CONFIG) or _sha256(config_file) != _CONFIG_FILE_SHA256:
        raise ValueError("V47 config path or bytes changed")
    if output_path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V47 output namespace changed")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("V47 one-shot output already exists")
    config = load_config(config_file)
    v47_settings(config)
    contract = v47_contract(config)
    terminal = require_v46_terminal(v46_terminal_sha256)
    v46_evidence = require_v46_report()
    protected = _resolve(_PROTECTED_REPORT)
    if protected.is_symlink() or _sha256(protected) != _PROTECTED_REPORT_SHA256:
        raise ValueError("V47 protected selection report changed")
    audit = FileAccessAudit(
        _preflight_forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        source_tensors, source_metadata, source_audit = v46._source_evidence()
        loader = v41_loader_config(config)
        assert_deferred_final_scenes_absent(loader)
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
        schedule, schedule_audit, broad_records = build_v47_schedule(records, units, config=config)
        if (
            len(source_tensors) != 179
            or source_metadata.get("optimizer_step") != 4
            or len(records) != 384
            or len(units) != 25
            or len(schedule) != 4
            or len(broad_records) != 48
        ):
            raise RuntimeError("V47 preflight inventory changed")
    audit.assert_clean()
    return {
        "schema_version": 1,
        "artifact": "v47_book_continuation_preflight",
        "passed": True,
        "terminal": terminal,
        "v46_evidence": {
            "path": v46_evidence["path"],
            "sha256": v46_evidence["sha256"],
            "candidate_id": _CANDIDATE_ID,
            "candidate_full_tensor_state_sha256": _CANDIDATE_FULL_SHA256,
        },
        "contract": contract,
        "source_audit": source_audit,
        "train_question_count": len(records),
        "changed_pair_unit_count": len(units),
        "schedule": schedule_audit,
        "qa_audit": qa_audit,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "candidate_reconstructed": False,
        "optimizer_constructed": False,
        "checkpoint_written": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_executed": False,
        "runtime_promotion_executed": False,
        "forbidden_file_accesses": audit.forbidden_accesses(),
    }


def _run_impl(
    *,
    config_path: str | Path,
    output: str | Path,
    v46_terminal_sha256: str,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    output_path = _resolve(output)
    if config_file != _resolve(DEFAULT_CONFIG) or _sha256(config_file) != _CONFIG_FILE_SHA256:
        raise ValueError("V47 config path or bytes changed")
    if output_path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V47 output namespace changed")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("V47 one-shot output already exists")
    config = load_config(config_file)
    settings = v47_settings(config)
    contract = v47_contract(config)
    terminal = require_v46_terminal(v46_terminal_sha256)
    v46_evidence = require_v46_report()
    source_full, source_metadata, base_audit = v46._source_evidence()
    loader = v41_loader_config(config)
    assert_deferred_final_scenes_absent(loader)
    records, qa_audit = load_v35_train_qa_records(loader)
    units = build_exact_question_pair_units(records)
    units_by_key = _unit_index(units)
    schedule, schedule_audit, broad_records = build_v47_schedule(records, units, config=config)

    construction = v44_contract(config)
    v41_tensors, v41_metadata = _v41_source_tensors(construction)
    if tensor_state_sha256(v41_tensors) != _V41_FULL_SHA256:
        raise RuntimeError("V47 V41 construction source changed")
    approved = require_approved_v29_source(loader)
    bundle, block_core, loaded_v41, loader_transition = load_v41_bundle(
        config, approved, construction.source_checkpoint, v41_tensors
    )
    if loaded_v41 != v41_metadata:
        raise RuntimeError("V47 V41 construction metadata changed")
    loaded_u4 = load_adapter_checkpoint(
        _resolve(_BASE_CHECKPOINT), bundle.checkpoint_modules, device="cpu"
    )
    if loaded_u4 != source_metadata:
        raise RuntimeError("V47 exact V45 update-four overlay metadata changed")
    named = freeze_for_v44(bundle, block_core)
    if (
        module_collection_state_sha256(bundle.checkpoint_modules)
        != _V45_U4_CONSTRUCTION_FULL_SHA256
        or frozen_v44_state_sha256(bundle) != _FROZEN_SHA256
    ):
        raise RuntimeError("V47 live V45 update-four base state changed")

    split = v31_contract(loader)
    manifest_ids = (*split.train_scene_ids, *split.validation_scene_ids)
    caches, cache_audit = cache_v41_train_scenes(
        config=loader,
        bundle=bundle,
        source_metadata=source_metadata,
        scene_ids=split.train_scene_ids,
        manifest_scene_ids=manifest_ids,
    )
    cache_audit.update(
        {
            "scene_scope": "training_only",
            "validation_scene_ids_loaded": [],
            "validation_environment_maps_loaded": False,
            "deferred_final_scene_ids_loaded": [],
        }
    )
    cache_boundary = validate_v37_training_cache_boundary(
        cache_audit=cache_audit,
        caches=caches,
        config=loader,
        train_scene_ids=split.train_scene_ids,
        validation_scene_ids=split.validation_scene_ids,
    )
    prefix_evidence = _prefix_replay_attestation(
        caches=caches,
        block_cross_residual=block_core,
        bundle=bundle,
        expected_scene_ids=split.train_scene_ids,
    )
    if len(caches) != 16:
        raise RuntimeError("V47 must cache all 16 training scenes")

    g5_unit = units_by_key["cfq_5c84a2c27d2be251"]
    g5, gradient_row = v46._selected_side_gradient(
        unit=g5_unit,
        expected_pair_id="pair_000006",
        expected_question_key="cfq_5c84a2c27d2be251",
        side_index=0,
        caches=caches,
        block_core=block_core,
        bundle=bundle,
        named=named,
    )
    base_surface = {
        name: source_full[name].detach().float().cpu().clone() for name in _PARAMETER_NAMES
    }
    candidate = v46.candidate_from_sign_line(
        base_surface,
        g5,
        direction_id=_CANDIDATE_DIRECTION,
        alpha=_CANDIDATE_ALPHA,
    )
    v46._copy_candidate(named, candidate)
    reconstructed_full = module_collection_state_sha256(bundle.checkpoint_modules)
    reconstructed_authorized = tensor_state_sha256(
        {name: value.detach().cpu() for name, value in named.items()}
    )
    reconstructed_frozen = frozen_v44_state_sha256(bundle)
    if (
        reconstructed_full != _CANDIDATE_FULL_SHA256
        or reconstructed_authorized != _CANDIDATE_AUTHORIZED_SHA256
        or reconstructed_frozen != _CANDIDATE_FROZEN_SHA256
    ):
        raise RuntimeError("V47 reconstructed candidate hash attestation failed before write")

    candidate_row = _mapping(v46_evidence["candidate"], "V47 candidate evidence")
    candidate_pair, candidate_nll = training_pair_gate_diagnostics(
        units=units,
        caches=caches,
        block_cross_residual=block_core,
        bundle=bundle,
        settings=settings,
    )
    candidate_broad = training_broad_nll(
        records=broad_records,
        caches=caches,
        block_cross_residual=block_core,
        bundle=bundle,
    )
    validate_per_unit_nll_diagnostics(candidate_nll, candidate_pair)
    candidate_replay = {
        "pair_metrics": v46._numeric_close(candidate_pair, candidate_row["pair_metrics"]),
        "per_unit_nll": v46._numeric_close(
            candidate_nll, candidate_row["per_unit_nll_diagnostics"]
        ),
        "broad_nll": math.isclose(
            candidate_broad,
            float(candidate_row["broad_nll"]),
            rel_tol=0.0,
            abs_tol=_TOLERANCE,
        ),
    }
    candidate_replay["passed"] = all(candidate_replay.values())
    if candidate_replay["passed"] is not True:
        raise RuntimeError("V47 reconstructed candidate diagnostic replay changed")

    with torch.inference_mode():
        candidate_scene_tokens = {
            scene_id: current_scene_tokens(
                caches[scene_id], block_core, device=bundle.language.device
            )
            .detach()
            .cpu()
            .clone()
            for scene_id in sorted(caches)
        }
    candidate_prefix_hashes = {
        scene_id: tensor_state_sha256({"scene_tokens": value})
        for scene_id, value in candidate_scene_tokens.items()
    }
    source_scene_hash = tensor_state_sha256(
        {_PARAMETER_NAMES[0]: named[_PARAMETER_NAMES[0]].detach().cpu()}
    )
    source_query_hash = tensor_state_sha256(
        {name: named[name].detach().cpu() for name in (_PARAMETER_NAMES[1], _PARAMETER_NAMES[2])}
    )
    freeze_for_v44(bundle, block_core)
    optimizer = v45_optimizer(
        [named[_PARAMETER_NAMES[0]]],
        [named[_PARAMETER_NAMES[1]], named[_PARAMETER_NAMES[2]]],
        settings,
    )
    optimizer_audit = v45_optimizer_audit(optimizer)
    assert_v44_trainable_surface(bundle, block_core, optimizer=optimizer)
    source_audit = {
        **base_audit,
        "v46_report_sha256": _V46_REPORT_SHA256,
        "candidate_reconstruction": {
            "candidate_id": _CANDIDATE_ID,
            "direction_id": _CANDIDATE_DIRECTION,
            "alpha": _CANDIDATE_ALPHA,
            "gradient": gradient_row,
            "full_tensor_state_sha256": reconstructed_full,
            "authorized_surface_state_sha256": reconstructed_authorized,
            "frozen_state_sha256": reconstructed_frozen,
            "replay": candidate_replay,
        },
        "candidate_prefix_reference_computed_before_optimizer_construction": True,
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
    }
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "candidate_pair_metrics": candidate_pair,
            "candidate_per_unit_nll_diagnostics": candidate_nll,
            "candidate_broad_train_nll": candidate_broad,
            "candidate_replay": candidate_replay,
            "candidate_prefix_trust_rms": 0.0,
            "authorized_state_sha256": reconstructed_authorized,
            "scene_readout_state_sha256": source_scene_hash,
            "query_state_sha256": source_query_hash,
            "frozen_state_sha256": _CANDIDATE_FROZEN_SHA256,
            "validation_qa_loaded": False,
            "oracle_environment_files_loaded": False,
            "saved_checkpoint": True,
        }
    ]
    gate2: Mapping[str, Any] | None = None
    gate4: Mapping[str, Any] | None = None
    completed_steps = 0
    stop_reason: str | None = None
    output_path.mkdir(parents=True, exist_ok=False)
    metadata0 = _metadata(
        source_metadata=source_metadata,
        config=config,
        terminal=terminal,
        v46_evidence=v46_evidence,
        history=history,
        optimizer_step=0,
        bundle=bundle,
        block_core=block_core,
        candidate_prefix_hashes=candidate_prefix_hashes,
        source_audit=source_audit,
        schedule_audit=schedule_audit,
        gate2=None,
        gate4=None,
    )
    _save(output_path / "update_000", bundle=bundle, metadata=metadata0, optimizer=None)

    for item in schedule:
        step = item.optimizer_step
        freeze_for_v44(bundle, block_core)
        assert_v44_trainable_surface(bundle, block_core, optimizer=optimizer)
        optimizer.zero_grad(set_to_none=True)
        broad_tokens = current_scene_tokens(
            caches[item.broad_record.scene_id],
            block_core,
            device=bundle.language.device,
        )
        broad = broad_answer_nll(scene_tokens=broad_tokens, record=item.broad_record, bundle=bundle)
        broad_value = float(broad.detach().cpu())
        (settings.broad_nll_weight * broad).backward()
        del broad, broad_tokens

        target_tokens = _unit_tokens(
            item.target_unit,
            caches=caches,
            block_core=block_core,
            device=bundle.language.device,
        )
        pair_nll, side_hinge, cross_hinge, diagnostics = paired_cross_prefix_objective(
            unit=item.target_unit,
            scene_tokens=target_tokens,
            bundle=bundle,
            side_margin=settings.target_side_hinge_margin,
            cross_prefix_margin=settings.target_cross_prefix_margin,
        )
        pair_value = float(pair_nll.detach().cpu())
        side_value = float(side_hinge.detach().cpu())
        cross_value = float(cross_hinge.detach().cpu())
        target_loss = (
            settings.pair_correct_nll_weight * pair_nll
            + settings.target_side_hinge_weight * side_hinge
            + settings.target_cross_prefix_weight * cross_hinge
        )
        target_loss.backward()
        del target_loss, pair_nll, side_hinge, cross_hinge, diagnostics, target_tokens
        _mps_empty_cache(bundle.language.device)

        retention_values = _backward_v45_retention(
            units_by_key=units_by_key,
            caches=caches,
            block_core=block_core,
            bundle=bundle,
            settings=settings,
        )
        trust, _trust_rms = source_prefix_trust_penalty(
            caches=caches,
            references=candidate_scene_tokens,
            block_core=block_core,
            device=bundle.language.device,
            scale=settings.source_prefix_trust_scale,
        )
        trust_value = float(trust.detach().cpu())
        (settings.source_prefix_trust_weight * trust).backward()
        del trust
        loss_value = (
            settings.broad_nll_weight * broad_value
            + settings.pair_correct_nll_weight * pair_value
            + settings.target_side_hinge_weight * side_value
            + settings.target_cross_prefix_weight * cross_value
            + float(retention_values["weighted_retention_loss"])
            + settings.source_prefix_trust_weight * trust_value
        )
        if any(
            parameter.grad is None
            or not torch.isfinite(parameter.grad).all()
            or torch.count_nonzero(parameter.grad).item() == 0
            for parameter in named.values()
        ):
            raise RuntimeError("V47 active gradient is absent, zero, or nonfinite")
        scene_preclip = float(
            torch.nn.utils.clip_grad_norm_(
                (named[_PARAMETER_NAMES[0]],), settings.gradient_clip_norm
            )
        )
        query_preclip = float(
            torch.nn.utils.clip_grad_norm_(
                (named[_PARAMETER_NAMES[1]], named[_PARAMETER_NAMES[2]]),
                settings.gradient_clip_norm,
            )
        )
        if not math.isfinite(scene_preclip) or not math.isfinite(query_preclip):
            raise RuntimeError("V47 gradient norm is nonfinite")
        optimizer.step()
        completed_steps = step
        if frozen_v44_state_sha256(bundle) != _CANDIDATE_FROZEN_SHA256:
            raise RuntimeError("V47 changed a frozen tensor or buffer")

        pair_metrics: Mapping[str, Any] | None = None
        per_unit_nll: list[dict[str, Any]] | None = None
        broad_diagnostic: float | None = None
        greedy: Mapping[str, Any] | None = None
        with torch.inference_mode():
            _trust_diagnostic, trust_rms_tensor = source_prefix_trust_penalty(
                caches=caches,
                references=candidate_scene_tokens,
                block_core=block_core,
                device=bundle.language.device,
                scale=settings.source_prefix_trust_scale,
            )
        trust_rms = float(trust_rms_tensor.detach().cpu())
        if step in (2, 4):
            pair_metrics, per_unit_nll = training_pair_gate_diagnostics(
                units=units,
                caches=caches,
                block_cross_residual=block_core,
                bundle=bundle,
                settings=settings,
            )
            broad_diagnostic = training_broad_nll(
                records=broad_records,
                caches=caches,
                block_cross_residual=block_core,
                bundle=bundle,
            )
            validate_per_unit_nll_diagnostics(per_unit_nll, pair_metrics)
        if step == 4:
            greedy = training_greedy_metrics(
                units=units,
                broad_records=broad_records,
                caches=caches,
                block_cross_residual=block_core,
                bundle=bundle,
                config=loader,
            )
        authorized_hash = tensor_state_sha256(
            {name: value.detach().cpu() for name, value in named.items()}
        )
        scene_hash = tensor_state_sha256(
            {_PARAMETER_NAMES[0]: named[_PARAMETER_NAMES[0]].detach().cpu()}
        )
        query_hash = tensor_state_sha256(
            {
                name: named[name].detach().cpu()
                for name in (_PARAMETER_NAMES[1], _PARAMETER_NAMES[2])
            }
        )
        scene_changed = scene_hash != source_scene_hash
        query_changed = query_hash != source_query_hash
        if step == 2:
            assert pair_metrics is not None and broad_diagnostic is not None
            gate2 = v47_update2_gate(
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                scene_changed=scene_changed,
                query_changed=query_changed,
                frozen_exact=True,
                trust_rms=trust_rms,
            )
        if step == 4:
            assert (
                gate2 is not None
                and pair_metrics is not None
                and broad_diagnostic is not None
                and greedy is not None
            )
            gate4 = v47_update4_gate(
                update2_gate=gate2,
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                greedy_metrics=greedy,
                scene_changed=scene_changed,
                query_changed=query_changed,
                frozen_exact=True,
                trust_rms=trust_rms,
            )
        history.append(
            {
                "optimizer_update": step,
                "true_optimizer_step": True,
                "target_pair_id": item.target_unit.pair_id,
                "target_question_key": item.target_unit.question_key,
                "broad_question_id": item.broad_record.question_id,
                "train_loss": loss_value,
                "train_broad_nll": broad_value,
                "train_pair_correct_nll": pair_value,
                "train_target_side_hinge": side_value,
                "train_target_cross_prefix_hinge": cross_value,
                "train_retention": retention_values,
                "train_candidate_prefix_trust_penalty": trust_value,
                "candidate_prefix_trust_rms": trust_rms,
                "scene_readout_preclip_gradient_norm": scene_preclip,
                "query_preclip_gradient_norm": query_preclip,
                "pair_metrics": pair_metrics,
                "per_unit_nll_diagnostics": per_unit_nll,
                "broad_diagnostic_nll": broad_diagnostic,
                "training_greedy_metrics": greedy,
                "update2_integrity_gate": None if gate2 is None else dict(gate2),
                "update4_final_train_only_gate": None if gate4 is None else dict(gate4),
                "authorized_state_sha256": authorized_hash,
                "scene_readout_state_sha256": scene_hash,
                "query_state_sha256": query_hash,
                "scene_readout_state_changed": scene_changed,
                "query_state_changed": query_changed,
                "frozen_state_sha256": _CANDIDATE_FROZEN_SHA256,
                "validation_qa_loaded": False,
                "oracle_environment_files_loaded": False,
                "saved_checkpoint": step in _SAVED_STEPS,
            }
        )
        if step not in _SAVED_STEPS:
            continue
        metadata = _metadata(
            source_metadata=source_metadata,
            config=config,
            terminal=terminal,
            v46_evidence=v46_evidence,
            history=history,
            optimizer_step=step,
            bundle=bundle,
            block_core=block_core,
            candidate_prefix_hashes=candidate_prefix_hashes,
            source_audit=source_audit,
            schedule_audit=schedule_audit,
            gate2=gate2,
            gate4=gate4,
        )
        _save(
            output_path / f"update_{step:03d}",
            bundle=bundle,
            metadata=metadata,
            optimizer=optimizer,
        )
        print(
            json.dumps(
                {
                    "phase": "v47_book_continuation_checkpoint",
                    "optimizer_step": step,
                    "update2_integrity_gate_passed": None if gate2 is None else gate2.get("passed"),
                    "update4_final_gate_passed": None if gate4 is None else gate4.get("passed"),
                    "candidate_prefix_trust_rms": trust_rms,
                    "validation_qa_loaded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if step == 2 and gate2 is not None and gate2.get("passed") is not True:
            stop_reason = "update2_integrity_or_catastrophic_gate_failed"
            break
        if step == 4 and gate4 is not None and gate4.get("passed") is not True:
            stop_reason = "update4_final_train_only_gate_failed"

    return {
        "schema_version": 1,
        "artifact": "v47_book_continuation_train_only_pilot",
        "passed": gate4 is not None and gate4.get("passed") is True,
        "bounded_training_completed": completed_steps == 4,
        "stopped_at_train_only_gate": stop_reason,
        "output": str(output_path),
        "optimizer_updates": completed_steps,
        "saved_optimizer_steps": [
            int(row["optimizer_update"]) for row in history if row.get("saved_checkpoint") is True
        ],
        "terminal": terminal,
        "v46_evidence": {
            "path": v46_evidence["path"],
            "sha256": v46_evidence["sha256"],
            "candidate_id": _CANDIDATE_ID,
        },
        "candidate_reconstruction": source_audit["candidate_reconstruction"],
        "optimizer": optimizer_audit,
        "contract": contract,
        "schedule": schedule_audit,
        "update2_integrity_gate": gate2,
        "update4_final_train_only_gate": gate4,
        "loader_transition": loader_transition,
        "cache_boundary": cache_boundary,
        "scene_prefix_evidence": prefix_evidence,
        "qa_audit": qa_audit,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
    }


def run_v47(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path = DEFAULT_OUTPUT,
    v46_terminal_sha256: str,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    config = load_config(config_file)
    protected = _resolve(_PROTECTED_REPORT)
    if protected.is_symlink() or _sha256(protected) != _PROTECTED_REPORT_SHA256:
        raise ValueError("V47 protected selection report changed before training")
    protected_before = _sha256(protected)
    audit = FileAccessAudit(
        _training_forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        result = _run_impl(
            config_path=config_file,
            output=output,
            v46_terminal_sha256=v46_terminal_sha256,
        )
    audit.assert_clean()
    if _sha256(protected) != protected_before:
        raise RuntimeError("V47 changed the protected selection report")
    loader = v41_loader_config(config)
    split = v31_contract(loader)
    expected_maps = {
        str((artifact_root(loader, "maps") / scene_id / "voxel_map.npz").resolve())
        for scene_id in split.train_scene_ids
    }
    observed_maps = {path for path in audit.unique_paths if path.endswith("/voxel_map.npz")}
    if observed_maps != expected_maps:
        raise RuntimeError("V47 did not read exactly all 16 training maps")
    optimizer_reads = [path for path in audit.unique_paths if path.endswith("/optimizer.pt")]
    if optimizer_reads:
        raise RuntimeError(f"V47 opened forbidden source optimizer state: {optimizer_reads}")
    result["file_access_audit"] = {
        "passed": True,
        "loaded_files": audit.unique_paths,
        "loaded_training_maps": sorted(observed_maps),
        "forbidden_file_accesses": [],
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "protected_report_sha256_before_and_after": protected_before,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v46-terminal-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = (
        _preflight(
            config_path=args.config,
            output=args.output,
            v46_terminal_sha256=args.v46_terminal_sha256,
        )
        if args.preflight_only
        else run_v47(
            config_path=args.config,
            output=args.output,
            v46_terminal_sha256=args.v46_terminal_sha256,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("passed") is True else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_v47_schedule",
    "require_v46_report",
    "require_v46_terminal",
    "run_v47",
    "v47_contract",
    "v47_settings",
    "v47_update2_gate",
    "v47_update4_gate",
]
