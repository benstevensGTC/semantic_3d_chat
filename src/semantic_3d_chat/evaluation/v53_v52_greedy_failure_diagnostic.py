"""Detailed train-only diagnosis of the sole V52 alpha-2.0625 failure.

V53 authenticates the exact failed V52 report, reconstructs only its fixed
alpha-2.0625 candidate from immutable V47 update 004, and repeats the locked
training-only greedy evaluation while retaining row-level evidence.  It emits
normalized expected and generated answers for the 25 changed pair units and
the 48 unchanged broad-retention rows.  Training question text is used
transiently for generation but is not serialized.  Scene descriptions, oracle
metadata, validation data, and deferred-final data are never loaded.

This is a report-only diagnostic.  It constructs no optimizer, performs no
backward pass, writes no parameter state, and authorizes no checkpoint,
selector, promotion, chat, or embodied action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v51_query_alpha_grid as v51
from semantic_3d_chat.evaluation import v52_query_alpha_refinement as v52

AUTHORIZATION_ID = "v53_v52_greedy_failure_diagnostic"
V52_REPORT = Path("reports/gemma4/metrics/v52_query_alpha_refinement.json")
DEFAULT_REPORT = Path(
    "reports/gemma4/metrics/v53_v52_greedy_failure_diagnostic.json"
)
DEFAULT_CONFIG = v52.DEFAULT_CONFIG
SOURCE_CHECKPOINT = v52.SOURCE_CHECKPOINT
PREFIX_REFERENCE_CHECKPOINT = v52.PREFIX_REFERENCE_CHECKPOINT
PROTECTED_REPORT = v52.PROTECTED_REPORT

V52_REPORT_SHA256 = "6b653d1ff69d6b0dfe2cce6968478e5b47cef627550b282a2c8e7bc3cc197fd9"
TARGET_CANDIDATE_ID = "guarded_scene_alpha_1p0_query_alpha_2p0625"
TARGET_SCENE_ALPHA = 1.0
TARGET_QUERY_ALPHA = 2.0625
TARGET_SPEC = next(
    dict(value)
    for value in v52.CANDIDATE_GRID
    if value["candidate_id"] == TARGET_CANDIDATE_ID
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_EXPECTED_GREEDY = {
    "changed_unit_count": 25,
    "changed_row_count": 50,
    "changed_rows_exact_correct": 24,
    "complete_units": 4,
    "complete_physical_pair_coverage": 3,
    "complete_units_by_family": {
        "book_support": 0,
        "mirror_lr": 2,
        "picture_support": 0,
    },
    "broad_row_count": 48,
    "broad_exact_correct": 23,
}


@dataclass(frozen=True)
class DiagnosticPaths:
    predecessor: Path = V52_REPORT
    report: Path = DEFAULT_REPORT
    config: Path = DEFAULT_CONFIG


class DiagnosticBackend(Protocol):
    def authenticate_and_prepare(self) -> Mapping[str, Any]: ...

    def reconstruct_candidate(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def evaluate_non_greedy(self, candidate_id: str) -> Mapping[str, Any]: ...

    def detailed_greedy(
        self, candidate_id: str, teacher_metrics: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def restore_source(self) -> Mapping[str, Any]: ...

    def access_audit(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    combined = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(combined))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _locked_hash(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} is unavailable or unsafe: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{field} changed: expected {expected}, observed {observed}")


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


def _target_row(report: Mapping[str, Any]) -> Mapping[str, Any]:
    grid = _mapping(report.get("candidate_grid"), "V52 candidate grid")
    rows = _sequence(grid.get("candidates"), "V52 candidate rows")
    matches = [
        _mapping(value, "V52 candidate row")
        for value in rows
        if _mapping(value, "V52 candidate row").get("candidate") == TARGET_SPEC
    ]
    if len(matches) != 1:
        raise ValueError("V52 alpha-2.0625 candidate inventory changed")
    return matches[0]


def authenticate_predecessor(
    expected_sha256: str, path: str | Path = V52_REPORT
) -> dict[str, Any]:
    """Authenticate the exact V52 failure and its sole near-passing candidate."""

    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V53 expected V52 report SHA256 must be lowercase hexadecimal")
    if expected_sha256 != V52_REPORT_SHA256:
        raise ValueError("V53 invocation did not name the pinned V52 report SHA256")
    predecessor = _resolve(path)
    if predecessor != _resolve(V52_REPORT):
        raise ValueError("V53 predecessor path is pinned")
    if predecessor.is_symlink() or not predecessor.is_file():
        raise FileNotFoundError("V53 exact V52 report is unavailable or unsafe")
    payload = predecessor.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError("V53 V52 report differs from the explicit invocation SHA256")
    report = _mapping(json.loads(payload), "V52 report")
    target = _target_row(report)
    pre = _mapping(target.get("non_greedy_pre_gate"), "V52 target pre-gate")
    pre_checks = _mapping(pre.get("checks"), "V52 target pre-gate checks")
    greedy = _mapping(target.get("greedy_gate"), "V52 target greedy gate")
    greedy_checks = _mapping(greedy.get("checks"), "V52 target greedy checks")
    evidence = _mapping(greedy.get("evidence"), "V52 target greedy evidence")
    restoration = _mapping(target.get("source_restoration"), "V52 target restoration")
    final_restoration = _mapping(
        report.get("final_source_restoration"), "V52 final restoration"
    )
    access = _mapping(report.get("access_audit"), "V52 access audit")
    selection = _mapping(report.get("selection"), "V52 selection")
    checkpoint = _mapping(report.get("checkpoint"), "V52 checkpoint")
    expected_greedy = all(evidence.get(key) == value for key, value in _EXPECTED_GREEDY.items())
    checks = {
        "artifact": report.get("artifact") == "v52_query_alpha_refinement",
        "failed_without_winner": report.get("passed") is False
        and selection.get("winner") is None
        and selection.get("passing_candidate_ids") == [],
        "target_spec_exact": target.get("candidate") == TARGET_SPEC,
        "target_non_greedy_all_pass": pre.get("evaluated") is True
        and pre.get("passed") is True
        and bool(pre_checks)
        and all(value is True for value in pre_checks.values()),
        "target_only_full_failure_is_greedy_complete_units": greedy.get("authorized")
        is True
        and greedy.get("executed") is True
        and greedy.get("passed") is False
        and greedy.get("skipped_due_pre_gate") is False
        and {
            str(name)
            for name, passed in greedy_checks.items()
            if passed is not True
        }
        == {"train_greedy_complete_units_at_least_5"}
        and target.get("full_gate_passed") is False,
        "target_greedy_evidence_exact": expected_greedy
        and evidence.get("broad_exact_accuracy") == 23 / 48,
        "target_source_restored_exact": restoration.get("passed") is True
        and restoration.get("full_tensor_state_sha256") == v51._SOURCE_FULL_SHA256
        and restoration.get("authorized_surface_state_sha256")
        == v51._SOURCE_AUTHORIZED_SHA256
        and restoration.get("frozen_state_sha256") == v51._FROZEN_SHA256
        and restoration.get("all_parameter_gradients_absent") is True
        and target.get("evaluation_error") is None,
        "final_source_restored_exact": final_restoration.get("passed") is True
        and final_restoration.get("full_tensor_state_sha256")
        == v51._SOURCE_FULL_SHA256
        and final_restoration.get("authorized_surface_state_sha256")
        == v51._SOURCE_AUTHORIZED_SHA256
        and final_restoration.get("frozen_state_sha256") == v51._FROZEN_SHA256,
        "access_clean": access.get("passed") is True
        and access.get("training_map_count") == 16
        and access.get("optimizer_file_reads") == []
        and access.get("forbidden_file_accesses") == []
        and access.get("validation_qa_loaded") is False
        and access.get("oracle_loaded") is False
        and access.get("final_test_loaded") is False,
        "no_checkpoint_or_restricted_action": checkpoint.get("written") is False
        and checkpoint.get("inventory") is None
        and report.get("optimizer_constructed_or_loaded") is False
        and report.get("optimizer_state_file_opened") is False
        and report.get("optimizer_step_executed") is False
        and report.get("validation_qa_loaded") is False
        and report.get("validation_environment_maps_loaded") is False
        and report.get("oracle_loaded") is False
        and report.get("final_test_scenes_touched") is False
        and report.get("selector_executed") is False
        and report.get("runtime_promotion_executed") is False
        and report.get("chat_promotion_executed") is False
        and report.get("embodied_promotion_executed") is False,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"V52 report does not authorize V53: {failed}")
    return {
        "path": str(V52_REPORT),
        "sha256": observed,
        "authorization_id": AUTHORIZATION_ID,
        "checks": checks,
        "target": {
            "candidate": dict(TARGET_SPEC),
            "recorded_reconstruction": dict(
                _mapping(
                    target.get("candidate_reconstruction"),
                    "V52 target reconstruction",
                )
            ),
            "recorded_non_greedy": dict(pre),
            "recorded_greedy": dict(greedy),
        },
    }


def _family(pair_id: str) -> str:
    return {
        "pair_000015": "book_support",
        "pair_000016": "mirror_lr",
        "pair_000017": "picture_support",
    }.get(pair_id, "other")


def _teacher_index(pair_metrics: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = _sequence(pair_metrics.get("units"), "teacher unit rows")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for value in rows:
        row = _mapping(value, "teacher unit row")
        key = (str(row.get("pair_id")), str(row.get("question_key")))
        if key in result:
            raise ValueError(f"Duplicate teacher unit: {key}")
        result[key] = row
    if len(result) != 25:
        raise ValueError("V53 requires exactly 25 teacher unit rows")
    return result


def summarize_detailed_rows(
    *,
    units: Sequence[Mapping[str, Any]],
    broad_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate row detail and reproduce the locked aggregate greedy metrics."""

    if len(units) != 25 or len(broad_rows) != 48:
        raise ValueError("V53 detailed inventory must be 25 units and 48 broad rows")
    legacy_changed_correct = 0
    type_aware_changed_correct = 0
    legacy_complete_units = 0
    type_aware_complete_units = 0
    legacy_complete_pairs: set[str] = set()
    type_aware_complete_pairs: set[str] = set()
    legacy_by_family = {"book_support": 0, "mirror_lr": 0, "picture_support": 0}
    type_aware_by_family = {"book_support": 0, "mirror_lr": 0, "picture_support": 0}
    legacy_classifications = {
        "complete_success": 0,
        "one_sided_failure": 0,
        "complete_failure": 0,
    }
    type_aware_classifications = dict(legacy_classifications)
    observed_keys: set[tuple[str, str]] = set()
    teacher_greedy_disagreements = 0
    reordered_list_rescue_rows = 0
    reordered_list_rescue_units = 0
    for value in units:
        unit = _mapping(value, "detailed unit")
        key = (str(unit.get("pair_id")), str(unit.get("question_key")))
        if key in observed_keys:
            raise ValueError(f"Duplicate detailed unit: {key}")
        observed_keys.add(key)
        sides = _sequence(unit.get("sides"), "detailed sides")
        if len(sides) != 2:
            raise ValueError("Each detailed pair unit must contain two sides")
        legacy_correct = [
            _mapping(side, "detailed side").get("legacy_exact_correct") is True
            for side in sides
        ]
        type_aware_correct = [
            _mapping(side, "detailed side").get("type_aware_correct") is True
            for side in sides
        ]
        legacy_count = sum(legacy_correct)
        type_aware_count = sum(type_aware_correct)
        legacy_classification = (
            "complete_success"
            if legacy_count == 2
            else "one_sided_failure"
            if legacy_count == 1
            else "complete_failure"
        )
        type_aware_classification = (
            "complete_success"
            if type_aware_count == 2
            else "one_sided_failure"
            if type_aware_count == 1
            else "complete_failure"
        )
        if unit.get("legacy_failure_classification") != legacy_classification:
            raise ValueError("V53 legacy failure classification changed")
        if unit.get("type_aware_failure_classification") != type_aware_classification:
            raise ValueError("V53 type-aware failure classification changed")
        if unit.get("legacy_greedy_complete") is not (legacy_count == 2):
            raise ValueError("V53 legacy greedy-complete flag changed")
        if unit.get("type_aware_greedy_complete") is not (type_aware_count == 2):
            raise ValueError("V53 type-aware greedy-complete flag changed")
        legacy_classifications[legacy_classification] += 1
        type_aware_classifications[type_aware_classification] += 1
        legacy_changed_correct += legacy_count
        type_aware_changed_correct += type_aware_count
        legacy_complete_units += int(legacy_count == 2)
        type_aware_complete_units += int(type_aware_count == 2)
        rescued_sides = sum(
            _mapping(side, "detailed side").get("reordered_list_rescue") is True
            for side in sides
        )
        reordered_list_rescue_rows += rescued_sides
        reordered_list_rescue_units += int(rescued_sides > 0)
        if legacy_count == 2:
            pair_id = key[0]
            legacy_complete_pairs.add(pair_id)
            family = str(unit.get("family"))
            if family in legacy_by_family:
                legacy_by_family[family] += 1
        if type_aware_count == 2:
            pair_id = key[0]
            type_aware_complete_pairs.add(pair_id)
            family = str(unit.get("family"))
            if family in type_aware_by_family:
                type_aware_by_family[family] += 1
        teacher_greedy_disagreements += int(
            unit.get("teacher_complete") is True and legacy_count != 2
        )
    legacy_broad_correct = sum(
        _mapping(value, "broad row").get("legacy_exact_correct") is True
        for value in broad_rows
    )
    type_aware_broad_correct = sum(
        _mapping(value, "broad row").get("type_aware_correct") is True
        for value in broad_rows
    )
    broad_reordered_list_rescues = sum(
        _mapping(value, "broad row").get("reordered_list_rescue") is True
        for value in broad_rows
    )
    return {
        "schema_version": 1,
        "changed_unit_count": len(units),
        "changed_row_count": 2 * len(units),
        # Locked V52-compatible legacy aliases.
        "changed_rows_exact_correct": legacy_changed_correct,
        "complete_units": legacy_complete_units,
        "complete_physical_pair_coverage": len(legacy_complete_pairs),
        "complete_units_by_family": legacy_by_family,
        "broad_exact_correct": legacy_broad_correct,
        "broad_exact_accuracy": legacy_broad_correct / len(broad_rows),
        # Explicit dual-scoring evidence.
        "legacy_exact": {
            "changed_rows_correct": legacy_changed_correct,
            "complete_units": legacy_complete_units,
            "complete_physical_pair_coverage": len(legacy_complete_pairs),
            "complete_units_by_family": legacy_by_family,
            "failure_classification_counts": legacy_classifications,
            "broad_rows_correct": legacy_broad_correct,
            "broad_accuracy": legacy_broad_correct / len(broad_rows),
        },
        "type_aware": {
            "changed_rows_correct": type_aware_changed_correct,
            "complete_units": type_aware_complete_units,
            "complete_physical_pair_coverage": len(type_aware_complete_pairs),
            "complete_units_by_family": type_aware_by_family,
            "failure_classification_counts": type_aware_classifications,
            "broad_rows_correct": type_aware_broad_correct,
            "broad_accuracy": type_aware_broad_correct / len(broad_rows),
        },
        "reordered_list_rescue_rows": reordered_list_rescue_rows,
        "reordered_list_rescue_units": reordered_list_rescue_units,
        "broad_reordered_list_rescue_rows": broad_reordered_list_rescues,
        "teacher_complete_but_greedy_incomplete_units": teacher_greedy_disagreements,
        "broad_row_count": len(broad_rows),
    }


class RealDiagnosticBackend(v52.RealRefinementBackend):
    """Exact V52 reconstruction plus a row-retaining greedy evaluator."""

    def detailed_greedy(
        self, candidate_id: str, teacher_metrics: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        import torch

        from semantic_3d_chat.evaluation.metrics import (
            LIST_ANSWER_TYPES,
            exact_normalized_match,
            list_order_insensitive_match,
            normalize_answer,
        )
        from semantic_3d_chat.evaluation.scene_signal_audit import (
            _question_logits_and_answer,
        )
        from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
            current_scene_tokens,
        )

        if candidate_id != TARGET_CANDIDATE_ID or self._active_candidate_id != candidate_id:
            raise RuntimeError("V53 detailed greedy requires the live alpha-2.0625 candidate")
        delegate = self._delegate
        if not isinstance(teacher_metrics, Mapping):
            raise TypeError("V53 teacher metrics must be available before greedy detail")
        teacher_by_key = _teacher_index(teacher_metrics)
        units = tuple(sorted(delegate._units, key=lambda value: (value.pair_id, value.question_key)))
        broad_records = tuple(delegate._broad_records)
        if len(units) != 25 or len(broad_records) != 48:
            raise RuntimeError("V53 exact train-only row inventory changed")
        scene_ids = sorted(
            {record.scene_id for unit in units for record in unit.records}
            | {record.scene_id for record in broad_records}
        )
        model_dtype = next(delegate._bundle.language.model.parameters()).dtype
        delegate._block_core.eval()
        with torch.inference_mode():
            prefixes = {
                scene_id: delegate._bundle.composer.scene_prefix(
                    current_scene_tokens(
                        delegate._caches[scene_id],
                        delegate._block_core,
                        device=delegate._bundle.language.device,
                    ).to(model_dtype)
                )
                for scene_id in scene_ids
            }
        decoder = delegate._bundle.language.decoder_module
        was_training = bool(decoder.training)
        decoder.eval()
        detail_units: list[dict[str, Any]] = []
        broad_rows: list[dict[str, Any]] = []
        try:
            with torch.inference_mode():
                for unit in units:
                    key = (unit.pair_id, unit.question_key)
                    teacher_row = teacher_by_key[key]
                    side_margins = list(
                        _sequence(teacher_row.get("side_margins"), "teacher margins")
                    )
                    cross_margins = list(
                        _sequence(
                            teacher_row.get("cross_prefix_margins"),
                            "teacher cross margins",
                        )
                    )
                    sides: list[dict[str, Any]] = []
                    for side_index, record in enumerate(unit.records):
                        _, prediction, _generation = _question_logits_and_answer(
                            delegate._bundle.language,
                            prefixes[record.scene_id],
                            dict(delegate._loader),
                            record.question,
                        )
                        expected = normalize_answer(record.answer)
                        generated = normalize_answer(prediction)
                        legacy_correct = exact_normalized_match(
                            prediction, record.answer
                        )
                        type_aware_correct = (
                            list_order_insensitive_match(prediction, record.answer)
                            if record.answer_type in LIST_ANSWER_TYPES
                            else legacy_correct
                        )
                        sides.append(
                            {
                                "side_index": side_index,
                                "scene_id": record.scene_id,
                                "question_id": record.question_id,
                                "expected_normalized_answer": expected,
                                "generated_normalized_answer": generated,
                                "answer_type": record.answer_type,
                                "legacy_exact_correct": legacy_correct,
                                "type_aware_correct": type_aware_correct,
                                "reordered_list_rescue": bool(
                                    type_aware_correct and not legacy_correct
                                ),
                                "teacher_side_margin": float(side_margins[side_index]),
                                "teacher_cross_prefix_margin": float(
                                    cross_margins[side_index]
                                ),
                                "teacher_side_positive": float(
                                    side_margins[side_index]
                                )
                                > 0.0,
                            }
                        )
                    legacy_count = sum(
                        side["legacy_exact_correct"] is True for side in sides
                    )
                    type_aware_count = sum(
                        side["type_aware_correct"] is True for side in sides
                    )
                    legacy_classification = (
                        "complete_success"
                        if legacy_count == 2
                        else "one_sided_failure"
                        if legacy_count == 1
                        else "complete_failure"
                    )
                    type_aware_classification = (
                        "complete_success"
                        if type_aware_count == 2
                        else "one_sided_failure"
                        if type_aware_count == 1
                        else "complete_failure"
                    )
                    detail_units.append(
                        {
                            "pair_id": unit.pair_id,
                            "question_key": unit.question_key,
                            "scene_ids": list(unit.scene_ids),
                            "family": _family(unit.pair_id),
                            "teacher_complete": teacher_row.get("complete") is True,
                            "teacher_cross_prefix_complete": teacher_row.get(
                                "cross_prefix_complete"
                            )
                            is True,
                            "legacy_greedy_complete": legacy_count == 2,
                            "type_aware_greedy_complete": type_aware_count == 2,
                            "legacy_failure_classification": legacy_classification,
                            "type_aware_failure_classification": (
                                type_aware_classification
                            ),
                            "sides": sides,
                        }
                    )
                for record in broad_records:
                    _, prediction, _generation = _question_logits_and_answer(
                        delegate._bundle.language,
                        prefixes[record.scene_id],
                        dict(delegate._loader),
                        record.question,
                    )
                    expected = normalize_answer(record.answer)
                    generated = normalize_answer(prediction)
                    legacy_correct = exact_normalized_match(
                        prediction, record.answer
                    )
                    type_aware_correct = (
                        list_order_insensitive_match(prediction, record.answer)
                        if record.answer_type in LIST_ANSWER_TYPES
                        else legacy_correct
                    )
                    broad_rows.append(
                        {
                            "scene_id": record.scene_id,
                            "question_id": record.question_id,
                            "expected_normalized_answer": expected,
                            "generated_normalized_answer": generated,
                            "answer_type": record.answer_type,
                            "legacy_exact_correct": legacy_correct,
                            "type_aware_correct": type_aware_correct,
                            "reordered_list_rescue": bool(
                                type_aware_correct and not legacy_correct
                            ),
                        }
                    )
        finally:
            decoder.train(was_training)
        summary = summarize_detailed_rows(units=detail_units, broad_rows=broad_rows)
        return {
            "schema_version": 1,
            "summary": summary,
            "pair_units": detail_units,
            "broad_rows": broad_rows,
            "pair_units_sha256": _canonical_sha256(detail_units),
            "broad_rows_sha256": _canonical_sha256(broad_rows),
            "contains_question_text": False,
            "contains_normalized_training_answers": True,
            "training_scenes_only": True,
            "validation_qa_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
        }


def _restoration_exact(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("passed") is True
        and value.get("full_tensor_state_sha256") == v51._SOURCE_FULL_SHA256
        and value.get("authorized_surface_state_sha256")
        == v51._SOURCE_AUTHORIZED_SHA256
        and value.get("frozen_state_sha256") == v51._FROZEN_SHA256
        and value.get("all_parameter_gradients_absent") is True
    )


def execute_diagnostic(
    *, predecessor: Mapping[str, Any], backend: DiagnosticBackend
) -> dict[str, Any]:
    """Run one authenticated alpha-2.0625 reconstruction and detailed probe."""

    preparation: Mapping[str, Any] = {}
    reconstruction: Mapping[str, Any] = {}
    non_greedy: Mapping[str, Any] = {}
    detail: Mapping[str, Any] = {}
    restoration: Mapping[str, Any] = {"attempted": False, "passed": False}
    access: Mapping[str, Any] = {"passed": False}
    errors: list[dict[str, str]] = []
    try:
        preparation = backend.authenticate_and_prepare()
        reconstruction = backend.reconstruct_candidate(TARGET_SPEC)
        non_greedy = backend.evaluate_non_greedy(TARGET_CANDIDATE_ID)
        pair_metrics = _mapping(non_greedy.get("pair_metrics"), "V53 pair metrics")
        predecessor_target = _mapping(predecessor.get("target"), "V53 predecessor target")
        recorded_reconstruction = _mapping(
            predecessor_target.get("recorded_reconstruction"),
            "V53 recorded reconstruction",
        )
        for field in (
            "full_tensor_state_sha256",
            "authorized_surface_state_sha256",
            "scene_readout_state_sha256",
            "query_state_sha256",
            "frozen_state_sha256",
        ):
            if reconstruction.get(field) != recorded_reconstruction.get(field):
                raise RuntimeError(f"V53 reconstructed candidate changed: {field}")
        recorded_pre = _mapping(
            predecessor_target.get("recorded_non_greedy"),
            "V53 recorded non-greedy gate",
        )
        recorded_evidence = _mapping(
            recorded_pre.get("evidence"), "V53 recorded non-greedy evidence"
        )
        if _canonical_sha256(pair_metrics) != recorded_evidence.get(
            "pair_metrics_sha256"
        ):
            raise RuntimeError("V53 teacher pair metrics differ from exact V52")
        per_unit = list(
            _sequence(
                non_greedy.get("per_unit_nll_diagnostics"),
                "V53 per-unit NLL diagnostics",
            )
        )
        if _canonical_sha256(per_unit) != recorded_evidence.get("per_unit_nll_sha256"):
            raise RuntimeError("V53 teacher per-unit NLL differs from exact V52")
        if (
            non_greedy.get("broad_nll") != recorded_evidence.get("broad_nll")
            or non_greedy.get("broad_row_count") != 48
        ):
            raise RuntimeError("V53 broad teacher evidence differs from exact V52")
        pre_checks = v51.non_greedy_pre_gate_checks(reconstruction, non_greedy)
        if not pre_checks or not all(pre_checks.values()):
            raise RuntimeError("V53 alpha-2.0625 no longer passes every non-greedy gate")
        detail = backend.detailed_greedy(TARGET_CANDIDATE_ID, pair_metrics)
        summary = _mapping(detail.get("summary"), "V53 detailed summary")
        if any(summary.get(key) != value for key, value in _EXPECTED_GREEDY.items()):
            raise RuntimeError("V53 detailed greedy aggregate differs from exact V52")
        if summary.get("broad_exact_accuracy") != 23 / 48:
            raise RuntimeError("V53 detailed broad accuracy differs from exact V52")
    except Exception as exc:  # noqa: BLE001 - diagnostic errors are sealed
        errors.append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        try:
            restoration = {"attempted": True, **dict(backend.restore_source())}
        except Exception as exc:  # noqa: BLE001
            restoration = {
                "attempted": True,
                "passed": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            errors.append(
                {"type": type(exc).__name__, "message": f"source restoration failed: {exc}"}
            )
        try:
            access = dict(backend.access_audit())
        except Exception as exc:  # noqa: BLE001
            access = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            errors.append(
                {"type": type(exc).__name__, "message": f"access audit failed: {exc}"}
            )
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001
            errors.append({"type": type(exc).__name__, "message": f"close failed: {exc}"})
    access_exact = bool(
        access.get("passed") is True
        and access.get("training_map_count") == 16
        and access.get("optimizer_file_reads") == []
        and access.get("forbidden_file_accesses") == []
        and access.get("validation_qa_loaded") is False
        and access.get("oracle_loaded") is False
        and access.get("final_test_loaded") is False
    )
    passed = bool(
        detail
        and _restoration_exact(restoration)
        and access_exact
        and not errors
    )
    return {
        "schema_version": 1,
        "artifact": AUTHORIZATION_ID,
        "passed": passed,
        "authorization": {
            "predecessor_path": predecessor["path"],
            "predecessor_sha256": predecessor["sha256"],
            "authorization_id": AUTHORIZATION_ID,
            "checks": dict(_mapping(predecessor.get("checks"), "predecessor checks")),
        },
        "scope": {
            "target_candidate": dict(TARGET_SPEC),
            "only_one_candidate_reconstructed": True,
            "report_only": True,
            "checkpoint_written": False,
            "optimizer_constructed_or_loaded": False,
            "optimizer_state_file_opened": False,
            "optimizer_step_executed": False,
            "parameter_gradient_accumulation": False,
            "validation_qa_loaded": False,
            "validation_environment_maps_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
            "selector_executed": False,
            "runtime_promotion_executed": False,
            "chat_promotion_executed": False,
            "embodied_promotion_executed": False,
        },
        "preparation": dict(preparation),
        "candidate_reconstruction": dict(reconstruction),
        "non_greedy": {
            "pair_metrics_sha256": (
                _canonical_sha256(non_greedy.get("pair_metrics")) if non_greedy else None
            ),
            "broad_nll": non_greedy.get("broad_nll"),
            "broad_row_count": non_greedy.get("broad_row_count"),
            "all_v51_checks_passed": bool(non_greedy)
            and all(
                v51.non_greedy_pre_gate_checks(reconstruction, non_greedy).values()
            ),
        },
        "detailed_greedy": dict(detail),
        "source_restoration": dict(restoration),
        "access_audit": dict(access),
        "execution_errors": errors,
        "checkpoint_written": False,
        "optimizer_constructed_or_loaded": False,
        "optimizer_state_file_opened": False,
        "optimizer_step_executed": False,
        "validation_qa_loaded": False,
        "validation_environment_maps_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_executed": False,
        "runtime_promotion_executed": False,
        "chat_promotion_executed": False,
        "embodied_promotion_executed": False,
    }


def _resolved_paths(paths: DiagnosticPaths | None) -> DiagnosticPaths:
    selected = DiagnosticPaths() if paths is None else paths
    resolved = DiagnosticPaths(
        predecessor=_resolve(selected.predecessor),
        report=_resolve(selected.report),
        config=_resolve(selected.config),
    )
    expected = DiagnosticPaths(
        predecessor=_resolve(V52_REPORT),
        report=_resolve(DEFAULT_REPORT),
        config=_resolve(DEFAULT_CONFIG),
    )
    if resolved != expected:
        raise ValueError("V53 predecessor, report, and config paths are pinned")
    return resolved


def preflight(
    *, expected_v52_report_sha256: str, paths: DiagnosticPaths | None = None
) -> dict[str, Any]:
    """Authenticate V53 without loading Gemma, QA records, or scene maps."""

    resolved = _resolved_paths(paths)
    if resolved.report.is_symlink() or resolved.report.exists():
        raise FileExistsError("V53 report is one-shot and already exists")
    predecessor = authenticate_predecessor(
        expected_v52_report_sha256, resolved.predecessor
    )
    _locked_hash(resolved.config, v51._CONFIG_SHA256, "V53 config")
    _locked_hash(
        _resolve(PROTECTED_REPORT), v51._PROTECTED_REPORT_SHA256, "V53 protected report"
    )
    source = _resolve(SOURCE_CHECKPOINT)
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("V53 source checkpoint is unavailable")
    if sorted(path.name for path in source.iterdir()) != sorted(v51._SOURCE_FILES):
        raise ValueError("V53 source checkpoint inventory changed")
    for name, digest in v51._SOURCE_FILES.items():
        if name != "optimizer.pt":
            _locked_hash(source / name, digest, f"V53 source {name}")
    prefix = _resolve(PREFIX_REFERENCE_CHECKPOINT)
    if prefix.is_symlink() or not prefix.is_dir():
        raise FileNotFoundError("V53 prefix reference checkpoint is unavailable")
    if sorted(path.name for path in prefix.iterdir()) != sorted(
        v51._PREFIX_REFERENCE_FILES
    ):
        raise ValueError("V53 prefix reference checkpoint inventory changed")
    for name, digest in v51._PREFIX_REFERENCE_FILES.items():
        _locked_hash(prefix / name, digest, f"V53 prefix reference {name}")
    return {
        "schema_version": 1,
        "artifact": f"{AUTHORIZATION_ID}_preflight",
        "passed": True,
        "predecessor": predecessor,
        "target_candidate": dict(TARGET_SPEC),
        "model_loaded": False,
        "qa_loaded": False,
        "maps_loaded": False,
        "checkpoint_written": False,
        "optimizer_constructed_or_loaded": False,
        "optimizer_state_file_opened": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_loaded": False,
    }


def run_diagnostic(
    *,
    expected_v52_report_sha256: str,
    paths: DiagnosticPaths | None = None,
    backend_factory: Callable[[Mapping[str, Any], v51.GridPaths], DiagnosticBackend]
    | None = None,
) -> dict[str, Any]:
    """Run the exact one-shot V53 detailed train-only diagnostic."""

    resolved = _resolved_paths(paths)
    if resolved.report.is_symlink() or resolved.report.exists():
        raise FileExistsError("V53 report is one-shot and will not be overwritten")
    _locked_hash(resolved.config, v51._CONFIG_SHA256, "V53 config")
    predecessor = authenticate_predecessor(
        expected_v52_report_sha256, resolved.predecessor
    )
    engine_paths = v51.GridPaths(
        terminal=resolved.predecessor,
        report=resolved.report,
        checkpoint_root=_resolve(v52.DEFAULT_CHECKPOINT_ROOT),
        config=resolved.config,
    )
    factory = RealDiagnosticBackend if backend_factory is None else backend_factory
    with v52.scoped_v51_refinement():
        backend = factory(predecessor, engine_paths)
        report = execute_diagnostic(predecessor=predecessor, backend=backend)
    _atomic_json(resolved.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-v52-report-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--predecessor", type=Path, default=V52_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    paths = DiagnosticPaths(
        predecessor=args.predecessor,
        report=args.report,
        config=args.config,
    )
    if args.preflight:
        result = preflight(
            expected_v52_report_sha256=args.expected_v52_report_sha256,
            paths=paths,
        )
        summary = {
            "artifact": result["artifact"],
            "passed": result["passed"],
            "predecessor_sha256": result["predecessor"]["sha256"],
            "target_candidate": result["target_candidate"]["candidate_id"],
            "model_loaded": False,
            "qa_loaded": False,
            "maps_loaded": False,
        }
    else:
        result = run_diagnostic(
            expected_v52_report_sha256=args.expected_v52_report_sha256,
            paths=paths,
        )
        summary = {
            "artifact": result["artifact"],
            "passed": result["passed"],
            "report": str(DEFAULT_REPORT),
            "report_sha256": _sha256(_resolve(DEFAULT_REPORT)),
            "complete_units": result.get("detailed_greedy", {})
            .get("summary", {})
            .get("complete_units"),
            "broad_exact_correct": result.get("detailed_greedy", {})
            .get("summary", {})
            .get("broad_exact_correct"),
        }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_ID",
    "DEFAULT_REPORT",
    "TARGET_CANDIDATE_ID",
    "TARGET_SPEC",
    "V52_REPORT_SHA256",
    "DiagnosticBackend",
    "DiagnosticPaths",
    "RealDiagnosticBackend",
    "authenticate_predecessor",
    "execute_diagnostic",
    "main",
    "preflight",
    "run_diagnostic",
    "summarize_detailed_rows",
]
