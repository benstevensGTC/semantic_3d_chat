"""Pure teacher-forced functional audit for a predicted V21 update.

The helper deliberately does not load a model, checkpoint, scene, or QA file.
A V21 preflight supplies paired before/after measurements produced by the real
Gemma forward path.  This module validates that evidence, summarizes candidate
and full-vocabulary correctness, and applies the predeclared functional gate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

V21_FUNCTIONAL_AUDIT_TYPE = "v21_predicted_update_teacher_forced_functional_audit"

_UNIT_KEYS = frozenset({"pair_id", "question_key", "sides"})
_SIDE_KEYS = frozenset({"scene_id", "candidate_margin", "full_vocab_margin"})
_POLICY_KEYS = frozenset(
    {
        "candidate_target_margin",
        "candidate_hinge_weight",
        "full_vocab_target_margin",
        "full_vocab_hinge_weight",
    }
)


class V21PredictedUpdateAuditViolation(ValueError):
    """Raised when functional evidence is incomplete or internally inconsistent."""


def _fail(message: str) -> None:
    raise V21PredictedUpdateAuditViolation(message)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a nonempty string")
    return value


def _finite_number(value: object, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail(f"{field} must be finite")
    if nonnegative and parsed < 0.0:
        _fail(f"{field} must be nonnegative")
    return parsed


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(f"{field} must be a positive integer")
    return value


def _normalize_policies(
    policies: Mapping[str, Mapping[str, object]],
    pair_ids: tuple[str, str],
) -> dict[str, dict[str, float]]:
    raw = _mapping(policies, "policies")
    if set(raw) != set(pair_ids):
        _fail("policies must contain exactly the color and mirror pair IDs")
    normalized: dict[str, dict[str, float]] = {}
    for pair_id in pair_ids:
        policy = _mapping(raw[pair_id], f"policies.{pair_id}")
        if set(policy) != _POLICY_KEYS:
            _fail(f"policies.{pair_id} keys mismatch")
        values = {
            "candidate_target_margin": _finite_number(
                policy["candidate_target_margin"],
                f"policies.{pair_id}.candidate_target_margin",
                nonnegative=True,
            ),
            "candidate_hinge_weight": _finite_number(
                policy["candidate_hinge_weight"],
                f"policies.{pair_id}.candidate_hinge_weight",
                nonnegative=True,
            ),
            "full_vocab_target_margin": _finite_number(
                policy["full_vocab_target_margin"],
                f"policies.{pair_id}.full_vocab_target_margin",
                nonnegative=True,
            ),
            "full_vocab_hinge_weight": _finite_number(
                policy["full_vocab_hinge_weight"],
                f"policies.{pair_id}.full_vocab_hinge_weight",
                nonnegative=True,
            ),
        }
        if values["candidate_hinge_weight"] + values["full_vocab_hinge_weight"] <= 0.0:
            _fail(f"policies.{pair_id} must assign positive aggregate hinge weight")
        normalized[pair_id] = values
    return normalized


def _normalize_measurements(
    measurements: Sequence[Mapping[str, object]],
    *,
    field: str,
    pair_ids: tuple[str, str],
    expected_units_per_pair: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    units = _sequence(measurements, field)
    normalized: dict[tuple[str, str], dict[str, Any]] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    pair_scene_ids: dict[str, tuple[str, str]] = {}
    for index, raw_unit in enumerate(units):
        unit = _mapping(raw_unit, f"{field}[{index}]")
        if set(unit) != _UNIT_KEYS:
            _fail(f"{field}[{index}] keys mismatch")
        pair_id = _identifier(unit["pair_id"], f"{field}[{index}].pair_id")
        if pair_id not in pair_ids:
            _fail(f"{field}[{index}] has an unexpected pair ID")
        question_key = _identifier(unit["question_key"], f"{field}[{index}].question_key")
        identity = (pair_id, question_key)
        if identity in normalized:
            _fail(f"{field} contains duplicate unit {identity}")
        raw_sides = _sequence(unit["sides"], f"{field}[{index}].sides")
        if len(raw_sides) != 2:
            _fail(f"{field}[{index}] must contain exactly two scene sides")
        sides: list[dict[str, Any]] = []
        for side_index, raw_side in enumerate(raw_sides):
            side = _mapping(raw_side, f"{field}[{index}].sides[{side_index}]")
            if set(side) != _SIDE_KEYS:
                _fail(f"{field}[{index}].sides[{side_index}] keys mismatch")
            sides.append(
                {
                    "scene_id": _identifier(
                        side["scene_id"],
                        f"{field}[{index}].sides[{side_index}].scene_id",
                    ),
                    "candidate_margin": _finite_number(
                        side["candidate_margin"],
                        f"{field}[{index}].sides[{side_index}].candidate_margin",
                    ),
                    "full_vocab_margin": _finite_number(
                        side["full_vocab_margin"],
                        f"{field}[{index}].sides[{side_index}].full_vocab_margin",
                    ),
                }
            )
        sides.sort(key=lambda item: item["scene_id"])
        scene_ids = tuple(side["scene_id"] for side in sides)
        if len(set(scene_ids)) != 2:
            _fail(f"{field}[{index}] scene sides must be distinct")
        previous_scene_ids = pair_scene_ids.setdefault(pair_id, scene_ids)
        if scene_ids != previous_scene_ids:
            _fail(f"{field}.{pair_id} units do not share one exact scene pair")
        normalized[identity] = {
            "pair_id": pair_id,
            "question_key": question_key,
            "sides": sides,
        }
        counts[pair_id] += 1
    expected_counts = {pair_id: expected_units_per_pair for pair_id in pair_ids}
    if dict(counts) != expected_counts:
        _fail(f"{field} unit counts mismatch: expected={expected_counts} observed={dict(counts)}")
    return normalized


def _metric_summary(
    units: Sequence[Mapping[str, Any]],
    *,
    margin_key: str,
    target_margin: float,
) -> dict[str, Any]:
    margins = [
        float(side[margin_key])
        for unit in units
        for side in _sequence(unit["sides"], "normalized unit sides")
    ]
    unit_correct = sum(
        all(float(side[margin_key]) > 0.0 for side in unit["sides"]) for unit in units
    )
    hinges = [max(0.0, target_margin - margin) for margin in margins]
    return {
        "total_sides": len(margins),
        "correct_sides": sum(margin > 0.0 for margin in margins),
        "total_units": len(units),
        "correct_units": unit_correct,
        "minimum_margin": min(margins),
        "mean_margin": sum(margins) / len(margins),
        "target_margin": target_margin,
        "mean_margin_hinge": sum(hinges) / len(hinges),
        "maximum_margin_hinge": max(hinges),
    }


def _pair_summary(
    normalized: Mapping[tuple[str, str], Mapping[str, Any]],
    pair_id: str,
    policy: Mapping[str, float],
) -> dict[str, Any]:
    units = [normalized[key] for key in sorted(normalized) if key[0] == pair_id]
    candidate = _metric_summary(
        units,
        margin_key="candidate_margin",
        target_margin=policy["candidate_target_margin"],
    )
    full_vocab = _metric_summary(
        units,
        margin_key="full_vocab_margin",
        target_margin=policy["full_vocab_target_margin"],
    )
    objective = (
        policy["candidate_hinge_weight"] * candidate["mean_margin_hinge"]
        + policy["full_vocab_hinge_weight"] * full_vocab["mean_margin_hinge"]
    )
    return {
        "pair_id": pair_id,
        "scene_ids": [side["scene_id"] for side in units[0]["sides"]],
        "question_keys": [unit["question_key"] for unit in units],
        "candidate": candidate,
        "full_vocab": full_vocab,
        "weighted_margin_hinge_objective": objective,
        "objective_components": {
            "candidate_hinge_weight": policy["candidate_hinge_weight"],
            "weighted_candidate_mean_hinge": (
                policy["candidate_hinge_weight"] * candidate["mean_margin_hinge"]
            ),
            "full_vocab_hinge_weight": policy["full_vocab_hinge_weight"],
            "weighted_full_vocab_mean_hinge": (
                policy["full_vocab_hinge_weight"] * full_vocab["mean_margin_hinge"]
            ),
        },
    }


def _strict_pair_pass(summary: Mapping[str, Any], *, sides: int, units: int) -> bool:
    candidate = _mapping(summary["candidate"], "candidate summary")
    full_vocab = _mapping(summary["full_vocab"], "full-vocabulary summary")
    return bool(
        candidate["total_sides"] == sides
        and candidate["correct_sides"] == sides
        and candidate["total_units"] == units
        and candidate["correct_units"] == units
        and candidate["minimum_margin"] > 0.0
        and full_vocab["total_sides"] == sides
        and full_vocab["correct_sides"] == sides
        and full_vocab["total_units"] == units
        and full_vocab["correct_units"] == units
        and full_vocab["minimum_margin"] > 0.0
    )


def evaluate_v21_predicted_update(
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
    *,
    policies: Mapping[str, Mapping[str, object]],
    color_pair_id: str,
    mirror_pair_id: str,
    expected_units_per_pair: int = 6,
) -> dict[str, Any]:
    """Validate and gate one isolated predicted update using Gemma measurements.

    Color must be a strict 12-side/6-unit pass both before and after the
    predicted update.  Mirror need not pass immediately; its weighted mean
    candidate/full-vocabulary margin-hinge objective must strictly decrease.
    """

    color_pair_id = _identifier(color_pair_id, "color_pair_id")
    mirror_pair_id = _identifier(mirror_pair_id, "mirror_pair_id")
    if color_pair_id == mirror_pair_id:
        _fail("color_pair_id and mirror_pair_id must differ")
    units_per_pair = _positive_int(expected_units_per_pair, "expected_units_per_pair")
    if units_per_pair != 6:
        _fail("V21 functional audit requires exactly six units per pair")
    pair_ids = (color_pair_id, mirror_pair_id)
    normalized_policies = _normalize_policies(policies, pair_ids)
    before_units = _normalize_measurements(
        before,
        field="before",
        pair_ids=pair_ids,
        expected_units_per_pair=units_per_pair,
    )
    after_units = _normalize_measurements(
        after,
        field="after",
        pair_ids=pair_ids,
        expected_units_per_pair=units_per_pair,
    )
    if set(before_units) != set(after_units):
        _fail("Before/after measurement unit identities differ")
    identity_mismatches = [
        identity
        for identity in sorted(before_units)
        if [side["scene_id"] for side in before_units[identity]["sides"]]
        != [side["scene_id"] for side in after_units[identity]["sides"]]
    ]
    if identity_mismatches:
        _fail(f"Before/after scene-side identities differ: {identity_mismatches}")

    summaries = {
        phase: {
            pair_id: _pair_summary(units, pair_id, normalized_policies[pair_id])
            for pair_id in pair_ids
        }
        for phase, units in (("before", before_units), ("after", after_units))
    }
    expected_sides = 2 * units_per_pair
    color_before = summaries["before"][color_pair_id]
    color_after = summaries["after"][color_pair_id]
    mirror_before = summaries["before"][mirror_pair_id]
    mirror_after = summaries["after"][mirror_pair_id]
    mirror_before_objective = float(mirror_before["weighted_margin_hinge_objective"])
    mirror_after_objective = float(mirror_after["weighted_margin_hinge_objective"])
    improvement = mirror_before_objective - mirror_after_objective
    checks = {
        "color_before_strict_positive_12_sides_6_units": _strict_pair_pass(
            color_before, sides=expected_sides, units=units_per_pair
        ),
        "color_after_strict_positive_12_sides_6_units": _strict_pair_pass(
            color_after, sides=expected_sides, units=units_per_pair
        ),
        "mirror_weighted_margin_hinge_objective_strictly_improved": improvement > 0.0,
    }
    normalized_measurements = {
        "before": [before_units[key] for key in sorted(before_units)],
        "after": [after_units[key] for key in sorted(after_units)],
    }
    return {
        "schema_version": 1,
        "audit_type": V21_FUNCTIONAL_AUDIT_TYPE,
        "pair_roles": {"color": color_pair_id, "mirror": mirror_pair_id},
        "expected_units_per_pair": units_per_pair,
        "expected_sides_per_pair": expected_sides,
        "policies": normalized_policies,
        "measurements": normalized_measurements,
        "summaries": summaries,
        "mirror_objective_change": {
            "before": mirror_before_objective,
            "after": mirror_after_objective,
            "absolute_improvement": improvement,
            "strictly_improved": improvement > 0.0,
        },
        "gate_policy": {
            "color_must_remain_strict_positive": True,
            "mirror_immediate_full_pass_required": False,
            "mirror_aggregate_objective_must_strictly_improve": True,
        },
        "gate_checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "V21_FUNCTIONAL_AUDIT_TYPE",
    "V21PredictedUpdateAuditViolation",
    "evaluate_v21_predicted_update",
]
