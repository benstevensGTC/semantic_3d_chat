from __future__ import annotations

import copy
import math

import pytest

from semantic_3d_chat.evaluation.v21_predicted_update_audit import (
    V21_FUNCTIONAL_AUDIT_TYPE,
    V21PredictedUpdateAuditViolation,
    evaluate_v21_predicted_update,
)

COLOR_PAIR_ID = "pair_000001"
MIRROR_PAIR_ID = "pair_000003"


def _policies() -> dict[str, dict[str, float]]:
    return {
        COLOR_PAIR_ID: {
            "candidate_target_margin": 0.25,
            "candidate_hinge_weight": 8.0,
            "full_vocab_target_margin": 0.25,
            "full_vocab_hinge_weight": 2.0,
        },
        MIRROR_PAIR_ID: {
            "candidate_target_margin": 1.0,
            "candidate_hinge_weight": 8.0,
            "full_vocab_target_margin": 1.0,
            "full_vocab_hinge_weight": 2.0,
        },
    }


def _unit(
    pair_id: str,
    index: int,
    candidate: tuple[float, float],
    full_vocab: tuple[float, float],
) -> dict:
    scene_ids = ("scene_000003", "scene_000004")
    if pair_id == MIRROR_PAIR_ID:
        scene_ids = ("scene_000007", "scene_000008")
    return {
        "pair_id": pair_id,
        "question_key": f"cfq_{pair_id[-1]}_{index:02d}",
        "sides": [
            {
                "scene_id": scene_ids[0],
                "candidate_margin": candidate[0],
                "full_vocab_margin": full_vocab[0],
            },
            {
                "scene_id": scene_ids[1],
                "candidate_margin": candidate[1],
                "full_vocab_margin": full_vocab[1],
            },
        ],
    }


def _measurements(*, mirror_margin: float) -> list[dict]:
    units = [_unit(COLOR_PAIR_ID, index, (0.5, 0.75), (0.375, 0.625)) for index in range(6)]
    units.extend(
        _unit(
            MIRROR_PAIR_ID,
            index,
            (mirror_margin, mirror_margin - 0.25),
            (mirror_margin - 0.125, mirror_margin - 0.375),
        )
        for index in range(6)
    )
    return units


def _evaluate(before: list[dict], after: list[dict]) -> dict:
    return evaluate_v21_predicted_update(
        before,
        after,
        policies=_policies(),
        color_pair_id=COLOR_PAIR_ID,
        mirror_pair_id=MIRROR_PAIR_ID,
    )


def test_passes_on_strict_color_retention_and_partial_mirror_objective_improvement() -> None:
    before = _measurements(mirror_margin=-0.5)
    after = _measurements(mirror_margin=-0.25)

    report = _evaluate(before, after)

    assert report["audit_type"] == V21_FUNCTIONAL_AUDIT_TYPE
    assert report["passed"] is True
    assert report["gate_policy"]["mirror_immediate_full_pass_required"] is False
    assert report["summaries"]["after"][COLOR_PAIR_ID]["candidate"]["correct_sides"] == 12
    assert report["summaries"]["after"][COLOR_PAIR_ID]["full_vocab"]["correct_units"] == 6
    mirror_after = report["summaries"]["after"][MIRROR_PAIR_ID]
    assert mirror_after["candidate"]["correct_sides"] == 0
    assert mirror_after["full_vocab"]["correct_units"] == 0
    assert report["mirror_objective_change"]["absolute_improvement"] == pytest.approx(2.5)


def test_weighted_objective_matches_candidate_and_full_vocab_training_policy() -> None:
    before = _measurements(mirror_margin=0.0)
    after = _measurements(mirror_margin=0.125)

    report = _evaluate(before, after)
    mirror = report["summaries"]["before"][MIRROR_PAIR_ID]

    assert mirror["candidate"]["mean_margin_hinge"] == pytest.approx(1.125)
    assert mirror["full_vocab"]["mean_margin_hinge"] == pytest.approx(1.25)
    assert mirror["weighted_margin_hinge_objective"] == pytest.approx(11.5)
    assert mirror["objective_components"]["weighted_candidate_mean_hinge"] == pytest.approx(9.0)
    assert mirror["objective_components"]["weighted_full_vocab_mean_hinge"] == pytest.approx(2.5)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_color_must_be_strict_before_and_after(phase: str) -> None:
    before = _measurements(mirror_margin=-0.5)
    after = _measurements(mirror_margin=-0.25)
    selected = before if phase == "before" else after
    selected[0]["sides"][0]["full_vocab_margin"] = 0.0

    report = _evaluate(before, after)

    assert report["passed"] is False
    assert report["gate_checks"][f"color_{phase}_strict_positive_12_sides_6_units"] is False
    other = "after" if phase == "before" else "before"
    assert report["gate_checks"][f"color_{other}_strict_positive_12_sides_6_units"] is True


@pytest.mark.parametrize("after_margin", [-0.5, -0.75])
def test_mirror_objective_must_strictly_improve(after_margin: float) -> None:
    before = _measurements(mirror_margin=-0.5)
    after = _measurements(mirror_margin=after_margin)

    report = _evaluate(before, after)

    assert report["passed"] is False
    assert (
        report["gate_checks"]["mirror_weighted_margin_hinge_objective_strictly_improved"] is False
    )


def test_measurement_order_is_irrelevant_but_identities_must_match() -> None:
    before = _measurements(mirror_margin=-0.5)
    after = list(reversed(_measurements(mirror_margin=-0.25)))
    for unit in after:
        unit["sides"].reverse()

    assert _evaluate(before, after)["passed"] is True

    after[0]["question_key"] = "cfq_changed"
    with pytest.raises(V21PredictedUpdateAuditViolation, match="identities differ"):
        _evaluate(before, after)


def test_gate_cardinality_is_pinned_to_six_units_and_twelve_sides() -> None:
    with pytest.raises(V21PredictedUpdateAuditViolation, match="exactly six"):
        evaluate_v21_predicted_update(
            _measurements(mirror_margin=-0.5),
            _measurements(mirror_margin=-0.25),
            policies=_policies(),
            color_pair_id=COLOR_PAIR_ID,
            mirror_pair_id=MIRROR_PAIR_ID,
            expected_units_per_pair=5,
        )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda values: values.append(copy.deepcopy(values[0])), "duplicate unit"),
        (
            lambda values: values[0]["sides"][0].__setitem__("candidate_margin", math.inf),
            "must be finite",
        ),
        (lambda values: values[0]["sides"].pop(), "exactly two"),
        (
            lambda values: values[1]["sides"][0].__setitem__("scene_id", "scene_changed"),
            "exact scene pair",
        ),
    ],
)
def test_malformed_rich_measurements_fail_closed(mutate, match: str) -> None:
    before = _measurements(mirror_margin=-0.5)
    mutate(before)

    with pytest.raises(V21PredictedUpdateAuditViolation, match=match):
        _evaluate(before, _measurements(mirror_margin=-0.25))
