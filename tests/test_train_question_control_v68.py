from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    TRAIN_PAIR_IDS,
)
from semantic_3d_chat.evaluation.v68_regularized_pair_preregistration import (
    V68_ARM_GRID,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training import train_question_control_v68 as v68


def _passing_metrics() -> dict[str, object]:
    threshold = v68.V67_NUMERIC_SCREEN_THRESHOLDS
    return {
        "supported_class_exact": threshold.held_supported_class_exact_minimum,
        "supported_total": threshold.held_supported_total,
        "unsupported_total": threshold.held_unsupported_total,
        "inventory_total": 576,
        "changed_class_exact": threshold.held_changed_class_exact_minimum,
        "changed_total": threshold.held_changed_total,
        "complete_class_units": threshold.held_complete_class_units_minimum,
        "complete_unit_total": threshold.held_complete_unit_total,
        "prediction_change_units": threshold.held_prediction_change_units_minimum,
        "pair_delta_cosine_sum": threshold.mean_pair_delta_cosine_minimum
        * threshold.held_complete_unit_total,
        "positive_pair_delta_units": threshold.positive_pair_delta_units_minimum,
        "own_over_opposite_margin_sum": (
            threshold.mean_own_over_opposite_margin_minimum * threshold.fully_supported_pair_sides
        ),
        "positive_own_over_opposite_sides": (threshold.positive_own_over_opposite_sides_minimum),
        "fully_supported_pair_sides": threshold.fully_supported_pair_sides,
        "mean_pair_delta_cosine": threshold.mean_pair_delta_cosine_minimum,
        "mean_own_over_opposite_margin": (threshold.mean_own_over_opposite_margin_minimum),
        "answer_or_question_text_stored": False,
        "gemma_generation_used": False,
    }


def _fold_metrics_from_aggregate(aggregate: dict[str, object]) -> list[dict[str, object]]:
    sum_fields = (
        "supported_class_exact",
        "supported_total",
        "unsupported_total",
        "inventory_total",
        "changed_class_exact",
        "changed_total",
        "complete_class_units",
        "complete_unit_total",
        "prediction_change_units",
        "pair_delta_cosine_sum",
        "positive_pair_delta_units",
        "own_over_opposite_margin_sum",
        "positive_own_over_opposite_sides",
        "fully_supported_pair_sides",
    )
    folds: list[dict[str, object]] = []
    for index in range(12):
        fold: dict[str, object] = {}
        for field in sum_fields:
            value = aggregate[field]
            if isinstance(value, float):
                fold[field] = value / 12.0
            else:
                quotient, remainder = divmod(int(value), 12)
                fold[field] = quotient + (1 if index < remainder else 0)
        fold["question_or_answer_text_stored"] = False
        fold["gemma_generation_used"] = False
        folds.append(fold)
    return folds


def _arm_result(arm: dict[str, object], *, passing: bool) -> dict[str, object]:
    metrics = _passing_metrics()
    if not passing:
        metrics["complete_class_units"] = 0
    fold_metrics = _fold_metrics_from_aggregate(metrics)
    aggregate = v68.v67.aggregate_numeric_screens_v67(fold_metrics)
    checks = v68.v67.assess_numeric_screen_v67(aggregate)
    return {
        "arm_id": arm["arm_id"],
        "arm_sha256": v68._arm_sha256(arm),
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "metrics": aggregate,
        "checks": checks,
        "gemma_generation_used": False,
        "folds": [
            {
                "held_pair_id": pair_id,
                "numeric_screen": metric,
            }
            for pair_id, metric in zip(TRAIN_PAIR_IDS, fold_metrics, strict=True)
        ],
    }


def test_v68_parser_has_training_only_boundary_and_no_protected_inputs() -> None:
    destinations = {action.dest for action in v68._parser()._actions}

    assert {
        "mode",
        "baseline_lock",
        "preregistration",
        "training_baseline_lock",
        "filtered_train_qa",
        "teacher_cache",
        "supplemental_teacher_cache",
        "prefix_cache",
        "base_runtime_config",
        "base_checkpoint",
        "source_v60_checkpoint",
        "work_directory",
        "output_checkpoint",
        "training_report",
        "screen_authorization",
    } <= destinations
    assert {
        "validation_questions",
        "scorer_references",
        "oracle",
        "fresh_development",
        "final_scenes",
        "internal_validation",
    }.isdisjoint(destinations)


def test_v68_locked_fit_args_ignore_unregistered_values() -> None:
    args = SimpleNamespace(user_supplied_learning_rate=999.0)
    locked = v68._locked_fit_args(args, V68_ARM_GRID[0])

    assert locked.seed == 680068
    assert locked.epochs == 160
    assert locked.learning_rate == 0.0003
    assert locked.pair_delta_weight == 0.0
    assert locked.route_weight == 0.0
    assert locked.v68_arm == dict(V68_ARM_GRID[0])
    assert not hasattr(locked, "validation_questions")


@pytest.mark.parametrize("scope", ["all_value", "interaction_only"])
def test_v68_regularized_optimizer_scope_is_explicit(scope: str) -> None:
    control = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        16,
        torch.eye(4, 16),
        control_tokens=2,
        expected_environment_latents=4,
        moment_count=2,
        interaction_dim=4,
        trunk_dim=8,
    )
    named = v68._regularized_parameters(control, optimizer_scope=scope)
    names = {name for name, _parameter in named}

    assert names
    assert all(not name.startswith("route_") for name in names)
    assert all(not name.startswith("question_norm.") for name in names)
    assert not any(parameter.requires_grad for parameter in control.question_norm.parameters())
    if scope == "interaction_only":
        assert not any(name.startswith("coefficient_output.") for name in names)
        assert not any(name.startswith("magnitude_output.") for name in names)
    else:
        assert any(name.startswith("coefficient_output.") for name in names)
        assert any(name.startswith("magnitude_output.") for name in names)


def test_v68_screen_authorization_enforces_first_pass_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v68, "implementation_source_hashes_v68", lambda: {"x": "1"})
    first = _arm_result(dict(V68_ARM_GRID[0]), passing=False)
    second = _arm_result(dict(V68_ARM_GRID[1]), passing=True)
    third = v68._skipped_arm_result(V68_ARM_GRID[2])
    payload = {
        "artifact": v68._SCREEN_ARTIFACT,
        "passed": True,
        "checkpoint_published": False,
        "gemma_generation_used": False,
        "training_identity_sha256": "a" * 64,
        "preregistration_sha256": "b" * 64,
        "implementation_source_hashes": {"x": "1"},
        "thresholds": v68.asdict(v68.V67_NUMERIC_SCREEN_THRESHOLDS),
        "selection": {
            "rule": v68._SELECTION_RULE,
            "selected_arm_id": V68_ARM_GRID[1]["arm_id"],
            "selected_arm_sha256": v68._arm_sha256(V68_ARM_GRID[1]),
        },
        "arm_results": [first, second, third],
        "scope": {
            "question_or_answer_text_stored": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "internal_validation_loaded": False,
            "deferred_final_loaded": False,
        },
    }
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    accepted = v68.validate_screen_authorization_v68(
        path,
        expected_training_identity_sha256="a" * 64,
        expected_preregistration_sha256="b" * 64,
    )
    assert accepted["selected_arm_id"] == V68_ARM_GRID[1]["arm_id"]

    payload["selection"]["selected_arm_id"] = V68_ARM_GRID[2]["arm_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selected arm"):
        v68.validate_screen_authorization_v68(
            path,
            expected_training_identity_sha256="a" * 64,
            expected_preregistration_sha256="b" * 64,
        )


def test_v68_launcher_and_make_targets_are_wired() -> None:
    launcher = Path("scripts/run_gemma4_v68_regularized_pair.sh").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert '--mode "$V68_MODE"' in launcher
    assert '--screen-authorization "$V68_SCREEN_REPORT"' in launcher
    assert "v68_regularized_pair_preregistration.json" in launcher
    assert "gemma4-v68-preregister:" in makefile
    assert "gemma4-v68-screen:" in makefile
    assert "gemma4-v68-full:" in makefile
