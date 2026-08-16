from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training import train_question_control_v67 as v67


def _passing_fold() -> dict[str, object]:
    return {
        "supported_class_exact": 26,
        "supported_total": 48,
        "unsupported_total": 0,
        "inventory_total": 48,
        "changed_class_exact": 4,
        "changed_total": 6,
        "complete_class_units": 2,
        "complete_unit_total": 3,
        "prediction_change_units": 2,
        "pair_delta_cosine_sum": 1.5,
        "positive_pair_delta_units": 3,
        "own_over_opposite_margin_sum": 0.3,
        "positive_own_over_opposite_sides": 5,
        "fully_supported_pair_sides": 6,
        "question_or_answer_text_stored": False,
        "gemma_generation_used": False,
    }


def test_v67_parser_has_training_only_boundary_and_no_protected_inputs() -> None:
    destinations = {action.dest for action in v67._parser()._actions}

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


def test_v67_uses_only_preregistered_hyperparameters() -> None:
    args = SimpleNamespace(user_supplied_learning_rate=999.0)
    locked = v67._locked_fit_args(args)

    assert locked.seed == 670067
    assert locked.epochs == 160
    assert locked.learning_rate == 0.0003
    assert locked.pair_delta_weight == 0.0  # dedicated V67 objective owns this term
    assert locked.route_weight == 0.0
    assert not hasattr(locked, "validation_questions")


def test_v67_pair_refinement_restores_frozen_question_norm_scope() -> None:
    control = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        16,
        torch.eye(4, 16),
        control_tokens=2,
        expected_environment_latents=4,
        moment_count=2,
        interaction_dim=4,
        trunk_dim=8,
    )
    assert any(parameter.requires_grad for parameter in control.question_norm.parameters())

    trainable = v67._pair_refinement_parameters(control)
    trainable_ids = {id(parameter) for parameter in trainable}

    assert trainable
    assert all(
        not parameter.requires_grad for parameter in control.question_norm.parameters()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in control.named_parameters()
        if name.startswith("route_")
    )
    assert all(id(parameter) not in trainable_ids for parameter in control.question_norm.parameters())


def test_v67_preregistration_validator_matches_pre_generation_lock() -> None:
    payload = v67.validate_v67_preregistration(
        "reports/gemma4/metrics/v67_pair_objective_preregistration.json"
    )

    assert payload["failed_predecessor"]["held_changed_side_exact"] == 37
    assert payload["numeric_screen"][
        "required_before_any_greedy_generation"
    ] is True


def test_v67_numeric_screen_gate_recomputes_all_locked_dimensions() -> None:
    folds = [_passing_fold() for _ in range(12)]
    # Match the exact preregistered inventory while retaining comfortably
    # passing evidence counts.
    for index in range(5):
        folds[index]["supported_total"] = 47
        folds[index]["unsupported_total"] = 1
    folds[0]["changed_total"] = 7
    folds[0]["complete_unit_total"] = 2
    folds[0]["fully_supported_pair_sides"] = 4
    folds[1]["changed_total"] = 2
    folds[1]["complete_unit_total"] = 1
    folds[1]["fully_supported_pair_sides"] = 2
    # Current totals: changed=69; add six, complete=33; add two.
    folds[2]["changed_total"] = 12
    folds[2]["complete_unit_total"] = 5
    folds[2]["fully_supported_pair_sides"] = 10

    metrics = v67.aggregate_numeric_screens_v67(folds)
    # Set the exact inventory fields independently of synthetic hit counts.
    metrics["changed_total"] = 75
    metrics["complete_unit_total"] = 35
    metrics["fully_supported_pair_sides"] = 70
    metrics["mean_pair_delta_cosine"] = 0.5
    metrics["mean_own_over_opposite_margin"] = 0.05
    metrics["changed_class_exact"] = 48
    metrics["complete_class_units"] = 24
    metrics["prediction_change_units"] = 24
    metrics["positive_pair_delta_units"] = 30
    metrics["positive_own_over_opposite_sides"] = 56

    checks = v67.assess_numeric_screen_v67(metrics)

    assert all(checks.values())
    metrics["prediction_change_units"] = 19
    assert v67.assess_numeric_screen_v67(metrics)[
        "held_prediction_change_units"
    ] is False


def test_v67_full_authorization_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "screen.json"
    identity = "a" * 64
    threshold = v67.V67_NUMERIC_SCREEN_THRESHOLDS
    metrics = {
        "supported_class_exact": threshold.held_supported_class_exact_minimum,
        "supported_total": threshold.held_supported_total,
        "unsupported_total": threshold.held_unsupported_total,
        "inventory_total": 576,
        "changed_class_exact": threshold.held_changed_class_exact_minimum,
        "changed_total": threshold.held_changed_total,
        "complete_class_units": threshold.held_complete_class_units_minimum,
        "complete_unit_total": threshold.held_complete_unit_total,
        "prediction_change_units": threshold.held_prediction_change_units_minimum,
        "mean_pair_delta_cosine": threshold.mean_pair_delta_cosine_minimum,
        "positive_pair_delta_units": threshold.positive_pair_delta_units_minimum,
        "mean_own_over_opposite_margin": (
            threshold.mean_own_over_opposite_margin_minimum
        ),
        "positive_own_over_opposite_sides": (
            threshold.positive_own_over_opposite_sides_minimum
        ),
        "fully_supported_pair_sides": threshold.fully_supported_pair_sides,
        "answer_or_question_text_stored": False,
        "gemma_generation_used": False,
    }
    checks = v67.assess_numeric_screen_v67(metrics)
    payload = {
        "artifact": v67._SCREEN_ARTIFACT,
        "passed": True,
        "training_identity_sha256": identity,
        "preregistration_sha256": v67._PINNED_PREREGISTRATION_SHA256,
        "thresholds": v67.asdict(threshold),
        "metrics": metrics,
        "checks": checks,
        "checkpoint_published": False,
        "gemma_generation_used": False,
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
    path.write_text(json.dumps(payload), encoding="utf-8")

    accepted = v67.validate_screen_authorization_v67(
        path, expected_training_identity_sha256=identity
    )
    assert accepted["passed"] is True

    payload["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        v67.validate_screen_authorization_v67(
            path, expected_training_identity_sha256=identity
        )


def test_v67_launcher_and_make_targets_are_wired() -> None:
    launcher = Path("scripts/run_gemma4_v67_pair_objective.sh").read_text(
        encoding="utf-8"
    )
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "--mode \"$V67_MODE\"" in launcher
    assert "--screen-authorization \"$V67_SCREEN_REPORT\"" in launcher
    assert "v67_pair_objective_preregistration.json" in launcher
    assert "gemma4-v67-screen:" in makefile
    assert "gemma4-v67-full:" in makefile
