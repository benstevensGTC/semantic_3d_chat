from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    TRAIN_PAIR_IDS,
)
from semantic_3d_chat.evaluation.v69_pair_augmentation_preregistration import (
    V69_ARM_GRID,
    V69_FOUNDATION_ARM,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training import train_question_control_v69 as v69
from semantic_3d_chat.training.question_control_v66_prototypes import (
    HybridAnswerPrototypeCodebookV66,
    answer_class_id_v66,
)
from semantic_3d_chat.training.train_question_control_v63 import FitResult, V63Row


def _passing_metrics() -> dict[str, object]:
    threshold = v69.V67_NUMERIC_SCREEN_THRESHOLDS
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
        "pair_delta_cosine_sum": (
            threshold.mean_pair_delta_cosine_minimum * threshold.held_complete_unit_total
        ),
        "positive_pair_delta_units": threshold.positive_pair_delta_units_minimum,
        "own_over_opposite_margin_sum": (
            threshold.mean_own_over_opposite_margin_minimum
            * threshold.fully_supported_pair_sides
        ),
        "positive_own_over_opposite_sides": (
            threshold.positive_own_over_opposite_sides_minimum
        ),
        "fully_supported_pair_sides": threshold.fully_supported_pair_sides,
        "mean_pair_delta_cosine": threshold.mean_pair_delta_cosine_minimum,
        "mean_own_over_opposite_margin": threshold.mean_own_over_opposite_margin_minimum,
        "answer_or_question_text_stored": False,
        "gemma_generation_used": False,
    }


def _fold_metrics(aggregate: dict[str, object]) -> list[dict[str, object]]:
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
    fold_metrics = _fold_metrics(metrics)
    aggregate = v69.v67.aggregate_numeric_screens_v67(fold_metrics)
    checks = v69.v67.assess_numeric_screen_v67(aggregate)
    return {
        "arm_id": arm["arm_id"],
        "arm_sha256": v69._arm_sha256(arm),
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "metrics": aggregate,
        "checks": checks,
        "gemma_generation_used": False,
        "folds": [
            {"held_pair_id": pair_id, "numeric_screen": metric}
            for pair_id, metric in zip(TRAIN_PAIR_IDS, fold_metrics, strict=True)
        ],
    }


def test_v69_parser_exposes_only_training_inputs() -> None:
    destinations = {action.dest for action in v69._parser()._actions}

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


def test_v69_locked_fit_args_keep_exact_v68_strong_foundation() -> None:
    locked = v69._locked_fit_args(SimpleNamespace(user_value=123))

    assert locked.seed == 680068
    assert locked.v68_arm == V69_FOUNDATION_ARM
    assert locked.learning_rate == 0.0003
    assert locked.epochs == 160
    assert locked.v69_seed == 680068


def test_v69_transition_inventory_is_ordered_and_rejects_non_changes() -> None:
    left = V63Row("s1", "q1", "same?", "p", "k", True, answer="yes")
    right = V63Row("s2", "q2", "same?", "p", "k", True, answer="no")
    transitions = v69._transition_inventory(((left, right),))

    assert len(transitions) == 1
    assert transitions[0][0] != transitions[0][1]
    with pytest.raises(ValueError, match="transition"):
        v69._transition_inventory(((left, V63Row(**{**right.__dict__, "answer": "yes"})),))


def test_v69_one_step_augmentation_runs_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden = 1536
    basis = torch.eye(2, hidden)
    control = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        hidden,
        basis,
        control_tokens=4,
        expected_environment_latents=2,
        moment_count=2,
        interaction_dim=4,
        trunk_dim=8,
        maximum_control_rms=0.25,
        initial_control_rms=0.075,
    )
    scene_ids = ("s1", "s2", "s3", "s4")
    signatures = {
        scene_id: torch.randn(1, 2, hidden) for scene_id in scene_ids
    }
    prefix = {scene_id: torch.randn(1, 4, hidden) for scene_id in scene_ids}
    rows = (
        V63Row("s1", "q1", "first?", "p1", "k1", True, answer="yes"),
        V63Row("s2", "q2", "first?", "p1", "k1", True, answer="no"),
        V63Row("s3", "q3", "second?", "p2", "k2", True, answer="yes"),
        V63Row("s4", "q4", "second?", "p2", "k2", True, answer="no"),
        V63Row("s1", "q5", "stable?", "p1", "k3", False, answer="yes"),
    )
    yes_id = answer_class_id_v66("yes")
    no_id = answer_class_id_v66("no")
    yes = torch.zeros(1, 4, hidden)
    no = torch.zeros(1, 4, hidden)
    yes[..., 0] = 0.1
    no[..., 1] = 0.1
    targets = {
        row.key: (yes.clone() if row.answer == "yes" else no.clone()) for row in rows
    }
    codebook = HybridAnswerPrototypeCodebookV66(
        prototypes={yes_id: yes, no_id: no},
        targets=targets,
        class_by_key={
            row.key: (yes_id if row.answer == "yes" else no_id) for row in rows
        },
        manifest={},
        sha256="a" * 64,
    )
    base_fit = FitResult(
        control=control,
        signatures=signatures,
        basis_reconstruction={},
        elapsed_seconds=0.0,
        optimizer_steps=0,
        maximum_preclip_gradient_norm=0.0,
        final_route_loss=0.0,
        question_norm_sha256="b" * 64,
        question_norm_frozen=True,
    )
    base = v69.v66.V66FitResult(
        control=control,
        signatures=signatures,
        base_fit=base_fit,
        classification_optimizer_steps=0,
        numeric_prototype_top1_accuracy=1.0,
        numeric_prototype_mean_margin=1.0,
    )
    foundation = v69.v68.V68FitResult(
        base=base,
        refinement_optimizer_steps=0,
        refinement_elapsed_seconds=0.0,
        train_pair_diagnostics={},
        train_prototype_top1_accuracy=1.0,
        train_hard_negative_diagnostics={},
        optimizer_scope=(),
        anchor_state_sha256="c" * 64,
    )
    monkeypatch.setattr(v69.v68, "_fit_regularized_pair", lambda **_kwargs: foundation)
    questions = {row.key: torch.randn(1, 2, hidden) for row in rows}
    questions[rows[1].key] = questions[rows[0].key].clone()
    questions[rows[3].key] = questions[rows[2].key].clone()
    arm = {
        "arm_id": "test",
        "augmentation_epochs": 1,
        "augmentation_repeats": 1,
        "augmentation_learning_rate": 1e-5,
        "signature_expansion": 0.1,
        "question_mix_weight": 0.1,
    }

    result = v69._fit_augmented_pair(
        rows=rows,
        codebook=codebook,
        preflight=SimpleNamespace(prefixes=prefix),
        questions=questions,
        basis=basis,
        args=SimpleNamespace(),
        arm=arm,
        seed=680068,
        phase="cpu_unit",
    )

    assert result.augmentation_optimizer_steps >= 2
    assert result.balanced_unit_count == 2
    assert result.transition_bucket_sizes == (2,)
    assert result.question_partner_inventory_sha256
    assert all(not name.startswith("question_norm.") for name in result.optimizer_scope)


def test_v69_screen_authorization_enforces_first_pass_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v69, "implementation_source_hashes_v69", lambda: {"x": "1"})
    first = _arm_result(dict(V69_ARM_GRID[0]), passing=False)
    second = _arm_result(dict(V69_ARM_GRID[1]), passing=True)
    third = v69._skipped_arm_result(V69_ARM_GRID[2])
    payload = {
        "artifact": v69._SCREEN_ARTIFACT,
        "passed": True,
        "checkpoint_published": False,
        "gemma_generation_used": False,
        "training_identity_sha256": "a" * 64,
        "preregistration_sha256": "b" * 64,
        "implementation_source_hashes": {"x": "1"},
        "thresholds": v69.asdict(v69.V67_NUMERIC_SCREEN_THRESHOLDS),
        "selection": {
            "rule": v69._SELECTION_RULE,
            "selected_arm_id": V69_ARM_GRID[1]["arm_id"],
            "selected_arm_sha256": v69._arm_sha256(V69_ARM_GRID[1]),
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

    accepted = v69.validate_screen_authorization_v69(
        path,
        expected_training_identity_sha256="a" * 64,
        expected_preregistration_sha256="b" * 64,
    )
    assert accepted["selected_arm_id"] == V69_ARM_GRID[1]["arm_id"]

    payload["selection"]["selected_arm_id"] = V69_ARM_GRID[2]["arm_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selected arm"):
        v69.validate_screen_authorization_v69(
            path,
            expected_training_identity_sha256="a" * 64,
            expected_preregistration_sha256="b" * 64,
        )


def test_v69_launcher_and_make_targets_are_wired() -> None:
    launcher = Path("scripts/run_gemma4_v69_pair_augmentation.sh").read_text(
        encoding="utf-8"
    )
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert '--mode "$V69_MODE"' in launcher
    assert '--screen-authorization "$V69_SCREEN_REPORT"' in launcher
    assert "v69_pair_augmentation_preregistration.json" in launcher
    assert "gemma4-v69-preregister:" in makefile
    assert "gemma4-v69-screen:" in makefile
    assert "gemma4-v69-full:" in makefile
