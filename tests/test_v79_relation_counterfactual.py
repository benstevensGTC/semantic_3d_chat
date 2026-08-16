from __future__ import annotations

from collections import Counter

import pytest
import torch
from safetensors.torch import load_file

from semantic_3d_chat.evaluation.v79_relation_counterfactual import (
    EXPECTED_SCREEN_ROWS,
    EXPECTED_SCREEN_UNITS,
    full_decision_v79,
    screen_decision_v79,
    select_screen_rows_v79,
)
from semantic_3d_chat.training.finetune_v77_historical_repair import (
    canonical_alternatives_v77,
    deterministic_training_schedule_v77,
)
from semantic_3d_chat.training.finetune_v79_relation_counterfactual import (
    LOCKED_SETTINGS_V79,
    V79_CHANGED_SIDES,
    V79_OPTIMIZER_STEPS,
    V79_PREREGISTRATION_SHA256,
    V79_SEED,
    V79_SELECTED_ROWS,
    candidate_metadata_v79,
    guard_input_v79,
    guard_output_v79,
    load_preregistration_v79,
    relation_objective_v79,
    select_historical_relation_rows_v79,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    _sha256_file,
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)


def _rows():
    _path, prereg = load_preregistration_v79()
    config = load_config_v73(prereg["sources"]["v73_split_config"]["path"])
    return split_rows_v73(load_training_rows_v73(config["training_qa"]))


def test_preregistration_is_authenticated_and_sources_are_locked() -> None:
    path, prereg = load_preregistration_v79()
    assert _sha256_file(path) == V79_PREREGISTRATION_SHA256
    assert prereg["optimization"]["optimizer_steps"] == V79_OPTIMIZER_STEPS
    assert prereg["screen"]["expected_rows"] == EXPECTED_SCREEN_ROWS
    assert prereg["conditional_full_evaluation"]["run_only_if_screen_passes"]
    for source in prereg["sources"].values():
        if isinstance(source, dict) and set(source) >= {"path", "sha256"}:
            resolved = guard_input_v79(source["path"], "preregistered source")
            assert _sha256_file(resolved) == source["sha256"]


def test_v75_source_is_tensor_exact_sealed_controller() -> None:
    source = load_file(
        "reports/gemma4/artifacts/v75_gemma_nll_balanced_train_diagnostic.safetensors",
        device="cpu",
    )
    sealed = load_file(
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1/control.safetensors",
        device="cpu",
    )
    assert set(source) == set(sealed)
    assert all(torch.equal(source[name], sealed[name]) for name in source)


def test_relation_selection_is_exhaustive_and_scene_disjoint() -> None:
    train, held = _rows()
    selected = select_historical_relation_rows_v79(train, held)
    assert len(selected) == V79_SELECTED_ROWS
    assert sum(row.expected_change for row in selected) == V79_CHANGED_SIDES
    assert Counter(row.answer_type for row in selected) == {"spatial_relation": V79_SELECTED_ROWS}
    assert {row.scene_id for row in selected}.isdisjoint({row.scene_id for row in held})
    alternatives = canonical_alternatives_v77(train)
    schedule = deterministic_training_schedule_v77(selected, alternatives, cycles=1, seed=V79_SEED)
    assert len(schedule) == V79_SELECTED_ROWS
    assert len({item.row.key for item in schedule}) == V79_SELECTED_ROWS
    assert len(schedule) // 8 + bool(len(schedule) % 8) == V79_OPTIMIZER_STEPS


def test_screen_selection_is_all_held_changed_relation_sides() -> None:
    _train, held = _rows()
    selected = select_screen_rows_v79(held)
    assert len(selected) == EXPECTED_SCREEN_ROWS
    assert all(row.expected_change for row in selected)
    assert {row.answer_type for row in selected} == {"spatial_relation"}
    assert len({(row.pair_id, row.question_key) for row in selected}) == (EXPECTED_SCREEN_UNITS)
    assert Counter(row.change_type for row in selected) == {
        "book_support": 4,
        "mirror_lr": 8,
        "object_relocation": 8,
        "picture_support": 8,
    }


def test_relation_objective_matches_locked_formula_and_has_gradients() -> None:
    correct = torch.tensor(2.0, requires_grad=True)
    negative = torch.tensor(2.2, requires_grad=True)
    paired = torch.tensor(2.1, requires_grad=True)
    wrong_scene = torch.tensor(2.0, requires_grad=True)
    anchor = torch.tensor(0.4, requires_grad=True)
    total, diagnostics = relation_objective_v79(
        correct_answer_nll=correct,
        negative_answer_nll=negative,
        paired_answer_nll=paired,
        wrong_scene_answer_nll=wrong_scene,
        source_output_mse=anchor,
    )
    expected = (
        2.0
        + 0.15 * (0.5 + 2.0 - 2.2)
        + 0.5 * (0.5 + 2.0 - 2.1)
        + 0.75 * (0.25 + 2.0 - 2.0)
        + 0.05 * 0.4
    )
    assert float(total.detach()) == pytest.approx(expected)
    assert float(diagnostics["paired_answer_margin"].detach()) == pytest.approx(0.1)
    assert float(diagnostics["wrong_scene_answer_margin"].detach()) == pytest.approx(0.0)
    total.backward()
    assert correct.grad is not None and float(correct.grad) > 1.0
    assert negative.grad is not None and float(negative.grad) < 0.0
    assert paired.grad is not None and float(paired.grad) < 0.0
    assert wrong_scene.grad is not None and float(wrong_scene.grad) < 0.0
    assert anchor.grad is not None and float(anchor.grad) == pytest.approx(0.05)


def test_stable_relation_objective_has_no_counterfactual_terms() -> None:
    total, diagnostics = relation_objective_v79(
        correct_answer_nll=torch.tensor(1.0),
        negative_answer_nll=torch.tensor(3.0),
        paired_answer_nll=None,
        wrong_scene_answer_nll=None,
        source_output_mse=torch.tensor(0.0),
        settings=LOCKED_SETTINGS_V79,
    )
    assert float(total) == pytest.approx(1.0)
    assert float(diagnostics["paired_answer_margin_hinge"]) == 0.0
    assert float(diagnostics["wrong_scene_margin_hinge"]) == 0.0
    with pytest.raises(ValueError, match="supplied together"):
        relation_objective_v79(
            correct_answer_nll=torch.tensor(1.0),
            negative_answer_nll=torch.tensor(3.0),
            paired_answer_nll=torch.tensor(2.0),
            wrong_scene_answer_nll=None,
            source_output_mse=torch.tensor(0.0),
        )


def test_diagnostic_metadata_contains_no_codebook_or_label_payload() -> None:
    metadata = candidate_metadata_v79()
    assert metadata["answer_codebook_serialized"] == "false"
    assert metadata["category_codebook_serialized"] == "false"
    assert metadata["questions_answers_or_labels_serialized"] == "false"
    assert metadata["runtime_publication_artifact"] == "false"
    assert "spatial_relation" not in metadata.values()


def test_path_guards_reject_forbidden_or_mutating_destinations() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        guard_input_v79("reports/gemma4/metrics/fake_validation.json", "input")
    with pytest.raises(ValueError, match="forbidden"):
        guard_output_v79(
            "data_gemma4/runtime/checkpoints/v79.safetensors",
            suffix=".safetensors",
        )


def test_screen_decision_requires_all_preregistered_conditions() -> None:
    def summary(correct: int, wrong: int, changes: int) -> dict[str, object]:
        return {
            "correct_scene": {"correct": correct},
            "correct_minus_wrong_count": correct - wrong,
            "correct_scene_prediction_changing_units": changes,
        }

    passed = screen_decision_v79(
        {
            "v75": summary(18, 5, 10),
            "v77": summary(19, 6, 11),
            "v79": summary(20, 6, 11),
        }
    )
    assert passed["screen_passed"]
    failed = screen_decision_v79(
        {
            "v75": summary(18, 5, 10),
            "v77": summary(19, 6, 11),
            "v79": summary(19, 4, 12),
        }
    )
    assert not failed["screen_passed"]
    assert not failed["conditions"]["correct_count_strictly_above_v77"]


def test_full_decision_enforces_retention_and_causal_gates() -> None:
    baseline = {
        "attribute": 47,
        "count": 61,
        "metric": 15,
        "orientation": 13,
        "presence": 63,
        "spatial_relation": 50,
        "support": 46,
    }
    candidate = {**baseline, "spatial_relation": 55}
    summary = {
        "correct_scene": {"correct": sum(candidate.values())},
        "correct_minus_wrong_count": 18,
        "correct_scene_prediction_changing_units": 35,
        "by_answer_type": {
            "correct_scene": {key: {"correct": value} for key, value in candidate.items()}
        },
    }
    decision = full_decision_v79(summary, baseline)
    assert decision["advancement_gate_passed"]
    candidate["attribute"] = 44
    summary["by_answer_type"]["correct_scene"]["attribute"]["correct"] = 44
    decision = full_decision_v79(summary, baseline)
    assert not decision["advancement_gate_passed"]
    assert not decision["conditions"]["maximum_answer_type_correct_drop_from_v75_at_most_2"]
