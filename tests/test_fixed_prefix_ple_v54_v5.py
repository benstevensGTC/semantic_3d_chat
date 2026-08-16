from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v5_preregistration import (
    V4_RESULT_SHA256,
    authenticate_v4_terminal,
    build_preregistration,
    v5_implementation_hashes,
    write_preregistration,
)
from semantic_3d_chat.training.train_fixed_prefix_ple_v54 import load_training_records
from semantic_3d_chat.training.train_fixed_prefix_ple_v54_v5 import (
    _teacher_checks,
    build_v5_schedule,
    learning_rate,
    symmetric_pair_objective,
)


def test_v5_authenticates_v4_scene_discrimination_failure() -> None:
    result = authenticate_v4_terminal()
    diagnostic = build_preregistration()["v4_diagnostic"]

    assert result["status"] == "failed_no_checkpoint"
    assert result["checkpoint_published"] is False
    assert diagnostic["result_sha256"] == V4_RESULT_SHA256
    assert diagnostic["answer_nll_after"] < diagnostic["answer_nll_before"]
    assert diagnostic["positive_margin_sides_after"] < diagnostic["positive_margin_sides_before"]
    assert diagnostic["complete_pair_units_after"] < diagnostic["complete_pair_units_before"]


def test_v5_pair_objective_is_side_swap_symmetric_and_exact() -> None:
    correct = torch.tensor([0.8, 1.1], requires_grad=True)
    wrong = torch.tensor([0.9, 1.8], requires_grad=True)
    loss, diagnostics = symmetric_pair_objective(correct, wrong)
    expected_margins = wrong - correct
    expected_hinges = torch.relu(0.5 - expected_margins)
    expected = 0.5 * correct.mean() + 4.0 * expected_hinges.mean()
    swapped, _ = symmetric_pair_objective(correct.flip(0), wrong.flip(0))

    assert torch.equal(diagnostics["wrong_prefix_margins"], expected_margins)
    assert torch.equal(loss, expected)
    assert torch.equal(swapped, loss)
    loss.backward()
    assert correct.grad is not None and torch.isfinite(correct.grad).all()
    assert wrong.grad is not None and torch.isfinite(wrong.grad).all()


def test_v5_schedule_covers_every_pair_twice_and_every_broad_row_once() -> None:
    schedule = build_v5_schedule(load_training_records())
    pair_counts = Counter(
        (update.pair[0].pair_id, update.pair[0].pair_question_key) for update in schedule
    )
    broad_keys = [(row.scene_id, row.question_id) for update in schedule for row in update.broad]

    assert len(schedule) == 80
    assert len(pair_counts) == 40
    assert set(pair_counts.values()) == {2}
    assert [len(update.broad) for update in schedule[:64]] == [6] * 64
    assert [len(update.broad) for update in schedule[64:]] == [7] * 16
    assert len(broad_keys) == len(set(broad_keys)) == 496
    assert build_v5_schedule(load_training_records()) == schedule


def test_v5_learning_rate_is_locked_warmup_then_cosine_decay() -> None:
    values = [learning_rate(update) for update in range(1, 81)]

    assert values[0] == pytest.approx(1.25e-5)
    assert values[7] == pytest.approx(1e-4)
    assert values[-1] == pytest.approx(1e-5)
    assert values[:8] == sorted(values[:8])
    assert values[7:] == sorted(values[7:], reverse=True)
    with pytest.raises(ValueError, match=r"\[1, 80\]"):
        learning_rate(0)


def test_v5_keeps_every_v4_gate_and_defers_both_holdouts() -> None:
    contract = build_preregistration()
    gates = contract["unchanged_v4_promotion_gates"]
    arm = contract["single_v5_arm"]

    assert gates == {
        "validation_answer_nll_improvement_minimum": 0.03,
        "changed_wrong_prefix_positive_margin_rate_minimum": 0.65,
        "changed_wrong_prefix_positive_margin_rate_delta_minimum": 0.10,
        "changed_pair_complete_unit_delta_minimum": 3,
        "greedy_exact_accuracy_delta_minimum": 0.02,
        "retention_mean_ce_increase_nats_maximum": 0.03,
        "retention_mean_kl_nats_maximum": 0.02,
        "retention_next_token_top1_agreement_minimum": 0.98,
        "all_required": True,
        "failed_run_publishes_no_checkpoint": True,
    }
    assert arm["one_arm_only"] is True
    assert arm["intermediate_selection"] is False
    assert arm["intermediate_checkpoint"] is False
    assert arm["candidate_state_for_all_gates"] == "single_final_state_after_update_80_only"
    assert arm["best_loss_selection"] is False
    assert arm["post_hoc_state_selection"] is False
    assert contract["unchanged_from_v4"]["base"].startswith("structurally_authenticated_v54_")
    assert contract["deferred_holdout"][
        "qa_not_accessed_for_v5_design_training_or_internal_selection"
    ]
    assert contract["final_split"]["accessed"] is False


def test_v5_teacher_checks_are_not_weaker_than_v4() -> None:
    baseline = {
        "answer_nll_mean": 3.0,
        "changed_positive_margin_rate": 30 / 52,
        "changed_complete_units": 12,
    }
    candidate = {
        "answer_nll_mean": 2.96,
        "changed_positive_margin_rate": 36 / 52,
        "changed_complete_units": 15,
    }
    retention = {
        "mean_ce_increase_nats": 0.03,
        "mean_kl_nats": 0.02,
        "next_token_top1_agreement": 0.98,
    }

    assert all(_teacher_checks(baseline, candidate, retention).values())
    candidate["changed_positive_margin_rate"] = 35 / 52
    assert (
        _teacher_checks(baseline, candidate, retention)[
            "changed_wrong_prefix_positive_margin_rate_delta"
        ]
        is False
    )


def test_v5_preregistration_is_create_once_and_source_locked(tmp_path: Path) -> None:
    destination = tmp_path / "v5.json"
    path, digest = write_preregistration(destination)
    contract = json.loads(destination.read_text(encoding="utf-8"))

    assert path == destination.resolve()
    assert len(digest) == 64
    assert contract == build_preregistration()
    assert contract["v5_implementation_source_hashes"] == v5_implementation_hashes()
    with pytest.raises(FileExistsError, match="exists"):
        write_preregistration(destination)


def test_v5_training_source_uses_streamed_logits_and_has_no_holdout_loader() -> None:
    source = Path("src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v5.py").read_text(
        encoding="utf-8"
    )
    launcher = Path("scripts/run_gemma4_v54_fixed_prefix_ple_reader_v5.sh").read_text(
        encoding="utf-8"
    )

    assert "v4.streamed_answer_nlls" in source
    assert "v4.evaluate_teacher_forcing_streamed" in source
    assert "question_dependent_retrieval" in source
    assert "load_holdout" not in source
    assert "reports/gemma4/questions/test.json" not in source
    assert "preregister|preflight|smoke|train|authenticate" in launcher
