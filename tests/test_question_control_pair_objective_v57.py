from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl
from semantic_3d_chat.training.question_control_pair_objective_v57 import (
    V57PairObjectiveSettings,
    _aligned_candidate_nll,
    answer_absolute_alignment_loss,
    answer_alignment_hinge,
    answer_delta_alignment_loss,
    attention_entropy_hinge,
    attention_logit_spread_penalty,
    control_delta_hinge,
    full_scene_control_with_attention,
    normalized_attention_entropy,
    pair_and_cross_prefix_hinges,
    relative_control_delta,
)


def test_v57_training_forward_is_exactly_runtime_v1_forward() -> None:
    torch.manual_seed(57)
    control = FullSceneQuestionControl(
        12,
        attention_dim=6,
        control_tokens=3,
        uniform_floor=0.1,
        output_scale=0.25,
    )
    scene = torch.randn(2, 9, 12)
    question = torch.randn(2, 5, 12)
    expected = control(scene, question)
    expected_attention = control._last_attention
    actual, attention = full_scene_control_with_attention(control, scene, question)

    assert torch.equal(actual, expected)
    assert expected_attention is not None
    assert torch.equal(attention, expected_attention)
    assert attention.requires_grad


def test_v57_attention_entropy_penalizes_collapse_but_not_uniform() -> None:
    uniform = torch.full((2, 3, 8), 1.0 / 8.0)
    entropy = normalized_attention_entropy(uniform)
    loss, returned = attention_entropy_hinge(
        uniform, minimum_normalized_entropy=0.55
    )
    assert torch.allclose(entropy, torch.ones_like(entropy))
    assert torch.equal(returned, entropy)
    assert loss.item() == pytest.approx(0.0)

    collapsed = torch.full((2, 3, 8), 0.01 / 7.0)
    collapsed[:, :, 0] = 0.99
    collapsed_loss, collapsed_entropy = attention_entropy_hinge(
        collapsed, minimum_normalized_entropy=0.55
    )
    assert collapsed_loss.item() > 0.4
    assert collapsed_entropy.max().item() < 0.1


def test_v57_pre_softmax_spread_penalty_has_gradient_after_saturation() -> None:
    logits = torch.zeros(2, 3, 8, requires_grad=True)
    with torch.no_grad():
        logits[:, :, 0] = 20.0
    loss, rms = attention_logit_spread_penalty(logits, maximum_rms=1.0)
    assert loss.item() > 1.0
    assert rms.min().item() > 1.0
    loss.backward()
    assert logits.grad is not None
    assert torch.linalg.vector_norm(logits.grad).item() > 0.0


def test_v57_relative_control_delta_hinge_detects_scene_invariance() -> None:
    same = torch.ones(2, 2, 4)
    same_loss, same_delta = control_delta_hinge(
        same, minimum_relative_delta=0.03
    )
    assert same_delta.item() == pytest.approx(0.0)
    assert same_loss.item() == pytest.approx(0.03)

    changed = same.clone()
    changed[1] *= -1.0
    changed_loss, changed_delta = control_delta_hinge(
        changed, minimum_relative_delta=0.03
    )
    assert relative_control_delta(changed).item() == pytest.approx(2.0)
    assert changed_delta.item() == pytest.approx(2.0)
    assert changed_loss.item() == pytest.approx(0.0)


def test_v57_answer_alignment_is_paired_and_continuous() -> None:
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    aligned_controls = targets[:, None, :].repeat(1, 3, 1).requires_grad_(True)
    loss, margins, similarities = answer_alignment_hinge(
        aligned_controls, targets, margin=0.1
    )
    assert loss.item() == pytest.approx(0.0)
    assert torch.allclose(margins, torch.ones(2))
    assert torch.equal(similarities, torch.eye(2))

    collapsed = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], requires_grad=True)
    collapsed_loss, collapsed_margins, _ = answer_alignment_hinge(
        collapsed, targets, margin=0.1
    )
    assert collapsed_loss.item() == pytest.approx(0.55)
    assert collapsed_margins.tolist() == pytest.approx([1.0, -1.0])
    collapsed_loss.backward()
    assert collapsed.grad is not None
    assert torch.isfinite(collapsed.grad).all()


def test_v57_absolute_answer_alignment_rewards_own_dense_target() -> None:
    perfect = torch.eye(2)
    loss, own = answer_absolute_alignment_loss(perfect)
    assert loss.item() == pytest.approx(0.0)
    assert own.tolist() == pytest.approx([1.0, 1.0])

    ambiguous = torch.zeros(2, 2, requires_grad=True)
    ambiguous_loss, ambiguous_own = answer_absolute_alignment_loss(ambiguous)
    assert ambiguous_loss.item() == pytest.approx(1.0)
    assert ambiguous_own.tolist() == pytest.approx([0.0, 0.0])
    ambiguous_loss.backward()
    assert ambiguous.grad is not None


def test_v57_answer_delta_alignment_targets_scene_difference_direction() -> None:
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    aligned = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], requires_grad=True)
    loss, cosine = answer_delta_alignment_loss(aligned, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert cosine.item() == pytest.approx(1.0, abs=1e-6)

    reversed_controls = torch.tensor(
        [[[0.0, 1.0]], [[1.0, 0.0]]], requires_grad=True
    )
    reversed_loss, reversed_cosine = answer_delta_alignment_loss(
        reversed_controls, targets
    )
    assert reversed_loss.item() == pytest.approx(2.0, abs=1e-6)
    assert reversed_cosine.item() == pytest.approx(-1.0, abs=1e-6)


def test_v57_side_and_true_cross_prefix_margins_use_two_by_two_scores() -> None:
    correct = torch.tensor([0.2, 0.3])
    swapped = torch.tensor([1.0, 1.1])
    side_hinge, side_margins, cross_hinge, cross_margins = (
        pair_and_cross_prefix_hinges(
            correct_rank_nll=correct,
            swapped_rank_nll=swapped,
            side_margin=0.5,
            cross_prefix_margin=0.1,
        )
    )
    assert side_hinge.item() == pytest.approx(0.0)
    assert side_margins.tolist() == pytest.approx([0.8, 0.8])
    assert cross_hinge.item() == pytest.approx(0.0)
    assert cross_margins.tolist() == pytest.approx([0.9, 0.7])


def test_v57_aligned_candidate_scores_need_only_one_lm_forward() -> None:
    # Labels supervise two answer tokens per row.  The first answer token is the
    # only difference; the common second token stands in for EOS.
    labels = torch.tensor(
        [
            [-100, -100, 3, 9],
            [-100, -100, 7, 9],
        ]
    )
    logits = torch.zeros(2, 4, 10)
    logits[0, 1, 3] = 4.0
    logits[0, 1, 7] = 1.0
    logits[1, 1, 3] = -1.0
    logits[1, 1, 7] = 3.0
    correct, swapped = _aligned_candidate_nll(
        logits,
        labels,
        torch.tensor([[3, 9]]),
        torch.tensor([[7, 9]]),
    )
    assert torch.all(swapped > correct)
    assert (swapped - correct).tolist() == pytest.approx([3.0, 4.0])

    with pytest.raises(ValueError, match="exactly one differing"):
        _aligned_candidate_nll(
            logits,
            labels,
            torch.tensor([[3, 8]]),
            torch.tensor([[7, 9]]),
        )


def test_v57_settings_reject_invalid_entropy_and_empty_objective() -> None:
    with pytest.raises(ValueError, match="entropy"):
        V57PairObjectiveSettings(minimum_normalized_attention_entropy=1.01)
    with pytest.raises(ValueError, match="no answer-sensitive"):
        V57PairObjectiveSettings(
            answer_nll_weight=0.0,
            side_hinge_weight=0.0,
            cross_prefix_hinge_weight=0.0,
            control_delta_weight=0.0,
            answer_alignment_weight=0.0,
            answer_absolute_alignment_weight=0.0,
            answer_delta_alignment_weight=0.0,
        )
