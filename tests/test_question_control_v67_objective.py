from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.training.question_control_v67_objective import (
    paired_scene_dependence_loss_v67,
)


def test_v67_pair_objective_prefers_correct_scene_to_teacher_assignment() -> None:
    targets = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]], [[[0.0, 1.0]], [[-1.0, 0.0]]]]
    )

    correct, diagnostics = paired_scene_dependence_loss_v67(targets, targets)
    swapped, swapped_diagnostics = paired_scene_dependence_loss_v67(
        targets.flip(dims=(1,)), targets
    )

    assert float(correct) < float(swapped)
    assert float(diagnostics.mean_own_over_opposite_margin) > 0.0
    assert float(diagnostics.mean_delta_cosine) == pytest.approx(1.0)
    assert float(swapped_diagnostics.mean_delta_cosine) == pytest.approx(-1.0)


def test_v67_pair_objective_backpropagates_scene_separation() -> None:
    targets = torch.randn(3, 2, 4, 16)
    predicted = torch.zeros_like(targets, requires_grad=True)

    loss, diagnostics = paired_scene_dependence_loss_v67(predicted, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum()) > 0.0
    assert float(diagnostics.positive_delta_fraction) == 0.0


@pytest.mark.parametrize(
    "predicted,targets,message",
    [
        (torch.zeros(2, 3), torch.zeros(2, 3), "matching"),
        (torch.zeros(1, 3, 2, 4), torch.zeros(1, 3, 2, 4), "two-sided"),
        (
            torch.full((1, 2, 1, 2), float("nan")),
            torch.zeros(1, 2, 1, 2),
            "nonfinite",
        ),
    ],
)
def test_v67_pair_objective_rejects_invalid_inputs(
    predicted: torch.Tensor, targets: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_scene_dependence_loss_v67(predicted, targets)
