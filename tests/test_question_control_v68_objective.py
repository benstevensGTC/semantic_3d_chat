from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.training.question_control_v68_objective import (
    hard_negative_prototype_margin_loss_v68,
    relative_parameter_anchor_loss_v68,
)


def test_v68_hard_negative_prefers_own_over_hardest_wrong_prototype() -> None:
    bank = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ]
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    correct_loss, correct = hard_negative_prototype_margin_loss_v68(
        bank[:2], bank, labels, margin=0.2
    )
    wrong_loss, wrong = hard_negative_prototype_margin_loss_v68(
        bank[:2].flip(dims=(0,)), bank, labels, margin=0.2
    )

    assert float(correct_loss) == pytest.approx(0.0)
    assert float(wrong_loss) > float(correct_loss)
    assert float(correct.positive_margin_fraction) == 1.0
    assert float(wrong.positive_margin_fraction) == 0.0


def test_v68_hard_negative_backpropagates_when_margin_is_violated() -> None:
    bank = torch.randn(4, 2, 8)
    predicted = torch.zeros(3, 2, 8, requires_grad=True)
    loss, diagnostics = hard_negative_prototype_margin_loss_v68(
        predicted,
        bank,
        torch.tensor([0, 1, 2]),
        margin=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum()) > 0.0
    assert torch.isfinite(diagnostics.mean_own_over_hardest_wrong_margin)


def test_v68_relative_parameter_anchor_is_zero_then_increases() -> None:
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    bias = torch.nn.Parameter(torch.tensor([0.5]))
    named = (("weight", weight), ("bias", bias))
    anchors = {name: parameter.detach().clone() for name, parameter in named}

    zero = relative_parameter_anchor_loss_v68(named, anchors)
    with torch.no_grad():
        weight.add_(0.25)
    changed = relative_parameter_anchor_loss_v68(named, anchors)
    changed.backward()

    assert float(zero.detach()) == pytest.approx(0.0)
    assert float(changed.detach()) > 0.0
    assert weight.grad is not None
    assert float(weight.grad.abs().sum()) > 0.0


@pytest.mark.parametrize(
    "predicted,bank,labels,match",
    [
        (torch.zeros(2, 3), torch.zeros(2, 1, 3), torch.tensor([0, 1]), "must be"),
        (
            torch.zeros(2, 1, 3),
            torch.zeros(2, 2, 3),
            torch.tensor([0, 1]),
            "shapes differ",
        ),
        (
            torch.zeros(2, 1, 3),
            torch.zeros(2, 1, 3),
            torch.tensor([0, 2]),
            "outside",
        ),
    ],
)
def test_v68_hard_negative_rejects_invalid_inputs(
    predicted: torch.Tensor,
    bank: torch.Tensor,
    labels: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        hard_negative_prototype_margin_loss_v68(predicted, bank, labels, margin=0.1)


def test_v68_anchor_rejects_changed_inventory() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="inventory"):
        relative_parameter_anchor_loss_v68((("weight", parameter),), {"other": torch.ones(2)})
