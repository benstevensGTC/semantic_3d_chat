from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.training.question_control_v66_objective import (
    numeric_prototype_classification_loss,
)


def test_numeric_prototype_loss_rewards_the_matching_continuous_prompt() -> None:
    prototypes = torch.zeros(3, 2, 6)
    prototypes[0, :, 0] = 1.0
    prototypes[1, :, 1] = 1.0
    prototypes[2, :, 2] = 1.0
    indices = torch.tensor([0, 2], dtype=torch.long)

    matching = prototypes[indices].clone().requires_grad_(True)
    wrong = prototypes[torch.tensor([1, 1])]
    matching_loss, matching_audit = numeric_prototype_classification_loss(
        matching, prototypes, indices, temperature=0.05
    )
    wrong_loss, wrong_audit = numeric_prototype_classification_loss(
        wrong, prototypes, indices, temperature=0.05
    )

    assert matching_loss < 1e-6
    assert matching_audit.top1_accuracy.item() == 1.0
    assert matching_audit.mean_margin.item() == pytest.approx(1.0)
    assert wrong_loss > 10.0
    assert wrong_audit.top1_accuracy.item() == 0.0
    matching_loss.backward()
    assert matching.grad is not None and torch.isfinite(matching.grad).all()


def test_numeric_prototype_loss_rejects_invalid_inputs() -> None:
    predicted = torch.ones(2, 4, 8)
    prototypes = torch.ones(3, 4, 8)
    indices = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="shapes"):
        numeric_prototype_classification_loss(predicted, prototypes[:, :3], indices)
    with pytest.raises(ValueError, match="int64"):
        numeric_prototype_classification_loss(predicted, prototypes, indices.float())
    with pytest.raises(ValueError, match="outside"):
        numeric_prototype_classification_loss(
            predicted, prototypes, torch.tensor([0, 3], dtype=torch.long)
        )
    with pytest.raises(ValueError, match="temperature"):
        numeric_prototype_classification_loss(
            predicted, prototypes, indices, temperature=0.0
        )
