from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.training.question_control_v69_objective import (
    balanced_transition_indices_v69,
    extrapolate_pair_signatures_v69,
    mix_transition_questions_v69,
)


def test_v69_signature_extrapolation_is_symmetric_and_expands_delta() -> None:
    signatures = torch.tensor([[[[1.0, 2.0]], [[3.0, 6.0]]]])
    augmented = extrapolate_pair_signatures_v69(signatures, expansion=0.25)

    assert torch.equal(augmented.mean(dim=1), signatures.mean(dim=1))
    original_delta = signatures[:, 0] - signatures[:, 1]
    augmented_delta = augmented[:, 0] - augmented[:, 1]
    assert torch.allclose(augmented_delta, 1.5 * original_delta)


def test_v69_transition_balancing_is_reproducible_and_balanced() -> None:
    transitions = (("a", "b"), ("a", "b"), ("x", "y"))
    first = balanced_transition_indices_v69(transitions, seed=69)
    second = balanced_transition_indices_v69(transitions, seed=69)

    assert first == second
    assert len(first) == 4
    assert sum(transitions[index] == ("a", "b") for index in first) == 2
    assert sum(transitions[index] == ("x", "y") for index in first) == 2


def test_v69_question_mix_never_crosses_transition_and_is_auditable() -> None:
    questions = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    transitions = (("a", "b"), ("a", "b"), ("a", "b"), ("x", "y"))
    mixed, partners = mix_transition_questions_v69(
        questions,
        transitions,
        mix_weight=0.2,
        seed=6901,
    )

    assert partners[3] == 3
    assert torch.allclose(mixed[3], questions[3])
    for index, partner in enumerate(partners):
        assert transitions[index] == transitions[partner]
        if index < 3:
            assert partner != index
        expected = 0.8 * questions[index] + 0.2 * questions[partner]
        assert torch.allclose(mixed[index], expected)


@pytest.mark.parametrize(
    "signatures,expansion,match",
    [
        (torch.zeros(2, 3), 0.1, "shape"),
        (torch.zeros(1, 2, 2, 3), 0.6, "lie in"),
        (torch.full((1, 2, 2, 3), torch.nan), 0.1, "finite"),
    ],
)
def test_v69_signature_extrapolation_rejects_invalid_inputs(
    signatures: torch.Tensor,
    expansion: float,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        extrapolate_pair_signatures_v69(signatures, expansion=expansion)


def test_v69_transition_helpers_reject_bad_or_misaligned_inventories() -> None:
    with pytest.raises(ValueError, match="endpoints"):
        balanced_transition_indices_v69((("same", "same"),), seed=1)
    with pytest.raises(ValueError, match="inventories"):
        mix_transition_questions_v69(
            torch.zeros(2, 1, 3),
            (("a", "b"),),
            mix_weight=0.1,
            seed=1,
        )
