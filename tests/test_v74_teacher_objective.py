from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.training.v74_teacher_objective import (
    normalized_unclipped_teacher_delta_loss_v74,
    normalized_unclipped_teacher_value_loss_v74,
    raw_prompt_rms_from_coefficients_v74,
    teacher_coefficients_v74,
)


def test_v74_coefficient_loss_equals_native_loss_in_orthonormal_basis() -> None:
    torch.manual_seed(7401)
    basis = torch.linalg.qr(torch.randn(12, 5)).Q.T.contiguous()
    target_coefficients = torch.randn(3, 4, 5)
    raw_coefficients = torch.randn(3, 4, 5, requires_grad=True)
    target = torch.einsum("bcr,rh->bch", target_coefficients, basis)
    raw = torch.einsum("bcr,rh->bch", raw_coefficients, basis)

    coefficient_loss = normalized_unclipped_teacher_value_loss_v74(
        raw_coefficients, teacher_coefficients_v74(target, basis)
    )
    native_loss = (
        (raw - target).square().sum(dim=(1, 2))
        / target.square().sum(dim=(1, 2))
    ).mean()

    assert torch.allclose(coefficient_loss, native_loss, atol=1e-6, rtol=1e-6)
    coefficient_loss.backward()
    assert raw_coefficients.grad is not None
    assert float(raw_coefficients.grad.abs().sum()) > 0.0


def test_v74_unclipped_objective_retains_radial_gradient_beyond_runtime_cap() -> None:
    target = torch.full((1, 2, 3), 0.1)
    raw = (target * 10.0).requires_grad_()
    loss = normalized_unclipped_teacher_value_loss_v74(raw, target)
    loss.backward()

    assert float(loss.detach()) == pytest.approx(81.0)
    assert raw.grad is not None
    # A positive gradient reduces the over-large raw radius under gradient descent.
    assert bool((raw.grad > 0.0).all())


def test_v74_coefficient_rms_matches_native_prompt_rms() -> None:
    torch.manual_seed(7402)
    basis = torch.linalg.qr(torch.randn(16, 6)).Q.T.contiguous()
    coefficients = torch.randn(2, 4, 6)
    native = torch.einsum("bcr,rh->bch", coefficients, basis)

    observed = raw_prompt_rms_from_coefficients_v74(
        coefficients, hidden_size=16
    )

    assert torch.allclose(
        observed, native.square().mean(dim=-1).sqrt(), atol=1e-6, rtol=1e-6
    )


def test_v74_delta_loss_is_zero_only_for_matching_counterfactual_delta() -> None:
    left_target = torch.randn(2, 4, 5)
    right_target = torch.randn(2, 4, 5)
    shared_offset = torch.randn(2, 4, 5)
    matching = normalized_unclipped_teacher_delta_loss_v74(
        left_target + shared_offset,
        right_target + shared_offset,
        left_target,
        right_target,
    )
    wrong = normalized_unclipped_teacher_delta_loss_v74(
        left_target,
        left_target,
        left_target,
        right_target,
    )

    assert float(matching) == pytest.approx(0.0, abs=1e-12)
    assert float(wrong) == pytest.approx(1.0, rel=1e-6)

@pytest.mark.parametrize("bad", [torch.zeros(1, 4), torch.zeros(1, 4, 2, 1)])
def test_v74_teacher_objective_rejects_wrong_rank(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        normalized_unclipped_teacher_value_loss_v74(bad, bad)
