"""Teacher-space objectives for the V74 dense continuous controller."""

from __future__ import annotations

import torch


def teacher_coefficients_v74(
    teacher_prompts: torch.Tensor,
    output_basis: torch.Tensor,
) -> torch.Tensor:
    """Project native-width teacher prompts into an orthonormal output basis."""

    if teacher_prompts.ndim != 3 or output_basis.ndim != 2:
        raise ValueError("V74 teacher prompts/basis must be [B,C,H] and [R,H]")
    if teacher_prompts.shape[-1] != output_basis.shape[-1]:
        raise ValueError("V74 teacher prompt and output-basis widths differ")
    if not teacher_prompts.is_floating_point() or not output_basis.is_floating_point():
        raise TypeError("V74 teacher prompt and output basis must be floating point")
    if not torch.isfinite(teacher_prompts).all() or not torch.isfinite(output_basis).all():
        raise ValueError("V74 teacher prompt or output basis is nonfinite")
    return torch.einsum(
        "bch,rh->bcr", teacher_prompts.float(), output_basis.float()
    )


def normalized_unclipped_teacher_value_loss_v74(
    raw_coefficients: torch.Tensor,
    target_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Match teachers before the controller's hard safety RMS clipping.

    Because the basis rows are orthonormal, the ratio of coefficient-space
    squared errors is exactly the normalized native-prompt MSE for targets in
    that basis.  Unlike a loss applied after hard RMS clipping, this retains a
    radial gradient when an output has crossed the safety cap.
    """

    if (
        raw_coefficients.ndim != 3
        or raw_coefficients.shape != target_coefficients.shape
    ):
        raise ValueError("V74 raw and target coefficients must share [B,C,R]")
    if not raw_coefficients.is_floating_point() or not target_coefficients.is_floating_point():
        raise TypeError("V74 coefficient tensors must be floating point")
    if not torch.isfinite(raw_coefficients).all() or not torch.isfinite(
        target_coefficients
    ).all():
        raise ValueError("V74 coefficient tensors are nonfinite")
    target_energy = target_coefficients.square().sum(dim=(1, 2))
    if bool((target_energy <= 1e-12).any()):
        raise ValueError("V74 teacher coefficient target has zero energy")
    error_energy = (raw_coefficients - target_coefficients).square().sum(
        dim=(1, 2)
    )
    return (error_energy / target_energy).mean()


def normalized_unclipped_teacher_delta_loss_v74(
    left_raw_coefficients: torch.Tensor,
    right_raw_coefficients: torch.Tensor,
    left_target_coefficients: torch.Tensor,
    right_target_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Match counterfactual teacher deltas before hard runtime clipping."""

    tensors = (
        left_raw_coefficients,
        right_raw_coefficients,
        left_target_coefficients,
        right_target_coefficients,
    )
    if any(value.ndim != 3 for value in tensors) or any(
        value.shape != tensors[0].shape for value in tensors[1:]
    ):
        raise ValueError("V74 delta coefficients must share shape [U,C,R]")
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("V74 delta coefficients must be floating point")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("V74 delta coefficients are nonfinite")
    predicted_delta = right_raw_coefficients - left_raw_coefficients
    target_delta = right_target_coefficients - left_target_coefficients
    target_energy = target_delta.square().sum(dim=(1, 2))
    if bool((target_energy <= 1e-12).any()):
        raise ValueError("V74 counterfactual teacher delta has zero energy")
    error_energy = (predicted_delta - target_delta).square().sum(dim=(1, 2))
    return (error_energy / target_energy).mean()


def raw_prompt_rms_from_coefficients_v74(
    coefficients: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    """Return per-control-token native RMS for orthonormal-basis coefficients."""

    if coefficients.ndim != 3:
        raise ValueError("V74 coefficients must have shape [B,C,R]")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("V74 hidden size must be a positive integer")
    return (coefficients.square().sum(dim=-1) / float(hidden_size)).sqrt()


__all__ = [
    "normalized_unclipped_teacher_delta_loss_v74",
    "normalized_unclipped_teacher_value_loss_v74",
    "raw_prompt_rms_from_coefficients_v74",
    "teacher_coefficients_v74",
]
