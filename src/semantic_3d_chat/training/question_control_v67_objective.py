"""Paired-scene objectives for the V67 continuous controller.

The objective sees only native-width continuous teacher tensors, continuous
scene signatures, and frozen numeric question embeddings.  Answer strings are
used upstream to construct a training-only fold-local codebook, never by this
module or by the runtime checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairSceneObjectiveDiagnosticsV67:
    mean_own_cosine: torch.Tensor
    mean_opposite_cosine: torch.Tensor
    mean_own_over_opposite_margin: torch.Tensor
    positive_own_over_opposite_fraction: torch.Tensor
    mean_delta_cosine: torch.Tensor
    positive_delta_fraction: torch.Tensor


def paired_scene_dependence_loss_v67(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    *,
    opposite_margin: float = 0.15,
    value_weight: float = 1.0,
    delta_weight: float = 4.0,
    opposite_weight: float = 4.0,
) -> tuple[torch.Tensor, PairSceneObjectiveDiagnosticsV67]:
    """Force an identical question to resolve from its paired scene tensor.

    ``predicted`` and ``targets`` are ``[U,2,C,H]``.  Each unit contains the
    two physical counterfactual scenes with byte-identical questions and
    different canonical answers.  Three complementary terms are used:

    * native-width value preservation for each own-side numeric teacher;
    * alignment of the predicted scene-to-scene delta with the teacher delta;
    * a margin requiring each prediction to be closer to its own teacher than
      to the exact paired-opposite teacher.

    No class name, answer text, token ID, or runtime lookup table is involved.
    """

    if predicted.ndim != 4 or targets.ndim != 4 or predicted.shape != targets.shape:
        raise ValueError("V67 pair objective expects matching [U,2,C,H] tensors")
    if predicted.shape[0] < 1 or predicted.shape[1] != 2:
        raise ValueError("V67 pair objective requires at least one two-sided unit")
    if not predicted.is_floating_point() or not targets.is_floating_point():
        raise TypeError("V67 pair objective requires floating tensors")
    if not torch.isfinite(predicted).all() or not torch.isfinite(targets).all():
        raise ValueError("V67 pair objective received nonfinite tensors")
    for field, value in (
        ("opposite_margin", opposite_margin),
        ("value_weight", value_weight),
        ("delta_weight", delta_weight),
        ("opposite_weight", opposite_weight),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"V67 {field} must be finite and nonnegative")
    if not 0.0 <= float(opposite_margin) <= 2.0:
        raise ValueError("V67 opposite margin must lie in [0,2]")
    if float(value_weight) + float(delta_weight) + float(opposite_weight) <= 0.0:
        raise ValueError("V67 pair objective enables no loss term")

    predicted_flat = F.normalize(predicted.float().flatten(2), dim=-1, eps=1e-8)
    target_flat = F.normalize(targets.float().flatten(2), dim=-1, eps=1e-8)
    own_cosine = (predicted_flat * target_flat).sum(dim=-1)
    opposite_targets = target_flat.flip(dims=(1,))
    opposite_cosine = (predicted_flat * opposite_targets).sum(dim=-1)
    own_margin = own_cosine - opposite_cosine
    margin_loss = F.relu(float(opposite_margin) - own_margin).mean()

    target_power = targets.float().square().mean(dim=(2, 3)).clamp_min(1e-8)
    relative_value_mse = (
        (predicted.float() - targets.float()).square().mean(dim=(2, 3))
        / target_power
    ).mean()
    value_loss = 1.0 - own_cosine.mean() + 0.10 * relative_value_mse

    predicted_delta = predicted[:, 0].float() - predicted[:, 1].float()
    target_delta = targets[:, 0].float() - targets[:, 1].float()
    delta_cosine = F.cosine_similarity(
        predicted_delta.flatten(1), target_delta.flatten(1), dim=-1, eps=1e-8
    )
    target_delta_power = target_delta.square().mean(dim=(1, 2)).clamp_min(1e-8)
    relative_delta_mse = (
        (predicted_delta - target_delta).square().mean(dim=(1, 2))
        / target_delta_power
    )
    delta_loss = (1.0 - delta_cosine + 0.10 * relative_delta_mse).mean()

    total = (
        float(value_weight) * value_loss
        + float(delta_weight) * delta_loss
        + float(opposite_weight) * margin_loss
    )
    diagnostics = PairSceneObjectiveDiagnosticsV67(
        mean_own_cosine=own_cosine.detach().mean(),
        mean_opposite_cosine=opposite_cosine.detach().mean(),
        mean_own_over_opposite_margin=own_margin.detach().mean(),
        positive_own_over_opposite_fraction=(own_margin.detach() > 0.0)
        .float()
        .mean(),
        mean_delta_cosine=delta_cosine.detach().mean(),
        positive_delta_fraction=(delta_cosine.detach() > 0.0).float().mean(),
    )
    if not torch.isfinite(total) or not all(
        torch.isfinite(value)
        for value in (
            diagnostics.mean_own_cosine,
            diagnostics.mean_opposite_cosine,
            diagnostics.mean_own_over_opposite_margin,
            diagnostics.positive_own_over_opposite_fraction,
            diagnostics.mean_delta_cosine,
            diagnostics.positive_delta_fraction,
        )
    ):
        raise RuntimeError("V67 pair objective produced a nonfinite result")
    return total, diagnostics


__all__ = [
    "PairSceneObjectiveDiagnosticsV67",
    "paired_scene_dependence_loss_v67",
]
