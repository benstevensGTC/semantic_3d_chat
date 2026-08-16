"""Regularizers for the preregistered V68 paired-scene screen.

V67 fit its seen counterfactual pairs almost perfectly but missed three
pair-held-out numeric gates.  V68 keeps V67's native-width paired objective
unchanged and adds two deliberately small generalization pressures:

* a hardest-wrong-prototype margin over the fold-local numeric teachers; and
* a scale-normalized anchor to the all-row pre-refinement controller.

Both operate only on continuous tensors.  They neither consume nor serialize
question text, answer text, token IDs, validation data, or oracle metadata.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HardNegativeDiagnosticsV68:
    mean_own_cosine: torch.Tensor
    mean_hardest_wrong_cosine: torch.Tensor
    mean_own_over_hardest_wrong_margin: torch.Tensor
    positive_margin_fraction: torch.Tensor


def hard_negative_prototype_margin_loss_v68(
    predicted: torch.Tensor,
    prototype_bank: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, HardNegativeDiagnosticsV68]:
    """Require each continuous prediction to beat its hardest wrong class.

    ``predicted`` is ``[B,C,H]`` and ``prototype_bank`` is ``[K,C,H]``.
    ``target_indices`` contains the fold-local numeric class index for every
    row.  The loss is a cosine hinge against the maximum non-target prototype;
    no answer strings or runtime codebook are involved.
    """

    if predicted.ndim != 3 or prototype_bank.ndim != 3:
        raise ValueError("V68 hard-negative tensors must be [B,C,H] and [K,C,H]")
    if predicted.shape[0] < 1 or prototype_bank.shape[0] < 2:
        raise ValueError("V68 hard-negative loss requires rows and two classes")
    if predicted.shape[1:] != prototype_bank.shape[1:]:
        raise ValueError("V68 prediction and prototype token shapes differ")
    if target_indices.shape != (predicted.shape[0],):
        raise ValueError("V68 target indices must have shape [B]")
    if not predicted.is_floating_point() or not prototype_bank.is_floating_point():
        raise TypeError("V68 hard-negative features must be floating tensors")
    if target_indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("V68 target indices must be integral")
    if not torch.isfinite(predicted).all() or not torch.isfinite(prototype_bank).all():
        raise ValueError("V68 hard-negative tensors must be finite")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or not 0.0 <= float(margin) <= 2.0
    ):
        raise ValueError("V68 hard-negative margin must lie in [0,2]")
    labels = target_indices.to(device=predicted.device, dtype=torch.long)
    if torch.any(labels < 0) or torch.any(labels >= prototype_bank.shape[0]):
        raise ValueError("V68 target index is outside the prototype bank")

    values = F.normalize(predicted.float().flatten(1), dim=-1, eps=1e-8)
    bank = F.normalize(
        prototype_bank.to(device=predicted.device).float().flatten(1),
        dim=-1,
        eps=1e-8,
    )
    similarities = values @ bank.T
    own = similarities.gather(1, labels[:, None]).squeeze(1)
    mask = F.one_hot(labels, num_classes=prototype_bank.shape[0]).bool()
    hardest_wrong = similarities.masked_fill(mask, -torch.inf).max(dim=1).values
    own_margin = own - hardest_wrong
    loss = F.relu(float(margin) - own_margin).mean()
    diagnostics = HardNegativeDiagnosticsV68(
        mean_own_cosine=own.detach().mean(),
        mean_hardest_wrong_cosine=hardest_wrong.detach().mean(),
        mean_own_over_hardest_wrong_margin=own_margin.detach().mean(),
        positive_margin_fraction=(own_margin.detach() > 0.0).float().mean(),
    )
    if not torch.isfinite(loss) or not all(
        torch.isfinite(value)
        for value in (
            diagnostics.mean_own_cosine,
            diagnostics.mean_hardest_wrong_cosine,
            diagnostics.mean_own_over_hardest_wrong_margin,
            diagnostics.positive_margin_fraction,
        )
    ):
        raise RuntimeError("V68 hard-negative objective produced a nonfinite result")
    return loss, diagnostics


def relative_parameter_anchor_loss_v68(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    anchors: Mapping[str, torch.Tensor],
    *,
    scale_floor: float = 1e-4,
) -> torch.Tensor:
    """Penalize refinement drift relative to the deterministic V66 base fit."""

    if (
        isinstance(scale_floor, bool)
        or not isinstance(scale_floor, (int, float))
        or not math.isfinite(float(scale_floor))
        or float(scale_floor) <= 0.0
    ):
        raise ValueError("V68 anchor scale floor must be finite and positive")
    parameters = tuple(named_parameters)
    if not parameters:
        raise ValueError("V68 anchor loss requires trainable parameters")
    if len({name for name, _parameter in parameters}) != len(parameters):
        raise ValueError("V68 anchor parameter names must be unique")
    if set(anchors) != {name for name, _parameter in parameters}:
        raise ValueError("V68 anchor inventory differs from optimizer scope")
    terms: list[torch.Tensor] = []
    for name, parameter in parameters:
        anchor = anchors[name]
        if anchor.shape != parameter.shape:
            raise ValueError(f"V68 anchor shape changed for {name}")
        if not parameter.is_floating_point() or not anchor.is_floating_point():
            raise TypeError("V68 anchor tensors must be floating")
        if not torch.isfinite(parameter).all() or not torch.isfinite(anchor).all():
            raise ValueError("V68 anchor tensors must be finite")
        reference = anchor.to(device=parameter.device, dtype=torch.float32)
        scale = reference.square().mean().clamp_min(float(scale_floor))
        terms.append((parameter.float() - reference).square().mean() / scale)
    result = torch.stack(terms).mean()
    if not torch.isfinite(result):
        raise RuntimeError("V68 parameter anchor produced a nonfinite result")
    return result


__all__ = [
    "HardNegativeDiagnosticsV68",
    "hard_negative_prototype_margin_loss_v68",
    "relative_parameter_anchor_loss_v68",
]
