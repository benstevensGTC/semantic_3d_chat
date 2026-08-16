"""Numeric class-separation objective for the V66 continuous adapter.

The class bank consists only of verified continuous Gemma prompt tensors.  The
loss never receives class names or answer text, and it is used only while
training.  Runtime still emits native-width continuous control tokens directly
into the frozen language model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NumericPrototypeDiagnostics:
    mean_own_cosine: torch.Tensor
    mean_best_other_cosine: torch.Tensor
    mean_margin: torch.Tensor
    top1_accuracy: torch.Tensor


def numeric_prototype_classification_loss(
    predicted: torch.Tensor,
    prototypes: torch.Tensor,
    class_indices: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, NumericPrototypeDiagnostics]:
    """Classify native-width controls against a verified numeric prompt bank.

    ``predicted`` has shape ``[B,C,H]`` and ``prototypes`` has shape
    ``[K,C,H]``.  Flattened cosine logits preserve all control-token coordinates
    and avoid introducing a low-dimensional semantic bottleneck.
    """

    if predicted.ndim != 3 or prototypes.ndim != 3:
        raise ValueError("V66 prototype objective expects [B,C,H] and [K,C,H]")
    if predicted.shape[1:] != prototypes.shape[1:] or prototypes.shape[0] < 2:
        raise ValueError("V66 prototype objective shapes or class count changed")
    if class_indices.shape != (predicted.shape[0],) or class_indices.dtype != torch.long:
        raise ValueError("V66 class indices must be int64 [B]")
    if (
        class_indices.numel() < 1
        or int(class_indices.min()) < 0
        or int(class_indices.max()) >= prototypes.shape[0]
    ):
        raise ValueError("V66 class index is outside the numeric prototype bank")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("V66 prototype temperature must be finite and positive")
    if not predicted.is_floating_point() or not prototypes.is_floating_point():
        raise TypeError("V66 prototype objective requires floating tensors")
    if not torch.isfinite(predicted).all() or not torch.isfinite(prototypes).all():
        raise ValueError("V66 prototype objective received nonfinite tensors")

    predicted_flat = F.normalize(predicted.float().flatten(1), dim=-1, eps=1e-8)
    prototype_flat = F.normalize(
        prototypes.to(device=predicted.device, dtype=torch.float32).flatten(1),
        dim=-1,
        eps=1e-8,
    )
    cosine = predicted_flat @ prototype_flat.T
    logits = cosine / float(temperature)
    indices = class_indices.to(device=predicted.device)
    loss = F.cross_entropy(logits, indices)
    own = cosine.gather(1, indices[:, None]).squeeze(1)
    other = cosine.masked_fill(
        F.one_hot(indices, num_classes=prototypes.shape[0]).bool(),
        -torch.inf,
    ).max(dim=1).values
    diagnostics = NumericPrototypeDiagnostics(
        mean_own_cosine=own.detach().mean(),
        mean_best_other_cosine=other.detach().mean(),
        mean_margin=(own - other).detach().mean(),
        top1_accuracy=(cosine.argmax(dim=1) == indices).detach().float().mean(),
    )
    if not torch.isfinite(loss) or not all(
        torch.isfinite(value)
        for value in (
            diagnostics.mean_own_cosine,
            diagnostics.mean_best_other_cosine,
            diagnostics.mean_margin,
            diagnostics.top1_accuracy,
        )
    ):
        raise RuntimeError("V66 numeric prototype objective produced a nonfinite value")
    return loss, diagnostics


__all__ = [
    "NumericPrototypeDiagnostics",
    "numeric_prototype_classification_loss",
]
