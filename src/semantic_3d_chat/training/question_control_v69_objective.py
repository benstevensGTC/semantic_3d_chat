"""Deterministic continuous augmentations for the V69 pair-sensitivity screen.

V69 expands each *training* counterfactual pair away from its decision
boundary in scene-signature space and optionally interpolates questions only
within an identical ordered numeric answer transition.  The helpers operate
exclusively on continuous tensors and opaque numeric class identifiers.  They
never read or serialize question text, answer text, oracle metadata, or held
pair tensors.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence

import torch


def extrapolate_pair_signatures_v69(
    pair_signatures: torch.Tensor,
    *,
    expansion: float,
) -> torch.Tensor:
    """Move both pair endpoints outward by a fixed fraction of their delta.

    ``pair_signatures`` is ``[U,2,M,H]``.  For endpoints ``left`` and
    ``right`` this returns ``left + e*(left-right)`` and
    ``right + e*(right-left)``.  The transform is symmetric, preserves the
    pair midpoint exactly, and introduces no synthetic semantic labels.
    """

    if pair_signatures.ndim != 4 or pair_signatures.shape[1] != 2:
        raise ValueError("V69 pair signatures must have shape [U,2,M,H]")
    if pair_signatures.shape[0] < 1 or pair_signatures.shape[2] < 1:
        raise ValueError("V69 pair signatures cannot be empty")
    if not pair_signatures.is_floating_point():
        raise TypeError("V69 pair signatures must be floating")
    if not torch.isfinite(pair_signatures).all():
        raise ValueError("V69 pair signatures must be finite")
    if (
        isinstance(expansion, bool)
        or not isinstance(expansion, (int, float))
        or not math.isfinite(float(expansion))
        or not 0.0 <= float(expansion) <= 0.5
    ):
        raise ValueError("V69 signature expansion must lie in [0,0.5]")
    midpoint = pair_signatures.mean(dim=1, keepdim=True)
    centered = pair_signatures - midpoint
    result = midpoint + (1.0 + 2.0 * float(expansion)) * centered
    if not torch.isfinite(result).all():
        raise RuntimeError("V69 signature extrapolation produced a nonfinite tensor")
    return result


def balanced_transition_indices_v69(
    transitions: Sequence[tuple[str, str]],
    *,
    seed: int,
) -> tuple[int, ...]:
    """Deterministically oversample every ordered transition to equal size.

    Transition values are opaque numeric-class identifiers.  No answer strings
    or natural-language content are needed.  Each nonempty transition bucket
    contributes the size of the largest bucket, with deterministic cycling and
    one deterministic global shuffle.
    """

    if not transitions:
        raise ValueError("V69 transition balancing requires at least one unit")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("V69 transition-balancing seed must be an integer")
    buckets: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, transition in enumerate(transitions):
        if (
            not isinstance(transition, tuple)
            or len(transition) != 2
            or any(not isinstance(value, str) or not value for value in transition)
        ):
            raise ValueError("V69 transitions must be pairs of opaque nonempty IDs")
        if transition[0] == transition[1]:
            raise ValueError("V69 counterfactual transition endpoints must differ")
        buckets[transition].append(index)
    maximum = max(len(indices) for indices in buckets.values())
    balanced: list[int] = []
    for transition in sorted(buckets):
        indices = buckets[transition]
        offset = random.Random(f"{seed}:{transition[0]}:{transition[1]}").randrange(
            len(indices)
        )
        balanced.extend(indices[(offset + item) % len(indices)] for item in range(maximum))
    random.Random(seed).shuffle(balanced)
    return tuple(balanced)


def mix_transition_questions_v69(
    questions: torch.Tensor,
    transitions: Sequence[tuple[str, str]],
    *,
    mix_weight: float,
    seed: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Mix each question with a deterministic same-transition partner.

    ``questions`` is one complete embedded question per pair unit, shaped
    ``[U,Q,H]``.  Buckets of size one are left unchanged.  A partner is always
    different when a bucket has multiple members, and the returned partner
    inventory makes the augmentation auditable without exposing text.
    """

    if questions.ndim != 3 or questions.shape[0] < 1 or questions.shape[1] < 1:
        raise ValueError("V69 questions must have shape [U,Q,H]")
    if len(transitions) != questions.shape[0]:
        raise ValueError("V69 question and transition inventories differ")
    if not questions.is_floating_point() or not torch.isfinite(questions).all():
        raise ValueError("V69 questions must be finite floating tensors")
    if (
        isinstance(mix_weight, bool)
        or not isinstance(mix_weight, (int, float))
        or not math.isfinite(float(mix_weight))
        or not 0.0 <= float(mix_weight) <= 0.25
    ):
        raise ValueError("V69 question mix weight must lie in [0,0.25]")
    # Reuse validation and canonical bucket ordering from the balancing helper.
    balanced_transition_indices_v69(transitions, seed=seed)
    buckets: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, transition in enumerate(transitions):
        buckets[transition].append(index)
    partners = list(range(len(transitions)))
    for transition in sorted(buckets):
        indices = buckets[transition]
        if len(indices) < 2:
            continue
        rotation = 1 + random.Random(
            f"{seed}:{transition[0]}:{transition[1]}:partner"
        ).randrange(len(indices) - 1)
        for offset, index in enumerate(indices):
            partners[index] = indices[(offset + rotation) % len(indices)]
    partner_tensor = questions[torch.tensor(partners, device=questions.device)]
    mixed = (1.0 - float(mix_weight)) * questions + float(mix_weight) * partner_tensor
    if not torch.isfinite(mixed).all():
        raise RuntimeError("V69 question interpolation produced a nonfinite tensor")
    return mixed, tuple(partners)


__all__ = [
    "balanced_transition_indices_v69",
    "extrapolate_pair_signatures_v69",
    "mix_transition_questions_v69",
]
