"""Uncertainty for the small samples these evaluations actually produce.

A bare proportion over fifty-odd queries invites a reader to take a nine-point
difference seriously. A Wilson interval says how wide the honest range is, and
because every condition answers the *same* questions, McNemar's exact test on
the paired disagreements is the right comparison -- it asks how often one
condition got an item the other missed, rather than treating the two arms as
independent samples.
"""

from __future__ import annotations

import math

__all__ = ["mcnemar_exact", "wilson_interval"]


def wilson_interval(hits: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """A 95% interval for a proportion that stays sane near 0 and 1."""

    if total <= 0:
        return (0.0, 1.0)
    proportion = hits / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def mcnemar_exact(first: list[float], second: list[float]) -> dict[str, float]:
    """Paired comparison of two conditions scored on the same items."""

    if len(first) != len(second):
        raise ValueError("paired conditions must cover the same items")
    only_first = sum(1 for a, b in zip(first, second, strict=True) if a > b)
    only_second = sum(1 for a, b in zip(first, second, strict=True) if b > a)
    discordant = only_first + only_second
    if discordant == 0:
        return {"p_value": 1.0, "only_first": 0, "only_second": 0}
    smaller = min(only_first, only_second)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    p_value = min(1.0, 2.0 * tail / (2.0 ** discordant))
    return {
        "p_value": round(p_value, 4),
        "only_first": only_first,
        "only_second": only_second,
    }
