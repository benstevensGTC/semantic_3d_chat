"""The shuffled-scene-token causal control.

Zeroing the scene prefix answers "does the controller use the room at all".
It cannot distinguish a controller that genuinely reads *where* things are from
one that has merely learned which features a room contains.  Permuting the 256
content latents keeps the exact multiset and destroys only the arrangement, so
a controller that survives the shuffle unchanged is not spatially grounded.

The control is opt-in so that every previously recorded four-condition report
keeps its exact shape.
"""

from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.training.gemma_waypoint_policy import (
    CONTROL_CONDITIONS,
    DEFAULT_CONTROL_CONDITIONS,
    _controlled_prefix,
)


def _prefix() -> torch.Tensor:
    # Distinct per-token content so a permutation is detectable, plus the two
    # native Gemma boundary tokens the protocol requires.
    content = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1) * torch.ones(
        1, 8, 4
    )
    boi = torch.full((1, 1, 4), -1.0)
    eoi = torch.full((1, 1, 4), -2.0)
    return torch.cat([boi, content, eoi], dim=1)


def test_default_battery_excludes_the_new_control() -> None:
    assert DEFAULT_CONTROL_CONDITIONS == (
        "primary",
        "wrong_scene_prefix",
        "zero_scene_prefix",
        "zero_history",
    )
    assert "shuffled_scene_prefix" in CONTROL_CONDITIONS
    assert CONTROL_CONDITIONS[: len(DEFAULT_CONTROL_CONDITIONS)] == (
        DEFAULT_CONTROL_CONDITIONS
    )


def test_primary_and_unrelated_conditions_pass_the_prefix_through() -> None:
    prefix = _prefix()
    for condition in ("primary", "wrong_scene_prefix", "zero_history"):
        assert torch.equal(_controlled_prefix(prefix, condition), prefix)


def test_zero_control_clears_content_but_keeps_native_boundaries() -> None:
    prefix = _prefix()
    zeroed = _controlled_prefix(prefix, "zero_scene_prefix")
    assert torch.equal(zeroed[:, 0], prefix[:, 0])
    assert torch.equal(zeroed[:, -1], prefix[:, -1])
    assert torch.count_nonzero(zeroed[:, 1:-1]) == 0
    assert torch.equal(prefix, _prefix()), "control must not mutate its input"


def test_shuffle_preserves_the_multiset_and_only_reorders_it() -> None:
    prefix = _prefix()
    shuffled = _controlled_prefix(prefix, "shuffled_scene_prefix")

    # Boundaries untouched.
    assert torch.equal(shuffled[:, 0], prefix[:, 0])
    assert torch.equal(shuffled[:, -1], prefix[:, -1])
    # Same tokens, different order.
    original = prefix[0, 1:-1, 0]
    permuted = shuffled[0, 1:-1, 0]
    assert sorted(permuted.tolist()) == sorted(original.tolist())
    assert not torch.equal(permuted, original)
    # Norm-preserving: the shuffle cannot be detected as an energy change.
    assert torch.linalg.vector_norm(shuffled) == pytest.approx(
        float(torch.linalg.vector_norm(prefix)), rel=1e-6
    )
    assert torch.equal(prefix, _prefix()), "control must not mutate its input"


def test_shuffle_is_deterministic_across_calls() -> None:
    first = _controlled_prefix(_prefix(), "shuffled_scene_prefix")
    second = _controlled_prefix(_prefix(), "shuffled_scene_prefix")
    assert torch.equal(first, second)
