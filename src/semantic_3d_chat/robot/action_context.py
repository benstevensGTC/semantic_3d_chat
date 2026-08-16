"""Fail-closed binding between action prefixes, robot state, and map geometry."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.state_encoder import (
    NumericRobotState,
    robot_state_vector,
    robot_state_vector_sha256,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ContinuousActionContext:
    """One immutable continuous context used to choose a bounded action."""

    active_prefix: torch.Tensor
    binding: dict[str, Any]
    numeric_state: NumericRobotState
    state_features: torch.Tensor

    @property
    def map_sha256(self) -> str:
        return str(self.binding["map_sha256"])


def _room_bounds(room_size_m: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor]:
    room = torch.as_tensor(room_size_m, dtype=torch.float32)
    if room.shape != (3,) or not torch.isfinite(room).all() or torch.any(room <= 0.0):
        raise ValueError("Action-context room size must contain three finite positive values")
    minimum = torch.tensor([-room[0] / 2.0, -room[1] / 2.0, 0.0])
    maximum = torch.tensor([room[0] / 2.0, room[1] / 2.0, room[2]])
    return minimum, maximum


def capture_continuous_action_context(
    runtime: Any,
    room_size_m: Sequence[float],
) -> ContinuousActionContext:
    """Capture and authenticate the inputs used by one action decision.

    Production runtimes expose an atomic snapshot.  A compatibility fallback
    remains for small test doubles, but it is accepted only when the same
    numeric-state digest is present in the continuous-prefix binding.
    """

    atomic = getattr(runtime, "continuous_action_context_snapshot", None)
    if callable(atomic):
        active, raw_binding, numeric_state = atomic()
    else:
        snapshot = getattr(runtime, "active_prefix_snapshot", None)
        simulator = getattr(runtime, "simulator", None)
        numeric = getattr(simulator, "numeric_state", None)
        if not callable(snapshot) or not callable(numeric):
            raise TypeError("Action runtime lacks a continuous context snapshot")
        active, raw_binding = snapshot()
        numeric_state = numeric()

    if not isinstance(active, torch.Tensor) or active.ndim != 3 or active.shape[0] != 1:
        raise RuntimeError("Action runtime returned an invalid continuous prefix")
    if not torch.isfinite(active.float()).all():
        raise RuntimeError("Action runtime returned nonfinite continuous tokens")
    if not isinstance(raw_binding, Mapping):
        raise TypeError("Action runtime returned an invalid prefix binding")
    if not isinstance(numeric_state, NumericRobotState):
        raise TypeError("Action runtime returned an invalid numeric robot state")
    binding = dict(raw_binding)
    required_digests = (
        "active_prefix_sha256",
        "scene_prefix_sha256",
        "map_sha256",
        "robot_state_sha256",
        "robot_tokens_sha256",
    )
    if any(
        not isinstance(binding.get(name), str)
        or _SHA256.fullmatch(str(binding[name])) is None
        for name in required_digests
    ):
        raise RuntimeError("Action runtime lacks a complete continuous-context binding")
    if prefix_sha256(active) != binding["active_prefix_sha256"]:
        raise RuntimeError("Action prefix differs from its authenticated binding")

    minimum, maximum = _room_bounds(room_size_m)
    state_features = robot_state_vector(numeric_state, minimum, maximum)
    if robot_state_vector_sha256(state_features) != binding["robot_state_sha256"]:
        raise RuntimeError("Numeric action state differs from the bound robot tokens")
    return ContinuousActionContext(
        active_prefix=active,
        binding=binding,
        numeric_state=numeric_state,
        state_features=state_features,
    )


def require_grounding_map_binding(
    context: ContinuousActionContext,
    *,
    grounding_map_sha256: str,
    scored_voxels: int,
    available_voxels: int,
) -> None:
    """Require target grounding to use the same complete map as the prefix."""

    if grounding_map_sha256 != context.map_sha256:
        raise RuntimeError("Target grounding map differs from the action scene prefix")
    if (
        isinstance(scored_voxels, bool)
        or isinstance(available_voxels, bool)
        or not isinstance(scored_voxels, int)
        or not isinstance(available_voxels, int)
        or scored_voxels < 1
        or scored_voxels != available_voxels
    ):
        raise RuntimeError("Target grounding did not score the complete active map")


__all__ = [
    "ContinuousActionContext",
    "capture_continuous_action_context",
    "require_grounding_map_binding",
]
