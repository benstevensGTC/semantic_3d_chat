"""The rover's body: exact motion, and the only place that says "no".

Gemma decides where to go. This module decides only whether the requested move
is physically legal, executes it exactly if so, and reports what happened. It
never substitutes a different destination, never re-routes around an obstacle
and never invents a stop -- a rejected proposal comes straight back to the model
as a fact to reason about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from semantic_3d_chat.spatial_lens.reasoning import RoverPose, normalize_degrees
from semantic_3d_chat.spatial_lens.scene_graph import SceneGraph


@dataclass
class StepReceipt:
    step: int
    action: str
    requested: dict[str, Any]
    accepted: bool
    error_code: str | None
    pose_before: dict[str, float]
    pose_after: dict[str, float]
    distance_m: float
    reasoning: str

    def summary(self) -> str:
        if not self.accepted:
            return f"{self.action} rejected ({self.error_code})"
        if self.action in {"MOVE_TO", "MOVE_TOWARD"}:
            return (
                f"{self.action} reached ({self.pose_after['x_m']:+.2f}, "
                f"{self.pose_after['y_m']:+.2f}) accepted, moved "
                f"{self.distance_m:.2f} m"
            )
        if self.action == "FACE":
            return f"FACE {self.pose_after['yaw_degrees']:+.1f} deg accepted"
        return "STOP accepted"

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "requested": self.requested,
            "accepted": self.accepted,
            "error_code": self.error_code,
            "pose_before": self.pose_before,
            "pose_after": self.pose_after,
            "distance_m": round(self.distance_m, 4),
            "reasoning": self.reasoning,
        }


@dataclass
class Rover:
    """A disc robot on the perceived free-space grid."""

    graph: SceneGraph
    pose: RoverPose
    max_step_m: float = 0.5
    max_turn_degrees: float = 180.0
    path: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.path:
            self.path = [(self.pose.x_m, self.pose.y_m)]

    def _segment_clear(self, target: tuple[float, float]) -> bool:
        """Sample the straight line so the rover cannot tunnel through a leg."""

        start = (self.pose.x_m, self.pose.y_m)
        distance = math.dist(start, target)
        samples = max(2, int(distance / (self.graph.grid_resolution_m * 0.5)) + 1)
        for index in range(samples + 1):
            ratio = index / samples
            point = (
                start[0] + (target[0] - start[0]) * ratio,
                start[1] + (target[1] - start[1]) * ratio,
            )
            if not self.graph.is_free(point[0], point[1]):
                return False
        return True

    def move_to(self, x: float, y: float) -> tuple[bool, str | None, float]:
        if not all(math.isfinite(value) for value in (x, y)):
            return False, "E_NOT_FINITE", 0.0
        distance = math.dist((self.pose.x_m, self.pose.y_m), (x, y))
        if distance > self.max_step_m + 1e-6:
            return False, "E_STEP_TOO_LONG", 0.0
        if not self.graph.is_free(x, y):
            return False, "E_TARGET_BLOCKED", 0.0
        if not self._segment_clear((x, y)):
            return False, "E_PATH_BLOCKED", 0.0
        self.pose = RoverPose(x_m=x, y_m=y, yaw_degrees=self.pose.yaw_degrees)
        self.path.append((x, y))
        return True, None, distance

    def move_toward(self, x: float, y: float) -> tuple[bool, str | None, float]:
        """Advance along the straight line to a chosen point, up to one step.

        The destination is Gemma's; this only truncates the travel to the step
        limit. It performs no search and no detour -- if the truncated segment
        is blocked the move is refused and the model has to choose elsewhere.
        """

        if not all(math.isfinite(value) for value in (x, y)):
            return False, "E_NOT_FINITE", 0.0
        start = (self.pose.x_m, self.pose.y_m)
        distance = math.dist(start, (x, y))
        if distance < 1e-6:
            return False, "E_ALREADY_THERE", 0.0
        ratio = min(1.0, self.max_step_m / distance)
        target = (
            start[0] + (x - start[0]) * ratio,
            start[1] + (y - start[1]) * ratio,
        )
        if not self.graph.is_free(*target) or not self._segment_clear(target):
            return False, "E_PATH_BLOCKED", 0.0
        travelled = math.dist(start, target)
        self.pose = RoverPose(
            x_m=target[0], y_m=target[1], yaw_degrees=self.pose.yaw_degrees
        )
        self.path.append(target)
        return True, None, travelled

    def face(self, yaw_degrees: float) -> tuple[bool, str | None, float]:
        if not math.isfinite(yaw_degrees):
            return False, "E_NOT_FINITE", 0.0
        delta = abs(normalize_degrees(yaw_degrees - self.pose.yaw_degrees))
        if delta > self.max_turn_degrees + 1e-6:
            return False, "E_TURN_TOO_LARGE", 0.0
        self.pose = RoverPose(
            x_m=self.pose.x_m,
            y_m=self.pose.y_m,
            yaw_degrees=normalize_degrees(yaw_degrees),
        )
        return True, None, 0.0

    @property
    def path_length_m(self) -> float:
        return sum(
            math.dist(self.path[index - 1], self.path[index])
            for index in range(1, len(self.path))
        )


def choose_start_pose(graph: SceneGraph) -> RoverPose:
    """Put the rover on the free cell closest to the middle of the room."""

    start = graph.nearest_free(0.0, 0.0, max_radius_m=max(graph.room_size_m[:2]))
    if start is None:
        raise ValueError("The perceived room has no free floor for the rover")
    return RoverPose(x_m=start[0], y_m=start[1], yaw_degrees=0.0)


__all__ = ["Rover", "StepReceipt", "choose_start_pose", "probe_directions"]


def probe_directions(
    rover: Rover, *, count: int = 8
) -> list[tuple[float, tuple[float, float], bool]]:
    """Which straight-line moves are physically available right now.

    This is proximity sensing, not path planning: it reports whether a single
    bounded step along each compass bearing is clear, and where it would land.
    The model still decides which one, if any, serves the goal.
    """

    results: list[tuple[float, tuple[float, float], bool]] = []
    for index in range(count):
        yaw = -180.0 + index * (360.0 / count)
        radians = math.radians(yaw)
        # Project convention: yaw 0 faces +Y, yaw -90 faces +X.
        target = (
            rover.pose.x_m - math.sin(radians) * rover.max_step_m,
            rover.pose.y_m + math.cos(radians) * rover.max_step_m,
        )
        clear = rover.graph.is_free(*target) and rover._segment_clear(target)
        results.append((yaw, target, clear))
    return results
