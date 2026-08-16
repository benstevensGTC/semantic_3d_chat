"""Runtime-safe numeric history rows for Gemma waypoint decisions.

The same function is used by offline expert generation and the live rejection-
only executor.  It imports no training, planner, oracle, simulator-metadata, or
semantic-retrieval code, so history tokens cannot become an environmental text
side channel.

The V1 12D row has the following stable layout::

    0:3   requested action one-hot [move_to, face, stop]
    3:5   resulting world XY normalized to the room's [-1, 1] interval
    5:7   sin/cos of resulting body yaw
    7:9   selected robot-frame [right, forward] movement / maximum step
    9:11  sin/cos of selected or action-neutral heading
    11    success (1 accepted/executed, 0 rejected)

Only parameters consumed by the selected action enter history.  MOVE_TO keeps
its requested robot-frame waypoint and uses the resulting/current body yaw as
a neutral heading.  FACE keeps its requested heading and uses a zero waypoint.
STOP uses a zero waypoint and the resulting/current yaw.  The raw model heads
remain unchanged in the decision receipt; this canonicalization applies only
to the next decision's numeric history and prevents inactive-head values from
creating a train/runtime distribution shift.

For a rejected MOVE_TO the active requested waypoint remains in the row while
the resulting pose equals the unchanged pre-action pose.  This lets Gemma
choose a recovery on its next forward instead of hiding failure or silently
replacing the rejected decision with a planner result.

V2 appends four goal-local continuous values.  They are derived only from the
numeric pose at the start of the conversational goal and exact action receipts:
normalized accepted path length, normalized signed swept area, normalized
distance back to the start, and normalized consecutive rejection streak.  The
ledger is state supplied *to* Gemma; it never chooses, replaces, or gates an
action and contains no route, target, label, oracle value, or STOP rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

HISTORY_FEATURE_DIM_V1: Final[int] = 12
HISTORY_FEATURE_DIM_V2: Final[int] = 16
HISTORY_PARAMETERIZATION_V1: Final[str] = "selected_action_parameters_v1"
HISTORY_PARAMETERIZATION_V2: Final[str] = (
    "selected_action_parameters_goal_progress_v2"
)
# Backward-compatible aliases used by all already-trained V1 checkpoints.
HISTORY_FEATURE_DIM: Final[int] = HISTORY_FEATURE_DIM_V1
HISTORY_PARAMETERIZATION: Final[str] = HISTORY_PARAMETERIZATION_V1
GOAL_PROGRESS_FEATURE_DIM: Final[int] = 4
HISTORY_ACTION_NAMES: Final[tuple[str, ...]] = ("move_to", "face", "stop")
HISTORY_ACTION_TO_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(HISTORY_ACTION_NAMES)
}


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _normalized_action(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("action must be a string")
    result = value.strip().casefold().replace("-", "_")
    if result not in HISTORY_ACTION_TO_INDEX:
        raise ValueError(f"action must be one of {HISTORY_ACTION_NAMES}")
    return result


def _cross_xy(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


@dataclass
class WaypointGoalProgressLedger:
    """Goal-scoped numeric consequences of exact Gemma action receipts.

    The mutable ledger is deliberately tiny and non-semantic.  Callers supply
    the exact pose immediately before and after one attempted primitive plus
    its boolean receipt status.  Failed receipts must be pose preserving.
    """

    start_xy_m: tuple[float, float]
    current_xy_m: tuple[float, float]
    cumulative_accepted_path_m: float = 0.0
    accepted_edge_cross_sum_m2: float = 0.0
    consecutive_rejections: int = 0

    @classmethod
    def from_initial_pose(
        cls, initial_pose_xy_yaw: Sequence[float]
    ) -> WaypointGoalProgressLedger:
        pose = np.asarray(initial_pose_xy_yaw, dtype=np.float64)
        if pose.shape not in {(2,), (3,)} or not np.isfinite(pose).all():
            raise ValueError("initial_pose_xy_yaw must contain two or three finite values")
        xy = float(pose[0]), float(pose[1])
        return cls(start_xy_m=xy, current_xy_m=xy)

    def __post_init__(self) -> None:
        start = _finite_vector(self.start_xy_m, 2, "start_xy_m")
        current = _finite_vector(self.current_xy_m, 2, "current_xy_m")
        for name, value in (
            ("cumulative_accepted_path_m", self.cumulative_accepted_path_m),
            ("accepted_edge_cross_sum_m2", self.accepted_edge_cross_sum_m2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if float(self.cumulative_accepted_path_m) < 0.0:
            raise ValueError("cumulative_accepted_path_m cannot be negative")
        if (
            isinstance(self.consecutive_rejections, bool)
            or not isinstance(self.consecutive_rejections, int)
            or self.consecutive_rejections < 0
        ):
            raise ValueError("consecutive_rejections must be a nonnegative integer")
        self.start_xy_m = float(start[0]), float(start[1])
        self.current_xy_m = float(current[0]), float(current[1])
        self.cumulative_accepted_path_m = float(self.cumulative_accepted_path_m)
        self.accepted_edge_cross_sum_m2 = float(self.accepted_edge_cross_sum_m2)

    @property
    def signed_swept_area_m2(self) -> float:
        """Signed area of accepted edges closed from current pose to start."""

        current = np.asarray(self.current_xy_m, dtype=np.float64)
        start = np.asarray(self.start_xy_m, dtype=np.float64)
        result = 0.5 * (
            self.accepted_edge_cross_sum_m2 + _cross_xy(current, start)
        )
        if not math.isfinite(result):
            raise RuntimeError("Goal-progress swept area became non-finite")
        return result

    @property
    def distance_to_start_m(self) -> float:
        result = math.dist(self.current_xy_m, self.start_xy_m)
        if not math.isfinite(result):
            raise RuntimeError("Goal-progress return distance became non-finite")
        return result

    def record_receipt(
        self,
        *,
        before_pose_xy_yaw: Sequence[float],
        after_pose_xy_yaw: Sequence[float],
        success: bool,
    ) -> None:
        """Advance from one exact numeric receipt without selecting an action."""

        before_pose = np.asarray(before_pose_xy_yaw, dtype=np.float64)
        after_pose = np.asarray(after_pose_xy_yaw, dtype=np.float64)
        if (
            before_pose.shape not in {(2,), (3,)}
            or after_pose.shape not in {(2,), (3,)}
            or not np.isfinite(before_pose).all()
            or not np.isfinite(after_pose).all()
        ):
            raise ValueError("Goal-progress receipt poses must contain finite XY[/yaw]")
        if not isinstance(success, bool):
            raise TypeError("Goal-progress receipt success must be boolean")
        before = before_pose[:2]
        after = after_pose[:2]
        current = np.asarray(self.current_xy_m, dtype=np.float64)
        if not np.allclose(before, current, rtol=0.0, atol=1e-9):
            raise ValueError("Goal-progress receipt does not start at the current pose")
        distance = float(np.linalg.norm(after - before))
        if not success:
            if distance > 1e-9:
                raise ValueError("Rejected goal-progress receipt changed position")
            self.consecutive_rejections += 1
            return
        self.cumulative_accepted_path_m += distance
        self.accepted_edge_cross_sum_m2 += _cross_xy(before, after)
        self.current_xy_m = float(after[0]), float(after[1])
        self.consecutive_rejections = 0

    def normalized_features(
        self,
        *,
        room_size_m: Sequence[float],
        rejection_streak_scale: int,
    ) -> tuple[float, float, float, float]:
        """Return the stable bounded V2 tail in its checkpointed field order."""

        room = _finite_vector(room_size_m, 3, "room_size_m")
        if np.any(room <= 0.0):
            raise ValueError("room_size_m must be positive")
        if (
            isinstance(rejection_streak_scale, bool)
            or not isinstance(rejection_streak_scale, int)
            or rejection_streak_scale < 1
        ):
            raise ValueError("rejection_streak_scale must be a positive integer")
        perimeter = 2.0 * float(room[0] + room[1])
        floor_area = float(room[0] * room[1])
        diagonal = math.hypot(float(room[0]), float(room[1]))
        values = (
            math.tanh(self.cumulative_accepted_path_m / perimeter),
            math.tanh(self.signed_swept_area_m2 / floor_area),
            min(self.distance_to_start_m / diagonal, 1.0),
            math.tanh(self.consecutive_rejections / rejection_streak_scale),
        )
        if not np.isfinite(values).all():
            raise RuntimeError("Goal-progress normalization produced non-finite values")
        if not (
            0.0 <= values[0] < 1.0
            and -1.0 < values[1] < 1.0
            and 0.0 <= values[2] <= 1.0
            and 0.0 <= values[3] < 1.0
        ):
            raise RuntimeError("Goal-progress normalization left its bounded domain")
        return tuple(float(value) for value in values)


def encode_waypoint_history_transition(
    *,
    action: str,
    result_pose_xy_yaw: Sequence[float],
    requested_waypoint_delta_robot_m: Sequence[float],
    requested_heading_degrees: float,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    success: bool,
) -> tuple[float, ...]:
    """Encode an accepted or rejected policy proposal as one 12D row.

    The raw model decision is not modified.  This function selects only the
    parameter(s) consumed by ``action`` for the history row:

    * MOVE_TO: requested waypoint plus resulting/current yaw as neutral heading;
    * FACE: zero waypoint plus the requested heading;
    * STOP: zero waypoint plus resulting/current yaw.

    This is intentionally shared by offline trace generation and live runtime.
    A rejected MOVE_TO therefore retains the waypoint that was actually rejected.
    """

    normalized_action = _normalized_action(action)
    pose = _finite_vector(result_pose_xy_yaw, 3, "result_pose_xy_yaw")
    requested_delta = _finite_vector(
        requested_waypoint_delta_robot_m,
        2,
        "requested_waypoint_delta_robot_m",
    )
    room = _finite_vector(room_size_m, 3, "room_size_m")
    if np.any(room <= 0.0):
        raise ValueError("room_size_m must be positive")
    if (
        isinstance(max_waypoint_step_m, bool)
        or not isinstance(max_waypoint_step_m, (int, float))
        or not math.isfinite(float(max_waypoint_step_m))
        or float(max_waypoint_step_m) <= 0.0
    ):
        raise ValueError("max_waypoint_step_m must be finite and positive")
    if (
        isinstance(requested_heading_degrees, bool)
        or not isinstance(requested_heading_degrees, (int, float))
        or not math.isfinite(float(requested_heading_degrees))
    ):
        raise ValueError("requested_heading_degrees must be finite")
    if not isinstance(success, bool):
        raise TypeError("success must be boolean")

    if normalized_action == "move_to":
        history_delta = requested_delta
        history_heading_degrees = float(pose[2])
    elif normalized_action == "face":
        history_delta = np.zeros(2, dtype=np.float64)
        history_heading_degrees = float(requested_heading_degrees)
    else:
        history_delta = np.zeros(2, dtype=np.float64)
        history_heading_degrees = float(pose[2])

    maximum = float(max_waypoint_step_m)
    # The policy head bounds each robot-frame component independently.  Keep
    # that exact square action domain instead of imposing a new circular clamp.
    if np.any(np.abs(requested_delta) > maximum + 1e-6):
        raise ValueError("requested waypoint component exceeds the policy bound")
    normalized_xy = np.asarray(
        [2.0 * pose[0] / room[0], 2.0 * pose[1] / room[1]],
        dtype=np.float64,
    )
    if np.any(np.abs(normalized_xy) > 1.0 + 1e-5):
        raise ValueError("resulting pose lies outside the room bounds")
    yaw_radians = math.radians(float(pose[2]))
    heading_radians = math.radians(history_heading_degrees)
    one_hot = [0.0, 0.0, 0.0]
    one_hot[HISTORY_ACTION_TO_INDEX[normalized_action]] = 1.0
    row = (
        *one_hot,
        float(normalized_xy[0]),
        float(normalized_xy[1]),
        math.sin(yaw_radians),
        math.cos(yaw_radians),
        float(history_delta[0] / maximum),
        float(history_delta[1] / maximum),
        math.sin(heading_radians),
        math.cos(heading_radians),
        float(success),
    )
    if len(row) != HISTORY_FEATURE_DIM or not np.isfinite(row).all():
        raise RuntimeError("Waypoint history encoder produced an invalid row")
    return tuple(float(value) for value in row)


def encode_waypoint_history_transition_v2(
    *,
    action: str,
    result_pose_xy_yaw: Sequence[float],
    requested_waypoint_delta_robot_m: Sequence[float],
    requested_heading_degrees: float,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    success: bool,
    goal_progress: Sequence[float],
) -> tuple[float, ...]:
    """Encode the V1 transition followed by the explicit four-value ledger."""

    progress = _finite_vector(
        goal_progress,
        GOAL_PROGRESS_FEATURE_DIM,
        "goal_progress",
    )
    if not (
        0.0 <= progress[0] < 1.0
        and -1.0 < progress[1] < 1.0
        and 0.0 <= progress[2] <= 1.0
        and 0.0 <= progress[3] < 1.0
    ):
        raise ValueError("goal_progress lies outside the V2 normalized domain")
    row = (
        *encode_waypoint_history_transition(
            action=action,
            result_pose_xy_yaw=result_pose_xy_yaw,
            requested_waypoint_delta_robot_m=requested_waypoint_delta_robot_m,
            requested_heading_degrees=requested_heading_degrees,
            room_size_m=room_size_m,
            max_waypoint_step_m=max_waypoint_step_m,
            success=success,
        ),
        *(float(value) for value in progress),
    )
    if len(row) != HISTORY_FEATURE_DIM_V2 or not np.isfinite(row).all():
        raise RuntimeError("V2 waypoint history encoder produced an invalid row")
    return tuple(float(value) for value in row)


def encode_rejected_waypoint_history_transition(
    *,
    action: str,
    unchanged_pose_xy_yaw: Sequence[float],
    requested_waypoint_delta_robot_m: Sequence[float],
    requested_heading_degrees: float,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
) -> tuple[float, ...]:
    """Record a rejected proposal without changing or replacing its target."""

    return encode_waypoint_history_transition(
        action=action,
        result_pose_xy_yaw=unchanged_pose_xy_yaw,
        requested_waypoint_delta_robot_m=requested_waypoint_delta_robot_m,
        requested_heading_degrees=requested_heading_degrees,
        room_size_m=room_size_m,
        max_waypoint_step_m=max_waypoint_step_m,
        success=False,
    )


__all__ = [
    "GOAL_PROGRESS_FEATURE_DIM",
    "HISTORY_ACTION_NAMES",
    "HISTORY_ACTION_TO_INDEX",
    "HISTORY_FEATURE_DIM",
    "HISTORY_FEATURE_DIM_V1",
    "HISTORY_FEATURE_DIM_V2",
    "HISTORY_PARAMETERIZATION",
    "HISTORY_PARAMETERIZATION_V1",
    "HISTORY_PARAMETERIZATION_V2",
    "WaypointGoalProgressLedger",
    "encode_rejected_waypoint_history_transition",
    "encode_waypoint_history_transition",
    "encode_waypoint_history_transition_v2",
]
