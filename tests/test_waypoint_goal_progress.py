from __future__ import annotations

import math
from itertools import pairwise

import pytest

from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V1,
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V1,
    HISTORY_PARAMETERIZATION_V2,
    WaypointGoalProgressLedger,
    encode_waypoint_history_transition,
    encode_waypoint_history_transition_v2,
)

ROOM_SIZE_M = (6.0, 5.0, 3.0)


def _record_move(
    ledger: WaypointGoalProgressLedger,
    before: tuple[float, float],
    after: tuple[float, float],
    *,
    success: bool = True,
) -> None:
    ledger.record_receipt(
        before_pose_xy_yaw=(*before, 0.0),
        after_pose_xy_yaw=(*after, 0.0),
        success=success,
    )


def test_v2_contract_is_explicit_and_v1_contract_is_unchanged() -> None:
    assert HISTORY_FEATURE_DIM_V1 == 12
    assert HISTORY_FEATURE_DIM_V2 == 16
    assert HISTORY_PARAMETERIZATION_V1 == "selected_action_parameters_v1"
    assert (
        HISTORY_PARAMETERIZATION_V2
        == "selected_action_parameters_goal_progress_v2"
    )


def test_goal_progress_closed_square_has_exact_path_area_and_return() -> None:
    ledger = WaypointGoalProgressLedger.from_initial_pose((0.0, 0.0, -90.0))
    points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    for before, after in pairwise(points):
        _record_move(ledger, before, after)

    assert ledger.cumulative_accepted_path_m == pytest.approx(4.0)
    assert ledger.signed_swept_area_m2 == pytest.approx(1.0)
    assert ledger.distance_to_start_m == pytest.approx(0.0)
    assert ledger.consecutive_rejections == 0
    features = ledger.normalized_features(
        room_size_m=ROOM_SIZE_M,
        rejection_streak_scale=16,
    )
    assert features == pytest.approx(
        (math.tanh(4.0 / 22.0), math.tanh(1.0 / 30.0), 0.0, 0.0)
    )


def test_rejections_change_only_streak_and_success_resets_it() -> None:
    ledger = WaypointGoalProgressLedger.from_initial_pose((0.0, 0.0))
    _record_move(ledger, (0.0, 0.0), (0.25, 0.0))
    _record_move(ledger, (0.25, 0.0), (0.25, 0.0), success=False)
    _record_move(ledger, (0.25, 0.0), (0.25, 0.0), success=False)

    assert ledger.cumulative_accepted_path_m == pytest.approx(0.25)
    assert ledger.signed_swept_area_m2 == pytest.approx(0.0)
    assert ledger.distance_to_start_m == pytest.approx(0.25)
    assert ledger.consecutive_rejections == 2
    assert ledger.normalized_features(
        room_size_m=ROOM_SIZE_M,
        rejection_streak_scale=16,
    )[-1] == pytest.approx(math.tanh(2.0 / 16.0))

    # An accepted pose-preserving FACE receipt resets the rejection streak; no
    # action is selected or corrected by the ledger.
    ledger.record_receipt(
        before_pose_xy_yaw=(0.25, 0.0, 0.0),
        after_pose_xy_yaw=(0.25, 0.0, 40.0),
        success=True,
    )
    assert ledger.consecutive_rejections == 0
    assert ledger.cumulative_accepted_path_m == pytest.approx(0.25)


def test_rejection_normalization_does_not_alias_after_history_window_fills() -> None:
    full_window = WaypointGoalProgressLedger(
        start_xy_m=(0.0, 0.0),
        current_xy_m=(0.0, 0.0),
        consecutive_rejections=16,
    )
    longer_streak = WaypointGoalProgressLedger(
        start_xy_m=(0.0, 0.0),
        current_xy_m=(0.0, 0.0),
        consecutive_rejections=26,
    )
    at_16 = full_window.normalized_features(
        room_size_m=ROOM_SIZE_M, rejection_streak_scale=16
    )[-1]
    at_26 = longer_streak.normalized_features(
        room_size_m=ROOM_SIZE_M, rejection_streak_scale=16
    )[-1]

    assert at_16 == pytest.approx(math.tanh(1.0))
    assert at_26 == pytest.approx(math.tanh(26.0 / 16.0))
    assert 0.0 < at_16 < at_26 < 1.0


def test_ledger_fails_closed_on_pose_discontinuity_or_moving_rejection() -> None:
    ledger = WaypointGoalProgressLedger.from_initial_pose((0.0, 0.0))
    with pytest.raises(ValueError, match="does not start at the current pose"):
        _record_move(ledger, (0.1, 0.0), (0.2, 0.0))
    with pytest.raises(ValueError, match="Rejected.*changed position"):
        _record_move(ledger, (0.0, 0.0), (0.1, 0.0), success=False)


def test_v2_encoder_appends_only_bounded_progress_to_identical_v1_row() -> None:
    arguments = {
        "action": "move_to",
        "result_pose_xy_yaw": (0.2, -0.1, 15.0),
        "requested_waypoint_delta_robot_m": (0.2, -0.1),
        "requested_heading_degrees": 15.0,
        "room_size_m": ROOM_SIZE_M,
        "max_waypoint_step_m": 0.5,
        "success": True,
    }
    v1 = encode_waypoint_history_transition(**arguments)
    progress = (0.2, -0.1, 0.05, 0.25)
    v2 = encode_waypoint_history_transition_v2(
        **arguments,
        goal_progress=progress,
    )

    assert len(v1) == HISTORY_FEATURE_DIM_V1
    assert len(v2) == HISTORY_FEATURE_DIM_V2
    assert v2[:HISTORY_FEATURE_DIM_V1] == v1
    assert v2[HISTORY_FEATURE_DIM_V1:] == progress


@pytest.mark.parametrize(
    "progress",
    [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, -0.1, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.1),
        (0.0, float("nan"), 0.0, 0.0),
    ],
)
def test_v2_encoder_rejects_out_of_domain_progress(
    progress: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError):
        encode_waypoint_history_transition_v2(
            action="face",
            result_pose_xy_yaw=(0.0, 0.0, 0.0),
            requested_waypoint_delta_robot_m=(0.0, 0.0),
            requested_heading_degrees=0.0,
            room_size_m=ROOM_SIZE_M,
            max_waypoint_step_m=0.5,
            success=True,
            goal_progress=progress,
        )
