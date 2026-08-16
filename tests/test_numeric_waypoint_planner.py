from __future__ import annotations

import json

import numpy as np
import pytest

from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.planner import NumericWaypointPlanner


def _map(points: np.ndarray, *, radius: float = 0.16) -> NumericCollisionMap:
    return NumericCollisionMap(
        points.astype(np.float32),
        room_min_xy_m=(-3.0, -2.5),
        room_max_xy_m=(3.0, 2.5),
        robot_radius_m=radius,
        surface_padding_m=0.02,
    )


def test_direct_numeric_approach_stops_at_standoff_with_bounded_steps() -> None:
    collision = _map(np.asarray([[0.0, 0.0]], dtype=np.float32))
    planner = NumericWaypointPlanner(
        collision,
        grid_resolution_m=0.10,
        standoff_m=0.60,
        standoff_tolerance_m=0.10,
        max_waypoint_step_m=0.25,
    )

    plan = planner.plan((-2.0, 0.0), (0.0, 0.0))

    assert plan.direct_path is True
    assert plan.expanded_nodes == 0
    assert plan.target_distance_m == pytest.approx(0.60)
    prior = np.asarray([-2.0, 0.0])
    for waypoint in plan.waypoints_xy_m:
        current = np.asarray(waypoint)
        assert np.linalg.norm(current - prior) <= 0.25 + 1e-9
        assert collision.segment_check(prior, current).collision is False
        prior = current


def test_a_star_routes_around_anonymous_geometry() -> None:
    wall = np.stack(
        (
            np.zeros(33, dtype=np.float32),
            np.linspace(-0.8, 0.8, 33, dtype=np.float32),
        ),
        axis=1,
    )
    target_surface = np.asarray([[1.5, 0.0]], dtype=np.float32)
    collision = _map(np.concatenate((wall, target_surface)), radius=0.17)
    planner = NumericWaypointPlanner(
        collision,
        grid_resolution_m=0.10,
        standoff_m=0.60,
        standoff_tolerance_m=0.05,
        max_waypoint_step_m=0.30,
    )

    plan = planner.plan((-1.5, 0.0), (1.5, 0.0))

    assert plan.direct_path is False
    assert plan.expanded_nodes > 0
    assert abs(plan.target_distance_m - 0.60) <= 0.05 + 1e-6
    assert any(abs(point[1]) > 0.9 for point in plan.waypoints_xy_m)
    prior = np.asarray([-1.5, 0.0])
    for waypoint in plan.waypoints_xy_m:
        current = np.asarray(waypoint)
        assert collision.segment_check(prior, current).collision is False
        assert np.linalg.norm(current - prior) <= 0.30 + 1e-9
        prior = current


def test_plan_payload_is_numeric_and_contains_no_semantic_environment_fields() -> None:
    collision = _map(np.asarray([[0.0, 0.0]], dtype=np.float32))
    plan = NumericWaypointPlanner(collision).plan((-1.5, 0.0), (0.0, 0.0))

    encoded = json.dumps(plan.as_dict(), sort_keys=True, allow_nan=False)
    for prohibited in (
        "object",
        "category",
        "label",
        "caption",
        "relationship",
        "oracle",
    ):
        assert prohibited not in encoded.casefold()


def test_planner_rejects_colliding_start() -> None:
    collision = _map(np.asarray([[0.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="start position is in collision"):
        NumericWaypointPlanner(collision).plan((0.0, 0.0), (1.0, 0.0))


def test_free_point_plan_reaches_exact_goal_around_anonymous_geometry() -> None:
    wall = np.stack(
        (
            np.zeros(33, dtype=np.float32),
            np.linspace(-0.8, 0.8, 33, dtype=np.float32),
        ),
        axis=1,
    )
    collision = _map(wall, radius=0.17)
    planner = NumericWaypointPlanner(
        collision,
        grid_resolution_m=0.10,
        max_waypoint_step_m=0.30,
    )

    plan = planner.plan_to_free_point((-1.5, 0.0), (1.5, 0.0))

    assert plan.direct_path is False
    assert plan.target_distance_m == 0.0
    assert plan.goal_xy_m == pytest.approx((1.5, 0.0))
    assert plan.waypoints_xy_m[-1] == pytest.approx((1.5, 0.0))
    prior = np.asarray([-1.5, 0.0])
    for waypoint in plan.waypoints_xy_m:
        current = np.asarray(waypoint)
        assert collision.segment_check(prior, current).collision is False
        assert np.linalg.norm(current - prior) <= 0.30 + 1e-9
        prior = current


def test_free_point_plan_rejects_occupied_goal() -> None:
    collision = _map(np.asarray([[0.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="goal is in collision"):
        NumericWaypointPlanner(collision).plan_to_free_point((-1.5, 0.0), (0.0, 0.0))
