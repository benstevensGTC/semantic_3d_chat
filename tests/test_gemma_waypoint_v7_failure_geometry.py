from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.planner import NumericWaypointPlanner
from semantic_3d_chat.robot.semantic_patrol import NumericPatrolPlanner
from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
    _canonical_sha256,
    _effective_settings,
    _lap_routes,
    _live_failure_dagger_augmentations,
    _live_failure_dagger_rows,
    _LiveFailureDaggerAugmentation,
    _replay_authenticated_live_prefix,
    _SyntheticTraceBuilder,
)

CONFIG_PATH = "configs/experiments/gemma_waypoint_policy_operator_dagger_v7.yaml"
REPORT_PATH = Path(
    "reports/gemma4/metrics/gemma_waypoint_dagger_v7_live_acceptance.json"
)
REPORT_SHA256 = "2c6e556de9459a8e2db4539781fe8e9b294cfc450cca7fde4f34865f930197db"
CHECKPOINT_SHA256 = (
    "8d373c09d8e15edeffde154540b156abd33f72997715a0c3593e0b1f92d126cc"
)
SCENE_PREFIX_SHA256 = (
    "52c33298140845d341fa2b4568f2c6e960279495890e08455caafa7d5bbc9c95"
)
RUNTIME_BINDING_SHA256 = (
    "800110e1545d77c1e374624ab837f6ed82cac54b269bf0359b840030db9e02cf"
)
PRE_D14_INPUT_SHA256 = (
    "c61251adb948d22ffa73c294b8eafd28d4ed0f33d027d88dabfa41a693b659d3"
)
POST_D15_INPUT_SHA256 = (
    "6ac247cc2df0783dd8993331b4155eb8c921d90c7f0452b0f43e7a84d0bc0234"
)
PRE_D14_POSE = (
    -1.8917434156060955,
    -0.06519614902193381,
    -90.00435638427734,
)
FAILED_D15_TARGET = (-1.576410768628845, 0.21046962982599612)
RESUME_WAYPOINT_INDEX = 5


@dataclass(frozen=True)
class _AuditContext:
    config: dict[str, Any]
    settings: dict[str, Any]
    report: dict[str, Any]
    source: _LiveFailureDaggerAugmentation
    expert_map: NumericCollisionMap
    runtime_map: NumericCollisionMap
    recovery_planner: NumericWaypointPlanner
    route: tuple[tuple[float, float], ...]


def _collision_map(
    config: dict[str, Any], settings: dict[str, Any], *, padding_m: float
) -> NumericCollisionMap:
    robot = config["robot"]
    return NumericCollisionMap.from_voxel_map(
        PROJECT_ROOT / "data_gemma4/maps/scene_000001/voxel_map.npz",
        room_size_m=config["scene"]["room_size_m"],
        robot_radius_m=float(robot["radius_m"]),
        collision_z_min_m=float(robot["collision_z_min_m"]),
        collision_z_max_m=float(robot["collision_z_max_m"]),
        surface_padding_m=padding_m,
    )


@pytest.fixture(scope="module")
def audit() -> _AuditContext:
    config = load_config(CONFIG_PATH)
    settings, _, _ = _effective_settings(config, "operator")
    report_bytes = (PROJECT_ROOT / REPORT_PATH).read_bytes()
    assert hashlib.sha256(report_bytes).hexdigest() == REPORT_SHA256
    report = json.loads(report_bytes)

    inherited = _live_failure_dagger_augmentations(settings)[-1]
    source = replace(
        inherited,
        report_path=PROJECT_ROOT / REPORT_PATH,
        report_sha256=REPORT_SHA256,
        checkpoint_sha256=CHECKPOINT_SHA256,
        first_rejection_step=15,
        failed_target_xy_m=FAILED_D15_TARGET,
        resume_waypoint_index=RESUME_WAYPOINT_INDEX,
    )
    expert_map = _collision_map(
        config,
        settings,
        padding_m=float(settings["expert_surface_padding_m"]),
    )
    runtime_map = _collision_map(
        config,
        settings,
        padding_m=float(config["robot"]["surface_padding_m"]),
    )
    patrol = NumericPatrolPlanner(
        expert_map,
        anchor_count=int(settings["lap_anchor_count"]),
        wall_margin_m=float(settings["lap_wall_margin_m"]),
        grid_resolution_m=float(settings["planner_grid_resolution_m"]),
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        max_waypoints=int(settings["lap_max_waypoints"]),
    )
    route = tuple(
        _lap_routes(np.asarray(source.initial_pose_xy_yaw[:2]), patrol)[
            "clockwise"
        ]
    )
    recovery_planner = NumericWaypointPlanner(
        expert_map,
        grid_resolution_m=float(settings["planner_grid_resolution_m"]),
        standoff_m=max(0.20, expert_map.inflated_radius_m),
        standoff_tolerance_m=0.05,
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
    )
    return _AuditContext(
        config=config,
        settings=settings,
        report=report,
        source=source,
        expert_map=expert_map,
        runtime_map=runtime_map,
        recovery_planner=recovery_planner,
        route=route,
    )


def _nominal_lap_rows(audit: _AuditContext) -> list[dict[str, Any]]:
    initial = audit.source.initial_pose_xy_yaw
    assert initial is not None
    builder = _SyntheticTraceBuilder(
        episode_id="v7_failure_geometry_audit",
        scene_id="scene_000001",
        split="train",
        family="lap",
        task_variant="lap_clockwise",
        instruction="Do a lap around the room.",
        start_xy=initial[:2],
        initial_yaw=float(initial[2]),
        episode_goal_xy_m=(float(initial[0]), float(initial[1])),
        training_target_xyz_m=None,
        collision_map=audit.expert_map,
        room_size_m=audit.config["scene"]["room_size_m"],
        max_waypoint_step_m=float(audit.settings["max_waypoint_step_m"]),
        max_turn_degrees=(
            float(audit.config["robot"]["max_turn_degrees"])
            - float(audit.settings["expert_turn_margin_degrees"])
        ),
        history_length=int(audit.settings["history_length"]),
    )
    for waypoint in audit.route:
        builder.move_to(
            waypoint,
            fixed_face_step_degrees=float(
                audit.settings["lap_fixed_face_step_degrees"]
            ),
        )
    builder.stop()
    return builder.rows


def _surface_distance_to_segment(
    collision_map: NumericCollisionMap,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> float:
    delta = end_xy - start_xy
    points = collision_map.obstacle_points_xy_m.astype(np.float64)
    fractions = np.clip(
        ((points - start_xy) @ delta) / float(np.dot(delta, delta)), 0.0, 1.0
    )
    closest = start_xy + fractions[:, None] * delta
    return float(np.min(np.linalg.norm(points - closest, axis=1)))


def test_v7_d14_is_the_first_action_class_divergence_and_wp5_is_unvisited(
    audit: _AuditContext,
) -> None:
    snapshot = audit.report["runtime_snapshot"]
    decisions = snapshot["model_decisions"]
    assert audit.report["gemma_runtime_binding_sha256"] == RUNTIME_BINDING_SHA256
    assert snapshot["scene_prefix_hash"] == SCENE_PREFIX_SHA256
    assert snapshot["control"]["navigation_checkpoint_sha256"] == (
        CHECKPOINT_SHA256
    )

    nominal = _nominal_lap_rows(audit)
    live_classes = [str(item["model_action"]).upper() for item in decisions]
    expert_classes = [str(item["expert_action"]) for item in nominal]
    assert live_classes[:13] == expert_classes[:13]
    assert expert_classes[13] == "MOVE_TO"
    assert nominal[13]["expert_xy_m"] == pytest.approx(audit.route[5])

    d14 = decisions[13]
    assert d14["step"] == 14
    assert d14["model_action"] == "face"
    assert d14["accepted"] is True
    assert d14["model_desired_heading_degrees"] == pytest.approx(
        -52.20322036743164
    )
    assert d14["model_action_logits"] == pytest.approx(
        [-0.9933091402053833, 39.35074996948242, -49.00237274169922]
    )

    replay = _replay_authenticated_live_prefix(
        augmentation=audit.source,
        decisions=decisions,
        transition_count=13,
        terminal_rejection_error_code=None,
        room_size_m=audit.config["scene"]["room_size_m"],
        max_waypoint_step_m=float(audit.settings["max_waypoint_step_m"]),
        fixed_face_step_degrees=float(
            audit.settings["lap_fixed_face_step_degrees"]
        ),
    )
    assert replay.pose.triple() == pytest.approx(PRE_D14_POSE)

    accepted_move_targets = [
        item["derived_world_waypoint_xy_m"]
        for item in decisions[:13]
        if item["model_action"] == "move_to" and item["accepted"] is True
    ]
    assert len(accepted_move_targets) == 5
    nearest_route_indices = [
        int(
            np.argmin(
                np.linalg.norm(
                    np.asarray(audit.route) - np.asarray(target), axis=1
                )
            )
        )
        for target in accepted_move_targets
    ]
    assert nearest_route_indices == [0, 1, 2, 3, 4]
    assert np.linalg.norm(
        np.asarray(accepted_move_targets[-1]) - np.asarray(audit.route[4])
    ) == pytest.approx(0.02801933701305601)

    direct_check = audit.expert_map.segment_check(
        np.asarray(PRE_D14_POSE[:2]), np.asarray(audit.route[5])
    )
    assert direct_check.collision is False
    assert direct_check.clearance_m == pytest.approx(0.013727277852233821)

    correction = _SyntheticTraceBuilder(
        episode_id="v7_pre_d14_geometry_audit",
        scene_id="scene_000001",
        split="train",
        family="lap_live_divergence_correction",
        task_variant="lap_clockwise_live_divergence_correction",
        instruction="Do a lap around the room.",
        start_xy=replay.pose.triple()[:2],
        initial_yaw=replay.pose.yaw,
        episode_goal_xy_m=(-0.5, -0.25),
        training_target_xyz_m=None,
        collision_map=audit.expert_map,
        room_size_m=audit.config["scene"]["room_size_m"],
        max_waypoint_step_m=float(audit.settings["max_waypoint_step_m"]),
        max_turn_degrees=float(audit.settings["lap_fixed_face_step_degrees"]),
        history_length=int(audit.settings["history_length"]),
        source_sample_sha256=REPORT_SHA256,
    )
    correction.pose = replay.pose
    correction.seed_training_history(
        poses=replay.poses,
        action_codes=replay.action_codes,
        numeric_rows=replay.numeric_history,
    )
    correction.move_to_direct(audit.route[5])
    first = correction.rows[0]
    assert len(first["history"]) == 13
    assert first["waypoint_delta_robot_m"] == pytest.approx(
        [0.00977216459382028, 0.4167441598201297]
    )
    assert _canonical_sha256(
        {"state_features": first["state_features"], "history": first["history"]}
    ) == PRE_D14_INPUT_SHA256


def test_v7_d15_collision_and_post_rejection_branch_have_exact_geometry(
    audit: _AuditContext,
) -> None:
    decisions = audit.report["runtime_snapshot"]["model_decisions"]
    first_rejected_index = next(
        index for index, item in enumerate(decisions) if item["accepted"] is False
    )
    assert first_rejected_index == 14
    d15 = decisions[first_rejected_index]
    assert d15["step"] == 15
    assert d15["model_action"] == "move_to"
    assert d15["error_code"] == "E_MODEL_COLLISION"
    assert d15["derived_world_waypoint_xy_m"] == pytest.approx(FAILED_D15_TARGET)
    assert d15["model_waypoint_delta_robot_m"] == pytest.approx(
        [-0.024572594091296196, 0.41811779141426086]
    )
    assert d15["model_desired_heading_degrees"] == pytest.approx(
        -58.870036125183105
    )
    assert d15["model_action_logits"] == pytest.approx(
        [129.1050262451172, -39.962825775146484, -104.18399047851562]
    )

    start = np.asarray(PRE_D14_POSE[:2])
    failed = np.asarray(FAILED_D15_TARGET)
    for collision_map, expected_penetration in (
        (audit.expert_map, 0.25249400611760586),
        (audit.runtime_map, 0.20249400611760587),
    ):
        assert collision_map.segment_check(start, failed).collision is True
        surface_distance = _surface_distance_to_segment(
            collision_map, start, failed
        )
        assert collision_map.inflated_radius_m - surface_distance == pytest.approx(
            expected_penetration
        )

    rows, metrics = _live_failure_dagger_rows(
        augmentation=audit.source,
        route_waypoints=audit.route,
        collision_map=audit.expert_map,
        recovery_planner=audit.recovery_planner,
        room_size_m=audit.config["scene"]["room_size_m"],
        max_waypoint_step_m=float(audit.settings["max_waypoint_step_m"]),
        fixed_face_step_degrees=float(
            audit.settings["lap_fixed_face_step_degrees"]
        ),
        history_length=int(audit.settings["history_length"]),
        episode_discriminator="v7_geometry_audit",
    )
    assert len(rows) == metrics["continuation_sample_count"] == 64
    assert metrics["total_decision_count"] == 79
    assert metrics["first_recovery_input_sha256"] == POST_D15_INPUT_SHA256
    assert metrics["path_length_m"] == pytest.approx(18.78404761689654)
    assert metrics["signed_winding_area_m2"] == pytest.approx(-4.211235961445963)
    assert metrics["return_error_m"] == pytest.approx(0.0)
    assert metrics["minimum_padded_map_clearance_m"] == pytest.approx(
        0.012784024478080025
    )
    assert rows[0]["expert_action"] == "MOVE_TO"
    assert rows[0]["expert_xy_m"] == pytest.approx(audit.route[5])
    assert rows[0]["waypoint_delta_robot_m"] == pytest.approx(
        [0.26315337011177264, 0.3232978406111784]
    )
    recovered_move_targets = [
        row["expert_xy_m"]
        for row in rows
        if row["expert_action"] == "MOVE_TO"
    ]
    assert np.asarray(recovered_move_targets) == pytest.approx(
        np.asarray(audit.route[5:])
    )
    assert rows[-1]["expert_action"] == "STOP"
    assert rows[-1]["history_pose_xy_yaw"][-1][:2] == pytest.approx(
        [-0.5, -0.25]
    )
