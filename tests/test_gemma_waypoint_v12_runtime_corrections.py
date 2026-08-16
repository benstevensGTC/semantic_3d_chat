from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.planner import NumericWaypointPlanner
from semantic_3d_chat.robot.semantic_patrol import NumericPatrolPlanner
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
    _canonical_sha256,
    _effective_settings,
    _HistoryEncoding,
    _lap_routes,
    _live_failure_dagger_augmentations,
    _live_failure_dagger_rows,
    _live_object_goal_dagger_augmentations,
    _live_object_goal_dagger_rows,
    _live_pre_divergence_dagger_augmentations,
    _load_training_oracle_object_geometry,
    load_gemma_waypoint_trace_dataset,
)

CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v12.yaml"
DATASET = Path("data_gemma4/training/gemma_waypoint_policy_v2_operator_dagger_v12")
LAP_RUNTIME_SHA256 = (
    "ef756eca81f4da644cbe8728ec9c005ad484967c68e87f1369d2a9ad08bcfc39"
)
OBJECT_RUNTIME_SHA256 = (
    "a853478a05e07ef5e1b8acec14f014fec41172dcf676ac8eeb8a30e46ee52aad"
)
CHECKPOINT_SHA256 = (
    "dcc8eb080c2a418d2ae1b6de8fc4665387ccf90f56f2add3bb1873a998606368"
)
SCENE_PREFIX_SHA256 = (
    "52c33298140845d341fa2b4568f2c6e960279495890e08455caafa7d5bbc9c95"
)
OBJECT_D7_INPUT_SHA256 = (
    "e3fefe1dd9d1cd43364715ee9f4b4e25ce3ea72028abab6ac23ef597840fd190"
)
OBJECT_NEIGHBOR_INPUT_SHA256 = (
    "9dafbbc8cc274311e21812f20aa75e02ae4f87eb52889400d424f09be19f0c4f",
    "fc4e2671602f7947e4f1752c6552f9efa561eb93f3d7db2106db79b4f385da7f",
    "9cb202ccbecee920b14627879551b36d010c4252ddb38029cbde374b2de0c41f",
    "043c09ff01120f6ebe373fe3bfbdfb4305b93b9ff0e69dfa28f65d5877245eae",
)
LAP_D53_INPUT_SHA256 = (
    "6cbe07f0df41883b99153d8ad15b211da8c72965a841452981121f493347550f"
)
LAP_D77_INPUT_SHA256 = (
    "30e8c1816e13e3151d6bf43a6d9fbedc1c6af6553de823c4d85128ce222de81e"
)


@dataclass(frozen=True)
class _Audit:
    config: dict[str, Any]
    settings: dict[str, Any]
    expert_collision_map: NumericCollisionMap
    runtime_collision_map: NumericCollisionMap
    recovery_planner: NumericWaypointPlanner
    approach_planner: NumericWaypointPlanner
    route: tuple[tuple[float, float], ...]
    encoding: _HistoryEncoding
    lap_source: Any
    object_source: Any
    target: Any
    bbox_min: Any
    bbox_max: Any


@pytest.fixture(scope="module")
def audit() -> _Audit:
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    robot = config["robot"]
    room = config["scene"]["room_size_m"]
    map_path = PROJECT_ROOT / "data_gemma4/maps/scene_000001/voxel_map.npz"

    def collision_map(surface_padding_m: float) -> NumericCollisionMap:
        return NumericCollisionMap.from_voxel_map(
            map_path,
            room_size_m=room,
            robot_radius_m=float(robot["radius_m"]),
            collision_z_min_m=float(robot["collision_z_min_m"]),
            collision_z_max_m=float(robot["collision_z_max_m"]),
            surface_padding_m=surface_padding_m,
        )

    expert_collision_map = collision_map(float(settings["expert_surface_padding_m"]))
    runtime_collision_map = collision_map(float(robot.get("surface_padding_m", 0.035)))
    patrol = NumericPatrolPlanner(
        expert_collision_map,
        anchor_count=int(settings["lap_anchor_count"]),
        wall_margin_m=float(settings["lap_wall_margin_m"]),
        grid_resolution_m=float(settings["planner_grid_resolution_m"]),
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        max_waypoints=int(settings["lap_max_waypoints"]),
    )
    target, bbox_min, bbox_max = _load_training_oracle_object_geometry(
        PROJECT_ROOT / "data/oracle/scene_000001/oracle.json",
        scene_id="scene_000001",
        category="chair",
        instance_id="i_000101",
    )
    lap_sources = _live_failure_dagger_augmentations(settings)
    object_sources = _live_object_goal_dagger_augmentations(settings)
    return _Audit(
        config=config,
        settings=settings,
        expert_collision_map=expert_collision_map,
        runtime_collision_map=runtime_collision_map,
        recovery_planner=NumericWaypointPlanner(
            expert_collision_map,
            grid_resolution_m=float(settings["planner_grid_resolution_m"]),
            standoff_m=max(0.20, expert_collision_map.inflated_radius_m),
            standoff_tolerance_m=0.05,
            max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        ),
        approach_planner=NumericWaypointPlanner(
            expert_collision_map,
            grid_resolution_m=float(settings["planner_grid_resolution_m"]),
            standoff_m=float(settings["approach_standoff_m"]),
            standoff_tolerance_m=float(settings["approach_standoff_tolerance_m"]),
            max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
            angular_samples=int(settings["approach_angular_samples"]),
        ),
        route=tuple(_lap_routes(np.asarray([-0.5, -0.25]), patrol)["clockwise"]),
        encoding=_HistoryEncoding.from_settings(
            settings, history_length=int(settings["history_length"])
        ),
        lap_source=next(
            source for source in lap_sources if source.report_sha256 == LAP_RUNTIME_SHA256
        ),
        object_source=next(
            source
            for source in object_sources
            if source.runtime_report_sha256 == OBJECT_RUNTIME_SHA256
        ),
        target=target,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def _input_sha256(row: dict[str, Any]) -> str:
    return _canonical_sha256(
        {"state_features": row["state_features"], "history": row["history"]}
    )


def _lap_rows(audit: _Audit) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _live_failure_dagger_rows(
        augmentation=audit.lap_source,
        route_waypoints=audit.route,
        collision_map=audit.expert_collision_map,
        recovery_planner=audit.recovery_planner,
        room_size_m=audit.config["scene"]["room_size_m"],
        max_waypoint_step_m=float(audit.settings["max_waypoint_step_m"]),
        fixed_face_step_degrees=float(
            audit.settings["lap_fixed_face_step_degrees"]
        ),
        history_length=int(audit.settings["history_length"]),
        history_encoding=audit.encoding,
    )


def _object_rows(audit: _Audit) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _live_object_goal_dagger_rows(
        augmentation=audit.object_source,
        target_xyz_m=audit.target,
        target_bbox_min_xy_m=audit.bbox_min,
        target_bbox_max_xy_m=audit.bbox_max,
        collision_map=audit.expert_collision_map,
        runtime_collision_map=audit.runtime_collision_map,
        approach_planner=audit.approach_planner,
        room_size_m=audit.config["scene"]["room_size_m"],
        max_waypoint_step_m=float(audit.settings["max_waypoint_step_m"]),
        max_turn_degrees=(
            float(audit.config["robot"]["max_turn_degrees"])
            - float(audit.settings["expert_turn_margin_degrees"])
        ),
        history_length=int(audit.settings["history_length"]),
        history_encoding=audit.encoding,
    )


def test_v12_cumulatively_retains_sources_and_v2_regularization(
    audit: _Audit,
) -> None:
    lap_sources = _live_failure_dagger_augmentations(audit.settings)
    divergence_branches = _live_pre_divergence_dagger_augmentations(
        audit.settings, lap_sources
    )
    object_sources = _live_object_goal_dagger_augmentations(audit.settings)
    assert len(lap_sources) == 7
    assert len({source.report_sha256 for source in lap_sources}) == 7
    assert lap_sources[-1].report_sha256 == LAP_RUNTIME_SHA256
    assert len(divergence_branches) == 6
    assert len(object_sources) == 2
    assert len({source.runtime_report_sha256 for source in object_sources}) == 2
    assert object_sources[-1].runtime_report_sha256 == OBJECT_RUNTIME_SHA256
    assert audit.lap_source.first_rejection_step == 52
    assert audit.object_source.correction_decision_step == 7
    assert audit.object_source.expected_expert_first_action == "STOP"
    assert audit.object_source.observed_action_accepted is False
    policy = audit.config["gemma_waypoint_policy"]
    assert policy["history_dim"] == HISTORY_FEATURE_DIM_V2
    assert policy["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert policy["action_refit_max_iter"] == 500
    assert policy["action_refit_learning_rate"] == pytest.approx(0.5)
    assert policy["action_refit_l2_weight"] == pytest.approx(1e-5)
    assert policy["retention_joint_training_epochs"] == 0
    assert policy["retention_maximum_shared_centered_logit_rmse"] == pytest.approx(
        0.25
    )
    assert policy["waypoint_branch_refit_enabled"] is True
    assert policy["waypoint_branch_refit_steps"] == 1000
    assert policy["retention_waypoint_weight"] == pytest.approx(20.0)
    assert policy[
        "waypoint_branch_refit_minimum_new_within_tolerance_fraction"
    ] == pytest.approx(1.0)
    assert policy["minimum_training_action_accuracy"] == pytest.approx(0.995)


def test_v12_successor_and_stop_neighborhood_parsers_are_fail_closed() -> None:
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")

    invalid_successor = copy.deepcopy(settings)
    invalid_successor["successor_live_failure_dagger_augmentations"] = ["not-a-map"]
    with pytest.raises(TypeError, match="successor_live_failure.*list of mappings"):
        _live_failure_dagger_augmentations(invalid_successor)

    duplicate_successor = copy.deepcopy(settings)
    successor = duplicate_successor["successor_live_failure_dagger_augmentations"][0]
    duplicate_successor["successor_live_failure_dagger_augmentations"].append(
        copy.deepcopy(successor)
    )
    with pytest.raises(ValueError, match="report SHA-256 values must be unique"):
        _live_failure_dagger_augmentations(duplicate_successor)

    for mutation in (
        {"observed_action_accepted": 0},
        {"stop_position_neighborhood_offsets_m": [[0.051, 0.0]]},
        {"stop_position_neighborhood_offsets_m": [[0.01, 0.0], [0.01, 0.0]]},
        {"expected_expert_first_action": "MOVE_TO"},
    ):
        invalid_object = copy.deepcopy(settings)
        invalid_object["additional_live_object_goal_dagger_augmentations"][0].update(
            mutation
        )
        with pytest.raises(ValueError, match="object-goal DAgger.*invalid"):
            _live_object_goal_dagger_augmentations(invalid_object)


def test_v12_exact_d7_and_numeric_drift_neighborhood_are_stop_supervised(
    audit: _Audit,
) -> None:
    rows, metrics = _object_rows(audit)
    assert len(rows) == 5
    assert [row["expert_action"] for row in rows] == ["STOP"] * 5
    assert [_input_sha256(row) for row in rows] == [
        OBJECT_D7_INPUT_SHA256,
        *OBJECT_NEIGHBOR_INPUT_SHA256,
    ]
    assert rows[0]["history_pose_xy_yaw"][-1] == pytest.approx(
        [-1.3479610430683904, -0.01765689076636459, 67.99934768676758]
    )
    np.testing.assert_allclose(
        [row["history_pose_xy_yaw"][-1][:2] for row in rows[1:]],
        [
            [-1.3629610430683903, -0.01765689076636459],
            [-1.3329610430683905, -0.01765689076636459],
            [-1.3479610430683904, -0.03265689076636459],
            [-1.3479610430683904, -0.01265689076636459],
        ],
        rtol=0.0,
        atol=1e-12,
    )
    assert metrics["first_correction_input_sha256"] == OBJECT_D7_INPUT_SHA256
    assert metrics["stop_neighborhood_input_sha256"] == list(
        OBJECT_NEIGHBOR_INPUT_SHA256
    )
    assert metrics["continuation_sample_count"] == 5
    assert metrics["continuation_episode_count"] == 5
    assert metrics["total_decision_count"] == 7
    assert metrics["path_length_m"] == pytest.approx(0.8792163569208261)
    assert metrics["target_center_progress_m"] == pytest.approx(0.5163528069996339)
    assert metrics["final_target_center_distance_m"] == pytest.approx(
        0.552303588846067
    )
    assert metrics["final_oracle_bbox_standoff_m"] == pytest.approx(
        0.2698761207029643
    )
    geometry = metrics["stop_neighborhood_geometry"]
    np.testing.assert_allclose(
        [item["offset_xy_m"] for item in geometry],
        [[-0.015, 0.0], [0.015, 0.0], [0.0, -0.015], [0.0, 0.005]],
        rtol=0.0,
        atol=1e-12,
    )
    assert min(item["target_center_progress_m"] for item in geometry) == pytest.approx(
        0.501648426137689
    )
    assert max(item["oracle_bbox_standoff_m"] for item in geometry) == pytest.approx(
        0.2848761207029643
    )
    assert metrics["minimum_padded_map_clearance_m"] == pytest.approx(
        0.0035718238906297106
    )
    assert metrics["offline_planner_used_for_labels_only"] is False
    assert metrics["runtime_planner_available"] is False
    assert metrics["runtime_oracle_available"] is False


def test_v12_d52_lap_recovery_is_exact_complete_and_model_stop_labeled(
    audit: _Audit,
) -> None:
    rows, metrics = _lap_rows(audit)
    assert len(rows) == 25
    assert metrics["first_rejection_step"] == 52
    assert metrics["first_rejected_target_xy_m"] == pytest.approx(
        [1.4742031455556566, -0.08804275167477256]
    )
    assert metrics["first_recovery_input_sha256"] == LAP_D53_INPUT_SHA256
    assert _input_sha256(rows[0]) == LAP_D53_INPUT_SHA256
    assert _input_sha256(rows[-1]) == LAP_D77_INPUT_SHA256
    assert [row["expert_action"] for row in rows[:3]] == [
        "MOVE_TO",
        "MOVE_TO",
        "MOVE_TO",
    ]
    np.testing.assert_allclose(
        [row["expert_xy_m"] for row in rows[:3]],
        [
            [1.5333333333333319, -0.20000000000000076],
            [1.9166666666666659, -0.10000000000000038],
            [2.3, 0.0],
        ],
        rtol=0.0,
        atol=1e-12,
    )
    assert rows[3]["expert_action"] == "FACE"
    assert rows[3]["expert_heading_degrees"] == pytest.approx(-129.9996452331543)
    assert rows[-1]["expert_action"] == "STOP"
    assert rows[-1]["history_pose_xy_yaw"][-1] == pytest.approx(
        [-0.5, -0.25, -9.999645233154297]
    )
    assert metrics["total_decision_count"] == 77
    assert metrics["path_length_m"] == pytest.approx(18.770158025417196)
    assert metrics["signed_winding_area_m2"] == pytest.approx(-4.271112417396937)
    assert metrics["return_error_m"] == pytest.approx(0.0)
    assert metrics["minimum_padded_map_clearance_m"] == pytest.approx(
        0.01733888318088672
    )
    assert metrics["model_labeled_stop"] is True
    assert metrics["offline_planner_used_for_labels_only"] is True
    assert metrics["runtime_planner_available"] is False
    assert metrics["source_checkpoint_sha256"] == CHECKPOINT_SHA256
    assert metrics["source_scene_prefix_sha256"] == SCENE_PREFIX_SHA256


def test_v12_generated_manifest_if_dataset_has_been_materialized() -> None:
    root = PROJECT_ROOT / DATASET
    if not (root / "manifest.json").is_file():
        pytest.skip("V12 dataset is intentionally not generated by this test module")
    manifest, rows = load_gemma_waypoint_trace_dataset(root)
    assert manifest["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert manifest["history_feature_dim"] == HISTORY_FEATURE_DIM_V2
    assert manifest["contradictory_exact_input_count"] == 0
    assert manifest["sample_count"] == 7816
    assert manifest["episode_count"] == 209
    assert manifest["live_failure_dagger_episode_count"] == 7
    assert manifest["live_failure_dagger_sample_count"] == 355
    assert manifest["live_divergence_dagger_episode_count"] == 6
    assert manifest["live_object_goal_dagger_episode_count"] == 6
    assert manifest["live_object_goal_dagger_sample_count"] == 12
    assert manifest["live_failure_dagger_all_sources_retained"] is True
    assert manifest["live_object_goal_dagger_all_sources_retained"] is True
    assert manifest["runtime_recovery_planner_available"] is False
    assert manifest["runtime_preprogrammed_lap_function"] is False
    assert len(rows) == manifest["sample_count"]
    v12_lap_rows = [
        row
        for row in rows
        if row.get("source_sample_sha256") == LAP_RUNTIME_SHA256
        and row["family"] == "lap_live_failure_recovery"
    ]
    v12_object_rows = [
        row
        for row in rows
        if row.get("source_sample_sha256") == OBJECT_RUNTIME_SHA256
        and row["family"] == "object_goal_live_divergence_correction"
    ]
    assert len(v12_lap_rows) == 25
    assert len(v12_object_rows) == 5
    assert [_input_sha256(row) for row in v12_object_rows] == [
        OBJECT_D7_INPUT_SHA256,
        *OBJECT_NEIGHBOR_INPUT_SHA256,
    ]
    assert _input_sha256(v12_lap_rows[0]) == LAP_D53_INPUT_SHA256
    assert _input_sha256(v12_lap_rows[-1]) == LAP_D77_INPUT_SHA256
    assert all(
        len(history_row) == HISTORY_FEATURE_DIM_V2
        for row in rows
        for history_row in row["history"]
    )
