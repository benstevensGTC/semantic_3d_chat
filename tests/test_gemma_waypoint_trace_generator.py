from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.gemma_waypoint_policy import (
    HISTORY_FEATURE_DIM as CORE_HISTORY_FEATURE_DIM,
)
from semantic_3d_chat.robot.state_encoder import NumericRobotState, robot_state_vector
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM,
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V2,
    encode_rejected_waypoint_history_transition,
)
from semantic_3d_chat.training.gemma_waypoint_policy import (
    load_waypoint_trace_jsonl,
)
from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
    ACTION_TO_CODE,
    _HistoryEncoding,
    _SyntheticTraceBuilder,
    convert_v3_action_to_absolute,
    generate_gemma_waypoint_trace_dataset,
    load_gemma_waypoint_trace_dataset,
)
from semantic_3d_chat.training.navigation_target_trace_v3 import (
    MANIFEST_SCHEMA as V3_MANIFEST_SCHEMA,
)
from semantic_3d_chat.training.navigation_target_trace_v3 import (
    TRACE_SCHEMA as V3_TRACE_SCHEMA,
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(x: float, y: float, yaw: float) -> list[float]:
    state = NumericRobotState(
        position_m=(x, y, 0.0),
        body_yaw_degrees=yaw,
        camera_yaw_degrees=yaw,
        pitch_degrees=0.0,
        linear_velocity_xy_m=(0.0, 0.0),
        angular_velocity_degrees=0.0,
        collision=False,
        last_movement_delta_m=(0.0, 0.0, 0.0),
        scan_coverage=0.0,
        stopped=False,
    )
    return robot_state_vector(
        state,
        torch.tensor([-3.0, -2.5, 0.0]),
        torch.tensor([3.0, 2.5, 3.0]),
    ).tolist()


def _v3_row(
    *,
    index: int,
    scene_id: str,
    split: str,
    step: int,
    state: list[float],
    action: str,
    argument: float,
) -> dict[str, object]:
    return {
        "schema": V3_TRACE_SCHEMA,
        "sample_id": f"g_{index:08d}",
        "episode_id": f"e_{scene_id[6:]}_00",
        "scene_id": scene_id,
        "split": split,
        "family": "approach",
        "instruction": "Move toward the chair and stop.",
        "step_index": step,
        "state_features": state,
        "action_name": action,
        "argument_target_normalized": argument,
        "target_state_available": True,
        "oracle_target_xyz_m": [1.0, 0.0, 0.5],
        "target_query_sha256": hashlib.sha256(b"chair").hexdigest(),
        "target_coordinates_training_only": True,
        "oracle_available_at_runtime": False,
    }


def _write_v3_dataset(root: Path, train_scene: str, validation_scene: str) -> None:
    root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for scene_id, split in (
        (train_scene, "train"),
        (validation_scene, "validation"),
    ):
        yaw = 45.0
        distance = 0.5
        end_x = -math.sin(math.radians(yaw)) * distance
        end_y = math.cos(math.radians(yaw)) * distance
        base = len(rows)
        rows.extend(
            (
                _v3_row(
                    index=base,
                    scene_id=scene_id,
                    split=split,
                    step=0,
                    state=_state(0.0, 0.0, 0.0),
                    action="turn",
                    argument=1.0,
                ),
                _v3_row(
                    index=base + 1,
                    scene_id=scene_id,
                    split=split,
                    step=1,
                    state=_state(0.0, 0.0, yaw),
                    action="move_forward",
                    argument=1.0,
                ),
                _v3_row(
                    index=base + 2,
                    scene_id=scene_id,
                    split=split,
                    step=2,
                    state=_state(end_x, end_y, yaw),
                    action="stop",
                    argument=0.0,
                ),
            )
        )
    traces = root / "traces.jsonl"
    traces.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    body = {
        "schema": V3_MANIFEST_SCHEMA,
        "sample_count": len(rows),
        "episode_count": 2,
        "train_scene_ids": [train_scene],
        "validation_scene_ids": [validation_scene],
        "scene_splits_disjoint": True,
        "target_coordinates_oracle_derived": True,
        "target_coordinates_training_tree_only": True,
        "checkpoint_contains_object_labels": False,
        "runtime_oracle_inputs": False,
        "traces_sha256": _file_sha256(traces),
    }
    manifest = {**body, "dataset_sha256": _canonical_sha256(body)}
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_scene_inputs(root: Path, scene_id: str) -> tuple[Path, Path]:
    oracle_root = root / "oracle"
    map_root = root / "maps"
    oracle_dir = oracle_root / scene_id
    map_dir = map_root / scene_id
    oracle_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)
    oracle = {
        "scene_id": scene_id,
        "instances": [
            {
                "instance_id": "i_opaque_1",
                "category": "chair",
                "expected_center_xyz_m": [0.5, 0.0, 0.5],
            },
            {
                "instance_id": "i_opaque_2",
                "category": "table",
                "expected_center_xyz_m": [1.5, 0.0, 0.5],
            },
        ],
    }
    (oracle_dir / "oracle.json").write_text(
        json.dumps(oracle, sort_keys=True), encoding="utf-8"
    )
    # A far-away anonymous point satisfies the nonempty geometry contract while
    # room bounds provide the open-room collision boundary for these unit tests.
    np.savez_compressed(
        map_dir / "voxel_map.npz",
        centers_world=np.asarray([[10.0, 10.0, 1.0]], dtype=np.float32),
    )
    return oracle_root, map_root


def _config(tmp_path: Path) -> dict[str, object]:
    train_scene = "scene_000011"
    validation_scene = "scene_000031"
    source = tmp_path / "training" / "v3"
    _write_v3_dataset(source, train_scene, validation_scene)
    oracle_root, map_root = _write_scene_inputs(tmp_path, train_scene)
    _write_scene_inputs(tmp_path, validation_scene)
    return {
        "seed": 7,
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "robot": {
            "radius_m": 0.25,
            "max_move_m": 0.5,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.8,
            "surface_padding_m": 0.035,
        },
        "gemma_waypoint_traces": {
            "source_trace_dataset": str(source),
            "oracle_root": str(oracle_root),
            "map_root": str(map_root),
            "train_scene_ids": [train_scene],
            "validation_scene_ids": [validation_scene],
            "history_length": 8,
            "max_waypoint_step_m": 0.5,
            "planner_grid_resolution_m": 0.15,
            "lap_anchor_count": 8,
            "lap_wall_margin_m": 0.45,
            "lap_max_waypoints": 64,
            "lap_fixed_face_step_degrees": 40.0,
            "between_search_resolution_m": 0.05,
            "seed": 11,
            "profiles": {
                "smoke": {
                    "train_scene_limit": 1,
                    "validation_scene_limit": 1,
                    "start_pose_count": 1,
                    "initial_yaw_degrees": [0.0],
                    "between_pairs_per_scene": 1,
                },
                "full": {
                    "start_pose_count": 2,
                    "initial_yaw_degrees": [0.0, 90.0],
                    "between_pairs_per_scene": 1,
                },
                "live": {
                    "start_pose_count": 1,
                    "initial_yaw_degrees": [0.0],
                    "training_initial_yaw_jitter_degrees": [-0.5, 0.0, 0.5],
                    "between_pairs_per_scene": 1,
                    "object_targets_per_scene": 1,
                    "include_source_rows": False,
                    "lap_execution_drift_augmentation": {
                        "enabled": True,
                        "nominal_initial_yaws_only": True,
                        "post_face_rows_only": True,
                        "turn_magnitude_profiles_degrees": [
                            [40.0, 39.9959716796875, 39.999839782714844],
                            [
                                39.99665451049805,
                                39.56814193725586,
                                39.72727966308594,
                            ],
                        ],
                    },
                    "lap_recovery_augmentation": {
                        "enabled": True,
                        "full_post_recovery_continuation": True,
                        "nominal_initial_yaws_only": True,
                        "max_states_per_episode": 4,
                        "minimum_states_per_episode": 2,
                        "position_perturbation_m": 0.10,
                        "yaw_offsets_degrees": [-8.0, 8.0],
                        "rejected_proposal_distance_m": 0.45,
                        "rejection_streak_lengths": [1, 4, 8],
                    },
                },
            },
        },
    }


def _signed_area(start: list[float], waypoints: list[list[float]]) -> float:
    points = [np.asarray(start), *(np.asarray(point) for point in waypoints)]
    return 0.5 * sum(
        float(first[0] * second[1] - first[1] * second[0])
        for first, second in pairwise(points)
    )


def test_v3_coordinate_conversion_produces_exact_absolute_targets() -> None:
    base = {
        "state_features": _state(1.0, -0.5, 30.0),
        "argument_target_normalized": 0.5,
    }
    face = convert_v3_action_to_absolute(
        {**base, "action_name": "turn"},
        room_size_m=(6.0, 5.0, 3.0),
        max_turn_degrees=40.0,
        max_move_m=0.6,
    )
    assert face is not None
    assert face.action == "FACE"
    assert face.heading_degrees == pytest.approx(50.0)
    assert face.xy_m is None

    forward = convert_v3_action_to_absolute(
        {**base, "action_name": "move_forward"},
        room_size_m=(6.0, 5.0, 3.0),
        max_turn_degrees=40.0,
        max_move_m=0.6,
    )
    assert forward is not None
    # normalized .5 is a .45 m movement in the simulator's yaw convention.
    assert forward.xy_m == pytest.approx((0.775, -0.1102885683))

    backward = convert_v3_action_to_absolute(
        {**base, "action_name": "move_backward"},
        room_size_m=(6.0, 5.0, 3.0),
        max_turn_degrees=40.0,
        max_move_m=0.6,
    )
    assert backward is not None
    assert backward.xy_m == pytest.approx((1.225, -0.8897114317))

    stop = convert_v3_action_to_absolute(
        {**base, "action_name": "stop"},
        room_size_m=(6.0, 5.0, 3.0),
        max_turn_degrees=40.0,
        max_move_m=0.6,
    )
    assert stop is not None and stop.action == "STOP"
    assert stop.xy_m is None and stop.heading_degrees is None
    assert (
        convert_v3_action_to_absolute(
            {**base, "action_name": "scan"},
            room_size_m=(6.0, 5.0, 3.0),
            max_turn_degrees=40.0,
            max_move_m=0.6,
        )
        is None
    )


def test_history_encoding_matches_core_and_preserves_rejected_proposal() -> None:
    assert HISTORY_FEATURE_DIM == CORE_HISTORY_FEATURE_DIM == 12
    row = encode_rejected_waypoint_history_transition(
        action="move_to",
        unchanged_pose_xy_yaw=(1.5, -1.0, 30.0),
        requested_waypoint_delta_robot_m=(0.25, -0.5),
        requested_heading_degrees=-60.0,
        room_size_m=(6.0, 5.0, 3.0),
        max_waypoint_step_m=0.5,
    )
    assert row[:3] == (1.0, 0.0, 0.0)
    assert row[3:5] == pytest.approx((0.5, -0.4))
    assert row[5:7] == pytest.approx((0.5, math.sqrt(3.0) / 2.0))
    assert row[7:9] == pytest.approx((0.5, -1.0))
    # Rejected MOVE retains its active waypoint, while its inactive heading
    # head is canonicalized to the unchanged/current body yaw.
    assert row[9:11] == pytest.approx((0.5, math.sqrt(3.0) / 2.0))
    assert row[11] == 0.0


def test_execution_drift_prefix_keeps_fourth_face_label(
    tmp_path: Path,
) -> None:
    _, map_root = _write_scene_inputs(tmp_path, "scene_000011")
    collision_map = NumericCollisionMap.from_voxel_map(
        map_root / "scene_000011" / "voxel_map.npz",
        room_size_m=(6.0, 5.0, 3.0),
        robot_radius_m=0.25,
        collision_z_min_m=0.12,
        collision_z_max_m=1.8,
        surface_padding_m=0.035,
    )

    def fourth_face_row(profile: list[float]) -> dict[str, object]:
        builder = _SyntheticTraceBuilder(
            episode_id="lap_execution_drift_test",
            scene_id="scene_000011",
            split="train",
            family="lap_execution_drift",
            task_variant="lap_clockwise_execution_drift",
            instruction="Do a lap around the room.",
            start_xy=(-0.5, -0.25),
            initial_yaw=-90.0,
            episode_goal_xy_m=(-0.5, -0.25),
            training_target_xyz_m=None,
            collision_map=collision_map,
            room_size_m=(6.0, 5.0, 3.0),
            max_waypoint_step_m=0.5,
            max_turn_degrees=40.0,
            history_length=16,
        )
        for magnitude in profile:
            current = builder.pose.yaw
            builder.face_with_execution_drift(
                current + 40.0,
                executed_delta_degrees=magnitude,
            )
        current = builder.pose.yaw
        builder.face_with_execution_drift(
            current + 40.0,
            executed_delta_degrees=40.0,
        )
        return builder.rows[-1]

    tiny = fourth_face_row([40.0, 39.9959716796875, 39.999839782714844])
    moderate = fourth_face_row(
        [39.99665451049805, 39.56814193725586, 39.72727966308594]
    )
    assert tiny["history_pose_xy_yaw"][-1][2] == pytest.approx(29.9958114624)
    assert moderate["history_pose_xy_yaw"][-1][2] == pytest.approx(29.2920761108)
    for row in (tiny, moderate):
        assert row["expert_action"] == "FACE"
        current_yaw = float(row["history_pose_xy_yaw"][-1][2])
        relative_label = (
            float(row["expert_heading_degrees"]) - current_yaw + 180.0
        ) % 360.0 - 180.0
        assert relative_label == pytest.approx(40.0)
        assert row["history_action_codes"][-1] == ACTION_TO_CODE["FACE"]


def test_smoke_generator_is_deterministic_split_isolated_and_fully_labeled(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = tmp_path / "training" / "waypoints_a"
    second = tmp_path / "training" / "waypoints_b"
    manifest_a = generate_gemma_waypoint_trace_dataset(
        config, first, profile="smoke"
    )
    manifest_b = generate_gemma_waypoint_trace_dataset(
        config, second, profile="smoke"
    )
    assert manifest_a["dataset_sha256"] == manifest_b["dataset_sha256"]
    assert manifest_a["traces_sha256"] == manifest_b["traces_sha256"]
    assert (first / "traces.jsonl").read_bytes() == (
        second / "traces.jsonl"
    ).read_bytes()

    manifest, rows = load_gemma_waypoint_trace_dataset(first)
    assert manifest["runtime_preprogrammed_lap_function"] is False
    assert manifest["policy_selects_all_headings_and_waypoints_at_runtime"] is True
    assert manifest["lap_fixed_face_step_degrees"] == pytest.approx(40.0)
    assert manifest["lap_face_actions_fixed_magnitude"] is True
    assert manifest["lap_residual_face_actions_omitted"] is True
    assert manifest["synthetic_variant_episode_counts"]["lap_clockwise"] == 2
    assert manifest["synthetic_variant_episode_counts"]["lap_counterclockwise"] == 2
    assert manifest["synthetic_family_episode_counts"]["between"] == 2
    assert manifest["synthetic_family_episode_counts"]["face"] == 2
    assert manifest["synthetic_family_episode_counts"]["approach"] == 2
    assert manifest["history_feature_dim"] == CORE_HISTORY_FEATURE_DIM
    assert manifest["history_parameterization"] == "selected_action_parameters_v1"
    assert {row["expert_action"] for row in rows} == {"FACE", "MOVE_TO", "STOP"}

    trainer_dataset = load_waypoint_trace_jsonl(
        first,
        state_dim=18,
        history_dim=CORE_HISTORY_FEATURE_DIM,
        max_history_tokens=8,
        max_waypoint_step_m=0.5,
    )
    assert len(trainer_dataset.samples) == len(rows)
    assert trainer_dataset.scene_splits == {
        "train": ("scene_000011",),
        "validation": ("scene_000031",),
    }

    split_for_scene = {
        "scene_000011": "train",
        "scene_000031": "validation",
    }
    assert all(row["split"] == split_for_scene[row["scene_id"]] for row in rows)
    assert all(
        len(row["history_pose_xy_yaw"])
        == len(row["history_action_codes"]) + 1
        for row in rows
    )
    assert all(
        row["expert_action_code"] == ACTION_TO_CODE[row["expert_action"]]
        for row in rows
    )
    for row in rows:
        if row["expert_action"] != "FACE":
            continue
        current_yaw = float(row["history_pose_xy_yaw"][-1][2])
        target_yaw = float(row["expert_heading_degrees"])
        delta = (target_yaw - current_yaw + 180.0) % 360.0 - 180.0
        assert abs(delta) <= 45.0 + 1e-6

    episodes: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        episodes[str(row["episode_id"])].append(row)
    for episode_rows in episodes.values():
        if episode_rows[0]["task_variant"] == "converted_v3":
            continue
        for prior, current in pairwise(episode_rows):
            if (
                current["expert_action"] == "MOVE_TO"
                and current["family"] != "lap"
            ):
                assert prior["expert_action"] == "FACE"
    lap_signs: dict[str, list[float]] = defaultdict(list)
    lap_coarse_heading_residuals: list[float] = []
    for episode_rows in episodes.values():
        if episode_rows[0]["family"] != "lap":
            continue
        for row in episode_rows:
            if row["expert_action"] == "FACE":
                current_yaw = float(row["history_pose_xy_yaw"][-1][2])
                target_yaw = float(row["expert_heading_degrees"])
                delta = (target_yaw - current_yaw + 180.0) % 360.0 - 180.0
                assert abs(delta) == pytest.approx(40.0)
            elif row["expert_action"] == "MOVE_TO":
                current = row["history_pose_xy_yaw"][-1]
                target = row["expert_xy_m"]
                world_delta = np.asarray(target) - np.asarray(current[:2])
                target_yaw = math.degrees(
                    math.atan2(-float(world_delta[0]), float(world_delta[1]))
                )
                residual = (target_yaw - float(current[2]) + 180.0) % 360.0 - 180.0
                lap_coarse_heading_residuals.append(abs(residual))
                assert abs(residual) < 40.0 + 1e-6
                radians = math.radians(float(current[2]))
                expected_robot_delta = (
                    float(
                        np.dot(
                            world_delta,
                            np.asarray([math.cos(radians), math.sin(radians)]),
                        )
                    ),
                    float(
                        np.dot(
                            world_delta,
                            np.asarray([-math.sin(radians), math.cos(radians)]),
                        )
                    ),
                )
                assert row["waypoint_delta_robot_m"] == pytest.approx(
                    expected_robot_delta
                )
        start = episode_rows[0]["history_pose_xy_yaw"][-1][:2]
        waypoints = [
            row["expert_xy_m"]
            for row in episode_rows
            if row["expert_action"] == "MOVE_TO"
        ]
        area = _signed_area(start, waypoints)
        lap_signs[str(episode_rows[0]["task_variant"])].append(area)
    assert all(area < -0.5 for area in lap_signs["lap_clockwise"])
    assert all(area > 0.5 for area in lap_signs["lap_counterclockwise"])
    assert any(residual > 1.0 for residual in lap_coarse_heading_residuals)

    between_episodes = [
        episode
        for episode in episodes.values()
        if episode[0]["family"] == "between"
    ]
    assert len(between_episodes) == 2
    for episode in between_episodes:
        moves = [row for row in episode if row["expert_action"] == "MOVE_TO"]
        assert moves
        # The midpoint [1, 0] is free in the fixture, so the planner's exact
        # final waypoint label must equal it rather than a prose target.
        assert moves[-1]["expert_xy_m"] == pytest.approx([1.0, 0.0])
        assert episode[-1]["expert_action"] == "STOP"

    live = tmp_path / "training" / "waypoints_live"
    live_manifest = generate_gemma_waypoint_trace_dataset(
        config, live, profile="live"
    )
    _, live_rows = load_gemma_waypoint_trace_dataset(live)
    assert live_manifest["source_rows_included"] is False
    assert live_manifest["source_converted_episode_count"] == 0
    assert all(row["task_variant"] != "converted_v3" for row in live_rows)
    assert {row["family"] for row in live_rows} == {
        "lap",
        "lap_execution_drift",
        "lap_recovery",
        "between",
        "face",
        "approach",
    }
    assert live_manifest["lap_execution_drift_augmentation_enabled"] is True
    assert live_manifest["lap_execution_drift_nominal_initial_yaws_only"] is True
    assert live_manifest["lap_execution_drift_post_face_rows_only"] is True
    assert live_manifest["lap_execution_drift_transition_aligned_histories"] is True
    assert live_manifest["lap_execution_drift_train_split_only"] is True
    drift_rows = [
        row for row in live_rows if row["family"] == "lap_execution_drift"
    ]
    assert drift_rows
    assert len(drift_rows) == live_manifest["lap_execution_drift_sample_count"]
    assert {row["split"] for row in drift_rows} == {"train"}
    for row in drift_rows:
        assert row["history_action_codes"][-1] == ACTION_TO_CODE["FACE"]
        assert len(row["history_pose_xy_yaw"]) == len(
            row["history_action_codes"]
        ) + 1
        assert len(row["history"]) == len(row["history_action_codes"])
        current_yaw = math.radians(float(row["history_pose_xy_yaw"][-1][2]))
        assert row["state_features"][3:5] == pytest.approx(
            [math.sin(current_yaw), math.cos(current_yaw)]
        )
    assert live_manifest["lap_recovery_augmentation_enabled"] is True
    assert live_manifest["lap_recovery_full_post_recovery_continuation"] is True
    assert live_manifest["lap_recovery_nominal_initial_yaws_only"] is True
    assert live_manifest["lap_recovery_correct_stop_supervised"] is True
    assert live_manifest["lap_recovery_sample_count"] > 0
    assert live_manifest["lap_recovery_train_split_only"] is True
    assert live_manifest["offline_dagger_recovery_labels_only"] is True
    assert live_manifest["runtime_recovery_planner_available"] is False

    nominal_laps = [row for row in live_rows if row["family"] == "lap"]
    initial_yaws_by_split: defaultdict[str, set[float]] = defaultdict(set)
    for row in nominal_laps:
        if row["step_index"] == 0:
            initial_yaws_by_split[str(row["split"])].add(
                round(float(row["history_pose_xy_yaw"][0][2]), 3)
            )
    assert initial_yaws_by_split["train"] == {-0.5, 0.0, 0.5}
    assert initial_yaws_by_split["validation"] == {0.0}

    recovery_rows = [row for row in live_rows if row["family"] == "lap_recovery"]
    assert recovery_rows
    assert {row["split"] for row in recovery_rows} == {"train"}
    assert {row["expert_action"] for row in recovery_rows} == {
        "FACE",
        "MOVE_TO",
        "STOP",
    }
    recovery_episodes: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in recovery_rows:
        recovery_episodes[str(row["episode_id"])].append(row)
    assert len(recovery_episodes) == live_manifest["lap_recovery_episode_count"]
    assert len(recovery_rows) == live_manifest["lap_recovery_sample_count"]

    first_rejection_streaks: set[int] = set()
    for episode in recovery_episodes.values():
        assert len(episode) > 1
        assert [int(row["step_index"]) for row in episode] == list(
            range(len(episode))
        )
        first = episode[0]
        trailing_rejections = 0
        for history_row in reversed(first["history"]):
            if history_row[-1] != 0.0:
                break
            trailing_rejections += 1
        first_rejection_streaks.add(trailing_rejections)
        assert episode[-1]["expert_action"] == "STOP"
        assert episode[-1]["history_pose_xy_yaw"][-1][:2] == pytest.approx(
            episode[-1]["episode_goal_xy_m"]
        )
    assert first_rejection_streaks == {1, 4, 8}

    collision_map = NumericCollisionMap.from_voxel_map(
        Path(config["gemma_waypoint_traces"]["map_root"])
        / "scene_000011"
        / "voxel_map.npz",
        room_size_m=(6.0, 5.0, 3.0),
        robot_radius_m=0.25,
        collision_z_min_m=0.12,
        collision_z_max_m=1.8,
        surface_padding_m=0.035,
    )
    for episode in recovery_episodes.values():
        first = episode[0]
        pose = first["history_pose_xy_yaw"][-1]
        rejection = first["history"][-1]
        yaw = math.radians(float(pose[2]))
        requested_right = float(rejection[7]) * 0.5
        requested_forward = float(rejection[8]) * 0.5
        rejected_world = np.asarray(pose[:2]) + (
            requested_right * np.asarray([math.cos(yaw), math.sin(yaw)])
            + requested_forward * np.asarray([-math.sin(yaw), math.cos(yaw)])
        )
        assert collision_map.segment_check(
            np.asarray(pose[:2]), rejected_world
        ).collision
        for row in episode:
            if row["expert_action"] == "MOVE_TO":
                assert not collision_map.segment_check(
                    np.asarray(row["history_pose_xy_yaw"][-1][:2]),
                    np.asarray(row["expert_xy_m"]),
                ).collision

    runtime_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "semantic_3d_chat"
        / "robot"
        / "gemma_waypoint_runtime.py"
    )
    runtime_tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        module.startswith("semantic_3d_chat.training")
        or module in {
            "semantic_3d_chat.robot.planner",
            "semantic_3d_chat.robot.semantic_patrol",
        }
        for module in imported_modules
    )


def test_generator_rejects_nontraining_destination(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="training tree"):
        generate_gemma_waypoint_trace_dataset(
            config, tmp_path / "runtime" / "waypoints", profile="smoke"
        )


def test_generator_rejects_lap_face_step_outside_expert_bound(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    traces = config["gemma_waypoint_traces"]
    assert isinstance(traces, dict)
    traces["lap_fixed_face_step_degrees"] = 45.1
    with pytest.raises(ValueError, match="Lap fixed FACE step"):
        generate_gemma_waypoint_trace_dataset(
            config,
            tmp_path / "training" / "invalid_lap_turns",
            profile="smoke",
        )


def test_v2_smoke_dataset_rebuilds_every_numeric_history_with_progress(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    traces = config["gemma_waypoint_traces"]
    assert isinstance(traces, dict)
    traces["history_parameterization"] = HISTORY_PARAMETERIZATION_V2
    traces["history_feature_dim"] = HISTORY_FEATURE_DIM_V2
    destination = tmp_path / "training" / "waypoints_v2"

    manifest = generate_gemma_waypoint_trace_dataset(
        config,
        destination,
        profile="smoke",
    )
    authenticated, rows = load_gemma_waypoint_trace_dataset(destination)

    assert authenticated == manifest
    assert manifest["history_feature_dim"] == 16
    assert manifest["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert manifest["history_goal_progress_from_numeric_receipts_only"] is True
    assert manifest["history_goal_progress_question_independent"] is True
    assert manifest["contradictory_exact_input_count"] == 0
    nonempty = [row["history"] for row in rows if row["history"]]
    assert nonempty
    assert all(len(history[-1]) == 16 for history in nonempty)
    assert all(history[-1][11] in {0.0, 1.0} for history in nonempty)
    assert any(history[-1][12] > 0.0 for history in nonempty)
    assert all(0.0 <= history[-1][15] < 1.0 for history in nonempty)


def test_history_parameterization_rejects_crossed_dimension_pairs() -> None:
    with pytest.raises(ValueError, match="parameterization/dimension pair"):
        _HistoryEncoding.from_settings(
            {
                "history_parameterization": HISTORY_PARAMETERIZATION_V2,
                "history_feature_dim": HISTORY_FEATURE_DIM,
            },
            history_length=16,
        )
