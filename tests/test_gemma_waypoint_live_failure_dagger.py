from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.planner import NumericWaypointPlanner
from semantic_3d_chat.robot.semantic_patrol import NumericPatrolPlanner
from semantic_3d_chat.training.gemma_waypoint_hidden_reuse import (
    load_waypoint_dataset_for_hidden_reuse,
)
from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
    _effective_settings,
    _lap_routes,
    _live_failure_dagger_augmentation,
    _live_failure_dagger_augmentations,
    _live_failure_dagger_rows,
    _live_pre_divergence_dagger_augmentations,
    _live_pre_divergence_dagger_rows,
    _LiveFailureDaggerAugmentation,
    _LivePreDivergenceDaggerAugmentation,
    _load_authenticated_live_failure_report,
    _SyntheticTraceBuilder,
    _validate_live_failure_history_window,
    _validate_live_history_window,
    load_gemma_waypoint_trace_dataset,
)

V5_CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v5.yaml"
V6_CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v6.yaml"
V7_CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v7.yaml"
V8_CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v8.yaml"
V4_DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v1_operator_dagger_v4"
)
V5_DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v1_operator_dagger_v5"
)
V6_DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v1_operator_dagger_v6"
)
V7_DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v1_operator_dagger_v7"
)
V8_DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v1_operator_dagger_v8"
)


def _failure_rows(
    config_path: str,
    augmentation_index: int,
    *,
    augmentation_override: _LiveFailureDaggerAugmentation | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = load_config(config_path)
    settings, _, _ = _effective_settings(config, "operator")
    augmentation = augmentation_override or _live_failure_dagger_augmentations(
        settings
    )[augmentation_index]
    robot = config["robot"]
    room = config["scene"]["room_size_m"]
    collision_map = NumericCollisionMap.from_voxel_map(
        PROJECT_ROOT / "data_gemma4/maps/scene_000001/voxel_map.npz",
        room_size_m=room,
        robot_radius_m=float(robot["radius_m"]),
        collision_z_min_m=float(robot["collision_z_min_m"]),
        collision_z_max_m=float(robot["collision_z_max_m"]),
        surface_padding_m=float(settings["expert_surface_padding_m"]),
    )
    patrol = NumericPatrolPlanner(
        collision_map,
        anchor_count=int(settings["lap_anchor_count"]),
        wall_margin_m=float(settings["lap_wall_margin_m"]),
        grid_resolution_m=float(settings["planner_grid_resolution_m"]),
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        max_waypoints=int(settings["lap_max_waypoints"]),
    )
    assert augmentation.initial_pose_xy_yaw is not None
    assert augmentation.route_direction is not None
    routes = _lap_routes(
        np.asarray(augmentation.initial_pose_xy_yaw[:2], dtype=np.float64),
        patrol,
    )
    planner = NumericWaypointPlanner(
        collision_map,
        grid_resolution_m=float(settings["planner_grid_resolution_m"]),
        standoff_m=max(0.20, collision_map.inflated_radius_m),
        standoff_tolerance_m=0.05,
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
    )
    return _live_failure_dagger_rows(
        augmentation=augmentation,
        route_waypoints=routes[augmentation.route_direction],
        collision_map=collision_map,
        recovery_planner=planner,
        room_size_m=room,
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        fixed_face_step_degrees=float(settings["lap_fixed_face_step_degrees"]),
        history_length=int(settings["history_length"]),
    )


def _v5_failure_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    return _failure_rows(V5_CONFIG, 0)


def _v6_failure_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    return _failure_rows(V6_CONFIG, 1)


def _v7_failure_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    return _failure_rows(V7_CONFIG, 2)


def _v8_live_source() -> _LiveFailureDaggerAugmentation:
    config = load_config(V7_CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    source = _live_failure_dagger_augmentations(settings)[-1]
    return replace(
        source,
        report_path=(
            PROJECT_ROOT
            / "reports/gemma4/metrics/gemma_waypoint_dagger_v7_live_acceptance.json"
        ),
        report_sha256=(
            "2c6e556de9459a8e2db4539781fe8e9b294cfc450cca7fde4f34865f930197db"
        ),
        checkpoint_sha256=(
            "8d373c09d8e15edeffde154540b156abd33f72997715a0c3593e0b1f92d126cc"
        ),
        first_rejection_step=15,
        failed_target_xy_m=(-1.576410768628845, 0.21046962982599612),
        resume_waypoint_index=5,
    )


def _v8_post_rejection_candidate_rows() -> tuple[
    list[dict[str, object]], dict[str, object]
]:
    return _failure_rows(
        V7_CONFIG,
        2,
        augmentation_override=_v8_live_source(),
    )


def _divergence_rows(
    augmentation: _LivePreDivergenceDaggerAugmentation,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = load_config(V7_CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    robot = config["robot"]
    room = config["scene"]["room_size_m"]
    collision_map = NumericCollisionMap.from_voxel_map(
        PROJECT_ROOT / "data_gemma4/maps/scene_000001/voxel_map.npz",
        room_size_m=room,
        robot_radius_m=float(robot["radius_m"]),
        collision_z_min_m=float(robot["collision_z_min_m"]),
        collision_z_max_m=float(robot["collision_z_max_m"]),
        surface_padding_m=float(settings["expert_surface_padding_m"]),
    )
    patrol = NumericPatrolPlanner(
        collision_map,
        anchor_count=int(settings["lap_anchor_count"]),
        wall_margin_m=float(settings["lap_wall_margin_m"]),
        grid_resolution_m=float(settings["planner_grid_resolution_m"]),
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        max_waypoints=int(settings["lap_max_waypoints"]),
    )
    assert augmentation.source.initial_pose_xy_yaw is not None
    assert augmentation.source.route_direction is not None
    routes = _lap_routes(
        np.asarray(augmentation.source.initial_pose_xy_yaw[:2], dtype=np.float64),
        patrol,
    )
    return _live_pre_divergence_dagger_rows(
        augmentation=augmentation,
        route_waypoints=routes[augmentation.source.route_direction],
        collision_map=collision_map,
        room_size_m=room,
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        fixed_face_step_degrees=float(settings["lap_fixed_face_step_degrees"]),
        history_length=int(settings["history_length"]),
    )


def _v7_divergence_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    config = load_config(V7_CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    live_sources = _live_failure_dagger_augmentations(settings)
    augmentation = _live_pre_divergence_dagger_augmentations(
        settings, live_sources
    )[0]
    return _divergence_rows(augmentation)


def _v8_pre_divergence_candidate_rows() -> tuple[
    list[dict[str, object]], dict[str, object]
]:
    augmentation = _LivePreDivergenceDaggerAugmentation(
        source=_v8_live_source(),
        branch_id="pre_d14_move_wp05",
        correction_decision_step=14,
        observed_model_action="face",
        expected_pose_xy_yaw=(
            -1.8917434156060955,
            -0.06519614902193381,
            -90.00435638427734,
        ),
        expected_input_sha256=(
            "c61251adb948d22ffa73c294b8eafd28d4ed0f33d027d88dabfa41a693b659d3"
        ),
        resume_waypoint_index=5,
        maximum_total_decisions=128,
    )
    return _divergence_rows(augmentation)


def test_v5_exact_live_failure_history_has_safe_complete_stop_continuation() -> None:
    rows, metrics = _v5_failure_rows()

    assert len(rows) == metrics["continuation_sample_count"] == 63
    assert metrics["first_rejection_step"] == 22
    assert metrics["exact_prefix_transition_count"] == 22
    assert metrics["exact_history_parameterization"] == (
        "selected_action_parameters_v1"
    )
    assert metrics["total_decision_count"] == 85
    assert metrics["path_length_m"] == pytest.approx(19.0031849643)
    assert metrics["signed_winding_area_m2"] == pytest.approx(-4.2327997437)
    assert metrics["return_error_m"] == pytest.approx(0.0)
    assert metrics["minimum_padded_map_clearance_m"] > 0.0
    assert metrics["model_labeled_stop"] is True
    assert metrics["runtime_planner_available"] is False

    first = rows[0]
    assert first["family"] == "lap_live_failure_recovery"
    assert first["split"] == "train"
    assert first["expert_action"] == "MOVE_TO"
    assert first["history_pose_xy_yaw"][-1] == pytest.approx(
        [-0.5142774038215565, 0.5953953545128375, 29.978729248046875]
    )
    assert len(first["history"]) == 16
    # The collision preflight does not call the simulator. The unchanged
    # robot-state token remains collision=false; only this history row records
    # the rejected active MOVE proposal.
    assert first["state_features"][12] == pytest.approx(0.0)
    assert first["history"][-1][0:3] == [1.0, 0.0, 0.0]
    assert first["history"][-1][-1] == 0.0
    assert rows[-1]["expert_action"] == "STOP"
    assert rows[-1]["history_pose_xy_yaw"][-1][:2] == pytest.approx(
        [-0.5, -0.25]
    )
    assert {row["split"] for row in rows} == {"train"}
    assert all(row["expert_planner_available_at_runtime"] is False for row in rows)


def test_live_failure_history_window_grows_then_rolls_and_fails_short() -> None:
    successful = [0.0] * 11 + [1.0]
    rejected = [0.0] * 12
    early_history = [list(successful) for _ in range(14)] + [rejected]
    mature_history = [list(successful) for _ in range(15)] + [rejected]

    _validate_live_failure_history_window(
        early_history,
        exact_prefix_transition_count=15,
        history_length=16,
    )
    _validate_live_failure_history_window(
        mature_history,
        exact_prefix_transition_count=22,
        history_length=16,
    )
    with pytest.raises(RuntimeError, match="exact transition window"):
        _validate_live_failure_history_window(
            early_history[:-1],
            exact_prefix_transition_count=15,
            history_length=16,
        )
    with pytest.raises(RuntimeError, match="exact transition window"):
        _validate_live_failure_history_window(
            mature_history[:-1],
            exact_prefix_transition_count=22,
            history_length=16,
        )
    malformed_success_terminal = [
        *[list(successful) for _ in range(14)],
        list(successful),
    ]
    with pytest.raises(RuntimeError, match="exact transition window"):
        _validate_live_failure_history_window(
            malformed_success_terminal,
            exact_prefix_transition_count=15,
            history_length=16,
        )


def test_real_step15_live_rejection_keeps_exact_15_transition_history() -> None:
    rows, metrics = _v8_post_rejection_candidate_rows()

    assert metrics["first_rejection_step"] == 15
    assert metrics["exact_prefix_transition_count"] == 15
    assert len(rows[0]["history"]) == 15
    assert rows[0]["history"][-1][-1] == 0.0
    assert metrics["first_recovery_input_sha256"] == (
        "6ac247cc2df0783dd8993331b4155eb8c921d90c7f0452b0f43e7a84d0bc0234"
    )


def test_real_pre_d14_divergence_keeps_exact_13_successful_transitions() -> None:
    rows, metrics = _v8_pre_divergence_candidate_rows()

    assert metrics["correction_decision_step"] == 14
    assert metrics["exact_prefix_transition_count"] == 13
    assert metrics["observed_model_action"] == "face"
    assert metrics["expert_first_action"] == "MOVE_TO"
    assert len(rows[0]["history"]) == 13
    assert rows[0]["history"][-1][-1] == 1.0
    assert rows[0]["waypoint_delta_robot_m"] == pytest.approx(
        [0.00977216459382028, 0.4167441598201297]
    )
    assert metrics["first_correction_input_sha256"] == (
        "c61251adb948d22ffa73c294b8eafd28d4ed0f33d027d88dabfa41a693b659d3"
    )
    with pytest.raises(RuntimeError, match="exact transition window"):
        _validate_live_history_window(
            rows[0]["history"][:-1],
            exact_prefix_transition_count=13,
            history_length=16,
            expected_terminal_success=True,
        )
    malformed_terminal = [list(item) for item in rows[0]["history"]]
    malformed_terminal[-1][-1] = 0.0
    with pytest.raises(RuntimeError, match="exact transition window"):
        _validate_live_history_window(
            malformed_terminal,
            exact_prefix_transition_count=13,
            history_length=16,
            expected_terminal_success=True,
        )


def test_v6_appends_v5_failure_without_dropping_v4_failure_source() -> None:
    config = load_config(V6_CONFIG)
    settings, train_scenes, validation_scenes = _effective_settings(
        config, "operator"
    )
    augmentations = _live_failure_dagger_augmentations(settings)

    assert [item.report_sha256 for item in augmentations] == [
        "3ec5458d763cd7d27533e981d309fb2e4ea58da592f6d5f980f6a7cc80c5680f",
        "670fd0eb87f471e084ade548d8f9fa78c961aa926b4d79913bccdcd49ba0adc4",
    ]
    assert [item.checkpoint_sha256 for item in augmentations] == [
        "b28642f2ccb3ef440ada4636566b5b5e237985ce50d9323196679780d626b2f1",
        "3f6785f082442c37ccb23db308e0f1e21b31dda28abe1c5f197d7be511baa199",
    ]
    assert train_scenes == ["scene_000001"]
    assert set(train_scenes).isdisjoint(validation_scenes)


def test_v6_exact_v5_state_gets_complete_safe_return_and_stop_labels() -> None:
    rows, metrics = _v6_failure_rows()

    assert len(rows) == metrics["continuation_sample_count"] == 60
    assert metrics["source_report_sha256"] == (
        "670fd0eb87f471e084ade548d8f9fa78c961aa926b4d79913bccdcd49ba0adc4"
    )
    assert metrics["source_checkpoint_sha256"] == (
        "3f6785f082442c37ccb23db308e0f1e21b31dda28abe1c5f197d7be511baa199"
    )
    assert metrics["source_scene_prefix_sha256"] == (
        "52c33298140845d341fa2b4568f2c6e960279495890e08455caafa7d5bbc9c95"
    )
    assert metrics["first_rejection_step"] == 22
    assert metrics["exact_prefix_transition_count"] == 22
    assert metrics["first_rejected_target_xy_m"] == pytest.approx(
        [-0.8826844833092204, 0.41691211338913026]
    )
    assert metrics["first_recovery_input_sha256"] == (
        "f74d3fe6dfdd0b428c9a8d5d6492005cc3f040cc33712cd468ae2500c7757e60"
    )
    assert metrics["total_decision_count"] == 82
    assert metrics["path_length_m"] == pytest.approx(18.7801968197)
    assert metrics["signed_winding_area_m2"] == pytest.approx(-4.2154355624)
    assert metrics["return_error_m"] == pytest.approx(0.0)
    assert metrics["minimum_padded_map_clearance_m"] > 0.0
    assert metrics["model_labeled_stop"] is True
    assert metrics["runtime_planner_available"] is False

    first = rows[0]
    assert first["family"] == "lap_live_failure_recovery"
    assert first["source_sample_sha256"] == metrics["source_report_sha256"]
    assert first["history_pose_xy_yaw"][-1] == pytest.approx(
        [-0.5600231068373551, 0.30971998670610523, 69.9988899230957]
    )
    assert len(first["history"]) == 16
    assert first["history"][-1][-1] == 0.0
    assert rows[-1]["expert_action"] == "STOP"
    assert rows[-1]["history_pose_xy_yaw"][-1][:2] == pytest.approx(
        [-0.5, -0.25]
    )
    assert {row["split"] for row in rows} == {"train"}
    assert all(row["expert_planner_available_at_runtime"] is False for row in rows)


def test_duplicate_cumulative_live_failure_source_is_rejected() -> None:
    config = load_config(V6_CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    settings["additional_live_failure_dagger_augmentations"] = [
        dict(settings["live_failure_dagger_augmentation"])
    ]

    with pytest.raises(ValueError, match="SHA-256 values must be unique"):
        _live_failure_dagger_augmentations(settings)


def test_v7_appends_v6_failure_without_dropping_prior_sources() -> None:
    config = load_config(V7_CONFIG)
    settings, train_scenes, validation_scenes = _effective_settings(
        config, "operator"
    )
    augmentations = _live_failure_dagger_augmentations(settings)

    assert [item.report_sha256 for item in augmentations] == [
        "3ec5458d763cd7d27533e981d309fb2e4ea58da592f6d5f980f6a7cc80c5680f",
        "670fd0eb87f471e084ade548d8f9fa78c961aa926b4d79913bccdcd49ba0adc4",
        "a3ccd49891451d8e5863d90ef89718ed3489a6822abea725f3423dd43902ee97",
    ]
    assert [item.checkpoint_sha256 for item in augmentations] == [
        "b28642f2ccb3ef440ada4636566b5b5e237985ce50d9323196679780d626b2f1",
        "3f6785f082442c37ccb23db308e0f1e21b31dda28abe1c5f197d7be511baa199",
        "7366fced7601b726c7e0d03483446aef4e41bc7bd5bcad224176a3c2543f428a",
    ]
    assert [item.first_rejection_step for item in augmentations] == [22, 22, 47]
    assert train_scenes == ["scene_000001"]
    assert set(train_scenes).isdisjoint(validation_scenes)


def test_v8_appends_v7_recovery_and_divergence_with_prior_sources_intact() -> None:
    config = load_config(V8_CONFIG)
    settings, train_scenes, validation_scenes = _effective_settings(
        config, "operator"
    )
    sources = _live_failure_dagger_augmentations(settings)
    branches = _live_pre_divergence_dagger_augmentations(settings, sources)

    assert [item.report_sha256 for item in sources] == [
        "3ec5458d763cd7d27533e981d309fb2e4ea58da592f6d5f980f6a7cc80c5680f",
        "670fd0eb87f471e084ade548d8f9fa78c961aa926b4d79913bccdcd49ba0adc4",
        "a3ccd49891451d8e5863d90ef89718ed3489a6822abea725f3423dd43902ee97",
        "2c6e556de9459a8e2db4539781fe8e9b294cfc450cca7fde4f34865f930197db",
    ]
    assert [item.checkpoint_sha256 for item in sources] == [
        "b28642f2ccb3ef440ada4636566b5b5e237985ce50d9323196679780d626b2f1",
        "3f6785f082442c37ccb23db308e0f1e21b31dda28abe1c5f197d7be511baa199",
        "7366fced7601b726c7e0d03483446aef4e41bc7bd5bcad224176a3c2543f428a",
        "8d373c09d8e15edeffde154540b156abd33f72997715a0c3593e0b1f92d126cc",
    ]
    assert [(item.branch_id, item.correction_decision_step) for item in branches] == [
        ("pre_d45_move_wp25", 45),
        ("pre_d14_move_wp05", 14),
    ]
    assert branches[-1].expected_input_sha256 == (
        "c61251adb948d22ffa73c294b8eafd28d4ed0f33d027d88dabfa41a693b659d3"
    )
    assert train_scenes == ["scene_000001"]
    assert set(train_scenes).isdisjoint(validation_scenes)


def test_v7_exact_v6_state_gets_complete_safe_return_and_stop_labels() -> None:
    rows, metrics = _v7_failure_rows()

    assert len(rows) == metrics["continuation_sample_count"] == 32
    assert metrics["source_report_sha256"] == (
        "a3ccd49891451d8e5863d90ef89718ed3489a6822abea725f3423dd43902ee97"
    )
    assert metrics["source_checkpoint_sha256"] == (
        "7366fced7601b726c7e0d03483446aef4e41bc7bd5bcad224176a3c2543f428a"
    )
    assert metrics["source_scene_prefix_sha256"] == (
        "52c33298140845d341fa2b4568f2c6e960279495890e08455caafa7d5bbc9c95"
    )
    assert metrics["first_rejection_step"] == 47
    assert metrics["exact_prefix_transition_count"] == 47
    assert metrics["first_rejected_target_xy_m"] == pytest.approx(
        [0.23210880891888863, 0.15496964295076576]
    )
    assert metrics["first_recovery_input_sha256"] == (
        "3553c6e3f7939943bc80d84f49489223f8ca89c92d524200e3b2a48b2f4269f9"
    )
    assert metrics["total_decision_count"] == 79
    assert metrics["path_length_m"] == pytest.approx(18.6500994894)
    assert metrics["signed_winding_area_m2"] == pytest.approx(-4.2163834844)
    assert metrics["return_error_m"] == pytest.approx(0.0)
    assert metrics["minimum_padded_map_clearance_m"] > 0.0
    assert metrics["model_labeled_stop"] is True
    assert metrics["runtime_planner_available"] is False

    first = rows[0]
    assert first["family"] == "lap_live_failure_recovery"
    assert first["source_sample_sha256"] == metrics["source_report_sha256"]
    assert first["history_pose_xy_yaw"][-1] == pytest.approx(
        [-0.20516896210234745, 0.1940730574124565, -130.0002670288086]
    )
    assert len(first["history"]) == 16
    assert first["history"][-1][-1] == 0.0
    assert rows[-1]["expert_action"] == "STOP"
    assert rows[-1]["history_pose_xy_yaw"][-1][:2] == pytest.approx(
        [-0.5, -0.25]
    )
    assert {row["split"] for row in rows} == {"train"}
    assert all(row["expert_planner_available_at_runtime"] is False for row in rows)


def test_v7_pre_d45_branch_corrects_face_to_direct_move_before_collision() -> None:
    rows, metrics = _v7_divergence_rows()

    assert len(rows) == metrics["continuation_sample_count"] == 32
    assert metrics["branch_id"] == "pre_d45_move_wp25"
    assert metrics["source_report_sha256"] == (
        "a3ccd49891451d8e5863d90ef89718ed3489a6822abea725f3423dd43902ee97"
    )
    assert metrics["source_checkpoint_sha256"] == (
        "7366fced7601b726c7e0d03483446aef4e41bc7bd5bcad224176a3c2543f428a"
    )
    assert metrics["correction_decision_step"] == 45
    assert metrics["exact_prefix_transition_count"] == 44
    assert metrics["observed_model_action"] == "face"
    assert metrics["expert_first_action"] == "MOVE_TO"
    assert metrics["first_correction_input_sha256"] == (
        "6bdaef05a353bdfa6d1ed935cb4c0dbc6f4720200c525c67bc993ea11dd315ad"
    )
    assert metrics["total_decision_count"] == 76
    assert metrics["path_length_m"] == pytest.approx(18.6500994777)
    assert metrics["signed_winding_area_m2"] == pytest.approx(-4.2163834808)
    assert metrics["return_error_m"] == pytest.approx(0.0)
    assert metrics["minimum_padded_map_clearance_m"] > 0.0
    assert metrics["model_labeled_stop"] is True
    assert metrics["runtime_planner_available"] is False

    first = rows[0]
    assert first["expert_action"] == "MOVE_TO"
    assert first["expert_xy_m"] == pytest.approx([-0.2, -0.15])
    assert first["waypoint_delta_robot_m"] == pytest.approx(
        [-0.17651435787778125, 0.2953907047782081]
    )
    assert first["history_pose_xy_yaw"][-1] == pytest.approx(
        [-0.20516896210234745, 0.1940730574124565, 149.9997329711914]
    )
    assert rows[-1]["expert_action"] == "STOP"
    assert rows[-1]["history_pose_xy_yaw"][-1][:2] == pytest.approx(
        [-0.5, -0.25]
    )
    assert {row["split"] for row in rows} == {"train"}
    assert all(row["expert_planner_available_at_runtime"] is False for row in rows)


def test_v5_report_authentication_fails_before_relabeling(tmp_path: Path) -> None:
    config = load_config(V5_CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    augmentation = _live_failure_dagger_augmentation(settings)
    assert augmentation.report_path is not None
    copied = tmp_path / "reports" / "failure.json"
    copied.parent.mkdir()
    value = json.loads(augmentation.report_path.read_text(encoding="utf-8"))
    value["instruction"] = "tampered"
    copied.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 differs"):
        _load_authenticated_live_failure_report(
            replace(augmentation, report_path=copied)
        )


def test_v5_face_teacher_preserves_runtime_last_movement_delta() -> None:
    collision_map = NumericCollisionMap(
        np.asarray([[10.0, 10.0]], dtype=np.float32),
        room_min_xy_m=(-3.0, -2.5),
        room_max_xy_m=(3.0, 2.5),
        robot_radius_m=0.25,
        surface_padding_m=0.035,
    )
    builder = _SyntheticTraceBuilder(
        episode_id="runtime_state_fidelity",
        scene_id="scene_000001",
        split="train",
        family="lap",
        task_variant="lap_clockwise",
        instruction="Do a lap around the room.",
        start_xy=(-0.5, -0.25),
        initial_yaw=0.0,
        episode_goal_xy_m=(-0.5, -0.25),
        training_target_xyz_m=None,
        collision_map=collision_map,
        room_size_m=(6.0, 5.0, 3.0),
        max_waypoint_step_m=0.5,
        max_turn_degrees=40.0,
        history_length=16,
    )
    builder.move_to((-0.5, 0.0), fixed_face_step_degrees=40.0)
    movement = builder.pose.last_delta
    builder.face(builder.pose.yaw + 40.0)

    assert builder.pose.last_delta == movement
    assert builder.pose.linear_velocity == (0.0, 0.0)
    assert builder.pose.angular_velocity == pytest.approx(40.0)
    assert builder.rows[-1]["state_features"][13:16] == pytest.approx(
        [movement[0] / 6.0, movement[1] / 5.0, movement[2] / 3.0]
    )


def test_v4_canonical_dataset_is_authenticated_for_exact_input_reuse() -> None:
    config = load_config(V5_CONFIG)
    policy = config["gemma_waypoint_policy"]
    loaded = load_waypoint_dataset_for_hidden_reuse(
        V4_DATASET,
        state_dim=int(policy["state_dim"]),
        history_dim=int(policy["history_dim"]),
        max_history_tokens=int(policy["max_history_tokens"]),
        max_waypoint_step_m=float(policy["max_waypoint_step_m"]),
    )
    strict_manifest, strict_rows = load_gemma_waypoint_trace_dataset(V4_DATASET)

    assert loaded.sha256 == strict_manifest["dataset_sha256"]
    assert loaded.traces_sha256 == strict_manifest["traces_sha256"]
    assert len(loaded.samples) == len(strict_rows) == 7601


def test_generated_v5_dataset_is_strict_and_train_validation_isolated() -> None:
    manifest, rows = load_gemma_waypoint_trace_dataset(V5_DATASET)
    failure_rows = [
        row for row in rows if row["family"] == "lap_live_failure_recovery"
    ]

    assert manifest["dataset_sha256"] == (
        "3b32dd1541c9cf918a2ee4de178f41be18b1ec84857a411031cf16e6c8482c29"
    )
    assert manifest["traces_sha256"] == (
        "11214e6e099861a1e357e9414089ef7becb9704b91cc776e60ccca264d770284"
    )
    assert manifest["expert_surface_padding_m"] == pytest.approx(0.085)
    assert manifest["runtime_surface_padding_m"] == pytest.approx(0.035)
    assert manifest["minimum_runtime_clearance_margin_m"] == pytest.approx(0.05)
    assert len(failure_rows) == manifest["live_failure_dagger_sample_count"] == 63
    assert {row["split"] for row in failure_rows} == {"train"}
    assert not any(
        row["family"] == "lap_live_failure_recovery"
        and row["scene_id"] in set(manifest["validation_scene_ids"])
        for row in rows
    )


def test_generated_v6_dataset_authenticates_both_cumulative_corrections() -> None:
    manifest, rows = load_gemma_waypoint_trace_dataset(V6_DATASET)
    source_report_sha256 = [
        "3ec5458d763cd7d27533e981d309fb2e4ea58da592f6d5f980f6a7cc80c5680f",
        "670fd0eb87f471e084ade548d8f9fa78c961aa926b4d79913bccdcd49ba0adc4",
    ]
    failure_rows = [
        row for row in rows if row["family"] == "lap_live_failure_recovery"
    ]
    rows_by_source = {
        digest: [
            row
            for row in failure_rows
            if row["source_sample_sha256"] == digest
        ]
        for digest in source_report_sha256
    }

    assert manifest["dataset_sha256"] == (
        "68ca090b9ea87ed9b65b515230fa9826d8f7d83dc2bb83eaf4d13754688d42a9"
    )
    assert manifest["traces_sha256"] == (
        "212c4a5c1fe28b7c0a2a6dbbc1261842861529358f88a98675366ffbb7b0ac8c"
    )
    assert manifest["sample_count"] == len(rows) == 7334
    assert manifest["live_failure_dagger_cumulative_sources"] is True
    assert manifest["live_failure_dagger_source_report_sha256"] == (
        source_report_sha256
    )
    assert manifest["live_failure_dagger_all_sources_retained"] is True
    assert manifest["live_failure_dagger_sample_count"] == 123
    assert manifest["live_failure_dagger_episode_count"] == 2
    assert len(rows_by_source[source_report_sha256[0]]) == 63
    assert len(rows_by_source[source_report_sha256[1]]) == 60
    assert set(manifest["live_failure_dagger_metrics_by_report_sha256"]) == set(
        source_report_sha256
    )
    assert all(
        source_rows[-1]["expert_action"] == "STOP"
        and source_rows[-1]["history_pose_xy_yaw"][-1][:2]
        == pytest.approx([-0.5, -0.25])
        for source_rows in rows_by_source.values()
    )
    assert {row["split"] for row in failure_rows} == {"train"}
    assert not any(
        row["scene_id"] in set(manifest["validation_scene_ids"])
        for row in failure_rows
    )
    assert set(manifest["train_scene_ids"]).isdisjoint(
        manifest["validation_scene_ids"]
    )


def test_generated_v7_dataset_separates_recovery_and_pre_divergence_branches() -> None:
    manifest, rows = load_gemma_waypoint_trace_dataset(V7_DATASET)
    source_report_sha256 = [
        "3ec5458d763cd7d27533e981d309fb2e4ea58da592f6d5f980f6a7cc80c5680f",
        "670fd0eb87f471e084ade548d8f9fa78c961aa926b4d79913bccdcd49ba0adc4",
        "a3ccd49891451d8e5863d90ef89718ed3489a6822abea725f3423dd43902ee97",
    ]
    branch_key = f"{source_report_sha256[2]}:pre_d45_move_wp25"
    failure_rows = [
        row for row in rows if row["family"] == "lap_live_failure_recovery"
    ]
    divergence_rows = [
        row
        for row in rows
        if row["family"] == "lap_live_divergence_correction"
    ]

    assert manifest["dataset_sha256"] == (
        "3ab3cb1a7239a121cfc4f2f85144dc922f578ec9b493a6053f2fcc260f518cb2"
    )
    assert manifest["traces_sha256"] == (
        "6abb5c4881acd79faff297c9536e5caf5bac20db52fc41c037c26df33ead0fe6"
    )
    assert manifest["sample_count"] == len(rows) == 7398
    assert manifest["live_failure_dagger_source_report_sha256"] == (
        source_report_sha256
    )
    assert manifest["live_failure_dagger_all_sources_retained"] is True
    assert manifest["live_failure_dagger_sample_count"] == len(failure_rows) == 155
    assert manifest["live_failure_dagger_episode_count"] == 3
    assert manifest["live_divergence_dagger_augmentation_enabled"] is True
    assert manifest["live_divergence_dagger_sample_count"] == (
        len(divergence_rows)
    ) == 32
    assert manifest["live_divergence_dagger_episode_count"] == 1
    assert set(manifest["live_divergence_dagger_metrics_by_branch_id"]) == {
        branch_key
    }
    assert (
        manifest["live_divergence_dagger_metrics_by_branch_id"][branch_key][
            "first_correction_input_sha256"
        ]
        == "6bdaef05a353bdfa6d1ed935cb4c0dbc6f4720200c525c67bc993ea11dd315ad"
    )
    assert sum(
        row["source_sample_sha256"] == source_report_sha256[0]
        for row in failure_rows
    ) == 63
    assert sum(
        row["source_sample_sha256"] == source_report_sha256[1]
        for row in failure_rows
    ) == 60
    assert sum(
        row["source_sample_sha256"] == source_report_sha256[2]
        for row in failure_rows
    ) == 32
    assert {row["source_sample_sha256"] for row in divergence_rows} == {
        source_report_sha256[2]
    }
    assert divergence_rows[0]["expert_action"] == "MOVE_TO"
    assert divergence_rows[-1]["expert_action"] == "STOP"
    assert {row["split"] for row in [*failure_rows, *divergence_rows]} == {
        "train"
    }
    assert not any(
        row["scene_id"] in set(manifest["validation_scene_ids"])
        for row in [*failure_rows, *divergence_rows]
    )
    assert set(manifest["train_scene_ids"]).isdisjoint(
        manifest["validation_scene_ids"]
    )


def test_generated_v8_dataset_authenticates_early_growing_history_branches() -> None:
    manifest, rows = load_gemma_waypoint_trace_dataset(V8_DATASET)
    source_sha = (
        "2c6e556de9459a8e2db4539781fe8e9b294cfc450cca7fde4f34865f930197db"
    )
    branch_key = f"{source_sha}:pre_d14_move_wp05"
    post_rows = [
        row
        for row in rows
        if row["family"] == "lap_live_failure_recovery"
        and row["source_sample_sha256"] == source_sha
    ]
    pre_rows = [
        row
        for row in rows
        if row["family"] == "lap_live_divergence_correction"
        and row["source_sample_sha256"] == source_sha
    ]

    assert manifest["dataset_sha256"] == (
        "bfecb2af533cc380dd87b76273568612b572042d840bff3d3be57e573dc7b092"
    )
    assert manifest["traces_sha256"] == (
        "419cb9eb8727b1ea6719ff47b86e2af3255572e48eabe62fb0d84b4897cf7dbe"
    )
    assert manifest["sample_count"] == len(rows) == 7525
    assert manifest["live_failure_dagger_sample_count"] == 219
    assert manifest["live_failure_dagger_episode_count"] == 4
    assert manifest["live_divergence_dagger_sample_count"] == 95
    assert manifest["live_divergence_dagger_episode_count"] == 2
    assert manifest["live_failure_dagger_all_sources_retained"] is True
    assert branch_key in manifest["live_divergence_dagger_metrics_by_branch_id"]
    assert len(post_rows) == 64
    assert len(pre_rows) == 63
    assert len(post_rows[0]["history"]) == 15
    assert post_rows[0]["history"][-1][-1] == 0.0
    assert len(pre_rows[0]["history"]) == 13
    assert pre_rows[0]["history"][-1][-1] == 1.0
    assert post_rows[-1]["expert_action"] == "STOP"
    assert pre_rows[-1]["expert_action"] == "STOP"
    assert post_rows[-1]["history_pose_xy_yaw"][-1][:2] == pytest.approx(
        [-0.5, -0.25]
    )
    assert pre_rows[-1]["history_pose_xy_yaw"][-1][:2] == pytest.approx(
        [-0.5, -0.25]
    )
    assert {row["split"] for row in [*post_rows, *pre_rows]} == {"train"}
    assert set(manifest["train_scene_ids"]).isdisjoint(
        manifest["validation_scene_ids"]
    )
