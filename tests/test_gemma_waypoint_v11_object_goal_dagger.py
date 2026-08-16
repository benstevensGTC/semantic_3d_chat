from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.planner import NumericWaypointPlanner
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
    _effective_settings,
    _HistoryEncoding,
    _live_failure_dagger_augmentations,
    _live_object_goal_dagger_augmentation,
    _live_object_goal_dagger_rows,
    _live_pre_divergence_dagger_augmentations,
    _load_authenticated_live_object_goal_reports,
    _load_training_oracle_object_geometry,
    load_gemma_waypoint_trace_dataset,
)

CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v11.yaml"
DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v2_operator_dagger_v11"
)
RUNTIME_SHA256 = (
    "a42a7445e1cdfa424b184f1a5958db197b5f7dcc1a0f15ed5ff4dffe521e71e6"
)
SCORE_SHA256 = (
    "51b6ebd72849343094b39f33377a8d9a705223c7f49d5a3ee42b3a06e550b3d7"
)
D2_INPUT_SHA256 = (
    "d5f4822df96b50edb3727198119cb12d1a1c34b91ccb7ed73ab7e892b5774bb6"
)


@dataclass(frozen=True)
class _Audit:
    config: dict[str, Any]
    settings: dict[str, Any]
    augmentation: Any
    collision_map: NumericCollisionMap
    planner: NumericWaypointPlanner
    encoding: _HistoryEncoding
    target: Any
    bbox_min: Any
    bbox_max: Any


@pytest.fixture(scope="module")
def audit() -> _Audit:
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    augmentation = _live_object_goal_dagger_augmentation(settings)
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
    target, bbox_min, bbox_max = _load_training_oracle_object_geometry(
        PROJECT_ROOT / "data/oracle/scene_000001/oracle.json",
        scene_id="scene_000001",
        category="chair",
        instance_id="i_000101",
    )
    return _Audit(
        config=config,
        settings=settings,
        augmentation=augmentation,
        collision_map=collision_map,
        planner=NumericWaypointPlanner(
            collision_map,
            grid_resolution_m=float(settings["planner_grid_resolution_m"]),
            standoff_m=float(settings["approach_standoff_m"]),
            standoff_tolerance_m=float(
                settings["approach_standoff_tolerance_m"]
            ),
            max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
            angular_samples=int(settings["approach_angular_samples"]),
        ),
        encoding=_HistoryEncoding.from_settings(
            settings, history_length=int(settings["history_length"])
        ),
        target=target,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def _rows(audit: _Audit) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _live_object_goal_dagger_rows(
        augmentation=audit.augmentation,
        target_xyz_m=audit.target,
        target_bbox_min_xy_m=audit.bbox_min,
        target_bbox_max_xy_m=audit.bbox_max,
        collision_map=audit.collision_map,
        approach_planner=audit.planner,
        room_size_m=audit.config["scene"]["room_size_m"],
        max_waypoint_step_m=float(audit.settings["max_waypoint_step_m"]),
        max_turn_degrees=(
            float(audit.config["robot"]["max_turn_degrees"])
            - float(audit.settings["expert_turn_margin_degrees"])
        ),
        history_length=int(audit.settings["history_length"]),
        history_encoding=audit.encoding,
    )


def test_v11_preserves_v10_lap_sources_and_regularized_refit(audit: _Audit) -> None:
    sources = _live_failure_dagger_augmentations(audit.settings)
    branches = _live_pre_divergence_dagger_augmentations(audit.settings, sources)
    assert len(sources) == 6
    assert len(branches) == 6
    assert audit.augmentation.runtime_report_sha256 == RUNTIME_SHA256
    assert audit.augmentation.score_report_sha256 == SCORE_SHA256
    assert audit.augmentation.correction_decision_step == 2
    assert audit.augmentation.observed_model_action == "move_to"
    assert audit.augmentation.expected_expert_first_action == "FACE"
    policy = audit.config["gemma_waypoint_policy"]
    assert policy["history_dim"] == HISTORY_FEATURE_DIM_V2
    assert policy["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert policy["action_refit_max_iter"] == 500
    assert policy["action_refit_learning_rate"] == pytest.approx(0.5)
    assert policy["action_refit_l2_weight"] == pytest.approx(0.00001)
    assert policy["minimum_training_action_accuracy"] == pytest.approx(0.995)


def test_live_object_goal_parser_is_fail_closed() -> None:
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    for mutation in (
        {"correction_decision_step": 0},
        {"observed_model_action": "face"},
        {"runtime_report_sha256": "0" * 63},
        {"target_instance_id": "chair"},
    ):
        invalid = copy.deepcopy(settings)
        invalid["live_object_goal_dagger_augmentation"].update(mutation)
        with pytest.raises(ValueError, match="object-goal DAgger.*invalid"):
            _live_object_goal_dagger_augmentation(invalid)


def test_runtime_must_authenticate_before_score_is_opened(
    audit: _Audit, tmp_path: Path
) -> None:
    runtime, score, _ = _load_authenticated_live_object_goal_reports(
        audit.augmentation
    )
    tampered_runtime = copy.deepcopy(runtime)
    tampered_runtime["fallback_used"] = True
    runtime_bytes = json.dumps(tampered_runtime, sort_keys=True).encode("utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_bytes(runtime_bytes)
    source = replace(
        audit.augmentation,
        runtime_report_path=runtime_path,
        runtime_report_sha256=hashlib.sha256(runtime_bytes).hexdigest(),
        score_report_path=tmp_path / "missing_score.json",
    )
    with pytest.raises(ValueError, match="runtime record contract"):
        _load_authenticated_live_object_goal_reports(source)

    tampered_score = copy.deepcopy(score)
    tampered_score["runtime_process_read_oracle"] = True
    score_bytes = json.dumps(tampered_score, sort_keys=True).encode("utf-8")
    score_path = tmp_path / "score.json"
    score_path.write_bytes(score_bytes)
    source = replace(
        audit.augmentation,
        score_report_path=score_path,
        score_report_sha256=hashlib.sha256(score_bytes).hexdigest(),
    )
    with pytest.raises(ValueError, match="score provenance differs"):
        _load_authenticated_live_object_goal_reports(source)


def test_v10_d1_is_close_and_d2_is_exact_first_semantic_divergence(
    audit: _Audit,
) -> None:
    runtime, _, _ = _load_authenticated_live_object_goal_reports(
        audit.augmentation
    )
    decisions = runtime["model_decisions"]
    assert decisions[0]["model_action"] == "face"
    assert decisions[0]["model_desired_heading_degrees"] == pytest.approx(
        -52.29376220703125
    )
    assert abs(decisions[0]["model_desired_heading_degrees"] - (-50.0)) < 2.30
    assert decisions[1]["model_action"] == "move_to"
    rows, metrics = _rows(audit)
    assert rows[0]["expert_action"] == "FACE"
    assert rows[0]["expert_heading_degrees"] == pytest.approx(
        -12.29376220703125
    )
    assert metrics["correction_decision_step"] == 2
    assert metrics["first_correction_input_sha256"] == D2_INPUT_SHA256


def test_v11_exact_d2_teacher_reaches_chair_safely_and_model_stops(
    audit: _Audit,
) -> None:
    rows, metrics = _rows(audit)
    assert len(rows) == 7
    assert [row["expert_action"] for row in rows] == [
        "FACE",
        "FACE",
        "FACE",
        "FACE",
        "MOVE_TO",
        "MOVE_TO",
        "STOP",
    ]
    assert rows[4]["expert_xy_m"] == pytest.approx(
        [-0.8366545954537721, -0.16154848743518224]
    )
    assert rows[5]["expert_xy_m"] == pytest.approx(
        [-1.1733091909075442, -0.07309697487036448]
    )
    assert metrics["total_decision_count"] == 8
    assert metrics["path_length_m"] == pytest.approx(0.6961608627756865)
    assert metrics["target_center_progress_m"] == pytest.approx(
        0.46865639584570096
    )
    assert metrics["final_target_center_distance_m"] == pytest.approx(0.6)
    assert metrics["final_oracle_bbox_standoff_m"] == pytest.approx(
        0.3253162048069642
    )
    assert metrics["minimum_padded_map_clearance_m"] > 0.0
    assert metrics["runtime_planner_available"] is False
    assert metrics["runtime_oracle_available"] is False
    assert rows[-1]["expert_action"] == "STOP"


def test_v11_generated_dataset_retains_v10_and_adds_one_object_branch() -> None:
    manifest, rows = load_gemma_waypoint_trace_dataset(PROJECT_ROOT / DATASET)
    assert manifest["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert manifest["history_feature_dim"] == HISTORY_FEATURE_DIM_V2
    assert manifest["contradictory_exact_input_count"] == 0
    assert manifest["live_failure_dagger_episode_count"] == 6
    assert manifest["live_divergence_dagger_episode_count"] == 6
    assert manifest["live_object_goal_dagger_episode_count"] == 1
    assert manifest["live_object_goal_dagger_sample_count"] == 7
    assert manifest["runtime_recovery_planner_available"] is False
    assert manifest["runtime_preprogrammed_lap_function"] is False
    assert len(rows) == manifest["sample_count"]
    correction_rows = [
        row
        for row in rows
        if row["family"] == "object_goal_live_divergence_correction"
    ]
    assert len(correction_rows) == 7
    assert all(row["split"] == "train" for row in correction_rows)
    assert correction_rows[-1]["expert_action"] == "STOP"
