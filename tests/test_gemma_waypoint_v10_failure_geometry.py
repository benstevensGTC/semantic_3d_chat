from __future__ import annotations

import copy
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
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
    _effective_settings,
    _HistoryEncoding,
    _lap_routes,
    _live_failure_dagger_augmentations,
    _live_failure_dagger_rows,
    _live_pre_divergence_dagger_augmentations,
    _live_pre_divergence_dagger_rows,
    _load_authenticated_live_failure_report,
    load_gemma_waypoint_trace_dataset,
)

CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v10.yaml"
DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v2_operator_dagger_v10"
)
V9_REPORT_SHA256 = (
    "ee4fb2f6e75f90a4472a14b5a324d7cd7acca22a0db29799699ce6a5bafb7b43"
)
D18_INPUT_SHA256 = (
    "cb874cabec503368aa48799713746e2f75d90ad5ddfd898e4e362104f6214199"
)
D29_RECOVERY_INPUT_SHA256 = (
    "71c2a99ee815e477e748ddb21998b191722c7e4d334a1c0f08baaca47030bff8"
)
D107_INPUT_SHA256 = (
    "4f2a2529173429322fac3bcd20a153a596068444e8ce32a8e8a9ccdc6a84b93c"
)


@dataclass(frozen=True)
class _Audit:
    config: dict[str, Any]
    settings: dict[str, Any]
    collision_map: NumericCollisionMap
    planner: NumericWaypointPlanner
    route: tuple[tuple[float, float], ...]
    encoding: _HistoryEncoding
    source: Any
    branches: tuple[Any, ...]


@pytest.fixture(scope="module")
def audit() -> _Audit:
    config = load_config(CONFIG)
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
    sources = _live_failure_dagger_augmentations(settings)
    source = next(
        value for value in sources if value.report_sha256 == V9_REPORT_SHA256
    )
    branches = tuple(
        value
        for value in _live_pre_divergence_dagger_augmentations(settings, sources)
        if value.source.report_sha256 == V9_REPORT_SHA256
    )
    return _Audit(
        config=config,
        settings=settings,
        collision_map=collision_map,
        planner=NumericWaypointPlanner(
            collision_map,
            grid_resolution_m=float(settings["planner_grid_resolution_m"]),
            standoff_m=max(0.20, collision_map.inflated_radius_m),
            standoff_tolerance_m=0.05,
            max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        ),
        route=tuple(
            _lap_routes(np.asarray([-0.5, -0.25]), patrol)["clockwise"]
        ),
        encoding=_HistoryEncoding.from_settings(
            settings,
            history_length=int(settings["history_length"]),
        ),
        source=source,
        branches=branches,
    )


def _failure_rows(audit: _Audit) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _live_failure_dagger_rows(
        augmentation=audit.source,
        route_waypoints=audit.route,
        collision_map=audit.collision_map,
        recovery_planner=audit.planner,
        room_size_m=audit.config["scene"]["room_size_m"],
        max_waypoint_step_m=float(audit.settings["max_waypoint_step_m"]),
        fixed_face_step_degrees=float(
            audit.settings["lap_fixed_face_step_degrees"]
        ),
        history_length=int(audit.settings["history_length"]),
        history_encoding=audit.encoding,
    )


def _divergence_rows(
    audit: _Audit, branch_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    branch = next(value for value in audit.branches if value.branch_id == branch_id)
    return _live_pre_divergence_dagger_rows(
        augmentation=branch,
        route_waypoints=audit.route,
        collision_map=audit.collision_map,
        recovery_planner=audit.planner,
        room_size_m=audit.config["scene"]["room_size_m"],
        max_waypoint_step_m=float(audit.settings["max_waypoint_step_m"]),
        fixed_face_step_degrees=float(
            audit.settings["lap_fixed_face_step_degrees"]
        ),
        history_length=int(audit.settings["history_length"]),
        history_encoding=audit.encoding,
    )


def test_v10_preserves_v4_through_v8_and_adds_both_v9_branches(
    audit: _Audit,
) -> None:
    sources = _live_failure_dagger_augmentations(audit.settings)
    all_branches = _live_pre_divergence_dagger_augmentations(
        audit.settings, sources
    )
    assert len(sources) == 6
    assert len({source.report_sha256 for source in sources}) == 6
    assert [source.report_sha256 for source in sources][-1] == V9_REPORT_SHA256
    assert {branch.branch_id for branch in audit.branches} == {
        "pre_d18_forced_face_wp08",
        "pre_d107_face_recover_wp33",
    }
    assert len(all_branches) == 6
    d18 = next(branch for branch in audit.branches if "d18" in branch.branch_id)
    d107 = next(branch for branch in audit.branches if "d107" in branch.branch_id)
    assert d18.force_first_face is True
    assert d18.recovery_plan_to_resume is False
    assert d107.force_first_face is False
    assert d107.recovery_plan_to_resume is True
    policy = audit.config["gemma_waypoint_policy"]
    assert policy["history_dim"] == HISTORY_FEATURE_DIM_V2
    assert policy["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert policy["action_refit_max_iter"] == 500
    assert policy["action_refit_learning_rate"] == pytest.approx(0.5)
    assert policy["action_refit_l2_weight"] == pytest.approx(0.00001)
    assert policy["minimum_training_action_accuracy"] == pytest.approx(0.995)


def test_force_first_face_parser_is_fail_closed() -> None:
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    sources = _live_failure_dagger_augmentations(settings)
    for mutation in (
        {"expected_expert_first_action": "MOVE_TO"},
        {"recovery_plan_to_resume": True},
        {"force_first_face": 1},
    ):
        invalid = copy.deepcopy(settings)
        branch = invalid["additional_live_failure_dagger_augmentations"][-1][
            "pre_divergence_dagger_augmentations"
        ][0]
        branch.update(mutation)
        with pytest.raises(ValueError, match="divergence DAgger contract"):
            _live_pre_divergence_dagger_augmentations(invalid, sources)


def test_v9_no_stop_report_requires_all_numeric_prohibitions(
    audit: _Audit, tmp_path: Path
) -> None:
    report = _load_authenticated_live_failure_report(audit.source)
    assert report["decision_authentication_failure"] == (
        "AssertionError: Gemma did not select and execute the terminal STOP"
    )
    tampered = copy.deepcopy(report)
    tampered["runtime_snapshot"]["control"]["fallback_used"] = True
    path = tmp_path / "v9_tampered.json"
    encoded = json.dumps(tampered, sort_keys=True).encode("utf-8")
    path.write_bytes(encoded)
    source = replace(
        audit.source,
        report_path=path,
        report_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    with pytest.raises(ValueError, match="provenance differs"):
        _load_authenticated_live_failure_report(source)


def test_v9_d29_rejection_has_exact_safe_clockwise_recovery(
    audit: _Audit,
) -> None:
    rows, metrics = _failure_rows(audit)
    assert len(rows) == 49
    assert metrics["first_rejection_step"] == 29
    assert metrics["first_rejected_target_xy_m"] == pytest.approx(
        [-0.15690739294684342, 2.164268335815707]
    )
    assert metrics["first_recovery_input_sha256"] == D29_RECOVERY_INPUT_SHA256
    assert metrics["total_decision_count"] == 78
    assert metrics["path_length_m"] == pytest.approx(18.929776744438126)
    assert metrics["signed_winding_area_m2"] == pytest.approx(
        -4.345836351279321
    )
    assert metrics["return_error_m"] == pytest.approx(0.0)
    assert metrics["minimum_padded_map_clearance_m"] > 0.0
    assert rows[0]["expert_action"] == "FACE"
    assert rows[0]["expert_heading_degrees"] == pytest.approx(
        -90.00100708007812
    )
    assert rows[-1]["expert_action"] == "STOP"


def test_v9_d18_forces_one_fixed_face_then_safe_continuation(
    audit: _Audit,
) -> None:
    rows, metrics = _divergence_rows(audit, "pre_d18_forced_face_wp08")
    assert len(rows) == 59
    assert metrics["first_correction_input_sha256"] == D18_INPUT_SHA256
    assert metrics["offline_forced_first_face"] is True
    assert metrics["offline_recovery_plan_to_resume"] is False
    assert metrics["total_decision_count"] == 76
    assert metrics["path_length_m"] == pytest.approx(18.77240013501142)
    assert metrics["signed_winding_area_m2"] == pytest.approx(
        -4.221185663635645
    )
    assert metrics["return_error_m"] == pytest.approx(0.0)
    assert rows[0]["expert_action"] == "FACE"
    assert rows[0]["expert_heading_degrees"] == pytest.approx(
        -10.002151489257812
    )
    assert rows[1]["expert_action"] == "MOVE_TO"
    assert rows[1]["expert_xy_m"] == pytest.approx(audit.route[8])
    assert rows[-1]["expert_action"] == "STOP"


def test_v9_d107_branch_returns_and_stops_inside_decision_budget(
    audit: _Audit,
) -> None:
    rows, metrics = _divergence_rows(audit, "pre_d107_face_recover_wp33")
    assert len(rows) == 20
    assert metrics["first_correction_input_sha256"] == D107_INPUT_SHA256
    assert metrics["offline_forced_first_face"] is False
    assert metrics["offline_recovery_plan_to_resume"] is True
    assert metrics["total_decision_count"] == 126
    assert metrics["path_length_m"] == pytest.approx(17.360472516223528)
    assert metrics["signed_winding_area_m2"] == pytest.approx(
        -8.129577871767038
    )
    assert metrics["return_error_m"] == pytest.approx(0.0)
    assert metrics["minimum_padded_map_clearance_m"] > 0.0
    assert rows[0]["expert_action"] == "FACE"
    assert rows[0]["expert_heading_degrees"] == pytest.approx(
        127.96941137313843
    )
    assert rows[-1]["expert_action"] == "STOP"
    assert rows[-1]["history_pose_xy_yaw"][-1][:2] == pytest.approx(
        [-0.5, -0.25]
    )


def test_v10_generated_dataset_has_no_exact_input_contradictions() -> None:
    manifest, rows = load_gemma_waypoint_trace_dataset(PROJECT_ROOT / DATASET)
    assert manifest["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert manifest["history_feature_dim"] == HISTORY_FEATURE_DIM_V2
    assert manifest["contradictory_exact_input_count"] == 0
    assert manifest["live_failure_dagger_episode_count"] == 6
    assert manifest["live_divergence_dagger_episode_count"] == 6
    assert manifest["runtime_recovery_planner_available"] is False
    assert manifest["runtime_preprogrammed_lap_function"] is False
    assert len(rows) == manifest["sample_count"]
    assert all(
        len(history_row) == HISTORY_FEATURE_DIM_V2
        for row in rows
        for history_row in row["history"]
    )
