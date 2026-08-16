"""Contracts for the unseen-room closed-loop harness.

These tests never load Gemma.  They cover the two properties that make the
held-out measurement meaningful:

* the task file handed to the rollout process carries no target geometry, so a
  rollout cannot be scored by anything it was itself given, and
* a goal the model never terminated is scored as a failure no matter where the
  rover happened to stop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v15_heldout_closed_loop import (
    PLAN_SCHEMA,
    ROLLOUT_SCHEMA,
    SCORE_SCHEMA,
    TARGET_SCHEMA,
    plan_heldout_tasks,
    score_heldout_rollouts,
)

ROOT = Path(__file__).resolve().parents[1]
ORACLE_ROOT = ROOT / "data" / "oracle"
SEALED = ("scene_000051", "scene_000052")

pytestmark = pytest.mark.skipif(
    not (ORACLE_ROOT / SEALED[0] / "oracle.json").is_file(),
    reason="Sealed held-out oracle scenes are not materialized in this checkout",
)


def _plan():
    return plan_heldout_tasks(
        SEALED, oracle_root=ORACLE_ROOT, object_goals_per_scene=2, include_lap=True
    )


def test_plan_splits_instructions_from_geometry() -> None:
    tasks, targets = _plan()
    assert tasks["schema"] == PLAN_SCHEMA
    assert targets["schema"] == TARGET_SCHEMA
    assert tasks["contains_target_geometry"] is False
    # Two object goals per scene produce a face and an approach goal each, plus
    # one lap goal per scene.
    assert tasks["task_count"] == len(SEALED) * (2 * 2 + 1)
    assert {row["goal_id"] for row in tasks["tasks"]} == {
        row["goal_id"] for row in targets["targets"]
    }

    serialized = json.dumps(tasks)
    for target in targets["targets"]:
        for key in (
            "target_center_xyz_m",
            "target_bbox_min_xyz_m",
            "target_bbox_max_xyz_m",
        ):
            assert key not in serialized
            if key in target:
                assert len(target[key]) == 3
    for task in tasks["tasks"]:
        assert set(task) == {"goal_id", "scene_id", "instruction", "max_steps"}
        assert task["scene_id"] in SEALED
        assert task["instruction"].strip() == task["instruction"]


def test_plan_only_names_unambiguous_objects() -> None:
    _, targets = _plan()
    for target in targets["targets"]:
        if target["metric"] == "lap_circuit":
            continue
        oracle = json.loads(
            (ORACLE_ROOT / target["scene_id"] / "oracle.json").read_text(
                encoding="utf-8"
            )
        )
        matches = [
            item
            for item in oracle["instances"]
            if item.get("category") == target["target_category"]
        ]
        assert len(matches) == 1, target["target_category"]
        assert target["target_category"] not in {"floor", "wall", "ceiling", "door"}


def _rollout_report(**overrides):
    base = {
        "goal_id": "scene_000051_face_chair",
        "scene_id": "scene_000051",
        "instruction": "Face the chair, then stop.",
        "max_steps": 32,
        "termination": "model_stop",
        "error_code": None,
        "model_stop_emitted": True,
        "controller_success": True,
        "decision_count": 4,
        "accepted_decision_count": 4,
        "rejected_decision_count": 0,
        "action_counts": {"face": 3, "stop": 1},
        "start_pose_xy_yaw": [0.0, 0.0, 0.0],
        "final_pose_xy_yaw": [0.0, 0.0, 0.0],
        "elapsed_seconds": 1.0,
        "checkpoint_sha256": "0" * 64,
        "scene_prefix_sha256": "1" * 64,
        "path_length_m": 0.0,
        "signed_swept_area_m2": 0.0,
        "absolute_swept_area_m2": 0.0,
        "return_distance_m": 0.0,
    }
    base.update(overrides)
    return {
        "schema": ROLLOUT_SCHEMA,
        "rollout_process_read_oracle": False,
        "oracle_reads_blocked_by_audit": True,
        "renderer_used": False,
        "deterministic_route_planner_used": False,
        "model_selected_every_action": True,
        "navigation_checkpoint": "data_gemma4/checkpoints/example",
        "scene_identities": {},
        "elapsed_seconds": 1.0,
        "rollout_count": 1,
        "rollouts": [base],
    }


def _targets(metric: str = "face_yaw"):
    return {
        "schema": TARGET_SCHEMA,
        "scene_ids": ["scene_000051"],
        "oracle_root": "data/oracle",
        "oracle_file_sha256": {"scene_000051": "2" * 64},
        "targets": [
            {
                "goal_id": "scene_000051_face_chair",
                "scene_id": "scene_000051",
                "metric": metric,
                "target_category": "chair",
                "target_instance_id": "i_000101",
                # Directly ahead: project convention is yaw 0 faces +Y.
                "target_center_xyz_m": [0.0, 1.5, 0.5],
                "target_bbox_min_xyz_m": [-0.2, 1.3, 0.0],
                "target_bbox_max_xyz_m": [0.2, 1.7, 1.0],
            }
        ],
    }


def test_face_goal_passes_only_when_aimed_at_the_object() -> None:
    aimed = score_heldout_rollouts(_rollout_report(), _targets())
    assert aimed["schema"] == SCORE_SCHEMA
    assert aimed["goals"][0]["oracle_yaw_error_degrees"] == pytest.approx(0.0)
    assert aimed["goals"][0]["passed"] is True
    assert aimed["pass_rate"] == pytest.approx(1.0)

    turned = score_heldout_rollouts(
        _rollout_report(final_pose_xy_yaw=[0.0, 0.0, 90.0]), _targets()
    )
    assert turned["goals"][0]["oracle_yaw_error_degrees"] == pytest.approx(90.0)
    assert turned["goals"][0]["passed"] is False


def test_a_goal_the_model_never_terminated_cannot_pass() -> None:
    """Stopping in the right place by running out of steps is not success."""

    report = score_heldout_rollouts(
        _rollout_report(
            termination="max_steps",
            model_stop_emitted=False,
            controller_success=False,
            error_code="E_MAX_STEPS",
        ),
        _targets(),
    )
    row = report["goals"][0]
    assert row["checks"]["maximum_oracle_yaw_error"] is True
    assert row["checks"]["model_selected_terminal_stop"] is False
    assert row["passed"] is False
    assert report["model_selected_terminal_stop_rate"] == pytest.approx(0.0)


def test_approach_goal_requires_progress_and_standoff() -> None:
    close = score_heldout_rollouts(
        _rollout_report(
            start_pose_xy_yaw=[0.0, -1.0, 0.0], final_pose_xy_yaw=[0.0, 0.9, 0.0]
        ),
        _targets("approach_standoff"),
    )
    row = close["goals"][0]
    assert row["checks"]["minimum_oracle_center_progress"] is True
    assert row["final_oracle_bbox_standoff_m"] == pytest.approx(0.4)
    assert row["passed"] is True

    far = score_heldout_rollouts(
        _rollout_report(
            start_pose_xy_yaw=[0.0, -1.0, 0.0], final_pose_xy_yaw=[0.0, -1.05, 0.0]
        ),
        _targets("approach_standoff"),
    )
    assert far["goals"][0]["checks"]["minimum_oracle_center_progress"] is False
    assert far["goals"][0]["passed"] is False


def test_lap_goal_requires_a_real_circuit() -> None:
    circuit = score_heldout_rollouts(
        _rollout_report(
            goal_id="scene_000051_lap",
            path_length_m=12.0,
            absolute_swept_area_m2=4.0,
            return_distance_m=0.1,
        ),
        {
            **_targets(),
            "targets": [
                {
                    "goal_id": "scene_000051_lap",
                    "scene_id": "scene_000051",
                    "metric": "lap_circuit",
                }
            ],
        },
    )
    assert circuit["goals"][0]["passed"] is True

    shuffle = score_heldout_rollouts(
        _rollout_report(
            goal_id="scene_000051_lap",
            path_length_m=12.0,
            absolute_swept_area_m2=0.01,
            return_distance_m=0.1,
        ),
        {
            **_targets(),
            "targets": [
                {
                    "goal_id": "scene_000051_lap",
                    "scene_id": "scene_000051",
                    "metric": "lap_circuit",
                }
            ],
        },
    )
    # Long path, no enclosed area: pacing back and forth is not a lap.
    assert shuffle["goals"][0]["checks"]["minimum_absolute_swept_area"] is False
    assert shuffle["goals"][0]["passed"] is False


def test_rollout_refuses_a_plan_carrying_geometry() -> None:
    from semantic_3d_chat.evaluation.v15_heldout_closed_loop import run_heldout_rollouts

    tasks, _ = _plan()
    leaky = {**tasks, "contains_target_geometry": True}
    with pytest.raises(ValueError, match="target geometry"):
        run_heldout_rollouts(leaky, navigation_checkpoint="unused")
