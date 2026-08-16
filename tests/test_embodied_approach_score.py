from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.embodied_approach_score import score_approach_results


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _runtime(*, scene_id: str = "scene_000001", valid: bool = True) -> dict:
    approach = {
        "terminal_approach_requested": True,
        "goal_satisfied": True,
        "stop_applied": True,
        "reason": "fresh_grounding_approach_goal_satisfied",
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "initial_robot_position_xy_m": [0.0, 0.0],
        "target_xyz_m": [1.0, 0.0, 0.5],
        "target_distance_m": 0.4,
        "target_standoff_m": 0.5,
        "actual_progress_m": 0.7,
    }
    grounding = {
        "all_map_voxels_scored": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "numeric_approach_interlock": approach,
    }
    receipt = {
        "success": True,
        "collision": False,
        "stopped": True,
        "position_m": [0.7, 0.0, 0.0],
        "map_version": 3,
        "processed_voxels": 100,
    }
    return {
        "schema": "semantic_3d_chat.embodied_conversation_result.v1",
        "scene_id": scene_id,
        "environmental_text_inputs": [],
        "passed_runtime_audit": valid,
        "forbidden_access_count": 0,
        "turns": [
            {
                "success": True,
                "termination_reason": "stop",
                "prefix_refresh_verified": True,
                "primary_static_scene_retrieval": False,
                "static_scene_prefix_question_independent": True,
                "environmental_text_inputs": [],
                "step_count": 3,
                "continuous_grounding_attestations": [grounding],
                "action_receipts": [receipt],
            }
        ],
    }


def _oracle(scene_id: str = "scene_000001") -> dict:
    return {
        "scene_id": scene_id,
        "instances": [
            {
                "instance_id": "i_opaque",
                "category": "chair",
                "expected_center_xyz_m": [1.0, 0.0, 0.5],
                "bbox": {"min_xyz_m": [0.8, -0.2, 0.0], "max_xyz_m": [1.2, 0.2, 1.0]},
            }
        ],
    }


def test_scores_continuous_approach_after_runtime_validation(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    oracle = tmp_path / "oracle" / "scene_000001" / "oracle.json"
    _write(result, _runtime())
    _write(oracle, _oracle())

    report = score_approach_results(
        [("scene_000001", result)], oracle_root=tmp_path / "oracle", target_category="chair"
    )

    assert report["all_passed"] is True
    assert report["runtime_evidence_validated_before_oracle_open"] is True
    assert report["runtime_process_read_oracle"] is False
    assert report["scenes"][0]["oracle_center_progress_m"] == pytest.approx(0.7)
    assert report["scenes"][0]["final_oracle_bbox_standoff_m"] == pytest.approx(0.1)


def test_rejects_bad_runtime_before_parsing_oracle(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    oracle = tmp_path / "oracle" / "scene_000001" / "oracle.json"
    _write(result, _runtime(valid=False))
    oracle.parent.mkdir(parents=True)
    oracle.write_text("not valid JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime approach contract"):
        score_approach_results(
            [("scene_000001", result)],
            oracle_root=tmp_path / "oracle",
            target_category="chair",
        )


def test_scores_numeric_collision_limited_safe_stop(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    oracle = tmp_path / "oracle" / "scene_000001" / "oracle.json"
    runtime = _runtime()
    approach = runtime["turns"][0]["continuous_grounding_attestations"][0][
        "numeric_approach_interlock"
    ]
    approach.update(
        {
            "goal_satisfied": False,
            "completion_satisfied": True,
            "completion_mode": "collision_limited_safe_stop",
            "reason": "collision_limited_safe_stop",
            "target_distance_m": 0.8,
            "collision_limited_interlock": {
                "safe_closest_reachable": True,
                "numeric_collision_map_only": True,
            },
        }
    )
    _write(result, runtime)
    _write(oracle, _oracle())

    report = score_approach_results(
        [("scene_000001", result)], oracle_root=tmp_path / "oracle", target_category="chair"
    )

    assert report["all_passed"] is True
    assert report["scenes"][0]["safe_collision_limited_completion"] is True
    assert report["scenes"][0]["semantic_standoff_completed"] is False


def test_preserves_valid_collision_failure_as_failed_score(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    oracle = tmp_path / "oracle" / "scene_000001" / "oracle.json"
    runtime = _runtime()
    turn = runtime["turns"][0]
    turn["success"] = False
    turn["termination_reason"] = "action_failure"
    approach = turn["continuous_grounding_attestations"][0]["numeric_approach_interlock"]
    approach.update(
        {
            "goal_satisfied": False,
            "stop_applied": False,
            "reason": "learned_action_not_stalled",
            "target_distance_m": 0.8,
        }
    )
    receipt = turn["action_receipts"][0]
    receipt.update({"success": False, "collision": True, "stopped": False})
    _write(result, runtime)
    _write(oracle, _oracle())

    report = score_approach_results(
        [("scene_000001", result)], oracle_root=tmp_path / "oracle", target_category="chair"
    )

    assert report["all_passed"] is False
    assert report["passed_count"] == 0
    assert report["scenes"][0]["runtime_completed"] is False
    assert report["scenes"][0]["checks"]["collision_free"] is False
