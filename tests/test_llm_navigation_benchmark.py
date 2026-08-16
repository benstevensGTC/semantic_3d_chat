from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation.llm_navigation_benchmark import (
    _authenticated_policy_provenance,
    canonical_sha256,
    parse_task_manifest,
    run_navigation_manifest,
    score_navigation_journal,
    validate_navigation_journal,
)
from semantic_3d_chat.robot.llm_tool_policy import (
    GeneratedToolProposal,
    LocalGemmaToolPolicy,
)


def _config() -> dict[str, Any]:
    return {
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "robot": {
            "radius_m": 0.25,
            "camera_height_m": 1.2,
            "max_move_m": 0.5,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
        },
    }


def _manifest(*tasks: dict[str, Any]):
    return parse_task_manifest(
        {
            "schema": "semantic_3d_chat.llm_navigation_tasks.v1",
            "scene_id": "scene_000001",
            "seed": 7,
            "tasks": list(tasks),
        }
    )


def _task(task_id: str, family: str, instruction: str, max_steps: int = 8):
    return {
        "task_id": task_id,
        "family": family,
        "instruction": instruction,
        "max_steps": max_steps,
    }


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class FakeRuntime:
    def __init__(self) -> None:
        self.counter = 0
        self.reset_scene("scene_000001", 7)

    def _refresh(self, *, scene: bool = False) -> None:
        self.counter += 1
        self.active_hash = _digest(f"active-{self.counter}")
        self.robot_hash = _digest(f"robot-{self.counter}")
        if scene:
            self.scene_hash = _digest(f"scene-{self.counter}")
        self.binding_hash = _digest(
            f"{self.active_hash}-{self.scene_hash}-{self.robot_hash}"
        )

    def prefix_binding(self):
        return {
            "active_prefix_sha256": self.active_hash,
            "scene_prefix_sha256": self.scene_hash,
            "robot_tokens_sha256": self.robot_hash,
            "map_sha256": self.map_hash,
            "map_version": self.scene_version,
            "binding_sha256": self.binding_hash,
        }

    def _result(self, success: bool = True, **updates: Any):
        result = {
            "success": success,
            "error_code": None,
            "scene_id": self.scene_id,
            "scene_version": self.scene_version,
            "position_m": [self.x, self.y, 0.0],
            "camera_position_m": [self.x, self.y, 1.2],
            "body_yaw_degrees": self.body_yaw,
            "camera_yaw_degrees": self.body_yaw + self.camera_offset,
            "pitch_degrees": self.pitch,
            "collision": False,
            "last_movement_delta_m": list(self.last_delta),
            "distance_moved": 0.0,
            "turn_degrees": 0.0,
            "scan_coverage": 0.1 * self.scene_version,
            "scan_count": self.scene_version,
            "visible_voxels": 0,
            "valid_depth_pixels": 0,
            "clearance_m": 0.5,
            "action_count": self.action_count,
            "stopped": self.stopped,
            **self.prefix_binding(),
        }
        result.update(updates)
        return result

    def get_robot_state(self):
        return self._result()

    def reset_scene(self, scene_id: str, seed: int):
        assert scene_id == "scene_000001" and seed == 7
        self.scene_id = scene_id
        self.x = self.y = 0.0
        self.body_yaw = self.camera_offset = self.pitch = 0.0
        self.scene_version = 0
        self.action_count = 0
        self.stopped = False
        self.last_delta = (0.0, 0.0, 0.0)
        self.scene_hash = _digest("scene-base")
        self.map_hash = _digest("map-base")
        self._refresh()
        return self._result()

    def turn(self, angle: float):
        self.body_yaw = (self.body_yaw + float(angle) + 180.0) % 360.0 - 180.0
        self.action_count += 1
        self._refresh()
        return self._result(turn_degrees=float(angle))

    def look(self, yaw: float, pitch: float):
        self.camera_offset += float(yaw)
        self.pitch += float(pitch)
        self.action_count += 1
        self._refresh()
        return self._result(turn_degrees=float(yaw))

    def move_forward(self, distance: float):
        yaw = math.radians(self.body_yaw)
        dx = -math.sin(yaw) * float(distance)
        dy = math.cos(yaw) * float(distance)
        self.x += dx
        self.y += dy
        self.last_delta = (dx, dy, 0.0)
        self.action_count += 1
        self._refresh()
        return self._result(distance_moved=float(distance))

    def move_backward(self, distance: float):
        return self.move_forward(-float(distance))

    def move_to(self, x: float, y: float):
        distance = math.hypot(float(x) - self.x, float(y) - self.y)
        self.last_delta = (float(x) - self.x, float(y) - self.y, 0.0)
        self.x, self.y = float(x), float(y)
        self.action_count += 1
        self._refresh()
        return self._result(distance_moved=distance)

    def scan(self):
        self.scene_version += 1
        self.map_hash = _digest(f"map-{self.scene_version}")
        self.action_count += 1
        self._refresh(scene=True)
        return self._result(
            visible_voxels=12,
            valid_depth_pixels=12,
            scan_count=self.scene_version,
        )

    def stop(self):
        self.stopped = True
        self.action_count += 1
        self._refresh()
        return self._result()


class PrefixAwareQueueBackend:
    def __init__(
        self,
        runtime: FakeRuntime,
        outputs: list[str],
        *,
        training_status: str = "untrained_tool_selection_seam",
    ) -> None:
        self.runtime = runtime
        self.outputs = list(outputs)
        self.inputs: list[str] = []
        self.training_status = training_status

    def generate(self, instruction: str, *, correction_code: str | None):
        del correction_code
        self.inputs.append(instruction)
        binding = self.runtime.prefix_binding()
        return GeneratedToolProposal(
            text=self.outputs.pop(0),
            active_prefix_sha256=binding["active_prefix_sha256"],
            scene_prefix_sha256=binding["scene_prefix_sha256"],
            robot_tokens_sha256=binding["robot_tokens_sha256"],
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=True,
            training_status=self.training_status,
        )


def _policy(
    runtime: FakeRuntime,
    outputs: list[str],
    *,
    training_status: str = "untrained_tool_selection_seam",
):
    backend = PrefixAwareQueueBackend(
        runtime, outputs, training_status=training_status
    )
    return (
        LocalGemmaToolPolicy(
            backend,
            _config(),
            robot_state_provider=runtime.get_robot_state,
            max_retries=0,
            fallback_policy="fail_closed",
        ),
        backend,
    )


def _call(name: str, **arguments: Any) -> str:
    return json.dumps({"tool": name, "arguments": arguments})


def test_v4_provenance_requires_exact_numeric_clearance_safety_contract() -> None:
    status = "supervised_continuous_semantic_clearance_navigation_policy_v4"
    contract = {
        "tool_policy_training_status": status,
        "navigation_policy_checkpoint_tree_sha256": "1" * 64,
        "navigation_policy_source_sha256": "2" * 64,
        "fallback_policy": "fail_closed",
        "strict_fixed_environment_embedding_input": True,
        "question_conditioned_scene_readout_tokens": False,
        "continuous_semantic_grounding_required": True,
        "all_map_voxels_scored_for_grounding": True,
        "query_dependent_grounding_navigation_only": True,
        "oracle_inputs_at_runtime": False,
        "environmental_text_inputs_at_runtime": [],
        "numeric_clearance_state_required": True,
        "clearance_from_sanitized_geometry_only": True,
        "clearance_ray_count": 24,
        "clearance_max_range_m": 1.0,
        "exact_collision_mask_required": True,
        "unsafe_motion_fallback": "highest_safe_nonterminal_action",
        "collision_interlock_required": True,
        "static_scene_prefix_question_independent": True,
    }
    journal = {
        "header": {
            "tool_policy_training_status": status,
            "run_contract": contract,
        }
    }
    authenticated = _authenticated_policy_provenance(journal)
    assert authenticated["claimed_trained_navigation_policy"] is True
    assert authenticated["policy_status"] == status
    del contract["exact_collision_mask_required"]
    with pytest.raises(ValueError, match="clearance-safety"):
        _authenticated_policy_provenance(journal)


def test_task_manifest_supports_six_families_without_oracle_fields() -> None:
    manifest = _manifest(
        _task("nav_000", "face", "Face the chair."),
        _task("nav_001", "approach", "Approach the table."),
        _task("nav_002", "stop", "Move and stop."),
        _task("nav_003", "obstacle", "Go around it."),
        _task("nav_004", "left_right", "Use the shorter turn."),
        _task("nav_005", "update_after_scan", "Scan and continue."),
    )
    assert {task.family for task in manifest.tasks} == {
        "face",
        "approach",
        "stop",
        "obstacle",
        "left_right",
        "update_after_scan",
    }
    with pytest.raises(ValueError, match="exactly"):
        parse_task_manifest(
            {
                "schema": "semantic_3d_chat.llm_navigation_tasks.v1",
                "scene_id": "scene_000001",
                "seed": 7,
                "tasks": [
                    {
                        **_task("nav_000", "face", "Face it."),
                        "target_instance_id": "i_000101",
                    }
                ],
            }
        )


def test_closed_loop_journal_contains_hashes_not_user_environment_prose(tmp_path: Path) -> None:
    manifest = _manifest(
        _task("nav_000", "stop", "Move toward the forbidden-zebra and stop.", 3)
    )
    runtime = FakeRuntime()
    policy, backend = _policy(runtime, [_call("move_forward", distance_meters=0.25), _call("stop")])
    journal_path = tmp_path / "predictions.json"
    journal = run_navigation_manifest(
        runtime,
        policy,
        manifest,
        journal_path=journal_path,
        run_contract={"checkpoint_sha256": "a" * 64},
    )
    assert journal["status"] == "complete"
    assert validate_navigation_journal(
        json.loads(journal_path.read_text()), require_complete=True
    )["journal_sha256"] == journal["journal_sha256"]
    serialized = json.dumps(journal).casefold()
    assert "forbidden-zebra" not in serialized
    assert journal["episodes"][0]["environmental_text_inputs"] == []
    assert journal["episodes"][0]["tool_policy_training_status"] == (
        "untrained_tool_selection_seam"
    )
    assert "numeric result from the preceding bounded action" in backend.inputs[1].casefold()
    assert "position_m" in backend.inputs[1]
    assert len(journal["episodes"][0]["steps"]) == 2

    tampered = json.loads(journal_path.read_text())
    tampered["episodes"][0]["initial_prefix_binding"][
        "active_prefix_sha256"
    ] = "f" * 64
    episode = tampered["episodes"][0]
    episode["episode_sha256"] = canonical_sha256(
        {key: value for key, value in episode.items() if key != "episode_sha256"}
    )
    tampered["journal_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "journal_sha256"}
    )
    with pytest.raises(ValueError, match="prefix chain is discontinuous"):
        validate_navigation_journal(tampered, require_complete=True)


def test_journal_resume_restarts_only_unsealed_episode_and_tamper_fails(tmp_path: Path) -> None:
    manifest = _manifest(
        _task("nav_000", "stop", "Stop.", 1),
        _task("nav_001", "stop", "Stop again.", 1),
    )
    path = tmp_path / "journal.json"
    runtime = FakeRuntime()
    policy, _ = _policy(runtime, [_call("stop"), _call("stop")])

    def interrupt(_episode):
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        run_navigation_manifest(
            runtime,
            policy,
            manifest,
            journal_path=path,
            run_contract={"version": 1},
            after_episode=interrupt,
        )
    partial = validate_navigation_journal(json.loads(path.read_text()))
    assert partial["status"] == "in_progress"
    assert [row["task_id"] for row in partial["episodes"]] == ["nav_000"]

    resumed_runtime = FakeRuntime()
    resumed_policy, _ = _policy(resumed_runtime, [_call("stop")])
    resumed = run_navigation_manifest(
        resumed_runtime,
        resumed_policy,
        manifest,
        journal_path=path,
        run_contract={"version": 1},
        resume=True,
    )
    assert resumed["status"] == "complete"
    assert [row["task_id"] for row in resumed["episodes"]] == ["nav_000", "nav_001"]

    tampered = json.loads(path.read_text())
    tampered["episodes"][0]["final_state"]["position_m"][0] = 99
    with pytest.raises(ValueError, match="episode.*hash"):
        validate_navigation_journal(tampered)


def test_physically_separate_scorer_scores_all_families(tmp_path: Path) -> None:
    manifest = _manifest(
        _task("nav_000", "face", "Face target.", 4),
        _task("nav_001", "approach", "Approach target.", 4),
        _task("nav_002", "stop", "Move 0.25 and stop.", 3),
        _task("nav_003", "obstacle", "Go around obstacle.", 4),
        _task("nav_004", "left_right", "Turn the shorter direction.", 4),
        _task("nav_005", "update_after_scan", "Scan, approach, and stop.", 4),
    )
    outputs = [
        _call("turn", angle_degrees=45),
        _call("turn", angle_degrees=45),
        _call("stop"),
        _call("move_forward", distance_meters=0.5),
        _call("move_forward", distance_meters=0.5),
        _call("stop"),
        _call("move_forward", distance_meters=0.25),
        _call("stop"),
        _call("move_forward", distance_meters=0.5),
        _call("move_forward", distance_meters=0.5),
        _call("stop"),
        _call("turn", angle_degrees=45),
        _call("turn", angle_degrees=45),
        _call("stop"),
        _call("scan"),
        _call("move_forward", distance_meters=0.5),
        _call("stop"),
    ]
    runtime = FakeRuntime()
    policy, _ = _policy(runtime, outputs)
    journal_path = tmp_path / "journal.json"
    journal = run_navigation_manifest(
        runtime,
        policy,
        manifest,
        journal_path=journal_path,
        run_contract={"version": 1},
        runtime_file_audit={
            "passed": True,
            "blocking_enabled": True,
            "forbidden_accesses": [],
        },
    )
    assert journal["status"] == "complete"

    oracle = {
        "scene_id": "scene_000001",
        "instances": [
            {
                "instance_id": "i_000001",
                "bbox": {"min_xyz_m": [-1.1, -0.1, 0], "max_xyz_m": [-0.9, 0.1, 1]},
            },
            {
                "instance_id": "i_000002",
                "bbox": {"min_xyz_m": [-0.1, 0.9, 0], "max_xyz_m": [0.1, 1.1, 1]},
            },
            {
                "instance_id": "i_000003",
                "bbox": {"min_xyz_m": [0.9, 0.4, 0], "max_xyz_m": [1.1, 0.6, 1]},
            },
        ],
    }
    spec = {
        "schema": "semantic_3d_chat.llm_navigation_oracle.v1",
        "scene_id": "scene_000001",
        "task_manifest_sha256": manifest.canonical_sha256,
        "tasks": [
            {
                "task_id": "nav_000",
                "family": "face",
                "target_instance_id": "i_000001",
                "maximum_heading_error_degrees": 1,
                "require_stopped": True,
            },
            {
                "task_id": "nav_001",
                "family": "approach",
                "target_instance_id": "i_000002",
                "maximum_target_standoff_m": 0.1,
                "minimum_target_progress_m": 0.8,
                "require_stopped": True,
            },
            {
                "task_id": "nav_002",
                "family": "stop",
                "minimum_displacement_m": 0.24,
                "maximum_displacement_m": 0.26,
                "require_stopped": True,
            },
            {
                "task_id": "nav_003",
                "family": "obstacle",
                "target_instance_id": "i_000002",
                "obstacle_instance_id": "i_000003",
                "maximum_target_standoff_m": 0.1,
                "minimum_target_progress_m": 0.8,
                "minimum_obstacle_bbox_clearance_m": 0.8,
                "require_stopped": True,
            },
            {
                "task_id": "nav_004",
                "family": "left_right",
                "target_instance_id": "i_000001",
                "maximum_heading_error_degrees": 1,
                "require_stopped": True,
            },
            {
                "task_id": "nav_005",
                "family": "update_after_scan",
                "target_instance_id": "i_000002",
                "maximum_target_standoff_m": 0.5,
                "minimum_target_progress_m": 0.4,
                "require_stopped": True,
            },
        ],
    }
    oracle_path = tmp_path / "oracle.json"
    spec_path = tmp_path / "scorer.json"
    oracle_path.write_text(json.dumps(oracle))
    spec_path.write_text(json.dumps(spec))
    score = score_navigation_journal(journal_path, spec_path, oracle_path)
    assert score["passed"]
    assert score["policy_status"] == "untrained_tool_selection_seam"
    assert score["claimed_trained_navigation_policy"] is False
    assert score["navigation_policy_checkpoint_tree_sha256"] is None
    assert score["metrics"] == {
        "task_count": 6,
        "success_count": 6,
        "success_rate": 1.0,
        "collision_count": 0,
        "action_failure_count": 0,
        "policy_rejection_count": 0,
        "executed_action_count": 17,
    }
    context = score["continuous_context_evidence"]
    assert context["passed"] is True
    assert context["step_count"] == 17
    assert context["decision_context_match_count"] == 17
    assert context["prefix_chain_match_count"] == 17
    assert context["numeric_state_change_count"] == 17
    assert context["robot_token_refresh_count"] == 17
    assert context["map_update_count"] == 1
    assert context["scene_prefix_refresh_count"] == 1
    assert context["next_decision_count"] == 11
    assert context["refreshed_context_consumed_count"] == 11
    assert context["oracle_inputs_used"] is False
    assert context["environmental_text_inputs"] == []
    assert set(score["by_family"]) == {
        "face",
        "approach",
        "stop",
        "obstacle",
        "left_right",
        "update_after_scan",
    }
    update = score["tasks"][-1]
    assert update["checks"]["successful_scan"]
    assert update["checks"]["updated_prefix_consumed"]
    assert update["checks"]["post_scan_motion"]

    learned_runtime = FakeRuntime()
    learned_policy, _ = _policy(
        learned_runtime,
        outputs,
        training_status="supervised_continuous_navigation_policy_v1",
    )
    learned_path = tmp_path / "learned.json"
    learned = run_navigation_manifest(
        learned_runtime,
        learned_policy,
        manifest,
        journal_path=learned_path,
        run_contract={
            "tool_policy_training_status": (
                "supervised_continuous_navigation_policy_v1"
            ),
            "navigation_policy_checkpoint_tree_sha256": "1" * 64,
            "navigation_policy_source_sha256": "2" * 64,
            "fallback_policy": "fail_closed",
            "strict_fixed_environment_embedding_input": True,
            "question_conditioned_scene_readout_tokens": False,
        },
        runtime_file_audit={
            "passed": True,
            "blocking_enabled": True,
            "forbidden_accesses": [],
        },
        policy_training_status="supervised_continuous_navigation_policy_v1",
    )
    assert learned["status"] == "complete"
    learned_score = score_navigation_journal(learned_path, spec_path, oracle_path)
    assert learned_score["passed"]
    assert (
        learned_score["policy_status"]
        == "supervised_continuous_navigation_policy_v1"
    )
    assert learned_score["claimed_trained_navigation_policy"] is True
    assert learned_score["navigation_policy_checkpoint_tree_sha256"] == "1" * 64
    assert all("not trained" not in value for value in learned_score["limitations"])


def test_scorer_never_opens_oracle_when_journal_is_not_sealed(tmp_path: Path) -> None:
    journal = tmp_path / "bad.json"
    journal.write_text(
        json.dumps(
            {
                "schema": "semantic_3d_chat.llm_navigation_journal.v1",
                "status": "complete",
                "header": {},
                "episodes": [],
                "runtime_file_audit": {},
                "journal_sha256": "0" * 64,
            }
        )
    )
    missing_spec = tmp_path / "oracle-sidecar-must-not-open.json"
    missing_oracle = tmp_path / "scene-oracle-must-not-open.json"
    with pytest.raises(ValueError, match="root hash"):
        score_navigation_journal(journal, missing_spec, missing_oracle)
    assert not missing_spec.exists() and not missing_oracle.exists()


def test_stop_displacement_failure_is_policy_behavior_not_tolerance_bug(
    tmp_path: Path,
) -> None:
    manifest = _manifest(_task("nav_000", "stop", "Move 0.25 and stop.", 2))
    runtime = FakeRuntime()
    policy, _ = _policy(runtime, [_call("stop")])
    journal = tmp_path / "journal.json"
    run_navigation_manifest(
        runtime,
        policy,
        manifest,
        journal_path=journal,
        run_contract={"version": 1},
        runtime_file_audit={
            "passed": True,
            "blocking_enabled": True,
            "forbidden_accesses": [],
        },
    )
    oracle = tmp_path / "scene.json"
    scorer = tmp_path / "scorer.json"
    oracle.write_text(json.dumps({"scene_id": "scene_000001", "instances": []}))
    scorer.write_text(
        json.dumps(
            {
                "schema": "semantic_3d_chat.llm_navigation_oracle.v1",
                "scene_id": "scene_000001",
                "task_manifest_sha256": manifest.canonical_sha256,
                "tasks": [
                    {
                        "task_id": "nav_000",
                        "family": "stop",
                        "minimum_displacement_m": 0.20,
                        "maximum_displacement_m": 0.30,
                        "require_stopped": True,
                    }
                ],
            }
        )
    )
    result = score_navigation_journal(journal, scorer, oracle)
    task = result["tasks"][0]
    assert not task["passed"]
    assert task["checks"]["required_stop"]
    assert not task["checks"]["displacement"]
    assert task["metrics"]["displacement_m"] == 0.0
    assert task["metrics"]["minimum_displacement_m"] == 0.20
    assert task["metrics"]["maximum_displacement_m"] == 0.30
    assert task["metrics"]["first_executed_tool"] == "stop"


def test_scorer_rejects_pending_runtime_audit_before_oracle_open(tmp_path: Path) -> None:
    manifest = _manifest(_task("nav_000", "stop", "Stop.", 1))
    runtime = FakeRuntime()
    policy, _ = _policy(runtime, [_call("stop")])
    journal_path = tmp_path / "journal.json"
    run_navigation_manifest(
        runtime,
        policy,
        manifest,
        journal_path=journal_path,
        run_contract={"version": 1},
        runtime_file_audit={"status": "pending_until_runtime_exit"},
    )
    missing_spec = tmp_path / "scoring-spec-must-not-open.json"
    missing_oracle = tmp_path / "oracle-must-not-open.json"
    with pytest.raises(ValueError, match="clean file audit"):
        score_navigation_journal(journal_path, missing_spec, missing_oracle)
    assert not missing_spec.exists() and not missing_oracle.exists()


def test_actual_task_manifest_and_oracle_sidecar_are_hash_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_payload = json.loads(
        (root / "configs/benchmarks/llm_navigation_scene_000001.json").read_text()
    )
    sidecar = json.loads(
        (
            root / "configs/benchmarks/oracle/llm_navigation_scene_000001.json"
        ).read_text()
    )
    manifest = parse_task_manifest(manifest_payload)
    assert manifest.canonical_sha256 == sidecar["task_manifest_sha256"]
    runtime_rows = json.dumps(manifest_payload).casefold()
    assert "target_instance_id" not in runtime_rows
    assert "bbox" not in runtime_rows
    assert "expected" not in runtime_rows
    assert canonical_sha256(manifest_payload) == (
        "8a59fd86c4f166e3361ae4204620fc3908b1b3f47d24bf7453526611a5b9c69a"
    )


def test_v2_benchmark_is_distinct_preregistered_and_feasible() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_payload = json.loads(
        (root / "configs/benchmarks/llm_navigation_v2_scene_000001.json").read_text()
    )
    sidecar = json.loads(
        (
            root
            / "configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json"
        ).read_text()
    )
    manifest = parse_task_manifest(manifest_payload)
    assert manifest.canonical_sha256 == sidecar["task_manifest_sha256"]
    assert manifest.canonical_sha256 != (
        "8a59fd86c4f166e3361ae4204620fc3908b1b3f47d24bf7453526611a5b9c69a"
    )
    assert sidecar["benchmark_version"] == 2
    assert sidecar["expected_initial_position_xy_m"] == [0.0, -0.5]
    assert sidecar["inference_config_sha256"] == hashlib.sha256(
        (root / "configs/runtime/embodied_navigation_v2.yaml").read_bytes()
    ).hexdigest()
    feasibility = sidecar["feasibility_justification"]
    assert feasibility["prepared_before_v2_inference"] is True
    assert feasibility["criteria_changed_from_v1"] is False
    assert feasibility["approach_progress_feasibility_margin_m"] > 0.4
    runtime_rows = json.dumps(manifest_payload).casefold()
    assert "target_instance_id" not in runtime_rows
    assert "bbox" not in runtime_rows
