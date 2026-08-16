from __future__ import annotations

import hashlib
import threading
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.llm_tool_policy import (
    GeneratedToolProposal,
    LocalGemmaToolPolicy,
)
from semantic_3d_chat.robot.practical_rover import (
    PracticalRoverController,
    practical_rover_preflight,
)


def _config() -> dict[str, Any]:
    return {
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "scene_encoder": {"global_latents": 4},
        "robot": {
            "radius_m": 0.25,
            "max_move_m": 0.5,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 45.0,
            "max_camera_yaw_offset_degrees": 90.0,
            "max_pitch_degrees": 45.0,
        },
    }


class FakeRuntime:
    def __init__(self) -> None:
        self.simulator = SimpleNamespace(history=deque(maxlen=64))
        self.version = 0
        self.yaw = 0.0
        self.position = [0.0, 0.0, 1.2]
        self.stopped = False
        self.forward_attempts = 0
        self.fail_forward_on_attempt: int | None = None
        base = SimpleNamespace(
            map_data=SimpleNamespace(
                feature_dim=3072,
                source_voxel_count=16,
                voxel_count=12,
            ),
            checkpoint_metadata={"semantic_dim": 3072},
        )
        self.prefix_refresher = SimpleNamespace(
            runtime=SimpleNamespace(
                base=base,
                control_metadata={"saved_runtime_training_gate_passed": True},
            )
        )

    def _scene_prefix(self) -> torch.Tensor:
        values = torch.linspace(0.1, 4.8, 48, dtype=torch.float32)
        return values.reshape(1, 6, 8) + self.version * 0.01

    def _robot_tokens(self) -> torch.Tensor:
        state = torch.tensor(
            [self.position[0], self.position[1], self.yaw, float(self.stopped)],
            dtype=torch.float32,
        )
        return state.repeat(4).reshape(1, 2, 8) + 0.01

    def _active_prefix(self) -> torch.Tensor:
        scene = self._scene_prefix()
        return torch.cat((scene[:, :-1], self._robot_tokens(), scene[:, -1:]), dim=1)

    def prefix_binding(self) -> dict[str, Any]:
        scene = self._scene_prefix()
        active = self._active_prefix()
        return {
            "map_version": self.version,
            "scene_prefix_sha256": prefix_sha256(scene),
            "active_prefix_sha256": prefix_sha256(active),
            "robot_tokens_sha256": prefix_sha256(self._robot_tokens()),
            "source_voxels": 16,
            "processed_voxels": 12,
        }

    def active_prefix_snapshot(self) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._active_prefix().clone(), self.prefix_binding()

    def get_robot_state(self) -> dict[str, Any]:
        return {
            "success": True,
            "scene_id": "scene_000001",
            "scene_version": self.version,
            "position_m": list(self.position),
            "body_yaw_degrees": self.yaw,
            "camera_yaw_degrees": self.yaw,
            "pitch_degrees": 0.0,
            "collision": False,
            "scan_count": self.version,
            "action_count": len(self.simulator.history),
            "stopped": self.stopped,
            **self.prefix_binding(),
        }

    def _receipt(self, name: str, *, scan: bool = False) -> dict[str, Any]:
        if scan:
            self.version += 1
        receipt = {
            **self.get_robot_state(),
            "success": True,
            "valid_depth_pixels": 100 if scan else 0,
            "observation_id": f"o_{self.version:06d}" if scan else None,
            "command": name,
        }
        self.simulator.history.append(receipt)
        return receipt

    def scan(self) -> dict[str, Any]:
        return self._receipt("scan", scan=True)

    def turn(self, angle: float) -> dict[str, Any]:
        self.yaw += angle
        return self._receipt("turn")

    def move_forward(self, distance: float) -> dict[str, Any]:
        self.forward_attempts += 1
        if self.forward_attempts == self.fail_forward_on_attempt:
            receipt = {
                **self.get_robot_state(),
                "success": False,
                "collision": True,
                "error_code": "E_COLLISION",
                "valid_depth_pixels": 0,
                "command": "move_forward",
            }
            self.simulator.history.append(receipt)
            return receipt
        self.position[1] += distance
        return self._receipt("move_forward")

    def move_backward(self, distance: float) -> dict[str, Any]:
        self.position[1] -= distance
        return self._receipt("move_backward")

    def look(self, yaw: float, pitch: float) -> dict[str, Any]:
        del pitch
        self.yaw += yaw
        return self._receipt("look")

    def move_to(self, x: float, y: float) -> dict[str, Any]:
        self.position[:2] = [x, y]
        return self._receipt("move_to")

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        return self._receipt("stop")

    def reset_scene(self, scene_id: str, seed: int) -> dict[str, Any]:
        assert scene_id == "scene_000001" and seed >= 0
        self.stopped = False
        self.position[:2] = [0.0, 0.0]
        self.version = 0
        return self._receipt("reset_scene")


class AutoScanningFakeRuntime(FakeRuntime):
    """Small behavioral double for RefreshingEmbodiedChatRuntime._action."""

    def __init__(self) -> None:
        super().__init__()
        self.auto_scan_after_motion = True
        self._lock = threading.RLock()
        self.scan_invocations = 0
        self.fail_scan = False

    def _after_motion(self, receipt: dict[str, Any]) -> dict[str, Any]:
        if receipt.get("success") is not True or not self.auto_scan_after_motion:
            return receipt
        scan_receipt = self.scan()
        for field in ("distance_moved", "turn_degrees", "clearance_m"):
            if field in receipt:
                scan_receipt[field] = receipt[field]
        return scan_receipt

    def scan(self) -> dict[str, Any]:
        self.scan_invocations += 1
        if self.fail_scan:
            receipt = {
                **self.get_robot_state(),
                "success": False,
                "error_code": "E_MAP_UPDATE",
                "valid_depth_pixels": 0,
                "command": "scan",
            }
            self.simulator.history.append(receipt)
            return receipt
        return super().scan()

    def turn(self, angle: float) -> dict[str, Any]:
        return self._after_motion(super().turn(angle))

    def move_forward(self, distance: float) -> dict[str, Any]:
        return self._after_motion(super().move_forward(distance))

    def move_backward(self, distance: float) -> dict[str, Any]:
        return self._after_motion(super().move_backward(distance))

    def look(self, yaw: float, pitch: float) -> dict[str, Any]:
        return self._after_motion(super().look(yaw, pitch))

    def move_to(self, x: float, y: float) -> dict[str, Any]:
        return self._after_motion(super().move_to(x, y))


@dataclass
class FakeAnswer:
    answer: str = "a local continuous answer"


class FakeSemanticAgent:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime
        self.text_encoder = object()
        self.calls: list[str] = []
        self.answer_calls: list[str] = []

    def handle(self, text: str) -> dict[str, Any]:
        self.calls.append(text)
        if text.casefold().startswith("face"):
            receipt = self.runtime.turn(12.0)
            return {
                "kind": "navigation",
                "command": "face",
                "success": True,
                "action_receipts": [receipt],
            }
        return {"kind": "answer", "answer": FakeAnswer().answer}

    def answer_question(self, text: str) -> dict[str, Any]:
        self.answer_calls.append(text)
        return {"kind": "answer", "answer": FakeAnswer().answer}


class DynamicBackend:
    def __init__(self, runtime: FakeRuntime, output: str) -> None:
        self.runtime = runtime
        self.output = output
        self.calls = 0

    def generate(self, instruction: str, *, correction_code: str | None):
        del instruction, correction_code
        self.calls += 1
        binding = self.runtime.prefix_binding()
        return GeneratedToolProposal(
            text=self.output,
            active_prefix_sha256=binding["active_prefix_sha256"],
            scene_prefix_sha256=binding["scene_prefix_sha256"],
            robot_tokens_sha256=binding["robot_tokens_sha256"],
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=True,
        )


def _controller(
    output: str,
    *,
    initial_scan: bool = True,
    runtime: FakeRuntime | None = None,
):
    runtime = FakeRuntime() if runtime is None else runtime
    backend = DynamicBackend(runtime, output)
    policy = LocalGemmaToolPolicy(
        backend,
        _config(),
        robot_state_provider=runtime.get_robot_state,
        max_retries=0,
    )
    controller = PracticalRoverController(
        runtime,
        FakeSemanticAgent(runtime),  # type: ignore[arg-type]
        _config(),
        llm_tool_policy=policy,
        initial_scan=initial_scan,
    )
    return controller, runtime, backend


def test_startup_scans_once_and_is_idempotent() -> None:
    controller, runtime, _ = _controller('{"tool":"scan","arguments":{}}')
    first = controller.startup()
    second = controller.startup()
    assert first == second
    assert first["initial_scan_performed"] is True
    assert first["scene_memory_refreshed"] is True
    assert first["map_version"] == 1
    assert len(runtime.simulator.history) == 1
    assert first["cloud_model_used"] is False
    memory = first["scene_memory"]
    assert memory["tensor_shape"] == [1, 6, 8]
    assert memory["active_tensor_shape"] == [1, 8, 8]
    assert memory["token_count"] == 6
    assert memory["model_dim"] == 8
    assert memory["robot_state_token_count"] == 2
    assert memory["sha256"] == first["scene_prefix_hash"]
    assert memory["active_sha256"] == first["active_prefix_hash"]
    assert memory["semantic_feature_dim"] == 3072
    assert memory["source_voxels"] == 16
    assert memory["processed_voxels"] == 12
    assert memory["l2_norm"] > 0.0
    assert memory["rms"] > 0.0
    assert memory["all_runtime_voxels_encoded"] is True
    assert memory["control_training_gate_passed"] is True
    assert memory["question_dependent_scene_retrieval"] is False
    assert memory["loaded_file_audit"] == {
        "enabled": False,
        "loaded_file_count": 0,
        "loaded_file_inventory_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        ),
        "forbidden_access_count": 0,
        "passed": True,
    }
    assert memory["environmental_text_inputs"] == []
    controller.close()


def test_scene_memory_diagnostics_fail_closed_on_tensor_binding_mismatch() -> None:
    class TamperedRuntime(FakeRuntime):
        def active_prefix_snapshot(self) -> tuple[torch.Tensor, dict[str, Any]]:
            tensor, binding = super().active_prefix_snapshot()
            tensor[0, 0, 0] += 1.0
            return tensor, binding

    controller, _runtime, _backend = _controller(
        "unused",
        initial_scan=False,
        runtime=TamperedRuntime(),
    )
    with pytest.raises(RuntimeError, match="differs from its binding"):
        controller.startup()
    controller.close()


def test_conforming_local_gemma_action_executes_without_fallback() -> None:
    controller, runtime, backend = _controller(
        '{"tool":"turn","arguments":{"angle_degrees":-15}}',
        initial_scan=False,
    )
    result = controller.handle_instruction("Turn right 15 degrees")
    assert result["success"] is True
    assert result["gemma_attempted"] is True
    assert result["gemma_accepted"] is True
    assert result["fallback_used"] is False
    assert result["control_mode"] == "local_gemma_constrained_json"
    assert runtime.yaw == -15.0
    assert backend.calls == 1
    controller.close()


def test_nonconforming_proposal_uses_explicit_safe_semantic_fallback() -> None:
    controller, runtime, backend = _controller(
        '{"tool":"move_forward","arguments":{"distance_meters":0.5}}',
        initial_scan=False,
    )
    result = controller.handle_instruction("Face the chair")
    assert result["success"] is True
    assert result["gemma_attempted"] is True
    assert result["gemma_accepted"] is False
    assert result["fallback_used"] is True
    assert result["control_mode"].startswith("local_gemma_all_voxel_semantic_grounding")
    assert runtime.yaw == 12.0
    assert backend.calls == 1
    controller.close()


def test_friendly_shorthand_falls_back_to_bounded_validated_action() -> None:
    controller, runtime, _ = _controller("not json", initial_scan=False)
    result = controller.handle_instruction("please drive forward")
    assert result["success"] is True
    assert result["gemma_attempted"] is True
    assert result["gemma_accepted"] is False
    assert result["fallback_used"] is True
    assert runtime.position[1] == 0.25
    controller.close()


def test_greeting_never_becomes_an_arbitrary_scene_qa_token() -> None:
    controller, _runtime, backend = _controller("3", initial_scan=False)
    result = controller.handle_instruction("hi")
    assert result["success"] is True
    assert result["reply"].startswith("Hi!")
    assert "outcome-level goal" in result["reply"]
    assert result["control_mode"] == "deterministic_non_environmental_greeting"
    assert result["gemma_attempted"] is False
    assert result["gemma_accepted"] is False
    assert result["fallback_used"] is False
    assert backend.calls == 0
    assert controller.semantic_agent.calls == []
    controller.close()


def test_polite_compound_command_is_chunked_and_executed_in_order() -> None:
    controller, runtime, backend = _controller("ients", initial_scan=False)
    result = controller.handle_instruction(
        "can you please move forward 3 and then turn right"
    )
    assert result["success"] is True
    assert result["planned_action_count"] == 8
    assert result["completed_action_count"] == 8
    assert len(result["actions"]) == 8
    assert runtime.position[1] == 3.0
    assert runtime.yaw == -90.0
    assert result["gemma_attempted"] is True
    assert result["gemma_accepted"] is False
    assert result["fallback_used"] is True
    assert result["control_mode"] == "deterministic_compound_parser_fallback"
    assert backend.calls == 1
    assert controller.semantic_agent.calls == []
    controller.close()


def test_live_compound_motion_defers_auto_scan_until_final_pose() -> None:
    runtime = AutoScanningFakeRuntime()
    controller, runtime, _backend = _controller(
        "ients",
        initial_scan=False,
        runtime=runtime,
    )
    result = controller.handle_instruction(
        "can you please move forward 3 and then turn right"
    )

    assert result["success"] is True
    assert result["planned_action_count"] == 8
    assert result["completed_action_count"] == 8
    assert result["final_refresh_performed"] is True
    assert result["final_refresh_succeeded"] is True
    assert result["final_refresh_error_code"] is None
    assert result["scene_memory_refreshed"] is True
    assert runtime.position[1] == 3.0
    assert runtime.yaw == -90.0
    assert runtime.scan_invocations == 1
    assert runtime.version == 1
    assert runtime.auto_scan_after_motion is True
    assert [item["command"] for item in runtime.simulator.history] == [
        *("move_forward" for _ in range(6)),
        "turn",
        "turn",
        "scan",
    ]
    # The response preserves the complete bounded trajectory plus its one
    # question-independent scene-memory refresh receipt.
    assert len(result["actions"]) == 9
    controller.close()


def test_live_compound_collision_scans_once_after_partial_progress() -> None:
    runtime = AutoScanningFakeRuntime()
    runtime.fail_forward_on_attempt = 3
    controller, runtime, _backend = _controller(
        "not json",
        initial_scan=False,
        runtime=runtime,
    )
    result = controller.handle_instruction(
        "Would you please drive forward 2 metres, then rotate right?"
    )

    assert result["success"] is False
    assert result["error_code"] == "E_COLLISION"
    assert result["planned_action_count"] == 6
    assert result["completed_action_count"] == 2
    assert result["final_refresh_performed"] is True
    assert result["final_refresh_succeeded"] is True
    assert result["scene_memory_refreshed"] is True
    assert runtime.position[1] == 1.0
    assert runtime.yaw == 0.0
    assert runtime.scan_invocations == 1
    assert runtime.version == 1
    assert runtime.auto_scan_after_motion is True
    assert [item["command"] for item in runtime.simulator.history] == [
        "move_forward",
        "move_forward",
        "move_forward",
        "scan",
    ]
    assert "stopped safely" in result["reply"]
    controller.close()


def test_live_compound_final_refresh_failure_is_reported_and_flag_restored() -> None:
    runtime = AutoScanningFakeRuntime()
    runtime.fail_scan = True
    controller, runtime, _backend = _controller(
        "not json",
        initial_scan=False,
        runtime=runtime,
    )
    result = controller.handle_instruction("move forward 1 metre then turn left")

    assert result["success"] is False
    assert result["planned_action_count"] == 4
    assert result["completed_action_count"] == 4
    assert result["error_code"] == "E_MAP_UPDATE"
    assert result["final_refresh_performed"] is True
    assert result["final_refresh_succeeded"] is False
    assert result["final_refresh_error_code"] == "E_MAP_UPDATE"
    assert result["scene_memory_refreshed"] is False
    assert runtime.scan_invocations == 1
    assert runtime.version == 0
    assert runtime.auto_scan_after_motion is True
    controller.close()


def test_single_tool_call_keeps_auto_scan_per_call_behavior() -> None:
    runtime = AutoScanningFakeRuntime()
    controller, runtime, backend = _controller(
        '{"tool":"turn","arguments":{"angle_degrees":-15}}',
        initial_scan=False,
        runtime=runtime,
    )
    result = controller.handle_instruction("Turn right 15 degrees")

    assert result["success"] is True
    assert result["final_refresh_performed"] is False
    assert runtime.scan_invocations == 1
    assert runtime.version == 1
    assert runtime.auto_scan_after_motion is True
    assert [item["command"] for item in runtime.simulator.history] == ["turn", "scan"]
    assert backend.calls == 1
    controller.close()


def test_compound_motion_stops_before_later_steps_after_collision() -> None:
    controller, runtime, _backend = _controller("not json", initial_scan=False)
    runtime.fail_forward_on_attempt = 3
    result = controller.handle_instruction(
        "Would you please drive forward 2 metres, then rotate right?"
    )
    assert result["success"] is False
    assert result["error_code"] == "E_COLLISION"
    assert result["planned_action_count"] == 6
    assert result["completed_action_count"] == 2
    assert len(result["actions"]) == 3
    assert runtime.position[1] == 1.0
    assert runtime.yaw == 0.0
    assert "stopped safely" in result["reply"]
    controller.close()


def test_oversized_repeated_motion_is_rejected_before_model_or_execution() -> None:
    controller, runtime, backend = _controller("unused", initial_scan=False)
    result = controller.handle_instruction("please move forward 1000 metres")
    assert result["success"] is False
    assert result["error_code"] == "E_PLAN_LIMIT"
    assert result["gemma_attempted"] is False
    assert result["fallback_used"] is False
    assert runtime.position[1] == 0.0
    assert backend.calls == 0
    controller.close()


def test_direct_ui_tool_is_double_validated_and_never_invokes_gemma() -> None:
    controller, runtime, backend = _controller(
        '{"tool":"turn","arguments":{"angle_degrees":10}}',
        initial_scan=False,
    )
    rejected = controller.dispatch_tool("turn", {"angle_degrees": 500})
    assert rejected["success"] is False
    assert rejected["error_code"] == "E_LIMIT"
    assert runtime.yaw == 0.0
    accepted = controller.dispatch_tool("turn", {"angle_degrees": -20})
    assert accepted["success"] is True
    assert accepted["gemma_attempted"] is False
    assert runtime.yaw == -20.0
    assert backend.calls == 0
    controller.close()


def test_question_uses_local_continuous_scene_answer_contract() -> None:
    controller, _runtime, backend = _controller("unused", initial_scan=False)
    result = controller.handle_instruction("What is around you?")
    assert result["reply"] == "a local continuous answer"
    assert result["control_mode"] == "local_gemma_continuous_scene_answer"
    assert result["gemma_attempted"] is True
    assert result["gemma_accepted"] is True
    assert backend.calls == 0
    controller.close()


def test_model_free_preflight_reports_actual_local_inputs() -> None:
    # The promoted default must itself satisfy the model-only control contract;
    # an explicit historical checkpoint here would let a stale default escape.
    result = practical_rover_preflight()
    assert result["ready"] is True
    assert result["loads_language_model"] is False
    assert result["renders_scene"] is False
    assert result["changes_robot_or_map_state"] is False
    assert result["model_id"] == "google/gemma-4-E2B-it"
    assert result["navigation_policy_task_trained"] is True
    assert result["navigation_policy_actual_gemma_causal_forward"] is True
    assert result["navigation_scene_token_count"] == 258
    assert result["navigation_checkpoint_sha256"] == (
        "149f5e04de1d8305e642909443f03b96894edc3ece67e4500eacec8f5ca81e7c"
    )
    assert result["navigation_history_feature_dim"] == 16
    assert result["navigation_history_parameterization"] == (
        "selected_action_parameters_goal_progress_v2"
    )
    assert result["every_scene_token_processed_per_navigation_decision"] is True
    assert result["every_map_voxel_influences_scene_prefix"] is True
    assert result["question_dependent_target_grounding"] is False
    assert result["model_selects_every_waypoint_and_heading"] is True
    assert result["model_selects_stop"] is True
    assert result["deterministic_route_planner_allowed_at_runtime"] is False
    assert result["fallback"] is None
    assert result["high_level_natural_language_only"] is True
    assert result["initial_scan"] is False
    assert result["auto_scan_after_motion"] is False
    assert result["untrained_json_backend_enabled"] is False
    assert result["environmental_text_inputs"] == []


class FakeGemmaWaypointGoal:
    def __init__(self, instruction: str) -> None:
        self.instruction = instruction

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "semantic_3d_chat.gemma_waypoint_goal.v1",
            "success": True,
            "termination": "model_stop",
            "error_code": None,
            "instruction_sha256": hashlib.sha256(
                self.instruction.encode("utf-8")
            ).hexdigest(),
            "checkpoint_sha256": "b" * 64,
            "model_stop_emitted": True,
            "step_count": 1,
            "steps": [
                {
                    "schema": "semantic_3d_chat.gemma_waypoint_step.v2",
                    "step": 1,
                    "model_action": "stop",
                    "model_waypoint_delta_robot_m": [0.0, 0.0],
                    "model_desired_heading_degrees": 0.0,
                    "primitive_tool": "stop",
                    "primitive_arguments": {},
                    "execution": {
                        "success": True,
                        "executed": True,
                        "substitution_applied": False,
                        "synthetic_stop_applied": False,
                    },
                    "actual_gemma_causal_forward": True,
                    "model_selected_every_waypoint_and_heading": True,
                    "deterministic_route_planner_used": False,
                    "substitution_applied": False,
                    "synthetic_stop_applied": False,
                }
            ],
            "synthetic_stop_applied": False,
            "substitution_applied": False,
            "deterministic_route_planner_used": False,
        }


class FakeGemmaWaypointController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.checkpoint_sha256 = "b" * 64
        self.metadata = {"gemma_runtime_binding_sha256": "c" * 64}

    def run(self, instruction: str, *, max_steps: int) -> FakeGemmaWaypointGoal:
        self.calls.append((instruction, max_steps))
        return FakeGemmaWaypointGoal(instruction)


@pytest.mark.parametrize(
    "instruction",
    [
        "Do a lap around the room",
        "Stop between the chair and the table",
        "Face the floor lamp",
        "Move forward 1 metre then turn left",
        "Please take me to the floor lamp",
        "Tour the room",
        "Between the chair and the table",
    ],
)
def test_production_navigation_routes_raw_text_only_to_gemma_waypoint_closed_loop(
    instruction: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.robot import practical_rover

    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    waypoint = FakeGemmaWaypointController()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("legacy deterministic navigation became reachable")

    monkeypatch.setattr(practical_rover, "execute_numeric_patrol", forbidden)
    monkeypatch.setattr(practical_rover, "execute_semantic_between_goal", forbidden)
    monkeypatch.setattr(practical_rover, "execute_grounded_goal_fallback", forbidden)
    monkeypatch.setattr(practical_rover, "execute_trained_goal", forbidden)
    monkeypatch.setattr(practical_rover, "parse_semantic_goal", forbidden)
    monkeypatch.setattr(practical_rover, "parse_navigation_instruction", forbidden)
    monkeypatch.setattr(practical_rover, "should_offer_llm_tool_policy", forbidden)
    controller = PracticalRoverController(
        runtime,
        FakeSemanticAgent(runtime),  # type: ignore[arg-type]
        _config(),
        gemma_waypoint_controller=waypoint,  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    startup = controller.startup()
    result = controller.handle_instruction(instruction)

    assert waypoint.calls == [(instruction, 128)]
    assert startup["control_mode"] == (
        "actual_local_gemma_model_only_waypoint_policy"
    )
    assert startup["model_selects_every_waypoint_and_heading"] is True
    assert startup["model_selects_stop"] is True
    assert startup["deterministic_route_planner_used"] is False
    assert result["control_mode"] == (
        "actual_local_gemma_model_only_waypoint_policy"
    )
    assert result["model_decisions"][0]["model_action"] == "stop"
    assert result["gemma_attempted"] is True
    assert result["gemma_accepted"] is True
    assert result["fallback_used"] is False
    assert result["deterministic_route_planner_used"] is False
    assert result["synthetic_stop_applied"] is False
    assert result["model_stop_emitted"] is True
    assert result["initial_scan_performed"] is False
    assert controller.semantic_agent.calls == []
    assert list(runtime.simulator.history) == []
    controller.close()


def test_production_scene_question_does_not_enter_waypoint_loop() -> None:
    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    waypoint = FakeGemmaWaypointController()
    controller = PracticalRoverController(
        runtime,
        FakeSemanticAgent(runtime),  # type: ignore[arg-type]
        _config(),
        gemma_waypoint_controller=waypoint,  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    result = controller.handle_instruction("What is hanging on the wall?")

    assert waypoint.calls == []
    assert result["reply"] == "a local continuous answer"
    assert result["control_mode"] == "local_gemma_continuous_scene_answer"
    assert controller.semantic_agent.calls == []
    assert controller.semantic_agent.answer_calls == ["What is hanging on the wall?"]
    controller.close()


def test_production_navigation_fails_closed_on_substitution_attestation() -> None:
    class SubstitutionGoal(FakeGemmaWaypointGoal):
        def as_dict(self) -> dict[str, Any]:
            result = super().as_dict()
            result["substitution_applied"] = True
            return result

    class SubstitutionController(FakeGemmaWaypointController):
        def run(self, instruction: str, *, max_steps: int) -> SubstitutionGoal:
            self.calls.append((instruction, max_steps))
            return SubstitutionGoal(instruction)

    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    controller = PracticalRoverController(
        runtime,
        FakeSemanticAgent(runtime),  # type: ignore[arg-type]
        _config(),
        gemma_waypoint_controller=SubstitutionController(),  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    with pytest.raises(RuntimeError, match="no-substitution contract"):
        controller.handle_instruction("Do a lap around the room")

    assert list(runtime.simulator.history) == []
    controller.close()


def test_model_only_controller_identity_tamper_cannot_reach_legacy_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.robot import practical_rover

    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    waypoint = FakeGemmaWaypointController()
    controller = PracticalRoverController(
        runtime,
        FakeSemanticAgent(runtime),  # type: ignore[arg-type]
        _config(),
        gemma_waypoint_controller=waypoint,  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("legacy planner became reachable after controller tamper")

    monkeypatch.setattr(practical_rover, "execute_numeric_patrol", forbidden)
    controller.gemma_waypoint_controller = None

    with pytest.raises(
        RuntimeError, match="model-only Gemma controller identity changed"
    ):
        controller.handle_instruction("Do a lap around the room")

    assert waypoint.calls == []
    assert list(runtime.simulator.history) == []
    controller.gemma_waypoint_controller = waypoint  # restore for audited close
    controller.close()


def test_waypoint_controller_requires_exclusive_high_level_mode() -> None:
    runtime = FakeRuntime()
    waypoint = FakeGemmaWaypointController()

    with pytest.raises(ValueError, match="requires the exclusive high-level mode"):
        PracticalRoverController(
            runtime,
            FakeSemanticAgent(runtime),  # type: ignore[arg-type]
            _config(),
            gemma_waypoint_controller=waypoint,  # type: ignore[arg-type]
            high_level_only=False,
        )

    runtime.auto_scan_after_motion = False
    with pytest.raises(ValueError, match="cannot coexist with a legacy goal planner"):
        PracticalRoverController(
            runtime,
            FakeSemanticAgent(runtime),  # type: ignore[arg-type]
            _config(),
            trained_goal_policy=object(),  # type: ignore[arg-type]
            gemma_waypoint_controller=waypoint,  # type: ignore[arg-type]
            high_level_only=True,
        )


def test_successful_model_only_goal_requires_executed_gemma_stop() -> None:
    class NoStopGoal(FakeGemmaWaypointGoal):
        def as_dict(self) -> dict[str, Any]:
            result = super().as_dict()
            terminal = result["steps"][-1]
            terminal["model_action"] = "move_to"
            terminal["primitive_tool"] = "move_to"
            terminal["primitive_arguments"] = {"x": 0.0, "y": 0.0}
            return result

    class NoStopController(FakeGemmaWaypointController):
        def run(self, instruction: str, *, max_steps: int) -> NoStopGoal:
            self.calls.append((instruction, max_steps))
            return NoStopGoal(instruction)

    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    controller = PracticalRoverController(
        runtime,
        FakeSemanticAgent(runtime),  # type: ignore[arg-type]
        _config(),
        gemma_waypoint_controller=NoStopController(),  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    with pytest.raises(RuntimeError, match="lacks an executed Gemma STOP"):
        controller.handle_instruction("Do a lap around the room")

    assert list(runtime.simulator.history) == []
    controller.close()


def test_high_level_target_goal_uses_task_trained_static_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.robot import practical_rover

    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    runtime.simulator.settings = {"auto_scan_after_motion": False}
    trained_bundle = object()
    observed: dict[str, Any] = {}

    def fake_execute(
        selected_runtime: Any,
        selected_bundle: Any,
        *,
        kind: str,
        target_text: str,
    ) -> dict[str, Any]:
        observed.update(
            runtime=selected_runtime,
            bundle=selected_bundle,
            kind=kind,
            target_text=target_text,
        )
        receipt = selected_runtime.turn(12.0)
        return {
            "kind": "navigation",
            "command": kind,
            "success": True,
            "error_code": None,
            "action_receipts": [receipt],
        }

    monkeypatch.setattr(practical_rover, "execute_trained_goal", fake_execute)
    controller = PracticalRoverController(
        runtime,
        FakeSemanticAgent(runtime),  # type: ignore[arg-type]
        _config(),
        trained_goal_policy=trained_bundle,  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    startup = controller.startup()
    result = controller.handle_instruction("Please face the chair")

    assert startup["initial_scan_performed"] is False
    assert startup["high_level_natural_language_only"] is True
    assert startup["task_trained_navigation"] is True
    assert startup["untrained_json_backend_enabled"] is False
    assert observed == {
        "runtime": runtime,
        "bundle": trained_bundle,
        "kind": "face",
        "target_text": "chair",
    }
    assert result["success"] is True
    assert result["control_mode"] == (
        "task_trained_local_gemma_full_scene_goal_policy_v3"
    )
    assert result["gemma_attempted"] is True
    assert result["gemma_accepted"] is True
    assert runtime.yaw == 12.0
    assert runtime.version == 0
    controller.close()


def test_failed_trained_goal_falls_back_to_camera_free_all_voxel_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.robot import practical_rover

    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    runtime.simulator.settings = {"auto_scan_after_motion": False}
    semantic_agent = FakeSemanticAgent(runtime)
    trained_calls: list[str] = []
    fallback_calls: list[str] = []

    def fake_trained(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        trained_calls.append(kwargs["target_text"])
        return {
            "schema": "trained",
            "success": False,
            "error_code": "E_SENSOR_ACTION",
            "termination_reason": "sensor_action_rejected",
            "step_count": 1,
            "training_status": "task_trained",
            "camera_observations_during_goal": 0,
            "static_scene_prefix_unchanged": True,
            "action_receipts": [],
        }

    def fake_fallback(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        fallback_calls.append(kwargs["target_text"])
        receipt = runtime.turn(15.0)
        return {
            "kind": "navigation",
            "command": "semantic_grounded_face",
            "success": True,
            "error_code": None,
            "action_receipts": [receipt],
            "camera_observations_during_goal": 0,
            "static_scene_prefix_unchanged": True,
        }

    monkeypatch.setattr(practical_rover, "execute_trained_goal", fake_trained)
    monkeypatch.setattr(
        practical_rover,
        "execute_grounded_goal_fallback",
        fake_fallback,
    )
    controller = PracticalRoverController(
        runtime,
        semantic_agent,  # type: ignore[arg-type]
        _config(),
        trained_goal_policy=object(),  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    result = controller.handle_instruction("Face the floor lamp")

    assert trained_calls == ["floor lamp"]
    assert fallback_calls == ["floor lamp"]
    assert result["success"] is True
    assert result["fallback_used"] is True
    assert result["gemma_accepted"] is True
    assert result["control_mode"] == (
        "task_trained_policy_then_all_voxel_semantic_geometry_fallback"
    )
    assert "all-voxel semantic grounding" in result["reply"]
    assert runtime.version == 0
    controller.close()


def test_high_level_lap_uses_global_map_patrol_not_camera_or_qa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.robot import practical_rover

    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    runtime.simulator.settings = {"auto_scan_after_motion": False}
    observed: dict[str, Any] = {}

    def fake_patrol(selected_runtime: Any, selected_config: Any) -> dict[str, Any]:
        observed.update(runtime=selected_runtime, config=selected_config)
        receipts = [
            selected_runtime.move_to(0.4, 0.0),
            selected_runtime.move_to(0.0, 0.0),
        ]
        return {
            "kind": "navigation",
            "command": "semantic_map_patrol",
            "success": True,
            "error_code": None,
            "planned_action_count": 2,
            "completed_action_count": 2,
            "plan": {"path_length_m": 0.8},
            "action_receipts": receipts,
            "camera_observations_during_goal": 0,
            "static_scene_prefix_unchanged": True,
        }

    monkeypatch.setattr(practical_rover, "execute_numeric_patrol", fake_patrol)
    config = _config()
    controller = PracticalRoverController(
        runtime,
        FakeSemanticAgent(runtime),  # type: ignore[arg-type]
        config,
        trained_goal_policy=object(),  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    result = controller.handle_instruction("Do a lap around the room")

    assert observed == {"runtime": runtime, "config": config}
    assert result["success"] is True
    assert result["control_mode"] == "global_static_3d_map_patrol_planner"
    assert result["planned_action_count"] == 2
    assert "internally planned waypoints" in result["reply"]
    assert runtime.version == 0
    assert controller.semantic_agent.calls == []
    controller.close()


def test_high_level_between_uses_all_voxel_static_semantic_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.robot import practical_rover

    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    runtime.simulator.settings = {"auto_scan_after_motion": False}
    semantic_agent = FakeSemanticAgent(runtime)
    observed: dict[str, Any] = {}

    def fake_between(
        selected_runtime: Any,
        selected_config: Any,
        *,
        first_target_text: str,
        second_target_text: str,
        text_encoder: Any,
    ) -> dict[str, Any]:
        observed.update(
            runtime=selected_runtime,
            config=selected_config,
            first=first_target_text,
            second=second_target_text,
            encoder=text_encoder,
        )
        receipt = selected_runtime.move_to(0.2, -0.1)
        return {
            "kind": "navigation",
            "command": "semantic_between",
            "success": True,
            "error_code": None,
            "action_receipts": [receipt],
            "planned_action_count": 1,
            "completed_action_count": 1,
            "camera_observations_during_goal": 0,
            "static_scene_prefix_unchanged": True,
        }

    monkeypatch.setattr(practical_rover, "execute_semantic_between_goal", fake_between)
    config = _config()
    controller = PracticalRoverController(
        runtime,
        semantic_agent,  # type: ignore[arg-type]
        config,
        trained_goal_policy=object(),  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    result = controller.handle_instruction("Stop between the chair and the table")

    assert observed == {
        "runtime": runtime,
        "config": config,
        "first": "chair",
        "second": "table",
        "encoder": semantic_agent.text_encoder,
    }
    assert result["success"] is True
    assert result["control_mode"] == "local_gemma_all_voxel_between_goal_planner"
    assert result["gemma_attempted"] is True
    assert result["gemma_accepted"] is True
    assert "between the two regions" in result["reply"]
    assert runtime.version == 0
    controller.close()


def test_high_level_mode_rejects_every_low_level_ui_or_chat_command() -> None:
    runtime = FakeRuntime()
    runtime.auto_scan_after_motion = False
    runtime.simulator.settings = {"auto_scan_after_motion": False}
    controller = PracticalRoverController(
        runtime,
        FakeSemanticAgent(runtime),  # type: ignore[arg-type]
        _config(),
        trained_goal_policy=object(),  # type: ignore[arg-type]
        high_level_only=True,
        initial_scan=False,
    )

    direct = controller.dispatch_tool("turn", {"angle_degrees": 15.0})
    scan = controller.handle_instruction("scan the room")
    turn = controller.handle_instruction("turn right 15 degrees")
    compound = controller.handle_instruction("move forward 1 metre then turn left")

    assert direct["success"] is False
    assert direct["error_code"] == "E_HIGH_LEVEL_ONLY"
    assert scan["success"] is False
    assert scan["error_code"] == "E_HIGH_LEVEL_ONLY"
    assert turn["error_code"] == "E_HIGH_LEVEL_ONLY"
    assert compound["error_code"] == "E_HIGH_LEVEL_ONLY"
    assert runtime.yaw == 0.0
    assert runtime.version == 0
    assert list(runtime.simulator.history) == []
    controller.close()
