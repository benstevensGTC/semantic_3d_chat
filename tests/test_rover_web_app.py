from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from semantic_3d_chat.robot.rover_web_app import (
    create_rover_web_app,
    serve_rover_web_app,
)


class FakeRoverSession:
    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.instructions: list[str] = []
        self.tools: list[tuple[str, dict[str, float]]] = []
        self.x = 0.0
        self.yaw = 0.0
        self.map_version = 1
        self.action_count = 0

    def _result(
        self,
        reply: str = "",
        actions: list[dict[str, Any]] | None = None,
        model_decisions: list[dict[str, Any]] | None = None,
    ) -> dict:
        gemma_attempted = model_decisions is not None
        return {
            "reply": reply,
            "state": {
                "scene_id": "scene_000001",
                "position_xy_m": [self.x, 0.0],
                "body_yaw_degrees": self.yaw,
                "camera_yaw_degrees": self.yaw,
                "pitch_degrees": 0.0,
                "collision": False,
                "stopped": False,
                "scan_coverage": 0.0,
                "scan_count": 0,
                "scene_version": self.map_version,
                "map_version": self.map_version,
                "action_count": self.action_count,
                "scene_prefix_hash": "a" * 64,
            },
            "actions": actions or [],
            "model_decisions": model_decisions or [],
            "scene_prefix_hash": "a" * 64,
            "map_version": self.map_version,
            "scene_memory": {
                "schema": "semantic_3d_chat.scene_memory_diagnostics.v1",
                "tensor_shape": [1, 258, 1536],
                "sha256": "a" * 64,
                "l2_norm": 512.25,
                "rms": 0.81,
                "token_count": 258,
                "model_dim": 1536,
                "active_tensor_shape": [1, 262, 1536],
                "active_sha256": "b" * 64,
                "active_l2_norm": 516.5,
                "robot_state_token_count": 4,
                "map_version": self.map_version,
                "source_voxels": 74_699,
                "processed_voxels": 10_421,
                "semantic_feature_dim": 3072,
                "all_runtime_voxels_encoded": True,
                "base_adapter_weights_loaded": True,
                "control_weights_loaded": True,
                "control_training_gate_passed": True,
                "question_dependent_scene_retrieval": False,
                "loaded_file_audit": {
                    "enabled": True,
                    "loaded_file_count": 5,
                    "loaded_file_inventory_sha256": "c" * 64,
                    "forbidden_access_count": 0,
                    "passed": True,
                },
                "environmental_text_inputs": [],
            },
            "control_mode": "actual_local_gemma_model_only_waypoint_policy",
            "navigation_control_mode": "actual_local_gemma_model_only_waypoint_policy",
            "navigation_checkpoint_sha256": "d" * 64,
            "gemma_runtime_binding_sha256": "e" * 64,
            "gemma_attempted": gemma_attempted,
            "gemma_accepted": gemma_attempted,
            "fallback_used": False,
            "local_inference": True,
            "cloud_model_used": False,
            "initial_scan_performed": False,
            "scene_memory_refreshed": False,
            "high_level_natural_language_only": True,
            "task_trained_navigation": True,
            "model_selects_every_waypoint_and_heading": True,
            "model_selects_stop": True,
            "deterministic_route_planner_used": False,
            "substitution_applied": False,
            "synthetic_stop_applied": False,
            "untrained_json_backend_enabled": False,
            "static_precomputed_scene_memory": True,
            "camera_control_input": False,
        }

    def startup(self) -> dict:
        self.started += 1
        return self._result("Rover ready")

    def handle_instruction(self, text: str) -> dict:
        self.instructions.append(text)
        self.x += 0.25
        self.action_count += 1
        return self._result(
            "I completed the outcome-level goal locally.",
            [
                {
                    "tool": "move_to",
                    "arguments": {"x": self.x, "y": 0.0},
                    "success": True,
                }
            ],
            [
                {
                    "step": 1,
                    "model_action": "move_to",
                    "model_action_logits": [8.0, -8.0, -8.0],
                    "model_action_probabilities": [1.0, 0.0, 0.0],
                    "decision_tensor_sha256": "f" * 64,
                    "instruction_sha256": "1" * 64,
                    "active_prefix_sha256": "2" * 64,
                    "scene_prefix_sha256": "a" * 64,
                    "robot_tokens_sha256": "3" * 64,
                    "checkpoint_sha256": "d" * 64,
                    "scene_token_count": 258,
                    "robot_token_count": 4,
                    "history_token_count": 0,
                    "prompt_token_count": 12,
                    "decision_position": 273,
                    "model_waypoint_delta_robot_m": [0.0, 0.25],
                    "model_turn_delta_degrees": 0.0,
                    "model_desired_heading_degrees": 0.0,
                    "primitive_tool": "move_to",
                    "primitive_arguments": {"x": self.x, "y": 0.0},
                    "execution": {
                        "success": True,
                        "executed": True,
                        "error_code": None,
                        "substitution_applied": False,
                        "synthetic_stop_applied": False,
                    },
                    "actual_gemma_causal_forward": True,
                    "model_selected_every_waypoint_and_heading": True,
                    "deterministic_route_planner_used": False,
                    "substitution_applied": False,
                    "synthetic_stop_applied": False,
                    "object_name": "must never cross the web boundary",
                },
                {
                    "step": 2,
                    "model_action": "stop",
                    "model_action_logits": [-8.0, -8.0, 8.0],
                    "model_action_probabilities": [0.0, 0.0, 1.0],
                    "decision_tensor_sha256": "4" * 64,
                    "instruction_sha256": "1" * 64,
                    "active_prefix_sha256": "b" * 64,
                    "scene_prefix_sha256": "a" * 64,
                    "robot_tokens_sha256": "5" * 64,
                    "checkpoint_sha256": "d" * 64,
                    "scene_token_count": 258,
                    "robot_token_count": 4,
                    "history_token_count": 1,
                    "prompt_token_count": 12,
                    "decision_position": 274,
                    "model_waypoint_delta_robot_m": [0.0, 0.0],
                    "model_turn_delta_degrees": 0.0,
                    "model_desired_heading_degrees": 0.0,
                    "primitive_tool": "stop",
                    "primitive_arguments": {},
                    "execution": {
                        "success": True,
                        "executed": True,
                        "error_code": None,
                        "substitution_applied": False,
                        "synthetic_stop_applied": False,
                    },
                    "actual_gemma_causal_forward": True,
                    "model_selected_every_waypoint_and_heading": True,
                    "deterministic_route_planner_used": False,
                    "substitution_applied": False,
                    "synthetic_stop_applied": False,
                },
            ],
        )

    def dispatch_tool(self, tool: str, arguments: dict[str, float]) -> dict:
        self.tools.append((tool, arguments))
        self.action_count += 1
        if tool == "turn":
            self.yaw += arguments["angle_degrees"]
        if tool == "scan":
            self.map_version += 1
        return self._result(
            "",
            [{"tool": tool, "arguments": arguments, "success": True}],
        )

    def close(self) -> None:
        self.closed += 1


def _write_visual(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-raster")


def test_rover_web_lifecycle_instruction_and_no_direct_tool_route(tmp_path: Path) -> None:
    figure_root = tmp_path / "reports/gemma4/figures"
    overview = figure_root / "scan_montage.png"
    map_preview = figure_root / "scene_000001/map_rgb.png"
    _write_visual(overview)
    _write_visual(map_preview)
    session = FakeRoverSession()
    app = create_rover_web_app(
        session,
        [6.0, 5.0, 3.0],
        visual_assets={"overview": overview, "map": map_preview},
        figure_root=figure_root,
    )

    with TestClient(app, base_url="http://127.0.0.1") as client:
        initial = client.get("/api/state")
        instruction = client.post(
            "/api/instruction", json={"instruction": "Do a lap around the room."}
        )
        direct_tool = client.post(
            "/api/tool",
            json={"tool": "turn", "arguments": {"angle_degrees": -30.0}},
        )
        asset = client.get("/assets/map")

    assert session.started == session.closed == 1
    assert initial.status_code == instruction.status_code == 200
    assert direct_tool.status_code == 404
    assert initial.json()["state"]["position_xy_m"] == [0.0, 0.0]
    assert instruction.json()["reply"] == "I completed the outcome-level goal locally."
    assert instruction.json()["state"]["position_xy_m"] == [0.25, 0.0]
    assert instruction.json()["state"]["map_version"] == 1
    decisions = instruction.json()["model_decisions"]
    assert decisions[0] == (
        {
            "step": 1,
            "model_action": "move_to",
            "model_action_logits": [8.0, -8.0, -8.0],
            "model_action_probabilities": [1.0, 0.0, 0.0],
            "scene_token_count": 258,
            "robot_token_count": 4,
            "history_token_count": 0,
            "prompt_token_count": 12,
            "decision_position": 273,
            "decision_tensor_sha256": "f" * 64,
            "instruction_sha256": "1" * 64,
            "active_prefix_sha256": "2" * 64,
            "scene_prefix_sha256": "a" * 64,
            "robot_tokens_sha256": "3" * 64,
            "checkpoint_sha256": "d" * 64,
            "model_waypoint_delta_robot_m": [0.0, 0.25],
            "model_turn_delta_degrees": 0.0,
            "model_desired_heading_degrees": 0.0,
            "derived_absolute_facing_heading_degrees": 0.0,
            "derived_world_waypoint_xy_m": [0.25, 0.0],
            "primitive_tool": "move_to",
            "primitive_arguments": {"x": 0.25, "y": 0.0},
            "accepted": True,
            "executed": True,
            "error_code": None,
            "actual_gemma_causal_forward": True,
            "model_selected_every_waypoint_and_heading": True,
            "deterministic_route_planner_used": False,
            "substitution_applied": False,
            "synthetic_stop_applied": False,
        }
    )
    assert decisions[1]["step"] == 2
    assert decisions[1]["model_action"] == "stop"
    assert decisions[1]["primitive_tool"] == "stop"
    assert decisions[1]["accepted"] is True
    assert decisions[1]["executed"] is True
    assert "object_name" not in instruction.text
    assert instruction.json()["scene_memory"] == {
        "schema": "semantic_3d_chat.scene_memory_diagnostics.v1",
        "tensor_shape": [1, 258, 1536],
        "sha256": "a" * 64,
        "l2_norm": 512.25,
        "rms": 0.81,
        "token_count": 258,
        "model_dim": 1536,
        "active_tensor_shape": [1, 262, 1536],
        "active_sha256": "b" * 64,
        "active_l2_norm": 516.5,
        "robot_state_token_count": 4,
        "map_version": 1,
        "source_voxels": 74_699,
        "processed_voxels": 10_421,
        "semantic_feature_dim": 3072,
        "all_runtime_voxels_encoded": True,
        "base_adapter_weights_loaded": True,
        "control_weights_loaded": True,
        "control_training_gate_passed": True,
        "question_dependent_scene_retrieval": False,
        "loaded_file_audit": {
            "enabled": True,
            "loaded_file_count": 5,
            "loaded_file_inventory_sha256": "c" * 64,
            "forbidden_access_count": 0,
            "passed": True,
        },
        "environmental_text_inputs": [],
    }
    assert instruction.json()["control"] == {
        "control_mode": "actual_local_gemma_model_only_waypoint_policy",
        "navigation_control_mode": "actual_local_gemma_model_only_waypoint_policy",
        "navigation_checkpoint_sha256": "d" * 64,
        "gemma_runtime_binding_sha256": "e" * 64,
        "gemma_attempted": True,
        "gemma_accepted": True,
        "fallback_used": False,
        "local_inference": True,
        "cloud_model_used": False,
        "initial_scan_performed": False,
        "scene_memory_refreshed": False,
        "high_level_natural_language_only": True,
        "task_trained_navigation": True,
        "model_selects_every_waypoint_and_heading": True,
        "model_selects_stop": True,
        "deterministic_route_planner_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
        "untrained_json_backend_enabled": False,
        "static_precomputed_scene_memory": True,
        "camera_control_input": False,
    }
    assert session.instructions == ["Do a lap around the room."]
    assert session.tools == []
    assert asset.status_code == 200


def test_rover_web_fails_closed_on_tampered_scene_memory() -> None:
    class TamperedMemorySession(FakeRoverSession):
        def startup(self) -> dict:
            result = super().startup()
            result["scene_memory"]["sha256"] = "d" * 64
            return result

    session = TamperedMemorySession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    with (
        pytest.raises(ValueError, match="differs from the active map binding"),
        TestClient(app, base_url="http://127.0.0.1"),
    ):
        pass

    assert session.closed == 1


def test_rover_web_exposes_raw_face_delta_and_only_numeric_derivations() -> None:
    class FaceSession(FakeRoverSession):
        def handle_instruction(self, text: str) -> dict:
            result = super().handle_instruction(text)
            decision = result["model_decisions"][0]
            decision.update(
                {
                    "model_action": "face",
                    "model_action_logits": [-8.0, 8.0, -8.0],
                    "model_action_probabilities": [0.0, 1.0, 0.0],
                    "model_turn_delta_degrees": -12.25,
                    "model_desired_heading_degrees": 32.75,
                    "primitive_tool": "turn",
                    "primitive_arguments": {"angle_degrees": -12.25},
                }
            )
            return result

    app = create_rover_web_app(FaceSession(), [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/instruction", json={"instruction": "Face the requested target."}
        )

    assert response.status_code == 200
    decision = response.json()["model_decisions"][0]
    assert decision["model_turn_delta_degrees"] == pytest.approx(-12.25)
    assert decision["derived_absolute_facing_heading_degrees"] == pytest.approx(32.75)
    assert decision["derived_world_waypoint_xy_m"] is None
    assert decision["primitive_arguments"] == {"angle_degrees": -12.25}
    assert "object_name" not in response.text


def test_rover_web_rejects_face_execution_that_differs_from_gemma_output() -> None:
    class SubstitutedFaceSession(FakeRoverSession):
        def handle_instruction(self, text: str) -> dict:
            result = super().handle_instruction(text)
            decision = result["model_decisions"][0]
            decision.update(
                {
                    "model_action": "face",
                    "model_action_logits": [-8.0, 8.0, -8.0],
                    "model_action_probabilities": [0.0, 1.0, 0.0],
                    "model_turn_delta_degrees": 17.0,
                    "model_desired_heading_degrees": 17.0,
                    "primitive_tool": "turn",
                    "primitive_arguments": {"angle_degrees": 16.0},
                }
            )
            return result

    app = create_rover_web_app(SubstitutedFaceSession(), [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/instruction", json={"instruction": "Face the requested target."}
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": "Executed FACE turn differs from Gemma's raw turn delta"
    }


def test_rover_web_rejects_successful_navigation_without_model_decisions() -> None:
    class MissingDecisionsSession(FakeRoverSession):
        def handle_instruction(self, text: str) -> dict:
            result = super().handle_instruction(text)
            result["model_decisions"] = []
            return result

    app = create_rover_web_app(MissingDecisionsSession(), [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/instruction", json={"instruction": "Reach the requested outcome."}
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": "Successful Gemma navigation requires a nonempty model decision log"
    }


@pytest.mark.parametrize("terminal_kind", ["move", "rejected_stop"])
def test_rover_web_rejects_successful_navigation_without_executed_gemma_stop(
    terminal_kind: str,
) -> None:
    class InvalidTerminalSession(FakeRoverSession):
        def handle_instruction(self, text: str) -> dict:
            result = super().handle_instruction(text)
            if terminal_kind == "move":
                result["model_decisions"] = result["model_decisions"][:1]
            else:
                terminal = result["model_decisions"][-1]
                terminal["execution"].update(
                    {"success": False, "executed": False, "error_code": "E_MODEL_STOP"}
                )
            return result

    app = create_rover_web_app(InvalidTerminalSession(), [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/instruction", json={"instruction": "Reach the requested outcome."}
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": "Successful Gemma navigation must end in accepted, executed Gemma STOP"
    }


@pytest.mark.parametrize(
    ("flag", "unsafe_value"),
    (
        ("high_level_natural_language_only", False),
        ("task_trained_navigation", False),
        ("model_selects_every_waypoint_and_heading", False),
        ("model_selects_stop", False),
        ("deterministic_route_planner_used", True),
        ("fallback_used", True),
        ("substitution_applied", True),
        ("synthetic_stop_applied", True),
        ("untrained_json_backend_enabled", True),
        ("static_precomputed_scene_memory", False),
        ("camera_control_input", True),
    ),
)
def test_rover_web_fails_closed_on_unsafe_control_contract(
    flag: str,
    unsafe_value: bool,
) -> None:
    class UnsafeControlSession(FakeRoverSession):
        def startup(self) -> dict:
            result = super().startup()
            result[flag] = unsafe_value
            return result

    session = UnsafeControlSession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    with (
        pytest.raises(ValueError, match="high-level static-map control gate"),
        TestClient(app, base_url="http://127.0.0.1"),
    ):
        pass

    assert session.closed == 1


def test_rover_web_rejects_a_legacy_navigation_control_mode() -> None:
    class LegacyNavigationSession(FakeRoverSession):
        def startup(self) -> dict:
            result = super().startup()
            result["navigation_control_mode"] = (
                "high_level_task_trained_local_gemma_static_scene"
            )
            return result

    session = LegacyNavigationSession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    with (
        pytest.raises(ValueError, match="model-only navigation backend"),
        TestClient(app, base_url="http://127.0.0.1"),
    ):
        pass

    assert session.closed == 1


def test_rover_page_is_offline_responsive_control_surface() -> None:
    session = FakeRoverSession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        page = client.get("/")

    assert page.status_code == 200
    assert "Local Semantic 3D Rover" in page.text
    assert "Precomputed embedded 3D map" in page.text
    assert "Free-form high-level goal" in page.text
    assert "Gemma chooses every waypoint, turn, recovery, and STOP" in page.text
    assert "Send free-form goal to Gemma" in page.text
    assert "Patrol room" not in page.text
    assert "data-command" not in page.text
    assert "the rover camera is not a control input" in page.text
    assert "renderDecision" in page.text
    assert "Gemma raw turn" in page.text
    assert "deterministic absolute facing" in page.text
    assert "Gemma raw waypoint" in page.text
    assert "deterministic frame transform" in page.text
    assert "ACCEPTED · EXECUTED" in page.text
    assert "Gemma step" in page.text
    assert "model_action_probabilities" in page.text
    assert "causal context" in page.text
    assert "prompt tokens" in page.text
    assert "raw logits" in page.text
    assert "decision_tensor_sha256" in page.text
    assert "chair" not in page.text.casefold()
    assert "lamp" not in page.text.casefold()
    assert "Direct bounded controls" not in page.text
    assert "Action receipts" not in page.text
    assert "data-tool=" not in page.text
    assert "move_forward" not in page.text
    assert "camera-cone" not in page.text
    assert "sendTool" not in page.text
    assert "event.key.toLowerCase" not in page.text
    assert "https://" not in page.text
    assert "cdn." not in page.text
    assert page.headers["content-security-policy"].startswith("default-src 'self'")


def test_rover_web_rejects_tampered_raw_action_head_provenance() -> None:
    class TamperedHeadSession(FakeRoverSession):
        def handle_instruction(self, text: str) -> dict:
            result = super().handle_instruction(text)
            result["model_decisions"][0]["model_action_probabilities"] = [0.0, 1.0, 0.0]
            return result

    app = create_rover_web_app(TamperedHeadSession(), [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/instruction", json={"instruction": "Reach the requested outcome."}
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": "Gemma action probabilities differ from its raw logits"
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"tool": "delete_scene", "arguments": {}},
        {"tool": "turn", "arguments": {"angle_degrees": 91}},
        {"tool": "move_forward", "arguments": {"distance_meters": 0.75}},
        {"tool": "look", "arguments": {"yaw_delta_degrees": 90, "pitch_delta_degrees": 0}},
        {"tool": "move_to", "arguments": {"x": 3.1, "y": 0}},
        {"tool": "scan", "arguments": {"surprise": 1}},
    ],
)
def test_rover_web_has_no_direct_tool_dispatch_route(payload: dict) -> None:
    session = FakeRoverSession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/tool", json=payload)
    assert response.status_code == 404
    assert session.tools == []


def test_rover_web_rejects_bad_instructions_before_model_call() -> None:
    session = FakeRoverSession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        malformed = client.post(
            "/api/instruction",
            content=b"not-json",
            headers={"content-type": "application/json"},
        )
        empty = client.post("/api/instruction", json={"instruction": "  "})
        long = client.post("/api/instruction", json={"instruction": "x" * 2_049})
    assert malformed.status_code == empty.status_code == long.status_code == 400
    assert session.instructions == []


def test_rover_web_preserves_128_gemma_decisions_for_a_full_lap() -> None:
    class LongLapSession(FakeRoverSession):
        def handle_instruction(self, text: str) -> dict:
            result = super().handle_instruction(text)
            template = result["model_decisions"][0]
            terminal = result["model_decisions"][-1]
            result["model_decisions"] = [
                {**copy.deepcopy(template), "step": step}
                for step in range(1, 128)
            ]
            result["model_decisions"].append(
                {**copy.deepcopy(terminal), "step": 128}
            )
            result["actions"] = [
                {
                    "scene_id": "scene_000001",
                    "position_m": [0.25, 0.0, 0.0],
                    "body_yaw_degrees": 0.0,
                    "camera_yaw_degrees": 0.0,
                    "pitch_degrees": 0.0,
                    "collision": False,
                    "stopped": False,
                    "scene_version": 1,
                    "scan_count": 0,
                    "distance_moved": 0.0,
                    "success": True,
                }
                for _ in range(128)
            ]
            return result

    app = create_rover_web_app(LongLapSession(), [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/instruction", json={"instruction": "Do a complete lap."}
        )

    assert response.status_code == 200
    assert len(response.json()["model_decisions"]) == 128
    assert len(response.json()["actions"]) == 128


def test_rover_web_rejects_more_than_128_gemma_decisions() -> None:
    class UnboundedSession(FakeRoverSession):
        def handle_instruction(self, text: str) -> dict:
            result = super().handle_instruction(text)
            template = result["model_decisions"][0]
            result["model_decisions"] = [
                {**copy.deepcopy(template), "step": step}
                for step in range(1, 130)
            ]
            return result

    app = create_rover_web_app(UnboundedSession(), [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/instruction", json={"instruction": "Do a complete lap."}
        )

    assert response.status_code == 400
    assert response.json() == {"error": "Rover result contains too many Gemma decisions"}


def test_rover_web_close_is_idempotent() -> None:
    session = FakeRoverSession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        first = client.post("/api/close", json={})
        second = client.post("/api/close", json={})
    assert first.json() == second.json() == {"closed": True}
    assert session.closed == 1


def test_rover_web_closes_session_when_startup_fails() -> None:
    class FailingStartupSession(FakeRoverSession):
        def startup(self) -> dict:
            self.started += 1
            raise RuntimeError("startup failed")

    session = FailingStartupSession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    with (
        pytest.raises(RuntimeError, match="startup failed"),
        TestClient(app, base_url="http://127.0.0.1"),
    ):
        pass

    assert session.started == session.closed == 1


def test_rover_web_refuses_non_loopback_server(monkeypatch: pytest.MonkeyPatch) -> None:
    class NeverRun:
        pass

    with pytest.raises(ValueError, match="loopback"):
        serve_rover_web_app(NeverRun(), host="0.0.0.0", port=8770)  # type: ignore[arg-type]


def test_rover_web_rejects_cross_origin_and_simple_content_type_mutations() -> None:
    session = FakeRoverSession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    payload = '{"instruction":"Do a lap around the room."}'
    with TestClient(app, base_url="http://127.0.0.1:8770") as client:
        simple = client.post(
            "/api/instruction",
            content=payload,
            headers={"content-type": "text/plain"},
        )
        cross_origin = client.post(
            "/api/instruction",
            content=payload,
            headers={
                "content-type": "application/json",
                "origin": "https://attacker.example",
            },
        )
        wrong_port = client.post(
            "/api/instruction",
            content=payload,
            headers={
                "content-type": "application/json",
                "origin": "http://127.0.0.1:9999",
            },
        )
        accepted = client.post(
            "/api/instruction",
            content=payload,
            headers={
                "content-type": "application/json; charset=utf-8",
                "origin": "http://127.0.0.1:8770",
            },
        )

    assert simple.status_code == cross_origin.status_code == wrong_port.status_code == 400
    assert accepted.status_code == 200
    assert session.tools == []
    assert session.instructions == ["Do a lap around the room."]


def test_rover_web_rejects_non_loopback_host_before_dispatch() -> None:
    session = FakeRoverSession()
    app = create_rover_web_app(session, [6.0, 5.0, 3.0])
    with TestClient(app, base_url="http://127.0.0.1:8770") as client:
        state = client.get("/api/state", headers={"host": "attacker.example"})
        mutation = client.post(
            "/api/instruction",
            json={"instruction": "Do a lap around the room."},
            headers={"host": "attacker.example"},
        )

    assert state.status_code == mutation.status_code == 400
    assert session.tools == []


def test_rover_web_source_has_no_supervision_or_network_client_imports() -> None:
    source = Path("src/semantic_3d_chat/robot/rover_web_app.py").read_text(encoding="utf-8")
    assert "semantic_3d_chat.data" not in source
    assert "semantic_3d_chat.evaluation" not in source
    assert "import requests" not in source
    assert "import httpx" not in source
