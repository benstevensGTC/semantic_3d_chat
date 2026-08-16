from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.gemma_runtime_binding import (
    gemma_runtime_binding_sha256,
    raw_hf_gemma_runtime_binding,
)
from semantic_3d_chat.robot.gemma_waypoint_policy import (
    ACTION_NAMES,
    ActualGemmaWaypointPolicy,
    GemmaMotionAction,
    GemmaWaypointDecision,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    CHECKPOINT_ARCHITECTURE,
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_V2,
    HEADING_PARAMETERIZATION,
    GemmaWaypointClosedLoopController,
    load_gemma_waypoint_policy_checkpoint,
    robot_delta_to_world_xy,
)
from semantic_3d_chat.robot.state_encoder import NumericRobotState
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V1,
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V1,
    HISTORY_PARAMETERIZATION_V2,
)

_HIDDEN = 16
_CHECKPOINT = "c" * 64


def _gemma_binding() -> dict[str, Any]:
    return raw_hf_gemma_runtime_binding(
        model_id="fake/gemma-4",
        model_revision="fake-waypoint-revision",
        language_dtype="float32",
    )


def _config() -> dict[str, Any]:
    return {
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "robot": {
            "radius_m": 0.25,
            "max_move_m": 0.5,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
        },
    }


class _FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=_HIDDEN,
            num_hidden_layers=2,
            hidden_size_per_layer_input=4,
            vocab_size=128,
            pad_token_id=0,
            bos_token_id=2,
            use_bidirectional_attention=None,
        )
        self.ple = nn.Embedding(128, 8)

    def get_per_layer_inputs(
        self,
        token_ids: torch.Tensor,
        _embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.ple(token_ids).reshape(*token_ids.shape, 2, 4)


class _FakeBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeTextModel()


class _FakeGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, _HIDDEN)
        self.model = _FakeBase()
        self.config = SimpleNamespace(
            text_config=self.model.language_model.config,
            boi_token_id=10,
            image_token_id=11,
            eoi_token_id=12,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


def _backend() -> Gemma4PrefixBackend:
    tokenizer = SimpleNamespace(
        bos_token_id=2,
        pad_token_id=0,
        boi_token_id=10,
        image_token_id=11,
        eoi_token_id=12,
    )
    return Gemma4PrefixBackend(
        _FakeGemma(),
        tokenizer=tokenizer,
        model_revision="fake-waypoint-revision",
    )


class _CollisionMap:
    def __init__(self, blocked_x: float | None = None) -> None:
        self.blocked_x = blocked_x
        self.segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def segment_check(
        self,
        start: tuple[float, float],
        target: tuple[float, float],
    ) -> SimpleNamespace:
        normalized = (
            (float(start[0]), float(start[1])),
            (float(target[0]), float(target[1])),
        )
        self.segments.append(normalized)
        collision = self.blocked_x is not None and float(target[0]) >= self.blocked_x
        return SimpleNamespace(collision=collision, clearance_m=0.0 if collision else 1.0)


class _Runtime:
    def __init__(
        self,
        *,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        collision_map: _CollisionMap | None = None,
    ) -> None:
        self.auto_scan_after_motion = False
        self.x = x
        self.y = y
        self.yaw = yaw
        self.version = 0
        self.stopped = False
        self.stop_calls = 0
        self.calls: list[tuple[str, tuple[float, ...]]] = []
        self.simulator = SimpleNamespace(
            collision_map=collision_map if collision_map is not None else _CollisionMap()
        )
        generator = torch.Generator().manual_seed(412)
        self._base_prefix = torch.randn(1, 262, _HIDDEN, generator=generator)

    def _prefix(self) -> torch.Tensor:
        prefix = self._base_prefix.clone()
        prefix[:, -5:-1] += float(self.version)
        return prefix

    def prefix_binding(self) -> dict[str, Any]:
        active = self._prefix()
        scene = torch.cat((active[:, :257], active[:, -1:]), dim=1)
        robot = active[:, 257:261]
        return {
            "scene_id": "scene_000001",
            "map_version": 0,
            "map_sha256": "d" * 64,
            "source_voxels": 128,
            "processed_voxels": 64,
            "active_prefix_sha256": prefix_sha256(active),
            "scene_prefix_sha256": prefix_sha256(scene),
            "robot_tokens_sha256": prefix_sha256(robot),
        }

    def active_prefix_snapshot(self) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._prefix(), self.prefix_binding()

    def continuous_action_context_snapshot(
        self,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        active, binding = self.active_prefix_snapshot()
        return active, binding, self.get_robot_state()

    def get_robot_state(self) -> dict[str, Any]:
        return {
            "scene_id": "scene_000001",
            "position_m": [self.x, self.y, 0.0],
            "body_yaw_degrees": self.yaw,
            "camera_yaw_degrees": self.yaw,
            "pitch_degrees": 0.0,
            "stopped": self.stopped,
            **self.prefix_binding(),
        }

    def move_to(self, x: float, y: float) -> dict[str, Any]:
        self.calls.append(("move_to", (x, y)))
        self.x, self.y = float(x), float(y)
        self.version += 1
        return {"success": True, "error_code": None, **self.get_robot_state()}

    def turn(self, angle_degrees: float) -> dict[str, Any]:
        self.calls.append(("turn", (angle_degrees,)))
        self.yaw = (self.yaw + float(angle_degrees) + 180.0) % 360.0 - 180.0
        self.version += 1
        return {"success": True, "error_code": None, **self.get_robot_state()}

    def stop(self) -> dict[str, Any]:
        self.stop_calls += 1
        self.stopped = True
        return {"success": True, "error_code": None, **self.get_robot_state()}


def _policy_metadata(policy: ActualGemmaWaypointPolicy) -> dict[str, Any]:
    history_dim = policy.history_projector.feature_dim
    history_parameterization = {
        HISTORY_FEATURE_DIM_V1: HISTORY_PARAMETERIZATION_V1,
        HISTORY_FEATURE_DIM_V2: HISTORY_PARAMETERIZATION_V2,
    }[history_dim]
    return {
        "action_names": list(ACTION_NAMES),
        "model_selects_every_waypoint_and_heading": True,
        "deterministic_route_planner_allowed_at_runtime": False,
        "actual_gemma_causal_forward": True,
        "gemma_output_hidden_states": True,
        "complete_scene_prefix_required": True,
        "every_scene_token_processed": True,
        "numeric_state_and_history_required": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "history_dim": history_dim,
        "history_parameterization": history_parameterization,
        "scene_token_count": policy.scene_token_count,
        "robot_token_count": policy.robot_token_count,
        "hidden_size": policy.hidden_size,
        "max_history_tokens": policy.history_projector.max_history_tokens,
        "max_waypoint_step_m": policy.max_waypoint_step_m,
        "heading_parameterization": HEADING_PARAMETERIZATION,
        "max_turn_delta_degrees": policy.max_turn_delta_degrees,
    }


class _ScriptedPolicy(ActualGemmaWaypointPolicy):
    def __init__(
        self,
        decisions: list[tuple[str, tuple[float, float], float]],
        *,
        history_feature_dim: int = HISTORY_FEATURE_DIM_V1,
    ) -> None:
        super().__init__(
            hidden_size=_HIDDEN,
            scene_token_count=258,
            robot_token_count=4,
            history_feature_dim=history_feature_dim,
            max_history_tokens=16,
            head_hidden_dim=8,
            max_waypoint_step_m=0.5,
            max_turn_delta_degrees=45.0,
        )
        self.script = list(decisions)
        self.calls: list[dict[str, Any]] = []

    def decide(
        self,
        *,
        prefix_backend: Any,
        tokenizer: Any,
        active_scene_robot_prefix: torch.Tensor,
        instruction: str,
        history_features: torch.Tensor,
    ) -> GemmaWaypointDecision:
        del prefix_backend, tokenizer
        index = len(self.calls)
        if index >= len(self.script):
            raise RuntimeError("Scripted policy exhausted")
        action_name, delta, turn_delta_degrees = self.script[index]
        self.calls.append(
            {
                "instruction": instruction,
                "history": history_features.detach().clone(),
                "prefix_sha256": prefix_sha256(active_scene_robot_prefix),
            }
        )
        action_index = ACTION_NAMES.index(action_name)
        probabilities = [0.0, 0.0, 0.0]
        probabilities[action_index] = 1.0
        logits = [-8.0, -8.0, -8.0]
        logits[action_index] = 8.0
        return GemmaWaypointDecision(
            action=GemmaMotionAction(action_name),
            action_index=action_index,
            action_logits=tuple(logits),
            action_probabilities=tuple(probabilities),
            waypoint_delta_robot_m=delta,
            turn_delta_degrees=turn_delta_degrees,
            instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
            active_prefix_sha256=prefix_sha256(active_scene_robot_prefix),
            scene_token_count=258,
            robot_token_count=4,
            history_token_count=int(history_features.shape[1]),
            prompt_token_count=3,
            decision_position=1,
            actual_gemma_causal_forward=True,
        )


def _controller(
    runtime: _Runtime,
    policy: _ScriptedPolicy,
    *,
    metadata: dict[str, Any] | None = None,
) -> GemmaWaypointClosedLoopController:
    backend = _backend()
    return GemmaWaypointClosedLoopController(
        runtime=runtime,
        config=_config(),
        policy=policy,
        prefix_backend=backend,
        tokenizer=backend.tokenizer,
        checkpoint_sha256=_CHECKPOINT,
        checkpoint_metadata=_policy_metadata(policy) if metadata is None else metadata,
    )


def test_exact_coordinate_transform_does_not_plan_or_clamp() -> None:
    assert robot_delta_to_world_xy((1.0, 2.0), 0.0, (0.2, 0.3)) == pytest.approx(
        (1.2, 2.3)
    )
    assert robot_delta_to_world_xy((1.0, 2.0), 90.0, (0.2, 0.3)) == pytest.approx(
        (0.7, 2.2)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gemma_output_hidden_states", False),
        ("complete_scene_prefix_required", False),
        ("every_scene_token_processed", False),
        ("numeric_state_and_history_required", False),
        ("environmental_text_inputs", ["chair"]),
        ("oracle_inputs_at_runtime", True),
    ],
)
def test_closed_loop_rejects_noncontinuous_or_leaking_checkpoint_contract(
    field: str,
    value: object,
) -> None:
    runtime = _Runtime()
    policy = _ScriptedPolicy([("stop", (0.0, 0.0), 0.0)])
    metadata = _policy_metadata(policy)
    metadata[field] = value

    with pytest.raises(ValueError, match="metadata differs"):
        _controller(runtime, policy, metadata=metadata)

    assert policy.calls == []
    assert runtime.calls == []


def test_closed_loop_rejects_camera_driven_auto_scan_runtime() -> None:
    runtime = _Runtime()
    runtime.auto_scan_after_motion = True
    policy = _ScriptedPolicy([("stop", (0.0, 0.0), 0.0)])

    with pytest.raises(ValueError, match="auto_scan_after_motion=false"):
        _controller(runtime, policy)

    assert policy.calls == []
    assert runtime.calls == []


def test_closed_loop_requires_atomic_prefix_and_numeric_state_snapshot() -> None:
    runtime = _Runtime()
    runtime.continuous_action_context_snapshot = None  # type: ignore[assignment]
    policy = _ScriptedPolicy([("stop", (0.0, 0.0), 0.0)])

    with pytest.raises(TypeError, match="continuous_action_context_snapshot"):
        _controller(runtime, policy)

    assert policy.calls == []
    assert runtime.calls == []


def test_closed_loop_accepts_production_typed_atomic_numeric_state() -> None:
    class TypedAtomicRuntime(_Runtime):
        def continuous_action_context_snapshot(
            self,
        ) -> tuple[torch.Tensor, dict[str, Any], NumericRobotState]:
            active, binding = self.active_prefix_snapshot()
            return (
                active,
                binding,
                NumericRobotState(
                    position_m=(self.x, self.y, 0.0),
                    body_yaw_degrees=self.yaw,
                    camera_yaw_degrees=self.yaw,
                    pitch_degrees=0.0,
                    linear_velocity_xy_m=(0.0, 0.0),
                    angular_velocity_degrees=0.0,
                    collision=False,
                    last_movement_delta_m=(0.0, 0.0, 0.0),
                    scan_coverage=0.0,
                    stopped=self.stopped,
                ),
            )

    runtime = TypedAtomicRuntime()
    policy = _ScriptedPolicy([("stop", (0.0, 0.0), 0.0)])

    result = _controller(runtime, policy).run("Stop when you decide the goal is done.")

    assert result.success is True
    assert result.termination == "model_stop"
    assert result.receipts[-1].execution["numeric_tool_receipt"]["goal_settled"] is True
    assert runtime.stop_calls == 0


def test_forged_active_hash_cannot_hide_scene_slice_replacement() -> None:
    class SceneSliceReplacingRuntime(_Runtime):
        replace_scene_slice = False

        def active_prefix_snapshot(self) -> tuple[torch.Tensor, dict[str, Any]]:
            active, binding = super().active_prefix_snapshot()
            if self.replace_scene_slice:
                active[0, 0, 0] += 1.0
                binding["active_prefix_sha256"] = prefix_sha256(active)
            return active, binding

    runtime = SceneSliceReplacingRuntime()
    policy = _ScriptedPolicy([("stop", (0.0, 0.0), 0.0)])
    controller = _controller(runtime, policy)
    runtime.replace_scene_slice = True

    with pytest.raises(RuntimeError, match="replaced part of the fixed scene tensor"):
        controller.run("Use only the fixed scene memory.")

    assert policy.calls == []
    assert runtime.calls == []


def test_forged_active_hash_cannot_hide_robot_token_replacement() -> None:
    class RobotSliceReplacingRuntime(_Runtime):
        replace_robot_slice = False

        def active_prefix_snapshot(self) -> tuple[torch.Tensor, dict[str, Any]]:
            active, binding = super().active_prefix_snapshot()
            if self.replace_robot_slice:
                active[0, 257, 0] += 1.0
                binding["active_prefix_sha256"] = prefix_sha256(active)
            return active, binding

    runtime = RobotSliceReplacingRuntime()
    policy = _ScriptedPolicy([("stop", (0.0, 0.0), 0.0)])
    controller = _controller(runtime, policy)
    runtime.replace_robot_slice = True

    with pytest.raises(RuntimeError, match="robot tokens differ from their binding"):
        controller.run("Use only numeric robot state.")

    assert policy.calls == []
    assert runtime.calls == []
def test_fresh_model_decisions_control_move_face_and_goal_scoped_stop() -> None:
    instruction = "Choose every part of a lap around the room yourself."
    runtime = _Runtime()
    policy = _ScriptedPolicy(
        [
            ("move_to", (0.2, 0.3), 0.0),
            ("face", (0.0, 0.0), 30.0),
            ("stop", (0.0, 0.0), 30.0),
        ]
    )

    result = _controller(runtime, policy).run(instruction, max_steps=5)

    assert result.success is True
    assert result.termination == "model_stop"
    assert result.model_stop_emitted is True
    assert len(policy.calls) == 3
    assert [call["instruction"] for call in policy.calls] == [instruction] * 3
    assert [call["history"].shape for call in policy.calls] == [
        (1, 0, 12),
        (1, 1, 12),
        (1, 2, 12),
    ]
    assert len({call["prefix_sha256"] for call in policy.calls}) == 3
    assert runtime.calls[0][0] == "move_to"
    assert runtime.calls[0][1] == pytest.approx((0.2, 0.3))
    assert runtime.calls[1][0] == "turn"
    assert runtime.calls[1][1] == pytest.approx((30.0,))
    assert result.receipts[1].turn_delta_degrees == pytest.approx(30.0)
    assert result.receipts[1].desired_heading_degrees == pytest.approx(30.0)
    assert result.receipts[1].as_dict()["model_turn_delta_degrees"] == pytest.approx(
        30.0
    )
    assert result.receipts[1].as_dict()["model_desired_heading_degrees"] == pytest.approx(
        30.0
    )
    assert runtime.stop_calls == 0
    assert runtime.stopped is False
    assert result.receipts[-1].primitive_tool == "stop"
    assert result.receipts[-1].execution["executed_tool"] == "stop"
    assert result.receipts[-1].execution["synthetic_stop_applied"] is False
    assert all(receipt.bound_proposal is not None for receipt in result.receipts)

    # Model STOP settled only that goal; deterministic primitives remain usable.
    followup = runtime.move_to(0.0, 0.0)
    assert followup["success"] is True


def test_collision_rejection_is_history_zero_then_gemma_redecides() -> None:
    collision = _CollisionMap(blocked_x=0.3)
    runtime = _Runtime(collision_map=collision)
    policy = _ScriptedPolicy(
        [
            ("move_to", (0.4, 0.0), 37.0),
            ("move_to", (-0.2, 0.0), -22.0),
            ("stop", (0.3, -0.4), 11.0),
        ]
    )

    result = _controller(runtime, policy).run("Move safely, choosing the route.")

    assert result.success is True
    assert result.receipts[0].execution["error_code"] == "E_MODEL_COLLISION"
    assert result.receipts[0].execution["executed"] is False
    assert result.receipts[0].history_row[-1] == 0.0
    # A rejected MOVE preserves its active waypoint, but the inactive turn
    # head is canonicalized to the unchanged/current yaw only in history.
    assert result.receipts[0].waypoint_delta_robot_m == pytest.approx((0.4, 0.0))
    assert result.receipts[0].turn_delta_degrees == pytest.approx(37.0)
    assert result.receipts[0].history_row[7:9] == pytest.approx((0.8, 0.0))
    assert result.receipts[0].history_row[9:11] == pytest.approx((0.0, 1.0))
    assert policy.calls[1]["history"][0, -1, -1].item() == 0.0
    assert result.receipts[1].history_row[-1] == 1.0
    assert runtime.calls == [("move_to", (-0.2, 0.0))]
    assert collision.segments[0] == ((0.0, 0.0), (0.4, 0.0))
    assert runtime.stop_calls == 0


def test_v2_runtime_appends_only_receipt_derived_goal_progress() -> None:
    collision = _CollisionMap(blocked_x=0.3)
    runtime = _Runtime(collision_map=collision)
    policy = _ScriptedPolicy(
        [
            ("move_to", (0.2, 0.0), 0.0),
            ("move_to", (0.4, 0.0), 0.0),
            ("move_to", (-0.2, 0.2), 0.0),
            ("stop", (0.0, 0.0), 0.0),
        ],
        history_feature_dim=HISTORY_FEATURE_DIM_V2,
    )

    result = _controller(runtime, policy).run(
        "Choose the route and decide when to stop.", max_steps=4
    )

    assert result.success is True
    assert [call["history"].shape for call in policy.calls] == [
        (1, 0, HISTORY_FEATURE_DIM_V2),
        (1, 1, HISTORY_FEATURE_DIM_V2),
        (1, 2, HISTORY_FEATURE_DIM_V2),
        (1, 3, HISTORY_FEATURE_DIM_V2),
    ]
    first, rejected, recovered, stopped = result.receipts
    diagonal = math.hypot(6.0, 5.0)
    assert first.history_row[12:] == pytest.approx(
        (math.tanh(0.2 / 22.0), 0.0, 0.2 / diagonal, 0.0)
    )
    assert rejected.execution["success"] is False
    assert rejected.history_row[11] == 0.0
    assert rejected.history_row[12:] == pytest.approx(
        (
            math.tanh(0.2 / 22.0),
            0.0,
            0.2 / diagonal,
            math.tanh(1.0 / 16.0),
        )
    )
    recovered_path = 0.2 + math.hypot(0.2, 0.2)
    assert recovered.history_row[12:] == pytest.approx(
        (
            math.tanh(recovered_path / 22.0),
            math.tanh(0.02 / 30.0),
            0.2 / diagonal,
            0.0,
        )
    )
    assert stopped.history_row[12:] == pytest.approx(recovered.history_row[12:])
    assert result.as_dict()["deterministic_route_planner_used"] is False
    assert result.as_dict()["substitution_applied"] is False
    assert result.as_dict()["synthetic_stop_applied"] is False
    assert runtime.stop_calls == 0


def test_v2_goal_progress_resets_for_each_user_goal() -> None:
    runtime = _Runtime(x=0.4, y=-0.2)
    policy = _ScriptedPolicy(
        [
            ("stop", (0.0, 0.0), 0.0),
            ("stop", (0.0, 0.0), 0.0),
        ],
        history_feature_dim=HISTORY_FEATURE_DIM_V2,
    )
    controller = _controller(runtime, policy)

    first = controller.run("Choose whether the first goal is complete.")
    second = controller.run("Choose whether the second goal is complete.")

    assert first.receipts[0].history_row[12:] == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert second.receipts[0].history_row[12:] == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert policy.calls[0]["history"].shape == (1, 0, HISTORY_FEATURE_DIM_V2)
    assert policy.calls[1]["history"].shape == (1, 0, HISTORY_FEATURE_DIM_V2)


def test_runtime_history_canonicalizes_only_inactive_model_heads() -> None:
    runtime = _Runtime(yaw=10.0)
    policy = _ScriptedPolicy(
        [
            ("move_to", (0.2, 0.3), 17.0),
            ("face", (0.4, -0.4), 30.0),
            ("stop", (-0.3, 0.2), -15.0),
        ]
    )

    result = _controller(runtime, policy).run("Choose an exact short route.")

    move, face, stop = result.receipts
    # Raw Gemma outputs and exact execution provenance are untouched.
    assert move.waypoint_delta_robot_m == pytest.approx((0.2, 0.3))
    assert move.turn_delta_degrees == pytest.approx(17.0)
    assert face.waypoint_delta_robot_m == pytest.approx((0.4, -0.4))
    assert face.turn_delta_degrees == pytest.approx(30.0)
    assert stop.waypoint_delta_robot_m == pytest.approx((-0.3, 0.2))
    assert stop.turn_delta_degrees == pytest.approx(-15.0)
    assert [name for name, _arguments in runtime.calls] == ["move_to", "turn"]
    assert runtime.calls[0][1] == pytest.approx((0.144867, 0.330172), abs=1e-6)
    assert runtime.calls[1][1] == pytest.approx((30.0,))

    # MOVE keeps its active waypoint and uses current/result yaw as the
    # action-neutral heading. FACE keeps only its active requested heading.
    # STOP keeps neither inactive head and uses the current yaw.
    assert move.history_row[7:9] == pytest.approx((0.4, 0.6))
    assert move.history_row[9:11] == pytest.approx(
        (math.sin(math.radians(10.0)), math.cos(math.radians(10.0)))
    )
    assert face.history_row[7:9] == pytest.approx((0.0, 0.0))
    assert face.history_row[9:11] == pytest.approx(
        (math.sin(math.radians(40.0)), math.cos(math.radians(40.0)))
    )
    assert stop.history_row[7:9] == pytest.approx((0.0, 0.0))
    assert stop.history_row[9:11] == pytest.approx(
        (math.sin(math.radians(40.0)), math.cos(math.radians(40.0)))
    )
    assert torch.allclose(policy.calls[1]["history"][0, 0], torch.tensor(move.history_row))
    assert torch.allclose(policy.calls[2]["history"][0, 1], torch.tensor(face.history_row))


def test_relative_face_turn_is_exact_and_world_heading_wraps() -> None:
    runtime = _Runtime(yaw=170.0)
    policy = _ScriptedPolicy(
        [
            ("face", (0.0, 0.0), 20.0),
            ("stop", (0.0, 0.0), 0.0),
        ]
    )

    result = _controller(runtime, policy).run("Face where you judge best.")

    assert result.success is True
    first = result.receipts[0]
    assert first.turn_delta_degrees == pytest.approx(20.0)
    assert first.desired_heading_degrees == pytest.approx(-170.0)
    assert first.primitive_arguments == {"angle_degrees": 20.0}
    assert first.execution["success"] is True
    assert first.execution["substitution_applied"] is False
    assert runtime.calls == [("turn", (20.0,))]
    assert runtime.yaw == pytest.approx(-170.0)


def test_runtime_fails_closed_when_turn_receipt_claims_success_but_pose_differs() -> None:
    class SubstitutingTurnRuntime(_Runtime):
        def turn(self, angle_degrees: float) -> dict[str, Any]:
            self.calls.append(("turn", (angle_degrees,)))
            self.yaw = (self.yaw + float(angle_degrees) / 2.0 + 180.0) % 360.0 - 180.0
            self.version += 1
            return {"success": True, "error_code": None, **self.get_robot_state()}

    runtime = SubstitutingTurnRuntime()
    policy = _ScriptedPolicy([("face", (0.0, 0.0), 20.0)])

    with pytest.raises(RuntimeError, match="FACE differs from Gemma's exact turn delta"):
        _controller(runtime, policy).run("Face where you judge best.")

    assert runtime.calls == [("turn", (20.0,))]


def test_runtime_fails_closed_when_move_implicitly_changes_facing() -> None:
    class RotatingMoveRuntime(_Runtime):
        def move_to(self, x: float, y: float) -> dict[str, Any]:
            result = super().move_to(x, y)
            self.yaw = 5.0
            return result

    runtime = RotatingMoveRuntime()
    policy = _ScriptedPolicy([("move_to", (0.1, 0.2), 0.0)])

    with pytest.raises(
        RuntimeError,
        match="MOVE_TO changed facing without a Gemma FACE decision",
    ):
        _controller(runtime, policy).run("Choose the exact next waypoint.")


def test_runtime_fails_closed_if_motion_changes_static_scene_prefix() -> None:
    class SceneChangingRuntime(_Runtime):
        def prefix_binding(self) -> dict[str, Any]:
            result = super().prefix_binding()
            if self.version:
                result["scene_prefix_sha256"] = hashlib.sha256(
                    f"scene:{self.version}".encode()
                ).hexdigest()
            return result

    runtime = SceneChangingRuntime()
    policy = _ScriptedPolicy([("move_to", (0.1, 0.2), 0.0)])

    with pytest.raises(
        RuntimeError,
        match="Static scene prefix changed after a Gemma motion primitive",
    ):
        _controller(runtime, policy).run("Choose the exact next waypoint.")


def test_turn_delta_outside_model_head_bound_is_not_clamped_or_executed() -> None:
    runtime = _Runtime()
    policy = _ScriptedPolicy([("face", (0.0, 0.0), 45.01)])

    with pytest.raises(RuntimeError, match="violates its bounded head"):
        _controller(runtime, policy).run("Face where you judge best.")

    assert runtime.calls == []


def test_max_steps_is_failure_and_never_synthesizes_stop() -> None:
    runtime = _Runtime()
    policy = _ScriptedPolicy(
        [
            ("move_to", (0.1, 0.0), 0.0),
            ("move_to", (0.1, 0.0), 0.0),
        ]
    )

    result = _controller(runtime, policy).run("Keep moving.", max_steps=2)

    assert result.success is False
    assert result.termination == "max_steps"
    assert result.error_code == "E_MAX_STEPS"
    assert result.model_stop_emitted is False
    assert result.as_dict()["substitution_applied"] is False
    assert runtime.stop_calls == 0
    assert all(
        receipt.execution["synthetic_stop_applied"] is False
        for receipt in result.receipts
    )


def _checkpoint_metadata(
    policy: ActualGemmaWaypointPolicy,
    weights_sha256: str,
) -> dict[str, Any]:
    binding = _gemma_binding()
    history_dim = policy.history_projector.feature_dim
    schema, history_parameterization = {
        HISTORY_FEATURE_DIM_V1: (CHECKPOINT_SCHEMA, HISTORY_PARAMETERIZATION_V1),
        HISTORY_FEATURE_DIM_V2: (CHECKPOINT_SCHEMA_V2, HISTORY_PARAMETERIZATION_V2),
    }[history_dim]
    return {
        "schema": schema,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "action_names": list(ACTION_NAMES),
        "weights_sha256": weights_sha256,
        "saved_controller_tensors_only": True,
        "frozen_gemma_weights_saved": False,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "runtime_required_files": ["policy.safetensors", "runtime_metadata.json"],
        "model_id": "fake/gemma-4",
        "model_revision": "fake-waypoint-revision",
        "dataset_sha256": "d" * 64,
        "training_traces_sha256": "e" * 64,
        "training_sample_count": 9,
        "validation_sample_count": 3,
        "training_scene_count": 3,
        "validation_scene_count": 1,
        "scene_splits_disjoint": True,
        "scene_token_count": policy.scene_token_count,
        "robot_token_count": policy.robot_token_count,
        "hidden_size": policy.hidden_size,
        "state_dim": 18,
        "history_dim": history_dim,
        "history_parameterization": history_parameterization,
        "max_history_tokens": policy.history_projector.max_history_tokens,
        "context_token_count": 1,
        "head_hidden_dim": 8,
        "max_waypoint_step_m": policy.max_waypoint_step_m,
        "heading_parameterization": HEADING_PARAMETERIZATION,
        "max_turn_delta_degrees": policy.max_turn_delta_degrees,
        "history_projector_initialization_seed": (
            policy.history_projector.initialization_seed
        ),
        "numeric_heads_initialization_seed": policy.numeric_heads.initialization_seed,
        "action_refit_l2_weight": 0.0,
        "context_projection_frozen_during_training": True,
        "actual_gemma_causal_forward": True,
        "gemma_output_hidden_states": True,
        "complete_scene_prefix_required": True,
        "every_scene_token_processed": True,
        "numeric_state_and_history_required": True,
        "deterministic_route_planner_allowed_at_runtime": False,
        "model_selects_every_waypoint_and_heading": True,
        "gemma_runtime_binding": binding,
        "gemma_runtime_binding_sha256": gemma_runtime_binding_sha256(binding),
    }


def test_loader_accepts_only_sealed_v2_schema_dimension_parameterization_pair(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint_v2"
    checkpoint.mkdir()
    policy = ActualGemmaWaypointPolicy(
        hidden_size=_HIDDEN,
        scene_token_count=258,
        robot_token_count=4,
        history_feature_dim=HISTORY_FEATURE_DIM_V2,
        max_history_tokens=16,
        head_hidden_dim=8,
        max_waypoint_step_m=0.5,
        max_turn_delta_degrees=45.0,
    )
    weights = checkpoint / "policy.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in policy.state_dict().items()
        },
        str(weights),
    )
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = _checkpoint_metadata(policy, digest)
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    backend = _backend()

    loaded = load_gemma_waypoint_policy_checkpoint(
        checkpoint,
        prefix_backend=backend,
        expected_model_id="fake/gemma-4",
        expected_model_revision="fake-waypoint-revision",
        expected_gemma_runtime_binding=_gemma_binding(),
    )
    assert loaded.metadata["schema"] == CHECKPOINT_SCHEMA_V2
    assert loaded.metadata["history_dim"] == HISTORY_FEATURE_DIM_V2
    assert loaded.metadata["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert loaded.policy.history_projector.feature_dim == HISTORY_FEATURE_DIM_V2

    for field, value in (
        ("schema", CHECKPOINT_SCHEMA),
        ("history_parameterization", HISTORY_PARAMETERIZATION_V1),
        ("history_dim", HISTORY_FEATURE_DIM_V1),
    ):
        tampered = dict(metadata)
        tampered[field] = value
        metadata_path.write_text(
            json.dumps(tampered, sort_keys=True), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="contract mismatch"):
            load_gemma_waypoint_policy_checkpoint(
                checkpoint,
                prefix_backend=backend,
                expected_model_id="fake/gemma-4",
                expected_model_revision="fake-waypoint-revision",
                expected_gemma_runtime_binding=_gemma_binding(),
            )


def test_loader_reads_only_sanitized_files_and_binds_model_identity(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    policy = ActualGemmaWaypointPolicy(
        hidden_size=_HIDDEN,
        scene_token_count=258,
        robot_token_count=4,
        history_feature_dim=12,
        max_history_tokens=16,
        head_hidden_dim=8,
        max_waypoint_step_m=0.5,
        max_turn_delta_degrees=45.0,
    )
    weights = checkpoint / "policy.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in policy.state_dict().items()
        },
        str(weights),
    )
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    (checkpoint / "runtime_metadata.json").write_text(
        json.dumps(_checkpoint_metadata(policy, digest), sort_keys=True),
        encoding="utf-8",
    )
    backend = _backend()

    loaded = load_gemma_waypoint_policy_checkpoint(
        checkpoint,
        prefix_backend=backend,
        expected_model_id="fake/gemma-4",
        expected_model_revision="fake-waypoint-revision",
        expected_gemma_runtime_binding=_gemma_binding(),
    )

    assert loaded.checkpoint_sha256 == digest
    assert loaded.policy.context_projection_frozen is True
    assert loaded.policy.training is False
    assert loaded.policy.max_turn_delta_degrees == pytest.approx(45.0)
    assert loaded.metadata["action_refit_l2_weight"] == pytest.approx(0.0)
    for name, expected in policy.state_dict().items():
        assert torch.equal(loaded.policy.state_dict()[name].cpu(), expected)

    with pytest.raises(ValueError, match="contract mismatch"):
        load_gemma_waypoint_policy_checkpoint(
            checkpoint,
            prefix_backend=backend,
            expected_model_id="wrong/model",
            expected_model_revision="fake-waypoint-revision",
            expected_gemma_runtime_binding=_gemma_binding(),
        )

    different_stack = raw_hf_gemma_runtime_binding(
        model_id="fake/gemma-4",
        model_revision="different-runtime-revision",
        language_dtype="float32",
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        load_gemma_waypoint_policy_checkpoint(
            checkpoint,
            prefix_backend=backend,
            expected_model_id="fake/gemma-4",
            expected_model_revision="fake-waypoint-revision",
            expected_gemma_runtime_binding=different_stack,
        )

    wrong_heading_contract = _checkpoint_metadata(policy, digest)
    wrong_heading_contract["heading_parameterization"] = "absolute_heading_sincos"
    (checkpoint / "runtime_metadata.json").write_text(
        json.dumps(wrong_heading_contract, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        load_gemma_waypoint_policy_checkpoint(
            checkpoint,
            prefix_backend=backend,
            expected_model_id="fake/gemma-4",
            expected_model_revision="fake-waypoint-revision",
            expected_gemma_runtime_binding=_gemma_binding(),
        )
    (checkpoint / "runtime_metadata.json").write_text(
        json.dumps(_checkpoint_metadata(policy, digest), sort_keys=True),
        encoding="utf-8",
    )

    legacy_without_refit_metadata = _checkpoint_metadata(policy, digest)
    legacy_without_refit_metadata.pop("action_refit_l2_weight")
    (checkpoint / "runtime_metadata.json").write_text(
        json.dumps(legacy_without_refit_metadata, sort_keys=True),
        encoding="utf-8",
    )
    legacy_loaded = load_gemma_waypoint_policy_checkpoint(
        checkpoint,
        prefix_backend=backend,
        expected_model_id="fake/gemma-4",
        expected_model_revision="fake-waypoint-revision",
        expected_gemma_runtime_binding=_gemma_binding(),
    )
    assert "action_refit_l2_weight" not in legacy_loaded.metadata

    for invalid_refit_weight in (-0.1, float("inf"), float("nan"), True, "0.1"):
        invalid_refit_metadata = _checkpoint_metadata(policy, digest)
        invalid_refit_metadata["action_refit_l2_weight"] = invalid_refit_weight
        (checkpoint / "runtime_metadata.json").write_text(
            json.dumps(invalid_refit_metadata, sort_keys=True),
            encoding="utf-8",
        )
        with pytest.raises((TypeError, ValueError), match="action_refit_l2_weight"):
            load_gemma_waypoint_policy_checkpoint(
                checkpoint,
                prefix_backend=backend,
                expected_model_id="fake/gemma-4",
                expected_model_revision="fake-waypoint-revision",
                expected_gemma_runtime_binding=_gemma_binding(),
            )
    (checkpoint / "runtime_metadata.json").write_text(
        json.dumps(_checkpoint_metadata(policy, digest), sort_keys=True),
        encoding="utf-8",
    )

    old_history_contract = _checkpoint_metadata(policy, digest)
    old_history_contract["schema"] = "semantic_3d_chat.gemma_waypoint_checkpoint.v2"
    old_history_contract.pop("history_parameterization")
    (checkpoint / "runtime_metadata.json").write_text(
        json.dumps(old_history_contract, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metadata fields changed"):
        load_gemma_waypoint_policy_checkpoint(
            checkpoint,
            prefix_backend=backend,
            expected_model_id="fake/gemma-4",
            expected_model_revision="fake-waypoint-revision",
            expected_gemma_runtime_binding=_gemma_binding(),
        )
    (checkpoint / "runtime_metadata.json").write_text(
        json.dumps(_checkpoint_metadata(policy, digest), sort_keys=True),
        encoding="utf-8",
    )

    (checkpoint / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two"):
        load_gemma_waypoint_policy_checkpoint(
            checkpoint,
            prefix_backend=backend,
            expected_model_id="fake/gemma-4",
            expected_model_revision="fake-waypoint-revision",
            expected_gemma_runtime_binding=_gemma_binding(),
        )
