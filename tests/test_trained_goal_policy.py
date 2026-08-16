from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.llm_tool_policy import (
    ToolPolicyDecision,
    validate_tool_call_text,
)
from semantic_3d_chat.robot.navigation_policy_v3 import TRAINING_STATUS
from semantic_3d_chat.robot.navigation_policy_v3_3 import (
    SemanticGroundedActionBackendV33,
)
from semantic_3d_chat.robot.trained_goal_policy import (
    TrainedGoalPolicyBundle,
    canonical_terminal_goal,
    execute_trained_goal,
    load_trained_goal_policy,
)


def _config() -> dict[str, Any]:
    return {
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "robot": {
            "radius_m": 0.25,
            "max_move_m": 0.5,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 45.0,
            "max_camera_yaw_offset_degrees": 90.0,
            "max_pitch_degrees": 45.0,
            "auto_scan_after_motion": False,
        },
    }


def _digest(version: int, kind: int) -> str:
    return f"{version + kind:064x}"[-64:]


class _GoalRuntime:
    def __init__(
        self,
        *,
        refresh_after_action: bool = True,
        mutate_scene_after_action: bool = False,
    ) -> None:
        self.robot_version = 0
        self.scene_version = 0
        self.scan_count = 0
        self.yaw = 0.0
        self.actions: list[str] = []
        self.refresh_after_action = refresh_after_action
        self.mutate_scene_after_action = mutate_scene_after_action
        self.simulator = SimpleNamespace(
            settings={"auto_scan_after_motion": False},
        )

    def prefix_binding(self) -> dict[str, Any]:
        return {
            "scene_id": "scene_000001",
            "map_version": self.scene_version,
            "map_sha256": _digest(self.scene_version, 4),
            "scene_prefix_sha256": _digest(self.scene_version, 1),
            "active_prefix_sha256": _digest(
                self.robot_version + self.scene_version * 100,
                2,
            ),
            "robot_state_sha256": _digest(self.robot_version, 3),
            "robot_tokens_sha256": _digest(self.robot_version, 5),
            "source_voxels": 17,
            "processed_voxels": 17,
        }

    def get_robot_state(self) -> dict[str, Any]:
        return {
            "success": True,
            "scene_id": "scene_000001",
            "scene_version": self.scene_version,
            "position_m": [0.0, 0.0, 1.2],
            "body_yaw_degrees": self.yaw,
            "camera_yaw_degrees": self.yaw,
            "pitch_degrees": 0.0,
            "collision": False,
            "scan_count": self.scan_count,
            "stopped": False,
            **self.prefix_binding(),
        }

    def turn(self, angle_degrees: float) -> dict[str, Any]:
        self.actions.append("turn")
        self.yaw += angle_degrees
        if self.refresh_after_action:
            self.robot_version += 1
        if self.mutate_scene_after_action:
            self.scene_version += 1
        return {"command": "turn", "success": True, **self.get_robot_state()}


class _GoalBackend:
    runtime_interlock_version = "v3.3"

    def __init__(self) -> None:
        self.last_grounding: dict[str, Any] | None = None


class _SequencePolicy:
    def __init__(
        self,
        runtime: _GoalRuntime,
        backend: _GoalBackend,
        actions: list[str],
        *,
        training_status: str = TRAINING_STATUS,
        grounding_delta: int = 0,
        context_mismatch: bool = False,
    ) -> None:
        self.runtime = runtime
        self.backend = backend
        self.actions = list(actions)
        self.config = _config()
        self.training_status = training_status
        self.grounding_delta = grounding_delta
        self.context_mismatch = context_mismatch
        self.scan_counts_at_selection: list[int] = []

    def select(self, instruction: str) -> ToolPolicyDecision:
        assert instruction == "Face the floor lamp, then stop."
        self.scan_counts_at_selection.append(self.runtime.scan_count)
        binding = self.runtime.prefix_binding()
        self.backend.last_grounding = {
            "target_available": True,
            "target_xyz_m": [1.0, 0.0, 1.0],
            "scored_voxels": binding["source_voxels"] + self.grounding_delta,
            "eligible_voxels": 10,
            "continuous_context_verified": True,
            "active_prefix_sha256": binding["active_prefix_sha256"],
            "scene_prefix_sha256": binding["scene_prefix_sha256"],
            "robot_tokens_sha256": binding["robot_tokens_sha256"],
        }
        action = self.actions.pop(0)
        if action == "turn":
            payload = {"tool": "turn", "arguments": {"angle_degrees": 15.0}}
        elif action == "scan":
            payload = {"tool": "scan", "arguments": {}}
        else:
            payload = {"tool": "stop", "arguments": {}}
        validation = validate_tool_call_text(
            json.dumps(payload),
            self.config,
            robot_state=self.runtime.get_robot_state(),
        )
        assert validation.call is not None and validation.error_code is None
        robot_hash = str(binding["robot_tokens_sha256"])
        if self.context_mismatch:
            robot_hash = "f" * 64
        return ToolPolicyDecision(
            call=validation.call,
            attempts=1,
            validation_errors=(),
            proposal_sha256=("a" * 64,),
            active_prefix_sha256=str(binding["active_prefix_sha256"]),
            scene_prefix_sha256=str(binding["scene_prefix_sha256"]),
            robot_tokens_sha256=robot_hash,
            fallback_policy="fail_closed",
            training_status=self.training_status,
        )


def _bundle(
    runtime: _GoalRuntime,
    *,
    actions: list[str],
    training_status: str = TRAINING_STATUS,
    grounding_delta: int = 0,
    context_mismatch: bool = False,
) -> tuple[TrainedGoalPolicyBundle, _SequencePolicy]:
    backend = _GoalBackend()
    policy = _SequencePolicy(
        runtime,
        backend,
        actions,
        training_status=training_status,
        grounding_delta=grounding_delta,
        context_mismatch=context_mismatch,
    )
    bundle = TrainedGoalPolicyBundle(
        policy=policy,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        metadata={
            "scene_token_count": 258,
            "robot_token_count": 4,
            "every_scene_token_processed": True,
        },
        checkpoint=PROJECT_ROOT / "data_gemma4/checkpoints/navigation_policy_v3",
    )
    return bundle, policy


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("face", "Face the floor lamp, then stop."),
        ("approach", "Approach the floor lamp, then stop."),
    ],
)
def test_canonical_terminal_goal_uses_only_user_target(
    kind: str,
    expected: str,
) -> None:
    assert canonical_terminal_goal(kind, "  the   floor lamp?! ") == expected  # type: ignore[arg-type]


def test_canonical_terminal_goal_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="target phrase"):
        canonical_terminal_goal("face", "   ")
    with pytest.raises(ValueError, match="Unsupported"):
        canonical_terminal_goal("orbit", "lamp")  # type: ignore[arg-type]


def test_goal_loop_reasons_before_camera_action_and_refreshes_after_motion() -> None:
    runtime = _GoalRuntime()
    bundle, policy = _bundle(runtime, actions=["turn", "stop"])

    result = execute_trained_goal(
        runtime,
        bundle,
        kind="face",
        target_text="the floor lamp",
    )

    assert result["success"] is True
    assert result["termination_reason"] == "goal_settled"
    assert runtime.actions == ["turn"]
    assert policy.scan_counts_at_selection == [0, 0]
    assert result["internal_sensor_actions_before_first_policy_decision"] == 0
    assert result["initial_scan_count"] == 0
    assert result["final_scan_count"] == 0
    assert result["camera_observations_during_goal"] == 0
    assert result["static_scene_prefix_unchanged"] is True
    assert result["all_target_groundings_scored_complete_map"] is True
    assert result["goal_settled_without_episode_stop_latch"] is True
    assert "the floor lamp" not in json.dumps(result)


@pytest.mark.parametrize(
    ("bundle_options", "expected_error"),
    [
        ({"context_mismatch": True}, "E_CONTEXT"),
        ({"training_status": "untrained_tool_selection_seam"}, "E_CONTEXT"),
        ({"grounding_delta": -1}, "E_GROUNDING"),
    ],
)
def test_goal_loop_rejects_unbound_untrained_or_partial_decisions_without_action(
    bundle_options: dict[str, Any],
    expected_error: str,
) -> None:
    runtime = _GoalRuntime()
    bundle, _policy = _bundle(runtime, actions=["turn"], **bundle_options)

    result = execute_trained_goal(
        runtime,
        bundle,
        kind="face",
        target_text="the floor lamp",
    )

    assert result["success"] is False
    assert result["error_code"] == expected_error
    assert runtime.actions == []


def test_goal_loop_fails_closed_when_motion_does_not_refresh_prefix() -> None:
    runtime = _GoalRuntime(refresh_after_action=False)
    bundle, _policy = _bundle(runtime, actions=["turn"])

    result = execute_trained_goal(
        runtime,
        bundle,
        kind="face",
        target_text="the floor lamp",
    )

    assert result["success"] is False
    assert result["error_code"] == "E_ROBOT_PREFIX_STALE"
    assert result["termination_reason"] == "robot_prefix_refresh_rejected"
    assert runtime.actions == ["turn"]


def test_goal_loop_rejects_camera_scan_proposal_without_executing_it() -> None:
    runtime = _GoalRuntime()
    bundle, _policy = _bundle(runtime, actions=["scan"])

    result = execute_trained_goal(
        runtime,
        bundle,
        kind="face",
        target_text="the floor lamp",
    )

    assert result["success"] is False
    assert result["error_code"] == "E_SENSOR_ACTION"
    assert result["termination_reason"] == "sensor_action_rejected"
    assert runtime.actions == []
    assert result["initial_prefix_binding"]["scene_prefix_sha256"] == result[
        "final_prefix_binding"
    ]["scene_prefix_sha256"]


def test_goal_loop_fails_closed_if_precomputed_scene_memory_changes() -> None:
    runtime = _GoalRuntime(mutate_scene_after_action=True)
    bundle, _policy = _bundle(runtime, actions=["turn"])

    result = execute_trained_goal(
        runtime,
        bundle,
        kind="face",
        target_text="the floor lamp",
    )

    assert result["success"] is False
    assert result["error_code"] == "E_STATIC_SCENE_CHANGED"
    assert result["termination_reason"] == "static_scene_changed"
    assert result["static_scene_prefix_unchanged"] is False


class _DiagnosticTextEncoder:
    output_dim = 1536

    def encode_queries(self, queries: list[str] | tuple[str, ...]) -> np.ndarray:
        output = np.zeros((len(queries), self.output_dim), dtype=np.float32)
        output[:, 0] = 1.0
        return output


def test_checkpoint_factory_loads_task_trained_v3_3_goal_policy() -> None:
    config = load_config("configs/runtime/embodied_live.yaml")
    language = SimpleNamespace(hidden_size=1536)
    base = SimpleNamespace(
        language=language,
        config={"language": dict(config["language"])},
    )
    runtime = SimpleNamespace(
        prefix_refresher=SimpleNamespace(runtime=SimpleNamespace(base=base)),
        get_robot_state=dict,
    )

    bundle = load_trained_goal_policy(
        runtime,
        config,
        text_encoder=_DiagnosticTextEncoder(),
    )

    assert isinstance(bundle.backend, SemanticGroundedActionBackendV33)
    assert bundle.metadata["task_trained"] is True
    assert bundle.metadata["scene_token_count"] == 258
    assert bundle.metadata["every_scene_token_processed"] is True
    assert sum(parameter.numel() for parameter in bundle.backend.controller.parameters()) == 1_201_034
    assert bundle.summary()["current_camera_observation_required_before_first_decision"] is False
    assert bundle.summary()["environmental_text_inputs"] == []
