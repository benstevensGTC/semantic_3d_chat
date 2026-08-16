from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot import conversation_cli


def _required_cli() -> list[str]:
    return [
        "--base-checkpoint",
        "numeric_base",
        "--runtime-asset",
        "opaque.blend",
        "--robot-state-checkpoint",
        "numeric_robot",
    ]


def _runtime() -> SimpleNamespace:
    base = SimpleNamespace(
        language=SimpleNamespace(hidden_size=1536),
        config={
            "language": {
                "model_id": "local/test-gemma",
                "revision": "1" * 40,
            }
        },
    )
    return SimpleNamespace(
        prefix_refresher=SimpleNamespace(runtime=base),
        get_robot_state=dict,
    )


def test_parser_selects_v1_v3_or_v4_and_rejects_untrained_plus_learned() -> None:
    v1 = conversation_cli._parser().parse_args(
        [
            *_required_cli(),
            "--navigation-policy-checkpoint",
            "sanitized_v1",
            "--navigation-policy-version",
            "1",
        ]
    )
    v3 = conversation_cli._parser().parse_args(
        [
            *_required_cli(),
            "--navigation-policy-checkpoint",
            "sanitized_v3",
        ]
    )
    v4 = conversation_cli._parser().parse_args(
        [
            *_required_cli(),
            "--navigation-policy-checkpoint",
            "sanitized_v4",
            "--navigation-policy-version",
            "4",
        ]
    )
    assert v1.navigation_policy_version == 1
    assert v3.navigation_policy_version == 3
    assert v4.navigation_policy_version == 4

    with pytest.raises(SystemExit):
        conversation_cli._parser().parse_args(
            [
                *_required_cli(),
                "--llm-tool-policy",
                "--navigation-policy-checkpoint",
                "sanitized_v3",
            ]
        )


def test_finite_result_output_parser_and_atomic_writer(tmp_path: Path) -> None:
    args = conversation_cli._parser().parse_args(
        [
            *_required_cli(),
            "--command",
            "Face the target, then stop.",
            "--result-output",
            str(tmp_path / "result.json"),
        ]
    )
    assert args.result_output == str(tmp_path / "result.json")
    payload = {
        "schema": "semantic_3d_chat.embodied_conversation_result.v1",
        "environmental_text_inputs": [],
        "turns": [{"success": True}],
    }
    destination = tmp_path / "nested" / "result.json"
    conversation_cli._atomic_json(destination, payload)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert list(destination.parent.glob("*.tmp")) == []


@pytest.mark.parametrize("version", [1, 3, 4])
def test_learned_backend_selection_uses_sanitized_loader_and_loaded_model_binding(
    monkeypatch: pytest.MonkeyPatch,
    version: int,
) -> None:
    controller = object()
    metadata = {
        "task_trained": True,
        "complete_scene_prefix_required": True,
        "every_scene_token_processed": True,
        "numeric_robot_tokens_required": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }
    calls: dict[str, Any] = {}

    def loader(checkpoint: Path, **kwargs: Any) -> tuple[object, dict[str, Any]]:
        calls["checkpoint"] = checkpoint
        calls.update(kwargs)
        return controller, metadata

    class V1Backend:
        def __init__(self, runtime: Any, loaded: Any, values: Any) -> None:
            calls["backend"] = (1, runtime, loaded, values)

    class V3Backend:
        def __init__(
            self,
            runtime: Any,
            loaded: Any,
            values: Any,
            config: Any,
        ) -> None:
            calls["backend"] = (3, runtime, loaded, values, config)

    class V4Backend:
        def __init__(
            self,
            runtime: Any,
            loaded: Any,
            values: Any,
            config: Any,
        ) -> None:
            calls["backend"] = (4, runtime, loaded, values, config)

    wrong_loader = lambda *args, **kwargs: pytest.fail("wrong loader selected")
    if version == 1:
        monkeypatch.setattr(conversation_cli, "load_navigation_policy_checkpoint", loader)
        monkeypatch.setattr(
            conversation_cli,
            "load_navigation_policy_v3_checkpoint",
            wrong_loader,
        )
    elif version == 3:
        monkeypatch.setattr(
            conversation_cli,
            "load_navigation_policy_checkpoint",
            wrong_loader,
        )
        monkeypatch.setattr(conversation_cli, "load_navigation_policy_v3_checkpoint", loader)
        monkeypatch.setattr(
            conversation_cli,
            "load_navigation_policy_v4_checkpoint",
            wrong_loader,
        )
    else:
        monkeypatch.setattr(
            conversation_cli,
            "load_navigation_policy_checkpoint",
            wrong_loader,
        )
        monkeypatch.setattr(
            conversation_cli,
            "load_navigation_policy_v3_checkpoint",
            wrong_loader,
        )
        monkeypatch.setattr(conversation_cli, "load_navigation_policy_v4_checkpoint", loader)
    monkeypatch.setattr(conversation_cli, "LearnedContinuousActionBackend", V1Backend)
    monkeypatch.setattr(conversation_cli, "SemanticGroundedActionBackendV3", V3Backend)
    monkeypatch.setattr(conversation_cli, "SemanticClearanceActionBackendV4", V4Backend)

    audit = conversation_cli._runtime_file_audit()
    backend, loaded_metadata = conversation_cli._load_navigation_backend(
        _runtime(),
        {"safe": True},
        "data_gemma4/checkpoints/navigation_policy_v3",
        version,
        audit=audit,
    )

    expected = V1Backend if version == 1 else V3Backend if version == 3 else V4Backend
    assert isinstance(backend, expected)
    assert loaded_metadata == metadata
    assert calls["expected_hidden_size"] == 1536
    assert calls["expected_model_id"] == "local/test-gemma"
    assert calls["expected_model_revision"] == "1" * 40
    assert calls["device"] == "cpu"
    assert calls["audit"] is audit
    assert calls["backend"][0] == version


def test_conversation_audit_blocks_every_oracle_and_training_root() -> None:
    audit = conversation_cli._runtime_file_audit()
    expected = (
        PROJECT_ROOT / "data" / "oracle",
        PROJECT_ROOT / "data" / "qa",
        PROJECT_ROOT / "data_gemma4" / "training",
        PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
    )
    assert set(audit.forbidden_roots) == {path.resolve() for path in expected}
    assert audit.forbidden_component_names == {
        "oracle",
        "qa",
        "training",
        "scorer_only",
    }
    for root in expected:
        with audit, pytest.raises(PermissionError, match="Blocked forbidden"):
            audit.record(root / "never_opened.json")


def test_startup_truthfully_reports_trained_v3_and_all_map_contract() -> None:
    runtime = SimpleNamespace(
        prefix_refresher=SimpleNamespace(
            runtime=SimpleNamespace(startup_summary=lambda: {"device": "cpu"})
        ),
        prefix_binding=lambda: {"active_prefix_sha256": "a" * 64},
    )
    metadata = {
        "task_trained": True,
        "complete_scene_prefix_required": True,
        "every_scene_token_processed": True,
        "numeric_robot_tokens_required": True,
        "continuous_semantic_grounding_required": True,
        "all_map_voxels_scored_for_grounding": True,
        "query_dependent_grounding_navigation_only": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }
    startup = conversation_cli._startup(
        runtime,
        "scene_000001",
        navigation_policy_checkpoint="data_gemma4/checkpoints/navigation_policy_v3",
        navigation_policy_version=3,
        navigation_policy_metadata=metadata,
        navigation_max_steps=9,
    )

    assert startup["llm_tool_policy"]["backend"] == "learned_navigation_v3"
    assert (
        startup["llm_tool_policy"]["training_status"]
        == "supervised_continuous_semantic_grounded_navigation_policy_v3"
    )
    policy = startup["navigation_policy"]
    assert policy["task_trained"] is True
    assert policy["all_map_voxels_scored_for_grounding"] is True
    assert policy["grounding_performed_at_startup"] is False
    assert policy["environmental_text_inputs"] == []
    assert policy["oracle_inputs_at_runtime"] is False
    assert startup["learned_navigation_closed_loop"]["max_steps"] == 9

    legacy = conversation_cli._startup(runtime, "scene_000001")
    assert legacy["llm_tool_policy"]["enabled"] is False
    assert legacy["llm_tool_policy"]["backend"] == "disabled"
    assert legacy["navigation_policy"]["enabled"] is False


def test_startup_reports_numeric_alignment_interlock_without_environment_text() -> None:
    runtime = SimpleNamespace(
        prefix_refresher=SimpleNamespace(
            runtime=SimpleNamespace(startup_summary=lambda: {"device": "cpu"})
        ),
        prefix_binding=lambda: {"active_prefix_sha256": "a" * 64},
    )
    backend = SimpleNamespace(
        numeric_alignment_interlock_summary=lambda: {
            "enabled": True,
            "deadband_degrees": 3.0,
            "stalled_turn_degrees": 1.0,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }
    )
    startup = conversation_cli._startup(
        runtime,
        "scene_000001",
        navigation_policy_checkpoint="numeric_checkpoint",
        navigation_policy_version=3,
        navigation_policy_metadata={"task_trained": True},
        navigation_backend=backend,
    )

    interlock = startup["navigation_policy"][
        "numeric_alignment_convergence_interlock"
    ]
    assert interlock["enabled"] is True
    assert interlock["environmental_text_inputs"] == []
    assert interlock["oracle_inputs_at_runtime"] is False
    assert "numeric_alignment_goal_converged" in startup["learned_navigation_closed_loop"][
        "termination_conditions"
    ]
    assert "numeric_approach_goal_converged" in startup["learned_navigation_closed_loop"][
        "termination_conditions"
    ]


class _LoopRuntime:
    def __init__(self, initial_hash: str) -> None:
        self.active_hash = initial_hash

    def prefix_binding(self) -> dict[str, Any]:
        return {"active_prefix_sha256": self.active_hash}


class _LoopBackend:
    def __init__(self) -> None:
        self.last_grounding: dict[str, Any] | None = None


class _LoopAgent:
    def __init__(self, commands: list[str], *, fail_at: int | None = None) -> None:
        self.commands = commands
        self.fail_at = fail_at
        self.calls = 0
        self.backend = _LoopBackend()
        self.runtime = _LoopRuntime("a" * 64)

    def handle(self, text: str) -> dict[str, Any]:
        assert text.startswith("Move")
        index = self.calls
        self.calls += 1
        selected_hash = self.runtime.active_hash
        next_hash = chr(ord("b") + index) * 64
        self.runtime.active_hash = next_hash
        success = index != self.fail_at
        self.backend.last_grounding = {
            "target_available": True,
            "map_sha256": "m" * 64,
            "query_embedding_sha256": "q" * 64,
            "scored_voxels": 321,
            "eligible_voxels": 120,
        }
        command = self.commands[min(index, len(self.commands) - 1)]
        return {
            "kind": "navigation",
            "command": command,
            "success": success,
            "request_sha256": "r" * 64,
            "action_receipts": [
                {
                    "success": success,
                    "distance_moved_m": 0.2,
                    "scene_version": index,
                }
            ],
            "prefix_binding": {"active_prefix_sha256": next_hash},
            "tool_selection": {
                "active_prefix_sha256": selected_hash,
                "training_status": (
                    "supervised_continuous_semantic_grounded_navigation_policy_v3"
                ),
            },
            "environmental_text_inputs": [],
        }


def test_learned_navigation_loops_until_stop_and_attests_prefix_refresh() -> None:
    agent = _LoopAgent(["turn", "move_forward", "stop"])
    result = conversation_cli._handle_conversation_turn(
        agent,  # type: ignore[arg-type]
        "Move closer to the target, then stop.",
        navigation_backend=agent.backend,
        navigation_max_steps=12,
    )

    assert result["success"] is True
    assert result["termination_reason"] == "stop"
    assert result["step_count"] == 3
    assert len(result["action_receipts"]) == 3
    assert result["prefix_refresh_verified"] is True
    assert all(
        step["continuous_grounding"]["all_map_voxels_scored"] is True
        for step in result["steps"]
    )
    assert all(
        item["scored_voxels"] == 321
        for item in result["continuous_grounding_attestations"]
    )
    serialized = str(result["continuous_grounding_attestations"]).casefold()
    assert "chair" not in serialized and "bowl" not in serialized


def test_learned_navigation_stops_at_cap_and_old_modes_remain_one_step() -> None:
    capped = _LoopAgent(["turn"])
    result = conversation_cli._handle_conversation_turn(
        capped,  # type: ignore[arg-type]
        "Move closer to the target.",
        navigation_backend=capped.backend,
        navigation_max_steps=2,
    )
    assert result["success"] is False
    assert result["termination_reason"] == "max_steps"
    assert result["step_count"] == 2
    assert capped.calls == 2

    old_mode = _LoopAgent(["turn"])
    one = conversation_cli._handle_conversation_turn(
        old_mode,  # type: ignore[arg-type]
        "Move forward 0.2 meters.",
        navigation_backend=None,
        navigation_max_steps=12,
    )
    assert one["command"] == "turn"
    assert old_mode.calls == 1

    learned_question = _LoopAgent(["turn"])
    learned_question.handle = lambda text: {"kind": "answer", "answer": text}  # type: ignore[method-assign]
    answer = conversation_cli._handle_conversation_turn(
        learned_question,  # type: ignore[arg-type]
        "What is around you?",
        navigation_backend=learned_question.backend,
        navigation_max_steps=12,
    )
    assert answer["kind"] == "answer"


def test_learned_navigation_stops_on_failed_action() -> None:
    agent = _LoopAgent(["move_forward"], fail_at=1)
    result = conversation_cli._handle_conversation_turn(
        agent,  # type: ignore[arg-type]
        "Move closer to the target.",
        navigation_backend=agent.backend,
        navigation_max_steps=12,
    )
    assert result["success"] is False
    assert result["termination_reason"] == "action_failure"
    assert result["step_count"] == 2
