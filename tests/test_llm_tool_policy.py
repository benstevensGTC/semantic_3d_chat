from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.conversation import (
    ConversationalEmbodiedAgent,
    should_offer_llm_tool_policy,
)
from semantic_3d_chat.robot.conversation_cli import _parser, _startup
from semantic_3d_chat.robot.llm_tool_policy import (
    ContinuousPrefixGemmaToolBackend,
    GeneratedToolProposal,
    LocalGemmaToolPolicy,
    ValidatedToolCall,
    execute_validated_tool_call,
    tool_protocol_system_prompt,
    validate_tool_call_text,
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
        "language": {},
    }


def test_embodied_cli_defaults_to_strict_static_prefix_and_control_is_opt_in() -> None:
    base = [
        "--base-checkpoint",
        "numeric_base",
        "--runtime-asset",
        "opaque.blend",
        "--robot-state-checkpoint",
        "numeric_robot",
    ]
    strict = _parser().parse_args(base)
    enhanced = _parser().parse_args([*base, "--control-checkpoint", "numeric_control"])
    assert strict.control_checkpoint is None
    assert enhanced.control_checkpoint == "numeric_control"

    runtime = SimpleNamespace(
        prefix_refresher=SimpleNamespace(
            runtime=SimpleNamespace(startup_summary=lambda: {"device": "cpu"})
        ),
        prefix_binding=lambda: {"active_prefix_sha256": "a" * 64},
    )
    startup = _startup(runtime, "scene_000001")
    assert startup["strict_fixed_environment_embedding_input"] is True
    assert startup["question_conditioned_scene_readout_tokens"] is False


def _state(**updates: Any) -> dict[str, Any]:
    result = {
        "scene_id": "scene_000001",
        "position_m": [0.0, 0.0, 0.0],
        "body_yaw_degrees": 0.0,
        "camera_yaw_degrees": 0.0,
        "pitch_degrees": 0.0,
    }
    result.update(updates)
    return result


@pytest.mark.parametrize(
    "payload",
    [
        {"tool": "get_robot_state", "arguments": {}},
        {
            "tool": "look",
            "arguments": {"yaw_delta_degrees": 10, "pitch_delta_degrees": -5},
        },
        {"tool": "turn", "arguments": {"angle_degrees": -45}},
        {"tool": "move_forward", "arguments": {"distance_meters": 0.5}},
        {"tool": "move_backward", "arguments": {"distance_meters": 0.25}},
        {"tool": "move_to", "arguments": {"x": 0.5, "y": -0.25}},
        {"tool": "scan", "arguments": {}},
        {"tool": "stop", "arguments": {}},
        {
            "tool": "reset_scene",
            "arguments": {"scene_id": "scene_000001", "seed": 7},
        },
    ],
)
def test_all_nine_tool_envelopes_validate(payload: dict[str, Any]) -> None:
    import json

    result = validate_tool_call_text(json.dumps(payload), _config(), robot_state=_state())
    assert result.valid
    assert result.call is not None
    assert result.call.as_dict() == payload
    assert len(result.call.call_sha256) == 64


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('```json\n{"tool":"scan","arguments":{}}\n```', "E_JSON"),
        ('{"tool":"scan","tool":"stop","arguments":{}}', "E_DUPLICATE_KEY"),
        ('{"tool":"scan","arguments":{},"extra":0}', "E_ENVELOPE"),
        ('{"tool":"unknown","arguments":{}}', "E_TOOL"),
        ('{"tool":"turn","arguments":{"angle_degrees":true}}', "E_NUMERIC"),
        ('{"tool":"turn","arguments":{"angle_degrees":NaN}}', "E_JSON"),
        ('{"tool":"turn","arguments":{"angle_degrees":46}}', "E_LIMIT"),
        ('{"tool":"scan","arguments":{"extra":1}}', "E_SCHEMA"),
        (
            '{"tool":"reset_scene","arguments":{"scene_id":"chair","seed":1}}',
            "E_SCENE_ID",
        ),
    ],
)
def test_strict_json_validation_rejects_malformed_without_repair(
    text: str,
    error: str,
) -> None:
    result = validate_tool_call_text(text, _config(), robot_state=_state())
    assert not result.valid
    assert result.call is None
    assert result.error_code == error


def test_dynamic_pose_limits_are_checked_before_execution() -> None:
    look = validate_tool_call_text(
        '{"tool":"look","arguments":{"yaw_delta_degrees":20,"pitch_delta_degrees":0}}',
        _config(),
        robot_state=_state(camera_yaw_degrees=50.0),
    )
    move = validate_tool_call_text(
        '{"tool":"move_to","arguments":{"x":1.1,"y":0}}',
        _config(),
        robot_state=_state(),
    )
    reset = validate_tool_call_text(
        '{"tool":"reset_scene","arguments":{"scene_id":"scene_000002","seed":1}}',
        _config(),
        robot_state=_state(),
    )
    assert look.error_code == "E_LIMIT"
    assert move.error_code == "E_LIMIT"
    assert reset.error_code == "E_SCENE_UNAVAILABLE"


class FakeProposalBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str, str | None]] = []

    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal:
        self.calls.append((instruction, correction_code))
        return GeneratedToolProposal(
            text=self.outputs.pop(0),
            active_prefix_sha256="a" * 64,
            scene_prefix_sha256="b" * 64,
            robot_tokens_sha256="c" * 64,
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=True,
        )


def _policy(
    outputs: list[str],
    *,
    max_retries: int = 1,
    fallback: str = "fail_closed",
) -> tuple[LocalGemmaToolPolicy, FakeProposalBackend]:
    backend = FakeProposalBackend(outputs)
    return (
        LocalGemmaToolPolicy(
            backend,
            _config(),
            robot_state_provider=_state,
            max_retries=max_retries,
            fallback_policy=fallback,  # type: ignore[arg-type]
        ),
        backend,
    )


def test_policy_retries_only_after_rejection_and_never_more_than_two() -> None:
    policy, backend = _policy(
        ["not json", '{"tool":"turn","arguments":{"angle_degrees":15}}'],
        max_retries=2,
    )
    decision = policy.select("Turn right")
    assert decision.accepted
    assert decision.call is not None and decision.call.name == "turn"
    assert decision.attempts == 2
    assert decision.validation_errors == ("E_JSON",)
    assert backend.calls == [("Turn right", None), ("Turn right", "E_JSON")]
    assert decision.audit_payload()["raw_model_output_logged"] is False

    exhausted, exhausted_backend = _policy(["x", "y", "z"], max_retries=2)
    rejected = exhausted.select("Turn right")
    assert rejected.call is None and rejected.attempts == 3
    assert len(exhausted_backend.calls) == 3
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        _policy(["x"], max_retries=3)


def test_missing_robot_tokens_or_nonlocal_backend_fails_closed() -> None:
    class MissingContext(FakeProposalBackend):
        def generate(self, instruction: str, *, correction_code: str | None):
            proposal = super().generate(instruction, correction_code=correction_code)
            return GeneratedToolProposal(
                **{
                    **proposal.__dict__,
                    "robot_tokens_sha256": None,
                    "used_continuous_robot_tokens": False,
                }
            )

    backend = MissingContext(['{"tool":"stop","arguments":{}}'])
    policy = LocalGemmaToolPolicy(
        backend,
        _config(),
        robot_state_provider=_state,
        max_retries=0,
    )
    decision = policy.select("Stop")
    assert decision.call is None
    assert decision.validation_errors == ("E_ROBOT_CONTEXT",)


def test_changed_runtime_context_is_rejected_without_retry_or_execution() -> None:
    backend = FakeProposalBackend(['{"tool":"stop","arguments":{}}'])
    changed_state = _state(
        active_prefix_sha256="d" * 64,
        scene_prefix_sha256="b" * 64,
        robot_tokens_sha256="c" * 64,
    )
    policy = LocalGemmaToolPolicy(
        backend,
        _config(),
        robot_state_provider=lambda: changed_state,
        max_retries=2,
    )
    decision = policy.select("Stop")
    assert decision.call is None
    assert decision.attempts == 1
    assert decision.validation_errors == ("E_CONTEXT_CHANGED",)


@dataclass
class FakeAnswer:
    answer: str = "continuous answer"
    grounding_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grounding_confidence: float = 0.5
    prefix_hash: str = "b" * 64


class FakeAgentRuntime:
    def __init__(self) -> None:
        self.actions: list[tuple[str, tuple[Any, ...]]] = []

    def _action(self, name: str, *arguments: Any) -> dict[str, Any]:
        self.actions.append((name, arguments))
        return {"success": True, "scene_id": "scene_000001", "scene_version": 0}

    def get_robot_state(self):
        return _state(success=True)

    def prefix_binding(self):
        return {
            "active_prefix_sha256": "a" * 64,
            "scene_prefix_sha256": "b" * 64,
            "robot_tokens_sha256": "c" * 64,
        }

    def answer(self, question: str):
        self.actions.append(("answer", (question,)))
        return FakeAnswer()

    def look(self, yaw: float, pitch: float):
        return self._action("look", yaw, pitch)

    def turn(self, angle: float):
        return self._action("turn", angle)

    def move_forward(self, distance: float):
        return self._action("move_forward", distance)

    def move_backward(self, distance: float):
        return self._action("move_backward", distance)

    def move_to(self, x: float, y: float):
        return self._action("move_to", x, y)

    def scan(self):
        return self._action("scan")

    def stop(self):
        return self._action("stop")

    def reset_scene(self, scene_id: str, seed: int):
        return self._action("reset_scene", scene_id, seed)


class UnusedTextEncoder:
    output_dim = 4


def _agent(runtime: FakeAgentRuntime, policy: LocalGemmaToolPolicy):
    return ConversationalEmbodiedAgent(
        runtime,  # type: ignore[arg-type]
        UnusedTextEncoder(),  # type: ignore[arg-type]
        room_size_m=(6.0, 5.0, 3.0),
        feature_start=0,
        feature_dim=4,
        tool_policy=policy,
    )


def test_agent_executes_one_validated_model_call_and_never_malformed_output() -> None:
    runtime = FakeAgentRuntime()
    valid, _ = _policy(['{"tool":"turn","arguments":{"angle_degrees":15}}'])
    result = _agent(runtime, valid).handle("Turn toward the target")
    assert runtime.actions == [("turn", (15,))]
    assert result["success"] is True
    assert result["tool_selection"]["used_continuous_scene_prefix"] is True
    assert result["tool_selection"]["used_continuous_robot_tokens"] is True

    blocked_runtime = FakeAgentRuntime()
    invalid, _ = _policy(["malformed", "still malformed"])
    blocked = _agent(blocked_runtime, invalid).handle("Turn toward the target")
    assert blocked_runtime.actions == []
    assert blocked["error_code"] == "E_TOOL_POLICY_REJECTED"
    assert blocked["action_receipts"] == []


def test_explicit_deterministic_fallback_and_question_bypass() -> None:
    runtime = FakeAgentRuntime()
    fallback, _ = _policy(
        ["malformed", "still malformed"],
        fallback="deterministic_parser",
    )
    agent = _agent(runtime, fallback)
    moved = agent.handle("Turn right 10 degrees")
    assert runtime.actions == [("turn", (10.0,))]
    assert moved["tool_selection_fallback_used"] is True

    answered = agent.handle("What is around you?")
    assert answered["answer"] == "continuous answer"
    assert runtime.actions[-1][0] == "answer"
    assert should_offer_llm_tool_policy("Get robot state")
    assert not should_offer_llm_tool_policy("Is anything nearby?")


def test_executor_refuses_unvalidated_objects() -> None:
    with pytest.raises(TypeError, match="ValidatedToolCall"):
        execute_validated_tool_call(
            FakeAgentRuntime(),
            {"tool": "stop"},  # type: ignore[arg-type]
            config=_config(),
        )

    forged = ValidatedToolCall(
        name="turn",
        arguments={"angle_degrees": 90},
        canonical_json='{"arguments":{"angle_degrees":90},"tool":"turn"}',
        call_sha256="0" * 64,
    )
    runtime = FakeAgentRuntime()
    with pytest.raises(ValueError, match="integrity"):
        execute_validated_tool_call(runtime, forged, config=_config())
    assert runtime.actions == []


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        self.messages = list(messages)
        return torch.tensor([[1, 2]], dtype=torch.long)

    def decode(self, token_ids, *, skip_special_tokens: bool):
        assert token_ids == [7, 8] and skip_special_tokens
        return '{"tool":"scan","arguments":{}}'


class FakeLanguage:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.device = torch.device("cpu")
        self.backend_name = "gemma4"
        self.received_prefix: torch.Tensor | None = None

    def generate_from_scene_prefix(self, scene_prefix, prompt_ids, **kwargs):
        assert prompt_ids.tolist() == [[1, 2]]
        assert kwargs["max_new_tokens"] == 32
        self.received_prefix = scene_prefix.detach().clone()
        return torch.tensor([[7, 8]], dtype=torch.long)


class FakeBase:
    def __init__(self) -> None:
        self.language = FakeLanguage()
        self.config = {"language": {}}
        self._generation_function = object()

    @staticmethod
    def _eos_token_ids():
        return 1


class FakeContinuousRuntime:
    def __init__(self) -> None:
        self.prefix = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
        self.base = FakeBase()
        self.prefix_refresher = SimpleNamespace(runtime=self.base)

    def active_prefix_snapshot(self):
        return self.prefix.clone(), {
            "active_prefix_sha256": prefix_sha256(self.prefix),
            "scene_prefix_sha256": "b" * 64,
            "robot_tokens_sha256": "c" * 64,
        }


def test_production_backend_uses_exact_continuous_prefix_and_environment_free_prompt() -> None:
    runtime = FakeContinuousRuntime()
    backend = ContinuousPrefixGemmaToolBackend(runtime, _config(), max_new_tokens=32)
    proposal = backend.generate("Scan now", correction_code=None)
    assert proposal.text == '{"tool":"scan","arguments":{}}'
    assert proposal.used_continuous_scene_prefix
    assert proposal.used_continuous_robot_tokens
    assert torch.equal(runtime.base.language.received_prefix, runtime.prefix)
    messages = runtime.base.language.tokenizer.messages
    assert messages[1] == {"role": "user", "content": "Scan now"}
    assert "chair" not in messages[0]["content"].casefold()
    assert "bowl" not in messages[0]["content"].casefold()
    protocol = tool_protocol_system_prompt(_config())
    for tool in (
        "get_robot_state",
        "look",
        "turn",
        "move_forward",
        "move_backward",
        "move_to",
        "scan",
        "stop",
        "reset_scene",
    ):
        assert tool in protocol
