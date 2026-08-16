from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend
from semantic_3d_chat.robot.gemma_waypoint_policy import (
    ACTION_NAMES,
    ActualGemmaWaypointPolicy,
    GemmaMotionAction,
    NumericHistoryTokenProjector,
)

_HIDDEN = 16
_SCENE_TOKENS = 258
_ROBOT_TOKENS = 4
_HISTORY_FEATURES = 12


class _FakeTokenizer:
    bos_token_id = 2
    pad_token_id = 0
    boi_token_id = 10
    image_token_id = 11
    eoi_token_id = 12

    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> torch.Tensor:
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        self.last_messages = [dict(message) for message in messages]
        instruction = messages[-1]["content"]
        digest = hashlib.sha256(instruction.encode("utf-8")).digest()
        instruction_token = 16 + digest[0] % 96
        return torch.tensor([[self.bos_token_id, 6, instruction_token]])


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
        self.ple_embedding = nn.Embedding(128, 8)

    def get_per_layer_inputs(
        self,
        token_ids: torch.Tensor,
        _embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.ple_embedding(token_ids).reshape(*token_ids.shape, 2, 4)


class _FakeGemmaBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeTextModel()


class _FakeGemma4(nn.Module):
    """Tiny causal stand-in; the production method still calls model.forward."""

    def __init__(self, *, return_hidden_states: bool = True) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9_441)
            self.embedding = nn.Embedding(128, _HIDDEN)
            self.model = _FakeGemmaBase()
        self.config = SimpleNamespace(
            text_config=self.model.language_model.config,
            boi_token_id=10,
            image_token_id=11,
            eoi_token_id=12,
        )
        self.return_hidden_states = return_hidden_states
        self.forward_calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        self.forward_calls += 1
        self.last_kwargs = kwargs
        inputs = kwargs["inputs_embeds"]
        # The final token depends causally on every earlier continuous and text
        # token. This is intentionally simple, deterministic test machinery.
        final_layer = torch.tanh(inputs.cumsum(dim=1))
        hidden_states = (inputs, final_layer) if self.return_hidden_states else None
        return SimpleNamespace(hidden_states=hidden_states)


def _backend(
    *,
    return_hidden_states: bool = True,
) -> tuple[Gemma4PrefixBackend, _FakeGemma4, _FakeTokenizer]:
    tokenizer = _FakeTokenizer()
    model = _FakeGemma4(return_hidden_states=return_hidden_states)
    backend = Gemma4PrefixBackend(
        model,
        tokenizer=tokenizer,
        model_revision="fake-pinned-gemma4-policy-revision",
    )
    return backend, model, tokenizer


def _active_prefix(backend: Gemma4PrefixBackend, seed: int = 72) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # BOI + 256 scene latents + 4 numeric robot tokens + EOI. The active
    # runtime already inserts robot state immediately before the EOI boundary.
    interior = torch.randn(1, 260, _HIDDEN, generator=generator) * 0.1
    boi, eoi = backend.native_boundary_embeddings()
    return torch.cat((boi, interior, eoi), dim=1)


def _policy(**kwargs: Any) -> ActualGemmaWaypointPolicy:
    return ActualGemmaWaypointPolicy(
        hidden_size=_HIDDEN,
        scene_token_count=_SCENE_TOKENS,
        robot_token_count=_ROBOT_TOKENS,
        history_feature_dim=_HISTORY_FEATURES,
        max_history_tokens=4,
        head_hidden_dim=8,
        **kwargs,
    )


def test_numeric_history_projector_shapes_empty_and_rejects_bad_input() -> None:
    projector = NumericHistoryTokenProjector(
        feature_dim=_HISTORY_FEATURES,
        hidden_size=_HIDDEN,
        intermediate_dim=8,
        max_history_tokens=4,
    )

    assert projector(torch.empty(2, 0, _HISTORY_FEATURES)).shape == (2, 0, _HIDDEN)
    assert projector(torch.randn(2, 3, _HISTORY_FEATURES)).shape == (2, 3, _HIDDEN)
    with pytest.raises(ValueError, match="shape"):
        projector(torch.randn(2, 3, _HISTORY_FEATURES - 1))
    with pytest.raises(ValueError, match="token limit"):
        projector(torch.randn(1, 5, _HISTORY_FEATURES))
    invalid = torch.zeros(1, 1, _HISTORY_FEATURES)
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        projector(invalid)


def test_context_controls_are_deterministic_and_frozen_by_default() -> None:
    torch.manual_seed(1)
    first = _policy()
    torch.manual_seed(98_765)
    second = _policy()

    assert first.context_projection_frozen is True
    assert second.context_projection_frozen is True
    assert torch.equal(first.decision_token, second.decision_token)
    for first_value, second_value in zip(
        first.history_projector.state_dict().values(),
        second.history_projector.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(first_value, second_value)
    for first_value, second_value in zip(
        first.numeric_heads.state_dict().values(),
        second.numeric_heads.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(first_value, second_value)
    assert all(parameter.requires_grad for parameter in first.numeric_heads.parameters())

    first.set_context_projection_trainable(True)
    assert first.context_projection_frozen is False
    assert first.decision_token.requires_grad is True
    assert all(
        parameter.requires_grad for parameter in first.history_projector.parameters()
    )


def test_actual_gemma_forward_consumes_full_prefix_instruction_and_history() -> None:
    backend, model, tokenizer = _backend()
    policy = _policy()
    prefix = _active_prefix(backend)
    history = torch.randn(1, 3, _HISTORY_FEATURES)
    instruction = "Do a full lap around the room, choosing the route yourself."

    output = policy.forward_actual_gemma(
        prefix_backend=backend,
        tokenizer=tokenizer,
        active_scene_robot_prefix=prefix,
        instruction=instruction,
        history_features=history,
    )

    assert output.action_logits.shape == (1, 3)
    assert output.waypoint_delta_robot_m.shape == (1, 2)
    assert bool((output.waypoint_delta_robot_m.abs() <= 0.5).all())
    assert output.turn_delta_degrees.shape == (1, 1)
    assert bool((output.turn_delta_degrees.abs() <= 40.0).all())
    assert output.decision_hidden.shape == (1, _HIDDEN)
    assert output.actual_gemma_causal_forward is True
    assert output.raw_instruction_included is True
    assert output.scene_token_count == _SCENE_TOKENS
    assert output.robot_token_count == _ROBOT_TOKENS
    assert output.history_token_count == 3
    assert output.prompt_token_count == 3
    assert output.decision_position == 268
    assert tokenizer.last_messages is not None
    assert tokenizer.last_messages[-1] == {"role": "user", "content": instruction}

    assert model.forward_calls == 1
    assert model.last_kwargs is not None
    assert model.last_kwargs["output_hidden_states"] is True
    assert model.last_kwargs["return_dict"] is True
    assert model.last_kwargs["use_cache"] is False
    assert model.last_kwargs["inputs_embeds"].shape == (1, 269, _HIDDEN)
    assert torch.equal(model.last_kwargs["inputs_embeds"][:, 1:263], prefix)


def test_cached_actual_gemma_hidden_reproduces_head_output() -> None:
    backend, _model, _tokenizer = _backend()
    policy = _policy()
    output = policy.forward_actual_gemma(
        prefix_backend=backend,
        tokenizer=backend.tokenizer,
        active_scene_robot_prefix=_active_prefix(backend),
        instruction="Move through the room without using a prescribed route.",
        history_features=torch.randn(1, 2, _HISTORY_FEATURES),
    )

    cached = policy.forward_heads_from_cached_gemma_hidden(output.decision_hidden)
    assert torch.equal(cached.action_logits, output.action_logits)
    assert torch.equal(
        cached.waypoint_delta_robot_m,
        output.waypoint_delta_robot_m,
    )
    assert torch.equal(
        cached.turn_delta_degrees,
        output.turn_delta_degrees,
    )


def test_scene_and_raw_instruction_change_actual_gemma_decision_hidden() -> None:
    backend, model, tokenizer = _backend()
    policy = _policy()
    history = torch.zeros(1, 0, _HISTORY_FEATURES)
    first_prefix = _active_prefix(backend, seed=1)
    second_prefix = _active_prefix(backend, seed=2)

    first = policy.forward_actual_gemma(
        prefix_backend=backend,
        tokenizer=tokenizer,
        active_scene_robot_prefix=first_prefix,
        instruction="Move to a useful location in the room.",
        history_features=history,
    )
    scene_changed = policy.forward_actual_gemma(
        prefix_backend=backend,
        tokenizer=tokenizer,
        active_scene_robot_prefix=second_prefix,
        instruction="Move to a useful location in the room.",
        history_features=history,
    )
    instruction_changed = policy.forward_actual_gemma(
        prefix_backend=backend,
        tokenizer=tokenizer,
        active_scene_robot_prefix=first_prefix,
        instruction="Face a useful location instead of moving there.",
        history_features=history,
    )

    assert model.forward_calls == 3
    assert not torch.equal(first.decision_hidden, scene_changed.decision_hidden)
    assert not torch.equal(first.decision_hidden, instruction_changed.decision_hidden)


def test_decide_returns_only_learned_high_level_motion_vocabulary() -> None:
    backend, _model, tokenizer = _backend()
    policy = _policy()
    decision = policy.decide(
        prefix_backend=backend,
        tokenizer=tokenizer,
        active_scene_robot_prefix=_active_prefix(backend),
        instruction="Choose every waypoint for a lap around the room.",
        history_features=torch.zeros(1, 0, _HISTORY_FEATURES),
    )

    assert decision.action in set(GemmaMotionAction)
    assert tuple(action.value for action in GemmaMotionAction) == ACTION_NAMES
    assert decision.action.value == ACTION_NAMES[decision.action_index]
    assert sum(decision.action_probabilities) == pytest.approx(1.0)
    assert decision.actual_gemma_causal_forward is True


def test_invalid_prefix_or_missing_hidden_states_fails_closed() -> None:
    backend, model, tokenizer = _backend()
    policy = _policy()
    with pytest.raises(ValueError, match="complete bound tensor"):
        policy.forward_actual_gemma(
            prefix_backend=backend,
            tokenizer=tokenizer,
            active_scene_robot_prefix=_active_prefix(backend)[:, :-1],
            instruction="Move somewhere.",
            history_features=torch.zeros(1, 0, _HISTORY_FEATURES),
        )
    assert model.forward_calls == 0

    broken_backend, broken_model, broken_tokenizer = _backend(
        return_hidden_states=False
    )
    with pytest.raises(RuntimeError, match="did not return hidden_states"):
        policy.forward_actual_gemma(
            prefix_backend=broken_backend,
            tokenizer=broken_tokenizer,
            active_scene_robot_prefix=_active_prefix(broken_backend),
            instruction="Move somewhere.",
            history_features=torch.zeros(1, 0, _HISTORY_FEATURES),
        )
    assert broken_model.forward_calls == 1
