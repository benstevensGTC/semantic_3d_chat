"""Actual-Gemma closed-loop waypoint policy over continuous 3D memory.

This module deliberately stops at the learned decision boundary.  It does not
contain a path planner, semantic retriever, target-coordinate heuristic,
alignment interlock, collision correction, or completion rule.  One real
Gemma causal forward consumes the complete scene-plus-robot prefix, the user's
unchanged high-level instruction, numeric trajectory-history tokens, and a
learned decision token.  Small numeric heads then select exactly one of three
outcomes: ``MOVE_TO``, ``FACE``, or ``STOP``.

The eventual action executor may validate bounds and reject a colliding
proposal.  It must not replace a predicted waypoint or heading, and it must not
turn a rejected motion into a successful stop.  A subsequent policy forward is
responsible for choosing any recovery action.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

import torch
from torch import nn

from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    prefix_sha256,
)

HIDDEN_SIZE: Final[int] = 1536
SCENE_TOKEN_COUNT: Final[int] = 258
ROBOT_TOKEN_COUNT: Final[int] = 4
ACTIVE_PREFIX_TOKEN_COUNT: Final[int] = SCENE_TOKEN_COUNT + ROBOT_TOKEN_COUNT
HISTORY_FEATURE_DIM: Final[int] = 12
MAX_HISTORY_TOKEN_COUNT: Final[int] = 16
CONTEXT_INITIALIZATION_SEED: Final[int] = 46_421
HEAD_INITIALIZATION_SEED: Final[int] = 46_422
ACTION_NAMES: Final[tuple[str, ...]] = ("move_to", "face", "stop")
_SHA256 = re.compile(r"[0-9a-f]{64}")

# Stable protocol text contains no environmental facts.  The user's exact
# instruction is supplied separately through the chat template.
POLICY_SYSTEM_PROMPT: Final[str] = (
    "Choose the next robot outcome using the continuous 3D scene memory, "
    "numeric robot state, numeric action history, and the user's goal. "
    "The learned numeric policy head can choose MOVE_TO, FACE, or STOP."
)


class GemmaMotionAction(str, Enum):
    """The entire learned action vocabulary exposed by this module."""

    MOVE_TO = "move_to"
    FACE = "face"
    STOP = "stop"


def _positive_int(value: object, name: str, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_tuple(value: tuple[float, ...], size: int, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != size:
        raise TypeError(f"{name} must be a {size}-tuple")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"{name} must contain only numbers")
    if any(not math.isfinite(float(item)) for item in value):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class GemmaWaypointPolicyOutput:
    """Strict differentiable batch output from one authenticated Gemma pass."""

    action_logits: torch.Tensor
    waypoint_delta_robot_m: torch.Tensor
    turn_delta_degrees: torch.Tensor
    decision_hidden: torch.Tensor
    instruction_sha256: str
    active_prefix_sha256: str
    scene_token_count: int
    robot_token_count: int
    history_token_count: int
    prompt_token_count: int
    decision_position: int
    actual_gemma_causal_forward: bool
    raw_instruction_included: bool

    def __post_init__(self) -> None:
        tensors = (
            self.action_logits,
            self.waypoint_delta_robot_m,
            self.turn_delta_degrees,
            self.decision_hidden,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("Gemma policy tensor outputs must be tensors")
        batch = self.action_logits.shape[0] if self.action_logits.ndim == 2 else -1
        hidden = self.decision_hidden.shape[-1] if self.decision_hidden.ndim == 2 else -1
        if (
            self.action_logits.shape != (batch, len(ACTION_NAMES))
            or self.waypoint_delta_robot_m.shape != (batch, 2)
            or self.turn_delta_degrees.shape != (batch, 1)
            or self.decision_hidden.shape != (batch, hidden)
            or batch < 1
            or hidden < 1
        ):
            raise ValueError("Gemma policy tensor output shapes are invalid")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("Gemma policy outputs contain NaN or infinity")
        if _SHA256.fullmatch(self.instruction_sha256) is None:
            raise ValueError("Gemma policy instruction hash is invalid")
        if _SHA256.fullmatch(self.active_prefix_sha256) is None:
            raise ValueError("Gemma policy prefix hash is invalid")
        for name, value, minimum in (
            ("scene_token_count", self.scene_token_count, 3),
            ("robot_token_count", self.robot_token_count, 1),
            ("history_token_count", self.history_token_count, 0),
            ("prompt_token_count", self.prompt_token_count, 1),
            ("decision_position", self.decision_position, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Gemma policy {name} is invalid")
        if self.actual_gemma_causal_forward is not True:
            raise ValueError("Gemma policy output must attest an actual causal forward")
        if self.raw_instruction_included is not True:
            raise ValueError("Gemma policy output must include the raw user instruction")

    @property
    def batch_size(self) -> int:
        return int(self.action_logits.shape[0])


@dataclass(frozen=True)
class GemmaWaypointHeadOutput:
    """Trainable numeric-head output for cached final Gemma hidden states.

    This deliberately contains no execution method and no scene lookup.  It is
    the cheap training boundary: callers may cache the actual Gemma decision
    hidden returned by :meth:`ActualGemmaWaypointPolicy.forward_actual_gemma`
    while the context projector and Gemma checkpoint are frozen, then optimize
    only ``policy.numeric_heads.parameters()``.
    """

    action_logits: torch.Tensor
    waypoint_delta_robot_m: torch.Tensor
    turn_delta_degrees: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.action_logits,
            self.waypoint_delta_robot_m,
            self.turn_delta_degrees,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("Gemma waypoint head outputs must be tensors")
        batch = self.action_logits.shape[0] if self.action_logits.ndim == 2 else -1
        if (
            self.action_logits.shape != (batch, len(ACTION_NAMES))
            or self.waypoint_delta_robot_m.shape != (batch, 2)
            or self.turn_delta_degrees.shape != (batch, 1)
            or batch < 1
        ):
            raise ValueError("Gemma waypoint head output shapes are invalid")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("Gemma waypoint head outputs contain NaN or infinity")


@dataclass(frozen=True)
class GemmaWaypointDecision:
    """One inference decision safe to hand to a rejection-only action layer."""

    action: GemmaMotionAction
    action_index: int
    action_logits: tuple[float, float, float]
    action_probabilities: tuple[float, float, float]
    waypoint_delta_robot_m: tuple[float, float]
    turn_delta_degrees: float
    instruction_sha256: str
    active_prefix_sha256: str
    scene_token_count: int
    robot_token_count: int
    history_token_count: int
    prompt_token_count: int
    decision_position: int
    actual_gemma_causal_forward: bool

    def __post_init__(self) -> None:
        if not isinstance(self.action, GemmaMotionAction):
            raise TypeError("Gemma waypoint action must use GemmaMotionAction")
        if (
            isinstance(self.action_index, bool)
            or not isinstance(self.action_index, int)
            or not 0 <= self.action_index < len(ACTION_NAMES)
            or ACTION_NAMES[self.action_index] != self.action.value
        ):
            raise ValueError("Gemma waypoint action index and name differ")
        _finite_tuple(self.action_logits, len(ACTION_NAMES), "action_logits")
        _finite_tuple(
            self.action_probabilities,
            len(ACTION_NAMES),
            "action_probabilities",
        )
        probabilities = tuple(float(value) for value in self.action_probabilities)
        if any(value < 0.0 or value > 1.0 for value in probabilities) or not math.isclose(
            sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError("Gemma waypoint action probabilities are invalid")
        _finite_tuple(self.waypoint_delta_robot_m, 2, "waypoint_delta_robot_m")
        if isinstance(self.turn_delta_degrees, bool) or not isinstance(
            self.turn_delta_degrees, (int, float)
        ) or not math.isfinite(float(self.turn_delta_degrees)):
            raise ValueError("Gemma waypoint turn delta must be finite")
        if _SHA256.fullmatch(self.instruction_sha256) is None:
            raise ValueError("Gemma waypoint instruction hash is invalid")
        if _SHA256.fullmatch(self.active_prefix_sha256) is None:
            raise ValueError("Gemma waypoint prefix hash is invalid")
        for name, value, minimum in (
            ("scene_token_count", self.scene_token_count, 3),
            ("robot_token_count", self.robot_token_count, 1),
            ("history_token_count", self.history_token_count, 0),
            ("prompt_token_count", self.prompt_token_count, 1),
            ("decision_position", self.decision_position, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Gemma waypoint {name} is invalid")
        if self.actual_gemma_causal_forward is not True:
            raise ValueError("Gemma waypoint decision lacks a real Gemma forward")


class NumericHistoryTokenProjector(nn.Module):
    """Project bounded numeric trajectory rows into ordered Gemma-space tokens.

    This component accepts no text or semantic identifiers.  Each row is one
    numeric historical action/pose record.  Zero history is valid for the first
    policy decision and produces an empty token sequence.
    """

    def __init__(
        self,
        *,
        feature_dim: int = HISTORY_FEATURE_DIM,
        hidden_size: int = HIDDEN_SIZE,
        intermediate_dim: int = 256,
        max_history_tokens: int = MAX_HISTORY_TOKEN_COUNT,
        initialization_seed: int = CONTEXT_INITIALIZATION_SEED,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int(feature_dim, "history feature_dim", 4096)
        self.hidden_size = _positive_int(hidden_size, "history hidden_size", 65536)
        self.intermediate_dim = _positive_int(
            intermediate_dim, "history intermediate_dim", 65536
        )
        self.max_history_tokens = _positive_int(
            max_history_tokens, "max_history_tokens", 4096
        )
        self.initialization_seed = _positive_int(
            initialization_seed, "history initialization_seed", 2**31 - 1
        )
        # Construction is independent of ambient application RNG state.  This
        # is essential when final Gemma states are cached for head-only
        # training: rebuilding the policy must reproduce identical controls.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.network = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.intermediate_dim),
                nn.GELU(),
                nn.Linear(self.intermediate_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            )
            self.position_tokens = nn.Parameter(
                torch.randn(self.max_history_tokens, self.hidden_size) * 0.02
            )

    def forward(self, history_features: torch.Tensor) -> torch.Tensor:
        if not isinstance(history_features, torch.Tensor) or history_features.ndim != 3:
            raise ValueError("history_features must have shape [B,T,F]")
        batch, steps, features = history_features.shape
        if batch < 1 or features != self.feature_dim:
            raise ValueError(
                f"history_features must have shape [B,T,{self.feature_dim}]"
            )
        if steps > self.max_history_tokens:
            raise ValueError("history_features exceeds the configured token limit")
        if not bool(torch.isfinite(history_features).all()):
            raise ValueError("history_features contains NaN or infinity")
        if steps == 0:
            return history_features.new_empty((batch, 0, self.hidden_size))
        values = self.network(history_features.float())
        positions = self.position_tokens[:steps].unsqueeze(0).to(values)
        output = values + positions
        if output.shape != (batch, steps, self.hidden_size) or not bool(
            torch.isfinite(output).all()
        ):
            raise RuntimeError("Numeric history projector produced invalid tokens")
        return output


class GemmaWaypointHeads(nn.Module):
    """Small trainable heads over a cached *actual Gemma* final hidden state."""

    def __init__(
        self,
        *,
        hidden_size: int = HIDDEN_SIZE,
        intermediate_dim: int = 256,
        max_waypoint_step_m: float = 0.50,
        max_turn_delta_degrees: float = 40.0,
        initialization_seed: int = HEAD_INITIALIZATION_SEED,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, "head hidden_size", 65536)
        head_dim = _positive_int(intermediate_dim, "head intermediate_dim", 65536)
        self.max_waypoint_step_m = _finite_positive(
            max_waypoint_step_m, "max_waypoint_step_m"
        )
        self.max_turn_delta_degrees = _finite_positive(
            max_turn_delta_degrees, "max_turn_delta_degrees"
        )
        if self.max_turn_delta_degrees >= 180.0:
            raise ValueError("max_turn_delta_degrees must be below 180 degrees")
        self.initialization_seed = _positive_int(
            initialization_seed, "head initialization_seed", 2**31 - 1
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.input_norm = nn.LayerNorm(self.hidden_size)
            self.action = nn.Linear(self.hidden_size, len(ACTION_NAMES))
            self.waypoint = nn.Sequential(
                nn.Linear(self.hidden_size, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 2),
            )
            self.heading = nn.Sequential(
                nn.Linear(self.hidden_size, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 1),
            )
            # Zero is a defined initial robot-relative turn. The bounded tanh
            # parameterization is part of the learned output space; the action
            # executor never clamps or substitutes the model's proposal.
            final_heading = self.heading[-1]
            assert isinstance(final_heading, nn.Linear)
            nn.init.zeros_(final_heading.bias)

    def forward(self, final_gemma_hidden: torch.Tensor) -> GemmaWaypointHeadOutput:
        if (
            not isinstance(final_gemma_hidden, torch.Tensor)
            or final_gemma_hidden.ndim != 2
            or final_gemma_hidden.shape[0] < 1
            or final_gemma_hidden.shape[1] != self.hidden_size
        ):
            raise ValueError(
                f"final_gemma_hidden must have shape [B,{self.hidden_size}]"
            )
        if not bool(torch.isfinite(final_gemma_hidden.float()).all()):
            raise ValueError("final_gemma_hidden contains NaN or infinity")
        hidden = self.input_norm(final_gemma_hidden.float())
        action_logits = self.action(hidden)
        waypoint = torch.tanh(self.waypoint(hidden)) * self.max_waypoint_step_m
        turn_delta = (
            torch.tanh(self.heading(hidden).float()) * self.max_turn_delta_degrees
        )
        return GemmaWaypointHeadOutput(
            action_logits=action_logits,
            waypoint_delta_robot_m=waypoint,
            turn_delta_degrees=turn_delta,
        )


class ActualGemmaWaypointPolicy(nn.Module):
    """Use an actual full Gemma pass to choose the next waypoint outcome."""

    def __init__(
        self,
        *,
        hidden_size: int = HIDDEN_SIZE,
        scene_token_count: int = SCENE_TOKEN_COUNT,
        robot_token_count: int = ROBOT_TOKEN_COUNT,
        history_feature_dim: int = HISTORY_FEATURE_DIM,
        max_history_tokens: int = MAX_HISTORY_TOKEN_COUNT,
        head_hidden_dim: int = 256,
        max_waypoint_step_m: float = 0.50,
        max_turn_delta_degrees: float = 40.0,
        context_initialization_seed: int = CONTEXT_INITIALIZATION_SEED,
        head_initialization_seed: int = HEAD_INITIALIZATION_SEED,
        freeze_context_projection: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, "hidden_size", 65536)
        self.scene_token_count = _positive_int(
            scene_token_count, "scene_token_count", 65536
        )
        if self.scene_token_count < 3:
            raise ValueError("scene_token_count must include two boundaries and content")
        self.robot_token_count = _positive_int(
            robot_token_count, "robot_token_count", 4096
        )
        self.active_prefix_token_count = self.scene_token_count + self.robot_token_count
        self.max_waypoint_step_m = _finite_positive(
            max_waypoint_step_m, "max_waypoint_step_m"
        )
        self.max_turn_delta_degrees = _finite_positive(
            max_turn_delta_degrees, "max_turn_delta_degrees"
        )
        head_dim = _positive_int(head_hidden_dim, "head_hidden_dim", 65536)
        context_seed = _positive_int(
            context_initialization_seed, "context_initialization_seed", 2**31 - 2
        )
        self.history_projector = NumericHistoryTokenProjector(
            feature_dim=history_feature_dim,
            hidden_size=self.hidden_size,
            intermediate_dim=head_dim,
            max_history_tokens=max_history_tokens,
            initialization_seed=context_seed,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(context_seed + 1)
            self.decision_token = nn.Parameter(
                torch.randn(1, 1, self.hidden_size) * 0.02
            )
        self.numeric_heads = GemmaWaypointHeads(
            hidden_size=self.hidden_size,
            intermediate_dim=head_dim,
            max_waypoint_step_m=self.max_waypoint_step_m,
            max_turn_delta_degrees=self.max_turn_delta_degrees,
            initialization_seed=head_initialization_seed,
        )
        if not isinstance(freeze_context_projection, bool):
            raise TypeError("freeze_context_projection must be a boolean")
        self.set_context_projection_trainable(not freeze_context_projection)

    @property
    def context_projection_frozen(self) -> bool:
        parameters = [self.decision_token, *self.history_projector.parameters()]
        return all(not parameter.requires_grad for parameter in parameters)

    def set_context_projection_trainable(self, trainable: bool) -> None:
        """Freeze by default so cached actual-Gemma states stay reproducible."""

        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean")
        self.decision_token.requires_grad_(trainable)
        self.history_projector.requires_grad_(trainable)

    def forward_heads_from_cached_gemma_hidden(
        self,
        final_gemma_hidden: torch.Tensor,
    ) -> GemmaWaypointHeadOutput:
        """Train/evaluate numeric heads without rerunning a cached Gemma pass.

        Production action inference must call :meth:`forward_actual_gemma` or
        :meth:`decide`; this method is the explicit offline head-training seam.
        """

        return self.numeric_heads(final_gemma_hidden)

    def _validate_instruction(self, instruction: str) -> str:
        if not isinstance(instruction, str):
            raise TypeError("Gemma waypoint instruction must be text")
        if not instruction.strip() or instruction != instruction.strip():
            raise ValueError("Gemma waypoint instruction must be nonempty and unwrapped")
        if len(instruction) > 4096 or "\x00" in instruction:
            raise ValueError("Gemma waypoint instruction is too long or contains NUL")
        return instruction

    def _validate_prefix(self, active_prefix: torch.Tensor) -> None:
        if (
            not isinstance(active_prefix, torch.Tensor)
            or active_prefix.ndim != 3
            or active_prefix.shape[0] < 1
            or tuple(active_prefix.shape[1:])
            != (self.active_prefix_token_count, self.hidden_size)
        ):
            raise ValueError(
                "active_scene_robot_prefix must be the complete bound tensor "
                f"with shape [B,{self.active_prefix_token_count},{self.hidden_size}]"
            )
        if not bool(torch.isfinite(active_prefix.float()).all()):
            raise ValueError("active_scene_robot_prefix contains NaN or infinity")

    @staticmethod
    def _last_hidden(outputs: Any) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
            raise RuntimeError(
                "Actual Gemma forward did not return hidden_states; "
                "output_hidden_states=True is required"
            )
        final = hidden_states[-1]
        if not isinstance(final, torch.Tensor) or final.ndim != 3:
            raise RuntimeError("Actual Gemma final hidden state is invalid")
        return final

    def forward_actual_gemma(
        self,
        *,
        prefix_backend: Any,
        tokenizer: Any,
        active_scene_robot_prefix: torch.Tensor,
        instruction: str,
        history_features: torch.Tensor,
    ) -> GemmaWaypointPolicyOutput:
        """Run one real causal Gemma forward and return differentiable heads.

        ``prefix_backend`` is expected to be the project's
        :class:`~semantic_3d_chat.language.gemma4_backend.Gemma4PrefixBackend`.
        Duck typing is intentional so unit tests can use a tiny fake Gemma
        model while still exercising the real backend's ``prepare`` method.
        """

        literal = self._validate_instruction(instruction)
        self._validate_prefix(active_scene_robot_prefix)
        if not callable(getattr(prefix_backend, "prepare", None)) or not isinstance(
            getattr(prefix_backend, "model", None), nn.Module
        ):
            raise TypeError("Gemma waypoint policy requires Gemma4PrefixBackend")
        backend_hidden = getattr(prefix_backend, "hidden_size", None)
        if backend_hidden != self.hidden_size:
            raise ValueError("Gemma backend and waypoint policy hidden sizes differ")
        if (
            not isinstance(history_features, torch.Tensor)
            or history_features.ndim != 3
            or history_features.shape[0] != active_scene_robot_prefix.shape[0]
        ):
            raise ValueError("history_features must have shape [B,T,F]")

        device = active_scene_robot_prefix.device
        batch_size = int(active_scene_robot_prefix.shape[0])
        prompt_ids = prompt_token_ids(
            tokenizer,
            POLICY_SYSTEM_PROMPT,
            literal,
            device,
        )
        if prompt_ids.shape[0] != 1:
            raise ValueError("Gemma waypoint prompt template must produce one row")
        prompt_ids = prompt_ids.expand(batch_size, -1)
        history_tokens = self.history_projector(
            history_features.to(device=self.decision_token.device)
        ).to(active_scene_robot_prefix)
        decision = self.decision_token.expand(batch_size, -1, -1).to(
            active_scene_robot_prefix
        )
        control_tokens = torch.cat((history_tokens, decision), dim=1)
        prepared = prefix_backend.prepare(
            active_scene_robot_prefix,
            prompt_ids,
            scene_prefix_after_bos=True,
            scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
            control_tokens=control_tokens,
        )
        if prepared.scene_prefix_length != self.active_prefix_token_count:
            raise RuntimeError("Prepared Gemma input omitted part of the active prefix")
        expected_length = (
            self.active_prefix_token_count
            + int(prompt_ids.shape[1])
            + int(control_tokens.shape[1])
        )
        if tuple(prepared.inputs_embeds.shape) != (
            batch_size,
            expected_length,
            self.hidden_size,
        ):
            raise RuntimeError("Prepared Gemma waypoint input shape differs")
        decision_position = expected_length - 1
        expected_decision = decision.to(prepared.inputs_embeds)
        if not torch.equal(
            prepared.inputs_embeds[:, decision_position:], expected_decision
        ):
            raise RuntimeError("Learned decision token is not the final causal input")

        outputs = prefix_backend.model(
            inputs_embeds=prepared.inputs_embeds,
            per_layer_inputs=prepared.per_layer_inputs,
            attention_mask=prepared.attention_mask,
            mm_token_type_ids=prepared.mm_token_type_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        final_hidden = self._last_hidden(outputs)
        if tuple(final_hidden.shape) != (
            batch_size,
            expected_length,
            self.hidden_size,
        ):
            raise RuntimeError("Actual Gemma hidden-state sequence shape differs")
        decision_hidden = final_hidden[:, decision_position].float()
        head_output = self.forward_heads_from_cached_gemma_hidden(decision_hidden)
        return GemmaWaypointPolicyOutput(
            action_logits=head_output.action_logits,
            waypoint_delta_robot_m=head_output.waypoint_delta_robot_m,
            turn_delta_degrees=head_output.turn_delta_degrees,
            decision_hidden=decision_hidden,
            instruction_sha256=hashlib.sha256(literal.encode("utf-8")).hexdigest(),
            active_prefix_sha256=prefix_sha256(active_scene_robot_prefix),
            scene_token_count=self.scene_token_count,
            robot_token_count=self.robot_token_count,
            history_token_count=int(history_features.shape[1]),
            prompt_token_count=int(prompt_ids.shape[1]),
            decision_position=decision_position,
            actual_gemma_causal_forward=True,
            raw_instruction_included=True,
        )

    @torch.inference_mode()
    def decide(
        self,
        *,
        prefix_backend: Any,
        tokenizer: Any,
        active_scene_robot_prefix: torch.Tensor,
        instruction: str,
        history_features: torch.Tensor,
    ) -> GemmaWaypointDecision:
        """Return one strict MOVE_TO, FACE, or STOP decision."""

        output = self.forward_actual_gemma(
            prefix_backend=prefix_backend,
            tokenizer=tokenizer,
            active_scene_robot_prefix=active_scene_robot_prefix,
            instruction=instruction,
            history_features=history_features,
        )
        if output.batch_size != 1:
            raise RuntimeError("Interactive Gemma waypoint inference requires batch size one")
        probabilities = torch.softmax(output.action_logits[0].float(), dim=-1)
        index = int(torch.argmax(probabilities).item())
        return GemmaWaypointDecision(
            action=GemmaMotionAction(ACTION_NAMES[index]),
            action_index=index,
            action_logits=tuple(float(value) for value in output.action_logits[0].cpu()),
            action_probabilities=tuple(float(value) for value in probabilities.cpu()),
            waypoint_delta_robot_m=tuple(
                float(value) for value in output.waypoint_delta_robot_m[0].cpu()
            ),
            turn_delta_degrees=float(output.turn_delta_degrees[0, 0].cpu()),
            instruction_sha256=output.instruction_sha256,
            active_prefix_sha256=output.active_prefix_sha256,
            scene_token_count=output.scene_token_count,
            robot_token_count=output.robot_token_count,
            history_token_count=output.history_token_count,
            prompt_token_count=output.prompt_token_count,
            decision_position=output.decision_position,
            actual_gemma_causal_forward=output.actual_gemma_causal_forward,
        )


__all__ = [
    "ACTION_NAMES",
    "ACTIVE_PREFIX_TOKEN_COUNT",
    "CONTEXT_INITIALIZATION_SEED",
    "HEAD_INITIALIZATION_SEED",
    "HIDDEN_SIZE",
    "HISTORY_FEATURE_DIM",
    "MAX_HISTORY_TOKEN_COUNT",
    "POLICY_SYSTEM_PROMPT",
    "ROBOT_TOKEN_COUNT",
    "SCENE_TOKEN_COUNT",
    "ActualGemmaWaypointPolicy",
    "GemmaMotionAction",
    "GemmaWaypointDecision",
    "GemmaWaypointHeadOutput",
    "GemmaWaypointHeads",
    "GemmaWaypointPolicyOutput",
    "NumericHistoryTokenProjector",
]
