"""Preregistered continuous-context adapter pieces for Gemma-4 robot tools.

This module contains architecture and tensor-preparation code only.  It does
not load training traces, read oracle metadata, start optimization, or publish
a runtime checkpoint.  The deployable path is intended to consume:

* the complete fixed scene prefix, with numeric robot tokens already inserted;
* the literal user navigation instruction as ordinary user text;
* a ten-value target state grounded against every active semantic-map voxel;
* a 24-ray free-space state derived from anonymous numeric map geometry; and
* an exact minified JSON action as the teacher-forced answer suffix.

Object names in the user's own instruction are allowed user text.  No object
inventory, caption, scene graph, simulator label, or oracle relationship is an
input to this adapter at runtime.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import nn

from semantic_3d_chat.language.lora import LoRASettings
from semantic_3d_chat.robot.navigation_policy import ACTION_NAMES, tool_call_from_prediction

MODEL_ID: Final[str] = "google/gemma-4-E2B-it"
MODEL_REVISION: Final[str] = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
HIDDEN_SIZE: Final[int] = 1536
TARGET_STATE_DIM: Final[int] = 10
CLEARANCE_STATE_DIM: Final[int] = 24
TARGET_TOKEN_COUNT: Final[int] = 2
CLEARANCE_TOKEN_COUNT: Final[int] = 2
CONTROL_TOKEN_COUNT: Final[int] = TARGET_TOKEN_COUNT + CLEARANCE_TOKEN_COUNT
PROJECTOR_INITIALIZATION_SEED: Final[int] = 2026081217
PROJECTOR_INITIAL_OUTPUT_SCALE: Final[float] = 0.02

K_PROJECTION: Final[str] = "model.language_model.layers.34.self_attn.k_proj"
V_PROJECTION: Final[str] = "model.language_model.layers.34.self_attn.v_proj"
K_OR_V_IN_FEATURES: Final[int] = 1536
# V1 trusted the checkpoint tensor inventory before its later structural smoke
# proved that Transformers ignores these tensors in KV-sharing layers.
K_OR_V_OUT_FEATURES: Final[int] = 512
LORA_RANK: Final[int] = 4
LORA_ALPHA: Final[float] = 8.0
LORA_PARAMETER_COUNT: Final[int] = 2 * LORA_RANK * (
    K_OR_V_IN_FEATURES + K_OR_V_OUT_FEATURES
)
PROJECTOR_PARAMETER_COUNT: Final[int] = (
    TARGET_STATE_DIM * TARGET_TOKEN_COUNT * HIDDEN_SIZE
    + TARGET_TOKEN_COUNT * HIDDEN_SIZE
    + CLEARANCE_STATE_DIM * CLEARANCE_TOKEN_COUNT * HIDDEN_SIZE
    + CLEARANCE_TOKEN_COUNT * HIDDEN_SIZE
)
TOTAL_TRAINABLE_PARAMETER_COUNT: Final[int] = (
    LORA_PARAMETER_COUNT + PROJECTOR_PARAMETER_COUNT
)


def tool_decoder_lora_settings() -> LoRASettings:
    """Return the one fresh, exact-path, unmerged upper-decoder LoRA arm."""

    return LoRASettings(
        enabled=True,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=0.0,
        target_modules=(K_PROJECTION, V_PROJECTION),
    )


def validate_decoder_surface(model: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    """Fail closed if Gemma's preregistered final full-attention K/V changed."""

    projections: list[nn.Linear] = []
    for path in (K_PROJECTION, V_PROJECTION):
        try:
            module = model.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f"Missing preregistered tool-decoder projection: {path}") from exc
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Preregistered tool-decoder projection is not Linear: {path}")
        observed = (module.in_features, module.out_features, module.bias is None)
        expected = (K_OR_V_IN_FEATURES, K_OR_V_OUT_FEATURES, True)
        if observed != expected:
            raise ValueError(f"Tool-decoder projection contract changed: {observed} != {expected}")
        projections.append(module)

    config = getattr(getattr(model, "config", None), "text_config", None)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and (
        len(layer_types) != 35 or layer_types[34] != "full_attention"
    ):
        raise ValueError("Gemma layer 34 is no longer the preregistered full-attention layer")
    return projections[0], projections[1]


class NumericToolContextProjectorV1(nn.Module):
    """Project numeric target and anonymous free-space values into LM tokens.

    The two branches use separate direct projections, so token identity is
    structural rather than a textual marker.  Inputs are already engineered,
    bounded numeric values; keeping this bridge linear makes its entire
    trainable surface small and auditable.
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        *,
        target_token_count: int = TARGET_TOKEN_COUNT,
        clearance_token_count: int = CLEARANCE_TOKEN_COUNT,
        initialization_seed: int = PROJECTOR_INITIALIZATION_SEED,
        initial_output_scale: float = PROJECTOR_INITIAL_OUTPUT_SCALE,
    ) -> None:
        super().__init__()
        integers = (hidden_size, target_token_count, clearance_token_count)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integers):
            raise ValueError("Tool-context dimensions and token counts must be positive integers")
        if (
            isinstance(initialization_seed, bool)
            or not isinstance(initialization_seed, int)
            or initialization_seed < 0
        ):
            raise ValueError("Tool-context initialization seed must be nonnegative")
        if (
            isinstance(initial_output_scale, bool)
            or not isinstance(initial_output_scale, (int, float))
            or not math.isfinite(float(initial_output_scale))
            or not 0.0 < float(initial_output_scale) <= 0.1
        ):
            raise ValueError("Tool-context initial output scale must be in (0, 0.1]")
        self.hidden_size = hidden_size
        self.target_token_count = target_token_count
        self.clearance_token_count = clearance_token_count
        self.initialization_seed = initialization_seed
        self.initial_output_scale = float(initial_output_scale)
        self.target_projection = nn.Linear(
            TARGET_STATE_DIM, target_token_count * hidden_size
        )
        self.clearance_projection = nn.Linear(
            CLEARANCE_STATE_DIM, clearance_token_count * hidden_size
        )
        self._reset_parameters()

    @property
    def token_count(self) -> int:
        return self.target_token_count + self.clearance_token_count

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.initialization_seed)
        with torch.no_grad():
            for layer in (self.target_projection, self.clearance_projection):
                bound = 1.0 / math.sqrt(layer.in_features)
                weight = torch.empty(layer.weight.shape, dtype=torch.float32).uniform_(
                    -bound, bound, generator=generator
                )
                bias = torch.empty(layer.bias.shape, dtype=torch.float32).uniform_(
                    -bound, bound, generator=generator
                )
                layer.weight.copy_(weight.to(layer.weight) * self.initial_output_scale)
                layer.bias.copy_(bias.to(layer.bias) * self.initial_output_scale)

    def forward(
        self,
        target_state: torch.Tensor,
        clearance_state: torch.Tensor,
    ) -> torch.Tensor:
        target = torch.as_tensor(target_state)
        clearance = torch.as_tensor(clearance_state)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if clearance.ndim == 1:
            clearance = clearance.unsqueeze(0)
        if target.ndim != 2 or target.shape[1] != TARGET_STATE_DIM:
            raise ValueError(f"target_state must have shape [B, {TARGET_STATE_DIM}]")
        if clearance.shape != (target.shape[0], CLEARANCE_STATE_DIM):
            raise ValueError(
                f"clearance_state must have shape [B, {CLEARANCE_STATE_DIM}]"
            )
        if not torch.isfinite(target).all() or not torch.isfinite(clearance).all():
            raise ValueError("Tool-context numeric inputs contain NaN or infinity")
        if torch.any((clearance < 0.0) | (clearance > 1.0)):
            raise ValueError("Tool-context clearance values must be normalized to [0, 1]")
        device = self.target_projection.weight.device
        target = target.to(device=device, dtype=torch.float32)
        clearance = clearance.to(device=device, dtype=torch.float32)
        target_tokens = self.target_projection(target).reshape(
            len(target), self.target_token_count, self.hidden_size
        )
        clearance_tokens = self.clearance_projection(clearance).reshape(
            len(target), self.clearance_token_count, self.hidden_size
        )
        tokens = torch.cat((target_tokens, clearance_tokens), dim=1)
        if not torch.isfinite(tokens).all():
            raise RuntimeError("Tool-context projector produced NaN or infinity")
        return tokens


def tool_decoder_system_prompt(*, max_turn_degrees: float, max_move_m: float) -> str:
    """Return the short, environment-free, five-action protocol prompt."""

    turn = float(max_turn_degrees)
    move = float(max_move_m)
    if not math.isfinite(turn) or turn <= 0.0 or not math.isfinite(move) or move <= 0.0:
        raise ValueError("Tool bounds must be finite and positive")
    return (
        "Choose one next robot action using the continuous context. Output only one "
        "minified JSON object with exactly keys tool and arguments. Allowed forms: "
        '{"arguments":{},"tool":"stop"}; '
        '{"arguments":{},"tool":"scan"}; '
        f'{{"arguments":{{"angle_degrees":number from {-turn:g} to {turn:g}}},'
        '"tool":"turn"}}; '
        f'{{"arguments":{{"distance_meters":number from 0.02 to {move:g}}},'
        '"tool":"move_forward"}}; or the same arguments with tool "move_backward". '
        "No prose or Markdown."
    )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _rounded(value: float, digits: int) -> float:
    result = round(float(value), digits)
    return 0.0 if result == 0.0 else result


def canonical_tool_json_from_trace(
    row: Mapping[str, Any],
    *,
    max_turn_degrees: float,
    max_move_m: float,
) -> str:
    """Convert one authenticated V3 action row to its exact JSON answer label."""

    action_index = row.get("action_index")
    action_name = row.get("action_name")
    if (
        isinstance(action_index, bool)
        or not isinstance(action_index, int)
        or not 0 <= action_index < len(ACTION_NAMES)
        or action_name != ACTION_NAMES[action_index]
    ):
        raise ValueError("Trace action index and name differ from the fixed vocabulary")
    normalized = _finite_number(
        row.get("argument_target_normalized"), "argument_target_normalized"
    )
    if not -1.0 - 1e-6 <= normalized <= 1.0 + 1e-6:
        raise ValueError("Trace normalized argument is outside [-1, 1]")
    call = tool_call_from_prediction(
        action_index,
        normalized,
        max_turn_degrees=max_turn_degrees,
        max_move_m=max_move_m,
    )
    arguments = call["arguments"]
    if action_name == "turn":
        arguments["angle_degrees"] = _rounded(arguments["angle_degrees"], 3)
    elif action_name in {"move_forward", "move_backward"}:
        arguments["distance_meters"] = _rounded(arguments["distance_meters"], 3)
    return json.dumps(call, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_answer_token_ids(
    tokenizer: Any,
    canonical_json: str,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Tokenize only the JSON answer and append the model-derived EOS token."""

    if not isinstance(canonical_json, str) or not canonical_json:
        raise ValueError("Canonical JSON answer must be nonempty text")
    parsed = json.loads(canonical_json)
    if not isinstance(parsed, dict) or set(parsed) != {"tool", "arguments"}:
        raise ValueError("Canonical answer is not one exact tool envelope")
    encoded = tokenizer(canonical_json, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] < 1:
        raise ValueError("Tokenizer returned no canonical JSON answer tokens")
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0:
        raise ValueError("Tokenizer has no valid EOS token ID")
    suffix = torch.tensor([[eos]], dtype=torch.long)
    return torch.cat((ids.to(dtype=torch.long, device="cpu"), suffix), dim=1).to(device)


def prepare_tool_decoder_inputs(
    prefix_backend: Any,
    active_scene_robot_prefix: torch.Tensor,
    prompt_ids: torch.Tensor,
    projector: NumericToolContextProjectorV1,
    target_state: torch.Tensor,
    clearance_state: torch.Tensor,
    *,
    answer_ids: torch.Tensor | None = None,
    scene_prefix_after_bos: bool = True,
    scene_boundary_mode: str = "gemma4_native_image",
) -> Any:
    """Build Gemma PLE-aware inputs for training or cached local generation."""

    if active_scene_robot_prefix.ndim != 3:
        raise ValueError("Active scene-plus-robot prefix must be rank three")
    controls = projector(target_state, clearance_state).to(active_scene_robot_prefix)
    prepared = prefix_backend.prepare(
        active_scene_robot_prefix,
        prompt_ids,
        answer_ids,
        scene_prefix_after_bos=scene_prefix_after_bos,
        scene_boundary_mode=scene_boundary_mode,
        control_tokens=controls,
    )
    if prepared.inputs_embeds.shape[-1] != active_scene_robot_prefix.shape[-1]:
        raise RuntimeError("Prepared Gemma tool input hidden width changed")
    if answer_ids is not None:
        expected_ignored = prepared.inputs_embeds.shape[1] - answer_ids.shape[1]
        if (
            prepared.labels is None
            or not torch.all(prepared.labels[:, :expected_ignored] == -100)
            or not torch.equal(prepared.labels[:, expected_ignored:], answer_ids)
        ):
            raise RuntimeError("Gemma tool labels are not confined to the JSON answer suffix")
    return prepared


__all__ = [
    "CLEARANCE_STATE_DIM",
    "CLEARANCE_TOKEN_COUNT",
    "CONTROL_TOKEN_COUNT",
    "HIDDEN_SIZE",
    "K_PROJECTION",
    "LORA_PARAMETER_COUNT",
    "MODEL_ID",
    "MODEL_REVISION",
    "PROJECTOR_PARAMETER_COUNT",
    "TARGET_STATE_DIM",
    "TARGET_TOKEN_COUNT",
    "TOTAL_TRAINABLE_PARAMETER_COUNT",
    "V_PROJECTION",
    "NumericToolContextProjectorV1",
    "canonical_answer_token_ids",
    "canonical_tool_json_from_trace",
    "prepare_tool_decoder_inputs",
    "tool_decoder_lora_settings",
    "tool_decoder_system_prompt",
    "validate_decoder_surface",
]
