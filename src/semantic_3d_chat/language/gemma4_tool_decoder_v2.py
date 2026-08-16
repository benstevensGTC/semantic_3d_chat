"""Corrected Gemma-4 causal tool-decoder surface after the V1 KV failure.

V1 selected layer-34 K/V tensors from the checkpoint inventory, but Gemma-4
E2B shares K/V after layer 14 and Transformers intentionally does not
instantiate those modules.  V2 instead adapts the real final-layer MLP down
projection.  It is the highest disjoint content-dependent projection after
layer-34 full attention has integrated scene, robot, instruction, target, and
free-space context.
"""

from __future__ import annotations

from typing import Final

from torch import nn

from semantic_3d_chat.language.gemma4_tool_decoder_v1 import (
    CLEARANCE_STATE_DIM,
    CLEARANCE_TOKEN_COUNT,
    CONTROL_TOKEN_COUNT,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PROJECTOR_INITIAL_OUTPUT_SCALE,
    PROJECTOR_PARAMETER_COUNT,
    TARGET_STATE_DIM,
    TARGET_TOKEN_COUNT,
    NumericToolContextProjectorV1,
    canonical_answer_token_ids,
    canonical_tool_json_from_trace,
    prepare_tool_decoder_inputs,
    tool_decoder_system_prompt,
)
from semantic_3d_chat.language.lora import LoRASettings

TARGET_PROJECTION: Final[str] = "model.language_model.layers.34.mlp.down_proj"
TARGET_PROJECTION_IN_FEATURES: Final[int] = 12288
TARGET_PROJECTION_OUT_FEATURES: Final[int] = 1536
LORA_RANK: Final[int] = 4
LORA_ALPHA: Final[float] = 8.0
LORA_PARAMETER_COUNT: Final[int] = LORA_RANK * (
    TARGET_PROJECTION_IN_FEATURES + TARGET_PROJECTION_OUT_FEATURES
)
PROJECTOR_INITIALIZATION_SEED: Final[int] = 2026081218
INITIAL_LORA_STATE_SHA256: Final[str] = (
    "58b12ef8f2b5c9ffdf1a99d068b32b5414cefe50236013ebbedc831382114a13"
)
INITIAL_PROJECTOR_STATE_SHA256: Final[str] = (
    "cb83627478b01d69d0d5618c7cd0b04f440fab5bdb070e643f3d9ec03a12b199"
)
TOTAL_TRAINABLE_PARAMETER_COUNT: Final[int] = (
    LORA_PARAMETER_COUNT + PROJECTOR_PARAMETER_COUNT
)


class NumericToolContextProjectorV2(NumericToolContextProjectorV1):
    """V1's auditable numeric bridge with a distinct V2 initialization."""

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        *,
        target_token_count: int = TARGET_TOKEN_COUNT,
        clearance_token_count: int = CLEARANCE_TOKEN_COUNT,
        initialization_seed: int = PROJECTOR_INITIALIZATION_SEED,
        initial_output_scale: float = PROJECTOR_INITIAL_OUTPUT_SCALE,
    ) -> None:
        super().__init__(
            hidden_size,
            target_token_count=target_token_count,
            clearance_token_count=clearance_token_count,
            initialization_seed=initialization_seed,
            initial_output_scale=initial_output_scale,
        )


def tool_decoder_lora_settings_v2() -> LoRASettings:
    """Return V2's one exact, unmerged, upper-decoder adapter."""

    return LoRASettings(
        enabled=True,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=0.0,
        target_modules=(TARGET_PROJECTION,),
    )


def validate_decoder_surface_v2(model: nn.Module) -> nn.Linear:
    """Require the exact loaded final-layer MLP down projection."""

    try:
        projection = model.get_submodule(TARGET_PROJECTION)
    except AttributeError as exc:
        raise ValueError(
            f"Missing preregistered V2 tool-decoder projection: {TARGET_PROJECTION}"
        ) from exc
    if not isinstance(projection, nn.Linear):
        raise TypeError("V2 tool-decoder projection must be torch.nn.Linear")
    observed = (
        projection.in_features,
        projection.out_features,
        projection.bias is None,
    )
    expected = (
        TARGET_PROJECTION_IN_FEATURES,
        TARGET_PROJECTION_OUT_FEATURES,
        True,
    )
    if observed != expected:
        raise ValueError(f"V2 tool-decoder projection changed: {observed} != {expected}")
    config = getattr(getattr(model, "config", None), "text_config", None)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and (
        len(layer_types) != 35 or layer_types[34] != "full_attention"
    ):
        raise ValueError("Gemma layer 34 is no longer the expected full-attention layer")
    return projection


__all__ = [
    "CLEARANCE_STATE_DIM",
    "CLEARANCE_TOKEN_COUNT",
    "CONTROL_TOKEN_COUNT",
    "HIDDEN_SIZE",
    "INITIAL_LORA_STATE_SHA256",
    "INITIAL_PROJECTOR_STATE_SHA256",
    "LORA_ALPHA",
    "LORA_PARAMETER_COUNT",
    "LORA_RANK",
    "MODEL_ID",
    "MODEL_REVISION",
    "PROJECTOR_INITIALIZATION_SEED",
    "PROJECTOR_INITIAL_OUTPUT_SCALE",
    "PROJECTOR_PARAMETER_COUNT",
    "TARGET_PROJECTION",
    "TARGET_PROJECTION_IN_FEATURES",
    "TARGET_PROJECTION_OUT_FEATURES",
    "TARGET_STATE_DIM",
    "TARGET_TOKEN_COUNT",
    "TOTAL_TRAINABLE_PARAMETER_COUNT",
    "NumericToolContextProjectorV2",
    "canonical_answer_token_ids",
    "canonical_tool_json_from_trace",
    "prepare_tool_decoder_inputs",
    "tool_decoder_lora_settings_v2",
    "tool_decoder_system_prompt",
    "validate_decoder_surface_v2",
]
