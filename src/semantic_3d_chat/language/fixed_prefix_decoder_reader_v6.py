"""Exact upper-decoder surface for the unsealed V6 static-reader proposal."""

from __future__ import annotations

from typing import Final

from torch import nn

from semantic_3d_chat.language.lora import LoRASettings

MODEL_ID: Final[str] = "google/gemma-4-E2B-it"
MODEL_REVISION: Final[str] = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
TARGET_MODULES: Final[tuple[str, str]] = (
    "model.language_model.layers.32.mlp.down_proj",
    "model.language_model.layers.33.mlp.down_proj",
)
TARGET_IN_FEATURES: Final[int] = 12_288
TARGET_OUT_FEATURES: Final[int] = 1_536
LORA_RANK: Final[int] = 4
LORA_ALPHA: Final[float] = 8.0
LORA_PARAMETER_COUNT_PER_MODULE: Final[int] = LORA_RANK * (TARGET_IN_FEATURES + TARGET_OUT_FEATURES)
LORA_PARAMETER_COUNT: Final[int] = len(TARGET_MODULES) * LORA_PARAMETER_COUNT_PER_MODULE
INITIALIZATION_SEED: Final[int] = 720_054
INITIAL_STATE_SHA256: Final[str] = (
    "6d3570a60306d4d28aa2ca35962e161582a478b959098cf2bf89be2cce56f2e2"
)
EXPECTED_LAYER_TYPES: Final[tuple[str, ...]] = (
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
) * 7
SLIDING_WINDOW_TOKENS: Final[int] = 512


def decoder_reader_lora_settings_v6() -> LoRASettings:
    """Return the one exact, unmerged, rank-4 V6 decoder adapter."""

    return LoRASettings(
        enabled=True,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=0.0,
        target_modules=TARGET_MODULES,
    )


def validate_decoder_reader_surface_v6(model: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    """Require both projections and the complete pinned decoder routing contract."""

    projections: list[nn.Linear] = []
    for target in TARGET_MODULES:
        try:
            projection = model.get_submodule(target)
        except AttributeError as exc:
            raise ValueError(f"Missing V6 decoder projection: {target}") from exc
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"V6 decoder projection must be torch.nn.Linear: {target}")
        observed = (
            projection.in_features,
            projection.out_features,
            projection.bias is None,
        )
        expected = (TARGET_IN_FEATURES, TARGET_OUT_FEATURES, True)
        if observed != expected:
            raise ValueError(f"V6 decoder projection changed: {observed} != {expected}")
        projections.append(projection)
    text = getattr(getattr(model, "config", None), "text_config", None)
    if text is None:
        raise ValueError("V6 requires an explicit Gemma text_config")
    layer_types = getattr(text, "layer_types", None)
    if tuple(layer_types or ()) != EXPECTED_LAYER_TYPES:
        raise ValueError("V6 requires all 35 pinned Gemma decoder layer types")
    if getattr(text, "sliding_window", None) != SLIDING_WINDOW_TOKENS:
        raise ValueError("V6 requires Gemma's pinned 512-token sliding window")
    return projections[0], projections[1]


__all__ = [
    "EXPECTED_LAYER_TYPES",
    "INITIALIZATION_SEED",
    "INITIAL_STATE_SHA256",
    "LORA_ALPHA",
    "LORA_PARAMETER_COUNT",
    "LORA_PARAMETER_COUNT_PER_MODULE",
    "LORA_RANK",
    "MODEL_ID",
    "MODEL_REVISION",
    "SLIDING_WINDOW_TOKENS",
    "TARGET_IN_FEATURES",
    "TARGET_MODULES",
    "TARGET_OUT_FEATURES",
    "decoder_reader_lora_settings_v6",
    "validate_decoder_reader_surface_v6",
]
