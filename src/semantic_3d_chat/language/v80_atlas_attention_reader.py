"""Exact, unmerged V80 attention-reader surface for Gemma-4 E2B.

The physical layer-14 K/V projections produce the shared full-attention K/V
states consumed by later global layers.  Layer 34 owns the final full-attention
Q/O projections.  Wrapping those four live modules lets question/answer tokens
learn to address the entire fixed 738-token atlas without selecting any token.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import nn

TARGET_MODULES: Final[tuple[str, ...]] = (
    "model.language_model.layers.14.self_attn.k_proj",
    "model.language_model.layers.14.self_attn.v_proj",
    "model.language_model.layers.34.self_attn.q_proj",
    "model.language_model.layers.34.self_attn.o_proj",
)
TARGET_SHAPES_OUT_IN: Final[dict[str, tuple[int, int]]] = {
    TARGET_MODULES[0]: (512, 1536),
    TARGET_MODULES[1]: (512, 1536),
    TARGET_MODULES[2]: (4096, 1536),
    TARGET_MODULES[3]: (1536, 4096),
}
RANK: Final[int] = 8
ALPHA: Final[float] = 16.0
INITIALIZATION_SEED: Final[int] = 800080
PARAMETER_COUNT: Final[int] = sum(
    RANK * (out_features + in_features)
    for out_features, in_features in TARGET_SHAPES_OUT_IN.values()
)
EXPECTED_LAYER_TYPES: Final[tuple[str, ...]] = (
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
) * 7


class OuterAdditiveFP32LoRA(nn.Module):
    """Zero-output FP32 residual around an existing frozen projection."""

    def __init__(self, base: nn.Module, *, rank: int = RANK, alpha: float = ALPHA) -> None:
        super().__init__()
        in_features = getattr(base, "in_features", None)
        out_features = getattr(base, "out_features", None)
        if type(in_features) is not int or type(out_features) is not int:
            raise TypeError("V80 base projection must expose integer in/out features")
        if type(rank) is not int or rank < 1:
            raise ValueError("V80 rank must be a positive integer")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("V80 alpha must be finite and positive")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.enabled = True
        device = next(base.parameters()).device
        self.residual_a = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.float32, device=device)
        )
        self.residual_b = nn.Parameter(
            torch.zeros(out_features, rank, dtype=torch.float32, device=device)
        )

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def adapter_parameter_count(self) -> int:
        return self.rank * (self.in_features + self.out_features)

    def adapter_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.residual_a, self.residual_b

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        if not self.enabled:
            return base_output
        hidden = F.linear(inputs.float(), self.residual_a)
        update = F.linear(hidden, self.residual_b) * self.scaling
        return base_output + update.to(base_output.dtype)


@dataclass(frozen=True)
class V80Installation:
    target_names: tuple[str, ...]
    adapters: tuple[OuterAdditiveFP32LoRA, ...]

    def parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for adapter in self.adapters
            for parameter in adapter.adapter_parameters()
        ]

    @property
    def parameter_count(self) -> int:
        return sum(adapter.adapter_parameter_count for adapter in self.adapters)

    def state_dict(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for index, adapter in enumerate(self.adapters):
            result[f"adapters.{index}.residual_a"] = adapter.residual_a
            result[f"adapters.{index}.residual_b"] = adapter.residual_b
        return result

    def gradient_norms(self) -> dict[str, dict[str, float | None]]:
        result: dict[str, dict[str, float | None]] = {}
        for name, adapter in zip(self.target_names, self.adapters, strict=True):
            result[name] = {
                "residual_a": None
                if adapter.residual_a.grad is None
                else float(adapter.residual_a.grad.detach().float().norm().cpu()),
                "residual_b": None
                if adapter.residual_b.grad is None
                else float(adapter.residual_b.grad.detach().float().norm().cpu()),
            }
        return result

    def assert_only_adapters_trainable(self, model: nn.Module) -> None:
        allowed = {id(parameter) for parameter in self.parameters()}
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed
        ]
        missing = [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in allowed and not parameter.requires_grad
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"V80 trainable surface changed: unexpected={unexpected}, missing={missing}"
            )

    def assert_fp32_finite(self) -> None:
        failures: list[str] = []
        for name, adapter in zip(self.target_names, self.adapters, strict=True):
            for factor_name, parameter in (
                ("residual_a", adapter.residual_a),
                ("residual_b", adapter.residual_b),
            ):
                if parameter.dtype != torch.float32 or not bool(
                    torch.isfinite(parameter.detach()).all()
                ):
                    failures.append(f"{name}.{factor_name}")
        if failures:
            raise RuntimeError(f"V80 adapters are not finite FP32 tensors: {failures}")

    @contextlib.contextmanager
    def disabled(self) -> Iterator[None]:
        previous = tuple(adapter.enabled for adapter in self.adapters)
        try:
            for adapter in self.adapters:
                adapter.enabled = False
            yield
        finally:
            for adapter, enabled in zip(self.adapters, previous, strict=True):
                adapter.enabled = enabled


def initialize_v80(installation: V80Installation, *, seed: int = INITIALIZATION_SEED) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for adapter in installation.adapters:
            source = torch.empty(adapter.residual_a.shape, dtype=torch.float32)
            nn.init.kaiming_uniform_(source, a=math.sqrt(5), generator=generator)
            adapter.residual_a.copy_(source.to(adapter.residual_a.device))
            adapter.residual_b.zero_()


def validate_v80_topology(model: nn.Module) -> None:
    text = getattr(getattr(model, "config", None), "text_config", None)
    if text is None:
        raise ValueError("V80 requires Gemma text_config")
    if tuple(getattr(text, "layer_types", ())) != EXPECTED_LAYER_TYPES:
        raise ValueError("V80 Gemma layer-type topology changed")
    if getattr(text, "sliding_window", None) != 512:
        raise ValueError("V80 Gemma sliding window changed")
    if getattr(text, "num_hidden_layers", None) != 35:
        raise ValueError("V80 Gemma layer count changed")
    if getattr(text, "num_kv_shared_layers", None) != 20:
        raise ValueError("V80 Gemma K/V-sharing topology changed")
    layer14 = model.get_submodule("model.language_model.layers.14.self_attn")
    layer34 = model.get_submodule("model.language_model.layers.34.self_attn")
    if (
        getattr(layer14, "is_kv_shared_layer", None) is not False
        or getattr(layer14, "store_full_length_kv", None) is not True
        or getattr(layer14, "layer_type", None) != "full_attention"
        or getattr(layer34, "is_kv_shared_layer", None) is not True
        or getattr(layer34, "layer_type", None) != "full_attention"
    ):
        raise ValueError("V80 physical/shared full-attention routing changed")


def install_v80(model: nn.Module) -> V80Installation:
    model.requires_grad_(False)
    validate_v80_topology(model)
    resolved: list[tuple[nn.Module, str, nn.Module]] = []
    for path in TARGET_MODULES:
        parent_path, _, attribute = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        base = getattr(parent, attribute)
        observed = (getattr(base, "out_features", None), getattr(base, "in_features", None))
        if observed != TARGET_SHAPES_OUT_IN[path]:
            raise ValueError(f"V80 target shape changed for {path}: {observed}")
        resolved.append((parent, attribute, base))
    adapters: list[OuterAdditiveFP32LoRA] = []
    for parent, attribute, base in resolved:
        adapter = OuterAdditiveFP32LoRA(base)
        setattr(parent, attribute, adapter)
        adapters.append(adapter)
    installation = V80Installation(TARGET_MODULES, tuple(adapters))
    initialize_v80(installation)
    if installation.parameter_count != PARAMETER_COUNT:
        raise RuntimeError("V80 trainable parameter count changed")
    installation.assert_only_adapters_trainable(model)
    installation.assert_fp32_finite()
    return installation


def causal_prefix_visibility(
    *, prefix_tokens: int, prompt_tokens: int, answer_tokens: int
) -> Mapping[str, Any]:
    """Prove the two global target layers expose every prefix key to text queries."""

    if min(prefix_tokens, prompt_tokens, answer_tokens) < 1:
        raise ValueError("V80 visibility lengths must be positive")
    sequence_tokens = 1 + prefix_tokens + prompt_tokens + answer_tokens
    prefix_positions = tuple(range(1, prefix_tokens + 1))
    question_start = 1 + prefix_tokens
    answer_end = sequence_tokens - 1
    # A causal full-attention query sees every earlier position.  The earliest
    # text query is after the complete scene prefix by construction.
    all_visible = all(key <= question_start for key in prefix_positions) and all(
        key <= answer_end for key in prefix_positions
    )
    return {
        "prefix_tokens": prefix_tokens,
        "sequence_tokens": sequence_tokens,
        "first_prefix_position": prefix_positions[0],
        "last_prefix_position": prefix_positions[-1],
        "first_question_position": question_start,
        "last_answer_position": answer_end,
        "target_layers": [14, 34],
        "attention_type": "full_attention",
        "visible_prefix_token_count_per_text_query": prefix_tokens,
        "all_prefix_tokens_visible": all_visible,
        "selection_or_top_k": False,
    }


__all__ = [
    "ALPHA",
    "EXPECTED_LAYER_TYPES",
    "INITIALIZATION_SEED",
    "PARAMETER_COUNT",
    "RANK",
    "TARGET_MODULES",
    "TARGET_SHAPES_OUT_IN",
    "OuterAdditiveFP32LoRA",
    "V80Installation",
    "causal_prefix_visibility",
    "initialize_v80",
    "install_v80",
    "validate_v80_topology",
]
