"""Strict, merge-free LoRA support for a frozen local language model.

Adapters are installed only at complete module paths listed in configuration.
There is deliberately no suffix matching, regular expression, PEFT dependency,
or weight merge.  That makes the small trainable surface and checkpoint payload
straightforward to audit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Wrap a frozen linear layer with an unmerged FP32 low-rank residual.

    The base layer is retained by reference and frozen immediately. Adapter A
    is Kaiming-initialized and adapter B is exactly zero, so construction is an
    exact functional no-op. Forward evaluation never mutates or merges the base
    weight. The low-rank path is evaluated in float32 and cast to the base
    output dtype only for the final residual addition.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be torch.nn.Linear, got {type(base).__name__}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a finite positive number")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive number")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a number in [0, 1)")
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a number in [0, 1)")

        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        device = base.weight.device
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=device, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def adapter_parameter_count(self) -> int:
        return self.rank * (self.in_features + self.out_features)

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only A/B for an optimizer, excluding the frozen base layer."""

        yield self.lora_a
        yield self.lora_b

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> LoRALinear:
        """Honor device moves while keeping adapter storage in float32.

        Parent-module ``to(dtype=...)`` calls recurse through ``_apply``. Gemma
        may therefore move or cast a wrapped decoder after adapter insertion.
        The base follows that request, while A/B are restored to float32 on the
        resulting device.
        """

        super()._apply(fn, recurse=recurse)
        for parameter in (self.lora_a, self.lora_b):
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
            if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                parameter.grad.data = parameter.grad.data.float()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_inputs = self.dropout(inputs.float())
        hidden = F.linear(adapter_inputs, self.lora_a)
        update = F.linear(hidden, self.lora_b) * self.scaling
        return base_output + update.to(dtype=base_output.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha:g}, "
            f"dropout={self.dropout.p:g}, merge=False"
        )


@dataclass(frozen=True)
class LoRASettings:
    """Architecture-only LoRA settings shared by training and inference."""

    enabled: bool = False
    rank: int | None = None
    alpha: float | None = None
    dropout: float | None = None
    target_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("language.lora.enabled must be a boolean")
        if not self.enabled:
            if (
                any(value is not None for value in (self.rank, self.alpha, self.dropout))
                or self.target_modules
            ):
                raise ValueError("disabled LoRA settings cannot define adapter parameters")
            return
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("language.lora.rank must be a positive integer")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError("language.lora.alpha must be a finite positive number")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0.0:
            raise ValueError("language.lora.alpha must be a finite positive number")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("language.lora.dropout must be a number in [0, 1)")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("language.lora.dropout must be a number in [0, 1)")
        if not self.target_modules:
            raise ValueError("language.lora.target_modules must not be empty")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("language.lora.target_modules contains duplicate paths")
        for name in self.target_modules:
            if not isinstance(name, str) or not name or name.strip() != name:
                raise ValueError("LoRA target module paths must be non-empty trimmed strings")
            if any(character in name for character in "*?[]"):
                raise ValueError("LoRA target module paths must be exact; globs are forbidden")
            if name.startswith(".") or name.endswith(".") or ".." in name:
                raise ValueError(f"Invalid exact LoRA target module path: {name!r}")

    def contract(self) -> dict[str, Any]:
        if not self.enabled:
            return {"schema_version": 1, "enabled": False}
        return {
            "schema_version": 1,
            "enabled": True,
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "dropout": float(self.dropout),
            "target_modules": list(self.target_modules),
        }


@dataclass(frozen=True)
class LoRAOptimizerSettings:
    learning_rate: float
    weight_decay: float

    def contract(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


def lora_settings(config: Mapping[str, Any]) -> LoRASettings:
    """Parse a strict opt-in LoRA contract; missing configuration is disabled."""

    language = config.get("language")
    if not isinstance(language, Mapping):
        raise TypeError("config.language must be a mapping")
    raw = language.get("lora")
    if raw is None:
        return LoRASettings()
    if not isinstance(raw, Mapping):
        raise TypeError("language.lora must be a mapping")
    allowed = {"enabled", "rank", "alpha", "dropout", "target_modules"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown language.lora settings: {unknown}")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("language.lora.enabled must be a boolean")
    if not enabled:
        if set(raw) - {"enabled"}:
            raise ValueError("disabled language.lora may contain only enabled: false")
        return LoRASettings()
    missing = sorted({"rank", "alpha", "dropout", "target_modules"} - set(raw))
    if missing:
        raise ValueError(f"Enabled language.lora is missing settings: {missing}")
    if str(language.get("backend", "auto")).casefold() != "gemma4":
        raise ValueError("This controlled LoRA path requires language.backend: gemma4")
    targets = raw["target_modules"]
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise TypeError("language.lora.target_modules must be a sequence of exact paths")
    return LoRASettings(
        enabled=True,
        rank=raw["rank"],
        alpha=raw["alpha"],
        dropout=raw["dropout"],
        target_modules=tuple(targets),
    )


def lora_optimizer_settings(
    config: Mapping[str, Any], settings: LoRASettings
) -> LoRAOptimizerSettings | None:
    """Return the explicit optimizer contract for an enabled LoRA experiment."""

    training = config.get("training")
    if not settings.enabled:
        if training is None:
            return None
        if not isinstance(training, Mapping):
            raise TypeError("config.training must be a mapping")
        present = {key for key in ("lora_learning_rate", "lora_weight_decay") if key in training}
        if present:
            raise ValueError("LoRA optimizer settings are forbidden while LoRA is disabled")
        return None
    if not isinstance(training, Mapping):
        raise TypeError("config.training must be a mapping")
    missing = sorted({"lora_learning_rate", "lora_weight_decay"} - set(training))
    if missing:
        raise ValueError(f"Enabled LoRA is missing explicit optimizer settings: {missing}")
    learning_rate = training["lora_learning_rate"]
    weight_decay = training["lora_weight_decay"]
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError("training.lora_learning_rate must be a finite positive number")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("training.lora_learning_rate must be a finite positive number")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)):
        raise TypeError("training.lora_weight_decay must be a finite non-negative number")
    if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
        raise ValueError("training.lora_weight_decay must be a finite non-negative number")
    return LoRAOptimizerSettings(float(learning_rate), float(weight_decay))


class _LoRAParameterPair(nn.Module):
    """A checkpoint-only view of parameters already owned by a LoRALinear."""

    def __init__(self, adapter: LoRALinear) -> None:
        super().__init__()
        self.lora_a = adapter.lora_a
        self.lora_b = adapter.lora_b


class LoRAAdapterState(nn.Module):
    """Compact state module containing only LoRA A/B tensors, never base weights."""

    def __init__(self, target_names: Sequence[str], adapters: Sequence[LoRALinear]) -> None:
        super().__init__()
        if len(target_names) != len(adapters):
            raise ValueError("LoRA target/adaptor count mismatch")
        self.target_names = tuple(target_names)
        self.adapters = nn.ModuleList(_LoRAParameterPair(adapter) for adapter in adapters)


@dataclass
class LoRAInstallation:
    settings: LoRASettings
    adapters: tuple[LoRALinear, ...]
    state_module: LoRAAdapterState

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.settings.target_modules

    def parameters(self) -> list[nn.Parameter]:
        return [
            parameter for adapter in self.adapters for parameter in adapter.adapter_parameters()
        ]

    @property
    def training(self) -> bool:
        modes = {adapter.training for adapter in self.adapters}
        if len(modes) != 1:
            raise RuntimeError("Installed LoRA adapters have inconsistent train/eval modes")
        return modes.pop()

    def train(self, mode: bool = True) -> LoRAInstallation:
        if not isinstance(mode, bool):
            raise TypeError("LoRA train mode must be a boolean")
        for adapter in self.adapters:
            adapter.train(mode)
        return self

    def eval(self) -> LoRAInstallation:
        return self.train(False)

    @property
    def parameter_counts(self) -> dict[str, int]:
        return {
            name: adapter.adapter_parameter_count
            for name, adapter in zip(self.target_names, self.adapters, strict=True)
        }

    @property
    def parameter_count(self) -> int:
        return sum(self.parameter_counts.values())

    def assert_only_lora_trainable(self, model: nn.Module) -> None:
        adapter_ids = {id(parameter) for parameter in self.parameters()}
        missing = [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in adapter_ids and not parameter.requires_grad
        ]
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in adapter_ids
        ]
        if missing or unexpected:
            raise RuntimeError(
                "Invalid LoRA trainable parameter surface: "
                f"frozen_adapters={missing}, unexpected_trainable={unexpected}"
            )

    def gradient_norms(self) -> dict[str, Any]:
        squared_total = 0.0
        by_module: dict[str, dict[str, float | None]] = {}
        for name, adapter in zip(self.target_names, self.adapters, strict=True):
            values: dict[str, float | None] = {}
            module_squared = 0.0
            for short_name, parameter in (("lora_a", adapter.lora_a), ("lora_b", adapter.lora_b)):
                norm = (
                    None
                    if parameter.grad is None
                    else float(parameter.grad.detach().float().norm().cpu())
                )
                values[short_name] = norm
                if norm is not None:
                    module_squared += norm * norm
            values["total_l2"] = math.sqrt(module_squared)
            by_module[name] = values
            squared_total += module_squared
        return {"total_l2": math.sqrt(squared_total), "by_module": by_module}

    def state_sha256(self) -> str:
        return tensor_state_sha256(self.state_module.state_dict())

    def validate_state(self) -> None:
        for name, adapter in zip(self.target_names, self.adapters, strict=True):
            for short_name, parameter in (("lora_a", adapter.lora_a), ("lora_b", adapter.lora_b)):
                if parameter.dtype != torch.float32:
                    raise TypeError(f"{name}.{short_name} must remain float32")
                if not torch.isfinite(parameter).all():
                    raise ValueError(f"{name}.{short_name} contains NaN or infinity")


def install_lora_adapters(model: nn.Module, settings: LoRASettings) -> LoRAInstallation | None:
    """Replace exact configured linear modules after validating the full target set."""

    if not settings.enabled:
        return None
    resolved: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for path in settings.target_modules:
        parent_path, separator, attribute = path.rpartition(".")
        if not separator:
            raise ValueError(f"LoRA target must include its complete parent path: {path!r}")
        try:
            parent = model.get_submodule(parent_path)
        except AttributeError as exc:
            raise ValueError(f"LoRA target module does not exist: {path}") from exc
        base = getattr(parent, attribute, None)
        if isinstance(base, LoRALinear):
            raise TypeError(f"LoRA target is already wrapped: {path}")
        if not isinstance(base, nn.Linear):
            observed = type(base).__name__ if base is not None else "<missing>"
            raise TypeError(f"LoRA target must be torch.nn.Linear: {path} ({observed})")
        resolved.append((path, parent, attribute, base))

    adapters: list[LoRALinear] = []
    for _path, parent, attribute, base in resolved:
        adapter = LoRALinear(
            base,
            rank=int(settings.rank),
            alpha=float(settings.alpha),
            dropout=float(settings.dropout),
        )
        adapter.train(base.training)
        setattr(parent, attribute, adapter)
        adapters.append(adapter)
    installation = LoRAInstallation(
        settings=settings,
        adapters=tuple(adapters),
        state_module=LoRAAdapterState(settings.target_modules, adapters),
    )
    installation.assert_only_lora_trainable(model)
    installation.validate_state()
    return installation


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor identities, shapes, dtypes and exact bytes deterministically."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lora_checkpoint_contract_mismatch(
    metadata: Mapping[str, Any], expected_contract: Mapping[str, Any]
) -> dict[str, Any] | None:
    expected = dict(expected_contract)
    observed = metadata.get("lora", {"schema_version": 1, "enabled": False})
    if observed == expected:
        return None
    return {"checkpoint": observed, "runtime": expected}


def lora_checkpoint_contract(
    settings: LoRASettings,
    optimizer: LoRAOptimizerSettings | None,
    adapter_parameter_count: int,
) -> dict[str, Any]:
    """Build the single deterministic checkpoint/runtime LoRA contract."""

    if isinstance(adapter_parameter_count, bool) or not isinstance(adapter_parameter_count, int):
        raise TypeError("adapter_parameter_count must be a non-negative integer")
    if adapter_parameter_count < 0:
        raise ValueError("adapter_parameter_count must be a non-negative integer")
    if not settings.enabled:
        if optimizer is not None or adapter_parameter_count != 0:
            raise ValueError("disabled LoRA must have no optimizer or adapter parameters")
        return {"schema_version": 1, "enabled": False}
    if optimizer is None:
        raise ValueError("enabled LoRA requires an optimizer contract")
    if adapter_parameter_count == 0:
        raise ValueError("enabled LoRA must have trainable adapter parameters")
    return {
        **settings.contract(),
        "learning_rate": optimizer.learning_rate,
        "weight_decay": optimizer.weight_decay,
        "adapter_parameter_count": adapter_parameter_count,
    }


def validate_lora_checkpoint_state(
    metadata: Mapping[str, Any], installation: LoRAInstallation
) -> None:
    """Reject missing, incompatible, non-finite, or content-tampered LoRA state."""

    installation.validate_state()
    expected_names = list(installation.target_names)
    expected_counts = installation.parameter_counts
    observed_names = metadata.get("lora_wrapped_modules")
    observed_counts = metadata.get("lora_trainable_parameter_counts")
    observed_total = metadata.get("lora_trainable_parameter_count")
    expected_hash = metadata.get("lora_state_sha256")
    mismatches: dict[str, Any] = {}
    if observed_names != expected_names:
        mismatches["lora_wrapped_modules"] = {
            "checkpoint": observed_names,
            "runtime": expected_names,
        }
    if observed_counts != expected_counts:
        mismatches["lora_trainable_parameter_counts"] = {
            "checkpoint": observed_counts,
            "runtime": expected_counts,
        }
    if observed_total != installation.parameter_count:
        mismatches["lora_trainable_parameter_count"] = {
            "checkpoint": observed_total,
            "runtime": installation.parameter_count,
        }
    observed_hash = installation.state_sha256()
    if not isinstance(expected_hash, str) or expected_hash != observed_hash:
        mismatches["lora_state_sha256"] = {
            "checkpoint": expected_hash,
            "runtime": observed_hash,
        }
    if mismatches:
        raise ValueError(f"LoRA checkpoint state mismatch or tamper detected: {mismatches}")


__all__ = [
    "LoRAAdapterState",
    "LoRAInstallation",
    "LoRALinear",
    "LoRAOptimizerSettings",
    "LoRASettings",
    "install_lora_adapters",
    "lora_checkpoint_contract",
    "lora_checkpoint_contract_mismatch",
    "lora_optimizer_settings",
    "lora_settings",
    "tensor_state_sha256",
    "validate_lora_checkpoint_state",
]
