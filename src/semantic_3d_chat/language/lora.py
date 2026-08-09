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
import re
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


_LORA_BANK_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class LoRABankSettings:
    """One named, disjoint LoRA bank in a multi-bank installation."""

    name: str
    trainable: bool
    adapter: LoRASettings
    initialization_algorithm: str = "module_default"
    initialization_seed: int | None = None
    expected_initial_state_sha256: str | None = None

    def __post_init__(self) -> None:
        if not _LORA_BANK_NAME.fullmatch(self.name):
            raise ValueError(f"LoRA bank names must match [a-z][a-z0-9_]*: {self.name!r}")
        if not isinstance(self.trainable, bool):
            raise TypeError(f"LoRA bank {self.name!r} trainable must be a boolean")
        if not self.adapter.enabled:
            raise ValueError(f"LoRA bank {self.name!r} must contain an enabled adapter")
        algorithms = {
            "module_default",
            "cpu_kaiming_uniform_a_exact_zero_b",
            "checkpoint_overwrite",
        }
        if self.initialization_algorithm not in algorithms:
            raise ValueError(
                f"LoRA bank {self.name!r} has unsupported initialization algorithm: "
                f"{self.initialization_algorithm!r}"
            )
        if self.initialization_algorithm == "cpu_kaiming_uniform_a_exact_zero_b":
            if (
                isinstance(self.initialization_seed, bool)
                or not isinstance(self.initialization_seed, int)
                or self.initialization_seed < 0
            ):
                raise ValueError(
                    f"LoRA bank {self.name!r} deterministic initialization requires a "
                    "non-negative integer seed"
                )
        elif self.initialization_seed is not None:
            raise ValueError(
                f"LoRA bank {self.name!r} initialization_seed is only valid for "
                "cpu_kaiming_uniform_a_exact_zero_b"
            )
        if self.expected_initial_state_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.expected_initial_state_sha256
        ):
            raise ValueError(
                f"LoRA bank {self.name!r} expected_initial_state_sha256 must be lowercase hex"
            )


@dataclass(frozen=True)
class LoRABanksSettings:
    """Canonical collection contract, including legacy single-bank mode."""

    banks: tuple[LoRABankSettings, ...] = ()
    legacy_single_bank: bool = False

    def __post_init__(self) -> None:
        names = [bank.name for bank in self.banks]
        if len(set(names)) != len(names):
            raise ValueError("language.lora_banks contains duplicate bank names")
        targets = [target for bank in self.banks for target in bank.adapter.target_modules]
        if len(set(targets)) != len(targets):
            raise ValueError("LoRA target modules must be disjoint across all banks")
        if self.legacy_single_bank and len(self.banks) > 1:
            raise ValueError("Legacy LoRA mode can contain at most one bank")

    @property
    def enabled(self) -> bool:
        return bool(self.banks)

    @property
    def trainable(self) -> bool:
        return any(bank.trainable for bank in self.banks)

    def bank(self, name: str) -> LoRABankSettings:
        for bank in self.banks:
            if bank.name == name:
                return bank
        raise KeyError(f"Unknown LoRA bank: {name}")

    def contract(self) -> dict[str, Any]:
        if self.legacy_single_bank:
            return self.banks[0].adapter.contract() if self.banks else LoRASettings().contract()
        return {
            "schema_version": 2,
            "enabled": self.enabled,
            "banks": [
                {
                    "name": bank.name,
                    "trainable": bank.trainable,
                    "rank": int(bank.adapter.rank),
                    "alpha": float(bank.adapter.alpha),
                    "dropout": float(bank.adapter.dropout),
                    "target_modules": list(bank.adapter.target_modules),
                    "initialization_algorithm": bank.initialization_algorithm,
                    "initialization_seed": bank.initialization_seed,
                    "expected_initial_state_sha256": bank.expected_initial_state_sha256,
                }
                for bank in self.banks
            ],
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


def lora_banks_settings(config: Mapping[str, Any]) -> LoRABanksSettings:
    """Parse named banks while preserving the historical single-bank config.

    ``language.lora`` remains byte-for-byte compatible with schema-1
    checkpoints. New experiments may instead provide a mapping under
    ``language.lora_banks``. The two forms are intentionally mutually
    exclusive so a checkpoint can never depend on an implicit installation
    order or an accidentally duplicated residual path.
    """

    language = config.get("language")
    if not isinstance(language, Mapping):
        raise TypeError("config.language must be a mapping")
    if language.get("lora") is not None and language.get("lora_banks") is not None:
        raise ValueError("language.lora and language.lora_banks are mutually exclusive")
    raw_banks = language.get("lora_banks")
    if raw_banks is None:
        legacy = lora_settings(config)
        if not legacy.enabled:
            return LoRABanksSettings(legacy_single_bank=True)
        return LoRABanksSettings(
            banks=(LoRABankSettings("legacy", True, legacy),),
            legacy_single_bank=True,
        )
    if str(language.get("backend", "auto")).casefold() != "gemma4":
        raise ValueError("This controlled LoRA path requires language.backend: gemma4")
    if not isinstance(raw_banks, Mapping) or not raw_banks:
        raise TypeError("language.lora_banks must be a non-empty mapping keyed by bank name")

    banks: list[LoRABankSettings] = []
    allowed = {
        "trainable",
        "rank",
        "alpha",
        "dropout",
        "target_modules",
        "initialization_algorithm",
        "initialization_seed",
        "expected_initial_state_sha256",
    }
    required = allowed
    for name, raw in raw_banks.items():
        if not isinstance(name, str):
            raise TypeError("LoRA bank names must be strings")
        if not isinstance(raw, Mapping):
            raise TypeError(f"language.lora_banks.{name} must be a mapping")
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown language.lora_banks.{name} settings: {unknown}")
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"language.lora_banks.{name} is missing settings: {missing}")
        targets = raw["target_modules"]
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise TypeError(
                f"language.lora_banks.{name}.target_modules must be a sequence of exact paths"
            )
        adapter = LoRASettings(
            enabled=True,
            rank=raw["rank"],
            alpha=raw["alpha"],
            dropout=raw["dropout"],
            target_modules=tuple(targets),
        )
        banks.append(
            LoRABankSettings(
                name=name,
                trainable=raw["trainable"],
                adapter=adapter,
                initialization_algorithm=raw["initialization_algorithm"],
                initialization_seed=raw["initialization_seed"],
                expected_initial_state_sha256=raw["expected_initial_state_sha256"],
            )
        )
    return LoRABanksSettings(tuple(banks))


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


def lora_banks_optimizer_settings(
    config: Mapping[str, Any], settings: LoRABanksSettings
) -> LoRAOptimizerSettings | None:
    """Return the shared optimizer contract for all trainable named banks."""

    if settings.legacy_single_bank:
        legacy = settings.banks[0].adapter if settings.banks else LoRASettings()
        return lora_optimizer_settings(config, legacy)
    training = config.get("training")
    if not settings.trainable:
        if training is not None and not isinstance(training, Mapping):
            raise TypeError("config.training must be a mapping")
        return None
    # Reuse the strict legacy numeric parser with an enabled sentinel. The
    # actual targets/rank remain bank-specific and are never taken from it.
    sentinel = settings.banks[0].adapter
    return lora_optimizer_settings(config, sentinel)


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


@dataclass(frozen=True)
class InstalledLoRABank:
    settings: LoRABankSettings
    installation: LoRAInstallation


@dataclass
class LoRABankCollection:
    """Auditable collection of disjoint frozen and trainable LoRA banks."""

    settings: LoRABanksSettings
    banks: tuple[InstalledLoRABank, ...]

    def __post_init__(self) -> None:
        expected = [bank.name for bank in self.settings.banks]
        observed = [bank.settings.name for bank in self.banks]
        if observed != expected:
            raise ValueError(
                f"Installed LoRA bank order mismatch: configured={expected}, installed={observed}"
            )

    @property
    def bank_names(self) -> tuple[str, ...]:
        return tuple(bank.settings.name for bank in self.banks)

    @property
    def adapters(self) -> tuple[LoRALinear, ...]:
        """Flatten installed adapters for legacy audit/test introspection."""

        return tuple(adapter for bank in self.banks for adapter in bank.installation.adapters)

    def bank(self, name: str) -> InstalledLoRABank:
        for bank in self.banks:
            if bank.settings.name == name:
                return bank
        raise KeyError(f"Unknown installed LoRA bank: {name}")

    def parameters(self) -> list[nn.Parameter]:
        """Return only optimizer-authorized parameters."""

        return [
            parameter
            for bank in self.banks
            if bank.settings.trainable
            for parameter in bank.installation.parameters()
        ]

    def all_parameters(self) -> list[nn.Parameter]:
        return [parameter for bank in self.banks for parameter in bank.installation.parameters()]

    @property
    def parameter_counts(self) -> dict[str, int]:
        return {bank.settings.name: bank.installation.parameter_count for bank in self.banks}

    @property
    def parameter_count(self) -> int:
        return sum(self.parameter_counts.values())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            bank.installation.parameter_count for bank in self.banks if bank.settings.trainable
        )

    @property
    def wrapped_modules(self) -> dict[str, list[str]]:
        return {bank.settings.name: list(bank.installation.target_names) for bank in self.banks}

    @property
    def trainable_parameter_counts(self) -> dict[str, dict[str, int]]:
        return {
            bank.settings.name: bank.installation.parameter_counts
            for bank in self.banks
            if bank.settings.trainable
        }

    def state_modules(self) -> dict[str, nn.Module]:
        if self.settings.legacy_single_bank:
            return {} if not self.banks else {"lora": self.banks[0].installation.state_module}
        return {
            f"lora_banks.{bank.settings.name}": bank.installation.state_module
            for bank in self.banks
        }

    def state_sha256(self) -> dict[str, str]:
        return {bank.settings.name: bank.installation.state_sha256() for bank in self.banks}

    def checkpoint_metadata(self) -> dict[str, Any]:
        if self.settings.legacy_single_bank:
            if not self.banks:
                return {}
            installation = self.banks[0].installation
            return {
                "lora_wrapped_modules": list(installation.target_names),
                "lora_trainable_parameter_counts": installation.parameter_counts,
                "lora_trainable_parameter_count": installation.parameter_count,
                "lora_state_sha256": installation.state_sha256(),
            }
        return {
            "lora_bank_wrapped_modules": self.wrapped_modules,
            "lora_bank_parameter_counts": {
                bank.settings.name: bank.installation.parameter_counts for bank in self.banks
            },
            "lora_bank_state_sha256": self.state_sha256(),
            "lora_parameter_count": self.parameter_count,
            "lora_trainable_parameter_count": self.trainable_parameter_count,
        }

    def train(self, mode: bool = True) -> LoRABankCollection:
        if not isinstance(mode, bool):
            raise TypeError("LoRA train mode must be a boolean")
        for bank in self.banks:
            bank.installation.train(mode if bank.settings.trainable else False)
        return self

    @property
    def training(self) -> bool:
        modes = {bank.installation.training for bank in self.banks if bank.settings.trainable}
        if not modes:
            return False
        if len(modes) != 1:
            raise RuntimeError("Trainable LoRA banks have inconsistent train/eval modes")
        return modes.pop()

    def eval(self) -> LoRABankCollection:
        return self.train(False)

    def assert_trainable_surface(self, model: nn.Module) -> None:
        trainable_ids = {id(parameter) for parameter in self.parameters()}
        all_adapter_ids = {id(parameter) for parameter in self.all_parameters()}
        missing = [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in trainable_ids and not parameter.requires_grad
        ]
        frozen_bank_trainable = [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in all_adapter_ids
            and id(parameter) not in trainable_ids
            and parameter.requires_grad
        ]
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in trainable_ids
        ]
        if missing or frozen_bank_trainable or unexpected:
            raise RuntimeError(
                "Invalid multi-bank LoRA trainable parameter surface: "
                f"frozen_trainable={frozen_bank_trainable}, "
                f"missing_trainable={missing}, unexpected_trainable={unexpected}"
            )

    def gradient_norms(self) -> dict[str, Any]:
        by_bank: dict[str, Any] = {}
        squared_total = 0.0
        for bank in self.banks:
            if not bank.settings.trainable:
                continue
            values = bank.installation.gradient_norms()
            by_bank[bank.settings.name] = values
            squared_total += float(values["total_l2"]) ** 2
        return {"total_l2": math.sqrt(squared_total), "by_bank": by_bank}

    def validate_state(self) -> None:
        for bank in self.banks:
            bank.installation.validate_state()


def install_lora_banks(model: nn.Module, settings: LoRABanksSettings) -> LoRABankCollection | None:
    """Atomically install every exact target, then apply bank trainability."""

    if not settings.enabled:
        return None
    resolved: list[tuple[LoRABankSettings, str, nn.Module, str, nn.Linear]] = []
    for bank in settings.banks:
        for path in bank.adapter.target_modules:
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
            resolved.append((bank, path, parent, attribute, base))

    installed: list[InstalledLoRABank] = []
    for bank in settings.banks:
        adapters: list[LoRALinear] = []
        for resolved_bank, _path, parent, attribute, base in resolved:
            if resolved_bank.name != bank.name:
                continue
            adapter = LoRALinear(
                base,
                rank=int(bank.adapter.rank),
                alpha=float(bank.adapter.alpha),
                dropout=float(bank.adapter.dropout),
            )
            if bank.trainable:
                adapter.train(base.training)
            else:
                adapter.requires_grad_(False)
                adapter.eval()
            setattr(parent, attribute, adapter)
            adapters.append(adapter)
        installation = LoRAInstallation(
            settings=bank.adapter,
            adapters=tuple(adapters),
            state_module=LoRAAdapterState(bank.adapter.target_modules, adapters),
        )
        if bank.initialization_algorithm == "cpu_kaiming_uniform_a_exact_zero_b":
            assert bank.initialization_seed is not None
            initialize_lora_adapter_state(installation, seed=bank.initialization_seed)
            expected_hash = bank.expected_initial_state_sha256
            observed_hash = installation.state_sha256()
            if expected_hash is not None and observed_hash != expected_hash:
                raise ValueError(
                    f"LoRA bank {bank.name!r} deterministic initial-state hash mismatch: "
                    f"expected={expected_hash} observed={observed_hash}"
                )
        installed.append(InstalledLoRABank(bank, installation))
    collection = LoRABankCollection(settings=settings, banks=tuple(installed))
    collection.assert_trainable_surface(model)
    collection.validate_state()
    return collection


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


def initialize_lora_adapter_state(installation: LoRAInstallation, *, seed: int) -> None:
    """Reset A deterministically on CPU and every B tensor exactly to zero."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("LoRA initialization seed must be a non-negative integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for adapter in installation.adapters:
            source = torch.empty(adapter.lora_a.shape, dtype=torch.float32, device="cpu")
            nn.init.kaiming_uniform_(source, a=math.sqrt(5), generator=generator)
            adapter.lora_a.copy_(source.to(adapter.lora_a.device))
            adapter.lora_b.zero_()
    installation.validate_state()
    if any(torch.count_nonzero(adapter.lora_b).item() for adapter in installation.adapters):
        raise RuntimeError("LoRA bank did not initialize to exact zero output")


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
        # ``view(torch.uint8)`` rejects a zero-dimensional scalar when the
        # element sizes differ. Flattening first preserves the exact byte
        # order for every existing non-scalar state while making scalar model
        # buffers hashable as well.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
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


def lora_banks_checkpoint_contract(
    settings: LoRABanksSettings,
    optimizer: LoRAOptimizerSettings | None,
    parameter_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build schema 1 for legacy configs or schema 2 for named banks."""

    counts = dict(parameter_counts)
    expected_names = [bank.name for bank in settings.banks]
    if sorted(counts) != sorted(expected_names):
        raise ValueError(
            "LoRA bank parameter-count keys do not match configured banks: "
            f"counts={sorted(counts)} configured={sorted(expected_names)}"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise ValueError("Every enabled LoRA bank must have a positive integer parameter count")
    if settings.legacy_single_bank:
        legacy = settings.banks[0].adapter if settings.banks else LoRASettings()
        count = counts[settings.banks[0].name] if settings.banks else 0
        return lora_checkpoint_contract(legacy, optimizer, count)
    if not settings.enabled:
        if optimizer is not None or counts:
            raise ValueError("Disabled LoRA banks must have no optimizer or parameters")
        return {"schema_version": 2, "enabled": False, "banks": []}
    if settings.trainable and optimizer is None:
        raise ValueError("Trainable LoRA banks require an optimizer contract")
    architecture = settings.contract()
    architecture_by_name = {record["name"]: record for record in architecture["banks"]}
    banks: list[dict[str, Any]] = []
    for bank in settings.banks:
        record = {
            **architecture_by_name[bank.name],
            "adapter_parameter_count": counts[bank.name],
        }
        if bank.trainable:
            assert optimizer is not None
            record.update(optimizer.contract())
        banks.append(record)
    return {
        **architecture,
        "banks": banks,
        "adapter_parameter_count": sum(counts.values()),
        "trainable_adapter_parameter_count": sum(
            counts[bank.name] for bank in settings.banks if bank.trainable
        ),
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


def validate_lora_banks_checkpoint_state(
    metadata: Mapping[str, Any], collection: LoRABankCollection
) -> None:
    """Validate legacy state or every named bank against checkpoint hashes."""

    collection.validate_state()
    if collection.settings.legacy_single_bank:
        if collection.banks:
            validate_lora_checkpoint_state(metadata, collection.banks[0].installation)
        return
    expected_metadata = collection.checkpoint_metadata()
    mismatches: dict[str, Any] = {}
    for key, runtime_value in expected_metadata.items():
        checkpoint_value = metadata.get(key)
        if checkpoint_value != runtime_value:
            mismatches[key] = {"checkpoint": checkpoint_value, "runtime": runtime_value}
    for bank in collection.banks:
        expected_initial = bank.settings.expected_initial_state_sha256
        if not bank.settings.trainable and expected_initial is not None:
            observed = bank.installation.state_sha256()
            if observed != expected_initial:
                mismatches[f"{bank.settings.name}.frozen_expected_state_sha256"] = {
                    "checkpoint": observed,
                    "runtime": expected_initial,
                }
    if mismatches:
        raise ValueError(f"LoRA bank checkpoint state mismatch or tamper detected: {mismatches}")


__all__ = [
    "InstalledLoRABank",
    "LoRAAdapterState",
    "LoRABankCollection",
    "LoRABankSettings",
    "LoRABanksSettings",
    "LoRAInstallation",
    "LoRALinear",
    "LoRAOptimizerSettings",
    "LoRASettings",
    "initialize_lora_adapter_state",
    "install_lora_adapters",
    "install_lora_banks",
    "lora_banks_checkpoint_contract",
    "lora_banks_optimizer_settings",
    "lora_banks_settings",
    "lora_checkpoint_contract",
    "lora_checkpoint_contract_mismatch",
    "lora_optimizer_settings",
    "lora_settings",
    "tensor_state_sha256",
    "validate_lora_banks_checkpoint_state",
    "validate_lora_checkpoint_state",
]
