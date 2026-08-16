"""Model-free contract for an exact frozen scene-one direct-memory stack.

The versioned chat runtime supplies final state hashes only after a sealed model
gate passes.  This shared validator contains no experiment paths, scorer imports,
questions, answers, object labels, model loads, or artifact writes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

SCENE_ID: Final[str] = "scene_000001"
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_TARGET: Final[re.Pattern[str]] = re.compile(
    r"model\.language_model\.layers\.[0-9]+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:up|down|gate)_proj)"
)


@dataclass(frozen=True)
class FrozenRuntimeBankContract:
    name: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: float
    parameter_count: int
    state_sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "a").isalnum():
            raise ValueError("Strict runtime bank name is invalid")
        if not self.target_modules or any(
            _TARGET.fullmatch(target) is None for target in self.target_modules
        ):
            raise ValueError(f"Strict runtime targets are invalid: {self.name}")
        if self.rank < 1 or self.alpha <= 0.0 or self.parameter_count < 1:
            raise ValueError(f"Strict runtime topology is invalid: {self.name}")
        if _SHA256.fullmatch(self.state_sha256) is None:
            raise ValueError(f"Strict runtime state hash is invalid: {self.name}")


@dataclass(frozen=True)
class StrictScene1StackContract:
    banks: tuple[FrozenRuntimeBankContract, ...]
    expected_total_parameter_count: int

    def __post_init__(self) -> None:
        names = tuple(bank.name for bank in self.banks)
        if not names or len(set(names)) != len(names):
            raise ValueError("Strict scene-one stack bank names must be unique")
        targets = [target for bank in self.banks for target in bank.target_modules]
        if len(set(targets)) != len(targets):
            raise ValueError("Strict scene-one stack target modules must be disjoint")
        if sum(bank.parameter_count for bank in self.banks) != (
            self.expected_total_parameter_count
        ):
            raise ValueError("Strict scene-one stack parameter total changed")

    @property
    def bank_order(self) -> tuple[str, ...]:
        return tuple(bank.name for bank in self.banks)

    @property
    def state_by_bank(self) -> dict[str, str]:
        return {bank.name: bank.state_sha256 for bank in self.banks}


def validate_strict_scene1_stack(
    *,
    scene_id: str,
    runtime_config: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    contract: StrictScene1StackContract,
) -> None:
    """Require identical, frozen, ordered banks in config and checkpoint."""

    if scene_id != SCENE_ID:
        raise ValueError("Strict scene-one runtime accepts only scene_000001")
    language = runtime_config.get("language")
    configured = language.get("lora_banks") if isinstance(language, Mapping) else None
    lora = checkpoint_metadata.get("lora")
    metadata_banks = lora.get("banks") if isinstance(lora, Mapping) else None
    states = checkpoint_metadata.get("lora_bank_state_sha256")
    modules = checkpoint_metadata.get("lora_bank_wrapped_modules")
    counts = checkpoint_metadata.get("lora_bank_parameter_counts")
    if (
        not isinstance(configured, Mapping)
        or tuple(configured) != contract.bank_order
        or not isinstance(metadata_banks, list)
        or not all(isinstance(row, Mapping) for row in metadata_banks)
        or tuple(str(row.get("name")) for row in metadata_banks)
        != contract.bank_order
        or not isinstance(states, Mapping)
        or set(states) != set(contract.bank_order)
        or not isinstance(modules, Mapping)
        or set(modules) != set(contract.bank_order)
        or not isinstance(counts, Mapping)
        or set(counts) != set(contract.bank_order)
        or lora.get("adapter_parameter_count")
        != contract.expected_total_parameter_count
        or lora.get("trainable_adapter_parameter_count") != 0
        or checkpoint_metadata.get("lora_parameter_count")
        != contract.expected_total_parameter_count
        or checkpoint_metadata.get("lora_trainable_parameter_count") != 0
    ):
        raise ValueError("Strict scene-one frozen stack inventory changed")
    metadata_by_name = {str(row["name"]): row for row in metadata_banks}
    for bank in contract.banks:
        configured_row = configured[bank.name]
        metadata_row = metadata_by_name[bank.name]
        bank_counts = counts.get(bank.name)
        if (
            not isinstance(configured_row, Mapping)
            or not isinstance(bank_counts, Mapping)
            or configured_row.get("trainable") is not False
            or configured_row.get("rank") != bank.rank
            or float(configured_row.get("alpha", -1.0)) != bank.alpha
            or float(configured_row.get("dropout", -1.0)) != 0.0
            or tuple(configured_row.get("target_modules", ())) != bank.target_modules
            or configured_row.get("expected_initial_state_sha256")
            != bank.state_sha256
            or metadata_row.get("trainable") is not False
            or metadata_row.get("rank") != bank.rank
            or float(metadata_row.get("alpha", -1.0)) != bank.alpha
            or float(metadata_row.get("dropout", -1.0)) != 0.0
            or tuple(metadata_row.get("target_modules", ())) != bank.target_modules
            or metadata_row.get("adapter_parameter_count") != bank.parameter_count
            or states.get(bank.name) != bank.state_sha256
            or tuple(modules.get(bank.name, ())) != bank.target_modules
            or sum(int(value) for value in bank_counts.values())
            != bank.parameter_count
        ):
            raise ValueError(f"Strict scene-one bank changed: {bank.name}")


def extend_stack_contract(
    parent: StrictScene1StackContract,
    additions: Sequence[FrozenRuntimeBankContract],
) -> StrictScene1StackContract:
    """Create exact parent-plus-addition order without mutating the parent."""

    added = tuple(additions)
    return StrictScene1StackContract(
        banks=parent.banks + added,
        expected_total_parameter_count=parent.expected_total_parameter_count
        + sum(bank.parameter_count for bank in added),
    )


__all__ = [
    "SCENE_ID",
    "FrozenRuntimeBankContract",
    "StrictScene1StackContract",
    "extend_stack_contract",
    "validate_strict_scene1_stack",
]
