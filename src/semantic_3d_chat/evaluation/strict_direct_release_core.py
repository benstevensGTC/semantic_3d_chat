"""Reusable, model-free primitives for strict direct-memory runtime releases.

This module contains no experiment paths, behavior expectations, model-gate
thresholds, or promotion policy.  A versioned release wrapper must authenticate
those facts first and then supply a complete :class:`BridgeSourceContract` for
every added bank.  The core only composes byte-authenticated, unmerged LoRA
banks while preserving the complete parent archive.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import RUNTIME_METADATA_FILENAME

_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_BANK_NAME: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9_]{0,95}")
_TARGET_MODULE: Final[re.Pattern[str]] = re.compile(
    r"model\.language_model\.layers\.[0-9]+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:up|down|gate)_proj)"
)
_BRIDGE_FILES: Final[frozenset[str]] = frozenset(
    {"bridge.safetensors", RUNTIME_METADATA_FILENAME}
)
_SAFE_TENSOR_METADATA: Final[dict[str, str]] = {
    "environmental_memory_serialized": "false",
    "questions_or_answers_serialized": "false",
    "oracle_serialized": "false",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_exact_directory(root: str | Path, purpose: str) -> Path:
    unresolved = Path(os.path.abspath(Path(root).expanduser()))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} contains a symbolic-link component: {current}")
    if not unresolved.is_dir() or unresolved.is_symlink():
        raise FileNotFoundError(f"{purpose} is unavailable: {unresolved}")
    return unresolved


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


@dataclass(frozen=True)
class BridgeSourceContract:
    """Complete immutable identity of one fixed-final, unmerged LoRA bank."""

    root: Path
    artifact: str
    bank_name: str
    target_module: str
    rank: int
    alpha: float
    dropout: float
    parameter_count: int
    state_sha256: str
    weights_sha256: str
    metadata_sha256: str
    status: str = "fixed_final_awaiting_preregistered_acceptance_gates"

    def __post_init__(self) -> None:
        if _BANK_NAME.fullmatch(self.bank_name) is None:
            raise ValueError(f"Invalid strict-release bank name: {self.bank_name!r}")
        if _TARGET_MODULE.fullmatch(self.target_module) is None:
            raise ValueError(
                f"Invalid strict-release target module: {self.target_module!r}"
            )
        if self.rank < 1 or self.alpha <= 0.0 or self.dropout != 0.0:
            raise ValueError("Strict runtime bridges require positive rank/alpha and zero dropout")
        if self.parameter_count < 1:
            raise ValueError("Strict runtime bridge parameter count must be positive")
        for field in (self.state_sha256, self.weights_sha256, self.metadata_sha256):
            if _SHA256.fullmatch(field) is None:
                raise ValueError("Strict runtime bridge hashes must be lowercase SHA-256")


@dataclass(frozen=True)
class LoadedBridgeSource:
    contract: BridgeSourceContract
    state: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def load_bridge_source(contract: BridgeSourceContract) -> LoadedBridgeSource:
    """Authenticate one exact two-file bridge and return CPU float32 tensors."""

    root = _guard_exact_directory(contract.root, f"{contract.bank_name} bridge")
    if {item.name for item in root.iterdir()} != _BRIDGE_FILES:
        raise ValueError(f"{contract.bank_name} is not an exact two-file bridge")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError(f"{contract.bank_name} bridge files must be physical files")
    weights = root / "bridge.safetensors"
    metadata_path = root / RUNTIME_METADATA_FILENAME
    if (
        sha256_file(weights) != contract.weights_sha256
        or sha256_file(metadata_path) != contract.metadata_sha256
    ):
        raise ValueError(f"{contract.bank_name} source file bytes changed")
    metadata = _strict_json(metadata_path)
    if (
        metadata.get("artifact") != contract.artifact
        or metadata.get("status") != contract.status
        or metadata.get("bank_name") != contract.bank_name
        or metadata.get("target_module") != contract.target_module
        or metadata.get("rank") != contract.rank
        or float(metadata.get("alpha", -1.0)) != contract.alpha
        or float(metadata.get("dropout", -1.0)) != contract.dropout
        or metadata.get("parameter_count") != contract.parameter_count
        or metadata.get("state_sha256") != contract.state_sha256
        or metadata.get("weights_sha256") != contract.weights_sha256
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError(f"{contract.bank_name} bridge metadata changed")
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        tensor_metadata = handle.metadata()
        if not isinstance(tensor_metadata, dict) or any(
            tensor_metadata.get(key) != value
            for key, value in _SAFE_TENSOR_METADATA.items()
        ):
            raise ValueError(f"{contract.bank_name} bridge tensor metadata is unsafe")
        if set(handle.keys()) != {"lora_a", "lora_b"}:
            raise ValueError(f"{contract.bank_name} bridge tensor inventory changed")
        raw_a = handle.get_tensor("lora_a")
        raw_b = handle.get_tensor("lora_b")
    if (
        raw_a.ndim != 2
        or raw_b.ndim != 2
        or raw_a.shape[0] != contract.rank
        or raw_b.shape[1] != contract.rank
        or raw_a.dtype != torch.float32
        or raw_b.dtype != torch.float32
        or raw_a.numel() + raw_b.numel() != contract.parameter_count
        or not bool(torch.isfinite(raw_a).all())
        or not bool(torch.isfinite(raw_b).all())
    ):
        raise ValueError(f"{contract.bank_name} bridge tensor shape or dtype changed")
    state = {
        "adapters.0.lora_a": raw_a.contiguous(),
        "adapters.0.lora_b": raw_b.contiguous(),
    }
    if tensor_state_sha256(state) != contract.state_sha256:
        raise ValueError(f"{contract.bank_name} bridge tensor state changed")
    return LoadedBridgeSource(contract=contract, state=state, metadata=metadata)


def base_bank_order(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    lora = metadata.get("lora")
    banks = lora.get("banks") if isinstance(lora, Mapping) else None
    if not isinstance(banks, list) or not all(isinstance(row, Mapping) for row in banks):
        raise TypeError("Parent runtime named-bank metadata is malformed")
    names = tuple(str(row.get("name")) for row in banks)
    if len(set(names)) != len(names) or any(row.get("trainable") is not False for row in banks):
        raise ValueError("Parent runtime bank inventory is not uniquely frozen")
    return names


def compose_exact_bank_archive(
    *,
    base_checkpoint: str | Path,
    expected_base_banks: Sequence[str],
    added_bridges: Sequence[BridgeSourceContract],
    expected_final_banks: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compose an exact parent-plus-bridge archive without merging any weight."""

    root = _guard_exact_directory(base_checkpoint, "strict parent runtime checkpoint")
    if {item.name for item in root.iterdir()} != {
        "adapter.safetensors",
        RUNTIME_METADATA_FILENAME,
    }:
        raise ValueError("Strict parent runtime checkpoint is not an exact two-file package")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("Strict parent runtime checkpoint files must be physical files")
    metadata = _strict_json(root / RUNTIME_METADATA_FILENAME)
    observed_base = base_bank_order(metadata)
    expected_base = tuple(expected_base_banks)
    expected_final = tuple(expected_final_banks)
    addition_names = tuple(source.bank_name for source in added_bridges)
    if observed_base != expected_base:
        raise ValueError("Strict parent runtime bank order changed")
    if expected_final != expected_base + addition_names:
        raise ValueError("Strict final bank inventory is not exact parent-plus-additions")
    if len(set(expected_final)) != len(expected_final):
        raise ValueError("Strict final bank inventory contains duplicate names")

    base = load_file(str(root / "adapter.safetensors"), device="cpu")
    merged = {name: value.detach().cpu().contiguous() for name, value in base.items()}
    bridge_evidence: list[dict[str, Any]] = []
    for contract in added_bridges:
        loaded = load_bridge_source(contract)
        for suffix, value in loaded.state.items():
            key = f"lora_banks.{contract.bank_name}.{suffix}"
            if key in merged:
                raise ValueError(f"Strict release bank key already exists: {key}")
            merged[key] = value
        bridge_evidence.append(
            {
                "bank_name": contract.bank_name,
                "target_module": contract.target_module,
                "state_sha256": contract.state_sha256,
                "weights_sha256": contract.weights_sha256,
                "metadata_sha256": contract.metadata_sha256,
                "parameter_count": contract.parameter_count,
            }
        )
    retained = {name: merged[name] for name in base}
    if tensor_state_sha256(retained) != tensor_state_sha256(base):
        raise RuntimeError("Strict parent adapter tensors changed during composition")
    return merged, {
        "base_adapter_sha256": sha256_file(root / "adapter.safetensors"),
        "base_metadata_sha256": sha256_file(root / RUNTIME_METADATA_FILENAME),
        "base_bank_order": list(observed_base),
        "final_bank_order": list(expected_final),
        "base_tensor_count": len(base),
        "final_tensor_count": len(merged),
        "added_tensor_count": 2 * len(added_bridges),
        "base_tensors_byte_identical": True,
        "bridges": bridge_evidence,
    }


def extend_runtime_metadata(
    *,
    parent_metadata: Mapping[str, Any],
    added_bridges: Sequence[BridgeSourceContract],
    expected_final_banks: Sequence[str],
) -> dict[str, Any]:
    """Extend named-bank metadata while keeping every added bank frozen."""

    metadata = copy.deepcopy(dict(parent_metadata))
    observed_base = base_bank_order(metadata)
    final = tuple(expected_final_banks)
    if final != observed_base + tuple(source.bank_name for source in added_bridges):
        raise ValueError("Strict metadata final bank order changed")
    lora = metadata["lora"]
    banks = list(lora["banks"])
    states = dict(metadata["lora_bank_state_sha256"])
    modules = dict(metadata["lora_bank_wrapped_modules"])
    counts = dict(metadata["lora_bank_parameter_counts"])
    added_count = 0
    for source in added_bridges:
        banks.append(
            {
                "name": source.bank_name,
                "trainable": False,
                "rank": source.rank,
                "alpha": source.alpha,
                "dropout": source.dropout,
                "target_modules": [source.target_module],
                "initialization_algorithm": "checkpoint_overwrite",
                "initialization_seed": None,
                "expected_initial_state_sha256": source.state_sha256,
                "adapter_parameter_count": source.parameter_count,
            }
        )
        states[source.bank_name] = source.state_sha256
        modules[source.bank_name] = [source.target_module]
        counts[source.bank_name] = {source.target_module: source.parameter_count}
        added_count += source.parameter_count
    total = int(lora["adapter_parameter_count"]) + added_count
    metadata["lora"] = {
        "schema_version": 2,
        "enabled": True,
        "banks": banks,
        "adapter_parameter_count": total,
        "trainable_adapter_parameter_count": 0,
    }
    metadata["lora_bank_state_sha256"] = states
    metadata["lora_bank_wrapped_modules"] = modules
    metadata["lora_bank_parameter_counts"] = counts
    metadata["lora_parameter_count"] = total
    metadata["lora_trainable_parameter_count"] = 0
    if base_bank_order(metadata) != final:
        raise RuntimeError("Strict extended metadata lost exact bank order")
    return metadata


def extend_runtime_lora_config(
    *,
    parent_runtime_config: Mapping[str, Any],
    added_bridges: Sequence[BridgeSourceContract],
    expected_final_banks: Sequence[str],
) -> dict[str, Any]:
    """Create a sanitized runtime-config bank surface from authenticated states."""

    config = copy.deepcopy(dict(parent_runtime_config))
    language = config.get("language")
    banks = language.get("lora_banks") if isinstance(language, dict) else None
    if not isinstance(banks, dict):
        raise TypeError("Strict parent runtime config lacks named LoRA banks")
    base = tuple(banks)
    final = tuple(expected_final_banks)
    if final != base + tuple(source.bank_name for source in added_bridges):
        raise ValueError("Strict runtime-config final bank order changed")
    for source in added_bridges:
        banks[source.bank_name] = {
            "trainable": False,
            "rank": source.rank,
            "alpha": source.alpha,
            "dropout": source.dropout,
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": source.state_sha256,
            "target_modules": [source.target_module],
        }
    if tuple(banks) != final or any(row.get("trainable") is not False for row in banks.values()):
        raise RuntimeError("Strict runtime-config bank inventory is not exactly frozen")
    return config


def validate_runtime_bank_inventory(
    *,
    runtime_config: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    expected_bank_order: Sequence[str],
    expected_states: Mapping[str, str],
) -> None:
    """Authenticate the same exact frozen inventory in config and checkpoint."""

    expected = tuple(expected_bank_order)
    language = runtime_config.get("language")
    configured = language.get("lora_banks") if isinstance(language, Mapping) else None
    if not isinstance(configured, Mapping) or tuple(configured) != expected:
        raise ValueError("Strict runtime config bank inventory changed")
    if base_bank_order(checkpoint_metadata) != expected:
        raise ValueError("Strict runtime checkpoint bank inventory changed")
    lora = checkpoint_metadata.get("lora")
    metadata_banks = lora.get("banks") if isinstance(lora, Mapping) else None
    if not isinstance(metadata_banks, list) or not all(
        isinstance(row, Mapping) for row in metadata_banks
    ):
        raise TypeError("Strict runtime checkpoint bank contracts are malformed")
    metadata_by_name = {str(row.get("name")): row for row in metadata_banks}
    states = checkpoint_metadata.get("lora_bank_state_sha256")
    if not isinstance(states, Mapping) or set(states) != set(expected):
        raise ValueError("Strict runtime checkpoint bank-state inventory changed")
    for name in expected:
        state = expected_states.get(name)
        configured_row = configured[name]
        metadata_row = metadata_by_name[name]
        if (
            not isinstance(configured_row, Mapping)
            or configured_row.get("trainable") is not False
            or metadata_row.get("trainable") is not False
            or configured_row.get("expected_initial_state_sha256")
            != metadata_row.get("expected_initial_state_sha256")
            or states.get(name) != state
            or not isinstance(state, str)
            or _SHA256.fullmatch(state) is None
        ):
            raise ValueError(f"Strict runtime bank state changed: {name}")


__all__ = [
    "BridgeSourceContract",
    "LoadedBridgeSource",
    "base_bank_order",
    "compose_exact_bank_archive",
    "extend_runtime_lora_config",
    "extend_runtime_metadata",
    "load_bridge_source",
    "sha256_file",
    "validate_runtime_bank_inventory",
]
