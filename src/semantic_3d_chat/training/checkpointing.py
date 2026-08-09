from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from semantic_3d_chat.language.lora import tensor_state_sha256


def module_collection_state_sha256(modules: dict[str, nn.Module]) -> str:
    """Hash a named module collection without including optimizer state."""

    state = {
        f"{module_name}.{name}": value
        for module_name, module in modules.items()
        for name, value in module.state_dict().items()
    }
    return tensor_state_sha256(state)


def save_adapter_checkpoint(
    directory: str | Path,
    modules: dict[str, nn.Module],
    metadata: dict[str, Any],
) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for module_name, module in modules.items():
        for name, value in module.state_dict().items():
            tensors[f"{module_name}.{name}"] = value.detach().cpu().contiguous()
    save_file(tensors, destination / "adapter.safetensors")
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix="metadata.", suffix=".tmp", dir=destination
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination / "metadata.json")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_adapter_checkpoint(
    directory: str | Path,
    modules: dict[str, nn.Module],
    device: str = "cpu",
    *,
    allowed_missing_key_prefixes: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Load a checkpoint without silently dropping or inventing module state.

    ``allowed_missing_key_prefixes`` exists only for explicit architecture
    migrations such as adding an exact-zero residual to an older checkpoint.
    Normal resume and runtime loads remain fully strict.
    """

    source = Path(directory)
    tensors = load_file(source / "adapter.safetensors", device=device)
    consumed: set[str] = set()
    allowed_by_module = dict(allowed_missing_key_prefixes or {})
    unknown_modules = sorted(set(allowed_by_module) - set(modules))
    if unknown_modules:
        raise ValueError(f"Missing-key allowances reference unknown modules: {unknown_modules}")
    for module_name, module in modules.items():
        prefix = f"{module_name}."
        state = {
            key[len(prefix) :]: value for key, value in tensors.items() if key.startswith(prefix)
        }
        consumed.update(key for key in tensors if key.startswith(prefix))
        allowed_prefixes = tuple(allowed_by_module.get(module_name, ()))
        if not allowed_prefixes:
            module.load_state_dict(state, strict=True)
            continue
        result = module.load_state_dict(state, strict=False)
        unexpected = sorted(result.unexpected_keys)
        forbidden_missing = sorted(
            key
            for key in result.missing_keys
            if not any(key.startswith(allowed) for allowed in allowed_prefixes)
        )
        if unexpected or forbidden_missing:
            raise RuntimeError(
                f"Checkpoint migration mismatch for {module_name!r}: "
                f"unexpected={unexpected} forbidden_missing={forbidden_missing}"
            )
    unconsumed = sorted(set(tensors) - consumed)
    if unconsumed:
        raise RuntimeError(f"Checkpoint contains unconsumed tensor keys: {unconsumed}")
    return json.loads((source / "metadata.json").read_text(encoding="utf-8"))


def save_optimizer_checkpoint(directory: str | Path, optimizer: torch.optim.Optimizer) -> Path:
    """Atomically save resumable optimizer state beside adapter weights."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix="optimizer.", suffix=".tmp", dir=destination
    )
    os.close(temporary_fd)
    try:
        torch.save(optimizer.state_dict(), temporary_name)
        os.replace(temporary_name, destination / "optimizer.pt")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination / "optimizer.pt"


def load_optimizer_checkpoint(
    directory: str | Path,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> None:
    """Restore a trusted local optimizer checkpoint for an exact epoch resume."""

    source = Path(directory) / "optimizer.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Resume checkpoint has no optimizer state: {source}")
    state = torch.load(source, map_location=device, weights_only=True)
    optimizer.load_state_dict(state)
