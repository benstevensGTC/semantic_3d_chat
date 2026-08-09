from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn


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
) -> dict[str, Any]:
    source = Path(directory)
    tensors = load_file(source / "adapter.safetensors", device=device)
    for module_name, module in modules.items():
        prefix = f"{module_name}."
        state = {key[len(prefix) :]: value for key, value in tensors.items() if key.startswith(prefix)}
        module.load_state_dict(state, strict=True)
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
