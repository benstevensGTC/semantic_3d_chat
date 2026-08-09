from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    def load_recursive(current_path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        resolved = current_path.resolve()
        if resolved in stack:
            chain = " -> ".join(str(item) for item in (*stack, resolved))
            raise ValueError(f"Cyclic _base_ config inheritance: {chain}")
        with resolved.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            raise TypeError(f"Configuration must be a mapping: {resolved}")
        if base_name := loaded.get("_base_"):
            base = load_recursive(resolved.parent / str(base_name), (*stack, resolved))
            return _deep_merge(base, loaded)
        return loaded

    config = load_recursive(config_path, ())
    config["_config_path"] = str(config_path.resolve())
    return config


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    stable = {key: value for key, value in config.items() if not key.startswith("_")}
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def artifact_root(config: dict[str, Any], kind: str) -> Path:
    """Resolve an optional per-artifact root with ``data_root/kind`` fallback."""

    if not kind or Path(kind).name != kind:
        raise ValueError("Artifact kind must be one plain path component")
    paths = config["paths"]
    configured = paths.get(f"{kind}_root")
    root = Path(str(configured)) if configured is not None else Path(str(paths["data_root"])) / kind
    return root if root.is_absolute() else PROJECT_ROOT / root


def project_path(config: dict[str, Any], *parts: str) -> Path:
    if not parts:
        root = Path(str(config["paths"]["data_root"]))
        return root if root.is_absolute() else PROJECT_ROOT / root
    return artifact_root(config, parts[0]).joinpath(*parts[1:])


def reports_root(config: dict[str, Any]) -> Path:
    """Resolve the configured report tree independently of derived data roots."""

    root = Path(str(config["paths"]["reports_root"]))
    return root if root.is_absolute() else PROJECT_ROOT / root


def default_checkpoint_path(config: dict[str, Any]) -> Path:
    """Resolve the trained adapter selected by this configuration."""

    root = artifact_root(config, "checkpoints")
    namespace = config.get("training", {}).get("output_namespace")
    if namespace is not None:
        namespace = str(namespace)
        if not namespace or Path(namespace).name != namespace:
            raise ValueError("Training output_namespace must be one plain path component")
        root = root / namespace
    return root / "best"
