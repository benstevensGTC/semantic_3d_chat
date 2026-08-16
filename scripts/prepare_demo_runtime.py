#!/usr/bin/env python3
"""Materialize and authenticate the minimal local demo runtime checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.training.checkpointing import (
    validate_runtime_checkpoint_metadata,
)

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "source_checkpoint",
        "runtime_checkpoint",
        "inference_inventory",
        "files",
        "training_metadata_included",
        "environmental_text_inputs",
    }
)
_INVENTORY = frozenset({"adapter.safetensors", "runtime_metadata.json"})


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_regular(path: Path, purpose: str) -> Path:
    unresolved = Path(os.path.abspath(path))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} path contains a symbolic link: {current}")
    if not unresolved.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {unresolved}")
    return unresolved


def load_release_manifest(path: str | Path) -> dict[str, Any]:
    source = _safe_regular(_rooted(path), "Demo release manifest")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("Demo release manifest fields changed")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact") != "semantic_3d_chat_local_demo_runtime_release_v1"
        or payload.get("training_metadata_included") is not False
        or payload.get("environmental_text_inputs") != []
        or frozenset(payload.get("inference_inventory", ())) != _INVENTORY
    ):
        raise ValueError("Demo release manifest contract changed")
    files = payload.get("files")
    if not isinstance(files, Mapping) or set(files) != _INVENTORY:
        raise ValueError("Demo release file inventory changed")
    for name, entry in files.items():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"sha256", "size_bytes"}
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
            or isinstance(entry["size_bytes"], bool)
            or not isinstance(entry["size_bytes"], int)
            or entry["size_bytes"] < 1
        ):
            raise ValueError(f"Demo release manifest entry changed: {name}")
    return dict(payload)


def _validate_file(path: Path, expected: Mapping[str, Any], purpose: str) -> None:
    try:
        source = _safe_regular(path, purpose)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{purpose} is unavailable: {path} "
            f"(expected {expected['size_bytes']} bytes, SHA-256 {expected['sha256']})"
        ) from error
    if source.stat().st_size != expected["size_bytes"]:
        raise ValueError(f"{purpose} size changed")
    if _sha256(source) != expected["sha256"]:
        raise ValueError(f"{purpose} digest changed")


def validate_runtime_release(destination: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not destination.is_dir() or destination.is_symlink():
        raise FileNotFoundError(f"Demo runtime checkpoint is unavailable: {destination}")
    inventory = {item.name for item in destination.iterdir()}
    if inventory != _INVENTORY:
        raise ValueError(
            "Demo runtime checkpoint must contain exactly the two sanitized "
            f"inference files; observed={sorted(inventory)}"
        )
    for name in sorted(_INVENTORY):
        _validate_file(destination / name, manifest["files"][name], name)
    metadata = json.loads((destination / "runtime_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("Demo runtime metadata must be a JSON object")
    validate_runtime_checkpoint_metadata(metadata)
    return {
        "artifact": manifest["artifact"],
        "checkpoint": str(destination),
        "inventory": sorted(inventory),
        "training_metadata_included": False,
        "environmental_text_inputs": [],
        "validated": True,
    }


def prepare_runtime_release(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = _rooted(str(manifest["source_checkpoint"]))
    destination = _rooted(str(manifest["runtime_checkpoint"]))
    if destination.exists():
        return validate_runtime_release(destination, manifest)
    if not source.is_dir() or source.is_symlink():
        expected = ", ".join(
            f"{name}={manifest['files'][name]['size_bytes']} bytes/"
            f"{manifest['files'][name]['sha256']}"
            for name in sorted(_INVENTORY)
        )
        raise FileNotFoundError(f"Source checkpoint is unavailable: {source} (expected {expected})")
    for name in sorted(_INVENTORY):
        _validate_file(source / name, manifest["files"][name], f"source {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        for name in sorted(_INVENTORY):
            shutil.copyfile(source / name, temporary / name)
        validate_runtime_release(temporary, manifest)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_runtime_release(destination, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/runtime/demo_release_v1.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = load_release_manifest(args.manifest)
    result = (
        validate_runtime_release(_rooted(str(manifest["runtime_checkpoint"])), manifest)
        if args.check
        else prepare_runtime_release(manifest)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
