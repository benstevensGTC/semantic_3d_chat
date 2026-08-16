"""Symlink-safe, finite-JSON, create-once I/O for V96 evaluator revision v2."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT


def _reject_nonfinite_v96_v2(value: Any, *, source: Path) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite V96 v2 JSON number in {source}")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite_v96_v2(item, source=source)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_nonfinite_v96_v2(item, source=source)


def physical_path_v96_v2(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    candidate = raw if raw.is_absolute() else PROJECT_ROOT / raw
    absolute = Path(os.path.abspath(candidate))
    if any(component.is_symlink() for component in (absolute, *absolute.parents)):
        raise FileNotFoundError(f"V96 v2 physical path contains a symlink: {absolute}")
    return absolute


def read_json_strict_v96_v2(path: str | Path) -> dict[str, Any]:
    source = physical_path_v96_v2(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V96 v2 JSON key in {source}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite V96 v2 JSON constant in {source}: {value}")

    loaded = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(loaded, dict):
        raise TypeError(f"V96 v2 JSON must contain one object: {source}")
    _reject_nonfinite_v96_v2(loaded, source=source)
    return loaded


def read_jsonl_strict_v96_v2(path: str | Path) -> list[dict[str, Any]]:
    source = physical_path_v96_v2(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V96 v2 JSONL key in {source}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite V96 v2 JSONL constant in {source}: {value}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        loaded = json.loads(
            line,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
        if not isinstance(loaded, dict):
            raise TypeError(f"V96 v2 JSONL line {line_number} is not an object: {source}")
        _reject_nonfinite_v96_v2(loaded, source=source)
        rows.append(loaded)
    return rows


def _atomic_create_v96_v2(path: str | Path, encoded: bytes) -> None:
    raw = Path(path).expanduser()
    candidate = raw if raw.is_absolute() else PROJECT_ROOT / raw
    destination = Path(os.path.abspath(candidate))
    # Check before and after mkdir so neither a preexisting nor raced symlinked
    # ancestor can redirect publication outside the intended tree.
    physical_path_v96_v2(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    physical_path_v96_v2(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    parent_descriptor = -1
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        physical_path_v96_v2(destination)
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        parent_descriptor = os.open(destination.parent, flags)
        os.link(
            temporary,
            destination.name,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    finally:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        temporary.unlink(missing_ok=True)


def write_json_create_once_v96_v2(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_create_v96_v2(path, encoded)


def write_jsonl_create_once_v96_v2(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> None:
    encoded = "".join(
        json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")
    _atomic_create_v96_v2(path, encoded)


__all__ = [
    "physical_path_v96_v2",
    "read_json_strict_v96_v2",
    "read_jsonl_strict_v96_v2",
    "write_json_create_once_v96_v2",
    "write_jsonl_create_once_v96_v2",
]
