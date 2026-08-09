"""Small deterministic I/O helpers shared only by evaluation baselines."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected an object at {source}:{line_number}")
        records.append(value)
    return records


def atomic_write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        payload = "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records
        )
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def indexed_records(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    source = Path(path)
    records = read_jsonl(source) if source.is_file() else []
    return {
        (str(item["scene_id"]), str(item["question_id"])): item
        for item in records
    }


def text_fingerprint(*parts: str) -> str:
    payload = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
