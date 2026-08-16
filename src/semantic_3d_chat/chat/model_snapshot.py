"""Content-address the exact local model snapshot used by chat and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

_HUB_COMMIT = re.compile(r"[0-9a-f]{40}")
_REQUIRED_GEMMA_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=64)
def _sha256_regular_file(
    path: str,
    size_bytes: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    """Hash once per immutable-looking file identity within a process."""

    source = Path(path)
    stat = source.stat()
    if (
        stat.st_size != size_bytes
        or stat.st_mtime_ns != mtime_ns
        or stat.st_ctime_ns != ctime_ns
    ):
        raise RuntimeError(f"Local model file changed before hashing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    final_stat = source.stat()
    if (
        final_stat.st_size != size_bytes
        or final_stat.st_mtime_ns != mtime_ns
        or final_stat.st_ctime_ns != ctime_ns
    ):
        raise RuntimeError(f"Local model file changed while hashing: {source}")
    return digest.hexdigest()


def local_model_snapshot_identity(
    config: Mapping[str, Any],
    *,
    record_file: Callable[[str | Path], None] | None = None,
) -> dict[str, Any]:
    """Resolve offline and hash every byte in the pinned shared Gemma snapshot."""

    vision = config.get("vision")
    language = config.get("language")
    if not isinstance(vision, Mapping) or not isinstance(language, Mapping):
        raise TypeError("Runtime model snapshot requires vision/language mappings")
    model_id = vision.get("model_id")
    revision = vision.get("revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise TypeError("Runtime model_id/revision must be strings")
    if _HUB_COMMIT.fullmatch(revision) is None:
        raise ValueError("Runtime model revision must be an exact 40-hex Hub commit")
    if language.get("model_id") != model_id or language.get("revision") != revision:
        raise ValueError("Vision and language must use the same pinned local model snapshot")

    # Resolve the standard Hub cache layout directly.  Unlike
    # ``snapshot_download(local_files_only=True)``, this is guaranteed not to
    # create lock files or cache metadata during a read-only preflight.
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import repo_folder_name

    snapshot = (
        Path(HF_HUB_CACHE)
        / repo_folder_name(repo_id=model_id, repo_type="model")
        / "snapshots"
        / revision
    ).resolve()
    if not snapshot.is_dir() or snapshot.name != revision:
        raise RuntimeError(
            f"Pinned local model snapshot did not resolve to revision {revision}: {snapshot}"
        )
    logical_files = sorted(path for path in snapshot.rglob("*") if not path.is_dir())
    if not logical_files:
        raise FileNotFoundError(f"Pinned local model snapshot is empty: {snapshot}")
    relative_names = {path.relative_to(snapshot).as_posix() for path in logical_files}
    if missing := sorted(_REQUIRED_GEMMA_FILES - relative_names):
        raise FileNotFoundError(
            f"Pinned local Gemma snapshot is incomplete; missing={missing}"
        )
    model_cache_root = snapshot.parents[1]
    entries: list[dict[str, Any]] = []
    for logical in logical_files:
        resolved = logical.resolve(strict=True)
        if not resolved.is_relative_to(model_cache_root) or not resolved.is_file():
            raise FileNotFoundError(f"Local model snapshot entry is not a file: {logical}")
        if record_file is not None:
            record_file(logical)
        stat = resolved.stat()
        entries.append(
            {
                "path": logical.relative_to(snapshot).as_posix(),
                "sha256": _sha256_regular_file(
                    str(resolved),
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                ),
                "size_bytes": stat.st_size,
            }
        )
    return {
        "model_id": model_id,
        "revision": revision,
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "tree_sha256": _canonical_sha256(entries),
        "files": entries,
    }


__all__ = ["local_model_snapshot_identity"]
