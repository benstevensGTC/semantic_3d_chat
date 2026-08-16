"""V96-authorized projection of the sealed deferred QA selection.

V95 sealed the deterministic, answer-independent selector before the deferred
scenes existed, but its executable wrapper necessarily authenticates V95's own
unlock.  V95 did not pass its development gate, so that unlock can never exist.
This module is the deliberately narrow V96 successor: it authenticates the V96
unlock *before* opening the raw answer-bearing pool, then calls the unchanged
pure V95 selector and publishes the same two fixed outputs.

It never loads Gemma, Blender, a semantic map, or an oracle.  Importing this
module performs no file reads and creates no artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_qa import (
    FINAL_QA,
    RAW_QA,
    SELECTION_MANIFEST,
    select_exact_final_records_v95,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import CONFIG
from semantic_3d_chat.evaluation.v96_deferred_final import (
    _authenticate_deferred_final_unlock_under_guard_v96,
)

SCHEMA_VERSION: Final[int] = 96
ARTIFACT: Final[str] = "gemma4_v96_deferred_final_exact_qa_selection_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    result: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"V96 deferred QA line {line_number} is not an object")
        result.append(value)
    if not result:
        raise ValueError("V96 deferred raw QA pool is empty")
    return result


def _serialized_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "\n".join(json.dumps(dict(record), sort_keys=True, allow_nan=False) for record in records)
        + "\n"
    ).encode("utf-8")


def _create_or_authenticate(path: Path, encoded: bytes) -> bool:
    """Create one fixed output, permitting only the preregistered placeholder."""

    if path.is_symlink():
        raise ValueError(f"V96 deferred QA output may not be a symlink: {path}")
    if path.exists() and path.stat().st_size:
        if path.read_bytes() != encoded:
            raise FileExistsError(f"V96 deferred QA output differs: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def select_final_qa_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Authenticate V96 first, then run only V95's sealed pure selector."""

    # The outer V96 materializer owns the process lock while this child runs.
    # This internal authenticator performs the full immutable-evidence check
    # without attempting to acquire the already-held cross-process lock again.
    unlock = _authenticate_deferred_final_unlock_under_guard_v96(config_path)

    source = RAW_QA.resolve()
    destination = FINAL_QA.resolve()
    manifest_destination = SELECTION_MANIFEST.resolve()
    records = _read_jsonl(source)
    selected, selection = select_exact_final_records_v95(records)
    encoded = _serialized_jsonl(selected)
    qa_created = _create_or_authenticate(destination, encoded)
    payload = {
        **selection,
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "created_after_authenticated_v96_unlock",
        "unlock_file_sha256": unlock["unlock_file_sha256"],
        "unlock_identity_sha256": unlock["unlock_identity_sha256"],
        "candidate_fingerprint_sha256": unlock["candidate_fingerprint_sha256"],
        "raw_qa_sha256": sha256_file_v85(source),
        "final_qa_sha256": hashlib.sha256(encoded).hexdigest(),
        "raw_qa_path": source.relative_to(PROJECT_ROOT).as_posix(),
        "final_qa_path": destination.relative_to(PROJECT_ROOT).as_posix(),
        "authorization_checked_before_label_read": True,
        "pure_v95_selector_reused_unchanged": True,
        "v95_unlock_required": False,
        "prediction_process_imported": False,
        "model_loaded": False,
        "automatic_runtime_promotion": False,
    }
    manifest_encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_created = _create_or_authenticate(manifest_destination, manifest_encoded)
    return {
        **payload,
        "qa_created": qa_created,
        "selection_manifest_created": manifest_created,
        "selection_manifest_sha256": hashlib.sha256(manifest_encoded).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select",))
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = select_final_qa_v96(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ARTIFACT", "main", "select_final_qa_v96"]
