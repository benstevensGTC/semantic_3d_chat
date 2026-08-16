"""Create-once source seal and process lock for V95 known development.

The seal is intentionally independent of the sealed training configuration.
It binds every implementation module that can predict, authenticate, read
labels, measure NLL, or publish gate evidence.  Evaluation stages authenticate
this seal while holding one cross-process exclusive lock before doing any
stage work.  Creating or authenticating the seal opens no questions, labels,
model weights, predictions, or scores.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import functools
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, ParamSpec, TypeVar

from semantic_3d_chat.config import PROJECT_ROOT

ARTIFACT: Final[str] = "gemma4_v95_known_development_implementation_seal_v1"
SCHEMA_VERSION: Final[int] = 95
IMPLEMENTATION_SEAL: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v95_known_development_implementation_seal.json"
)
EVALUATION_LOCK: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/locks/gemma4_v95_known_development_evaluation.lock"
)
IMPLEMENTATION_SOURCES: Final[dict[str, Path]] = {
    "common": PROJECT_ROOT / "src/semantic_3d_chat/evaluation/v95_known_development_common.py",
    "predict": PROJECT_ROOT / "src/semantic_3d_chat/evaluation/predict_v95_known_development.py",
    "authenticate": PROJECT_ROOT
    / "src/semantic_3d_chat/evaluation/authenticate_v95_known_development.py",
    "score": PROJECT_ROOT / "src/semantic_3d_chat/evaluation/score_v95_known_development.py",
    "nll": PROJECT_ROOT / "src/semantic_3d_chat/evaluation/nll_v95_known_development.py",
    "seal": PROJECT_ROOT / "src/semantic_3d_chat/evaluation/seal_v95_known_development.py",
    "implementation_guard": Path(__file__).resolve(),
}
KNOWN_DEVELOPMENT_OUTPUTS: Final[tuple[Path, ...]] = (
    PROJECT_ROOT / "reports/gemma4/predictions/"
    "gemma4_v95_strict_causal_successor_known_development_question_only.jsonl",
    PROJECT_ROOT / "reports/gemma4/predictions/"
    "gemma4_v95_strict_causal_successor_known_development_question_only.jsonl.provenance.json",
    PROJECT_ROOT / "reports/gemma4/predictions/"
    "gemma4_v95_strict_causal_successor_known_development_question_only.jsonl.access.json",
    PROJECT_ROOT / "reports/gemma4/predictions/"
    "gemma4_v95_strict_causal_successor_known_development_question_only.jsonl.completion.json",
    PROJECT_ROOT / "reports/gemma4/metrics/"
    "gemma4_v95_strict_causal_successor_known_development_structured.json",
    PROJECT_ROOT / "reports/gemma4/metrics/"
    "gemma4_v95_strict_causal_successor_known_development_nll.json",
    PROJECT_ROOT / "reports/gemma4/metrics/"
    "gemma4_v95_strict_causal_successor_known_development_nll_access.json",
    PROJECT_ROOT / "reports/gemma4/metrics/"
    "gemma4_v95_strict_causal_successor_known_development_nll_completion.json",
    PROJECT_ROOT / "reports/gemma4/metrics/"
    "gemma4_v95_strict_causal_successor_known_development.json",
    PROJECT_ROOT / "reports/gemma4/metrics/"
    "gemma4_v95_strict_causal_successor_known_development_evidence.json",
)

_PROCESS_MUTEX = threading.Lock()
_THREAD_STATE = threading.local()
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _sha256_file(path: Path) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise FileNotFoundError(f"V95 implementation source is absent or linked: {candidate}")
    path = candidate.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V95 implementation source is absent or linked: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_source_inventory_v95(
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES,
) -> dict[str, dict[str, str]]:
    expected_names = {
        "common",
        "predict",
        "authenticate",
        "score",
        "nll",
        "seal",
        "implementation_guard",
    }
    if set(sources) != expected_names:
        raise ValueError("V95 evaluation implementation source inventory changed")
    inventory: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted(sources.items()):
        candidate = Path(raw_path).expanduser()
        if candidate.is_symlink():
            raise FileNotFoundError(f"V95 implementation source is absent or linked: {candidate}")
        path = candidate.resolve()
        try:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(f"V95 implementation source escapes project: {path}") from error
        inventory[name] = {
            "path": relative,
            "sha256": _sha256_file(path),
        }
    return inventory


def build_evaluation_implementation_seal_v95(
    *,
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES,
    outputs: Sequence[Path] = KNOWN_DEVELOPMENT_OUTPUTS,
) -> dict[str, Any]:
    inventory = implementation_source_inventory_v95(sources)
    present = [
        Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
        for path in outputs
        if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise RuntimeError(
            f"V95 implementation must be sealed before evaluation output creation: {present}"
        )
    return {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_before_known_development_evaluation",
        "source_count": len(inventory),
        "sources": inventory,
        "source_inventory_sha256": _canonical_sha256(inventory),
        "known_development_outputs_present_before_seal": [],
        "questions_opened": False,
        "labels_opened": False,
        "model_loaded": False,
        "runtime_promotion_authorized": False,
    }


def _strict_json(path: Path) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise FileNotFoundError(candidate)
    path = candidate.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V95 implementation-seal key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError("V95 implementation seal must contain one JSON object")
    return value


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> Path:
    candidate = Path(path).expanduser()
    parent = candidate.parent.resolve()
    destination = parent / candidate.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V95 create-once implementation seal exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def seal_evaluation_implementation_v95(
    *,
    seal_path: Path = IMPLEMENTATION_SEAL,
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES,
    outputs: Sequence[Path] = KNOWN_DEVELOPMENT_OUTPUTS,
) -> dict[str, Any]:
    """Create the implementation seal once without opening evaluation inputs."""

    payload = build_evaluation_implementation_seal_v95(
        sources=sources,
        outputs=outputs,
    )
    output = _atomic_create_json(seal_path, payload)
    return {
        **payload,
        "seal_path": output.relative_to(PROJECT_ROOT).as_posix(),
        "seal_sha256": _sha256_file(output),
    }


def authenticate_evaluation_implementation_v95(
    *,
    seal_path: Path = IMPLEMENTATION_SEAL,
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES,
) -> dict[str, Any]:
    """Fail closed unless every evaluator source matches the create-once seal."""

    candidate = Path(seal_path).expanduser()
    if candidate.is_symlink():
        raise FileNotFoundError(candidate)
    path = candidate.resolve()
    payload = _strict_json(path)
    inventory = implementation_source_inventory_v95(sources)
    if (
        payload.get("artifact") != ARTIFACT
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "sealed_before_known_development_evaluation"
        or payload.get("source_count") != len(inventory)
        or payload.get("sources") != inventory
        or payload.get("source_inventory_sha256") != _canonical_sha256(inventory)
        or payload.get("known_development_outputs_present_before_seal") != []
        or payload.get("questions_opened") is not False
        or payload.get("labels_opened") is not False
        or payload.get("model_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
        or set(payload)
        != {
            "artifact",
            "schema_version",
            "status",
            "source_count",
            "sources",
            "source_inventory_sha256",
            "known_development_outputs_present_before_seal",
            "questions_opened",
            "labels_opened",
            "model_loaded",
            "runtime_promotion_authorized",
        }
    ):
        raise ValueError("V95 known-development implementation seal changed")
    return {
        "artifact": ARTIFACT,
        "authenticated": True,
        "seal_sha256": _sha256_file(path),
        "source_inventory_sha256": payload["source_inventory_sha256"],
        "source_count": len(inventory),
        "questions_opened": False,
        "labels_opened": False,
        "model_loaded": False,
    }


@contextmanager
def exclusive_evaluation_lock_v95(
    lock_path: Path = EVALUATION_LOCK,
) -> Iterator[None]:
    """Acquire a nonblocking, reentrant-in-thread evaluation process lock."""

    depth = int(getattr(_THREAD_STATE, "depth", 0))
    if depth:
        _THREAD_STATE.depth = depth + 1
        try:
            yield
        finally:
            _THREAD_STATE.depth -= 1
        return
    if not _PROCESS_MUTEX.acquire(blocking=False):
        raise RuntimeError("V95 known-development evaluation is already active")
    descriptor = -1
    try:
        candidate = Path(lock_path).expanduser()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        path = candidate.parent.resolve() / candidate.name
        if path.is_symlink():
            raise RuntimeError("V95 evaluation lock path must not be a symlink")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError("V95 known-development evaluation is already active") from error
            raise
        _THREAD_STATE.depth = 1
        try:
            yield
        finally:
            _THREAD_STATE.depth = 0
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _PROCESS_MUTEX.release()


def hardened_evaluation_stage_v95(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Require the source seal and exclusive lock around a public stage."""

    @functools.wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with exclusive_evaluation_lock_v95():
            authenticate_evaluation_implementation_v95()
            return function(*args, **kwargs)

    return guarded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("seal", "authenticate"), nargs="?", default="authenticate"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "seal":
        with exclusive_evaluation_lock_v95():
            result = seal_evaluation_implementation_v95()
    else:
        with exclusive_evaluation_lock_v95():
            result = authenticate_evaluation_implementation_v95()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "EVALUATION_LOCK",
    "IMPLEMENTATION_SEAL",
    "IMPLEMENTATION_SOURCES",
    "KNOWN_DEVELOPMENT_OUTPUTS",
    "authenticate_evaluation_implementation_v95",
    "build_evaluation_implementation_seal_v95",
    "exclusive_evaluation_lock_v95",
    "hardened_evaluation_stage_v95",
    "implementation_source_inventory_v95",
    "main",
    "seal_evaluation_implementation_v95",
]
