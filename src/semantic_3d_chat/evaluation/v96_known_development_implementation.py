"""Create-once source seal and process lock for V96 evaluation.

The seal binds every evaluator module plus the sealed V96 config, preflight,
and trainer source.  It cannot be created while the V96 contract is draft or
while any evaluation output already exists.  Public evaluator stages require
the seal and hold one nonblocking cross-process lock for their full duration.
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
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    load_config_v96,
)

ARTIFACT: Final[str] = "gemma4_v96_known_development_implementation_seal_v1"
SCHEMA_VERSION: Final[int] = 96
IMPLEMENTATION_SEAL: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_known_development_implementation_seal.json"
)
EVALUATION_LOCK: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/locks/gemma4_v96_known_development_evaluation.lock"
)
IMPLEMENTATION_SOURCES: Final[dict[str, Path]] = {
    "common": PROJECT_ROOT
    / "src/semantic_3d_chat/evaluation/v96_known_development_common.py",
    "predict": PROJECT_ROOT
    / "src/semantic_3d_chat/evaluation/predict_v96_known_development.py",
    "authenticate": PROJECT_ROOT
    / "src/semantic_3d_chat/evaluation/authenticate_v96_known_development.py",
    "score": PROJECT_ROOT
    / "src/semantic_3d_chat/evaluation/score_v96_known_development.py",
    "nll": PROJECT_ROOT
    / "src/semantic_3d_chat/evaluation/nll_v96_known_development.py",
    "seal": PROJECT_ROOT
    / "src/semantic_3d_chat/evaluation/seal_v96_known_development.py",
    "implementation_guard": Path(__file__).resolve(),
}
KNOWN_DEVELOPMENT_OUTPUTS: Final[tuple[Path, ...]] = (
    PROJECT_ROOT
    / "reports/gemma4/predictions/"
    "gemma4_v96_atomic_pair_repair_known_development_question_only.jsonl",
    PROJECT_ROOT
    / "reports/gemma4/predictions/"
    "gemma4_v96_atomic_pair_repair_known_development_question_only.jsonl.provenance.json",
    PROJECT_ROOT
    / "reports/gemma4/predictions/"
    "gemma4_v96_atomic_pair_repair_known_development_question_only.jsonl.access.json",
    PROJECT_ROOT
    / "reports/gemma4/predictions/"
    "gemma4_v96_atomic_pair_repair_known_development_question_only.jsonl.completion.json",
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development_structured.json",
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development_nll.json",
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development_nll_access.json",
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development_nll_completion.json",
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development.json",
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development_evidence.json",
)

_PROCESS_MUTEX = threading.Lock()
_THREAD_STATE = threading.local()
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _sha256_file(path: Path) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise FileNotFoundError(f"V96 implementation source is absent or linked: {candidate}")
    source = candidate.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"V96 implementation source is absent or linked: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
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


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"V96 sealed source escapes project: {path}") from error


def implementation_source_inventory_v96(
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES,
) -> dict[str, dict[str, str]]:
    expected = {
        "common",
        "predict",
        "authenticate",
        "score",
        "nll",
        "seal",
        "implementation_guard",
    }
    if set(sources) != expected:
        raise ValueError("V96 evaluation implementation source inventory changed")
    inventory: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted(sources.items()):
        candidate = Path(raw_path).expanduser()
        if candidate.is_symlink():
            raise FileNotFoundError(
                f"V96 implementation source is absent or linked: {candidate}"
            )
        path = candidate.resolve()
        inventory[name] = {"path": _relative(path), "sha256": _sha256_file(path)}
    return inventory


def contract_source_inventory_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, dict[str, str]]:
    """Bind the now-sealed config/preflight/trainer; draft contracts fail closed."""

    config = load_config_v96(config_path, allow_draft=False)
    sources = config["sources"]
    paths = {
        "config": Path(config_path),
        "preflight": Path(sources["preflight_source"]),
        "trainer": Path(sources["trainer_source"]),
    }
    rooted = {
        name: path if path.is_absolute() else PROJECT_ROOT / path
        for name, path in paths.items()
    }
    if any(path.is_symlink() for path in rooted.values()):
        raise FileNotFoundError("V96 contract source must not be a symbolic link")
    resolved = {name: path.resolve() for name, path in rooted.items()}
    inventory = {
        name: {"path": _relative(path), "sha256": _sha256_file(path)}
        for name, path in sorted(resolved.items())
    }
    if (
        inventory["preflight"]["sha256"] != sources["preflight_source_sha256"]
        or inventory["trainer"]["sha256"] != sources["trainer_source_sha256"]
    ):
        raise ValueError("V96 config does not bind the current preflight/trainer sources")
    return inventory


def build_evaluation_implementation_seal_v96(
    *,
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES,
    outputs: Sequence[Path] = KNOWN_DEVELOPMENT_OUTPUTS,
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    implementation = implementation_source_inventory_v96(sources)
    contract = contract_source_inventory_v96(config_path)
    present = [
        _relative(Path(path))
        for path in outputs
        if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise RuntimeError(
            f"V96 implementation must be sealed before evaluation output creation: {present}"
        )
    return {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_before_known_development_evaluation",
        "source_count": len(implementation),
        "sources": implementation,
        "source_inventory_sha256": _canonical_sha256(implementation),
        "contract_source_count": len(contract),
        "contract_sources": contract,
        "contract_source_inventory_sha256": _canonical_sha256(contract),
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
    source = candidate.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V96 implementation-seal key: {key}")
            result[key] = value
        return result

    value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError("V96 implementation seal must contain one JSON object")
    return value


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> Path:
    candidate = Path(path).expanduser()
    destination = candidate.parent.resolve() / candidate.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V96 create-once implementation seal exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
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


def seal_evaluation_implementation_v96(
    *,
    seal_path: Path = IMPLEMENTATION_SEAL,
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES,
    outputs: Sequence[Path] = KNOWN_DEVELOPMENT_OUTPUTS,
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Create the seal once without opening questions, labels, or a model."""

    payload = build_evaluation_implementation_seal_v96(
        sources=sources, outputs=outputs, config_path=config_path
    )
    output = _atomic_create_json(seal_path, payload)
    return {
        **payload,
        "seal_path": _relative(output),
        "seal_sha256": _sha256_file(output),
    }


def authenticate_evaluation_implementation_v96(
    *,
    seal_path: Path = IMPLEMENTATION_SEAL,
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES,
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Fail unless evaluator and sealed training-contract sources match."""

    candidate = Path(seal_path).expanduser()
    if candidate.is_symlink():
        raise FileNotFoundError(candidate)
    source = candidate.resolve()
    payload = _strict_json(source)
    implementation = implementation_source_inventory_v96(sources)
    contract = contract_source_inventory_v96(config_path)
    expected_fields = {
        "artifact",
        "schema_version",
        "status",
        "source_count",
        "sources",
        "source_inventory_sha256",
        "contract_source_count",
        "contract_sources",
        "contract_source_inventory_sha256",
        "known_development_outputs_present_before_seal",
        "questions_opened",
        "labels_opened",
        "model_loaded",
        "runtime_promotion_authorized",
    }
    if (
        set(payload) != expected_fields
        or payload.get("artifact") != ARTIFACT
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "sealed_before_known_development_evaluation"
        or payload.get("source_count") != len(implementation)
        or payload.get("sources") != implementation
        or payload.get("source_inventory_sha256") != _canonical_sha256(implementation)
        or payload.get("contract_source_count") != len(contract)
        or payload.get("contract_sources") != contract
        or payload.get("contract_source_inventory_sha256") != _canonical_sha256(contract)
        or payload.get("known_development_outputs_present_before_seal") != []
        or payload.get("questions_opened") is not False
        or payload.get("labels_opened") is not False
        or payload.get("model_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 known-development implementation seal changed")
    return {
        "artifact": ARTIFACT,
        "authenticated": True,
        "seal_sha256": _sha256_file(source),
        "source_inventory_sha256": payload["source_inventory_sha256"],
        "contract_source_inventory_sha256": payload[
            "contract_source_inventory_sha256"
        ],
        "source_count": len(implementation),
        "questions_opened": False,
        "labels_opened": False,
        "model_loaded": False,
    }


@contextmanager
def exclusive_evaluation_lock_v96(
    lock_path: Path = EVALUATION_LOCK,
) -> Iterator[None]:
    """Acquire a nonblocking process lock, reentrant within one thread."""

    depth = int(getattr(_THREAD_STATE, "depth", 0))
    if depth:
        _THREAD_STATE.depth = depth + 1
        try:
            yield
        finally:
            _THREAD_STATE.depth -= 1
        return
    if not _PROCESS_MUTEX.acquire(blocking=False):
        raise RuntimeError("V96 known-development evaluation is already active")
    descriptor = -1
    try:
        candidate = Path(lock_path).expanduser()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        path = candidate.parent.resolve() / candidate.name
        if path.is_symlink():
            raise RuntimeError("V96 evaluation lock path must not be a symlink")
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
                raise RuntimeError(
                    "V96 known-development evaluation is already active"
                ) from error
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


def hardened_evaluation_stage_v96(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    @functools.wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with exclusive_evaluation_lock_v96():
            authenticate_evaluation_implementation_v96()
            return function(*args, **kwargs)

    return guarded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("seal", "authenticate"), nargs="?", default="authenticate"
    )
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "seal":
        with exclusive_evaluation_lock_v96():
            result = seal_evaluation_implementation_v96(config_path=args.config)
    else:
        with exclusive_evaluation_lock_v96():
            result = authenticate_evaluation_implementation_v96(config_path=args.config)
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
    "authenticate_evaluation_implementation_v96",
    "build_evaluation_implementation_seal_v96",
    "contract_source_inventory_v96",
    "exclusive_evaluation_lock_v96",
    "hardened_evaluation_stage_v96",
    "implementation_source_inventory_v96",
    "main",
    "seal_evaluation_implementation_v96",
]
