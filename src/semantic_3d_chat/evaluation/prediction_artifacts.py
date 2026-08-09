"""Content-addressed, crash-safe storage for local evaluation predictions.

The primary and control inference drivers can take minutes per question on a
Mac.  This module gives those drivers a shared resume contract without knowing
anything about models or questions: every completed opaque question key is
atomically checkpointed, and cached records are accepted only when the effective
configuration, adapter checkpoint, and exact question/reference file match.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.baseline_io import atomic_write_jsonl, read_jsonl, sha256_file

PROVENANCE_SCHEMA_VERSION: Final[int] = 1
_CHECKPOINT_INFERENCE_FILES: Final[tuple[str, ...]] = (
    "adapter.safetensors",
    "metadata.json",
)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def effective_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the merged, effective configuration rather than one YAML layer."""

    stable = {key: value for key, value in config.items() if not str(key).startswith("_")}
    return _canonical_json_sha256(stable)


def checkpoint_fingerprint(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash the inference-relevant contents of an adapter checkpoint.

    Optimizer state is deliberately excluded: it cannot affect inference and is
    much larger than the adapter.  A checkpoint directory must contain the
    adapter weights.  Metadata is included when present because it defines the
    architecture/load contract.
    """

    source = Path(path).expanduser().resolve()
    if source.is_file():
        files = [source]
        root = source.parent
    elif source.is_dir():
        adapter = source / "adapter.safetensors"
        if not adapter.is_file():
            raise FileNotFoundError(f"Adapter checkpoint is missing {adapter}")
        files = [source / name for name in _CHECKPOINT_INFERENCE_FILES]
        files = [item for item in files if item.is_file()]
        root = source
    else:
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")

    entries = [
        {
            "path": item.relative_to(root).as_posix() if item != root else item.name,
            "sha256": sha256_file(item),
            "size_bytes": item.stat().st_size,
        }
        for item in files
    ]
    return _canonical_json_sha256(entries), entries


@dataclass(frozen=True)
class PredictionProvenance:
    """Immutable identity of one prediction run."""

    config_path: str
    config_sha256: str
    config_file_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_files: tuple[dict[str, Any], ...]
    references_path: str
    references_sha256: str
    split: str
    run_kind: str
    condition: str | None = None

    def identity(self) -> dict[str, Any]:
        """Return only fields that determine whether predictions are reusable."""

        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "config_sha256": self.config_sha256,
            "config_file_sha256": self.config_file_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "references_sha256": self.references_sha256,
            "split": self.split,
            "run_kind": self.run_kind,
            "condition": self.condition,
        }

    @property
    def sha256(self) -> str:
        return _canonical_json_sha256(self.identity())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity(),
            "provenance_sha256": self.sha256,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_files": [dict(item) for item in self.checkpoint_files],
            "references_path": self.references_path,
        }


def build_prediction_provenance(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    references_path: str | Path,
    split: str,
    run_kind: str,
    condition: str | None = None,
) -> PredictionProvenance:
    """Resolve and fingerprint every input that can change predictions."""

    resolved_config = Path(config_path).expanduser().resolve()
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    resolved_references = Path(references_path).expanduser().resolve()
    for label, source in (
        ("Configuration", resolved_config),
        ("Reference/questions file", resolved_references),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"{label} does not exist: {source}")
    if not split:
        raise ValueError("split must be non-empty")
    if not run_kind:
        raise ValueError("run_kind must be non-empty")
    checkpoint_sha256, checkpoint_files = checkpoint_fingerprint(resolved_checkpoint)
    return PredictionProvenance(
        config_path=str(resolved_config),
        config_sha256=effective_config_sha256(config),
        config_file_sha256=sha256_file(resolved_config),
        checkpoint_path=str(resolved_checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_files=tuple(checkpoint_files),
        references_path=str(resolved_references),
        references_sha256=sha256_file(resolved_references),
        split=split,
        run_kind=run_kind,
        condition=condition,
    )


def provenance_path_for(prediction_path: str | Path) -> Path:
    destination = Path(prediction_path).expanduser().resolve()
    return destination.with_name(f"{destination.name}.provenance.json")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AtomicPredictionJournal:
    """Atomically persist and resume prediction records by opaque question key.

    The output file is replaced after every successful :meth:`append`.  If a
    process is killed while writing, the prior complete JSONL remains intact.
    A provenance sidecar prevents accidentally mixing records from different
    configs, adapters, reference files, splits, or control conditions.
    """

    def __init__(
        self,
        prediction_path: str | Path,
        provenance: PredictionProvenance,
        *,
        resume: bool = True,
        key_fields: Sequence[str] = ("scene_id", "question_id"),
    ) -> None:
        if not key_fields or len(set(key_fields)) != len(key_fields):
            raise ValueError("key_fields must be a non-empty unique sequence")
        self.path = Path(prediction_path).expanduser().resolve()
        self.provenance_path = provenance_path_for(self.path)
        self.provenance = provenance
        self.key_fields = tuple(key_fields)
        self._records: list[dict[str, Any]] = []
        self._keys: set[tuple[str, ...]] = set()

        if resume and self.path.exists():
            self._load_existing()
        else:
            _atomic_write_json(self.provenance_path, provenance.as_dict())
            atomic_write_jsonl(self.path, ())

    def _key(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        values: list[str] = []
        for field in self.key_fields:
            value = record.get(field)
            if value is None or not str(value):
                raise ValueError(f"Prediction record is missing non-empty {field!r}")
            values.append(str(value))
        return tuple(values)

    def _load_existing(self) -> None:
        if not self.provenance_path.is_file():
            raise RuntimeError(
                f"Cannot resume {self.path}: missing provenance sidecar "
                f"{self.provenance_path}"
            )
        stored = json.loads(self.provenance_path.read_text(encoding="utf-8"))
        if not isinstance(stored, dict):
            raise TypeError(f"Prediction provenance must be an object: {self.provenance_path}")
        expected = self.provenance.as_dict()
        if stored.get("provenance_sha256") != expected["provenance_sha256"]:
            raise RuntimeError(
                "Prediction resume provenance mismatch: cached predictions were produced "
                "by different config/checkpoint/reference inputs; use no-resume or a new output"
            )
        records = read_jsonl(self.path)
        for record in records:
            key = self._key(record)
            if key in self._keys:
                raise ValueError(f"Duplicate cached prediction key: {key}")
            record_hash = record.get("provenance_sha256")
            if record_hash != self.provenance.sha256:
                raise RuntimeError(f"Cached record {key} has incorrect prediction provenance")
            self._records.append(record)
            self._keys.add(key)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)

    @property
    def completed_keys(self) -> frozenset[tuple[str, ...]]:
        return frozenset(self._keys)

    def contains(self, *key_parts: str) -> bool:
        if len(key_parts) != len(self.key_fields):
            raise ValueError(f"Expected {len(self.key_fields)} key components")
        return tuple(str(item) for item in key_parts) in self._keys

    def append(self, record: Mapping[str, Any]) -> bool:
        """Persist one new result; return ``False`` when it was already cached."""

        stored = dict(record)
        key = self._key(stored)
        if key in self._keys:
            return False
        supplied_hash = stored.get("provenance_sha256")
        if supplied_hash not in (None, self.provenance.sha256):
            raise ValueError("Prediction record supplied an incompatible provenance_sha256")
        stored["provenance_sha256"] = self.provenance.sha256
        new_records = [*self._records, stored]
        atomic_write_jsonl(self.path, new_records)
        self._records = new_records
        self._keys.add(key)
        return True
