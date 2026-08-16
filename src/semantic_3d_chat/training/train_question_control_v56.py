"""Train a vocabulary-free question-control head on immutable V54 scene prefixes.

This is deliberately a narrow successor trainer.  It opens one explicitly named
training QA JSONL, caches complete question-independent prefixes from the frozen
V54 static-chat runtime, and optimizes only :class:`FullSceneQuestionControl`.
The runtime checkpoint contains no optimizer, QA, scene, or training metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import (
    RUNTIME_CONFIG_ROOT,
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT, project_path
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation.prediction_artifacts import (
    checkpoint_fingerprint,
)
from semantic_3d_chat.language.local_lm import prompt_token_ids, question_token_ids
from semantic_3d_chat.language.prefix_injection import (
    PrefixBatch,
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
    stack_prefix_batches,
)
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl
from semantic_3d_chat.training.train_adapter import forward_prefix_batch, tokenize_answer

_SCENE_ID = re.compile(r"scene_([0-9]{6})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRAINING_SCENE_NUMBERS: Final[frozenset[int]] = frozenset(
    (*range(11, 25), *range(31, 57))
)
_DEFERRED_FINAL_SCENE_NUMBERS: Final[frozenset[int]] = frozenset(range(25, 31))
_FRESH_DEVELOPMENT_SCENE_NUMBERS: Final[frozenset[int]] = frozenset(range(57, 63))
_RUNTIME_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "architecture",
        "hidden_size",
        "attention_dim",
        "control_tokens",
        "uniform_floor",
        "output_scale",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "question_dependent_scene_retrieval",
        "complete_scene_prefix_required",
        "environmental_text_inputs",
    }
)
_CACHE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "scene_count",
        "question_inputs_used",
        "question_dependent_scene_retrieval",
        "complete_scene_prefixes",
        "environmental_text_inputs",
        "scenes",
    }
)
_CACHE_SCENE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "filename",
        "file_sha256",
        "file_size_bytes",
        "prefix_sha256",
        "shape",
        "dtype",
    }
)
_FORBIDDEN_TRAINING_PATH_TOKENS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "development", "test", "final", "v55"}
)


@dataclass(frozen=True)
class CurriculumStep:
    """One optimizer step in the deterministic train-only curriculum."""

    epoch: int
    ordinal: int
    kind: str
    records: tuple[QARecord, ...]

    def signature(self) -> tuple[int, int, str, tuple[tuple[str, str], ...]]:
        return (
            self.epoch,
            self.ordinal,
            self.kind,
            tuple((record.scene_id, record.question_id) for record in self.records),
        )


@dataclass
class PrefixCacheResult:
    prefixes: dict[str, torch.Tensor]
    manifest: dict[str, Any]
    created: bool
    retained_runtime: Any | None = None


class StaticRuntimePrefixFactory:
    """Reuse one frozen V54 model stack while instantiating exact per-scene runtimes."""

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint: str | Path,
        bootstrap_scene_id: str,
    ) -> None:
        validate_training_scene_ids((bootstrap_scene_id,))
        self.config = config
        self.checkpoint = _resolve(checkpoint)
        self.bootstrap = StaticChatRuntime.load(
            config,
            bootstrap_scene_id,
            checkpoint=self.checkpoint,
            local_files_only=True,
        )

    def _map_data(self, scene_id: str) -> Any:
        path = project_path(self.config, "maps", scene_id, "voxel_map.npz")
        resolved = _resolve(path)
        _reject_symlink_components(resolved, "Numeric training map")
        if "oracle" in _scoped_path_tokens(resolved) or not resolved.is_file():
            raise FileNotFoundError(f"Sanitized numeric training map is unavailable: {resolved}")
        data = load_map_tensors(
            resolved,
            self.config["scene"]["room_size_m"],
            device="cpu",
            input_voxel_size_m=self.config["scene_encoder"].get(
                "input_voxel_size_m"
            ),
        )
        if data.feature_dim != int(self.bootstrap.checkpoint_metadata["semantic_dim"]):
            raise ValueError(f"Semantic map dimension changed for {scene_id}")
        return data.to(self.bootstrap.language.device)

    def load(self, scene_id: str) -> StaticChatRuntime:
        validate_training_scene_ids((scene_id,))
        if scene_id == self.bootstrap.scene_id:
            self.bootstrap.assert_prefix_unchanged()
            return self.bootstrap
        return StaticChatRuntime(
            config=self.config,
            scene_id=scene_id,
            checkpoint_path=self.bootstrap.checkpoint_path,
            checkpoint_metadata=self.bootstrap.checkpoint_metadata,
            language=self.bootstrap.language,
            map_data=self._map_data(scene_id),
            scene_model=self.bootstrap.scene_model,
            dense_aligner=self.bootstrap.dense_aligner,
            dense_sidecar_adapter=self.bootstrap.dense_sidecar_adapter,
            block_cross_residual=self.bootstrap.block_cross_residual,
            global_scene_residual=self.bootstrap.global_scene_residual,
            signed_x_scene_residual=self.bootstrap.signed_x_scene_residual,
            composer=self.bootstrap.composer,
            grounding=self.bootstrap.grounding,
            warnings=self.bootstrap.warnings,
            generation_function=self.bootstrap._generation_function,
        )


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _sha256_file(path: Path) -> str:
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


def _log_event(**payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _path_tokens(path: Path) -> set[str]:
    return {
        token
        for part in path.parts
        for token in re.split(r"[^a-z0-9]+", part.casefold())
        if token
    }


def _scoped_path_tokens(path: Path) -> set[str]:
    """Inspect repo-relative components without treating a pytest temp parent as data."""

    try:
        scoped = path.relative_to(PROJECT_ROOT)
    except ValueError:
        scoped = Path(path.name)
    return _path_tokens(scoped)


def _reject_symlink_components(path: Path, purpose: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} path must not contain symlinks: {current}")


def _validate_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def validate_training_scene_ids(scene_ids: Sequence[str]) -> tuple[str, ...]:
    """Accept only the preregistered V56 training scenes.

    V55's exhausted development scenes 19--24 are now explicitly training data.
    Deferred final scenes 25--30 and fresh development scenes 57--62 are never
    legal inputs to this trainer.
    """

    if not scene_ids:
        raise ValueError("At least one explicit training scene ID is required")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Training scene IDs must be unique")
    validated: list[str] = []
    for scene_id in scene_ids:
        match = _SCENE_ID.fullmatch(scene_id)
        if match is None:
            raise ValueError(f"Training scene ID is not opaque: {scene_id!r}")
        number = int(match.group(1))
        if number in _DEFERRED_FINAL_SCENE_NUMBERS:
            raise ValueError(f"Deferred final scene is forbidden during V56 training: {scene_id}")
        if number in _FRESH_DEVELOPMENT_SCENE_NUMBERS:
            raise ValueError(f"Fresh development scene is forbidden during V56 training: {scene_id}")
        if number not in _TRAINING_SCENE_NUMBERS:
            raise ValueError(f"Scene is outside the preregistered V56 training set: {scene_id}")
        validated.append(scene_id)
    return tuple(sorted(validated))


def _load_sanitized_runtime_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load an inference-only config or safe alias below ``configs/runtime``."""

    resolved = _resolve(path)
    _reject_symlink_components(resolved, "Base runtime config")
    try:
        resolved.relative_to(RUNTIME_CONFIG_ROOT)
    except ValueError as exc:
        raise ValueError("V56 requires a sanitized config below configs/runtime") from exc
    if not resolved.is_file() or resolved.suffix.casefold() not in {".yaml", ".yml"}:
        raise FileNotFoundError(f"Sanitized runtime config is unavailable: {resolved}")
    return load_runtime_config(resolved), resolved


def _validate_training_qa_path(path: str | Path) -> Path:
    resolved = _resolve(path)
    _reject_symlink_components(resolved, "Training QA")
    forbidden = sorted(_scoped_path_tokens(resolved) & _FORBIDDEN_TRAINING_PATH_TOKENS)
    if forbidden:
        raise ValueError(f"Training QA path contains forbidden split tokens: {forbidden}")
    if resolved.suffix.casefold() != ".jsonl" or not resolved.is_file():
        raise FileNotFoundError(f"Training QA JSONL is unavailable: {resolved}")
    return resolved


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"Training QA {field} must be a nonempty string or null")
    return value


def load_training_records(
    path: str | Path,
    *,
    scene_ids: Sequence[str],
) -> tuple[list[QARecord], str]:
    """Select explicit scenes from one train JSONL after validating every row's split."""

    allowed = set(validate_training_scene_ids(scene_ids))
    source = _validate_training_qa_path(path)
    records: list[QARecord] = []
    keys: set[tuple[str, str]] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"Training QA line {line_number} must be an object")
            required = ("scene_id", "question_id", "question", "answer", "answer_type")
            if any(not isinstance(value.get(field), str) or not value[field] for field in required):
                raise TypeError(f"Training QA line {line_number} has invalid required strings")
            scene_id = str(value["scene_id"])
            validate_training_scene_ids((scene_id,))
            key = (scene_id, str(value["question_id"]))
            if key in keys:
                raise ValueError(f"Training QA contains a duplicate opaque key: {key}")
            keys.add(key)
            if scene_id not in allowed:
                continue
            expected_change = value.get("counterfactual_expected_change")
            if expected_change is not None and not isinstance(expected_change, bool):
                raise TypeError("counterfactual_expected_change must be boolean or null")
            records.append(
                QARecord(
                    scene_id=scene_id,
                    question_id=str(value["question_id"]),
                    question=str(value["question"]),
                    answer=str(value["answer"]),
                    answer_type=str(value["answer_type"]),
                    target_xyz=None,
                    reference_xyz=None,
                    counterfactual_pair_id=_optional_string(
                        value.get("counterfactual_pair_id"), "counterfactual_pair_id"
                    ),
                    counterfactual_question_key=_optional_string(
                        value.get("counterfactual_question_key"),
                        "counterfactual_question_key",
                    ),
                    counterfactual_expected_change=expected_change,
                    counterfactual_role=_optional_string(
                        value.get("counterfactual_role"), "counterfactual_role"
                    ),
                    counterfactual_change_type=_optional_string(
                        value.get("counterfactual_change_type"),
                        "counterfactual_change_type",
                    ),
                )
            )
    if not records:
        raise ValueError("Training QA JSONL is empty")
    observed = {record.scene_id for record in records}
    if observed != allowed:
        raise ValueError(
            "Training QA does not exactly cover explicit scenes: "
            f"missing={sorted(allowed - observed)} extra={sorted(observed - allowed)}"
        )
    return records, _sha256_file(source)


def _changed_pair_units(records: Sequence[QARecord]) -> list[tuple[QARecord, QARecord]]:
    grouped: defaultdict[tuple[str, str], list[QARecord]] = defaultdict(list)
    for record in records:
        if record.counterfactual_expected_change is not True:
            continue
        if not record.counterfactual_pair_id or not record.counterfactual_question_key:
            raise ValueError("Changed counterfactual record lacks locked pair metadata")
        grouped[
            (record.counterfactual_pair_id, record.counterfactual_question_key)
        ].append(record)
    units: list[tuple[QARecord, QARecord]] = []
    for key, members in sorted(grouped.items()):
        ordered = sorted(
            members,
            key=lambda record: (
                0 if record.counterfactual_role == "reference" else 1,
                record.scene_id,
                record.question_id,
            ),
        )
        if (
            len(ordered) != 2
            or len({record.scene_id for record in ordered}) != 2
            or {record.counterfactual_role for record in ordered}
            != {"reference", "counterfactual"}
            or len({record.question for record in ordered}) != 1
            or len({record.answer_type for record in ordered}) != 1
        ):
            raise ValueError(f"Changed counterfactual unit is not a locked two-side pair: {key}")
        units.append((ordered[0], ordered[1]))
    if not units:
        raise ValueError("V56 curriculum requires changed counterfactual pair units")
    return units


def _shuffled_chunks(
    records: Sequence[QARecord],
    *,
    repeats: int,
    batch_size: int,
    rng: random.Random,
) -> list[tuple[QARecord, ...]]:
    expanded = [record for _ in range(repeats) for record in records]
    rng.shuffle(expanded)
    return [
        tuple(expanded[offset : offset + batch_size])
        for offset in range(0, len(expanded), batch_size)
    ]


def build_curriculum(
    records: Sequence[QARecord],
    *,
    epochs: int,
    seed: int,
    changed_pair_repeats: int = 4,
    count_replay_repeats: int = 2,
    broad_repeats: int = 1,
    replay_batch_size: int = 2,
) -> list[CurriculumStep]:
    """Interleave atomic changed pairs, count replay, and broad task replay."""

    integer_settings = {
        "epochs": epochs,
        "changed_pair_repeats": changed_pair_repeats,
        "count_replay_repeats": count_replay_repeats,
        "broad_repeats": broad_repeats,
        "replay_batch_size": replay_batch_size,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integer_settings.values()):
        raise ValueError(f"Curriculum integer settings must be positive: {integer_settings}")
    changed_units = _changed_pair_units(records)
    changed_ids = {id(record) for unit in changed_units for record in unit}
    count_rows = [
        record
        for record in records
        if id(record) not in changed_ids and record.answer_type == "count"
    ]
    broad_rows = [
        record
        for record in records
        if id(record) not in changed_ids and record.answer_type != "count"
    ]
    if not count_rows or not broad_rows:
        raise ValueError("V56 curriculum requires both count replay and broad training rows")

    schedule: list[CurriculumStep] = []
    ordinal = 0
    for epoch in range(epochs):
        rng = random.Random(seed + epoch * 1_000_003)
        pair_batches = [unit for _ in range(changed_pair_repeats) for unit in changed_units]
        rng.shuffle(pair_batches)
        count_batches = _shuffled_chunks(
            count_rows,
            repeats=count_replay_repeats,
            batch_size=replay_batch_size,
            rng=rng,
        )
        broad_batches = _shuffled_chunks(
            broad_rows,
            repeats=broad_repeats,
            batch_size=replay_batch_size,
            rng=rng,
        )
        width = max(len(pair_batches), len(count_batches), len(broad_batches))
        for index in range(width):
            for kind, batches in (
                ("changed_pair", pair_batches),
                ("count_replay", count_batches),
                ("broad", broad_batches),
            ):
                if index >= len(batches):
                    continue
                rows = tuple(batches[index])
                if kind == "changed_pair" and len(rows) != 2:
                    raise AssertionError("Changed-pair optimizer steps must keep both sides")
                schedule.append(CurriculumStep(epoch, ordinal, kind, rows))
                ordinal += 1
    return schedule


def curriculum_summary(schedule: Sequence[CurriculumStep]) -> dict[str, Any]:
    counts = Counter(step.kind for step in schedule)
    pair_units = {
        (
            step.records[0].counterfactual_pair_id,
            step.records[0].counterfactual_question_key,
        )
        for step in schedule
        if step.kind == "changed_pair"
    }
    signatures = [
        {
            "epoch": step.epoch,
            "ordinal": step.ordinal,
            "kind": step.kind,
            "opaque_keys": [
                [record.scene_id, record.question_id] for record in step.records
            ],
        }
        for step in schedule
    ]
    return {
        "step_count": len(schedule),
        "steps_by_kind": dict(sorted(counts.items())),
        "changed_pair_unit_count": len(pair_units),
        "schedule_sha256": _canonical_sha256(signatures),
        "paired_two_side_optimizer_steps": all(
            len(step.records) == 2
            and len({record.scene_id for record in step.records}) == 2
            for step in schedule
            if step.kind == "changed_pair"
        ),
    }


def _cache_path_guard(path: str | Path) -> Path:
    resolved = _resolve(path)
    _reject_symlink_components(resolved, "Prefix cache")
    forbidden = sorted(
        _scoped_path_tokens(resolved)
        & {"oracle", "validation", "development", "test", "final", "v55"}
    )
    if forbidden:
        raise ValueError(f"Prefix cache path contains forbidden tokens: {forbidden}")
    return resolved


def _tensor_dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _cache_entry(scene_id: str, path: Path, prefix: torch.Tensor) -> dict[str, Any]:
    return {
        "filename": f"{scene_id}.safetensors",
        "file_sha256": _sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "prefix_sha256": prefix_sha256(prefix),
        "shape": list(prefix.shape),
        "dtype": _tensor_dtype_name(prefix),
    }


def _validate_cache_manifest(
    value: object,
    *,
    scene_ids: Sequence[str],
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CACHE_MANIFEST_FIELDS:
        raise ValueError("Prefix cache manifest fields changed")
    expected_scenes = validate_training_scene_ids(scene_ids)
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("artifact") != "question_independent_scene_prefix_cache_v1"
        or value.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or value.get("base_runtime_config_sha256") != base_runtime_config_sha256
        or value.get("scene_count") != len(expected_scenes)
        or value.get("question_inputs_used") is not False
        or value.get("question_dependent_scene_retrieval") is not False
        or value.get("complete_scene_prefixes") is not True
        or value.get("environmental_text_inputs") != []
    ):
        raise ValueError("Prefix cache manifest contract mismatch")
    _validate_hash(value.get("base_checkpoint_sha256"), "cache base checkpoint")
    _validate_hash(value.get("base_runtime_config_sha256"), "cache runtime config")
    scenes = value.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != set(expected_scenes):
        raise ValueError("Prefix cache scene inventory changed")
    for scene_id, entry in scenes.items():
        if not isinstance(entry, Mapping) or set(entry) != _CACHE_SCENE_FIELDS:
            raise ValueError(f"Prefix cache scene fields changed: {scene_id}")
        if entry.get("filename") != f"{scene_id}.safetensors":
            raise ValueError("Prefix cache filenames must remain opaque and deterministic")
        _validate_hash(entry.get("file_sha256"), "cache file")
        _validate_hash(entry.get("prefix_sha256"), "cache prefix")
        size = entry.get("file_size_bytes")
        shape = entry.get("shape")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("Prefix cache file size must be positive")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape)
            or shape[0] != 1
        ):
            raise ValueError("Prefix cache tensor shape must be [1,S,H]")
        if entry.get("dtype") not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Prefix cache dtype is unsupported")
    return dict(value)


def load_prefix_cache(
    cache_path: str | Path,
    *,
    scene_ids: Sequence[str],
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    root = _cache_path_guard(cache_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Prefix cache is unavailable: {root}")
    expected_scenes = validate_training_scene_ids(scene_ids)
    expected_files = {"manifest.json", *(f"{scene_id}.safetensors" for scene_id in expected_scenes)}
    inventory = {item.name for item in root.iterdir()}
    if inventory != expected_files or any(item.is_symlink() for item in root.iterdir()):
        raise ValueError("Prefix cache contains an unexpected or symlinked file")
    manifest_path = root / "manifest.json"
    manifest = _validate_cache_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        scene_ids=expected_scenes,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=base_runtime_config_sha256,
    )
    prefixes: dict[str, torch.Tensor] = {}
    common_shape: tuple[int, ...] | None = None
    common_dtype: torch.dtype | None = None
    for scene_id in expected_scenes:
        entry = manifest["scenes"][scene_id]
        source = root / entry["filename"]
        if (
            not source.is_file()
            or source.stat().st_size != entry["file_size_bytes"]
            or _sha256_file(source) != entry["file_sha256"]
        ):
            raise ValueError(f"Cached prefix bytes changed: {scene_id}")
        state = load_file(str(source), device="cpu")
        if set(state) != {"scene_prefix"}:
            raise ValueError("Cached prefix safetensors must contain only scene_prefix")
        prefix = state["scene_prefix"].detach().contiguous()
        if (
            list(prefix.shape) != entry["shape"]
            or _tensor_dtype_name(prefix) != entry["dtype"]
            or prefix_sha256(prefix) != entry["prefix_sha256"]
            or not torch.isfinite(prefix).all()
        ):
            raise ValueError(f"Cached prefix tensor changed: {scene_id}")
        if common_shape is None:
            common_shape, common_dtype = tuple(prefix.shape), prefix.dtype
        elif tuple(prefix.shape) != common_shape or prefix.dtype != common_dtype:
            raise ValueError("Every cached scene prefix must share one exact shape and dtype")
        prefixes[scene_id] = prefix
    return prefixes, manifest


def ensure_prefix_cache(
    cache_path: str | Path,
    *,
    scene_ids: Sequence[str],
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    runtime_loader: Callable[[str], Any],
) -> PrefixCacheResult:
    """Build once from StaticChatRuntime or validate the immutable existing cache."""

    root = _cache_path_guard(cache_path)
    expected_scenes = validate_training_scene_ids(scene_ids)
    if root.exists():
        prefixes, manifest = load_prefix_cache(
            root,
            scene_ids=expected_scenes,
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
        )
        return PrefixCacheResult(prefixes, manifest, False)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    retained_runtime: Any | None = None
    try:
        entries: dict[str, Any] = {}
        # Load the first requested scene last so its language stack can be reused
        # for optimization without holding two Gemma copies during cache creation.
        build_order = (*expected_scenes[1:], expected_scenes[0])
        for scene_id in build_order:
            if retained_runtime is not None:
                del retained_runtime
                retained_runtime = None
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            runtime = runtime_loader(scene_id)
            runtime.assert_prefix_unchanged()
            prefix = runtime.scene_prefix.detach().cpu().contiguous()
            if (
                prefix.ndim != 3
                or prefix.shape[0] != 1
                or not torch.isfinite(prefix).all()
                or runtime.scene_prefix_hash != prefix_sha256(prefix)
            ):
                raise ValueError(f"Static runtime produced an invalid scene prefix: {scene_id}")
            destination = temporary / f"{scene_id}.safetensors"
            save_file({"scene_prefix": prefix}, destination)
            entries[scene_id] = _cache_entry(scene_id, destination, prefix)
            retained_runtime = runtime
            runtime = None
        manifest = {
            "schema_version": 1,
            "artifact": "question_independent_scene_prefix_cache_v1",
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "base_runtime_config_sha256": base_runtime_config_sha256,
            "scene_count": len(expected_scenes),
            "question_inputs_used": False,
            "question_dependent_scene_retrieval": False,
            "complete_scene_prefixes": True,
            "environmental_text_inputs": [],
            "scenes": {scene_id: entries[scene_id] for scene_id in expected_scenes},
        }
        _validate_cache_manifest(
            manifest,
            scene_ids=expected_scenes,
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
        )
        _write_json(temporary / "manifest.json", manifest)
        prefixes, validated = load_prefix_cache(
            temporary,
            scene_ids=expected_scenes,
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
        )
        os.rename(temporary, root)
        return PrefixCacheResult(prefixes, validated, True, retained_runtime)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _prepared_to_prefix_batch(prepared: Any) -> PrefixBatch:
    return PrefixBatch(
        inputs_embeds=prepared.inputs_embeds,
        attention_mask=prepared.attention_mask,
        labels=prepared.labels,
        scene_prefix_length=prepared.scene_prefix_length,
        per_layer_inputs=prepared.per_layer_inputs,
        mm_token_type_ids=prepared.mm_token_type_ids,
    )


def assert_answer_only_labels(labels: torch.Tensor | None, answer_ids: torch.Tensor) -> None:
    """Prove that teacher forcing supervises only the exact answer suffix."""

    if labels is None or labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("Teacher-forced labels must have shape [1,L]")
    answer_ids = answer_ids.to(labels.device)
    if answer_ids.ndim != 2 or answer_ids.shape[0] != 1 or answer_ids.shape[1] < 1:
        raise ValueError("Answer IDs must have shape [1,A] with A >= 1")
    answer_length = answer_ids.shape[1]
    if labels.shape[1] < answer_length:
        raise ValueError("Teacher-forced labels are shorter than the answer")
    if torch.any(labels[:, :-answer_length] != -100) or not torch.equal(
        labels[:, -answer_length:], answer_ids
    ):
        raise ValueError("Teacher-forced labels are not answer-only")


def question_control_answer_loss(
    *,
    runtime: Any,
    control: FullSceneQuestionControl,
    prefixes: Mapping[str, torch.Tensor],
    records: Sequence[QARecord],
) -> torch.Tensor:
    """Compute Gemma answer-only CE through continuous ``control_tokens``."""

    if not records:
        raise ValueError("A question-control training step cannot be empty")
    language = runtime.language
    backend = language.prefix_backend
    if backend is None or language.backend_name != "gemma4":
        raise RuntimeError("V56 question control requires the Gemma prefix backend")
    embedding_layer = language.model.get_input_embeddings()
    model_dtype = next(language.model.parameters()).dtype
    batches: list[PrefixBatch] = []
    for record in records:
        if record.scene_id not in prefixes:
            raise KeyError(f"Missing immutable prefix for {record.scene_id}")
        scene_prefix = prefixes[record.scene_id].to(
            device=language.device, dtype=model_dtype
        )
        prompt_ids = prompt_token_ids(
            language.tokenizer,
            str(runtime.config["language"]["system_prompt"]),
            record.question,
            language.device,
        )
        answer_ids = tokenize_answer(language.tokenizer, record.answer, language.device)
        question_ids = question_token_ids(
            language.tokenizer, record.question, language.device
        )
        with torch.no_grad():
            question_embeddings = embedding_layer(question_ids).detach().float()
        continuous_control = control(scene_prefix.float(), question_embeddings)
        prepared = backend.prepare(
            scene_prefix,
            prompt_ids,
            answer_ids,
            scene_prefix_after_bos=scene_prefix_after_bos_setting(runtime.config),
            scene_boundary_mode=scene_boundary_mode_setting(runtime.config),
            control_tokens=continuous_control.to(scene_prefix),
        )
        assert_answer_only_labels(prepared.labels, answer_ids)
        batches.append(_prepared_to_prefix_batch(prepared))
    stacked = stack_prefix_batches(
        batches,
        language.device,
        prefix_backend=backend,
    )
    output = forward_prefix_batch(language, stacked)
    loss = output.loss.float()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Question-control answer-only CE is nonfinite or nonscalar")
    return loss


def _select_training_device(runtime: Any, requested: str) -> torch.device:
    if requested not in {"auto", "mps", "cpu"}:
        raise ValueError("Training device must be auto, mps, or cpu")
    current = torch.device(runtime.language.device)
    if requested == "auto" or requested == current.type:
        return current
    if requested == "mps":
        raise RuntimeError("MPS was requested but the frozen runtime selected CPU")
    runtime.language.model.to(torch.device("cpu"))
    runtime.language.device = torch.device("cpu")
    runtime.scene_prefix = runtime.scene_prefix.cpu()
    return torch.device("cpu")


def freeze_base_runtime(runtime: Any) -> dict[str, int | bool]:
    """Freeze and count the complete V54 scene/LM stack before optimizer creation."""

    modules = [
        runtime.language.model,
        runtime.scene_model,
        runtime.composer,
        runtime.grounding,
        runtime.dense_aligner,
        runtime.dense_sidecar_adapter,
        runtime.block_cross_residual,
        runtime.global_scene_residual,
        runtime.signed_x_scene_residual,
    ]
    parameter_count = 0
    for module in modules:
        if module is None:
            continue
        module.requires_grad_(False)
        module.eval()
        parameter_count += sum(parameter.numel() for parameter in module.parameters())
    frozen = all(
        not parameter.requires_grad
        for module in modules
        if module is not None
        for parameter in module.parameters()
    )
    if not frozen:
        raise RuntimeError("V54 base stack did not freeze completely")
    return {"parameter_count": parameter_count, "all_parameters_frozen": True}


def build_runtime_metadata(
    control: FullSceneQuestionControl,
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
) -> dict[str, Any]:
    """Create the exact runtime loader contract with no training provenance."""

    metadata = {
        "schema_version": 1,
        "architecture": "full_scene_question_control_v1",
        "hidden_size": control.hidden_size,
        "attention_dim": control.attention_dim,
        "control_tokens": control.control_token_count,
        "uniform_floor": control.uniform_floor,
        "output_scale": control.output_scale,
        "weights_sha256": _validate_hash(weights_sha256, "control weights"),
        "base_checkpoint_sha256": _validate_hash(
            base_checkpoint_sha256, "base checkpoint"
        ),
        "base_runtime_config_sha256": _validate_hash(
            base_runtime_config_sha256, "base runtime config"
        ),
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
        "environmental_text_inputs": [],
    }
    if set(metadata) != _RUNTIME_METADATA_FIELDS:
        raise AssertionError("Question-control runtime metadata field contract changed")
    return metadata


def _safe_output_path(path: str | Path, purpose: str) -> Path:
    resolved = _resolve(path)
    _reject_symlink_components(resolved, purpose)
    if "v55" in resolved.as_posix().casefold():
        raise ValueError(f"{purpose} must not target immutable V55 artifacts")
    if resolved.exists():
        raise FileExistsError(f"{purpose} already exists; overwrite is forbidden: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _finite_float_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        if not tensor.is_floating_point():
            raise TypeError(f"Question-control state must be floating point: {name}")
        tensor = tensor.float().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Question-control state contains nonfinite values: {name}")
        state[name] = tensor
    return state


def save_control_checkpoint(
    checkpoint_path: str | Path,
    *,
    control: FullSceneQuestionControl,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
) -> dict[str, str]:
    """Write exactly two runtime files into a new, no-overwrite directory."""

    destination = _safe_output_path(checkpoint_path, "Control checkpoint")
    destination.mkdir(exist_ok=False)
    try:
        weights = destination / "control.safetensors"
        save_file(_finite_float_state(control), weights)
        weights_sha256 = _sha256_file(weights)
        # Reload before publication so malformed or nonfinite weights cannot be sealed.
        reloaded = load_file(str(weights), device="cpu")
        expected_state = _finite_float_state(control)
        if set(reloaded) != set(expected_state) or any(
            not torch.equal(reloaded[name], expected_state[name]) for name in expected_state
        ):
            raise RuntimeError("Saved question-control state failed exact reload validation")
        metadata = build_runtime_metadata(
            control,
            weights_sha256=weights_sha256,
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
        )
        _write_json(destination / "runtime_metadata.json", metadata)
        if {item.name for item in destination.iterdir()} != {
            "control.safetensors",
            "runtime_metadata.json",
        }:
            raise RuntimeError("Control checkpoint runtime inventory is not minimal")
        return {
            "weights_sha256": weights_sha256,
            "runtime_metadata_sha256": _sha256_file(
                destination / "runtime_metadata.json"
            ),
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _write_training_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = _safe_output_path(path, "Training report")
    _write_json(destination, report)
    return destination


def _epoch_loss_summary(losses: Sequence[tuple[int, float]]) -> list[dict[str, Any]]:
    grouped: defaultdict[int, list[float]] = defaultdict(list)
    for epoch, loss in losses:
        grouped[epoch].append(loss)
    return [
        {
            "epoch": epoch,
            "steps": len(values),
            "mean_answer_ce": sum(values) / len(values),
            "minimum_answer_ce": min(values),
            "maximum_answer_ce": max(values),
        }
        for epoch, values in sorted(grouped.items())
    ]


def train_question_control(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one no-resume, train-only V56 control-head run."""

    _validate_cli_numbers(args)
    scene_ids = validate_training_scene_ids(args.scene_id)
    raw_checkpoint_output = _resolve(args.output_checkpoint)
    raw_report_output = _resolve(args.training_report)
    if raw_checkpoint_output == raw_report_output or raw_report_output.is_relative_to(
        raw_checkpoint_output
    ):
        raise ValueError("Training report must remain outside the runtime checkpoint")
    checkpoint_output = _safe_output_path(raw_checkpoint_output, "Control checkpoint")
    report_output = _safe_output_path(raw_report_output, "Training report")

    config, config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, base_checkpoint_files = checkpoint_fingerprint(base_checkpoint)
    records, qa_sha256 = load_training_records(args.train_qa, scene_ids=scene_ids)
    schedule = build_curriculum(
        records,
        epochs=args.epochs,
        seed=args.seed,
        changed_pair_repeats=args.changed_pair_repeats,
        count_replay_repeats=args.count_replay_repeats,
        broad_repeats=args.broad_repeats,
        replay_batch_size=args.replay_batch_size,
    )

    _log_event(
        phase="v56_preflight_complete",
        scene_count=len(scene_ids),
        training_record_count=len(records),
        optimizer_step_count=len(schedule),
        base_checkpoint_sha256=base_checkpoint_sha256,
        runtime_config_sha256=runtime_config_sha256,
    )

    _log_event(phase="v56_base_runtime_load", scene_id=scene_ids[0])
    runtime_factory = StaticRuntimePrefixFactory(
        config,
        base_checkpoint,
        scene_ids[0],
    )

    cache_loads = 0

    def cache_runtime_loader(scene_id: str) -> StaticChatRuntime:
        nonlocal cache_loads
        cache_loads += 1
        _log_event(
            phase="v56_prefix_cache_build",
            scene_id=scene_id,
            scene_ordinal=cache_loads,
            scene_count=len(scene_ids),
        )
        return runtime_factory.load(scene_id)

    cache = ensure_prefix_cache(
        args.prefix_cache,
        scene_ids=scene_ids,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
        runtime_loader=cache_runtime_loader,
    )
    _log_event(
        phase="v56_prefix_cache_ready",
        created=cache.created,
        scene_count=len(cache.prefixes),
    )
    runtime = runtime_factory.bootstrap
    runtime.assert_prefix_unchanged()
    if not torch.equal(
        runtime.scene_prefix.detach().cpu(), cache.prefixes[scene_ids[0]]
    ):
        raise ValueError("Prefix cache does not exactly match StaticChatRuntime")
    device = _select_training_device(runtime, args.device)
    frozen_audit = freeze_base_runtime(runtime)
    runtime.language.enable_decoder_gradient_checkpointing()
    model_dtype = next(runtime.language.model.parameters()).dtype
    training_prefixes = {
        scene_id: prefix.to(device=device, dtype=model_dtype)
        for scene_id, prefix in cache.prefixes.items()
    }

    torch.manual_seed(args.seed)
    control = FullSceneQuestionControl(
        runtime.language.hidden_size,
        attention_dim=args.attention_dim,
        control_tokens=args.control_tokens,
        uniform_floor=args.uniform_floor,
        output_scale=args.output_scale,
    ).to(device=device, dtype=torch.float32)
    if control.parameter_count < 1 or any(
        not parameter.requires_grad for parameter in control.parameters()
    ):
        raise RuntimeError("Question-control head is not the exclusive trainable surface")
    optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    started = time.perf_counter()
    loss_log: list[tuple[int, float]] = []
    gradient_norms: list[float] = []
    control.train()
    for step in schedule:
        optimizer.zero_grad(set_to_none=True)
        loss = question_control_answer_loss(
            runtime=runtime,
            control=control,
            prefixes=training_prefixes,
            records=step.records,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            control.parameters(), args.gradient_clip_norm
        )
        gradient_value = float(gradient_norm.detach().float().cpu())
        if not math.isfinite(gradient_value):
            raise RuntimeError("Question-control gradient norm is nonfinite")
        optimizer.step()
        if any(not torch.isfinite(value).all() for value in control.state_dict().values()):
            raise RuntimeError("Question-control optimizer produced nonfinite state")
        loss_log.append((step.epoch, float(loss.detach().cpu())))
        gradient_norms.append(gradient_value)
        completed_steps = step.ordinal + 1
        if completed_steps % args.log_every == 0 or completed_steps == len(schedule):
            _log_event(
                phase="v56_training",
                completed_steps=completed_steps,
                optimizer_step_count=len(schedule),
                epoch=step.epoch,
                curriculum_kind=step.kind,
                answer_ce=loss_log[-1][1],
                preclip_gradient_norm=gradient_value,
            )
    control.eval()

    checkpoint_hashes = save_control_checkpoint(
        checkpoint_output,
        control=control,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    cache_manifest_path = _resolve(args.prefix_cache) / "manifest.json"
    report = {
        "schema_version": 1,
        "artifact": "v56_question_control_training",
        "passed": True,
        "base": {
            "checkpoint_sha256": base_checkpoint_sha256,
            "checkpoint_files": base_checkpoint_files,
            "runtime_config_effective_sha256": runtime_config_sha256,
            "runtime_config_file_sha256": _sha256_file(config_path),
        },
        "inputs": {
            "training_qa_sha256": qa_sha256,
            "training_record_count": len(records),
            "training_scene_ids": list(scene_ids),
            "prefix_cache_manifest_sha256": _sha256_file(cache_manifest_path),
            "prefix_sha256_by_scene": {
                scene_id: cache.manifest["scenes"][scene_id]["prefix_sha256"]
                for scene_id in scene_ids
            },
            "prefix_cache_created": cache.created,
        },
        "curriculum": curriculum_summary(schedule),
        "architecture": {
            "name": "full_scene_question_control_v1",
            "hidden_size": control.hidden_size,
            "attention_dim": control.attention_dim,
            "control_tokens": control.control_token_count,
            "uniform_floor": control.uniform_floor,
            "output_scale": control.output_scale,
            "parameter_count": control.parameter_count,
        },
        "optimization": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "optimizer_steps": len(schedule),
            "device": device.type,
            "elapsed_seconds": time.perf_counter() - started,
            "epoch_loss": _epoch_loss_summary(loss_log),
            "maximum_preclip_gradient_norm": max(gradient_norms),
        },
        "checkpoint": checkpoint_hashes,
        "scope": {
            "base_scene_stack_frozen": frozen_audit["all_parameters_frozen"],
            "base_parameter_count": frozen_audit["parameter_count"],
            "only_control_head_optimized": True,
            "answer_only_cross_entropy": True,
            "paired_two_side_optimizer_steps": True,
            "question_inputs_to_scene_prefix_cache": False,
            "question_dependent_scene_retrieval": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
            "optimizer_state_saved": False,
        },
    }
    _write_training_report(report_output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--train-qa", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=56056)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--changed-pair-repeats", type=int, default=4)
    parser.add_argument("--count-replay-repeats", type=int, default=2)
    parser.add_argument("--broad-repeats", type=int, default=1)
    parser.add_argument("--replay-batch-size", type=int, default=2)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--control-tokens", type=int, default=4)
    parser.add_argument("--uniform-floor", type=float, default=0.05)
    parser.add_argument("--output-scale", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def _validate_cli_numbers(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("V56 seed must be nonnegative")
    positive_ints = (
        "epochs",
        "changed_pair_repeats",
        "count_replay_repeats",
        "broad_repeats",
        "replay_batch_size",
        "attention_dim",
        "control_tokens",
        "log_every",
    )
    if any(getattr(args, field) < 1 for field in positive_ints):
        raise ValueError("V56 integer hyperparameters must be positive")
    positive_floats = ("uniform_floor", "output_scale", "learning_rate", "gradient_clip_norm")
    if any(
        not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0.0
        for field in positive_floats
    ):
        raise ValueError("V56 positive hyperparameters must be finite")
    if args.uniform_floor > 1.0:
        raise ValueError("uniform_floor must not exceed one")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_cli_numbers(args)
    report = train_question_control(args)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "optimizer_steps": report["optimization"]["optimizer_steps"],
                "checkpoint": str(_resolve(args.output_checkpoint)),
                "training_report": str(_resolve(args.training_report)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CurriculumStep",
    "PrefixCacheResult",
    "assert_answer_only_labels",
    "build_curriculum",
    "build_runtime_metadata",
    "curriculum_summary",
    "ensure_prefix_cache",
    "load_prefix_cache",
    "load_training_records",
    "main",
    "save_control_checkpoint",
    "train_question_control",
    "validate_training_scene_ids",
]
