"""Create the training-only V66 teacher bank for seven missing answer classes.

V62 optimized prompts only for counterfactual-changed rows.  Consequently its
otherwise verified cache has no prompt for seven canonical answers that occur
only in the remaining authenticated training population.  This module fills
that exact hole without looking at validation, scorer, oracle, fresh-development,
or deferred-final artifacts.

For every missing ``(answer class, training pair)`` group, one common numeric
``[1, 4, 1536]`` prompt is optimized against both authorized scene/question
rows in that group.  Publication is allowed only when the same prompt greedily
answers both rows.  Seven classes already present in V62 have teachers from
only one pair; this bank also adds one verified alternate-pair prototype for
each so pair-held-out training never borrows its held pair.  The final cache
contains opaque IDs, numeric tensors,
numeric diagnostics, and hashes; it never stores answer or question text and
is explicitly forbidden at chat/runtime inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.question_control_runtime import (
    _load_control_head,
    question_control_training_artifact_root,
)
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import exact_normalized_match, normalize_answer
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    PINNED_V62_PREREGISTRATION_SHA256,
    load_filtered_training_qa,
)
from semantic_3d_chat.training.soft_prompt_teacher_v62 import (
    PROMPT_SHAPE,
    _source_initial_prompt,
    load_v62_teacher_cache,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    _select_training_device,
    freeze_base_runtime,
    load_prefix_cache,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _disable_decoder_checkpointing,
    _generate_with_control,
    _pooled_question_embedding,
    _teacher_nll,
)
from semantic_3d_chat.training.train_question_control_v63 import (
    V63Row,
    _validated_rows,
    training_scene_ids,
)
from semantic_3d_chat.training.train_question_control_v65 import (
    validate_training_baseline_lock,
)

_SCHEMA: Final[int] = 1
_ARTIFACT: Final[str] = "v66_training_only_missing_answer_class_teacher_cache_v1"
_WORK_ARTIFACT: Final[str] = "v66_missing_answer_class_teacher_work_v1"
_RECORD_ARTIFACT: Final[str] = "v66_verified_answer_pair_teacher_record_v1"
_PROMPT_SHAPE: Final[tuple[int, int, int]] = PROMPT_SHAPE
_EXPECTED_MISSING_CLASS_COUNT: Final[int] = 7
_EXPECTED_ALTERNATE_CLASS_COUNT: Final[int] = 7
_EXPECTED_CLASS_COUNT: Final[int] = 14
_EXPECTED_PROTOTYPE_COUNT: Final[int] = 63
_EXPECTED_VERIFICATION_ROW_COUNT: Final[int] = 126
_EXPECTED_TRAIN_ROW_COUNT: Final[int] = 576
_EXPECTED_V62_TEACHER_COUNT: Final[int] = 80
_EXPECTED_MISSING_CLASS_IDS: Final[frozenset[str]] = frozenset(
    {
        "answer_16477688c0e00699c6cf",
        "answer_99f7855eb789dfecf9ea",
        "answer_9cb4bb5e93df46436db4",
        "answer_b1b886ce5f5750a00ae3",
        "answer_b1f51a511f1da0cd348b",
        "answer_b31420793ee136a0ece4",
        "answer_f3a22c1ce8e0a5a96393",
    }
)
_EXPECTED_ALTERNATE_CLASS_IDS: Final[frozenset[str]] = frozenset(
    {
        "answer_4d8aaa64f68587b18e27",
        "answer_8a798890fe93817163b1",
        "answer_9390298f3fb0c5b16049",
        "answer_a3e1f4935b0919b346c2",
        "answer_ba4788b226aa8dc2e6dc",
        "answer_c685a2c9bab235ccdd2a",
        "answer_d6d9ba45963232b7e73c",
    }
)
_EXPECTED_OPTIMIZED_CLASS_IDS: Final[frozenset[str]] = (
    _EXPECTED_MISSING_CLASS_IDS | _EXPECTED_ALTERNATE_CLASS_IDS
)
_SHA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "weights_sha256",
        "run_signature_sha256",
        "filtered_train_jsonl_sha256",
        "training_baseline_lock_sha256",
        "v62_teacher_metadata_sha256",
        "v62_teacher_weights_sha256",
        "base_checkpoint_sha256",
        "runtime_config_sha256",
        "source_control_checkpoint_sha256",
        "prefix_cache_manifest_sha256",
        "class_inventory_sha256",
        "source_key_inventory_sha256",
        "verification_key_inventory_sha256",
    }
)


@dataclass(frozen=True)
class V66TeacherGroup:
    """One missing answer class observed in one authorized training pair."""

    answer_class_id: str
    pair_id: str
    rows: tuple[V63Row, V63Row]
    source_purpose: str = "missing_class_all_pairs"

    @property
    def source(self) -> V63Row:
        return min(self.rows, key=lambda row: row.key)

    @property
    def key(self) -> tuple[str, str]:
        return self.answer_class_id, self.pair_id


@dataclass(frozen=True)
class V66CompletedPrototype:
    group: V66TeacherGroup
    prompt: torch.Tensor
    target_token_ids_sha256: str
    optimization: dict[str, Any]


@dataclass(frozen=True)
class V66TeacherPreflight:
    config: dict[str, Any]
    runtime_config_sha256: str
    base_checkpoint: Path
    base_checkpoint_sha256: str
    source_control: torch.nn.Module
    source_control_checkpoint_sha256: str
    source_control_metadata: dict[str, Any]
    prefixes: dict[str, torch.Tensor]
    prefix_cache_manifest_sha256: str
    rows: tuple[V63Row, ...]
    groups: tuple[V66TeacherGroup, ...]
    filtered_train_jsonl_sha256: str
    training_baseline_lock_sha256: str
    v62_teacher_metadata_sha256: str
    v62_teacher_weights_sha256: str
    work_directory: Path
    output_artifact: Path
    optimizer: dict[str, int | float]
    run_manifest: dict[str, Any]


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


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().float().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _answer_class_id(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V66 canonical training answer normalizes to empty")
    return f"answer_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:20]}"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_symlink_components(path: Path, purpose: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V66 {purpose} path contains a symlink: {current}")


def _training_output_path(
    path: str | Path,
    *,
    training_root: Path,
    purpose: str,
) -> Path:
    output = _resolve(path)
    _reject_symlink_components(output, purpose)
    try:
        relative = output.relative_to(training_root)
    except ValueError as exc:
        raise ValueError(f"V66 {purpose} must remain below {training_root}") from exc
    if not relative.parts:
        raise ValueError(f"V66 {purpose} cannot replace the training root")
    return output


def _group_missing_answer_classes(
    rows: Sequence[V63Row],
    v62_teacher_keys: set[tuple[str, str]],
    *,
    enforce_production_inventory: bool = True,
) -> tuple[V66TeacherGroup, ...]:
    """Derive missing classes plus alternate sources for sparse V62 classes."""

    by_key = {row.key: row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("V66 training rows contain duplicate opaque keys")
    if not v62_teacher_keys or not v62_teacher_keys.issubset(by_key):
        raise ValueError("V66 V62 teacher keys escape the authenticated training rows")
    all_classes = {_answer_class_id(row.answer) for row in rows}
    teacher_classes = {_answer_class_id(by_key[key].answer) for key in v62_teacher_keys}
    missing = all_classes - teacher_classes
    if enforce_production_inventory and missing != _EXPECTED_MISSING_CLASS_IDS:
        raise ValueError("V66 missing answer-class inventory differs from its pin")

    all_grouped: defaultdict[tuple[str, str], list[V63Row]] = defaultdict(list)
    for row in rows:
        class_id = _answer_class_id(row.answer)
        all_grouped[(class_id, row.pair_id)].append(row)
    teacher_pairs: defaultdict[str, set[str]] = defaultdict(set)
    for key in v62_teacher_keys:
        row = by_key[key]
        teacher_pairs[_answer_class_id(row.answer)].add(row.pair_id)

    selected: list[tuple[str, str, str]] = []
    selected.extend(
        (class_id, pair_id, "missing_class_all_pairs")
        for class_id, pair_id in all_grouped
        if class_id in missing
    )
    for class_id in sorted(teacher_pairs):
        all_pairs = {pair_id for candidate, pair_id in all_grouped if candidate == class_id}
        if len(teacher_pairs[class_id]) == 1 and len(all_pairs) > 1:
            alternate_pair = min(all_pairs - teacher_pairs[class_id])
            selected.append((class_id, alternate_pair, "alternate_pair_coverage"))

    result: list[V66TeacherGroup] = []
    for class_id, pair_id, source_purpose in sorted(selected):
        members = all_grouped[(class_id, pair_id)]
        by_scene: defaultdict[str, list[V63Row]] = defaultdict(list)
        for row in members:
            by_scene[row.scene_id].append(row)
        ordered = tuple(
            min(scene_rows, key=lambda row: row.key)
            for _scene_id, scene_rows in sorted(by_scene.items())
        )
        if len(ordered) != 2:
            raise ValueError(
                "V66 each missing class/pair prototype requires exactly two scenes"
            )
        normalized = {normalize_answer(row.answer) for row in ordered}
        if len(normalized) != 1:
            raise RuntimeError("V66 answer-class digest collision")
        result.append(
            V66TeacherGroup(
                answer_class_id=class_id,
                pair_id=pair_id,
                rows=(ordered[0], ordered[1]),
                source_purpose=source_purpose,
            )
        )
    groups = tuple(result)
    if enforce_production_inventory:
        if (
            len(rows) != _EXPECTED_TRAIN_ROW_COUNT
            or len(v62_teacher_keys) != _EXPECTED_V62_TEACHER_COUNT
            or len(groups) != _EXPECTED_PROTOTYPE_COUNT
            or sum(len(group.rows) for group in groups)
            != _EXPECTED_VERIFICATION_ROW_COUNT
            or {group.answer_class_id for group in groups}
            != _EXPECTED_OPTIMIZED_CLASS_IDS
            or sum(group.source_purpose == "missing_class_all_pairs" for group in groups)
            != 56
            or sum(group.source_purpose == "alternate_pair_coverage" for group in groups)
            != _EXPECTED_ALTERNATE_CLASS_COUNT
        ):
            raise ValueError("V66 missing answer prototype inventory changed")
        class_pair_counts: defaultdict[str, int] = defaultdict(int)
        for group in groups:
            class_pair_counts[group.answer_class_id] += 1
        if any(
            class_pair_counts[class_id] < 5 for class_id in _EXPECTED_MISSING_CLASS_IDS
        ):
            raise ValueError("V66 every missing class requires at least five source pairs")
    return groups


def _opaque_keys_sha256(keys: Sequence[tuple[str, str]]) -> str:
    return _canonical_sha256(
        [
            {"scene_id": scene_id, "question_id": question_id}
            for scene_id, question_id in sorted(keys)
        ]
    )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"V66 {field} must be a positive integer")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V66 {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"V66 {field} must be finite and positive")
    return result


def _optimizer_settings(args: argparse.Namespace) -> dict[str, int | float]:
    minimum = _positive_int(args.teacher_min_steps, "minimum teacher steps")
    maximum = _positive_int(args.teacher_max_steps, "maximum teacher steps")
    if minimum > maximum:
        raise ValueError("V66 minimum teacher steps exceed maximum teacher steps")
    return {
        "learning_rate": _positive_float(
            args.teacher_learning_rate, "teacher learning rate"
        ),
        "minimum_steps": minimum,
        "maximum_steps": maximum,
        "nll_threshold": _positive_float(
            args.teacher_nll_threshold, "teacher NLL threshold"
        ),
        "gradient_clip_norm": _positive_float(
            args.teacher_gradient_clip_norm, "teacher gradient clip norm"
        ),
    }


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_v66_teacher_preflight(args: argparse.Namespace) -> V66TeacherPreflight:
    """Authenticate every training-only dependency before loading Gemma."""

    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("V66 seed must be a nonnegative integer")
    optimizer = _optimizer_settings(args)

    # First read: the create-once hash-only training baseline authorization.
    baseline_path = _resolve(args.training_baseline_lock)
    baseline = validate_training_baseline_lock(baseline_path)
    baseline_sha256 = _sha256_file(baseline_path)

    filtered_path = _resolve(args.filtered_train_qa)
    _reject_symlink_components(filtered_path, "filtered training JSONL")
    raw_rows = load_filtered_training_qa(filtered_path)
    rows = _validated_rows(raw_rows)
    if set(baseline.required_output_hashes) != {row.key for row in rows}:
        raise ValueError("V66 training baseline and authenticated row inventory differ")
    filtered_sha256 = _sha256_file(filtered_path)

    v62_root = _resolve(args.v62_teacher_cache)
    v62_teachers, v62_metadata = load_v62_teacher_cache(v62_root)
    if (
        v62_metadata.get("filtered_train_jsonl_sha256") != filtered_sha256
        or v62_metadata.get("preregistration_sha256")
        != PINNED_V62_PREREGISTRATION_SHA256
        or v62_metadata.get("target_count") != _EXPECTED_V62_TEACHER_COUNT
        or v62_metadata.get("runtime_load_permitted") is not False
        or v62_metadata.get("validation_inputs_used") is not False
        or v62_metadata.get("held_out_inputs_used") is not False
    ):
        raise ValueError("V66 V62 teacher cache provenance changed")
    groups = _group_missing_answer_classes(rows, set(v62_teachers))

    config, _config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, _base_files = checkpoint_fingerprint(base_checkpoint)
    if (
        base_checkpoint_sha256 != v62_metadata.get("base_checkpoint_sha256")
        or base_checkpoint_sha256 != baseline.payload.get("v54_checkpoint_sha256")
        or runtime_config_sha256 != v62_metadata.get("runtime_config_sha256")
        or runtime_config_sha256
        != baseline.payload.get("v54_runtime_config_effective_sha256")
    ):
        raise ValueError("V66 frozen V54 runtime provenance differs from its locks")

    source_checkpoint = _resolve(args.source_control_checkpoint)
    source_sha256 = _control_checkpoint_sha256(source_checkpoint)
    source_control, source_metadata = _load_control_head(
        source_checkpoint,
        hidden_size=_PROMPT_SHAPE[2],
        device=torch.device("cpu"),
    )
    if (
        source_metadata.get("hidden_size") != _PROMPT_SHAPE[2]
        or source_metadata.get("control_tokens") != _PROMPT_SHAPE[1]
        or source_metadata.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or source_metadata.get("base_runtime_config_sha256")
        != runtime_config_sha256
    ):
        raise ValueError("V66 initialization controller is incompatible")

    scene_ids = training_scene_ids()
    prefix_root = _resolve(args.prefix_cache)
    prefixes, _prefix_manifest = load_prefix_cache(
        prefix_root,
        scene_ids=scene_ids,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    prefix_manifest_sha256 = _sha256_file(prefix_root / "manifest.json")
    if (
        prefix_manifest_sha256
        != v62_metadata.get("prefix_cache_manifest_sha256")
        or any(
            tuple(prefix.shape) != (1, 258, _PROMPT_SHAPE[2])
            or not torch.isfinite(prefix).all()
            for prefix in prefixes.values()
        )
    ):
        raise ValueError("V66 complete-scene prefix cache provenance changed")

    training_root = question_control_training_artifact_root(config).resolve()
    work_directory = _training_output_path(
        args.work_directory,
        training_root=training_root,
        purpose="resumable work directory",
    )
    output_artifact = _training_output_path(
        args.output_artifact,
        training_root=training_root,
        purpose="final teacher cache",
    )
    if (
        work_directory == output_artifact
        or work_directory.is_relative_to(output_artifact)
        or output_artifact.is_relative_to(work_directory)
    ):
        raise ValueError("V66 work and final cache paths must be disjoint")

    group_identity = [
        {
            "answer_class_id": group.answer_class_id,
            "source_pair_id": group.pair_id,
            "source_scene_id": group.source.scene_id,
            "source_question_id": group.source.question_id,
            "source_purpose": group.source_purpose,
            "verification_keys": [
                {"scene_id": row.scene_id, "question_id": row.question_id}
                for row in group.rows
            ],
        }
        for group in groups
    ]
    identity: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "artifact": _WORK_ARTIFACT,
        "filtered_train_jsonl_sha256": filtered_sha256,
        "training_baseline_lock_sha256": baseline_sha256,
        "v62_teacher_metadata_sha256": _sha256_file(v62_root / "metadata.json"),
        "v62_teacher_weights_sha256": _sha256_file(
            v62_root / "teachers.safetensors"
        ),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "source_control_checkpoint_sha256": source_sha256,
        "prefix_cache_manifest_sha256": prefix_manifest_sha256,
        "prompt_shape": list(_PROMPT_SHAPE),
        "answer_class_count": _EXPECTED_CLASS_COUNT,
        "missing_answer_class_count": _EXPECTED_MISSING_CLASS_COUNT,
        "alternate_pair_class_count": _EXPECTED_ALTERNATE_CLASS_COUNT,
        "prototype_count": _EXPECTED_PROTOTYPE_COUNT,
        "verification_row_count": _EXPECTED_VERIFICATION_ROW_COUNT,
        "groups_sha256": _canonical_sha256(group_identity),
        "optimizer": optimizer,
        "seed": args.seed,
        "answer_or_question_text_stored": False,
        "runtime_load_permitted": False,
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_inputs_used": False,
        "fresh_development_inputs_used": False,
        "deferred_final_inputs_used": False,
    }
    run_manifest = {**identity, "run_signature_sha256": _canonical_sha256(identity)}
    return V66TeacherPreflight(
        config=config,
        runtime_config_sha256=runtime_config_sha256,
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
        source_control=source_control,
        source_control_checkpoint_sha256=source_sha256,
        source_control_metadata=source_metadata,
        prefixes=prefixes,
        prefix_cache_manifest_sha256=prefix_manifest_sha256,
        rows=rows,
        groups=groups,
        filtered_train_jsonl_sha256=filtered_sha256,
        training_baseline_lock_sha256=baseline_sha256,
        v62_teacher_metadata_sha256=_sha256_file(v62_root / "metadata.json"),
        v62_teacher_weights_sha256=_sha256_file(v62_root / "teachers.safetensors"),
        work_directory=work_directory,
        output_artifact=output_artifact,
        optimizer=optimizer,
        run_manifest=run_manifest,
    )


def _record_name(group: V66TeacherGroup) -> str:
    return f"record_{_canonical_sha256(list(group.key))[:24]}"


def _prepare_work_directory(preflight: V66TeacherPreflight) -> None:
    root = preflight.work_directory
    if root.exists():
        if not root.is_dir() or root.is_symlink():
            raise ValueError("V66 work path is not a regular directory")
        if {item.name for item in root.iterdir()} != {"manifest.json", "records"}:
            raise ValueError("V66 work inventory changed")
        observed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if observed != preflight.run_manifest:
            raise ValueError("V66 resumable work manifest differs from this run")
        records = root / "records"
        if not records.is_dir() or records.is_symlink():
            raise ValueError("V66 resumable records path is invalid")
        for partial in records.glob(".record-*.partial-*"):
            if partial.is_dir() and not partial.is_symlink():
                shutil.rmtree(partial)
            else:
                raise ValueError("V66 stale partial record is unsafe")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.partial-", dir=root.parent))
    try:
        (temporary / "records").mkdir()
        _write_json_new(temporary / "manifest.json", preflight.run_manifest)
        os.rename(temporary, root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validated_optimization(value: object) -> dict[str, Any]:
    fields = {
        "steps",
        "initial_mean_nll",
        "final_mean_nll",
        "maximum_preclip_gradient_norm",
        "initial_rms",
        "final_rms",
        "learning_rate",
        "attempt_count",
        "attempt_learning_rates",
        "total_forward_steps",
        "training_row_count",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("V66 optimization fields changed")
    result = dict(value)
    for field in ("steps", "attempt_count", "total_forward_steps", "training_row_count"):
        _positive_int(result[field], f"optimization {field}")
    rates = result["attempt_learning_rates"]
    if (
        not isinstance(rates, list)
        or len(rates) != result["attempt_count"]
        or any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(float(rate))
            or float(rate) <= 0.0
            for rate in rates
        )
    ):
        raise ValueError("V66 adaptive learning-rate inventory changed")
    for field in (
        "initial_mean_nll",
        "final_mean_nll",
        "maximum_preclip_gradient_norm",
        "initial_rms",
        "final_rms",
        "learning_rate",
    ):
        number = result[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0.0
        ):
            raise ValueError(f"V66 optimization {field} is invalid")
    return result


def _save_work_record(
    preflight: V66TeacherPreflight,
    completed: V66CompletedPrototype,
) -> V66CompletedPrototype:
    prompt = completed.prompt.detach().cpu().float().contiguous()
    if tuple(prompt.shape) != _PROMPT_SHAPE or not torch.isfinite(prompt).all():
        raise ValueError("V66 completed prompt must be finite [1,4,1536]")
    _validated_optimization(completed.optimization)
    if not _is_sha256(completed.target_token_ids_sha256):
        raise ValueError("V66 target token digest is invalid")
    destination = preflight.work_directory / "records" / _record_name(completed.group)
    if destination.exists():
        raise FileExistsError("V66 completed work record already exists")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        weights = temporary / "prompt.safetensors"
        save_file({"prompt": prompt}, weights)
        metadata = {
            "schema_version": _SCHEMA,
            "artifact": _RECORD_ARTIFACT,
            "run_signature_sha256": preflight.run_manifest[
                "run_signature_sha256"
            ],
            "answer_class_id": completed.group.answer_class_id,
            "source_pair_id": completed.group.pair_id,
            "source_scene_id": completed.group.source.scene_id,
            "source_question_id": completed.group.source.question_id,
            "source_purpose": completed.group.source_purpose,
            "verification_keys": [
                {"scene_id": row.scene_id, "question_id": row.question_id}
                for row in completed.group.rows
            ],
            "target_token_ids_sha256": completed.target_token_ids_sha256,
            "weights_sha256": _sha256_file(weights),
            "prompt_sha256": _tensor_sha256(prompt),
            "shape": list(_PROMPT_SHAPE),
            "rms": float(prompt.square().mean().sqrt()),
            "greedy_canonical_exact": 2,
            "greedy_canonical_total": 2,
            "optimization": completed.optimization,
        }
        _write_json_new(temporary / "metadata.json", metadata)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _load_work_record(preflight, completed.group)


def _load_work_record(
    preflight: V66TeacherPreflight,
    group: V66TeacherGroup,
) -> V66CompletedPrototype:
    source = preflight.work_directory / "records" / _record_name(group)
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError("V66 completed work record is unavailable")
    if {item.name for item in source.iterdir()} != {
        "prompt.safetensors",
        "metadata.json",
    } or any(item.is_symlink() for item in source.iterdir()):
        raise ValueError("V66 completed work record inventory changed")
    weights = source / "prompt.safetensors"
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "artifact",
        "run_signature_sha256",
        "answer_class_id",
        "source_pair_id",
        "source_scene_id",
        "source_question_id",
        "source_purpose",
        "verification_keys",
        "target_token_ids_sha256",
        "weights_sha256",
        "prompt_sha256",
        "shape",
        "rms",
        "greedy_canonical_exact",
        "greedy_canonical_total",
        "optimization",
    }
    expected_verification = [
        {"scene_id": row.scene_id, "question_id": row.question_id}
        for row in group.rows
    ]
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != fields
        or metadata.get("schema_version") != _SCHEMA
        or metadata.get("artifact") != _RECORD_ARTIFACT
        or metadata.get("run_signature_sha256")
        != preflight.run_manifest["run_signature_sha256"]
        or metadata.get("answer_class_id") != group.answer_class_id
        or metadata.get("source_pair_id") != group.pair_id
        or metadata.get("source_scene_id") != group.source.scene_id
        or metadata.get("source_question_id") != group.source.question_id
        or metadata.get("source_purpose") != group.source_purpose
        or metadata.get("verification_keys") != expected_verification
        or metadata.get("weights_sha256") != _sha256_file(weights)
        or metadata.get("shape") != list(_PROMPT_SHAPE)
        or metadata.get("greedy_canonical_exact") != 2
        or metadata.get("greedy_canonical_total") != 2
        or not _is_sha256(metadata.get("target_token_ids_sha256"))
        or not _is_sha256(metadata.get("prompt_sha256"))
    ):
        raise ValueError("V66 completed work record contract changed")
    state = load_file(str(weights), device="cpu")
    if set(state) != {"prompt"}:
        raise ValueError("V66 completed work tensor inventory changed")
    prompt = state["prompt"].detach().float().contiguous()
    if (
        tuple(prompt.shape) != _PROMPT_SHAPE
        or not torch.isfinite(prompt).all()
        or _tensor_sha256(prompt) != metadata["prompt_sha256"]
        or abs(float(metadata["rms"]) - float(prompt.square().mean().sqrt())) > 1e-7
    ):
        raise ValueError("V66 completed work prompt changed")
    return V66CompletedPrototype(
        group=group,
        prompt=prompt,
        target_token_ids_sha256=str(metadata["target_token_ids_sha256"]),
        optimization=_validated_optimization(metadata["optimization"]),
    )


def _existing_work_records(
    preflight: V66TeacherPreflight,
) -> dict[tuple[str, str], V66CompletedPrototype]:
    expected = {_record_name(group): group for group in preflight.groups}
    records_root = preflight.work_directory / "records"
    observed = {item.name for item in records_root.iterdir()}
    if observed - set(expected):
        raise ValueError("V66 work cache contains unexpected records")
    result: dict[tuple[str, str], V66CompletedPrototype] = {}
    for name, group in expected.items():
        if (records_root / name).exists():
            completed = _load_work_record(preflight, group)
            result[group.key] = completed
    return result


def _joint_teacher_nll(
    *,
    runtime: Any,
    prefixes: Mapping[str, torch.Tensor],
    group: V66TeacherGroup,
    prompt: torch.Tensor,
) -> torch.Tensor:
    losses = [
        _teacher_nll(
            runtime=runtime,
            scene_prefix=prefixes[row.scene_id],
            record=row,
            free_prompt=prompt,
        )
        for row in group.rows
    ]
    loss = torch.stack(losses).mean()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("V66 joint teacher NLL is nonfinite or nonscalar")
    return loss


def _optimize_joint_prompt_once(
    *,
    runtime: Any,
    prefixes: Mapping[str, torch.Tensor],
    group: V66TeacherGroup,
    initial_prompt: torch.Tensor,
    learning_rate: float,
    min_steps: int,
    max_steps: int,
    nll_threshold: float,
    gradient_clip_norm: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prompt = torch.nn.Parameter(initial_prompt.detach().float().clone())
    optimizer = torch.optim.Adam((prompt,), lr=learning_rate)
    losses: list[float] = []
    gradients: list[float] = []
    best_prompt = initial_prompt.detach().float().clone()
    best_loss = math.inf
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _joint_teacher_nll(
            runtime=runtime,
            prefixes=prefixes,
            group=group,
            prompt=prompt,
        )
        value = float(loss.detach().cpu())
        losses.append(value)
        if value < best_loss:
            best_loss = value
            best_prompt = prompt.detach().clone()
        if step + 1 >= min_steps and value <= nll_threshold:
            break
        if step >= 1 and value > losses[0] * 1.5 and best_loss >= losses[0] * 0.99:
            break
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_((prompt,), gradient_clip_norm)
        gradient_value = float(gradient.detach().float().cpu())
        if not math.isfinite(gradient_value):
            raise RuntimeError("V66 joint teacher gradient is nonfinite")
        optimizer.step()
        if not torch.isfinite(prompt).all():
            raise RuntimeError("V66 joint teacher optimizer produced nonfinite prompt")
        gradients.append(gradient_value)
    return best_prompt, {
        "steps": len(losses),
        "initial_mean_nll": losses[0],
        "final_mean_nll": best_loss,
        "maximum_preclip_gradient_norm": max(gradients, default=0.0),
        "initial_rms": float(initial_prompt.detach().float().square().mean().sqrt()),
        "final_rms": float(best_prompt.detach().float().square().mean().sqrt()),
        "learning_rate": learning_rate,
        "training_row_count": len(group.rows),
    }


def _optimize_joint_prompt_adaptive(**kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    base_rate = float(kwargs.pop("learning_rate"))
    threshold = float(kwargs["nll_threshold"])
    rates = tuple(
        dict.fromkeys(
            (base_rate, base_rate / 3.0, base_rate / 10.0, base_rate / 30.0)
        )
    )
    attempts: list[dict[str, Any]] = []
    best_prompt: torch.Tensor | None = None
    best_metrics: dict[str, Any] | None = None
    for rate in rates:
        prompt, metrics = _optimize_joint_prompt_once(
            **kwargs,
            learning_rate=rate,
        )
        attempts.append(metrics)
        if best_metrics is None or metrics["final_mean_nll"] < best_metrics[
            "final_mean_nll"
        ]:
            best_prompt, best_metrics = prompt, metrics
        if metrics["final_mean_nll"] <= threshold:
            break
    if best_prompt is None or best_metrics is None:
        raise RuntimeError("V66 adaptive joint teacher made no optimization attempt")
    return best_prompt, {
        **best_metrics,
        "attempt_count": len(attempts),
        "attempt_learning_rates": [item["learning_rate"] for item in attempts],
        "total_forward_steps": sum(item["steps"] for item in attempts),
    }


def _target_token_ids_sha256(runtime: Any, answer: str) -> str:
    normalized = normalize_answer(answer)
    encoded = runtime.language.tokenizer(
        normalized,
        add_special_tokens=False,
        return_tensors="pt",
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] < 1:
        raise ValueError("V66 target answer produced invalid token IDs")
    values = ids.detach().cpu().to(dtype=torch.int64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(values.shape)).encode())
    digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _load_runtime(
    preflight: V66TeacherPreflight,
    requested_device: str,
) -> tuple[Any, torch.device, torch.dtype]:
    first_scene = training_scene_ids()[0]
    factory = StaticRuntimePrefixFactory(
        preflight.config,
        preflight.base_checkpoint,
        first_scene,
    )
    runtime = factory.bootstrap
    if not torch.equal(
        runtime.scene_prefix.detach().cpu(),
        preflight.prefixes[first_scene].detach().cpu(),
    ):
        raise ValueError("V66 cached prefix differs from frozen V54 runtime")
    freeze_base_runtime(runtime)
    device = _select_training_device(runtime, requested_device)
    model_dtype = next(runtime.language.model.parameters()).dtype
    return runtime, device, model_dtype


def _final_metadata(
    preflight: V66TeacherPreflight,
    completed: Sequence[V66CompletedPrototype],
    *,
    weights_sha256: str,
) -> dict[str, Any]:
    ordered = sorted(completed, key=lambda item: item.group.key)
    source_keys = [item.group.source.key for item in ordered]
    verification_keys = [row.key for item in ordered for row in item.group.rows]
    class_ids = sorted({item.group.answer_class_id for item in ordered})
    return {
        "schema_version": _SCHEMA,
        "artifact": _ARTIFACT,
        "weights_sha256": weights_sha256,
        "run_signature_sha256": preflight.run_manifest["run_signature_sha256"],
        "filtered_train_jsonl_sha256": preflight.filtered_train_jsonl_sha256,
        "training_baseline_lock_sha256": preflight.training_baseline_lock_sha256,
        "v62_teacher_metadata_sha256": preflight.v62_teacher_metadata_sha256,
        "v62_teacher_weights_sha256": preflight.v62_teacher_weights_sha256,
        "base_checkpoint_sha256": preflight.base_checkpoint_sha256,
        "runtime_config_sha256": preflight.runtime_config_sha256,
        "source_control_checkpoint_sha256": (
            preflight.source_control_checkpoint_sha256
        ),
        "prefix_cache_manifest_sha256": preflight.prefix_cache_manifest_sha256,
        "class_inventory_sha256": _canonical_sha256(class_ids),
        "source_key_inventory_sha256": _opaque_keys_sha256(source_keys),
        "verification_key_inventory_sha256": _opaque_keys_sha256(
            verification_keys
        ),
        "prompt_shape": list(_PROMPT_SHAPE),
        "answer_class_count": len(class_ids),
        "missing_answer_class_count": _EXPECTED_MISSING_CLASS_COUNT,
        "alternate_pair_class_count": _EXPECTED_ALTERNATE_CLASS_COUNT,
        "prototype_count": len(ordered),
        "verified_training_row_count": len(verification_keys),
        "greedy_canonical_exact": len(verification_keys),
        "greedy_canonical_total": len(verification_keys),
        "answer_text_used_training_only": True,
        "answer_or_question_text_stored": False,
        "runtime_load_permitted": False,
        "environmental_text_inputs": [],
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_inputs_used": False,
        "fresh_development_inputs_used": False,
        "deferred_final_inputs_used": False,
        "records": [
            {
                "tensor_key": f"prototype_{index:06d}",
                "answer_class_id": item.group.answer_class_id,
                "source_pair_id": item.group.pair_id,
                "source_scene_id": item.group.source.scene_id,
                "source_question_id": item.group.source.question_id,
                "source_purpose": item.group.source_purpose,
                "verification_keys": [
                    {"scene_id": row.scene_id, "question_id": row.question_id}
                    for row in item.group.rows
                ],
                "target_token_ids_sha256": item.target_token_ids_sha256,
                "prompt_sha256": _tensor_sha256(item.prompt),
                "shape": list(_PROMPT_SHAPE),
                "rms": float(item.prompt.float().square().mean().sqrt()),
                "greedy_canonical_exact": 2,
                "greedy_canonical_total": 2,
                "optimization": item.optimization,
            }
            for index, item in enumerate(ordered)
        ],
    }


def save_v66_answer_class_teacher_cache(
    preflight: V66TeacherPreflight,
    completed: Sequence[V66CompletedPrototype],
) -> dict[str, Any]:
    """Atomically publish the complete create-once numeric cache."""

    ordered = tuple(sorted(completed, key=lambda item: item.group.key))
    if (
        {item.group.key for item in ordered} != {group.key for group in preflight.groups}
        or len(ordered) != len(preflight.groups)
    ):
        raise ValueError("V66 cannot publish an incomplete prototype inventory")
    destination = preflight.output_artifact
    if destination.exists():
        _loaded, metadata = load_v66_answer_class_teacher_cache(destination)
        if metadata.get("run_signature_sha256") != preflight.run_manifest.get(
            "run_signature_sha256"
        ):
            raise FileExistsError("Existing V66 cache belongs to a different run")
        return metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        weights = temporary / "teachers.safetensors"
        save_file(
            {
                f"prototype_{index:06d}": item.prompt.detach()
                .cpu()
                .float()
                .contiguous()
                for index, item in enumerate(ordered)
            },
            weights,
        )
        metadata = _final_metadata(
            preflight,
            ordered,
            weights_sha256=_sha256_file(weights),
        )
        _write_json_new(temporary / "metadata.json", metadata)
        loaded, validated = load_v66_answer_class_teacher_cache(temporary)
        expected = {item.group.source.key: item.prompt.detach().cpu().float() for item in ordered}
        if (
            validated != metadata
            or set(loaded) != set(expected)
            or any(not torch.equal(loaded[key], value) for key, value in expected.items())
        ):
            raise RuntimeError("V66 final cache failed exact strict reload")
        os.rename(temporary, destination)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_v66_answer_class_teacher_cache(
    source: str | Path,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]:
    """Strict training-side loader returning one verified prompt per source key."""

    root = _resolve(source)
    _reject_symlink_components(root, "final teacher cache")
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"V66 answer-class teacher cache is unavailable: {root}")
    if {item.name for item in root.iterdir()} != {
        "teachers.safetensors",
        "metadata.json",
    } or any(item.is_symlink() for item in root.iterdir()):
        raise ValueError("V66 final cache inventory changed")
    weights = root / "teachers.safetensors"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "artifact",
        *_SHA_FIELDS,
        "prompt_shape",
        "answer_class_count",
        "missing_answer_class_count",
        "alternate_pair_class_count",
        "prototype_count",
        "verified_training_row_count",
        "greedy_canonical_exact",
        "greedy_canonical_total",
        "answer_text_used_training_only",
        "answer_or_question_text_stored",
        "runtime_load_permitted",
        "environmental_text_inputs",
        "validation_inputs_used",
        "scorer_inputs_used",
        "oracle_inputs_used",
        "fresh_development_inputs_used",
        "deferred_final_inputs_used",
        "records",
    }
    records = metadata.get("records") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != required
        or metadata.get("schema_version") != _SCHEMA
        or metadata.get("artifact") != _ARTIFACT
        or metadata.get("weights_sha256") != _sha256_file(weights)
        or metadata.get("prompt_shape") != list(_PROMPT_SHAPE)
        or metadata.get("answer_class_count") != _EXPECTED_CLASS_COUNT
        or metadata.get("missing_answer_class_count") != _EXPECTED_MISSING_CLASS_COUNT
        or metadata.get("alternate_pair_class_count")
        != _EXPECTED_ALTERNATE_CLASS_COUNT
        or metadata.get("prototype_count") != _EXPECTED_PROTOTYPE_COUNT
        or metadata.get("verified_training_row_count")
        != _EXPECTED_VERIFICATION_ROW_COUNT
        or metadata.get("greedy_canonical_exact")
        != _EXPECTED_VERIFICATION_ROW_COUNT
        or metadata.get("greedy_canonical_total")
        != _EXPECTED_VERIFICATION_ROW_COUNT
        or metadata.get("answer_text_used_training_only") is not True
        or metadata.get("answer_or_question_text_stored") is not False
        or metadata.get("runtime_load_permitted") is not False
        or metadata.get("environmental_text_inputs") != []
        or any(
            metadata.get(field) is not False
            for field in (
                "validation_inputs_used",
                "scorer_inputs_used",
                "oracle_inputs_used",
                "fresh_development_inputs_used",
                "deferred_final_inputs_used",
            )
        )
        or not isinstance(records, list)
        or len(records) != _EXPECTED_PROTOTYPE_COUNT
    ):
        raise ValueError("V66 final cache contract changed")
    if any(not _is_sha256(metadata.get(field)) for field in _SHA_FIELDS):
        raise ValueError("V66 final cache contains an invalid provenance digest")

    state = load_file(str(weights), device="cpu")
    if set(state) != {record.get("tensor_key") for record in records if isinstance(record, Mapping)}:
        raise ValueError("V66 final cache tensor inventory changed")
    record_fields = {
        "tensor_key",
        "answer_class_id",
        "source_pair_id",
        "source_scene_id",
        "source_question_id",
        "source_purpose",
        "verification_keys",
        "target_token_ids_sha256",
        "prompt_sha256",
        "shape",
        "rms",
        "greedy_canonical_exact",
        "greedy_canonical_total",
        "optimization",
    }
    result: dict[tuple[str, str], torch.Tensor] = {}
    class_ids: set[str] = set()
    verification_keys: list[tuple[str, str]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError("V66 final cache record fields changed")
        verification = record.get("verification_keys")
        if (
            not isinstance(record.get("answer_class_id"), str)
            or record["answer_class_id"] not in _EXPECTED_OPTIMIZED_CLASS_IDS
            or not isinstance(record.get("source_pair_id"), str)
            or not str(record["source_pair_id"]).startswith("pair_")
            or not isinstance(record.get("source_scene_id"), str)
            or not str(record["source_scene_id"]).startswith("scene_")
            or not isinstance(record.get("source_question_id"), str)
            or not str(record["source_question_id"]).startswith("q_")
            or record.get("source_purpose")
            not in {"missing_class_all_pairs", "alternate_pair_coverage"}
            or not isinstance(verification, list)
            or len(verification) != 2
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"scene_id", "question_id"}
                or not isinstance(item["scene_id"], str)
                or not isinstance(item["question_id"], str)
                for item in verification
            )
            or record.get("shape") != list(_PROMPT_SHAPE)
            or record.get("greedy_canonical_exact") != 2
            or record.get("greedy_canonical_total") != 2
            or not _is_sha256(record.get("target_token_ids_sha256"))
            or not _is_sha256(record.get("prompt_sha256"))
        ):
            raise ValueError("V66 final cache opaque record contract changed")
        key = str(record["source_scene_id"]), str(record["source_question_id"])
        if key in result or not any(
            item["scene_id"] == key[0] and item["question_id"] == key[1]
            for item in verification
        ):
            raise ValueError("V66 final cache source key changed")
        tensor_key = record["tensor_key"]
        prompt = state[tensor_key].detach().float().contiguous()
        if (
            tuple(prompt.shape) != _PROMPT_SHAPE
            or not torch.isfinite(prompt).all()
            or _tensor_sha256(prompt) != record["prompt_sha256"]
            or abs(float(record["rms"]) - float(prompt.square().mean().sqrt()))
            > 1e-7
        ):
            raise ValueError("V66 final cache numeric prompt changed")
        _validated_optimization(record["optimization"])
        result[key] = prompt
        class_ids.add(str(record["answer_class_id"]))
        verification_keys.extend(
            (str(item["scene_id"]), str(item["question_id"])) for item in verification
        )
    if (
        class_ids != _EXPECTED_OPTIMIZED_CLASS_IDS
        or _canonical_sha256(sorted(class_ids))
        != metadata["class_inventory_sha256"]
        or _opaque_keys_sha256(tuple(result))
        != metadata["source_key_inventory_sha256"]
        or len(set(verification_keys)) != _EXPECTED_VERIFICATION_ROW_COUNT
        or _opaque_keys_sha256(verification_keys)
        != metadata["verification_key_inventory_sha256"]
    ):
        raise ValueError("V66 final cache aggregate inventory changed")
    return result, dict(metadata)


def generate_v66_answer_class_teacher_cache(
    args: argparse.Namespace,
    *,
    runtime_provider: Callable[
        [V66TeacherPreflight, str], tuple[Any, torch.device, torch.dtype]
    ]
    | None = None,
    optimizer_fn: Callable[..., tuple[torch.Tensor, dict[str, Any]]] | None = None,
    generator_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Generate/resume the cache; hooks keep unit tests independent of Gemma."""

    preflight = build_v66_teacher_preflight(args)
    inventory = {
        "passed": True,
        "mode": "dry_run_inventory" if args.dry_run_inventory else "generate",
        "gemma_loaded": False,
        "answer_class_count": _EXPECTED_CLASS_COUNT,
        "missing_answer_class_count": _EXPECTED_MISSING_CLASS_COUNT,
        "alternate_pair_class_count": _EXPECTED_ALTERNATE_CLASS_COUNT,
        "prototype_count": len(preflight.groups),
        "verification_row_count": sum(len(group.rows) for group in preflight.groups),
        "prompt_shape": list(_PROMPT_SHAPE),
        "run_signature_sha256": preflight.run_manifest["run_signature_sha256"],
        "answer_or_question_text_stored": False,
        "runtime_load_permitted": False,
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_inputs_used": False,
        "fresh_development_inputs_used": False,
        "deferred_final_inputs_used": False,
    }
    if args.dry_run_inventory:
        return inventory

    _prepare_work_directory(preflight)
    completed = _existing_work_records(preflight)
    if preflight.output_artifact.exists():
        metadata = save_v66_answer_class_teacher_cache(
            preflight, tuple(completed.values())
        )
        return {
            **inventory,
            "mode": "already_complete",
            "completed_prototype_count": len(completed),
            "weights_sha256": metadata["weights_sha256"],
        }

    provider = runtime_provider or _load_runtime
    optimize = optimizer_fn or _optimize_joint_prompt_adaptive
    generate = generator_fn or _generate_with_control
    runtime, device, model_dtype = provider(preflight, args.device)
    source_control = preflight.source_control.to(device=device, dtype=torch.float32).eval()
    prefixes = {
        scene_id: prefix.to(device=device, dtype=model_dtype)
        for scene_id, prefix in preflight.prefixes.items()
    }
    resumed = len(completed)
    runtime.language.enable_decoder_gradient_checkpointing()
    checkpointing_enabled = True
    try:
        for ordinal, group in enumerate(preflight.groups, start=1):
            if group.key in completed:
                continue
            record_seed = int(
                _canonical_sha256([args.seed, *group.key])[:16],
                16,
            )
            random.seed(record_seed)
            torch.manual_seed(record_seed % (2**63 - 1))
            initial_members: list[torch.Tensor] = []
            for row in group.rows:
                pooled = _pooled_question_embedding(runtime, row.question).to(device=device)
                initial_members.append(
                    _source_initial_prompt(
                        source_control,
                        prefixes[row.scene_id],
                        pooled,
                    )
                )
            initial = torch.stack(initial_members).mean(dim=0)
            prompt, metrics = optimize(
                runtime=runtime,
                prefixes=prefixes,
                group=group,
                initial_prompt=initial,
                learning_rate=preflight.optimizer["learning_rate"],
                min_steps=preflight.optimizer["minimum_steps"],
                max_steps=preflight.optimizer["maximum_steps"],
                nll_threshold=preflight.optimizer["nll_threshold"],
                gradient_clip_norm=preflight.optimizer["gradient_clip_norm"],
            )
            _disable_decoder_checkpointing(runtime.language)
            checkpointing_enabled = False
            exact = 0
            for row in group.rows:
                generated = generate(
                    runtime=runtime,
                    scene_prefix=prefixes[row.scene_id],
                    question=row.question,
                    control_tokens=prompt,
                )
                exact += int(exact_normalized_match(generated, row.answer))
            if exact != len(group.rows):
                raise RuntimeError(
                    "V66 shared answer-pair prompt failed greedy canonical verification: "
                    f"answer_class_id={group.answer_class_id} pair_id={group.pair_id} "
                    f"exact={exact}/{len(group.rows)}"
                )
            target_digests = {
                _target_token_ids_sha256(runtime, row.answer) for row in group.rows
            }
            if len(target_digests) != 1:
                raise RuntimeError("V66 shared class rows tokenize inconsistently")
            completed[group.key] = _save_work_record(
                preflight,
                V66CompletedPrototype(
                    group=group,
                    prompt=prompt,
                    target_token_ids_sha256=next(iter(target_digests)),
                    optimization=_validated_optimization(metrics),
                ),
            )
            print(
                json.dumps(
                    {
                        "phase": "v66_answer_class_teacher_complete",
                        "completed": len(completed),
                        "total": len(preflight.groups),
                        "ordinal": ordinal,
                        "answer_class_id": group.answer_class_id,
                        "source_pair_id": group.pair_id,
                        "greedy_canonical_exact": exact,
                        "greedy_canonical_total": len(group.rows),
                        "final_mean_nll": metrics["final_mean_nll"],
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            if len(completed) < len(preflight.groups):
                runtime.language.enable_decoder_gradient_checkpointing()
                checkpointing_enabled = True
    finally:
        if checkpointing_enabled:
            _disable_decoder_checkpointing(runtime.language)

    metadata = save_v66_answer_class_teacher_cache(
        preflight, tuple(completed.values())
    )
    return {
        **inventory,
        "mode": "complete",
        "gemma_loaded": True,
        "resumed_prototype_count": resumed,
        "new_prototype_count": len(completed) - resumed,
        "completed_prototype_count": len(completed),
        "greedy_canonical_exact": metadata["greedy_canonical_exact"],
        "greedy_canonical_total": metadata["greedy_canonical_total"],
        "output_artifact": str(preflight.output_artifact),
        "weights_sha256": metadata["weights_sha256"],
        "metadata_sha256": _sha256_file(preflight.output_artifact / "metadata.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-baseline-lock", required=True)
    parser.add_argument("--filtered-train-qa", required=True)
    parser.add_argument("--v62-teacher-cache", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--source-control-checkpoint", required=True)
    parser.add_argument("--work-directory", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=66066)
    parser.add_argument("--teacher-learning-rate", type=float, default=0.03)
    parser.add_argument("--teacher-min-steps", type=int, default=5)
    parser.add_argument("--teacher-max-steps", type=int, default=30)
    parser.add_argument("--teacher-nll-threshold", type=float, default=1e-3)
    parser.add_argument("--teacher-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--dry-run-inventory", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = generate_v66_answer_class_teacher_cache(args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V66CompletedPrototype",
    "V66TeacherGroup",
    "V66TeacherPreflight",
    "build_v66_teacher_preflight",
    "generate_v66_answer_class_teacher_cache",
    "load_v66_answer_class_teacher_cache",
    "save_v66_answer_class_teacher_cache",
]
