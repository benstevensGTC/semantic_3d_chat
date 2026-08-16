"""Resumable training-only Gemma soft-prompt cache generation for V62.

This command consumes one hash-bound, preregistered *training* JSONL and
selects exactly rows whose ``counterfactual_expected_change`` value is true.
It initializes a numeric ``[1, 4, 1536]`` prompt from a sanitized controller,
optimizes that prompt through the frozen local Gemma decoder using V58's
proven adaptive routine, and requires greedy canonical-answer verification.

Completed records are published atomically into a resumable work directory.
The final artifact contains only opaque identities, numeric tensors, numeric
optimization diagnostics, and strict provenance hashes.  It is explicitly
training-only and has no chat/runtime integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
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
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation.metrics import exact_normalized_match
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    PINNED_V62_PREREGISTRATION_SHA256,
    add_baseline_lock_authorization_argument,
    add_filtered_training_data_argument,
    load_filtered_training_qa,
    validate_baseline_lock,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    _select_training_device,
    freeze_base_runtime,
    load_prefix_cache,
    validate_training_scene_ids,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _disable_decoder_checkpointing,
    _generate_with_control,
    _optimize_teacher_prompt,
    _optimize_teacher_prompt_adaptive,
    _pooled_question_embedding,
)

PROMPT_SHAPE: Final[tuple[int, int, int]] = (1, 4, 1536)
_SHA256_LENGTH: Final[int] = 64
_SELECTION_PREDICATE: Final[str] = "counterfactual_expected_change=true"
_WORK_ARTIFACT: Final[str] = "v62_soft_prompt_teacher_generation_work_v1"
_FINAL_ARTIFACT: Final[str] = "v62_training_only_changed_soft_prompt_teacher_cache"
_RECORD_ARTIFACT: Final[str] = "v62_verified_soft_prompt_record_v1"
_OPTIMIZATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "steps",
        "initial_nll",
        "final_nll",
        "minimum_nll",
        "maximum_preclip_gradient_norm",
        "initial_rms",
        "final_rms",
        "learning_rate",
        "attempt_count",
        "attempt_learning_rates",
        "total_forward_steps",
    }
)


@dataclass(frozen=True)
class V62TeacherPreflight:
    """Validated inventory required before Gemma may be loaded."""

    config: dict[str, Any]
    config_path: Path
    runtime_config_sha256: str
    base_checkpoint: Path
    base_checkpoint_sha256: str
    source_control_checkpoint: Path
    source_control_checkpoint_sha256: str
    source_control: torch.nn.Module
    source_control_metadata: dict[str, Any]
    filtered_train_jsonl: Path
    filtered_train_jsonl_sha256: str
    preregistration_sha256: str
    baseline_lock: Path
    baseline_lock_sha256: str
    baseline_lock_authorization: dict[str, Any]
    scene_ids: tuple[str, ...]
    records: tuple[QARecord, ...]
    total_filtered_rows: int
    selection_sha256: str
    prefixes: dict[str, torch.Tensor]
    prefix_cache: Path
    prefix_cache_manifest_sha256: str
    work_directory: Path
    output_artifact: Path
    optimizer: dict[str, int | float]
    run_manifest: dict[str, Any]


@dataclass(frozen=True)
class V62CompletedTeacher:
    """One verified numeric prompt recovered from the resumable work cache."""

    scene_id: str
    question_id: str
    prompt: torch.Tensor
    optimization: dict[str, Any]

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.question_id


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


def _digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"V62 {field} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().float().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _reject_symlink_components(path: Path, purpose: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V62 {purpose} path contains a symlink: {current}")


def _validated_filtered_train_path(
    actual: str | Path,
    expected_sha256: str,
) -> tuple[Path, str, tuple[dict[str, Any], ...]]:
    """Authenticate through the sole public V62 training-data loader."""

    source = _resolve(actual)
    _reject_symlink_components(source, "filtered training JSONL")
    if source.suffix.casefold() != ".jsonl" or not source.is_file():
        raise FileNotFoundError(f"V62 filtered training JSONL is unavailable: {source}")
    observed = _sha256_file(source)
    if observed != _digest(expected_sha256, "filtered training JSONL"):
        raise ValueError("V62 filtered training JSONL digest changed")
    # This independent public boundary pins the exact canonical artifact bytes,
    # all 12 train-pair IDs, 24 train scenes, and every row's schema.
    rows = load_filtered_training_qa(source)
    return source, observed, rows


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"V62 training row {field} must be a nonempty string or null")
    return value


def _load_changed_training_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    scene_ids: Sequence[str],
) -> tuple[tuple[QARecord, ...], int, str]:
    """Return exactly expected-change rows from the authenticated boundary."""

    allowed = set(validate_training_scene_ids(scene_ids))
    all_scenes: set[str] = set()
    keys: set[tuple[str, str]] = set()
    changed: list[QARecord] = []
    total = len(rows)
    for line_number, value in enumerate(rows, start=1):
        if not isinstance(value, Mapping):
            raise TypeError(f"V62 training line {line_number} must be an object")
        required = ("scene_id", "question_id", "question", "answer", "answer_type")
        if any(
            not isinstance(value.get(field), str) or not value[field]
            for field in required
        ):
            raise TypeError(f"V62 training line {line_number} has invalid strings")
        scene_id = str(value["scene_id"])
        validate_training_scene_ids((scene_id,))
        if scene_id not in allowed:
            raise ValueError(
                "V62 filtered training JSONL contains a scene outside the exact "
                f"requested inventory: {scene_id}"
            )
        question_id = str(value["question_id"])
        key = (scene_id, question_id)
        if key in keys:
            raise ValueError(f"V62 filtered training JSONL duplicates opaque key: {key}")
        keys.add(key)
        all_scenes.add(scene_id)
        expected_change = value.get("counterfactual_expected_change")
        if not isinstance(expected_change, bool):
            raise TypeError(
                "V62 filtered training rows require boolean "
                "counterfactual_expected_change"
            )
        if expected_change is not True:
            continue
        changed.append(
            QARecord(
                scene_id=scene_id,
                question_id=question_id,
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
                counterfactual_expected_change=True,
                counterfactual_role=_optional_string(
                    value.get("counterfactual_role"), "counterfactual_role"
                ),
                counterfactual_change_type=_optional_string(
                    value.get("counterfactual_change_type"),
                    "counterfactual_change_type",
                ),
            )
        )
    if total < 1 or not changed:
        raise ValueError("V62 filtered training JSONL has no expected-change records")
    if all_scenes != allowed:
        raise ValueError(
            "V62 filtered training JSONL does not exactly cover requested scenes: "
            f"missing={sorted(allowed - all_scenes)}"
        )
    ordered = tuple(sorted(changed, key=lambda item: (item.scene_id, item.question_id)))
    selection_sha256 = _canonical_sha256(
        {
            "predicate": _SELECTION_PREDICATE,
            "opaque_keys": [[record.scene_id, record.question_id] for record in ordered],
        }
    )
    return ordered, total, selection_sha256


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"V62 {field} must be a positive integer")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V62 {field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"V62 {field} must be a finite positive number")
    return result


def _optimizer_settings(args: argparse.Namespace) -> dict[str, int | float]:
    minimum = _positive_int(args.teacher_min_steps, "teacher minimum steps")
    maximum = _positive_int(args.teacher_max_steps, "teacher maximum steps")
    if minimum > maximum:
        raise ValueError("V62 teacher minimum steps exceed maximum steps")
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


def _training_output_path(path: str | Path, training_root: Path, purpose: str) -> Path:
    result = _resolve(path)
    _reject_symlink_components(result, purpose)
    try:
        relative = result.relative_to(training_root)
    except ValueError as exc:
        raise ValueError(f"V62 {purpose} must remain below {training_root}") from exc
    if not relative.parts:
        raise ValueError(f"V62 {purpose} cannot replace the training root")
    return result


def _source_contract_shape(metadata: Mapping[str, Any]) -> tuple[int, int, int]:
    hidden = metadata.get("hidden_size")
    controls = metadata.get("control_tokens")
    if hidden != PROMPT_SHAPE[2] or controls != PROMPT_SHAPE[1]:
        raise ValueError(
            "V62 source controller must emit exact [1,4,1536] prompts: "
            f"control_tokens={controls} hidden_size={hidden}"
        )
    return PROMPT_SHAPE


def _build_run_manifest(
    *,
    filtered_train_jsonl: Path,
    filtered_train_jsonl_sha256: str,
    baseline_lock_sha256: str,
    preregistration_sha256: str,
    scene_ids: Sequence[str],
    total_filtered_rows: int,
    selected_record_count: int,
    selection_sha256: str,
    runtime_config_sha256: str,
    base_checkpoint_sha256: str,
    source_control_checkpoint_sha256: str,
    prefix_cache_manifest_sha256: str,
    optimizer: Mapping[str, int | float],
    seed: int,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema_version": 1,
        "artifact": _WORK_ARTIFACT,
        "filtered_train_jsonl": str(filtered_train_jsonl),
        "filtered_train_jsonl_sha256": filtered_train_jsonl_sha256,
        "preregistration_sha256": preregistration_sha256,
        "baseline_lock_sha256": baseline_lock_sha256,
        "scene_ids": list(scene_ids),
        "scene_ids_sha256": _canonical_sha256(list(scene_ids)),
        "total_filtered_rows": total_filtered_rows,
        "selected_record_count": selected_record_count,
        "selection_predicate": _SELECTION_PREDICATE,
        "selection_sha256": selection_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "source_control_checkpoint_sha256": source_control_checkpoint_sha256,
        "prefix_cache_manifest_sha256": prefix_cache_manifest_sha256,
        "prompt_shape": list(PROMPT_SHAPE),
        "optimizer": dict(optimizer),
        "seed": seed,
        "greedy_canonical_verification_required": True,
        "runtime_load_permitted": False,
        "validation_inputs_used": False,
        "held_out_inputs_used": False,
    }
    return {**identity, "run_signature_sha256": _canonical_sha256(identity)}


def build_v62_teacher_preflight(args: argparse.Namespace) -> V62TeacherPreflight:
    """Validate all inventories and hashes without loading Gemma."""

    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("V62 seed must be a nonnegative integer")
    optimizer = _optimizer_settings(args)
    baseline_lock = _resolve(args.baseline_lock)
    _reject_symlink_components(baseline_lock, "baseline lock")
    # Validate this hash-only prerequisite before the training JSONL is opened.
    baseline_authorization = validate_baseline_lock(baseline_lock)
    baseline_lock_sha256 = _sha256_file(baseline_lock)
    preregistration_sha256 = PINNED_V62_PREREGISTRATION_SHA256
    scene_ids = validate_training_scene_ids(args.scene_id)
    train_path, train_sha256, authenticated_rows = _validated_filtered_train_path(
        args.filtered_train_qa,
        args.filtered_train_sha256,
    )
    records, total_rows, selection_sha256 = _load_changed_training_records(
        authenticated_rows, scene_ids=scene_ids
    )

    config, config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, _base_files = checkpoint_fingerprint(base_checkpoint)
    if baseline_authorization.get("v54_checkpoint_sha256") != base_checkpoint_sha256:
        raise ValueError("V62 baseline lock authorizes a different V54 checkpoint")
    source_control_checkpoint = _resolve(args.source_control_checkpoint)
    source_control_checkpoint_sha256 = _control_checkpoint_sha256(
        source_control_checkpoint
    )
    source_control, source_metadata = _load_control_head(
        source_control_checkpoint,
        hidden_size=PROMPT_SHAPE[2],
        device=torch.device("cpu"),
    )
    _source_contract_shape(source_metadata)
    if (
        source_metadata.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or source_metadata.get("base_runtime_config_sha256") != runtime_config_sha256
    ):
        raise ValueError("V62 source controller belongs to a different frozen V54 runtime")

    prefix_cache = _resolve(args.prefix_cache)
    prefixes, _prefix_manifest = load_prefix_cache(
        prefix_cache,
        scene_ids=scene_ids,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    if any(
        prefix.ndim != 3
        or prefix.shape[0] != 1
        or prefix.shape[-1] != PROMPT_SHAPE[2]
        or not torch.isfinite(prefix).all()
        for prefix in prefixes.values()
    ):
        raise ValueError("V62 prefix cache does not contain finite 1536D scene prefixes")
    prefix_cache_manifest_sha256 = _sha256_file(prefix_cache / "manifest.json")

    training_root = question_control_training_artifact_root(config).resolve()
    work_directory = _training_output_path(
        args.work_directory, training_root, "resumable work directory"
    )
    output_artifact = _training_output_path(
        args.output_artifact, training_root, "final teacher artifact"
    )
    if (
        work_directory == output_artifact
        or work_directory.is_relative_to(output_artifact)
        or output_artifact.is_relative_to(work_directory)
    ):
        raise ValueError("V62 work and final artifact paths must be disjoint")

    run_manifest = _build_run_manifest(
        filtered_train_jsonl=train_path,
        filtered_train_jsonl_sha256=train_sha256,
        baseline_lock_sha256=baseline_lock_sha256,
        preregistration_sha256=preregistration_sha256,
        scene_ids=scene_ids,
        total_filtered_rows=total_rows,
        selected_record_count=len(records),
        selection_sha256=selection_sha256,
        runtime_config_sha256=runtime_config_sha256,
        base_checkpoint_sha256=base_checkpoint_sha256,
        source_control_checkpoint_sha256=source_control_checkpoint_sha256,
        prefix_cache_manifest_sha256=prefix_cache_manifest_sha256,
        optimizer=optimizer,
        seed=args.seed,
    )
    return V62TeacherPreflight(
        config=config,
        config_path=config_path,
        runtime_config_sha256=runtime_config_sha256,
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
        source_control_checkpoint=source_control_checkpoint,
        source_control_checkpoint_sha256=source_control_checkpoint_sha256,
        source_control=source_control,
        source_control_metadata=source_metadata,
        filtered_train_jsonl=train_path,
        filtered_train_jsonl_sha256=train_sha256,
        preregistration_sha256=preregistration_sha256,
        baseline_lock=baseline_lock,
        baseline_lock_sha256=baseline_lock_sha256,
        baseline_lock_authorization=baseline_authorization,
        scene_ids=scene_ids,
        records=records,
        total_filtered_rows=total_rows,
        selection_sha256=selection_sha256,
        prefixes=prefixes,
        prefix_cache=prefix_cache,
        prefix_cache_manifest_sha256=prefix_cache_manifest_sha256,
        work_directory=work_directory,
        output_artifact=output_artifact,
        optimizer=optimizer,
        run_manifest=run_manifest,
    )


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _prepare_work_directory(preflight: V62TeacherPreflight) -> None:
    root = preflight.work_directory
    expected_manifest = preflight.run_manifest
    if root.exists():
        if not root.is_dir() or root.is_symlink():
            raise ValueError("V62 work path is not a regular directory")
        inventory = {item.name for item in root.iterdir()}
        if inventory != {"manifest.json", "records"}:
            raise ValueError("V62 resumable work inventory changed")
        observed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if observed != expected_manifest:
            raise ValueError("V62 resumable work manifest does not match this exact run")
        records = root / "records"
        if not records.is_dir() or records.is_symlink():
            raise ValueError("V62 resumable records directory is invalid")
        for partial in records.glob(".record-*.partial-*"):
            if partial.is_dir() and not partial.is_symlink():
                shutil.rmtree(partial)
            else:
                raise ValueError(f"V62 stale partial record is unsafe: {partial}")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.partial-", dir=root.parent))
    try:
        (temporary / "records").mkdir()
        _write_json_new(temporary / "manifest.json", expected_manifest)
        os.rename(temporary, root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _record_name(scene_id: str, question_id: str) -> str:
    digest = _canonical_sha256([scene_id, question_id])
    return f"record_{digest[:24]}"


def _validated_optimization_metrics(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _OPTIMIZATION_FIELDS:
        raise ValueError("V62 optimization metric fields changed")
    result = dict(value)
    for field in ("steps", "attempt_count", "total_forward_steps"):
        _positive_int(result[field], f"optimization {field}")
    for field in (
        "initial_nll",
        "final_nll",
        "minimum_nll",
        "maximum_preclip_gradient_norm",
        "initial_rms",
        "final_rms",
        "learning_rate",
    ):
        item = result[field]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"V62 optimization {field} must be numeric")
        if not math.isfinite(float(item)) or float(item) < 0.0:
            raise ValueError(f"V62 optimization {field} must be finite and nonnegative")
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
        raise ValueError("V62 adaptive attempt learning rates are invalid")
    return result


def _save_completed_record(
    preflight: V62TeacherPreflight,
    record: QARecord,
    prompt: torch.Tensor,
    optimization: Mapping[str, Any],
) -> V62CompletedTeacher:
    numeric = prompt.detach().cpu().float().contiguous()
    if tuple(numeric.shape) != PROMPT_SHAPE or not torch.isfinite(numeric).all():
        raise ValueError("V62 completed prompt must be finite [1,4,1536]")
    metrics = _validated_optimization_metrics(optimization)
    records_root = preflight.work_directory / "records"
    destination = records_root / _record_name(record.scene_id, record.question_id)
    if destination.exists():
        raise FileExistsError(f"V62 completed record already exists: {destination}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.partial-",
            dir=records_root,
        )
    )
    try:
        weights = temporary / "prompt.safetensors"
        save_file({"prompt": numeric}, weights)
        metadata = {
            "schema_version": 1,
            "artifact": _RECORD_ARTIFACT,
            "run_signature_sha256": preflight.run_manifest["run_signature_sha256"],
            "scene_id": record.scene_id,
            "question_id": record.question_id,
            "prompt_sha256": _tensor_sha256(numeric),
            "weights_sha256": _sha256_file(weights),
            "shape": list(PROMPT_SHAPE),
            "rms": float(numeric.square().mean().sqrt()),
            "greedy_canonical_exact": True,
            "optimization": metrics,
        }
        _write_json_new(temporary / "metadata.json", metadata)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _load_completed_record(preflight, record)


def _load_completed_record(
    preflight: V62TeacherPreflight,
    record: QARecord,
) -> V62CompletedTeacher:
    source = preflight.work_directory / "records" / _record_name(
        record.scene_id, record.question_id
    )
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"V62 completed record is unavailable: {source}")
    if {item.name for item in source.iterdir()} != {
        "prompt.safetensors",
        "metadata.json",
    } or any(item.is_symlink() for item in source.iterdir()):
        raise ValueError("V62 completed record inventory changed")
    weights = source / "prompt.safetensors"
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "artifact",
        "run_signature_sha256",
        "scene_id",
        "question_id",
        "prompt_sha256",
        "weights_sha256",
        "shape",
        "rms",
        "greedy_canonical_exact",
        "optimization",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise ValueError("V62 completed record metadata fields changed")
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != 1
        or metadata["artifact"] != _RECORD_ARTIFACT
        or metadata["run_signature_sha256"]
        != preflight.run_manifest["run_signature_sha256"]
        or metadata["scene_id"] != record.scene_id
        or metadata["question_id"] != record.question_id
        or metadata["shape"] != list(PROMPT_SHAPE)
        or metadata["greedy_canonical_exact"] is not True
        or metadata["weights_sha256"] != _sha256_file(weights)
    ):
        raise ValueError("V62 completed record contract changed")
    _digest(metadata["prompt_sha256"], "completed prompt")
    state = load_file(str(weights), device="cpu")
    if set(state) != {"prompt"}:
        raise ValueError("V62 completed record tensor inventory changed")
    prompt = state["prompt"].detach().float().contiguous()
    if (
        tuple(prompt.shape) != PROMPT_SHAPE
        or not torch.isfinite(prompt).all()
        or _tensor_sha256(prompt) != metadata["prompt_sha256"]
        or abs(float(metadata["rms"]) - float(prompt.square().mean().sqrt())) > 1e-7
    ):
        raise ValueError("V62 completed record numeric prompt changed")
    optimization = _validated_optimization_metrics(metadata["optimization"])
    return V62CompletedTeacher(
        scene_id=record.scene_id,
        question_id=record.question_id,
        prompt=prompt,
        optimization=optimization,
    )


def _existing_completed_records(
    preflight: V62TeacherPreflight,
) -> dict[tuple[str, str], V62CompletedTeacher]:
    expected = {
        _record_name(record.scene_id, record.question_id): record
        for record in preflight.records
    }
    records_root = preflight.work_directory / "records"
    observed = {item.name for item in records_root.iterdir()}
    unexpected = sorted(observed - set(expected))
    if unexpected:
        raise ValueError(f"V62 work cache contains unexpected records: {unexpected}")
    return {
        completed.key: completed
        for name, record in expected.items()
        if (records_root / name).exists()
        for completed in (_load_completed_record(preflight, record),)
    }


def _source_initial_prompt(
    control: torch.nn.Module,
    scene_prefix: torch.Tensor,
    pooled_question: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        output = control(scene_prefix.float(), pooled_question.float())
    prompt = output if isinstance(output, torch.Tensor) else getattr(output, "control_tokens", None)
    if not isinstance(prompt, torch.Tensor):
        raise TypeError("V62 source controller did not return numeric control tokens")
    prompt = prompt.detach().float()
    if tuple(prompt.shape) != PROMPT_SHAPE or not torch.isfinite(prompt).all():
        raise ValueError("V62 source controller did not emit finite [1,4,1536] prompts")
    return prompt


def _load_training_runtime(
    preflight: V62TeacherPreflight,
    requested_device: str,
) -> tuple[Any, torch.device, torch.dtype]:
    factory = StaticRuntimePrefixFactory(
        preflight.config,
        preflight.base_checkpoint,
        preflight.scene_ids[0],
    )
    runtime = factory.bootstrap
    cached = preflight.prefixes[preflight.scene_ids[0]].detach().cpu()
    if not torch.equal(runtime.scene_prefix.detach().cpu(), cached):
        raise ValueError("V62 cached prefix differs from the frozen V54 runtime")
    freeze_base_runtime(runtime)
    device = _select_training_device(runtime, requested_device)
    model_dtype = next(runtime.language.model.parameters()).dtype
    return runtime, device, model_dtype


def _final_metadata(
    preflight: V62TeacherPreflight,
    completed: Sequence[V62CompletedTeacher],
    weights_sha256: str,
) -> dict[str, Any]:
    ordered = sorted(completed, key=lambda item: item.key)
    return {
        "schema_version": 1,
        "artifact": _FINAL_ARTIFACT,
        "weights_sha256": weights_sha256,
        "run_signature_sha256": preflight.run_manifest["run_signature_sha256"],
        "filtered_train_jsonl": str(preflight.filtered_train_jsonl),
        "filtered_train_jsonl_sha256": preflight.filtered_train_jsonl_sha256,
        "preregistration_sha256": preflight.preregistration_sha256,
        "baseline_lock_sha256": preflight.baseline_lock_sha256,
        "scene_ids": list(preflight.scene_ids),
        "scene_ids_sha256": _canonical_sha256(list(preflight.scene_ids)),
        "selection_predicate": _SELECTION_PREDICATE,
        "selection_sha256": preflight.selection_sha256,
        "base_checkpoint_sha256": preflight.base_checkpoint_sha256,
        "runtime_config_sha256": preflight.runtime_config_sha256,
        "source_control_checkpoint_sha256": (
            preflight.source_control_checkpoint_sha256
        ),
        "prefix_cache_manifest_sha256": preflight.prefix_cache_manifest_sha256,
        "prompt_shape": list(PROMPT_SHAPE),
        "target_count": len(ordered),
        "greedy_canonical_exact": len(ordered),
        "greedy_canonical_total": len(ordered),
        "runtime_load_permitted": False,
        "environmental_text_inputs": [],
        "validation_inputs_used": False,
        "held_out_inputs_used": False,
        "records": [
            {
                "tensor_key": f"prompt_{index:06d}",
                "scene_id": target.scene_id,
                "question_id": target.question_id,
                "prompt_sha256": _tensor_sha256(target.prompt),
                "shape": list(PROMPT_SHAPE),
                "rms": float(target.prompt.float().square().mean().sqrt()),
                "greedy_canonical_exact": True,
                "optimization": target.optimization,
            }
            for index, target in enumerate(ordered)
        ],
    }


def save_v62_teacher_cache(
    preflight: V62TeacherPreflight,
    completed: Sequence[V62CompletedTeacher],
) -> dict[str, Any]:
    """Atomically publish one strict, numeric, training-only final artifact."""

    ordered = sorted(completed, key=lambda item: item.key)
    expected_keys = {(record.scene_id, record.question_id) for record in preflight.records}
    if {target.key for target in ordered} != expected_keys or len(ordered) != len(expected_keys):
        raise ValueError("V62 cannot publish an incomplete or duplicate teacher inventory")
    destination = preflight.output_artifact
    if destination.exists():
        loaded, metadata = load_v62_teacher_cache(destination)
        if (
            metadata["run_signature_sha256"]
            != preflight.run_manifest["run_signature_sha256"]
            or set(loaded) != expected_keys
        ):
            raise FileExistsError("Existing V62 final artifact belongs to another run")
        return metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        state = {
            f"prompt_{index:06d}": target.prompt.detach().cpu().float().contiguous()
            for index, target in enumerate(ordered)
        }
        weights = temporary / "teachers.safetensors"
        save_file(state, weights)
        metadata = _final_metadata(preflight, ordered, _sha256_file(weights))
        _write_json_new(temporary / "metadata.json", metadata)
        loaded, validated = load_v62_teacher_cache(temporary)
        if validated != metadata or set(loaded) != expected_keys or any(
            not torch.equal(loaded[target.key], target.prompt.detach().cpu().float())
            for target in ordered
        ):
            raise RuntimeError("V62 final teacher cache failed exact reload")
        os.rename(temporary, destination)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_v62_teacher_cache(
    source: str | Path,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]:
    """Strict training-side loader; chat/runtime code must never import this."""

    root = _resolve(source)
    _reject_symlink_components(root, "final teacher cache")
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"V62 teacher cache is unavailable: {root}")
    if {item.name for item in root.iterdir()} != {
        "teachers.safetensors",
        "metadata.json",
    } or any(item.is_symlink() for item in root.iterdir()):
        raise ValueError("V62 final teacher cache inventory changed")
    weights = root / "teachers.safetensors"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "artifact",
        "weights_sha256",
        "run_signature_sha256",
        "filtered_train_jsonl",
        "filtered_train_jsonl_sha256",
        "preregistration_sha256",
        "baseline_lock_sha256",
        "scene_ids",
        "scene_ids_sha256",
        "selection_predicate",
        "selection_sha256",
        "base_checkpoint_sha256",
        "runtime_config_sha256",
        "source_control_checkpoint_sha256",
        "prefix_cache_manifest_sha256",
        "prompt_shape",
        "target_count",
        "greedy_canonical_exact",
        "greedy_canonical_total",
        "runtime_load_permitted",
        "environmental_text_inputs",
        "validation_inputs_used",
        "held_out_inputs_used",
        "records",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise ValueError("V62 final teacher metadata fields changed")
    records = metadata.get("records")
    if (
        type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != 1
        or metadata.get("artifact") != _FINAL_ARTIFACT
        or metadata.get("weights_sha256") != _sha256_file(weights)
        or metadata.get("selection_predicate") != _SELECTION_PREDICATE
        or metadata.get("prompt_shape") != list(PROMPT_SHAPE)
        or metadata.get("runtime_load_permitted") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("validation_inputs_used") is not False
        or metadata.get("held_out_inputs_used") is not False
        or not isinstance(records, list)
        or metadata.get("target_count") != len(records)
        or metadata.get("greedy_canonical_exact") != len(records)
        or metadata.get("greedy_canonical_total") != len(records)
    ):
        raise ValueError("V62 final teacher cache contract changed")
    for field in (
        "weights_sha256",
        "run_signature_sha256",
        "filtered_train_jsonl_sha256",
        "preregistration_sha256",
        "baseline_lock_sha256",
        "scene_ids_sha256",
        "selection_sha256",
        "base_checkpoint_sha256",
        "runtime_config_sha256",
        "source_control_checkpoint_sha256",
        "prefix_cache_manifest_sha256",
    ):
        _digest(metadata[field], field)
    scene_ids = metadata.get("scene_ids")
    if (
        not isinstance(scene_ids, list)
        or validate_training_scene_ids(scene_ids) != tuple(scene_ids)
        or _canonical_sha256(scene_ids) != metadata["scene_ids_sha256"]
    ):
        raise ValueError("V62 final teacher scene inventory changed")
    state = load_file(str(weights), device="cpu")
    if set(state) != {record.get("tensor_key") for record in records}:
        raise ValueError("V62 final teacher tensor inventory changed")
    result: dict[tuple[str, str], torch.Tensor] = {}
    record_fields = {
        "tensor_key",
        "scene_id",
        "question_id",
        "prompt_sha256",
        "shape",
        "rms",
        "greedy_canonical_exact",
        "optimization",
    }
    for record in records:
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError("V62 final teacher record fields changed")
        scene_id = record.get("scene_id")
        question_id = record.get("question_id")
        if (
            not isinstance(scene_id, str)
            or scene_id not in scene_ids
            or not isinstance(question_id, str)
            or not question_id
            or record.get("shape") != list(PROMPT_SHAPE)
            or record.get("greedy_canonical_exact") is not True
        ):
            raise ValueError("V62 final teacher opaque record identity changed")
        tensor_key = record.get("tensor_key")
        prompt = state[tensor_key].detach().float().contiguous()
        key = (scene_id, question_id)
        if (
            key in result
            or tuple(prompt.shape) != PROMPT_SHAPE
            or not torch.isfinite(prompt).all()
            or _tensor_sha256(prompt) != record.get("prompt_sha256")
            or abs(float(record.get("rms")) - float(prompt.square().mean().sqrt()))
            > 1e-7
        ):
            raise ValueError("V62 final teacher prompt changed")
        _validated_optimization_metrics(record.get("optimization"))
        result[key] = prompt
    return result, dict(metadata)


def generate_v62_teacher_cache(
    args: argparse.Namespace,
    *,
    runtime_provider: Callable[
        [V62TeacherPreflight, str], tuple[Any, torch.device, torch.dtype]
    ]
    | None = None,
    question_embedder: Callable[[Any, str], torch.Tensor] | None = None,
    optimizer_fn: Callable[..., tuple[torch.Tensor, dict[str, Any]]] | None = None,
    fallback_optimizer_fn: Callable[..., tuple[torch.Tensor, dict[str, Any]]]
    | None = None,
    generator_fn: Callable[..., str] | None = None,
    disable_checkpointing_fn: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Generate or resume V62 teachers; dependency hooks support offline tests."""

    preflight = build_v62_teacher_preflight(args)
    inventory = {
        "passed": True,
        "mode": "dry_run_inventory" if args.dry_run_inventory else "generate",
        "gemma_loaded": False,
        "filtered_train_jsonl": str(preflight.filtered_train_jsonl),
        "filtered_train_jsonl_sha256": preflight.filtered_train_jsonl_sha256,
        "preregistration_sha256": preflight.preregistration_sha256,
        "baseline_lock_sha256": preflight.baseline_lock_sha256,
        "scene_ids": list(preflight.scene_ids),
        "total_filtered_rows": preflight.total_filtered_rows,
        "selected_record_count": len(preflight.records),
        "selection_sha256": preflight.selection_sha256,
        "base_checkpoint_sha256": preflight.base_checkpoint_sha256,
        "runtime_config_sha256": preflight.runtime_config_sha256,
        "source_control_checkpoint_sha256": (
            preflight.source_control_checkpoint_sha256
        ),
        "source_control_architecture": preflight.source_control_metadata["architecture"],
        "prefix_cache_manifest_sha256": preflight.prefix_cache_manifest_sha256,
        "prompt_shape": list(PROMPT_SHAPE),
        "run_signature_sha256": preflight.run_manifest["run_signature_sha256"],
        "validation_inputs_used": False,
        "held_out_inputs_used": False,
        "runtime_load_permitted": False,
    }
    if args.dry_run_inventory:
        return inventory

    _prepare_work_directory(preflight)
    completed = _existing_completed_records(preflight)
    if preflight.output_artifact.exists():
        metadata = save_v62_teacher_cache(preflight, tuple(completed.values()))
        return {
            **inventory,
            "mode": "already_complete",
            "completed_record_count": len(completed),
            "resumed_record_count": len(completed),
            "output_artifact": str(preflight.output_artifact),
            "weights_sha256": metadata["weights_sha256"],
        }

    provider = runtime_provider or _load_training_runtime
    embed_question = question_embedder or _pooled_question_embedding
    optimize = optimizer_fn or _optimize_teacher_prompt_adaptive
    # A few strongly prior-biased answers need a much smaller step size than
    # the normal adaptive grid.  Keep this expensive path verification-driven:
    # it runs only after the standard prompt fails an actual greedy decode.
    # Dependency-injected tests retain their old fail-closed behavior unless
    # they explicitly provide a fallback optimizer.
    fallback_optimize = fallback_optimizer_fn
    if fallback_optimize is None and optimizer_fn is None and generator_fn is None:
        fallback_optimize = _optimize_teacher_prompt
    generate = generator_fn or _generate_with_control
    disable_checkpointing = disable_checkpointing_fn or _disable_decoder_checkpointing
    runtime, device, model_dtype = provider(preflight, args.device)
    source_control = preflight.source_control.to(device=device, dtype=torch.float32).eval()
    prefixes = {
        scene_id: prefix.to(device=device, dtype=model_dtype)
        for scene_id, prefix in preflight.prefixes.items()
    }
    resumed_count = len(completed)
    runtime.language.enable_decoder_gradient_checkpointing()
    checkpointing_enabled = True
    try:
        for ordinal, record in enumerate(preflight.records, start=1):
            key = (record.scene_id, record.question_id)
            if key in completed:
                continue
            record_seed = int(
                _canonical_sha256([args.seed, record.scene_id, record.question_id])[:16],
                16,
            )
            random.seed(record_seed)
            torch.manual_seed(record_seed % (2**63 - 1))
            scene_prefix = prefixes[record.scene_id]
            pooled_question = embed_question(runtime, record.question).to(device=device)
            initial_prompt = _source_initial_prompt(
                source_control,
                scene_prefix,
                pooled_question,
            )
            prompt, metrics = optimize(
                runtime=runtime,
                scene_prefix=scene_prefix,
                record=record,
                initial_prompt=initial_prompt,
                learning_rate=preflight.optimizer["learning_rate"],
                min_steps=preflight.optimizer["minimum_steps"],
                max_steps=preflight.optimizer["maximum_steps"],
                nll_threshold=preflight.optimizer["nll_threshold"],
                gradient_clip_norm=preflight.optimizer["gradient_clip_norm"],
            )
            disable_checkpointing(runtime.language)
            checkpointing_enabled = False
            generated = generate(
                runtime=runtime,
                scene_prefix=scene_prefix,
                question=record.question,
                control_tokens=prompt,
            )
            if (
                not exact_normalized_match(generated, record.answer)
                and fallback_optimize is not None
            ):
                runtime.language.enable_decoder_gradient_checkpointing()
                checkpointing_enabled = True
                fallback_rate = float(preflight.optimizer["learning_rate"]) / 300.0
                fallback_steps = max(
                    100, int(preflight.optimizer["maximum_steps"]) * 5
                )
                prompt, fallback_metrics = fallback_optimize(
                    runtime=runtime,
                    scene_prefix=scene_prefix,
                    record=record,
                    initial_prompt=initial_prompt,
                    learning_rate=fallback_rate,
                    min_steps=max(10, int(preflight.optimizer["minimum_steps"])),
                    max_steps=fallback_steps,
                    nll_threshold=min(
                        1e-4, float(preflight.optimizer["nll_threshold"])
                    ),
                    gradient_clip_norm=preflight.optimizer[
                        "gradient_clip_norm"
                    ],
                )
                metrics = {
                    **fallback_metrics,
                    "attempt_count": 1,
                    "attempt_learning_rates": [fallback_rate],
                    "total_forward_steps": fallback_metrics["steps"],
                }
                disable_checkpointing(runtime.language)
                checkpointing_enabled = False
                generated = generate(
                    runtime=runtime,
                    scene_prefix=scene_prefix,
                    question=record.question,
                    control_tokens=prompt,
                )
            if not exact_normalized_match(generated, record.answer):
                raise RuntimeError(
                    "V62 optimized soft prompt failed greedy canonical verification: "
                    f"scene_id={record.scene_id} question_id={record.question_id}; "
                    f"generated={generated!r}"
                )
            completed[key] = _save_completed_record(
                preflight,
                record,
                prompt,
                metrics,
            )
            print(
                json.dumps(
                    {
                        "phase": "v62_teacher_record_complete",
                        "completed": len(completed),
                        "total": len(preflight.records),
                        "ordinal": ordinal,
                        "scene_id": record.scene_id,
                        "question_id": record.question_id,
                        "final_nll": metrics["final_nll"],
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
            if len(completed) < len(preflight.records):
                runtime.language.enable_decoder_gradient_checkpointing()
                checkpointing_enabled = True
    finally:
        if checkpointing_enabled:
            disable_checkpointing(runtime.language)

    metadata = save_v62_teacher_cache(preflight, tuple(completed.values()))
    return {
        **inventory,
        "mode": "complete",
        "gemma_loaded": True,
        "completed_record_count": len(completed),
        "resumed_record_count": resumed_count,
        "new_record_count": len(completed) - resumed_count,
        "greedy_canonical_exact": len(completed),
        "greedy_canonical_total": len(completed),
        "output_artifact": str(preflight.output_artifact),
        "weights_sha256": metadata["weights_sha256"],
        "metadata_sha256": _sha256_file(preflight.output_artifact / "metadata.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_filtered_training_data_argument(parser)
    parser.add_argument("--filtered-train-sha256", required=True)
    add_baseline_lock_authorization_argument(parser)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--source-control-checkpoint", required=True)
    parser.add_argument("--work-directory", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=62062)
    parser.add_argument("--teacher-learning-rate", type=float, default=0.03)
    parser.add_argument("--teacher-min-steps", type=int, default=5)
    parser.add_argument("--teacher-max-steps", type=int, default=20)
    parser.add_argument("--teacher-nll-threshold", type=float, default=1e-3)
    parser.add_argument("--teacher-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--dry-run-inventory", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = generate_v62_teacher_cache(args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROMPT_SHAPE",
    "V62CompletedTeacher",
    "V62TeacherPreflight",
    "build_v62_teacher_preflight",
    "generate_v62_teacher_cache",
    "load_v62_teacher_cache",
    "save_v62_teacher_cache",
]
