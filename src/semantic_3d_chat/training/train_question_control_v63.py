"""Pair-disjoint expanded continuous-value distillation for V62/V63.

This trainer is deliberately training-only.  It authenticates the hash-only
V62 baseline lock before opening any training input, loads the exact 12-pair
filtered training boundary and the exact 80 numeric Gemma prompt teachers, and
fits a V3-compatible full-scene value controller.  Gemma is frozen and used
only to embed training questions; generation and decoder backpropagation are
not part of this command.

V5 routing normalizes pooled question embeddings with an inherited V3 layer.
To make later route-only composition exact, every fold copies and freezes the
``question_norm`` tensors from the authenticated V60 source used by V62.  All
other fold value parameters remain freshly initialized.

Before an all-training fit is allowed, every counterfactual pair is held out in
turn.  Each fold constructs its output basis from the other eleven pairs, fits
from scratch, and evaluates numeric prompt reconstruction on the unseen pair.
No checkpoint or report is published unless the immutable aggregate CV gates
and final-fit gates pass.
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
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    PAIR_INVENTORY,
    PINNED_V62_PREREGISTRATION_SHA256,
    TRAIN_PAIR_IDS,
    add_baseline_lock_authorization_argument,
    add_filtered_training_data_argument,
    load_filtered_training_qa,
    validate_baseline_lock,
)
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
    teacher_output_basis,
)
from semantic_3d_chat.training.question_control_v3_checkpoint import (
    save_v3_control_checkpoint,
)
from semantic_3d_chat.training.soft_prompt_teacher_v62 import (
    PROMPT_SHAPE,
    load_v62_teacher_cache,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    _log_event,
    _resolve,
    _safe_output_path,
    _select_training_device,
    _sha256_file,
    _write_json,
    freeze_base_runtime,
    load_prefix_cache,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _pooled_question_embedding,
)

_EXPECTED_RECORDS: Final[int] = 576
_EXPECTED_SCENES: Final[int] = 24
_EXPECTED_PAIRS: Final[int] = 12
_EXPECTED_CHANGED_SIDES: Final[int] = 80
_EXPECTED_CHANGED_UNITS: Final[int] = 40
_EXPECTED_CONTROL_TOKENS: Final[int] = 4
_EXPECTED_HIDDEN_SIZE: Final[int] = 1536


@dataclass(frozen=True)
class V63CVThresholds:
    """Immutable pre-run gates over pair-held-out numeric reconstruction."""

    mean_prompt_cosine: float = 0.90
    minimum_prompt_cosine: float = 0.60
    minimum_fold_mean_prompt_cosine: float = 0.82
    prompt_root_mean_square_error: float = 0.030
    mean_prompt_rms_absolute_error: float = 0.015
    mean_pair_delta_cosine: float = 0.50
    minimum_positive_pair_delta_fraction: float = 0.75


@dataclass(frozen=True)
class V63FinalThresholds:
    """All-training reconstruction gates checked before publication."""

    mean_prompt_cosine: float = 0.97
    minimum_prompt_cosine: float = 0.80
    prompt_root_mean_square_error: float = 0.020
    mean_prompt_rms_absolute_error: float = 0.010
    mean_pair_delta_cosine: float = 0.75


PREREGISTERED_CV_THRESHOLDS: Final[V63CVThresholds] = V63CVThresholds()
PREREGISTERED_FINAL_THRESHOLDS: Final[V63FinalThresholds] = V63FinalThresholds()


@dataclass(frozen=True)
class V63Row:
    scene_id: str
    question_id: str
    question: str
    pair_id: str
    question_key: str
    route_label: bool
    answer: str = ""
    answer_type: str = ""
    answer_items: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.question_id


@dataclass(frozen=True)
class V63Preflight:
    baseline_lock_sha256: str
    baseline_authorization: dict[str, Any]
    filtered_train_sha256: str
    rows: tuple[V63Row, ...]
    scene_ids: tuple[str, ...]
    teacher_targets: dict[tuple[str, str], torch.Tensor]
    teacher_metadata: dict[str, Any]
    teacher_metadata_sha256: str
    teacher_weights_sha256: str
    config: dict[str, Any]
    config_path: Path
    runtime_config_sha256: str
    base_checkpoint: Path
    base_checkpoint_sha256: str
    base_checkpoint_files: list[dict[str, Any]]
    source_v60_checkpoint: Path
    source_v60_checkpoint_sha256: str
    source_v60_metadata: dict[str, Any]
    source_v60_question_norm_state: dict[str, torch.Tensor]
    source_v60_question_norm_sha256: str
    prefixes: dict[str, torch.Tensor]
    prefix_manifest: dict[str, Any]
    prefix_manifest_sha256: str
    output_checkpoint: Path
    training_report: Path


@dataclass(frozen=True)
class ReconstructionMeasurements:
    prompt_cosines: tuple[float, ...]
    prompt_rms_absolute_errors: tuple[float, ...]
    squared_error_sum: float
    element_count: int
    pair_delta_cosines: tuple[float, ...]

    def summary(self) -> dict[str, float | int]:
        if (
            not self.prompt_cosines
            or not self.prompt_rms_absolute_errors
            or self.element_count < 1
            or not self.pair_delta_cosines
        ):
            raise ValueError("V63 reconstruction measurements are incomplete")
        return {
            "teacher_side_count": len(self.prompt_cosines) // _EXPECTED_CONTROL_TOKENS,
            "prompt_token_count": len(self.prompt_cosines),
            "changed_pair_unit_count": len(self.pair_delta_cosines),
            "mean_prompt_cosine": _mean(self.prompt_cosines),
            "minimum_prompt_cosine": min(self.prompt_cosines),
            "prompt_root_mean_square_error": math.sqrt(
                self.squared_error_sum / self.element_count
            ),
            "mean_prompt_rms_absolute_error": _mean(
                self.prompt_rms_absolute_errors
            ),
            "maximum_prompt_rms_absolute_error": max(
                self.prompt_rms_absolute_errors
            ),
            "mean_pair_delta_cosine": _mean(self.pair_delta_cosines),
            "minimum_pair_delta_cosine": min(self.pair_delta_cosines),
            "positive_pair_delta_count": sum(
                value > 0.0 for value in self.pair_delta_cosines
            ),
            "positive_pair_delta_fraction": _mean(
                [float(value > 0.0) for value in self.pair_delta_cosines]
            ),
        }


@dataclass(frozen=True)
class FitResult:
    control: TeacherBasisFullSceneQuestionControlV3
    signatures: dict[str, torch.Tensor]
    basis_reconstruction: dict[str, float]
    elapsed_seconds: float
    optimizer_steps: int
    maximum_preclip_gradient_norm: float
    final_route_loss: float
    question_norm_sha256: str
    question_norm_frozen: bool


class V63OfflineGateError(RuntimeError):
    """Raised without publishing model outputs when offline evidence is inadequate."""

    def __init__(
        self,
        message: str,
        diagnostics: Mapping[str, Any],
        *,
        failure_stage: str,
        provenance: Mapping[str, Any],
        scope: Mapping[str, Any],
        gemma_audit: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)
        self.failure_stage = failure_stage
        self.provenance = dict(provenance)
        self.scope = dict(scope)
        self.gemma_audit = dict(gemma_audit)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty V63 metric population")
    return sum(values) / len(values)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _question_norm_state(
    control: TeacherBasisFullSceneQuestionControlV3,
) -> dict[str, torch.Tensor]:
    state = {
        name.removeprefix("question_norm."): value.detach().cpu().float().clone()
        for name, value in control.state_dict().items()
        if name.startswith("question_norm.")
    }
    if set(state) != {"weight", "bias"} or any(
        tuple(value.shape) != (_EXPECTED_HIDDEN_SIZE,)
        or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("V63 source V60 question_norm state changed")
    return state


def _train_specs() -> tuple[Any, ...]:
    specs = {spec.pair_id: spec for spec in PAIR_INVENTORY}
    return tuple(specs[pair_id] for pair_id in TRAIN_PAIR_IDS)


def training_scene_ids() -> tuple[str, ...]:
    return tuple(sorted(scene_id for spec in _train_specs() for scene_id in spec.scene_ids))


def _validate_hyperparameters(args: argparse.Namespace) -> None:
    integer_fields = (
        "seed",
        "basis_rank",
        "moment_count",
        "interaction_dim",
        "trunk_dim",
        "epochs",
        "batch_size",
        "changed_repeats",
        "log_every",
    )
    for field in integer_fields:
        value = getattr(args, field)
        minimum = 0 if field == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"V63 {field} must be an integer >= {minimum}")
    float_fields = (
        "maximum_control_rms",
        "initial_control_rms",
        "gate_threshold",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "coefficient_weight",
        "log_rms_weight",
        "reconstruction_weight",
        "relative_mse_weight",
        "pair_delta_weight",
        "route_weight",
    )
    for field in float_fields:
        value = getattr(args, field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"V63 {field} must be finite and nonnegative")
    if not 0.0 < args.initial_control_rms < args.maximum_control_rms <= 1.0:
        raise ValueError("V63 RMS bounds require 0 < initial < maximum <= 1")
    if not 0.0 < args.gate_threshold < 1.0:
        raise ValueError("V63 gate threshold must lie in (0,1)")
    if not any(
        getattr(args, field) > 0.0
        for field in (
            "coefficient_weight",
            "log_rms_weight",
            "reconstruction_weight",
            "relative_mse_weight",
        )
    ):
        raise ValueError("V63 enables no continuous-value loss")


def _validated_rows(raw_rows: Sequence[Mapping[str, Any]]) -> tuple[V63Row, ...]:
    if len(raw_rows) != _EXPECTED_RECORDS:
        raise ValueError(
            f"V63 requires exactly {_EXPECTED_RECORDS} filtered training rows"
        )
    expected_pairs = set(TRAIN_PAIR_IDS)
    expected_scenes = set(training_scene_ids())
    rows: list[V63Row] = []
    opaque_keys: set[tuple[str, str]] = set()
    pair_counts: defaultdict[str, int] = defaultdict(int)
    pair_scenes: defaultdict[str, set[str]] = defaultdict(set)
    for raw in raw_rows:
        row = V63Row(
            scene_id=str(raw["scene_id"]),
            question_id=str(raw["question_id"]),
            question=str(raw["question"]),
            pair_id=str(raw["counterfactual_pair_id"]),
            question_key=str(raw["counterfactual_question_key"]),
            route_label=bool(raw["counterfactual_expected_change"]),
            answer=str(raw["answer"]),
            answer_type=str(raw.get("answer_type", "exact")),
            answer_items=tuple(
                str(item)
                for item in (
                    raw.get("answer_items")
                    if isinstance(raw.get("answer_items"), list)
                    else ()
                )
            ),
        )
        if row.key in opaque_keys:
            raise ValueError(f"V63 duplicate opaque row key: {row.key}")
        if row.pair_id not in expected_pairs or row.scene_id not in expected_scenes:
            raise ValueError("V63 filtered training inventory escaped its pair boundary")
        if not row.answer.strip():
            raise ValueError("V63 filtered training row has an empty canonical answer")
        if not row.answer_type.strip():
            raise ValueError("V63 filtered training row has an empty answer type")
        if raw.get("answer_items") is not None and (
            not isinstance(raw.get("answer_items"), list)
            or not row.answer_items
            or any(not item.strip() for item in row.answer_items)
        ):
            raise ValueError("V63 filtered training row has invalid answer items")
        opaque_keys.add(row.key)
        pair_counts[row.pair_id] += 1
        pair_scenes[row.pair_id].add(row.scene_id)
        rows.append(row)
    if set(pair_counts) != expected_pairs or any(value != 48 for value in pair_counts.values()):
        raise ValueError("Every V63 counterfactual pair must contain exactly 48 sides")
    spec_scenes = {spec.pair_id: set(spec.scene_ids) for spec in _train_specs()}
    if any(pair_scenes[pair_id] != spec_scenes[pair_id] for pair_id in expected_pairs):
        raise ValueError("V63 pair-to-scene binding changed")
    changed = [row for row in rows if row.route_label]
    if len(changed) != _EXPECTED_CHANGED_SIDES:
        raise ValueError("V63 requires exactly 80 changed-side teachers")
    units: defaultdict[tuple[str, str], list[V63Row]] = defaultdict(list)
    for row in changed:
        units[(row.pair_id, row.question_key)].append(row)
    if len(units) != _EXPECTED_CHANGED_UNITS:
        raise ValueError("V63 changed paired-unit inventory changed")
    for key, sides in units.items():
        if len(sides) != 2 or len({side.scene_id for side in sides}) != 2:
            raise ValueError(f"V63 changed unit is not a two-scene pair: {key}")
        if len({side.question.encode("utf-8") for side in sides}) != 1:
            raise ValueError(f"V63 paired questions are not byte-identical: {key}")
    return tuple(sorted(rows, key=lambda row: (row.pair_id, row.scene_id, row.question_id)))


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_output_isolation(
    *,
    output_checkpoint: str | Path,
    training_report: str | Path,
    inputs: Sequence[Path],
) -> tuple[Path, Path]:
    checkpoint = _resolve(output_checkpoint)
    report = _resolve(training_report)
    if _paths_overlap(checkpoint, report):
        raise ValueError("V63 checkpoint and report destinations must be disjoint")
    if checkpoint.exists() or report.exists():
        raise FileExistsError("V63 output overwrite is forbidden")
    for source in inputs:
        if _paths_overlap(checkpoint, source):
            raise ValueError("V63 checkpoint destination overlaps an input")
        if _paths_overlap(report, source):
            raise ValueError("V63 report destination overlaps an input")
    return checkpoint, report


def _validate_diagnostics_destination(args: argparse.Namespace) -> Path | None:
    """Reserve a disjoint create-once destination before expensive V63 work."""

    raw_output = getattr(args, "diagnostics_output", None)
    if raw_output is None:
        return None
    output = _resolve(raw_output)
    protected = (
        _resolve(args.output_checkpoint),
        _resolve(args.training_report),
        _resolve(args.baseline_lock),
        _resolve(args.filtered_train_qa),
        _resolve(args.teacher_cache),
        _resolve(args.prefix_cache),
        _resolve(args.base_runtime_config),
        _resolve(args.base_checkpoint),
        _resolve(args.source_v60_checkpoint),
    )
    if any(_paths_overlap(output, path) for path in protected):
        raise ValueError("V63 failure diagnostics destination overlaps an input/output")
    return _safe_output_path(output, "V63 failure diagnostics")


def _failure_provenance(
    preflight: V63Preflight,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return deterministic hashes identifying the exact training-only run."""

    return {
        "authorization": {
            "baseline_lock_sha256": preflight.baseline_lock_sha256,
            "preregistration_sha256": PINNED_V62_PREREGISTRATION_SHA256,
            "baseline_validated_before_training_data": True,
        },
        "base": {
            "checkpoint_sha256": preflight.base_checkpoint_sha256,
            "runtime_config_effective_sha256": preflight.runtime_config_sha256,
            "runtime_config_file_sha256": _sha256_file(preflight.config_path),
        },
        "source_v60": {
            "checkpoint_sha256": preflight.source_v60_checkpoint_sha256,
            "weights_sha256": preflight.source_v60_metadata["weights_sha256"],
            "runtime_metadata_sha256": _sha256_file(
                preflight.source_v60_checkpoint / "runtime_metadata.json"
            ),
            "question_norm_sha256": preflight.source_v60_question_norm_sha256,
            "question_norm_copied_tensor_exact": True,
            "question_norm_frozen_in_every_fit": True,
        },
        "inputs": {
            "filtered_training_qa_sha256": preflight.filtered_train_sha256,
            "training_scene_ids": list(preflight.scene_ids),
            "training_record_count": len(preflight.rows),
            "counterfactual_pair_count": len(TRAIN_PAIR_IDS),
            "changed_teacher_side_count": len(preflight.teacher_targets),
            "changed_paired_unit_count": len(_changed_units(preflight.rows)),
            "teacher_metadata_sha256": preflight.teacher_metadata_sha256,
            "teacher_weights_sha256": preflight.teacher_weights_sha256,
            "teacher_selection_sha256": preflight.teacher_metadata[
                "selection_sha256"
            ],
            "prefix_cache_manifest_sha256": preflight.prefix_manifest_sha256,
        },
        "run_configuration": {
            "seed": args.seed,
            "requested_device": args.device,
            "basis_rank": args.basis_rank,
            "moment_count": args.moment_count,
            "interaction_dim": args.interaction_dim,
            "trunk_dim": args.trunk_dim,
            "maximum_control_rms": args.maximum_control_rms,
            "initial_control_rms": args.initial_control_rms,
            "gate_threshold": args.gate_threshold,
            "epochs_per_fold_and_final": args.epochs,
            "batch_size": args.batch_size,
            "changed_repeats": args.changed_repeats,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "loss_weights": {
                "coefficient": args.coefficient_weight,
                "log_rms": args.log_rms_weight,
                "reconstruction_cosine": args.reconstruction_weight,
                "relative_mse": args.relative_mse_weight,
                "pair_delta": args.pair_delta_weight,
                "provisional_route": args.route_weight,
            },
            "controller_training_device": "cpu",
        },
    }


def _failure_scope() -> dict[str, bool]:
    return {
        "training_only": True,
        "base_scene_stack_frozen": True,
        "gemma_backward_used": False,
        "gemma_generation_used": False,
        "numeric_teacher_cache_only": True,
        "teacher_cache_runtime_access": False,
        "complete_scene_prefix_retained": True,
        "question_dependent_scene_retrieval": False,
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
        "report_contains_question_or_answer_text": False,
        "checkpoint_published": False,
        "training_report_published": False,
    }


def _failure_report(exc: V63OfflineGateError) -> dict[str, Any]:
    """Build the strict, text-free, deterministic V63 failure artifact."""

    report = {
        "schema_version": 1,
        "artifact": "v63_pair_disjoint_expanded_value_distillation_failure",
        "passed": False,
        "promotion_eligible": False,
        "failure_stage": exc.failure_stage,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "diagnostics": exc.diagnostics,
        "provenance": exc.provenance,
        "gemma_audit": exc.gemma_audit,
        "publication": {
            "failure_diagnostics_create_once": True,
            "checkpoint_published": False,
            "training_report_published": False,
        },
        "scope": exc.scope,
    }
    # Validate the complete payload before any destination is opened.
    json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return report


def _publish_failure_report_create_once(
    path: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Atomically expose a complete JSON file and never replace a prior result."""

    output = _safe_output_path(path, "V63 failure diagnostics")
    payload = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def build_v63_preflight(args: argparse.Namespace) -> V63Preflight:
    """Authenticate the locked boundary before Gemma or optimization is used."""

    _validate_hyperparameters(args)

    # This is intentionally the first file load.  Future refactors must retain
    # this ordering: no training/cache/config input is authorized beforehand.
    baseline_path = _resolve(args.baseline_lock)
    baseline_authorization = validate_baseline_lock(baseline_path)
    baseline_lock_sha256 = _sha256_file(baseline_path)

    filtered_path = _resolve(args.filtered_train_qa)
    raw_rows = load_filtered_training_qa(filtered_path)
    filtered_train_sha256 = _sha256_file(filtered_path)
    rows = _validated_rows(raw_rows)
    scene_ids = training_scene_ids()

    config, config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, base_checkpoint_files = checkpoint_fingerprint(
        base_checkpoint
    )
    if baseline_authorization.get("v54_checkpoint_sha256") != base_checkpoint_sha256:
        raise ValueError("V63 baseline lock authorizes a different frozen V54 checkpoint")

    source_v60_checkpoint = _resolve(args.source_v60_checkpoint)
    source_v60_checkpoint_sha256 = _control_checkpoint_sha256(
        source_v60_checkpoint
    )
    source_v60_raw, source_v60_metadata = _load_control_head(
        source_v60_checkpoint,
        hidden_size=_EXPECTED_HIDDEN_SIZE,
        device=torch.device("cpu"),
    )
    if type(source_v60_raw) is not TeacherBasisFullSceneQuestionControlV3:
        raise TypeError("V63 source V60 must be the exact V3 architecture")
    source_v60 = source_v60_raw
    if (
        source_v60_metadata.get("base_checkpoint_sha256")
        != base_checkpoint_sha256
        or source_v60_metadata.get("base_runtime_config_sha256")
        != runtime_config_sha256
        or source_v60_metadata.get("hidden_size") != _EXPECTED_HIDDEN_SIZE
        or source_v60_metadata.get("control_tokens") != _EXPECTED_CONTROL_TOKENS
        or source_v60_metadata.get("expected_environment_latents") != 256
        or source_v60_metadata.get("moment_count") != args.moment_count
        or source_v60_metadata.get("gate_threshold") != args.gate_threshold
    ):
        raise ValueError("V63 source V60 architecture/base/runtime is incompatible")
    source_v60_question_norm_state = _question_norm_state(source_v60)
    source_v60_question_norm_sha256 = _tensor_state_sha256(
        source_v60_question_norm_state
    )

    teacher_root = _resolve(args.teacher_cache)
    targets, teacher_metadata = load_v62_teacher_cache(teacher_root)
    expected_teacher_keys = {row.key for row in rows if row.route_label}
    if set(targets) != expected_teacher_keys or len(targets) != _EXPECTED_CHANGED_SIDES:
        raise ValueError("V63 numeric teacher inventory differs from all 80 changed sides")
    teacher_metadata_path = teacher_root / "metadata.json"
    teacher_weights_path = teacher_root / "teachers.safetensors"
    if (
        teacher_metadata.get("filtered_train_jsonl_sha256")
        != filtered_train_sha256
        or teacher_metadata.get("preregistration_sha256")
        != PINNED_V62_PREREGISTRATION_SHA256
        or teacher_metadata.get("baseline_lock_sha256") != baseline_lock_sha256
        or teacher_metadata.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or teacher_metadata.get("runtime_config_sha256") != runtime_config_sha256
        or teacher_metadata.get("scene_ids") != list(scene_ids)
        or teacher_metadata.get("target_count") != _EXPECTED_CHANGED_SIDES
        or teacher_metadata.get("greedy_canonical_exact") != _EXPECTED_CHANGED_SIDES
        or teacher_metadata.get("runtime_load_permitted") is not False
        or teacher_metadata.get("validation_inputs_used") is not False
        or teacher_metadata.get("held_out_inputs_used") is not False
    ):
        raise ValueError("V63 numeric teacher provenance is not the exact locked V62 run")
    if any(
        tuple(target.shape) != PROMPT_SHAPE or not torch.isfinite(target).all()
        for target in targets.values()
    ):
        raise ValueError("V63 teachers must remain finite native [1,4,1536] prompts")
    maximum_teacher_rms = max(
        float(target.float().square().mean(dim=-1).sqrt().max())
        for target in targets.values()
    )
    if maximum_teacher_rms >= args.maximum_control_rms:
        raise ValueError(
            "V63 maximum control RMS must exceed every authorized teacher RMS"
        )

    prefix_root = _resolve(args.prefix_cache)
    prefixes, prefix_manifest = load_prefix_cache(
        prefix_root,
        scene_ids=scene_ids,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    prefix_manifest_sha256 = _sha256_file(prefix_root / "manifest.json")
    if (
        teacher_metadata.get("prefix_cache_manifest_sha256")
        != prefix_manifest_sha256
    ):
        raise ValueError("V63 teacher cache and full-scene prefix cache are not bound")
    if any(
        tuple(prefix.shape) != (1, 258, _EXPECTED_HIDDEN_SIZE)
        or not torch.isfinite(prefix).all()
        for prefix in prefixes.values()
    ):
        raise ValueError("V63 requires 256 complete 1536D scene latents plus boundaries")

    output_checkpoint, training_report = _validate_output_isolation(
        output_checkpoint=args.output_checkpoint,
        training_report=args.training_report,
        inputs=(
            baseline_path,
            filtered_path,
            teacher_root,
            prefix_root,
            config_path,
            base_checkpoint,
            source_v60_checkpoint,
        ),
    )
    return V63Preflight(
        baseline_lock_sha256=baseline_lock_sha256,
        baseline_authorization=baseline_authorization,
        filtered_train_sha256=filtered_train_sha256,
        rows=rows,
        scene_ids=scene_ids,
        teacher_targets={key: value.detach().cpu().float() for key, value in targets.items()},
        teacher_metadata=teacher_metadata,
        teacher_metadata_sha256=_sha256_file(teacher_metadata_path),
        teacher_weights_sha256=_sha256_file(teacher_weights_path),
        config=config,
        config_path=config_path,
        runtime_config_sha256=runtime_config_sha256,
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_checkpoint_files=base_checkpoint_files,
        source_v60_checkpoint=source_v60_checkpoint,
        source_v60_checkpoint_sha256=source_v60_checkpoint_sha256,
        source_v60_metadata=source_v60_metadata,
        source_v60_question_norm_state=source_v60_question_norm_state,
        source_v60_question_norm_sha256=source_v60_question_norm_sha256,
        prefixes=prefixes,
        prefix_manifest=prefix_manifest,
        prefix_manifest_sha256=prefix_manifest_sha256,
        output_checkpoint=output_checkpoint,
        training_report=training_report,
    )


def _compute_frozen_question_embeddings(
    preflight: V63Preflight,
    *,
    requested_device: str,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]:
    """Use frozen Gemma input embeddings only; never generate or backpropagate."""

    runtime = StaticRuntimePrefixFactory(
        preflight.config,
        preflight.base_checkpoint,
        preflight.scene_ids[0],
    ).bootstrap
    if not torch.equal(
        runtime.scene_prefix.detach().cpu(),
        preflight.prefixes[preflight.scene_ids[0]].detach().cpu(),
    ):
        raise ValueError("V63 cached prefix differs from the frozen base runtime")
    frozen = freeze_base_runtime(runtime)
    device = _select_training_device(runtime, requested_device)
    unique_questions = sorted({row.question for row in preflight.rows})
    by_text: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for question in unique_questions:
            value = _pooled_question_embedding(runtime, question).detach().cpu().float()
            if tuple(value.shape) != (1, 1, _EXPECTED_HIDDEN_SIZE):
                raise ValueError("V63 pooled Gemma question embedding shape changed")
            if not torch.isfinite(value).all():
                raise ValueError("V63 pooled Gemma question embedding is nonfinite")
            by_text[question] = value
    if any(parameter.requires_grad for parameter in runtime.language.model.parameters()):
        raise RuntimeError("V63 Gemma stack was not completely frozen")
    result = {row.key: by_text[row.question] for row in preflight.rows}
    audit = {
        "device": str(device),
        "unique_question_count": len(unique_questions),
        "opaque_row_count": len(result),
        "pooled_question_shape": [1, 1, _EXPECTED_HIDDEN_SIZE],
        "base_stack_parameter_count": frozen["parameter_count"],
        "base_stack_all_parameters_frozen": frozen["all_parameters_frozen"],
        "gemma_backward_used": False,
        "gemma_generation_used": False,
        "answer_tokens_embedded": False,
    }
    del runtime
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result, audit


def _effective_basis(targets: Mapping[tuple[str, str], torch.Tensor], rank: int) -> torch.Tensor:
    ordered = torch.cat([targets[key] for key in sorted(targets)], dim=0)
    maximum = min(ordered.shape[0] * ordered.shape[1], ordered.shape[2])
    return teacher_output_basis(ordered, rank=min(rank, maximum))


def _basis_coverage(
    targets: Mapping[tuple[str, str], torch.Tensor], basis: torch.Tensor
) -> dict[str, float]:
    cosines: list[float] = []
    for key in sorted(targets):
        target = targets[key].float()
        rms = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        direction = target / rms
        projected = torch.einsum(
            "bch,rh,rk->bck", direction, basis.float(), basis.float()
        )
        cosines.extend(
            F.cosine_similarity(direction, projected, dim=-1).flatten().tolist()
        )
    return {
        "mean_cosine": _mean(cosines),
        "minimum_cosine": min(cosines),
    }


def _basis_targets(
    targets: Mapping[tuple[str, str], torch.Tensor],
    basis: torch.Tensor,
) -> tuple[
    dict[tuple[str, str], torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
]:
    coefficients: dict[tuple[str, str], torch.Tensor] = {}
    rms_values: dict[tuple[str, str], torch.Tensor] = {}
    for key, target in targets.items():
        value = target.float()
        rms = value.square().mean(dim=-1).sqrt()
        direction = value / rms.unsqueeze(-1).clamp_min(1e-8)
        raw = torch.einsum("bch,rh->bcr", direction, basis.float())
        coefficients[key] = F.normalize(raw, dim=-1, eps=1e-8)
        rms_values[key] = rms
    return coefficients, rms_values


def _make_control(
    *,
    basis: torch.Tensor,
    source_question_norm_state: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> TeacherBasisFullSceneQuestionControlV3:
    control = TeacherBasisFullSceneQuestionControlV3(
        _EXPECTED_HIDDEN_SIZE,
        basis,
        control_tokens=_EXPECTED_CONTROL_TOKENS,
        expected_environment_latents=256,
        moment_count=args.moment_count,
        interaction_dim=args.interaction_dim,
        trunk_dim=args.trunk_dim,
        maximum_control_rms=args.maximum_control_rms,
        initial_control_rms=args.initial_control_rms,
        gate_threshold=args.gate_threshold,
    ).cpu().float()
    control.question_norm.load_state_dict(dict(source_question_norm_state), strict=True)
    control.question_norm.requires_grad_(False)
    observed_norm = _question_norm_state(control)
    if (
        set(observed_norm) != set(source_question_norm_state)
        or any(
            not torch.equal(
                observed_norm[name], source_question_norm_state[name].detach().cpu().float()
            )
            for name in observed_norm
        )
        or any(parameter.requires_grad for parameter in control.question_norm.parameters())
    ):
        raise RuntimeError("V63 failed to copy and freeze source V60 question_norm")
    if control.trainable_parameter_count >= 1_500_000:
        raise ValueError("V63 controller exceeds the 1.5M trainable-parameter ceiling")
    return control


def _scene_signatures(
    control: TeacherBasisFullSceneQuestionControlV3,
    prefixes: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        return {
            scene_id: control.encode_scene(prefix.detach().cpu().float())
            for scene_id, prefix in prefixes.items()
        }


def _changed_units(rows: Sequence[V63Row]) -> tuple[tuple[V63Row, V63Row], ...]:
    grouped: defaultdict[tuple[str, str], list[V63Row]] = defaultdict(list)
    for row in rows:
        if row.route_label:
            grouped[(row.pair_id, row.question_key)].append(row)
    result: list[tuple[V63Row, V63Row]] = []
    for key, sides in sorted(grouped.items()):
        if len(sides) != 2:
            raise ValueError(f"V63 changed unit lost a paired side: {key}")
        ordered = sorted(sides, key=lambda row: (row.scene_id, row.question_id))
        result.append((ordered[0], ordered[1]))
    return tuple(result)


def _value_batch_loss(
    control: TeacherBasisFullSceneQuestionControlV3,
    rows: Sequence[V63Row],
    *,
    signatures: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    targets: Mapping[tuple[str, str], torch.Tensor],
    coefficient_targets: Mapping[tuple[str, str], torch.Tensor],
    rms_targets: Mapping[tuple[str, str], torch.Tensor],
    args: argparse.Namespace,
) -> torch.Tensor:
    scene = torch.cat([signatures[row.scene_id] for row in rows])
    question = torch.cat([questions[row.key] for row in rows])
    target = torch.cat([targets[row.key] for row in rows])
    target_coefficients = torch.cat([coefficient_targets[row.key] for row in rows])
    target_rms = torch.cat([rms_targets[row.key] for row in rows])
    output = control.forward_from_signature(scene, question)
    coefficient_loss = 1.0 - F.cosine_similarity(
        output.coefficient_directions, target_coefficients, dim=-1
    ).mean()
    log_rms_loss = F.mse_loss(
        output.control_rms.clamp_min(1e-7).log(),
        target_rms.clamp_min(1e-7).log(),
    )
    reconstruction_loss = 1.0 - F.cosine_similarity(
        output.control_tokens, target, dim=-1
    ).mean()
    target_power = target.square().mean().clamp_min(1e-8)
    relative_mse = F.mse_loss(output.control_tokens, target) / target_power
    return (
        args.coefficient_weight * coefficient_loss
        + args.log_rms_weight * log_rms_loss
        + args.reconstruction_weight * reconstruction_loss
        + args.relative_mse_weight * relative_mse
    )


def _pair_delta_loss(
    control: TeacherBasisFullSceneQuestionControlV3,
    units: Sequence[tuple[V63Row, V63Row]],
    *,
    signatures: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    targets: Mapping[tuple[str, str], torch.Tensor],
) -> torch.Tensor:
    flat = [row for unit in units for row in unit]
    output = control.forward_from_signature(
        torch.cat([signatures[row.scene_id] for row in flat]),
        torch.cat([questions[row.key] for row in flat]),
    ).control_tokens.reshape(len(units), 2, _EXPECTED_CONTROL_TOKENS, -1)
    target = torch.cat([targets[row.key] for row in flat]).reshape(
        len(units), 2, _EXPECTED_CONTROL_TOKENS, -1
    )
    predicted_delta = output[:, 0] - output[:, 1]
    target_delta = target[:, 0] - target[:, 1]
    target_power = target_delta.square().mean(dim=(1, 2)).clamp_min(1e-8)
    cosine = F.cosine_similarity(
        predicted_delta.flatten(1), target_delta.flatten(1), dim=-1
    )
    relative_mse = (predicted_delta - target_delta).square().mean(dim=(1, 2))
    return (1.0 - cosine + 0.10 * relative_mse / target_power).mean()


def _route_loss(
    control: TeacherBasisFullSceneQuestionControlV3,
    rows: Sequence[V63Row],
    questions: Mapping[tuple[str, str], torch.Tensor],
) -> torch.Tensor:
    embeddings = torch.cat([questions[row.key] for row in rows])
    labels = torch.tensor([float(row.route_label) for row in rows])
    # Detaching normalized frozen-question features keeps this provisional route
    # objective from modifying the continuous-value question layer norm.
    normalized = control.normalized_question(embeddings).detach()
    logits = control.route_logits_from_normalized_question(normalized)
    positives = labels.sum().clamp_min(1.0)
    negatives = (1.0 - labels).sum().clamp_min(1.0)
    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=(negatives / positives),
    )


def _optimizer_step(
    *,
    loss: torch.Tensor,
    control: TeacherBasisFullSceneQuestionControlV3,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float,
) -> float:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("V63 optimization loss is nonfinite or nonscalar")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [parameter for parameter in control.parameters() if parameter.requires_grad]
    gradient = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip_norm)
    value = float(gradient.detach())
    if not math.isfinite(value):
        raise RuntimeError("V63 controller gradient is nonfinite")
    optimizer.step()
    return value


def _fit_controller(
    *,
    rows: Sequence[V63Row],
    targets: Mapping[tuple[str, str], torch.Tensor],
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    source_question_norm_state: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    seed: int,
    log_phase: str,
    fixed_output_basis: torch.Tensor | None = None,
) -> FitResult:
    if not targets:
        raise ValueError("V63 fit requires at least one numeric teacher")
    torch.manual_seed(seed)
    random.seed(seed)
    basis = (
        _effective_basis(targets, args.basis_rank)
        if fixed_output_basis is None
        else fixed_output_basis.detach().cpu().float().contiguous()
    )
    if (
        basis.ndim != 2
        or basis.shape[1] != _EXPECTED_HIDDEN_SIZE
        or not torch.isfinite(basis).all()
    ):
        raise ValueError("V63 fixed output basis must be finite [R,1536]")
    control = _make_control(
        basis=basis,
        source_question_norm_state=source_question_norm_state,
        args=args,
    )
    signatures = _scene_signatures(control, prefixes)
    changed_rows = [row for row in rows if row.key in targets]
    if {row.key for row in changed_rows} != set(targets):
        raise ValueError("V63 fit rows and numeric targets do not match")
    positive_questions = torch.cat(
        [control.normalized_question(questions[row.key]) for row in rows if row.route_label]
    )
    negative_questions = torch.cat(
        [control.normalized_question(questions[row.key]) for row in rows if not row.route_label]
    )
    control.initialize_route_prototypes(positive_questions, negative_questions)
    coefficient_targets, rms_targets = _basis_targets(targets, basis)
    units = _changed_units(changed_rows)
    optimizer_parameters = [
        parameter for parameter in control.parameters() if parameter.requires_grad
    ]
    if not optimizer_parameters or any(
        parameter.requires_grad for parameter in control.question_norm.parameters()
    ):
        raise RuntimeError("V63 optimizer scope includes frozen source question_norm")
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    gradients: list[float] = []
    ordinal = 0
    final_route_loss = math.inf
    started = time.perf_counter()
    for epoch in range(args.epochs):
        rng = random.Random(seed + epoch * 1_000_003)
        expanded = [row for row in changed_rows for _ in range(args.changed_repeats)]
        rng.shuffle(expanded)
        for offset in range(0, len(expanded), args.batch_size):
            batch = expanded[offset : offset + args.batch_size]
            loss = _value_batch_loss(
                control,
                batch,
                signatures=signatures,
                questions=questions,
                targets=targets,
                coefficient_targets=coefficient_targets,
                rms_targets=rms_targets,
                args=args,
            )
            gradients.append(
                _optimizer_step(
                    loss=loss,
                    control=control,
                    optimizer=optimizer,
                    gradient_clip_norm=args.gradient_clip_norm,
                )
            )
            ordinal += 1
        if args.pair_delta_weight > 0.0:
            pair_loss = args.pair_delta_weight * _pair_delta_loss(
                control,
                units,
                signatures=signatures,
                questions=questions,
                targets=targets,
            )
            gradients.append(
                _optimizer_step(
                    loss=pair_loss,
                    control=control,
                    optimizer=optimizer,
                    gradient_clip_norm=args.gradient_clip_norm,
                )
            )
            ordinal += 1
        if args.route_weight > 0.0:
            route = _route_loss(control, rows, questions)
            final_route_loss = float(route.detach())
            gradients.append(
                _optimizer_step(
                    loss=args.route_weight * route,
                    control=control,
                    optimizer=optimizer,
                    gradient_clip_norm=args.gradient_clip_norm,
                )
            )
            ordinal += 1
        if (epoch + 1) % args.log_every == 0:
            _log_event(
                phase=log_phase,
                epoch=epoch + 1,
                optimizer_steps=ordinal,
                teacher_sides=len(targets),
            )
    control.eval()
    observed_norm = _question_norm_state(control)
    expected_norm_sha256 = _tensor_state_sha256(source_question_norm_state)
    observed_norm_sha256 = _tensor_state_sha256(observed_norm)
    norm_frozen = all(
        not parameter.requires_grad for parameter in control.question_norm.parameters()
    ) and all(parameter.grad is None for parameter in control.question_norm.parameters())
    if observed_norm_sha256 != expected_norm_sha256 or not norm_frozen:
        raise RuntimeError("V63 source V60 question_norm changed during fitting")
    return FitResult(
        control=control,
        signatures=signatures,
        basis_reconstruction=_basis_coverage(targets, basis),
        elapsed_seconds=time.perf_counter() - started,
        optimizer_steps=ordinal,
        maximum_preclip_gradient_norm=max(gradients),
        final_route_loss=final_route_loss,
        question_norm_sha256=observed_norm_sha256,
        question_norm_frozen=norm_frozen,
    )


def _measure_reconstruction(
    control: TeacherBasisFullSceneQuestionControlV3,
    rows: Sequence[V63Row],
    *,
    signatures: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    targets: Mapping[tuple[str, str], torch.Tensor],
) -> ReconstructionMeasurements:
    changed = [row for row in rows if row.key in targets]
    predicted: dict[tuple[str, str], torch.Tensor] = {}
    prompt_cosines: list[float] = []
    rms_errors: list[float] = []
    squared_error_sum = 0.0
    element_count = 0
    with torch.inference_mode():
        for offset in range(0, len(changed), 32):
            batch = changed[offset : offset + 32]
            output = control.forward_from_signature(
                torch.cat([signatures[row.scene_id] for row in batch]),
                torch.cat([questions[row.key] for row in batch]),
            ).control_tokens.cpu()
            expected = torch.cat([targets[row.key] for row in batch]).float()
            prompt_cosines.extend(
                F.cosine_similarity(output, expected, dim=-1).flatten().tolist()
            )
            observed_rms = output.square().mean(dim=-1).sqrt()
            expected_rms = expected.square().mean(dim=-1).sqrt()
            rms_errors.extend((observed_rms - expected_rms).abs().flatten().tolist())
            error = output - expected
            squared_error_sum += float(error.square().sum())
            element_count += error.numel()
            for index, row in enumerate(batch):
                predicted[row.key] = output[index : index + 1]
    pair_cosines: list[float] = []
    for left, right in _changed_units(changed):
        predicted_delta = predicted[left.key] - predicted[right.key]
        target_delta = targets[left.key] - targets[right.key]
        if float(target_delta.square().sum()) <= 1e-12:
            raise ValueError("V63 teacher pair has a zero numeric delta")
        pair_cosines.append(
            float(
                F.cosine_similarity(
                    predicted_delta.flatten()[None],
                    target_delta.flatten()[None],
                    dim=-1,
                ).item()
            )
        )
    return ReconstructionMeasurements(
        prompt_cosines=tuple(prompt_cosines),
        prompt_rms_absolute_errors=tuple(rms_errors),
        squared_error_sum=squared_error_sum,
        element_count=element_count,
        pair_delta_cosines=tuple(pair_cosines),
    )


def _route_metrics(
    control: TeacherBasisFullSceneQuestionControlV3,
    rows: Sequence[V63Row],
    questions: Mapping[tuple[str, str], torch.Tensor],
) -> dict[str, float | int]:
    with torch.inference_mode():
        embeddings = torch.cat([questions[row.key] for row in rows])
        probabilities = torch.sigmoid(
            control.route_logits_from_normalized_question(
                control.normalized_question(embeddings)
            )
        ).tolist()
    labels = [row.route_label for row in rows]
    correct = sum(
        (probability >= control.gate_threshold) == label
        for probability, label in zip(probabilities, labels, strict=True)
    )
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "provisional_not_a_promotion_gate": True,
    }


def _combine_measurements(
    measurements: Sequence[ReconstructionMeasurements],
) -> ReconstructionMeasurements:
    return ReconstructionMeasurements(
        prompt_cosines=tuple(
            value for measurement in measurements for value in measurement.prompt_cosines
        ),
        prompt_rms_absolute_errors=tuple(
            value
            for measurement in measurements
            for value in measurement.prompt_rms_absolute_errors
        ),
        squared_error_sum=sum(value.squared_error_sum for value in measurements),
        element_count=sum(value.element_count for value in measurements),
        pair_delta_cosines=tuple(
            value
            for measurement in measurements
            for value in measurement.pair_delta_cosines
        ),
    )


def evaluate_cv_checks(
    aggregate: Mapping[str, float | int],
    *,
    fold_mean_cosines: Sequence[float],
    thresholds: V63CVThresholds = PREREGISTERED_CV_THRESHOLDS,
) -> dict[str, bool]:
    return {
        "mean_prompt_cosine": float(aggregate["mean_prompt_cosine"])
        >= thresholds.mean_prompt_cosine,
        "minimum_prompt_cosine": float(aggregate["minimum_prompt_cosine"])
        >= thresholds.minimum_prompt_cosine,
        "minimum_fold_mean_prompt_cosine": min(fold_mean_cosines)
        >= thresholds.minimum_fold_mean_prompt_cosine,
        "prompt_root_mean_square_error": float(
            aggregate["prompt_root_mean_square_error"]
        )
        <= thresholds.prompt_root_mean_square_error,
        "mean_prompt_rms_absolute_error": float(
            aggregate["mean_prompt_rms_absolute_error"]
        )
        <= thresholds.mean_prompt_rms_absolute_error,
        "mean_pair_delta_cosine": float(aggregate["mean_pair_delta_cosine"])
        >= thresholds.mean_pair_delta_cosine,
        "positive_pair_delta_fraction": float(
            aggregate["positive_pair_delta_fraction"]
        )
        >= thresholds.minimum_positive_pair_delta_fraction,
        "complete_teacher_side_coverage": aggregate["teacher_side_count"]
        == _EXPECTED_CHANGED_SIDES,
        "complete_changed_unit_coverage": aggregate["changed_pair_unit_count"]
        == _EXPECTED_CHANGED_UNITS,
        "complete_fold_coverage": len(fold_mean_cosines) == _EXPECTED_PAIRS,
    }


def evaluate_final_checks(
    summary: Mapping[str, float | int],
    *,
    thresholds: V63FinalThresholds = PREREGISTERED_FINAL_THRESHOLDS,
) -> dict[str, bool]:
    return {
        "mean_prompt_cosine": float(summary["mean_prompt_cosine"])
        >= thresholds.mean_prompt_cosine,
        "minimum_prompt_cosine": float(summary["minimum_prompt_cosine"])
        >= thresholds.minimum_prompt_cosine,
        "prompt_root_mean_square_error": float(
            summary["prompt_root_mean_square_error"]
        )
        <= thresholds.prompt_root_mean_square_error,
        "mean_prompt_rms_absolute_error": float(
            summary["mean_prompt_rms_absolute_error"]
        )
        <= thresholds.mean_prompt_rms_absolute_error,
        "mean_pair_delta_cosine": float(summary["mean_pair_delta_cosine"])
        >= thresholds.mean_pair_delta_cosine,
        "complete_teacher_side_coverage": summary["teacher_side_count"]
        == _EXPECTED_CHANGED_SIDES,
        "complete_changed_unit_coverage": summary["changed_pair_unit_count"]
        == _EXPECTED_CHANGED_UNITS,
    }


def run_pair_disjoint_cross_validation(
    *,
    preflight: V63Preflight,
    questions: Mapping[tuple[str, str], torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    measurements: list[ReconstructionMeasurements] = []
    heldout_keys: set[tuple[str, str]] = set()
    for fold_index, heldout_pair_id in enumerate(TRAIN_PAIR_IDS):
        train_rows = tuple(
            row for row in preflight.rows if row.pair_id != heldout_pair_id
        )
        heldout_rows = tuple(
            row for row in preflight.rows if row.pair_id == heldout_pair_id
        )
        train_targets = {
            key: target
            for key, target in preflight.teacher_targets.items()
            if any(row.key == key and row.pair_id != heldout_pair_id for row in preflight.rows)
        }
        heldout_targets = {
            row.key: preflight.teacher_targets[row.key]
            for row in heldout_rows
            if row.route_label
        }
        if set(train_targets) & set(heldout_targets):
            raise AssertionError("V63 CV teacher partitions overlap")
        if heldout_keys & set(heldout_targets):
            raise AssertionError("V63 CV evaluated a teacher more than once")
        heldout_keys.update(heldout_targets)
        fit = _fit_controller(
            rows=train_rows,
            targets=train_targets,
            prefixes=preflight.prefixes,
            questions=questions,
            source_question_norm_state=preflight.source_v60_question_norm_state,
            args=args,
            seed=args.seed + (fold_index + 1) * 100_003,
            log_phase=f"v63_cv_{heldout_pair_id}",
        )
        measured = _measure_reconstruction(
            fit.control,
            heldout_rows,
            signatures=fit.signatures,
            questions=questions,
            targets=heldout_targets,
        )
        measurements.append(measured)
        fold_summary = measured.summary()
        folds.append(
            {
                "fold_index": fold_index,
                "heldout_pair_id": heldout_pair_id,
                "training_pair_count": _EXPECTED_PAIRS - 1,
                "training_teacher_side_count": len(train_targets),
                "heldout_teacher_side_count": len(heldout_targets),
                "heldout_changed_unit_count": len(_changed_units(heldout_rows)),
                "fold_output_basis_rank": fit.control.output_basis_rank,
                "training_basis_reconstruction": fit.basis_reconstruction,
                "heldout_basis_coverage": _basis_coverage(
                    heldout_targets, fit.control.output_basis
                ),
                "heldout_reconstruction": fold_summary,
                "heldout_route": _route_metrics(
                    fit.control, heldout_rows, questions
                ),
                "optimizer_steps": fit.optimizer_steps,
                "elapsed_seconds": fit.elapsed_seconds,
                "maximum_preclip_gradient_norm": (
                    fit.maximum_preclip_gradient_norm
                ),
                "source_v60_question_norm_sha256": fit.question_norm_sha256,
                "source_v60_question_norm_frozen": fit.question_norm_frozen,
            }
        )
    if heldout_keys != set(preflight.teacher_targets):
        raise AssertionError("V63 CV did not evaluate every teacher exactly once")
    aggregate = _combine_measurements(measurements).summary()
    fold_means = [
        float(fold["heldout_reconstruction"]["mean_prompt_cosine"])
        for fold in folds
    ]
    checks = evaluate_cv_checks(aggregate, fold_mean_cosines=fold_means)
    norm_exact_every_fold = all(
        fold["source_v60_question_norm_frozen"] is True
        and fold["source_v60_question_norm_sha256"]
        == preflight.source_v60_question_norm_sha256
        for fold in folds
    )
    checks["source_v60_question_norm_exact_and_frozen_in_every_fold"] = (
        norm_exact_every_fold
    )
    return {
        "protocol": "deterministic_leave_one_counterfactual_pair_out",
        "pair_count": len(folds),
        "fold_specific_output_basis": True,
        "heldout_teacher_used_in_fold_basis": False,
        "heldout_teacher_used_in_fold_optimization": False,
        "each_teacher_evaluated_once": True,
        "source_v60_question_norm_sha256": (
            preflight.source_v60_question_norm_sha256
        ),
        "source_v60_question_norm_exact_and_frozen_in_every_fold": norm_exact_every_fold,
        "thresholds": asdict(PREREGISTERED_CV_THRESHOLDS),
        "aggregate": aggregate,
        "checks": checks,
        "passed": all(checks.values()),
        "folds": folds,
    }


def _publish_checkpoint_and_report(
    *,
    preflight: V63Preflight,
    control: TeacherBasisFullSceneQuestionControlV3,
    report_without_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    checkpoint = _safe_output_path(preflight.output_checkpoint, "V63 checkpoint")
    report = _safe_output_path(preflight.training_report, "V63 training report")
    staging_root = Path(
        tempfile.mkdtemp(prefix=".v63-publish-", dir=checkpoint.parent)
    )
    published_checkpoint = False
    published_report = False
    try:
        staged_checkpoint = staging_root / "checkpoint"
        checkpoint_hashes = save_v3_control_checkpoint(
            staged_checkpoint,
            control=control,
            base_checkpoint_sha256=preflight.base_checkpoint_sha256,
            base_runtime_config_sha256=preflight.runtime_config_sha256,
        )
        final_report = {
            **dict(report_without_checkpoint),
            "checkpoint": checkpoint_hashes,
        }
        staged_report = staging_root / "training_report.json"
        _write_json(staged_report, final_report)
        os.rename(staged_checkpoint, checkpoint)
        published_checkpoint = True
        os.rename(staged_report, report)
        published_report = True
        return checkpoint_hashes, final_report
    except BaseException:
        if published_report:
            report.unlink(missing_ok=True)
        if published_checkpoint:
            shutil.rmtree(checkpoint, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def train_v63(args: argparse.Namespace) -> dict[str, Any]:
    preflight = build_v63_preflight(args)
    questions, gemma_audit = _compute_frozen_question_embeddings(
        preflight,
        requested_device=args.device,
    )
    cv = run_pair_disjoint_cross_validation(
        preflight=preflight,
        questions=questions,
        args=args,
    )
    if not cv["passed"]:
        raise V63OfflineGateError(
            "V63 pair-disjoint numeric reconstruction gate failed; "
            "no checkpoint or training report published",
            {"cross_validation": cv},
            failure_stage="pair_disjoint_cross_validation",
            provenance=_failure_provenance(preflight, args),
            scope=_failure_scope(),
            gemma_audit=gemma_audit,
        )

    final_fit = _fit_controller(
        rows=preflight.rows,
        targets=preflight.teacher_targets,
        prefixes=preflight.prefixes,
        questions=questions,
        source_question_norm_state=preflight.source_v60_question_norm_state,
        args=args,
        seed=args.seed + 9_999_991,
        log_phase="v63_final_all_training_fit",
    )
    final_measurements = _measure_reconstruction(
        final_fit.control,
        preflight.rows,
        signatures=final_fit.signatures,
        questions=questions,
        targets=preflight.teacher_targets,
    )
    final_summary = final_measurements.summary()
    final_checks = evaluate_final_checks(final_summary)
    final_checks["source_v60_question_norm_exact"] = (
        final_fit.question_norm_sha256
        == preflight.source_v60_question_norm_sha256
    )
    final_checks["source_v60_question_norm_frozen"] = final_fit.question_norm_frozen
    if not all(final_checks.values()):
        raise V63OfflineGateError(
            "V63 all-training numeric reconstruction gate failed; "
            "no checkpoint or training report published",
            {
                "cross_validation": cv,
                "final_fit": {
                    "summary": final_summary,
                    "checks": final_checks,
                },
            },
            failure_stage="all_training_fit",
            provenance=_failure_provenance(preflight, args),
            scope=_failure_scope(),
            gemma_audit=gemma_audit,
        )

    report_without_checkpoint = {
        "schema_version": 1,
        "artifact": "v63_pair_disjoint_expanded_value_distillation",
        "offline_checks_passed": True,
        "promotion_eligible": False,
        "successor_factorized_route_required": True,
        "base": {
            "checkpoint_sha256": preflight.base_checkpoint_sha256,
            "checkpoint_files": preflight.base_checkpoint_files,
            "runtime_config_effective_sha256": preflight.runtime_config_sha256,
            "runtime_config_file_sha256": _sha256_file(preflight.config_path),
        },
        "authorization": {
            "baseline_lock_sha256": preflight.baseline_lock_sha256,
            "preregistration_sha256": PINNED_V62_PREREGISTRATION_SHA256,
            "baseline_validated_before_training_data": True,
        },
        "source_v60": {
            "checkpoint_sha256": preflight.source_v60_checkpoint_sha256,
            "weights_sha256": preflight.source_v60_metadata["weights_sha256"],
            "runtime_metadata_sha256": _sha256_file(
                preflight.source_v60_checkpoint / "runtime_metadata.json"
            ),
            "architecture": preflight.source_v60_metadata["architecture"],
            "question_norm_sha256": preflight.source_v60_question_norm_sha256,
            "question_norm_copied_tensor_exact": True,
            "question_norm_frozen_in_every_fit": True,
        },
        "inputs": {
            "filtered_training_qa_sha256": preflight.filtered_train_sha256,
            "training_scene_ids": list(preflight.scene_ids),
            "training_record_count": len(preflight.rows),
            "counterfactual_pair_count": len(TRAIN_PAIR_IDS),
            "changed_teacher_side_count": len(preflight.teacher_targets),
            "changed_paired_unit_count": len(_changed_units(preflight.rows)),
            "teacher_metadata_sha256": preflight.teacher_metadata_sha256,
            "teacher_weights_sha256": preflight.teacher_weights_sha256,
            "teacher_selection_sha256": preflight.teacher_metadata["selection_sha256"],
            "prefix_cache_manifest_sha256": preflight.prefix_manifest_sha256,
            "prefix_shape": preflight.prefix_manifest["scenes"][
                preflight.scene_ids[0]
            ]["shape"],
        },
        "architecture": {
            "name": "teacher_basis_full_scene_question_control_v3",
            "runtime_schema_version": 3,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in final_fit.control.parameters()
                if parameter.requires_grad
            ),
            "saved_parameter_count": final_fit.control.parameter_count,
            "hidden_size": final_fit.control.hidden_size,
            "control_tokens": final_fit.control.control_token_count,
            "basis_rank_requested": args.basis_rank,
            "basis_rank_effective": final_fit.control.output_basis_rank,
            "scene_moment_count": final_fit.control.moment_count,
            "all_256_scene_latents_used": True,
            "boundary_tokens_excluded": True,
            "question_dependent_scene_retrieval": False,
            "softmax_scene_attention_used": False,
            "control_values_scene_question_bilinear": True,
            "native_prompt_shape": list(PROMPT_SHAPE),
            "source_v60_question_norm_frozen": True,
        },
        "cross_validation": cv,
        "final_fit": {
            "thresholds": asdict(PREREGISTERED_FINAL_THRESHOLDS),
            "checks": final_checks,
            "summary": final_summary,
            "basis_reconstruction": final_fit.basis_reconstruction,
            "route": _route_metrics(final_fit.control, preflight.rows, questions),
            "optimizer_steps": final_fit.optimizer_steps,
            "elapsed_seconds": final_fit.elapsed_seconds,
            "maximum_preclip_gradient_norm": (
                final_fit.maximum_preclip_gradient_norm
            ),
            "final_route_loss": final_fit.final_route_loss,
        },
        "gemma_audit": gemma_audit,
        "optimization": {
            "seed": args.seed,
            "epochs_per_fold_and_final": args.epochs,
            "batch_size": args.batch_size,
            "changed_repeats": args.changed_repeats,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "loss_weights": {
                "coefficient": args.coefficient_weight,
                "log_rms": args.log_rms_weight,
                "reconstruction_cosine": args.reconstruction_weight,
                "relative_mse": args.relative_mse_weight,
                "pair_delta": args.pair_delta_weight,
                "provisional_route": args.route_weight,
            },
            "controller_training_device": "cpu",
        },
        "scope": {
            "base_scene_stack_frozen": True,
            "gemma_backward_used": False,
            "gemma_generation_used": False,
            "numeric_teacher_cache_only": True,
            "teacher_cache_runtime_access": False,
            "complete_scene_prefix_retained": True,
            "question_dependent_scene_retrieval": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
            "report_contains_question_or_answer_text": False,
            "source_v60_question_norm_exact_and_frozen": True,
        },
    }
    _checkpoint_hashes, report = _publish_checkpoint_and_report(
        preflight=preflight,
        control=final_fit.control,
        report_without_checkpoint=report_without_checkpoint,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_baseline_lock_authorization_argument(parser)
    add_filtered_training_data_argument(parser)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--source-v60-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument(
        "--diagnostics-output",
        help=(
            "Optional create-once JSON destination written only when an offline "
            "gate fails; no checkpoint or training report is published."
        ),
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=630063)
    parser.add_argument("--basis-rank", type=int, default=128)
    parser.add_argument("--moment-count", type=int, default=8)
    parser.add_argument("--interaction-dim", type=int, default=32)
    parser.add_argument("--trunk-dim", type=int, default=192)
    parser.add_argument("--maximum-control-rms", type=float, default=0.25)
    parser.add_argument("--initial-control-rms", type=float, default=0.075)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--changed-repeats", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--coefficient-weight", type=float, default=3.0)
    parser.add_argument("--log-rms-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=2.0)
    parser.add_argument("--relative-mse-weight", type=float, default=0.25)
    parser.add_argument("--pair-delta-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    diagnostics_output = _validate_diagnostics_destination(args)
    try:
        report = train_v63(args)
    except V63OfflineGateError as exc:
        written_diagnostics: str | None = None
        if diagnostics_output is not None:
            if _resolve(args.output_checkpoint).exists() or _resolve(
                args.training_report
            ).exists():
                raise RuntimeError(
                    "V63 refused failure diagnostics after a model artifact appeared"
                ) from exc
            written_diagnostics = str(
                _publish_failure_report_create_once(
                    diagnostics_output,
                    _failure_report(exc),
                )
            )
        print(
            json.dumps(
                {
                    "passed": False,
                    "checkpoint_published": False,
                    "training_report_published": False,
                    "failure_diagnostics": written_diagnostics,
                    "error": str(exc),
                    "diagnostics": exc.diagnostics,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "passed": True,
                "promotion_eligible": False,
                "checkpoint": str(_resolve(args.output_checkpoint)),
                "training_report": str(_resolve(args.training_report)),
                "cross_validation": report["cross_validation"]["aggregate"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PREREGISTERED_CV_THRESHOLDS",
    "PREREGISTERED_FINAL_THRESHOLDS",
    "ReconstructionMeasurements",
    "V63CVThresholds",
    "V63FinalThresholds",
    "V63OfflineGateError",
    "V63Preflight",
    "V63Row",
    "build_v63_preflight",
    "evaluate_cv_checks",
    "evaluate_final_checks",
    "main",
    "run_pair_disjoint_cross_validation",
    "train_v63",
    "training_scene_ids",
]
