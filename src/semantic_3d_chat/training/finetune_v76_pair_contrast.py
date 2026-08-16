"""Quarantined V76 all-pair Gemma answer-contrast training screen.

V76 starts from one exact, numerically screened V75 dense full-scene reader.
Each optimizer step is an atomic changed counterfactual unit: the same question
is paired with two immutable continuous scene prefixes and two different
answers.  Frozen Gemma supplies both the correct-answer NLL and the NLL of the
paired alternative answer for each scene.  The trainable V75 reader must make
the correct answer cheaper by a configurable margin.

Only historical training-pool pairs are optimized.  No official validation,
official test, deferred-final, or oracle path is accepted.  The resulting
safetensors file is diagnostic-only and deliberately cannot satisfy the chat
runtime's checkpoint schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.prefix_injection import stack_prefix_batches
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.finetune_v74_gemma_nll import (
    V75_STATE_FIELDS,
    _guard_training_input,
    _load_initial_candidate,
    _question_embeddings,
    assert_dense_reader_exact_zero_scene,
    assert_exclusive_dense_reader_trainable_surface,
)
from semantic_3d_chat.training.pair_curriculum import token_normalized_nll
from semantic_3d_chat.training.question_control_pair_objective_v57 import (
    _compose_batch,
)
from semantic_3d_chat.training.train_adapter import forward_prefix_batch
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    _select_training_device,
    freeze_base_runtime,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _disable_decoder_checkpointing,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    EXPECTED_HIDDEN_SIZE,
    HELD_PAIR_IDS,
    TRAIN_PAIR_IDS,
    ChangedUnitV73,
    RowV73,
    _sha256_file,
    changed_units_v73,
    load_config_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)

V76_INITIAL_CANDIDATE: Final[str] = (
    "reports/gemma4/artifacts/v75_nonlinear_h768_p12_w2_passed_diagnostic.safetensors"
)
V76_INITIAL_CANDIDATE_SHA256: Final[str] = (
    "182481dd77645cd2a467b3585dd7b060fcea578cc013eebc21486e1915ce9c17"
)
EXPECTED_CHANGED_UNIT_COUNT: Final[int] = 40
EXPECTED_CHANGED_SIDE_COUNT: Final[int] = 80
SELECTION_SALT: Final[str] = "semantic_3d_chat.v76.all_pair_contrast.v1"
EXPECTED_CHANGE_TYPE_COUNTS: Final[dict[str, int]] = {
    "book_support": 8,
    "chair_orientation": 1,
    "color_swap": 4,
    "cube_support": 3,
    "mirror_lr": 8,
    "object_count": 1,
    "object_relocation": 4,
    "object_removal": 3,
    "picture_support": 8,
}
_SOURCE_METADATA: Final[dict[str, str]] = {
    "artifact": "v75_verified_teacher_dense_reader_candidate_v1",
    "training_pool_only": "true",
    "runtime_promotion_forbidden_until_gemma_gate": "true",
    "numeric_gate_passed": "true",
    "answer_codebook_serialized": "false",
    "environmental_text_inputs": "0",
}
_FORBIDDEN_OUTPUT_TOKENS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "validate", "test", "deferred", "final", "runtime"}
)


@dataclass(frozen=True)
class V76LossSettings:
    """Loss coefficients for one atomic two-scene training unit."""

    answer_nll_weight: float = 1.0
    pair_contrast_weight: float = 2.0
    pair_contrast_margin: float = 0.5
    source_anchor_weight: float = 0.01

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"V76 {field} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"V76 {field} must be finite and nonnegative")
        if self.answer_nll_weight == 0.0 and self.pair_contrast_weight == 0.0:
            raise ValueError("V76 must enable answer NLL or paired-answer contrast")


@dataclass(frozen=True)
class ScheduledPairV76:
    cycle: int
    step_in_cycle: int
    unit: ChangedUnitV73


def _unit_digest(unit: ChangedUnitV73) -> str:
    return hashlib.sha256(
        f"{SELECTION_SALT}|{unit.pair_id}|{unit.question_key}".encode()
    ).hexdigest()


def select_historical_changed_units_v76(
    train_rows: Sequence[RowV73], *, max_units: int = EXPECTED_CHANGED_UNIT_COUNT
) -> tuple[ChangedUnitV73, ...]:
    """Return a deterministic subset of the exact 40 historical changed units.

    The default is exhaustive.  ``max_units`` exists solely for bounded local
    diagnostics; it never permits a held pair and is recorded in the output.
    """

    if isinstance(max_units, bool) or not isinstance(max_units, int):
        raise TypeError("V76 max_units must be an integer")
    if not 1 <= max_units <= EXPECTED_CHANGED_UNIT_COUNT:
        raise ValueError(f"V76 max_units must be in [1, {EXPECTED_CHANGED_UNIT_COUNT}]")
    units = changed_units_v73(train_rows)
    pair_ids = {unit.pair_id for unit in units}
    if not pair_ids <= set(TRAIN_PAIR_IDS) or pair_ids & set(HELD_PAIR_IDS):
        raise ValueError("V76 optimization escaped historical training pairs")
    if len(units) != EXPECTED_CHANGED_UNIT_COUNT:
        raise ValueError("V76 historical changed-unit inventory changed")
    if len({row.key for unit in units for row in (unit.left, unit.right)}) != (
        EXPECTED_CHANGED_SIDE_COUNT
    ):
        raise ValueError("V76 historical changed-side inventory changed")
    if Counter(unit.change_type for unit in units) != Counter(EXPECTED_CHANGE_TYPE_COUNTS):
        raise ValueError("V76 historical changed-family inventory changed")
    for unit in units:
        if (
            unit.left.question != unit.right.question
            or unit.left.answer == unit.right.answer
            or unit.left.paired_scene_id != unit.right.scene_id
            or unit.right.paired_scene_id != unit.left.scene_id
        ):
            raise ValueError("V76 changed unit lost exact same-question pairing")
    if max_units == EXPECTED_CHANGED_UNIT_COUNT:
        return tuple(units)
    selected = sorted(units, key=lambda unit: (_unit_digest(unit), unit.pair_id))[:max_units]
    return tuple(sorted(selected, key=lambda unit: (unit.pair_id, unit.question_key)))


def deterministic_pair_schedule_v76(
    units: Sequence[ChangedUnitV73], *, cycles: int, seed: int
) -> tuple[ScheduledPairV76, ...]:
    """Shuffle complete atomic units deterministically once per complete cycle."""

    if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= 12:
        raise ValueError("V76 cycles must be an integer in [1, 12]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("V76 seed must be a nonnegative integer")
    keys = [(unit.pair_id, unit.question_key) for unit in units]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("V76 schedule requires unique changed pair units")
    result: list[ScheduledPairV76] = []
    for cycle in range(cycles):
        order = list(units)
        random.Random(seed + cycle).shuffle(order)
        result.extend(
            ScheduledPairV76(cycle + 1, index + 1, unit) for index, unit in enumerate(order)
        )
    return tuple(result)


def paired_answer_contrast_objective_v76(
    correct_answer_nll: torch.Tensor,
    alternative_answer_nll: torch.Tensor,
    source_anchor_l2: torch.Tensor,
    settings: V76LossSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine answer NLL, own-vs-paired-answer hinge, and source anchoring.

    A positive preference margin means the paired alternative has greater NLL
    than the scene's own answer.  Both scene sides contribute symmetrically.
    """

    if (
        correct_answer_nll.ndim != 1
        or alternative_answer_nll.shape != correct_answer_nll.shape
        or correct_answer_nll.numel() < 1
    ):
        raise ValueError("V76 answer NLL inputs must be matching nonempty vectors")
    if source_anchor_l2.ndim != 0:
        raise ValueError("V76 source anchor must be a scalar")
    values = (correct_answer_nll, alternative_answer_nll, source_anchor_l2)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("V76 objective inputs must be finite")
    if bool((correct_answer_nll < 0.0).any()) or bool((alternative_answer_nll < 0.0).any()):
        raise ValueError("V76 answer NLL cannot be negative")
    if bool(source_anchor_l2 < 0.0):
        raise ValueError("V76 source anchor cannot be negative")
    preference_margin = alternative_answer_nll - correct_answer_nll
    contrast_hinge = F.relu(float(settings.pair_contrast_margin) - preference_margin).mean()
    correct_mean = correct_answer_nll.mean()
    alternative_mean = alternative_answer_nll.mean()
    total = (
        float(settings.answer_nll_weight) * correct_mean
        + float(settings.pair_contrast_weight) * contrast_hinge
        + float(settings.source_anchor_weight) * source_anchor_l2
    )
    if total.ndim != 0 or not bool(torch.isfinite(total)):
        raise RuntimeError("V76 objective became nonfinite or nonscalar")
    return total, {
        "correct_answer_nll": correct_mean,
        "alternative_answer_nll": alternative_mean,
        "paired_alternative_margin": preference_margin.mean(),
        "minimum_paired_alternative_margin": preference_margin.min(),
        "positive_preference_sides": (preference_margin > 0.0).sum(),
        "margin_satisfied_sides": (preference_margin >= float(settings.pair_contrast_margin)).sum(),
        "pair_contrast_hinge": contrast_hinge,
        "source_anchor_l2": source_anchor_l2,
    }


def snapshot_source_parameters_v76(
    model: DenseFullSceneContinuousControlV75,
) -> dict[str, torch.Tensor]:
    if type(model) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V76 source snapshot requires the exact V75 reader")
    return {
        name: parameter.detach().float().clone() for name, parameter in model.named_parameters()
    }


def source_weight_anchor_l2_v76(
    model: torch.nn.Module, source_parameters: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Mean per-parameter-element squared displacement from V75 initialization."""

    current = dict(model.named_parameters())
    if set(current) != set(source_parameters) or not current:
        raise ValueError("V76 source-anchor parameter inventory changed")
    squared_sum: torch.Tensor | None = None
    element_count = 0
    for name, parameter in current.items():
        source = source_parameters[name]
        if source.shape != parameter.shape or not bool(torch.isfinite(source).all()):
            raise ValueError(f"V76 source-anchor tensor changed: {name}")
        difference = parameter.float() - source.to(parameter.device).float()
        contribution = difference.square().sum()
        squared_sum = contribution if squared_sum is None else squared_sum + contribution
        element_count += parameter.numel()
    assert squared_sum is not None
    return squared_sum / element_count


def _path_tokens(path: Path) -> set[str]:
    try:
        scoped = path.relative_to(PROJECT_ROOT)
    except ValueError:
        scoped = path
    return {
        token
        for part in scoped.parts
        for token in part.casefold().replace("-", "_").split("_")
        if token
    }


def _guard_input_v76(path: str | Path, purpose: str) -> Path:
    value = Path(path).expanduser()
    source = Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))
    try:
        source.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("V76 inputs must remain below the project root") from error
    cursor = source
    while True:
        if cursor.exists() and cursor.is_symlink():
            raise ValueError(f"V76 {purpose} path cannot traverse a symlink")
        if cursor == PROJECT_ROOT:
            break
        cursor = cursor.parent
    return _guard_training_input(source, purpose)


def _guard_output_v76(path: str | Path, *, suffix: str) -> Path:
    value = Path(path).expanduser()
    destination = Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))
    try:
        destination.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("V76 diagnostic outputs must remain below the project root") from error
    forbidden = sorted(_path_tokens(destination) & _FORBIDDEN_OUTPUT_TOKENS)
    if forbidden:
        raise ValueError(f"V76 output crosses forbidden boundaries: {forbidden}")
    if destination.suffix != suffix:
        raise ValueError(f"V76 output must use the {suffix} suffix")
    cursor = destination.parent
    while True:
        if cursor.exists() and cursor.is_symlink():
            raise ValueError("V76 output path cannot traverse a symlink")
        if cursor == PROJECT_ROOT:
            break
        cursor = cursor.parent
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    return destination


def assert_exact_v75_source_v76(path: str | Path) -> tuple[Path, dict[str, str]]:
    """Authenticate the exact all-gates V75 candidate, not a compatible look-alike."""

    source = _guard_input_v76(path, "exact V75 source candidate")
    digest = _sha256_file(source)
    if digest != V76_INITIAL_CANDIDATE_SHA256:
        raise ValueError("V76 source is not the exact locked V75 candidate")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        fields = frozenset(handle.keys())
    if fields != V75_STATE_FIELDS or metadata != _SOURCE_METADATA:
        raise ValueError("V76 exact V75 source contract changed")
    return source, metadata


def _side_answer_nlls_v76(
    *,
    runtime: Any,
    model: DenseFullSceneContinuousControlV75,
    row: RowV73,
    opposite: RowV73,
    prefixes: Mapping[str, torch.Tensor],
    question_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return own-answer and paired-alternative NLL for one physical scene."""

    if row.question != opposite.question or row.answer == opposite.answer:
        raise ValueError("V76 pair objective requires a changed same-question unit")
    language = runtime.language
    model_dtype = next(language.model.parameters()).dtype
    try:
        scene = prefixes[row.scene_id].to(device=language.device, dtype=model_dtype)
    except KeyError as error:
        raise KeyError(f"V76 lacks immutable prefix for {row.scene_id}") from error
    control = model(scene.float(), question_embedding).control_tokens
    own_batch = _compose_batch(
        runtime=runtime,
        scene_prefix=scene,
        record=row,
        answer=row.answer,
        control_tokens=control,
    )[0]
    alternative_batch = _compose_batch(
        runtime=runtime,
        scene_prefix=scene,
        record=row,
        answer=opposite.answer,
        control_tokens=control,
    )[0]
    stacked = stack_prefix_batches(
        (own_batch, alternative_batch),
        language.device,
        prefix_backend=language.prefix_backend,
    )
    output = forward_prefix_batch(language, stacked)
    if stacked.labels is None:
        raise RuntimeError("V76 answer-contrast batch lacks labels")
    nll = token_normalized_nll(output.logits, stacked.labels)
    if nll.shape != (2,):
        raise RuntimeError("V76 answer-contrast NLL shape changed")
    return nll[0], nll[1]


def _pair_answer_nlls_v76(
    *,
    runtime: Any,
    model: DenseFullSceneContinuousControlV75,
    unit: ChangedUnitV73,
    prefixes: Mapping[str, torch.Tensor],
    question_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two own-answer and two paired-alternative NLL scalars."""

    values = tuple(
        _side_answer_nlls_v76(
            runtime=runtime,
            model=model,
            row=row,
            opposite=opposite,
            prefixes=prefixes,
            question_embedding=question_embedding,
        )
        for row, opposite in ((unit.left, unit.right), (unit.right, unit.left))
    )
    return (
        torch.stack(tuple(value[0] for value in values)),
        torch.stack(tuple(value[1] for value in values)),
    )


def _aggregate_metrics_v76(
    correct: Sequence[float], alternative: Sequence[float], *, margin: float
) -> dict[str, float | int]:
    if len(correct) != len(alternative) or not correct:
        raise ValueError("V76 aggregate metrics require aligned nonempty sides")
    margins = [alternate - own for own, alternate in zip(correct, alternative, strict=True)]
    return {
        "side_count": len(correct),
        "mean_correct_answer_nll": sum(correct) / len(correct),
        "mean_alternative_answer_nll": sum(alternative) / len(alternative),
        "mean_paired_alternative_margin": sum(margins) / len(margins),
        "minimum_paired_alternative_margin": min(margins),
        "positive_preference_sides": sum(value > 0.0 for value in margins),
        "margin_satisfied_sides": sum(value >= margin for value in margins),
        "mean_pair_contrast_hinge": sum(max(0.0, margin - value) for value in margins)
        / len(margins),
    }


def _measure_units_v76(
    *,
    runtime: Any,
    model: DenseFullSceneContinuousControlV75,
    units: Sequence[ChangedUnitV73],
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    settings: V76LossSettings,
) -> dict[str, float | int]:
    model.eval()
    correct: list[float] = []
    alternative: list[float] = []
    with torch.no_grad():
        for unit in units:
            own, paired = _pair_answer_nlls_v76(
                runtime=runtime,
                model=model,
                unit=unit,
                prefixes=prefixes,
                question_embedding=questions[unit.left.question],
            )
            correct.extend(float(value) for value in own.detach().cpu())
            alternative.extend(float(value) for value in paired.detach().cpu())
    return _aggregate_metrics_v76(correct, alternative, margin=settings.pair_contrast_margin)


def _finite_v75_state(model: DenseFullSceneContinuousControlV75) -> dict[str, torch.Tensor]:
    state = {
        key: value.detach().cpu().float().contiguous() for key, value in model.state_dict().items()
    }
    if frozenset(state) != V75_STATE_FIELDS:
        raise ValueError("V76 V75 state inventory changed")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("V76 V75 state became nonfinite")
    return state


def save_v76_diagnostic(
    path: str | Path,
    model: DenseFullSceneContinuousControlV75,
    *,
    optimizer_steps: int,
    selected_unit_count: int,
    cycles: int,
    source_sha256: str = V76_INITIAL_CANDIDATE_SHA256,
) -> dict[str, Any]:
    """Atomically save a minimal, explicitly non-runtime V76 diagnostic."""

    if type(model) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V76 diagnostic requires the exact V75 architecture")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (optimizer_steps, selected_unit_count, cycles)
    ):
        raise ValueError("V76 diagnostic counts must be positive integers")
    if (
        selected_unit_count > EXPECTED_CHANGED_UNIT_COUNT
        or optimizer_steps != selected_unit_count * cycles
    ):
        raise ValueError("V76 diagnostic counts do not describe complete pair cycles")
    if source_sha256 != V76_INITIAL_CANDIDATE_SHA256:
        raise ValueError("V76 diagnostic source hash is not the exact V75 candidate")
    destination = _guard_output_v76(path, suffix=".safetensors")
    zero_audit = assert_dense_reader_exact_zero_scene(model)
    if (
        model.environment_latents != 256
        or model.hidden_size != EXPECTED_HIDDEN_SIZE
        or not zero_audit["all_environment_latents_attended"]
    ):
        raise RuntimeError("V76 full-256-latent contract failed")
    metadata = {
        "artifact": "v76_all_historical_pair_contrast_diagnostic_v1",
        "controller_architecture": "v75",
        "source_candidate_sha256": source_sha256,
        "training_pool_only": "true",
        "historical_train_pairs_only": "true",
        "selected_changed_units": str(selected_unit_count),
        "selected_changed_sides": str(2 * selected_unit_count),
        "exhaustive_historical_changed_units": str(
            selected_unit_count == EXPECTED_CHANGED_UNIT_COUNT
        ).lower(),
        "cycles": str(cycles),
        "optimizer_steps": str(optimizer_steps),
        "held_optimization_rows": "0",
        "runtime_promotion_forbidden_until_gemma_gate": "true",
        "runtime_publication_artifact": "false",
        "numeric_gate_passed": "unverified_after_v76_training",
        "answer_codebook_serialized": "false",
        "environmental_text_inputs": "0",
        "official_validation_loaded": "false",
        "official_test_loaded": "false",
        "oracle_loaded": "false",
        "exact_zero_scene_verified": "true",
        "question_only_output_path_exists": "false",
        "all_256_environment_latents_attended": "true",
        "question_dependent_retrieval": "false",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        state = _finite_v75_state(model)
        save_file(state, temporary, metadata=metadata)
        reloaded = load_file(str(temporary), device="cpu")
        if set(reloaded) != set(state) or any(
            not torch.equal(reloaded[key], state[key]) for key in state
        ):
            raise RuntimeError("V76 diagnostic failed exact tensor reload")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination.relative_to(PROJECT_ROOT)),
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "metadata": metadata,
        "exact_zero_audit": zero_audit,
    }


def _write_report_v76(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = _guard_output_v76(path, suffix=".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
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


def validate_args_v76(args: argparse.Namespace) -> V76LossSettings:
    if (
        isinstance(args.max_units, bool)
        or not isinstance(args.max_units, int)
        or not 1 <= args.max_units <= 40
    ):
        raise ValueError("V76 max_units must be in [1, 40]")
    if (
        isinstance(args.cycles, bool)
        or not isinstance(args.cycles, int)
        or not 1 <= args.cycles <= 12
    ):
        raise ValueError("V76 cycles must be in [1, 12]")
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or args.seed < 0
        or isinstance(args.log_every, bool)
        or not isinstance(args.log_every, int)
        or args.log_every < 1
    ):
        raise ValueError("V76 seed/log interval is invalid")
    for field, upper in (
        ("learning_rate", 1e-3),
        ("gradient_clip_norm", 10.0),
    ):
        value = getattr(args, field)
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 < value <= upper:
            raise ValueError(f"V76 {field} must be in (0, {upper}]")
    if (
        isinstance(args.weight_decay, bool)
        or not math.isfinite(float(args.weight_decay))
        or not 0.0 <= args.weight_decay <= 1.0
    ):
        raise ValueError("V76 weight_decay must be in [0, 1]")
    return V76LossSettings(
        answer_nll_weight=args.answer_nll_weight,
        pair_contrast_weight=args.pair_contrast_weight,
        pair_contrast_margin=args.pair_contrast_margin,
        source_anchor_weight=args.source_anchor_weight,
    )


def run_v76_pair_contrast_screen(args: argparse.Namespace) -> dict[str, Any]:
    """Run the all-historical-pair V76 screen; never called during import/tests."""

    settings = validate_args_v76(args)
    candidate_path, source_metadata = assert_exact_v75_source_v76(args.initial_candidate)
    runtime_config_path = _guard_input_v76(args.runtime_config, "runtime config")
    base_checkpoint = _guard_input_v76(args.base_checkpoint, "base checkpoint")
    v73_config_path = _guard_input_v76(args.v73_config, "V73 config")
    output_candidate = _guard_output_v76(args.output_candidate, suffix=".safetensors")
    output_report = _guard_output_v76(args.output_report, suffix=".json")
    if len({candidate_path, output_candidate, output_report}) != 3:
        raise ValueError("V76 source and outputs must be distinct")

    v73 = load_config_v73(v73_config_path)
    training_qa_path = _guard_input_v76(v73["training_qa"], "training QA")
    prefix_cache_path = _guard_input_v76(v73["prefix_cache"], "immutable prefix cache")
    all_rows = load_training_rows_v73(training_qa_path)
    train_rows, held_rows = split_rows_v73(all_rows)
    units = select_historical_changed_units_v76(train_rows, max_units=args.max_units)
    selected_rows = tuple(row for unit in units for row in (unit.left, unit.right))
    if {row.scene_id for row in selected_rows} & {row.scene_id for row in held_rows}:
        raise RuntimeError("V76 selected a held scene")
    schedule = deterministic_pair_schedule_v76(units, cycles=args.cycles, seed=args.seed)
    prefixes, prefix_manifest = load_prefixes_v73(
        prefix_cache_path, {row.scene_id for row in selected_rows}
    )

    runtime_config, _ = _load_sanitized_runtime_config(runtime_config_path)
    runtime = StaticRuntimePrefixFactory(
        runtime_config, base_checkpoint, selected_rows[0].scene_id
    ).bootstrap
    freeze_audit = freeze_base_runtime(runtime)
    _disable_decoder_checkpointing(runtime.language)
    device = _select_training_device(runtime, args.device)
    loaded, _ = _load_initial_candidate(candidate_path, device)
    if type(loaded) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V76 exact source did not instantiate the exact V75 reader")
    model = loaded
    if model.environment_latents != 256 or model.hidden_size != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("V76 exact V75 source lost the full-256-latent contract")
    source_zero_audit = assert_dense_reader_exact_zero_scene(model)
    trainable_audit = assert_exclusive_dense_reader_trainable_surface(runtime, model)
    if not torch.equal(
        runtime.scene_prefix.detach().cpu().float(),
        prefixes[selected_rows[0].scene_id].detach().cpu().float(),
    ):
        raise ValueError("V76 cached prefix differs from the frozen V54 runtime")
    questions = {
        question: _question_embeddings(runtime, question)
        for question in sorted({unit.left.question for unit in units})
    }
    source_parameters = snapshot_source_parameters_v76(model)
    before = _measure_units_v76(
        runtime=runtime,
        model=model,
        units=units,
        prefixes=prefixes,
        questions=questions,
        settings=settings,
    )

    runtime.language.enable_decoder_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, Any]] = []
    cycle_values: dict[int, dict[str, list[float]]] = {}
    started = time.perf_counter()
    model.train()
    for step, scheduled in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        own_parts: list[torch.Tensor] = []
        alternative_parts: list[torch.Tensor] = []
        # Backpropagate the physical sides separately, then take one atomic
        # optimizer step. This retains only one frozen-Gemma activation graph
        # at a time and is algebraically the same two-side mean objective.
        for row, opposite in (
            (scheduled.unit.left, scheduled.unit.right),
            (scheduled.unit.right, scheduled.unit.left),
        ):
            own, alternative = _side_answer_nlls_v76(
                runtime=runtime,
                model=model,
                row=row,
                opposite=opposite,
                prefixes=prefixes,
                question_embedding=questions[scheduled.unit.left.question],
            )
            side_objective, _ = paired_answer_contrast_objective_v76(
                own.reshape(1),
                alternative.reshape(1),
                own.new_zeros(()),
                settings,
            )
            (0.5 * side_objective).backward()
            own_parts.append(own.detach())
            alternative_parts.append(alternative.detach())
        anchor = source_weight_anchor_l2_v76(model, source_parameters)
        if settings.source_anchor_weight > 0.0:
            (float(settings.source_anchor_weight) * anchor).backward()
        correct_nll = torch.stack(own_parts)
        alternative_nll = torch.stack(alternative_parts)
        loss, diagnostics = paired_answer_contrast_objective_v76(
            correct_nll, alternative_nll, anchor.detach(), settings
        )
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            .detach()
            .float()
            .cpu()
        )
        if not math.isfinite(gradient):
            raise RuntimeError("V76 preclip gradient norm became nonfinite")
        optimizer.step()
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise RuntimeError("V76 optimizer produced nonfinite state")

        own_values = [float(value) for value in correct_nll.detach().cpu()]
        alternate_values = [float(value) for value in alternative_nll.detach().cpu()]
        bucket = cycle_values.setdefault(scheduled.cycle, {"correct": [], "alternative": []})
        bucket["correct"].extend(own_values)
        bucket["alternative"].extend(alternate_values)
        if step == 1 or step % args.log_every == 0 or step == len(schedule):
            event = {
                "step": step,
                "optimizer_steps": len(schedule),
                "cycle": scheduled.cycle,
                "step_in_cycle": scheduled.step_in_cycle,
                "pair_id": scheduled.unit.pair_id,
                "question_key": scheduled.unit.question_key,
                "correct_answer_nll": float(diagnostics["correct_answer_nll"].detach().cpu()),
                "alternative_answer_nll": float(
                    diagnostics["alternative_answer_nll"].detach().cpu()
                ),
                "paired_alternative_margin": float(
                    diagnostics["paired_alternative_margin"].detach().cpu()
                ),
                "pair_contrast_hinge": float(diagnostics["pair_contrast_hinge"].detach().cpu()),
                "source_anchor_l2": float(anchor.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
                "preclip_gradient_norm": gradient,
            }
            history.append(event)
            print(
                json.dumps({"event": "v76_pair_contrast_train", **event}, sort_keys=True),
                flush=True,
            )

    _disable_decoder_checkpointing(runtime.language)
    model.eval()
    after = _measure_units_v76(
        runtime=runtime,
        model=model,
        units=units,
        prefixes=prefixes,
        questions=questions,
        settings=settings,
    )
    final_anchor = float(source_weight_anchor_l2_v76(model, source_parameters).detach().cpu())
    after_zero_audit = assert_dense_reader_exact_zero_scene(model)
    diagnostic = save_v76_diagnostic(
        output_candidate,
        model,
        optimizer_steps=len(schedule),
        selected_unit_count=len(units),
        cycles=args.cycles,
    )
    cycle_metrics = {
        str(cycle): _aggregate_metrics_v76(
            values["correct"],
            values["alternative"],
            margin=settings.pair_contrast_margin,
        )
        for cycle, values in sorted(cycle_values.items())
    }
    train_objective_improved = float(after["mean_correct_answer_nll"]) < float(
        before["mean_correct_answer_nll"]
    ) and float(after["mean_paired_alternative_margin"]) > float(
        before["mean_paired_alternative_margin"]
    )
    held_smoke_authorized = train_objective_improved and len(units) == EXPECTED_CHANGED_UNIT_COUNT
    report = {
        "artifact": "v76_all_historical_pair_contrast_screen_v1",
        "scope": {
            "training_pool_only": True,
            "historical_train_pairs_only": True,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
            "checkpoint_published": False,
            "runtime_promotion_authorized": False,
            "held_optimization_rows": 0,
        },
        "source_candidate": {
            "path": str(candidate_path.relative_to(PROJECT_ROOT)),
            "sha256": V76_INITIAL_CANDIDATE_SHA256,
            "metadata": source_metadata,
        },
        "selection": {
            "available_changed_units": EXPECTED_CHANGED_UNIT_COUNT,
            "selected_changed_units": len(units),
            "selected_changed_sides": len(selected_rows),
            "default_exhaustive_selection": args.max_units == EXPECTED_CHANGED_UNIT_COUNT,
            "selection_salt": SELECTION_SALT,
            "change_type_counts": dict(Counter(unit.change_type for unit in units)),
            "unit_keys": [
                {
                    "pair_id": unit.pair_id,
                    "question_key": unit.question_key,
                    "side_keys": [list(unit.left.key), list(unit.right.key)],
                }
                for unit in units
            ],
        },
        "schedule": {
            "cycles": args.cycles,
            "optimizer_steps": len(schedule),
            "atomic_two_scene_units": True,
            "each_selected_unit_once_per_cycle": True,
            "seed": args.seed,
        },
        "loss_settings": asdict(settings),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
        },
        "before": before,
        "per_cycle_pre_update": cycle_metrics,
        "after": after,
        "mean_correct_answer_nll_delta": float(after["mean_correct_answer_nll"])
        - float(before["mean_correct_answer_nll"]),
        "mean_paired_alternative_margin_delta": float(after["mean_paired_alternative_margin"])
        - float(before["mean_paired_alternative_margin"]),
        "final_source_weight_anchor_l2": final_anchor,
        "train_objective_improved": train_objective_improved,
        "held_smoke_authorized": held_smoke_authorized,
        "elapsed_training_seconds": time.perf_counter() - started,
        "diagnostic_candidate": diagnostic,
        "base_freeze_audit": freeze_audit,
        "exclusive_trainable_audit": trainable_audit,
        "source_exact_zero_audit": source_zero_audit,
        "after_exact_zero_audit": after_zero_audit,
        "fit_history": history,
        "prefix_manifest_base_checkpoint_sha256": prefix_manifest["base_checkpoint_sha256"],
    }
    _write_report_v76(output_report, report)
    print(
        json.dumps(
            {
                "event": "v76_pair_contrast_complete",
                "output_candidate": diagnostic["path"],
                "output_report": str(output_report.relative_to(PROJECT_ROOT)),
                "selected_changed_units": len(units),
                "before_correct_answer_nll": before["mean_correct_answer_nll"],
                "after_correct_answer_nll": after["mean_correct_answer_nll"],
                "before_paired_alternative_margin": before["mean_paired_alternative_margin"],
                "after_paired_alternative_margin": after["mean_paired_alternative_margin"],
                "train_objective_improved": train_objective_improved,
                "held_smoke_authorized": held_smoke_authorized,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-candidate", default=V76_INITIAL_CANDIDATE)
    parser.add_argument("--runtime-config", default="configs/runtime/gemma4_v54.yaml")
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
    )
    parser.add_argument(
        "--v73-config",
        default="configs/experiments/gemma4_v73_fullscene_controller.yaml",
    )
    parser.add_argument("--max-units", type=int, default=EXPECTED_CHANGED_UNIT_COUNT)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=760176)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--answer-nll-weight", type=float, default=1.0)
    parser.add_argument("--pair-contrast-weight", type=float, default=2.0)
    parser.add_argument("--pair-contrast-margin", type=float, default=0.5)
    parser.add_argument("--source-anchor-weight", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--output-candidate",
        default=("reports/gemma4/artifacts/v76_v75_all40_pair_contrast_diagnostic.safetensors"),
    )
    parser.add_argument(
        "--output-report",
        default="reports/gemma4/metrics/v76_v75_all40_pair_contrast_screen.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_v76_pair_contrast_screen(args)
    return 0


__all__ = [
    "EXPECTED_CHANGED_SIDE_COUNT",
    "EXPECTED_CHANGED_UNIT_COUNT",
    "V76_INITIAL_CANDIDATE",
    "V76_INITIAL_CANDIDATE_SHA256",
    "ScheduledPairV76",
    "V76LossSettings",
    "assert_exact_v75_source_v76",
    "build_parser",
    "deterministic_pair_schedule_v76",
    "paired_answer_contrast_objective_v76",
    "run_v76_pair_contrast_screen",
    "save_v76_diagnostic",
    "select_historical_changed_units_v76",
    "snapshot_source_parameters_v76",
    "source_weight_anchor_l2_v76",
    "validate_args_v76",
]


if __name__ == "__main__":
    raise SystemExit(main())
