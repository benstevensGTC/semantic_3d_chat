"""Quarantined V77 all-row historical answer repair for the V75 reader.

V77 starts from the exact V75 answer-NLL candidate that passed the broad
internal Gemma behavior screen.  It trains only the compact V75 continuous
scene reader while the Gemma decoder, the V54 scene stack, and every immutable
pre-question full-scene prefix remain frozen.

The default schedule consumes all 576 rows from the preregistered historical
optimization split once.  Rows are interleaved round-robin by canonical answer
class and by a hash of the question template so the frequent ``1``/yes/no
targets cannot dominate long contiguous stretches.  Each row receives:

* token-normalized answer NLL through frozen Gemma;
* a deterministic same-answer-type negative-answer NLL margin;
* an optional V76-style paired-answer margin on changed counterfactual rows;
* optional source-control-output and source-weight L2 anchors.

Question and answer text exist only inside this supervised training process.
The output safetensors contains the six V75 numeric tensors plus fixed audit
metadata.  It never contains an answer codebook, questions, answers, labels, or
runtime publication metadata, and it is not directly loadable as a production
checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
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
    _question_embeddings,
    assert_dense_reader_exact_zero_scene,
    assert_exclusive_dense_reader_trainable_surface,
)
from semantic_3d_chat.training.finetune_v76_pair_contrast import (
    snapshot_source_parameters_v76,
    source_weight_anchor_l2_v76,
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
    EXPECTED_TRAIN_ROWS,
    HELD_PAIR_IDS,
    TRAIN_PAIR_IDS,
    RowV73,
    _sha256_file,
    changed_units_v73,
    load_config_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)

V77_INITIAL_CANDIDATE: Final[str] = (
    "reports/gemma4/artifacts/v75_gemma_nll_balanced_train_diagnostic.safetensors"
)
V77_INITIAL_CANDIDATE_SHA256: Final[str] = (
    "d01275538489b3493a8e1ff080109d1db46832be6ca2a26f6d89d161c597188a"
)
EXPECTED_ANSWER_CLASS_COUNT: Final[int] = 28
EXPECTED_QUESTION_TEMPLATE_COUNT: Final[int] = 96
EXPECTED_CHANGED_SIDE_COUNT: Final[int] = 80
SCHEDULE_SALT: Final[str] = "semantic_3d_chat.v77.answer_template_balanced.v1"
NEGATIVE_SALT: Final[str] = "semantic_3d_chat.v77.canonical_negative.v1"

_SOURCE_METADATA: Final[dict[str, str]] = {
    "answer_codebook_serialized": "false",
    "artifact": "v75_historical_train_gemma_nll_diagnostic_v1",
    "controller_architecture": "v75",
    "environmental_text_inputs": "0",
    "exact_zero_scene_verified": "true",
    "held_optimization_rows": "0",
    "historical_train_pairs_only": "true",
    "numeric_gate_passed": "unverified_after_gemma_nll",
    "official_test_loaded": "false",
    "official_validation_loaded": "false",
    "optimizer_steps": "54",
    "oracle_loaded": "false",
    "question_only_output_path_exists": "false",
    "runtime_promotion_forbidden_until_gemma_gate": "true",
    "runtime_publication_artifact": "false",
    "source_candidate_sha256": ("182481dd77645cd2a467b3585dd7b060fcea578cc013eebc21486e1915ce9c17"),
    "train_behavior_improved": "true",
    "training_pool_only": "true",
}
_FORBIDDEN_PATH_TOKENS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "validate", "test", "deferred", "final"}
)
_FORBIDDEN_OUTPUT_TOKENS: Final[frozenset[str]] = _FORBIDDEN_PATH_TOKENS | {
    "runtime",
    "release",
    "production",
}


@dataclass(frozen=True)
class V77LossSettings:
    """Weights for one historical row; every term is training-only."""

    answer_nll_weight: float = 1.0
    negative_margin_weight: float = 0.25
    negative_margin: float = 0.50
    changed_pair_margin_weight: float = 0.25
    changed_pair_margin: float = 0.50
    source_output_anchor_weight: float = 0.05
    source_weight_anchor_weight: float = 0.01

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"V77 {field} must be finite and nonnegative")
        if self.answer_nll_weight <= 0.0:
            raise ValueError("V77 answer_nll_weight must be positive")


@dataclass(frozen=True)
class ScheduledRowV77:
    cycle: int
    step_in_cycle: int
    row: RowV73
    template_id: str
    negative_answer: str


def _absolute_below_project(path: str | Path) -> Path:
    value = Path(path).expanduser()
    absolute = Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))
    try:
        absolute.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("V77 paths must remain below the project root") from error
    return absolute


def _path_tokens(path: Path) -> set[str]:
    try:
        scoped = path.relative_to(PROJECT_ROOT)
    except ValueError:
        scoped = Path(path.name)
    return {
        token
        for part in scoped.parts
        for token in re.split(r"[^a-z0-9]+", part.casefold())
        if token
    }


def _reject_symlink_components(path: Path, *, include_leaf: bool) -> None:
    stop = path if include_leaf else path.parent
    cursor = stop
    while True:
        if cursor.is_symlink():
            raise ValueError("V77 paths cannot traverse symlinks")
        if cursor == PROJECT_ROOT:
            return
        if cursor.parent == cursor:
            raise ValueError("V77 path ancestry escaped the project root")
        cursor = cursor.parent


def _guard_input_v77(path: str | Path, purpose: str) -> Path:
    source = _absolute_below_project(path)
    forbidden = sorted(_path_tokens(source) & _FORBIDDEN_PATH_TOKENS)
    if forbidden:
        raise ValueError(f"V77 {purpose} crosses forbidden path tokens: {forbidden}")
    _reject_symlink_components(source, include_leaf=True)
    if not source.exists() or source.is_symlink():
        raise FileNotFoundError(f"V77 {purpose} is unavailable: {source}")
    return source


def _guard_output_v77(path: str | Path, *, suffix: str) -> Path:
    destination = _absolute_below_project(path)
    forbidden = sorted(_path_tokens(destination) & _FORBIDDEN_OUTPUT_TOKENS)
    if forbidden:
        raise ValueError(f"V77 output crosses forbidden path tokens: {forbidden}")
    if destination.suffix != suffix:
        raise ValueError(f"V77 output must use the {suffix} suffix")
    _reject_symlink_components(destination, include_leaf=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    return destination


def question_template_id_v77(question: str) -> str:
    """Return a stable opaque template ID without retaining question text."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("V77 question template cannot be empty")
    normalized = " ".join(question.casefold().split())
    return "template_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _row_digest(row: RowV73, *, seed: int, cycle: int) -> str:
    return hashlib.sha256(
        (
            f"{SCHEDULE_SALT}|{seed}|{cycle}|{row.answer_class}|"
            f"{question_template_id_v77(row.question)}|{row.scene_id}|{row.question_id}"
        ).encode()
    ).hexdigest()


def _balanced_order_v77(rows: Sequence[RowV73], *, seed: int, cycle: int) -> tuple[RowV73, ...]:
    """Round-robin answer classes, and round-robin templates inside each class."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("V77 schedule seed must be a nonnegative integer")
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0:
        raise ValueError("V77 schedule cycle must be a nonnegative integer")
    if not rows or len({row.key for row in rows}) != len(rows):
        raise ValueError("V77 schedule requires nonempty unique rows")
    if any(row.pair_id not in TRAIN_PAIR_IDS or row.pair_id in HELD_PAIR_IDS for row in rows):
        raise ValueError("V77 schedule escaped the historical optimization split")

    grouped: dict[str, dict[str, list[RowV73]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row.answer_class][question_template_id_v77(row.question)].append(row)
    for templates in grouped.values():
        for members in templates.values():
            members.sort(key=lambda row: _row_digest(row, seed=seed, cycle=cycle))

    class_order = sorted(
        grouped,
        key=lambda answer_class: hashlib.sha256(
            f"{SCHEDULE_SALT}|class|{seed}|{cycle}|{answer_class}".encode()
        ).hexdigest(),
    )
    template_orders: dict[str, list[str]] = {}
    template_offsets: dict[str, int] = {}
    for answer_class in class_order:
        template_orders[answer_class] = sorted(
            grouped[answer_class],
            key=lambda template: hashlib.sha256(
                f"{SCHEDULE_SALT}|template|{seed}|{cycle}|{answer_class}|{template}".encode()
            ).hexdigest(),
        )
        template_offsets[answer_class] = 0

    result: list[RowV73] = []
    remaining = len(rows)
    while remaining:
        made_progress = False
        for answer_class in class_order:
            templates = template_orders[answer_class]
            if not templates:
                continue
            offset = template_offsets[answer_class] % len(templates)
            chosen_index: int | None = None
            for delta in range(len(templates)):
                index = (offset + delta) % len(templates)
                if grouped[answer_class][templates[index]]:
                    chosen_index = index
                    break
            if chosen_index is None:
                continue
            template = templates[chosen_index]
            result.append(grouped[answer_class][template].pop())
            template_offsets[answer_class] = chosen_index + 1
            remaining -= 1
            made_progress = True
        if not made_progress:
            raise RuntimeError("V77 balanced scheduler stalled")
    if len(result) != len(rows) or {row.key for row in result} != {row.key for row in rows}:
        raise RuntimeError("V77 balanced scheduler lost or duplicated rows")
    return tuple(result)


def select_balanced_historical_rows_v77(
    train_rows: Sequence[RowV73], *, max_rows: int, seed: int
) -> tuple[RowV73, ...]:
    """Select an answer/template-interleaved subset of the locked 576 rows."""

    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"V77 requires exactly {EXPECTED_TRAIN_ROWS} historical rows")
    if (
        isinstance(max_rows, bool)
        or not isinstance(max_rows, int)
        or not 1 <= max_rows <= EXPECTED_TRAIN_ROWS
    ):
        raise ValueError(f"V77 max_rows must be in [1, {EXPECTED_TRAIN_ROWS}]")
    answer_classes = {row.answer_class for row in train_rows}
    templates = {question_template_id_v77(row.question) for row in train_rows}
    if len(answer_classes) != EXPECTED_ANSWER_CLASS_COUNT:
        raise ValueError("V77 historical answer-class inventory changed")
    if len(templates) != EXPECTED_QUESTION_TEMPLATE_COUNT:
        raise ValueError("V77 historical question-template inventory changed")
    return _balanced_order_v77(train_rows, seed=seed, cycle=0)[:max_rows]


def canonical_alternatives_v77(
    train_rows: Sequence[RowV73],
) -> dict[str, tuple[str, ...]]:
    """Build an in-memory same-answer-type negative pool from canonical targets."""

    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError("V77 canonical alternatives require all historical rows")
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        grouped[row.answer_type].add(row.answer)
    result = {key: tuple(sorted(values)) for key, values in sorted(grouped.items())}
    if not result or any(len(values) < 2 for values in result.values()):
        raise ValueError("V77 every answer type requires at least two canonical alternatives")
    return result


def sample_negative_answer_v77(
    row: RowV73,
    alternatives: Mapping[str, Sequence[str]],
    *,
    seed: int,
    cycle: int,
) -> str:
    """Choose one deterministic canonical alternative, never the correct answer."""

    candidates = tuple(
        value for value in alternatives.get(row.answer_type, ()) if value != row.answer
    )
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("V77 row lacks unique same-type negative answers")
    digest = hashlib.sha256(
        f"{NEGATIVE_SALT}|{seed}|{cycle}|{row.scene_id}|{row.question_id}".encode()
    ).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def deterministic_training_schedule_v77(
    selected_rows: Sequence[RowV73],
    alternatives: Mapping[str, Sequence[str]],
    *,
    cycles: int,
    seed: int,
) -> tuple[ScheduledRowV77, ...]:
    if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= 6:
        raise ValueError("V77 cycles must be an integer in [1, 6]")
    if not selected_rows or len({row.key for row in selected_rows}) != len(selected_rows):
        raise ValueError("V77 schedule requires unique selected rows")
    result: list[ScheduledRowV77] = []
    for cycle in range(cycles):
        order = _balanced_order_v77(selected_rows, seed=seed, cycle=cycle + 1)
        result.extend(
            ScheduledRowV77(
                cycle=cycle + 1,
                step_in_cycle=index + 1,
                row=row,
                template_id=question_template_id_v77(row.question),
                negative_answer=sample_negative_answer_v77(
                    row, alternatives, seed=seed, cycle=cycle + 1
                ),
            )
            for index, row in enumerate(order)
        )
    return tuple(result)


def changed_opposites_v77(train_rows: Sequence[RowV73]) -> dict[tuple[str, str], RowV73]:
    units = changed_units_v73(train_rows)
    result: dict[tuple[str, str], RowV73] = {}
    for unit in units:
        result[unit.left.key] = unit.right
        result[unit.right.key] = unit.left
    if len(result) != EXPECTED_CHANGED_SIDE_COUNT:
        raise ValueError("V77 changed-side inventory changed")
    return result


def row_objective_v77(
    *,
    correct_answer_nll: torch.Tensor,
    negative_answer_nll: torch.Tensor,
    changed_pair_answer_nll: torch.Tensor | None,
    source_output_mse: torch.Tensor,
    settings: V77LossSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine token-normalized NLL, negative margins, and output anchoring."""

    scalars = (correct_answer_nll, negative_answer_nll, source_output_mse)
    if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in scalars):
        raise ValueError("V77 row objective inputs must be finite scalars")
    if any(bool(value < 0.0) for value in scalars):
        raise ValueError("V77 row objective inputs cannot be negative")
    if changed_pair_answer_nll is not None and (
        changed_pair_answer_nll.ndim != 0
        or not bool(torch.isfinite(changed_pair_answer_nll))
        or bool(changed_pair_answer_nll < 0.0)
    ):
        raise ValueError("V77 changed-pair answer NLL must be a finite nonnegative scalar")

    negative_hinge = F.relu(
        float(settings.negative_margin) + correct_answer_nll - negative_answer_nll
    )
    pair_hinge = correct_answer_nll.new_zeros(())
    if changed_pair_answer_nll is not None:
        pair_hinge = F.relu(
            float(settings.changed_pair_margin) + correct_answer_nll - changed_pair_answer_nll
        )
    total = (
        float(settings.answer_nll_weight) * correct_answer_nll
        + float(settings.negative_margin_weight) * negative_hinge
        + float(settings.changed_pair_margin_weight) * pair_hinge
        + float(settings.source_output_anchor_weight) * source_output_mse
    )
    if total.ndim != 0 or not bool(torch.isfinite(total)):
        raise RuntimeError("V77 row objective became nonfinite")
    return total, {
        "correct_answer_nll": correct_answer_nll,
        "negative_answer_nll": negative_answer_nll,
        "negative_answer_margin": negative_answer_nll - correct_answer_nll,
        "negative_margin_hinge": negative_hinge,
        "changed_pair_margin_hinge": pair_hinge,
        "source_output_mse": source_output_mse,
    }


def assert_exact_v75_nll_source_v77(path: str | Path) -> tuple[Path, dict[str, str]]:
    """Authenticate the exact successful d012 V75-NLL candidate."""

    source = _guard_input_v77(path, "exact V75-NLL source candidate")
    if not source.is_file() or _sha256_file(source) != V77_INITIAL_CANDIDATE_SHA256:
        raise ValueError("V77 source is not the exact locked d012 V75-NLL candidate")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        fields = frozenset(handle.keys())
    if fields != V75_STATE_FIELDS or metadata != _SOURCE_METADATA:
        raise ValueError("V77 exact V75-NLL source contract changed")
    return source, metadata


def load_exact_v75_nll_source_v77(
    path: str | Path, device: torch.device
) -> tuple[DenseFullSceneContinuousControlV75, dict[str, str]]:
    source, metadata = assert_exact_v75_nll_source_v77(path)
    state = load_file(str(source), device="cpu")
    expected_shapes = {
        "output_basis": (112, EXPECTED_HIDDEN_SIZE),
        "key.weight": (128, EXPECTED_HIDDEN_SIZE),
        "value.weight": (128, EXPECTED_HIDDEN_SIZE),
        "query.weight": (512, EXPECTED_HIDDEN_SIZE),
        "coefficient_hidden.weight": (768, 512),
        "coefficient_output.weight": (448, 768),
    }
    if {key: tuple(value.shape) for key, value in state.items()} != expected_shapes:
        raise ValueError("V77 exact source tensor shapes changed")
    if any(
        not value.is_floating_point() or not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("V77 exact source contains a nonfinite or nonfloat tensor")
    model = DenseFullSceneContinuousControlV75(
        EXPECTED_HIDDEN_SIZE,
        state["output_basis"],
        environment_latents=256,
        query_count=4,
        model_dimension=128,
        coefficient_decoder_hidden_dimension=768,
    )
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32)
    assert_dense_reader_exact_zero_scene(model)
    return model, metadata


def _candidate_nlls_v77(
    *,
    runtime: Any,
    model: DenseFullSceneContinuousControlV75,
    source_model: DenseFullSceneContinuousControlV75,
    scheduled: ScheduledRowV77,
    prefixes: Mapping[str, torch.Tensor],
    question_embedding: torch.Tensor,
    opposite: RowV73 | None,
    settings: V77LossSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    row = scheduled.row
    language = runtime.language
    model_dtype = next(language.model.parameters()).dtype
    try:
        scene = prefixes[row.scene_id].to(device=language.device, dtype=model_dtype)
    except KeyError as error:
        raise KeyError(f"V77 lacks immutable prefix for {row.scene_id}") from error
    control = model(scene.float(), question_embedding).control_tokens
    with torch.no_grad():
        source_control = source_model(scene.float(), question_embedding).control_tokens
    source_output_mse = F.mse_loss(control.float(), source_control.float())

    answer_candidates = [row.answer, scheduled.negative_answer]
    pair_index: int | None = None
    if settings.changed_pair_margin_weight > 0.0 and opposite is not None:
        if row.question != opposite.question or row.answer == opposite.answer:
            raise ValueError("V77 changed-pair auxiliary requires a changed same-question unit")
        if opposite.answer == scheduled.negative_answer:
            pair_index = 1
        else:
            pair_index = len(answer_candidates)
            answer_candidates.append(opposite.answer)
    batches = tuple(
        _compose_batch(
            runtime=runtime,
            scene_prefix=scene,
            record=row,
            answer=answer,
            control_tokens=control,
        )[0]
        for answer in answer_candidates
    )
    stacked = stack_prefix_batches(
        batches,
        language.device,
        prefix_backend=language.prefix_backend,
    )
    output = forward_prefix_batch(language, stacked)
    if stacked.labels is None:
        raise RuntimeError("V77 candidate batch lacks answer labels")
    nll = token_normalized_nll(output.logits, stacked.labels)
    if nll.shape != (len(answer_candidates),):
        raise RuntimeError("V77 candidate NLL shape changed")
    pair_nll = None if pair_index is None else nll[pair_index]
    return nll[0], nll[1], pair_nll, source_output_mse


def _measurement_v77(
    *,
    runtime: Any,
    model: DenseFullSceneContinuousControlV75,
    source_model: DenseFullSceneContinuousControlV75,
    scheduled_rows: Sequence[ScheduledRowV77],
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    opposites: Mapping[tuple[str, str], RowV73],
    settings: V77LossSettings,
) -> dict[str, float | int]:
    model.eval()
    correct: list[float] = []
    negative: list[float] = []
    pair_correct: list[float] = []
    pair_alternative: list[float] = []
    output_mse: list[float] = []
    with torch.no_grad():
        for scheduled in scheduled_rows:
            own, wrong, paired, anchor = _candidate_nlls_v77(
                runtime=runtime,
                model=model,
                source_model=source_model,
                scheduled=scheduled,
                prefixes=prefixes,
                question_embedding=questions[scheduled.row.question],
                opposite=opposites.get(scheduled.row.key),
                settings=settings,
            )
            correct.append(float(own.detach().cpu()))
            negative.append(float(wrong.detach().cpu()))
            output_mse.append(float(anchor.detach().cpu()))
            if paired is not None:
                pair_correct.append(correct[-1])
                pair_alternative.append(float(paired.detach().cpu()))
    margins = [wrong - own for own, wrong in zip(correct, negative, strict=True)]
    pair_margins = [wrong - own for own, wrong in zip(pair_correct, pair_alternative, strict=True)]
    return {
        "row_count": len(correct),
        "mean_correct_answer_nll": sum(correct) / len(correct),
        "mean_negative_answer_nll": sum(negative) / len(negative),
        "mean_negative_answer_margin": sum(margins) / len(margins),
        "negative_margin_satisfied_rows": sum(
            margin >= settings.negative_margin for margin in margins
        ),
        "changed_pair_side_count": len(pair_margins),
        "mean_changed_pair_answer_margin": (
            sum(pair_margins) / len(pair_margins) if pair_margins else 0.0
        ),
        "changed_pair_margin_satisfied_sides": sum(
            margin >= settings.changed_pair_margin for margin in pair_margins
        ),
        "mean_source_output_mse": sum(output_mse) / len(output_mse),
    }


def _finite_v75_state_v77(
    model: DenseFullSceneContinuousControlV75,
) -> dict[str, torch.Tensor]:
    if type(model) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V77 diagnostic requires the exact V75 architecture")
    state = {
        key: value.detach().cpu().float().contiguous() for key, value in model.state_dict().items()
    }
    if frozenset(state) != V75_STATE_FIELDS:
        raise ValueError("V77 output tensor inventory changed")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("V77 output state became nonfinite")
    return state


def save_v77_diagnostic(
    path: str | Path,
    model: DenseFullSceneContinuousControlV75,
    *,
    selected_row_count: int,
    cycles: int,
    optimizer_steps: int,
    source_sha256: str = V77_INITIAL_CANDIDATE_SHA256,
) -> dict[str, Any]:
    """Atomically save only numeric V75 tensors and fixed non-runtime metadata."""

    for field, value in {
        "selected_row_count": selected_row_count,
        "cycles": cycles,
        "optimizer_steps": optimizer_steps,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"V77 {field} must be a positive integer")
    if not 1 <= selected_row_count <= EXPECTED_TRAIN_ROWS:
        raise ValueError("V77 selected row count exceeds the historical split")
    if source_sha256 != V77_INITIAL_CANDIDATE_SHA256:
        raise ValueError("V77 diagnostic source is not the exact d012 candidate")
    destination = _guard_output_v77(path, suffix=".safetensors")
    zero_audit = assert_dense_reader_exact_zero_scene(model)
    if model.environment_latents != 256 or model.hidden_size != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("V77 full-256-latent scene contract failed")
    metadata = {
        "artifact": "v77_all_historical_answer_repair_diagnostic_v1",
        "controller_architecture": "v75",
        "source_candidate_sha256": source_sha256,
        "training_pool_only": "true",
        "historical_train_pairs_only": "true",
        "selected_historical_rows": str(selected_row_count),
        "exhaustive_576_row_selection": str(selected_row_count == EXPECTED_TRAIN_ROWS).lower(),
        "cycles": str(cycles),
        "optimizer_steps": str(optimizer_steps),
        "held_optimization_rows": "0",
        "runtime_promotion_forbidden_until_gemma_gate": "true",
        "runtime_publication_artifact": "false",
        "answer_codebook_serialized": "false",
        "negative_answer_codebook_serialized": "false",
        "questions_or_answers_serialized": "false",
        "environmental_text_inputs_at_inference": "0",
        "official_validation_loaded": "false",
        "official_test_loaded": "false",
        "deferred_final_loaded": "false",
        "oracle_loaded": "false",
        "exact_zero_scene_verified": "true",
        "question_only_output_path_exists": "false",
        "all_256_environment_latents_attended": "true",
        "question_dependent_retrieval": "false",
    }
    state = _finite_v75_state_v77(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        save_file(state, temporary, metadata=metadata)
        reloaded = load_file(str(temporary), device="cpu")
        if set(reloaded) != set(state) or any(
            not torch.equal(reloaded[key], state[key]) for key in state
        ):
            raise RuntimeError("V77 diagnostic failed exact tensor reload")
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


def _write_report_v77(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = _guard_output_v77(path, suffix=".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
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


def validate_args_v77(args: argparse.Namespace) -> V77LossSettings:
    integer_ranges = {
        "max_rows": (1, EXPECTED_TRAIN_ROWS),
        "cycles": (1, 6),
        "gradient_accumulation_rows": (1, 64),
        "measurement_rows": (1, EXPECTED_TRAIN_ROWS),
        "log_every": (1, 10_000),
    }
    for field, (minimum, maximum) in integer_ranges.items():
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"V77 {field} must be in [{minimum}, {maximum}]")
    if args.measurement_rows > args.max_rows:
        raise ValueError("V77 measurement_rows cannot exceed max_rows")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("V77 seed must be a nonnegative integer")
    for field, upper in (("learning_rate", 1e-3), ("gradient_clip_norm", 10.0)):
        value = getattr(args, field)
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 < value <= upper:
            raise ValueError(f"V77 {field} must be in (0, {upper}]")
    if (
        isinstance(args.weight_decay, bool)
        or not math.isfinite(float(args.weight_decay))
        or not 0.0 <= args.weight_decay <= 1.0
    ):
        raise ValueError("V77 weight_decay must be in [0, 1]")
    return V77LossSettings(
        answer_nll_weight=args.answer_nll_weight,
        negative_margin_weight=args.negative_margin_weight,
        negative_margin=args.negative_margin,
        changed_pair_margin_weight=args.changed_pair_margin_weight,
        changed_pair_margin=args.changed_pair_margin,
        source_output_anchor_weight=args.source_output_anchor_weight,
        source_weight_anchor_weight=args.source_weight_anchor_weight,
    )


def _selection_summary_v77(rows: Sequence[RowV73]) -> dict[str, Any]:
    class_counts = Counter(row.answer_class for row in rows)
    template_counts = Counter(question_template_id_v77(row.question) for row in rows)
    type_counts = Counter(row.answer_type for row in rows)
    return {
        "selected_rows": len(rows),
        "distinct_answer_classes": len(class_counts),
        "distinct_question_templates": len(template_counts),
        "minimum_rows_per_observed_answer_class": min(class_counts.values()),
        "maximum_rows_per_observed_answer_class": max(class_counts.values()),
        "minimum_rows_per_observed_template": min(template_counts.values()),
        "maximum_rows_per_observed_template": max(template_counts.values()),
        "answer_type_counts": dict(sorted(type_counts.items())),
        "answer_class_values_serialized": False,
        "question_template_text_serialized": False,
        "schedule_salt": SCHEDULE_SALT,
        "negative_sampling_salt": NEGATIVE_SALT,
    }


def run_v77_historical_repair(args: argparse.Namespace) -> dict[str, Any]:
    """Run V77 locally. Importing this module never initializes Gemma."""

    settings = validate_args_v77(args)
    source_path, source_metadata = assert_exact_v75_nll_source_v77(args.initial_candidate)
    runtime_config_path = _guard_input_v77(args.runtime_config, "runtime config")
    base_checkpoint = _guard_input_v77(args.base_checkpoint, "base checkpoint")
    v73_config_path = _guard_input_v77(args.v73_config, "V73 config")
    output_candidate = _guard_output_v77(args.output_candidate, suffix=".safetensors")
    output_report = _guard_output_v77(args.output_report, suffix=".json")
    if len({source_path, output_candidate, output_report}) != 3:
        raise ValueError("V77 source and outputs must be distinct")

    v73 = load_config_v73(v73_config_path)
    training_qa_path = _guard_input_v77(v73["training_qa"], "historical training QA")
    prefix_cache_path = _guard_input_v77(v73["prefix_cache"], "immutable prefix cache")
    all_rows = load_training_rows_v73(training_qa_path)
    train_rows, held_rows = split_rows_v73(all_rows)
    selected = select_balanced_historical_rows_v77(
        train_rows, max_rows=args.max_rows, seed=args.seed
    )
    if {row.scene_id for row in selected} & {row.scene_id for row in held_rows}:
        raise RuntimeError("V77 selected an internal held scene")
    alternatives = canonical_alternatives_v77(train_rows)
    opposites = changed_opposites_v77(train_rows)
    schedule = deterministic_training_schedule_v77(
        selected, alternatives, cycles=args.cycles, seed=args.seed
    )
    measurement_schedule = deterministic_training_schedule_v77(
        selected[: args.measurement_rows], alternatives, cycles=1, seed=args.seed + 77
    )
    prefixes, prefix_manifest = load_prefixes_v73(
        prefix_cache_path, {row.scene_id for row in selected}
    )

    runtime_config, _ = _load_sanitized_runtime_config(runtime_config_path)
    runtime = StaticRuntimePrefixFactory(
        runtime_config, base_checkpoint, selected[0].scene_id
    ).bootstrap
    freeze_audit = freeze_base_runtime(runtime)
    _disable_decoder_checkpointing(runtime.language)
    device = _select_training_device(runtime, args.device)
    model, _ = load_exact_v75_nll_source_v77(source_path, device)
    source_model = copy.deepcopy(model).eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    source_zero_audit = assert_dense_reader_exact_zero_scene(model)
    trainable_audit = assert_exclusive_dense_reader_trainable_surface(runtime, model)
    if not torch.equal(
        runtime.scene_prefix.detach().cpu().float(),
        prefixes[selected[0].scene_id].detach().cpu().float(),
    ):
        raise ValueError("V77 cached prefix differs from the frozen V54 runtime")
    questions = {
        question: _question_embeddings(runtime, question)
        for question in sorted({row.question for row in selected})
    }
    source_parameters = snapshot_source_parameters_v76(model)
    before = _measurement_v77(
        runtime=runtime,
        model=model,
        source_model=source_model,
        scheduled_rows=measurement_schedule,
        prefixes=prefixes,
        questions=questions,
        opposites=opposites,
        settings=settings,
    )

    runtime.language.enable_decoder_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    fit_history: list[dict[str, Any]] = []
    training_correct_nll: list[float] = []
    training_negative_margin: list[float] = []
    optimizer_steps = 0
    started = time.perf_counter()
    model.train()
    for offset in range(0, len(schedule), args.gradient_accumulation_rows):
        chunk = schedule[offset : offset + args.gradient_accumulation_rows]
        optimizer.zero_grad(set_to_none=True)
        chunk_losses: list[float] = []
        chunk_pair_sides = 0
        for scheduled in chunk:
            own, negative, paired, output_anchor = _candidate_nlls_v77(
                runtime=runtime,
                model=model,
                source_model=source_model,
                scheduled=scheduled,
                prefixes=prefixes,
                question_embedding=questions[scheduled.row.question],
                opposite=opposites.get(scheduled.row.key),
                settings=settings,
            )
            loss, diagnostics = row_objective_v77(
                correct_answer_nll=own,
                negative_answer_nll=negative,
                changed_pair_answer_nll=paired,
                source_output_mse=output_anchor,
                settings=settings,
            )
            (loss / len(chunk)).backward()
            chunk_losses.append(float(loss.detach().cpu()))
            training_correct_nll.append(float(own.detach().cpu()))
            training_negative_margin.append(
                float(diagnostics["negative_answer_margin"].detach().cpu())
            )
            chunk_pair_sides += paired is not None

        weight_anchor = source_weight_anchor_l2_v76(model, source_parameters)
        if settings.source_weight_anchor_weight > 0.0:
            (float(settings.source_weight_anchor_weight) * weight_anchor).backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            .detach()
            .float()
            .cpu()
        )
        if not math.isfinite(gradient):
            raise RuntimeError("V77 preclip gradient norm became nonfinite")
        optimizer.step()
        optimizer_steps += 1
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise RuntimeError("V77 optimizer produced nonfinite state")
        if (
            optimizer_steps == 1
            or optimizer_steps % args.log_every == 0
            or offset + len(chunk) == len(schedule)
        ):
            event = {
                "optimizer_step": optimizer_steps,
                "row_updates_completed": offset + len(chunk),
                "row_updates_total": len(schedule),
                "cycle": chunk[-1].cycle,
                "mean_chunk_loss": sum(chunk_losses) / len(chunk_losses),
                "changed_pair_auxiliary_sides": chunk_pair_sides,
                "source_weight_anchor_l2": float(weight_anchor.detach().cpu()),
                "preclip_gradient_norm": gradient,
            }
            fit_history.append(event)
            print(
                json.dumps({"event": "v77_historical_repair", **event}, sort_keys=True), flush=True
            )

    _disable_decoder_checkpointing(runtime.language)
    model.eval()
    after = _measurement_v77(
        runtime=runtime,
        model=model,
        source_model=source_model,
        scheduled_rows=measurement_schedule,
        prefixes=prefixes,
        questions=questions,
        opposites=opposites,
        settings=settings,
    )
    final_weight_anchor = float(
        source_weight_anchor_l2_v76(model, source_parameters).detach().cpu()
    )
    after_zero_audit = assert_dense_reader_exact_zero_scene(model)
    diagnostic = save_v77_diagnostic(
        output_candidate,
        model,
        selected_row_count=len(selected),
        cycles=args.cycles,
        optimizer_steps=optimizer_steps,
    )
    report = {
        "artifact": "v77_all_historical_answer_repair_screen_v1",
        "scope": {
            "historical_training_pool_only": True,
            "available_historical_optimization_rows": EXPECTED_TRAIN_ROWS,
            "selected_historical_optimization_rows": len(selected),
            "historical_internal_held_optimization_rows": 0,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
            "runtime_promotion_authorized": False,
            "checkpoint_published": False,
        },
        "source_candidate": {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "sha256": V77_INITIAL_CANDIDATE_SHA256,
            "metadata": source_metadata,
        },
        "selection": _selection_summary_v77(selected),
        "schedule": {
            "cycles": args.cycles,
            "row_updates": len(schedule),
            "optimizer_steps": optimizer_steps,
            "gradient_accumulation_rows": args.gradient_accumulation_rows,
            "answer_class_round_robin": True,
            "template_round_robin_within_answer_class": True,
            "each_selected_row_once_per_cycle": True,
            "deterministic_seed": args.seed,
        },
        "canonical_negative_sampling": {
            "same_answer_type_only": True,
            "correct_answer_excluded": True,
            "deterministic_hash_sampling": True,
            "answer_values_serialized": False,
            "answer_type_count": len(alternatives),
        },
        "loss_settings": asdict(settings),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
        },
        "measurement": {
            "balanced_row_count": len(measurement_schedule),
            "before": before,
            "after": after,
            "mean_correct_answer_nll_delta": float(after["mean_correct_answer_nll"])
            - float(before["mean_correct_answer_nll"]),
            "mean_negative_answer_margin_delta": float(after["mean_negative_answer_margin"])
            - float(before["mean_negative_answer_margin"]),
        },
        "training_online": {
            "mean_preupdate_correct_answer_nll": sum(training_correct_nll)
            / len(training_correct_nll),
            "mean_preupdate_negative_answer_margin": sum(training_negative_margin)
            / len(training_negative_margin),
        },
        "final_source_weight_anchor_l2": final_weight_anchor,
        "elapsed_training_seconds": time.perf_counter() - started,
        "diagnostic_candidate": diagnostic,
        "base_freeze_audit": freeze_audit,
        "exclusive_trainable_audit": trainable_audit,
        "source_exact_zero_audit": source_zero_audit,
        "after_exact_zero_audit": after_zero_audit,
        "fit_history": fit_history,
        "prefix_manifest_base_checkpoint_sha256": prefix_manifest["base_checkpoint_sha256"],
    }
    _write_report_v77(output_report, report)
    print(
        json.dumps(
            {
                "event": "v77_historical_repair_complete",
                "output_candidate": diagnostic["path"],
                "output_report": str(output_report.relative_to(PROJECT_ROOT)),
                "selected_rows": len(selected),
                "row_updates": len(schedule),
                "optimizer_steps": optimizer_steps,
                "before_correct_answer_nll": before["mean_correct_answer_nll"],
                "after_correct_answer_nll": after["mean_correct_answer_nll"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-candidate", default=V77_INITIAL_CANDIDATE)
    parser.add_argument("--runtime-config", default="configs/runtime/gemma4_v54.yaml")
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
    )
    parser.add_argument(
        "--v73-config",
        default="configs/experiments/gemma4_v73_fullscene_controller.yaml",
    )
    parser.add_argument("--max-rows", type=int, default=EXPECTED_TRAIN_ROWS)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--gradient-accumulation-rows", type=int, default=8)
    parser.add_argument("--measurement-rows", type=int, default=48)
    parser.add_argument("--seed", type=int, default=770177)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--answer-nll-weight", type=float, default=1.0)
    parser.add_argument("--negative-margin-weight", type=float, default=0.25)
    parser.add_argument("--negative-margin", type=float, default=0.5)
    parser.add_argument("--changed-pair-margin-weight", type=float, default=0.25)
    parser.add_argument("--changed-pair-margin", type=float, default=0.5)
    parser.add_argument("--source-output-anchor-weight", type=float, default=0.05)
    parser.add_argument("--source-weight-anchor-weight", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--output-candidate",
        default=(
            "reports/gemma4/artifacts/"
            "v77_v75_all576_historical_answer_repair_diagnostic.safetensors"
        ),
    )
    parser.add_argument(
        "--output-report",
        default=("reports/gemma4/metrics/v77_v75_all576_historical_answer_repair_screen.json"),
    )
    return parser


def main() -> int:
    run_v77_historical_repair(build_parser().parse_args())
    return 0


__all__ = [
    "EXPECTED_ANSWER_CLASS_COUNT",
    "EXPECTED_CHANGED_SIDE_COUNT",
    "EXPECTED_QUESTION_TEMPLATE_COUNT",
    "V77_INITIAL_CANDIDATE",
    "V77_INITIAL_CANDIDATE_SHA256",
    "ScheduledRowV77",
    "V77LossSettings",
    "assert_exact_v75_nll_source_v77",
    "build_parser",
    "canonical_alternatives_v77",
    "changed_opposites_v77",
    "deterministic_training_schedule_v77",
    "load_exact_v75_nll_source_v77",
    "question_template_id_v77",
    "row_objective_v77",
    "run_v77_historical_repair",
    "sample_negative_answer_v77",
    "save_v77_diagnostic",
    "select_balanced_historical_rows_v77",
    "validate_args_v77",
]


if __name__ == "__main__":
    raise SystemExit(main())
