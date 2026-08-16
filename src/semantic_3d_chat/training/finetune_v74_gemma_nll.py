"""Short, training-pool-only answer-NLL tuning for V74/V75 dense readers.

The frozen local Gemma decoder supplies answer-token gradients, but every
Gemma, V54 scene-stack, and cached full-scene-prefix parameter remains frozen.
Only one authenticated zero-safe V74 or V75 scene reader is optimized.  The
output is always quarantined as a diagnostic and is never a runtime promotion
artifact.
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
from pathlib import Path
from typing import Any, Final, TypeAlias

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import (
    LIST_ANSWER_TYPES,
    list_order_insensitive_match,
    normalize_answer,
)
from semantic_3d_chat.evaluation.v55_development_score import (
    canonical_type_specific_match,
)
from semantic_3d_chat.language.local_lm import question_token_ids
from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    _select_training_device,
    freeze_base_runtime,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _disable_decoder_checkpointing,
    _generate_with_control,
    _teacher_nll,
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

SELECTION_SALT: Final[str] = "semantic_3d_chat.v74.gemma_nll.balanced.v1"
EXPECTED_TRAIN_FAMILIES: Final[tuple[str, ...]] = (
    "book_support",
    "chair_orientation",
    "color_swap",
    "cube_support",
    "mirror_lr",
    "object_count",
    "object_relocation",
    "object_removal",
    "picture_support",
)
V74_STATE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "output_basis",
        "key.weight",
        "value.weight",
        "query.weight",
        "coefficient_output.weight",
    }
)
V75_STATE_FIELDS: Final[frozenset[str]] = V74_STATE_FIELDS | {
    "coefficient_hidden.weight"
}
# Backward-compatible public name used by the original V74-only tests.
EXPECTED_STATE_FIELDS: Final[frozenset[str]] = V74_STATE_FIELDS
SOURCE_CANDIDATE_ARTIFACTS: Final[dict[str, str]] = {
    "v74": "v74_verified_teacher_dense_reader_candidate_v1",
    "v75": "v75_verified_teacher_dense_reader_candidate_v1",
}
FORBIDDEN_INPUT_PARTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "validate", "test", "deferred", "final"}
)

DenseReaderControl: TypeAlias = (
    DenseFullSceneContinuousControlV74 | DenseFullSceneContinuousControlV75
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _guard_training_input(path: str | Path, purpose: str) -> Path:
    source = _resolve(path)
    scoped = (
        source.relative_to(PROJECT_ROOT)
        if source.is_relative_to(PROJECT_ROOT)
        else source
    )
    tokens = {
        token
        for part in scoped.parts
        for token in part.casefold().replace("-", "_").split("_")
        if token
    }
    forbidden = sorted(tokens & FORBIDDEN_INPUT_PARTS)
    if forbidden:
        raise ValueError(f"V74 Gemma-NLL {purpose} crosses forbidden split tokens: {forbidden}")
    if not source.exists() or source.is_symlink():
        raise FileNotFoundError(
            f"V74 Gemma-NLL {purpose} is unavailable or symlinked: {source}"
        )
    return source


def _selection_digest(unit: ChangedUnitV73) -> str:
    return hashlib.sha256(
        f"{SELECTION_SALT}|{unit.pair_id}|{unit.question_key}".encode()
    ).hexdigest()


def select_balanced_historical_units_v74(
    train_rows: Sequence[RowV73],
) -> tuple[ChangedUnitV73, ...]:
    """Select one deterministic two-sided changed unit per training family."""

    units = changed_units_v73(train_rows)
    if not units:
        raise ValueError("V74 Gemma-NLL selection requires changed training units")
    observed_pairs = {unit.pair_id for unit in units}
    if not observed_pairs <= set(TRAIN_PAIR_IDS) or observed_pairs & set(HELD_PAIR_IDS):
        raise ValueError("V74 Gemma-NLL optimization escaped historical train pairs")
    by_family: dict[str, list[ChangedUnitV73]] = {}
    for unit in units:
        by_family.setdefault(unit.change_type, []).append(unit)
    if tuple(sorted(by_family)) != EXPECTED_TRAIN_FAMILIES:
        raise ValueError("V74 Gemma-NLL training-family inventory changed")
    selected = tuple(
        min(by_family[family], key=lambda unit: (_selection_digest(unit), unit.pair_id))
        for family in EXPECTED_TRAIN_FAMILIES
    )
    rows = [row for unit in selected for row in (unit.left, unit.right)]
    if len(selected) != 9 or len(rows) != 18 or len({row.key for row in rows}) != 18:
        raise RuntimeError("V74 Gemma-NLL balanced selection lost paired coverage")
    if any(unit.left.question != unit.right.question for unit in selected):
        raise RuntimeError("V74 Gemma-NLL selection lost same-question pairing")
    return selected


def deterministic_training_schedule_v74(
    units: Sequence[ChangedUnitV73], *, cycles: int, seed: int
) -> tuple[RowV73, ...]:
    """Return complete balanced cycles, with each side visited once per cycle."""

    if cycles < 1 or seed < 0:
        raise ValueError("V74 Gemma-NLL cycles must be positive and seed nonnegative")
    base = [row for unit in units for row in (unit.left, unit.right)]
    if not base or len({row.key for row in base}) != len(base):
        raise ValueError("V74 Gemma-NLL schedule requires unique paired rows")
    result: list[RowV73] = []
    for cycle in range(cycles):
        order = list(base)
        random.Random(seed + cycle).shuffle(order)
        result.extend(order)
    return tuple(result)


def dense_reader_architecture(model: DenseReaderControl) -> str:
    """Return the exact supported architecture, rejecting look-alike subclasses."""

    if type(model) is DenseFullSceneContinuousControlV74:
        return "v74"
    if type(model) is DenseFullSceneContinuousControlV75:
        return "v75"
    raise TypeError("Gemma-NLL supports only the exact V74 or V75 dense reader")


def assert_dense_reader_exact_zero_scene(
    model: DenseReaderControl,
) -> dict[str, int | bool | str]:
    """Prove the controller cannot emit a question-only environmental prompt."""

    architecture = dense_reader_architecture(model)
    parameter = next(model.parameters())
    was_training = model.training
    model.eval()
    scene = torch.zeros(
        1,
        model.environment_latents + 2,
        model.hidden_size,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    first_question = torch.arange(
        model.hidden_size, device=parameter.device, dtype=parameter.dtype
    ).reshape(1, 1, model.hidden_size)
    second_question = -3.0 * first_question.roll(1, dims=-1)
    try:
        with torch.no_grad():
            first = model(scene, first_question).control_tokens
            second = model(scene, second_question).control_tokens
            audit = model.audit()
    finally:
        model.train(was_training)
    if int(torch.count_nonzero(first)) or int(torch.count_nonzero(second)):
        raise RuntimeError("Dense-reader Gemma-NLL zero-scene guarantee failed")
    if (
        audit.question_only_output_path_exists
        or not audit.zero_scene_produces_exact_zero_controls
        or not audit.all_latents_receive_positive_weight
        or audit.question_dependent_retrieval
    ):
        raise RuntimeError("Dense-reader Gemma-NLL structural audit failed")
    return {
        "architecture_version": architecture,
        "zero_scene_nonzero_controls": 0,
        "question_only_output_path_exists": False,
        "all_environment_latents_attended": True,
        "question_dependent_retrieval": False,
        "exact_zero_scene_verified": True,
    }


def assert_exclusive_dense_reader_trainable_surface(
    runtime: Any, model: DenseReaderControl
) -> dict[str, int | bool | str]:
    """Prove that the supported dense reader is the only trainable surface."""

    architecture = dense_reader_architecture(model)
    base_parameters = tuple(runtime.language.model.parameters())
    model_parameters = tuple(model.parameters())
    if not base_parameters or not model_parameters:
        raise RuntimeError("Dense-reader Gemma-NLL trainable audit found an empty model")
    base_trainable = sum(parameter.numel() for parameter in base_parameters if parameter.requires_grad)
    controller_trainable = sum(
        parameter.numel() for parameter in model_parameters if parameter.requires_grad
    )
    if base_trainable != 0:
        raise RuntimeError("Frozen Gemma/base stack unexpectedly has trainable parameters")
    if controller_trainable != sum(
        parameter.numel() for parameter in model_parameters
    ):
        raise RuntimeError(
            "Every dense-reader parameter must remain trainable during this screen"
        )
    return {
        "architecture_version": architecture,
        "base_parameter_count": sum(parameter.numel() for parameter in base_parameters),
        "base_trainable_parameter_count": base_trainable,
        "controller_parameter_count": sum(
            parameter.numel() for parameter in model_parameters
        ),
        "controller_trainable_parameter_count": controller_trainable,
        "only_dense_reader_trainable": True,
    }


def assert_exclusive_v74_trainable_surface(
    runtime: Any, model: DenseFullSceneContinuousControlV74
) -> dict[str, int | bool]:
    """Backward-compatible exact-V74 wrapper around the generic audit."""

    if type(model) is not DenseFullSceneContinuousControlV74:
        raise TypeError("Legacy V74 trainable audit requires exact V74")
    generic = assert_exclusive_dense_reader_trainable_surface(runtime, model)
    return {
        "base_parameter_count": int(generic["base_parameter_count"]),
        "base_trainable_parameter_count": int(
            generic["base_trainable_parameter_count"]
        ),
        "v74_parameter_count": int(generic["controller_parameter_count"]),
        "v74_trainable_parameter_count": int(
            generic["controller_trainable_parameter_count"]
        ),
        "only_v74_trainable": True,
    }


def _load_initial_candidate(
    path: Path,
    device: torch.device,
    *,
    hidden_size: int = EXPECTED_HIDDEN_SIZE,
    environment_latents: int = 256,
) -> tuple[DenseReaderControl, dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        fields = frozenset(handle.keys())
    if fields == V74_STATE_FIELDS:
        architecture = "v74"
    elif fields == V75_STATE_FIELDS:
        architecture = "v75"
    else:
        raise ValueError(
            "Dense-reader Gemma-NLL source candidate has an unsupported state layout"
        )
    required = {
        "artifact": SOURCE_CANDIDATE_ARTIFACTS[architecture],
        "training_pool_only": "true",
        "runtime_promotion_forbidden_until_gemma_gate": "true",
        "numeric_gate_passed": "true",
        "answer_codebook_serialized": "false",
        "environmental_text_inputs": "0",
    }
    if set(metadata) != set(required) or any(
        metadata.get(key) != value for key, value in required.items()
    ):
        raise ValueError(
            "Dense-reader Gemma-NLL source candidate quarantine contract changed"
        )
    state = load_file(str(path), device="cpu")
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("Dense-reader Gemma-NLL source state is nonfinite or nonfloat")
    basis = state["output_basis"]
    key = state["key.weight"]
    value = state["value.weight"]
    query = state["query.weight"]
    if (
        basis.ndim != 2
        or basis.shape[1] != hidden_size
        or key.ndim != 2
        or key.shape[1] != hidden_size
        or value.shape != key.shape
        or query.ndim != 2
        or query.shape[1] != hidden_size
    ):
        raise ValueError("Dense-reader Gemma-NLL source common tensor shapes changed")
    model_dimension = int(key.shape[0])
    output_basis_rank = int(basis.shape[0])
    if (
        model_dimension < 1
        or model_dimension > 1024
        or output_basis_rank < 1
        or output_basis_rank > hidden_size
        or query.shape[0] % model_dimension
    ):
        raise ValueError("Dense-reader Gemma-NLL source dimensions are invalid")
    query_count = int(query.shape[0] // model_dimension)
    if query_count < 1 or query_count > 32:
        raise ValueError("Dense-reader Gemma-NLL query count is invalid")
    coefficient_output = state["coefficient_output.weight"]
    expected_output_rows = query_count * output_basis_rank
    if architecture == "v74":
        if coefficient_output.shape != (
            expected_output_rows,
            query_count * model_dimension,
        ):
            raise ValueError("V74 Gemma-NLL coefficient tensor shape changed")
        model: DenseReaderControl = DenseFullSceneContinuousControlV74(
            hidden_size,
            basis,
            environment_latents=environment_latents,
            query_count=query_count,
            model_dimension=model_dimension,
        )
    else:
        coefficient_hidden = state["coefficient_hidden.weight"]
        if (
            coefficient_hidden.ndim != 2
            or coefficient_hidden.shape[1] != query_count * model_dimension
            or coefficient_hidden.shape[0] < 1
            or coefficient_hidden.shape[0] > 4096
            or coefficient_output.shape
            != (expected_output_rows, coefficient_hidden.shape[0])
        ):
            raise ValueError("V75 Gemma-NLL coefficient tensor shapes changed")
        model = DenseFullSceneContinuousControlV75(
            hidden_size,
            basis,
            environment_latents=environment_latents,
            query_count=query_count,
            model_dimension=model_dimension,
            coefficient_decoder_hidden_dimension=int(coefficient_hidden.shape[0]),
        )
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32)
    assert_dense_reader_exact_zero_scene(model)
    return model, metadata


def _question_embeddings(runtime: Any, question: str) -> torch.Tensor:
    ids = question_token_ids(
        runtime.language.tokenizer, question, runtime.language.device
    )
    with torch.no_grad():
        value = runtime.language.model.get_input_embeddings()(ids).detach().float()
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("V74 Gemma-NLL question embedding shape changed")
    return value


def _load_answer_items(path: str | Path) -> dict[tuple[str, str], tuple[str, ...]]:
    source = _guard_training_input(path, "training QA answer-items scorer")
    result: dict[tuple[str, str], tuple[str, ...]] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        key = value.get("scene_id"), value.get("question_id")
        if not all(isinstance(item, str) and item for item in key):
            raise ValueError(f"V74 Gemma-NLL scoring key changed at line {line_number}")
        items = value.get("answer_items")
        if items is None:
            continue
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item for item in items
        ):
            raise ValueError(f"V74 Gemma-NLL answer-items changed at line {line_number}")
        result[(str(key[0]), str(key[1]))] = tuple(items)
    return result


def _answer_matches(
    row: RowV73, prediction: str, answer_items: Mapping[tuple[str, str], Sequence[str]]
) -> bool:
    if row.answer_type in LIST_ANSWER_TYPES:
        reference: str | Sequence[str] = answer_items.get(row.key, (row.answer,))
        return list_order_insensitive_match(prediction, reference)
    return canonical_type_specific_match(row.answer_type, prediction, row.answer)


def _pair_change_count(
    units: Sequence[ChangedUnitV73], predictions: Mapping[tuple[str, str], str]
) -> int:
    return sum(
        normalize_answer(predictions[unit.left.key])
        != normalize_answer(predictions[unit.right.key])
        for unit in units
    )


def _behavior_snapshot(
    *,
    runtime: Any,
    model: DenseReaderControl,
    units: Sequence[ChangedUnitV73],
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    answer_items: Mapping[tuple[str, str], Sequence[str]],
) -> dict[str, Any]:
    model.eval()
    model_dtype = next(runtime.language.model.parameters()).dtype
    records: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], str] = {}
    with torch.inference_mode():
        for unit in units:
            for row in (unit.left, unit.right):
                scene = prefixes[row.scene_id].to(
                    device=runtime.language.device, dtype=model_dtype
                )
                control = model(scene.float(), questions[row.question]).control_tokens
                prediction = _generate_with_control(
                    runtime=runtime,
                    scene_prefix=scene,
                    question=row.question,
                    control_tokens=control,
                )
                predictions[row.key] = prediction
                records.append(
                    {
                        "scene_id": row.scene_id,
                        "question_id": row.question_id,
                        "pair_id": row.pair_id,
                        "question_key": row.question_key,
                        "change_type": row.change_type,
                        "answer_type": row.answer_type,
                        "reference": row.answer,
                        "prediction": prediction,
                        "correct": _answer_matches(row, prediction, answer_items),
                        "control_rms": float(control.float().square().mean().sqrt().cpu()),
                    }
                )
    correct = sum(bool(record["correct"]) for record in records)
    complete = sum(
        all(
            next(record for record in records if (record["scene_id"], record["question_id"]) == row.key)[
                "correct"
            ]
            for row in (unit.left, unit.right)
        )
        for unit in units
    )
    return {
        "correct_sides": correct,
        "side_count": len(records),
        "accuracy": correct / len(records),
        "complete_units": complete,
        "unit_count": len(units),
        "prediction_change_units": _pair_change_count(units, predictions),
        "mean_control_rms": sum(float(record["control_rms"]) for record in records)
        / len(records),
        "records": records,
    }


def _mean_answer_nll(
    *,
    runtime: Any,
    model: DenseReaderControl,
    rows: Sequence[RowV73],
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    model_dtype = next(runtime.language.model.parameters()).dtype
    values: list[float] = []
    with torch.no_grad():
        for row in rows:
            scene = prefixes[row.scene_id].to(
                device=runtime.language.device, dtype=model_dtype
            )
            control = model(scene.float(), questions[row.question]).control_tokens
            loss = _teacher_nll(
                runtime=runtime,
                scene_prefix=scene,
                record=row,
                free_prompt=control,
            )
            values.append(float(loss.detach().cpu()))
    return {
        "mean": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
        "row_count": len(values),
    }


def _finite_state(model: DenseReaderControl) -> dict[str, torch.Tensor]:
    architecture = dense_reader_architecture(model)
    state = {
        key: value.detach().cpu().float().contiguous()
        for key, value in model.state_dict().items()
    }
    expected = V74_STATE_FIELDS if architecture == "v74" else V75_STATE_FIELDS
    if frozenset(state) != expected:
        raise ValueError("Dense-reader Gemma-NLL diagnostic state inventory changed")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("V74 Gemma-NLL diagnostic state is nonfinite")
    return state


def save_dense_reader_gemma_nll_diagnostic(
    path: str | Path,
    model: DenseReaderControl,
    *,
    source_sha256: str,
    optimizer_steps: int,
    train_behavior_improved: bool,
) -> dict[str, Any]:
    """Create one quarantined safetensors diagnostic without overwrite."""

    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("Dense-reader Gemma-NLL source hash must be lowercase SHA-256")
    if isinstance(optimizer_steps, bool) or not isinstance(optimizer_steps, int):
        raise TypeError("Dense-reader Gemma-NLL optimizer steps must be an integer")
    if optimizer_steps < 1:
        raise ValueError("Dense-reader Gemma-NLL optimizer steps must be positive")
    if type(train_behavior_improved) is not bool:
        raise TypeError("Dense-reader Gemma-NLL behavior result must be boolean")
    architecture = dense_reader_architecture(model)
    zero_scene_audit = assert_dense_reader_exact_zero_scene(model)
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = _finite_state(model)
    metadata = {
        "artifact": f"{architecture}_historical_train_gemma_nll_diagnostic_v1",
        "controller_architecture": architecture,
        "training_pool_only": "true",
        "historical_train_pairs_only": "true",
        "held_optimization_rows": "0",
        "runtime_promotion_forbidden_until_gemma_gate": "true",
        "numeric_gate_passed": "unverified_after_gemma_nll",
        "answer_codebook_serialized": "false",
        "environmental_text_inputs": "0",
        "official_validation_loaded": "false",
        "official_test_loaded": "false",
        "oracle_loaded": "false",
        "exact_zero_scene_verified": "true",
        "question_only_output_path_exists": "false",
        "runtime_publication_artifact": "false",
        "source_candidate_sha256": source_sha256,
        "optimizer_steps": str(optimizer_steps),
        "train_behavior_improved": str(train_behavior_improved).lower(),
    }
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
            raise RuntimeError("V74 Gemma-NLL diagnostic failed exact reload")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        display_path = str(destination.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(destination)
    return {
        "path": display_path,
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "metadata": metadata,
        "zero_scene_audit": zero_scene_audit,
    }


def save_v74_gemma_nll_diagnostic(
    path: str | Path,
    model: DenseFullSceneContinuousControlV74,
    *,
    source_sha256: str,
    optimizer_steps: int,
    train_behavior_improved: bool,
) -> dict[str, Any]:
    """Backward-compatible exact-V74 diagnostic writer."""

    if type(model) is not DenseFullSceneContinuousControlV74:
        raise TypeError("Legacy V74 diagnostic writer requires exact V74")
    return save_dense_reader_gemma_nll_diagnostic(
        path,
        model,
        source_sha256=source_sha256,
        optimizer_steps=optimizer_steps,
        train_behavior_improved=train_behavior_improved,
    )


def _write_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run_dense_reader_gemma_nll_screen(args: argparse.Namespace) -> dict[str, Any]:
    if args.cycles < 1 or args.cycles > 6:
        raise ValueError("V74 Gemma-NLL cycles must be in [1, 6]")
    if not 0.0 < args.learning_rate <= 1e-3:
        raise ValueError("V74 Gemma-NLL learning rate must be in (0, 1e-3]")
    if not 0.0 < args.gradient_clip_norm <= 10.0:
        raise ValueError("V74 Gemma-NLL gradient clip must be in (0, 10]")
    if args.seed < 0 or args.log_every < 1 or not 0.0 <= args.weight_decay <= 1.0:
        raise ValueError("V74 Gemma-NLL seed/log interval/weight decay is invalid")

    candidate_path = _guard_training_input(args.initial_candidate, "source candidate")
    runtime_config_path = _guard_training_input(args.runtime_config, "runtime config")
    base_checkpoint = _guard_training_input(args.base_checkpoint, "base checkpoint")
    v73_config_path = _guard_training_input(args.v73_config, "V73 config")
    output_candidate = _resolve(args.output_candidate)
    output_report = _resolve(args.output_report)
    if output_candidate == output_report or output_candidate == candidate_path:
        raise ValueError("V74 Gemma-NLL outputs must be distinct new artifacts")
    if output_candidate.exists() or output_candidate.is_symlink():
        raise FileExistsError(output_candidate)
    if output_report.exists() or output_report.is_symlink():
        raise FileExistsError(output_report)
    source_sha256 = _sha256_file(candidate_path)

    v73 = load_config_v73(v73_config_path)
    all_rows = load_training_rows_v73(v73["training_qa"])
    train_rows, held_rows = split_rows_v73(all_rows)
    units = select_balanced_historical_units_v74(train_rows)
    selected_rows = tuple(row for unit in units for row in (unit.left, unit.right))
    if {row.scene_id for row in selected_rows} & {row.scene_id for row in held_rows}:
        raise RuntimeError("V74 Gemma-NLL selected a held scene")
    schedule = deterministic_training_schedule_v74(
        units, cycles=args.cycles, seed=args.seed
    )
    prefixes, prefix_manifest = load_prefixes_v73(
        v73["prefix_cache"], {row.scene_id for row in selected_rows}
    )
    answer_items = _load_answer_items(v73["training_qa"])

    runtime_config, _ = _load_sanitized_runtime_config(runtime_config_path)
    runtime = StaticRuntimePrefixFactory(
        runtime_config, base_checkpoint, selected_rows[0].scene_id
    ).bootstrap
    freeze_audit = freeze_base_runtime(runtime)
    _disable_decoder_checkpointing(runtime.language)
    device = _select_training_device(runtime, args.device)
    model, source_metadata = _load_initial_candidate(candidate_path, device)
    architecture = dense_reader_architecture(model)
    source_zero_scene_audit = assert_dense_reader_exact_zero_scene(model)
    trainable_audit = assert_exclusive_dense_reader_trainable_surface(runtime, model)
    if not torch.equal(
        runtime.scene_prefix.detach().cpu().float(),
        prefixes[selected_rows[0].scene_id].detach().cpu().float(),
    ):
        raise ValueError("V74 Gemma-NLL cached prefix differs from frozen V54 runtime")

    questions = {
        question: _question_embeddings(runtime, question)
        for question in sorted({row.question for row in selected_rows})
    }
    before_nll = _mean_answer_nll(
        runtime=runtime,
        model=model,
        rows=selected_rows,
        prefixes=prefixes,
        questions=questions,
    )
    before_behavior = _behavior_snapshot(
        runtime=runtime,
        model=model,
        units=units,
        prefixes=prefixes,
        questions=questions,
        answer_items=answer_items,
    )

    runtime.language.enable_decoder_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model_dtype = next(runtime.language.model.parameters()).dtype
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    for index, row in enumerate(schedule, 1):
        scene = prefixes[row.scene_id].to(device=device, dtype=model_dtype)
        control = model(scene.float(), questions[row.question]).control_tokens
        optimizer.zero_grad(set_to_none=True)
        loss = _teacher_nll(
            runtime=runtime,
            scene_prefix=scene,
            record=row,
            free_prompt=control,
        )
        loss.backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            .detach()
            .float()
            .cpu()
        )
        if not math.isfinite(gradient):
            raise RuntimeError("V74 Gemma-NLL gradient became nonfinite")
        optimizer.step()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise RuntimeError("V74 Gemma-NLL optimizer produced nonfinite state")
        if index == 1 or index % args.log_every == 0 or index == len(schedule):
            event = {
                "step": index,
                "optimizer_steps": len(schedule),
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "answer_nll": float(loss.detach().cpu()),
                "preclip_gradient_norm": gradient,
            }
            history.append(event)
            print(
                json.dumps(
                    {"event": f"{architecture}_gemma_nll_train", **event},
                    sort_keys=True,
                ),
                flush=True,
            )

    _disable_decoder_checkpointing(runtime.language)
    model.eval()
    after_nll = _mean_answer_nll(
        runtime=runtime,
        model=model,
        rows=selected_rows,
        prefixes=prefixes,
        questions=questions,
    )
    after_behavior = _behavior_snapshot(
        runtime=runtime,
        model=model,
        units=units,
        prefixes=prefixes,
        questions=questions,
        answer_items=answer_items,
    )
    behavior_improved = (
        after_behavior["correct_sides"] > before_behavior["correct_sides"]
        and after_behavior["complete_units"] >= before_behavior["complete_units"]
        and after_nll["mean"] < before_nll["mean"]
    )
    diagnostic = save_dense_reader_gemma_nll_diagnostic(
        args.output_candidate,
        model,
        source_sha256=source_sha256,
        optimizer_steps=len(schedule),
        train_behavior_improved=behavior_improved,
    )
    report = {
        "artifact": f"{architecture}_historical_train_gemma_nll_screen_v1",
        "controller_architecture": architecture,
        "seed": args.seed,
        "device": str(device),
        "cycles": args.cycles,
        "optimizer_steps": len(schedule),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "elapsed_training_seconds": time.perf_counter() - started,
        "source_candidate": {
            "path": str(candidate_path.relative_to(PROJECT_ROOT)),
            "sha256": source_sha256,
            "metadata": source_metadata,
        },
        "diagnostic_candidate": diagnostic,
        "selection": {
            "algorithm": "sha256_minimum_changed_unit_per_historical_train_family",
            "salt": SELECTION_SALT,
            "family_count": len(units),
            "side_count": len(selected_rows),
            "families": dict(Counter(unit.change_type for unit in units)),
            "units": [
                {
                    "pair_id": unit.pair_id,
                    "question_key": unit.question_key,
                    "change_type": unit.change_type,
                    "side_keys": [list(unit.left.key), list(unit.right.key)],
                    "selection_sha256": _selection_digest(unit),
                }
                for unit in units
            ],
        },
        "before": {"answer_nll": before_nll, "behavior": before_behavior},
        "after": {"answer_nll": after_nll, "behavior": after_behavior},
        "answer_nll_mean_delta": after_nll["mean"] - before_nll["mean"],
        "train_exact_side_gain": (
            after_behavior["correct_sides"] - before_behavior["correct_sides"]
        ),
        "train_complete_unit_gain": (
            after_behavior["complete_units"] - before_behavior["complete_units"]
        ),
        "train_behavior_improved": behavior_improved,
        "held_smoke_authorized": behavior_improved,
        "held_optimization_rows": 0,
        "base_freeze_audit": freeze_audit,
        "exclusive_trainable_audit": trainable_audit,
        "source_zero_scene_audit": source_zero_scene_audit,
        "saved_zero_scene_audit": diagnostic["zero_scene_audit"],
        "fit_history": history,
        "prefix_manifest_base_checkpoint_sha256": prefix_manifest[
            "base_checkpoint_sha256"
        ],
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "checkpoint_published": False,
        "runtime_promotion_authorized": False,
    }
    _write_report(args.output_report, report)
    print(
        json.dumps(
            {
                "event": f"{architecture}_gemma_nll_complete",
                "output_report": str(_resolve(args.output_report)),
                "output_candidate": diagnostic["path"],
                "before_mean_nll": before_nll["mean"],
                "after_mean_nll": after_nll["mean"],
                "before_correct_sides": before_behavior["correct_sides"],
                "after_correct_sides": after_behavior["correct_sides"],
                "train_behavior_improved": behavior_improved,
                "held_smoke_authorized": behavior_improved,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def run_v74_gemma_nll_screen(args: argparse.Namespace) -> dict[str, Any]:
    """Backward-compatible entry point; source state selects V74 or V75."""

    return run_dense_reader_gemma_nll_screen(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-candidate",
        default="reports/gemma4/artifacts/v74_teacher_unclipped_p4_passed_diagnostic.safetensors",
    )
    parser.add_argument("--runtime-config", default="configs/runtime/gemma4_v54.yaml")
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
    )
    parser.add_argument(
        "--v73-config",
        default="configs/experiments/gemma4_v73_fullscene_controller.yaml",
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=740176)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=9)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--output-candidate",
        default="reports/gemma4/artifacts/v74_gemma_nll_historical_train_diagnostic.safetensors",
    )
    parser.add_argument(
        "--output-report",
        default="reports/gemma4/metrics/v74_gemma_nll_historical_train_screen.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dense_reader_gemma_nll_screen(args)
    return 0


__all__ = [
    "EXPECTED_TRAIN_FAMILIES",
    "V74_STATE_FIELDS",
    "V75_STATE_FIELDS",
    "assert_dense_reader_exact_zero_scene",
    "assert_exclusive_dense_reader_trainable_surface",
    "assert_exclusive_v74_trainable_surface",
    "dense_reader_architecture",
    "deterministic_training_schedule_v74",
    "run_dense_reader_gemma_nll_screen",
    "run_v74_gemma_nll_screen",
    "save_dense_reader_gemma_nll_diagnostic",
    "save_v74_gemma_nll_diagnostic",
    "select_balanced_historical_units_v74",
]


if __name__ == "__main__":
    raise SystemExit(main())
