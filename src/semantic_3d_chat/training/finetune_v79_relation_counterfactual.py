"""Preregistered historical-only V79 relation/counterfactual repair.

V79 begins from the exact sealed-source V75 diagnostic controller and updates
only that compact numeric controller.  Gemma, the V54 scene stack, and every
immutable pre-question full-scene prefix stay frozen.  Optimization is an
exhaustive single pass over the 120 spatial-relation rows in the historical
V73 training fold; the pair- and scene-disjoint internal held fold is never
opened by this command.

Changed counterfactual sides receive two explicitly scene-sensitive terms in
addition to answer NLL: the correct scene must prefer its own answer over the
paired answer, and the correct answer must be cheaper with the correct scene
than with the paired wrong scene.  Stable relation rows retain the broad
relation language behavior without being assigned a false scene-swap margin.

The output is a quarantined numeric diagnostic containing exactly the six V75
controller tensors.  It contains no answer/category codebook, question,
answer, label, or runtime publication payload.  All constants and the
conditional held evaluation are fixed in the authenticated V79
preregistration before this trainer is run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
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
    _question_embeddings,
    assert_dense_reader_exact_zero_scene,
    assert_exclusive_dense_reader_trainable_surface,
)
from semantic_3d_chat.training.finetune_v76_pair_contrast import (
    snapshot_source_parameters_v76,
    source_weight_anchor_l2_v76,
)
from semantic_3d_chat.training.finetune_v77_historical_repair import (
    ScheduledRowV77,
    assert_exact_v75_nll_source_v77,
    canonical_alternatives_v77,
    changed_opposites_v77,
    deterministic_training_schedule_v77,
    load_exact_v75_nll_source_v77,
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
    load_config_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)

V79_PREREGISTRATION: Final[str] = (
    "configs/experiments/v79_historical_relation_counterfactual_preregistration.json"
)
V79_PREREGISTRATION_SHA256: Final[str] = (
    "05ce28bb7ce5f592dab4163a905f00e247b935b91dcb8700a6dfa8f9e4145cba"
)
V79_SOURCE_SHA256: Final[str] = "d01275538489b3493a8e1ff080109d1db46832be6ca2a26f6d89d161c597188a"
V79_RELATION_ANSWER_TYPE: Final[str] = "spatial_relation"
V79_SELECTED_ROWS: Final[int] = 120
V79_CHANGED_SIDES: Final[int] = 48
V79_STABLE_ROWS: Final[int] = 72
V79_CYCLES: Final[int] = 1
V79_GRADIENT_ACCUMULATION_ROWS: Final[int] = 8
V79_OPTIMIZER_STEPS: Final[int] = 15
V79_MEASUREMENT_ROWS: Final[int] = 24
V79_SEED: Final[int] = 790179

_FORBIDDEN_INPUT_TOKENS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "validate", "test", "deferred", "final"}
)
_FORBIDDEN_OUTPUT_TOKENS: Final[frozenset[str]] = _FORBIDDEN_INPUT_TOKENS | {
    "runtime",
    "release",
    "production",
}


@dataclass(frozen=True)
class V79LossSettings:
    """Locked V79 loss weights from the authenticated preregistration."""

    answer_nll_weight: float = 1.0
    negative_margin_weight: float = 0.15
    negative_margin: float = 0.50
    paired_answer_margin_weight: float = 0.50
    paired_answer_margin: float = 0.50
    wrong_scene_margin_weight: float = 0.75
    wrong_scene_margin: float = 0.25
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
                raise ValueError(f"V79 {field} must be finite and nonnegative")
        if self.answer_nll_weight <= 0.0:
            raise ValueError("V79 answer_nll_weight must be positive")


LOCKED_SETTINGS_V79: Final[V79LossSettings] = V79LossSettings()


def _absolute_below_project_v79(path: str | Path) -> Path:
    value = Path(path).expanduser()
    absolute = Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))
    try:
        absolute.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("V79 paths must remain below the project root") from error
    return absolute


def _path_tokens_v79(path: Path) -> set[str]:
    scoped = path.relative_to(PROJECT_ROOT)
    return {
        token
        for part in scoped.parts
        for token in re.split(r"[^a-z0-9]+", part.casefold())
        if token
    }


def _reject_symlink_components_v79(path: Path, *, include_leaf: bool) -> None:
    cursor = path if include_leaf else path.parent
    while True:
        if cursor.is_symlink():
            raise ValueError("V79 paths cannot traverse symlinks")
        if cursor == PROJECT_ROOT:
            return
        if cursor.parent == cursor:
            raise ValueError("V79 path ancestry escaped the project root")
        cursor = cursor.parent


def guard_input_v79(path: str | Path, purpose: str) -> Path:
    source = _absolute_below_project_v79(path)
    forbidden = sorted(_path_tokens_v79(source) & _FORBIDDEN_INPUT_TOKENS)
    if forbidden:
        raise ValueError(f"V79 {purpose} crosses forbidden path tokens: {forbidden}")
    _reject_symlink_components_v79(source, include_leaf=True)
    if not source.exists() or source.is_symlink():
        raise FileNotFoundError(f"V79 {purpose} is unavailable: {source}")
    return source


def guard_output_v79(path: str | Path, *, suffix: str) -> Path:
    destination = _absolute_below_project_v79(path)
    forbidden = sorted(_path_tokens_v79(destination) & _FORBIDDEN_OUTPUT_TOKENS)
    if forbidden:
        raise ValueError(f"V79 output crosses forbidden path tokens: {forbidden}")
    if destination.suffix != suffix:
        raise ValueError(f"V79 output must use the {suffix} suffix")
    _reject_symlink_components_v79(destination, include_leaf=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    return destination


def load_preregistration_v79(
    path: str | Path = V79_PREREGISTRATION,
) -> tuple[Path, dict[str, Any]]:
    """Authenticate the preregistration and its locked experiment contract."""

    source = guard_input_v79(path, "preregistration")
    if _sha256_file(source) != V79_PREREGISTRATION_SHA256:
        raise ValueError("V79 preregistration hash changed")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V79 preregistration payload changed")
    source_spec = payload.get("sources", {}).get("v75_initial_candidate", {})
    optimization = payload.get("optimization", {})
    row_filter = optimization.get("row_filter", {})
    loss = optimization.get("loss", {})
    expected = {
        "artifact": payload.get("artifact"),
        "source_path": source_spec.get("path"),
        "source_sha256": source_spec.get("sha256"),
        "answer_type": row_filter.get("answer_type"),
        "selected_rows": row_filter.get("expected_rows"),
        "changed_sides": row_filter.get("expected_changed_sides"),
        "stable_rows": row_filter.get("expected_stable_rows"),
        "cycles": optimization.get("cycles"),
        "row_updates": optimization.get("row_updates"),
        "accumulation": optimization.get("gradient_accumulation_rows"),
        "optimizer_steps": optimization.get("optimizer_steps"),
        "seed": optimization.get("seed"),
        "measurement_rows": optimization.get("measurement_rows"),
        "candidate_output": optimization.get("candidate_output"),
        "report_output": optimization.get("training_report_output"),
        "loss": loss,
    }
    locked = {
        "artifact": "v79_historical_relation_counterfactual_preregistration_v1",
        "source_path": (
            "reports/gemma4/artifacts/v75_gemma_nll_balanced_train_diagnostic.safetensors"
        ),
        "source_sha256": V79_SOURCE_SHA256,
        "answer_type": V79_RELATION_ANSWER_TYPE,
        "selected_rows": V79_SELECTED_ROWS,
        "changed_sides": V79_CHANGED_SIDES,
        "stable_rows": V79_STABLE_ROWS,
        "cycles": V79_CYCLES,
        "row_updates": V79_SELECTED_ROWS,
        "accumulation": V79_GRADIENT_ACCUMULATION_ROWS,
        "optimizer_steps": V79_OPTIMIZER_STEPS,
        "seed": V79_SEED,
        "measurement_rows": V79_MEASUREMENT_ROWS,
        "candidate_output": (
            "reports/gemma4/artifacts/v79_v75_relation_counterfactual_diagnostic.safetensors"
        ),
        "report_output": ("reports/gemma4/metrics/v79_relation_counterfactual_training.json"),
        "loss": {
            "answer_nll_weight": 1.0,
            "same_type_negative_margin_weight": 0.15,
            "same_type_negative_margin": 0.5,
            "paired_answer_margin_weight": 0.5,
            "paired_answer_margin": 0.5,
            "paired_answer_margin_expected_change_only": True,
            "wrong_scene_answer_margin_weight": 0.75,
            "wrong_scene_answer_margin": 0.25,
            "wrong_scene_answer_margin_expected_change_only": True,
            "source_output_anchor_weight": 0.05,
            "source_weight_anchor_weight": 0.01,
        },
    }
    if expected != locked:
        raise ValueError("V79 preregistration experiment constants changed")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(key) is not False
        for key in (
            "official_validation_loaded",
            "official_test_loaded",
            "deferred_final_loaded",
            "oracle_loaded",
            "runtime_publication_authorized",
        )
    ):
        raise ValueError("V79 preregistration scope contract changed")
    return source, payload


def select_historical_relation_rows_v79(
    train_rows: Sequence[RowV73], held_rows: Sequence[RowV73]
) -> tuple[RowV73, ...]:
    """Select the exhaustive relation subset without crossing held scenes."""

    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError("V79 requires the exact 576-row historical train fold")
    selected = tuple(row for row in train_rows if row.answer_type == V79_RELATION_ANSWER_TYPE)
    if (
        len(selected) != V79_SELECTED_ROWS
        or len({row.key for row in selected}) != V79_SELECTED_ROWS
        or sum(row.expected_change for row in selected) != V79_CHANGED_SIDES
        or sum(not row.expected_change for row in selected) != V79_STABLE_ROWS
    ):
        raise ValueError("V79 historical relation inventory changed")
    if any(row.pair_id not in TRAIN_PAIR_IDS or row.pair_id in HELD_PAIR_IDS for row in selected):
        raise ValueError("V79 relation selection escaped the train-pair fold")
    if {row.scene_id for row in selected} & {row.scene_id for row in held_rows}:
        raise ValueError("V79 relation selection overlaps internal held scenes")
    return selected


def relation_objective_v79(
    *,
    correct_answer_nll: torch.Tensor,
    negative_answer_nll: torch.Tensor,
    paired_answer_nll: torch.Tensor | None,
    wrong_scene_answer_nll: torch.Tensor | None,
    source_output_mse: torch.Tensor,
    settings: V79LossSettings = LOCKED_SETTINGS_V79,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine answer quality, relation contrast, and source anchoring."""

    required = (correct_answer_nll, negative_answer_nll, source_output_mse)
    if any(
        value.ndim != 0 or not bool(torch.isfinite(value)) or bool(value < 0.0)
        for value in required
    ):
        raise ValueError("V79 objective inputs must be finite nonnegative scalars")
    for name, value in (
        ("paired_answer_nll", paired_answer_nll),
        ("wrong_scene_answer_nll", wrong_scene_answer_nll),
    ):
        if value is not None and (
            value.ndim != 0 or not bool(torch.isfinite(value)) or bool(value < 0.0)
        ):
            raise ValueError(f"V79 {name} must be a finite nonnegative scalar")
    if (paired_answer_nll is None) != (wrong_scene_answer_nll is None):
        raise ValueError("V79 changed-side contrast terms must be supplied together")

    negative_hinge = F.relu(
        float(settings.negative_margin) + correct_answer_nll - negative_answer_nll
    )
    paired_hinge = correct_answer_nll.new_zeros(())
    wrong_scene_hinge = correct_answer_nll.new_zeros(())
    if paired_answer_nll is not None and wrong_scene_answer_nll is not None:
        paired_hinge = F.relu(
            float(settings.paired_answer_margin) + correct_answer_nll - paired_answer_nll
        )
        wrong_scene_hinge = F.relu(
            float(settings.wrong_scene_margin) + correct_answer_nll - wrong_scene_answer_nll
        )
    total = (
        float(settings.answer_nll_weight) * correct_answer_nll
        + float(settings.negative_margin_weight) * negative_hinge
        + float(settings.paired_answer_margin_weight) * paired_hinge
        + float(settings.wrong_scene_margin_weight) * wrong_scene_hinge
        + float(settings.source_output_anchor_weight) * source_output_mse
    )
    if total.ndim != 0 or not bool(torch.isfinite(total)):
        raise RuntimeError("V79 objective became nonfinite")
    return total, {
        "correct_answer_nll": correct_answer_nll,
        "negative_answer_margin": negative_answer_nll - correct_answer_nll,
        "negative_margin_hinge": negative_hinge,
        "paired_answer_margin": (
            correct_answer_nll.new_zeros(())
            if paired_answer_nll is None
            else paired_answer_nll - correct_answer_nll
        ),
        "paired_answer_margin_hinge": paired_hinge,
        "wrong_scene_answer_margin": (
            correct_answer_nll.new_zeros(())
            if wrong_scene_answer_nll is None
            else wrong_scene_answer_nll - correct_answer_nll
        ),
        "wrong_scene_margin_hinge": wrong_scene_hinge,
        "source_output_mse": source_output_mse,
    }


def _candidate_nlls_v79(
    *,
    runtime: Any,
    model: DenseFullSceneContinuousControlV75,
    source_model: DenseFullSceneContinuousControlV75,
    scheduled: ScheduledRowV77,
    prefixes: Mapping[str, torch.Tensor],
    question_embedding: torch.Tensor,
    opposite: RowV73 | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    row = scheduled.row
    language = runtime.language
    model_dtype = next(language.model.parameters()).dtype
    scene = prefixes[row.scene_id].to(device=language.device, dtype=model_dtype)
    control = model(scene.float(), question_embedding).control_tokens
    with torch.no_grad():
        source_control = source_model(scene.float(), question_embedding).control_tokens
    output_anchors = [F.mse_loss(control.float(), source_control.float())]

    batch_specs: list[tuple[torch.Tensor, torch.Tensor, str]] = [
        (scene, control, row.answer),
        (scene, control, scheduled.negative_answer),
    ]
    paired_index: int | None = None
    wrong_scene_index: int | None = None
    if row.expected_change:
        if (
            opposite is None
            or row.question != opposite.question
            or row.answer == opposite.answer
            or row.paired_scene_id != opposite.scene_id
        ):
            raise ValueError("V79 changed row lost its paired opposite")
        if opposite.answer == scheduled.negative_answer:
            paired_index = 1
        else:
            paired_index = len(batch_specs)
            batch_specs.append((scene, control, opposite.answer))
        wrong_scene = prefixes[row.paired_scene_id].to(device=language.device, dtype=model_dtype)
        wrong_control = model(wrong_scene.float(), question_embedding).control_tokens
        with torch.no_grad():
            wrong_source_control = source_model(
                wrong_scene.float(), question_embedding
            ).control_tokens
        output_anchors.append(F.mse_loss(wrong_control.float(), wrong_source_control.float()))
        wrong_scene_index = len(batch_specs)
        batch_specs.append((wrong_scene, wrong_control, row.answer))
    elif opposite is not None:
        raise ValueError("V79 stable row unexpectedly has a changed opposite")

    batches = tuple(
        _compose_batch(
            runtime=runtime,
            scene_prefix=batch_scene,
            record=row,
            answer=answer,
            control_tokens=batch_control,
        )[0]
        for batch_scene, batch_control, answer in batch_specs
    )
    stacked = stack_prefix_batches(
        batches,
        language.device,
        prefix_backend=language.prefix_backend,
    )
    output = forward_prefix_batch(language, stacked)
    if stacked.labels is None:
        raise RuntimeError("V79 candidate batch lacks answer labels")
    nll = token_normalized_nll(output.logits, stacked.labels)
    if nll.shape != (len(batch_specs),):
        raise RuntimeError("V79 candidate NLL shape changed")
    paired = None if paired_index is None else nll[paired_index]
    wrong_scene_nll = None if wrong_scene_index is None else nll[wrong_scene_index]
    return nll[0], nll[1], paired, wrong_scene_nll, torch.stack(output_anchors).mean()


@torch.inference_mode()
def _measure_v79(
    *,
    runtime: Any,
    model: DenseFullSceneContinuousControlV75,
    source_model: DenseFullSceneContinuousControlV75,
    schedule: Sequence[ScheduledRowV77],
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    opposites: Mapping[tuple[str, str], RowV73],
    settings: V79LossSettings,
) -> dict[str, float | int]:
    model.eval()
    correct_values: list[float] = []
    negative_margins: list[float] = []
    paired_margins: list[float] = []
    wrong_scene_margins: list[float] = []
    output_mse: list[float] = []
    for scheduled in schedule:
        correct, negative, paired, wrong_scene, anchor = _candidate_nlls_v79(
            runtime=runtime,
            model=model,
            source_model=source_model,
            scheduled=scheduled,
            prefixes=prefixes,
            question_embedding=questions[scheduled.row.question],
            opposite=opposites.get(scheduled.row.key),
        )
        correct_value = float(correct.detach().cpu())
        correct_values.append(correct_value)
        negative_margins.append(float(negative.detach().cpu()) - correct_value)
        output_mse.append(float(anchor.detach().cpu()))
        if paired is not None and wrong_scene is not None:
            paired_margins.append(float(paired.detach().cpu()) - correct_value)
            wrong_scene_margins.append(float(wrong_scene.detach().cpu()) - correct_value)
    return {
        "row_count": len(correct_values),
        "changed_side_count": len(paired_margins),
        "mean_correct_answer_nll": sum(correct_values) / len(correct_values),
        "mean_negative_answer_margin": sum(negative_margins) / len(negative_margins),
        "negative_margin_satisfied_rows": sum(
            value >= settings.negative_margin for value in negative_margins
        ),
        "mean_paired_answer_margin": (
            sum(paired_margins) / len(paired_margins) if paired_margins else 0.0
        ),
        "paired_margin_satisfied_sides": sum(
            value >= settings.paired_answer_margin for value in paired_margins
        ),
        "mean_wrong_scene_answer_margin": (
            sum(wrong_scene_margins) / len(wrong_scene_margins) if wrong_scene_margins else 0.0
        ),
        "wrong_scene_margin_satisfied_sides": sum(
            value >= settings.wrong_scene_margin for value in wrong_scene_margins
        ),
        "mean_source_output_mse": sum(output_mse) / len(output_mse),
    }


def candidate_metadata_v79() -> dict[str, str]:
    """Return fixed diagnostic metadata; no category values are serialized."""

    return {
        "artifact": "v79_historical_relation_counterfactual_diagnostic_v1",
        "controller_architecture": "v75",
        "source_candidate_sha256": V79_SOURCE_SHA256,
        "preregistration_sha256": V79_PREREGISTRATION_SHA256,
        "training_pool_only": "true",
        "historical_train_pairs_only": "true",
        "selected_historical_rows": str(V79_SELECTED_ROWS),
        "selected_changed_sides": str(V79_CHANGED_SIDES),
        "cycles": str(V79_CYCLES),
        "optimizer_steps": str(V79_OPTIMIZER_STEPS),
        "held_optimization_rows": "0",
        "runtime_promotion_forbidden_until_gemma_gate": "true",
        "runtime_publication_artifact": "false",
        "answer_codebook_serialized": "false",
        "category_codebook_serialized": "false",
        "questions_answers_or_labels_serialized": "false",
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


def _finite_state_v79(
    model: DenseFullSceneContinuousControlV75,
) -> dict[str, torch.Tensor]:
    if type(model) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V79 diagnostic requires the exact V75 architecture")
    state = {
        key: value.detach().cpu().float().contiguous() for key, value in model.state_dict().items()
    }
    if frozenset(state) != V75_STATE_FIELDS:
        raise ValueError("V79 output tensor inventory changed")
    if any(
        not bool(value.is_floating_point() and torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("V79 output state became nonfinite or nonfloat")
    return state


def _atomic_create_v79(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_diagnostic_v79(
    path: str | Path, model: DenseFullSceneContinuousControlV75
) -> dict[str, Any]:
    destination = guard_output_v79(path, suffix=".safetensors")
    zero_audit = assert_dense_reader_exact_zero_scene(model)
    if model.environment_latents != 256 or model.hidden_size != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("V79 full-scene controller contract failed")
    state = _finite_state_v79(model)
    metadata = candidate_metadata_v79()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        save_file(state, temporary, metadata=metadata)
        with safe_open(str(temporary), framework="pt", device="cpu") as handle:
            if frozenset(handle.keys()) != V75_STATE_FIELDS:
                raise RuntimeError("V79 diagnostic reload inventory changed")
            if dict(handle.metadata() or {}) != metadata:
                raise RuntimeError("V79 diagnostic reload metadata changed")
        reloaded = load_file(str(temporary), device="cpu")
        if any(not torch.equal(reloaded[key], state[key]) for key in state):
            raise RuntimeError("V79 diagnostic failed exact tensor reload")
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


def run_v79_relation_counterfactual(
    *,
    preregistration: str | Path = V79_PREREGISTRATION,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Execute the fixed V79 optimizer. Importing the module initializes nothing."""

    if device_name not in {"auto", "mps", "cpu"}:
        raise ValueError("V79 device must be auto, mps, or cpu")
    prereg_path, prereg = load_preregistration_v79(preregistration)
    sources = prereg["sources"]
    optimization = prereg["optimization"]
    source_path = guard_input_v79(sources["v75_initial_candidate"]["path"], "exact V75 source")
    source_path, source_metadata = assert_exact_v75_nll_source_v77(source_path)
    v73_config_path = guard_input_v79(sources["v73_split_config"]["path"], "V73 split config")
    runtime_config_path = guard_input_v79(sources["runtime_config"]["path"], "runtime config")
    base_checkpoint = guard_input_v79(
        "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
        "base checkpoint",
    )
    output_candidate = guard_output_v79(optimization["candidate_output"], suffix=".safetensors")
    output_report = guard_output_v79(optimization["training_report_output"], suffix=".json")

    for source_name, source_spec in sources.items():
        if not isinstance(source_spec, Mapping) or "path" not in source_spec:
            continue
        input_path = guard_input_v79(source_spec["path"], source_name)
        if _sha256_file(input_path) != source_spec["sha256"]:
            raise ValueError(f"V79 preregistered source hash changed: {source_name}")

    v73 = load_config_v73(v73_config_path)
    qa_path = guard_input_v79(v73["training_qa"], "historical training QA")
    prefix_cache = guard_input_v79(v73["prefix_cache"], "immutable prefix cache")
    all_rows = load_training_rows_v73(qa_path)
    train_rows, held_rows = split_rows_v73(all_rows)
    selected = select_historical_relation_rows_v79(train_rows, held_rows)
    alternatives = canonical_alternatives_v77(train_rows)
    opposites = changed_opposites_v77(train_rows)
    schedule = deterministic_training_schedule_v77(
        selected,
        alternatives,
        cycles=V79_CYCLES,
        seed=V79_SEED,
    )
    if len(schedule) != V79_SELECTED_ROWS:
        raise RuntimeError("V79 locked schedule row count changed")
    measurement_schedule = deterministic_training_schedule_v77(
        selected,
        alternatives,
        cycles=1,
        seed=V79_SEED + 79,
    )[:V79_MEASUREMENT_ROWS]
    prefixes, prefix_manifest = load_prefixes_v73(
        prefix_cache,
        {row.scene_id for row in selected} | {row.paired_scene_id for row in selected},
    )
    if prefix_manifest["base_checkpoint_sha256"] != sources["base_checkpoint_sha256"]:
        raise ValueError("V79 base checkpoint hash changed")

    runtime_config, _ = _load_sanitized_runtime_config(runtime_config_path)
    runtime = StaticRuntimePrefixFactory(
        runtime_config, base_checkpoint, selected[0].scene_id
    ).bootstrap
    freeze_audit = freeze_base_runtime(runtime)
    _disable_decoder_checkpointing(runtime.language)
    device = _select_training_device(runtime, device_name)
    model, _ = load_exact_v75_nll_source_v77(source_path, device)
    source_model = copy.deepcopy(model).eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    source_zero_audit = assert_dense_reader_exact_zero_scene(source_model)
    trainable_audit = assert_exclusive_dense_reader_trainable_surface(runtime, model)
    if not torch.equal(
        runtime.scene_prefix.detach().cpu().float(),
        prefixes[selected[0].scene_id].detach().cpu().float(),
    ):
        raise ValueError("V79 cached prefix differs from frozen V54 runtime")
    questions = {
        question: _question_embeddings(runtime, question)
        for question in sorted({row.question for row in selected})
    }
    settings = LOCKED_SETTINGS_V79
    source_parameters = snapshot_source_parameters_v76(model)
    before = _measure_v79(
        runtime=runtime,
        model=model,
        source_model=source_model,
        schedule=measurement_schedule,
        prefixes=prefixes,
        questions=questions,
        opposites=opposites,
        settings=settings,
    )

    runtime.language.enable_decoder_gradient_checkpointing()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    fit_history: list[dict[str, Any]] = []
    online_correct: list[float] = []
    online_paired: list[float] = []
    online_wrong_scene: list[float] = []
    optimizer_steps = 0
    started = time.perf_counter()
    model.train()
    for offset in range(0, len(schedule), V79_GRADIENT_ACCUMULATION_ROWS):
        chunk = schedule[offset : offset + V79_GRADIENT_ACCUMULATION_ROWS]
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        changed_sides = 0
        for scheduled in chunk:
            correct, negative, paired, wrong_scene, output_anchor = _candidate_nlls_v79(
                runtime=runtime,
                model=model,
                source_model=source_model,
                scheduled=scheduled,
                prefixes=prefixes,
                question_embedding=questions[scheduled.row.question],
                opposite=opposites.get(scheduled.row.key),
            )
            loss, diagnostics = relation_objective_v79(
                correct_answer_nll=correct,
                negative_answer_nll=negative,
                paired_answer_nll=paired,
                wrong_scene_answer_nll=wrong_scene,
                source_output_mse=output_anchor,
                settings=settings,
            )
            (loss / len(chunk)).backward()
            losses.append(float(loss.detach().cpu()))
            online_correct.append(float(correct.detach().cpu()))
            if paired is not None and wrong_scene is not None:
                changed_sides += 1
                online_paired.append(float(diagnostics["paired_answer_margin"].detach().cpu()))
                online_wrong_scene.append(
                    float(diagnostics["wrong_scene_answer_margin"].detach().cpu())
                )

        weight_anchor = source_weight_anchor_l2_v76(model, source_parameters)
        (float(settings.source_weight_anchor_weight) * weight_anchor).backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().float().cpu()
        )
        if not math.isfinite(gradient):
            raise RuntimeError("V79 preclip gradient norm became nonfinite")
        optimizer.step()
        optimizer_steps += 1
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise RuntimeError("V79 optimizer produced nonfinite state")
        event = {
            "optimizer_step": optimizer_steps,
            "row_updates_completed": offset + len(chunk),
            "row_updates_total": len(schedule),
            "mean_chunk_loss": sum(losses) / len(losses),
            "changed_counterfactual_sides": changed_sides,
            "source_weight_anchor_l2": float(weight_anchor.detach().cpu()),
            "preclip_gradient_norm": gradient,
        }
        fit_history.append(event)
        print(json.dumps({"event": "v79_fit", **event}, sort_keys=True), flush=True)

    if optimizer_steps != V79_OPTIMIZER_STEPS:
        raise RuntimeError("V79 optimizer-step count violated preregistration")
    _disable_decoder_checkpointing(runtime.language)
    model.eval()
    after = _measure_v79(
        runtime=runtime,
        model=model,
        source_model=source_model,
        schedule=measurement_schedule,
        prefixes=prefixes,
        questions=questions,
        opposites=opposites,
        settings=settings,
    )
    final_weight_anchor = float(
        source_weight_anchor_l2_v76(model, source_parameters).detach().cpu()
    )
    diagnostic = save_diagnostic_v79(output_candidate, model)
    report = {
        "artifact": "v79_historical_relation_counterfactual_training_v1",
        "preregistration": {
            "path": str(prereg_path.relative_to(PROJECT_ROOT)),
            "sha256": V79_PREREGISTRATION_SHA256,
            "authenticated_before_training": True,
        },
        "scope": {
            "historical_training_pool_only": True,
            "historical_optimization_rows": V79_SELECTED_ROWS,
            "historical_internal_held_optimization_rows": 0,
            "train_scene_count": len({row.scene_id for row in selected}),
            "held_scene_count": len({row.scene_id for row in held_rows}),
            "train_held_scene_overlap": 0,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
            "runtime_promotion_authorized": False,
            "checkpoint_published": False,
        },
        "source_candidate": {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "sha256": V79_SOURCE_SHA256,
            "metadata": source_metadata,
        },
        "selection": {
            "selected_rows": len(selected),
            "changed_sides": sum(row.expected_change for row in selected),
            "stable_rows": sum(not row.expected_change for row in selected),
            "distinct_question_templates": len({row.question for row in selected}),
            "distinct_answer_classes": len({row.answer_class for row in selected}),
            "change_type_counts": dict(
                sorted(Counter(row.change_type for row in selected).items())
            ),
            "answer_class_values_serialized": False,
            "question_or_answer_text_serialized": False,
        },
        "schedule": {
            "cycles": V79_CYCLES,
            "row_updates": len(schedule),
            "gradient_accumulation_rows": V79_GRADIENT_ACCUMULATION_ROWS,
            "optimizer_steps": optimizer_steps,
            "seed": V79_SEED,
            "exhaustive_within_locked_filter": True,
        },
        "loss_settings": asdict(settings),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-6,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
        },
        "measurement": {
            "locked_row_count": V79_MEASUREMENT_ROWS,
            "before": before,
            "after": after,
            "mean_correct_answer_nll_delta": float(after["mean_correct_answer_nll"])
            - float(before["mean_correct_answer_nll"]),
            "mean_paired_answer_margin_delta": float(after["mean_paired_answer_margin"])
            - float(before["mean_paired_answer_margin"]),
            "mean_wrong_scene_answer_margin_delta": float(after["mean_wrong_scene_answer_margin"])
            - float(before["mean_wrong_scene_answer_margin"]),
        },
        "training_online": {
            "mean_preupdate_correct_answer_nll": sum(online_correct) / len(online_correct),
            "changed_side_updates": len(online_paired),
            "mean_preupdate_paired_answer_margin": sum(online_paired) / len(online_paired),
            "mean_preupdate_wrong_scene_answer_margin": sum(online_wrong_scene)
            / len(online_wrong_scene),
        },
        "final_source_weight_anchor_l2": final_weight_anchor,
        "elapsed_training_seconds": time.perf_counter() - started,
        "diagnostic_candidate": diagnostic,
        "base_freeze_audit": freeze_audit,
        "exclusive_trainable_audit": trainable_audit,
        "source_exact_zero_audit": source_zero_audit,
        "after_exact_zero_audit": assert_dense_reader_exact_zero_scene(model),
        "prefix_manifest_base_checkpoint_sha256": prefix_manifest["base_checkpoint_sha256"],
        "fit_history": fit_history,
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_create_v79(output_report, encoded)
    print(
        json.dumps(
            {
                "event": "v79_training_complete",
                "candidate": diagnostic["path"],
                "candidate_sha256": diagnostic["sha256"],
                "report": str(output_report.relative_to(PROJECT_ROOT)),
                "optimizer_steps": optimizer_steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=V79_PREREGISTRATION)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_v79_relation_counterfactual(preregistration=args.preregistration, device_name=args.device)
    return 0


__all__ = [
    "LOCKED_SETTINGS_V79",
    "V79_CHANGED_SIDES",
    "V79_MEASUREMENT_ROWS",
    "V79_OPTIMIZER_STEPS",
    "V79_PREREGISTRATION",
    "V79_PREREGISTRATION_SHA256",
    "V79_RELATION_ANSWER_TYPE",
    "V79_SEED",
    "V79_SELECTED_ROWS",
    "V79_SOURCE_SHA256",
    "V79LossSettings",
    "candidate_metadata_v79",
    "guard_input_v79",
    "guard_output_v79",
    "load_preregistration_v79",
    "relation_objective_v79",
    "run_v79_relation_counterfactual",
    "save_diagnostic_v79",
    "select_historical_relation_rows_v79",
]


if __name__ == "__main__":
    raise SystemExit(main())
