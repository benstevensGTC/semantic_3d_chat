"""V67 pair-aware continuous-controller training and fail-closed publication.

V67 is a failure-driven successor to V66b.  It keeps the same all-row numeric
teacher targets and the same V7 runtime architecture, then adds an explicit
atomic paired-scene refinement.  Identical questions on opposite physical
scenes must produce the corresponding native-width teacher, align their
scene-to-scene output delta, and beat the exact paired-opposite teacher by a
fixed cosine margin.

The ``screen`` mode performs all twelve leave-one-pair-out fits and numeric
checks without greedy generation.  ``full`` refuses to start generation unless
that create-once screen report passed the preregistered gate.  A runtime
checkpoint is published only after unchanged V66b behavioral, paired-opposite,
strict-reload, and saved-runtime generation gates all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    TRAIN_PAIR_IDS,
)
from semantic_3d_chat.evaluation.v67_pair_objective_preregistration import (
    V67_HYPERPARAMETERS,
    V67_NUMERIC_SCREEN_THRESHOLDS,
    build_v67_preregistration,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training import train_question_control_v66 as v66
from semantic_3d_chat.training.question_control_v66_objective import (
    numeric_prototype_classification_loss,
)
from semantic_3d_chat.training.question_control_v66_prototypes import (
    HybridAnswerPrototypeCodebookV66,
    answer_class_id_v66,
    build_hybrid_answer_prototype_codebook_v66,
)
from semantic_3d_chat.training.question_control_v67_objective import (
    paired_scene_dependence_loss_v67,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _resolve,
    _sha256_file,
    _write_training_report,
)
from semantic_3d_chat.training.train_question_control_v63 import (
    V63Preflight,
    V63Row,
    _changed_units,
    _optimizer_step,
    _scene_signatures,
    build_v63_preflight,
)
from semantic_3d_chat.training.train_question_control_v65 import (
    V65RuntimeBundle,
    _load_runtime,
    _tensor_sha256,
    validate_training_baseline_lock,
)

_EXPECTED_ROWS: Final[int] = 576
_EXPECTED_CLASSES: Final[int] = 28
_PINNED_PREREGISTRATION_SHA256: Final[str] = (
    "a87ad59102c48da95390659839b76707c3d32af726034ab930fae5e01ba7ab8f"
)
_WORK_ARTIFACT: Final[str] = "v67_pair_objective_work_v1"
_FOLD_ARTIFACT: Final[str] = "v67_pair_objective_fold_v1"
_SCREEN_ARTIFACT: Final[str] = "v67_pair_objective_numeric_screen_v1"
_FULL_ARTIFACT: Final[str] = "v67_pair_objective_behavioral_training_v1"


@dataclass(frozen=True)
class V67FitResult:
    base: v66.V66FitResult
    refinement_optimizer_steps: int
    refinement_elapsed_seconds: float
    train_pair_diagnostics: dict[str, float | int]
    train_prototype_top1_accuracy: float

    @property
    def control(self) -> AlwaysOnTeacherBasisFullSceneQuestionControlV7:
        return self.base.control

    @property
    def signatures(self) -> dict[str, torch.Tensor]:
        return self.base.signatures


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def validate_v67_preregistration(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V67 preregistration is unavailable")
    if _sha256_file(source) != _PINNED_PREREGISTRATION_SHA256:
        raise ValueError("V67 preregistration differs from its pre-run pin")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload != build_v67_preregistration():
        raise ValueError("V67 preregistration semantic contract changed")
    return payload


def _locked_fit_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the V63/V66-compatible namespace from preregistered constants."""

    hp = V67_HYPERPARAMETERS
    values = vars(args).copy()
    values.update(
        {
            "seed": hp["seed"],
            "basis_rank": hp["basis_rank"],
            "moment_count": hp["moment_count"],
            "interaction_dim": hp["interaction_dim"],
            "trunk_dim": hp["trunk_dim"],
            "maximum_control_rms": hp["maximum_control_rms"],
            "initial_control_rms": hp["initial_control_rms"],
            "gate_threshold": 0.5,
            "epochs": hp["base_epochs"],
            "batch_size": hp["base_batch_size"],
            "changed_repeats": 1,
            "learning_rate": hp["base_learning_rate"],
            "weight_decay": hp["base_weight_decay"],
            "gradient_clip_norm": hp["gradient_clip_norm"],
            "coefficient_weight": 3.0,
            "log_rms_weight": 1.0,
            "reconstruction_weight": 2.0,
            "relative_mse_weight": 0.25,
            "prototype_classification_epochs": hp[
                "prototype_classification_epochs"
            ],
            "prototype_classification_weight": hp[
                "prototype_classification_weight"
            ],
            "prototype_classification_temperature": hp[
                "prototype_classification_temperature"
            ],
            "prototype_value_preservation_weight": hp[
                "prototype_value_preservation_weight"
            ],
            "pair_delta_weight": 0.0,
            "route_weight": 0.0,
            "log_every": 20,
        }
    )
    return argparse.Namespace(**values)


def _prototype_bank(
    codebook: HybridAnswerPrototypeCodebookV66,
) -> tuple[list[str], torch.Tensor, dict[str, int]]:
    class_ids = sorted(codebook.prototypes)
    if len(class_ids) < 2:
        raise ValueError("V67 requires at least two fold-local numeric classes")
    bank = torch.cat([codebook.prototypes[class_id] for class_id in class_ids])
    if bank.ndim != 3 or bank.shape[1:] != (4, 1536) or not torch.isfinite(bank).all():
        raise ValueError("V67 fold-local prototype bank changed shape or finiteness")
    return class_ids, bank.float(), {
        class_id: index for index, class_id in enumerate(class_ids)
    }


def _simple_value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("V67 value preservation expects matching [B,C,H]")
    cosine = F.cosine_similarity(predicted.float(), target.float(), dim=-1)
    power = target.float().square().mean().clamp_min(1e-8)
    return 1.0 - cosine.mean() + 0.10 * F.mse_loss(
        predicted.float(), target.float()
    ) / power


def _measure_train_pairs(
    control: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    units: Sequence[tuple[V63Row, V63Row]],
    *,
    signatures: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    targets: Mapping[tuple[str, str], torch.Tensor],
) -> dict[str, float | int]:
    flat = [row for unit in units for row in unit]
    with torch.inference_mode():
        predicted = control.forward_from_signature(
            torch.cat([signatures[row.scene_id] for row in flat]),
            torch.cat([questions[row.key] for row in flat]),
        ).control_tokens.reshape(len(units), 2, 4, 1536)
        expected = torch.cat([targets[row.key] for row in flat]).reshape(
            len(units), 2, 4, 1536
        )
        _loss, diagnostics = paired_scene_dependence_loss_v67(
            predicted,
            expected,
            opposite_margin=float(V67_HYPERPARAMETERS["pair_opposite_margin"]),
            value_weight=1.0,
            delta_weight=1.0,
            opposite_weight=1.0,
        )
    return {
        "changed_unit_count": len(units),
        "changed_side_count": len(flat),
        "mean_own_cosine": float(diagnostics.mean_own_cosine.cpu()),
        "mean_opposite_cosine": float(diagnostics.mean_opposite_cosine.cpu()),
        "mean_own_over_opposite_margin": float(
            diagnostics.mean_own_over_opposite_margin.cpu()
        ),
        "positive_own_over_opposite_fraction": float(
            diagnostics.positive_own_over_opposite_fraction.cpu()
        ),
        "mean_delta_cosine": float(diagnostics.mean_delta_cosine.cpu()),
        "positive_delta_fraction": float(diagnostics.positive_delta_fraction.cpu()),
    }


def _pair_refinement_parameters(
    control: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
) -> list[torch.nn.Parameter]:
    """Freeze inherited normalization/routing state and return value parameters.

    ``AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3`` copies tensor
    values through a state dict, which intentionally does not carry PyTorch's
    ``requires_grad`` flags.  The source V3 question normalization is frozen,
    so restore that training contract explicitly before the V67 refinement
    optimizer is constructed.
    """

    control.question_norm.requires_grad_(False)
    for name, parameter in control.named_parameters():
        if name.startswith("route_"):
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in control.parameters() if parameter.requires_grad]
    if (
        not trainable
        or any(parameter.requires_grad for parameter in control.question_norm.parameters())
        or any(
            parameter.requires_grad
            for name, parameter in control.named_parameters()
            if name.startswith("route_")
        )
    ):
        raise RuntimeError("V67 optimizer scope changed")
    return trainable


def _fit_pair_aware(
    *,
    rows: Sequence[V63Row],
    codebook: HybridAnswerPrototypeCodebookV66,
    preflight: V63Preflight,
    questions: Mapping[tuple[str, str], torch.Tensor],
    basis: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    phase: str,
) -> V67FitResult:
    """Run V66 all-row fitting followed by atomic changed-pair refinement."""

    base = v66._fit_always_on(
        rows=rows,
        targets=codebook.targets,
        preflight=preflight,
        questions=questions,
        basis=basis,
        args=args,
        seed=seed,
        phase=f"{phase}_base",
    )
    control = base.control.train()
    signatures = base.signatures
    units = _changed_units(tuple(row for row in rows if row.route_label))
    if not units:
        raise ValueError("V67 pair refinement requires changed counterfactual units")
    for left, right in units:
        if (
            left.question.encode("utf-8") != right.question.encode("utf-8")
            or not torch.equal(questions[left.key], questions[right.key])
            or answer_class_id_v66(left.answer) == answer_class_id_v66(right.answer)
            or _tensor_sha256(codebook.targets[left.key])
            == _tensor_sha256(codebook.targets[right.key])
        ):
            raise ValueError("V67 changed unit is not an exact paired opposite")

    _class_ids, prototype_bank, class_index = _prototype_bank(codebook)
    retention = [row for row in rows if not row.route_label]
    if not retention:
        raise ValueError("V67 requires stable rows for retention")
    trainable = _pair_refinement_parameters(control)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(V67_HYPERPARAMETERS["pair_learning_rate"]),
        weight_decay=float(V67_HYPERPARAMETERS["base_weight_decay"]),
    )
    steps = 0
    started = time.perf_counter()
    epochs = int(V67_HYPERPARAMETERS["pair_refinement_epochs"])
    repeats = int(V67_HYPERPARAMETERS["pair_refinement_repeats"])
    unit_batch_size = int(V67_HYPERPARAMETERS["pair_unit_batch_size"])
    retention_batch_size = int(V67_HYPERPARAMETERS["retention_batch_size"])
    retention_batches = int(V67_HYPERPARAMETERS["retention_batches_per_epoch"])
    for epoch in range(epochs):
        for repeat in range(repeats):
            ordered_units = list(units)
            random.Random(seed + epoch * 1_000_003 + repeat * 10_007).shuffle(
                ordered_units
            )
            for offset in range(0, len(ordered_units), unit_batch_size):
                batch_units = ordered_units[offset : offset + unit_batch_size]
                flat = [row for unit in batch_units for row in unit]
                output = control.forward_from_signature(
                    torch.cat([signatures[row.scene_id] for row in flat]),
                    torch.cat([questions[row.key] for row in flat]),
                ).control_tokens
                target = torch.cat([codebook.targets[row.key] for row in flat])
                pair_loss, _diagnostics = paired_scene_dependence_loss_v67(
                    output.reshape(len(batch_units), 2, 4, 1536),
                    target.reshape(len(batch_units), 2, 4, 1536),
                    opposite_margin=float(
                        V67_HYPERPARAMETERS["pair_opposite_margin"]
                    ),
                    value_weight=float(V67_HYPERPARAMETERS["pair_value_weight"]),
                    delta_weight=float(V67_HYPERPARAMETERS["pair_delta_weight"]),
                    opposite_weight=float(
                        V67_HYPERPARAMETERS["pair_opposite_weight"]
                    ),
                )
                indices = torch.tensor(
                    [class_index[codebook.class_by_key[row.key]] for row in flat],
                    dtype=torch.long,
                )
                class_loss, _class_diagnostics = numeric_prototype_classification_loss(
                    output,
                    prototype_bank,
                    indices,
                    temperature=float(
                        V67_HYPERPARAMETERS[
                            "prototype_classification_temperature"
                        ]
                    ),
                )
                _optimizer_step(
                    loss=(
                        pair_loss
                        + float(V67_HYPERPARAMETERS["pair_classification_weight"])
                        * class_loss
                    ),
                    control=control,
                    optimizer=optimizer,
                    gradient_clip_norm=float(
                        V67_HYPERPARAMETERS["gradient_clip_norm"]
                    ),
                )
                steps += 1

        ordered_retention = list(retention)
        random.Random(seed + 70_000_019 + epoch * 1_000_003).shuffle(
            ordered_retention
        )
        for batch_index in range(retention_batches):
            start = batch_index * retention_batch_size
            batch = ordered_retention[start : start + retention_batch_size]
            if not batch:
                break
            output = control.forward_from_signature(
                torch.cat([signatures[row.scene_id] for row in batch]),
                torch.cat([questions[row.key] for row in batch]),
            ).control_tokens
            target = torch.cat([codebook.targets[row.key] for row in batch])
            indices = torch.tensor(
                [class_index[codebook.class_by_key[row.key]] for row in batch],
                dtype=torch.long,
            )
            class_loss, _diagnostics = numeric_prototype_classification_loss(
                output,
                prototype_bank,
                indices,
                temperature=float(
                    V67_HYPERPARAMETERS["prototype_classification_temperature"]
                ),
            )
            retention_loss = _simple_value_loss(output, target) + class_loss
            _optimizer_step(
                loss=float(V67_HYPERPARAMETERS["retention_weight"])
                * retention_loss,
                control=control,
                optimizer=optimizer,
                gradient_clip_norm=float(V67_HYPERPARAMETERS["gradient_clip_norm"]),
            )
            steps += 1

    control.eval()
    signatures = _scene_signatures(
        control,
        {
            scene_id: preflight.prefixes[scene_id]
            for scene_id in sorted({row.scene_id for row in rows})
        },
    )
    pair_diagnostics = _measure_train_pairs(
        control,
        units,
        signatures=signatures,
        questions=questions,
        targets=codebook.targets,
    )
    with torch.inference_mode():
        predicted = control.forward_from_signature(
            torch.cat([signatures[row.scene_id] for row in rows]),
            torch.cat([questions[row.key] for row in rows]),
        ).control_tokens
        indices = torch.tensor(
            [class_index[codebook.class_by_key[row.key]] for row in rows],
            dtype=torch.long,
        )
        _loss, prototype_diagnostics = numeric_prototype_classification_loss(
            predicted,
            prototype_bank,
            indices,
            temperature=float(
                V67_HYPERPARAMETERS["prototype_classification_temperature"]
            ),
        )
    rebuilt_base = v66.V66FitResult(
        control=control,
        signatures=signatures,
        base_fit=base.base_fit,
        classification_optimizer_steps=base.classification_optimizer_steps,
        numeric_prototype_top1_accuracy=base.numeric_prototype_top1_accuracy,
        numeric_prototype_mean_margin=base.numeric_prototype_mean_margin,
    )
    return V67FitResult(
        base=rebuilt_base,
        refinement_optimizer_steps=steps,
        refinement_elapsed_seconds=time.perf_counter() - started,
        train_pair_diagnostics=pair_diagnostics,
        train_prototype_top1_accuracy=float(
            prototype_diagnostics.top1_accuracy.cpu()
        ),
    )


def numeric_screen_fold_v67(
    fit: V67FitResult,
    held_rows: Sequence[V63Row],
    *,
    codebook: HybridAnswerPrototypeCodebookV66,
    questions: Mapping[tuple[str, str], torch.Tensor],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Classify held controls against a fold-local numeric bank, without Gemma."""

    class_ids, prototype_bank, class_index = _prototype_bank(codebook)
    supported = [
        row for row in held_rows if answer_class_id_v66(row.answer) in class_index
    ]
    unsupported = len(held_rows) - len(supported)
    with torch.inference_mode():
        output = fit.control.forward_from_signature(
            torch.cat([fit.signatures[row.scene_id] for row in supported]),
            torch.cat([questions[row.key] for row in supported]),
        ).control_tokens
    predicted_flat = F.normalize(output.float().flatten(1), dim=-1, eps=1e-8)
    bank_flat = F.normalize(prototype_bank.float().flatten(1), dim=-1, eps=1e-8)
    predicted_indices = (predicted_flat @ bank_flat.T).argmax(dim=1).tolist()
    row_records: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], tuple[V63Row, torch.Tensor, str, str]] = {}
    for row, prediction, predicted_index in zip(
        supported, output, predicted_indices, strict=True
    ):
        target_class = answer_class_id_v66(row.answer)
        predicted_class = class_ids[int(predicted_index)]
        row_records.append(
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "pair_id": row.pair_id,
                "question_key": row.question_key,
                "changed_side": row.route_label,
                "target_class_id": target_class,
                "predicted_class_id": predicted_class,
                "class_exact": predicted_class == target_class,
                "control_sha256": _tensor_sha256(prediction.unsqueeze(0)),
            }
        )
        by_key[row.key] = (row, prediction, target_class, predicted_class)

    grouped: defaultdict[tuple[str, str], list[V63Row]] = defaultdict(list)
    for row in held_rows:
        if row.route_label and row.key in by_key:
            grouped[(row.pair_id, row.question_key)].append(row)
    pair_records: list[dict[str, Any]] = []
    for (pair_id, question_key), sides in sorted(grouped.items()):
        if len(sides) != 2:
            continue
        left, right = sorted(sides, key=lambda row: (row.scene_id, row.question_id))
        left_info, right_info = by_key[left.key], by_key[right.key]
        predicted = torch.stack([left_info[1], right_info[1]])
        target = torch.cat(
            [
                codebook.prototypes[left_info[2]],
                codebook.prototypes[right_info[2]],
            ]
        )
        predicted_delta = predicted[0] - predicted[1]
        target_delta = target[0] - target[1]
        delta_cosine = float(
            F.cosine_similarity(
                predicted_delta.flatten().unsqueeze(0),
                target_delta.flatten().unsqueeze(0),
                dim=-1,
                eps=1e-8,
            )[0]
        )
        predicted_normalized = F.normalize(predicted.flatten(1), dim=-1, eps=1e-8)
        target_normalized = F.normalize(target.flatten(1), dim=-1, eps=1e-8)
        own = (predicted_normalized * target_normalized).sum(dim=-1)
        opposite = (predicted_normalized * target_normalized.flip(dims=(0,))).sum(
            dim=-1
        )
        margins = own - opposite
        pair_records.append(
            {
                "pair_id": pair_id,
                "question_key": question_key,
                "left_scene_id": left.scene_id,
                "right_scene_id": right.scene_id,
                "both_class_exact": left_info[3] == left_info[2]
                and right_info[3] == right_info[2],
                "prediction_changed": left_info[3] != right_info[3],
                "pair_delta_cosine": delta_cosine,
                "own_over_opposite_margins": [float(value) for value in margins],
            }
        )
    metrics = _numeric_metrics_from_records(
        row_records, pair_records, unsupported_count=unsupported
    )
    evidence = tuple(
        [{"kind": "row", **record} for record in row_records]
        + [{"kind": "pair", **record} for record in pair_records]
    )
    return metrics, evidence


def _numeric_metrics_from_records(
    row_records: Sequence[Mapping[str, Any]],
    pair_records: Sequence[Mapping[str, Any]],
    *,
    unsupported_count: int,
) -> dict[str, Any]:
    changed_rows = [record for record in row_records if record["changed_side"] is True]
    deltas = [float(record["pair_delta_cosine"]) for record in pair_records]
    margins = [
        float(margin)
        for record in pair_records
        for margin in record["own_over_opposite_margins"]
    ]
    return {
        "supported_class_exact": sum(record["class_exact"] is True for record in row_records),
        "supported_total": len(row_records),
        "unsupported_total": unsupported_count,
        "inventory_total": len(row_records) + unsupported_count,
        "changed_class_exact": sum(record["class_exact"] is True for record in changed_rows),
        "changed_total": len(changed_rows),
        "complete_class_units": sum(
            record["both_class_exact"] is True for record in pair_records
        ),
        "complete_unit_total": len(pair_records),
        "prediction_change_units": sum(
            record["prediction_changed"] is True for record in pair_records
        ),
        "pair_delta_cosine_sum": sum(deltas),
        "positive_pair_delta_units": sum(value > 0.0 for value in deltas),
        "own_over_opposite_margin_sum": sum(margins),
        "positive_own_over_opposite_sides": sum(value > 0.0 for value in margins),
        "fully_supported_pair_sides": len(margins),
        "question_or_answer_text_stored": False,
        "gemma_generation_used": False,
    }


def aggregate_numeric_screens_v67(
    fold_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(fold_metrics) != len(TRAIN_PAIR_IDS):
        raise ValueError("V67 numeric screen requires all twelve held-pair folds")
    sum_fields = (
        "supported_class_exact",
        "supported_total",
        "unsupported_total",
        "inventory_total",
        "changed_class_exact",
        "changed_total",
        "complete_class_units",
        "complete_unit_total",
        "prediction_change_units",
        "pair_delta_cosine_sum",
        "positive_pair_delta_units",
        "own_over_opposite_margin_sum",
        "positive_own_over_opposite_sides",
        "fully_supported_pair_sides",
    )
    aggregate = {
        field: sum(float(fold[field]) for fold in fold_metrics) for field in sum_fields
    }
    for field in sum_fields:
        if field.endswith("_sum"):
            continue
        aggregate[field] = int(aggregate[field])
    pair_total = int(aggregate["complete_unit_total"])
    side_total = int(aggregate["fully_supported_pair_sides"])
    aggregate["mean_pair_delta_cosine"] = (
        float(aggregate["pair_delta_cosine_sum"]) / pair_total
    )
    aggregate["mean_own_over_opposite_margin"] = (
        float(aggregate["own_over_opposite_margin_sum"]) / side_total
    )
    aggregate["answer_or_question_text_stored"] = False
    aggregate["gemma_generation_used"] = False
    return aggregate


def assess_numeric_screen_v67(metrics: Mapping[str, Any]) -> dict[str, bool]:
    threshold = V67_NUMERIC_SCREEN_THRESHOLDS
    return {
        "held_supported_class_exact": int(metrics["supported_class_exact"])
        >= threshold.held_supported_class_exact_minimum,
        "held_supported_total": int(metrics["supported_total"])
        == threshold.held_supported_total,
        "held_unsupported_total": int(metrics["unsupported_total"])
        == threshold.held_unsupported_total,
        "complete_inventory": int(metrics["inventory_total"]) == _EXPECTED_ROWS,
        "held_changed_class_exact": int(metrics["changed_class_exact"])
        >= threshold.held_changed_class_exact_minimum,
        "held_changed_total": int(metrics["changed_total"])
        == threshold.held_changed_total,
        "held_complete_class_units": int(metrics["complete_class_units"])
        >= threshold.held_complete_class_units_minimum,
        "held_complete_unit_total": int(metrics["complete_unit_total"])
        == threshold.held_complete_unit_total,
        "held_prediction_change_units": int(metrics["prediction_change_units"])
        >= threshold.held_prediction_change_units_minimum,
        "mean_pair_delta_cosine": float(metrics["mean_pair_delta_cosine"])
        >= threshold.mean_pair_delta_cosine_minimum,
        "positive_pair_delta_units": int(metrics["positive_pair_delta_units"])
        >= threshold.positive_pair_delta_units_minimum,
        "mean_own_over_opposite_margin": float(
            metrics["mean_own_over_opposite_margin"]
        )
        >= threshold.mean_own_over_opposite_margin_minimum,
        "positive_own_over_opposite_sides": int(
            metrics["positive_own_over_opposite_sides"]
        )
        >= threshold.positive_own_over_opposite_sides_minimum,
        "fully_supported_pair_sides": int(metrics["fully_supported_pair_sides"])
        == threshold.fully_supported_pair_sides,
        "no_text_stored": metrics.get("answer_or_question_text_stored") is False,
        "no_generation_used": metrics.get("gemma_generation_used") is False,
    }


def _training_identity(
    preflight: V63Preflight,
    teacher_audit: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "artifact": "v67_pair_objective_training_identity_v1",
        "preregistration_sha256": _sha256_file(_resolve(args.preregistration)),
        "filtered_training_qa_sha256": preflight.filtered_train_sha256,
        "training_baseline_lock_sha256": _sha256_file(
            _resolve(args.training_baseline_lock)
        ),
        "base_checkpoint_sha256": preflight.base_checkpoint_sha256,
        "runtime_config_sha256": preflight.runtime_config_sha256,
        "prefix_cache_manifest_sha256": preflight.prefix_manifest_sha256,
        "combined_teacher_audit_sha256": _canonical_sha256(teacher_audit),
        "fixed_hyperparameters": dict(V67_HYPERPARAMETERS),
        "pair_ids": list(TRAIN_PAIR_IDS),
        "implementation_files_sha256": {
            "trainer": _sha256_file(Path(__file__).resolve()),
            "pair_objective": _sha256_file(
                Path(paired_scene_dependence_loss_v67.__code__.co_filename).resolve()
            ),
            "v66_base_trainer": _sha256_file(Path(v66.__file__).resolve()),
        },
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "internal_validation_loaded": False,
        "deferred_final_loaded": False,
    }
    return {**identity, "sha256": _canonical_sha256(identity)}


def validate_screen_authorization_v67(
    path: str | Path,
    *,
    expected_training_identity_sha256: str,
) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V67 full run requires a regular screen report")
    payload = json.loads(source.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    checks = payload.get("checks") if isinstance(payload, dict) else None
    expected_checks = (
        assess_numeric_screen_v67(metrics)
        if isinstance(metrics, Mapping)
        else None
    )
    scope = payload.get("scope") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("artifact") != _SCREEN_ARTIFACT
        or payload.get("passed") is not True
        or payload.get("training_identity_sha256")
        != expected_training_identity_sha256
        or payload.get("preregistration_sha256")
        != _PINNED_PREREGISTRATION_SHA256
        or payload.get("thresholds") != asdict(V67_NUMERIC_SCREEN_THRESHOLDS)
        or not isinstance(checks, Mapping)
        or checks != expected_checks
        or not all(checks.values())
        or payload.get("checkpoint_published") is not False
        or payload.get("gemma_generation_used") is not False
        or not isinstance(scope, Mapping)
        or scope.get("question_or_answer_text_stored") is not False
        or any(
            scope.get(field) is not False
            for field in (
                "validation_inputs_used",
                "scorer_inputs_used",
                "oracle_loaded",
                "fresh_development_loaded",
                "internal_validation_loaded",
                "deferred_final_loaded",
            )
        )
    ):
        raise ValueError("V67 screen authorization did not pass the locked contract")
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "training_identity_sha256": expected_training_identity_sha256,
        "passed": True,
    }


def _fit_audit(fit: V67FitResult) -> dict[str, Any]:
    return {
        "base_optimizer_steps": fit.base.base_fit.optimizer_steps,
        "base_classification_optimizer_steps": fit.base.classification_optimizer_steps,
        "pair_refinement_optimizer_steps": fit.refinement_optimizer_steps,
        "base_elapsed_seconds": fit.base.base_fit.elapsed_seconds,
        "pair_refinement_elapsed_seconds": fit.refinement_elapsed_seconds,
        "question_norm_sha256": fit.base.base_fit.question_norm_sha256,
        "question_norm_frozen": fit.base.base_fit.question_norm_frozen,
        "base_numeric_prototype_top1_accuracy": (
            fit.base.numeric_prototype_top1_accuracy
        ),
        "post_refinement_numeric_prototype_top1_accuracy": (
            fit.train_prototype_top1_accuracy
        ),
        "train_pair_diagnostics": fit.train_pair_diagnostics,
    }


def _validate_cached_fold_v67(
    payload: object,
    *,
    mode: str,
    held_pair_id: str,
    run_signature_sha256: str,
    held_rows: Sequence[V63Row],
    codebook: HybridAnswerPrototypeCodebookV66,
    basis: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("V67 cached fold must be an object")
    required = {
        "schema_version",
        "artifact",
        "run_signature_sha256",
        "held_pair_id",
        "held_rows_used_for_optimization",
        "held_teacher_sources_used",
        "fold_codebook_sha256",
        "fold_basis_sha256",
        "fit",
        "numeric_screen",
        "numeric_evidence",
    }
    if mode == "full":
        required |= {"behavior", "records"}
    if set(payload) != required:
        raise ValueError("V67 cached fold fields changed")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact") != _FOLD_ARTIFACT
        or payload.get("run_signature_sha256") != run_signature_sha256
        or payload.get("held_pair_id") != held_pair_id
        or payload.get("held_rows_used_for_optimization") is not False
        or payload.get("held_teacher_sources_used") is not False
        or payload.get("fold_codebook_sha256") != codebook.sha256
        or payload.get("fold_basis_sha256") != _tensor_sha256(basis)
    ):
        raise ValueError("V67 cached fold provenance changed")
    evidence = payload.get("numeric_evidence")
    if not isinstance(evidence, list):
        raise TypeError("V67 cached numeric evidence must be a list")
    row_records: list[Mapping[str, Any]] = []
    pair_records: list[Mapping[str, Any]] = []
    for raw in evidence:
        if not isinstance(raw, dict) or raw.get("kind") not in {"row", "pair"}:
            raise ValueError("V67 cached numeric evidence record changed")
        record = {key: value for key, value in raw.items() if key != "kind"}
        (row_records if raw["kind"] == "row" else pair_records).append(record)
    by_key = {row.key: row for row in held_rows}
    expected_supported = {
        row.key
        for row in held_rows
        if answer_class_id_v66(row.answer) in codebook.prototypes
    }
    observed: set[tuple[str, str]] = set()
    for record in row_records:
        key = str(record.get("scene_id")), str(record.get("question_id"))
        if key in observed or key not in expected_supported:
            raise ValueError("V67 cached numeric row inventory changed")
        row = by_key[key]
        predicted_class = record.get("predicted_class_id")
        expected = {
            "pair_id": row.pair_id,
            "question_key": row.question_key,
            "changed_side": row.route_label,
            "target_class_id": answer_class_id_v66(row.answer),
            "class_exact": predicted_class == answer_class_id_v66(row.answer),
        }
        if (
            predicted_class not in codebook.prototypes
            or any(record.get(field) != value for field, value in expected.items())
            or not v66._is_sha256(record.get("control_sha256"))
        ):
            raise ValueError("V67 cached numeric row differs from held inventory")
        observed.add(key)
    if observed != expected_supported:
        raise ValueError("V67 cached numeric row coverage changed")
    expected_pair_keys: set[tuple[str, str]] = set()
    grouped: defaultdict[tuple[str, str], list[V63Row]] = defaultdict(list)
    for row in held_rows:
        if row.route_label and row.key in expected_supported:
            grouped[(row.pair_id, row.question_key)].append(row)
    expected_pair_keys = {
        key for key, sides in grouped.items() if len(sides) == 2
    }
    observed_pairs: set[tuple[str, str]] = set()
    for record in pair_records:
        key = str(record.get("pair_id")), str(record.get("question_key"))
        margins = record.get("own_over_opposite_margins")
        if (
            key in observed_pairs
            or key not in expected_pair_keys
            or not isinstance(record.get("both_class_exact"), bool)
            or not isinstance(record.get("prediction_changed"), bool)
            or not isinstance(record.get("pair_delta_cosine"), (int, float))
            or not math.isfinite(float(record["pair_delta_cosine"]))
            or not isinstance(margins, list)
            or len(margins) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in margins
            )
        ):
            raise ValueError("V67 cached pair evidence changed")
        observed_pairs.add(key)
    if observed_pairs != expected_pair_keys:
        raise ValueError("V67 cached pair evidence coverage changed")
    recomputed_numeric = _numeric_metrics_from_records(
        row_records,
        pair_records,
        unsupported_count=len(held_rows) - len(expected_supported),
    )
    if payload.get("numeric_screen") != recomputed_numeric:
        raise ValueError("V67 cached numeric metrics differ from evidence")
    if mode == "full":
        records = payload.get("records")
        if not isinstance(records, list):
            raise TypeError("V67 cached behavioral records must be a list")
        behavior = v66.behavior_metrics_v66(
            records,
            unsupported_count=len(held_rows) - len(expected_supported),
        )
        if payload.get("behavior") != behavior:
            raise ValueError("V67 cached behavior differs from record evidence")
        behavior_keys = {
            (str(record.get("scene_id")), str(record.get("question_id")))
            for record in records
        }
        if behavior_keys != expected_supported or len(behavior_keys) != len(records):
            raise ValueError("V67 cached behavioral row coverage changed")
        for record in records:
            row = by_key[(str(record["scene_id"]), str(record["question_id"]))]
            if (
                record.get("pair_id") != row.pair_id
                or record.get("question_key") != row.question_key
                or record.get("answer_class_id") != answer_class_id_v66(row.answer)
                or record.get("reference_canonical_sha256")
                != v66._row_reference_sha256(row)
                or record.get("scoring_contract_sha256")
                != v66._row_scoring_contract_sha256(row)
            ):
                raise ValueError("V67 cached behavior differs from held row")
    return payload


def _screen_report(
    *,
    preflight: V63Preflight,
    training_identity: Mapping[str, Any],
    teacher_audit: Mapping[str, Any],
    fold_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = aggregate_numeric_screens_v67(
        [fold["numeric_screen"] for fold in fold_payloads]
    )
    checks = assess_numeric_screen_v67(metrics)
    return {
        "schema_version": 1,
        "artifact": _SCREEN_ARTIFACT,
        "passed": all(checks.values()),
        "promotion_eligible": False,
        "checkpoint_published": False,
        "gemma_generation_used": False,
        "terminal_reason": (
            "numeric_screen_passed_full_behavioral_run_authorized"
            if all(checks.values())
            else "numeric_screen_failed_no_generation_or_checkpoint_authorized"
        ),
        "preregistration_sha256": _PINNED_PREREGISTRATION_SHA256,
        "training_identity_sha256": training_identity["sha256"],
        "authorization": {
            "baseline_lock_sha256": preflight.baseline_lock_sha256,
            "filtered_training_qa_sha256": preflight.filtered_train_sha256,
            "teacher_audit_sha256": _canonical_sha256(teacher_audit),
        },
        "thresholds": asdict(V67_NUMERIC_SCREEN_THRESHOLDS),
        "metrics": metrics,
        "checks": checks,
        "folds": [
            {
                key: fold[key]
                for key in (
                    "held_pair_id",
                    "held_rows_used_for_optimization",
                    "held_teacher_sources_used",
                    "fold_codebook_sha256",
                    "fold_basis_sha256",
                    "fit",
                    "numeric_screen",
                )
            }
            for fold in fold_payloads
        ],
        "scope": {
            "training_only": True,
            "numeric_teacher_and_prefix_cache_only": True,
            "question_or_answer_text_stored": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "internal_validation_loaded": False,
            "deferred_final_loaded": False,
        },
    }


def train_v67(
    args: argparse.Namespace,
    *,
    runtime_provider: Callable[..., V65RuntimeBundle] | None = None,
    generator_fn: Callable[..., str] | None = None,
    supplemental_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    fit_args = _locked_fit_args(args)
    validate_v67_preregistration(args.preregistration)
    preflight = build_v63_preflight(fit_args)
    if len(preflight.rows) != _EXPECTED_ROWS:
        raise ValueError("V67 authenticated training inventory changed")
    validate_training_baseline_lock(
        args.training_baseline_lock, expected_rows=preflight.rows
    )
    teachers, teacher_audit = v66.load_combined_verified_teachers_v66(
        preflight,
        args.supplemental_teacher_cache,
        supplemental_loader=supplemental_loader,
    )
    training_identity = _training_identity(
        preflight, teacher_audit, fit_args
    )
    if args.mode == "full" and not args.screen_authorization:
        raise ValueError("V67 full mode requires a passed screen authorization")
    screen_authorization = (
        None
        if args.mode == "screen"
        else validate_screen_authorization_v67(
            args.screen_authorization,
            expected_training_identity_sha256=str(training_identity["sha256"]),
        )
    )
    work = _resolve(args.work_directory)
    if work in {preflight.output_checkpoint, preflight.training_report}:
        raise ValueError("V67 work directory overlaps a publication destination")
    run_manifest = {
        "schema_version": 1,
        "artifact": _WORK_ARTIFACT,
        "mode": args.mode,
        "training_identity_sha256": training_identity["sha256"],
        "screen_authorization_sha256": (
            None if screen_authorization is None else screen_authorization["sha256"]
        ),
    }
    run_manifest["run_signature_sha256"] = _canonical_sha256(run_manifest)
    v66._prepare_work_directory(work, run_manifest)
    bundle = (runtime_provider or _load_runtime)(
        preflight, requested_device=args.device
    )
    generation = generator_fn or v66._generate_with_control
    fold_payloads: list[dict[str, Any]] = []
    for fold_index, held_pair in enumerate(TRAIN_PAIR_IDS):
        train_rows = tuple(row for row in preflight.rows if row.pair_id != held_pair)
        held_rows = tuple(row for row in preflight.rows if row.pair_id == held_pair)
        train_keys = {row.key for row in train_rows}
        fold_teachers = {key: value for key, value in teachers.items() if key in train_keys}
        codebook = build_hybrid_answer_prototype_codebook_v66(
            train_rows,
            fold_teachers,
            expected_class_count=None,
            scope=f"v67_fold_{held_pair}",
            forbidden_pair_id=held_pair,
        )
        basis = v66._codebook_basis(codebook, int(fit_args.basis_rank))
        fold_path = work / f"fold_{held_pair}.json"
        if fold_path.exists():
            cached = _validate_cached_fold_v67(
                json.loads(fold_path.read_text(encoding="utf-8")),
                mode=args.mode,
                held_pair_id=held_pair,
                run_signature_sha256=str(run_manifest["run_signature_sha256"]),
                held_rows=held_rows,
                codebook=codebook,
                basis=basis,
            )
            fold_payloads.append(cached)
            continue
        fit = _fit_pair_aware(
            rows=train_rows,
            codebook=codebook,
            preflight=preflight,
            questions=bundle.question_embeddings,
            basis=basis,
            args=fit_args,
            seed=int(fit_args.seed) + (fold_index + 1) * 100_003,
            phase=f"v67_cv_{held_pair}",
        )
        held_signatures = _scene_signatures(
            fit.control,
            {
                scene_id: preflight.prefixes[scene_id]
                for scene_id in sorted({row.scene_id for row in held_rows})
            },
        )
        held_base = v66.V66FitResult(
            control=fit.control,
            signatures=held_signatures,
            base_fit=fit.base.base_fit,
            classification_optimizer_steps=fit.base.classification_optimizer_steps,
            numeric_prototype_top1_accuracy=fit.base.numeric_prototype_top1_accuracy,
            numeric_prototype_mean_margin=fit.base.numeric_prototype_mean_margin,
        )
        held_fit = V67FitResult(
            base=held_base,
            refinement_optimizer_steps=fit.refinement_optimizer_steps,
            refinement_elapsed_seconds=fit.refinement_elapsed_seconds,
            train_pair_diagnostics=fit.train_pair_diagnostics,
            train_prototype_top1_accuracy=fit.train_prototype_top1_accuracy,
        )
        numeric_metrics, numeric_evidence = numeric_screen_fold_v67(
            held_fit,
            held_rows,
            codebook=codebook,
            questions=bundle.question_embeddings,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact": _FOLD_ARTIFACT,
            "run_signature_sha256": run_manifest["run_signature_sha256"],
            "held_pair_id": held_pair,
            "held_rows_used_for_optimization": False,
            "held_teacher_sources_used": False,
            "fold_codebook_sha256": codebook.sha256,
            "fold_basis_sha256": _tensor_sha256(basis),
            "fit": _fit_audit(held_fit),
            "numeric_screen": numeric_metrics,
            "numeric_evidence": list(numeric_evidence),
        }
        if args.mode == "full":
            records = v66.generate_supported_rows_v66(
                held_fit.base,
                held_rows,
                questions=bundle.question_embeddings,
                supported_classes=set(codebook.prototypes),
                bundle=bundle,
                prefixes=preflight.prefixes,
                generator_fn=generation,
            )
            payload["behavior"] = v66.behavior_metrics_v66(
                records, unsupported_count=len(held_rows) - len(records)
            )
            payload["records"] = list(records)
        v66._atomic_new_json(fold_path, payload)
        fold_payloads.append(payload)

    screen = _screen_report(
        preflight=preflight,
        training_identity=training_identity,
        teacher_audit=teacher_audit,
        fold_payloads=fold_payloads,
    )
    if args.mode == "screen":
        _write_training_report(preflight.training_report, screen)
        return screen
    if not screen["passed"]:
        raise RuntimeError(
            "V67 full run reproduced a failed numeric screen after authorization"
        )

    all_records = [record for fold in fold_payloads for record in fold["records"]]
    unsupported = sum(
        int(fold["behavior"]["unsupported_total"]) for fold in fold_payloads
    )
    cv_metrics = v66.behavior_metrics_v66(
        all_records, unsupported_count=unsupported
    )
    cv_checks = v66.assess_cv_v66(cv_metrics)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": _FULL_ARTIFACT,
        "promotion_eligible": False,
        "checkpoint": None,
        "terminal_reason": "pair_disjoint_behavior_gate_failed",
        "preregistration_sha256": _PINNED_PREREGISTRATION_SHA256,
        "training_identity_sha256": training_identity["sha256"],
        "screen_authorization": validate_screen_authorization_v67(
            args.screen_authorization,
            expected_training_identity_sha256=str(training_identity["sha256"]),
        ),
        "architecture": {
            "name": "always_on_teacher_basis_full_scene_control_v7",
            "complete_scene_prefix": True,
            "scene_latents": 256,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "runtime_answer_codebook": False,
        },
        "fixed_hyperparameters": dict(V67_HYPERPARAMETERS),
        "teacher_audit": teacher_audit,
        "numeric_screen_reproduced": {
            key: screen[key] for key in ("metrics", "checks", "passed")
        },
        "cv": {
            "protocol": "leave_one_counterfactual_pair_out_pair_aware_allrow",
            "metrics": cv_metrics,
            "checks": cv_checks,
            "passed": all(cv_checks.values()),
            "folds": [
                {
                    key: fold[key]
                    for key in (
                        "held_pair_id",
                        "held_rows_used_for_optimization",
                        "held_teacher_sources_used",
                        "fold_codebook_sha256",
                        "fold_basis_sha256",
                        "fit",
                        "numeric_screen",
                        "behavior",
                    )
                }
                for fold in fold_payloads
            ],
        },
        "scope": {
            "training_only": True,
            "gemma_frozen": True,
            "gemma_backward_used": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "internal_validation_loaded": False,
            "deferred_final_loaded": False,
            "question_or_answer_text_stored": False,
        },
    }
    if not report["cv"]["passed"]:
        _write_training_report(preflight.training_report, report)
        return report

    all_codebook = build_hybrid_answer_prototype_codebook_v66(
        preflight.rows,
        teachers,
        expected_class_count=_EXPECTED_CLASSES,
        scope="v67_final_all_training",
    )
    final_basis = v66._codebook_basis(all_codebook, int(fit_args.basis_rank))
    final_fit = _fit_pair_aware(
        rows=preflight.rows,
        codebook=all_codebook,
        preflight=preflight,
        questions=bundle.question_embeddings,
        basis=final_basis,
        args=fit_args,
        seed=int(fit_args.seed) + 9_999_991,
        phase="v67_final_all_training",
    )
    final_records = v66.generate_supported_rows_v66(
        final_fit.base,
        preflight.rows,
        questions=bundle.question_embeddings,
        supported_classes=set(all_codebook.prototypes),
        bundle=bundle,
        prefixes=preflight.prefixes,
        generator_fn=generation,
    )
    final_metrics = v66.behavior_metrics_v66(final_records, unsupported_count=0)
    final_checks = v66.assess_final_v66(final_metrics)
    report["final_fit"] = {
        "metrics": final_metrics,
        "checks": final_checks,
        "passed": all(final_checks.values()),
        "codebook_sha256": all_codebook.sha256,
        "basis_sha256": _tensor_sha256(final_basis),
        "fit_state_sha256": _canonical_sha256(
            {
                name: _tensor_sha256(value)
                for name, value in final_fit.control.state_dict().items()
            }
        ),
        "fit": _fit_audit(final_fit),
    }
    if not report["final_fit"]["passed"]:
        report["terminal_reason"] = "all_training_behavior_gate_failed"
        _write_training_report(preflight.training_report, report)
        return report

    paired_records = v66.generate_paired_opposite_scene_rows_v66(
        final_fit.base,
        preflight.rows,
        questions=bundle.question_embeddings,
        bundle=bundle,
        prefixes=preflight.prefixes,
        generator_fn=generation,
    )
    dependence_metrics = v66.paired_opposite_metrics_v66(paired_records)
    dependence_checks = v66.paired_opposite_checks_v66(dependence_metrics)
    report["paired_opposite_scene_dependence"] = {
        "protocol": (
            "exact_counterfactual_paired_opposite_scene_prefix_and_signature_"
            "same_byte_identical_question"
        ),
        "metrics": dependence_metrics,
        "checks": dependence_checks,
        "passed": all(dependence_checks.values()),
    }
    if not report["paired_opposite_scene_dependence"]["passed"]:
        report["terminal_reason"] = "paired_opposite_scene_dependence_gate_failed"
        _write_training_report(preflight.training_report, report)
        return report

    report["terminal_reason"] = (
        "all_v67_training_gates_passed_checkpoint_saved"
    )
    return v66._publish_v66_checkpoint_and_report(
        preflight=preflight,
        control=final_fit.control,
        report_without_checkpoint={
            key: value for key, value in report.items() if key != "checkpoint"
        },
        bundle=bundle,
        rows=preflight.rows,
        prefixes=preflight.prefixes,
        generator_fn=generation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "full"), required=True)
    parser.add_argument("--baseline-lock", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--training-baseline-lock", required=True)
    parser.add_argument("--filtered-train-qa", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--supplemental-teacher-cache", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--source-v60-checkpoint", required=True)
    parser.add_argument("--work-directory", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--screen-authorization")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "full" and not args.screen_authorization:
        raise ValueError("V67 full mode requires --screen-authorization")
    if args.mode == "screen" and args.screen_authorization:
        raise ValueError("V67 screen mode cannot consume a screen authorization")
    random.seed(int(V67_HYPERPARAMETERS["seed"]))
    torch.manual_seed(int(V67_HYPERPARAMETERS["seed"]))
    report = train_v67(args)
    print(json.dumps(report, sort_keys=True))
    if args.mode == "screen":
        return 0 if report.get("passed") is True else 1
    return (
        0
        if report.get("terminal_reason")
        == "all_v67_training_gates_passed_checkpoint_saved"
        and isinstance(report.get("checkpoint"), Mapping)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V67FitResult",
    "aggregate_numeric_screens_v67",
    "assess_numeric_screen_v67",
    "numeric_screen_fold_v67",
    "train_v67",
    "validate_screen_authorization_v67",
    "validate_v67_preregistration",
]
