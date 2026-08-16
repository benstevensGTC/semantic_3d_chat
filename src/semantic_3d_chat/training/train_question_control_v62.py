"""Train V5's route head with train-only pair-disjoint cross-validation.

The V62 trainer is intentionally incapable of reading internal-validation
questions, scorer references, or prediction answers.  Its only data argument
is the create-once filtered 12-pair training JSONL.  Before that file is
opened, a hash-only V54 baseline lock must authenticate the already frozen
evaluation boundary.

Gemma is loaded locally only to cache frozen pooled question embeddings.  It
is never generated from, differentiated through, or updated.  Every inherited
V60 tensor remains byte-identical; only ``factorized_route`` is optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    TRAIN_PAIR_IDS,
    V62_PROHIBITED_TRAINER_DATA_ARGUMENTS,
    add_baseline_lock_authorization_argument,
    add_filtered_training_data_argument,
    load_filtered_training_qa,
    validate_baseline_lock,
)
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v5 import (
    FactorizedRouteFeaturesV5,
    NormalizedFactorizedSceneQuestionControlV5,
)
from semantic_3d_chat.training.question_control_v5_checkpoint import (
    inherited_v60_state_sha256,
    save_v5_control_checkpoint,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    _log_event,
    _resolve,
    _safe_output_path,
    _select_training_device,
    _sha256_file,
    _write_training_report,
    freeze_base_runtime,
    load_prefix_cache,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _pooled_question_embedding,
)

_PINNED_FILTERED_TRAIN_SHA256: Final[str] = (
    "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1"
)


# This train-only model-selection rule is source controlled and included by
# digest in every report.  It does not consult the internal-validation split.
V62_ROUTE_CV_THRESHOLDS: Final[dict[str, int | float]] = {
    "minimum_held_natural_accuracy": 0.90,
    "minimum_held_changed_recall": 0.70,
    "minimum_held_retention_specificity": 0.95,
    "minimum_held_changed_unit_completeness": 0.60,
    "minimum_held_cartesian_accuracy": 0.90,
    "minimum_held_cartesian_negative_specificity": 0.95,
    "minimum_fold_natural_accuracy": 0.75,
    "minimum_fold_cartesian_accuracy": 0.75,
    "minimum_folds_with_changed_hit": 10,
    "required_fold_count": 12,
    "all_fold_training_natural_exact": 1,
    "all_fold_training_dense_exact": 1,
}


@dataclass(frozen=True)
class RouteExampleV62:
    """One route-only example; ``question`` is never serialized to reports."""

    scene_id: str
    question: str
    label: bool
    pair_id: str
    question_id: str | None
    unit_id: str | None
    population: str


@dataclass(frozen=True)
class PairDisjointFoldV62:
    pair_id: str
    training_pair_ids: tuple[str, ...]
    held_pair_ids: tuple[str, ...]
    training_natural: tuple[RouteExampleV62, ...]
    training_dense: tuple[RouteExampleV62, ...]
    held_natural: tuple[RouteExampleV62, ...]
    held_cartesian: tuple[RouteExampleV62, ...]


@dataclass(frozen=True)
class RouteFitConfigV62:
    epochs: int = 2400
    minimum_epochs: int = 200
    success_patience: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 2.0
    minimum_signed_logit_margin: float = 3.0
    margin_weight: float = 0.1
    natural_loss_weight: float = 1.0
    dense_loss_weight: float = 1.0
    log_every: int = 200

    def validate(self) -> None:
        for field in ("epochs", "minimum_epochs", "success_patience", "log_every"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"V62 {field} must be a positive integer")
        if self.minimum_epochs > self.epochs:
            raise ValueError("V62 minimum_epochs cannot exceed epochs")
        for field in (
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "minimum_signed_logit_margin",
            "margin_weight",
            "natural_loss_weight",
            "dense_loss_weight",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V62 {field} must be finite and nonnegative")
        if self.learning_rate == 0.0 or self.gradient_clip_norm == 0.0:
            raise ValueError("V62 learning rate and gradient clip norm must be positive")
        if self.natural_loss_weight + self.dense_loss_weight == 0.0:
            raise ValueError("V62 route objective has no enabled population")


@dataclass(frozen=True)
class RouteFitResultV62:
    control: NormalizedFactorizedSceneQuestionControlV5
    completed_epochs: int
    best_epoch: int
    best_minimum_signed_logit_margin: float
    best_objective_loss: float
    maximum_preclip_gradient_norm: float
    natural_metrics: dict[str, int | float | bool]
    dense_metrics: dict[str, int | float | bool]
    elapsed_seconds: float
    route_device: str


@dataclass(frozen=True)
class BaselineAuthorizationV62:
    payload: dict[str, Any]
    sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json_bytes(dict(row)))
    return digest.hexdigest()


def load_v62_baseline_authorization(path: str | Path) -> BaselineAuthorizationV62:
    """Authenticate the answer-free lock without opening its source artifacts."""

    source = _resolve(path)
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V62 baseline authorization path contains a symlink: {current}")
    if not source.is_file() or source.suffix.casefold() != ".json":
        raise FileNotFoundError(f"V62 baseline authorization is unavailable: {source}")
    raw = source.read_bytes()
    # The boundary-owned validator pins the preregistration, question
    # manifest, question inventory, V54 checkpoint, and all 384 output hashes.
    # It has no prediction-answer or preregistration-path parameter.
    payload = validate_baseline_lock(source)
    return BaselineAuthorizationV62(payload=dict(payload), sha256=hashlib.sha256(raw).hexdigest())


def _train_specs() -> tuple[Any, ...]:
    by_id = {spec.pair_id: spec for spec in PAIR_INVENTORY}
    return tuple(by_id[pair_id] for pair_id in TRAIN_PAIR_IDS)


def authorized_training_scene_ids() -> tuple[str, ...]:
    return tuple(scene_id for spec in _train_specs() for scene_id in spec.scene_ids)


def natural_route_examples(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[RouteExampleV62, ...]:
    examples = tuple(
        RouteExampleV62(
            scene_id=str(row["scene_id"]),
            question=str(row["question"]),
            label=bool(row["counterfactual_expected_change"]),
            pair_id=str(row["counterfactual_pair_id"]),
            question_id=str(row["question_id"]),
            unit_id=str(row["counterfactual_question_key"]),
            population="natural",
        )
        for row in rows
    )
    if not examples:
        raise ValueError("V62 natural route population is empty")
    return examples


def dense_cartesian_route_examples(
    question_rows: Sequence[Mapping[str, Any]],
    *,
    scene_ids: Sequence[str],
    label_rows: Sequence[Mapping[str, Any]] | None = None,
    population: str = "dense_cartesian",
) -> tuple[RouteExampleV62, ...]:
    """Cross every changed question text with every requested scene.

    A cell is positive only when that exact ``(scene_id, question text)`` is a
    changed natural row in ``label_rows``.  No semantic matching or retrieval
    is performed.
    """

    scenes = tuple(sorted(scene_ids))
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("V62 dense Cartesian scenes must be nonempty and unique")
    changed_questions = tuple(
        sorted(
            {
                str(row["question"])
                for row in question_rows
                if row["counterfactual_expected_change"] is True
            }
        )
    )
    if not changed_questions:
        raise ValueError("V62 dense Cartesian population has no changed questions")
    labels_from = question_rows if label_rows is None else label_rows
    positives = {
        (str(row["scene_id"]), str(row["question"]))
        for row in labels_from
        if row["counterfactual_expected_change"] is True
    }
    return tuple(
        RouteExampleV62(
            scene_id=scene_id,
            question=question,
            label=(scene_id, question) in positives,
            pair_id="cartesian",
            question_id=None,
            unit_id=None,
            population=population,
        )
        for question in changed_questions
        for scene_id in scenes
    )


def route_population_sha256(examples: Sequence[RouteExampleV62]) -> str:
    rows = [
        {
            "scene_id": example.scene_id,
            "question_sha256": hashlib.sha256(example.question.encode("utf-8")).hexdigest(),
            "label": example.label,
            "pair_id": example.pair_id,
            "question_id": example.question_id,
            "unit_id": example.unit_id,
            "population": example.population,
        }
        for example in examples
    ]
    return _canonical_jsonl_sha256(rows)


def leave_one_pair_out_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    all_scene_ids: Sequence[str],
) -> tuple[PairDisjointFoldV62, ...]:
    expected = tuple(pair_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("V62 CV pair IDs must be nonempty and unique")
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["counterfactual_pair_id"])].append(row)
    if set(grouped) != set(expected):
        raise ValueError("V62 CV rows differ from the exact pair inventory")
    folds: list[PairDisjointFoldV62] = []
    for held_pair in expected:
        train_rows = tuple(
            row for pair_id in expected if pair_id != held_pair for row in grouped[pair_id]
        )
        held_rows = tuple(grouped[held_pair])
        train_scene_ids = tuple(sorted({str(row["scene_id"]) for row in train_rows}))
        training_ids = tuple(pair_id for pair_id in expected if pair_id != held_pair)
        fold = PairDisjointFoldV62(
            pair_id=held_pair,
            training_pair_ids=training_ids,
            held_pair_ids=(held_pair,),
            training_natural=natural_route_examples(train_rows),
            training_dense=dense_cartesian_route_examples(
                train_rows,
                scene_ids=train_scene_ids,
                label_rows=train_rows,
                population="fold_training_dense",
            ),
            held_natural=natural_route_examples(held_rows),
            held_cartesian=dense_cartesian_route_examples(
                held_rows,
                scene_ids=all_scene_ids,
                label_rows=rows,
                population="fold_held_cartesian",
            ),
        )
        if set(fold.training_pair_ids) & set(fold.held_pair_ids):
            raise AssertionError("V62 CV pair leakage detected")
        folds.append(fold)
    return tuple(folds)


def _route_threshold_logit(control: NormalizedFactorizedSceneQuestionControlV5) -> float:
    probability = float(control.gate_threshold)
    return math.log(probability / (1.0 - probability))


def _population_logits(
    control: NormalizedFactorizedSceneQuestionControlV5,
    examples: Sequence[RouteExampleV62],
    question_inputs: Mapping[str, torch.Tensor],
    scene_inputs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    questions = tuple(sorted({example.question for example in examples}))
    scenes = tuple(sorted({example.scene_id for example in examples}))
    missing_questions = set(questions) - set(question_inputs)
    missing_scenes = set(scenes) - set(scene_inputs)
    if missing_questions or missing_scenes:
        raise ValueError(
            "V62 route cache is incomplete: "
            f"questions={len(missing_questions)} scenes={sorted(missing_scenes)}"
        )
    question_index = {value: index for index, value in enumerate(questions)}
    scene_index = {value: index for index, value in enumerate(scenes)}
    route_devices = {parameter.device for parameter in control.factorized_route.parameters()}
    if len(route_devices) != 1:
        raise RuntimeError("V62 factorized route parameters span multiple devices")
    route_device = next(iter(route_devices))
    normalized_questions = torch.stack(
        [
            question_inputs[value]
            .detach()
            .to(device=route_device, dtype=torch.float32)
            for value in questions
        ]
    )
    signatures = torch.cat(
        [
            scene_inputs[value].detach().to(device=route_device, dtype=torch.float32)
            for value in scenes
        ],
        dim=0,
    )
    question_factors = control.factorized_route.encode_question(normalized_questions)
    scene_factors = control.factorized_route.encode_scene(signatures)
    q_indices = torch.tensor(
        [question_index[item.question] for item in examples],
        device=route_device,
    )
    s_indices = torch.tensor(
        [scene_index[item.scene_id] for item in examples],
        device=route_device,
    )
    return control.route_logits_from_features(
        FactorizedRouteFeaturesV5(
            question=question_factors[q_indices],
            scene=scene_factors[s_indices],
        )
    )


def route_metrics(
    control: NormalizedFactorizedSceneQuestionControlV5,
    examples: Sequence[RouteExampleV62],
    question_inputs: Mapping[str, torch.Tensor],
    scene_inputs: Mapping[str, torch.Tensor],
) -> tuple[dict[str, int | float | bool], tuple[bool, ...]]:
    with torch.inference_mode():
        logits = _population_logits(control, examples, question_inputs, scene_inputs)
    labels = torch.tensor([example.label for example in examples], dtype=torch.bool)
    centered = logits.detach().float().cpu() - _route_threshold_logit(control)
    predicted = centered.ge(0.0)
    correct = predicted.eq(labels)
    positive = labels
    negative = ~labels
    signed = torch.where(labels, centered, -centered)
    result: dict[str, int | float | bool] = {
        "correct": int(correct.sum()),
        "total": len(examples),
        "accuracy": float(correct.float().mean()),
        "positive_correct": int((correct & positive).sum()),
        "positive_total": int(positive.sum()),
        "negative_correct": int((correct & negative).sum()),
        "negative_total": int(negative.sum()),
        "minimum_signed_logit_margin": float(signed.min()),
        "exact": bool(correct.all()),
    }
    return result, tuple(bool(value) for value in predicted.tolist())


def _balanced_route_loss(centered: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = labels == 1
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("V62 route populations require both route classes")
    return 0.5 * (F.softplus(-centered[positive]).mean() + F.softplus(centered[negative]).mean())


def _route_objective(
    natural_centered: torch.Tensor,
    dense_centered: torch.Tensor,
    natural_labels: torch.Tensor,
    dense_labels: torch.Tensor,
    config: RouteFitConfigV62,
) -> torch.Tensor:
    natural_signs = natural_labels.mul(2.0).sub(1.0)
    dense_signs = dense_labels.mul(2.0).sub(1.0)
    natural_loss = _balanced_route_loss(natural_centered, natural_labels.bool())
    dense_loss = _balanced_route_loss(dense_centered, dense_labels.bool())
    natural_margin = F.relu(
        config.minimum_signed_logit_margin - natural_signs * natural_centered
    ).square().mean()
    dense_margin = F.relu(
        config.minimum_signed_logit_margin - dense_signs * dense_centered
    ).square().mean()
    return config.natural_loss_weight * (
        natural_loss + config.margin_weight * natural_margin
    ) + config.dense_loss_weight * (dense_loss + config.margin_weight * dense_margin)


def _factorized_route_cpu_state(
    control: NormalizedFactorizedSceneQuestionControlV5,
) -> dict[str, torch.Tensor]:
    """Snapshot only the trainable route head, never inherited V60 values."""

    return {
        name: value.detach().float().cpu().clone()
        for name, value in control.factorized_route.state_dict().items()
    }


def fit_route_only(
    source_v60: TeacherBasisFullSceneQuestionControlV3,
    *,
    natural_examples: Sequence[RouteExampleV62],
    dense_examples: Sequence[RouteExampleV62],
    question_inputs: Mapping[str, torch.Tensor],
    scene_inputs: Mapping[str, torch.Tensor],
    route_factor_rank: int,
    seed: int,
    config: RouteFitConfigV62,
    log_phase: str | None = None,
    device: torch.device | str | None = None,
) -> RouteFitResultV62:
    """Fit a fresh gate; no inherited value tensor can receive a gradient."""

    config.validate()
    torch.manual_seed(seed)
    route_device = torch.device("cpu") if device is None else torch.device(device)
    if route_device.type not in {"cpu", "mps"}:
        raise ValueError("V62 route device must be CPU or MPS")
    if route_device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("V62 MPS route device was requested but is unavailable")
        torch.mps.manual_seed(seed)
    control = NormalizedFactorizedSceneQuestionControlV5.from_v60(
        source_v60, route_factor_rank=route_factor_rank
    ).to(device=route_device, dtype=torch.float32)
    trainable = {
        name for name, parameter in control.named_parameters() if parameter.requires_grad
    }
    if not trainable or any(not name.startswith("factorized_route.") for name in trainable):
        raise RuntimeError("V62 optimizer scope is not exactly factorized_route")
    inherited_before = inherited_v60_state_sha256(control)
    optimizer = torch.optim.AdamW(
        control.factorized_route.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    natural_labels = torch.tensor(
        [float(example.label) for example in natural_examples],
        dtype=torch.float32,
        device=route_device,
    )
    dense_labels = torch.tensor(
        [float(example.label) for example in dense_examples],
        dtype=torch.float32,
        device=route_device,
    )
    threshold = _route_threshold_logit(control)
    maximum_gradient = 0.0
    stable_success = 0
    completed_epochs = 0
    best_epoch = 0
    best_score: tuple[int, float, float] | None = None
    best_minimum_margin = -math.inf
    best_objective_loss = math.inf
    best_route_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        natural_logits = _population_logits(
            control, natural_examples, question_inputs, scene_inputs
        )
        dense_logits = _population_logits(control, dense_examples, question_inputs, scene_inputs)
        natural_centered = natural_logits - threshold
        dense_centered = dense_logits - threshold
        loss = _route_objective(
            natural_centered,
            dense_centered,
            natural_labels,
            dense_labels,
            config,
        )
        loss.backward()
        if any(
            parameter.grad is not None
            for name, parameter in control.named_parameters()
            if not name.startswith("factorized_route.")
        ):
            raise RuntimeError("V62 backward reached an inherited V60 parameter")
        gradient = torch.nn.utils.clip_grad_norm_(
            control.factorized_route.parameters(), config.gradient_clip_norm
        )
        gradient_value = float(gradient.detach().float().cpu())
        if not math.isfinite(gradient_value):
            raise RuntimeError("V62 route gradient is nonfinite")
        maximum_gradient = max(maximum_gradient, gradient_value)
        optimizer.step()
        completed_epochs = epoch + 1
        with torch.inference_mode():
            natural_after = _population_logits(
                control, natural_examples, question_inputs, scene_inputs
            ) - threshold
            dense_after = _population_logits(
                control, dense_examples, question_inputs, scene_inputs
            ) - threshold
            natural_signs = natural_labels.mul(2.0).sub(1.0)
            dense_signs = dense_labels.mul(2.0).sub(1.0)
            natural_signed = natural_signs * natural_after
            dense_signed = dense_signs * dense_after
            natural_correct = natural_after.ge(0.0).eq(natural_labels.bool())
            dense_correct = dense_after.ge(0.0).eq(dense_labels.bool())
            correct_count = int(
                (natural_correct.sum() + dense_correct.sum()).detach().cpu()
            )
            exact = correct_count == len(natural_examples) + len(dense_examples)
            minimum_margin = float(
                torch.cat((natural_signed, dense_signed)).min().detach().float().cpu()
            )
            scored_loss = float(
                _route_objective(
                    natural_after,
                    dense_after,
                    natural_labels,
                    dense_labels,
                    config,
                )
                .detach()
                .float()
                .cpu()
            )
        # Route correctness dominates margin, which dominates objective loss.
        # Strict comparison deliberately retains the earliest state on ties.
        score = (correct_count, minimum_margin, -scored_loss)
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = completed_epochs
            best_minimum_margin = minimum_margin
            best_objective_loss = scored_loss
            best_route_state = _factorized_route_cpu_state(control)
        stable_success = (
            stable_success + 1
            if exact
            and minimum_margin >= config.minimum_signed_logit_margin
            and completed_epochs >= config.minimum_epochs
            else 0
        )
        if log_phase is not None and (
            completed_epochs % config.log_every == 0 or stable_success == 1
        ):
            _log_event(
                phase=log_phase,
                epoch=completed_epochs,
                loss=float(loss.detach().float().cpu()),
                routes_exact=exact,
                minimum_signed_logit_margin=minimum_margin,
            )
        if stable_success >= config.success_patience:
            break
    if best_route_state is None or best_epoch < 1 or best_score is None:
        raise RuntimeError("V62 route fit did not produce a selectable epoch")
    control.factorized_route.load_state_dict(best_route_state, strict=True)
    control.eval()
    natural_metrics, _ = route_metrics(
        control, natural_examples, question_inputs, scene_inputs
    )
    dense_metrics, _ = route_metrics(control, dense_examples, question_inputs, scene_inputs)
    if inherited_v60_state_sha256(control) != inherited_before:
        raise RuntimeError("V62 route fit changed inherited V60 bytes")
    return RouteFitResultV62(
        control=control,
        completed_epochs=completed_epochs,
        best_epoch=best_epoch,
        best_minimum_signed_logit_margin=best_minimum_margin,
        best_objective_loss=best_objective_loss,
        maximum_preclip_gradient_norm=maximum_gradient,
        natural_metrics=natural_metrics,
        dense_metrics=dense_metrics,
        elapsed_seconds=time.perf_counter() - started,
        route_device=str(route_device),
    )


def _unit_completeness(
    examples: Sequence[RouteExampleV62], predictions: Sequence[bool]
) -> tuple[int, int]:
    units: defaultdict[tuple[str, str], list[bool]] = defaultdict(list)
    for example, predicted in zip(examples, predictions, strict=True):
        if example.label:
            if example.unit_id is None:
                raise ValueError("V62 changed natural example lacks paired-unit identity")
            units[(example.pair_id, example.unit_id)].append(predicted)
    if any(len(values) != 2 for values in units.values()):
        raise ValueError("V62 changed paired units must contain exactly two sides")
    return sum(all(values) for values in units.values()), len(units)


def assess_route_cv(
    fold_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(fold_reports) != int(V62_ROUTE_CV_THRESHOLDS["required_fold_count"]):
        raise ValueError("V62 route CV requires exactly 12 leave-one-pair-out folds")
    natural_correct = sum(int(fold["held_natural"]["correct"]) for fold in fold_reports)
    natural_total = sum(int(fold["held_natural"]["total"]) for fold in fold_reports)
    changed_correct = sum(
        int(fold["held_natural"]["positive_correct"]) for fold in fold_reports
    )
    changed_total = sum(
        int(fold["held_natural"]["positive_total"]) for fold in fold_reports
    )
    retention_correct = sum(
        int(fold["held_natural"]["negative_correct"]) for fold in fold_reports
    )
    retention_total = sum(
        int(fold["held_natural"]["negative_total"]) for fold in fold_reports
    )
    unit_correct = sum(int(fold["changed_units_complete"]) for fold in fold_reports)
    unit_total = sum(int(fold["changed_units_total"]) for fold in fold_reports)
    cart_correct = sum(int(fold["held_cartesian"]["correct"]) for fold in fold_reports)
    cart_total = sum(int(fold["held_cartesian"]["total"]) for fold in fold_reports)
    cart_negative_correct = sum(
        int(fold["held_cartesian"]["negative_correct"]) for fold in fold_reports
    )
    cart_negative_total = sum(
        int(fold["held_cartesian"]["negative_total"]) for fold in fold_reports
    )

    def ratio(numerator: int, denominator: int) -> float:
        if denominator < 1:
            raise ValueError("V62 CV metric has an empty denominator")
        return numerator / denominator

    metrics = {
        "held_natural": {
            "correct": natural_correct,
            "total": natural_total,
            "accuracy": ratio(natural_correct, natural_total),
        },
        "held_changed": {
            "correct": changed_correct,
            "total": changed_total,
            "recall": ratio(changed_correct, changed_total),
        },
        "held_retention": {
            "correct": retention_correct,
            "total": retention_total,
            "specificity": ratio(retention_correct, retention_total),
        },
        "held_changed_units": {
            "complete": unit_correct,
            "total": unit_total,
            "completeness": ratio(unit_correct, unit_total),
        },
        "held_cartesian": {
            "correct": cart_correct,
            "total": cart_total,
            "accuracy": ratio(cart_correct, cart_total),
            "negative_correct": cart_negative_correct,
            "negative_total": cart_negative_total,
            "negative_specificity": ratio(cart_negative_correct, cart_negative_total),
        },
        "minimum_fold_natural_accuracy": min(
            float(fold["held_natural"]["accuracy"]) for fold in fold_reports
        ),
        "minimum_fold_cartesian_accuracy": min(
            float(fold["held_cartesian"]["accuracy"]) for fold in fold_reports
        ),
        "folds_with_changed_hit": sum(
            int(fold["held_natural"]["positive_correct"] > 0) for fold in fold_reports
        ),
    }
    threshold = V62_ROUTE_CV_THRESHOLDS
    checks = {
        "all_fold_training_natural_exact": all(
            fold["training_natural"]["exact"] is True for fold in fold_reports
        ),
        "all_fold_training_dense_exact": all(
            fold["training_dense"]["exact"] is True for fold in fold_reports
        ),
        "held_natural_accuracy": metrics["held_natural"]["accuracy"]
        >= threshold["minimum_held_natural_accuracy"],
        "held_changed_recall": metrics["held_changed"]["recall"]
        >= threshold["minimum_held_changed_recall"],
        "held_retention_specificity": metrics["held_retention"]["specificity"]
        >= threshold["minimum_held_retention_specificity"],
        "held_changed_unit_completeness": metrics["held_changed_units"]["completeness"]
        >= threshold["minimum_held_changed_unit_completeness"],
        "held_cartesian_accuracy": metrics["held_cartesian"]["accuracy"]
        >= threshold["minimum_held_cartesian_accuracy"],
        "held_cartesian_negative_specificity": metrics["held_cartesian"]
        ["negative_specificity"]
        >= threshold["minimum_held_cartesian_negative_specificity"],
        "minimum_fold_natural_accuracy": metrics["minimum_fold_natural_accuracy"]
        >= threshold["minimum_fold_natural_accuracy"],
        "minimum_fold_cartesian_accuracy": metrics["minimum_fold_cartesian_accuracy"]
        >= threshold["minimum_fold_cartesian_accuracy"],
        "folds_with_changed_hit": metrics["folds_with_changed_hit"]
        >= threshold["minimum_folds_with_changed_hit"],
    }
    return {
        "thresholds": dict(threshold),
        "thresholds_sha256": hashlib.sha256(
            _canonical_json_bytes(threshold)
        ).hexdigest(),
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_pair_disjoint_cv(
    source_v60: TeacherBasisFullSceneQuestionControlV3,
    *,
    rows: Sequence[Mapping[str, Any]],
    question_inputs: Mapping[str, torch.Tensor],
    scene_inputs: Mapping[str, torch.Tensor],
    route_factor_rank: int,
    seed: int,
    config: RouteFitConfigV62,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    folds = leave_one_pair_out_folds(
        rows,
        pair_ids=TRAIN_PAIR_IDS,
        all_scene_ids=authorized_training_scene_ids(),
    )
    reports: list[dict[str, Any]] = []
    for ordinal, fold in enumerate(folds):
        fit = fit_route_only(
            source_v60,
            natural_examples=fold.training_natural,
            dense_examples=fold.training_dense,
            question_inputs=question_inputs,
            scene_inputs=scene_inputs,
            route_factor_rank=route_factor_rank,
            seed=seed + ordinal * 1_000_003,
            config=config,
            log_phase=f"v62_route_cv_{fold.pair_id}",
            device=device,
        )
        held_natural, natural_predictions = route_metrics(
            fit.control, fold.held_natural, question_inputs, scene_inputs
        )
        held_cartesian, _ = route_metrics(
            fit.control, fold.held_cartesian, question_inputs, scene_inputs
        )
        units_complete, units_total = _unit_completeness(
            fold.held_natural, natural_predictions
        )
        reports.append(
            {
                "fold_ordinal": ordinal,
                "held_pair_id": fold.pair_id,
                "training_pair_count": len(fold.training_pair_ids),
                "held_pair_count": len(fold.held_pair_ids),
                "pair_disjoint": not bool(
                    set(fold.training_pair_ids) & set(fold.held_pair_ids)
                ),
                "training_natural_sha256": route_population_sha256(
                    fold.training_natural
                ),
                "training_dense_sha256": route_population_sha256(fold.training_dense),
                "held_natural_sha256": route_population_sha256(fold.held_natural),
                "held_cartesian_sha256": route_population_sha256(fold.held_cartesian),
                "training_natural": fit.natural_metrics,
                "training_dense": fit.dense_metrics,
                "held_natural": held_natural,
                "held_cartesian": held_cartesian,
                "changed_units_complete": units_complete,
                "changed_units_total": units_total,
                "completed_epochs": fit.completed_epochs,
                "best_epoch": fit.best_epoch,
                "best_minimum_signed_logit_margin": (
                    fit.best_minimum_signed_logit_margin
                ),
                "best_objective_loss": fit.best_objective_loss,
                "maximum_preclip_gradient_norm": fit.maximum_preclip_gradient_norm,
                "elapsed_seconds": fit.elapsed_seconds,
                "route_device": fit.route_device,
            }
        )
    aggregate = assess_route_cv(reports)
    return {
        "method": "deterministic_leave_one_counterfactual_pair_out",
        "fold_count": len(reports),
        "pair_disjoint": all(report["pair_disjoint"] for report in reports),
        "folds": reports,
        "aggregate": aggregate,
    }


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, raw in state.items():
        value = raw.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_source_report(
    path: str | Path,
    *,
    source_weights_sha256: str,
    base_checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> tuple[dict[str, Any], str]:
    source = _resolve(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("V62 source V60 report must be a JSON object")
    checks = value.get("offline_checks")
    optimization = value.get("optimization")
    base = value.get("base")
    if (
        value.get("artifact") != "v60_teacher_basis_control_training"
        or not isinstance(checks, Mapping)
        or not isinstance(optimization, Mapping)
        or not isinstance(base, Mapping)
        or any(
            checks.get(field) is not True
            for field in (
                "basis_mean_cosine",
                "basis_minimum_cosine",
                "mean_prompt_cosine",
                "minimum_prompt_cosine",
                "mean_rms_absolute_error",
                "mean_pair_delta_cosine",
            )
        )
        or value.get("checkpoint", {}).get("weights_sha256") != source_weights_sha256
        or base.get("checkpoint_sha256") != base_checkpoint_sha256
        or base.get("runtime_config_effective_sha256") != runtime_config_sha256
    ):
        raise ValueError("V62 source V60 prompt/value evidence changed")
    return value, _sha256_file(source)


def _fit_config(args: argparse.Namespace) -> RouteFitConfigV62:
    return RouteFitConfigV62(
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        success_patience=args.success_patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        minimum_signed_logit_margin=args.minimum_signed_logit_margin,
        margin_weight=args.margin_weight,
        natural_loss_weight=args.natural_loss_weight,
        dense_loss_weight=args.dense_loss_weight,
        log_every=args.log_every,
    )


def _validate_args(args: argparse.Namespace) -> RouteFitConfigV62:
    config = _fit_config(args)
    config.validate()
    if isinstance(args.route_factor_rank, bool) or args.route_factor_rank < 1:
        raise ValueError("V62 route_factor_rank must be a positive integer")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise TypeError("V62 seed must be an integer")
    paths = {
        _resolve(args.base_checkpoint),
        _resolve(args.source_v60_checkpoint),
        _resolve(args.output_checkpoint),
        _resolve(args.training_report),
    }
    if len(paths) != 4:
        raise ValueError("V62 base/source/output/report paths must be distinct")
    return config


def _base_report(
    *,
    authorization: BaselineAuthorizationV62,
    rows: Sequence[Mapping[str, Any]],
    scene_ids: Sequence[str],
    natural: Sequence[RouteExampleV62],
    dense: Sequence[RouteExampleV62],
    cv: Mapping[str, Any],
    source: Mapping[str, Any],
    base: Mapping[str, Any],
    prefix_manifest_sha256: str,
    architecture: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "v62_normalized_factorized_route_training",
        "offline_checks_passed": False,
        "promotion_eligible": False,
        "saved_runtime_generation_gate_required": True,
        "authorization": {
            "baseline_lock_sha256": authorization.sha256,
            "preregistration_sha256": authorization.payload["preregistration_sha256"],
            "questions_manifest_sha256": authorization.payload[
                "questions_manifest_sha256"
            ],
            "required_output_hashes_sha256": authorization.payload[
                "required_output_hashes_sha256"
            ],
            "output_hash_count": authorization.payload["question_count"],
            "answer_text_read": False,
            "validated_before_training_input": True,
        },
        "source": dict(source),
        "base": dict(base),
        "inputs": {
            "filtered_training_qa_sha256": _PINNED_FILTERED_TRAIN_SHA256,
            "training_scene_ids": list(scene_ids),
            "training_pair_ids": list(TRAIN_PAIR_IDS),
            "natural_row_count": len(rows),
            "natural_changed_count": sum(example.label for example in natural),
            "natural_retention_count": sum(not example.label for example in natural),
            "natural_population_sha256": route_population_sha256(natural),
            "dense_cartesian_count": len(dense),
            "dense_changed_question_count": len({item.question for item in dense}),
            "dense_positive_count": sum(example.label for example in dense),
            "dense_negative_count": sum(not example.label for example in dense),
            "dense_population_sha256": route_population_sha256(dense),
            "prefix_cache_manifest_sha256": prefix_manifest_sha256,
        },
        "architecture": dict(architecture),
        "cross_validation": dict(cv),
        "checkpoint": None,
        "scope": dict(scope),
    }


def train_v62(args: argparse.Namespace) -> dict[str, Any]:
    """Execute V62 only after strict hash-only authorization succeeds."""

    fit_config = _validate_args(args)
    # This is deliberately the first input open.  In particular, the filtered
    # training QA loader cannot run before the baseline/preregistration lock.
    authorization = load_v62_baseline_authorization(args.baseline_lock)
    output_checkpoint = _safe_output_path(args.output_checkpoint, "V62 checkpoint")
    output_report = _safe_output_path(args.training_report, "V62 report")

    config, config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, base_files = checkpoint_fingerprint(base_checkpoint)
    if (
        base_checkpoint_sha256 != authorization.payload["v54_checkpoint_sha256"]
        or base_files != authorization.payload["v54_checkpoint_files"]
    ):
        raise ValueError("V62 base checkpoint differs from the baseline authorization")

    rows = load_filtered_training_qa(args.filtered_train_qa)
    scene_ids = tuple(sorted({str(row["scene_id"]) for row in rows}))
    expected_scene_ids = tuple(sorted(authorized_training_scene_ids()))
    if scene_ids != expected_scene_ids or len(rows) != 576:
        raise ValueError("V62 filtered training population changed after authentication")
    prefixes, _prefix_manifest = load_prefix_cache(
        args.prefix_cache,
        scene_ids=scene_ids,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )

    runtime = StaticRuntimePrefixFactory(config, base_checkpoint, scene_ids[0]).bootstrap
    route_device = _select_training_device(runtime, args.device)
    frozen = freeze_base_runtime(runtime)
    source_path = _resolve(args.source_v60_checkpoint)
    source_checkpoint_sha256 = _control_checkpoint_sha256(source_path)
    source_control, source_metadata = _load_control_head(
        source_path,
        hidden_size=runtime.language.hidden_size,
        device=torch.device("cpu"),
    )
    if type(source_control) is not TeacherBasisFullSceneQuestionControlV3:
        raise TypeError("V62 source must be the exact V3/V60 architecture")
    if (
        source_metadata.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or source_metadata.get("base_runtime_config_sha256") != runtime_config_sha256
    ):
        raise ValueError("V62 source V60 belongs to a different frozen base")
    source_weights_sha256 = _sha256_file(source_path / "control.safetensors")
    source_report, source_report_sha256 = _load_source_report(
        args.source_v60_report,
        source_weights_sha256=source_weights_sha256,
        base_checkpoint_sha256=base_checkpoint_sha256,
        runtime_config_sha256=runtime_config_sha256,
    )
    source_state = source_control.state_dict()
    source_state_sha256 = _tensor_state_sha256(source_state)

    signatures = {
        scene_id: source_control.encode_scene(prefix.float().cpu())
        for scene_id, prefix in prefixes.items()
    }
    raw_questions: dict[str, torch.Tensor] = {}
    normalized_questions: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for question in sorted({str(row["question"]) for row in rows}):
            raw = _pooled_question_embedding(runtime, question).float().cpu()
            raw_questions[question] = raw
            normalized_questions[question] = source_control.normalized_question(raw).squeeze(0)

    natural = natural_route_examples(rows)
    dense = dense_cartesian_route_examples(
        rows,
        scene_ids=scene_ids,
        label_rows=rows,
    )
    if (
        len(natural) != 576
        or sum(example.label for example in natural) != 80
        or sum(not example.label for example in natural) != 496
        or len({example.question for example in dense}) != 31
        or len(dense) != 744
        or sum(example.label for example in dense) != 80
        or sum(not example.label for example in dense) != 664
    ):
        raise RuntimeError("V62 natural or dense Cartesian route inventory changed")
    cv = run_pair_disjoint_cv(
        source_control,
        rows=rows,
        question_inputs=normalized_questions,
        scene_inputs=signatures,
        route_factor_rank=args.route_factor_rank,
        seed=args.seed,
        config=fit_config,
        device=route_device,
    )
    retained_metrics = {
        key: source_report["optimization"][key]
        for key in (
            "mean_prompt_cosine",
            "minimum_prompt_cosine",
            "mean_rms_absolute_error",
            "mean_pair_delta_cosine",
        )
    }
    source_summary = {
        "v60_checkpoint_sha256": source_checkpoint_sha256,
        "v60_weights_sha256": source_weights_sha256,
        "v60_state_sha256": source_state_sha256,
        "v60_report_sha256": source_report_sha256,
        "retained_prompt_metrics": retained_metrics,
    }
    base_summary = {
        "checkpoint_sha256": base_checkpoint_sha256,
        "checkpoint_files": base_files,
        "runtime_config_effective_sha256": runtime_config_sha256,
        "runtime_config_file_sha256": _sha256_file(config_path),
    }
    scope = {
        "base_scene_stack_frozen": frozen["all_parameters_frozen"],
        "gemma_backward_used": False,
        "gemma_generation_used": False,
        "gemma_use": "frozen_pooled_question_embeddings_only",
        "only_factorized_route_trained": True,
        "v60_values_frozen": True,
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_retained": True,
        "internal_validation_questions_loaded": False,
        "scorer_references_loaded": False,
        "prediction_answers_loaded": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    architecture = {
        "name": "normalized_factorized_scene_question_route_v5",
        "route_factor_rank": args.route_factor_rank,
        "route_device": str(route_device),
        "separate_question_scene_route_projections": True,
        "normalized_route_factors": True,
        "low_rank_bilinear_route": True,
        "all_scene_moments_consumed": True,
        "inherited_value_trunk_used_by_route": False,
    }
    report = _base_report(
        authorization=authorization,
        rows=rows,
        scene_ids=scene_ids,
        natural=natural,
        dense=dense,
        cv=cv,
        source=source_summary,
        base=base_summary,
        prefix_manifest_sha256=_sha256_file(_resolve(args.prefix_cache) / "manifest.json"),
        architecture=architecture,
        scope=scope,
    )
    if cv["aggregate"]["passed"] is not True:
        report["terminal_reason"] = "train_only_pair_disjoint_cv_gate_failed"
        _write_training_report(output_report, report)
        return report

    final_fit = fit_route_only(
        source_control,
        natural_examples=natural,
        dense_examples=dense,
        question_inputs=normalized_questions,
        scene_inputs=signatures,
        route_factor_rank=args.route_factor_rank,
        seed=args.seed + 99_000_001,
        config=fit_config,
        log_phase="v62_route_final_fit",
        device=route_device,
    )
    inherited_sha256 = inherited_v60_state_sha256(final_fit.control)
    inherited_exact = (
        inherited_sha256 == source_state_sha256
        and set(source_state) == set(final_fit.control.inherited_state_names)
        and all(
            torch.equal(
                source_state[name].detach().cpu(),
                final_fit.control.state_dict()[name].detach().cpu(),
            )
            for name in source_state
        )
    )
    values_exact = True
    source_control.to(device=route_device, dtype=torch.float32)
    with torch.inference_mode():
        for example in natural:
            signature = signatures[example.scene_id].to(
                device=route_device, dtype=torch.float32
            )
            question = raw_questions[example.question].to(
                device=route_device, dtype=torch.float32
            )
            source_output = source_control.forward_from_signature(
                signature, question
            )
            candidate_output = final_fit.control.forward_from_signature(
                signature, question
            )
            values_exact = values_exact and all(
                torch.equal(first.detach().cpu(), second.detach().cpu())
                for first, second in (
                    (source_output.control_tokens, candidate_output.control_tokens),
                    (
                        source_output.coefficient_directions,
                        candidate_output.coefficient_directions,
                    ),
                    (source_output.control_rms, candidate_output.control_rms),
                )
            )
            if not values_exact:
                break
    final_checks = {
        "pair_disjoint_cv_passed": cv["aggregate"]["passed"] is True,
        "all_576_natural_training_routes_exact": final_fit.natural_metrics["exact"] is True,
        "all_dense_cartesian_training_routes_exact": final_fit.dense_metrics["exact"] is True,
        "natural_minimum_margin": final_fit.natural_metrics[
            "minimum_signed_logit_margin"
        ]
        >= fit_config.minimum_signed_logit_margin,
        "dense_minimum_margin": final_fit.dense_metrics[
            "minimum_signed_logit_margin"
        ]
        >= fit_config.minimum_signed_logit_margin,
        "inherited_v60_tensors_exact": inherited_exact,
        "v60_value_outputs_exact_on_all_576": values_exact,
        "only_factorized_route_trainable": final_fit.control.inherited_v60_state_frozen
        and all(
            name.startswith("factorized_route.")
            for name, parameter in final_fit.control.named_parameters()
            if parameter.requires_grad
        ),
    }
    report["final_fit"] = {
        "completed_epochs": final_fit.completed_epochs,
        "best_epoch": final_fit.best_epoch,
        "best_minimum_signed_logit_margin": (
            final_fit.best_minimum_signed_logit_margin
        ),
        "best_objective_loss": final_fit.best_objective_loss,
        "elapsed_seconds": final_fit.elapsed_seconds,
        "maximum_preclip_gradient_norm": final_fit.maximum_preclip_gradient_norm,
        "route_device": final_fit.route_device,
        "natural": final_fit.natural_metrics,
        "dense_cartesian": final_fit.dense_metrics,
        "checks": final_checks,
    }
    report["offline_checks_passed"] = all(final_checks.values())
    if report["offline_checks_passed"] is not True:
        report["terminal_reason"] = "final_train_route_or_v60_identity_gate_failed"
        _write_training_report(output_report, report)
        return report

    checkpoint_hashes = save_v5_control_checkpoint(
        output_checkpoint,
        control=final_fit.control,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
        source_v60_checkpoint_sha256=source_checkpoint_sha256,
        expected_inherited_state_sha256=source_state_sha256,
    )
    try:
        loaded, loaded_metadata = _load_control_head(
            output_checkpoint,
            hidden_size=runtime.language.hidden_size,
            device=torch.device("cpu"),
        )
        strict_reload = (
            isinstance(loaded, NormalizedFactorizedSceneQuestionControlV5)
            and set(loaded.state_dict()) == set(final_fit.control.state_dict())
            and all(
                torch.equal(
                    loaded.state_dict()[name].detach().cpu(),
                    final_fit.control.state_dict()[name].detach().cpu(),
                )
                for name in final_fit.control.state_dict()
            )
            and loaded_metadata.get("inherited_value_state_sha256")
            == source_state_sha256
        )
        if not strict_reload:
            raise RuntimeError("V62 saved checkpoint failed strict runtime reload")
    except BaseException:
        # This directory was created by this invocation and has never been
        # promoted.  Do not leave a runtime artifact after a failed seal test.
        shutil.rmtree(output_checkpoint, ignore_errors=True)
        raise
    report["checkpoint"] = checkpoint_hashes
    report["final_fit"]["checks"]["strict_saved_runtime_reload_exact"] = True
    report["terminal_reason"] = "train_only_gates_passed_checkpoint_saved"
    _write_training_report(output_report, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--source-v60-checkpoint", required=True)
    parser.add_argument("--source-v60-report", required=True)
    add_filtered_training_data_argument(parser)
    parser.add_argument("--prefix-cache", required=True)
    add_baseline_lock_authorization_argument(parser)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--seed", type=int, default=62062)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--route-factor-rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2400)
    parser.add_argument("--minimum-epochs", type=int, default=200)
    parser.add_argument("--success-patience", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=2.0)
    parser.add_argument("--minimum-signed-logit-margin", type=float, default=3.0)
    parser.add_argument("--margin-weight", type=float, default=0.1)
    parser.add_argument("--natural-loss-weight", type=float, default=1.0)
    parser.add_argument("--dense-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    destinations = {action.dest for action in parser._actions}
    if destinations & V62_PROHIBITED_TRAINER_DATA_ARGUMENTS:
        raise AssertionError("V62 trainer parser exposes a prohibited data boundary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = train_v62(args)
    print(
        json.dumps(
            {
                "offline_checks_passed": report["offline_checks_passed"],
                "promotion_eligible": False,
                "checkpoint_saved": report["checkpoint"] is not None,
            },
            sort_keys=True,
        )
    )
    return 0 if report["offline_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V62_ROUTE_CV_THRESHOLDS",
    "BaselineAuthorizationV62",
    "PairDisjointFoldV62",
    "RouteExampleV62",
    "RouteFitConfigV62",
    "RouteFitResultV62",
    "assess_route_cv",
    "authorized_training_scene_ids",
    "dense_cartesian_route_examples",
    "fit_route_only",
    "leave_one_pair_out_folds",
    "load_v62_baseline_authorization",
    "main",
    "natural_route_examples",
    "route_metrics",
    "route_population_sha256",
    "run_pair_disjoint_cv",
    "train_v62",
]
