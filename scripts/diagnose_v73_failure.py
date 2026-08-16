#!/usr/bin/env python3
"""Train-only diagnosis of the V73 full-scene reader failure.

This script is deliberately restricted to the already opened 40-scene
training pool and its locked historical/replicated pair split.  It never reads
official validation, test, deferred-final, or oracle data and never writes a
runtime checkpoint.

The diagnostic compares:

* the released V73 value-only two-hop reader;
* a one-hop full-scene reader whose attended values are multiplicatively gated
  by its question queries (still exact-zero for an all-zero scene); and
* the same one-hop reader with that multiplicative interaction removed.

The gated reader is evaluated both as a cheap answer-class probe and through
V73's exact LM-native four-token prototype objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.training.train_question_control_v73 import (
    EXPECTED_CONTROL_TOKENS,
    EXPECTED_HIDDEN_SIZE,
    RowV73,
    _cosine_to_class,
    _fit_reader_v73,
    _prefix_batch,
    _predict_v73,
    _question_batch,
    _select_device,
    _targets,
    build_preflight_v73,
    build_prototype_bank_v73,
    changed_units_v73,
    load_config_v73,
    load_embedding_assets_v73,
    load_prefixes_v73,
)
from semantic_3d_chat.scene_encoder.question_control_v73 import (
    FullSceneSetAttentionQuestionControlV73,
)


class DenseQuestionSceneCore(nn.Module):
    """One full-image-prefix attention read with an optional bilinear gate."""

    def __init__(
        self,
        *,
        model_dimension: int = 128,
        query_count: int = 4,
        uniform_floor_mass: float = 0.05,
        gated: bool,
    ) -> None:
        super().__init__()
        self.model_dimension = int(model_dimension)
        self.query_count = int(query_count)
        self.uniform_floor_mass = float(uniform_floor_mass)
        self.gated = bool(gated)
        self.key = nn.Linear(EXPECTED_HIDDEN_SIZE, model_dimension, bias=False)
        self.value = nn.Linear(EXPECTED_HIDDEN_SIZE, model_dimension, bias=False)
        self.query = nn.Linear(
            EXPECTED_HIDDEN_SIZE, query_count * model_dimension, bias=False
        )

    def encode_scene(
        self, scene_prefix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        environment = F.layer_norm(
            scene_prefix[:, 1:-1].float(), (EXPECTED_HIDDEN_SIZE,)
        )
        return self.key(environment), self.value(environment)

    def encode_question(
        self,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = question_attention_mask.to(question_embeddings).unsqueeze(-1)
        question_mean = (question_embeddings.float() * mask).sum(dim=1) / mask.sum(
            dim=1
        ).clamp_min(1.0)
        question_mean = F.layer_norm(question_mean, (EXPECTED_HIDDEN_SIZE,))
        return self.query(question_mean).reshape(
            -1, self.query_count, self.model_dimension
        )

    def read_encoded(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query: torch.Tensor,
        scene_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scene_index is not None:
            key = key[scene_index]
            value = value[scene_index]
        score = torch.einsum("bqd,bld->bql", query, key) / math.sqrt(
            float(self.model_dimension)
        )
        probability = torch.softmax(score.float(), dim=-1).to(value)
        probability = (
            (1.0 - self.uniform_floor_mass) * probability
            + self.uniform_floor_mass / float(key.shape[1])
        )
        context = torch.einsum("bql,bld->bqd", probability, value)
        if self.gated:
            context = context * torch.tanh(query)
        return context

    def forward(
        self,
        scene_prefix: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        key, value = self.encode_scene(scene_prefix)
        query = self.encode_question(question_embeddings, question_attention_mask)
        return self.read_encoded(key, value, query)


class DenseClassProbe(nn.Module):
    def __init__(self, class_count: int, *, gated: bool) -> None:
        super().__init__()
        self.core = DenseQuestionSceneCore(gated=gated)
        self.output = nn.Linear(
            self.core.query_count * self.core.model_dimension,
            class_count,
            bias=False,
        )

    def forward(
        self,
        scene_prefix: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.output(
            self.core(
                scene_prefix, question_embeddings, question_attention_mask
            ).flatten(1)
        )


class DenseNativePrototypeReader(nn.Module):
    """Gated dense read followed by V73's native continuous output basis."""

    def __init__(
        self, output_basis: torch.Tensor, *, maximum_control_rms: float = 0.25
    ) -> None:
        super().__init__()
        self.core = DenseQuestionSceneCore(gated=True)
        self.output_basis_rank = int(output_basis.shape[0])
        self.coefficient_output = nn.Linear(
            self.core.query_count * self.core.model_dimension,
            EXPECTED_CONTROL_TOKENS * self.output_basis_rank,
            bias=False,
        )
        self.register_buffer("output_basis", output_basis.float().clone())
        self.maximum_control_rms = float(maximum_control_rms)

    def output_from_context(self, context: torch.Tensor) -> SimpleNamespace:
        context = context.flatten(1)
        coefficient = self.coefficient_output(context).reshape(
            -1, EXPECTED_CONTROL_TOKENS, self.output_basis_rank
        )
        raw = torch.einsum("bcr,rh->bch", coefficient, self.output_basis)
        raw_rms = raw.square().mean(dim=-1, keepdim=True).sqrt()
        scale = torch.clamp(
            self.maximum_control_rms / raw_rms.clamp_min(1e-8), max=1.0
        )
        output = raw * scale
        return SimpleNamespace(
            control_tokens=output,
            coefficient_directions=F.normalize(coefficient, dim=-1, eps=1e-8),
            control_rms=output.square().mean(dim=-1).sqrt(),
        )

    def forward(
        self,
        scene_prefix: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        if question_attention_mask is None:
            question_attention_mask = torch.ones(
                question_embeddings.shape[:2],
                dtype=torch.bool,
                device=question_embeddings.device,
            )
        return self.output_from_context(
            self.core(scene_prefix, question_embeddings, question_attention_mask)
        )


def fit_joint_native_reader(
    model: DenseNativePrototypeReader,
    train_rows: Sequence[RowV73],
    *,
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    bank: Any,
    device: torch.device,
    seed: int,
    steps: int = 400,
    pair_weight: float = 0.5,
    changed_side_cross_entropy_weight: float = 0.0,
    learning_rate: float = 0.003,
) -> dict[str, Any]:
    """Joint full-fold CE + paired-softplus optimization used by the best probe."""

    torch.manual_seed(seed)
    random.seed(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scene_ids = sorted({row.scene_id for row in train_rows})
    scene_index_by_id = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    scene_prefix = torch.cat([prefixes[scene_id] for scene_id in scene_ids]).to(device)
    row_scene_index = torch.tensor(
        [scene_index_by_id[row.scene_id] for row in train_rows],
        dtype=torch.long,
        device=device,
    )
    question, mask = _question_batch(train_rows, questions, device)
    _target, class_ids = _targets(train_rows, bank, device)
    row_index = {row.key: index for index, row in enumerate(train_rows)}
    pair_sides: list[tuple[int, int, int]] = []
    for unit in changed_units_v73(train_rows):
        left = row_index[unit.left.key]
        right = row_index[unit.right.key]
        pair_sides.extend(
            (
                (left, int(class_ids[left]), int(class_ids[right])),
                (right, int(class_ids[right]), int(class_ids[left])),
            )
        )
    side_index = torch.tensor([item[0] for item in pair_sides], device=device)
    own_class = torch.tensor([item[1] for item in pair_sides], device=device)
    opposite_class = torch.tensor([item[2] for item in pair_sides], device=device)
    prototypes = bank.prototypes.to(device)
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    maximum_gradient = 0.0
    for step in range(steps):
        key, value = model.core.encode_scene(scene_prefix)
        query = model.core.encode_question(question, mask)
        context = model.core.read_encoded(key, value, query, row_scene_index)
        output = model.output_from_context(context).control_tokens
        logits = _cosine_to_class(output, prototypes) / 0.20
        row_loss = F.cross_entropy(logits, class_ids)
        side_margin = (
            logits[side_index, own_class] - logits[side_index, opposite_class]
        )
        pair_loss = F.softplus(-side_margin).mean()
        changed_side_loss = F.cross_entropy(
            logits[side_index], class_ids[side_index]
        )
        loss = (
            row_loss
            + pair_weight * pair_loss
            + changed_side_cross_entropy_weight * changed_side_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        if step in {0, 1, 9, 49, 99, 199, steps - 1}:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "row_cross_entropy": float(row_loss.detach().cpu()),
                    "pair_softplus": float(pair_loss.detach().cpu()),
                    "changed_side_cross_entropy": float(
                        changed_side_loss.detach().cpu()
                    ),
                }
            )
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "steps": steps,
        "pair_weight": pair_weight,
        "changed_side_cross_entropy_weight": changed_side_cross_entropy_weight,
        "learning_rate": learning_rate,
        "maximum_preclip_gradient_norm": maximum_gradient,
        "history": history,
    }


def _class_targets(rows: Sequence[RowV73], class_index: Mapping[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [class_index[row.answer_class] for row in rows],
        dtype=torch.long,
        device=device,
    )


def fit_class_probe(
    model: DenseClassProbe,
    train_rows: Sequence[RowV73],
    *,
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    class_index: Mapping[str, int],
    device: torch.device,
    seed: int,
    epochs: int = 80,
    pair_weight: float = 0.5,
    gradient_clip_norm: float = 1.0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    units = changed_units_v73(train_rows)
    started = time.perf_counter()
    losses: list[float] = []
    maximum_gradient = 0.0
    for _epoch in range(epochs):
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        for offset in range(0, len(order), 48):
            batch = [train_rows[index] for index in order[offset : offset + 48]]
            question, mask = _question_batch(batch, questions, device)
            logits = model(_prefix_batch(batch, prefixes, device), question, mask)
            loss = F.cross_entropy(logits, _class_targets(batch, class_index, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm if gradient_clip_norm > 0.0 else float("inf"),
                )
            )
            maximum_gradient = max(maximum_gradient, gradient)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        unit_order = torch.randperm(len(units), generator=generator).tolist()
        for offset in range(0, len(unit_order), 20):
            selected = [units[index] for index in unit_order[offset : offset + 20]]
            batch = [row for unit in selected for row in (unit.left, unit.right)]
            question, mask = _question_batch(batch, questions, device)
            logits = model(_prefix_batch(batch, prefixes, device), question, mask)
            own = _class_targets(batch, class_index, device)
            opposite = own.reshape(-1, 2).flip(1).reshape(-1)
            index = torch.arange(len(batch), device=device)
            margin = logits[index, own] - logits[index, opposite]
            loss = pair_weight * F.softplus(-margin).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm if gradient_clip_norm > 0.0 else float("inf"),
                )
            )
            maximum_gradient = max(maximum_gradient, gradient)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "steps": len(losses),
        "initial_loss": losses[0],
        "last_loss": losses[-1],
        "maximum_preclip_gradient_norm": maximum_gradient,
    }


@torch.inference_mode()
def predict_class_probe(
    model: DenseClassProbe,
    rows: Sequence[RowV73],
    *,
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    device: torch.device,
    wrong_scene: bool = False,
    zero_scene: bool = False,
) -> torch.Tensor:
    model.eval().to(device)
    result: list[torch.Tensor] = []
    for offset in range(0, len(rows), 48):
        batch = rows[offset : offset + 48]
        question, mask = _question_batch(batch, questions, device)
        prefix = _prefix_batch(batch, prefixes, device, wrong_scene=wrong_scene)
        if zero_scene:
            prefix = torch.zeros_like(prefix)
        result.append(model(prefix, question, mask).detach().cpu())
    return torch.cat(result)


def class_metrics(
    model: DenseClassProbe,
    rows: Sequence[RowV73],
    *,
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    class_index: Mapping[str, int],
    device: torch.device,
) -> dict[str, Any]:
    logits = predict_class_probe(
        model, rows, prefixes=prefixes, questions=questions, device=device
    )
    wrong = predict_class_probe(
        model,
        rows,
        prefixes=prefixes,
        questions=questions,
        device=device,
        wrong_scene=True,
    )
    zero = predict_class_probe(
        model,
        rows[:16],
        prefixes=prefixes,
        questions=questions,
        device=device,
        zero_scene=True,
    )
    prediction = logits.argmax(dim=-1)
    supported = [row.answer_class in class_index for row in rows]
    correct = [
        bool(ok and int(prediction[index]) == class_index[row.answer_class])
        for index, (row, ok) in enumerate(zip(rows, supported, strict=True))
    ]
    index_by_key = {row.key: index for index, row in enumerate(rows)}
    complete = changes = positive = supported_sides = 0
    margins: list[float] = []
    wrong_drops: list[float] = []
    family: dict[str, list[float]] = defaultdict(list)
    for unit in changed_units_v73(rows):
        indexes = [index_by_key[unit.left.key], index_by_key[unit.right.key]]
        if not all(supported[index] for index in indexes):
            continue
        supported_sides += 2
        classes = [class_index[unit.left.answer_class], class_index[unit.right.answer_class]]
        complete += int(all(correct[index] for index in indexes))
        changes += int(int(prediction[indexes[0]]) != int(prediction[indexes[1]]))
        for index, own, opposite in zip(indexes, classes, reversed(classes), strict=True):
            margin = float(logits[index, own] - logits[index, opposite])
            wrong_margin = float(wrong[index, own] - wrong[index, opposite])
            margins.append(margin)
            wrong_drops.append(margin - wrong_margin)
            family[unit.change_type].append(margin)
            positive += int(margin > 0.0)
    supported_total = sum(supported)
    changed_exact = sum(
        correct[index] for index, row in enumerate(rows) if row.expected_change
    )
    return {
        "supported_accuracy": sum(correct) / max(supported_total, 1),
        "changed_supported_accuracy": changed_exact / max(supported_sides, 1),
        "complete_class_units": complete,
        "prediction_change_units": changes,
        "positive_own_over_opposite_sides": positive,
        "mean_own_over_opposite_margin": sum(margins) / max(len(margins), 1),
        "mean_correct_over_wrong_scene_margin": sum(wrong_drops) / max(len(wrong_drops), 1),
        "zero_scene_maximum_absolute_logit": float(zero.abs().max()),
        "distinct_predicted_classes": int(prediction.unique().numel()),
        "family_mean_margins": {
            key: sum(value) / len(value) for key, value in sorted(family.items())
        },
    }


def native_metrics(
    model: nn.Module,
    rows: Sequence[RowV73],
    *,
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    bank: Any,
    device: torch.device,
) -> dict[str, Any]:
    output = _predict_v73(
        model,
        rows,
        prefixes=prefixes,
        questions=questions,
        batch_size=48,
        device=device,
    )
    wrong = _predict_v73(
        model,
        rows,
        prefixes=prefixes,
        questions=questions,
        batch_size=48,
        device=device,
        wrong_scene=True,
    )
    zero = _predict_v73(
        model,
        rows[:16],
        prefixes=prefixes,
        questions=questions,
        batch_size=48,
        device=device,
        zero_scene=True,
    )
    similarity = _cosine_to_class(output, bank.prototypes)
    wrong_similarity = _cosine_to_class(wrong, bank.prototypes)
    prediction = similarity.argmax(dim=-1)
    supported = [row.answer_class in bank.class_index for row in rows]
    correct = [
        bool(ok and int(prediction[index]) == bank.class_index[row.answer_class])
        for index, (row, ok) in enumerate(zip(rows, supported, strict=True))
    ]
    index_by_key = {row.key: index for index, row in enumerate(rows)}
    complete = changes = positive = supported_sides = 0
    margins: list[float] = []
    wrong_drops: list[float] = []
    scene_delta_rms: list[float] = []
    scene_cosines: list[float] = []
    family: dict[str, list[float]] = defaultdict(list)
    for unit in changed_units_v73(rows):
        indexes = [index_by_key[unit.left.key], index_by_key[unit.right.key]]
        if not all(supported[index] for index in indexes):
            continue
        supported_sides += 2
        classes = [bank.class_index[unit.left.answer_class], bank.class_index[unit.right.answer_class]]
        complete += int(all(correct[index] for index in indexes))
        changes += int(int(prediction[indexes[0]]) != int(prediction[indexes[1]]))
        scene_delta_rms.append(
            float((output[indexes[1]] - output[indexes[0]]).square().mean().sqrt())
        )
        scene_cosines.append(
            float(F.cosine_similarity(output[indexes[0]].flatten(), output[indexes[1]].flatten(), dim=0))
        )
        for index, own, opposite in zip(indexes, classes, reversed(classes), strict=True):
            margin = float(similarity[index, own] - similarity[index, opposite])
            wrong_margin = float(wrong_similarity[index, own] - wrong_similarity[index, opposite])
            margins.append(margin)
            wrong_drops.append(margin - wrong_margin)
            family[unit.change_type].append(margin)
            positive += int(margin > 0.0)
    supported_total = sum(supported)
    changed_exact = sum(
        correct[index] for index, row in enumerate(rows) if row.expected_change
    )
    return {
        "supported_accuracy": sum(correct) / max(supported_total, 1),
        "changed_supported_accuracy": changed_exact / max(supported_sides, 1),
        "complete_class_units": complete,
        "prediction_change_units": changes,
        "positive_own_over_opposite_sides": positive,
        "mean_own_over_opposite_margin": sum(margins) / max(len(margins), 1),
        "mean_correct_over_wrong_scene_margin": sum(wrong_drops) / max(len(wrong_drops), 1),
        "zero_scene_maximum_absolute_control": float(zero.abs().max()),
        "distinct_predicted_classes": int(prediction.unique().numel()),
        "mean_output_rms": float(output.square().mean(dim=(1, 2)).sqrt().mean()),
        "mean_changed_pair_control_delta_rms": sum(scene_delta_rms) / max(len(scene_delta_rms), 1),
        "mean_changed_pair_control_cosine": sum(scene_cosines) / max(len(scene_cosines), 1),
        "family_mean_margins": {
            key: sum(value) / len(value) for key, value in sorted(family.items())
        },
    }


def native_candidate_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed absolute screen required before preserving a quarantined candidate."""

    thresholds: dict[str, float | int] = {
        "supported_accuracy": 0.80,
        "changed_supported_accuracy": 0.65,
        "complete_class_units": 8,
        "prediction_change_units": 13,
        "positive_own_over_opposite_sides": 34,
        "mean_own_over_opposite_margin": 0.20,
        "mean_correct_over_wrong_scene_margin": 0.02,
        "zero_scene_maximum_absolute_control": 0.0,
    }
    checks = {
        key: (
            metrics[key] == threshold
            if key == "zero_scene_maximum_absolute_control"
            else metrics[key] >= threshold
        )
        for key, threshold in thresholds.items()
    }
    return {
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v73_fullscene_controller.yaml",
    )
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/v73_failure_diagnostic.json",
    )
    parser.add_argument("--skip-v73-rerun", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--joint-native", action="store_true")
    parser.add_argument("--joint-native-steps", type=int, default=400)
    parser.add_argument("--joint-native-pair-weight", type=float, default=0.5)
    parser.add_argument(
        "--joint-native-changed-side-ce-weight", type=float, default=0.0
    )
    parser.add_argument("--candidate-output")
    parser.add_argument(
        "--class-variant",
        choices=("both", "gated", "attention_only"),
        default="both",
    )
    parser.add_argument("--class-epochs", type=int, default=80)
    parser.add_argument("--class-pair-weight", type=float, default=0.5)
    parser.add_argument("--class-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "mps"))
    args = parser.parse_args()

    config = load_config_v73(args.config)
    preflight, state = build_preflight_v73(config)
    rows = state["rows"]
    train = state["train"]
    held = state["held"]
    prefixes, _manifest = load_prefixes_v73(
        config["prefix_cache"], {row.scene_id for row in rows}
    )
    answers = {row.answer_class: row.answer for row in train}
    assets = load_embedding_assets_v73(
        config["gemma_snapshot"], {row.question for row in rows}, answers
    )
    bank = build_prototype_bank_v73(
        train,
        assets.answers,
        target_rms=float(config["fit"]["prototype_rms"]),
        basis_rank=int(config["architecture"]["output_basis_rank"]),
    )
    device = _select_device(args.device or str(config["device"]))
    seed = int(config["seed"])
    result: dict[str, Any] = {
        "artifact": "v73_failure_diagnostic_train_pool_only_v1",
        "scope": {
            "training_pool_only": True,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
            "checkpoint_published": False,
            "gemma_generation_used": False,
        },
        "preflight": preflight,
        "device": str(device),
    }

    class_results: dict[str, Any] = {}
    variants = {
        "both": (("gated", True), ("attention_only", False)),
        "gated": (("gated", True),),
        "attention_only": (("attention_only", False),),
    }[args.class_variant]
    for name, gated in variants:
        torch.manual_seed(seed)
        model = DenseClassProbe(len(bank.class_ids), gated=gated)
        fit = fit_class_probe(
            model,
            train,
            prefixes=prefixes,
            questions=assets.questions,
            class_index=bank.class_index,
            device=device,
            seed=seed,
            epochs=args.class_epochs,
            pair_weight=args.class_pair_weight,
            gradient_clip_norm=args.class_gradient_clip_norm,
        )
        class_results[name] = {
            "fit": fit,
            "train": class_metrics(
                model,
                train,
                prefixes=prefixes,
                questions=assets.questions,
                class_index=bank.class_index,
                device=device,
            ),
            "held": class_metrics(
                model,
                held,
                prefixes=prefixes,
                questions=assets.questions,
                class_index=bank.class_index,
                device=device,
            ),
        }
    result["class_probe"] = class_results

    if not args.skip_native:
        torch.manual_seed(seed)
        native = DenseNativePrototypeReader(
            bank.output_basis,
            maximum_control_rms=float(config["architecture"]["maximum_control_rms"]),
        )
        if args.joint_native:
            native_fit = fit_joint_native_reader(
                native,
                train,
                prefixes=prefixes,
                questions=assets.questions,
                bank=bank,
                device=device,
                seed=seed,
                steps=args.joint_native_steps,
                pair_weight=args.joint_native_pair_weight,
                changed_side_cross_entropy_weight=(
                    args.joint_native_changed_side_ce_weight
                ),
            )
        else:
            native_fit = _fit_reader_v73(
                native,
                train,
                prefixes=prefixes,
                questions=assets.questions,
                bank=bank,
                config=config,
                seed=seed,
                device=device,
            )
        train_native = native_metrics(
            native,
            train,
            prefixes=prefixes,
            questions=assets.questions,
            bank=bank,
            device=device,
        )
        held_native = native_metrics(
            native,
            held,
            prefixes=prefixes,
            questions=assets.questions,
            bank=bank,
            device=device,
        )
        candidate_gate = native_candidate_gate(held_native)
        result["gated_native_prototype"] = {
            "joint_full_fold_objective": args.joint_native,
            "trainable_parameters": sum(
                value.numel() for value in native.parameters() if value.requires_grad
            ),
            "fit": native_fit,
            "train": train_native,
            "held": held_native,
            "candidate_gate": candidate_gate,
        }
        if args.candidate_output:
            if not args.joint_native:
                raise ValueError("candidate preservation requires joint-native mode")
            if not candidate_gate["passed"]:
                raise RuntimeError("candidate failed the locked absolute gate")
            candidate = Path(args.candidate_output)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            if candidate.exists():
                raise FileExistsError(candidate)
            candidate.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in native.state_dict().items()
                },
                str(candidate),
                metadata={
                    "artifact": "v74_gated_native_train_pool_candidate_v1",
                    "seed": str(seed),
                    "training_pool_only": "true",
                    "runtime_promotion_forbidden": "true",
                    "question_or_answer_text_serialized": "false",
                    "environmental_text_inputs": "0",
                    "official_validation_loaded": "false",
                    "official_test_loaded": "false",
                    "oracle_loaded": "false",
                },
            )
            result["candidate"] = {
                "artifact": "v74_gated_native_train_pool_candidate_v1",
                "path": str(candidate.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(candidate),
                "runtime_promotion_forbidden": True,
                "checkpoint_published": False,
                "gate": candidate_gate,
            }

    if not args.skip_v73_rerun:
        torch.manual_seed(seed)
        architecture = config["architecture"]
        v73 = FullSceneSetAttentionQuestionControlV73(
            EXPECTED_HIDDEN_SIZE,
            bank.output_basis,
            expected_environment_latents=256,
            control_token_count=4,
            model_dimension=int(architecture["model_dimension"]),
            head_count=int(architecture["head_count"]),
            feedforward_dimension=int(architecture["feedforward_dimension"]),
            scene_encoder_layers=int(architecture["scene_encoder_layers"]),
            scene_cross_attention_layers=int(architecture["scene_cross_attention_layers"]),
            internal_reader_slots=int(architecture["internal_reader_slots"]),
            uniform_floor_mass=float(architecture["uniform_floor_mass"]),
            maximum_control_rms=float(architecture["maximum_control_rms"]),
            initial_control_rms=float(architecture["initial_control_rms"]),
        )
        v73_fit = _fit_reader_v73(
            v73,
            train,
            prefixes=prefixes,
            questions=assets.questions,
            bank=bank,
            config=config,
            seed=seed,
            device=device,
        )
        result["v73_rerun"] = {
            "fit": v73_fit,
            "train": native_metrics(
                v73,
                train,
                prefixes=prefixes,
                questions=assets.questions,
                bank=bank,
                device=device,
            ),
            "held": native_metrics(
                v73,
                held,
                prefixes=prefixes,
                questions=assets.questions,
                bank=bank,
                device=device,
            ),
        }

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
