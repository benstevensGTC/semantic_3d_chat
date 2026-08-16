#!/usr/bin/env python3
"""Train-only diagnostic for a minimal dense all-latent causal reader.

This deliberately small model answers one question with one full positive-floor
attention pass over all 256 immutable scene latents.  Its class logits are
strictly scene-value dependent: a zero scene produces exact zero logits for
every question.  The script is development-only and never opens official
validation, test, deferred-final, or oracle artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from semantic_3d_chat.training.train_question_control_v73 import (
    changed_units_v73,
    load_config_v73,
    load_embedding_assets_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)


class DenseReaderDiagnostic(nn.Module):
    """Bias-free 4-query bilinear reader with no question-only output path."""

    def __init__(self, class_count: int, *, hidden_size: int = 1536, width: int = 128):
        super().__init__()
        self.query_count = 4
        self.width = width
        self.scene_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.question_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.key = nn.Linear(hidden_size, width, bias=False)
        self.value = nn.Linear(hidden_size, width, bias=False)
        self.query = nn.Linear(hidden_size, width * self.query_count, bias=False)
        self.classifier = nn.Linear(width * self.query_count, class_count, bias=False)

    def forward(self, scene: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        key, value = self.encode_scene(scene)
        return self.forward_encoded(key, value, question)

    def encode_scene(self, scene: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, latent_count, _ = scene.shape
        key = self.key(self.scene_norm(scene))
        value = self.value(self.scene_norm(scene))
        return key, value

    def forward_encoded(
        self, key: torch.Tensor, value: torch.Tensor, question: torch.Tensor
    ) -> torch.Tensor:
        batch, latent_count = question.shape[0], key.shape[1]
        query = self.query(self.question_norm(question)).reshape(
            batch, self.query_count, self.width
        )
        score = torch.einsum("bqd,bld->bql", query, key) / math.sqrt(self.width)
        weight = 0.95 * torch.softmax(score, dim=-1) + 0.05 / latent_count
        context = torch.einsum("bql,bld->bqd", weight, value)
        interaction = (context * torch.tanh(query)).reshape(batch, -1)
        return self.classifier(interaction)


def _batches(values: list[int], size: int):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=730074)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.pair_weight < 0:
        raise ValueError("epochs must be positive and pair weight nonnegative")

    config = load_config_v73("configs/experiments/gemma4_v73_fullscene_controller.yaml")
    rows = load_training_rows_v73(config["training_qa"])
    train, held = split_rows_v73(rows)
    prefixes, manifest = load_prefixes_v73(
        config["prefix_cache"], {row.scene_id for row in rows}
    )
    assets = load_embedding_assets_v73(
        config["gemma_snapshot"],
        {row.question for row in rows},
        {row.answer_class: row.answer for row in train},
    )
    classes = tuple(sorted({row.answer_class for row in train}))
    class_index = {value: index for index, value in enumerate(classes)}
    scenes = {
        scene_id: prefix[0, 1:-1].float().contiguous()
        for scene_id, prefix in prefixes.items()
    }
    questions = {
        text: embedding.float().mean(dim=0).contiguous()
        for text, embedding in assets.questions.items()
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = DenseReaderDiagnostic(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    units = changed_units_v73(train)

    def tensors(batch):
        scene = torch.stack([scenes[row.scene_id] for row in batch]).to(device)
        question = torch.stack([questions[row.question] for row in batch]).to(device)
        target = torch.tensor(
            [class_index.get(row.answer_class, -1) for row in batch],
            dtype=torch.long,
            device=device,
        )
        return scene, question, target

    started = time.perf_counter()
    # One deterministic joint full-fold update avoids the oscillation caused by
    # alternating pure classification and pure pair-margin optimizer steps.
    # Scene K/V are computed once per unique scene and gathered to all rows, so
    # this remains a cheap diagnostic rather than duplicating 256 latents 576x.
    ordered_train = tuple(train)
    unique_scene_ids = tuple(sorted({row.scene_id for row in ordered_train}))
    unique_scene_index = {scene_id: index for index, scene_id in enumerate(unique_scene_ids)}
    unique_scenes = torch.stack([scenes[scene_id] for scene_id in unique_scene_ids]).to(device)
    row_scene_index = torch.tensor(
        [unique_scene_index[row.scene_id] for row in ordered_train], device=device
    )
    all_questions = torch.stack([questions[row.question] for row in ordered_train]).to(device)
    all_targets = torch.tensor(
        [class_index[row.answer_class] for row in ordered_train],
        dtype=torch.long,
        device=device,
    )
    row_index = {row.key: index for index, row in enumerate(ordered_train)}
    pair_rows = torch.tensor(
        [[row_index[unit.left.key], row_index[unit.right.key]] for unit in units],
        dtype=torch.long,
        device=device,
    )
    pair_targets = all_targets[pair_rows]
    for _epoch in range(args.epochs):
        model.train()
        key, value = model.encode_scene(unique_scenes)
        logits = model.forward_encoded(
            key[row_scene_index], value[row_scene_index], all_questions
        )
        pair_logits = logits[pair_rows]
        own = pair_logits.gather(2, pair_targets[:, :, None]).squeeze(-1)
        opposite = pair_logits.gather(
            2, pair_targets.flip(1)[:, :, None]
        ).squeeze(-1)
        loss = F.cross_entropy(logits, all_targets) + args.pair_weight * F.softplus(
            -(own - opposite)
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    @torch.inference_mode()
    def evaluate(eval_rows):
        model.eval()
        output = []
        for offset in range(0, len(eval_rows), 64):
            scene, question, _target = tensors(eval_rows[offset : offset + 64])
            output.append(model(scene, question).cpu())
        logits = torch.cat(output)
        prediction = logits.argmax(-1)
        supported = [row.answer_class in class_index for row in eval_rows]
        correct = [
            supported[index]
            and int(prediction[index]) == class_index[row.answer_class]
            for index, row in enumerate(eval_rows)
        ]
        by_key = {row.key: index for index, row in enumerate(eval_rows)}
        complete = changes = positive = 0
        margins = []
        wrong_drops = []
        families = defaultdict(list)
        supported_changed = 0
        for unit in changed_units_v73(eval_rows):
            indexes = [by_key[unit.left.key], by_key[unit.right.key]]
            if not all(supported[index] for index in indexes):
                continue
            supported_changed += 2
            targets = [
                class_index[unit.left.answer_class],
                class_index[unit.right.answer_class],
            ]
            complete += int(all(correct[index] for index in indexes))
            changes += int(prediction[indexes[0]] != prediction[indexes[1]])
            side_margins = []
            for index, own, opposite in zip(indexes, targets, reversed(targets), strict=True):
                margin = float(logits[index, own] - logits[index, opposite])
                margins.append(margin)
                side_margins.append(margin)
                families[unit.change_type].append(margin)
                positive += int(margin > 0)
            # Swapping to the paired scene swaps these same-question rows.
            wrong_drops.extend([sum(side_margins), sum(side_margins)])
        changed_correct = sum(
            correct[index]
            for index, row in enumerate(eval_rows)
            if row.expected_change and supported[index]
        )
        return {
            "supported_accuracy": sum(correct) / max(sum(supported), 1),
            "changed_supported_accuracy": changed_correct / max(supported_changed, 1),
            "complete_class_units": complete,
            "prediction_change_units": changes,
            "positive_own_over_opposite_sides": positive,
            "mean_own_over_opposite_margin": sum(margins) / max(len(margins), 1),
            "mean_correct_over_wrong_scene_margin": sum(wrong_drops)
            / max(len(wrong_drops), 1),
            "family_mean_margins": {
                key: sum(value) / len(value) for key, value in sorted(families.items())
            },
        }

    train_metrics = evaluate(train)
    held_metrics = evaluate(held)
    with torch.inference_mode():
        _scene, question, _target = tensors(held[:1])
        zero_maximum = float(
            model(torch.zeros_like(_scene), question).abs().max().detach().cpu()
        )
    payload = {
        "artifact": "v74_dense_all_latent_train_pool_diagnostic_v1",
        "seed": args.seed,
        "epochs": args.epochs,
        "pair_softplus_weight": args.pair_weight,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "train": train_metrics,
        "held": held_metrics,
        "zero_scene_maximum_absolute_logit": zero_maximum,
        "all_256_latents_receive_positive_attention_floor": True,
        "minimum_attention_weight": 0.05 / 256,
        "question_only_output_path": False,
        "question_dependent_retrieval": False,
        "immutable_prefix_manifest_sha256": manifest.get("base_checkpoint_sha256"),
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "oracle_loaded": False,
        "checkpoint_published": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
