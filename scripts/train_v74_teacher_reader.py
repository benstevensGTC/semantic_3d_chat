#!/usr/bin/env python3
"""Train-only V74 reader against verified Gemma continuous soft prompts.

The training code may use QA text and verified answer-class teachers, but the
saved candidate contains only learned tensors and an orthonormal numeric output
basis.  It never contains answers, labels, a codebook, or environmental text.
Official validation/test/oracle inputs are outside this command's path graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.scene_encoder.question_control_v3 import teacher_output_basis
from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.soft_prompt_teacher_v62 import load_v62_teacher_cache
from semantic_3d_chat.training.soft_prompt_teacher_v66 import (
    load_v66_answer_class_teacher_cache,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    PrototypeBankV73,
    _cosine_to_class,
    _predict_v73,
    _question_batch,
    changed_units_v73,
    load_config_v73,
    load_embedding_assets_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)
from semantic_3d_chat.training.v74_teacher_objective import (
    normalized_unclipped_teacher_delta_loss_v74,
    normalized_unclipped_teacher_value_loss_v74,
    raw_prompt_rms_from_coefficients_v74,
    teacher_coefficients_v74,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _teacher_bank(train, teacher_values) -> PrototypeBankV73:
    by_key = {row.key: row for row in train}
    grouped: defaultdict[str, list[tuple[tuple[str, str], torch.Tensor]]] = defaultdict(list)
    for key, value in teacher_values.items():
        if key in by_key:
            grouped[by_key[key].answer_class].append((key, value.float()))
    class_ids = tuple(sorted({row.answer_class for row in train}))
    if set(grouped) != set(class_ids):
        raise ValueError("V74 verified teachers do not cover every training class")
    prototypes = []
    for class_id in class_ids:
        ordered = sorted(grouped[class_id], key=lambda item: item[0])
        flat = torch.stack([F.normalize(value.flatten(), dim=0) for _, value in ordered])
        similarity = flat @ flat.T
        mean = similarity.mean(dim=1)
        best = max(
            (index for index, score in enumerate(mean) if score >= mean.max() - 1e-8),
            key=lambda index: tuple(-ord(char) for char in str(ordered[index][0])),
        )
        prototypes.append(ordered[best][1][0].contiguous())
    prototype_tensor = torch.stack(prototypes)
    basis = teacher_output_basis(prototype_tensor, rank=112)
    return PrototypeBankV73(
        class_ids=class_ids,
        prototypes=prototype_tensor,
        class_index={value: index for index, value in enumerate(class_ids)},
        output_basis=basis,
    )


def _metrics(model, rows, prefixes, questions, bank, device):
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
        batch_size=16,
        device=device,
        zero_scene=True,
    )
    similarity = _cosine_to_class(output, bank.prototypes)
    wrong_similarity = _cosine_to_class(wrong, bank.prototypes)
    prediction = similarity.argmax(-1)
    supported = [row.answer_class in bank.class_index for row in rows]
    correct = [
        ok and int(prediction[index]) == bank.class_index[row.answer_class]
        for index, (row, ok) in enumerate(zip(rows, supported, strict=True))
    ]
    index = {row.key: position for position, row in enumerate(rows)}
    complete = changes = positive = changed_correct = changed_total = 0
    margins = []
    wrong_drops = []
    deltas = []
    normalized_teacher_delta_mse = []
    teacher_delta_cosine = []
    for unit in changed_units_v73(rows):
        sides = [index[unit.left.key], index[unit.right.key]]
        if not all(supported[side] for side in sides):
            continue
        classes = [bank.class_index[unit.left.answer_class], bank.class_index[unit.right.answer_class]]
        complete += int(all(correct[side] for side in sides))
        changes += int(prediction[sides[0]] != prediction[sides[1]])
        changed_correct += sum(correct[side] for side in sides)
        changed_total += 2
        predicted_delta = output[sides[1]] - output[sides[0]]
        target_delta = bank.prototypes[classes[1]] - bank.prototypes[classes[0]]
        deltas.append(float(predicted_delta.square().mean().sqrt()))
        target_delta_energy = target_delta.square().sum().clamp_min(1e-8)
        normalized_teacher_delta_mse.append(
            float((predicted_delta - target_delta).square().sum() / target_delta_energy)
        )
        teacher_delta_cosine.append(
            float(
                F.cosine_similarity(
                    predicted_delta.flatten(), target_delta.flatten(), dim=0
                )
            )
        )
        for side, own, opposite in zip(sides, classes, reversed(classes), strict=True):
            margin = float(similarity[side, own] - similarity[side, opposite])
            wrong_margin = float(wrong_similarity[side, own] - wrong_similarity[side, opposite])
            margins.append(margin)
            wrong_drops.append(margin - wrong_margin)
            positive += int(margin > 0)
    supported_index = torch.tensor(
        [index for index, ok in enumerate(supported) if ok], dtype=torch.long
    )
    supported_output = output[supported_index]
    supported_target = torch.stack(
        [
            bank.prototypes[bank.class_index[rows[index].answer_class]]
            for index in supported_index.tolist()
        ]
    )
    normalized_teacher_mse = (
        (supported_output - supported_target).square().mean(dim=(1, 2))
        / supported_target.square().mean(dim=(1, 2)).clamp_min(1e-8)
    )
    output_token_rms = supported_output.square().mean(dim=-1).sqrt()
    return {
        "supported_accuracy": sum(correct) / max(sum(supported), 1),
        "changed_supported_accuracy": changed_correct / max(changed_total, 1),
        "complete_class_units": complete,
        "prediction_change_units": changes,
        "positive_own_over_opposite_sides": positive,
        "mean_own_over_opposite_margin": sum(margins) / max(len(margins), 1),
        "mean_correct_over_wrong_scene_margin": sum(wrong_drops) / max(len(wrong_drops), 1),
        "mean_changed_pair_control_delta_rms": sum(deltas) / max(len(deltas), 1),
        "mean_normalized_teacher_delta_mse": sum(normalized_teacher_delta_mse)
        / max(len(normalized_teacher_delta_mse), 1),
        "mean_teacher_delta_cosine": sum(teacher_delta_cosine)
        / max(len(teacher_delta_cosine), 1),
        "zero_scene_maximum_absolute_control": float(zero.abs().max()),
        "mean_normalized_teacher_mse": float(normalized_teacher_mse.mean()),
        "mean_flat_teacher_cosine": float(
            F.cosine_similarity(
                supported_output.flatten(1), supported_target.flatten(1)
            ).mean()
        ),
        "mean_output_token_rms": float(output_token_rms.mean()),
        "fraction_output_tokens_at_rms_cap": float(
            (output_token_rms >= model.maximum_control_rms - 1e-6)
            .float()
            .mean()
        ),
    }


def main(*, architecture_version: str = "v74") -> int:
    if architecture_version not in {"v74", "v75"}:
        raise ValueError("Verified-teacher reader architecture must be V74 or V75")
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--changed-side-ce-weight", type=float, default=2.0)
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--unclipped-value-weight", type=float, default=0.0)
    parser.add_argument("--unclipped-delta-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--coefficient-decoder-hidden-dimension", type=int, default=768
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--save-failed-diagnostic", action="store_true")
    args = parser.parse_args()
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
    primary, primary_metadata = load_v62_teacher_cache(
        "data_gemma4/training/v62_changed_teachers"
    )
    supplemental, supplemental_metadata = load_v66_answer_class_teacher_cache(
        "data_gemma4/training/v66_answer_class_teachers"
    )
    if set(primary) & set(supplemental):
        raise ValueError("V74 verified teacher sources overlap")
    bank = _teacher_bank(train, {**primary, **supplemental})
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    seed = 740074
    torch.manual_seed(seed)
    random.seed(seed)
    if args.coefficient_decoder_hidden_dimension < 1:
        raise ValueError("Coefficient-decoder hidden dimension must be positive")
    if architecture_version == "v75":
        model = DenseFullSceneContinuousControlV75(
            1536,
            bank.output_basis,
            coefficient_decoder_hidden_dimension=(
                args.coefficient_decoder_hidden_dimension
            ),
        ).to(device)
    else:
        model = DenseFullSceneContinuousControlV74(1536, bank.output_basis).to(
            device
        )
    if args.learning_rate <= 0.0:
        raise ValueError("V74 learning rate must be positive")
    if args.unclipped_value_weight < 0.0:
        raise ValueError("V74 unclipped value weight must be nonnegative")
    if args.unclipped_delta_weight < 0.0:
        raise ValueError("V74 unclipped delta weight must be nonnegative")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scenes = tuple(sorted({row.scene_id for row in train}))
    scene_index = {scene_id: index for index, scene_id in enumerate(scenes)}
    scene_prefix = torch.cat([prefixes[scene_id] for scene_id in scenes]).to(device)
    row_scene = torch.tensor([scene_index[row.scene_id] for row in train], device=device)
    questions, masks = _question_batch(train, assets.questions, device)
    classes = torch.tensor([bank.class_index[row.answer_class] for row in train], device=device)
    units = changed_units_v73(train)
    row_index = {row.key: index for index, row in enumerate(train)}
    sides = []
    pair_rows = []
    for unit in units:
        left, right = row_index[unit.left.key], row_index[unit.right.key]
        pair_rows.append((left, right))
        sides.extend(((left, int(classes[left]), int(classes[right])), (right, int(classes[right]), int(classes[left]))))
    side_index = torch.tensor([value[0] for value in sides], device=device)
    own_class = torch.tensor([value[1] for value in sides], device=device)
    opposite_class = torch.tensor([value[2] for value in sides], device=device)
    pair_left_index = torch.tensor(
        [value[0] for value in pair_rows], device=device
    )
    pair_right_index = torch.tensor(
        [value[1] for value in pair_rows], device=device
    )
    prototypes = bank.prototypes.to(device)
    prototype_coefficients = teacher_coefficients_v74(
        prototypes, model.output_basis
    )
    captured_coefficients: list[torch.Tensor] = []

    def capture_coefficients(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        result: torch.Tensor,
    ) -> None:
        captured_coefficients.append(
            result.reshape(
                -1, model.control_token_count, model.output_basis_rank
            )
        )

    coefficient_hook = model.coefficient_output.register_forward_hook(
        capture_coefficients
    )
    started = time.perf_counter()
    history = []
    for step in range(args.steps):
        key, value = model.encode_scene(scene_prefix)
        output = model.forward_encoded(
            key[row_scene], value[row_scene], questions, masks
        ).control_tokens
        if len(captured_coefficients) != 1:
            raise RuntimeError("V74 failed to capture one pre-clip coefficient tensor")
        raw_coefficients = captured_coefficients.pop()
        logits = _cosine_to_class(output, prototypes) / 0.20
        row_loss = F.cross_entropy(logits, classes)
        margin = logits[side_index, own_class] - logits[side_index, opposite_class]
        pair_loss = F.softplus(-margin).mean()
        changed_loss = F.cross_entropy(logits[side_index], classes[side_index])
        target = prototypes[classes]
        target_power = target.square().mean(dim=(1, 2)).clamp_min(1e-8)
        value_loss = ((output - target).square().mean(dim=(1, 2)) / target_power).mean()
        unclipped_value_loss = normalized_unclipped_teacher_value_loss_v74(
            raw_coefficients,
            prototype_coefficients[classes],
        )
        unclipped_delta_loss = normalized_unclipped_teacher_delta_loss_v74(
            raw_coefficients[pair_left_index],
            raw_coefficients[pair_right_index],
            prototype_coefficients[classes[pair_left_index]],
            prototype_coefficients[classes[pair_right_index]],
        )
        raw_token_rms = raw_prompt_rms_from_coefficients_v74(
            raw_coefficients, hidden_size=model.hidden_size
        )
        loss = (
            row_loss
            + args.pair_weight * pair_loss
            + args.changed_side_ce_weight * changed_loss
            + args.value_weight * value_loss
            + args.unclipped_value_weight * unclipped_value_loss
            + args.unclipped_delta_weight * unclipped_delta_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step in {0, 1, 9, 49, 99, 199, args.steps - 1}:
            history.append({
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "row_ce": float(row_loss.detach().cpu()),
                "pair_softplus": float(pair_loss.detach().cpu()),
                "changed_ce": float(changed_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "unclipped_value_loss": float(
                    unclipped_value_loss.detach().cpu()
                ),
                "unclipped_delta_loss": float(
                    unclipped_delta_loss.detach().cpu()
                ),
                "mean_raw_output_token_rms": float(
                    raw_token_rms.detach().mean().cpu()
                ),
                "fraction_raw_output_tokens_above_rms_cap": float(
                    (
                        raw_token_rms.detach()
                        >= model.maximum_control_rms
                    )
                    .float()
                    .mean()
                    .cpu()
                ),
                "preclip_gradient_norm": gradient,
            })
    coefficient_hook.remove()
    train_metrics = _metrics(model, train, prefixes, assets.questions, bank, device)
    held_metrics = _metrics(model, held, prefixes, assets.questions, bank, device)
    gates = {
        "supported_accuracy": held_metrics["supported_accuracy"] >= 0.80,
        "changed_supported_accuracy": held_metrics["changed_supported_accuracy"] >= 0.65,
        "complete_class_units": held_metrics["complete_class_units"] >= 8,
        "prediction_change_units": held_metrics["prediction_change_units"] >= 13,
        "positive_sides": held_metrics["positive_own_over_opposite_sides"] >= 34,
        "margin": held_metrics["mean_own_over_opposite_margin"] >= 0.20,
        "wrong_scene": held_metrics["mean_correct_over_wrong_scene_margin"] >= 0.02,
        "zero_scene": held_metrics["zero_scene_maximum_absolute_control"] == 0.0,
    }
    payload = {
        "artifact": (
            f"{architecture_version}_verified_teacher_dense_reader_"
            "train_pool_screen_v1"
        ),
        "architecture_version": architecture_version,
        "seed": seed,
        "device": str(device),
        "steps": args.steps,
        "pair_weight": args.pair_weight,
        "changed_side_ce_weight": args.changed_side_ce_weight,
        "value_weight": args.value_weight,
        "unclipped_value_weight": args.unclipped_value_weight,
        "unclipped_delta_weight": args.unclipped_delta_weight,
        "learning_rate": args.learning_rate,
        "coefficient_decoder_hidden_dimension": (
            args.coefficient_decoder_hidden_dimension
            if architecture_version == "v75"
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "fit_history": history,
        "train": train_metrics,
        "held": held_metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "architecture": model.audit().__dict__,
        "verified_teacher_count": len(primary) + len(supplemental),
        "verified_teacher_class_count": len(bank.class_ids),
        "teacher_metadata_sha256": hashlib.sha256(
            json.dumps([primary_metadata, supplemental_metadata], sort_keys=True).encode()
        ).hexdigest(),
        "prefix_manifest_base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "oracle_loaded": False,
        "checkpoint_published": False,
    }
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.candidate:
        if not payload["passed"] and not args.save_failed_diagnostic:
            raise RuntimeError("V74 verified-teacher candidate failed its numeric gates")
        candidate = args.candidate if args.candidate.is_absolute() else PROJECT_ROOT / args.candidate
        if candidate.exists():
            raise FileExistsError(candidate)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {key: value.detach().cpu().float() for key, value in model.state_dict().items()},
            candidate,
            metadata={
                "artifact": (
                    f"{architecture_version}_verified_teacher_dense_reader_"
                    "candidate_v1"
                ),
                "training_pool_only": "true",
                "runtime_promotion_forbidden_until_gemma_gate": "true",
                "numeric_gate_passed": str(payload["passed"]).lower(),
                "answer_codebook_serialized": "false",
                "environmental_text_inputs": "0",
            },
        )
        payload["candidate_sha256"] = _sha256(candidate)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
