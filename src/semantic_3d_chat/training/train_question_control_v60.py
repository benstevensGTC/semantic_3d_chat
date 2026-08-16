"""Cheap cached-teacher training for the V3 orthogonal-basis controller.

This successor never backpropagates through Gemma.  It loads only the already
verified numeric V58/V59 teacher caches, immutable full-scene prefixes, and
question embeddings from the frozen local language stack.  It must pass strict
offline reconstruction, prompt-cosine, RMS, route, and pair-delta gates before
emitting a saved-runtime candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v59_multiscene_train_gate import (
    ANCHOR_PAIR_ID,
    EXPANSION_PAIR_IDS,
    LOCKED_SCENE_IDS,
)
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
    teacher_output_basis,
)
from semantic_3d_chat.training.question_control_v3_checkpoint import (
    save_v3_control_checkpoint,
)
from semantic_3d_chat.training.soft_prompt_teacher_v58 import (
    load_teacher_artifact,
)
from semantic_3d_chat.training.soft_prompt_teacher_v59 import (
    load_expansion_teachers,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _changed_pair_units,
    _load_sanitized_runtime_config,
    _log_event,
    _resolve,
    _safe_output_path,
    _select_training_device,
    _sha256_file,
    _write_training_report,
    freeze_base_runtime,
    load_prefix_cache,
    load_training_records,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _pooled_question_embedding,
)

_EXPECTED_RECORDS: Final[int] = 144
_EXPECTED_CHANGED: Final[int] = 22


@dataclass(frozen=True)
class V3Example:
    scene_id: str
    question_id: str
    signature: torch.Tensor
    question: torch.Tensor
    target: torch.Tensor | None
    route_label: float
    group: str

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.question_id


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _changed_keys(records: Sequence[Any]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    anchor = {
        (record.scene_id, record.question_id)
        for record in records
        if record.counterfactual_expected_change is True
        and record.counterfactual_pair_id == ANCHOR_PAIR_ID
    }
    expansion = {
        (record.scene_id, record.question_id)
        for record in records
        if record.counterfactual_expected_change is True
        and record.counterfactual_pair_id in EXPANSION_PAIR_IDS
    }
    if len(records) != _EXPECTED_RECORDS or len(anchor) != 8 or len(expansion) != 14:
        raise ValueError("V60 locked six-scene inventory changed")
    return anchor, expansion


def _validate_args(args: argparse.Namespace) -> None:
    if tuple(sorted(args.scene_id)) != LOCKED_SCENE_IDS:
        raise ValueError("V60 requires the exact locked six training scenes")
    for field in (
        "basis_rank",
        "moment_count",
        "interaction_dim",
        "trunk_dim",
        "epochs",
        "batch_size",
        "changed_repeats",
        "retention_repeats",
        "log_every",
    ):
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"V60 {field} must be a positive integer")
    for field in (
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "coefficient_weight",
        "log_rms_weight",
        "reconstruction_weight",
        "pair_delta_weight",
        "route_weight",
        "maximum_control_rms",
        "initial_control_rms",
        "gate_threshold",
    ):
        value = float(getattr(args, field))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"V60 {field} must be finite and nonnegative")
    if not 0.0 < args.initial_control_rms < args.maximum_control_rms <= 1.0:
        raise ValueError("V60 RMS bounds are invalid")


def _teacher_targets(
    *,
    anchor_path: str | Path,
    expansion_path: str | Path,
    anchor_keys: set[tuple[str, str]],
    expansion_keys: set[tuple[str, str]],
    base_checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]:
    anchor, anchor_meta = load_teacher_artifact(anchor_path)
    expansion, expansion_meta = load_expansion_teachers(expansion_path)
    roles = {
        (str(row["scene_id"]), str(row["question_id"])): str(row["role"])
        for row in anchor_meta["records"]
    }
    if (
        {key for key, role in roles.items() if role == "changed_teacher"} != anchor_keys
        or set(expansion) != expansion_keys
        or anchor_meta["base_checkpoint_sha256"] != base_checkpoint_sha256
        or expansion_meta["base_checkpoint_sha256"] != base_checkpoint_sha256
        or anchor_meta["base_runtime_config_sha256"] != runtime_config_sha256
        or expansion_meta["base_runtime_config_sha256"] != runtime_config_sha256
    ):
        raise ValueError("V60 numeric teacher provenance changed")
    targets = {key: anchor[key].float() for key in anchor_keys}
    targets.update({key: expansion[key].float() for key in expansion_keys})
    return targets, {
        "anchor_metadata_sha256": _sha256_file(_resolve(anchor_path) / "metadata.json"),
        "expansion_metadata_sha256": _sha256_file(
            _resolve(expansion_path) / "metadata.json"
        ),
        "selection_sha256": _sha256_json([list(key) for key in sorted(targets)]),
    }


def _basis_targets(
    targets: dict[tuple[str, str], torch.Tensor], basis: torch.Tensor
) -> tuple[
    dict[tuple[str, str], torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
    dict[str, float],
]:
    coefficients = {}
    rms_values = {}
    cosines = []
    for key, target in targets.items():
        rms = target.square().mean(dim=-1).sqrt()
        directions = target / rms.unsqueeze(-1).clamp_min(1e-8)
        raw_coefficients = torch.einsum("bch,rh->bcr", directions, basis)
        coefficient_direction = F.normalize(raw_coefficients, dim=-1, eps=1e-8)
        reconstructed = torch.einsum("bcr,rh->bch", coefficient_direction, basis)
        cosine = F.cosine_similarity(directions, reconstructed, dim=-1)
        coefficients[key] = coefficient_direction
        rms_values[key] = rms
        cosines.extend(cosine.flatten().tolist())
    return coefficients, rms_values, {
        "mean_cosine": sum(cosines) / len(cosines),
        "minimum_cosine": min(cosines),
    }


def _batch_loss(
    control: TeacherBasisFullSceneQuestionControlV3,
    examples: Sequence[V3Example],
    coefficient_targets: dict[tuple[str, str], torch.Tensor],
    rms_targets: dict[tuple[str, str], torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    signatures = torch.cat([example.signature for example in examples])
    questions = torch.cat([example.question for example in examples])
    output = control.forward_from_signature(signatures, questions)
    labels = torch.tensor(
        [example.route_label for example in examples],
        device=output.gate_logits.device,
        dtype=output.gate_logits.dtype,
    )
    route_loss = F.binary_cross_entropy_with_logits(output.gate_logits, labels)
    changed = [index for index, example in enumerate(examples) if example.target is not None]
    if changed:
        target_coefficients = torch.cat(
            [coefficient_targets[examples[index].key] for index in changed]
        ).to(output.control_tokens)
        target_rms = torch.cat(
            [rms_targets[examples[index].key] for index in changed]
        ).to(output.control_tokens)
        target_prompts = torch.cat(
            [examples[index].target for index in changed if examples[index].target is not None]
        ).to(output.control_tokens)
        predicted_coefficients = output.coefficient_directions[changed]
        predicted_rms = output.control_rms[changed]
        coefficient_loss = (
            1.0
            - F.cosine_similarity(
                predicted_coefficients, target_coefficients, dim=-1
            ).mean()
        )
        log_rms_loss = F.mse_loss(
            predicted_rms.clamp_min(1e-6).log(), target_rms.clamp_min(1e-6).log()
        )
        reconstruction_loss = (
            1.0
            - F.cosine_similarity(
                output.control_tokens[changed], target_prompts, dim=-1
            ).mean()
        )
    else:
        zero = output.control_tokens.sum() * 0.0
        coefficient_loss = log_rms_loss = reconstruction_loss = zero
    total = (
        args.coefficient_weight * coefficient_loss
        + args.log_rms_weight * log_rms_loss
        + args.reconstruction_weight * reconstruction_loss
        + args.route_weight * route_loss
    )
    return total, {
        "coefficient_loss": float(coefficient_loss.detach().cpu()),
        "log_rms_loss": float(log_rms_loss.detach().cpu()),
        "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
        "route_loss": float(route_loss.detach().cpu()),
    }


def _pair_loss(
    control: TeacherBasisFullSceneQuestionControlV3,
    pair: tuple[V3Example, V3Example],
    coefficient_targets: dict[tuple[str, str], torch.Tensor],
) -> tuple[torch.Tensor, float]:
    output = control.forward_from_signature(
        torch.cat([pair[0].signature, pair[1].signature]),
        torch.cat([pair[0].question, pair[1].question]),
    )
    target = torch.cat(
        [coefficient_targets[pair[0].key], coefficient_targets[pair[1].key]]
    ).to(output.coefficient_directions)
    predicted_delta = output.coefficient_directions[0] - output.coefficient_directions[1]
    target_delta = target[0] - target[1]
    cosine = F.cosine_similarity(
        predicted_delta.flatten()[None], target_delta.flatten()[None], dim=-1
    ).squeeze()
    scale = target_delta.square().mean().clamp_min(1e-6)
    mse = (predicted_delta - target_delta).square().mean() / scale
    return mse + (1.0 - cosine), float(cosine.detach().cpu())


def _prompt_cosines(
    predicted: torch.Tensor, target: torch.Tensor
) -> list[float]:
    """Compare prompts after moving cached CPU teachers to prediction device."""

    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("V60 prompt cosine tensors must share [B,C,H]")
    values = F.cosine_similarity(predicted, target.to(predicted), dim=-1)
    return values.detach().float().cpu().flatten().tolist()


def train_v60(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    output_checkpoint = _safe_output_path(args.output_checkpoint, "V60 checkpoint")
    output_report = _safe_output_path(args.training_report, "V60 report")
    config, config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, base_files = checkpoint_fingerprint(base_checkpoint)
    records, qa_sha256 = load_training_records(args.train_qa, scene_ids=LOCKED_SCENE_IDS)
    records = sorted(records, key=lambda record: (record.scene_id, record.question_id))
    anchor_keys, expansion_keys = _changed_keys(records)
    changed_keys = anchor_keys | expansion_keys
    targets, teacher_provenance = _teacher_targets(
        anchor_path=args.anchor_teacher_artifact,
        expansion_path=args.expansion_teacher_cache,
        anchor_keys=anchor_keys,
        expansion_keys=expansion_keys,
        base_checkpoint_sha256=base_checkpoint_sha256,
        runtime_config_sha256=runtime_config_sha256,
    )
    target_stack = torch.cat([targets[key] for key in sorted(targets)])
    basis = teacher_output_basis(target_stack, rank=args.basis_rank)
    coefficient_targets, rms_targets, reconstruction = _basis_targets(targets, basis)
    if reconstruction["mean_cosine"] < 0.995 or reconstruction["minimum_cosine"] < 0.96:
        raise RuntimeError(f"V60 basis reconstruction gate failed: {reconstruction}")

    prefixes, prefix_manifest = load_prefix_cache(
        args.prefix_cache,
        scene_ids=LOCKED_SCENE_IDS,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    runtime = StaticRuntimePrefixFactory(
        config, base_checkpoint, LOCKED_SCENE_IDS[0]
    ).bootstrap
    device = _select_training_device(runtime, args.device)
    frozen = freeze_base_runtime(runtime)
    model_dtype = next(runtime.language.model.parameters()).dtype
    prefixes = {
        key: value.to(device=device, dtype=model_dtype) for key, value in prefixes.items()
    }
    questions = {}
    with torch.inference_mode():
        for record in records:
            questions[(record.scene_id, record.question_id)] = _pooled_question_embedding(
                runtime, record.question
            )
    control = TeacherBasisFullSceneQuestionControlV3(
        runtime.language.hidden_size,
        basis,
        control_tokens=int(target_stack.shape[1]),
        expected_environment_latents=int(config["scene_encoder"]["global_latents"]),
        moment_count=args.moment_count,
        interaction_dim=args.interaction_dim,
        trunk_dim=args.trunk_dim,
        maximum_control_rms=args.maximum_control_rms,
        initial_control_rms=args.initial_control_rms,
        gate_threshold=args.gate_threshold,
    ).to(device=device, dtype=torch.float32)
    if control.trainable_parameter_count >= 1_000_000:
        raise ValueError("V60 trainable parameter count exceeds one million")
    signatures = {
        key: control.encode_scene(value.float()) for key, value in prefixes.items()
    }
    examples = [
        V3Example(
            record.scene_id,
            record.question_id,
            signatures[record.scene_id],
            questions[(record.scene_id, record.question_id)],
            targets.get((record.scene_id, record.question_id), None),
            float((record.scene_id, record.question_id) in changed_keys),
            (
                "anchor_changed"
                if (record.scene_id, record.question_id) in anchor_keys
                else "expansion_changed"
                if (record.scene_id, record.question_id) in expansion_keys
                else "retention"
            ),
        )
        for record in records
    ]
    # Initialize semantic task-route prototypes from normalized frozen question
    # embeddings, then keep them learnable under BCE for paraphrase-friendly
    # separation.
    positive = torch.cat(
        [control.normalized_question(example.question) for example in examples if example.route_label]
    )
    negative = torch.cat(
        [control.normalized_question(example.question) for example in examples if not example.route_label]
    )
    control.initialize_route_prototypes(positive, negative)
    examples_by_key = {example.key: example for example in examples}
    pair_examples = [
        (
            examples_by_key[(pair[0].scene_id, pair[0].question_id)],
            examples_by_key[(pair[1].scene_id, pair[1].question_id)],
        )
        for pair in _changed_pair_units(records)
    ]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in control.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    started = time.perf_counter()
    gradients = []
    steps = []
    ordinal = 0
    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch * 1_000_003)
        expanded = [
            example
            for example in examples
            for _ in range(
                args.changed_repeats if example.route_label else args.retention_repeats
            )
        ]
        rng.shuffle(expanded)
        batches = [
            expanded[offset : offset + args.batch_size]
            for offset in range(0, len(expanded), args.batch_size)
        ]
        for kind, work in (("batch", batches), ("pair", pair_examples)):
            for batch in work:
                optimizer.zero_grad(set_to_none=True)
                if kind == "batch":
                    loss, diagnostics = _batch_loss(
                        control, batch, coefficient_targets, rms_targets, args
                    )
                else:
                    pair_loss, cosine = _pair_loss(
                        control, batch, coefficient_targets
                    )
                    loss = args.pair_delta_weight * pair_loss
                    diagnostics = {"pair_delta_cosine": cosine}
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    control.parameters(), args.gradient_clip_norm
                )
                gradient_value = float(gradient.detach().cpu())
                if not math.isfinite(gradient_value):
                    raise RuntimeError("V60 gradient is nonfinite")
                optimizer.step()
                ordinal += 1
                gradients.append(gradient_value)
                steps.append({
                    "epoch": epoch,
                    "ordinal": ordinal,
                    "kind": kind,
                    "loss": float(loss.detach().cpu()),
                    **diagnostics,
                })
                if ordinal % args.log_every == 0:
                    _log_event(phase="v60_basis_distillation", **steps[-1])
    control.eval()
    elapsed = time.perf_counter() - started

    route_correct = defaultdict(list)
    prompt_cosines = defaultdict(list)
    rms_errors = defaultdict(list)
    with torch.inference_mode():
        for example in examples:
            output = control.forward_from_signature(example.signature, example.question)
            route_correct[example.group].append(
                (float(output.gate_probabilities.item()) >= args.gate_threshold)
                == bool(example.route_label)
            )
            if example.target is not None:
                prompt_cosines[example.group].extend(
                    _prompt_cosines(output.control_tokens, example.target)
                )
                rms_errors[example.group].extend(
                    (output.control_rms - rms_targets[example.key].to(output.control_rms))
                    .abs()
                    .flatten()
                    .tolist()
                )
        pair_cosines = [
            _pair_loss(control, pair, coefficient_targets)[1] for pair in pair_examples
        ]
    all_prompt_cosines = [value for values in prompt_cosines.values() for value in values]
    all_rms_errors = [value for values in rms_errors.values() for value in values]
    offline_checks = {
        "basis_mean_cosine": reconstruction["mean_cosine"] >= 0.995,
        "basis_minimum_cosine": reconstruction["minimum_cosine"] >= 0.96,
        "all_training_routes_exact": all(all(values) for values in route_correct.values()),
        "mean_prompt_cosine": sum(all_prompt_cosines) / len(all_prompt_cosines) >= 0.95,
        "minimum_prompt_cosine": min(all_prompt_cosines) >= 0.8,
        "mean_rms_absolute_error": sum(all_rms_errors) / len(all_rms_errors) <= 0.015,
        "mean_pair_delta_cosine": sum(pair_cosines) / len(pair_cosines) >= 0.75,
    }
    checkpoint_hashes = save_v3_control_checkpoint(
        output_checkpoint,
        control=control,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    mean = lambda values: sum(values) / len(values)
    report = {
        "schema_version": 1,
        "artifact": "v60_teacher_basis_control_training",
        "offline_checks_passed": all(offline_checks.values()),
        "offline_checks": offline_checks,
        "promotion_eligible": False,
        "saved_runtime_gate_required": True,
        "base": {
            "checkpoint_sha256": base_checkpoint_sha256,
            "checkpoint_files": base_files,
            "runtime_config_effective_sha256": runtime_config_sha256,
            "runtime_config_file_sha256": _sha256_file(config_path),
        },
        "inputs": {
            "training_qa_sha256": qa_sha256,
            "training_scene_ids": list(LOCKED_SCENE_IDS),
            "training_record_count": len(records),
            "changed_record_count": len(changed_keys),
            "prefix_cache_manifest_sha256": _sha256_file(
                _resolve(args.prefix_cache) / "manifest.json"
            ),
            "prefix_shape": prefix_manifest["scenes"][LOCKED_SCENE_IDS[0]]["shape"],
            "teacher_provenance": teacher_provenance,
        },
        "architecture": {
            "name": "teacher_basis_full_scene_question_control_v3",
            "trainable_parameter_count": control.trainable_parameter_count,
            "saved_parameter_count": control.parameter_count,
            "basis_rank": control.output_basis_rank,
            "scene_signature_shape": list(signatures[LOCKED_SCENE_IDS[0]].shape),
            "boundary_tokens_excluded": True,
            "softmax_scene_attention_used": False,
            "control_values_scene_question_bilinear": True,
        },
        "basis_reconstruction": reconstruction,
        "optimization": {
            "epochs": args.epochs,
            "optimizer_steps": ordinal,
            "elapsed_seconds": elapsed,
            "maximum_preclip_gradient_norm": max(gradients),
            "mean_prompt_cosine": mean(all_prompt_cosines),
            "minimum_prompt_cosine": min(all_prompt_cosines),
            "mean_rms_absolute_error": mean(all_rms_errors),
            "mean_pair_delta_cosine": mean(pair_cosines),
            "route_accuracy": {
                key: mean([float(value) for value in values])
                for key, values in sorted(route_correct.items())
            },
        },
        "checkpoint": checkpoint_hashes,
        "scope": {
            "base_scene_stack_frozen": frozen["all_parameters_frozen"],
            "gemma_backward_used": False,
            "cached_numeric_teachers_only": True,
            "runtime_teacher_access": False,
            "question_dependent_scene_retrieval": False,
            "complete_scene_prefix_retained": True,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
        },
    }
    _write_training_report(output_report, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--anchor-teacher-artifact", required=True)
    parser.add_argument("--expansion-teacher-cache", required=True)
    parser.add_argument("--train-qa", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=60060)
    parser.add_argument("--basis-rank", type=int, default=80)
    parser.add_argument("--moment-count", type=int, default=8)
    parser.add_argument("--interaction-dim", type=int, default=24)
    parser.add_argument("--trunk-dim", type=int, default=128)
    parser.add_argument("--maximum-control-rms", type=float, default=0.2)
    parser.add_argument("--initial-control-rms", type=float, default=0.075)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--changed-repeats", type=int, default=4)
    parser.add_argument("--retention-repeats", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--coefficient-weight", type=float, default=4.0)
    parser.add_argument("--log-rms-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--pair-delta-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=400)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = train_v60(args)
    print(
        json.dumps(
            {
                "offline_checks_passed": report["offline_checks_passed"],
                "promotion_eligible": False,
                "checkpoint": str(_resolve(args.output_checkpoint)),
            },
            sort_keys=True,
        )
    )
    return 0 if report["offline_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "train_v60"]
