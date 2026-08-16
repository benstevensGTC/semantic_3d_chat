"""Train a delta-sensitive V57 objective on the V1 question controller.

The runtime surface is intentionally identical to V56: a checkpoint contains
only ``FullSceneQuestionControl`` V1 weights plus its strict runtime metadata.
V57 changes the supervised objective, not inference.  Changed counterfactual
pairs receive explicit within-prefix, cross-prefix, control-delta, attention,
and frozen answer-embedding losses; broad/count replay retains answer-only CE.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl
from semantic_3d_chat.training.question_control_pair_objective_v57 import (
    V57PairObjectiveSettings,
    paired_question_control_objective,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    _log_event,
    _resolve,
    _safe_output_path,
    _sha256_file,
    _write_training_report,
    build_curriculum,
    curriculum_summary,
    ensure_prefix_cache,
    freeze_base_runtime,
    load_training_records,
    question_control_answer_loss,
    save_control_checkpoint,
    validate_training_scene_ids,
)


def _pair_settings(args: argparse.Namespace) -> V57PairObjectiveSettings:
    return V57PairObjectiveSettings(
        answer_nll_weight=args.answer_nll_weight,
        side_hinge_weight=args.side_hinge_weight,
        side_margin=args.side_margin,
        cross_prefix_hinge_weight=args.cross_prefix_hinge_weight,
        cross_prefix_margin=args.cross_prefix_margin,
        control_delta_weight=args.control_delta_weight,
        minimum_relative_control_delta=args.minimum_relative_control_delta,
        attention_entropy_weight=args.attention_entropy_weight,
        minimum_normalized_attention_entropy=(
            args.minimum_normalized_attention_entropy
        ),
        attention_logit_spread_weight=args.attention_logit_spread_weight,
        maximum_attention_logit_rms=args.maximum_attention_logit_rms,
        answer_alignment_weight=args.answer_alignment_weight,
        answer_alignment_margin=args.answer_alignment_margin,
        answer_absolute_alignment_weight=args.answer_absolute_alignment_weight,
        answer_delta_alignment_weight=args.answer_delta_alignment_weight,
    )


def _conditioning_pair_settings(
    args: argparse.Namespace,
    base: V57PairObjectiveSettings,
) -> V57PairObjectiveSettings:
    """Stage 1 learns a scene-sensitive control surface before LM ranking."""

    return replace(
        base,
        answer_nll_weight=args.conditioning_answer_nll_weight,
        side_hinge_weight=0.0,
        cross_prefix_hinge_weight=0.0,
        control_delta_weight=args.conditioning_control_delta_weight,
        attention_entropy_weight=args.conditioning_attention_entropy_weight,
        attention_logit_spread_weight=(
            args.conditioning_attention_logit_spread_weight
        ),
        answer_alignment_weight=args.conditioning_answer_alignment_weight,
        answer_absolute_alignment_weight=(
            args.conditioning_answer_absolute_alignment_weight
        ),
        answer_delta_alignment_weight=(
            args.conditioning_answer_delta_alignment_weight
        ),
    )


def _validate_cli_numbers(args: argparse.Namespace) -> None:
    if isinstance(args.seed, bool) or args.seed < 0:
        raise ValueError("V57 seed must be nonnegative")
    positive_ints = (
        "epochs",
        "changed_pair_repeats",
        "count_replay_repeats",
        "broad_repeats",
        "replay_batch_size",
        "attention_dim",
        "control_tokens",
        "log_every",
    )
    if any(
        isinstance(getattr(args, field), bool) or getattr(args, field) < 1
        for field in positive_ints
    ):
        raise ValueError("V57 integer hyperparameters must be positive")
    positive_floats = (
        "uniform_floor",
        "output_scale",
        "learning_rate",
        "gradient_clip_norm",
    )
    if any(
        not math.isfinite(float(getattr(args, field)))
        or float(getattr(args, field)) <= 0.0
        for field in positive_floats
    ):
        raise ValueError("V57 positive hyperparameters must be finite")
    if args.uniform_floor > 1.0:
        raise ValueError("V57 uniform_floor must not exceed one")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("V57 weight_decay must be finite and nonnegative")
    if (
        isinstance(args.conditioning_epochs, bool)
        or args.conditioning_epochs < 0
        or args.conditioning_epochs > args.epochs
    ):
        raise ValueError("V57 conditioning_epochs must be in [0, epochs]")
    nonnegative_floats = (
        "conditioning_replay_weight",
        "conditioning_answer_nll_weight",
        "conditioning_control_delta_weight",
        "conditioning_attention_entropy_weight",
        "conditioning_attention_logit_spread_weight",
        "conditioning_answer_alignment_weight",
        "conditioning_answer_absolute_alignment_weight",
        "conditioning_answer_delta_alignment_weight",
    )
    if any(
        not math.isfinite(float(getattr(args, field)))
        or float(getattr(args, field)) < 0.0
        for field in nonnegative_floats
    ):
        raise ValueError("V57 conditioning weights must be finite and nonnegative")
    _pair_settings(args)


def _initialize_control(
    *,
    args: argparse.Namespace,
    runtime: Any,
    device: torch.device,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
) -> tuple[FullSceneQuestionControl, dict[str, Any] | None]:
    if args.initial_control_checkpoint is None:
        torch.manual_seed(args.seed)
        return (
            FullSceneQuestionControl(
                runtime.language.hidden_size,
                attention_dim=args.attention_dim,
                control_tokens=args.control_tokens,
                uniform_floor=args.uniform_floor,
                output_scale=args.output_scale,
            ).to(device=device, dtype=torch.float32),
            None,
        )
    control, metadata = _load_control_head(
        args.initial_control_checkpoint,
        hidden_size=runtime.language.hidden_size,
        device=device,
    )
    expected_architecture = {
        "attention_dim": args.attention_dim,
        "control_tokens": args.control_tokens,
        "uniform_floor": args.uniform_floor,
        "output_scale": args.output_scale,
    }
    observed_architecture = {
        field: metadata[field] for field in expected_architecture
    }
    if observed_architecture != expected_architecture:
        raise ValueError(
            "V57 warm-start architecture disagrees with explicit CLI settings: "
            f"expected={expected_architecture} observed={observed_architecture}"
        )
    if (
        metadata["base_checkpoint_sha256"] != base_checkpoint_sha256
        or metadata["base_runtime_config_sha256"] != base_runtime_config_sha256
    ):
        raise ValueError("V57 warm start belongs to a different frozen base runtime")
    return control, {
        "checkpoint_sha256": _control_checkpoint_sha256(
            _resolve(args.initial_control_checkpoint)
        ),
        "weights_sha256": metadata["weights_sha256"],
    }


def _tensor_mean(value: torch.Tensor | bool) -> float | bool:
    if isinstance(value, bool):
        return value
    return float(value.detach().float().mean().cpu())


def _pair_diagnostic_row(
    *,
    epoch: int,
    ordinal: int,
    diagnostics: Mapping[str, torch.Tensor | bool],
) -> dict[str, float | int | bool]:
    return {
        "epoch": epoch,
        "ordinal": ordinal,
        "correct_answer_nll": _tensor_mean(diagnostics["correct_answer_nll"]),
        "side_hinge": _tensor_mean(diagnostics["side_hinge"]),
        "minimum_side_margin": float(
            diagnostics["side_margins"].detach().float().min().cpu()  # type: ignore[union-attr]
        ),
        "cross_prefix_hinge": _tensor_mean(diagnostics["cross_prefix_hinge"]),
        "minimum_cross_prefix_margin": float(
            diagnostics["cross_prefix_margins"].detach().float().min().cpu()  # type: ignore[union-attr]
        ),
        "control_delta_hinge": _tensor_mean(diagnostics["control_delta_hinge"]),
        "relative_control_delta": _tensor_mean(
            diagnostics["relative_control_delta"]
        ),
        "attention_entropy_hinge": _tensor_mean(
            diagnostics["attention_entropy_hinge"]
        ),
        "minimum_normalized_attention_entropy": float(
            diagnostics["normalized_attention_entropy"]
            .detach()  # type: ignore[union-attr]
            .float()
            .min()
            .cpu()
        ),
        "answer_alignment_hinge": _tensor_mean(
            diagnostics["answer_alignment_hinge"]
        ),
        "answer_absolute_alignment_loss": _tensor_mean(
            diagnostics["answer_absolute_alignment_loss"]
        ),
        "minimum_own_answer_similarity": float(
            diagnostics["own_answer_similarities"]
            .detach()  # type: ignore[union-attr]
            .float()
            .min()
            .cpu()
        ),
        "minimum_answer_alignment_margin": float(
            diagnostics["answer_alignment_margins"]
            .detach()  # type: ignore[union-attr]
            .float()
            .min()
            .cpu()
        ),
        "attention_logit_spread_penalty": _tensor_mean(
            diagnostics["attention_logit_spread_penalty"]
        ),
        "maximum_attention_logit_rms": float(
            diagnostics["attention_logit_rms"]
            .detach()  # type: ignore[union-attr]
            .float()
            .max()
            .cpu()
        ),
        "answer_delta_alignment_loss": _tensor_mean(
            diagnostics["answer_delta_alignment_loss"]
        ),
        "answer_delta_alignment_cosine": _tensor_mean(
            diagnostics["answer_delta_alignment_cosine"]
        ),
        "single_forward_candidate_scoring": bool(
            diagnostics["single_forward_candidate_scoring"]
        ),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("V57 cannot summarize an empty metric")
    return sum(values) / len(values)


def _optimization_summary(
    step_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_epoch: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        by_epoch[int(row["epoch"])].append(row)
    return {
        "epoch_loss": [
            {
                "epoch": epoch,
                "steps": len(rows),
                "mean_total_loss": _mean([float(row["total_loss"]) for row in rows]),
                "mean_loss_by_kind": {
                    kind: _mean(
                        [
                            float(row["total_loss"])
                            for row in rows
                            if row["kind"] == kind
                        ]
                    )
                    for kind in sorted({str(row["kind"]) for row in rows})
                },
            }
            for epoch, rows in sorted(by_epoch.items())
        ],
        "final_pair_diagnostics": (
            None
            if not pair_rows
            else {
                field: float(pair_rows[-1][field])
                for field in (
                    "correct_answer_nll",
                    "minimum_side_margin",
                    "minimum_cross_prefix_margin",
                    "relative_control_delta",
                    "minimum_normalized_attention_entropy",
                    "minimum_answer_alignment_margin",
                    "minimum_own_answer_similarity",
                    "maximum_attention_logit_rms",
                    "answer_delta_alignment_cosine",
                )
            }
        ),
        "pair_candidate_scoring_steps": dict(
            sorted(
                Counter(
                    "single_forward"
                    if bool(row["single_forward_candidate_scoring"])
                    else "full_sequence_fallback"
                    for row in pair_rows
                ).items()
            )
        ),
    }


def train_question_control_v57(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one deterministic, no-resume V57 training run."""

    _validate_cli_numbers(args)
    scene_ids = validate_training_scene_ids(args.scene_id)
    raw_checkpoint_output = _resolve(args.output_checkpoint)
    raw_report_output = _resolve(args.training_report)
    if raw_checkpoint_output == raw_report_output or raw_report_output.is_relative_to(
        raw_checkpoint_output
    ):
        raise ValueError("V57 training report must remain outside the runtime checkpoint")
    checkpoint_output = _safe_output_path(raw_checkpoint_output, "V57 control checkpoint")
    report_output = _safe_output_path(raw_report_output, "V57 training report")

    config, config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, base_checkpoint_files = checkpoint_fingerprint(base_checkpoint)
    records, qa_sha256 = load_training_records(args.train_qa, scene_ids=scene_ids)
    schedule = build_curriculum(
        records,
        epochs=args.epochs,
        seed=args.seed,
        changed_pair_repeats=args.changed_pair_repeats,
        count_replay_repeats=args.count_replay_repeats,
        broad_repeats=args.broad_repeats,
        replay_batch_size=args.replay_batch_size,
    )
    settings = _pair_settings(args)
    conditioning_settings = _conditioning_pair_settings(args, settings)
    _log_event(
        phase="v57_preflight_complete",
        scene_count=len(scene_ids),
        training_record_count=len(records),
        optimizer_step_count=len(schedule),
        changed_pair_step_count=sum(
            step.kind == "changed_pair" for step in schedule
        ),
        base_checkpoint_sha256=base_checkpoint_sha256,
        runtime_config_sha256=runtime_config_sha256,
    )

    _log_event(phase="v57_base_runtime_load", scene_id=scene_ids[0])
    runtime_factory = StaticRuntimePrefixFactory(config, base_checkpoint, scene_ids[0])
    cache_loads = 0

    def cache_runtime_loader(scene_id: str) -> StaticChatRuntime:
        nonlocal cache_loads
        cache_loads += 1
        _log_event(
            phase="v57_prefix_cache_build",
            scene_id=scene_id,
            scene_ordinal=cache_loads,
            scene_count=len(scene_ids),
        )
        return runtime_factory.load(scene_id)

    cache = ensure_prefix_cache(
        args.prefix_cache,
        scene_ids=scene_ids,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
        runtime_loader=cache_runtime_loader,
    )
    runtime = runtime_factory.bootstrap
    runtime.assert_prefix_unchanged()
    if not torch.equal(
        runtime.scene_prefix.detach().cpu(), cache.prefixes[scene_ids[0]]
    ):
        raise ValueError("V57 prefix cache does not exactly match StaticChatRuntime")
    from semantic_3d_chat.training.train_question_control_v56 import (  # local to keep API narrow
        _select_training_device,
    )

    device = _select_training_device(runtime, args.device)
    frozen_audit = freeze_base_runtime(runtime)
    runtime.language.enable_decoder_gradient_checkpointing()
    model_dtype = next(runtime.language.model.parameters()).dtype
    training_prefixes = {
        scene_id: prefix.to(device=device, dtype=model_dtype)
        for scene_id, prefix in cache.prefixes.items()
    }
    control, warm_start = _initialize_control(
        args=args,
        runtime=runtime,
        device=device,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    if control.parameter_count < 1 or any(
        not parameter.requires_grad for parameter in control.parameters()
    ):
        raise RuntimeError("V57 question controller is not the trainable surface")
    optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    started = time.perf_counter()
    step_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    gradient_norms: list[float] = []
    control.train()
    for step in schedule:
        optimizer.zero_grad(set_to_none=True)
        if step.kind == "changed_pair":
            active_settings = (
                conditioning_settings
                if step.epoch < args.conditioning_epochs
                else settings
            )
            loss, diagnostics = paired_question_control_objective(
                runtime=runtime,
                control=control,
                prefixes=training_prefixes,
                records=(step.records[0], step.records[1]),
                settings=active_settings,
            )
            pair_row = _pair_diagnostic_row(
                epoch=step.epoch,
                ordinal=step.ordinal,
                diagnostics=diagnostics,
            )
            pair_rows.append(pair_row)
        else:
            loss = question_control_answer_loss(
                runtime=runtime,
                control=control,
                prefixes=training_prefixes,
                records=step.records,
            )
            if step.epoch < args.conditioning_epochs:
                loss = loss * args.conditioning_replay_weight
            pair_row = None
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError("V57 optimizer loss is nonfinite or nonscalar")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            control.parameters(), args.gradient_clip_norm
        )
        gradient_value = float(gradient_norm.detach().float().cpu())
        if not math.isfinite(gradient_value):
            raise RuntimeError("V57 controller gradient norm is nonfinite")
        optimizer.step()
        if any(not torch.isfinite(value).all() for value in control.state_dict().values()):
            raise RuntimeError("V57 optimizer produced nonfinite controller state")
        loss_value = float(loss.detach().cpu())
        step_rows.append(
            {
                "epoch": step.epoch,
                "ordinal": step.ordinal,
                "kind": step.kind,
                "stage": (
                    "conditioning"
                    if step.epoch < args.conditioning_epochs
                    else "answer_ranking"
                ),
                "total_loss": loss_value,
                "preclip_gradient_norm": gradient_value,
            }
        )
        gradient_norms.append(gradient_value)
        completed_steps = step.ordinal + 1
        if completed_steps % args.log_every == 0 or completed_steps == len(schedule):
            event: dict[str, Any] = {
                "phase": "v57_training",
                "completed_steps": completed_steps,
                "optimizer_step_count": len(schedule),
                "epoch": step.epoch,
                "curriculum_kind": step.kind,
                "training_stage": step_rows[-1]["stage"],
                "total_loss": loss_value,
                "preclip_gradient_norm": gradient_value,
            }
            if pair_row is not None:
                event.update(
                    {
                        "minimum_side_margin": pair_row["minimum_side_margin"],
                        "minimum_cross_prefix_margin": pair_row[
                            "minimum_cross_prefix_margin"
                        ],
                        "relative_control_delta": pair_row[
                            "relative_control_delta"
                        ],
                        "minimum_attention_entropy": pair_row[
                            "minimum_normalized_attention_entropy"
                        ],
                    }
                )
            _log_event(**event)
    control.eval()

    checkpoint_hashes = save_control_checkpoint(
        checkpoint_output,
        control=control,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    cache_manifest_path = _resolve(args.prefix_cache) / "manifest.json"
    summaries = _optimization_summary(step_rows, pair_rows)
    report = {
        "schema_version": 1,
        "artifact": "v57_question_control_pair_margin_training",
        "passed": True,
        "base": {
            "checkpoint_sha256": base_checkpoint_sha256,
            "checkpoint_files": base_checkpoint_files,
            "runtime_config_effective_sha256": runtime_config_sha256,
            "runtime_config_file_sha256": _sha256_file(config_path),
            "warm_start": warm_start,
        },
        "inputs": {
            "training_qa_sha256": qa_sha256,
            "training_record_count": len(records),
            "training_scene_ids": list(scene_ids),
            "prefix_cache_manifest_sha256": _sha256_file(cache_manifest_path),
            "prefix_sha256_by_scene": {
                scene_id: cache.manifest["scenes"][scene_id]["prefix_sha256"]
                for scene_id in scene_ids
            },
            "prefix_cache_created": cache.created,
        },
        "curriculum": curriculum_summary(schedule),
        "architecture": {
            "name": "full_scene_question_control_v1",
            "runtime_compatible_with_v56": True,
            "hidden_size": control.hidden_size,
            "attention_dim": control.attention_dim,
            "control_tokens": control.control_token_count,
            "uniform_floor": control.uniform_floor,
            "output_scale": control.output_scale,
            "parameter_count": control.parameter_count,
        },
        "objective": {
            "name": "delta_sensitive_counterfactual_pair_v57",
            "conditioning_epochs": args.conditioning_epochs,
            "conditioning_replay_weight": args.conditioning_replay_weight,
            "conditioning_pair": conditioning_settings.contract(),
            "answer_ranking_pair": settings.contract(),
            "same_question_two_scene_candidate_ranking": True,
            "true_cross_prefix_answer_scores": True,
            "full_sequence_fallback_for_unaligned_answers": True,
            "answer_embedding_alignment_is_train_only": True,
            "pre_softmax_attention_collapse_guard": True,
            "explicit_control_delta_to_answer_delta_alignment": True,
        },
        "optimization": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "optimizer_steps": len(schedule),
            "device": device.type,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_preclip_gradient_norm": max(gradient_norms),
            **summaries,
        },
        "checkpoint": checkpoint_hashes,
        "scope": {
            "base_scene_stack_frozen": frozen_audit["all_parameters_frozen"],
            "base_parameter_count": frozen_audit["parameter_count"],
            "only_control_head_optimized": True,
            "language_ce_labels_are_answer_only": True,
            "paired_two_side_optimizer_steps": True,
            "complete_scene_prefix_retained": True,
            "question_inputs_to_scene_prefix_cache": False,
            "question_dependent_scene_retrieval": False,
            "runtime_answer_or_label_inputs": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
            "optimizer_state_saved": False,
        },
    }
    _write_training_report(report_output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--train-qa", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--initial-control-checkpoint")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=57057)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--changed-pair-repeats", type=int, default=4)
    parser.add_argument("--count-replay-repeats", type=int, default=2)
    parser.add_argument("--broad-repeats", type=int, default=1)
    parser.add_argument("--replay-batch-size", type=int, default=2)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--control-tokens", type=int, default=4)
    parser.add_argument("--uniform-floor", type=float, default=0.05)
    parser.add_argument("--output-scale", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--answer-nll-weight", type=float, default=1.0)
    parser.add_argument("--side-hinge-weight", type=float, default=0.5)
    parser.add_argument("--side-margin", type=float, default=0.5)
    parser.add_argument("--cross-prefix-hinge-weight", type=float, default=1.0)
    parser.add_argument("--cross-prefix-margin", type=float, default=0.1)
    parser.add_argument("--control-delta-weight", type=float, default=8.0)
    parser.add_argument("--minimum-relative-control-delta", type=float, default=0.03)
    parser.add_argument("--attention-entropy-weight", type=float, default=2.0)
    parser.add_argument(
        "--minimum-normalized-attention-entropy", type=float, default=0.55
    )
    parser.add_argument("--attention-logit-spread-weight", type=float, default=1.0)
    parser.add_argument("--maximum-attention-logit-rms", type=float, default=1.0)
    parser.add_argument("--answer-alignment-weight", type=float, default=2.0)
    parser.add_argument("--answer-alignment-margin", type=float, default=0.1)
    parser.add_argument("--answer-absolute-alignment-weight", type=float, default=1.0)
    parser.add_argument("--answer-delta-alignment-weight", type=float, default=2.0)
    parser.add_argument("--conditioning-epochs", type=int, default=1)
    parser.add_argument("--conditioning-replay-weight", type=float, default=0.25)
    parser.add_argument("--conditioning-answer-nll-weight", type=float, default=0.25)
    parser.add_argument("--conditioning-control-delta-weight", type=float, default=12.0)
    parser.add_argument(
        "--conditioning-attention-entropy-weight", type=float, default=2.0
    )
    parser.add_argument(
        "--conditioning-attention-logit-spread-weight", type=float, default=2.0
    )
    parser.add_argument(
        "--conditioning-answer-alignment-weight", type=float, default=4.0
    )
    parser.add_argument(
        "--conditioning-answer-absolute-alignment-weight", type=float, default=2.0
    )
    parser.add_argument(
        "--conditioning-answer-delta-alignment-weight", type=float, default=4.0
    )
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_cli_numbers(args)
    report = train_question_control_v57(args)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "optimizer_steps": report["optimization"]["optimizer_steps"],
                "checkpoint": str(_resolve(args.output_checkpoint)),
                "training_report": str(_resolve(args.training_report)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "train_question_control_v57"]
