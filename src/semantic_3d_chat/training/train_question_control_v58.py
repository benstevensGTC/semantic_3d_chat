"""Learn Gemma-native soft-prompt teachers, then distill the V1 scene controller.

Stage A follows the proven causal recipe exactly: initialize one free
``[1,4,1536]`` prompt from the accepted V57 controller, optimize it through the
frozen Gemma decoder with answer-only NLL, Adam(lr=0.03), and unit gradient
clipping.  Stage B removes Gemma from the backward path and trains the shared
``FullSceneQuestionControl`` to reproduce those prompts.  Non-changed records
retain their original controller outputs as replay targets.

The final checkpoint remains the strict two-file V1 runtime checkpoint.  QA,
answers, and teacher artifacts are training-only and are never loaded by chat.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.evaluation.metrics import exact_normalized_match
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.local_lm import prompt_token_ids, question_token_ids
from semantic_3d_chat.language.prefix_injection import (
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl
from semantic_3d_chat.training.pair_curriculum import token_normalized_nll
from semantic_3d_chat.training.question_control_pair_objective_v57 import _compose_batch
from semantic_3d_chat.training.soft_prompt_teacher_v58 import (
    SoftPromptTarget,
    normalized_prompt_distillation_loss,
    pair_delta_distillation_loss,
    save_teacher_artifact,
)
from semantic_3d_chat.training.train_adapter import forward_prefix_batch
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
    ensure_prefix_cache,
    freeze_base_runtime,
    load_training_records,
    save_control_checkpoint,
    validate_training_scene_ids,
)


@dataclass(frozen=True)
class DistillationExample:
    scene_id: str
    question_id: str
    scene_prefix: torch.Tensor
    pooled_question: torch.Tensor
    target: torch.Tensor
    changed: bool

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.question_id


def _positive_int(args: argparse.Namespace, field: str) -> int:
    value = getattr(args, field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"V58 {field} must be a positive integer")
    return value


def _finite_nonnegative(args: argparse.Namespace, field: str) -> float:
    value = float(getattr(args, field))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"V58 {field} must be finite and nonnegative")
    return value


def _validate_args(args: argparse.Namespace) -> None:
    if isinstance(args.seed, bool) or args.seed < 0:
        raise ValueError("V58 seed must be nonnegative")
    for field in (
        "teacher_min_steps",
        "teacher_max_steps",
        "distill_epochs",
        "distill_batch_size",
        "changed_repeats",
        "log_every",
    ):
        _positive_int(args, field)
    if args.teacher_min_steps > args.teacher_max_steps:
        raise ValueError("V58 teacher_min_steps cannot exceed teacher_max_steps")
    for field in (
        "teacher_learning_rate",
        "teacher_gradient_clip_norm",
        "teacher_nll_threshold",
        "distill_learning_rate",
        "distill_weight_decay",
        "distill_gradient_clip_norm",
        "distill_mse_weight",
        "distill_cosine_weight",
        "pair_delta_weight",
        "pair_delta_mse_weight",
        "pair_delta_cosine_weight",
    ):
        _finite_nonnegative(args, field)
    for field in (
        "teacher_learning_rate",
        "teacher_gradient_clip_norm",
        "teacher_nll_threshold",
        "distill_learning_rate",
        "distill_gradient_clip_norm",
    ):
        if getattr(args, field) <= 0.0:
            raise ValueError(f"V58 {field} must be positive")
    if args.distill_mse_weight == 0.0 and args.distill_cosine_weight == 0.0:
        raise ValueError("V58 distillation enables no prompt loss")


def _disable_decoder_checkpointing(language: Any) -> None:
    decoder = language.decoder_module
    disable = getattr(decoder, "gradient_checkpointing_disable", None)
    if not callable(disable):
        raise TypeError("V58 decoder cannot disable gradient checkpointing")
    disable()
    decoder.eval()
    language.model.eval()
    for config in (getattr(language.model, "config", None), getattr(decoder, "config", None)):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = True
    language.decoder_gradient_checkpointing_enabled = False


def _pooled_question_embedding(runtime: Any, question: str) -> torch.Tensor:
    language = runtime.language
    ids = question_token_ids(language.tokenizer, question, language.device)
    with torch.no_grad():
        embeddings = language.model.get_input_embeddings()(ids).detach().float()
    return embeddings.mean(dim=1, keepdim=True)


def _teacher_nll(
    *,
    runtime: Any,
    scene_prefix: torch.Tensor,
    record: Any,
    free_prompt: torch.Tensor,
) -> torch.Tensor:
    batch, _answer_ids = _compose_batch(
        runtime=runtime,
        scene_prefix=scene_prefix,
        record=record,
        answer=record.answer,
        control_tokens=free_prompt,
    )
    output = forward_prefix_batch(runtime.language, batch)
    if batch.labels is None:
        raise RuntimeError("V58 teacher batch lacks answer labels")
    loss = token_normalized_nll(output.logits, batch.labels).mean()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("V58 teacher NLL is nonfinite or nonscalar")
    return loss


def _optimize_teacher_prompt(
    *,
    runtime: Any,
    scene_prefix: torch.Tensor,
    record: Any,
    initial_prompt: torch.Tensor,
    learning_rate: float,
    min_steps: int,
    max_steps: int,
    nll_threshold: float,
    gradient_clip_norm: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prompt = torch.nn.Parameter(initial_prompt.detach().float().clone())
    optimizer = torch.optim.Adam([prompt], lr=learning_rate)
    losses: list[float] = []
    gradients: list[float] = []
    best_prompt = initial_prompt.detach().float().clone()
    best_loss = math.inf
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _teacher_nll(
            runtime=runtime,
            scene_prefix=scene_prefix,
            record=record,
            free_prompt=prompt,
        )
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_prompt = prompt.detach().clone()
        if step + 1 >= min_steps and loss_value <= nll_threshold:
            break
        if (
            step >= 1
            and loss_value > losses[0] * 1.25
            and best_loss >= losses[0] * 0.99
        ):
            break
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_((prompt,), gradient_clip_norm)
        gradient_value = float(gradient.detach().float().cpu())
        if not math.isfinite(gradient_value):
            raise RuntimeError("V58 teacher prompt gradient is nonfinite")
        optimizer.step()
        if not torch.isfinite(prompt).all():
            raise RuntimeError("V58 teacher optimizer produced nonfinite prompt")
        gradients.append(gradient_value)
    return best_prompt, {
        "steps": len(losses),
        "initial_nll": losses[0],
        "final_nll": best_loss,
        "minimum_nll": best_loss,
        "maximum_preclip_gradient_norm": max(gradients, default=0.0),
        "initial_rms": float(initial_prompt.detach().float().square().mean().sqrt()),
        "final_rms": float(best_prompt.detach().float().square().mean().sqrt()),
        "learning_rate": learning_rate,
    }


def _optimize_teacher_prompt_adaptive(
    *,
    runtime: Any,
    scene_prefix: torch.Tensor,
    record: Any,
    initial_prompt: torch.Tensor,
    learning_rate: float,
    min_steps: int,
    max_steps: int,
    nll_threshold: float,
    gradient_clip_norm: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Keep the proven LR first, then back off deterministically on overshoot."""

    rates = tuple(dict.fromkeys((learning_rate, learning_rate / 3.0, learning_rate / 10.0, learning_rate / 30.0)))
    best_prompt: torch.Tensor | None = None
    best_metrics: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    for rate in rates:
        prompt, metrics = _optimize_teacher_prompt(
            runtime=runtime,
            scene_prefix=scene_prefix,
            record=record,
            initial_prompt=initial_prompt,
            learning_rate=rate,
            min_steps=min_steps,
            max_steps=max_steps,
            nll_threshold=nll_threshold,
            gradient_clip_norm=gradient_clip_norm,
        )
        attempts.append(dict(metrics))
        if best_metrics is None or metrics["final_nll"] < best_metrics["final_nll"]:
            best_prompt, best_metrics = prompt, metrics
        if metrics["final_nll"] <= nll_threshold:
            break
    if best_prompt is None or best_metrics is None:
        raise RuntimeError("V58 adaptive teacher optimization made no attempt")
    return best_prompt, {
        **best_metrics,
        "attempt_count": len(attempts),
        "attempt_learning_rates": [attempt["learning_rate"] for attempt in attempts],
        "total_forward_steps": sum(attempt["steps"] for attempt in attempts),
    }


@torch.inference_mode()
def _generate_with_control(
    *,
    runtime: Any,
    scene_prefix: torch.Tensor,
    question: str,
    control_tokens: torch.Tensor | None,
) -> str:
    language = runtime.language
    backend = language.prefix_backend
    if backend is None or language.backend_name != "gemma4":
        raise RuntimeError("V58 verification requires the Gemma prefix backend")
    prompt_ids = prompt_token_ids(
        language.tokenizer,
        str(runtime.config["language"]["system_prompt"]),
        question,
        language.device,
    )
    prepared = backend.prepare(
        scene_prefix,
        prompt_ids,
        scene_prefix_after_bos=scene_prefix_after_bos_setting(runtime.config),
        scene_boundary_mode=scene_boundary_mode_setting(runtime.config),
        control_tokens=(
            None if control_tokens is None else control_tokens.to(scene_prefix)
        ),
    )
    generated = backend.generate(
        prepared,
        max_new_tokens=int(runtime.config["language"]["max_answer_tokens"]),
        eos_token_ids=runtime._eos_token_ids(),
    )
    return language.tokenizer.decode(
        generated[0].detach().cpu().tolist(), skip_special_tokens=True
    ).strip() or "unknown"


def _distill_minibatch(
    control: FullSceneQuestionControl,
    examples: Sequence[DistillationExample],
    *,
    mse_weight: float,
    cosine_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scene = torch.cat(tuple(example.scene_prefix for example in examples), dim=0).float()
    question = torch.cat(tuple(example.pooled_question for example in examples), dim=0)
    target = torch.cat(tuple(example.target for example in examples), dim=0)
    predicted = control(scene, question)
    return normalized_prompt_distillation_loss(
        predicted,
        target,
        mse_weight=mse_weight,
        cosine_weight=cosine_weight,
    )


def _distill_pair_delta(
    control: FullSceneQuestionControl,
    examples: tuple[DistillationExample, DistillationExample],
    *,
    mse_weight: float,
    cosine_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scene = torch.cat((examples[0].scene_prefix, examples[1].scene_prefix), dim=0).float()
    question = torch.cat(
        (examples[0].pooled_question, examples[1].pooled_question), dim=0
    )
    target = torch.cat((examples[0].target, examples[1].target), dim=0)
    predicted = control(scene, question)
    return pair_delta_distillation_loss(
        predicted,
        target,
        mse_weight=mse_weight,
        cosine_weight=cosine_weight,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("V58 cannot summarize an empty sequence")
    return sum(values) / len(values)


def train_question_control_v58(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    scene_ids = validate_training_scene_ids(args.scene_id)
    output_checkpoint = _safe_output_path(args.output_checkpoint, "V58 runtime checkpoint")
    output_report = _safe_output_path(args.training_report, "V58 training report")
    teacher_artifact = _resolve(args.teacher_artifact)
    if teacher_artifact.exists():
        raise FileExistsError(
            f"V58 teacher artifact already exists; overwrite is forbidden: {teacher_artifact}"
        )
    if output_report.is_relative_to(output_checkpoint) or teacher_artifact.is_relative_to(
        output_checkpoint
    ) or output_report == teacher_artifact:
        raise ValueError("V58 training artifacts must remain outside runtime checkpoint")

    config, config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, base_checkpoint_files = checkpoint_fingerprint(base_checkpoint)
    source_control_sha256 = _control_checkpoint_sha256(args.source_control_checkpoint)
    records, qa_sha256 = load_training_records(args.train_qa, scene_ids=scene_ids)
    ordered_records = sorted(records, key=lambda record: (record.scene_id, record.question_id))
    changed_units = _changed_pair_units(ordered_records)
    changed_keys = {
        (record.scene_id, record.question_id)
        for unit in changed_units
        for record in unit
    }
    _log_event(
        phase="v58_preflight_complete",
        scene_count=len(scene_ids),
        record_count=len(ordered_records),
        changed_record_count=len(changed_keys),
        changed_pair_count=len(changed_units),
    )

    runtime_factory = StaticRuntimePrefixFactory(config, base_checkpoint, scene_ids[0])
    loads = 0

    def cache_runtime_loader(scene_id: str) -> StaticChatRuntime:
        nonlocal loads
        loads += 1
        _log_event(
            phase="v58_prefix_cache_build",
            scene_id=scene_id,
            scene_ordinal=loads,
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
    device = _select_training_device(runtime, args.device)
    frozen_audit = freeze_base_runtime(runtime)
    model_dtype = next(runtime.language.model.parameters()).dtype
    prefixes = {
        scene_id: prefix.to(device=device, dtype=model_dtype)
        for scene_id, prefix in cache.prefixes.items()
    }
    source_control, source_metadata = _load_control_head(
        args.source_control_checkpoint,
        hidden_size=runtime.language.hidden_size,
        device=device,
    )
    if (
        source_metadata["base_checkpoint_sha256"] != base_checkpoint_sha256
        or source_metadata["base_runtime_config_sha256"] != runtime_config_sha256
    ):
        raise ValueError("V58 source controller belongs to another frozen runtime")

    pooled_questions: dict[tuple[str, str], torch.Tensor] = {}
    baseline_prompts: dict[tuple[str, str], torch.Tensor] = {}
    source_control.eval()
    with torch.inference_mode():
        for record in ordered_records:
            key = (record.scene_id, record.question_id)
            pooled = _pooled_question_embedding(runtime, record.question)
            pooled_questions[key] = pooled
            baseline_prompts[key] = source_control(
                prefixes[record.scene_id].float(), pooled
            ).detach()

    runtime.language.enable_decoder_gradient_checkpointing()
    teacher_prompts = dict(baseline_prompts)
    teacher_rows: list[dict[str, Any]] = []
    teacher_started = time.perf_counter()
    for ordinal, record in enumerate(
        (
            record
            for record in ordered_records
            if (record.scene_id, record.question_id) in changed_keys
        ),
        start=1,
    ):
        key = (record.scene_id, record.question_id)
        prompt, metrics = _optimize_teacher_prompt_adaptive(
            runtime=runtime,
            scene_prefix=prefixes[record.scene_id],
            record=record,
            initial_prompt=baseline_prompts[key],
            learning_rate=args.teacher_learning_rate,
            min_steps=args.teacher_min_steps,
            max_steps=args.teacher_max_steps,
            nll_threshold=args.teacher_nll_threshold,
            gradient_clip_norm=args.teacher_gradient_clip_norm,
        )
        teacher_prompts[key] = prompt
        row = {
            "scene_id": record.scene_id,
            "question_id": record.question_id,
            **metrics,
        }
        teacher_rows.append(row)
        _log_event(
            phase="v58_teacher_optimization",
            completed=ordinal,
            total=len(changed_keys),
            scene_id=record.scene_id,
            question_id=record.question_id,
            steps=metrics["steps"],
            final_nll=metrics["final_nll"],
            final_rms=metrics["final_rms"],
        )
    _disable_decoder_checkpointing(runtime.language)

    teacher_verification: list[dict[str, Any]] = []
    records_by_key = {
        (record.scene_id, record.question_id): record for record in ordered_records
    }
    for key in sorted(
        item for item in records_by_key if item in changed_keys
    ):
        record = records_by_key[key]
        generated = _generate_with_control(
            runtime=runtime,
            scene_prefix=prefixes[record.scene_id],
            question=record.question,
            control_tokens=teacher_prompts[key],
        )
        teacher_verification.append(
            {
                "scene_id": record.scene_id,
                "question_id": record.question_id,
                "exact_normalized": exact_normalized_match(generated, record.answer),
            }
        )
    teacher_exact = sum(row["exact_normalized"] for row in teacher_verification)
    if teacher_exact != len(teacher_verification):
        raise RuntimeError(
            "V58 free soft-prompt teachers failed greedy verification: "
            f"{teacher_exact}/{len(teacher_verification)}"
        )
    teacher_elapsed = time.perf_counter() - teacher_started

    teacher_targets = [
        SoftPromptTarget(
            record.scene_id,
            record.question_id,
            (
                "changed_teacher"
                if (record.scene_id, record.question_id) in changed_keys
                else "retention_baseline"
            ),
            teacher_prompts[(record.scene_id, record.question_id)],
        )
        for record in ordered_records
    ]
    teacher_hashes = save_teacher_artifact(
        teacher_artifact,
        targets=teacher_targets,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
        source_control_checkpoint_sha256=source_control_sha256,
    )

    examples = [
        DistillationExample(
            scene_id=record.scene_id,
            question_id=record.question_id,
            scene_prefix=prefixes[record.scene_id],
            pooled_question=pooled_questions[(record.scene_id, record.question_id)],
            target=teacher_prompts[(record.scene_id, record.question_id)],
            changed=(record.scene_id, record.question_id) in changed_keys,
        )
        for record in ordered_records
    ]
    examples_by_key = {example.key: example for example in examples}
    pair_examples = [
        (
            examples_by_key[(unit[0].scene_id, unit[0].question_id)],
            examples_by_key[(unit[1].scene_id, unit[1].question_id)],
        )
        for unit in changed_units
    ]
    distill_control = copy.deepcopy(source_control).to(device=device, dtype=torch.float32)
    distill_control.train()
    optimizer = torch.optim.AdamW(
        distill_control.parameters(),
        lr=args.distill_learning_rate,
        weight_decay=args.distill_weight_decay,
    )
    distill_rows: list[dict[str, Any]] = []
    gradient_norms: list[float] = []
    distill_started = time.perf_counter()
    step_ordinal = 0
    for epoch in range(args.distill_epochs):
        rng = random.Random(args.seed + epoch * 1_000_003)
        expanded = [
            example
            for example in examples
            for _ in range(args.changed_repeats if example.changed else 1)
        ]
        rng.shuffle(expanded)
        minibatches = [
            expanded[offset : offset + args.distill_batch_size]
            for offset in range(0, len(expanded), args.distill_batch_size)
        ]
        for kind, batches in (
            ("prompt", minibatches),
            ("pair_delta", list(pair_examples)),
        ):
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                if kind == "prompt":
                    loss, diagnostics = _distill_minibatch(
                        distill_control,
                        batch,
                        mse_weight=args.distill_mse_weight,
                        cosine_weight=args.distill_cosine_weight,
                    )
                    diagnostic_value = float(
                        diagnostics["mean_token_cosine"].detach().cpu()
                    )
                else:
                    pair_loss, diagnostics = _distill_pair_delta(
                        distill_control,
                        batch,
                        mse_weight=args.pair_delta_mse_weight,
                        cosine_weight=args.pair_delta_cosine_weight,
                    )
                    loss = args.pair_delta_weight * pair_loss
                    diagnostic_value = float(diagnostics["delta_cosine"].detach().cpu())
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    distill_control.parameters(), args.distill_gradient_clip_norm
                )
                gradient_value = float(gradient.detach().float().cpu())
                if not math.isfinite(gradient_value):
                    raise RuntimeError("V58 distillation gradient is nonfinite")
                optimizer.step()
                if any(
                    not torch.isfinite(value).all()
                    for value in distill_control.state_dict().values()
                ):
                    raise RuntimeError("V58 distillation produced nonfinite state")
                step_ordinal += 1
                row = {
                    "epoch": epoch,
                    "ordinal": step_ordinal,
                    "kind": kind,
                    "loss": float(loss.detach().cpu()),
                    "diagnostic_cosine": diagnostic_value,
                    "preclip_gradient_norm": gradient_value,
                }
                distill_rows.append(row)
                gradient_norms.append(gradient_value)
                if step_ordinal % args.log_every == 0:
                    _log_event(phase="v58_distillation", **row)
    distill_control.eval()
    distill_elapsed = time.perf_counter() - distill_started

    final_prompt_metrics: defaultdict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for example in examples:
            predicted = distill_control(
                example.scene_prefix.float(), example.pooled_question
            )
            _loss, diagnostics = normalized_prompt_distillation_loss(
                predicted,
                example.target,
                mse_weight=args.distill_mse_weight,
                cosine_weight=args.distill_cosine_weight,
            )
            group = "changed" if example.changed else "retention"
            final_prompt_metrics[f"{group}_normalized_mse"].append(
                float(diagnostics["normalized_mse"].cpu())
            )
            final_prompt_metrics[f"{group}_token_cosine"].append(
                float(diagnostics["mean_token_cosine"].cpu())
            )
        final_pair_cosines = []
        for pair in pair_examples:
            _loss, diagnostics = _distill_pair_delta(
                distill_control,
                pair,
                mse_weight=args.pair_delta_mse_weight,
                cosine_weight=args.pair_delta_cosine_weight,
            )
            final_pair_cosines.append(float(diagnostics["delta_cosine"].cpu()))

    distilled_verification: list[dict[str, Any]] = []
    with torch.inference_mode():
        for key in sorted(item for item in records_by_key if item in changed_keys):
            record = records_by_key[key]
            control_tokens = distill_control(
                prefixes[record.scene_id].float(), pooled_questions[key]
            )
            generated = _generate_with_control(
                runtime=runtime,
                scene_prefix=prefixes[record.scene_id],
                question=record.question,
                control_tokens=control_tokens,
            )
            distilled_verification.append(
                {
                    "scene_id": record.scene_id,
                    "question_id": record.question_id,
                    "exact_normalized": exact_normalized_match(
                        generated, record.answer
                    ),
                }
            )
    distilled_exact = sum(row["exact_normalized"] for row in distilled_verification)
    checkpoint_hashes = save_control_checkpoint(
        output_checkpoint,
        control=distill_control,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    report = {
        "schema_version": 1,
        "artifact": "v58_soft_prompt_teacher_distillation_training",
        "passed": distilled_exact == len(distilled_verification),
        "base": {
            "checkpoint_sha256": base_checkpoint_sha256,
            "checkpoint_files": base_checkpoint_files,
            "runtime_config_effective_sha256": runtime_config_sha256,
            "runtime_config_file_sha256": _sha256_file(config_path),
            "source_control_checkpoint_sha256": source_control_sha256,
        },
        "inputs": {
            "training_qa_sha256": qa_sha256,
            "training_scene_ids": list(scene_ids),
            "training_record_count": len(records),
            "changed_record_count": len(changed_keys),
            "prefix_cache_manifest_sha256": _sha256_file(
                _resolve(args.prefix_cache) / "manifest.json"
            ),
        },
        "teacher": {
            "optimizer": "Adam",
            "learning_rate": args.teacher_learning_rate,
            "gradient_clip_norm": args.teacher_gradient_clip_norm,
            "minimum_steps": args.teacher_min_steps,
            "maximum_steps": args.teacher_max_steps,
            "nll_threshold": args.teacher_nll_threshold,
            "elapsed_seconds": teacher_elapsed,
            "mean_final_nll": _mean([row["final_nll"] for row in teacher_rows]),
            "mean_final_rms": _mean([row["final_rms"] for row in teacher_rows]),
            "selected_learning_rates": {
                str(rate): sum(row["learning_rate"] == rate for row in teacher_rows)
                for rate in sorted({row["learning_rate"] for row in teacher_rows})
            },
            "total_forward_steps": sum(
                row["total_forward_steps"] for row in teacher_rows
            ),
            "greedy_exact": teacher_exact,
            "greedy_total": len(teacher_verification),
            "artifact": teacher_hashes,
        },
        "distillation": {
            "epochs": args.distill_epochs,
            "optimizer_steps": step_ordinal,
            "learning_rate": args.distill_learning_rate,
            "batch_size": args.distill_batch_size,
            "changed_repeats": args.changed_repeats,
            "elapsed_seconds": distill_elapsed,
            "maximum_preclip_gradient_norm": max(gradient_norms),
            "final_metrics": {
                key: _mean(values) for key, values in sorted(final_prompt_metrics.items())
            },
            "mean_pair_delta_cosine": _mean(final_pair_cosines),
            "greedy_exact": distilled_exact,
            "greedy_total": len(distilled_verification),
        },
        "checkpoint": checkpoint_hashes,
        "scope": {
            "runtime_architecture": "full_scene_question_control_v1",
            "base_scene_stack_frozen": frozen_audit["all_parameters_frozen"],
            "only_control_head_saved": True,
            "teacher_prompts_training_only": True,
            "teacher_artifact_runtime_load_permitted": False,
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
    parser.add_argument("--source-control-checkpoint", required=True)
    parser.add_argument("--train-qa", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--teacher-artifact", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=58058)
    parser.add_argument("--teacher-learning-rate", type=float, default=0.03)
    parser.add_argument("--teacher-min-steps", type=int, default=5)
    parser.add_argument("--teacher-max-steps", type=int, default=20)
    parser.add_argument("--teacher-nll-threshold", type=float, default=1e-3)
    parser.add_argument("--teacher-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--distill-epochs", type=int, default=100)
    parser.add_argument("--distill-batch-size", type=int, default=8)
    parser.add_argument("--changed-repeats", type=int, default=4)
    parser.add_argument("--distill-learning-rate", type=float, default=1e-3)
    parser.add_argument("--distill-weight-decay", type=float, default=0.0)
    parser.add_argument("--distill-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--distill-mse-weight", type=float, default=1.0)
    parser.add_argument("--distill-cosine-weight", type=float, default=1.0)
    parser.add_argument("--pair-delta-weight", type=float, default=2.0)
    parser.add_argument("--pair-delta-mse-weight", type=float, default=1.0)
    parser.add_argument("--pair-delta-cosine-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = train_question_control_v58(args)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "teacher_exact": report["teacher"]["greedy_exact"],
                "distilled_exact": report["distillation"]["greedy_exact"],
                "checkpoint": str(_resolve(args.output_checkpoint)),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "train_question_control_v58"]
