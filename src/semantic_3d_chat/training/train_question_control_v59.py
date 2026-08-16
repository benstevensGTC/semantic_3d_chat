"""Train the compact V2 controller on a locked six-scene training gate.

The only Gemma-backprop stage optimizes 14 numeric per-record prompt teachers
for two expansion counterfactual pairs.  Those prompts are cached separately
from runtime artifacts.  The V2 controller then distills numeric prompts and a
scene-by-question route label without running Gemma backward.  Anchor changed
rows reuse the already-verified V58 numeric teachers; ordinary anchor rows get
the exact no-control route and are never supplied as semantic text at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.evaluation.metrics import exact_normalized_match
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v59_multiscene_train_gate import (
    ANCHOR_PAIR_ID,
    EXPANSION_PAIR_IDS,
    LOCKED_SCENE_IDS,
)
from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl
from semantic_3d_chat.scene_encoder.question_control_v2 import (
    BoundedFullSceneQuestionControlV2,
)
from semantic_3d_chat.training.question_control_v2_checkpoint import (
    save_v2_control_checkpoint,
)
from semantic_3d_chat.training.soft_prompt_teacher_v58 import (
    load_teacher_artifact,
    normalized_prompt_distillation_loss,
    pair_delta_distillation_loss,
)
from semantic_3d_chat.training.soft_prompt_teacher_v59 import (
    ExpansionTeacherTarget,
    load_expansion_teachers,
    save_expansion_teachers,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _cache_entry,
    _cache_path_guard,
    _changed_pair_units,
    _load_sanitized_runtime_config,
    _log_event,
    _resolve,
    _safe_output_path,
    _select_training_device,
    _sha256_file,
    _write_json,
    _write_training_report,
    freeze_base_runtime,
    load_prefix_cache,
    load_training_records,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _disable_decoder_checkpointing,
    _generate_with_control,
    _optimize_teacher_prompt_adaptive,
    _pooled_question_embedding,
)

_EXPECTED_RECORD_COUNT: Final[int] = 144
_EXPECTED_CHANGED_COUNT: Final[int] = 22
_EXPECTED_ANCHOR_CHANGED: Final[int] = 8
_EXPECTED_EXPANSION_CHANGED: Final[int] = 14


@dataclass(frozen=True)
class V2Example:
    scene_id: str
    question_id: str
    scene_signature: torch.Tensor
    pooled_question: torch.Tensor
    target: torch.Tensor | None
    route_label: float
    group: str

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.question_id


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    if tuple(sorted(args.scene_id)) != LOCKED_SCENE_IDS:
        raise ValueError("V59 trainer requires the exact preregistered six scenes")
    integers = (
        "teacher_min_steps",
        "teacher_max_steps",
        "distill_epochs",
        "distill_batch_size",
        "changed_repeats",
        "retention_repeats",
        "moment_count",
        "interaction_dim",
        "output_rank",
        "log_every",
    )
    for field in integers:
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"V59 {field} must be a positive integer")
    if args.teacher_min_steps > args.teacher_max_steps:
        raise ValueError("V59 teacher_min_steps exceeds teacher_max_steps")
    floats = (
        "teacher_learning_rate",
        "teacher_nll_threshold",
        "teacher_gradient_clip_norm",
        "distill_learning_rate",
        "distill_weight_decay",
        "distill_gradient_clip_norm",
        "distill_mse_weight",
        "distill_cosine_weight",
        "pair_delta_weight",
        "route_loss_weight",
        "maximum_control_rms",
        "initial_control_rms",
        "gate_threshold",
    )
    for field in floats:
        value = float(getattr(args, field))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"V59 {field} must be finite and nonnegative")
    if not 0.0 < args.maximum_control_rms <= 1.0:
        raise ValueError("V59 maximum_control_rms must be in (0,1]")
    if not 0.0 < args.initial_control_rms < args.maximum_control_rms:
        raise ValueError("V59 initial_control_rms must be in (0, maximum)")
    if not 0.0 < args.gate_threshold < 1.0:
        raise ValueError("V59 gate_threshold must be in (0,1)")


def _derive_subset_prefix_cache(
    *,
    source_cache: str | Path,
    destination_cache: str | Path,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], bool]:
    destination = _cache_path_guard(destination_cache)
    if destination.exists():
        prefixes, manifest = load_prefix_cache(
            destination,
            scene_ids=LOCKED_SCENE_IDS,
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
        )
        return prefixes, manifest, False
    source = _cache_path_guard(source_cache)
    manifest_path = source / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("V59 full prefix-cache manifest is unavailable")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_scene_ids = tuple(sorted(source_manifest.get("scenes", {})))
    prefixes, _validated = load_prefix_cache(
        source,
        scene_ids=source_scene_ids,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=base_runtime_config_sha256,
    )
    if not set(LOCKED_SCENE_IDS).issubset(prefixes):
        raise ValueError("V59 full prefix cache lacks locked scenes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        entries = {}
        for scene_id in LOCKED_SCENE_IDS:
            tensor = prefixes[scene_id].detach().cpu().contiguous()
            path = temporary / f"{scene_id}.safetensors"
            save_file({"scene_prefix": tensor}, path)
            entries[scene_id] = _cache_entry(scene_id, path, tensor)
        manifest = {
            "schema_version": 1,
            "artifact": "question_independent_scene_prefix_cache_v1",
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "base_runtime_config_sha256": base_runtime_config_sha256,
            "scene_count": len(LOCKED_SCENE_IDS),
            "question_inputs_used": False,
            "question_dependent_scene_retrieval": False,
            "complete_scene_prefixes": True,
            "environmental_text_inputs": [],
            "scenes": entries,
        }
        _write_json(temporary / "manifest.json", manifest)
        validated_prefixes, validated_manifest = load_prefix_cache(
            temporary,
            scene_ids=LOCKED_SCENE_IDS,
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
        )
        temporary.rename(destination)
        return validated_prefixes, validated_manifest, True
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _locked_inventory(records: Sequence[Any]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    changed = [record for record in records if record.counterfactual_expected_change is True]
    anchor = {
        (record.scene_id, record.question_id)
        for record in changed
        if record.counterfactual_pair_id == ANCHOR_PAIR_ID
    }
    expansion = {
        (record.scene_id, record.question_id)
        for record in changed
        if record.counterfactual_pair_id in EXPANSION_PAIR_IDS
    }
    if (
        len(records) != _EXPECTED_RECORD_COUNT
        or len(changed) != _EXPECTED_CHANGED_COUNT
        or len(anchor) != _EXPECTED_ANCHOR_CHANGED
        or len(expansion) != _EXPECTED_EXPANSION_CHANGED
        or len(anchor | expansion) != _EXPECTED_CHANGED_COUNT
    ):
        raise ValueError("V59 locked six-scene QA inventory changed")
    return anchor, expansion


def _teacher_selection_sha256(expansion_keys: set[tuple[str, str]]) -> str:
    return _canonical_sha256([list(key) for key in sorted(expansion_keys)])


def _load_or_create_expansion_teachers(
    *,
    args: argparse.Namespace,
    runtime: Any,
    prefixes: Mapping[str, torch.Tensor],
    records_by_key: Mapping[tuple[str, str], Any],
    source_prompts: Mapping[tuple[str, str], torch.Tensor],
    expansion_keys: set[tuple[str, str]],
    base_checkpoint_sha256: str,
    runtime_config_sha256: str,
    source_control_sha256: str,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]:
    selection_sha256 = _teacher_selection_sha256(expansion_keys)
    cache_path = _resolve(args.teacher_cache)
    if cache_path.exists():
        targets, metadata = load_expansion_teachers(cache_path)
        if (
            set(targets) != expansion_keys
            or metadata["base_checkpoint_sha256"] != base_checkpoint_sha256
            or metadata["base_runtime_config_sha256"] != runtime_config_sha256
            or metadata["source_control_checkpoint_sha256"] != source_control_sha256
            or metadata["selection_sha256"] != selection_sha256
        ):
            raise ValueError("V59 expansion teacher cache belongs to another run")
        return targets, {"cache_reused": True, "metadata": metadata}

    runtime.language.enable_decoder_gradient_checkpointing()
    targets: dict[tuple[str, str], torch.Tensor] = {}
    rows = []
    started = time.perf_counter()
    for ordinal, key in enumerate(sorted(expansion_keys), 1):
        record = records_by_key[key]
        prompt, metrics = _optimize_teacher_prompt_adaptive(
            runtime=runtime,
            scene_prefix=prefixes[record.scene_id],
            record=record,
            initial_prompt=source_prompts[key],
            learning_rate=args.teacher_learning_rate,
            min_steps=args.teacher_min_steps,
            max_steps=args.teacher_max_steps,
            nll_threshold=args.teacher_nll_threshold,
            gradient_clip_norm=args.teacher_gradient_clip_norm,
        )
        targets[key] = prompt.detach()
        rows.append({"scene_id": key[0], "question_id": key[1], **metrics})
        _log_event(
            phase="v59_expansion_teacher",
            completed=ordinal,
            total=len(expansion_keys),
            scene_id=key[0],
            question_id=key[1],
            final_nll=metrics["final_nll"],
            total_forward_steps=metrics["total_forward_steps"],
        )
    _disable_decoder_checkpointing(runtime.language)
    verification = []
    for key in sorted(expansion_keys):
        record = records_by_key[key]
        generated = _generate_with_control(
            runtime=runtime,
            scene_prefix=prefixes[record.scene_id],
            question=record.question,
            control_tokens=targets[key],
        )
        exact = exact_normalized_match(generated, record.answer)
        verification.append(exact)
    if not all(verification):
        raise RuntimeError(
            f"V59 expansion teachers failed greedy verification: {sum(verification)}/{len(verification)}"
        )
    hashes = save_expansion_teachers(
        cache_path,
        targets=[ExpansionTeacherTarget(*key, targets[key]) for key in sorted(targets)],
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
        source_control_checkpoint_sha256=source_control_sha256,
        selection_sha256=selection_sha256,
    )
    return targets, {
        "cache_reused": False,
        "artifact": hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "total_forward_steps": sum(row["total_forward_steps"] for row in rows),
        "mean_final_nll": sum(row["final_nll"] for row in rows) / len(rows),
        "greedy_exact": sum(verification),
        "greedy_total": len(verification),
    }


def _batch_loss(
    control: BoundedFullSceneQuestionControlV2,
    examples: Sequence[V2Example],
    *,
    mse_weight: float,
    cosine_weight: float,
    route_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    signatures = torch.cat([example.scene_signature for example in examples])
    questions = torch.cat([example.pooled_question for example in examples])
    output = control.forward_from_signature(signatures, questions)
    labels = torch.tensor(
        [example.route_label for example in examples],
        device=output.gate_logits.device,
        dtype=output.gate_logits.dtype,
    )
    route_loss = F.binary_cross_entropy_with_logits(output.gate_logits, labels)
    changed_indices = [index for index, example in enumerate(examples) if example.target is not None]
    if changed_indices:
        predicted = output.control_tokens[changed_indices]
        target = torch.cat(
            [examples[index].target for index in changed_indices if examples[index].target is not None]
        )
        prompt_loss, diagnostics = normalized_prompt_distillation_loss(
            predicted,
            target,
            mse_weight=mse_weight,
            cosine_weight=cosine_weight,
        )
        cosine = float(diagnostics["mean_token_cosine"].detach().cpu())
    else:
        prompt_loss = output.control_tokens.sum() * 0.0
        cosine = 0.0
    total = prompt_loss + route_weight * route_loss
    return total, {
        "prompt_loss": float(prompt_loss.detach().cpu()),
        "route_loss": float(route_loss.detach().cpu()),
        "prompt_cosine": cosine,
    }


def _pair_loss(
    control: BoundedFullSceneQuestionControlV2,
    pair: tuple[V2Example, V2Example],
    *,
    weight: float,
) -> tuple[torch.Tensor, float]:
    signatures = torch.cat([pair[0].scene_signature, pair[1].scene_signature])
    questions = torch.cat([pair[0].pooled_question, pair[1].pooled_question])
    targets = torch.cat([pair[0].target, pair[1].target])
    predicted = control.forward_from_signature(signatures, questions).control_tokens
    loss, diagnostics = pair_delta_distillation_loss(predicted, targets)
    return weight * loss, float(diagnostics["delta_cosine"].detach().cpu())


def train_v59(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_checkpoint = _safe_output_path(args.output_checkpoint, "V59 V2 checkpoint")
    output_report = _safe_output_path(args.training_report, "V59 training report")
    config, config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, base_checkpoint_files = checkpoint_fingerprint(base_checkpoint)
    source_control_sha256 = _control_checkpoint_sha256(args.source_control_checkpoint)
    anchor_teacher_source_sha256 = _control_checkpoint_sha256(
        args.anchor_teacher_source_control_checkpoint
    )
    records, qa_sha256 = load_training_records(args.train_qa, scene_ids=LOCKED_SCENE_IDS)
    records = sorted(records, key=lambda record: (record.scene_id, record.question_id))
    anchor_keys, expansion_keys = _locked_inventory(records)
    changed_keys = anchor_keys | expansion_keys
    records_by_key = {(record.scene_id, record.question_id): record for record in records}

    prefixes, prefix_manifest, cache_created = _derive_subset_prefix_cache(
        source_cache=args.full_prefix_cache,
        destination_cache=args.subset_prefix_cache,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )
    runtime_factory = StaticRuntimePrefixFactory(config, base_checkpoint, LOCKED_SCENE_IDS[0])
    runtime = runtime_factory.bootstrap
    device = _select_training_device(runtime, args.device)
    frozen = freeze_base_runtime(runtime)
    model_dtype = next(runtime.language.model.parameters()).dtype
    prefixes = {
        scene_id: prefix.to(device=device, dtype=model_dtype)
        for scene_id, prefix in prefixes.items()
    }
    source_control, source_metadata = _load_control_head(
        args.source_control_checkpoint,
        hidden_size=runtime.language.hidden_size,
        device=device,
    )
    if not isinstance(source_control, FullSceneQuestionControl):
        raise TypeError("V59 source controller must be the accepted V58 V1")
    if (
        source_metadata["base_checkpoint_sha256"] != base_checkpoint_sha256
        or source_metadata["base_runtime_config_sha256"] != runtime_config_sha256
    ):
        raise ValueError("V59 source controller belongs to another frozen runtime")
    imported, imported_metadata = load_teacher_artifact(args.anchor_teacher_artifact)
    anchor_scene_keys = {
        (record.scene_id, record.question_id)
        for record in records
        if record.scene_id in {"scene_000031", "scene_000032"}
    }
    imported_roles = {
        (str(row["scene_id"]), str(row["question_id"])): str(row["role"])
        for row in imported_metadata["records"]
    }
    if (
        set(imported) != anchor_scene_keys
        or set(imported_roles) != anchor_scene_keys
        or {
            key for key, role in imported_roles.items() if role == "changed_teacher"
        }
        != anchor_keys
        or imported_metadata["target_count"] != len(anchor_scene_keys)
        or imported_metadata["base_checkpoint_sha256"] != base_checkpoint_sha256
        or imported_metadata["base_runtime_config_sha256"] != runtime_config_sha256
        or imported_metadata["source_control_checkpoint_sha256"]
        != anchor_teacher_source_sha256
    ):
        raise ValueError("V59 anchor teacher artifact contract changed")

    pooled_questions = {}
    source_prompts = {}
    source_control.eval()
    with torch.inference_mode():
        for record in records:
            key = (record.scene_id, record.question_id)
            pooled = _pooled_question_embedding(runtime, record.question)
            pooled_questions[key] = pooled
            source_prompts[key] = source_control(
                prefixes[record.scene_id].float(), pooled
            ).detach()
    expansion_teachers, teacher_report = _load_or_create_expansion_teachers(
        args=args,
        runtime=runtime,
        prefixes=prefixes,
        records_by_key=records_by_key,
        source_prompts=source_prompts,
        expansion_keys=expansion_keys,
        base_checkpoint_sha256=base_checkpoint_sha256,
        runtime_config_sha256=runtime_config_sha256,
        source_control_sha256=source_control_sha256,
    )
    targets = {key: imported[key].to(device) for key in anchor_keys}
    targets.update({key: value.to(device) for key, value in expansion_teachers.items()})
    target_rms_values = [
        float(value.detach().float().square().mean().sqrt().cpu())
        for value in targets.values()
    ]
    if max(target_rms_values) > args.maximum_control_rms:
        raise ValueError(
            "V59 maximum_control_rms is below an observed teacher target: "
            f"maximum_teacher={max(target_rms_values):.6f} "
            f"bound={args.maximum_control_rms:.6f}"
        )

    control = BoundedFullSceneQuestionControlV2(
        runtime.language.hidden_size,
        control_tokens=source_control.control_token_count,
        expected_environment_latents=int(config["scene_encoder"]["global_latents"]),
        moment_count=args.moment_count,
        interaction_dim=args.interaction_dim,
        output_rank=args.output_rank,
        maximum_control_rms=args.maximum_control_rms,
        initial_control_rms=args.initial_control_rms,
        gate_threshold=args.gate_threshold,
    ).to(device=device, dtype=torch.float32)
    if control.trainable_parameter_count >= 1_000_000:
        raise ValueError("V59 V2 trainable parameter count must remain below one million")
    signatures = {
        scene_id: control.encode_scene(prefix.float()) for scene_id, prefix in prefixes.items()
    }
    examples = [
        V2Example(
            record.scene_id,
            record.question_id,
            signatures[record.scene_id],
            pooled_questions[(record.scene_id, record.question_id)],
            targets.get((record.scene_id, record.question_id)),
            1.0 if (record.scene_id, record.question_id) in changed_keys else 0.0,
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
    examples_by_key = {example.key: example for example in examples}
    pair_examples = [
        (examples_by_key[(unit[0].scene_id, unit[0].question_id)], examples_by_key[(unit[1].scene_id, unit[1].question_id)])
        for unit in _changed_pair_units(records)
    ]
    optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=args.distill_learning_rate,
        weight_decay=args.distill_weight_decay,
    )
    control.train()
    started = time.perf_counter()
    rows = []
    gradients = []
    step = 0
    for epoch in range(args.distill_epochs):
        rng = random.Random(args.seed + epoch * 1_000_003)
        expanded = [
            example
            for example in examples
            for _ in range(
                args.changed_repeats
                if example.route_label == 1.0
                else args.retention_repeats
            )
        ]
        rng.shuffle(expanded)
        batches = [
            expanded[offset : offset + args.distill_batch_size]
            for offset in range(0, len(expanded), args.distill_batch_size)
        ]
        for kind, work in (("prompt_route", batches), ("pair_delta", pair_examples)):
            for batch in work:
                optimizer.zero_grad(set_to_none=True)
                if kind == "prompt_route":
                    loss, diagnostics = _batch_loss(
                        control,
                        batch,
                        mse_weight=args.distill_mse_weight,
                        cosine_weight=args.distill_cosine_weight,
                        route_weight=args.route_loss_weight,
                    )
                else:
                    loss, cosine = _pair_loss(control, batch, weight=args.pair_delta_weight)
                    diagnostics = {"pair_delta_cosine": cosine}
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    control.parameters(), args.distill_gradient_clip_norm
                )
                gradient_value = float(gradient.detach().cpu())
                if not math.isfinite(gradient_value):
                    raise RuntimeError("V59 V2 distillation gradient is nonfinite")
                optimizer.step()
                if any(not torch.isfinite(value).all() for value in control.state_dict().values()):
                    raise RuntimeError("V59 V2 distillation produced nonfinite state")
                step += 1
                gradients.append(gradient_value)
                rows.append({
                    "epoch": epoch,
                    "step": step,
                    "kind": kind,
                    "loss": float(loss.detach().cpu()),
                    **diagnostics,
                })
                if step % args.log_every == 0:
                    _log_event(phase="v59_v2_distillation", **rows[-1])
    control.eval()
    distill_elapsed = time.perf_counter() - started

    route_metrics: defaultdict[str, list[float]] = defaultdict(list)
    prompt_metrics: defaultdict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for example in examples:
            output = control.forward_from_signature(
                example.scene_signature, example.pooled_question
            )
            route_metrics[f"{example.group}_probability"].append(
                float(output.gate_probabilities.item())
            )
            route_metrics[f"{example.group}_correct"].append(
                float((output.gate_probabilities.item() >= args.gate_threshold) == bool(example.route_label))
            )
            if example.target is not None:
                _loss, diagnostics = normalized_prompt_distillation_loss(
                    output.control_tokens,
                    example.target,
                    mse_weight=args.distill_mse_weight,
                    cosine_weight=args.distill_cosine_weight,
                )
                prompt_metrics[f"{example.group}_cosine"].append(
                    float(diagnostics["mean_token_cosine"].cpu())
                )
        pair_cosines = []
        for pair in pair_examples:
            _loss, cosine = _pair_loss(control, pair, weight=1.0)
            pair_cosines.append(cosine)
    retention_route_exact = all(
        probability < args.gate_threshold
        for key, probabilities in route_metrics.items()
        if key == "retention_probability"
        for probability in probabilities
    )
    changed_route_exact = all(
        probability >= args.gate_threshold
        for key, probabilities in route_metrics.items()
        if key.endswith("changed_probability")
        for probability in probabilities
    )
    checkpoint_hashes = save_v2_control_checkpoint(
        output_checkpoint,
        control=control,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
        source_control_checkpoint_sha256=source_control_sha256,
    )
    mean = lambda values: sum(values) / len(values) if values else None
    report = {
        "schema_version": 1,
        "artifact": "v59_bounded_global_scene_question_control_training",
        "training_objectives_met": retention_route_exact and changed_route_exact,
        "promotion_eligible": False,
        "saved_runtime_gate_required": True,
        "base": {
            "checkpoint_sha256": base_checkpoint_sha256,
            "checkpoint_files": base_checkpoint_files,
            "runtime_config_effective_sha256": runtime_config_sha256,
            "runtime_config_file_sha256": _sha256_file(config_path),
            "source_control_checkpoint_sha256": source_control_sha256,
            "anchor_teacher_source_control_checkpoint_sha256": (
                anchor_teacher_source_sha256
            ),
        },
        "inputs": {
            "training_qa_sha256": qa_sha256,
            "training_scene_ids": list(LOCKED_SCENE_IDS),
            "training_record_count": len(records),
            "anchor_changed_count": len(anchor_keys),
            "expansion_changed_count": len(expansion_keys),
            "prefix_cache_created": cache_created,
            "prefix_cache_manifest_sha256": _sha256_file(
                _resolve(args.subset_prefix_cache) / "manifest.json"
            ),
            "prefix_shape": prefix_manifest["scenes"][LOCKED_SCENE_IDS[0]]["shape"],
        },
        "teacher": teacher_report,
        "teacher_target_rms": {
            "minimum": min(target_rms_values),
            "mean": sum(target_rms_values) / len(target_rms_values),
            "maximum": max(target_rms_values),
        },
        "architecture": {
            "name": "bounded_global_scene_question_control_v2",
            "trainable_parameter_count": control.trainable_parameter_count,
            "moment_count": control.moment_count,
            "interaction_dim": control.interaction_dim,
            "output_rank": control.output_rank,
            "maximum_control_rms": control.maximum_control_rms,
            "initial_control_rms": control.initial_control_rms,
            "scene_signature_shape": list(signatures[LOCKED_SCENE_IDS[0]].shape),
            "boundary_tokens_excluded": True,
            "softmax_scene_attention_used": False,
        },
        "distillation": {
            "epochs": args.distill_epochs,
            "optimizer_steps": step,
            "elapsed_seconds": distill_elapsed,
            "maximum_preclip_gradient_norm": max(gradients),
            "route_metrics": {key: mean(values) for key, values in sorted(route_metrics.items())},
            "prompt_metrics": {key: mean(values) for key, values in sorted(prompt_metrics.items())},
            "mean_pair_delta_cosine": mean(pair_cosines),
            "retention_route_all_off": retention_route_exact,
            "changed_route_all_on": changed_route_exact,
        },
        "checkpoint": checkpoint_hashes,
        "scope": {
            "base_scene_stack_frozen": frozen["all_parameters_frozen"],
            "only_controller_saved": True,
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
    parser.add_argument("--source-control-checkpoint", required=True)
    parser.add_argument("--anchor-teacher-artifact", required=True)
    parser.add_argument("--anchor-teacher-source-control-checkpoint", required=True)
    parser.add_argument("--train-qa", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--full-prefix-cache", required=True)
    parser.add_argument("--subset-prefix-cache", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=59059)
    parser.add_argument("--teacher-learning-rate", type=float, default=0.03)
    parser.add_argument("--teacher-min-steps", type=int, default=5)
    parser.add_argument("--teacher-max-steps", type=int, default=20)
    parser.add_argument("--teacher-nll-threshold", type=float, default=1e-3)
    parser.add_argument("--teacher-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--distill-epochs", type=int, default=80)
    parser.add_argument("--distill-batch-size", type=int, default=16)
    parser.add_argument("--changed-repeats", type=int, default=4)
    parser.add_argument("--retention-repeats", type=int, default=1)
    parser.add_argument("--distill-learning-rate", type=float, default=1e-3)
    parser.add_argument("--distill-weight-decay", type=float, default=0.0)
    parser.add_argument("--distill-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--distill-mse-weight", type=float, default=1.0)
    parser.add_argument("--distill-cosine-weight", type=float, default=1.0)
    parser.add_argument("--pair-delta-weight", type=float, default=2.0)
    parser.add_argument("--route-loss-weight", type=float, default=4.0)
    parser.add_argument("--moment-count", type=int, default=8)
    parser.add_argument("--interaction-dim", type=int, default=24)
    parser.add_argument("--output-rank", type=int, default=64)
    parser.add_argument("--maximum-control-rms", type=float, default=0.2)
    parser.add_argument("--initial-control-rms", type=float, default=0.075)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = train_v59(args)
    print(json.dumps({"training_objectives_met": report["training_objectives_met"], "promotion_eligible": False, "checkpoint": str(_resolve(args.output_checkpoint))}, sort_keys=True))
    return 0 if report["training_objectives_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "train_v59"]
