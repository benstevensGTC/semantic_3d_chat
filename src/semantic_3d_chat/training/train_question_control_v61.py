"""Train only V4's compact scene-conditioned gate over frozen V60 values.

This trainer consumes the locked six-scene training QA, immutable full-scene
prefixes, and the sanitized V60 runtime checkpoint/report.  It has no argument
for—and never opens—the V61 paraphrase gate or its V54 baseline.  Gemma is used
only to obtain frozen question embeddings; no Gemma backward or answer
generation occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
from semantic_3d_chat.evaluation.v59_multiscene_train_gate import (
    ANCHOR_PAIR_ID,
    EXPANSION_PAIR_IDS,
    LOCKED_SCENE_IDS,
)
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v4 import (
    SceneConditionedGateTeacherBasisControlV4,
)
from semantic_3d_chat.training.question_control_v4_checkpoint import (
    inherited_value_state_sha256,
    save_v4_control_checkpoint,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    _log_event,
    _resolve,
    _safe_output_path,
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
class GateExampleV4:
    scene_id: str
    question_id: str
    feature: torch.Tensor
    label: float
    group: str


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, raw in state.items():
        value = raw.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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
    if (
        len(records) != _EXPECTED_RECORDS
        or len(anchor) != 8
        or len(expansion) != 14
        or len(anchor | expansion) != _EXPECTED_CHANGED
    ):
        raise ValueError("V61 locked six-scene route inventory changed")
    return anchor, expansion


def _validate_args(args: argparse.Namespace) -> None:
    if tuple(sorted(args.scene_id)) != LOCKED_SCENE_IDS:
        raise ValueError("V61 requires the exact locked six training scenes")
    for field in (
        "gate_hidden_dim",
        "epochs",
        "minimum_epochs",
        "success_patience",
        "log_every",
    ):
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"V61 {field} must be a positive integer")
    if args.minimum_epochs > args.epochs:
        raise ValueError("V61 minimum_epochs cannot exceed epochs")
    for field in (
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "minimum_signed_logit_margin",
        "margin_weight",
    ):
        value = float(getattr(args, field))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"V61 {field} must be finite and nonnegative")
    forbidden = {
        _resolve(args.source_v60_checkpoint),
        _resolve(args.base_checkpoint),
    }
    if _resolve(args.output_checkpoint) in forbidden:
        raise ValueError("V61 output must not overwrite V60 or the base checkpoint")


def _load_source_report(
    path: str | Path,
    *,
    source_control_sha256: str,
    source_weights_sha256: str,
) -> tuple[dict[str, Any], str]:
    source = _resolve(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("V61 source V60 report must be a JSON object")
    checks = value.get("offline_checks")
    optimization = value.get("optimization")
    if (
        value.get("artifact") != "v60_teacher_basis_control_training"
        or not isinstance(checks, Mapping)
        or not isinstance(optimization, Mapping)
        or checks.get("basis_mean_cosine") is not True
        or checks.get("basis_minimum_cosine") is not True
        or checks.get("mean_prompt_cosine") is not True
        or checks.get("minimum_prompt_cosine") is not True
        or checks.get("mean_rms_absolute_error") is not True
        or checks.get("mean_pair_delta_cosine") is not True
        or value.get("checkpoint", {}).get("weights_sha256") != source_weights_sha256
    ):
        raise ValueError("V61 source V60 prompt/value evidence changed")
    # Bind the report to both the exact runtime checkpoint aggregate and its
    # weights; the aggregate itself is recorded in V61 output.
    if len(source_control_sha256) != 64:
        raise ValueError("V61 source V60 checkpoint digest is invalid")
    return value, _sha256_file(source)


def _group(label: bool, key: tuple[str, str], anchor: set[tuple[str, str]]) -> str:
    if not label:
        return "retention"
    return "anchor_changed" if key in anchor else "expansion_changed"


def _route_diagnostics(
    control: SceneConditionedGateTeacherBasisControlV4,
    examples: Sequence[GateExampleV4],
) -> tuple[dict[str, dict[str, float | int]], float, bool]:
    grouped: defaultdict[str, list[tuple[bool, float]]] = defaultdict(list)
    exact = True
    minimum_margin = math.inf
    with torch.inference_mode():
        for example in examples:
            logit = control.scene_question_gate(example.feature).squeeze()
            probability = float(torch.sigmoid(logit).cpu())
            predicted = probability >= control.gate_threshold
            correct = predicted == bool(example.label)
            signed = float(logit.cpu()) * (1.0 if example.label else -1.0)
            minimum_margin = min(minimum_margin, signed)
            exact = exact and correct
            grouped[example.group].append((correct, probability))
    metrics = {
        group: {
            "correct": sum(int(item[0]) for item in items),
            "total": len(items),
            "minimum_probability": min(item[1] for item in items),
            "maximum_probability": max(item[1] for item in items),
        }
        for group, items in sorted(grouped.items())
    }
    return metrics, minimum_margin, exact


def train_v61(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    output_checkpoint = _safe_output_path(args.output_checkpoint, "V61 checkpoint")
    output_report = _safe_output_path(args.training_report, "V61 report")
    config, config_path = _load_sanitized_runtime_config(args.base_runtime_config)
    runtime_config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _resolve(args.base_checkpoint)
    base_checkpoint_sha256, base_files = checkpoint_fingerprint(base_checkpoint)
    records, qa_sha256 = load_training_records(args.train_qa, scene_ids=LOCKED_SCENE_IDS)
    records = sorted(records, key=lambda record: (record.scene_id, record.question_id))
    anchor_keys, expansion_keys = _changed_keys(records)
    changed_keys = anchor_keys | expansion_keys
    prefixes, prefix_manifest = load_prefix_cache(
        args.prefix_cache,
        scene_ids=LOCKED_SCENE_IDS,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
    )

    runtime = StaticRuntimePrefixFactory(config, base_checkpoint, LOCKED_SCENE_IDS[0]).bootstrap
    frozen = freeze_base_runtime(runtime)
    source_path = _resolve(args.source_v60_checkpoint)
    source_control_sha256 = _control_checkpoint_sha256(source_path)
    source_control, source_metadata = _load_control_head(
        source_path,
        hidden_size=runtime.language.hidden_size,
        device=torch.device("cpu"),
    )
    if type(source_control) is not TeacherBasisFullSceneQuestionControlV3:
        raise TypeError("V61 source must be the exact V3/V60 architecture")
    if (
        source_metadata.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or source_metadata.get("base_runtime_config_sha256") != runtime_config_sha256
    ):
        raise ValueError("V61 source V60 belongs to a different frozen base")
    source_weights_sha256 = _sha256_file(source_path / "control.safetensors")
    source_report, source_report_sha256 = _load_source_report(
        args.source_v60_report,
        source_control_sha256=source_control_sha256,
        source_weights_sha256=source_weights_sha256,
    )
    source_state = source_control.state_dict()
    source_state_sha256 = _tensor_state_sha256(source_state)

    control = SceneConditionedGateTeacherBasisControlV4.from_v60(
        source_control, gate_hidden_dim=args.gate_hidden_dim
    ).float()
    inherited_before = inherited_value_state_sha256(control)
    inherited_tensors_exact_before = set(source_state) == set(
        control.inherited_state_names
    ) and all(torch.equal(source_state[name], control.state_dict()[name]) for name in source_state)
    if not inherited_tensors_exact_before or inherited_before != source_state_sha256:
        raise RuntimeError("V61 did not begin from a byte-identical V60 value state")
    trainable_names = tuple(
        name for name, parameter in control.named_parameters() if parameter.requires_grad
    )
    if not trainable_names or any(
        not name.startswith("scene_question_gate.") for name in trainable_names
    ):
        raise RuntimeError("V61 trainable parameter scope is not gate-only")

    signatures = {
        scene_id: control.encode_scene(prefix.float().cpu())
        for scene_id, prefix in prefixes.items()
    }
    question_cache: dict[str, torch.Tensor] = {}
    examples: list[GateExampleV4] = []
    with torch.inference_mode():
        for record in records:
            if record.question not in question_cache:
                question_cache[record.question] = (
                    _pooled_question_embedding(runtime, record.question).float().cpu()
                )
            question = question_cache[record.question]
            normalized = control.normalized_question(question)
            value_trunk = control._value_trunk(signatures[record.scene_id], normalized)
            key = record.scene_id, record.question_id
            label = key in changed_keys
            examples.append(
                GateExampleV4(
                    scene_id=record.scene_id,
                    question_id=record.question_id,
                    feature=value_trunk.detach().flatten(1),
                    label=float(label),
                    group=_group(label, key, anchor_keys),
                )
            )
    features = torch.cat([example.feature for example in examples])
    labels = torch.tensor([example.label for example in examples], dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        control.scene_question_gate.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    started = time.perf_counter()
    maximum_gradient = 0.0
    stable_success = 0
    completed_epochs = 0
    for epoch in range(args.epochs):
        # Full-batch optimization is deterministic; a fixed permutation keeps
        # the training contract explicit without changing the exact objective.
        order = list(range(len(examples)))
        random.Random(args.seed + epoch * 1_000_003).shuffle(order)
        batch_features = features[order]
        batch_labels = labels[order]
        optimizer.zero_grad(set_to_none=True)
        logits = control.scene_question_gate(batch_features).squeeze(-1)
        positive_loss = F.softplus(-logits[batch_labels == 1]).mean()
        negative_loss = F.softplus(logits[batch_labels == 0]).mean()
        signs = batch_labels.mul(2.0).sub(1.0)
        margin_loss = F.relu(args.minimum_signed_logit_margin - signs * logits).square().mean()
        loss = 0.5 * (positive_loss + negative_loss) + args.margin_weight * margin_loss
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            control.scene_question_gate.parameters(), args.gradient_clip_norm
        )
        gradient_value = float(gradient.detach().cpu())
        if not math.isfinite(gradient_value):
            raise RuntimeError("V61 gate gradient is nonfinite")
        maximum_gradient = max(maximum_gradient, gradient_value)
        optimizer.step()
        completed_epochs = epoch + 1
        with torch.inference_mode():
            observed = control.scene_question_gate(features).squeeze(-1)
            signed_margin = (labels.mul(2.0).sub(1.0) * observed).min().item()
            route_exact = bool(torch.equal(observed.ge(0.0), labels.bool()))
        stable_success = (
            stable_success + 1
            if (
                route_exact
                and signed_margin >= args.minimum_signed_logit_margin
                and completed_epochs >= args.minimum_epochs
            )
            else 0
        )
        if completed_epochs % args.log_every == 0 or stable_success == 1:
            _log_event(
                phase="v61_scene_conditioned_gate",
                epoch=completed_epochs,
                loss=float(loss.detach()),
                route_exact=route_exact,
                minimum_signed_logit_margin=signed_margin,
            )
        if stable_success >= args.success_patience:
            break
    elapsed = time.perf_counter() - started
    control.eval()

    route_metrics, minimum_margin, routes_exact = _route_diagnostics(control, examples)
    inherited_after = inherited_value_state_sha256(control)
    inherited_tensors_exact_after = set(source_state) == set(control.inherited_state_names) and all(
        torch.equal(source_state[name], control.state_dict()[name]) for name in source_state
    )
    value_outputs_exact = True
    with torch.inference_mode():
        for record, example in zip(records, examples, strict=True):
            question = question_cache[record.question]
            source_output = source_control.forward_from_signature(
                signatures[record.scene_id], question
            )
            candidate_output = control.forward_from_signature(signatures[record.scene_id], question)
            value_outputs_exact = value_outputs_exact and all(
                torch.equal(first, second)
                for first, second in (
                    (source_output.control_tokens, candidate_output.control_tokens),
                    (
                        source_output.coefficient_directions,
                        candidate_output.coefficient_directions,
                    ),
                    (source_output.control_rms, candidate_output.control_rms),
                )
            )
            if not value_outputs_exact:
                break
    source_optimization = source_report["optimization"]
    retained_prompt_metrics = {
        key: source_optimization[key]
        for key in (
            "mean_prompt_cosine",
            "minimum_prompt_cosine",
            "mean_rms_absolute_error",
            "mean_pair_delta_cosine",
        )
    }
    offline_checks = {
        "inherited_tensors_exact_before_training": inherited_tensors_exact_before,
        "inherited_tensors_exact_after_training": inherited_tensors_exact_after,
        "inherited_state_sha256_unchanged": (
            inherited_before == inherited_after == source_state_sha256
        ),
        "value_outputs_exact_on_all_144": value_outputs_exact,
        "only_gate_trainable": control.inherited_v60_state_frozen
        and all(name.startswith("scene_question_gate.") for name in trainable_names),
        "all_144_training_routes_exact": routes_exact,
        "minimum_signed_logit_margin": minimum_margin >= args.minimum_signed_logit_margin,
        "retained_v60_mean_prompt_cosine": retained_prompt_metrics["mean_prompt_cosine"] >= 0.95,
        "retained_v60_minimum_prompt_cosine": retained_prompt_metrics["minimum_prompt_cosine"]
        >= 0.8,
        "retained_v60_mean_rms_absolute_error": retained_prompt_metrics["mean_rms_absolute_error"]
        <= 0.015,
        "retained_v60_mean_pair_delta_cosine": retained_prompt_metrics["mean_pair_delta_cosine"]
        >= 0.75,
    }
    checkpoint_hashes = save_v4_control_checkpoint(
        output_checkpoint,
        control=control,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=runtime_config_sha256,
        source_v60_checkpoint_sha256=source_control_sha256,
        expected_inherited_state_sha256=source_state_sha256,
    )
    loaded, loaded_metadata = _load_control_head(
        output_checkpoint,
        hidden_size=runtime.language.hidden_size,
        device=torch.device("cpu"),
    )
    strict_reload_exact = (
        isinstance(loaded, SceneConditionedGateTeacherBasisControlV4)
        and set(loaded.state_dict()) == set(control.state_dict())
        and all(
            torch.equal(loaded.state_dict()[name], control.state_dict()[name])
            for name in control.state_dict()
        )
        and loaded_metadata.get("inherited_value_state_sha256") == source_state_sha256
    )
    offline_checks["strict_saved_runtime_reload_exact"] = strict_reload_exact
    report = {
        "schema_version": 1,
        "artifact": "v61_scene_conditioned_gate_training",
        "offline_checks_passed": all(offline_checks.values()),
        "offline_checks": offline_checks,
        "promotion_eligible": False,
        "saved_runtime_generation_gate_required": True,
        "source": {
            "v60_checkpoint_sha256": source_control_sha256,
            "v60_weights_sha256": source_weights_sha256,
            "v60_state_sha256": source_state_sha256,
            "v60_report_sha256": source_report_sha256,
            "retained_prompt_metrics": retained_prompt_metrics,
        },
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
            "v61_generalization_gate_loaded": False,
            "v61_preregistration_loaded": False,
            "v61_baseline_lock_loaded": False,
        },
        "architecture": {
            "name": "scene_conditioned_gate_teacher_basis_control_v4",
            "gate_parameter_count": control.gate_parameter_count,
            "saved_parameter_count": control.parameter_count,
            "inherited_value_state_frozen": control.inherited_v60_state_frozen,
            "scene_signature_shape": list(signatures[LOCKED_SCENE_IDS[0]].shape),
            "boundary_tokens_excluded": True,
            "softmax_scene_attention_used": False,
            "control_values_scene_question_bilinear": True,
            "gate_scene_question_conditioned": True,
            "question_dependent_scene_retrieval": False,
        },
        "optimization": {
            "configured_epochs": args.epochs,
            "completed_epochs": completed_epochs,
            "elapsed_seconds": elapsed,
            "maximum_preclip_gradient_norm": maximum_gradient,
            "minimum_signed_logit_margin": minimum_margin,
            "route_metrics": route_metrics,
        },
        "checkpoint": checkpoint_hashes,
        "scope": {
            "base_scene_stack_frozen": frozen["all_parameters_frozen"],
            "gemma_backward_used": False,
            "gemma_generation_used": False,
            "cached_v60_values_only": True,
            "only_scene_question_gate_trained": True,
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
    parser.add_argument("--source-v60-checkpoint", required=True)
    parser.add_argument("--source-v60-report", required=True)
    parser.add_argument("--train-qa", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--seed", type=int, default=61061)
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--minimum-epochs", type=int, default=200)
    parser.add_argument("--success-patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=2.0)
    parser.add_argument("--minimum-signed-logit-margin", type=float, default=4.0)
    parser.add_argument("--margin-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = train_v61(args)
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


__all__ = ["main", "train_v61"]
