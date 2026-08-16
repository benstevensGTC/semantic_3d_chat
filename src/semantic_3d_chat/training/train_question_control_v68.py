"""V68 regularized paired-scene grid and fail-closed publication.

V68 is a failure-driven successor to V67.  It changes no numeric or behavior
gate.  Instead it screens three preregistered regularization arms in a fixed
order using only leave-one-counterfactual-pair-out numeric evidence.  The
first arm passing every V67 numeric gate is selected and later arms are
skipped.  No Gemma generation is permitted until that create-once screen
authorizes an exact selected arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    TRAIN_PAIR_IDS,
)
from semantic_3d_chat.evaluation.v67_pair_objective_preregistration import (
    V67_NUMERIC_SCREEN_THRESHOLDS,
)
from semantic_3d_chat.evaluation.v68_regularized_pair_preregistration import (
    V68_ARM_GRID,
    V68_COMMON_HYPERPARAMETERS,
    build_v68_preregistration,
    implementation_source_hashes_v68,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training import train_question_control_v66 as v66
from semantic_3d_chat.training import train_question_control_v67 as v67
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
from semantic_3d_chat.training.question_control_v68_objective import (
    hard_negative_prototype_margin_loss_v68,
    relative_parameter_anchor_loss_v68,
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
_WORK_ARTIFACT: Final[str] = "v68_regularized_pair_work_v1"
_FOLD_ARTIFACT: Final[str] = "v68_regularized_pair_fold_v1"
_SCREEN_ARTIFACT: Final[str] = "v68_regularized_pair_numeric_grid_v1"
_FULL_ARTIFACT: Final[str] = "v68_regularized_pair_behavioral_training_v1"
_SELECTION_RULE: Final[str] = "run_in_declared_order_and_select_first_all_gate_pass"


@dataclass(frozen=True)
class V68FitResult:
    base: v66.V66FitResult
    refinement_optimizer_steps: int
    refinement_elapsed_seconds: float
    train_pair_diagnostics: dict[str, float | int]
    train_prototype_top1_accuracy: float
    train_hard_negative_diagnostics: dict[str, float]
    optimizer_scope: tuple[str, ...]
    anchor_state_sha256: str

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


def _arm_by_id(arm_id: str) -> dict[str, str | int | float]:
    matches = [dict(arm) for arm in V68_ARM_GRID if arm["arm_id"] == arm_id]
    if len(matches) != 1:
        raise ValueError(f"V68 arm is not preregistered: {arm_id}")
    return matches[0]


def _arm_sha256(arm: Mapping[str, Any]) -> str:
    return _canonical_sha256(dict(arm))


def validate_v68_preregistration(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V68 preregistration is unavailable")
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected = build_v68_preregistration()
    if payload != expected:
        raise ValueError("V68 preregistration or an implementation source differs from its lock")
    if payload.get("implementation_source_hashes") != (implementation_source_hashes_v68()):
        raise ValueError("V68 implementation source lock changed")
    return payload


def _locked_fit_args(
    args: argparse.Namespace,
    arm: Mapping[str, str | int | float],
) -> argparse.Namespace:
    """Build a V63/V66-compatible namespace from only preregistered values."""

    hp = V68_COMMON_HYPERPARAMETERS
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
            "prototype_classification_epochs": hp["prototype_classification_epochs"],
            "prototype_classification_weight": hp["prototype_classification_weight"],
            "prototype_classification_temperature": hp["prototype_classification_temperature"],
            "prototype_value_preservation_weight": hp["prototype_value_preservation_weight"],
            "pair_delta_weight": 0.0,
            "route_weight": 0.0,
            "log_every": 20,
            "v68_arm": dict(arm),
        }
    )
    return argparse.Namespace(**values)


def _regularized_parameters(
    control: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    *,
    optimizer_scope: str,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Apply one of the two preregistered, explicitly bounded scopes."""

    for parameter in control.parameters():
        parameter.requires_grad_(False)
    all_value_prefixes = (
        "scene_projection.",
        "question_projection.",
        "control_trunk.",
        "coefficient_output.",
        "magnitude_output.",
    )
    interaction_prefixes = (
        "scene_projection.",
        "question_projection.",
        "control_trunk.",
    )
    if optimizer_scope == "all_value":
        allowed = all_value_prefixes
    elif optimizer_scope == "interaction_only":
        allowed = interaction_prefixes
    else:
        raise ValueError("V68 optimizer scope is not preregistered")
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in control.named_parameters():
        if name.startswith(allowed):
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    selected_names = {name for name, _parameter in selected}
    expected_names = {
        name for name, _parameter in control.named_parameters() if name.startswith(allowed)
    }
    if (
        not selected
        or selected_names != expected_names
        or any(parameter.requires_grad for parameter in control.question_norm.parameters())
        or any(
            parameter.requires_grad
            for name, parameter in control.named_parameters()
            if name.startswith("route_")
        )
        or any(
            parameter.requires_grad
            for name, parameter in control.named_parameters()
            if not name.startswith(allowed)
        )
    ):
        raise RuntimeError("V68 optimizer scope changed")
    return tuple(selected)


def _anchor_sha256(anchors: Mapping[str, torch.Tensor]) -> str:
    return _canonical_sha256(
        {name: _tensor_sha256(value) for name, value in sorted(anchors.items())}
    )


def _fit_regularized_pair(
    *,
    rows: Sequence[V63Row],
    codebook: HybridAnswerPrototypeCodebookV66,
    preflight: V63Preflight,
    questions: Mapping[tuple[str, str], torch.Tensor],
    basis: torch.Tensor,
    args: argparse.Namespace,
    arm: Mapping[str, str | int | float],
    seed: int,
    phase: str,
) -> V68FitResult:
    """Run the unchanged V66 base fit, then one locked V68 refinement arm."""

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
        raise ValueError("V68 pair refinement requires changed counterfactual units")
    for left, right in units:
        if (
            left.question.encode("utf-8") != right.question.encode("utf-8")
            or not torch.equal(questions[left.key], questions[right.key])
            or answer_class_id_v66(left.answer) == answer_class_id_v66(right.answer)
            or _tensor_sha256(codebook.targets[left.key])
            == _tensor_sha256(codebook.targets[right.key])
        ):
            raise ValueError("V68 changed unit is not an exact paired opposite")

    _class_ids, prototype_bank, class_index = v67._prototype_bank(codebook)
    retention = [row for row in rows if not row.route_label]
    if not retention:
        raise ValueError("V68 requires stable rows for retention")
    named_parameters = _regularized_parameters(control, optimizer_scope=str(arm["optimizer_scope"]))
    anchors = {
        name: parameter.detach().cpu().float().contiguous().clone()
        for name, parameter in named_parameters
    }
    anchor_state_sha256 = _anchor_sha256(anchors)
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in named_parameters],
        lr=float(arm["pair_learning_rate"]),
        weight_decay=float(V68_COMMON_HYPERPARAMETERS["base_weight_decay"]),
    )
    steps = 0
    started = time.perf_counter()
    epochs = int(arm["pair_refinement_epochs"])
    repeats = int(arm["pair_refinement_repeats"])
    unit_batch_size = int(V68_COMMON_HYPERPARAMETERS["pair_unit_batch_size"])
    retention_batch_size = int(V68_COMMON_HYPERPARAMETERS["retention_batch_size"])
    retention_batches = int(arm["retention_batches_per_epoch"])
    for epoch in range(epochs):
        for repeat in range(repeats):
            ordered_units = list(units)
            random.Random(seed + epoch * 1_000_003 + repeat * 10_007).shuffle(ordered_units)
            for offset in range(0, len(ordered_units), unit_batch_size):
                batch_units = ordered_units[offset : offset + unit_batch_size]
                flat = [row for unit in batch_units for row in unit]
                output = control.forward_from_signature(
                    torch.cat([signatures[row.scene_id] for row in flat]),
                    torch.cat([questions[row.key] for row in flat]),
                ).control_tokens
                target = torch.cat([codebook.targets[row.key] for row in flat])
                pair_loss, _pair_diagnostics = paired_scene_dependence_loss_v67(
                    output.reshape(len(batch_units), 2, 4, 1536),
                    target.reshape(len(batch_units), 2, 4, 1536),
                    opposite_margin=float(arm["pair_opposite_margin"]),
                    value_weight=float(arm["pair_value_weight"]),
                    delta_weight=float(arm["pair_delta_weight"]),
                    opposite_weight=float(arm["pair_opposite_weight"]),
                )
                indices = torch.tensor(
                    [class_index[codebook.class_by_key[row.key]] for row in flat],
                    dtype=torch.long,
                    device=output.device,
                )
                class_loss, _class_diagnostics = numeric_prototype_classification_loss(
                    output,
                    prototype_bank,
                    indices,
                    temperature=float(
                        V68_COMMON_HYPERPARAMETERS["prototype_classification_temperature"]
                    ),
                )
                hard_loss, _hard_diagnostics = hard_negative_prototype_margin_loss_v68(
                    output,
                    prototype_bank,
                    indices,
                    margin=float(arm["hard_negative_margin"]),
                )
                anchor_loss = relative_parameter_anchor_loss_v68(
                    named_parameters,
                    anchors,
                    scale_floor=float(V68_COMMON_HYPERPARAMETERS["anchor_scale_floor"]),
                )
                loss = (
                    pair_loss
                    + float(arm["pair_classification_weight"]) * class_loss
                    + float(arm["hard_negative_weight"]) * hard_loss
                    + float(arm["anchor_weight"]) * anchor_loss
                )
                _optimizer_step(
                    loss=loss,
                    control=control,
                    optimizer=optimizer,
                    gradient_clip_norm=float(V68_COMMON_HYPERPARAMETERS["gradient_clip_norm"]),
                )
                steps += 1

        ordered_retention = list(retention)
        random.Random(seed + 70_000_019 + epoch * 1_000_003).shuffle(ordered_retention)
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
                device=output.device,
            )
            class_loss, _class_diagnostics = numeric_prototype_classification_loss(
                output,
                prototype_bank,
                indices,
                temperature=float(
                    V68_COMMON_HYPERPARAMETERS["prototype_classification_temperature"]
                ),
            )
            hard_loss, _hard_diagnostics = hard_negative_prototype_margin_loss_v68(
                output,
                prototype_bank,
                indices,
                margin=float(arm["hard_negative_margin"]),
            )
            anchor_loss = relative_parameter_anchor_loss_v68(
                named_parameters,
                anchors,
                scale_floor=float(V68_COMMON_HYPERPARAMETERS["anchor_scale_floor"]),
            )
            retention_loss = (
                v67._simple_value_loss(output, target)
                + class_loss
                + float(arm["hard_negative_weight"]) * hard_loss
                + float(arm["anchor_weight"]) * anchor_loss
            )
            _optimizer_step(
                loss=float(arm["retention_weight"]) * retention_loss,
                control=control,
                optimizer=optimizer,
                gradient_clip_norm=float(V68_COMMON_HYPERPARAMETERS["gradient_clip_norm"]),
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
    pair_diagnostics = v67._measure_train_pairs(
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
            device=predicted.device,
        )
        _loss, prototype_diagnostics = numeric_prototype_classification_loss(
            predicted,
            prototype_bank,
            indices,
            temperature=float(V68_COMMON_HYPERPARAMETERS["prototype_classification_temperature"]),
        )
        _hard_loss, hard_diagnostics = hard_negative_prototype_margin_loss_v68(
            predicted,
            prototype_bank,
            indices,
            margin=float(arm["hard_negative_margin"]),
        )
    rebuilt_base = v66.V66FitResult(
        control=control,
        signatures=signatures,
        base_fit=base.base_fit,
        classification_optimizer_steps=base.classification_optimizer_steps,
        numeric_prototype_top1_accuracy=base.numeric_prototype_top1_accuracy,
        numeric_prototype_mean_margin=base.numeric_prototype_mean_margin,
    )
    return V68FitResult(
        base=rebuilt_base,
        refinement_optimizer_steps=steps,
        refinement_elapsed_seconds=time.perf_counter() - started,
        train_pair_diagnostics=pair_diagnostics,
        train_prototype_top1_accuracy=float(prototype_diagnostics.top1_accuracy.cpu()),
        train_hard_negative_diagnostics={
            "mean_own_cosine": float(hard_diagnostics.mean_own_cosine.cpu()),
            "mean_hardest_wrong_cosine": float(hard_diagnostics.mean_hardest_wrong_cosine.cpu()),
            "mean_own_over_hardest_wrong_margin": float(
                hard_diagnostics.mean_own_over_hardest_wrong_margin.cpu()
            ),
            "positive_margin_fraction": float(hard_diagnostics.positive_margin_fraction.cpu()),
        },
        optimizer_scope=tuple(name for name, _parameter in named_parameters),
        anchor_state_sha256=anchor_state_sha256,
    )


def _fit_audit(fit: V68FitResult) -> dict[str, Any]:
    return {
        "base_optimizer_steps": fit.base.base_fit.optimizer_steps,
        "base_classification_optimizer_steps": fit.base.classification_optimizer_steps,
        "pair_refinement_optimizer_steps": fit.refinement_optimizer_steps,
        "base_elapsed_seconds": fit.base.base_fit.elapsed_seconds,
        "pair_refinement_elapsed_seconds": fit.refinement_elapsed_seconds,
        "question_norm_sha256": fit.base.base_fit.question_norm_sha256,
        "question_norm_frozen": fit.base.base_fit.question_norm_frozen,
        "base_numeric_prototype_top1_accuracy": (fit.base.numeric_prototype_top1_accuracy),
        "post_refinement_numeric_prototype_top1_accuracy": (fit.train_prototype_top1_accuracy),
        "train_pair_diagnostics": fit.train_pair_diagnostics,
        "train_hard_negative_diagnostics": fit.train_hard_negative_diagnostics,
        "optimizer_parameter_names": list(fit.optimizer_scope),
        "anchor_state_sha256": fit.anchor_state_sha256,
    }


def _training_identity(
    preflight: V63Preflight,
    teacher_audit: Mapping[str, Any],
    args: argparse.Namespace,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "artifact": "v68_regularized_pair_training_identity_v1",
        "preregistration_sha256": _sha256_file(_resolve(args.preregistration)),
        "filtered_training_qa_sha256": preflight.filtered_train_sha256,
        "training_baseline_lock_sha256": _sha256_file(_resolve(args.training_baseline_lock)),
        "base_checkpoint_sha256": preflight.base_checkpoint_sha256,
        "runtime_config_sha256": preflight.runtime_config_sha256,
        "prefix_cache_manifest_sha256": preflight.prefix_manifest_sha256,
        "combined_teacher_audit_sha256": _canonical_sha256(teacher_audit),
        "common_hyperparameters": dict(V68_COMMON_HYPERPARAMETERS),
        "ordered_arm_grid": [dict(arm) for arm in V68_ARM_GRID],
        "implementation_source_hashes": dict(preregistration["implementation_source_hashes"]),
        "pair_ids": list(TRAIN_PAIR_IDS),
        "selection_rule": _SELECTION_RULE,
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "internal_validation_loaded": False,
        "deferred_final_loaded": False,
    }
    return {**identity, "sha256": _canonical_sha256(identity)}


def _validate_cached_fold_v68(
    payload: object,
    *,
    mode: str,
    arm: Mapping[str, Any],
    held_pair_id: str,
    run_signature_sha256: str,
    held_rows: Sequence[V63Row],
    codebook: HybridAnswerPrototypeCodebookV66,
    basis: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("V68 cached fold must be an object")
    expected_arm_sha = _arm_sha256(arm)
    if (
        payload.get("artifact") != _FOLD_ARTIFACT
        or payload.get("arm_id") != arm["arm_id"]
        or payload.get("arm_sha256") != expected_arm_sha
    ):
        raise ValueError("V68 cached fold arm provenance changed")
    v67_payload = {
        key: value for key, value in payload.items() if key not in {"arm_id", "arm_sha256"}
    }
    v67_payload["artifact"] = v67._FOLD_ARTIFACT
    v67._validate_cached_fold_v67(
        v67_payload,
        mode=mode,
        held_pair_id=held_pair_id,
        run_signature_sha256=run_signature_sha256,
        held_rows=held_rows,
        codebook=codebook,
        basis=basis,
    )
    return payload


def _held_fit(
    fit: V68FitResult,
    held_rows: Sequence[V63Row],
    preflight: V63Preflight,
) -> V68FitResult:
    signatures = _scene_signatures(
        fit.control,
        {
            scene_id: preflight.prefixes[scene_id]
            for scene_id in sorted({row.scene_id for row in held_rows})
        },
    )
    base = v66.V66FitResult(
        control=fit.control,
        signatures=signatures,
        base_fit=fit.base.base_fit,
        classification_optimizer_steps=fit.base.classification_optimizer_steps,
        numeric_prototype_top1_accuracy=fit.base.numeric_prototype_top1_accuracy,
        numeric_prototype_mean_margin=fit.base.numeric_prototype_mean_margin,
    )
    return V68FitResult(
        base=base,
        refinement_optimizer_steps=fit.refinement_optimizer_steps,
        refinement_elapsed_seconds=fit.refinement_elapsed_seconds,
        train_pair_diagnostics=fit.train_pair_diagnostics,
        train_prototype_top1_accuracy=fit.train_prototype_top1_accuracy,
        train_hard_negative_diagnostics=fit.train_hard_negative_diagnostics,
        optimizer_scope=fit.optimizer_scope,
        anchor_state_sha256=fit.anchor_state_sha256,
    )


def _run_arm_folds(
    *,
    mode: str,
    arm: Mapping[str, str | int | float],
    preflight: V63Preflight,
    teachers: Mapping[tuple[str, str], torch.Tensor],
    bundle: V65RuntimeBundle,
    args: argparse.Namespace,
    training_identity: Mapping[str, Any],
    screen_authorization_sha256: str | None,
    generator_fn: Callable[..., str],
) -> dict[str, Any]:
    fit_args = _locked_fit_args(args, arm)
    arm_id = str(arm["arm_id"])
    arm_hash = _arm_sha256(arm)
    work = _resolve(args.work_directory) / arm_id
    run_manifest = {
        "schema_version": 1,
        "artifact": _WORK_ARTIFACT,
        "mode": mode,
        "training_identity_sha256": training_identity["sha256"],
        "arm_id": arm_id,
        "arm_sha256": arm_hash,
        "screen_authorization_sha256": screen_authorization_sha256,
    }
    run_manifest["run_signature_sha256"] = _canonical_sha256(run_manifest)
    v66._prepare_work_directory(work, run_manifest)
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
            scope=f"v68_{arm_id}_{held_pair}",
            forbidden_pair_id=held_pair,
        )
        basis = v66._codebook_basis(codebook, int(fit_args.basis_rank))
        fold_path = work / f"fold_{held_pair}.json"
        if fold_path.exists():
            cached = _validate_cached_fold_v68(
                json.loads(fold_path.read_text(encoding="utf-8")),
                mode=mode,
                arm=arm,
                held_pair_id=held_pair,
                run_signature_sha256=str(run_manifest["run_signature_sha256"]),
                held_rows=held_rows,
                codebook=codebook,
                basis=basis,
            )
            fold_payloads.append(cached)
            continue
        fit = _fit_regularized_pair(
            rows=train_rows,
            codebook=codebook,
            preflight=preflight,
            questions=bundle.question_embeddings,
            basis=basis,
            args=fit_args,
            arm=arm,
            seed=int(fit_args.seed) + (fold_index + 1) * 100_003,
            phase=f"v68_{arm_id}_{held_pair}",
        )
        held_fit = _held_fit(fit, held_rows, preflight)
        numeric_metrics, numeric_evidence = v67.numeric_screen_fold_v67(
            held_fit,
            held_rows,
            codebook=codebook,
            questions=bundle.question_embeddings,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact": _FOLD_ARTIFACT,
            "run_signature_sha256": run_manifest["run_signature_sha256"],
            "arm_id": arm_id,
            "arm_sha256": arm_hash,
            "held_pair_id": held_pair,
            "held_rows_used_for_optimization": False,
            "held_teacher_sources_used": False,
            "fold_codebook_sha256": codebook.sha256,
            "fold_basis_sha256": _tensor_sha256(basis),
            "fit": _fit_audit(held_fit),
            "numeric_screen": numeric_metrics,
            "numeric_evidence": list(numeric_evidence),
        }
        if mode == "full":
            records = v66.generate_supported_rows_v66(
                held_fit.base,
                held_rows,
                questions=bundle.question_embeddings,
                supported_classes=set(codebook.prototypes),
                bundle=bundle,
                prefixes=preflight.prefixes,
                generator_fn=generator_fn,
            )
            payload["behavior"] = v66.behavior_metrics_v66(
                records, unsupported_count=len(held_rows) - len(records)
            )
            payload["records"] = list(records)
        v66._atomic_new_json(fold_path, payload)
        fold_payloads.append(payload)

    metrics = v67.aggregate_numeric_screens_v67([fold["numeric_screen"] for fold in fold_payloads])
    checks = v67.assess_numeric_screen_v67(metrics)
    result: dict[str, Any] = {
        "arm_id": arm_id,
        "arm_sha256": arm_hash,
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "metrics": metrics,
        "checks": checks,
        "gemma_generation_used": mode == "full",
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
        "_fold_payloads": fold_payloads,
    }
    return result


def _public_arm_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "_fold_payloads"}


def _skipped_arm_result(arm: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arm_id": arm["arm_id"],
        "arm_sha256": _arm_sha256(arm),
        "status": "skipped_after_first_pass",
        "passed": None,
        "metrics": None,
        "checks": None,
        "gemma_generation_used": False,
        "folds": [],
    }


def _screen_report(
    *,
    preflight: V63Preflight,
    preregistration_sha256: str,
    training_identity: Mapping[str, Any],
    teacher_audit: Mapping[str, Any],
    arm_results: Sequence[Mapping[str, Any]],
    selected_arm_id: str | None,
) -> dict[str, Any]:
    passed = selected_arm_id is not None
    return {
        "schema_version": 1,
        "artifact": _SCREEN_ARTIFACT,
        "passed": passed,
        "promotion_eligible": False,
        "checkpoint_published": False,
        "gemma_generation_used": False,
        "terminal_reason": (
            "numeric_grid_passed_selected_arm_authorized"
            if passed
            else "all_numeric_grid_arms_failed_no_generation_or_checkpoint_authorized"
        ),
        "preregistration_sha256": preregistration_sha256,
        "training_identity_sha256": training_identity["sha256"],
        "implementation_source_hashes": dict(training_identity["implementation_source_hashes"]),
        "authorization": {
            "baseline_lock_sha256": preflight.baseline_lock_sha256,
            "filtered_training_qa_sha256": preflight.filtered_train_sha256,
            "teacher_audit_sha256": _canonical_sha256(teacher_audit),
        },
        "selection": {
            "rule": _SELECTION_RULE,
            "selected_arm_id": selected_arm_id,
            "selected_arm_sha256": (
                None if selected_arm_id is None else _arm_sha256(_arm_by_id(selected_arm_id))
            ),
            "later_arms_skipped_after_first_pass": True,
        },
        "thresholds": asdict(V67_NUMERIC_SCREEN_THRESHOLDS),
        "arm_results": [dict(result) for result in arm_results],
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


def _recompute_arm_checks(result: Mapping[str, Any]) -> dict[str, bool]:
    folds = result.get("folds")
    if not isinstance(folds, list) or len(folds) != len(TRAIN_PAIR_IDS):
        raise ValueError("V68 screen arm fold inventory changed")
    observed_pairs = [fold.get("held_pair_id") for fold in folds if isinstance(fold, dict)]
    if observed_pairs != list(TRAIN_PAIR_IDS):
        raise ValueError("V68 screen arm fold order changed")
    metrics = v67.aggregate_numeric_screens_v67([fold["numeric_screen"] for fold in folds])
    if result.get("metrics") != metrics:
        raise ValueError("V68 screen arm metrics differ from its folds")
    return v67.assess_numeric_screen_v67(metrics)


def validate_screen_authorization_v68(
    path: str | Path,
    *,
    expected_training_identity_sha256: str,
    expected_preregistration_sha256: str,
) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V68 full run requires a regular screen report")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V68 screen authorization must be an object")
    selection = payload.get("selection")
    results = payload.get("arm_results")
    scope = payload.get("scope")
    if (
        payload.get("artifact") != _SCREEN_ARTIFACT
        or payload.get("passed") is not True
        or payload.get("checkpoint_published") is not False
        or payload.get("gemma_generation_used") is not False
        or payload.get("training_identity_sha256") != expected_training_identity_sha256
        or payload.get("preregistration_sha256") != expected_preregistration_sha256
        or payload.get("implementation_source_hashes") != implementation_source_hashes_v68()
        or payload.get("thresholds") != asdict(V67_NUMERIC_SCREEN_THRESHOLDS)
        or not isinstance(selection, dict)
        or selection.get("rule") != _SELECTION_RULE
        or not isinstance(results, list)
        or len(results) != len(V68_ARM_GRID)
        or not isinstance(scope, dict)
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
        raise ValueError("V68 screen authorization changed or did not pass")
    selected_arm_id = selection.get("selected_arm_id")
    selected_index: int | None = None
    for index, (arm, result) in enumerate(zip(V68_ARM_GRID, results, strict=True)):
        if not isinstance(result, dict):
            raise TypeError("V68 screen arm result must be an object")
        if (
            result.get("arm_id") != arm["arm_id"]
            or result.get("arm_sha256") != _arm_sha256(arm)
            or result.get("gemma_generation_used") is not False
        ):
            raise ValueError("V68 screen arm provenance changed")
        if result.get("status") == "skipped_after_first_pass":
            if (
                any(result.get(field) is not None for field in ("passed", "metrics", "checks"))
                or result.get("folds") != []
            ):
                raise ValueError("V68 skipped arm contains result evidence")
            if selected_index is None:
                raise ValueError("V68 skipped an arm before a passing selection")
            continue
        if selected_index is not None:
            raise ValueError("V68 executed an arm after the first passing arm")
        expected_checks = _recompute_arm_checks(result)
        if (
            result.get("checks") != expected_checks
            or result.get("passed") != all(expected_checks.values())
            or result.get("status") != ("passed" if all(expected_checks.values()) else "failed")
        ):
            raise ValueError("V68 screen arm checks changed")
        if all(expected_checks.values()):
            selected_index = index
    if selected_index is None:
        raise ValueError("V68 screen authorization has no passing arm")
    selected_arm = V68_ARM_GRID[selected_index]
    if selected_arm_id != selected_arm["arm_id"] or selection.get(
        "selected_arm_sha256"
    ) != _arm_sha256(selected_arm):
        raise ValueError("V68 selected arm differs from first passing arm")
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "training_identity_sha256": expected_training_identity_sha256,
        "selected_arm_id": selected_arm_id,
        "selected_arm_sha256": _arm_sha256(selected_arm),
        "passed": True,
    }


def train_v68(
    args: argparse.Namespace,
    *,
    runtime_provider: Callable[..., V65RuntimeBundle] | None = None,
    generator_fn: Callable[..., str] | None = None,
    supplemental_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    preregistration = validate_v68_preregistration(args.preregistration)
    preregistration_sha256 = _sha256_file(_resolve(args.preregistration))
    first_fit_args = _locked_fit_args(args, V68_ARM_GRID[0])
    preflight = build_v63_preflight(first_fit_args)
    if len(preflight.rows) != _EXPECTED_ROWS:
        raise ValueError("V68 authenticated training inventory changed")
    validate_training_baseline_lock(args.training_baseline_lock, expected_rows=preflight.rows)
    teachers, teacher_audit = v66.load_combined_verified_teachers_v66(
        preflight,
        args.supplemental_teacher_cache,
        supplemental_loader=supplemental_loader,
    )
    training_identity = _training_identity(
        preflight, teacher_audit, first_fit_args, preregistration
    )
    if args.mode == "full" and not args.screen_authorization:
        raise ValueError("V68 full mode requires a passed screen authorization")
    screen_authorization = (
        None
        if args.mode == "screen"
        else validate_screen_authorization_v68(
            args.screen_authorization,
            expected_training_identity_sha256=str(training_identity["sha256"]),
            expected_preregistration_sha256=preregistration_sha256,
        )
    )
    work = _resolve(args.work_directory)
    if work in {preflight.output_checkpoint, preflight.training_report}:
        raise ValueError("V68 work directory overlaps a publication destination")
    bundle = (runtime_provider or _load_runtime)(preflight, requested_device=args.device)
    generation = generator_fn or v66._generate_with_control

    if args.mode == "screen":
        public_results: list[dict[str, Any]] = []
        selected_arm_id: str | None = None
        for arm in V68_ARM_GRID:
            if selected_arm_id is not None:
                public_results.append(_skipped_arm_result(arm))
                continue
            result = _run_arm_folds(
                mode="screen",
                arm=arm,
                preflight=preflight,
                teachers=teachers,
                bundle=bundle,
                args=args,
                training_identity=training_identity,
                screen_authorization_sha256=None,
                generator_fn=generation,
            )
            public_results.append(_public_arm_result(result))
            if result["passed"] is True:
                selected_arm_id = str(arm["arm_id"])
        report = _screen_report(
            preflight=preflight,
            preregistration_sha256=preregistration_sha256,
            training_identity=training_identity,
            teacher_audit=teacher_audit,
            arm_results=public_results,
            selected_arm_id=selected_arm_id,
        )
        _write_training_report(preflight.training_report, report)
        return report

    if screen_authorization is None:
        raise RuntimeError("V68 full mode lost its screen authorization")
    selected_arm = _arm_by_id(str(screen_authorization["selected_arm_id"]))
    arm_result = _run_arm_folds(
        mode="full",
        arm=selected_arm,
        preflight=preflight,
        teachers=teachers,
        bundle=bundle,
        args=args,
        training_identity=training_identity,
        screen_authorization_sha256=str(screen_authorization["sha256"]),
        generator_fn=generation,
    )
    if arm_result["passed"] is not True:
        raise RuntimeError("V68 full run reproduced a failed numeric screen after authorization")
    fold_payloads = arm_result["_fold_payloads"]
    all_records = [record for fold in fold_payloads for record in fold["records"]]
    unsupported = sum(int(fold["behavior"]["unsupported_total"]) for fold in fold_payloads)
    cv_metrics = v66.behavior_metrics_v66(all_records, unsupported_count=unsupported)
    cv_checks = v66.assess_cv_v66(cv_metrics)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": _FULL_ARTIFACT,
        "promotion_eligible": False,
        "checkpoint": None,
        "terminal_reason": "pair_disjoint_behavior_gate_failed",
        "preregistration_sha256": preregistration_sha256,
        "training_identity_sha256": training_identity["sha256"],
        "implementation_source_hashes": dict(training_identity["implementation_source_hashes"]),
        "screen_authorization": screen_authorization,
        "selected_arm": dict(selected_arm),
        "architecture": {
            "name": "always_on_teacher_basis_full_scene_control_v7",
            "complete_scene_prefix": True,
            "scene_latents": 256,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "runtime_answer_codebook": False,
        },
        "common_hyperparameters": dict(V68_COMMON_HYPERPARAMETERS),
        "teacher_audit": teacher_audit,
        "numeric_screen_reproduced": {
            key: arm_result[key] for key in ("metrics", "checks", "passed")
        },
        "cv": {
            "protocol": "leave_one_counterfactual_pair_out_v68_selected_arm",
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

    fit_args = _locked_fit_args(args, selected_arm)
    all_codebook = build_hybrid_answer_prototype_codebook_v66(
        preflight.rows,
        teachers,
        expected_class_count=_EXPECTED_CLASSES,
        scope="v68_final_all_training",
    )
    final_basis = v66._codebook_basis(all_codebook, int(fit_args.basis_rank))
    final_fit = _fit_regularized_pair(
        rows=preflight.rows,
        codebook=all_codebook,
        preflight=preflight,
        questions=bundle.question_embeddings,
        basis=final_basis,
        args=fit_args,
        arm=selected_arm,
        seed=int(fit_args.seed) + 9_999_991,
        phase="v68_final_all_training",
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
            {name: _tensor_sha256(value) for name, value in final_fit.control.state_dict().items()}
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

    report["terminal_reason"] = "all_v68_training_gates_passed_checkpoint_saved"
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
        raise ValueError("V68 full mode requires --screen-authorization")
    if args.mode == "screen" and args.screen_authorization:
        raise ValueError("V68 screen mode cannot consume a screen authorization")
    random.seed(int(V68_COMMON_HYPERPARAMETERS["seed"]))
    torch.manual_seed(int(V68_COMMON_HYPERPARAMETERS["seed"]))
    report = train_v68(args)
    print(json.dumps(report, sort_keys=True))
    if args.mode == "screen":
        return 0 if report.get("passed") is True else 1
    return (
        0
        if report.get("terminal_reason") == "all_v68_training_gates_passed_checkpoint_saved"
        and isinstance(report.get("checkpoint"), Mapping)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V68FitResult",
    "train_v68",
    "validate_screen_authorization_v68",
    "validate_v68_preregistration",
]
