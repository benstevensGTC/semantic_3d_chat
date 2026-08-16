"""Train-only development folds for V72 question-adaptive branch fusion.

This module intentionally has no Gemma generation or checkpoint publication
path.  It fits on eleven counterfactual-pair groups and evaluates the omitted
training pair only after optimization.  V72 first reproduces V71's two fitted
complete-scene branches, then calibrates only a small continuous question gate
against fold-local numeric teachers while both branches remain frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    TRAIN_PAIR_IDS,
)
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisControlOutput,
)
from semantic_3d_chat.scene_encoder.question_control_v72 import (
    AdaptiveMultiscaleTeacherBasisControlV72,
)
from semantic_3d_chat.training import train_question_control_v66 as v66
from semantic_3d_chat.training import train_question_control_v67 as v67
from semantic_3d_chat.training import train_question_control_v68 as v68
from semantic_3d_chat.training import train_question_control_v69 as v69
from semantic_3d_chat.training import train_question_control_v71 as v71
from semantic_3d_chat.training.question_control_v66_objective import (
    numeric_prototype_classification_loss,
)
from semantic_3d_chat.training.question_control_v66_prototypes import (
    build_hybrid_answer_prototype_codebook_v66,
)
from semantic_3d_chat.training.question_control_v67_objective import (
    paired_scene_dependence_loss_v67,
)
from semantic_3d_chat.training.question_control_v68_objective import (
    hard_negative_prototype_margin_loss_v68,
)
from semantic_3d_chat.training.train_question_control_v56 import _resolve
from semantic_3d_chat.training.train_question_control_v63 import (
    FitResult,
    V63Preflight,
    V63Row,
    _changed_units,
    _scene_signatures,
    build_v63_preflight,
)
from semantic_3d_chat.training.train_question_control_v65 import (
    V65RuntimeBundle,
    _load_runtime,
    validate_training_baseline_lock,
)


_EXPECTED_ROWS: Final[int] = 576
V72_GATE_HIDDEN_SIZE: Final[int] = 64
V72_CALIBRATION_EPOCHS: Final[int] = 100
V72_CALIBRATION_LEARNING_RATE: Final[float] = 0.002
V72_CALIBRATION_WEIGHT_DECAY: Final[float] = 0.001
V72_SELECTION_WEIGHT: Final[float] = 1.0
V72_PAIR_WEIGHT: Final[float] = 2.0
V72_CLASSIFICATION_WEIGHT: Final[float] = 0.5
V72_HARD_NEGATIVE_WEIGHT: Final[float] = 0.5
V72_RETENTION_WEIGHT: Final[float] = 0.5
V72_GRADIENT_CLIP_NORM: Final[float] = 1.0


@dataclass(frozen=True)
class FusionCalibrationAuditV72:
    optimizer_steps: int
    elapsed_seconds: float
    trainable_parameter_count: int
    branch_parameter_count: int
    branch_parameters_changed: bool
    initial_selection_accuracy: float
    final_selection_accuracy: float
    initial_fusion_weight_mean: float
    final_fusion_weight_mean: float
    final_fusion_weight_standard_deviation: float
    distinct_question_weight_vectors: int
    question_count: int
    maximum_preclip_gradient_norm: float
    held_rows_used_for_optimization: bool
    question_dependent_scene_retrieval: bool
    latent_selection_or_top_k_used: bool


def _fit_adaptive_multiscale_base(
    *,
    rows: Sequence[V63Row],
    targets: Mapping[tuple[str, str], torch.Tensor],
    preflight: V63Preflight,
    questions: Mapping[tuple[str, str], torch.Tensor],
    basis: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    phase: str,
) -> v66.V66FitResult:
    """Fit V71's two independent foundations, then install the V72 gate."""

    branch_8 = v66._fit_always_on(
        rows=rows,
        targets=targets,
        preflight=preflight,
        questions=questions,
        basis=basis,
        args=v71._branch_args(args, 8),
        seed=seed,
        phase=f"{phase}_branch_8",
    )
    branch_32 = v66._fit_always_on(
        rows=rows,
        targets=targets,
        preflight=preflight,
        questions=questions,
        basis=basis,
        args=v71._branch_args(args, 32),
        seed=seed + v71.V71_BRANCH_SEED_OFFSET,
        phase=f"{phase}_branch_32",
    )
    control = AdaptiveMultiscaleTeacherBasisControlV72(
        branch_8.control,
        branch_32.control,
        gate_hidden_size=V72_GATE_HIDDEN_SIZE,
    ).cpu().float()
    signatures = _scene_signatures(
        control,
        {
            scene_id: preflight.prefixes[scene_id]
            for scene_id in sorted({row.scene_id for row in rows})
        },
    )
    if branch_8.base_fit.question_norm_sha256 != branch_32.base_fit.question_norm_sha256:
        raise RuntimeError("V72 branches lost the authenticated question norm")
    combined = FitResult(
        control=control,
        signatures=signatures,
        basis_reconstruction={
            "branch_8_mean_cosine": branch_8.base_fit.basis_reconstruction[
                "mean_cosine"
            ],
            "branch_8_minimum_cosine": branch_8.base_fit.basis_reconstruction[
                "minimum_cosine"
            ],
            "branch_32_mean_cosine": branch_32.base_fit.basis_reconstruction[
                "mean_cosine"
            ],
            "branch_32_minimum_cosine": branch_32.base_fit.basis_reconstruction[
                "minimum_cosine"
            ],
        },
        elapsed_seconds=(
            branch_8.base_fit.elapsed_seconds + branch_32.base_fit.elapsed_seconds
        ),
        optimizer_steps=(
            branch_8.base_fit.optimizer_steps + branch_32.base_fit.optimizer_steps
        ),
        maximum_preclip_gradient_norm=max(
            branch_8.base_fit.maximum_preclip_gradient_norm,
            branch_32.base_fit.maximum_preclip_gradient_norm,
        ),
        final_route_loss=max(
            branch_8.base_fit.final_route_loss, branch_32.base_fit.final_route_loss
        ),
        question_norm_sha256=branch_8.base_fit.question_norm_sha256,
        question_norm_frozen=True,
    )
    return v66.V66FitResult(
        control=control,
        signatures=signatures,
        base_fit=combined,
        classification_optimizer_steps=(
            branch_8.classification_optimizer_steps
            + branch_32.classification_optimizer_steps
        ),
        numeric_prototype_top1_accuracy=(
            branch_8.numeric_prototype_top1_accuracy
            + branch_32.numeric_prototype_top1_accuracy
        )
        / 2.0,
        numeric_prototype_mean_margin=(
            branch_8.numeric_prototype_mean_margin
            + branch_32.numeric_prototype_mean_margin
        )
        / 2.0,
    )


def _fit_regularized_adaptive(**kwargs: Any) -> v68.V68FitResult:
    base = _fit_adaptive_multiscale_base(
        rows=kwargs["rows"],
        targets=kwargs["codebook"].targets,
        preflight=kwargs["preflight"],
        questions=kwargs["questions"],
        basis=kwargs["basis"],
        args=kwargs["args"],
        seed=kwargs["seed"],
        phase=f"{kwargs['phase']}_adaptive_base",
    )
    original = v66._fit_always_on

    def _provide_locked_base(**_ignored: Any) -> v66.V66FitResult:
        return base

    v66._fit_always_on = _provide_locked_base
    try:
        return v71._ORIGINAL_V68_FIT(**kwargs)
    finally:
        v66._fit_always_on = original


def _fit_augmented_adaptive(**kwargs: Any) -> v69.V69FitResult:
    original = v68._fit_regularized_pair
    v68._fit_regularized_pair = _fit_regularized_adaptive
    try:
        return v69._fit_augmented_pair(**kwargs)
    finally:
        v68._fit_regularized_pair = original


def _cached_branch_outputs(
    control: AdaptiveMultiscaleTeacherBasisControlV72,
    rows: Sequence[V63Row],
    *,
    signatures: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branch_8: list[torch.Tensor] = []
    branch_32: list[torch.Tensor] = []
    question_batches: list[torch.Tensor] = []
    control.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            signature = torch.cat([signatures[row.scene_id] for row in batch])
            question = torch.cat([questions[row.key] for row in batch])
            output_8, output_32 = control.branch_outputs_from_signature(
                signature, question
            )
            branch_8.append(output_8.detach().float())
            branch_32.append(output_32.detach().float())
            question_batches.append(question.detach().float())
    return torch.cat(branch_8), torch.cat(branch_32), torch.cat(question_batches)


def _calibrate_adaptive_fusion(
    fit: v69.V69FitResult,
    rows: Sequence[V63Row],
    *,
    codebook: Any,
    questions: Mapping[tuple[str, str], torch.Tensor],
    seed: int,
) -> FusionCalibrationAuditV72:
    control = fit.control
    if type(control) is not AdaptiveMultiscaleTeacherBasisControlV72:
        raise TypeError("V72 calibration requires the exact adaptive controller")
    row_list = list(rows)
    index_by_key = {row.key: index for index, row in enumerate(row_list)}
    if len(index_by_key) != len(row_list):
        raise ValueError("V72 calibration row keys are not unique")
    units = _changed_units(tuple(row for row in row_list if row.route_label))
    unit_indices = [
        (index_by_key[left.key], index_by_key[right.key]) for left, right in units
    ]
    if not unit_indices:
        raise ValueError("V72 calibration requires changed training units")

    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in control.named_parameters()
        if not name.startswith("coefficient_output.fusion_")
    }
    for parameter in control.parameters():
        parameter.requires_grad_(False)
    fusion_named = tuple(
        (name, parameter)
        for name, parameter in control.named_parameters()
        if name.startswith("coefficient_output.fusion_")
    )
    for _name, parameter in fusion_named:
        parameter.requires_grad_(True)
    if not fusion_named:
        raise RuntimeError("V72 fusion calibration parameter scope is empty")

    branch_8, branch_32, question_tensor = _cached_branch_outputs(
        control,
        row_list,
        signatures=fit.signatures,
        questions=questions,
    )
    targets = torch.cat([codebook.targets[row.key] for row in row_list]).float()
    basis = control.output_basis.detach().float()
    coefficient_8 = torch.einsum("nch,rh->ncr", branch_8, basis)
    coefficient_32 = torch.einsum("nch,rh->ncr", branch_32, basis)
    target_coefficients = torch.einsum("nch,rh->ncr", targets, basis)
    selection_targets = (
        (coefficient_8 - target_coefficients).square()
        < (coefficient_32 - target_coefficients).square()
    ).float()
    prototype_ids, prototype_bank, class_index = v67._prototype_bank(codebook)
    del prototype_ids
    class_targets = torch.tensor(
        [class_index[codebook.class_by_key[row.key]] for row in row_list],
        dtype=torch.long,
    )
    pair_flat_indices = torch.tensor(
        [index for unit in unit_indices for index in unit], dtype=torch.long
    )
    retention_indices = torch.tensor(
        [index for index, row in enumerate(row_list) if not row.route_label],
        dtype=torch.long,
    )
    if retention_indices.numel() == 0:
        raise ValueError("V72 calibration requires retention rows")

    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in fusion_named],
        lr=V72_CALIBRATION_LEARNING_RATE,
        weight_decay=V72_CALIBRATION_WEIGHT_DECAY,
    )
    with torch.inference_mode():
        initial_weights = control.fusion_weights(question_tensor)
        initial_selection_accuracy = float(
            ((initial_weights >= 0.5) == (selection_targets >= 0.5)).float().mean()
        )
        initial_weight_mean = float(initial_weights.mean())
    generator = torch.Generator().manual_seed(seed + 72_000_072)
    maximum_preclip = 0.0
    started = time.perf_counter()
    control.train()
    for _epoch in range(V72_CALIBRATION_EPOCHS):
        weights = control.fusion_weights(question_tensor)
        probability = ((weights - control.fusion_floor) / control.fusion_span).clamp(
            1e-6, 1.0 - 1e-6
        )
        selection_loss = F.binary_cross_entropy(probability, selection_targets)

        pair_8 = branch_8[pair_flat_indices]
        pair_32 = branch_32[pair_flat_indices]
        pair_questions = question_tensor[pair_flat_indices]
        pair_output, _directions, _rms, _weights = control.fuse_branch_outputs(
            pair_8, pair_32, pair_questions
        )
        pair_targets = targets[pair_flat_indices]
        pair_loss, _diagnostics = paired_scene_dependence_loss_v67(
            pair_output.reshape(len(unit_indices), 2, 4, 1536),
            pair_targets.reshape(len(unit_indices), 2, 4, 1536),
            opposite_margin=0.28,
            value_weight=1.0,
            delta_weight=8.0,
            opposite_weight=12.0,
        )
        pair_class_loss, _class_diagnostics = numeric_prototype_classification_loss(
            pair_output,
            prototype_bank,
            class_targets[pair_flat_indices],
            temperature=0.07,
        )
        pair_hard_loss, _hard_diagnostics = hard_negative_prototype_margin_loss_v68(
            pair_output,
            prototype_bank,
            class_targets[pair_flat_indices],
            margin=0.14,
        )

        permutation = retention_indices[
            torch.randperm(retention_indices.numel(), generator=generator)[:96]
        ]
        retention_output, _rd, _rr, _rw = control.fuse_branch_outputs(
            branch_8[permutation],
            branch_32[permutation],
            question_tensor[permutation],
        )
        retention_loss = v67._simple_value_loss(
            retention_output, targets[permutation]
        )
        loss = (
            V72_SELECTION_WEIGHT * selection_loss
            + V72_PAIR_WEIGHT * pair_loss
            + V72_CLASSIFICATION_WEIGHT * pair_class_loss
            + V72_HARD_NEGATIVE_WEIGHT * pair_hard_loss
            + V72_RETENTION_WEIGHT * retention_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        preclip = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for _name, parameter in fusion_named],
                V72_GRADIENT_CLIP_NORM,
            )
        )
        if not torch.isfinite(torch.tensor(preclip)):
            raise RuntimeError("V72 calibration gradient became nonfinite")
        maximum_preclip = max(maximum_preclip, preclip)
        optimizer.step()

    control.eval()
    after_changed = any(
        not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in control.named_parameters()
        if name in before
    )
    with torch.inference_mode():
        final_weights = control.fusion_weights(question_tensor)
        final_selection_accuracy = float(
            ((final_weights >= 0.5) == (selection_targets >= 0.5)).float().mean()
        )
        rounded = torch.round(final_weights.flatten(1) * 1_000_000).to(torch.int64)
        distinct = len({tensor.numpy().tobytes() for tensor in rounded})
    return FusionCalibrationAuditV72(
        optimizer_steps=V72_CALIBRATION_EPOCHS,
        elapsed_seconds=time.perf_counter() - started,
        trainable_parameter_count=sum(parameter.numel() for _, parameter in fusion_named),
        branch_parameter_count=sum(value.numel() for value in before.values()),
        branch_parameters_changed=after_changed,
        initial_selection_accuracy=initial_selection_accuracy,
        final_selection_accuracy=final_selection_accuracy,
        initial_fusion_weight_mean=initial_weight_mean,
        final_fusion_weight_mean=float(final_weights.mean()),
        final_fusion_weight_standard_deviation=float(
            final_weights.std(unbiased=False)
        ),
        distinct_question_weight_vectors=distinct,
        question_count=len(row_list),
        maximum_preclip_gradient_norm=maximum_preclip,
        held_rows_used_for_optimization=False,
        question_dependent_scene_retrieval=False,
        latent_selection_or_top_k_used=False,
    )


class _ForcedBranchV72(nn.Module):
    def __init__(
        self, control: AdaptiveMultiscaleTeacherBasisControlV72, branch: str
    ) -> None:
        super().__init__()
        if branch not in {"branch_8", "branch_32"}:
            raise ValueError("invalid V72 diagnostic branch")
        self.control = control
        self.branch = branch

    def forward_from_signature(
        self,
        signature: torch.Tensor,
        question: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> TeacherBasisControlOutput:
        branch_8, branch_32 = self.control.branch_outputs_from_signature(
            signature, question, question_attention_mask
        )
        value = branch_8 if self.branch == "branch_8" else branch_32
        coefficients = F.normalize(
            torch.einsum("bch,rh->bcr", value, self.control.output_basis),
            dim=-1,
            eps=1e-8,
        )
        rms = value.square().mean(dim=-1).sqrt()
        logits = torch.full(
            (value.shape[0],), 20.0, device=value.device, dtype=value.dtype
        )
        return TeacherBasisControlOutput(
            control_tokens=value,
            coefficient_directions=coefficients,
            control_rms=rms,
            gate_logits=logits,
            gate_probabilities=torch.sigmoid(logits),
        )


def _score_control(
    control: nn.Module,
    signatures: Mapping[str, torch.Tensor],
    held_rows: Sequence[V63Row],
    *,
    codebook: Any,
    questions: Mapping[tuple[str, str], torch.Tensor],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    fit = SimpleNamespace(control=control, signatures=signatures)
    return v67.numeric_screen_fold_v67(
        fit, held_rows, codebook=codebook, questions=questions
    )


def run_development_folds_v72(
    args: argparse.Namespace,
    *,
    runtime_provider: Any | None = None,
    supplemental_loader: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    fit_args = v71._fit_args(args)
    preflight = build_v63_preflight(fit_args)
    if len(preflight.rows) != _EXPECTED_ROWS:
        raise ValueError("V72 authenticated training inventory changed")
    validate_training_baseline_lock(
        args.training_baseline_lock, expected_rows=preflight.rows
    )
    teachers, teacher_audit = v66.load_combined_verified_teachers_v66(
        preflight,
        args.supplemental_teacher_cache,
        supplemental_loader=supplemental_loader,
    )
    held_pairs = tuple(args.held_pairs)
    if (
        not held_pairs
        or len(set(held_pairs)) != len(held_pairs)
        or any(pair not in TRAIN_PAIR_IDS for pair in held_pairs)
    ):
        raise ValueError("V72 development pairs must be distinct training pair IDs")
    if len(held_pairs) > 2:
        raise ValueError("V72 cheap development run is capped at two folds")
    bundle: V65RuntimeBundle = (runtime_provider or _load_runtime)(
        preflight, requested_device=args.device
    )
    folds: list[dict[str, Any]] = []
    for fold_index, held_pair in enumerate(held_pairs):
        train_rows = tuple(row for row in preflight.rows if row.pair_id != held_pair)
        held_rows = tuple(row for row in preflight.rows if row.pair_id == held_pair)
        train_keys = {row.key for row in train_rows}
        fold_teachers = {
            key: value for key, value in teachers.items() if key in train_keys
        }
        codebook = build_hybrid_answer_prototype_codebook_v66(
            train_rows,
            fold_teachers,
            expected_class_count=None,
            scope=f"v72_adaptive_dev_{held_pair}",
            forbidden_pair_id=held_pair,
        )
        basis = v66._codebook_basis(codebook, int(fit_args.basis_rank))
        fit = _fit_augmented_adaptive(
            rows=train_rows,
            codebook=codebook,
            preflight=preflight,
            questions=bundle.question_embeddings,
            basis=basis,
            args=fit_args,
            arm=v71.V71_AUGMENTATION_ARM,
            seed=int(v69.V69_COMMON_HYPERPARAMETERS["seed"])
            + (TRAIN_PAIR_IDS.index(held_pair) + 1) * 100_003,
            phase=f"v72_adaptive_dev_{held_pair}",
        )
        calibration = _calibrate_adaptive_fusion(
            fit,
            train_rows,
            codebook=codebook,
            questions=bundle.question_embeddings,
            seed=int(v69.V69_COMMON_HYPERPARAMETERS["seed"])
            + fold_index * 1_000_003,
        )
        held_signatures = _scene_signatures(
            fit.control,
            {
                scene_id: preflight.prefixes[scene_id]
                for scene_id in sorted({row.scene_id for row in held_rows})
            },
        )
        adaptive_metrics, adaptive_evidence = _score_control(
            fit.control,
            held_signatures,
            held_rows,
            codebook=codebook,
            questions=bundle.question_embeddings,
        )
        branch_metrics: dict[str, Any] = {}
        for branch in ("branch_8", "branch_32"):
            metrics, evidence = _score_control(
                _ForcedBranchV72(fit.control, branch),
                held_signatures,
                held_rows,
                codebook=codebook,
                questions=bundle.question_embeddings,
            )
            branch_metrics[branch] = {
                "metrics": metrics,
                "pair_evidence": [
                    record for record in evidence if record["kind"] == "pair"
                ],
            }
        held_questions = torch.cat(
            [bundle.question_embeddings[row.key] for row in held_rows]
        )
        with torch.inference_mode():
            held_weights = fit.control.fusion_weights(held_questions)
        folds.append(
            {
                "held_pair_id": held_pair,
                "held_rows_used_for_optimization": False,
                "held_teacher_sources_used": False,
                "calibration": asdict(calibration),
                "adaptive_metrics": adaptive_metrics,
                "adaptive_pair_evidence": [
                    record
                    for record in adaptive_evidence
                    if record["kind"] == "pair"
                ],
                "branch_diagnostics": branch_metrics,
                "held_fusion": {
                    "minimum": float(held_weights.min()),
                    "maximum": float(held_weights.max()),
                    "mean": float(held_weights.mean()),
                    "standard_deviation": float(
                        held_weights.std(unbiased=False)
                    ),
                    "distinct_row_vectors": len(
                        {
                            value.numpy().tobytes()
                            for value in torch.round(
                                held_weights.flatten(1) * 1_000_000
                            ).to(torch.int64)
                        }
                    ),
                    "row_count": len(held_rows),
                },
            }
        )
    report = {
        "schema_version": 1,
        "artifact": "v72_adaptive_fusion_train_only_development_v1",
        "status": "development_measurement_only",
        "checkpoint_published": False,
        "gemma_generation_used": False,
        "held_pairs": list(held_pairs),
        "architecture": {
            "complete_scene_prefix": True,
            "environment_latent_count": 256,
            "branch_moment_counts": [8, 32],
            "both_branches_process_every_environment_latent": True,
            "fusion": "question_conditioned_per_token_per_basis_continuous_gate",
            "fusion_bounds": [0.05, 0.95],
            "question_dependent_scene_retrieval": False,
            "latent_selection_or_top_k_used": False,
            "environmental_text_inputs": [],
        },
        "calibration_hyperparameters": {
            "gate_hidden_size": V72_GATE_HIDDEN_SIZE,
            "epochs": V72_CALIBRATION_EPOCHS,
            "learning_rate": V72_CALIBRATION_LEARNING_RATE,
            "weight_decay": V72_CALIBRATION_WEIGHT_DECAY,
            "selection_weight": V72_SELECTION_WEIGHT,
            "pair_weight": V72_PAIR_WEIGHT,
            "classification_weight": V72_CLASSIFICATION_WEIGHT,
            "hard_negative_weight": V72_HARD_NEGATIVE_WEIGHT,
            "retention_weight": V72_RETENTION_WEIGHT,
        },
        "folds": folds,
        "teacher_audit_loaded": bool(teacher_audit),
        "total_wall_time_seconds": time.perf_counter() - started,
        "scope": {
            "training_only": True,
            "pair_disjoint_held_measurement": True,
            "question_or_answer_text_stored_in_report": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "internal_validation_loaded": False,
            "deferred_final_loaded": False,
        },
    }
    output = _resolve(args.training_report)
    if output.exists():
        raise FileExistsError("V72 development report target already exists")
    v66._atomic_new_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-lock", required=True)
    parser.add_argument("--training-baseline-lock", required=True)
    parser.add_argument("--filtered-train-qa", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--supplemental-teacher-cache", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--source-v60-checkpoint", required=True)
    parser.add_argument("--held-pairs", nargs="+", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    random.seed(int(v69.V69_COMMON_HYPERPARAMETERS["seed"]))
    torch.manual_seed(int(v69.V69_COMMON_HYPERPARAMETERS["seed"]))
    report = run_development_folds_v72(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FusionCalibrationAuditV72",
    "run_development_folds_v72",
]
