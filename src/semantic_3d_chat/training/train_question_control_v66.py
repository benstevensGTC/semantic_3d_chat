"""Train-only all-row continuous adapter with pair-disjoint behavioral CV.

V66 abandons the V65 changed/retention route.  It learns one always-on,
bounded continuous full-scene+question controller from all 576 authorized
training rows.  Numeric answer prototypes come only from greedy-verified local
Gemma soft prompts.  Each leave-one-pair-out fold builds its codebook and
output basis after removing that pair's rows and teacher sources.

Every vocabulary-supported held row is answered by actual greedy local Gemma.
No validation, scorer, oracle, fresh-development, or deferred-final path is
accepted by this command.  Reports store opaque identities and hashes, not
question or answer text.  A checkpoint is published only after strict reload,
raw-token production-device generation, and a second all-training gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.evaluation.metrics import LIST_ANSWER_TYPES
from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import TRAIN_PAIR_IDS
from semantic_3d_chat.language.local_lm import question_token_ids
from semantic_3d_chat.scene_encoder.question_control_v3 import teacher_output_basis
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training.question_control_v7_checkpoint import (
    load_unsealed_v7_checkpoint_for_training_gate,
    save_v7_control_checkpoint,
    v7_value_state_sha256,
)
from semantic_3d_chat.training.question_control_v66_objective import (
    numeric_prototype_classification_loss,
)
from semantic_3d_chat.training.question_control_v66_prototypes import (
    HybridAnswerPrototypeCodebookV66,
    answer_class_id_v66,
    build_hybrid_answer_prototype_codebook_v66,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _resolve,
    _sha256_file,
    _write_training_report,
)
from semantic_3d_chat.training.train_question_control_v58 import _generate_with_control
from semantic_3d_chat.training.train_question_control_v63 import (
    FitResult,
    V63Preflight,
    V63Row,
    _fit_controller,
    _optimizer_step,
    _scene_signatures,
    build_v63_preflight,
)
from semantic_3d_chat.training.train_question_control_v65 import (
    V65RuntimeBundle,
    _canonical_scoring_sha256,
    _load_runtime,
    _row_reference_sha256,
    _row_scoring_contract_sha256,
    _tensor_sha256,
    validate_training_baseline_lock,
)

_EXPECTED_ROWS: Final[int] = 576
_EXPECTED_CLASSES: Final[int] = 28
_EXPECTED_SUPPORTED_CV_ROWS: Final[int] = 571
_EXPECTED_UNSUPPORTED_CV_ROWS: Final[int] = 5
_WORK_ARTIFACT: Final[str] = "v66b_allrow_pair_heldout_work_v2"
_FOLD_ARTIFACT: Final[str] = "v66b_allrow_pair_heldout_fold_v2"
_PINNED_PREREGISTRATION_SHA256: Final[str] = (
    "9c47e43e85b66bcf07794ccc206783db6a40b18af8ad29407475f081e60930bf"
)
_INVALIDATED_V66_PREREGISTRATION_SHA256: Final[str] = (
    "974f7049d2cf96670c77e6c19808a53fbca8b7c68e7cba7f9f5b184d0fc6ac4c"
)
_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "scene_id",
        "question_id",
        "pair_id",
        "question_key",
        "answer_class_id",
        "answer_type_sha256",
        "list_scoring",
        "counterfactual_changed_side",
        "fold_class_supported",
        "raw_prediction_sha256",
        "canonical_prediction_sha256",
        "reference_canonical_sha256",
        "scoring_contract_sha256",
        "canonical_exact",
    }
)
_PAIRED_OPPOSITE_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "scene_id",
        "question_id",
        "pair_id",
        "question_key",
        "paired_scene_id",
        "paired_question_id",
        "injected_scene_id",
        "question_sha256",
        "paired_question_sha256",
        "question_byte_identical",
        "original_prefix_sha256",
        "injected_prefix_sha256",
        "paired_opposite_prefix_sha256",
        "injected_prefix_matches_paired_opposite",
        "original_signature_sha256",
        "injected_signature_sha256",
        "paired_opposite_signature_sha256",
        "injected_signature_matches_paired_opposite",
        "injected_scene_is_paired_opposite",
        "original_answer_class_id",
        "injected_answer_class_id",
        "answer_type_sha256",
        "list_scoring",
        "raw_prediction_sha256",
        "canonical_prediction_sha256",
        "original_reference_canonical_sha256",
        "injected_reference_canonical_sha256",
        "original_scoring_contract_sha256",
        "injected_scoring_contract_sha256",
        "answer_follows_injected_scene",
        "answer_matches_original_scene",
    }
)


@dataclass(frozen=True)
class V66BehaviorThresholds:
    """Training-only preregistered gates, declared before generation."""

    held_supported_exact_minimum: int = 300
    held_supported_total: int = _EXPECTED_SUPPORTED_CV_ROWS
    held_unsupported_total: int = _EXPECTED_UNSUPPORTED_CV_ROWS
    eligible_fold_total: int = 12
    eligible_folds_with_exact_hit_minimum: int = 12
    held_changed_side_exact_minimum: int = 45
    held_changed_side_total: int = 75
    held_complete_unit_minimum: int = 15
    held_complete_unit_total: int = 35
    held_prediction_change_unit_minimum: int = 20
    held_prediction_change_unit_total: int = 35
    per_type_minimum_exact: tuple[tuple[str, int], ...] = (
        ("attribute", 35),
        ("count", 58),
        ("metric", 3),
        ("orientation", 20),
        ("presence", 60),
        ("spatial_relation", 65),
        ("support", 35),
    )
    final_exact_minimum: int = 520
    final_total: int = _EXPECTED_ROWS
    final_complete_unit_minimum: int = 36
    final_complete_unit_total: int = 40
    wrong_scene_exact_maximum: int = 250
    wrong_scene_changed_complete_maximum: int = 10


V66_BEHAVIOR_THRESHOLDS: Final[V66BehaviorThresholds] = V66BehaviorThresholds()


@dataclass(frozen=True)
class V66bBehaviorThresholds:
    """Successor gates with an identified paired-scene dependence control."""

    held_supported_exact_minimum: int = 300
    held_supported_total: int = _EXPECTED_SUPPORTED_CV_ROWS
    held_unsupported_total: int = _EXPECTED_UNSUPPORTED_CV_ROWS
    eligible_fold_total: int = 12
    eligible_folds_with_exact_hit_minimum: int = 12
    held_changed_side_exact_minimum: int = 45
    held_changed_side_total: int = 75
    held_complete_unit_minimum: int = 15
    held_complete_unit_total: int = 35
    held_prediction_change_unit_minimum: int = 20
    held_prediction_change_unit_total: int = 35
    per_type_minimum_exact: tuple[tuple[str, int], ...] = (
        ("attribute", 35),
        ("count", 58),
        ("metric", 3),
        ("orientation", 20),
        ("presence", 60),
        ("spatial_relation", 65),
        ("support", 35),
    )
    final_exact_minimum: int = 520
    final_total: int = _EXPECTED_ROWS
    final_complete_unit_minimum: int = 36
    final_complete_unit_total: int = 40
    paired_opposite_follows_side_minimum: int = 60
    paired_opposite_side_total: int = 80
    paired_opposite_follows_complete_minimum: int = 25
    paired_opposite_unit_total: int = 40
    paired_opposite_original_exact_maximum: int = 20
    paired_opposite_original_complete_maximum: int = 5


V66B_BEHAVIOR_THRESHOLDS: Final[V66bBehaviorThresholds] = (
    V66bBehaviorThresholds()
)


@dataclass(frozen=True)
class V66FitResult:
    control: AlwaysOnTeacherBasisFullSceneQuestionControlV7
    signatures: dict[str, torch.Tensor]
    base_fit: FitResult
    classification_optimizer_steps: int
    numeric_prototype_top1_accuracy: float
    numeric_prototype_mean_margin: float


def validate_v66_preregistration(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V66b preregistration is unavailable")
    if _sha256_file(source) != _PINNED_PREREGISTRATION_SHA256:
        raise ValueError("V66b preregistration differs from its pre-run pin")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V66b preregistration must be an object")
    invalidated = payload.get("invalidated_predecessor")
    controls = payload.get("controls")
    scope = payload.get("scope")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact")
        != "v66b_allrow_paired_opposite_training_preregistration"
        or payload.get("status")
        != "locked_before_v66b_controller_training_or_generation"
        or payload.get("thresholds")
        != json.loads(json.dumps(asdict(V66B_BEHAVIOR_THRESHOLDS), allow_nan=False))
        or not isinstance(invalidated, Mapping)
        or invalidated.get("artifact")
        != "v66_allrow_always_on_training_preregistration"
        or invalidated.get("sha256") != _INVALIDATED_V66_PREREGISTRATION_SHA256
        or invalidated.get("invalidated") is not True
        or invalidated.get("reason")
        != (
            "cyclic_wrong_scene_pairs_preserved_answers_for_340_of_409_"
            "exact_text_mappings"
        )
        or invalidated.get("preserved_answer_mappings") != 340
        or invalidated.get("exact_text_cyclic_mappings") != 409
        or invalidated.get("predecessor_artifact_bytes_modified") is not False
        or not isinstance(controls, Mapping)
        or controls.get("exact_paired_opposite_scene_prefix_and_signature")
        is not True
        or controls.get("same_question_byte_identity_required") is not True
        or controls.get("answer_follows_injected_scene_scored_against_opposite_reference")
        is not True
        or controls.get("cyclic_wrong_complete_scene_prefix_and_signature")
        is not False
        or controls.get(
            "unverified_native_answer_embedding_fallback_permitted"
        )
        is not False
        or not isinstance(scope, Mapping)
        or any(
            scope.get(field) is not False
            for field in (
                "validation_inputs_used",
                "scorer_inputs_used",
                "oracle_loaded",
                "fresh_development_loaded",
                "deferred_final_loaded",
            )
        )
    ):
        raise ValueError("V66b preregistration contract changed")
    return payload


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _supplemental_loader() -> Callable[
    [str | Path], tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]
]:
    try:
        from semantic_3d_chat.training.soft_prompt_teacher_v66 import (
            load_v66_answer_class_teacher_cache,
        )
    except ImportError as error:
        raise RuntimeError(
            "V66 supplemental verified-teacher module is not yet available"
        ) from error
    return load_v66_answer_class_teacher_cache


def load_combined_verified_teachers_v66(
    preflight: V63Preflight,
    supplemental_cache: str | Path,
    *,
    supplemental_loader: Callable[
        [str | Path], tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]
    ]
    | None = None,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]:
    """Merge strict V62 and V66 verified caches without duplicate sources."""

    loader = supplemental_loader or _supplemental_loader()
    supplemental, metadata = loader(_resolve(supplemental_cache))
    if not isinstance(metadata, dict):
        raise TypeError("V66 supplemental teacher metadata must be an object")
    known_rows = {row.key: row for row in preflight.rows}
    if (
        not supplemental
        or not set(supplemental).issubset(known_rows)
        or set(supplemental) & set(preflight.teacher_targets)
    ):
        raise ValueError("V66 supplemental teacher source inventory is invalid")
    if any(
        tuple(value.shape) != (1, 4, 1536) or not torch.isfinite(value).all()
        for value in supplemental.values()
    ):
        raise ValueError("V66 supplemental teachers must be finite [1,4,1536]")
    raw_records = metadata.get("records")
    if not isinstance(raw_records, list):
        raise TypeError("V66 supplemental metadata lacks opaque source records")
    supplemental_records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise TypeError("V66 supplemental metadata record must be an object")
        source_key = (
            str(record.get("source_scene_id")),
            str(record.get("source_question_id")),
        )
        verification = record.get("verification_keys")
        if (
            source_key in supplemental_records
            or source_key not in supplemental
            or not isinstance(verification, list)
            or len(verification) != 2
        ):
            raise ValueError("V66 supplemental opaque source inventory changed")
        source_row = known_rows[source_key]
        verification_keys = {
            (str(item.get("scene_id")), str(item.get("question_id")))
            for item in verification
            if isinstance(item, Mapping)
        }
        if (
            len(verification_keys) != 2
            or source_key not in verification_keys
            or record.get("answer_class_id")
            != answer_class_id_v66(source_row.answer)
            or record.get("source_pair_id") != source_row.pair_id
            or any(
                key not in known_rows
                or known_rows[key].pair_id != source_row.pair_id
                or answer_class_id_v66(known_rows[key].answer)
                != answer_class_id_v66(source_row.answer)
                for key in verification_keys
            )
        ):
            raise ValueError("V66 supplemental teacher is not bound to its QA class/pair")
        supplemental_records[source_key] = record
    if set(supplemental_records) != set(supplemental):
        raise ValueError("V66 supplemental tensor and metadata inventories differ")
    combined = {
        **preflight.teacher_targets,
        **{key: value.detach().cpu().float() for key, value in supplemental.items()},
    }
    covered_classes = {answer_class_id_v66(known_rows[key].answer) for key in combined}
    all_classes = {answer_class_id_v66(row.answer) for row in preflight.rows}
    if covered_classes != all_classes or len(all_classes) != _EXPECTED_CLASSES:
        raise ValueError("V66 verified caches do not cover all 28 answer classes")
    return combined, {
        "v62_teacher_metadata_sha256": preflight.teacher_metadata_sha256,
        "v62_teacher_weights_sha256": preflight.teacher_weights_sha256,
        "supplemental_cache_sha256": _canonical_sha256(metadata),
        "supplemental_metadata": metadata,
        "combined_teacher_count": len(combined),
        "answer_class_count": len(covered_classes),
        "every_answer_class_has_verified_teacher": True,
    }


def _codebook_basis(
    codebook: HybridAnswerPrototypeCodebookV66,
    requested_rank: int,
) -> torch.Tensor:
    ordered = torch.cat(
        [codebook.prototypes[class_id] for class_id in sorted(codebook.prototypes)],
        dim=0,
    )
    rank = min(requested_rank, ordered.shape[0] * ordered.shape[1], ordered.shape[2])
    return teacher_output_basis(ordered, rank=rank)


def _fit_always_on(
    *,
    rows: Sequence[V63Row],
    targets: Mapping[tuple[str, str], torch.Tensor],
    preflight: V63Preflight,
    questions: Mapping[tuple[str, str], torch.Tensor],
    basis: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    phase: str,
) -> V66FitResult:
    """Fit V3 values from all rows, then remove the route from runtime."""

    base = _fit_controller(
        rows=rows,
        targets=targets,
        prefixes={
            scene_id: preflight.prefixes[scene_id]
            for scene_id in sorted({row.scene_id for row in rows})
        },
        questions={row.key: questions[row.key] for row in rows},
        source_question_norm_state=preflight.source_v60_question_norm_state,
        args=args,
        seed=seed,
        log_phase=phase,
        fixed_output_basis=basis,
    )
    classification_steps = 0
    prototype_top1 = 0.0
    prototype_margin = 0.0
    if args.prototype_classification_weight > 0.0:
        class_ids = sorted({answer_class_id_v66(row.answer) for row in rows})
        class_index = {class_id: index for index, class_id in enumerate(class_ids)}
        representative: dict[str, torch.Tensor] = {}
        for row in rows:
            representative.setdefault(answer_class_id_v66(row.answer), targets[row.key])
        prototype_bank = torch.cat(
            [representative[class_id] for class_id in class_ids], dim=0
        ).float()
        optimizer = torch.optim.AdamW(
            [
                parameter
                for name, parameter in base.control.named_parameters()
                if parameter.requires_grad and not name.startswith("route_")
            ],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        signatures = _scene_signatures(
            base.control,
            {
                scene_id: preflight.prefixes[scene_id]
                for scene_id in sorted({row.scene_id for row in rows})
            },
        )
        ordered = list(rows)
        for epoch in range(args.prototype_classification_epochs):
            rng = random.Random(seed + 50_000_003 + epoch)
            rng.shuffle(ordered)
            for offset in range(0, len(ordered), args.batch_size):
                batch = ordered[offset : offset + args.batch_size]
                predicted = base.control.forward_from_signature(
                    torch.cat([signatures[row.scene_id] for row in batch]),
                    torch.cat([questions[row.key] for row in batch]),
                ).control_tokens
                indices = torch.tensor(
                    [class_index[answer_class_id_v66(row.answer)] for row in batch],
                    dtype=torch.long,
                )
                loss, _diagnostics = numeric_prototype_classification_loss(
                    predicted,
                    prototype_bank,
                    indices,
                    temperature=args.prototype_classification_temperature,
                )
                expected = torch.cat([targets[row.key] for row in batch])
                target_power = expected.square().mean().clamp_min(1e-8)
                preservation = 1.0 - F.cosine_similarity(
                    predicted,
                    expected,
                    dim=-1,
                ).mean()
                preservation = preservation + 0.10 * F.mse_loss(
                    predicted,
                    expected,
                ) / target_power
                _optimizer_step(
                    loss=(
                        args.prototype_classification_weight * loss
                        + args.prototype_value_preservation_weight * preservation
                    ),
                    control=base.control,
                    optimizer=optimizer,
                    gradient_clip_norm=args.gradient_clip_norm,
                )
                classification_steps += 1
        with torch.inference_mode():
            predicted_all = base.control.forward_from_signature(
                torch.cat([signatures[row.scene_id] for row in rows]),
                torch.cat([questions[row.key] for row in rows]),
            ).control_tokens
            all_indices = torch.tensor(
                [class_index[answer_class_id_v66(row.answer)] for row in rows],
                dtype=torch.long,
            )
            _loss, diagnostics = numeric_prototype_classification_loss(
                predicted_all,
                prototype_bank,
                all_indices,
                temperature=args.prototype_classification_temperature,
            )
            prototype_top1 = float(diagnostics.top1_accuracy.cpu())
            prototype_margin = float(diagnostics.mean_margin.cpu())
    control = AlwaysOnTeacherBasisFullSceneQuestionControlV7.from_v3(base.control)
    control.eval()
    signatures = _scene_signatures(
        control,
        {
            scene_id: preflight.prefixes[scene_id]
            for scene_id in sorted({row.scene_id for row in rows})
        },
    )
    return V66FitResult(
        control=control,
        signatures=signatures,
        base_fit=base,
        classification_optimizer_steps=classification_steps,
        numeric_prototype_top1_accuracy=prototype_top1,
        numeric_prototype_mean_margin=prototype_margin,
    )


def _canonical_record(
    row: V63Row,
    prediction: str,
    *,
    supported: bool,
) -> dict[str, Any]:
    prediction_digest = _canonical_scoring_sha256(
        prediction,
        answer_type=row.answer_type,
    )
    reference_digest = _row_reference_sha256(row)
    return {
        "scene_id": row.scene_id,
        "question_id": row.question_id,
        "pair_id": row.pair_id,
        "question_key": row.question_key,
        "answer_class_id": answer_class_id_v66(row.answer),
        "counterfactual_changed_side": row.route_label,
        "answer_type_sha256": hashlib.sha256(row.answer_type.encode()).hexdigest(),
        "list_scoring": row.answer_type in LIST_ANSWER_TYPES,
        "fold_class_supported": supported,
        "raw_prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
        "canonical_prediction_sha256": prediction_digest,
        "reference_canonical_sha256": reference_digest,
        "scoring_contract_sha256": _row_scoring_contract_sha256(row),
        "canonical_exact": prediction_digest == reference_digest,
    }


def generate_supported_rows_v66(
    fit: V66FitResult,
    rows: Sequence[V63Row],
    *,
    questions: Mapping[tuple[str, str], torch.Tensor],
    supported_classes: set[str],
    bundle: V65RuntimeBundle,
    prefixes: Mapping[str, torch.Tensor],
    generator_fn: Callable[..., str],
) -> tuple[dict[str, Any], ...]:
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in sorted(rows, key=lambda item: (item.scene_id, item.question_id)):
            class_id = answer_class_id_v66(row.answer)
            if class_id not in supported_classes:
                continue
            output = fit.control.forward_from_signature(
                fit.signatures[row.scene_id], questions[row.key]
            )
            if fit.control.audit().control_used is not True:
                raise RuntimeError("V66 always-on controller suppressed a held row")
            prediction = generator_fn(
                runtime=bundle.runtime,
                scene_prefix=prefixes[row.scene_id].to(
                    device=bundle.device,
                    dtype=bundle.model_dtype,
                ),
                question=row.question,
                control_tokens=output.control_tokens,
            )
            results.append(_canonical_record(row, prediction, supported=True))
    return tuple(results)


def _changed_pair_units_v66(
    rows: Sequence[V63Row],
) -> tuple[tuple[V63Row, V63Row], ...]:
    grouped: defaultdict[tuple[str, str], list[V63Row]] = defaultdict(list)
    for row in rows:
        if row.route_label:
            grouped[(row.pair_id, row.question_key)].append(row)
    if len(grouped) != 40 or sum(len(sides) for sides in grouped.values()) != 80:
        raise ValueError("V66b paired-scene control requires exactly 80 sides/40 units")
    units: list[tuple[V63Row, V63Row]] = []
    for key, raw_sides in sorted(grouped.items()):
        sides = sorted(raw_sides, key=lambda row: (row.scene_id, row.question_id))
        if (
            len(sides) != 2
            or sides[0].scene_id == sides[1].scene_id
            or sides[0].question.encode("utf-8") != sides[1].question.encode("utf-8")
            or sides[0].answer_type != sides[1].answer_type
            or _row_reference_sha256(sides[0]) == _row_reference_sha256(sides[1])
        ):
            raise ValueError(
                f"V66b changed unit is not an identified paired opposite: {key}"
            )
        units.append((sides[0], sides[1]))
    return tuple(units)


def generate_paired_opposite_scene_rows_v66(
    fit: V66FitResult,
    rows: Sequence[V63Row],
    *,
    questions: Mapping[tuple[str, str], torch.Tensor],
    bundle: V65RuntimeBundle,
    prefixes: Mapping[str, torch.Tensor],
    generator_fn: Callable[..., str],
) -> tuple[dict[str, Any], ...]:
    """Inject the exact paired opposite scene for each changed-side question."""

    units = _changed_pair_units_v66(rows)
    scene_ids = sorted({row.scene_id for unit in units for row in unit})
    if (
        not set(scene_ids).issubset(prefixes)
        or not set(scene_ids).issubset(fit.signatures)
    ):
        raise ValueError("V66b paired-scene control lacks an exact prefix/signature")
    recomputed_signatures = _scene_signatures(
        fit.control, {scene_id: prefixes[scene_id] for scene_id in scene_ids}
    )
    if any(
        not torch.equal(
            recomputed_signatures[scene_id].detach().cpu().float(),
            fit.signatures[scene_id].detach().cpu().float(),
        )
        for scene_id in scene_ids
    ):
        raise ValueError("V66b cached scene signature differs from the exact prefix")
    signatures = {scene_id: fit.signatures[scene_id] for scene_id in scene_ids}
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for left, right in units:
            if not torch.equal(questions[left.key], questions[right.key]):
                raise ValueError("V66b byte-identical question embeddings differ")
            for row, opposite in ((left, right), (right, left)):
                output = fit.control.forward_from_signature(
                    signatures[opposite.scene_id], questions[row.key]
                )
                if fit.control.audit().control_used is not True:
                    raise RuntimeError("V66b always-on controller suppressed paired control")
                prediction = generator_fn(
                    runtime=bundle.runtime,
                    scene_prefix=prefixes[opposite.scene_id].to(
                        device=bundle.device,
                        dtype=bundle.model_dtype,
                    ),
                    question=row.question,
                    control_tokens=output.control_tokens,
                )
                prediction_hash = _canonical_scoring_sha256(
                    prediction, answer_type=row.answer_type
                )
                original_reference = _row_reference_sha256(row)
                injected_reference = _row_reference_sha256(opposite)
                question_hash = hashlib.sha256(row.question.encode("utf-8")).hexdigest()
                paired_question_hash = hashlib.sha256(
                    opposite.question.encode("utf-8")
                ).hexdigest()
                original_prefix_hash = _tensor_sha256(prefixes[row.scene_id])
                injected_prefix_hash = _tensor_sha256(prefixes[opposite.scene_id])
                original_signature_hash = _tensor_sha256(signatures[row.scene_id])
                injected_signature_hash = _tensor_sha256(
                    signatures[opposite.scene_id]
                )
                results.append(
                    {
                        "scene_id": row.scene_id,
                        "question_id": row.question_id,
                        "pair_id": row.pair_id,
                        "question_key": row.question_key,
                        "paired_scene_id": opposite.scene_id,
                        "paired_question_id": opposite.question_id,
                        "injected_scene_id": opposite.scene_id,
                        "question_sha256": question_hash,
                        "paired_question_sha256": paired_question_hash,
                        "question_byte_identical": question_hash == paired_question_hash,
                        "original_prefix_sha256": original_prefix_hash,
                        "injected_prefix_sha256": injected_prefix_hash,
                        "paired_opposite_prefix_sha256": _tensor_sha256(
                            prefixes[opposite.scene_id]
                        ),
                        "original_signature_sha256": original_signature_hash,
                        "injected_signature_sha256": injected_signature_hash,
                        "paired_opposite_signature_sha256": _tensor_sha256(
                            signatures[opposite.scene_id]
                        ),
                        "injected_scene_is_paired_opposite": (
                            opposite.scene_id != row.scene_id
                        ),
                        "injected_prefix_matches_paired_opposite": (
                            injected_prefix_hash
                            == _tensor_sha256(prefixes[opposite.scene_id])
                        ),
                        "injected_signature_matches_paired_opposite": (
                            injected_signature_hash
                            == _tensor_sha256(signatures[opposite.scene_id])
                        ),
                        "original_answer_class_id": answer_class_id_v66(row.answer),
                        "injected_answer_class_id": answer_class_id_v66(
                            opposite.answer
                        ),
                        "answer_type_sha256": hashlib.sha256(
                            row.answer_type.encode()
                        ).hexdigest(),
                        "list_scoring": row.answer_type in LIST_ANSWER_TYPES,
                        "raw_prediction_sha256": hashlib.sha256(
                            prediction.encode("utf-8")
                        ).hexdigest(),
                        "canonical_prediction_sha256": prediction_hash,
                        "original_reference_canonical_sha256": original_reference,
                        "injected_reference_canonical_sha256": injected_reference,
                        "original_scoring_contract_sha256": (
                            _row_scoring_contract_sha256(row)
                        ),
                        "injected_scoring_contract_sha256": (
                            _row_scoring_contract_sha256(opposite)
                        ),
                        "answer_follows_injected_scene": (
                            prediction_hash == injected_reference
                        ),
                        "answer_matches_original_scene": (
                            prediction_hash == original_reference
                        ),
                    }
                )
    if len(results) != 80:
        raise RuntimeError("V66b paired-scene generation inventory changed")
    return tuple(results)


def behavior_metrics_v66(
    records: Sequence[Mapping[str, Any]],
    *,
    unsupported_count: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("V66 behavior metrics require generated rows")
    keys: set[tuple[str, str]] = set()
    pair_hits: defaultdict[str, int] = defaultdict(int)
    type_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    changed_units: defaultdict[tuple[str, str], list[tuple[bool, str]]] = defaultdict(list)
    exact = 0
    for record in records:
        key = str(record["scene_id"]), str(record["question_id"])
        if key in keys:
            raise ValueError("V66 behavior records contain duplicate opaque keys")
        keys.add(key)
        if record.get("fold_class_supported") is not True:
            raise ValueError("V66 generated a vocabulary-unsupported row")
        expected_exact = (
            record.get("canonical_prediction_sha256")
            == record.get("reference_canonical_sha256")
        )
        if record.get("canonical_exact") is not expected_exact:
            raise ValueError("V66 exact boolean differs from canonical hashes")
        hit = int(expected_exact)
        exact += hit
        pair_hits[str(record["pair_id"])] += hit
        type_key = str(record["answer_type_sha256"])
        type_counts[type_key][0] += hit
        type_counts[type_key][1] += 1
        if record.get("counterfactual_changed_side") is True:
            changed_units[(str(record["pair_id"]), str(record["question_key"]))].append(
                (expected_exact, str(record["canonical_prediction_sha256"]))
            )
    if any(len(sides) not in (1, 2) for sides in changed_units.values()):
        raise ValueError("V66 changed unit generated an invalid side inventory")
    complete_inventory_units = {
        key: sides for key, sides in changed_units.items() if len(sides) == 2
    }
    changed_side_exact = sum(
        int(exact_side)
        for sides in changed_units.values()
        for exact_side, _prediction in sides
    )
    complete_units = sum(
        all(exact_side for exact_side, _prediction in sides)
        for sides in complete_inventory_units.values()
    )
    prediction_change_units = sum(
        len({prediction for _exact_side, prediction in sides}) > 1
        for sides in complete_inventory_units.values()
    )
    total_inventory = len(records) + unsupported_count
    return {
        "supported_exact": exact,
        "supported_total": len(records),
        "supported_accuracy": exact / len(records),
        "unsupported_total": unsupported_count,
        "inventory_total": total_inventory,
        "eligible_fold_total": len(pair_hits),
        "eligible_folds_with_exact_hit": sum(value > 0 for value in pair_hits.values()),
        "changed_side_exact": changed_side_exact,
        "changed_side_total": sum(len(sides) for sides in changed_units.values()),
        "complete_changed_units": complete_units,
        "changed_unit_total": len(complete_inventory_units),
        "incomplete_unsupported_changed_units": len(changed_units)
        - len(complete_inventory_units),
        "prediction_change_units": prediction_change_units,
        "per_type_by_sha256": {
            key: {"exact": values[0], "total": values[1]}
            for key, values in sorted(type_counts.items())
        },
        "answer_or_question_text_stored": False,
    }


def assess_cv_v66(metrics: Mapping[str, Any]) -> dict[str, bool]:
    threshold = V66B_BEHAVIOR_THRESHOLDS
    per_type = metrics.get("per_type_by_sha256")
    if not isinstance(per_type, Mapping):
        raise TypeError("V66 CV per-type metrics are missing")
    per_type_checks = {
        f"per_type_{answer_type}": int(
            per_type.get(hashlib.sha256(answer_type.encode()).hexdigest(), {}).get(
                "exact", -1
            )
        )
        >= minimum
        for answer_type, minimum in threshold.per_type_minimum_exact
    }
    return {
        "held_supported_exact": int(metrics["supported_exact"])
        >= threshold.held_supported_exact_minimum,
        "held_supported_total": int(metrics["supported_total"])
        == threshold.held_supported_total,
        "held_unsupported_total": int(metrics["unsupported_total"])
        == threshold.held_unsupported_total,
        "complete_inventory": int(metrics["inventory_total"]) == _EXPECTED_ROWS,
        "eligible_fold_total": int(metrics["eligible_fold_total"])
        == threshold.eligible_fold_total,
        "eligible_folds_with_exact_hit": int(metrics["eligible_folds_with_exact_hit"])
        >= threshold.eligible_folds_with_exact_hit_minimum,
        "held_changed_side_exact": int(metrics["changed_side_exact"])
        >= threshold.held_changed_side_exact_minimum,
        "held_changed_side_total": int(metrics["changed_side_total"])
        == threshold.held_changed_side_total,
        "held_complete_units": int(metrics["complete_changed_units"])
        >= threshold.held_complete_unit_minimum,
        "held_complete_unit_total": int(metrics["changed_unit_total"])
        == threshold.held_complete_unit_total,
        "held_prediction_change_units": int(metrics["prediction_change_units"])
        >= threshold.held_prediction_change_unit_minimum,
        "held_prediction_change_unit_total": int(metrics["changed_unit_total"])
        == threshold.held_prediction_change_unit_total,
        **per_type_checks,
    }


def assess_final_v66(metrics: Mapping[str, Any]) -> dict[str, bool]:
    threshold = V66B_BEHAVIOR_THRESHOLDS
    return {
        "final_exact": int(metrics["supported_exact"])
        >= threshold.final_exact_minimum,
        "final_total": int(metrics["supported_total"]) == threshold.final_total,
        "no_unsupported_final_rows": int(metrics["unsupported_total"]) == 0,
        "complete_inventory": int(metrics["inventory_total"]) == threshold.final_total,
        "final_complete_units": int(metrics["complete_changed_units"])
        >= threshold.final_complete_unit_minimum,
        "final_complete_unit_total": int(metrics["changed_unit_total"])
        == threshold.final_complete_unit_total,
    }


def paired_opposite_metrics_v66(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the identified 80-side scene-dependence control from hashes."""

    if len(records) != V66B_BEHAVIOR_THRESHOLDS.paired_opposite_side_total:
        raise ValueError("V66b paired-scene control requires exactly 80 records")
    digest_fields = (
        "question_sha256",
        "paired_question_sha256",
        "original_prefix_sha256",
        "injected_prefix_sha256",
        "paired_opposite_prefix_sha256",
        "original_signature_sha256",
        "injected_signature_sha256",
        "paired_opposite_signature_sha256",
        "answer_type_sha256",
        "raw_prediction_sha256",
        "canonical_prediction_sha256",
        "original_reference_canonical_sha256",
        "injected_reference_canonical_sha256",
        "original_scoring_contract_sha256",
        "injected_scoring_contract_sha256",
    )
    identity_fields = (
        "scene_id",
        "question_id",
        "pair_id",
        "question_key",
        "paired_scene_id",
        "paired_question_id",
        "injected_scene_id",
        "original_answer_class_id",
        "injected_answer_class_id",
    )
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    keys: set[tuple[str, str]] = set()
    follows_injected = 0
    matches_original = 0
    question_identity_count = 0
    exact_paired_scene_count = 0
    exact_paired_prefix_count = 0
    exact_paired_signature_count = 0
    differing_reference_count = 0
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _PAIRED_OPPOSITE_RECORD_FIELDS:
            raise ValueError("V66b paired-scene record fields changed")
        if any(
            not isinstance(record.get(field), str) or not str(record[field])
            for field in identity_fields
        ):
            raise ValueError("V66b paired-scene opaque identity is invalid")
        if any(not _is_sha256(record.get(field)) for field in digest_fields):
            raise ValueError("V66b paired-scene digest is invalid")
        if not isinstance(record.get("list_scoring"), bool):
            raise TypeError("V66b paired-scene scoring mode is invalid")

        key = str(record["scene_id"]), str(record["question_id"])
        if key in keys:
            raise ValueError("V66b paired-scene records contain duplicate row keys")
        keys.add(key)
        grouped[(str(record["pair_id"]), str(record["question_key"]))].append(
            record
        )

        question_identical = (
            record["question_sha256"] == record["paired_question_sha256"]
        )
        exact_scene = (
            record["injected_scene_id"] == record["paired_scene_id"]
            and record["injected_scene_id"] != record["scene_id"]
        )
        exact_prefix = (
            record["injected_prefix_sha256"]
            == record["paired_opposite_prefix_sha256"]
            and record["injected_prefix_sha256"]
            != record["original_prefix_sha256"]
        )
        exact_signature = (
            record["injected_signature_sha256"]
            == record["paired_opposite_signature_sha256"]
            and record["injected_signature_sha256"]
            != record["original_signature_sha256"]
        )
        differing_reference = (
            record["injected_reference_canonical_sha256"]
            != record["original_reference_canonical_sha256"]
            and record["injected_answer_class_id"]
            != record["original_answer_class_id"]
        )
        injected_hit = (
            record["canonical_prediction_sha256"]
            == record["injected_reference_canonical_sha256"]
        )
        original_hit = (
            record["canonical_prediction_sha256"]
            == record["original_reference_canonical_sha256"]
        )
        derived_booleans = {
            "question_byte_identical": question_identical,
            "injected_scene_is_paired_opposite": exact_scene,
            "injected_prefix_matches_paired_opposite": exact_prefix,
            "injected_signature_matches_paired_opposite": exact_signature,
            "answer_follows_injected_scene": injected_hit,
            "answer_matches_original_scene": original_hit,
        }
        if any(record.get(field) is not value for field, value in derived_booleans.items()):
            raise ValueError("V66b paired-scene boolean differs from hashed evidence")
        if not (question_identical and exact_scene and exact_prefix and exact_signature):
            raise ValueError("V66b did not inject the exact paired scene and question")
        if not differing_reference:
            raise ValueError("V66b paired-scene references do not encode a changed fact")

        follows_injected += int(injected_hit)
        matches_original += int(original_hit)
        question_identity_count += int(question_identical)
        exact_paired_scene_count += int(exact_scene)
        exact_paired_prefix_count += int(exact_prefix)
        exact_paired_signature_count += int(exact_signature)
        differing_reference_count += int(differing_reference)

    if (
        len(grouped) != V66B_BEHAVIOR_THRESHOLDS.paired_opposite_unit_total
        or any(len(sides) != 2 for sides in grouped.values())
    ):
        raise ValueError("V66b paired-scene unit inventory is not exactly 40 pairs")

    follows_complete = 0
    original_complete = 0
    cross_swap_complete = 0
    for unit_key, sides in grouped.items():
        left, right = sorted(
            sides, key=lambda record: (str(record["scene_id"]), str(record["question_id"]))
        )
        cross_swap = (
            left["scene_id"] == right["paired_scene_id"]
            and right["scene_id"] == left["paired_scene_id"]
            and left["question_id"] == right["paired_question_id"]
            and right["question_id"] == left["paired_question_id"]
            and left["question_sha256"] == right["question_sha256"]
            and left["original_prefix_sha256"] == right["injected_prefix_sha256"]
            and right["original_prefix_sha256"] == left["injected_prefix_sha256"]
            and left["original_signature_sha256"]
            == right["injected_signature_sha256"]
            and right["original_signature_sha256"]
            == left["injected_signature_sha256"]
            and left["original_reference_canonical_sha256"]
            == right["injected_reference_canonical_sha256"]
            and right["original_reference_canonical_sha256"]
            == left["injected_reference_canonical_sha256"]
            and left["original_answer_class_id"] == right["injected_answer_class_id"]
            and right["original_answer_class_id"] == left["injected_answer_class_id"]
            and left["original_scoring_contract_sha256"]
            == right["injected_scoring_contract_sha256"]
            and right["original_scoring_contract_sha256"]
            == left["injected_scoring_contract_sha256"]
            and left["answer_type_sha256"] == right["answer_type_sha256"]
            and left["list_scoring"] is right["list_scoring"]
        )
        if not cross_swap:
            raise ValueError(f"V66b paired-scene cross-swap changed: {unit_key}")
        cross_swap_complete += 1
        follows_complete += int(
            left["answer_follows_injected_scene"] is True
            and right["answer_follows_injected_scene"] is True
        )
        original_complete += int(
            left["answer_matches_original_scene"] is True
            and right["answer_matches_original_scene"] is True
        )

    return {
        "answer_follows_injected_scene": follows_injected,
        "paired_opposite_side_total": len(records),
        "answer_follows_injected_scene_complete_units": follows_complete,
        "paired_opposite_unit_total": len(grouped),
        "answer_matches_original_reference": matches_original,
        "answer_matches_original_reference_complete_units": original_complete,
        "question_identity_count": question_identity_count,
        "exact_paired_scene_count": exact_paired_scene_count,
        "exact_paired_scene_prefix_count": exact_paired_prefix_count,
        "exact_paired_scene_signature_count": exact_paired_signature_count,
        "differing_reference_count": differing_reference_count,
        "cross_swap_complete_units": cross_swap_complete,
        "answer_or_question_text_stored": False,
    }


def paired_opposite_checks_v66(metrics: Mapping[str, Any]) -> dict[str, bool]:
    threshold = V66B_BEHAVIOR_THRESHOLDS
    return {
        "follows_injected_side_minimum": int(
            metrics["answer_follows_injected_scene"]
        )
        >= threshold.paired_opposite_follows_side_minimum,
        "paired_opposite_side_total": int(metrics["paired_opposite_side_total"])
        == threshold.paired_opposite_side_total,
        "follows_injected_complete_unit_minimum": int(
            metrics["answer_follows_injected_scene_complete_units"]
        )
        >= threshold.paired_opposite_follows_complete_minimum,
        "paired_opposite_unit_total": int(metrics["paired_opposite_unit_total"])
        == threshold.paired_opposite_unit_total,
        "original_reference_exact_ceiling": int(
            metrics["answer_matches_original_reference"]
        )
        <= threshold.paired_opposite_original_exact_maximum,
        "original_reference_complete_ceiling": int(
            metrics["answer_matches_original_reference_complete_units"]
        )
        <= threshold.paired_opposite_original_complete_maximum,
        "question_identity_complete": int(metrics["question_identity_count"])
        == threshold.paired_opposite_side_total,
        "exact_paired_scene_complete": int(metrics["exact_paired_scene_count"])
        == threshold.paired_opposite_side_total,
        "exact_paired_scene_prefix_complete": int(
            metrics["exact_paired_scene_prefix_count"]
        )
        == threshold.paired_opposite_side_total,
        "exact_paired_scene_signature_complete": int(
            metrics["exact_paired_scene_signature_count"]
        )
        == threshold.paired_opposite_side_total,
        "differing_reference_complete": int(metrics["differing_reference_count"])
        == threshold.paired_opposite_side_total,
        "cross_swap_complete": int(metrics["cross_swap_complete_units"])
        == threshold.paired_opposite_unit_total,
        "no_question_or_answer_text_stored": metrics.get(
            "answer_or_question_text_stored"
        )
        is False,
    }


def _atomic_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"V66 refuses to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode() + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_manifest(
    preflight: V63Preflight,
    teacher_audit: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "artifact": _WORK_ARTIFACT,
        "filtered_training_qa_sha256": preflight.filtered_train_sha256,
        "preregistration_sha256": _sha256_file(_resolve(args.preregistration)),
        "base_checkpoint_sha256": preflight.base_checkpoint_sha256,
        "runtime_config_sha256": preflight.runtime_config_sha256,
        "prefix_cache_manifest_sha256": preflight.prefix_manifest_sha256,
        "combined_teacher_audit_sha256": _canonical_sha256(teacher_audit),
        "implementation_files_sha256": {
            "trainer": _sha256_file(Path(__file__).resolve()),
            "controller": _sha256_file(
                Path(
                    AlwaysOnTeacherBasisFullSceneQuestionControlV7.__init__.__code__.co_filename
                ).resolve()
            ),
            "prototype_codebook": _sha256_file(
                Path(
                    build_hybrid_answer_prototype_codebook_v66.__code__.co_filename
                ).resolve()
            ),
            "prototype_objective": _sha256_file(
                Path(numeric_prototype_classification_loss.__code__.co_filename).resolve()
            ),
            "generator": _sha256_file(
                Path(_generate_with_control.__code__.co_filename).resolve()
            ),
        },
        "pair_ids": list(TRAIN_PAIR_IDS),
        "seed": args.seed,
        "hyperparameters": {
            field: getattr(args, field)
            for field in (
                "basis_rank",
                "moment_count",
                "interaction_dim",
                "trunk_dim",
                "maximum_control_rms",
                "initial_control_rms",
                "epochs",
                "batch_size",
                "changed_repeats",
                "learning_rate",
                "weight_decay",
                "gradient_clip_norm",
                "coefficient_weight",
                "log_rms_weight",
                "reconstruction_weight",
                "relative_mse_weight",
                "prototype_classification_epochs",
                "prototype_classification_weight",
                "prototype_classification_temperature",
                "prototype_value_preservation_weight",
            )
        },
        "thresholds": asdict(V66B_BEHAVIOR_THRESHOLDS),
        "fold_codebook_and_basis_built_after_pair_exclusion": True,
        "all_576_rows_train_the_value_function": True,
        "always_on_control": True,
        "unverified_numeric_prototype_fallback": False,
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    return {**identity, "run_signature_sha256": _canonical_sha256(identity)}


def _prepare_work_directory(path: Path, manifest: Mapping[str, Any]) -> None:
    normalized_manifest = json.loads(_canonical_bytes(dict(manifest)))
    if path.exists():
        existing_path = path / "manifest.json"
        if not existing_path.is_file():
            raise ValueError("V66 work directory lacks its manifest")
        if json.loads(existing_path.read_text(encoding="utf-8")) != normalized_manifest:
            raise ValueError("V66 work directory belongs to another run")
        return
    path.mkdir(parents=True)
    _atomic_new_json(path / "manifest.json", normalized_manifest)


def _fold_path(work: Path, pair_id: str) -> Path:
    if pair_id not in TRAIN_PAIR_IDS:
        raise ValueError("V66 fold pair is unauthorized")
    return work / f"fold_{pair_id}.json"


def _expected_supported_held_keys(
    rows: Sequence[V63Row],
    *,
    held_pair_id: str,
) -> set[tuple[str, str]]:
    train_classes = {
        answer_class_id_v66(row.answer) for row in rows if row.pair_id != held_pair_id
    }
    return {
        row.key
        for row in rows
        if row.pair_id == held_pair_id
        and answer_class_id_v66(row.answer) in train_classes
    }


def _validate_cached_fold_v66(
    payload: object,
    *,
    held_pair_id: str,
    run_signature_sha256: str,
    rows: Sequence[V63Row],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("V66 cached fold must be an object")
    required = {
        "schema_version",
        "artifact",
        "run_signature_sha256",
        "held_pair_id",
        "held_rows_used_for_optimization",
        "held_teacher_sources_used",
        "fold_codebook_sha256",
        "fold_basis_sha256",
        "fold_class_count",
        "fold_train_target_count",
        "fit",
        "behavior",
        "records",
    }
    if set(payload) != required:
        raise ValueError("V66 cached fold fields changed")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact") != _FOLD_ARTIFACT
        or payload.get("run_signature_sha256") != run_signature_sha256
        or payload.get("held_pair_id") != held_pair_id
        or payload.get("held_rows_used_for_optimization") is not False
        or payload.get("held_teacher_sources_used") is not False
    ):
        raise ValueError("V66 cached fold provenance changed")
    expected_train_rows = [row for row in rows if row.pair_id != held_pair_id]
    expected_class_count = len(
        {answer_class_id_v66(row.answer) for row in expected_train_rows}
    )
    if (
        not _is_sha256(payload.get("fold_codebook_sha256"))
        or not _is_sha256(payload.get("fold_basis_sha256"))
        or payload.get("fold_class_count") != expected_class_count
        or payload.get("fold_train_target_count") != len(expected_train_rows)
    ):
        raise ValueError("V66 cached fold codebook/basis inventory changed")
    fit = payload.get("fit")
    fit_fields = {
        "optimizer_steps",
        "classification_optimizer_steps",
        "elapsed_seconds",
        "question_norm_sha256",
        "question_norm_frozen",
        "numeric_prototype_top1_accuracy",
        "numeric_prototype_mean_margin",
    }
    if (
        not isinstance(fit, Mapping)
        or set(fit) != fit_fields
        or any(
            isinstance(fit.get(field), bool)
            or not isinstance(fit.get(field), int)
            or int(fit[field]) < 1
            for field in ("optimizer_steps", "classification_optimizer_steps")
        )
        or isinstance(fit.get("elapsed_seconds"), bool)
        or not isinstance(fit.get("elapsed_seconds"), (int, float))
        or not math.isfinite(float(fit["elapsed_seconds"]))
        or float(fit["elapsed_seconds"]) < 0.0
        or not _is_sha256(fit.get("question_norm_sha256"))
        or fit.get("question_norm_frozen") is not True
        or any(
            isinstance(fit.get(field), bool)
            or not isinstance(fit.get(field), (int, float))
            or not math.isfinite(float(fit[field]))
            for field in (
                "numeric_prototype_top1_accuracy",
                "numeric_prototype_mean_margin",
            )
        )
        or not 0.0 <= float(fit["numeric_prototype_top1_accuracy"]) <= 1.0
    ):
        raise ValueError("V66 cached fold fit diagnostics changed")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("V66 cached fold records changed")
    by_key = {row.key: row for row in rows if row.pair_id == held_pair_id}
    expected_keys = _expected_supported_held_keys(rows, held_pair_id=held_pair_id)
    observed_keys: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("V66 cached record must be an object")
        if set(record) != _RECORD_FIELDS:
            raise ValueError("V66 cached record fields changed")
        key = str(record.get("scene_id")), str(record.get("question_id"))
        if key in observed_keys or key not in expected_keys:
            raise ValueError("V66 cached record inventory changed")
        row = by_key[key]
        expected_fields = {
            "pair_id": row.pair_id,
            "question_key": row.question_key,
            "answer_class_id": answer_class_id_v66(row.answer),
            "answer_type_sha256": hashlib.sha256(row.answer_type.encode()).hexdigest(),
            "list_scoring": row.answer_type in LIST_ANSWER_TYPES,
            "counterfactual_changed_side": row.route_label,
            "reference_canonical_sha256": _row_reference_sha256(row),
            "scoring_contract_sha256": _row_scoring_contract_sha256(row),
            "fold_class_supported": True,
        }
        if any(record.get(field) != value for field, value in expected_fields.items()):
            raise ValueError("V66 cached record differs from authorized training row")
        for digest_field in (
            "raw_prediction_sha256",
            "canonical_prediction_sha256",
        ):
            value = record.get(digest_field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("V66 cached prediction digest is invalid")
        exact = (
            record["canonical_prediction_sha256"]
            == record["reference_canonical_sha256"]
        )
        if record.get("canonical_exact") is not exact:
            raise ValueError("V66 cached exact boolean differs from hashes")
        observed_keys.add(key)
    if observed_keys != expected_keys:
        raise ValueError("V66 cached fold does not cover exact supported inventory")
    recomputed = behavior_metrics_v66(
        records,
        unsupported_count=48 - len(expected_keys),
    )
    if payload.get("behavior") != recomputed:
        raise ValueError("V66 cached fold metrics differ from recomputation")
    return payload


def _fit_argument_copy(args: argparse.Namespace) -> argparse.Namespace:
    """Disable V63's changed-only pair/route objectives for all-row training."""

    values = vars(args).copy()
    values["pair_delta_weight"] = 0.0
    values["route_weight"] = 0.0
    # V63 validates this legacy value against V60, though V7 ignores routing.
    values["gate_threshold"] = 0.5
    return argparse.Namespace(**values)


def _publish_v66_checkpoint_and_report(
    *,
    preflight: V63Preflight,
    control: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    report_without_checkpoint: Mapping[str, Any],
    bundle: V65RuntimeBundle,
    rows: Sequence[V63Row],
    prefixes: Mapping[str, torch.Tensor],
    generator_fn: Callable[..., str],
) -> dict[str, Any]:
    """Strict-reload, behavior-gate, seal, and atomically publish V7."""

    checkpoint = preflight.output_checkpoint
    report_path = preflight.training_report
    if checkpoint.exists() or report_path.exists():
        raise FileExistsError("V66 create-once publication destination exists")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v66-publish-", dir=checkpoint.parent))
    published_checkpoint = False
    published_report = False
    try:
        staged_checkpoint = staging / "checkpoint"
        fit_state_sha256 = v7_value_state_sha256(control)
        save_v7_control_checkpoint(
            staged_checkpoint,
            control=control,
            base_checkpoint_sha256=preflight.base_checkpoint_sha256,
            base_runtime_config_sha256=preflight.runtime_config_sha256,
            expected_training_fit_state_sha256=fit_state_sha256,
        )
        loaded_cpu = load_unsealed_v7_checkpoint_for_training_gate(
            staged_checkpoint,
            hidden_size=1536,
        )
        if v7_value_state_sha256(loaded_cpu) != fit_state_sha256:
            raise RuntimeError("V66 staged V7 state changed after strict reload")
        loaded = loaded_cpu.to(device=bundle.device, dtype=torch.float32).eval()
        signatures = _scene_signatures(
            loaded,
            {
                scene_id: prefix.to(device=bundle.device, dtype=torch.float32)
                for scene_id, prefix in prefixes.items()
            },
        )
        reloaded_fit = V66FitResult(
            control=loaded,
            signatures=signatures,
            base_fit=FitResult(
                control=loaded,
                signatures=signatures,
                basis_reconstruction={},
                elapsed_seconds=0.0,
                optimizer_steps=0,
                maximum_preclip_gradient_norm=0.0,
                final_route_loss=0.0,
                question_norm_sha256="",
                question_norm_frozen=True,
            ),
            classification_optimizer_steps=0,
            numeric_prototype_top1_accuracy=0.0,
            numeric_prototype_mean_margin=0.0,
        )
        # Re-embed raw question tokens exactly as production does; do not reuse
        # the pooled training cache for the saved-runtime gate.
        raw_questions: dict[tuple[str, str], torch.Tensor] = {}
        with torch.inference_mode():
            for row in rows:
                ids = question_token_ids(
                    bundle.runtime.language.tokenizer,
                    row.question,
                    bundle.runtime.language.device,
                )
                raw_questions[row.key] = (
                    bundle.runtime.language.model.get_input_embeddings()(ids).float()
                )
        records = generate_supported_rows_v66(
            reloaded_fit,
            rows,
            questions=raw_questions,
            supported_classes={answer_class_id_v66(row.answer) for row in rows},
            bundle=bundle,
            prefixes=prefixes,
            generator_fn=generator_fn,
        )
        metrics = behavior_metrics_v66(records, unsupported_count=0)
        checks = assess_final_v66(metrics)
        if not all(checks.values()):
            raise RuntimeError("V66 strict-loaded production behavior gate failed")
        attestation = _canonical_sha256(
            {
                "schema_version": 1,
                "artifact": "v66_saved_runtime_training_gate_attestation",
                "training_fit_state_sha256": fit_state_sha256,
                "production_device": str(bundle.device),
                "raw_question_token_embeddings_used": True,
                "behavior": metrics,
                "checks": checks,
                "answer_or_question_text_stored": False,
            }
        )
        shutil.rmtree(staged_checkpoint)
        checkpoint_hashes = save_v7_control_checkpoint(
            staged_checkpoint,
            control=control,
            base_checkpoint_sha256=preflight.base_checkpoint_sha256,
            base_runtime_config_sha256=preflight.runtime_config_sha256,
            expected_training_fit_state_sha256=fit_state_sha256,
            saved_runtime_training_gate_passed=True,
            saved_runtime_training_gate_attestation_sha256=attestation,
        )
        public, metadata = _load_control_head(
            staged_checkpoint,
            hidden_size=1536,
            device=torch.device("cpu"),
        )
        if (
            type(public) is not AlwaysOnTeacherBasisFullSceneQuestionControlV7
            or v7_value_state_sha256(public) != fit_state_sha256
            or metadata["saved_runtime_training_gate_attestation_sha256"]
            != attestation
        ):
            raise RuntimeError("V66 sealed public checkpoint failed strict reload")
        final_report = {
            **dict(report_without_checkpoint),
            "checkpoint": checkpoint_hashes,
            "saved_runtime_reload": {
                "strict_loader_passed": True,
                "architecture": metadata["architecture"],
                "training_fit_state_sha256": fit_state_sha256,
                "gate_attestation_sha256": attestation,
                "reloaded_state_exact": True,
                "raw_question_token_embeddings_used": True,
                "production_device": str(bundle.device),
                "metrics": metrics,
                "checks": checks,
                "passed_before_publication": True,
            },
        }
        staged_report = staging / "training_report.json"
        _atomic_new_json(staged_report, final_report)
        os.rename(staged_checkpoint, checkpoint)
        published_checkpoint = True
        os.rename(staged_report, report_path)
        published_report = True
        return final_report
    except BaseException:
        if published_report:
            report_path.unlink(missing_ok=True)
        if published_checkpoint:
            shutil.rmtree(checkpoint, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def train_v66(
    args: argparse.Namespace,
    *,
    runtime_provider: Callable[..., V65RuntimeBundle] | None = None,
    generator_fn: Callable[..., str] | None = None,
    supplemental_loader: Callable[
        [str | Path], tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]
    ]
    | None = None,
) -> dict[str, Any]:
    if (
        isinstance(args.prototype_classification_epochs, bool)
        or not isinstance(args.prototype_classification_epochs, int)
        or args.prototype_classification_epochs < 1
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (
                args.prototype_classification_weight,
                args.prototype_classification_temperature,
                args.prototype_value_preservation_weight,
            )
        )
    ):
        raise ValueError("V66 numeric prototype objective hyperparameters are invalid")
    fit_args = _fit_argument_copy(args)
    preflight = build_v63_preflight(fit_args)
    if len(preflight.rows) != _EXPECTED_ROWS:
        raise ValueError("V66 authenticated training inventory changed")
    validate_v66_preregistration(args.preregistration)
    validate_training_baseline_lock(
        args.training_baseline_lock,
        expected_rows=preflight.rows,
    )
    teachers, teacher_audit = load_combined_verified_teachers_v66(
        preflight,
        args.supplemental_teacher_cache,
        supplemental_loader=supplemental_loader,
    )
    work = _resolve(args.work_directory)
    if work == preflight.output_checkpoint or work == preflight.training_report:
        raise ValueError("V66 work path overlaps a final output")
    manifest = _run_manifest(preflight, teacher_audit, fit_args)
    _prepare_work_directory(work, manifest)
    bundle = (runtime_provider or _load_runtime)(
        preflight, requested_device=args.device
    )
    generation = generator_fn or _generate_with_control

    fold_payloads: list[dict[str, Any]] = []
    for fold_index, held_pair in enumerate(TRAIN_PAIR_IDS):
        fold_file = _fold_path(work, held_pair)
        if fold_file.exists():
            cached = _validate_cached_fold_v66(
                json.loads(fold_file.read_text(encoding="utf-8")),
                held_pair_id=held_pair,
                run_signature_sha256=str(manifest["run_signature_sha256"]),
                rows=preflight.rows,
            )
            fold_payloads.append(cached)
            continue
        train_rows = tuple(row for row in preflight.rows if row.pair_id != held_pair)
        held_rows = tuple(row for row in preflight.rows if row.pair_id == held_pair)
        train_keys = {row.key for row in train_rows}
        fold_teachers = {key: value for key, value in teachers.items() if key in train_keys}
        codebook = build_hybrid_answer_prototype_codebook_v66(
            train_rows,
            fold_teachers,
            expected_class_count=None,
            scope=f"fold_{held_pair}",
            forbidden_pair_id=held_pair,
        )
        basis = _codebook_basis(codebook, fit_args.basis_rank)
        fit = _fit_always_on(
            rows=train_rows,
            targets=codebook.targets,
            preflight=preflight,
            questions=bundle.question_embeddings,
            basis=basis,
            args=fit_args,
            seed=fit_args.seed + (fold_index + 1) * 100_003,
            phase=f"v66_cv_{held_pair}",
        )
        held_signatures = _scene_signatures(
            fit.control,
            {
                scene_id: preflight.prefixes[scene_id]
                for scene_id in sorted({row.scene_id for row in held_rows})
            },
        )
        fit = V66FitResult(
            control=fit.control,
            signatures=held_signatures,
            base_fit=fit.base_fit,
            classification_optimizer_steps=fit.classification_optimizer_steps,
            numeric_prototype_top1_accuracy=fit.numeric_prototype_top1_accuracy,
            numeric_prototype_mean_margin=fit.numeric_prototype_mean_margin,
        )
        supported_classes = set(codebook.prototypes)
        records = generate_supported_rows_v66(
            fit,
            held_rows,
            questions=bundle.question_embeddings,
            supported_classes=supported_classes,
            bundle=bundle,
            prefixes=preflight.prefixes,
            generator_fn=generation,
        )
        unsupported_count = len(held_rows) - len(records)
        payload = {
            "schema_version": 1,
            "artifact": _FOLD_ARTIFACT,
            "run_signature_sha256": manifest["run_signature_sha256"],
            "held_pair_id": held_pair,
            "held_rows_used_for_optimization": False,
            "held_teacher_sources_used": False,
            "fold_codebook_sha256": codebook.sha256,
            "fold_basis_sha256": _tensor_sha256(basis),
            "fold_class_count": len(codebook.prototypes),
            "fold_train_target_count": len(codebook.targets),
            "fit": {
                "optimizer_steps": fit.base_fit.optimizer_steps,
                "classification_optimizer_steps": fit.classification_optimizer_steps,
                "elapsed_seconds": fit.base_fit.elapsed_seconds,
                "question_norm_sha256": fit.base_fit.question_norm_sha256,
                "question_norm_frozen": fit.base_fit.question_norm_frozen,
                "numeric_prototype_top1_accuracy": (
                    fit.numeric_prototype_top1_accuracy
                ),
                "numeric_prototype_mean_margin": fit.numeric_prototype_mean_margin,
            },
            "behavior": behavior_metrics_v66(
                records,
                unsupported_count=unsupported_count,
            ),
            "records": list(records),
        }
        _atomic_new_json(fold_file, payload)
        fold_payloads.append(payload)

    all_records = [record for fold in fold_payloads for record in fold["records"]]
    unsupported = sum(int(fold["behavior"]["unsupported_total"]) for fold in fold_payloads)
    cv_metrics = behavior_metrics_v66(all_records, unsupported_count=unsupported)
    cv_checks = assess_cv_v66(cv_metrics)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "v66b_allrow_paired_opposite_pair_disjoint_training",
        "promotion_eligible": False,
        "checkpoint": None,
        "authorization": {
            "baseline_lock_sha256": preflight.baseline_lock_sha256,
            "preregistration_sha256": _sha256_file(_resolve(args.preregistration)),
            "filtered_training_qa_sha256": preflight.filtered_train_sha256,
            "training_baseline_lock_sha256": _sha256_file(
                _resolve(args.training_baseline_lock)
            ),
        },
        "architecture": {
            "name": "always_on_teacher_basis_full_scene_control_v7",
            "complete_scene_prefix": True,
            "scene_latents": 256,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "runtime_answer_codebook": False,
        },
        "teacher_audit": teacher_audit,
        "work_manifest_sha256": manifest["run_signature_sha256"],
        "thresholds": asdict(V66B_BEHAVIOR_THRESHOLDS),
        "cv": {
            "protocol": "leave_one_counterfactual_pair_out_all_576_rows",
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
                        "fold_class_count",
                        "fold_train_target_count",
                        "fit",
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
            "deferred_final_loaded": False,
            "question_or_answer_text_stored": False,
        },
    }
    if not report["cv"]["passed"]:
        report["terminal_reason"] = "pair_disjoint_allrow_behavior_gate_failed"
        _write_training_report(preflight.training_report, report)
        return report

    all_codebook = build_hybrid_answer_prototype_codebook_v66(
        preflight.rows,
        teachers,
        expected_class_count=_EXPECTED_CLASSES,
        scope="final_all_training",
    )
    final_basis = _codebook_basis(all_codebook, fit_args.basis_rank)
    final_fit = _fit_always_on(
        rows=preflight.rows,
        targets=all_codebook.targets,
        preflight=preflight,
        questions=bundle.question_embeddings,
        basis=final_basis,
        args=fit_args,
        seed=fit_args.seed + 9_999_991,
        phase="v66_final_all_training",
    )
    final_records = generate_supported_rows_v66(
        final_fit,
        preflight.rows,
        questions=bundle.question_embeddings,
        supported_classes=set(all_codebook.prototypes),
        bundle=bundle,
        prefixes=preflight.prefixes,
        generator_fn=generation,
    )
    final_metrics = behavior_metrics_v66(final_records, unsupported_count=0)
    final_checks = assess_final_v66(final_metrics)
    report["final_fit"] = {
        "metrics": final_metrics,
        "checks": final_checks,
        "passed": all(final_checks.values()),
        "codebook_sha256": all_codebook.sha256,
        "basis_sha256": _tensor_sha256(final_basis),
        "fit_state_sha256": _canonical_sha256(
            {
                name: _tensor_sha256(value)
                for name, value in final_fit.control.state_dict().items()
            }
        ),
        "optimizer_steps": final_fit.base_fit.optimizer_steps,
        "classification_optimizer_steps": final_fit.classification_optimizer_steps,
        "elapsed_seconds": final_fit.base_fit.elapsed_seconds,
        "numeric_prototype_top1_accuracy": (
            final_fit.numeric_prototype_top1_accuracy
        ),
        "numeric_prototype_mean_margin": final_fit.numeric_prototype_mean_margin,
    }
    if not report["final_fit"]["passed"]:
        report["terminal_reason"] = "all_training_behavior_gate_failed"
        _write_training_report(preflight.training_report, report)
        return report

    paired_opposite_records = generate_paired_opposite_scene_rows_v66(
        final_fit,
        preflight.rows,
        questions=bundle.question_embeddings,
        bundle=bundle,
        prefixes=preflight.prefixes,
        generator_fn=generation,
    )
    dependence_metrics = paired_opposite_metrics_v66(paired_opposite_records)
    dependence_checks = paired_opposite_checks_v66(dependence_metrics)
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
    report["terminal_reason"] = (
        "training_and_paired_dependence_gates_passed_checkpoint_saved"
    )
    return _publish_v66_checkpoint_and_report(
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
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=660066)
    parser.add_argument("--basis-rank", type=int, default=112)
    parser.add_argument("--moment-count", type=int, default=8)
    parser.add_argument("--interaction-dim", type=int, default=32)
    parser.add_argument("--trunk-dim", type=int, default=192)
    parser.add_argument("--maximum-control-rms", type=float, default=0.25)
    parser.add_argument("--initial-control-rms", type=float, default=0.075)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--changed-repeats", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--coefficient-weight", type=float, default=3.0)
    parser.add_argument("--log-rms-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=2.0)
    parser.add_argument("--relative-mse-weight", type=float, default=0.25)
    parser.add_argument("--prototype-classification-epochs", type=int, default=40)
    parser.add_argument("--prototype-classification-weight", type=float, default=1.0)
    parser.add_argument(
        "--prototype-classification-temperature", type=float, default=0.07
    )
    parser.add_argument(
        "--prototype-value-preservation-weight", type=float, default=1.0
    )
    parser.add_argument("--pair-delta-weight", type=float, default=0.0)
    parser.add_argument("--route-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs < 1 or args.basis_rank < 1 or not math.isfinite(args.learning_rate):
        raise ValueError("V66 hyperparameters are invalid")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    report = train_v66(args)
    print(json.dumps(report, sort_keys=True))
    return (
        0
        if report.get("terminal_reason")
        == "training_and_paired_dependence_gates_passed_checkpoint_saved"
        and isinstance(report.get("checkpoint"), Mapping)
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V66B_BEHAVIOR_THRESHOLDS",
    "V66_BEHAVIOR_THRESHOLDS",
    "V66BehaviorThresholds",
    "V66bBehaviorThresholds",
    "assess_cv_v66",
    "assess_final_v66",
    "behavior_metrics_v66",
    "generate_paired_opposite_scene_rows_v66",
    "generate_supported_rows_v66",
    "load_combined_verified_teachers_v66",
    "paired_opposite_checks_v66",
    "paired_opposite_metrics_v66",
    "train_v66",
]
