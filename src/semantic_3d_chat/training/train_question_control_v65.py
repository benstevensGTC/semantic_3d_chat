"""Canonical-answer prototype distillation with contamination-free pair CV.

V63 established that independently optimized, answer-equivalent soft prompts
are not identifiable enough to regress across held pairs.  V65 removes that
arbitrary target variation using only the authenticated V62 training boundary:
verified numeric teachers are grouped into normalized-answer classes and a
deterministic medoid prompt becomes the numeric prototype for each class.
Crucially, every leave-one-pair-out fold builds both its codebook and output
basis from the other eleven pairs.  A held teacher, answer prototype, or basis
direction can therefore never reach that fold's optimization.

Some answer classes occur in only one counterfactual pair.  Those sides are
reported as vocabulary-unsupported and are excluded from the primary CV gate;
their exploratory generations are retained as hashes only.  The final
all-training fit uses the complete 21-class training vocabulary.  Runtime
checkpoints contain only the distilled continuous controller; answer strings
and training codebooks are absent.

Every fold is judged primarily by actual greedy local-Gemma answers on its held
changed sides.  Fold results are resumable, training-only, opaque/hash-only
artifacts.  No runtime checkpoint is emitted unless all pair-disjoint behavior
gates and the final all-training behavioral gate pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from semantic_3d_chat.chat.question_control_runtime import (
    _load_control_head,
    question_control_training_artifact_root,
)
from semantic_3d_chat.evaluation.metrics import (
    LIST_ANSWER_TYPES,
    normalize_answer,
    normalize_answer_items,
)
from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import TRAIN_PAIR_IDS
from semantic_3d_chat.language.local_lm import question_token_ids
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
    teacher_output_basis,
)
from semantic_3d_chat.scene_encoder.question_control_v6 import (
    MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
)
from semantic_3d_chat.training.question_control_v6_checkpoint import (
    load_unsealed_v6_checkpoint_for_training_gate,
    save_v6_control_checkpoint,
    v6_value_state_sha256,
)
from semantic_3d_chat.training.soft_prompt_teacher_v62 import PROMPT_SHAPE
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _resolve,
    _select_training_device,
    _sha256_file,
    _write_training_report,
    freeze_base_runtime,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _disable_decoder_checkpointing,
    _generate_with_control,
    _pooled_question_embedding,
)
from semantic_3d_chat.training.train_question_control_v63 import (
    FitResult,
    V63Preflight,
    V63Row,
    _basis_coverage,
    _changed_units,
    _fit_controller,
    _log_event,
    _measure_reconstruction,
    _optimizer_step,
    _scene_signatures,
    build_v63_preflight,
)

_EXPECTED_ROWS: Final[int] = 576
_EXPECTED_SCENES: Final[int] = 24
_EXPECTED_PAIRS: Final[int] = 12
_EXPECTED_CHANGED_SIDES: Final[int] = 80
_EXPECTED_CHANGED_UNITS: Final[int] = 40
_EXPECTED_ANSWER_CLASSES: Final[int] = 21
_EXPECTED_HIDDEN_SIZE: Final[int] = 1536
_PINNED_TRAINING_QUESTIONS_SHA256: Final[str] = (
    "038c06addb1ec7e50613ae8134607b62c7b35cdd0e216a58d5fc35676d2e01e6"
)
_PINNED_V54_CHECKPOINT_SHA256: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
_PINNED_V54_RUNTIME_CONFIG_FILE_SHA256: Final[str] = (
    "891c58faaaa5fcd2ed76c7e3871f14c5d8c5ae2e05d9fa4ddd5193773d40e56b"
)
_PINNED_V54_RUNTIME_CONFIG_EFFECTIVE_SHA256: Final[str] = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
_PINNED_V54_TRAINING_PREDICTIONS_SHA256: Final[str] = (
    "001789f1e5791444c0aa2794a2ec0b1ac09369677e5f72bea8c54121f69f3a98"
)
_PINNED_V54_TRAINING_PROVENANCE_SHA256: Final[str] = (
    "af3e80191c4b093222c19e9f6dccad6a92b740d6091228255758a439ad3ea09d"
)
_PINNED_TRAINING_SCENE_MAP_MANIFEST_SHA256: Final[str] = (
    "bc35bb9a3b677298db283e8650a552bcd4a5bdf0b47a4b50a366fc505a61c114"
)
_PINNED_V65_TRAINING_BASELINE_LOCK_SHA256: Final[str] = (
    "b1f20e64889116cceb0904ecb3842a6e43fcd6fa3cb0675c32a24f4d278e55e6"
)
_EXPECTED_TRAINING_SCENES: Final[frozenset[str]] = frozenset(
    f"scene_{number:06d}" for number in (*range(11, 25), *range(31, 39), 53, 54)
)
_TRAINING_BASELINE_SCHEMA: Final[str] = "semantic_3d_chat.v65.v54_training_baseline_lock.v1"
_WORK_ARTIFACT: Final[str] = "v65_canonical_answer_behavioral_cv_work_v2"
_FOLD_ARTIFACT: Final[str] = "v65_pair_heldout_behavioral_fold_v2"
_GENERATION_SEMANTICS: Final[str] = (
    "runtime_v6_magnitude_gate_checked_then_exact_control_or_no_token_path"
)
_BEHAVIOR_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "scene_id",
        "question_id",
        "pair_id",
        "question_key",
        "answer_class_id",
        "fold_class_supported",
        "raw_prediction_sha256",
        "canonical_prediction_sha256",
        "reference_canonical_sha256",
        "scoring_contract_sha256",
        "canonical_exact",
    }
)


@dataclass(frozen=True)
class V65BehaviorThresholds:
    held_supported_side_exact_minimum: int = 45
    held_supported_side_total: int = 60
    held_fully_supported_complete_unit_minimum: int = 19
    held_fully_supported_unit_total: int = 28
    eligible_folds_with_exact_hit_minimum: int = 7
    eligible_fold_total: int = 8
    inventory_fold_total: int = 12
    held_inventory_side_total: int = 80
    held_inventory_unit_total: int = 40
    held_unsupported_side_total: int = 20
    held_retention_exact_no_control_total: int = 496
    final_train_side_exact_minimum: int = 76
    final_train_side_total: int = 80
    final_train_complete_unit_minimum: int = 36
    final_train_complete_unit_total: int = 40
    final_retention_exact_no_control_total: int = 496


V65_BEHAVIOR_THRESHOLDS: Final[V65BehaviorThresholds] = V65BehaviorThresholds()


@dataclass(frozen=True)
class AnswerPrototypeCodebookV65:
    prototypes: dict[str, torch.Tensor]
    targets: dict[tuple[str, str], torch.Tensor]
    class_by_key: dict[tuple[str, str], str]
    manifest: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class V65RuntimeBundle:
    runtime: Any
    question_embeddings: dict[tuple[str, str], torch.Tensor]
    device: torch.device
    model_dtype: torch.dtype
    audit: dict[str, Any]


@dataclass(frozen=True)
class V65FitResult:
    control: MagnitudeGatedTeacherBasisFullSceneQuestionControlV6
    signatures: dict[str, torch.Tensor]
    base_fit: FitResult
    routing_optimizer_steps: int
    maximum_routing_gradient_norm: float


@dataclass(frozen=True)
class V65TrainingBaseline:
    lock_sha256: str
    required_output_hashes: dict[tuple[str, str], str]
    prefix_hashes: dict[str, str]
    payload: dict[str, Any]


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


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().float().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _opaque_key_inventory_sha256(keys: Sequence[tuple[str, str]]) -> str:
    return _canonical_sha256(
        [
            {"scene_id": scene_id, "question_id": question_id}
            for scene_id, question_id in sorted(keys)
        ]
    )


def validate_training_baseline_lock(
    path: str | Path,
    *,
    expected_rows: Sequence[V63Row] | None = None,
) -> V65TrainingBaseline:
    """Authenticate the hash-only V54 outputs for all 576 training rows."""

    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V65 training baseline lock is unavailable")
    if _sha256_file(source) != _PINNED_V65_TRAINING_BASELINE_LOCK_SHA256:
        raise ValueError("V65 training baseline lock differs from its create-once pin")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V65 training baseline lock must be a JSON object")
    required = {
        "schema",
        "schema_version",
        "artifact",
        "status",
        "questions_manifest_sha256",
        "predictions_sha256",
        "prediction_provenance_sha256",
        "prediction_provenance_identity_sha256",
        "v54_runtime_config_file_sha256",
        "v54_runtime_config_effective_sha256",
        "scene_map_manifest_sha256",
        "v54_checkpoint_sha256",
        "v54_checkpoint_files",
        "question_count",
        "scene_count",
        "question_key_inventory_sha256",
        "scene_prefix_hashes",
        "one_invariant_prefix_per_scene",
        "distinct_prefix_per_scene",
        "required_output_hashes",
        "required_output_hashes_sha256",
        "answer_or_question_text_stored",
        "training_scenes_only",
        "validation_inputs_loaded",
        "scorer_inputs_loaded",
        "oracle_loaded",
        "fresh_development_loaded",
        "deferred_final_loaded",
    }
    if set(payload) != required:
        raise ValueError("V65 training baseline lock fields changed")
    if (
        payload.get("schema") != _TRAINING_BASELINE_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("artifact") != "v65_v54_training_baseline_lock"
        or payload.get("status") != "locked_before_v65_training"
        or payload.get("questions_manifest_sha256") != _PINNED_TRAINING_QUESTIONS_SHA256
        or payload.get("v54_checkpoint_sha256") != _PINNED_V54_CHECKPOINT_SHA256
        or payload.get("v54_runtime_config_file_sha256") != _PINNED_V54_RUNTIME_CONFIG_FILE_SHA256
        or payload.get("v54_runtime_config_effective_sha256")
        != _PINNED_V54_RUNTIME_CONFIG_EFFECTIVE_SHA256
        or payload.get("predictions_sha256") != _PINNED_V54_TRAINING_PREDICTIONS_SHA256
        or payload.get("prediction_provenance_sha256") != _PINNED_V54_TRAINING_PROVENANCE_SHA256
        or payload.get("scene_map_manifest_sha256") != _PINNED_TRAINING_SCENE_MAP_MANIFEST_SHA256
        or payload.get("question_count") != _EXPECTED_ROWS
        or payload.get("scene_count") != _EXPECTED_SCENES
        or payload.get("one_invariant_prefix_per_scene") is not True
        or payload.get("distinct_prefix_per_scene") is not True
        or payload.get("answer_or_question_text_stored") is not False
        or payload.get("training_scenes_only") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "validation_inputs_loaded",
                "scorer_inputs_loaded",
                "oracle_loaded",
                "fresh_development_loaded",
                "deferred_final_loaded",
            )
        )
    ):
        raise ValueError("V65 training baseline prerequisite binding is invalid")
    for field in (
        "predictions_sha256",
        "prediction_provenance_sha256",
        "prediction_provenance_identity_sha256",
        "v54_runtime_config_file_sha256",
        "v54_runtime_config_effective_sha256",
        "scene_map_manifest_sha256",
        "question_key_inventory_sha256",
        "required_output_hashes_sha256",
    ):
        if not _is_sha256(payload.get(field)):
            raise ValueError("V65 training baseline contains an invalid digest")
    checkpoint_files = payload.get("v54_checkpoint_files")
    if (
        not isinstance(checkpoint_files, list)
        or [item.get("path") for item in checkpoint_files if isinstance(item, dict)]
        != ["adapter.safetensors", "metadata.json", "runtime_metadata.json"]
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "size_bytes"}
            or not _is_sha256(item.get("sha256"))
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 1
            for item in checkpoint_files
        )
    ):
        raise ValueError("V65 training baseline checkpoint-file inventory changed")
    prefixes = payload.get("scene_prefix_hashes")
    if (
        not isinstance(prefixes, dict)
        or set(prefixes) != _EXPECTED_TRAINING_SCENES
        or any(
            not isinstance(scene_id, str)
            or not scene_id.startswith("scene_")
            or not _is_sha256(prefix_hash)
            for scene_id, prefix_hash in prefixes.items()
        )
        or len(set(prefixes.values())) != _EXPECTED_SCENES
    ):
        raise ValueError("V65 training baseline prefix inventory changed")
    records = payload.get("required_output_hashes")
    if not isinstance(records, list) or len(records) != _EXPECTED_ROWS:
        raise ValueError("V65 training baseline output inventory changed")
    by_key: dict[tuple[str, str], str] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"scene_id", "question_id", "raw_output_sha256"}
            or not all(
                isinstance(record[field], str) and record[field]
                for field in ("scene_id", "question_id")
            )
            or not _is_sha256(record.get("raw_output_sha256"))
        ):
            raise ValueError("V65 training baseline has an invalid hash record")
        key = str(record["scene_id"]), str(record["question_id"])
        if key in by_key:
            raise ValueError("V65 training baseline contains duplicate opaque keys")
        by_key[key] = str(record["raw_output_sha256"])
    if payload["question_key_inventory_sha256"] != _opaque_key_inventory_sha256(
        tuple(by_key)
    ) or payload["required_output_hashes_sha256"] != _canonical_sha256(records):
        raise ValueError("V65 training baseline hash inventory changed")
    if {scene_id for scene_id, _question_id in by_key} != _EXPECTED_TRAINING_SCENES or any(
        sum(1 for scene_id, _question_id in by_key if scene_id == expected_scene) != 24
        for expected_scene in _EXPECTED_TRAINING_SCENES
    ):
        raise ValueError("V65 training baseline opaque scene inventory changed")
    if expected_rows is not None:
        expected_keys = {row.key for row in expected_rows}
        if set(by_key) != expected_keys or set(prefixes) != {row.scene_id for row in expected_rows}:
            raise ValueError("V65 training baseline differs from exact training inventory")
    return V65TrainingBaseline(
        lock_sha256=_sha256_file(source),
        required_output_hashes=by_key,
        prefix_hashes={str(key): str(value) for key, value in prefixes.items()},
        payload=payload,
    )


def _answer_class_id(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V65 canonical training answer normalizes to empty")
    return f"answer_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:20]}"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_scoring_value(value: object, *, answer_type: str) -> object:
    """Return the exact canonical value used by the hash-only fold scorer."""

    if answer_type in LIST_ANSWER_TYPES:
        return sorted(normalize_answer_items(value))
    return normalize_answer(value)


def _canonical_scoring_sha256(value: object, *, answer_type: str) -> str:
    return _canonical_sha256(
        {
            "scorer": (
                "order_insensitive_items_v1"
                if answer_type in LIST_ANSWER_TYPES
                else "normalized_exact_v1"
            ),
            "value": _canonical_scoring_value(value, answer_type=answer_type),
        }
    )


def _row_reference_sha256(row: V63Row) -> str:
    reference: object = (
        row.answer_items
        if row.answer_type in LIST_ANSWER_TYPES and row.answer_items
        else row.answer
    )
    return _canonical_scoring_sha256(reference, answer_type=row.answer_type)


def _row_scoring_contract_sha256(row: V63Row) -> str:
    return _canonical_sha256(
        {
            "scene_id": row.scene_id,
            "question_id": row.question_id,
            "pair_id": row.pair_id,
            "question_key": row.question_key,
            "answer_class_id": _answer_class_id(row.answer),
            "reference_canonical_sha256": _row_reference_sha256(row),
            "scorer": (
                "order_insensitive_items_v1"
                if row.answer_type in LIST_ANSWER_TYPES
                else "normalized_exact_v1"
            ),
        }
    )


def _validate_behavior_record(
    record: object,
    *,
    expected_row: V63Row | None = None,
    expected_supported: bool | None = None,
) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or set(record) != _BEHAVIOR_RECORD_FIELDS:
        raise ValueError("V65 behavioral record contract changed")
    opaque_fields = ("scene_id", "question_id", "pair_id", "question_key")
    if any(not isinstance(record[field], str) or not record[field] for field in opaque_fields):
        raise ValueError("V65 behavioral record has an invalid opaque identity")
    answer_class_id = record["answer_class_id"]
    if (
        not isinstance(answer_class_id, str)
        or not answer_class_id.startswith("answer_")
        or len(answer_class_id) != 27
        or any(character not in "0123456789abcdef" for character in answer_class_id[7:])
    ):
        raise ValueError("V65 behavioral record has an invalid numeric class ID")
    if type(record["fold_class_supported"]) is not bool:
        raise ValueError("V65 behavioral record has invalid vocabulary support")
    for digest_field in (
        "raw_prediction_sha256",
        "canonical_prediction_sha256",
        "reference_canonical_sha256",
        "scoring_contract_sha256",
    ):
        if not _is_sha256(record[digest_field]):
            raise ValueError("V65 behavioral record has an invalid prediction digest")
    if type(record["canonical_exact"]) is not bool:
        raise ValueError("V65 behavioral record has a non-boolean exact result")
    # Correctness is derivable from the two authenticated canonical digests;
    # never trust a cached boolean on its own.
    if record["canonical_exact"] is not (
        record["canonical_prediction_sha256"] == record["reference_canonical_sha256"]
    ):
        raise ValueError("V65 cached exact result differs from its canonical digests")
    if expected_row is not None:
        expected_identity = {
            "scene_id": expected_row.scene_id,
            "question_id": expected_row.question_id,
            "pair_id": expected_row.pair_id,
            "question_key": expected_row.question_key,
            "answer_class_id": _answer_class_id(expected_row.answer),
            "reference_canonical_sha256": _row_reference_sha256(expected_row),
            "scoring_contract_sha256": _row_scoring_contract_sha256(expected_row),
        }
        if any(record[field] != value for field, value in expected_identity.items()):
            raise ValueError("V65 cached behavior record differs from held inventory")
    if expected_supported is not None and record["fold_class_supported"] is not expected_supported:
        raise ValueError("V65 cached vocabulary-support status changed")
    return record


def _deterministic_medoid(
    members: Sequence[tuple[tuple[str, str], torch.Tensor]],
) -> tuple[tuple[str, str], torch.Tensor, dict[str, float]]:
    if not members:
        raise ValueError("V65 cannot select a medoid from an empty answer class")
    ordered = sorted(members, key=lambda item: item[0])
    if any(float(value.detach().float().square().sum()) <= 1e-12 for _key, value in ordered):
        raise ValueError("V65 canonical-answer medoid candidates must be nonzero")
    flattened = torch.stack(
        [F.normalize(value.detach().cpu().float().flatten(), dim=0) for _key, value in ordered]
    )
    similarities = flattened @ flattened.T
    mean_similarities = similarities.mean(dim=1)
    best_score = float(mean_similarities.max())
    # A stable opaque-key tie break avoids dependence on source order or BLAS
    # sorting behavior while retaining the exact verified teacher tensor.
    candidates = [
        index
        for index, score in enumerate(mean_similarities.tolist())
        if abs(score - best_score) <= 1e-8
    ]
    selected = min(candidates, key=lambda index: ordered[index][0])
    return (
        ordered[selected][0],
        ordered[selected][1].detach().cpu().float().clone(),
        {
            "selected_mean_flat_prompt_cosine": float(mean_similarities[selected]),
            "class_mean_pairwise_flat_prompt_cosine": float(similarities.mean()),
            "class_minimum_pairwise_flat_prompt_cosine": float(similarities.min()),
        },
    )


def build_answer_prototype_codebook(
    rows: Sequence[V63Row],
    teachers: Mapping[tuple[str, str], torch.Tensor],
    *,
    expected_class_count: int | None = _EXPECTED_ANSWER_CLASSES,
    expected_prompt_shape: tuple[int, int, int] = PROMPT_SHAPE,
    scope: str = "final_all_training",
    forbidden_pair_id: str | None = None,
) -> AnswerPrototypeCodebookV65:
    """Create numeric medoids using only the rows explicitly supplied."""

    if not scope or any(character.isspace() for character in scope):
        raise ValueError("V65 codebook scope must be one opaque token")
    if forbidden_pair_id is not None and any(row.pair_id == forbidden_pair_id for row in rows):
        raise AssertionError("V65 held pair reached its fold-local codebook")

    changed = [row for row in rows if row.route_label]
    if {row.key for row in changed} != set(teachers):
        raise ValueError("V65 changed rows and verified numeric teachers differ")
    groups: defaultdict[str, list[tuple[tuple[str, str], torch.Tensor]]] = defaultdict(list)
    normalized_by_class: dict[str, str] = {}
    class_by_key: dict[tuple[str, str], str] = {}
    for row in changed:
        normalized = normalize_answer(row.answer)
        class_id = _answer_class_id(row.answer)
        previous = normalized_by_class.setdefault(class_id, normalized)
        if previous != normalized:
            raise RuntimeError("V65 answer-class digest collision")
        teacher = teachers[row.key]
        if tuple(teacher.shape) != expected_prompt_shape or not torch.isfinite(teacher).all():
            raise ValueError("V65 numeric teacher shape or finiteness changed")
        groups[class_id].append((row.key, teacher))
        class_by_key[row.key] = class_id
    if expected_class_count is not None and len(groups) != expected_class_count:
        raise ValueError(
            f"V65 requires exactly {expected_class_count} normalized training answers; "
            f"observed={len(groups)}"
        )

    prototypes: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for class_id in sorted(groups):
        selected_key, prototype, diagnostics = _deterministic_medoid(groups[class_id])
        prototypes[class_id] = prototype
        records.append(
            {
                "answer_class_id": class_id,
                "member_count": len(groups[class_id]),
                "selected_medoid_scene_id": selected_key[0],
                "selected_medoid_question_id": selected_key[1],
                "prototype_sha256": _tensor_sha256(prototype),
                "prototype_shape": list(prototype.shape),
                **diagnostics,
            }
        )
    targets = {row.key: prototypes[class_by_key[row.key]] for row in changed}
    for class_id, members in groups.items():
        member_targets = [targets[key] for key, _teacher in members]
        if any(not torch.equal(member_targets[0], value) for value in member_targets[1:]):
            raise RuntimeError("V65 equal answers did not receive byte-identical targets")
        if _tensor_sha256(member_targets[0]) != _tensor_sha256(prototypes[class_id]):
            raise RuntimeError("V65 answer prototype changed during target assignment")

    source_pair_ids = sorted({row.pair_id for row in changed})
    manifest = {
        "schema_version": 2,
        "artifact": "v65_training_only_answer_numeric_codebook_v2",
        "selection": "maximum_mean_flat_prompt_cosine_medoid_then_opaque_key",
        "scope": scope,
        "fold_local": forbidden_pair_id is not None,
        "forbidden_pair_id": forbidden_pair_id,
        "source_pair_ids": source_pair_ids,
        "forbidden_pair_absent": (
            forbidden_pair_id is None or forbidden_pair_id not in source_pair_ids
        ),
        "scene_question_teachers_outside_source_pairs_used": False,
        "answer_strings_serialized": False,
        "runtime_load_permitted": False,
        "answer_class_count": len(records),
        "teacher_side_count": len(changed),
        "prompt_shape": list(expected_prompt_shape),
        "records": records,
    }
    return AnswerPrototypeCodebookV65(
        prototypes=prototypes,
        targets=targets,
        class_by_key=class_by_key,
        manifest=manifest,
        sha256=_canonical_sha256(manifest),
    )


def codebook_output_basis(
    codebook: AnswerPrototypeCodebookV65,
    *,
    requested_rank: int,
) -> torch.Tensor:
    stack = torch.cat(
        [codebook.prototypes[class_id] for class_id in sorted(codebook.prototypes)],
        dim=0,
    )
    maximum_rank = min(stack.shape[0] * stack.shape[1], stack.shape[2])
    rank = min(requested_rank, maximum_rank)
    return teacher_output_basis(stack, rank=rank)


def _pair_fold_partition(
    rows: Sequence[V63Row],
    targets: Mapping[tuple[str, str], torch.Tensor],
    *,
    held_pair_id: str,
) -> tuple[
    tuple[V63Row, ...],
    tuple[V63Row, ...],
    dict[tuple[str, str], torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
]:
    """Make the auditable example-level partition used by one behavioral fold."""

    if held_pair_id not in TRAIN_PAIR_IDS:
        raise ValueError(f"V65 fold pair is not authorized: {held_pair_id}")
    expected_target_keys = {row.key for row in rows if row.route_label}
    if set(targets) != expected_target_keys:
        raise ValueError("V65 fold targets differ from the changed training inventory")
    train_rows = tuple(row for row in rows if row.pair_id != held_pair_id)
    held_rows = tuple(row for row in rows if row.pair_id == held_pair_id)
    if not train_rows or not held_rows:
        raise ValueError("V65 fold partition lost its training or held rows")
    train_targets = {row.key: targets[row.key] for row in train_rows if row.route_label}
    held_targets = {row.key: targets[row.key] for row in held_rows if row.route_label}
    if (
        set(train_targets) & set(held_targets)
        or set(train_targets) | set(held_targets) != set(targets)
        or any(row.pair_id == held_pair_id for row in train_rows)
        or any(row.pair_id != held_pair_id for row in held_rows)
    ):
        raise AssertionError("V65 held scene/question examples reached optimization")
    return train_rows, held_rows, train_targets, held_targets


def _work_path(
    value: str | Path,
    *,
    config: Mapping[str, Any],
    output_checkpoint: Path,
    training_report: Path,
) -> Path:
    path = _resolve(value)
    training_root = question_control_training_artifact_root(config).resolve()
    try:
        relative = path.relative_to(training_root)
    except ValueError as exc:
        raise ValueError(f"V65 work directory must remain below {training_root}") from exc
    if not relative.parts:
        raise ValueError("V65 work directory cannot replace the training root")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"V65 work directory path contains a symlink: {current}")
    if (
        path == output_checkpoint
        or path == training_report
        or path in output_checkpoint.parents
        or output_checkpoint in path.parents
        or path in training_report.parents
        or training_report in path.parents
    ):
        raise ValueError("V65 work directory must be disjoint from final outputs")
    return path


def _run_manifest(
    *,
    preflight: V63Preflight,
    args: argparse.Namespace,
) -> dict[str, Any]:
    implementation_files = {
        "v65_trainer": _sha256_file(Path(__file__).resolve()),
        "v63_fitter": _sha256_file(Path(_fit_controller.__code__.co_filename).resolve()),
        "gemma_control_generator": _sha256_file(
            Path(_generate_with_control.__code__.co_filename).resolve()
        ),
        "canonical_scorer": _sha256_file(Path(normalize_answer.__code__.co_filename).resolve()),
        "v3_controller": _sha256_file(
            Path(TeacherBasisFullSceneQuestionControlV3.__init__.__code__.co_filename).resolve()
        ),
    }
    identity = {
        "schema_version": 2,
        "artifact": _WORK_ARTIFACT,
        "baseline_lock_sha256": preflight.baseline_lock_sha256,
        "training_baseline_lock_sha256": _sha256_file(_resolve(args.training_baseline_lock)),
        "filtered_training_qa_sha256": preflight.filtered_train_sha256,
        "teacher_metadata_sha256": preflight.teacher_metadata_sha256,
        "teacher_weights_sha256": preflight.teacher_weights_sha256,
        "prefix_cache_manifest_sha256": preflight.prefix_manifest_sha256,
        "base_checkpoint_sha256": preflight.base_checkpoint_sha256,
        "runtime_config_sha256": preflight.runtime_config_sha256,
        "source_v60_checkpoint_sha256": preflight.source_v60_checkpoint_sha256,
        "source_v60_question_norm_sha256": (preflight.source_v60_question_norm_sha256),
        "fold_codebooks_and_bases_built_after_partition": True,
        "held_pair_teachers_used_in_fold_codebook_or_basis": False,
        "implementation_files_sha256": implementation_files,
        "generator_revision_sha256": implementation_files["gemma_control_generator"],
        "generation_semantics": _GENERATION_SEMANTICS,
        "pair_ids": list(TRAIN_PAIR_IDS),
        "seed": args.seed,
        "fit": {
            "basis_rank_requested": args.basis_rank,
            "moment_count": args.moment_count,
            "interaction_dim": args.interaction_dim,
            "trunk_dim": args.trunk_dim,
            "maximum_control_rms": args.maximum_control_rms,
            "initial_control_rms": args.initial_control_rms,
            "gate_threshold": args.gate_threshold,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "changed_repeats": args.changed_repeats,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "coefficient_weight": args.coefficient_weight,
            "log_rms_weight": args.log_rms_weight,
            "reconstruction_weight": args.reconstruction_weight,
            "relative_mse_weight": args.relative_mse_weight,
            "pair_delta_weight": args.pair_delta_weight,
            "route_weight": args.route_weight,
            "activation_rms_threshold": args.activation_rms_threshold,
            "routing_epochs": args.routing_epochs,
            "routing_learning_rate": args.routing_learning_rate,
            "magnitude_route_weight": args.magnitude_route_weight,
            "routing_value_preservation_weight": (args.routing_value_preservation_weight),
            "changed_activation_multiplier": args.changed_activation_multiplier,
            "retention_activation_fraction": args.retention_activation_fraction,
        },
        "behavior_thresholds": asdict(V65_BEHAVIOR_THRESHOLDS),
        "runtime_checkpoint_may_be_written": False,
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_loaded": False,
    }
    return {**identity, "run_signature_sha256": _canonical_sha256(identity)}


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create one JSON file; never expose a partial resumable result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            raw = (
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic create-if-absent operation on the same
        # filesystem.  Unlike rename(), it cannot replace a concurrently
        # completed create-once artifact.
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_work_directory(path: Path, manifest: Mapping[str, Any]) -> None:
    expected = dict(manifest)
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise ValueError("V65 resumable work path is not a regular directory")
        if {item.name for item in path.iterdir()} != {"manifest.json", "folds"}:
            raise ValueError("V65 resumable work inventory changed")
        observed = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("V65 resumable work manifest differs from this exact run")
        folds = path / "folds"
        if not folds.is_dir() or folds.is_symlink():
            raise ValueError("V65 resumable fold directory is invalid")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.partial-", dir=path.parent))
    try:
        (temporary / "folds").mkdir()
        _write_new_json(temporary / "manifest.json", expected)
        os.rename(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_runtime(
    preflight: V63Preflight,
    *,
    requested_device: str,
) -> V65RuntimeBundle:
    runtime = StaticRuntimePrefixFactory(
        preflight.config,
        preflight.base_checkpoint,
        preflight.scene_ids[0],
    ).bootstrap
    if not torch.equal(
        runtime.scene_prefix.detach().cpu(),
        preflight.prefixes[preflight.scene_ids[0]].detach().cpu(),
    ):
        raise ValueError("V65 cached prefix differs from the frozen base runtime")
    frozen = freeze_base_runtime(runtime)
    device = _select_training_device(runtime, requested_device)
    model_dtype = next(runtime.language.model.parameters()).dtype
    question_by_text: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for question in sorted({row.question for row in preflight.rows}):
            embedding = _pooled_question_embedding(runtime, question).detach().cpu().float()
            if tuple(embedding.shape) != (1, 1, _EXPECTED_HIDDEN_SIZE):
                raise ValueError("V65 frozen pooled question embedding shape changed")
            question_by_text[question] = embedding
    questions = {row.key: question_by_text[row.question] for row in preflight.rows}
    _disable_decoder_checkpointing(runtime.language)
    if any(parameter.requires_grad for parameter in runtime.language.model.parameters()):
        raise RuntimeError("V65 Gemma parameters are not completely frozen")
    return V65RuntimeBundle(
        runtime=runtime,
        question_embeddings=questions,
        device=device,
        model_dtype=model_dtype,
        audit={
            "device": str(device),
            "model_dtype": str(model_dtype),
            "unique_question_count": len(question_by_text),
            "base_stack_parameter_count": frozen["parameter_count"],
            "base_stack_all_parameters_frozen": frozen["all_parameters_frozen"],
            "gemma_backward_used": False,
            "gemma_generation_use": "training_only_behavioral_gate",
        },
    )


def _behavior_rows(
    control: MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
    rows: Sequence[V63Row],
    *,
    signatures: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    class_by_key: Mapping[tuple[str, str], str],
    supported_keys: set[tuple[str, str]],
    runtime: Any,
    prefixes: Mapping[str, torch.Tensor],
    device: torch.device,
    model_dtype: torch.dtype,
    generator_fn: Callable[..., str],
) -> tuple[dict[str, Any], ...]:
    selected = sorted(
        [row for row in rows if row.route_label],
        key=lambda row: (row.pair_id, row.scene_id, row.question_id),
    )
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in selected:
            output = control.forward_from_signature(signatures[row.scene_id], questions[row.key])
            runtime_gate_audit = control.audit()
            prediction = generator_fn(
                runtime=runtime,
                scene_prefix=prefixes[row.scene_id].to(device=device, dtype=model_dtype),
                question=row.question,
                # Score the production decision, including its literal
                # no-control path.  A suppressed changed side is a measured
                # behavioral miss; it must not abort before the fold can emit
                # an auditable training-only failure report.
                control_tokens=(output.control_tokens if runtime_gate_audit.control_used else None),
            )
            prediction_digest = _canonical_scoring_sha256(prediction, answer_type=row.answer_type)
            reference_digest = _row_reference_sha256(row)
            results.append(
                {
                    "scene_id": row.scene_id,
                    "question_id": row.question_id,
                    "pair_id": row.pair_id,
                    "question_key": row.question_key,
                    "answer_class_id": class_by_key[row.key],
                    "fold_class_supported": row.key in supported_keys,
                    "raw_prediction_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
                    "canonical_prediction_sha256": prediction_digest,
                    "reference_canonical_sha256": reference_digest,
                    "scoring_contract_sha256": _row_scoring_contract_sha256(row),
                    "canonical_exact": prediction_digest == reference_digest,
                }
            )
    if len(results) != len(selected):
        raise RuntimeError("V65 behavioral result inventory changed")
    by_key = {row.key: row for row in selected}
    for record in results:
        key = str(record["scene_id"]), str(record["question_id"])
        _validate_behavior_record(
            record,
            expected_row=by_key[key],
            expected_supported=key in supported_keys,
        )
    return tuple(results)


def route_activation_summary(
    control: MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
    rows: Sequence[V63Row],
    *,
    signatures: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
) -> dict[str, Any]:
    """Measure the hard route without serializing questions or answers."""

    by_route: dict[bool, list[float]] = {True: [], False: []}
    with torch.inference_mode():
        for row in rows:
            output = control.forward_from_signature(signatures[row.scene_id], questions[row.key])
            activation = float(control.activation_rms(output.control_rms)[0].cpu())
            if not math.isfinite(activation):
                raise RuntimeError("V65 route activation is nonfinite")
            by_route[row.route_label].append(activation)

    def distribution(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            raise ValueError("V65 activation summary requires both route classes")
        tensor = torch.tensor(values, dtype=torch.float64)
        threshold = control.activation_rms_threshold
        active = int((tensor >= threshold).sum())
        return {
            "count": len(values),
            "active_count": active,
            "inactive_count": len(values) - active,
            "minimum": float(tensor.min()),
            "q05": float(torch.quantile(tensor, 0.05)),
            "q25": float(torch.quantile(tensor, 0.25)),
            "median": float(torch.quantile(tensor, 0.50)),
            "q75": float(torch.quantile(tensor, 0.75)),
            "q95": float(torch.quantile(tensor, 0.95)),
            "maximum": float(tensor.max()),
        }

    changed = distribution(by_route[True])
    retention = distribution(by_route[False])
    return {
        "activation_rms_threshold": control.activation_rms_threshold,
        "changed": changed,
        "retention": retention,
        "all_changed_active": changed["inactive_count"] == 0,
        "all_retention_inactive": retention["active_count"] == 0,
        "question_or_answer_text_serialized": False,
    }


def _retention_gate_records(
    control: MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
    rows: Sequence[V63Row],
    *,
    signatures: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    baseline_hashes: Mapping[tuple[str, str], str],
) -> tuple[dict[str, Any], ...]:
    """Prove runtime takes the exact no-token V54 path for retention rows."""

    selected = sorted(
        [row for row in rows if not row.route_label],
        key=lambda row: (row.pair_id, row.scene_id, row.question_id),
    )
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in selected:
            _output = control.forward_from_signature(signatures[row.scene_id], questions[row.key])
            audit = control.audit()
            no_control = not audit.control_used and audit.exact_no_control_below_threshold
            expected_hash = baseline_hashes.get(row.key)
            if not _is_sha256(expected_hash):
                raise ValueError("V65 retention row lacks its authenticated V54 hash")
            records.append(
                {
                    "scene_id": row.scene_id,
                    "question_id": row.question_id,
                    "pair_id": row.pair_id,
                    "baseline_raw_output_sha256": expected_hash,
                    "exact_no_control_route": no_control,
                    "runtime_output_identity_by_construction": no_control,
                    "activation_rms_below_threshold": no_control,
                }
            )
    if len(records) != len(selected):
        raise RuntimeError("V65 retention gate inventory changed")
    return tuple(records)


def behavioral_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("V65 behavioral metrics require at least one side")
    opaque_keys: set[tuple[str, str]] = set()
    units: defaultdict[tuple[str, str], list[bool]] = defaultdict(list)
    pair_hits: defaultdict[str, int] = defaultdict(int)
    correct = 0
    supported_correct = 0
    supported_total = 0
    for raw_record in records:
        record = _validate_behavior_record(raw_record)
        key = str(record["scene_id"]), str(record["question_id"])
        if key in opaque_keys:
            raise ValueError("V65 behavioral records contain a duplicate opaque key")
        opaque_keys.add(key)
        exact = bool(record["canonical_exact"])
        correct += int(exact)
        supported = bool(record["fold_class_supported"])
        supported_correct += int(exact and supported)
        supported_total += int(supported)
        pair_id = str(record["pair_id"])
        units[(pair_id, str(record["question_key"]))].append((exact, supported))
        pair_hits[pair_id] += int(exact and supported)
    if any(len(sides) != 2 for sides in units.values()):
        raise ValueError("V65 changed behavioral units require exactly two sides")
    complete = sum(all(exact for exact, _supported in sides) for sides in units.values())
    fully_supported = {
        key: sides for key, sides in units.items() if all(supported for _exact, supported in sides)
    }
    supported_complete = sum(
        all(exact for exact, _supported in sides) for sides in fully_supported.values()
    )
    eligible_pairs = {pair_id for pair_id, _question_key in fully_supported}
    return {
        "side_exact": correct,
        "side_total": len(records),
        "side_accuracy": correct / len(records),
        "supported_side_exact": supported_correct,
        "supported_side_total": supported_total,
        "supported_side_accuracy": (
            supported_correct / supported_total if supported_total else None
        ),
        "unsupported_side_total": len(records) - supported_total,
        "complete_units": complete,
        "unit_total": len(units),
        "complete_unit_accuracy": complete / len(units),
        "fully_supported_complete_units": supported_complete,
        "fully_supported_unit_total": len(fully_supported),
        "fully_supported_complete_unit_accuracy": (
            supported_complete / len(fully_supported) if fully_supported else None
        ),
        "pair_count": len(pair_hits),
        "eligible_fold_count": len(eligible_pairs),
        "eligible_folds_with_exact_hit": sum(pair_hits[pair_id] > 0 for pair_id in eligible_pairs),
        "answer_or_question_text_stored": False,
    }


def assess_cv_behavior(metrics: Mapping[str, Any]) -> dict[str, bool]:
    threshold = V65_BEHAVIOR_THRESHOLDS
    required = {
        "supported_side_exact",
        "supported_side_total",
        "unsupported_side_total",
        "fully_supported_complete_units",
        "fully_supported_unit_total",
        "eligible_folds_with_exact_hit",
        "eligible_fold_count",
        "side_total",
        "unit_total",
        "pair_count",
    }
    if any(type(metrics.get(field)) is not int for field in required):
        raise ValueError("V65 CV behavior metrics are incomplete or non-integral")
    return {
        "held_supported_side_exact": metrics["supported_side_exact"]
        >= threshold.held_supported_side_exact_minimum,
        "held_supported_side_total": metrics["supported_side_total"]
        == threshold.held_supported_side_total,
        "held_unsupported_side_total": metrics["unsupported_side_total"]
        == threshold.held_unsupported_side_total,
        "held_fully_supported_complete_units": metrics["fully_supported_complete_units"]
        >= threshold.held_fully_supported_complete_unit_minimum,
        "held_fully_supported_unit_total": metrics["fully_supported_unit_total"]
        == threshold.held_fully_supported_unit_total,
        "eligible_folds_with_exact_hit": metrics["eligible_folds_with_exact_hit"]
        >= threshold.eligible_folds_with_exact_hit_minimum,
        "eligible_fold_total": metrics["eligible_fold_count"] == threshold.eligible_fold_total,
        "held_inventory_side_total": metrics["side_total"] == threshold.held_inventory_side_total,
        "held_inventory_unit_total": metrics["unit_total"] == threshold.held_inventory_unit_total,
        "held_inventory_fold_total": metrics["pair_count"] == threshold.inventory_fold_total,
    }


def assess_final_behavior(metrics: Mapping[str, Any]) -> dict[str, bool]:
    threshold = V65_BEHAVIOR_THRESHOLDS
    required = {"side_exact", "side_total", "complete_units", "unit_total"}
    if any(type(metrics.get(field)) is not int for field in required):
        raise ValueError("V65 final behavior metrics are incomplete or non-integral")
    return {
        "train_side_exact": metrics["side_exact"] >= threshold.final_train_side_exact_minimum,
        "train_side_total": metrics["side_total"] == threshold.final_train_side_total,
        "train_complete_units": metrics["complete_units"]
        >= threshold.final_train_complete_unit_minimum,
        "train_complete_unit_total": metrics["unit_total"]
        == threshold.final_train_complete_unit_total,
    }


def assess_retention_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    final: bool,
    expected_count: int | None = None,
) -> dict[str, bool]:
    expected = (
        expected_count
        if expected_count is not None
        else (
            V65_BEHAVIOR_THRESHOLDS.final_retention_exact_no_control_total
            if final
            else V65_BEHAVIOR_THRESHOLDS.held_retention_exact_no_control_total
        )
    )
    keys: set[tuple[str, str]] = set()
    for record in records:
        required = {
            "scene_id",
            "question_id",
            "pair_id",
            "baseline_raw_output_sha256",
            "exact_no_control_route",
            "runtime_output_identity_by_construction",
            "activation_rms_below_threshold",
        }
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("V65 retention record fields changed")
        key = str(record["scene_id"]), str(record["question_id"])
        if key in keys or not _is_sha256(record["baseline_raw_output_sha256"]):
            raise ValueError("V65 retention record identity/hash changed")
        keys.add(key)
        route_fields = (
            "exact_no_control_route",
            "runtime_output_identity_by_construction",
            "activation_rms_below_threshold",
        )
        if any(type(record[field]) is not bool for field in route_fields):
            raise ValueError("V65 retention route evidence must be boolean")
        if len({record[field] for field in route_fields}) != 1:
            raise ValueError("V65 retention route evidence is internally inconsistent")
    exact_count = sum(bool(record["exact_no_control_route"]) for record in records)
    return {
        "retention_inventory_exact": len(records) == expected,
        "every_retention_row_exact_no_control": exact_count == expected,
        "base_output_identity_by_construction": exact_count == expected,
    }


def _fold_path(work_directory: Path, pair_id: str) -> Path:
    if pair_id not in TRAIN_PAIR_IDS:
        raise ValueError(f"V65 fold pair is not authorized: {pair_id}")
    return work_directory / "folds" / f"{pair_id}.json"


def _validate_fold_payload(
    payload: object,
    *,
    pair_id: str,
    run_signature_sha256: str,
    expected_rows: Sequence[V63Row] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("V65 fold payload must be a JSON object")
    required = {
        "schema_version",
        "artifact",
        "run_signature_sha256",
        "held_pair_id",
        "training_pair_count",
        "held_scene_question_examples_used_for_optimization",
        "fold_local_codebook",
        "held_teacher_used_in_codebook_or_basis",
        "generation_semantics",
        "fit",
        "prompt_reconstruction",
        "behavior",
        "changed_records",
        "retention",
        "held_retention_count",
        "retention_records",
    }
    records = payload.get("changed_records")
    retention_records = payload.get("retention_records")
    if (
        set(payload) != required
        or payload.get("schema_version") != 2
        or payload.get("artifact") != _FOLD_ARTIFACT
        or payload.get("run_signature_sha256") != run_signature_sha256
        or payload.get("held_pair_id") != pair_id
        or payload.get("training_pair_count") != 11
        or payload.get("held_scene_question_examples_used_for_optimization") is not False
        or not isinstance(payload.get("fold_local_codebook"), Mapping)
        or payload.get("held_teacher_used_in_codebook_or_basis") is not False
        or payload.get("generation_semantics") != _GENERATION_SEMANTICS
        or not isinstance(records, list)
        or not isinstance(retention_records, list)
    ):
        raise ValueError("V65 completed fold contract changed")
    held_rows = (
        [row for row in expected_rows if row.pair_id == pair_id]
        if expected_rows is not None
        else None
    )
    expected_changed = (
        {row.key: row for row in held_rows if row.route_label} if held_rows is not None else None
    )
    supported_classes = set(payload["fold_local_codebook"].get("class_ids", []))
    seen_changed: set[tuple[str, str]] = set()
    for record in records:
        key = str(record.get("scene_id")), str(record.get("question_id"))
        row = expected_changed.get(key) if expected_changed is not None else None
        validated_record = _validate_behavior_record(
            record,
            expected_row=row,
            expected_supported=(
                _answer_class_id(row.answer) in supported_classes if row is not None else None
            ),
        )
        if validated_record["pair_id"] != pair_id:
            raise ValueError("V65 completed fold contains another pair")
        seen_changed.add(key)
    if expected_changed is not None and seen_changed != set(expected_changed):
        raise ValueError("V65 completed fold changed inventory differs from held pair")
    metrics = behavioral_metrics(records)
    if payload.get("behavior") != metrics:
        raise ValueError("V65 completed fold behavioral metrics changed")
    if not isinstance(payload.get("held_retention_count"), int):
        raise TypeError("V65 completed fold retention count changed")
    if payload.get("retention") != assess_retention_gate(
        retention_records,
        final=False,
        expected_count=int(payload["held_retention_count"]),
    ):
        raise ValueError("V65 completed fold retention metrics changed")
    if held_rows is not None:
        expected_retention = {row.key for row in held_rows if not row.route_label}
        observed_retention = {
            (str(record.get("scene_id")), str(record.get("question_id")))
            for record in retention_records
        }
        if observed_retention != expected_retention:
            raise ValueError("V65 completed fold retention inventory differs from held pair")
        if payload["held_retention_count"] != len(expected_retention):
            raise ValueError("V65 completed fold retention count differs from held pair")
    return payload


def _load_completed_folds(
    work_directory: Path,
    *,
    run_signature_sha256: str,
    expected_rows: Sequence[V63Row] | None = None,
) -> dict[str, dict[str, Any]]:
    folds_root = work_directory / "folds"
    expected_names = {f"{pair_id}.json" for pair_id in TRAIN_PAIR_IDS}
    unexpected = {item.name for item in folds_root.iterdir()} - expected_names
    if unexpected:
        raise ValueError(f"V65 work directory has unexpected folds: {sorted(unexpected)}")
    completed: dict[str, dict[str, Any]] = {}
    for pair_id in TRAIN_PAIR_IDS:
        source = _fold_path(work_directory, pair_id)
        if not source.exists():
            continue
        if not source.is_file() or source.is_symlink():
            raise ValueError("V65 completed fold is not a regular file")
        payload = json.loads(source.read_text(encoding="utf-8"))
        completed[pair_id] = _validate_fold_payload(
            payload,
            pair_id=pair_id,
            run_signature_sha256=run_signature_sha256,
            expected_rows=expected_rows,
        )
    return completed


def _save_fold(
    work_directory: Path,
    payload: Mapping[str, Any],
    *,
    expected_rows: Sequence[V63Row] | None = None,
) -> None:
    pair_id = str(payload["held_pair_id"])
    destination = _fold_path(work_directory, pair_id)
    validated = _validate_fold_payload(
        dict(payload),
        pair_id=pair_id,
        run_signature_sha256=str(payload["run_signature_sha256"]),
        expected_rows=expected_rows,
    )
    _write_new_json(destination, validated)


def _prompt_summary(
    fit: V65FitResult,
    rows: Sequence[V63Row],
    *,
    signatures: Mapping[str, torch.Tensor] | None = None,
    questions: Mapping[tuple[str, str], torch.Tensor],
    targets: Mapping[tuple[str, str], torch.Tensor],
) -> dict[str, Any]:
    units: defaultdict[tuple[str, str], list[V63Row]] = defaultdict(list)
    for row in rows:
        if row.route_label:
            units[(row.pair_id, row.question_key)].append(row)
    complete_keys = {
        row.key
        for sides in units.values()
        if len(sides) == 2 and all(row.key in targets for row in sides)
        for row in sides
    }
    complete_rows = [row for row in rows if row.key in complete_keys]
    complete_targets = {key: value for key, value in targets.items() if key in complete_keys}
    if not complete_rows:
        return {
            "teacher_side_count": 0,
            "prompt_token_count": 0,
            "changed_pair_unit_count": 0,
            "vocabulary_supported_complete_units_only": True,
        }
    measurements = _measure_reconstruction(
        fit.control,
        complete_rows,
        signatures=fit.signatures if signatures is None else signatures,
        questions=questions,
        targets=complete_targets,
    )
    return {
        **measurements.summary(),
        "vocabulary_supported_complete_units_only": True,
    }


def _magnitude_route_loss(
    control: MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
    rows: Sequence[V63Row],
    *,
    signatures: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    changed_high_target: float,
    retention_low_target: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for offset in range(0, len(rows), 64):
        batch = rows[offset : offset + 64]
        output = control.forward_from_signature(
            torch.cat([signatures[row.scene_id] for row in batch]),
            torch.cat([questions[row.key] for row in batch]),
        )
        activation = control.activation_rms(output.control_rms)
        labels = torch.tensor([row.route_label for row in batch], dtype=torch.bool)
        changed = F.relu(changed_high_target - activation[labels]).square()
        retention = F.relu(activation[~labels] - retention_low_target).square()
        if changed.numel():
            losses.append(changed.mean())
        if retention.numel():
            losses.append(retention.mean())
    if not losses:
        raise ValueError("V65 magnitude route loss has no examples")
    return torch.stack(losses).mean()


def _fit_v65_controller(
    *,
    rows: Sequence[V63Row],
    targets: Mapping[tuple[str, str], torch.Tensor],
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[tuple[str, str], torch.Tensor],
    source_question_norm_state: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    seed: int,
    log_phase: str,
    fixed_output_basis: torch.Tensor,
) -> V65FitResult:
    """Fit V3 values, then jointly separate changed/retention magnitudes."""

    base = _fit_controller(
        rows=rows,
        targets=targets,
        prefixes=prefixes,
        questions=questions,
        source_question_norm_state=source_question_norm_state,
        args=args,
        seed=seed,
        log_phase=f"{log_phase}_value",
        fixed_output_basis=fixed_output_basis,
    )
    control = MagnitudeGatedTeacherBasisFullSceneQuestionControlV6.from_v65(
        base.control,
        activation_rms_threshold=args.activation_rms_threshold,
    )
    # Legacy route parameters are ignored by V6 and cannot affect its gate.
    for name, parameter in control.named_parameters():
        parameter.requires_grad_(
            not name.startswith("question_norm.") and not name.startswith("route_")
        )
    signatures = _scene_signatures(control, prefixes)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in control.parameters() if parameter.requires_grad],
        lr=args.routing_learning_rate,
        weight_decay=args.weight_decay,
    )
    gradients: list[float] = []
    changed_high = args.activation_rms_threshold * args.changed_activation_multiplier
    retention_low = args.activation_rms_threshold * args.retention_activation_fraction
    for epoch in range(args.routing_epochs):
        route_loss = args.magnitude_route_weight * _magnitude_route_loss(
            control,
            rows,
            signatures=signatures,
            questions=questions,
            changed_high_target=changed_high,
            retention_low_target=retention_low,
        )
        # Preserve numeric teacher values while learning the shared magnitude
        # surface.  This deliberately reuses V63's reconstruction loss through
        # a small supervised changed batch each epoch.
        changed_rows = [row for row in rows if row.key in targets]
        sample = changed_rows[
            (epoch * args.batch_size) % len(changed_rows) : (epoch * args.batch_size)
            % len(changed_rows)
            + args.batch_size
        ]
        if not sample:
            sample = changed_rows[: args.batch_size]
        predicted = control.forward_from_signature(
            torch.cat([signatures[row.scene_id] for row in sample]),
            torch.cat([questions[row.key] for row in sample]),
        ).control_tokens
        expected = torch.cat([targets[row.key] for row in sample])
        value_loss = args.routing_value_preservation_weight * (predicted - expected).square().mean()
        gradients.append(
            _optimizer_step(
                loss=route_loss + value_loss,
                control=control,
                optimizer=optimizer,
                gradient_clip_norm=args.gradient_clip_norm,
            )
        )
        if (epoch + 1) % args.log_every == 0:
            _log_event(
                phase=f"{log_phase}_magnitude_route",
                epoch=epoch + 1,
                optimizer_steps=epoch + 1,
            )
    control.eval()
    return V65FitResult(
        control=control,
        signatures=signatures,
        base_fit=base,
        routing_optimizer_steps=args.routing_epochs,
        maximum_routing_gradient_norm=max(gradients, default=0.0),
    )


def _aggregate_prompt_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    eligible = [item for item in summaries if int(item["teacher_side_count"]) > 0]
    side_total = sum(int(item["teacher_side_count"]) for item in summaries)
    unit_total = sum(int(item["changed_pair_unit_count"]) for item in summaries)
    if side_total != 56 or unit_total != 28:
        raise ValueError("V65 held prompt reconstruction coverage changed")
    return {
        "mean_prompt_cosine": sum(
            float(item["mean_prompt_cosine"]) * int(item["prompt_token_count"]) for item in eligible
        )
        / sum(int(item["prompt_token_count"]) for item in eligible),
        "minimum_prompt_cosine": min(float(item["minimum_prompt_cosine"]) for item in eligible),
        "mean_prompt_rms_absolute_error": sum(
            float(item["mean_prompt_rms_absolute_error"]) * int(item["prompt_token_count"])
            for item in eligible
        )
        / sum(int(item["prompt_token_count"]) for item in eligible),
        "mean_pair_delta_cosine": sum(
            float(item["mean_pair_delta_cosine"]) * int(item["changed_pair_unit_count"])
            for item in eligible
        )
        / unit_total,
    }


def _report_base(
    *,
    preflight: V63Preflight,
    codebook: AnswerPrototypeCodebookV65,
    basis: torch.Tensor,
    runtime_audit: Mapping[str, Any],
    work_manifest: Mapping[str, Any],
    cv: Mapping[str, Any],
    training_baseline: V65TrainingBaseline,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "artifact": "v65_magnitude_gated_canonical_answer_distillation",
        "offline_checks_passed": False,
        "promotion_eligible": False,
        "successor_factorized_route_required": False,
        "checkpoint": None,
        "authorization": {
            "baseline_lock_sha256": preflight.baseline_lock_sha256,
            "training_baseline_lock_sha256": training_baseline.lock_sha256,
            "filtered_training_qa_sha256": preflight.filtered_train_sha256,
            "baseline_validated_before_training_data": True,
            "training_v54_hash_inventory_count": len(training_baseline.required_output_hashes),
        },
        "base": {
            "checkpoint_sha256": preflight.base_checkpoint_sha256,
            "runtime_config_effective_sha256": preflight.runtime_config_sha256,
        },
        "source_v60": {
            "checkpoint_sha256": preflight.source_v60_checkpoint_sha256,
            "weights_sha256": preflight.source_v60_metadata["weights_sha256"],
            "question_norm_sha256": preflight.source_v60_question_norm_sha256,
            "question_norm_copied_exact_and_frozen": True,
        },
        "inputs": {
            "training_record_count": len(preflight.rows),
            "training_scene_count": len(preflight.scene_ids),
            "training_pair_count": len(TRAIN_PAIR_IDS),
            "changed_teacher_side_count": len(preflight.teacher_targets),
            "changed_paired_unit_count": len(_changed_units(preflight.rows)),
            "teacher_metadata_sha256": preflight.teacher_metadata_sha256,
            "teacher_weights_sha256": preflight.teacher_weights_sha256,
            "prefix_cache_manifest_sha256": preflight.prefix_manifest_sha256,
        },
        "codebook": {
            "sha256": codebook.sha256,
            "answer_class_count": len(codebook.prototypes),
            "teacher_side_count": len(codebook.targets),
            "selection": codebook.manifest["selection"],
            "final_all_training_only": True,
            "folds_use_separate_training_only_codebooks": True,
            "held_fold_label_codebook_visible": False,
            "held_teacher_used_in_fold_codebook_or_basis": False,
            "answer_strings_serialized_in_report_or_runtime": False,
            "prototype_manifest": codebook.manifest,
        },
        "architecture": {
            "name": "magnitude_gated_teacher_basis_full_scene_control_v6",
            "runtime_schema_version": 6,
            "hidden_size": _EXPECTED_HIDDEN_SIZE,
            "control_tokens": PROMPT_SHAPE[1],
            "global_scene_latents": 256,
            "moment_count": args.moment_count,
            "basis_rank_requested": args.basis_rank,
            "basis_rank_effective": int(basis.shape[0]),
            "final_all_training_basis_sha256": _tensor_sha256(basis),
            "activation_rms_threshold": args.activation_rms_threshold,
            "activation_rms_aggregation": "maximum_over_control_tokens",
            "exact_no_control_below_threshold": True,
            "unified_scene_question_value_and_route": True,
            "question_dependent_scene_retrieval": False,
            "complete_scene_prefix_retained": True,
        },
        "cross_validation": dict(cv),
        "runtime_audit": dict(runtime_audit),
        "work": {
            "run_signature_sha256": work_manifest["run_signature_sha256"],
            "resumable_pair_fold_artifact": True,
            "fold_artifacts_contain_answer_or_question_text": False,
        },
        "optimization": {
            "seed": args.seed,
            "epochs_per_fold_and_final": args.epochs,
            "controller_device": "cpu",
            "gemma_backward_used": False,
            "behavior_generation_device": runtime_audit["device"],
        },
        "scope": {
            "training_answers_used_only_to_build_numeric_codebook_and_score_training": True,
            "runtime_answer_strings": False,
            "training_v54_output_hashes_only": True,
            "gemma_backward_used": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "prediction_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
        },
    }


def _publish_v65_checkpoint_and_report(
    *,
    preflight: V63Preflight,
    control: MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
    report_without_checkpoint: Mapping[str, Any],
    bundle: V65RuntimeBundle,
    rows: Sequence[V63Row],
    prefixes: Mapping[str, torch.Tensor],
    class_by_key: Mapping[tuple[str, str], str],
    training_baseline: V65TrainingBaseline,
    generator_fn: Callable[..., str],
) -> dict[str, Any]:
    """Atomically publish a strict-reloaded, runtime-minimal V6 checkpoint."""

    checkpoint = preflight.output_checkpoint
    report = preflight.training_report
    if checkpoint.exists() or report.exists():
        raise FileExistsError("V65 create-once publication destination exists")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v65-publish-", dir=checkpoint.parent))
    published_checkpoint = False
    published_report = False
    try:
        staged_checkpoint = staging / "checkpoint"
        fit_state_sha256 = v6_value_state_sha256(control)
        checkpoint_hashes = save_v6_control_checkpoint(
            staged_checkpoint,
            control=control,
            base_checkpoint_sha256=preflight.base_checkpoint_sha256,
            base_runtime_config_sha256=preflight.runtime_config_sha256,
            expected_training_fit_state_sha256=fit_state_sha256,
        )
        reloaded_cpu = load_unsealed_v6_checkpoint_for_training_gate(
            staged_checkpoint, hidden_size=_EXPECTED_HIDDEN_SIZE
        )
        if v6_value_state_sha256(reloaded_cpu) != fit_state_sha256:
            raise RuntimeError("V65 staged V6 value state changed after strict reload")
        reloaded = reloaded_cpu.to(device=bundle.device, dtype=torch.float32).eval()
        signatures = {
            scene_id: reloaded.encode_scene(prefix.to(device=bundle.device, dtype=torch.float32))
            for scene_id, prefix in prefixes.items()
        }
        changed_records: list[dict[str, Any]] = []
        retention_records: list[dict[str, Any]] = []
        with torch.inference_mode():
            for row in sorted(
                rows, key=lambda item: (item.pair_id, item.scene_id, item.question_id)
            ):
                ids = question_token_ids(
                    bundle.runtime.language.tokenizer,
                    row.question,
                    bundle.runtime.language.device,
                )
                raw_embeddings = bundle.runtime.language.model.get_input_embeddings()(ids)
                output = reloaded.forward_from_signature(signatures[row.scene_id], raw_embeddings)
                audit = reloaded.audit()
                selected_control = output.control_tokens if audit.control_used else None
                if row.route_label:
                    if selected_control is None:
                        raise RuntimeError(
                            "V65 saved-runtime gate suppressed a changed-side control"
                        )
                    prediction = generator_fn(
                        runtime=bundle.runtime,
                        scene_prefix=prefixes[row.scene_id].to(
                            device=bundle.device, dtype=bundle.model_dtype
                        ),
                        question=row.question,
                        control_tokens=selected_control,
                    )
                    prediction_digest = _canonical_scoring_sha256(
                        prediction, answer_type=row.answer_type
                    )
                    reference_digest = _row_reference_sha256(row)
                    changed_records.append(
                        {
                            "scene_id": row.scene_id,
                            "question_id": row.question_id,
                            "pair_id": row.pair_id,
                            "question_key": row.question_key,
                            "answer_class_id": class_by_key[row.key],
                            "fold_class_supported": True,
                            "raw_prediction_sha256": hashlib.sha256(
                                prediction.encode("utf-8")
                            ).hexdigest(),
                            "canonical_prediction_sha256": prediction_digest,
                            "reference_canonical_sha256": reference_digest,
                            "scoring_contract_sha256": _row_scoring_contract_sha256(row),
                            "canonical_exact": prediction_digest == reference_digest,
                        }
                    )
                else:
                    if selected_control is not None:
                        raise RuntimeError(
                            "V65 saved-runtime retention would insert control tokens"
                        )
                    baseline_hash = training_baseline.required_output_hashes[row.key]
                    retention_records.append(
                        {
                            "scene_id": row.scene_id,
                            "question_id": row.question_id,
                            "pair_id": row.pair_id,
                            "baseline_raw_output_sha256": baseline_hash,
                            "exact_no_control_route": True,
                            "runtime_output_identity_by_construction": True,
                            "activation_rms_below_threshold": True,
                        }
                    )
        saved_behavior = behavioral_metrics(changed_records)
        saved_retention = assess_retention_gate(retention_records, final=True)
        saved_checks = {
            **assess_final_behavior(saved_behavior),
            **saved_retention,
        }
        if not all(saved_checks.values()):
            raise RuntimeError("V65 strict-loaded production-device behavior gate failed")
        gate_attestation = _canonical_sha256(
            {
                "schema_version": 1,
                "artifact": "v65_saved_runtime_training_gate_attestation",
                "training_fit_state_sha256": fit_state_sha256,
                "production_device": str(bundle.device),
                "raw_question_token_embeddings_used": True,
                "changed_behavior": saved_behavior,
                "retention": saved_retention,
                "checks": saved_checks,
                "answer_or_question_text_stored": False,
            }
        )
        # Replace only the staged private/ungated checkpoint with a sealed
        # create-once checkpoint whose public loader requires this attestation.
        shutil.rmtree(staged_checkpoint)
        checkpoint_hashes = save_v6_control_checkpoint(
            staged_checkpoint,
            control=control,
            base_checkpoint_sha256=preflight.base_checkpoint_sha256,
            base_runtime_config_sha256=preflight.runtime_config_sha256,
            expected_training_fit_state_sha256=fit_state_sha256,
            saved_runtime_training_gate_passed=True,
            saved_runtime_training_gate_attestation_sha256=gate_attestation,
        )
        public_reloaded, metadata = _load_control_head(
            staged_checkpoint,
            hidden_size=_EXPECTED_HIDDEN_SIZE,
            device=torch.device("cpu"),
        )
        if (
            type(public_reloaded) is not MagnitudeGatedTeacherBasisFullSceneQuestionControlV6
            or v6_value_state_sha256(public_reloaded) != fit_state_sha256
            or metadata["saved_runtime_training_gate_attestation_sha256"] != gate_attestation
        ):
            raise RuntimeError("V65 sealed public checkpoint failed strict reload")
        final_report = {
            **dict(report_without_checkpoint),
            "checkpoint": checkpoint_hashes,
            "saved_runtime_reload": {
                "strict_loader_passed": True,
                "architecture": metadata["architecture"],
                "training_fit_state_sha256": fit_state_sha256,
                "gate_attestation_sha256": gate_attestation,
                "reloaded_state_exact": True,
                "raw_question_token_embeddings_used": True,
                "production_device": str(bundle.device),
                "changed_behavior": saved_behavior,
                "retention": saved_retention,
                "checks": saved_checks,
                "passed_before_publication": True,
            },
            "promotion_eligible": False,
        }
        staged_report = staging / "training_report.json"
        _write_new_json(staged_report, final_report)
        os.rename(staged_checkpoint, checkpoint)
        published_checkpoint = True
        os.rename(staged_report, report)
        published_report = True
        return final_report
    except BaseException:
        if published_report:
            report.unlink(missing_ok=True)
        if published_checkpoint:
            shutil.rmtree(checkpoint, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def train_v65(
    args: argparse.Namespace,
    *,
    runtime_provider: Callable[..., V65RuntimeBundle] | None = None,
    generator_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Run resumable pair-held-out behavior gates, then optionally publish."""

    if (
        args.routing_epochs < 1
        or not 0.0 < args.activation_rms_threshold < args.maximum_control_rms
        or not 0.0 < args.retention_activation_fraction < 1.0
        or args.changed_activation_multiplier <= 1.0
        or args.routing_learning_rate <= 0.0
        or args.magnitude_route_weight <= 0.0
        or args.routing_value_preservation_weight <= 0.0
    ):
        raise ValueError("V65 magnitude-route hyperparameters are invalid")

    # The existing V62 baseline remains the very first open inside preflight.
    preflight = build_v63_preflight(args)
    training_baseline = validate_training_baseline_lock(
        args.training_baseline_lock,
        expected_rows=preflight.rows,
    )
    codebook = build_answer_prototype_codebook(
        preflight.rows,
        preflight.teacher_targets,
    )
    basis = codebook_output_basis(codebook, requested_rank=args.basis_rank)
    if _basis_coverage(codebook.prototypes, basis)["minimum_cosine"] < 0.999:
        raise RuntimeError("V65 numeric codebook basis does not retain its prototypes")
    work_directory = _work_path(
        args.work_directory,
        config=preflight.config,
        output_checkpoint=preflight.output_checkpoint,
        training_report=preflight.training_report,
    )
    work_manifest = _run_manifest(
        preflight=preflight,
        args=args,
    )
    prepare_work_directory(work_directory, work_manifest)

    provide_runtime = runtime_provider or _load_runtime
    generation = generator_fn or _generate_with_control
    bundle = provide_runtime(preflight, requested_device=args.device)
    completed = _load_completed_folds(
        work_directory,
        run_signature_sha256=work_manifest["run_signature_sha256"],
        expected_rows=preflight.rows,
    )
    for fold_index, held_pair_id in enumerate(TRAIN_PAIR_IDS):
        if held_pair_id in completed:
            continue
        train_rows = tuple(row for row in preflight.rows if row.pair_id != held_pair_id)
        held_rows = tuple(row for row in preflight.rows if row.pair_id == held_pair_id)
        fold_teachers = {
            row.key: preflight.teacher_targets[row.key] for row in train_rows if row.route_label
        }
        fold_codebook = build_answer_prototype_codebook(
            train_rows,
            fold_teachers,
            expected_class_count=None,
            scope=f"fold_{held_pair_id}",
            forbidden_pair_id=held_pair_id,
        )
        fold_basis = codebook_output_basis(fold_codebook, requested_rank=args.basis_rank)
        train_targets = fold_codebook.targets
        held_targets = {
            row.key: fold_codebook.prototypes[_answer_class_id(row.answer)]
            for row in held_rows
            if row.route_label and _answer_class_id(row.answer) in fold_codebook.prototypes
        }
        training_scene_ids = sorted({row.scene_id for row in train_rows})
        training_questions = {row.key: bundle.question_embeddings[row.key] for row in train_rows}
        fit = _fit_v65_controller(
            rows=train_rows,
            targets=train_targets,
            prefixes={scene_id: preflight.prefixes[scene_id] for scene_id in training_scene_ids},
            questions=training_questions,
            source_question_norm_state=preflight.source_v60_question_norm_state,
            args=args,
            seed=args.seed + (fold_index + 1) * 100_003,
            log_phase=f"v65_cv_{held_pair_id}",
            fixed_output_basis=fold_basis,
        )
        held_signatures = _scene_signatures(
            fit.control,
            {
                scene_id: preflight.prefixes[scene_id]
                for scene_id in sorted({row.scene_id for row in held_rows})
            },
        )
        records = _behavior_rows(
            fit.control,
            held_rows,
            signatures=held_signatures,
            questions=bundle.question_embeddings,
            class_by_key=codebook.class_by_key,
            supported_keys=set(held_targets),
            runtime=bundle.runtime,
            prefixes=preflight.prefixes,
            device=bundle.device,
            model_dtype=bundle.model_dtype,
            generator_fn=generation,
        )
        held_route_activation = route_activation_summary(
            fit.control,
            held_rows,
            signatures=held_signatures,
            questions=bundle.question_embeddings,
        )
        retention_records = _retention_gate_records(
            fit.control,
            held_rows,
            signatures=held_signatures,
            questions=bundle.question_embeddings,
            baseline_hashes=training_baseline.required_output_hashes,
        )
        payload = {
            "schema_version": 2,
            "artifact": _FOLD_ARTIFACT,
            "run_signature_sha256": work_manifest["run_signature_sha256"],
            "held_pair_id": held_pair_id,
            "training_pair_count": len(TRAIN_PAIR_IDS) - 1,
            "held_scene_question_examples_used_for_optimization": False,
            "fold_local_codebook": {
                "sha256": fold_codebook.sha256,
                "class_ids": sorted(fold_codebook.prototypes),
                "class_count": len(fold_codebook.prototypes),
                "basis_sha256": _tensor_sha256(fold_basis),
                "basis_rank": int(fold_basis.shape[0]),
            },
            "held_teacher_used_in_codebook_or_basis": False,
            "generation_semantics": _GENERATION_SEMANTICS,
            "fit": {
                "optimizer_steps": fit.base_fit.optimizer_steps,
                "routing_optimizer_steps": fit.routing_optimizer_steps,
                "elapsed_seconds": fit.base_fit.elapsed_seconds,
                "question_norm_sha256": fit.base_fit.question_norm_sha256,
                "question_norm_frozen": fit.base_fit.question_norm_frozen,
                "held_route_activation": held_route_activation,
            },
            "prompt_reconstruction": _prompt_summary(
                fit,
                held_rows,
                signatures=held_signatures,
                questions=bundle.question_embeddings,
                targets=held_targets,
            ),
            "behavior": behavioral_metrics(records),
            "changed_records": list(records),
            "retention": assess_retention_gate(
                retention_records,
                final=False,
                expected_count=len(retention_records),
            ),
            "held_retention_count": len(retention_records),
            "retention_records": list(retention_records),
        }
        _save_fold(work_directory, payload, expected_rows=preflight.rows)
        completed[held_pair_id] = payload

    if set(completed) != set(TRAIN_PAIR_IDS):
        raise RuntimeError("V65 did not complete all 12 behavioral folds")
    ordered_folds = [completed[pair_id] for pair_id in TRAIN_PAIR_IDS]
    held_records = [record for fold in ordered_folds for record in fold["changed_records"]]
    held_retention_records = [
        record for fold in ordered_folds for record in fold["retention_records"]
    ]
    held_behavior = behavioral_metrics(held_records)
    held_checks = assess_cv_behavior(held_behavior)
    cv = {
        "protocol": "deterministic_leave_one_counterfactual_pair_out",
        "pair_count": len(ordered_folds),
        "each_changed_training_side_generated_exactly_once": True,
        "fold_specific_training_only_codebook_and_basis": True,
        "held_teacher_used_in_fold_codebook_or_basis": False,
        "unsupported_closed_vocabulary_sides_excluded_from_primary_cv_gate": True,
        "held_scene_question_examples_used_for_fold_optimization": False,
        "thresholds": asdict(V65_BEHAVIOR_THRESHOLDS),
        "behavior": held_behavior,
        "checks": held_checks,
        "retention": assess_retention_gate(held_retention_records, final=False),
        "passed": all(held_checks.values())
        and all(assess_retention_gate(held_retention_records, final=False).values()),
        "secondary_prompt_reconstruction": _aggregate_prompt_summaries(
            [fold["prompt_reconstruction"] for fold in ordered_folds]
        ),
        "folds": [
            {
                key: fold[key]
                for key in (
                    "held_pair_id",
                    "training_pair_count",
                    "held_scene_question_examples_used_for_optimization",
                    "fold_local_codebook",
                    "held_teacher_used_in_codebook_or_basis",
                    "generation_semantics",
                    "fit",
                    "prompt_reconstruction",
                    "behavior",
                    "retention",
                )
            }
            for fold in ordered_folds
        ],
    }
    report = _report_base(
        preflight=preflight,
        codebook=codebook,
        basis=basis,
        runtime_audit=bundle.audit,
        work_manifest=work_manifest,
        cv=cv,
        training_baseline=training_baseline,
        args=args,
    )
    if not cv["passed"]:
        report["terminal_reason"] = "pair_disjoint_training_behavior_gate_failed"
        _write_training_report(preflight.training_report, report)
        return report

    final_fit = _fit_v65_controller(
        rows=preflight.rows,
        targets=codebook.targets,
        prefixes=preflight.prefixes,
        questions=bundle.question_embeddings,
        source_question_norm_state=preflight.source_v60_question_norm_state,
        args=args,
        seed=args.seed + 9_999_991,
        log_phase="v65_final_all_training_fit",
        fixed_output_basis=basis,
    )
    final_records = _behavior_rows(
        final_fit.control,
        preflight.rows,
        signatures=final_fit.signatures,
        questions=bundle.question_embeddings,
        class_by_key=codebook.class_by_key,
        supported_keys=set(codebook.targets),
        runtime=bundle.runtime,
        prefixes=preflight.prefixes,
        device=bundle.device,
        model_dtype=bundle.model_dtype,
        generator_fn=generation,
    )
    final_behavior = behavioral_metrics(final_records)
    final_retention_records = _retention_gate_records(
        final_fit.control,
        preflight.rows,
        signatures=final_fit.signatures,
        questions=bundle.question_embeddings,
        baseline_hashes=training_baseline.required_output_hashes,
    )
    final_checks = assess_final_behavior(final_behavior)
    final_checks["source_v60_question_norm_exact"] = (
        final_fit.base_fit.question_norm_sha256 == preflight.source_v60_question_norm_sha256
    )
    final_checks["source_v60_question_norm_frozen"] = final_fit.base_fit.question_norm_frozen
    final_checks.update(assess_retention_gate(final_retention_records, final=True))
    report["final_fit"] = {
        "behavior": final_behavior,
        "checks": final_checks,
        "passed": all(final_checks.values()),
        "secondary_prompt_reconstruction": _prompt_summary(
            final_fit,
            preflight.rows,
            questions=bundle.question_embeddings,
            targets=codebook.targets,
        ),
        "retention": assess_retention_gate(final_retention_records, final=True),
        "optimizer_steps": final_fit.base_fit.optimizer_steps,
        "routing_optimizer_steps": final_fit.routing_optimizer_steps,
        "elapsed_seconds": final_fit.base_fit.elapsed_seconds,
    }
    if not report["final_fit"]["passed"]:
        report["terminal_reason"] = "all_training_behavior_gate_failed"
        _write_training_report(preflight.training_report, report)
        return report

    report["offline_checks_passed"] = True
    report["terminal_reason"] = "training_behavior_gates_passed_checkpoint_saved"
    return _publish_v65_checkpoint_and_report(
        preflight=preflight,
        control=final_fit.control,
        report_without_checkpoint={
            key: value for key, value in report.items() if key != "checkpoint"
        },
        bundle=bundle,
        rows=preflight.rows,
        prefixes=preflight.prefixes,
        class_by_key=codebook.class_by_key,
        training_baseline=training_baseline,
        generator_fn=generation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-lock", required=True)
    parser.add_argument("--training-baseline-lock", required=True)
    parser.add_argument("--filtered-train-qa", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--source-v60-checkpoint", required=True)
    parser.add_argument("--work-directory", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=650065)
    parser.add_argument("--basis-rank", type=int, default=128)
    parser.add_argument("--moment-count", type=int, default=8)
    parser.add_argument("--interaction-dim", type=int, default=32)
    parser.add_argument("--trunk-dim", type=int, default=192)
    parser.add_argument("--maximum-control-rms", type=float, default=0.25)
    parser.add_argument("--initial-control-rms", type=float, default=0.075)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--changed-repeats", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--coefficient-weight", type=float, default=3.0)
    parser.add_argument("--log-rms-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=2.0)
    parser.add_argument("--relative-mse-weight", type=float, default=0.25)
    parser.add_argument("--pair-delta-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=0.25)
    parser.add_argument("--activation-rms-threshold", type=float, default=0.01)
    parser.add_argument("--routing-epochs", type=int, default=240)
    parser.add_argument("--routing-learning-rate", type=float, default=3e-4)
    parser.add_argument("--magnitude-route-weight", type=float, default=10.0)
    parser.add_argument("--routing-value-preservation-weight", type=float, default=2.0)
    parser.add_argument("--changed-activation-multiplier", type=float, default=2.0)
    parser.add_argument("--retention-activation-fraction", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=40)
    forbidden = {
        "validation_questions",
        "internal_validation_questions",
        "scorer_references",
        "scorer_sidecar",
        "questions_manifest",
        "predictions",
        "oracle",
        "heldout",
        "preregistration",
    }
    if {action.dest for action in parser._actions} & forbidden:
        raise AssertionError("V65 parser exposes a prohibited evaluation boundary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = train_v65(args)
    print(
        json.dumps(
            {
                "offline_checks_passed": report["offline_checks_passed"],
                "promotion_eligible": False,
                "checkpoint_saved": report.get("checkpoint") is not None,
                "terminal_reason": report["terminal_reason"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["offline_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V65_BEHAVIOR_THRESHOLDS",
    "AnswerPrototypeCodebookV65",
    "V65BehaviorThresholds",
    "V65RuntimeBundle",
    "assess_cv_behavior",
    "assess_final_behavior",
    "behavioral_metrics",
    "build_answer_prototype_codebook",
    "codebook_output_basis",
    "main",
    "prepare_work_directory",
    "train_v65",
]
