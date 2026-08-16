"""Public immutable contract for the one-shot V66 internal validation.

This module contains only public hashes, opaque scene/pair inventories, and
predeclared numeric thresholds.  It never accepts or opens questions, maps,
predictions, scorer references, QA, or oracle data.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation import v62_pair_disjoint_preregistration as boundary

SCHEMA: Final[str] = "semantic_3d_chat.v66.always_on_internal_validation_preregistration.v1"
ARTIFACT: Final[str] = "v66_always_on_internal_validation_preregistration"
PINNED_V66_INTERNAL_VALIDATION_PREREGISTRATION_SHA256: Final[str] = (
    "05d7faf616089e6af0fd01487398f0375d05b8383f37e8303e980b022eb56d74"
)
PINNED_V66_TRAINING_PREREGISTRATION_SHA256: Final[str] = (
    "9c47e43e85b66bcf07794ccc206783db6a40b18af8ad29407475f081e60930bf"
)
PINNED_V62_BASELINE_LOCK_SHA256: Final[str] = (
    "ff9aef64c85e243219216638163ab308d8aaf6492be7209ae43775fecd283d66"
)
PINNED_SCORER_REFERENCES_SHA256: Final[str] = (
    "4202e777ee57ab3f7da329f15589e56b8b0464b782fb4d856dd1a3281ff3115c"
)
PINNED_SCORER_RECORDS_SHA256: Final[str] = (
    "e140ea3a178d4f9bd8ac93c72148df73219ccac2867e5590ddd01f49e4d5b5a9"
)

INTERNAL_VALIDATION_THRESHOLDS: Final[dict[str, dict[str, int]]] = {
    "canonical_exact": {"minimum": 192, "total": 384},
    "changed_side_exact": {"minimum": 32, "total": 52},
    "changed_paired_unit_complete": {"minimum": 10, "total": 26},
    "changed_paired_unit_correct_direction": {"minimum": 15, "total": 26},
}
SAME_QUESTION_THRESHOLDS: Final[dict[str, dict[str, int]]] = {
    "complete_unit_coverage": {"minimum": 26, "total": 26},
    "distinct_scene_prefix_hashes": {"minimum": 26, "total": 26},
    "distinct_scene_signature_hashes": {"minimum": 26, "total": 26},
    "question_text_identity": {"minimum": 26, "total": 26},
}
SCENE_SWAP_THRESHOLDS: Final[dict[str, dict[str, int]]] = {
    "swapped_side_coverage": {"minimum": 52, "total": 52},
    "question_bytes_unchanged": {"minimum": 52, "total": 52},
    "opposite_prefix_hash_exact": {"minimum": 52, "total": 52},
    "opposite_signature_hash_exact": {"minimum": 52, "total": 52},
    "answer_follows_injected_scene": {"minimum": 32, "total": 52},
    "bidirectional_unit_complete": {"minimum": 10, "total": 26},
}
TRAINING_THRESHOLDS: Final[dict[str, Any]] = {
    "held_supported_exact_minimum": 300,
    "held_supported_total": 571,
    "held_unsupported_total": 5,
    "eligible_fold_total": 12,
    "eligible_folds_with_exact_hit_minimum": 12,
    "held_changed_side_exact_minimum": 45,
    "held_changed_side_total": 75,
    "held_complete_unit_minimum": 15,
    "held_complete_unit_total": 35,
    "held_prediction_change_unit_minimum": 20,
    "held_prediction_change_unit_total": 35,
    "final_exact_minimum": 520,
    "final_total": 576,
    "final_complete_unit_minimum": 36,
    "final_complete_unit_total": 40,
    "paired_opposite_follows_side_minimum": 60,
    "paired_opposite_side_total": 80,
    "paired_opposite_follows_complete_minimum": 25,
    "paired_opposite_unit_total": 40,
    "paired_opposite_original_exact_maximum": 20,
    "paired_opposite_original_complete_maximum": 5,
    "per_type_minimum_exact": [
        ["attribute", 35],
        ["count", 58],
        ["metric", 3],
        ["orientation", 20],
        ["presence", 60],
        ["spatial_relation", 65],
        ["support", 35],
    ],
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_regular_file(path: str | Path) -> Path:
    source = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V66 preregistration path contains a symlink: {current}")
    if not source.is_file():
        raise FileNotFoundError(f"V66 preregistration is unavailable: {source}")
    return source


def _expected_scene_ids() -> list[str]:
    specs = {spec.pair_id: spec for spec in boundary.PAIR_INVENTORY}
    return [
        scene_id
        for pair_id in boundary.INTERNAL_VALIDATION_PAIR_IDS
        for scene_id in specs[pair_id].scene_ids
    ]


def validate_v66_internal_validation_preregistration(
    path: str | Path,
) -> dict[str, Any]:
    """Authenticate the exact public V66 contract and return a copy."""

    source = _resolve_regular_file(path)
    if _sha256_file(source) != PINNED_V66_INTERNAL_VALIDATION_PREREGISTRATION_SHA256:
        raise ValueError("V66 internal-validation preregistration differs from its pin")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("V66 internal-validation preregistration is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("V66 internal-validation preregistration must be an object")
    source_boundary = value.get("source_boundary")
    candidate = value.get("candidate_contract")
    controls = value.get("controls")
    population = value.get("population")
    split = value.get("split")
    thresholds = value.get("thresholds")
    if (
        value.get("schema") != SCHEMA
        or value.get("schema_version") != 1
        or value.get("artifact") != ARTIFACT
        or value.get("status") != "locked_before_v66_internal_validation_candidate_generation"
        or not isinstance(source_boundary, Mapping)
        or source_boundary.get("parent_v62_preregistration_sha256")
        != boundary.PINNED_V62_PREREGISTRATION_SHA256
        or source_boundary.get("questions_manifest_sha256")
        != boundary.PINNED_V62_QUESTIONS_MANIFEST_SHA256
        or source_boundary.get("questions_sha256") != boundary.PINNED_V62_QUESTIONS_SHA256
        or source_boundary.get("baseline_lock_sha256") != PINNED_V62_BASELINE_LOCK_SHA256
        or source_boundary.get("scorer_references_sha256") != PINNED_SCORER_REFERENCES_SHA256
        or source_boundary.get("scorer_records_sha256") != PINNED_SCORER_RECORDS_SHA256
        or not isinstance(candidate, Mapping)
        or candidate.get("architecture") != "always_on_teacher_basis_full_scene_control_v7"
        or candidate.get("control_runtime_schema_version") != 7
        or candidate.get("environment_latent_count") != 256
        or candidate.get("environmental_text_inputs") != []
        or candidate.get("expected_checkpoint_path")
        != "data_gemma4/checkpoints/gemma4_v66b_allrow_always_on_control"
        or candidate.get("expected_training_report_path")
        != "reports/gemma4/metrics/v66b_allrow_always_on_distillation.json"
        or candidate.get("question_dependent_scene_retrieval") is not False
        or candidate.get("saved_runtime_training_gate_required") is not True
        or candidate.get("training_preregistration_sha256")
        != PINNED_V66_TRAINING_PREREGISTRATION_SHA256
        or candidate.get("training_report_artifact")
        != "v66b_allrow_paired_opposite_pair_disjoint_training"
        or controls
        != {
            "all_256_environment_latents_must_influence_signature": True,
            "all_rows_must_use_always_on_continuous_control": True,
            "bidirectional_scene_swap_required": True,
            "candidate_selection_after_reference_open_permitted": False,
            "natural_prefix_and_signature_immutable_across_questions": True,
            "reference_file_is_last_input_opened": True,
            "retry_permitted": False,
            "scene_swap_prefix_and_signature_must_match_injected_scene": True,
            "paired_opposite_training_dependence_gate_required": True,
        }
        or not isinstance(population, Mapping)
        or population
        != {
            "changed_paired_units": 26,
            "changed_sides": 52,
            "natural_question_count": 384,
            "paired_units": 192,
            "scene_count": 16,
            "scene_swap_question_count": 384,
        }
        or not isinstance(split, Mapping)
        or split.get("internal_validation_pair_ids") != list(boundary.INTERNAL_VALIDATION_PAIR_IDS)
        or split.get("internal_validation_scene_ids") != _expected_scene_ids()
        or split.get("training_pair_ids") != list(boundary.TRAIN_PAIR_IDS)
        or split.get("pair_disjoint") is not True
        or split.get("scene_disjoint") is not True
        or not isinstance(thresholds, Mapping)
        or thresholds.get("internal_validation") != INTERNAL_VALIDATION_THRESHOLDS
        or thresholds.get("same_question_different_scene") != SAME_QUESTION_THRESHOLDS
        or thresholds.get("scene_swap") != SCENE_SWAP_THRESHOLDS
        or thresholds.get("training") != TRAINING_THRESHOLDS
    ):
        raise ValueError("V66 internal-validation preregistration contract is invalid")
    protected = {f"scene_{number:06d}" for number in (*range(25, 31), *range(57, 63))}
    if protected.intersection(split["internal_validation_scene_ids"]):
        raise ValueError("V66 internal validation overlaps a protected scene split")
    return json.loads(json.dumps(value, allow_nan=False))


__all__ = [
    "ARTIFACT",
    "INTERNAL_VALIDATION_THRESHOLDS",
    "PINNED_SCORER_RECORDS_SHA256",
    "PINNED_SCORER_REFERENCES_SHA256",
    "PINNED_V62_BASELINE_LOCK_SHA256",
    "PINNED_V66_INTERNAL_VALIDATION_PREREGISTRATION_SHA256",
    "PINNED_V66_TRAINING_PREREGISTRATION_SHA256",
    "SAME_QUESTION_THRESHOLDS",
    "SCENE_SWAP_THRESHOLDS",
    "SCHEMA",
    "TRAINING_THRESHOLDS",
    "validate_v66_internal_validation_preregistration",
]
