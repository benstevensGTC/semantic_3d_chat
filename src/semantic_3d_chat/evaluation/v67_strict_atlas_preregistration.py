"""Immutable public contract for the one-shot V67 strict-atlas evaluation.

Only public source hashes, opaque scene IDs, numeric population counts, and
predeclared thresholds live here.  Validation never opens questions, maps,
predictions, answer references, training records, or oracle artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

SCHEMA: Final[str] = (
    "semantic_3d_chat.v67.strict_fixed_prefix_atlas_internal_validation_preregistration.v1"
)
ARTIFACT: Final[str] = "v67_strict_fixed_prefix_atlas_internal_validation_preregistration"
PINNED_V67_STRICT_ATLAS_PREREGISTRATION_SHA256: Final[str] = (
    "495d9a117bbb89a6072133e0c018eaa4a64e6aea6932360648062a4acb1c7f55"
)
PINNED_V67_TRAINING_PREREGISTRATION_SHA256: Final[str] = (
    "a87ad59102c48da95390659839b76707c3d32af726034ab930fae5e01ba7ab8f"
)
PINNED_SCORER_REFERENCES_SHA256: Final[str] = (
    "4202e777ee57ab3f7da329f15589e56b8b0464b782fb4d856dd1a3281ff3115c"
)
PINNED_SCORER_RECORDS_SHA256: Final[str] = (
    "e140ea3a178d4f9bd8ac93c72148df73219ccac2867e5590ddd01f49e4d5b5a9"
)
PINNED_ATLAS_PREDICTION_SOURCE_SHA256: Final[str] = (
    "144816a21dc7f7e06004565076919cda6568558d4463f2cbb2503d7bf378ac87"
)
PINNED_TERMINAL_GATE_SOURCE_SHA256: Final[str] = (
    "8cd354119b0ca5ed466999558904e82da14b325ae1d644f39de46cf2c037f9b3"
)

SCENE_IDS: Final[tuple[str, ...]] = tuple(
    [f"scene_{number:06d}" for number in range(39, 53)] + ["scene_000055", "scene_000056"]
)
POPULATION: Final[dict[str, int]] = {
    "changed_paired_units": 26,
    "changed_sides": 52,
    "natural_question_count": 384,
    "paired_units": 192,
    "scene_count": 16,
}
TERMINAL_THRESHOLDS: Final[dict[str, dict[str, int | float]]] = {
    "changed_paired_unit_complete": {"minimum": 10, "total": 26},
    "changed_paired_unit_correct_direction": {"minimum": 15, "total": 26},
    "changed_side_exact": {"minimum": 32, "total": 52},
    "natural_canonical_exact": {"minimum": 192, "total": 384},
    "normalized_exact_accuracy": {"minimum": 0.5, "total": 384},
    "prefix_invariance": {
        "questions_with_prebuilt_scene_prefix": 384,
        "scenes_with_one_prefix_hash": 16,
    },
}

_CANDIDATE_CONTRACT: Final[dict[str, Any]] = {
    "atlas_architecture": "fixed_scene_key_value_atlas_v1",
    "compiled_fixed_prefix_tokens": 738,
    "environmental_text_inputs": [],
    "expected_atlas_checkpoint": ("data_gemma4/checkpoints/gemma4_v67_strict_fixed_prefix_atlas"),
    "expected_atlas_predictions": (
        "reports/gemma4/predictions/v67_strict_fixed_prefix_atlas_internal_validation.jsonl"
    ),
    "expected_source_checkpoint": ("data_gemma4/checkpoints/gemma4_v67_pair_objective_control"),
    "expected_source_training_report": ("reports/gemma4/metrics/v67_pair_objective_training.json"),
    "global_environment_latents": 256,
    "probe_count": 96,
    "question_conditioned_scene_processing": False,
    "question_dependent_retrieval": False,
    "source_training_preregistration_sha256": (PINNED_V67_TRAINING_PREREGISTRATION_SHA256),
    "source_training_saved_runtime_gate_required": True,
}
_CONTROLS: Final[dict[str, bool]] = {
    "all_256_base_environment_latents_preserved": True,
    "all_compiled_atlas_tokens_present_before_question": True,
    "exact_same_prefix_for_every_question_in_scene": True,
    "no_question_selected_blocks_or_tokens": True,
    "oracle_and_training_directory_deletion_required": True,
    "prefix_hash_invariance_required": True,
    "reference_file_is_last_input_opened": True,
    "retry_permitted": False,
    "same_question_different_scene_prefixes_required": True,
}
_SOURCE_BOUNDARY: Final[dict[str, str]] = {
    "atlas_compiler_config_sha256": (
        "8c74d4ce3c1cda04eab9e65d2131544f36f8c458640432642fe30d466cc589f1"
    ),
    "atlas_compiler_source_sha256": (
        "22221fc4167d89d7b5e6822c80b6289312b3cb02a37650055c385a8b8750b438"
    ),
    "atlas_prediction_source_sha256": PINNED_ATLAS_PREDICTION_SOURCE_SHA256,
    "atlas_runtime_source_sha256": (
        "ba8fa693fc85355472b928ef874d6607689d143263f31356fcfe6ddb92eaca8c"
    ),
    "question_key_inventory_sha256": (
        "f36885e43100a5b7a3682ca38f7a06187c1f9b204095f5dc89b2e597e227ba27"
    ),
    "questions_manifest_sha256": (
        "078f65e1402e6e382a7bfdb2ad4b8a65d58e3164705a8a46cd222503aa201052"
    ),
    "questions_sha256": ("05bd92897b1888b92cfe7be651cc83f9b94cc4a36950c17cb58859ec73325167"),
    "scorer_records_sha256": PINNED_SCORER_RECORDS_SHA256,
    "scorer_references_sha256": PINNED_SCORER_REFERENCES_SHA256,
    "terminal_gate_source_sha256": PINNED_TERMINAL_GATE_SOURCE_SHA256,
}
_UNSCORED_REPORTED_METRICS: Final[list[str]] = [
    "attribute_accuracy",
    "count_accuracy",
    "grounding_coordinate_error_m",
    "presence_precision_recall_f1",
    "spatial_relation_accuracy",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_regular_file(path: str | Path) -> Path:
    source = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V67 preregistration path contains a symlink: {current}")
    if not source.is_file():
        raise FileNotFoundError(f"V67 preregistration is unavailable: {source}")
    return source


def validate_v67_strict_atlas_preregistration(path: str | Path) -> dict[str, Any]:
    """Authenticate the exact public V67 contract and return a detached copy."""

    source = _resolve_regular_file(path)
    if _sha256_file(source) != PINNED_V67_STRICT_ATLAS_PREREGISTRATION_SHA256:
        raise ValueError("V67 strict-atlas preregistration differs from its pin")
    try:
        value = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("V67 strict-atlas preregistration is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("V67 strict-atlas preregistration must be an object")

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
        or value.get("created_utc") != "2026-08-12T18:08:23Z"
        or value.get("status")
        != "locked_before_v67_training_generation_or_internal_validation_inference"
        or not isinstance(source_boundary, Mapping)
        or source_boundary != _SOURCE_BOUNDARY
        or not isinstance(candidate, Mapping)
        or candidate != _CANDIDATE_CONTRACT
        or not isinstance(controls, Mapping)
        or controls != _CONTROLS
        or not isinstance(population, Mapping)
        or population != POPULATION
        or not isinstance(split, Mapping)
        or split
        != {
            "pair_disjoint": True,
            "scene_disjoint": True,
            "scene_ids": list(SCENE_IDS),
        }
        or not isinstance(thresholds, Mapping)
        or thresholds != TERMINAL_THRESHOLDS
        or value.get("unscored_reported_metrics") != _UNSCORED_REPORTED_METRICS
    ):
        raise ValueError("V67 strict-atlas preregistration contract is invalid")

    protected = {f"scene_{number:06d}" for number in (*range(25, 31), *range(57, 63))}
    if protected.intersection(split["scene_ids"]):
        raise ValueError("V67 strict-atlas evaluation overlaps a protected scene split")
    return json.loads(json.dumps(value, allow_nan=False))


__all__ = [
    "ARTIFACT",
    "PINNED_ATLAS_PREDICTION_SOURCE_SHA256",
    "PINNED_SCORER_RECORDS_SHA256",
    "PINNED_SCORER_REFERENCES_SHA256",
    "PINNED_TERMINAL_GATE_SOURCE_SHA256",
    "PINNED_V67_STRICT_ATLAS_PREREGISTRATION_SHA256",
    "PINNED_V67_TRAINING_PREREGISTRATION_SHA256",
    "POPULATION",
    "SCENE_IDS",
    "SCHEMA",
    "TERMINAL_THRESHOLDS",
    "validate_v67_strict_atlas_preregistration",
]
