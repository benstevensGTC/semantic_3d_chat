"""Create the immutable V66b paired-opposite training preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.training.train_question_control_v66 import (
    V66B_BEHAVIOR_THRESHOLDS,
)

_FILTERED_TRAIN_SHA256: Final[str] = (
    "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1"
)
_TRAINING_BASELINE_LOCK_SHA256: Final[str] = (
    "b1f20e64889116cceb0904ecb3842a6e43fcd6fa3cb0675c32a24f4d278e55e6"
)
_INVALIDATED_V66_PREREGISTRATION_SHA256: Final[str] = (
    "974f7049d2cf96670c77e6c19808a53fbca8b7c68e7cba7f9f5b184d0fc6ac4c"
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_v66b_preregistration() -> dict[str, Any]:
    """Return the locked successor protocol without reading evaluation inputs."""

    return {
        "schema_version": 1,
        "artifact": "v66b_allrow_paired_opposite_training_preregistration",
        "status": "locked_before_v66b_controller_training_or_generation",
        "research_change": (
            "replace_unidentified_cyclic_scene_control_with_exact_"
            "counterfactual_paired_opposite_scene_injection"
        ),
        "invalidated_predecessor": {
            "artifact": "v66_allrow_always_on_training_preregistration",
            "path": "reports/gemma4/metrics/v66_allrow_preregistration.json",
            "sha256": _INVALIDATED_V66_PREREGISTRATION_SHA256,
            "invalidated": True,
            "reason": (
                "cyclic_wrong_scene_pairs_preserved_answers_for_340_of_409_"
                "exact_text_mappings"
            ),
            "preserved_answer_mappings": 340,
            "exact_text_cyclic_mappings": 409,
            "predecessor_artifact_bytes_modified": False,
        },
        "authorization": {
            "filtered_training_qa_sha256": _FILTERED_TRAIN_SHA256,
            "training_baseline_lock_sha256": _TRAINING_BASELINE_LOCK_SHA256,
            "training_rows": 576,
            "training_scenes": 24,
            "counterfactual_pairs": 12,
            "answer_classes": 28,
        },
        "frozen_v54_training_baseline": {
            "canonical_type_specific_exact": 227,
            "total": 576,
            "changed_side_exact": 35,
            "changed_side_total": 80,
            "complete_changed_units": 5,
            "changed_unit_total": 40,
            "prediction_change_units": 9,
            "per_type": {
                "attribute": {"exact": 25, "total": 120},
                "count": {"exact": 56, "total": 96},
                "metric": {"exact": 0, "total": 24},
                "orientation": {"exact": 21, "total": 22},
                "presence": {"exact": 56, "total": 100},
                "spatial_relation": {"exact": 62, "total": 120},
                "support": {"exact": 7, "total": 94},
            },
        },
        "pair_heldout_inventory": {
            "vocabulary_supported_rows": 571,
            "vocabulary_unsupported_singleton_rows": 5,
            "supported_changed_sides": 75,
            "fully_supported_changed_units": 35,
            "partly_unsupported_changed_units": 5,
            "every_supported_row_generated_once": True,
            "fold_codebook_and_basis_built_after_pair_exclusion": True,
            "held_pair_teacher_sources_used": False,
        },
        "paired_opposite_control": {
            "changed_sides": 80,
            "counterfactual_units": 40,
            "same_question_bytes_on_both_sides": True,
            "injected_scene_is_exact_counterfactual_pair_side": True,
            "exact_paired_opposite_scene_prefix_injected": True,
            "exact_paired_opposite_scene_signature_injected": True,
            "injected_and_original_canonical_references_differ": True,
            "follows_injected_scene_scored_against_opposite_reference": True,
            "original_reference_retention_scored_as_failure_control": True,
            "question_dependent_scene_retrieval": False,
        },
        "thresholds": json.loads(
            json.dumps(asdict(V66B_BEHAVIOR_THRESHOLDS), allow_nan=False)
        ),
        "controls": {
            "actual_greedy_local_gemma_primary": True,
            "canonical_type_specific_scoring": True,
            "same_question_counterfactual_prediction_change_required": True,
            "exact_paired_opposite_scene_prefix_and_signature": True,
            "same_question_byte_identity_required": True,
            "answer_follows_injected_scene_scored_against_opposite_reference": True,
            "cyclic_wrong_complete_scene_prefix_and_signature": False,
            "saved_runtime_raw_question_token_generation_required": True,
            "sealed_checkpoint_public_reload_required": True,
            "unverified_native_answer_embedding_fallback_permitted": False,
        },
        "scope": {
            "training_only": True,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
        },
    }


def write_v66b_preregistration(path: str | Path) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists():
        raise FileExistsError(f"V66b preregistration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            build_v66b_preregistration(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, _sha256_file(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path, digest = write_v66b_preregistration(args.output)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_v66b_preregistration", "write_v66b_preregistration"]
