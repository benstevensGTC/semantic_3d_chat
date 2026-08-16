"""Model-free, label-isolated structured scorer for V96 evaluator-auth v2."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation.score_v96_known_development import (
    stable_invariant_metrics_v96,
    structured_metrics_v96,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import CONFIG
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    QUESTION_COUNT,
    REFERENCE_SHA256,
    SCENE_IDS,
    SCHEMA_VERSION,
    STRUCTURED_SCORE_ARTIFACT,
    assert_aggregate_only_v96,
    authenticate_prediction_bundle_v96,
    canonical_sha256_v96,
    load_references_v96_v2,
    structured_score_forbidden_roots_v96,
    validate_structured_metrics_v96,
    write_json_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation_v2 import (
    hardened_evaluation_stage_v96_v2,
)


@hardened_evaluation_stage_v96_v2
def score_known_development_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    bundle = authenticate_prediction_bundle_v96(config_path)
    path = bundle["paths"].structured_score
    if path.exists() or path.is_symlink():
        raise FileExistsError("V96 v2 structured score is create-once")
    audit = FileAccessAudit(
        forbidden_roots=structured_score_forbidden_roots_v96(bundle["config"]),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        references = load_references_v96_v2(bundle["config"], bundle["questions"])
        metrics = structured_metrics_v96(references, bundle["rows"])
    audit.assert_clean()
    validate_structured_metrics_v96(metrics)
    report = {
        "artifact": STRUCTURED_SCORE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "measured_aggregate_only_not_yet_gated",
        "candidate_fingerprint_sha256": bundle["fixed"].candidate[
            "fingerprint_sha256"
        ],
        "candidate_attestation_file_sha256": bundle["fixed"].candidate[
            "attestation_file_sha256"
        ],
        "model_snapshot_inventory_sha256": bundle["fixed"].candidate[
            "model_snapshot_inventory_sha256"
        ],
        "frozen_v95_state_sha256": bundle["fixed"].candidate[
            "frozen_v95_state_sha256"
        ],
        "memory_manifest_sha256": bundle["fixed"].memory_manifest_sha256,
        "bound_memory_inventory_sha256": canonical_sha256_v96(
            bundle["fixed"].memory_hashes
        ),
        "question_manifest_sha256": bundle["questions"].manifest_sha256,
        "questions_sha256": bundle["questions"].questions_sha256,
        "reference_sha256": REFERENCE_SHA256,
        "prediction_sha256": bundle["prediction_sha256"],
        "prediction_provenance_sha256": bundle["provenance"]["provenance_sha256"],
        "prediction_access_sha256": bundle["access_sha256"],
        "prediction_completion_sha256": bundle["completion_sha256"],
        "row_count": QUESTION_COUNT,
        "scene_count": len(SCENE_IDS),
        "prediction_bundle_authenticated_before_labels_opened": True,
        "labels_opened_only_by_separate_scorer": True,
        "scorer_loaded_model": False,
        "row_level_content_serialized": False,
        "metrics": metrics,
        "runtime_promotion_authorized": False,
    }
    assert_aggregate_only_v96(report)
    write_json_create_once_v96(path, report)
    return {**report, "structured_score_sha256": sha256_file_v85(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(score_known_development_v96(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_references_v96_v2",
    "main",
    "score_known_development_v96",
    "stable_invariant_metrics_v96",
    "structured_metrics_v96",
]
