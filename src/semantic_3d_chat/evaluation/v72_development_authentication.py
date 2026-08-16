"""Authenticate the sealed V72 train-only development-negative result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT


V72_DEVELOPMENT_EVIDENCE: Final[str] = (
    "reports/gemma4/metrics/v72_adaptive_fusion_development_pair_000011.json"
)
V72_DEVELOPMENT_EVIDENCE_SHA256: Final[str] = (
    "7cbfdcbf3953aecad51e40c21bb1d8f14962f890d90337cbf9b796b11af6e72a"
)
V72_TERMINAL_MARKER: Final[str] = (
    "reports/gemma4/metrics/v72_adaptive_fusion_terminal.json"
)
V72_FORBIDDEN_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v72_adaptive_fusion_dev_forbidden"
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_v72_development_negative(
    *,
    evidence_path: str | Path = V72_DEVELOPMENT_EVIDENCE,
    marker_path: str | Path = V72_TERMINAL_MARKER,
    checkpoint_path: str | Path = V72_FORBIDDEN_CHECKPOINT,
) -> dict[str, Any]:
    """Fail closed unless evidence, terminal status, and no-publication agree."""

    evidence_source = _resolve(evidence_path)
    marker_source = _resolve(marker_path)
    checkpoint_source = _resolve(checkpoint_path)
    errors: list[str] = []
    evidence: Any = None
    marker: Any = None
    if (
        not evidence_source.is_file()
        or evidence_source.is_symlink()
        or _sha256(evidence_source) != V72_DEVELOPMENT_EVIDENCE_SHA256
    ):
        errors.append("development evidence digest differs or is unavailable")
    else:
        evidence = json.loads(evidence_source.read_text(encoding="utf-8"))
    if not marker_source.is_file() or marker_source.is_symlink():
        errors.append("terminal marker is unavailable")
    else:
        marker = json.loads(marker_source.read_text(encoding="utf-8"))
    if checkpoint_source.exists():
        errors.append("forbidden V72 checkpoint exists")

    fold = (
        evidence.get("folds", [None])[0]
        if isinstance(evidence, dict) and len(evidence.get("folds", [])) == 1
        else None
    )
    adaptive = fold.get("adaptive_metrics") if isinstance(fold, dict) else None
    branch = (
        fold.get("branch_diagnostics", {}).get("branch_32", {}).get("metrics")
        if isinstance(fold, dict)
        else None
    )
    calibration = fold.get("calibration") if isinstance(fold, dict) else None
    held_fusion = fold.get("held_fusion") if isinstance(fold, dict) else None
    if (
        not isinstance(evidence, dict)
        or evidence.get("status") != "development_measurement_only"
        or evidence.get("checkpoint_published") is not False
        or evidence.get("gemma_generation_used") is not False
        or not isinstance(fold, dict)
        or fold.get("held_pair_id") != "pair_000011"
        or fold.get("held_rows_used_for_optimization") is not False
        or fold.get("held_teacher_sources_used") is not False
        or not isinstance(adaptive, dict)
        or not isinstance(branch, dict)
        or not isinstance(calibration, dict)
        or not isinstance(held_fusion, dict)
        or adaptive.get("complete_class_units") != 1
        or adaptive.get("prediction_change_units") != 1
        or adaptive.get("positive_own_over_opposite_sides") != 5
        or branch.get("complete_class_units") != 2
        or branch.get("prediction_change_units") != 2
        or branch.get("positive_own_over_opposite_sides") != 6
        or calibration.get("branch_parameters_changed") is not False
        or calibration.get("distinct_question_weight_vectors", 0) <= 1
        or held_fusion.get("distinct_row_vectors", 0) <= 1
    ):
        errors.append("development evidence semantics differ")
    if (
        not isinstance(marker, dict)
        or marker.get("artifact")
        != "v72_adaptive_fusion_terminal_development_negative_v1"
        or marker.get("status") != "terminal_development_negative_no_checkpoint"
        or marker.get("promotion_eligible") is not False
        or marker.get("full_numeric_screen_authorized") is not False
        or marker.get("checkpoint_published") is not False
        or marker.get("gemma_generation_used") is not False
        or marker.get("development_evidence", {}).get("sha256")
        != V72_DEVELOPMENT_EVIDENCE_SHA256
        or not all(marker.get("unexecuted", {}).values())
    ):
        errors.append("terminal marker semantics differ")
    return {
        "measurement_authenticated": not errors,
        "status": (
            "authenticated_terminal_development_negative_no_checkpoint"
            if not errors
            else "authentication_failed"
        ),
        "errors": errors,
        "evidence_sha256": (
            _sha256(evidence_source) if evidence_source.is_file() else None
        ),
        "checkpoint_absent": not checkpoint_source.exists(),
        "adaptive_complete_class_units": (
            adaptive.get("complete_class_units") if isinstance(adaptive, dict) else None
        ),
        "branch_32_complete_class_units": (
            branch.get("complete_class_units") if isinstance(branch, dict) else None
        ),
    }


__all__ = [
    "V72_DEVELOPMENT_EVIDENCE_SHA256",
    "authenticate_v72_development_negative",
]
