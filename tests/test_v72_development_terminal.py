from __future__ import annotations

import json
from pathlib import Path

from semantic_3d_chat.evaluation.v72_development_authentication import (
    V72_DEVELOPMENT_EVIDENCE_SHA256,
    authenticate_v72_development_negative,
)


def test_v72_development_negative_authenticates_and_is_unpublished() -> None:
    result = authenticate_v72_development_negative()

    assert result["measurement_authenticated"] is True
    assert result["status"] == (
        "authenticated_terminal_development_negative_no_checkpoint"
    )
    assert result["evidence_sha256"] == V72_DEVELOPMENT_EVIDENCE_SHA256
    assert result["checkpoint_absent"] is True
    assert result["adaptive_complete_class_units"] == 1
    assert result["branch_32_complete_class_units"] == 2


def test_v72_terminal_marker_forbids_expansion_or_promotion() -> None:
    marker = json.loads(
        Path("reports/gemma4/metrics/v72_adaptive_fusion_terminal.json").read_text(
            encoding="utf-8"
        )
    )

    assert marker["promotion_eligible"] is False
    assert marker["full_numeric_screen_authorized"] is False
    assert marker["checkpoint_published"] is False
    assert marker["gemma_generation_used"] is False
    assert all(marker["unexecuted"].values())


def test_v72_authentication_fails_closed_on_tamper(tmp_path: Path) -> None:
    evidence = json.loads(
        Path(
            "reports/gemma4/metrics/"
            "v72_adaptive_fusion_development_pair_000011.json"
        ).read_text(encoding="utf-8")
    )
    evidence["folds"][0]["adaptive_metrics"]["complete_class_units"] = 4
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(evidence), encoding="utf-8")

    result = authenticate_v72_development_negative(evidence_path=tampered)
    assert result["measurement_authenticated"] is False
    assert "development evidence digest differs or is unavailable" in result["errors"]


def test_v72_authentication_fails_closed_if_checkpoint_exists(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result = authenticate_v72_development_negative(checkpoint_path=checkpoint)

    assert result["measurement_authenticated"] is False
    assert "forbidden V72 checkpoint exists" in result["errors"]
