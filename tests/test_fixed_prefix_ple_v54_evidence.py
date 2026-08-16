from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_evidence import (
    CHECKPOINT_PATHS,
    EVIDENCE_SHA256,
    authenticate_v1_v5_negative_results,
)


def _copy_evidence(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    for relative in EVIDENCE_SHA256:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative, destination)


def test_authenticates_terminal_v1_v5_negative_chain() -> None:
    evidence = authenticate_v1_v5_negative_results()

    assert evidence["status"] == "authenticated_terminal_negative_no_checkpoint"
    assert evidence["evidence_authenticated"] is True
    assert evidence["passed"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["checkpoint_published"] is False
    assert evidence["model_loaded"] is False
    assert evidence["mps_used"] is False
    assert all(evidence["checks"].values())
    assert all(evidence["checkpoint_absence"].values())

    v4 = evidence["versions"]["v4"]
    v5 = evidence["versions"]["v5"]
    assert v4["answer_nll_before"] == v5["answer_nll_before"]
    assert v4["positive_wrong_prefix_sides_before"] == 30
    assert v4["positive_wrong_prefix_sides_after"] == 28
    assert v4["complete_changed_units_before"] == 12
    assert v4["complete_changed_units_after"] == 10
    assert v5["positive_wrong_prefix_sides_after"] == 28
    assert v5["complete_changed_units_after"] == 9
    assert v5["broad_rows_consumed_exactly_once"] == 496
    assert v5["deferred_holdout_accessed"] is False
    assert v5["final_split_accessed"] is False


def test_authentication_rejects_any_reader_checkpoint(tmp_path: Path) -> None:
    _copy_evidence(tmp_path)
    checkpoint = tmp_path / CHECKPOINT_PATHS[-1]
    checkpoint.mkdir(parents=True)
    (checkpoint / "unexpected.bin").write_bytes(b"not an accepted checkpoint")

    with pytest.raises(RuntimeError, match="all_reader_checkpoints_absent"):
        authenticate_v1_v5_negative_results(tmp_path)


def test_evidence_allowlist_excludes_qa_oracle_and_holdout_paths() -> None:
    for path in EVIDENCE_SHA256:
        parts = set(path.parts)
        assert "oracle" not in parts
        assert "qa" not in parts
        assert "scorer_only" not in parts
        assert "training" not in parts
        assert "questions" not in parts
