from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_evidence import (
    CHECKPOINT,
    EVIDENCE_SHA256,
    TERMINAL_SMOKE,
    TRAINING_RELEASE,
    TRAINING_RESULT,
    authenticate_v6_terminal_smoke_failure,
)


def _copy_evidence(destination_root: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    for relative in EVIDENCE_SHA256:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative, destination)


def test_authenticates_terminal_v6_smoke_failure_without_training() -> None:
    evidence = authenticate_v6_terminal_smoke_failure()

    assert evidence["status"] == (
        "authenticated_terminal_smoke_failure_no_training_no_checkpoint"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["passed"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["failure_stage"] == (
        "byte_exact_full_vs_tail_answer_logit_equivalence"
    )
    assert evidence["gradient_computation_executed"] is False
    assert evidence["optimizer_constructed"] is False
    assert evidence["optimizer_steps"] == 0
    assert evidence["training_executed"] is False
    assert evidence["checkpoint_published"] is False
    assert evidence["checkpoint_absent"] is True
    assert evidence["training_release_absent"] is True
    assert evidence["training_result_absent"] is True
    assert evidence["forbidden_file_read_count"] == 0
    assert evidence["loaded_file_count"] == 233
    assert evidence["deferred_or_final_qa_accessed"] is False
    assert evidence["single_smoke_attempt_consumed"] is True
    assert evidence["maximum_smoke_attempts"] == 1
    assert evidence["planned_updates"] == 96
    assert evidence["evidence_sha256"][TERMINAL_SMOKE.as_posix()] == (
        EVIDENCE_SHA256[TERMINAL_SMOKE]
    )
    assert evidence["current_runtime_compatibility_claimed"] is False
    assert all(evidence["checks"].values())


def test_rejects_tampered_v6_terminal(tmp_path: Path) -> None:
    _copy_evidence(tmp_path)
    terminal = tmp_path / TERMINAL_SMOKE
    terminal.write_bytes(terminal.read_bytes() + b" ")

    with pytest.raises(ValueError, match="V6 evidence digest changed"):
        authenticate_v6_terminal_smoke_failure(tmp_path)


@pytest.mark.parametrize("unexpected", [TRAINING_RELEASE, TRAINING_RESULT, CHECKPOINT])
def test_rejects_any_unexpected_v6_successor_output(
    tmp_path: Path, unexpected: Path
) -> None:
    _copy_evidence(tmp_path)
    path = tmp_path / unexpected
    path.parent.mkdir(parents=True, exist_ok=True)
    if unexpected == CHECKPOINT:
        path.mkdir()
    else:
        path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError, match="training_release_result_and_checkpoint_absent"
    ):
        authenticate_v6_terminal_smoke_failure(tmp_path)


def test_v6_evidence_allowlist_excludes_v61_or_environmental_data() -> None:
    assert len(EVIDENCE_SHA256) == 4
    for path in EVIDENCE_SHA256:
        lowered = path.as_posix().casefold()
        assert "v6_1" not in lowered
        assert "v6.1" not in lowered
        assert "/oracle/" not in lowered
        assert "/qa/" not in lowered
        assert "/training/" not in lowered
        assert "/scorer_only/" not in lowered
