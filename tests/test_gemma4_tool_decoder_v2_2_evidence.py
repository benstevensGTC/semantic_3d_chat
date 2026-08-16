from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.gemma4_tool_decoder_v2_2_evidence import (
    CHECKPOINT,
    EVIDENCE_SHA256,
    TERMINAL_RESULT,
    authenticate_tool_decoder_v2_2_negative_result,
)


def _copy_evidence(destination_root: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    for relative in EVIDENCE_SHA256:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative, destination)


def test_authenticates_terminal_v2_2_negative_without_checkpoint() -> None:
    evidence = authenticate_tool_decoder_v2_2_negative_result()

    assert evidence["status"] == "authenticated_terminal_negative_no_runtime_checkpoint"
    assert evidence["evidence_authenticated"] is True
    assert evidence["passed"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["runtime_checkpoint_published"] is False
    assert evidence["runtime_checkpoint_absent"] is True
    assert evidence["greedy_generation_executed"] is False
    assert evidence["optimizer_updates"] == 64
    assert evidence["training_microbatches"] == 512
    assert evidence["training_loss_first"] == pytest.approx(2.414295881986618)
    assert evidence["training_loss_final"] == pytest.approx(0.2341814790852368)
    assert evidence["heldout_answer_token_nll"] == pytest.approx(0.37775762747489017)
    assert evidence["heldout_answer_token_accuracy"] == pytest.approx(0.8712881694434225)
    assert evidence["heldout_exact_sequence_accuracy"] == pytest.approx(0.17416225749559083)
    assert evidence["heldout_valid_schema_rate"] == pytest.approx(0.2641093474426808)
    assert evidence["heldout_tool_accuracy"] == pytest.approx(0.24118165784832452)
    assert evidence["terminal_result_sha256"] == EVIDENCE_SHA256[TERMINAL_RESULT]
    assert evidence["current_runtime_compatibility_claimed"] is False
    assert all(evidence["checks"].values())


def test_rejects_tampered_terminal_result(tmp_path: Path) -> None:
    _copy_evidence(tmp_path)
    terminal = tmp_path / TERMINAL_RESULT
    terminal.write_bytes(terminal.read_bytes() + b" ")

    with pytest.raises(ValueError, match="evidence digest changed"):
        authenticate_tool_decoder_v2_2_negative_result(tmp_path)


def test_rejects_unexpected_runtime_checkpoint(tmp_path: Path) -> None:
    _copy_evidence(tmp_path)
    checkpoint = tmp_path / CHECKPOINT
    checkpoint.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="runtime_checkpoint_absent"):
        authenticate_tool_decoder_v2_2_negative_result(tmp_path)


def test_read_allowlist_excludes_oracle_qa_maps_and_traces() -> None:
    for path in EVIDENCE_SHA256:
        parts = set(path.parts)
        assert "oracle" not in parts
        assert "qa" not in parts
        assert "maps" not in parts
        assert "traces" not in parts
