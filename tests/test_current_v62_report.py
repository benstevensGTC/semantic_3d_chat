from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_current_summary_authenticates_v62_terminal_training_rejection() -> None:
    summary = _summary()
    evidence = summary["fixed_prefix_decoder_reader_v6_2"]

    assert evidence["status"] == (
        "authenticated_terminal_training_gate_failure_no_checkpoint"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["passed"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["training_executed"] is True
    assert evidence["optimizer_updates"] == 96
    assert evidence["qa_forward_path"] == (
        "full_huggingface_forward_token_normalized_ce"
    )
    assert evidence["baseline_answer_nll"] == pytest.approx(3.235830537897224)
    assert evidence["candidate_answer_nll"] == pytest.approx(1.9157106683123857)
    assert evidence["baseline_expanded_positive_margin_rate"] == pytest.approx(
        0.6647058823529411
    )
    assert evidence["candidate_expanded_positive_margin_rate"] == pytest.approx(
        0.5941176470588235
    )
    assert evidence["baseline_curated_complete_units"] == 12
    assert evidence["candidate_curated_complete_units"] == 11
    assert evidence["baseline_orientation_positive_margin_rate"] == pytest.approx(
        0.8571428571428571
    )
    assert evidence["candidate_orientation_positive_margin_rate"] == pytest.approx(
        0.14285714285714285
    )
    assert evidence["retention"]["passed"] is True
    assert evidence["greedy_evaluation_executed"] is False
    assert evidence["checkpoint_published"] is False
    assert evidence["loaded_file_count"] == 246
    assert evidence["forbidden_file_read_count"] == 0
    assert evidence["deferred_or_final_qa_accessed"] is False
    assert evidence["environmental_text_inputs"] == []
    assert evidence["elapsed_seconds"] == pytest.approx(872.4142552500125)
    assert evidence["memory"] == {
        "peak_process_rss_bytes": 6_420_758_528,
        "mps_current_allocated_bytes": 257_912_320,
        "mps_driver_allocated_bytes": 15_822_618_624,
    }
    assert evidence["release_sha256"] == (
        "c2cc4110549bf6fca6c575a247ef0d3494f85458e7e644e24ad051a64d023258"
    )
    assert evidence["terminal_sha256"] == (
        "e86b417d5edeaedc5f541171845c37d3e740b5b24468fb0b2b062a2b8ae85f12"
    )
    assert all(evidence["checks"].values())
    for path, digest in evidence["evidence_sha256"].items():
        assert summary["source_artifacts"][path] == digest


def test_current_markdown_states_bounded_v62_negative() -> None:
    markdown = BUILDER["render_markdown"](_summary())

    assert "completed all 96" in markdown
    assert "3.2358" in markdown
    assert "1.9157" in markdown
    assert "0.6647" in markdown
    assert "0.5941" in markdown
    assert "orientation positive margins" in markdown
    assert "0.8571" in markdown
    assert "0.1429" in markdown
    assert "retention gates" in markdown
    assert "greedy evaluation was correctly" in markdown
    assert "no checkpoint was published" in markdown
    assert "zero forbidden reads" in markdown


@pytest.mark.parametrize("artifact_name", ["terminal", "audit", "manifest"])
def test_v62_report_authentication_fails_closed_on_publication_tamper(
    artifact_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_decoder_reader_v62_terminal_evidence"]
    constants = {
        "terminal": "DECODER_READER_V62_TERMINAL",
        "audit": "DECODER_READER_V62_AUDIT",
        "manifest": "DECODER_READER_V62_MANIFEST",
    }
    original = ROOT / inspector.__globals__[constants[artifact_name]]
    original_key = next(
        path
        for path in inspector.__globals__["DECODER_READER_V62_EVIDENCE_SHA256"]
        if path.name == original.name
    )
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    monkeypatch.setitem(inspector.__globals__, constants[artifact_name], tampered)
    evidence_hashes = dict(inspector.__globals__["DECODER_READER_V62_EVIDENCE_SHA256"])
    evidence_hashes[tampered] = evidence_hashes.pop(original_key)
    monkeypatch.setitem(
        inspector.__globals__, "DECODER_READER_V62_EVIDENCE_SHA256", evidence_hashes
    )

    result = inspector()

    assert result["status"] == "terminal_evidence_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["passed"] is False
    assert result["checkpoint_published"] is False
    assert "digest differs" in result["measurement_evidence_error"]


def test_v62_report_authentication_fails_closed_on_selection_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_decoder_reader_v62_terminal_evidence"]
    original = ROOT / BUILDER["DECODER_READER_V62_TERMINAL"]
    tampered = tmp_path / original.name
    payload = original.read_text(encoding="utf-8").replace(
        '"answer_nll_mean": 1.9157106683123857',
        '"answer_nll_mean": 1.8157106683123857',
        1,
    )
    assert payload != original.read_text(encoding="utf-8")
    tampered.write_text(payload, encoding="utf-8")
    monkeypatch.setitem(inspector.__globals__, "DECODER_READER_V62_TERMINAL", tampered)
    evidence_hashes = dict(inspector.__globals__["DECODER_READER_V62_EVIDENCE_SHA256"])
    original_key = next(path for path in evidence_hashes if path.name == original.name)
    evidence_hashes[tampered] = BUILDER["_sha256_file"](tampered)
    evidence_hashes.pop(original_key)
    monkeypatch.setitem(
        inspector.__globals__, "DECODER_READER_V62_EVIDENCE_SHA256", evidence_hashes
    )

    result = inspector()

    assert result["status"] == "terminal_evidence_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["passed"] is False
    assert result["checkpoint_published"] is False
