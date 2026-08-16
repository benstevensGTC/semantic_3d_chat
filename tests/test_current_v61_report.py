from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_current_summary_authenticates_v61_terminal_gradient_rejection() -> None:
    summary = _summary()
    evidence = summary["fixed_prefix_decoder_reader_v6_1"]

    assert evidence["status"] == (
        "authenticated_terminal_gradient_equivalence_failure_no_training_no_checkpoint"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["passed"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["failure_stage"] == "gradient_equivalence"
    assert evidence["objective_equivalence_passed"] is True
    assert evidence["raw_logits_max_abs_difference"] == pytest.approx(0.125)
    assert evidence["raw_logits_rms_difference"] == pytest.approx(0.0003781102763499204)
    assert evidence["nll_max_abs_difference"] == 0.0
    assert evidence["js_divergence_max"] == pytest.approx(1.63732948148244e-11)
    assert set(evidence["individual_gradient_comparisons"]) == {
        "correct",
        "wrong",
        "broad",
    }
    assert all(
        row["passed"] is True for row in evidence["individual_gradient_comparisons"].values()
    )
    aggregate = evidence["aggregate_gradient_comparison"]
    assert aggregate["passed"] is False
    assert aggregate["cosine_similarity"] == pytest.approx(0.9998925988237569)
    assert aggregate["cosine_similarity"] < aggregate["cosine_minimum"]
    assert aggregate["relative_l2"] == pytest.approx(0.014655751591121162)
    assert aggregate["relative_l2"] > aggregate["relative_l2_maximum"]
    assert aggregate["norm_ratio"] == pytest.approx(1.0000920072812416)
    assert aggregate["full_gradient_l2"] == pytest.approx(0.5059195382354968)
    assert evidence["optimizer_constructed"] is False
    assert evidence["optimizer_steps"] == 0
    assert evidence["training_executed"] is False
    assert evidence["checkpoint_published"] is False
    assert evidence["loaded_file_count"] == 240
    assert evidence["forbidden_file_read_count"] == 0
    assert evidence["release_sha256"] == (
        "4456ebd11d8cbb154236aa6962bfc5875499580ab326068b1b9581f2127e4b33"
    )
    assert evidence["attempt_sha256"] == (
        "ec462122b737cda9bd111afa2a66f187039711e3f211ea3901f2eaa15986e53a"
    )
    assert evidence["terminal_sha256"] == (
        "099c1fa684439814b58c17223781b745e406d17cc20c65c402159bd0ede18add"
    )
    assert all(evidence["checks"].values())
    assert evidence["successor_v6_2"] == {
        "status": "terminal_training_gate_failure_no_checkpoint",
        "method": "exact_full_forward_training_path",
        "run_claimed": True,
        "pass_claimed": False,
    }
    for path, digest in evidence["evidence_sha256"].items():
        assert summary["source_artifacts"][path] == digest


def test_current_markdown_states_bounded_v61_claim_and_v62_successor_result() -> None:
    markdown = BUILDER["render_markdown"](_summary())

    assert "Objective equivalence passed" in markdown
    assert "0.9998925988237569" in markdown
    assert "0.0146557515911212" in markdown
    assert "full gradient norm was nonzero" in markdown
    assert "240 files and zero forbidden" in markdown
    assert "no optimizer, ran no training, and published no checkpoint" in markdown
    assert "V6.2 then removed the disputed" in markdown
    assert "failed its causal scene-selectivity gate" in markdown


def test_v61_report_authentication_fails_closed_on_terminal_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_decoder_reader_v61_terminal_evidence"]
    terminal = ROOT / BUILDER["DECODER_READER_V61_TERMINAL"]
    tampered = tmp_path / terminal.name
    tampered.write_bytes(terminal.read_bytes() + b" ")
    monkeypatch.setitem(inspector.__globals__, "DECODER_READER_V61_TERMINAL", tampered)

    result = inspector()

    assert result["status"] == "terminal_evidence_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["passed"] is False
    assert result["checkpoint_published"] is False
    assert "digest differs" in result["measurement_evidence_error"]
