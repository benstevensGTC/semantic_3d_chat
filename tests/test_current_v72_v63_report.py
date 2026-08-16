from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_report_edit_surface_is_outside_v62_v63_and_v64_sealed_paths() -> None:
    release = json.loads((ROOT / BUILDER["DECODER_READER_V62_RELEASE"]).read_text(encoding="utf-8"))
    terminal = json.loads(
        (ROOT / BUILDER["ATTENTION_READER_V63_TERMINAL"]).read_text(encoding="utf-8")
    )
    v62_source = set(release["bound_source_sha256"])
    v62_assets = set(release["bound_training_asset_sha256"])
    v63_pinned = set(terminal["v6_3_pinned_sha256"])
    v63_pinned.update(
        {
            BUILDER["ATTENTION_READER_V63_AUTHENTICATION_SOURCE"].as_posix(),
            BUILDER["ATTENTION_READER_V63_AUTHENTICATION_TEST"].as_posix(),
            BUILDER["ATTENTION_READER_V63_TERMINAL"].as_posix(),
        }
    )
    v64_pinned = {path.as_posix() for path in BUILDER["ATTENTION_READER_V64_EVIDENCE_SHA256"]}
    intended = {
        "README.md",
        "reports/final_report.md",
        "reports/metrics/current_metrics.json",
        "scripts/build_current_report.py",
        "tests/test_current_v72_v63_report.py",
    }

    assert len(v62_source) == 97
    assert len(v62_assets) == 55
    assert intended.isdisjoint(v62_source | v62_assets)
    assert intended.isdisjoint(v63_pinned)
    assert intended.isdisjoint(v64_pinned)


def test_current_summary_authenticates_v72_terminal_negative() -> None:
    summary = _summary()
    evidence = summary["v72_adaptive_fusion_terminal"]

    assert evidence["status"] == ("authenticated_terminal_development_negative_no_checkpoint")
    assert evidence["evidence_authenticated"] is True
    assert evidence["passed"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["held_pair_id"] == "pair_000011"
    assert evidence["adaptive"] == {
        "complete_class_units": 1,
        "prediction_change_units": 1,
        "positive_own_over_opposite_sides": 5,
    }
    assert evidence["stronger_branch_32"] == {
        "complete_class_units": 2,
        "prediction_change_units": 2,
        "positive_own_over_opposite_sides": 6,
    }
    assert evidence["full_numeric_screen_executed"] is False
    assert evidence["internal_validation_executed"] is False
    assert evidence["gemma_generation_executed"] is False
    assert evidence["checkpoint_published"] is False
    assert all(evidence["checks"].values())


def test_current_summary_authenticates_v63_positive_nonpromotable_pilot() -> None:
    summary = _summary()
    evidence = summary["fixed_prefix_attention_reader_v6_3"]

    assert evidence["status"] == (
        "authenticated_positive_train_only_pilot_continuation_authorized_no_runtime_promotion"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["diagnostic_pass"] is True
    assert evidence["promotion_eligible"] is False
    assert evidence["runtime_checkpoint_promotion_authorized"] is False
    assert evidence["runtime_checkpoint_exists"] is False
    assert evidence["trainable_parameter_count"] == 30_720
    assert evidence["optimizer_updates"] == 8
    assert evidence["pair_unit_count"] == 40
    assert evidence["baseline"]["positive_margin_sides"] == 48
    assert evidence["candidate"]["positive_margin_sides"] == 49
    assert evidence["baseline"]["complete_units"] == 16
    assert evidence["candidate"]["complete_units"] == 18
    assert evidence["delta"]["mean_margin"] == pytest.approx(0.005331867933273338)
    assert evidence["retention"]["mean_kl_nats"] == pytest.approx(0.00034054686319865915)
    assert evidence["retention"]["top1_agreement"] == 1.0
    assert evidence["forbidden_access_count"] == 0
    assert evidence["oracle_accessed"] is False
    assert evidence["validation_or_deferred_final_accessed"] is False
    assert evidence["continuation"] == "v6_4_pair_disjoint_train_only_confirmation"
    assert all(evidence["checks"].values())
    for path, digest in evidence["evidence_sha256"].items():
        assert summary["source_artifacts"][path] == digest


def test_current_summary_authenticates_v64_pair_disjoint_terminal_negative() -> None:
    summary = _summary()
    evidence = summary["fixed_prefix_attention_reader_v6_4"]

    assert evidence["status"] == (
        "authenticated_failed_pair_disjoint_generalization_no_checkpoint_no_promotion"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["screen_pass"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["exact_attention_surface_continuation_authorized"] is False
    assert evidence["runtime_checkpoint_promotion_authorized"] is False
    assert evidence["runtime_checkpoint_exists"] is False
    assert evidence["split"]["physical_pair_disjoint"] is True
    assert evidence["split"]["scene_disjoint"] is True
    assert evidence["split"]["train_unit_count"] == 28
    assert evidence["split"]["held_unit_count"] == 12
    assert evidence["optimization"] == {
        "epochs": 3,
        "updates": 12,
        "units_per_update": 7,
        "pair_unit_exposures": 84,
    }
    assert evidence["baseline_train"]["complete_units"] == 13
    assert evidence["candidate_train"]["complete_units"] == 15
    assert evidence["baseline_held"]["complete_units"] == 3
    assert evidence["candidate_held"]["complete_units"] == 3
    assert evidence["held_delta"]["mean_margin"] == pytest.approx(-0.016738658150037125)
    assert evidence["held_delta"]["mean_margin_softplus"] == pytest.approx(0.011246321101983425)
    assert evidence["failed_checks"] == [
        "held_mean_margin_delta_at_least_0_002",
        "held_mean_margin_softplus_delta_at_most_minus_0_001",
    ]
    assert evidence["retention"]["mean_kl_nats"] == pytest.approx(0.00029075244895109383)
    assert evidence["forbidden_access_count"] == 0
    assert evidence["oracle_accessed"] is False
    assert evidence["internal_validation_or_deferred_final_accessed"] is False
    assert evidence["result_sha256"] == (
        "a909c71e10c2cca5757556dd462132a499b09f05576bb11119bf1b7f424f0414"
    )
    assert evidence["terminal_sha256"] == (
        "7e144231b81d0082d6c90956072f7d2564775005d3b805f2192ed7c57fec442e"
    )
    assert all(evidence["checks"].values())
    for path, digest in evidence["evidence_sha256"].items():
        assert summary["source_artifacts"][path] == digest


def test_current_markdown_keeps_v72_v63_and_v64_claims_bounded() -> None:
    markdown = BUILDER["render_markdown"](_summary())
    collapsed = " ".join(markdown.split())

    assert "V72 is an authenticated development-negative mechanism test" in collapsed
    assert "adaptive complete units were 1/4 versus 2/4" in collapsed
    assert "Positive wrong-prefix margins improved from 48/80 to 49/80" in collapsed
    assert "complete paired units improved from 16/40 to 18/40" in collapsed
    assert "authenticated positive diagnostic, not a promoted adapter" in collapsed
    assert "that confirmation has since completed and failed" in collapsed
    assert "held mean margin fell by 0.016739" in collapsed
    assert "held margin softplus worsened by 0.011246" in collapsed
    assert "authorizes no continuation of this exact attention surface" in collapsed
    assert "No V72, V6.3, or V6.4 checkpoint exists" in collapsed
    assert (
        "current static-reader blocker is causal pair-disjoint generalization itself" in collapsed
    )


def test_v72_report_fails_closed_on_evidence_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_v72_terminal_development_negative"]
    original = ROOT / BUILDER["V72_DEVELOPMENT_EVIDENCE"]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    hashes = dict(inspector.__globals__["V72_EVIDENCE_SHA256"])
    original_key = next(path for path in hashes if path.name == original.name)
    hashes[tampered] = hashes.pop(original_key)
    monkeypatch.setitem(inspector.__globals__, "V72_DEVELOPMENT_EVIDENCE", tampered)
    monkeypatch.setitem(inspector.__globals__, "V72_EVIDENCE_SHA256", hashes)

    result = inspector()

    assert result["status"] == "terminal_evidence_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["checkpoint_published"] is False
    assert "digest differs" in result["measurement_evidence_error"]


def test_v63_report_fails_closed_on_terminal_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_attention_reader_v63_terminal_evidence"]
    original = ROOT / BUILDER["ATTENTION_READER_V63_TERMINAL"]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    hashes = dict(inspector.__globals__["ATTENTION_READER_V63_EVIDENCE_SHA256"])
    original_key = next(path for path in hashes if path.name == original.name)
    hashes[tampered] = hashes.pop(original_key)
    monkeypatch.setitem(inspector.__globals__, "ATTENTION_READER_V63_TERMINAL", tampered)
    monkeypatch.setitem(inspector.__globals__, "ATTENTION_READER_V63_EVIDENCE_SHA256", hashes)

    result = inspector()

    assert result["status"] == "terminal_evidence_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_checkpoint_exists"] is False
    assert "digest differs" in result["measurement_evidence_error"]


def test_v64_report_fails_closed_on_terminal_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_attention_reader_v64_terminal_evidence"]
    original = ROOT / BUILDER["ATTENTION_READER_V64_TERMINAL"]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    hashes = dict(inspector.__globals__["ATTENTION_READER_V64_EVIDENCE_SHA256"])
    original_key = next(path for path in hashes if path.name == original.name)
    hashes[tampered] = hashes.pop(original_key)
    monkeypatch.setitem(inspector.__globals__, "ATTENTION_READER_V64_TERMINAL", tampered)
    monkeypatch.setitem(inspector.__globals__, "ATTENTION_READER_V64_EVIDENCE_SHA256", hashes)

    result = inspector()

    assert result["status"] == "terminal_evidence_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_checkpoint_exists"] is False
    assert "digest differs" in result["measurement_evidence_error"]
