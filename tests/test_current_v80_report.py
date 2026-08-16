from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_current_summary_authenticates_terminal_v80_mps_failure(
    summary: dict[str, Any],
) -> None:
    evidence = summary["v80_atlas_attention_reader"]

    assert evidence["status"] == (
        "authenticated_terminal_gradient_smoke_mps_oom_no_checkpoint_no_optimizer_update"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["preregistration_sha256"] == (
        "e44dc9aed1176cdfc30befe56d50e49a31f1638223a529a01bb086f5b3ea5894"
    )
    assert evidence["correction_v2_sha256"] == (
        "0f0d0183d4e6deed942465116305f5698d09717c5f233ef351445c828729c2cb"
    )
    assert evidence["terminal_sha256"] == (
        "41fdbfbc2fec970abe2dd1ac35b55947d3caf73788e7ef40d59a41db1c0448a0"
    )
    assert evidence["architecture"]["fixed_prefix_tokens"] == 738
    assert evidence["architecture"]["all_tokens_retained"] is True
    assert evidence["architecture"]["question_dependent_processing_or_selection"] is False
    assert evidence["architecture"]["trainable_parameter_count"] == 122_880

    corrected = evidence["prelaunch_correction"]
    assert corrected["authoritative_real_model_state"] == {
        "gradient_smoke_run": False,
        "loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
    }
    assert corrected["misleading_pass_boolean_corrected"] is True
    assert corrected["original_artifact_overwritten"] is False

    smoke = evidence["gradient_smoke"]
    assert smoke["phase"] == "gradient-smoke"
    assert smoke["passed"] is False
    assert smoke["error_type"] == "RuntimeError"
    assert "MPS backend out of memory" in smoke["error"]
    assert "3.75 GiB" in smoke["error"]
    assert "9.57 GiB" in smoke["error"]
    assert "13.32 GiB" in smoke["error"]
    assert smoke["optimizer_constructed"] is False
    assert smoke["optimizer_updates"] == 0
    assert smoke["terminal_artifact_optimizer_update_record"] == "unknown_or_zero"

    assert evidence["bounded_screen"] == {
        "launched": False,
        "report_present": False,
        "behavior_measured": False,
        "authorized": False,
    }
    assert evidence["checkpoint"]["published"] is False
    assert evidence["checkpoint"]["present"] is False
    assert evidence["protected_inputs"] == {
        "oracle_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
    }
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["behavioral_result_claimed"] is False
    assert all(evidence["preregistration_checks"].values())
    assert all(evidence["correction_v2_checks"].values())
    assert all(evidence["terminal_checks"].values())

    for path, digest in BUILDER["V80_ATLAS_ATTENTION_READER_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_report_bounds_v80_claims_and_describes_demo_modes(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "V80 atlas-attention-reader state" in markdown
    assert "correction-v2 artifact authenticates" in collapsed
    assert "real optimizer-free zero-step gradient smoke" in collapsed
    assert "3.75 GiB of MPS allocations plus 9.57 GiB" in collapsed
    assert "no optimizer update, checkpoint, behavioral result" in collapsed
    assert "protected data remained unopened" in collapsed
    assert "no behavioral or scene-causal improvement can be claimed" in collapsed
    assert "make demo                  # current V89: interactive on TTY, finite in CI" in markdown
    assert (
        "make demo-smoke            # finite promoted strict V89 three-question proof" in markdown
    )
    assert "make demo                  # finite promoted V75 continuous-scene chat" not in markdown


def test_v80_authentication_fails_closed_on_terminal_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v80_atlas_attention_reader_terminal"]
    globals_ = inspector.__globals__
    original_path = BUILDER["V80_ATLAS_ATTENTION_READER_TERMINAL"]
    original = ROOT / original_path
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    evidence = dict(BUILDER["V80_ATLAS_ATTENTION_READER_EVIDENCE_SHA256"])
    expected_digest = evidence.pop(original_path)
    evidence[tampered] = expected_digest
    monkeypatch.setitem(globals_, "V80_ATLAS_ATTENTION_READER_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["checkpoint"] == {"published": False, "present": False}
    assert result["runtime_promotion_authorized"] is False
    assert result["behavioral_result_claimed"] is False
    assert "Pinned evidence digest changed" in result["measurement_evidence_error"]
