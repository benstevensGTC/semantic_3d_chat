from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_current_summary_authenticates_v83_structure_and_negative_behavior(
    summary: dict[str, Any],
) -> None:
    evidence = summary["v83_strict_direct_memory"]

    assert evidence["status"] == (
        "authenticated_strict_direct_behavior_failed_not_promoted"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["official_validation_measured"] is False
    structure = evidence["structural_contract"]
    assert structure == {
        "fixed_environment_memory_shape": [1, 738, 1536],
        "continuous_payload_tokens": 736,
        "native_boundary_tokens": 2,
        "memory_compiled_and_bound_before_questions": True,
        "same_memory_reused_for_every_question": True,
        "fixed_memory_invariant": True,
        "tokens_supplied_directly_to_gemma": 738,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "question_dependent_scene_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "reader_enabled": False,
        "control_activation_tokens": 0,
        "pad_ple_and_native_boundaries_exact": True,
    }
    behavior = evidence["real_gemma_historical_development"]
    assert behavior["scores"] == {
        "v83_direct": {"correct": 6, "total": 16},
        "frozen_v54": {"correct": 6, "total": 16},
        "paired_wrong": {"correct": 7, "total": 16},
        "shuffled_atlas": {"correct": 6, "total": 16},
        "zero_payload": {"correct": 5, "total": 16},
    }
    assert behavior["prediction_change_units"]["v83_direct"] == 1
    assert behavior["prediction_change_units"]["total"] == 8
    assert behavior["passed"] is False
    assert all(evidence["checks"].values())
    assert evidence["isolation"]["reference_artifact_opened_by_report_builder"] is False
    assert summary["claim_scope"]["v83_runtime_promoted"] is False


def test_current_markdown_reports_v83_without_promotion(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "V83 strict direct-memory state" in markdown
    assert "V83 is the strictest direct fixed-memory mechanism tested" in collapsed
    assert "There is no separate question-conditioned environmental reader" in collapsed
    assert "direct V83, 6/16 for frozen V54, 7/16 with the paired wrong scene" in collapsed
    assert "1/8 counterfactual units" in collapsed
    assert "V83 is authenticated but **not promoted**" in markdown
    assert "make v83-check" in markdown


def test_v83_inspector_fails_closed_on_score_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v83_strict_direct_fixed_memory"]
    original = ROOT / BUILDER["V83_HISTORICAL_SCORE"]
    tampered = json.loads(original.read_text(encoding="utf-8"))
    tampered["arms"]["v83_direct"]["correct"] = 9
    path = tmp_path / original.name
    path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setitem(inspector.__globals__, "V83_HISTORICAL_SCORE", path)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "isolated_behavior_score" in result["measurement_evidence_error"]


def test_v83_inspector_fails_closed_on_prediction_hash_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v83_strict_direct_fixed_memory"]
    original = ROOT / BUILDER["V83_HISTORICAL_PREDICTIONS"]
    tampered = json.loads(original.read_text(encoding="utf-8"))
    tampered["architecture"]["question_derived_environmental_tokens"] = 1
    path = tmp_path / original.name
    path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setitem(inspector.__globals__, "V83_HISTORICAL_PREDICTIONS", path)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "strict_direct_prediction_artifact" in result["measurement_evidence_error"]
