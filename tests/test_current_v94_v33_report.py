from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


@pytest.fixture(scope="module")
def summary() -> dict[str, object]:
    return BUILDER["build_summary"]()


def test_v94_full_strong_causal_result_is_authenticated_and_claim_bounded(
    summary: dict[str, object],
) -> None:
    result = summary["v94_strong_causal_ablations"]
    assert isinstance(result, dict)
    assert result["status"] == "terminal_measured_posthoc_diagnostic_non_promotable"
    assert result["evidence_authenticated"] is True
    assert result["posthoc_non_preregistered"] is True
    assert result["runtime_promotion_authorized"] is False
    assert result["held_out_or_generalization_claim"] is False
    assert result["question_count"] == 216
    assert result["arms"]["primary"]["correct"] == 143
    assert result["arms"]["zero_full_scene"]["correct"] == 85
    assert result["arms"]["wrong_scene_swap"]["correct"] == 140
    assert result["arms"]["full_interior_token_permutation"]["correct"] == 143
    assert result["arms"]["position_spatial_shuffle"]["correct"] == 132
    assert result["arms"]["semantic_payload_shuffle"]["correct"] == 131
    assert result["arms"]["remove_rgb"]["correct"] == 147
    assert result["nll_measured"] is False


def test_v33_development_calibration_is_authenticated_without_generalization_claim(
    summary: dict[str, object],
) -> None:
    result = summary["navigation_policy_v3_3_development_calibration"]
    assert isinstance(result, dict)
    assert result["status"] == "accepted_development_calibration"
    assert result["evidence_authenticated"] is True
    assert result["development_calibration"] is True
    assert result["same_benchmark_used_for_diagnosis"] is True
    assert result["held_out_claim"] is False
    assert result["generalization_claim"] is False
    assert result["project_wide_runtime_promotion_claimed"] is False
    assert result["metrics"] == {
        "action_failure_count": 0,
        "collision_count": 0,
        "executed_action_count": 28,
        "policy_rejection_count": 0,
        "success_count": 6,
        "success_rate": 1.0,
        "task_count": 6,
    }
    assert result["continuous_context"]["step_count"] == 28
    assert result["continuous_context"]["map_update_count"] == 1
    assert result["calibration_task_metrics"]["final_target_standoff_m"] == pytest.approx(
        0.28686400627712905
    )
    assert result["calibration_task_metrics"]["target_progress_m"] == pytest.approx(
        2.1144650214101115
    )
    assert result["trajectory_journal"]["path"].endswith("learned_v3_3.json")


def test_current_markdown_states_v94_and_v33_boundaries(
    summary: dict[str, object],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "143/216 (66.20%)" in collapsed
    assert "paired wrong-scene memory fell only 1.39 points" in collapsed
    assert "full interior-token permutation had no accuracy drop at all" in collapsed
    assert "does not show useful incremental RGB dependence" in collapsed
    assert "completed all 6/6 tasks" in collapsed
    assert "stopped at 0.287 m" in collapsed
    assert "same one-scene six-task benchmark was used to diagnose" in collapsed
    assert "not held-out, cross-scene, or general conversational-navigation evidence" in collapsed


def test_new_report_inspectors_fail_closed_on_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v94_inspector = BUILDER["_inspect_v94_strong_causal_ablations"]
    monkeypatch.setitem(
        v94_inspector.__globals__, "V94_STRONG_CAUSAL_ABLATIONS_SHA256", "0" * 64
    )
    rejected_v94 = v94_inspector()
    assert rejected_v94["evidence_authenticated"] is False
    assert rejected_v94["runtime_promotion_authorized"] is False

    v33_inspector = BUILDER["_inspect_navigation_policy_v3_3_development_calibration"]
    hashes = dict(v33_inspector.__globals__["NAVIGATION_POLICY_V3_3_EVIDENCE_SHA256"])
    first_path = next(iter(hashes))
    hashes[first_path] = "0" * 64
    monkeypatch.setitem(
        v33_inspector.__globals__, "NAVIGATION_POLICY_V3_3_EVIDENCE_SHA256", hashes
    )
    rejected_v33 = v33_inspector()
    assert rejected_v33["evidence_authenticated"] is False
    assert rejected_v33["runtime_variant_authorized_for_development_benchmark"] is False
