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


def test_current_summary_authenticates_v82_without_promotion(
    summary: dict[str, Any],
) -> None:
    evidence = summary["v82_strict_dense_reader"]

    assert evidence["status"] == ("authenticated_historical_development_gate_failed_not_promoted")
    assert evidence["evidence_authenticated"] is True
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["official_validation_measured"] is False
    assert evidence["reader"]["fixed_memory_shape"] == [1, 738, 1536]
    assert evidence["reader"]["positive_payload_tokens"] == 640
    assert evidence["reader"]["non_payload_boundary_and_probe_key_tokens"] == 98
    assert evidence["reader"]["trainable_parameter_count"] == 688130
    numeric = evidence["numeric_development"]
    assert numeric["row_count"] == 384
    assert numeric["scene_count"] == 16
    assert numeric["pair_disjoint"] is True
    assert numeric["scene_disjoint"] is True
    assert numeric["mean_control_cosine"] == pytest.approx(0.9913522601127625)
    assert numeric["normalized_mse"] == pytest.approx(0.02923930250108242)
    assert numeric["zero_environment_maximum_absolute_control"] == 0.0
    behavior = evidence["real_gemma_historical_development"]
    assert behavior["scores"] == {
        "v82": {"correct": 8, "total": 16},
        "frozen_v54": {"correct": 6, "total": 16},
        "shuffled_atlas": {"correct": 3, "total": 16},
        "zero_environment": {"correct": 1, "total": 16},
        "wrong_scene": {"correct": 6, "total": 16},
    }
    assert behavior["gates"]["correct_minus_wrong_scene_at_least_2"] is True
    assert behavior["gates"]["candidate_correct_at_least_9"] is False
    assert behavior["gates"]["gain_over_frozen_v54_at_least_3"] is False
    assert behavior["passed"] is False
    assert all(evidence["checks"].values())


def test_current_summary_authenticates_hybrid_navigation_and_preserves_failure(
    summary: dict[str, Any],
) -> None:
    evidence = summary["embodied_hybrid_navigation"]

    assert evidence["status"] == ("passed_two_scene_hybrid_semantic_face_target_development")
    assert evidence["evidence_authenticated"] is True
    assert evidence["scene_ids"] == ["scene_000001", "scene_000031"]
    assert evidence["passed_count"] == evidence["scene_count"] == 2
    assert evidence["collision_count"] == 0
    assert evidence["forbidden_access_count"] == 0
    assert evidence["runtime_oracle_inputs"] is False
    assert evidence["runtime_environmental_text_inputs"] == []
    assert evidence["final_continuous_grounding_residual_degrees"] == pytest.approx(
        [0.2624104250426882, 0.1623973612254446]
    )
    assert evidence["oracle_heading_error_degrees"] == pytest.approx(
        [6.579150291502685, 3.3015028644879436]
    )
    assert evidence["question_independent_static_base_memory"] is True
    assert evidence["question_conditioned_control_tokens"] is True
    assert evidence["question_conditioned_navigation_grounding"] is True
    assert evidence["learned_plus_numeric_convergence_interlock"] is True
    assert evidence["learned_only_diagnostic"] == {
        "scene_count": 1,
        "passed_count": 0,
        "step_count": 12,
        "termination_reason": "max_steps",
        "preserved_failure": True,
    }
    assert all(evidence["checks"].values())


def test_current_markdown_bounds_v82_and_hybrid_claims(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "V82 learned dense-reader state" in markdown
    assert "688,130-parameter learned dense reader" in collapsed
    assert "real local-Gemma bounded V82 behavior run scored 8/16" in collapsed
    assert "missed both its 9/16 candidate minimum and +3-over-V54 gate" in collapsed
    assert "not promoted or officially validated" in collapsed
    assert "completes 2/2 episodes with zero collisions" in collapsed
    assert "Final continuous-grounding residuals were 0.262 and 0.162 degrees" in collapsed
    assert "heading errors of 6.579 and 3.302 degrees" in collapsed
    assert "preserved learned-only predecessor passed 0/1" in collapsed
    assert "not a strict identical-input prefix" in collapsed


def test_v82_inspector_fails_closed_on_behavior_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v82_strict_dense_learned_reader"]
    original = ROOT / BUILDER["V82_HISTORICAL_SCORE"]
    tampered = json.loads(original.read_text(encoding="utf-8"))
    tampered["arms"]["v82"]["correct"] = 9
    path = tmp_path / original.name
    path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setitem(inspector.__globals__, "V82_HISTORICAL_SCORE", path)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "real_gemma_historical_behavior" in result["measurement_evidence_error"]


def test_hybrid_navigation_inspector_fails_closed_on_environment_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_embodied_hybrid_semantic_navigation"]
    original_paths = BUILDER["EMBODIED_HYBRID_RESULTS"]
    tampered = json.loads((ROOT / original_paths[0]).read_text(encoding="utf-8"))
    tampered["environmental_text_inputs"] = ["forbidden label"]
    path = tmp_path / original_paths[0].name
    path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setitem(
        inspector.__globals__,
        "EMBODIED_HYBRID_RESULTS",
        (path, original_paths[1]),
    )

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "scene_000001" in result["measurement_evidence_error"]
