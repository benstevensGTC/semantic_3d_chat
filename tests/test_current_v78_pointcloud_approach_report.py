from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_current_summary_authenticates_exact_v78_held_pointcloud(
    summary: dict[str, Any],
) -> None:
    evidence = summary["v78_grounding_held_pointcloud"]

    assert evidence["status"] == (
        "authenticated_historical_internal_held_pointcloud_reproduction_"
        "evaluation_only_not_promoted"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["historical_internal_held_only"] is True
    assert evidence["official_validation"] is False
    assert evidence["runtime_path_executed"] is False
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["oracle_targets_evaluator_only"] is True
    assert evidence["environmental_text_inputs_to_primary_runtime"] == []
    assert evidence["metric_reproduction"]["count"] == 94
    assert evidence["metric_reproduction"]["mean_coordinate_error_m"] == pytest.approx(
        0.5270832777023315
    )
    assert evidence["metric_reproduction"]["within_1m_accuracy"] == pytest.approx(
        0.9255319237709045
    )
    assert evidence["numeric_map_support"]["mean_nearest_voxel_distance_m"] == (
        pytest.approx(0.17940254182473217)
    )
    assert evidence["readout_contract"] == {
        "complete_question_independent_scene_prefix": True,
        "scene_latent_count": 256,
        "scene_dimension": 1536,
        "every_scene_token_scored": True,
        "question_dependent_scene_retrieval": False,
        "top_k_scene_selection": False,
        "grounding_readout_is_question_conditioned": True,
        "strict_identical_total_environment_conditioned_input": False,
    }
    assert evidence["figure"] == {
        "path": "reports/gemma4/figures/v78_grounding_held_pointcloud_examples.png",
        "report_link": "gemma4/figures/v78_grounding_held_pointcloud_examples.png",
        "sha256": "8289bfa9d40097336c834a00555f43aef2e51dfe9b7cd04113f1e81876b0bfb2",
        "dimensions_px": [2745, 1927],
        "selected_count": 6,
        "selection_rule": (
            "lexicographically smallest question_id in each held scene; selected "
            "before inspecting predictions or coordinate errors"
        ),
        "oracle_markers_evaluation_only": True,
    }
    for path, digest in BUILDER[
        "V78_GROUNDING_HELD_POINTCLOUD_EVIDENCE_SHA256"
    ].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_summary_preserves_v2_failure_and_v3_completion_modes(
    summary: dict[str, Any],
) -> None:
    v2 = summary["embodied_hybrid_approach_v2"]
    v3 = summary["embodied_hybrid_approach_v3"]

    assert v2["status"] == (
        "authenticated_two_scene_v2_approach_development_one_of_two_passed"
    )
    assert v2["passed_count"] == 1
    assert v2["scene_count"] == 2
    assert v2["all_passed"] is False
    assert v2["runtime_oracle_inputs"] is False
    assert v2["runtime_environmental_text_inputs"] == []
    assert v2["runs"][1] == {
        "scene_id": "scene_000031",
        "passed": False,
        "runtime_completed": False,
        "stopped": False,
        "collision": True,
        "termination_reason": "action_failure",
        "numeric_translation_m": pytest.approx(1.2202412709593775),
        "oracle_center_progress_m": pytest.approx(1.2077359234198912),
        "final_oracle_bbox_standoff_m": pytest.approx(0.358838234313414),
    }

    assert v3["status"] == (
        "authenticated_two_scene_v3_approach_development_two_of_two_passed"
    )
    assert v3["passed_count"] == v3["scene_count"] == 2
    assert v3["all_passed"] is True
    assert v3["collision_count"] == 0
    assert v3["runtime_oracle_inputs"] is False
    assert v3["runtime_environmental_text_inputs"] == []
    assert v3["completion_modes"] == [
        "semantic_standoff",
        "collision_limited_safe_stop",
    ]
    assert v3["runs"][0]["semantic_target_distance_m"] == pytest.approx(
        0.4819971733403751
    )
    assert v3["runs"][1]["semantic_standoff_completed"] is False
    assert v3["runs"][1]["safe_collision_limited_completion"] is True
    assert v3["runs"][1]["semantic_target_distance_m"] == pytest.approx(
        0.7627565297837082
    )
    assert v3["runs"][1]["numeric_translation_m"] == pytest.approx(
        1.2867964024999163
    )
    assert v3["runs"][1]["oracle_center_progress_m"] == pytest.approx(
        1.2721308361074941
    )
    assert v3["runs"][1]["final_oracle_bbox_standoff_m"] == pytest.approx(
        0.29235282336232515
    )
    assert v3["historical_policy_source_snapshot"]["sha256"] == (
        "4e687161f6174192a2e44de160c847a70c6dbbab09f7f3277373f6bceed5fcc2"
    )
    for path, digest in BUILDER["EMBODIED_HYBRID_APPROACH_V3_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_markdown_bounds_v78_and_approach_claims(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "reconstructed all 94 coordinate rows with zero aggregate delta" in collapsed
    assert "mean error 0.527083278 m" in collapsed
    assert "92.55% within 1 m" in collapsed
    assert "mean nearest-voxel distance 0.179402542 m" in collapsed
    assert "Oracle target markers and target coordinates are evaluator-only" in collapsed
    assert "grounding readout itself is question-conditioned" in collapsed
    assert "not promoted" in collapsed
    assert "8289bfa9d40097336c834a00555f43aef2e51dfe9b7cd04113f1e81876b0bfb2" in markdown
    assert "](gemma4/figures/v78_grounding_held_pointcloud_examples.png)" in markdown

    assert "V2 passed 1/2" in collapsed
    assert "moved 1.220 m" in collapsed
    assert "caused `action_failure`" in collapsed
    assert "That failure is preserved, not overwritten" in collapsed
    assert "V3 successor passed 2/2" in collapsed
    assert "zero collisions, zero forbidden reads" in collapsed
    assert "final semantic target distance was 0.763 m" in collapsed
    assert "ordinary 0.5 m semantic-standoff goal remained false" in collapsed
    assert "`collision_limited_safe_stop`" in collapsed
    assert "This is 2/2 under the declared continuous-completion rule" in collapsed
    assert "not 2/2 ordinary-standoff completion" in collapsed


def test_v78_pointcloud_inspector_fails_closed_on_metric_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_v78_grounding_held_pointcloud"]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER["V78_GROUNDING_HELD_POINTCLOUD_REPORT"]
    tampered_value = json.loads(original.read_text(encoding="utf-8"))
    tampered_value["reproduced_metrics"]["count"] = 95
    tampered = tmp_path / original.name
    tampered.write_text(json.dumps(tampered_value), encoding="utf-8")
    evidence = dict(BUILDER["V78_GROUNDING_HELD_POINTCLOUD_EVIDENCE_SHA256"])
    del evidence[BUILDER["V78_GROUNDING_HELD_POINTCLOUD_REPORT"]]
    evidence[tampered] = _sha256(tampered)
    monkeypatch.setitem(globals_, "V78_GROUNDING_HELD_POINTCLOUD_REPORT", tampered)
    monkeypatch.setitem(
        globals_, "V78_GROUNDING_HELD_POINTCLOUD_EVIDENCE_SHA256", evidence
    )

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "contract or reproduced values differ" in result["measurement_evidence_error"]


def test_v3_approach_inspector_fails_closed_on_completion_mode_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_embodied_hybrid_approach_v3"]
    globals_ = inspector.__globals__
    original_score = ROOT / BUILDER["EMBODIED_HYBRID_APPROACH_V3_SCORE"]
    tampered_value = json.loads(original_score.read_text(encoding="utf-8"))
    tampered_value["scenes"][1]["completion_mode"] = "semantic_standoff"
    tampered = tmp_path / original_score.name
    tampered.write_text(json.dumps(tampered_value), encoding="utf-8")
    evidence = dict(BUILDER["EMBODIED_HYBRID_APPROACH_V3_EVIDENCE_SHA256"])
    del evidence[BUILDER["EMBODIED_HYBRID_APPROACH_V3_SCORE"]]
    evidence[tampered] = _sha256(tampered)
    monkeypatch.setitem(globals_, "EMBODIED_HYBRID_APPROACH_V3_SCORE", tampered)
    monkeypatch.setitem(globals_, "EMBODIED_HYBRID_APPROACH_V3_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "scene_000031" in result["measurement_evidence_error"]
