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


def test_current_summary_authenticates_v75_post_hoc_figures(
    summary: dict[str, Any],
) -> None:
    evidence = summary["v75_official_validation_figures"]

    assert evidence["status"] == ("authenticated_post_hoc_visualization_only_not_new_evaluation")
    assert evidence["evidence_authenticated"] is True
    assert evidence["post_hoc_visualization_only"] is True
    assert evidence["new_evaluation"] is False
    assert evidence["source_file_count"] == 1
    assert evidence["source_score_sha256"] == (
        "f6d9ceea78622c3a4851c3366ac06ed0835824f7b786424725fbfa5d5b978679"
    )
    assert evidence["model_loaded"] is False
    assert evidence["predictions_or_references_loaded"] is False
    assert evidence["scene_map_loaded"] is False
    assert evidence["qa_or_oracle_loaded"] is False
    assert evidence["unopened_split_loaded"] is False
    assert evidence["per_example_grounding_errors_available"] is False
    assert evidence["grounding_visualization"] == ("aggregate_summary_only_no_distribution")
    assert set(evidence["figures"]) == set(BUILDER["V75_OFFICIAL_VALIDATION_FIGURES"])
    for path, digest in BUILDER["V75_OFFICIAL_VALIDATION_FIGURE_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_summary_authenticates_bounded_v78_internal_figure(
    summary: dict[str, Any],
) -> None:
    evidence = summary["v78_grounding_internal_held_figure"]

    assert evidence["status"] == (
        "authenticated_historical_internal_held_post_hoc_figure_not_official_not_promoted"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["historical_internal_held_only"] is True
    assert evidence["new_evaluation"] is False
    assert evidence["official_validation"] is False
    assert evidence["runtime_evidence"] is False
    assert evidence["promotion_evidence"] is False
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["report_sha256"] == (
        "557cc497dd12bd74f45cecd3624e18649ddc548af091ed719c22c7998942b84b"
    )
    conditions = evidence["conditions"]
    assert conditions["historical_internal_held"]["mean_coordinate_error_m"] == (
        pytest.approx(0.5270832777023315)
    )
    assert conditions["paired_wrong_scene"]["mean_coordinate_error_m"] == (
        pytest.approx(0.5337055921554565)
    )
    assert evidence["paired_scene_causal"] == {
        "changed_target_sides": 10,
        "correct_scene_closer_to_original_fraction": 0.9,
        "paired_scene_follows_paired_target_fraction": 0.5,
    }
    for path, digest in BUILDER["V78_GROUNDING_INTERNAL_HELD_FIGURE_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_summary_authenticates_optional_v78_runtime_and_leakage(
    summary: dict[str, Any],
) -> None:
    evidence = summary["v78_grounding_runtime"]

    assert evidence["status"] == (
        "authenticated_optional_internal_diagnostic_runtime_not_promoted"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["checkpoint_inventory"] == ["grounding.safetensors", "metadata.json"]
    assert evidence["weights_sha256"] == (
        "3c7914a61e63d80617e7fcfca122e02eec30d15af5a43e910daa0cd6c0b501c4"
    )
    assert evidence["metadata_sha256"] == (
        "ea5536dc078b7707000404661c92bdb198dc0c40bbf73cf987d5f94b20464480"
    )
    assert evidence["strict_leakage_sha256"] == (
        "070feddd71141dfa75f8ca807ec47275225e8b056fce1e2f3862f88e89fc6215"
    )
    assert evidence["all_scene_tokens_scored"] is True
    assert evidence["answer_generation_unchanged"] is True
    assert evidence["official_validation_evidence"] is False
    assert evidence["runtime_promotion_authorized"] is False
    demo = evidence["static_real_demo"]
    assert demo["passed"] is True
    assert demo["loaded_file_count"] == 4_204
    assert demo["forbidden_access_count"] == 0
    assert demo["oracle_unavailable_during_inference"] is True
    assert demo["base_prefix_invariant"] is True
    assert demo["grounding_scene_tokens_invariant"] is True
    assert [row["v75_answer"] for row in demo["answers"]] == ["right", "red", "cube"]
    assert demo["answer_quality_claimed"] is False
    embodied = evidence["embodied_integration"]
    assert embodied["wiring_authenticated"] is True
    assert embodied["preflight_target_present"] is True
    assert embodied["finite_live_target_present"] is True
    assert embodied["sealed_v78_embodied_live_result_present"] is False
    assert all(embodied["checks"].values())
    for path, digest in BUILDER["V78_GROUNDING_RUNTIME_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_summary_authenticates_terminal_v79_screen(
    summary: dict[str, Any],
) -> None:
    evidence = summary["v79_relation_counterfactual"]

    assert evidence["status"] == (
        "authenticated_historical_scene_disjoint_screen_failed_no_promotion"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["candidate_sha256"] == (
        "bdd4b6f40a8a3ec99dd9cb7b3134caea0feb6c8c9e7298824ac33721aad877af"
    )
    assert evidence["training_report_sha256"] == (
        "2a1631e124c176184b7d8a80af1f720973d1f2daca1431cd570c790eda50e170"
    )
    assert evidence["screen_report_sha256"] == (
        "e4cadb61155ed6a3fb997b91a11106c5a119e6b44eb168bed361e0db75032394"
    )
    assert evidence["training"]["selected_rows"] == 120
    assert evidence["training"]["changed_sides"] == 48
    assert evidence["training"]["optimizer_steps"] == 15
    screen = evidence["screen"]
    assert screen["row_count"] == 28
    assert screen["unit_count"] == 14
    assert screen["screen_passed"] is False
    assert screen["failed_gate"] == "prediction_changing_units_at_least_best_baseline"
    assert screen["full_384_evaluation_authorized"] is False
    assert screen["runtime_promotion_authorized"] is False
    assert {
        name: (
            row["correct_scene"]["correct"],
            row["wrong_scene"]["correct"],
            row["correct_minus_wrong_count"],
            row["correct_scene_prediction_changing_units"],
        )
        for name, row in screen["summaries"].items()
    } == {
        "v75": (18, 5, 13, 9),
        "v77": (19, 5, 14, 10),
        "v79": (20, 5, 15, 9),
    }
    assert evidence["runtime_promotion_authorized"] is False
    assert all(evidence["training_checks"].values())
    assert all(evidence["screen_checks"].values())
    for path, digest in BUILDER["V79_RELATION_COUNTERFACTUAL_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_report_embeds_bounded_validation_figures(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    for record in summary["v75_official_validation_figures"]["figures"].values():
        assert f"]({record['report_link']})" in markdown
    assert "Post-hoc visualization of the already-sealed V75" in markdown
    assert "not a new evaluation" in markdown
    assert "no per-example errors, so no distribution is inferred" in collapsed

    v78 = summary["v78_grounding_internal_held_figure"]
    assert f"]({v78['figure']['report_link']})" in markdown
    assert "historical training-pool, pair- and scene-disjoint internal-held" in collapsed
    assert "not official validation" in collapsed
    assert "not authorized for promotion" in collapsed
    assert "paired-wrong-scene aggregate was nearly unchanged" in collapsed
    assert "because many rows do not move their target" in collapsed
    assert "not presented as a strong global causal control" in collapsed
    assert "Position/question shuffles and zero scene are the stronger controls" in collapsed
    assert "optional V78 numeric-grounding runtime" in collapsed
    assert "4,204 audited reads with zero forbidden accesses" in collapsed
    assert "two location questions still received V75's weak" in collapsed
    assert "no sealed V78-specific embodied transcript" in collapsed
    assert "V79 is a separately authenticated, terminal" in collapsed
    assert "correct-minus-wrong gaps of 13, 14, and 15" in collapsed
    assert "full 384-row evaluation and runtime publication were correctly blocked" in collapsed


@pytest.mark.parametrize(
    ("inspector_name", "evidence_name", "manifest_name"),
    (
        (
            "_inspect_v75_official_validation_figures",
            "V75_OFFICIAL_VALIDATION_FIGURE_EVIDENCE_SHA256",
            "V75_OFFICIAL_VALIDATION_FIGURE_MANIFEST",
        ),
        (
            "_inspect_v78_grounding_internal_held_figure",
            "V78_GROUNDING_INTERNAL_HELD_FIGURE_EVIDENCE_SHA256",
            "V78_GROUNDING_INTERNAL_HELD_FIGURE_MANIFEST",
        ),
    ),
)
def test_figure_authentication_fails_closed_on_manifest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inspector_name: str,
    evidence_name: str,
    manifest_name: str,
) -> None:
    inspector = BUILDER[inspector_name]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER[manifest_name]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    evidence = dict(BUILDER[evidence_name])
    del evidence[BUILDER[manifest_name]]
    evidence[tampered] = BUILDER[evidence_name][BUILDER[manifest_name]]
    monkeypatch.setitem(globals_, evidence_name, evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "digest differs" in result["measurement_evidence_error"]
