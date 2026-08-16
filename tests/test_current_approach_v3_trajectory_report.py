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


def test_current_summary_authenticates_exact_v3_trajectory_evidence(
    summary: dict[str, Any],
) -> None:
    evidence = summary["embodied_approach_v3_trajectories"]

    assert evidence["status"] == ("authenticated_runtime_only_v3_approach_trajectory_visualization")
    assert evidence["evidence_authenticated"] is True
    assert evidence["post_hoc_visualization_only"] is True
    assert evidence["new_inference"] is False
    assert evidence["runtime_result_files_only"] is True
    assert evidence["source_file_count"] == 2
    assert evidence["source_hashes_preserved"] is True
    assert evidence["oracle_files_opened"] is False
    assert evidence["qa_files_opened"] is False
    assert evidence["scene_metadata_files_opened"] is False
    assert evidence["semantic_map_files_opened"] is False
    assert evidence["model_files_opened"] is False
    assert evidence["environmental_text_inputs"] == []
    assert evidence["figure"] == {
        "height_px": 960,
        "path": "reports/gemma4/figures/embodied_approach_v3_trajectories.png",
        "report_link": "gemma4/figures/embodied_approach_v3_trajectories.png",
        "sha256": ("6bbe03c6dbd847469baff427121e5e3d01f0ead4899f8773dade6a3a561178a2"),
        "width_px": 2240,
    }
    assert evidence["machine_summary"] == {
        "path": "reports/gemma4/examples/embodied_approach_v3_trajectories.json",
        "sha256": ("2b1482c0364ac72fa912df8222714aefbd0d1a90d62c18c1b28a880b91acc72a"),
    }
    assert evidence["scenes"][0]["completion_mode"] == "semantic_standoff"
    assert evidence["scenes"][0]["semantic_standoff_satisfied"] is True
    assert evidence["scenes"][0]["final_continuous_target_distance_m"] == (
        pytest.approx(0.4819971733403751)
    )
    scene_31 = evidence["scenes"][1]
    assert scene_31["completion_mode"] == "collision_limited_safe_stop"
    assert scene_31["semantic_standoff_satisfied"] is False
    assert scene_31["collision_limited_completion"] is True
    assert scene_31["final_continuous_target_distance_m"] == pytest.approx(0.7627565297837082)
    assert scene_31["net_displacement_m"] == pytest.approx(1.2867964024999163)

    for path, digest in BUILDER["EMBODIED_APPROACH_V3_TRAJECTORY_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_markdown_distinguishes_both_v3_trajectory_stop_modes(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "hard-hash-authenticated, post-hoc runtime-only visualization" in collapsed
    assert "opened exactly two runtime result JSON files" in collapsed
    assert "Scene 1 shows ordinary `semantic_standoff` completion" in collapsed
    assert "Scene 31 shows the distinct `collision_limited_safe_stop` path" in collapsed
    assert "closest-safe collision-limited stop" in collapsed
    assert "not ordinary 0.5 m semantic-standoff success" in collapsed
    assert "6bbe03c6dbd847469baff427121e5e3d01f0ead4899f8773dade6a3a561178a2" in markdown
    assert "](gemma4/figures/embodied_approach_v3_trajectories.png)" in markdown


def test_v3_trajectory_inspector_fails_closed_on_figure_digest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_embodied_approach_v3_trajectories"]
    globals_ = inspector.__globals__
    original_figure = ROOT / BUILDER["EMBODIED_APPROACH_V3_TRAJECTORY_FIGURE"]
    tampered = tmp_path / original_figure.name
    tampered.write_bytes(original_figure.read_bytes() + b"tamper")
    evidence = {}
    for path, digest in BUILDER["EMBODIED_APPROACH_V3_TRAJECTORY_EVIDENCE_SHA256"].items():
        evidence[
            tampered if path == BUILDER["EMBODIED_APPROACH_V3_TRAJECTORY_FIGURE"] else path
        ] = digest
    monkeypatch.setitem(globals_, "EMBODIED_APPROACH_V3_TRAJECTORY_FIGURE", tampered)
    monkeypatch.setitem(globals_, "EMBODIED_APPROACH_V3_TRAJECTORY_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "evidence digest differs" in result["measurement_evidence_error"]


def test_v3_trajectory_inspector_fails_closed_on_rehashed_stop_mode_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_embodied_approach_v3_trajectories"]
    globals_ = inspector.__globals__
    original_summary = ROOT / BUILDER["EMBODIED_APPROACH_V3_TRAJECTORY_SUMMARY"]
    tampered_value = json.loads(original_summary.read_text(encoding="utf-8"))
    tampered_value["scenes"][1]["semantic_standoff_satisfied"] = True
    tampered = tmp_path / original_summary.name
    tampered.write_text(json.dumps(tampered_value), encoding="utf-8")
    evidence = {}
    for path, digest in BUILDER["EMBODIED_APPROACH_V3_TRAJECTORY_EVIDENCE_SHA256"].items():
        if path == BUILDER["EMBODIED_APPROACH_V3_TRAJECTORY_SUMMARY"]:
            evidence[tampered] = _sha256(tampered)
        else:
            evidence[path] = digest
    monkeypatch.setitem(globals_, "EMBODIED_APPROACH_V3_TRAJECTORY_SUMMARY", tampered)
    monkeypatch.setitem(globals_, "EMBODIED_APPROACH_V3_TRAJECTORY_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "scene_000031" in result["measurement_evidence_error"]
