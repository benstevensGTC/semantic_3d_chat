from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from semantic_3d_chat.evaluation import embodied_approach_v3_trajectories as trajectories

ROOT = Path(__file__).resolve().parents[1]


def _root_cases() -> tuple[trajectories.SealedCase, ...]:
    return tuple(
        trajectories.SealedCase(
            scene_id=case.scene_id,
            path=ROOT / case.path,
            sha256=case.sha256,
            completion_mode=case.completion_mode,
        )
        for case in trajectories.DEFAULT_CASES
    )


def test_two_hash_pinned_runtime_trajectories_cover_both_completion_modes() -> None:
    scenes = []
    for case in _root_cases():
        assert trajectories.file_sha256(case.path) == case.sha256
        payload = trajectories.load_sealed_result(case)
        scenes.append(trajectories.extract_trajectory(case, payload))

    by_scene = {scene["scene_id"]: scene for scene in scenes}
    scene_one = by_scene["scene_000001"]
    scene_thirty_one = by_scene["scene_000031"]

    assert scene_one["completion_mode"] == "semantic_standoff"
    assert scene_one["semantic_standoff_satisfied"] is True
    assert scene_one["collision_limited_completion"] is False
    assert scene_one["final_continuous_target_distance_m"] == pytest.approx(0.4819971733403751)
    assert scene_one["net_displacement_m"] == pytest.approx(0.7003040825966198)

    assert scene_thirty_one["completion_mode"] == "collision_limited_safe_stop"
    assert scene_thirty_one["semantic_standoff_satisfied"] is False
    assert scene_thirty_one["collision_limited_completion"] is True
    assert scene_thirty_one["final_continuous_target_distance_m"] == pytest.approx(
        0.7627565297837082
    )
    assert scene_thirty_one["net_displacement_m"] == pytest.approx(1.2867964024999163)
    clipped = scene_thirty_one["steps"][-2]
    assert clipped["interlock_reason"] == "collision_limited_safe_progress"
    assert clipped["collision_limited"]["executed_safe_distance_m"] == pytest.approx(
        0.06655513154053894
    )
    assert scene_thirty_one["steps"][-1]["collision_limited"]["safe_closest_reachable"] is True
    assert all(scene["all_map_voxels_scored_every_step"] for scene in scenes)
    assert all(scene["collision_count"] == 0 for scene in scenes)


def test_generated_figure_and_summary_are_deterministic_and_runtime_only(
    tmp_path: Path,
) -> None:
    first = trajectories.generate(_root_cases(), tmp_path / "first.png", tmp_path / "first.json")
    second = trajectories.generate(_root_cases(), tmp_path / "second.png", tmp_path / "second.json")

    assert first["figure"]["sha256"] == second["figure"]["sha256"]
    assert first["sources_sha256"] == second["sources_sha256"]
    with Image.open(tmp_path / "first.png") as image:
        assert image.format == "PNG"
        assert image.size == (2240, 960)
    scope = first["scope"]
    assert scope["runtime_result_files_loaded"] == 2
    assert scope["source_hashes_preserved"] is True
    assert scope["oracle_files_loaded"] is False
    assert scope["qa_files_loaded"] is False
    assert scope["scene_metadata_files_loaded"] is False
    assert scope["semantic_map_files_loaded"] is False
    assert scope["model_files_loaded"] is False
    assert scope["continuous_target_xyz_only"] is True


def test_modified_result_is_rejected_before_extraction(tmp_path: Path) -> None:
    original = _root_cases()[0]
    modified = tmp_path / "modified.json"
    modified.write_bytes(original.path.read_bytes() + b" ")
    case = trajectories.SealedCase(
        scene_id=original.scene_id,
        path=modified,
        sha256=original.sha256,
        completion_mode=original.completion_mode,
    )

    with pytest.raises(ValueError, match="embodied result digest differs"):
        trajectories.load_sealed_result(case)


def test_default_artifacts_and_make_target_are_reproducible() -> None:
    output = ROOT / trajectories.DEFAULT_OUTPUT
    figure = ROOT / trajectories.DEFAULT_FIGURE
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["figure"]["sha256"] == trajectories.file_sha256(figure)
    assert artifact["sources_sha256"] == {
        case.scene_id: case.sha256 for case in trajectories.DEFAULT_CASES
    }
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "embodied-approach-v3-trajectories:" in makefile
    assert "-m semantic_3d_chat.evaluation.embodied_approach_v3_trajectories" in makefile
