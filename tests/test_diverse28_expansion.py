from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_scene_batch import _select_plans
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config
from semantic_3d_chat.data.qa_generator import validate_exact_visibility_files
from semantic_3d_chat.data.scene_variants import batch_scene_plans, batch_scene_splits

DIVERSE20_CONFIG = "configs/experiments/diverse20.yaml"
DIVERSE28_CONFIG = "configs/experiments/diverse28.yaml"
NEW_SCENE_IDS = [f"scene_{index:06d}" for index in range(31, 39)]
FINAL_SCENE_IDS = [f"scene_{index:06d}" for index in range(25, 31)]


def test_diverse28_adds_only_four_atomic_training_pairs() -> None:
    baseline = load_config(DIVERSE20_CONFIG)
    expanded = load_config(DIVERSE28_CONFIG)
    plans = batch_scene_plans(expanded)
    splits = batch_scene_splits(expanded, plans)

    assert len(plans) == 28
    assert [plan.scene_id for plan in plans] == [
        *[f"scene_{index:06d}" for index in range(11, 31)],
        *NEW_SCENE_IDS,
    ]
    assert splits == {
        "train": [
            *[f"scene_{index:06d}" for index in range(11, 19)],
            *NEW_SCENE_IDS,
        ],
        "validation": [f"scene_{index:06d}" for index in range(19, 25)],
        "test": FINAL_SCENE_IDS,
    }
    assert expanded["batch"]["deferred_splits"] == ["test"]
    assert expanded["batch"]["require_visibility_evidence"] is True
    assert expanded["qa"]["balanced_selection"] == baseline["qa"]["balanced_selection"]

    scene_to_split = {
        scene_id: split_name for split_name, scene_ids in splits.items() for scene_id in scene_ids
    }
    new_plans = {plan.scene_id: plan for plan in plans if plan.scene_id in NEW_SCENE_IDS}
    groups: dict[str, list] = {}
    for plan in new_plans.values():
        groups.setdefault(str(plan.pair_id), []).append(plan)
    assert set(groups) == {
        "pair_000015",
        "pair_000016",
        "pair_000017",
        "pair_000018",
    }
    assert {members[0].change_type for members in groups.values()} == {
        "book_support",
        "mirror_lr",
        "picture_support",
        "object_removal",
    }
    assert len({members[0].seed for members in groups.values()}) == 4
    for members in groups.values():
        assert len(members) == 2
        assert members[0].seed == members[1].seed
        assert {member.pair_role for member in members} == {
            "reference",
            "counterfactual",
        }
        assert {scene_to_split[member.scene_id] for member in members} == {"train"}


def test_diverse28_preserves_existing_and_final_scene_definitions_exactly() -> None:
    baseline = load_config(DIVERSE20_CONFIG)
    expanded = load_config(DIVERSE28_CONFIG)

    for index in range(11, 31):
        scene_id = f"scene_{index:06d}"
        assert expanded["batch"]["scenes"][scene_id] == baseline["batch"]["scenes"][scene_id]
    assert expanded["batch"]["splits"]["validation"] == baseline["batch"]["splits"]["validation"]
    assert expanded["batch"]["splits"]["test"] == baseline["batch"]["splits"]["test"]


def test_diverse28_isolates_only_qa_and_reuses_existing_runtime_artifact_roots() -> None:
    baseline = load_config(DIVERSE20_CONFIG)
    expanded = load_config(DIVERSE28_CONFIG)

    assert artifact_root(expanded, "qa") == PROJECT_ROOT / "data_diverse28" / "qa"
    for key in ("data_root", "features_root", "maps_root", "checkpoints_root"):
        assert expanded["paths"][key] == baseline["paths"][key]


def test_diverse28_default_selection_excludes_all_final_scenes() -> None:
    config = load_config(DIVERSE28_CONFIG)
    plans = batch_scene_plans(config)

    development = _select_plans(
        config,
        plans,
        requested_scenes=None,
        requested_splits=None,
        include_deferred=False,
    )
    assert [plan.scene_id for plan in development] == [
        *[f"scene_{index:06d}" for index in range(11, 25)],
        *NEW_SCENE_IDS,
    ]
    assert not {plan.scene_id for plan in development} & set(FINAL_SCENE_IDS)
    with pytest.raises(ValueError, match="Deferred test scenes"):
        _select_plans(
            config,
            plans,
            requested_scenes=[FINAL_SCENE_IDS[0]],
            requested_splits=None,
            include_deferred=False,
        )


def _visibility(scene_id: str) -> dict:
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "method": "exact_depth_raycast",
        "minimum_visible_pixels": 1,
        "expected_instance_ids": ["i_000100", "i_000104"],
        "visible_pixel_counts": {"i_000100": 91, "i_000104": 7},
        "all_required_visible": True,
    }


def test_qa_visibility_gate_accepts_only_exact_complete_evidence(tmp_path: Path) -> None:
    oracle_root = tmp_path / "oracle"
    for scene_id in ("scene_000031", "scene_000032"):
        directory = oracle_root / scene_id
        directory.mkdir(parents=True)
        (directory / "visibility.json").write_text(
            json.dumps(_visibility(scene_id)), encoding="utf-8"
        )

    assert validate_exact_visibility_files(oracle_root, ["scene_000031", "scene_000032"]) == {
        "scene_000031": {"i_000100": 91, "i_000104": 7},
        "scene_000032": {"i_000100": 91, "i_000104": 7},
    }

    (oracle_root / "scene_000032" / "visibility.json").unlink()
    with pytest.raises(FileNotFoundError, match="scene_000032"):
        validate_exact_visibility_files(oracle_root, ["scene_000031", "scene_000032"])


def test_qa_visibility_gate_rejects_scene_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "oracle" / "scene_000031"
    directory.mkdir(parents=True)
    (directory / "visibility.json").write_text(
        json.dumps(_visibility("scene_000032")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="scene mismatch"):
        validate_exact_visibility_files(tmp_path / "oracle", ["scene_000031"])


def test_diverse28_make_and_docs_commands_never_unlock_final_test() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    for target in (
        "diverse28-dry-run",
        "diverse28-generate-expansion",
        "diverse28-render-expansion",
        "diverse28-generate-dataset",
    ):
        assert f"{target}:" in makefile
        assert f"make {target}" in readme
    diverse28_make_region = makefile[
        makefile.index("diverse28-dry-run:") : makefile.index("build-smoke-map:")
    ]
    assert "--include-deferred-test" not in diverse28_make_region
    assert "data_diverse28/" in gitignore
