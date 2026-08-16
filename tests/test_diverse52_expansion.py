from __future__ import annotations

import pytest

from scripts.generate_scene_batch import _select_plans
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config
from semantic_3d_chat.data.scene_variants import (
    CHANGE_TYPES,
    batch_scene_plans,
    batch_scene_splits,
)

DIVERSE28_CONFIG = "configs/experiments/diverse28.yaml"
DIVERSE52_CONFIG = "configs/experiments/diverse52.yaml"
NEW_TRAIN_SCENE_IDS = [f"scene_{index:06d}" for index in range(39, 57)]
FRESH_VALIDATION_SCENE_IDS = [f"scene_{index:06d}" for index in range(57, 63)]
NEW_SCENE_IDS = [*NEW_TRAIN_SCENE_IDS, *FRESH_VALIDATION_SCENE_IDS]
FINAL_SCENE_IDS = [f"scene_{index:06d}" for index in range(25, 31)]


def test_diverse52_uses_exhausted_development_only_for_training() -> None:
    expanded = load_config(DIVERSE52_CONFIG)
    plans = batch_scene_plans(expanded)
    splits = batch_scene_splits(expanded, plans)

    assert len(plans) == 52
    assert [plan.scene_id for plan in plans] == [
        *[f"scene_{index:06d}" for index in range(11, 39)],
        *NEW_SCENE_IDS,
    ]
    assert splits == {
        "train": [
            *[f"scene_{index:06d}" for index in range(11, 25)],
            *[f"scene_{index:06d}" for index in range(31, 57)],
        ],
        "validation": FRESH_VALIDATION_SCENE_IDS,
        "test": FINAL_SCENE_IDS,
    }
    assert expanded["batch"]["expected_split_counts"] == {
        "train": 40,
        "validation": 6,
        "test": 6,
    }
    assert expanded["batch"]["deferred_splits"] == ["test"]
    assert expanded["batch"]["require_visibility_evidence"] is True


def test_diverse52_new_pairs_are_atomic_and_cover_the_intended_families() -> None:
    config = load_config(DIVERSE52_CONFIG)
    plans = batch_scene_plans(config)
    splits = batch_scene_splits(config, plans)
    assert splits is not None
    scene_to_split = {
        scene_id: split_name
        for split_name, scene_ids in splits.items()
        for scene_id in scene_ids
    }
    groups: dict[str, list] = {}
    for plan in plans:
        if plan.scene_id in NEW_SCENE_IDS:
            groups.setdefault(str(plan.pair_id), []).append(plan)

    assert set(groups) == {f"pair_{index:06d}" for index in range(19, 31)}
    assert len({members[0].seed for members in groups.values()}) == 12
    for pair_id, members in groups.items():
        assert len(members) == 2, pair_id
        assert members[0].seed == members[1].seed
        assert {member.pair_role for member in members} == {
            "reference",
            "counterfactual",
        }
        assert members[0].paired_scene_id == members[1].scene_id
        assert members[1].paired_scene_id == members[0].scene_id
        assert len({scene_to_split[member.scene_id] for member in members}) == 1

    train_families = {
        members[0].change_type
        for members in groups.values()
        if scene_to_split[members[0].scene_id] == "train"
    }
    validation_families = {
        members[0].change_type
        for members in groups.values()
        if scene_to_split[members[0].scene_id] == "validation"
    }
    assert train_families == set(CHANGE_TYPES)
    assert validation_families == {
        "book_support",
        "mirror_lr",
        "picture_support",
    }


def test_diverse52_preserves_every_inherited_scene_and_final_definition() -> None:
    baseline = load_config(DIVERSE28_CONFIG)
    expanded = load_config(DIVERSE52_CONFIG)

    for index in range(11, 39):
        scene_id = f"scene_{index:06d}"
        assert expanded["batch"]["scenes"][scene_id] == baseline["batch"]["scenes"][
            scene_id
        ]
    assert expanded["batch"]["splits"]["test"] == baseline["batch"]["splits"]["test"]


def test_diverse52_isolates_qa_and_reuses_runtime_artifact_roots() -> None:
    baseline = load_config(DIVERSE28_CONFIG)
    expanded = load_config(DIVERSE52_CONFIG)

    assert artifact_root(expanded, "qa") == PROJECT_ROOT / "data_diverse52" / "qa"
    for key in ("data_root", "features_root", "maps_root", "checkpoints_root"):
        assert expanded["paths"][key] == baseline["paths"][key]
    assert expanded["qa"]["balanced_selection"]["per_scene"] == {
        "train": 24,
        "validation": 36,
        "test": 36,
    }


def test_diverse52_default_selection_cannot_visit_deferred_final_scenes() -> None:
    config = load_config(DIVERSE52_CONFIG)
    plans = batch_scene_plans(config)

    development = _select_plans(
        config,
        plans,
        requested_scenes=None,
        requested_splits=None,
        include_deferred=False,
    )
    selected = {plan.scene_id for plan in development}
    assert len(selected) == 46
    assert selected.isdisjoint(FINAL_SCENE_IDS)
    with pytest.raises(ValueError, match="Deferred test scenes"):
        _select_plans(
            config,
            plans,
            requested_scenes=[FINAL_SCENE_IDS[0]],
            requested_splits=None,
            include_deferred=False,
        )


def test_diverse52_local_dataset_root_is_ignored() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data_diverse52/" in gitignore


def test_diverse52_contract_contains_no_deferred_unlock() -> None:
    config_text = (PROJECT_ROOT / DIVERSE52_CONFIG).read_text(encoding="utf-8")
    assert "--include-deferred-test" not in config_text
    assert "reports/gemma4" not in config_text
    assert "v55_development" not in config_text
