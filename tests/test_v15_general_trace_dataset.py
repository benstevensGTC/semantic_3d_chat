"""Contracts for the 41-room V15 generalization trace datasets.

Scaling from one training room to twenty-seven exposed a case the single-room
profiles never hit: in some procedurally generated rooms the furniture blocks
the wall-following ring from one of the sampled collision-free starts.  The
generator used to abort the whole dataset.  It may now drop that one start, but
only when the profile opts in, and only while recording exactly what was
dropped -- a silent cap would make the room count a lie.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERAL = ROOT / "data_gemma4" / "training" / "gemma_waypoint_policy_v15_general"
SEALED = ROOT / "data_gemma4" / "training" / "gemma_waypoint_policy_v15_sealed"

HISTORY_PARAMETERIZATION = "selected_action_parameters_goal_progress_v2"


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


requires_general = pytest.mark.skipif(
    not (GENERAL / "manifest.json").is_file(),
    reason="V15 general trace dataset is not generated in this checkout",
)
requires_sealed = pytest.mark.skipif(
    not (SEALED / "manifest.json").is_file(),
    reason="V15 sealed trace dataset is not generated in this checkout",
)


@requires_general
def test_general_dataset_spans_many_disjoint_rooms() -> None:
    manifest = _manifest(GENERAL)
    train = manifest["train_scene_ids"]
    validation = manifest["validation_scene_ids"]
    assert manifest["train_scene_count"] == 27
    assert manifest["validation_scene_count"] == 8
    assert not set(train) & set(validation)
    assert manifest["scene_splits_disjoint"] is True
    # The live demonstration room stays in training so the promoted checkpoint
    # remains usable for the existing Blender demo.
    assert "scene_000001" in train
    # Deferred-final scenes must never enter any split.
    assert not {f"scene_{index:06d}" for index in range(25, 31)} & set(
        train + validation
    )


@requires_general
def test_general_dataset_uses_the_runtime_history_contract() -> None:
    manifest = _manifest(GENERAL)
    assert manifest["history_parameterization"] == HISTORY_PARAMETERIZATION
    assert manifest["history_feature_dim"] == 16
    assert manifest["expert_planners_available_at_runtime"] is False
    assert manifest["oracle_inputs_at_runtime"] is False
    assert manifest["policy_selects_all_headings_and_waypoints_at_runtime"] is True


@requires_general
def test_dropped_lap_starts_are_recorded_never_silent() -> None:
    manifest = _manifest(GENERAL)
    assert manifest["skip_unroutable_lap_starts"] is True
    dropped = manifest["unroutable_lap_starts"]
    assert manifest["unroutable_lap_start_count"] == len(dropped)
    assert dropped, "the 41-room sweep is expected to hit at least one blocked ring"
    known = set(manifest["train_scene_ids"]) | set(manifest["validation_scene_ids"])
    for entry in dropped:
        assert set(entry) == {
            "scene_id",
            "split",
            "start_index",
            "start_xy_m",
            "reason",
        }
        assert entry["scene_id"] in known
        assert entry["split"] in {"train", "validation"}
        assert len(entry["start_xy_m"]) == 2
        assert entry["reason"]


@requires_general
def test_every_room_still_contributes_both_lap_directions() -> None:
    """Dropping a start must never silently drop a whole room's lap family."""

    manifest = _manifest(GENERAL)
    rooms = manifest["train_scene_count"] + manifest["validation_scene_count"]
    episodes = manifest["synthetic_variant_episode_counts"]
    assert episodes["lap_clockwise"] >= rooms
    assert episodes["lap_counterclockwise"] >= rooms
    for family in ("approach", "between", "face"):
        assert manifest["synthetic_family_episode_counts"][family] >= rooms


@requires_general
def test_stop_supervision_exists_for_every_episode() -> None:
    manifest = _manifest(GENERAL)
    # Each expert episode terminates in exactly one model-labeled STOP, so the
    # STOP class is rare but never absent -- this is what the earlier one-room
    # runs lacked enough of to learn goal completion.
    assert manifest["action_sample_counts"]["STOP"] == manifest["episode_count"]
    assert manifest["action_sample_counts"]["STOP"] > 3_000


@requires_sealed
def test_sealed_rooms_are_disjoint_from_every_general_room() -> None:
    general = _manifest(GENERAL) if (GENERAL / "manifest.json").is_file() else None
    sealed = _manifest(SEALED)
    assert sealed["validation_scene_ids"] == [
        f"scene_{index:06d}" for index in range(51, 57)
    ]
    if general is not None:
        used = set(general["train_scene_ids"]) | set(general["validation_scene_ids"])
        assert not used & set(sealed["validation_scene_ids"])
