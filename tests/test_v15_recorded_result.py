"""Bind the README's V15 claims to the artifacts that produced them.

The numbers quoted in the README are only meaningful if they cannot silently
drift from the evidence.  These tests read the produced artifacts and assert the
exact claims, including the negative ones -- an accidental improvement to the
wrong-scene control would fail here and force the prose to be rewritten rather
than quietly overstate what was measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "gemma4" / "metrics"
TRAINING = METRICS / "gemma_waypoint_policy_v15_general_training.json"
SEALED = METRICS / "gemma_waypoint_v15_sealed_controls.json"
CLOSED_LOOP = METRICS / "gemma_waypoint_v15_heldout_score.json"
SUMMARY = METRICS / "gemma_waypoint_v15_summary.json"

pytestmark = pytest.mark.skipif(
    not TRAINING.is_file(),
    reason="V15 has not been run in this checkout",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_development_rooms_beat_the_one_room_v14_control() -> None:
    controls = _read(TRAINING)["controls"]
    primary = controls["conditions"]["primary"]
    assert primary["sample_count"] == 336
    # V14 reported 12.5% action accuracy and zero STOP recall on 24 rows.
    assert primary["action_accuracy"] == pytest.approx(0.6220238, abs=1e-6)
    assert primary["stop_recall"] == pytest.approx(0.7589286, abs=1e-6)
    assert primary["action_accuracy"] > 4 * 0.125
    assert primary["stop_recall"] > 0.0


def test_sealed_rooms_are_above_chance_but_far_from_solved() -> None:
    controls = _read(SEALED)
    primary = controls["conditions"]["primary"]
    assert primary["sample_count"] == 270
    assert primary["action_accuracy"] == pytest.approx(0.5111111, abs=1e-6)
    assert primary["stop_recall"] == pytest.approx(0.5888889, abs=1e-6)
    # Balanced three-way choice: chance is 1/3.
    assert primary["action_accuracy"] > 1.0 / 3.0


def test_scene_prefix_is_load_bearing_but_not_room_specific() -> None:
    """The central V15 finding, asserted in both directions."""

    controls = _read(SEALED)
    drop = controls["accuracy_drop_from_primary"]

    # Removing the scene costs a lot: the prefix is genuinely used.
    assert drop["zero_scene_prefix"] == pytest.approx(0.1296296, abs=1e-6)
    assert drop["zero_scene_prefix"] > 0.10

    # But substituting another room's prefix, or permuting the content latents
    # while preserving their exact multiset, does not degrade the action
    # choice. If either of these ever starts biting, this test must fail so the
    # README's conclusion gets rewritten.
    assert abs(drop["wrong_scene_prefix"]) < 0.02
    assert abs(drop["shuffled_scene_prefix"]) < 0.02

    # The numeric branches are more scene-sensitive than the classifier.
    change = controls["output_change_from_primary"]
    assert change["shuffled_scene_prefix"]["mean_heading_output_shift_degrees"] > 5.0
    assert change["zero_scene_prefix"]["mean_heading_output_shift_degrees"] > 25.0


def test_closed_loop_terminates_itself_yet_misses_the_goals() -> None:
    score = _read(CLOSED_LOOP)
    assert score["goal_count"] == 30
    assert len(score["scene_ids"]) == 6
    # Gemma chose to stop on its own in nearly every episode ...
    assert score["model_selected_terminal_stop_rate"] > 0.9
    # ... and still almost never satisfied the geometric goal.
    assert score["passed_count"] == 1
    assert score["pass_rate"] < 0.05
    assert score["per_metric"]["lap_circuit"]["passed"] == 0
    assert score["per_metric"]["face_yaw"]["passed"] == 0
    # Per-step accuracy above chance therefore does not imply closed-loop
    # navigation; that gap is the result, not a measurement bug.
    assert score["rollout_process_read_oracle"] is False


def test_failure_is_not_specific_to_unseen_rooms() -> None:
    """V15 also fails in scene_000001, which is inside its own training split.

    This is what rules out "the rooms were unseen" as the explanation and points
    at grounding plus missing on-policy correction instead.
    """

    path = METRICS / "gemma_waypoint_v15_live_room_closed_loop_score.json"
    if not path.is_file():
        pytest.skip("live-room V15 closed-loop score not present")
    score = _read(path)
    assert score["scene_ids"] == ["scene_000001"]
    assert score["goal_count"] == 5
    assert score["passed_count"] == 0
    assert score["model_selected_terminal_stop_rate"] == pytest.approx(1.0)


def test_summary_is_complete_and_separates_the_two_held_out_tiers() -> None:
    summary = _read(SUMMARY)
    assert summary["complete"] is True
    assert all(summary["stages_present"].values())
    rooms = summary["rooms"]
    assert rooms["train_room_count"] == 27
    assert rooms["development_room_count"] == 8
    assert rooms["sealed_room_count"] == 6
    train = set(rooms["train_scene_ids"])
    development = set(rooms["development_scene_ids"])
    sealed = set(rooms["sealed_scene_ids"])
    assert not train & development
    assert not train & sealed
    assert not development & sealed
    assert rooms["generated_rows"] == 67_331
    assert rooms["action_sample_counts"]["STOP"] == rooms["generated_episodes"]
    contract = summary["runtime_contract"]
    assert contract["history_dim"] == 16
    assert contract["scene_token_count"] == 258
    assert contract["oracle_inputs_at_runtime"] is False
    assert contract["deterministic_route_planner_allowed_at_runtime"] is False
    assert contract["training_scene_count"] == 27


def test_readme_states_the_negative_result() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    collapsed = " ".join(readme.split())
    assert "V15: what 27 training rooms actually bought" in collapsed
    assert "only 1 of 30 goals passed its geometric threshold" in collapsed
    assert "It is the *readout* that discards it" in collapsed
    assert "neither is decoder size" in collapsed
    assert "The failure is therefore not specific to unseen rooms" in collapsed
    assert "V15 is deliberately not promoted" in collapsed
    assert "A larger Gemma is not indicated by any measurement taken here" in collapsed
    # The README must not claim unseen-room navigation works.
    for overclaim in (
        "generalizes to unseen rooms",
        "solves unseen-room navigation",
        "broad embodied intelligence",
    ):
        assert overclaim not in collapsed


def test_geometry_is_present_in_the_scene_tokens() -> None:
    """The readout is the bottleneck, not the 3D representation.

    If this ever regresses the README's conclusion is wrong: a low R^2 would
    mean the bridge really does discard geometry, which is a redesign rather
    than a readout change.
    """

    path = METRICS / "gemma_waypoint_v15_scene_token_probe.json"
    if not path.is_file():
        pytest.skip("scene-token probe has not been run in this checkout")
    probe = _read(path)
    assert probe["gemma_forward_required"] is False
    assert probe["room_count"] == 41
    assert probe["anchor_regression"]["held_out_rooms"] is True
    for value in probe["anchor_regression"]["r_squared_xyz"]:
        assert value > 0.95
    identity = probe["room_identity_decoding"]
    assert identity["linear_accuracy"] > 0.5
    assert identity["linear_accuracy"] > 10 * identity["chance_accuracy"]
    assert probe["conclusion"] == "geometry_present_in_tokens"
