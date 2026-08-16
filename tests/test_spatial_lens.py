"""Contracts for the zero-training spatial-reasoning stack.

None of these load Gemma.  They cover the properties that decide whether the
pipeline's claims mean anything: that the author's words never reach perception,
that object discovery is geometric rather than looked up, and that the rover's
body refuses illegal motion instead of quietly correcting the model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from semantic_3d_chat.spatial_lens.discover import discover_objects
from semantic_3d_chat.spatial_lens.naming import (
    color_word,
    disambiguate,
    normalize_answer,
    vote,
)
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.reasoning import (
    RoverPose,
    heading_to,
    navigation_prompt,
    parse_decision,
)
from semantic_3d_chat.spatial_lens.room_spec import parse_room_spec
from semantic_3d_chat.spatial_lens.rover import Rover, choose_start_pose, probe_directions
from semantic_3d_chat.spatial_lens.scene_graph import (
    SceneGraph,
    SceneObject,
    build_free_grid,
)

ROOM = {
    "name": "unit_room",
    "size_m": [6.0, 5.0, 2.8],
    "objects": [
        {"name": "dining table", "shape": "table", "color": "wood", "position_m": [1.0, 1.0]},
        {"name": "reading chair", "shape": "chair", "color": "charcoal", "position_m": [1.0, -0.5]},
        {"name": "red ball", "shape": "sphere", "color": "red", "position_m": [-2.0, 1.5],
         "size_m": [0.3, 0.3, 0.3]},
    ],
}


# --------------------------------------------------------------- authoring
def test_build_payload_carries_no_author_words() -> None:
    """The whole experiment is void if Blender's input names the furniture."""

    spec = parse_room_spec(ROOM)
    serialized = json.dumps(spec.build_payload()).lower()
    for word in ("table", "chair", "ball", "dining", "reading", "red"):
        assert word not in serialized
    # ... while the scoring key keeps exactly that information.
    key = json.dumps(spec.key_payload()).lower()
    assert "dining table" in key and "red ball" in key


def test_composites_expand_into_primitives() -> None:
    spec = parse_room_spec(ROOM)
    by_name = {item.name: item for item in spec.objects}
    assert len(by_name["dining table"].parts) == 5  # top plus four legs
    assert len(by_name["reading chair"].parts) == 6  # seat, back, four legs
    assert len(by_name["red ball"].parts) == 1
    for item in spec.objects:
        for part in item.parts:
            assert part.kind in {"box", "cylinder", "sphere"}
            assert all(value > 0 for value in part.size_m)


def test_objects_must_fit_inside_the_room() -> None:
    outside = {**ROOM, "objects": [{**ROOM["objects"][0], "position_m": [2.9, 1.0]}]}
    with pytest.raises(ValueError, match="outside the room"):
        parse_room_spec(outside)


def test_overlapping_furniture_is_rejected() -> None:
    stacked = {
        **ROOM,
        "objects": [ROOM["objects"][0], {**ROOM["objects"][1], "position_m": [1.0, 1.0]}],
    }
    with pytest.raises(ValueError, match="overlap"):
        parse_room_spec(stacked)


# --------------------------------------------------------------- discovery
def _synthetic_cloud() -> SemanticCloud:
    """A room shell plus two separated blocks, on an exact voxel lattice.

    Real clouds come out of the fuser already snapped to voxel centres, so the
    fixture is built the same way: float ``arange`` spacing would silently skip
    lattice cells and split a solid block into disconnected slabs.
    """

    voxel = 0.05
    cells: set[tuple[int, int, int]] = set()

    def add(x: float, y: float, z: float) -> None:
        cells.add(
            (
                math.floor(round(x / voxel, 6)),
                math.floor(round(y / voxel, 6)),
                math.floor(round(z / voxel, 6)),
            )
        )

    # floor and one wall, which discovery must strip
    for ix in range(-58, 58, 2):
        for iy in range(-48, 48, 2):
            add(ix * voxel, iy * voxel, 0.02)
    for iy in range(-48, 48, 2):
        for iz in range(2, 50, 2):
            add(2.95, iy * voxel, iz * voxel)
    # two clearly separated solid blocks
    for cx, cy in ((-1.5, 0.0), (1.0, 1.2)):
        for ix in range(-5, 5):
            for iy in range(-5, 5):
                for iz in range(2, 14):
                    add(cx + ix * voxel, cy + iy * voxel, iz * voxel)

    # float64: at these magnitudes float32 rounding pushes a centre across a
    # lattice boundary, which would fragment a solid block into slabs.
    array = np.asarray(
        [[(i + 0.5) * voxel, (j + 0.5) * voxel, (k + 0.5) * voxel] for i, j, k in sorted(cells)],
        dtype=np.float64,
    )
    return SemanticCloud(
        centers_m=array,
        rgb=np.full((len(array), 3), 0.5, dtype=np.float32),
        features=np.zeros((len(array), 4), dtype=np.float16),
        counts=np.ones(len(array), dtype=np.int32),
        voxel_size_m=voxel,
        room_size_m=(6.0, 5.0, 2.8),
    )


def test_discovery_finds_the_blocks_and_drops_the_shell() -> None:
    proposals = discover_objects(_synthetic_cloud(), min_voxels=20)
    assert len(proposals) == 2
    centers = sorted(round(item.center_m[0], 1) for item in proposals)
    assert centers == [-1.5, 1.0]
    for item in proposals:
        # A shell point would blow the footprint up to room scale.
        assert item.footprint_m[0] < 1.0
        assert item.height_m > 0.3


# ------------------------------------------------------------------ naming
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Chair", "chair"),
        ("  A wooden Table.  ", "wooden table"),
        ("This is a bookshelf", "bookshelf"),
        ("**Floor lamp**", "floor lamp"),
        ("", ""),
    ],
)
def test_answer_normalization(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_vote_ignores_uninformative_answers() -> None:
    name, counts = vote(["shape", "chair", "chair", "object"])
    assert name == "chair"
    assert counts == {"chair": 2}
    assert vote(["shape", "object"])[0] == "unidentified object"


def test_colour_naming_uses_hue_not_brightness() -> None:
    assert color_word((0.52, 0.74, 0.74)) == "teal"
    assert color_word((0.82, 0.35, 0.30)) == "red"
    assert color_word((0.58, 0.39, 0.21)) == "brown"
    assert color_word((0.35, 0.38, 0.39)) == "gray"


def test_duplicate_names_become_addressable() -> None:
    result = disambiguate(["table", "table"], [(0.75, 0.62, 0.52), (0.52, 0.74, 0.74)])
    assert len(set(result)) == 2
    assert all("table" in name for name in result)


# ------------------------------------------------------------- scene graph
def _graph() -> SceneGraph:
    obstacle = SceneObject(
        object_id="p_001",
        name="tan table",
        center_m=(0.0, 0.0, 0.4),
        bbox_min_m=(-0.6, -0.4, 0.0),
        bbox_max_m=(0.6, 0.4, 0.8),
        mean_rgb=(0.7, 0.6, 0.5),
        voxel_count=500,
        name_confidence=1.0,
    )
    rug = SceneObject(
        object_id="p_002",
        name="rug",
        center_m=(2.0, 0.0, 0.01),
        bbox_min_m=(1.5, -0.5, 0.0),
        bbox_max_m=(2.5, 0.5, 0.02),
        mean_rgb=(0.3, 0.3, 0.6),
        voxel_count=200,
        name_confidence=1.0,
    )
    grid = build_free_grid(
        [obstacle, rug],
        (6.0, 5.0, 2.8),
        resolution_m=0.05,
        rover_radius_m=0.18,
        ignore_height_m=0.06,
    )
    return SceneGraph(
        room="unit",
        room_size_m=(6.0, 5.0, 2.8),
        objects=(obstacle, rug),
        free_grid=grid,
        grid_resolution_m=0.05,
        rover_radius_m=0.18,
    )


def test_free_grid_blocks_tall_objects_but_not_flat_ones() -> None:
    graph = _graph()
    assert not graph.is_free(0.0, 0.0)  # inside the table
    assert graph.is_free(2.0, 0.0)  # a rug is drivable
    assert not graph.is_free(2.95, 0.0)  # too close to the wall for the body
    assert graph.is_free(-2.0, -2.0)


def test_approach_point_is_free_floor_next_to_the_object() -> None:
    graph = _graph()
    table = graph.find("tan table")
    assert table is not None
    point = graph.approach_point(table)
    assert point is not None
    assert graph.is_free(*point)
    gap = max(
        abs(point[0]) - 0.6,
        abs(point[1]) - 0.4,
    )
    assert gap < 0.35  # adjacent, not across the room


def test_lookup_is_forgiving_about_perceived_names() -> None:
    graph = _graph()
    assert graph.find("tan table") is not None
    assert graph.find("table") is not None
    assert graph.find("spaceship") is None


# ------------------------------------------------------------------- rover
def test_rover_refuses_illegal_motion_without_correcting_it() -> None:
    graph = _graph()
    rover = Rover(graph=graph, pose=RoverPose(-2.0, 0.0, 0.0), max_step_m=0.5)

    accepted, code, _ = rover.move_to(2.0, 0.0)
    assert (accepted, code) == (False, "E_STEP_TOO_LONG")
    assert rover.pose.x_m == -2.0, "a rejected move must not shift the rover"

    accepted, code, _ = rover.move_to(0.0, 0.0)
    assert (accepted, code) == (False, "E_STEP_TOO_LONG")

    accepted, code, distance = rover.move_to(-1.6, 0.0)
    assert accepted and code is None
    assert distance == pytest.approx(0.4)
    assert rover.pose.x_m == pytest.approx(-1.6)


def test_move_toward_truncates_but_never_reroutes() -> None:
    graph = _graph()
    rover = Rover(graph=graph, pose=RoverPose(-2.5, 0.0, 0.0), max_step_m=0.5)
    accepted, _code, distance = rover.move_toward(-0.5, 0.0)
    assert accepted
    assert distance == pytest.approx(0.5, abs=1e-6)
    assert rover.pose.x_m == pytest.approx(-2.0)
    assert rover.pose.y_m == pytest.approx(0.0), "no lateral detour is invented"

    # Straight into the table: refused rather than steered around.
    blocked = Rover(graph=graph, pose=RoverPose(-0.9, 0.0, 0.0), max_step_m=0.5)
    accepted, code, _ = blocked.move_toward(0.0, 0.0)
    assert (accepted, code) == (False, "E_PATH_BLOCKED")
    assert blocked.pose.x_m == pytest.approx(-0.9)


def test_direction_probe_reports_both_clear_and_blocked() -> None:
    graph = _graph()
    rover = Rover(graph=graph, pose=RoverPose(-1.0, 0.0, 0.0), max_step_m=0.5)
    probes = probe_directions(rover)
    assert len(probes) == 8
    assert any(clear for _yaw, _point, clear in probes)
    assert any(not clear for _yaw, _point, clear in probes)
    for _yaw, point, clear in probes:
        if clear:
            assert graph.is_free(*point)


def test_start_pose_is_on_free_floor() -> None:
    graph = _graph()
    pose = choose_start_pose(graph)
    assert graph.is_free(pose.x_m, pose.y_m)


# --------------------------------------------------------------- reasoning
def test_heading_convention_matches_the_executor() -> None:
    assert heading_to((0.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert heading_to((0.0, 0.0), (1.0, 0.0)) == pytest.approx(-90.0)
    assert heading_to((0.0, 0.0), (-1.0, 0.0)) == pytest.approx(90.0)
    assert abs(heading_to((0.0, 0.0), (0.0, -1.0))) == pytest.approx(180.0)


@pytest.mark.parametrize(
    "reply",
    [
        '{"reasoning": "go", "action": "MOVE_TOWARD", "x": 1.0, "y": 2.0}',
        'Sure! ```json\n{"action":"MOVE_TOWARD","x":1.0,"y":2.0,"reasoning":"go"}\n```',
    ],
)
def test_decision_parsing_tolerates_chatter(reply: str) -> None:
    decision = parse_decision(reply)
    assert decision.action == "MOVE_TOWARD"
    assert (decision.x_m, decision.y_m) == (1.0, 2.0)


def test_decision_parsing_rejects_unknown_actions() -> None:
    with pytest.raises(ValueError, match="Unsupported action"):
        parse_decision('{"action": "TELEPORT", "x": 1, "y": 1}')
    with pytest.raises(ValueError, match="No JSON"):
        parse_decision("I would move to the left.")


def test_navigation_prompt_states_geometry_the_model_cannot_derive() -> None:
    graph = _graph()
    pose = RoverPose(-2.0, 0.0, 0.0)
    prompt = navigation_prompt(
        graph,
        pose,
        "reach the table",
        [],
        max_step_m=0.5,
        approach_points={"tan table": (-0.85, 0.0)},
        open_directions=probe_directions(Rover(graph=graph, pose=pose)),
    )
    assert "2.00 m away" in prompt  # distance to the table centre
    assert "stand at (-0.85, +0.00)" in prompt
    assert "blocked" in prompt or "clear" in prompt
    assert "reach the table" in prompt


# ---------------------------------------------------------- built artifacts
STUDIO = Path(__file__).resolve().parents[1] / "data" / "spatial_lens" / "studio"


@pytest.mark.skipif(
    not (STUDIO / "scene_graph.json").is_file(),
    reason="the studio demo room has not been perceived in this checkout",
)
def test_studio_graph_is_self_consistent() -> None:
    graph = SceneGraph.load(STUDIO / "scene_graph.json")
    assert len(graph.objects) >= 5
    assert 0.0 < graph.free_grid.mean() < 1.0
    for item in graph.objects:
        assert item.name and item.name != "unidentified object"
        for axis in range(3):
            assert item.bbox_min_m[axis] <= item.center_m[axis] <= item.bbox_max_m[axis]
        assert item.height_m > 0.0
        # Every object sits inside the room it was found in.
        assert abs(item.center_m[0]) <= graph.room_size_m[0] / 2
        assert abs(item.center_m[1]) <= graph.room_size_m[1] / 2


@pytest.mark.skipif(
    not (STUDIO / "scans" / "manifest.json").is_file(),
    reason="the studio demo room has not been scanned in this checkout",
)
def test_scan_declares_it_carries_no_labels() -> None:
    manifest = json.loads((STUDIO / "scans" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contains_instance_labels"] is False
    assert manifest["frame_count"] > 0
    for frame in manifest["frames"]:
        assert set(frame) >= {"intrinsics", "camera_to_world", "rgb_path", "depth_path"}
        assert len(frame["intrinsics"]) == 3
        assert len(frame["camera_to_world"]) == 4
        # A pose must be a rigid transform, or projection is meaningless.
        rotation = np.asarray(frame["camera_to_world"])[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        assert math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)


def test_readme_documents_the_capability_boundary() -> None:
    """The README must not claim navigation works in general -- it does not."""

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    collapsed = " ".join(readme.split())
    assert "Spatial Lens: author a room" in collapsed
    assert "with no on-device training at all" in collapsed
    assert "it cannot plan a detour around a large obstacle" in collapsed
    assert "Your words never reach the model" in collapsed
    for overclaim in (
        "navigates reliably",
        "solves navigation",
        "always reaches",
    ):
        assert overclaim not in collapsed
