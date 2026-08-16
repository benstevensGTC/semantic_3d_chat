from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from semantic_3d_chat.mapping.voxel_map import SparseVoxelMap
from semantic_3d_chat.robot.conversation import (
    ConversationalEmbodiedAgent,
    parse_navigation_instruction,
)
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator


class FakeTextEncoder:
    output_dim = 4

    def encode_queries(self, queries: list[str] | tuple[str, ...]) -> np.ndarray:
        values = {
            "fixture": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "a fixture": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "marker": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "a marker": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        }
        return np.stack([values[query] for query in queries])


def _write_map(path: Path) -> None:
    first = np.asarray(
        [[1.45, y, z] for y in (-0.05, 0.0, 0.05) for z in (0.45, 0.55)],
        dtype=np.float32,
    )
    second = first.copy()
    second[:, 0] = -1.45
    points = np.concatenate((first, second))
    features = np.zeros((len(points), 4), dtype=np.float32)
    features[: len(first), 0] = 1.0
    features[len(first) :, 1] = 1.0
    voxel_map = SparseVoxelMap(0.05, feature_dim=4)
    voxel_map.add_observations(
        points,
        features,
        rgb=np.full((len(points), 3), 120.0, dtype=np.float32),
        frame_id="f_000001",
    )
    voxel_map.save(path, metadata={"scene_id": "scene_000001"})


def _config(tmp_path: Path, map_path: Path) -> dict:
    return {
        "seed": 9,
        "paths": {
            "data_root": str(tmp_path / "runtime"),
            "maps_root": str(map_path.parents[1]),
        },
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "render": {"resolution": [16, 16], "horizontal_fov_degrees": 72.0},
        "robot": {
            "radius_m": 0.20,
            "camera_height_m": 1.20,
            "max_move_m": 0.50,
            "max_move_to_m": 0.50,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.80,
            "surface_padding_m": 0.02,
            "scan_depth_min_m": 0.10,
            "scan_depth_max_m": 6.0,
            "initial_position_xy_m": [0.0, 0.0],
            "history_length": 16,
        },
    }


@dataclass
class FakeAnswer:
    answer: str = "local answer"
    grounding_xyz_m: tuple[float, float, float] = (0.1, 0.2, 0.3)
    grounding_confidence: float = 0.75
    prefix_hash: str = "a" * 64


class FakeUpdater:
    def __init__(self, path: Path) -> None:
        self.base_map_path = path
        self.persistent_map_path = path.with_name("persistent.npz")


class FakeRefreshingRuntime:
    def __init__(self, simulator: EmbodiedCameraSimulator, path: Path) -> None:
        self.simulator = simulator
        self.map_updater = FakeUpdater(path)
        self.calls: list[str] = []

    def _call(self, name: str, *args: float) -> dict:
        self.calls.append(name)
        return getattr(self.simulator, name)(*args)

    def turn(self, value: float) -> dict:
        return self._call("turn", value)

    def move_forward(self, value: float) -> dict:
        return self._call("move_forward", value)

    def move_backward(self, value: float) -> dict:
        return self._call("move_backward", value)

    def move_to(self, x: float, y: float) -> dict:
        return self._call("move_to", x, y)

    def scan(self) -> dict:
        self.calls.append("scan")
        result = self.simulator.get_robot_state()
        result["success"] = True
        return result

    def stop(self) -> dict:
        return self._call("stop")

    def answer(self, question: str) -> FakeAnswer:
        assert question
        self.calls.append("answer")
        return FakeAnswer()

    def prefix_binding(self) -> dict:
        return {"scene_prefix_sha256": "a" * 64, "map_version": 0}


@pytest.mark.parametrize(
    ("text", "kind", "targets"),
    [
        ("Turn toward the bowl.", "face", ("bowl",)),
        ("Face the chair", "face", ("chair",)),
        ("Move closer to the table.", "approach", ("table",)),
        ("Go around the chair and stop beside the bowl.", "approach", ("bowl",)),
        ("Walk toward the red cube.", "approach", ("red cube",)),
        ("Look at the picture frame.", "face", ("picture frame",)),
        ("Move until the lamp is directly ahead.", "approach", ("lamp",)),
        ("Stop between the chair and the table.", "between", ("chair", "table")),
    ],
)
def test_navigation_instruction_parser(
    text: str,
    kind: str,
    targets: tuple[str, ...],
) -> None:
    parsed = parse_navigation_instruction(text)
    assert parsed is not None
    assert parsed.kind == kind
    assert parsed.targets == targets


def test_question_is_not_misclassified_as_imperative() -> None:
    assert parse_navigation_instruction("Which way should I turn to face the lamp?") is None


def test_conversational_agent_moves_through_runtime_and_falls_back_to_chat(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    _write_map(map_path)
    config = _config(tmp_path, map_path)
    runtime = FakeRefreshingRuntime(
        EmbodiedCameraSimulator(config, "scene_000001"),
        map_path,
    )
    agent = ConversationalEmbodiedAgent(
        runtime,
        FakeTextEncoder(),
        room_size_m=config["scene"]["room_size_m"],
        feature_start=0,
        feature_dim=4,
    )

    moved = agent.handle("Move closer to the fixture.")
    assert moved["kind"] == "navigation"
    assert moved["success"] is True
    assert "move_to" in runtime.calls and runtime.calls[-1] == "scan"
    assert moved["environmental_text_inputs"] == []
    assert "fixture" not in str(moved).casefold()

    answered = agent.handle("What is around you?")
    assert answered["kind"] == "answer"
    assert answered["answer"] == "local answer"
    assert runtime.calls[-1] == "answer"


def test_conversational_agent_faces_and_handles_numeric_actions(tmp_path: Path) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    _write_map(map_path)
    config = _config(tmp_path, map_path)
    runtime = FakeRefreshingRuntime(
        EmbodiedCameraSimulator(config, "scene_000001"),
        map_path,
    )
    agent = ConversationalEmbodiedAgent(
        runtime,
        FakeTextEncoder(),
        room_size_m=config["scene"]["room_size_m"],
        feature_start=0,
        feature_dim=4,
    )

    faced = agent.handle("Face the fixture.")
    assert faced["success"] is True
    assert "turn" in runtime.calls and runtime.calls[-1] == "scan"

    turned = agent.handle("Turn right 30 degrees.")
    assert turned["success"] is True
    assert turned["command"] == "turn"
