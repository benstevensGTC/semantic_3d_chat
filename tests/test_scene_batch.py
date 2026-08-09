from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_scene_batch import (
    _render_command,
    _write_batch_oracle_manifest,
)
from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.scene_variants import batch_scene_plans
from semantic_3d_chat.rendering_io import load_manifest


def test_render_command_receives_no_counterfactual_semantics() -> None:
    config = load_config("configs/experiments/multiscene.yaml")
    plan = next(
        plan for plan in batch_scene_plans(config) if plan.scene_id == "scene_000004"
    )
    command = _render_command(
        "/opt/local/bin/blender",
        Path("/project/configs/default.yaml"),
        Path("/project/data"),
        plan,
    )
    serialized = " ".join(command)

    assert "scene_000004" in serialized
    assert "--color-variant" not in command
    assert "--layout-variant" not in command
    assert "--remove-instance" not in command
    assert "pair_000001" not in serialized
    assert "swap_red_blue" not in serialized
    assert "i_000" not in serialized


def test_batch_oracle_manifest_is_full_and_oracle_side(tmp_path: Path) -> None:
    config = load_config("configs/experiments/multiscene.yaml")
    plans = batch_scene_plans(config)
    path = tmp_path / "oracle" / "batches" / "multiscene.json"

    _write_batch_oracle_manifest(
        path,
        plans,
        base_config=Path.cwd() / "configs" / "default.yaml",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scene_count"] == 10
    assert len(payload["scenes"]) == 10
    assert payload["scenes"][0]["scene_id"] == "scene_000001"
    assert payload["scenes"][-1]["scene_id"] == "scene_000010"


def test_runtime_manifest_requires_opaque_paths_and_rejects_semantic_keys(
    tmp_path: Path,
) -> None:
    safe = {
        "scene_id": "scene_000004",
        "frames": [
            {
                "frame_id": "f_000000",
                "rgb_path": "rgb/f_000000.png",
                "depth_path": "depth/f_000000.npy",
                "intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "camera_to_world": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(safe), encoding="utf-8")
    assert load_manifest(path)["scene_id"] == "scene_000004"

    unsafe_key = {**safe, "caption": "semantic text"}
    path.write_text(json.dumps(unsafe_key), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden semantic keys"):
        load_manifest(path)

    unsafe_filename = json.loads(json.dumps(safe))
    unsafe_filename["frames"][0]["rgb_path"] = "rgb/chair.png"
    path.write_text(json.dumps(unsafe_filename), encoding="utf-8")
    with pytest.raises(ValueError, match="opaque frame ID"):
        load_manifest(path)

    hidden_semantics = {**safe, "metadata": {"summary": "semantic text"}}
    path.write_text(json.dumps(hidden_semantics), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected keys"):
        load_manifest(path)
