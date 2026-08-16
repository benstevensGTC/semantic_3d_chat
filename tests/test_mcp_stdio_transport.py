from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from semantic_3d_chat.evaluation.mcp_transport_smoke import (
    EXPECTED_TOOLS,
    run_stdio_transport_smoke,
)


def _write_numeric_fixture(tmp_path: Path) -> Path:
    data_root = tmp_path / "runtime_data"
    map_path = data_root / "maps" / "scene_000001" / "voxel_map.npz"
    map_path.parent.mkdir(parents=True)
    centers = np.asarray(
        [
            [-2.9, -2.4, 0.7],
            [-2.9, 2.4, 0.7],
            [2.9, -2.4, 0.7],
            [2.9, 2.4, 0.7],
            [0.8, 0.8, 0.7],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        map_path,
        centers_world=centers,
        mean_rgb=np.full((len(centers), 3), 127.0, dtype=np.float32),
        observation_count=np.ones(len(centers), dtype=np.int32),
    )
    config = {
        "seed": 17,
        "paths": {
            "data_root": str(data_root),
            "reports_root": str(tmp_path / "reports"),
        },
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "render": {"resolution": [32, 24], "horizontal_fov_degrees": 72.0},
        "robot": {
            "radius_m": 0.2,
            "camera_height_m": 1.2,
            "initial_position_xy_m": [0.0, 0.0],
            "max_move_m": 0.5,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.8,
            "surface_padding_m": 0.02,
            "scan_depth_min_m": 0.1,
            "scan_depth_max_m": 6.0,
            "history_length": 16,
        },
    }
    config_path = tmp_path / "mcp_transport_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_official_mcp_sdk_stdio_transport(tmp_path: Path) -> None:
    report = asyncio.run(
        run_stdio_transport_smoke(
            _write_numeric_fixture(tmp_path),
            "scene_000001",
            python_executable=sys.executable,
        )
    )

    assert report["passed"] is True
    assert report["transport"] == "stdio"
    assert report["mcp_sdk_version"] == "2.0.0"
    assert report["tool_count"] == 9
    assert set(report["tools"]) == EXPECTED_TOOLS
    assert report["extra_argument_rejected"] is True
    assert report["out_of_bounds_rejected"] is True
    assert report["out_of_bounds_error_code"] == "E_LIMIT"
    assert report["state_unchanged_after_rejections"] is True
    assert report["same_scene_reset_passed"] is True
    assert report["reset_scene_version"] == 0
    assert report["semantic_result_leaks"] == []
