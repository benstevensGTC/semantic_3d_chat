from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.demo_check import inspect_manifest, inspect_map, inspect_promotion, resolve_checkpoint
from semantic_3d_chat.config import config_hash

ALIGNED_BYPASS_CONTRACT = {
    "language_aligned_tail_dim": 128,
    "native_aligned_coverage_scale": 0.75,
    "learned_scene_token_scale": 0.25,
    "learned_scene_token_rms_target": 0.9,
}


def _config(data_root: Path, *, namespace: str = "prepared") -> dict[str, object]:
    return {
        "paths": {"data_root": str(data_root)},
        "language": {"model_id": "local/lm", "revision": "abc"},
        "scene_encoder": {
            "architecture_version": "spatial_coverage_resampler_v2",
            "global_latents": 256,
            "model_dim": 384,
            "input_voxel_size_m": 0.15,
        },
        "training": {"output_namespace": namespace},
    }


def _checkpoint(
    path: Path,
    config: dict[str, object],
    *,
    architecture: str,
    metadata_updates: dict[str, object] | None = None,
) -> None:
    path.mkdir(parents=True)
    metadata = {
        "schema_version": 3,
        "config_hash": "test-config",
        "semantic_dim": 2048,
        "language_hidden_dim": 896,
        "language_model_id": "local/lm",
        "language_revision": "abc",
        "scene_latents": 256,
        "scene_model_dim": 384,
        "input_voxel_size_m": 0.15,
        "scene_encoder_architecture_version": architecture,
        "epoch": 1,
        "output_namespace": path.parent.name,
    }
    metadata.update(metadata_updates or {})
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "adapter.safetensors").write_bytes(b"prepared")


def test_resolve_checkpoint_rejects_legacy_and_prefers_configured_namespace(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    config = _config(data_root)
    _checkpoint(
        data_root / "checkpoints" / "best",
        config,
        architecture="legacy_resampler_v1",
    )
    expected = data_root / "checkpoints" / "prepared" / "best"
    _checkpoint(expected, config, architecture="spatial_coverage_resampler_v2")

    assert resolve_checkpoint(config) == expected.resolve()


def test_resolve_checkpoint_honors_isolated_checkpoint_root(tmp_path: Path) -> None:
    config = _config(tmp_path / "shared-data", namespace="gemma4_e2b")
    isolated = tmp_path / "candidate" / "checkpoints"
    config["paths"]["checkpoints_root"] = str(isolated)
    expected = isolated / "gemma4_e2b" / "best"
    _checkpoint(expected, config, architecture="spatial_coverage_resampler_v2")

    assert resolve_checkpoint(config) == expected.resolve()


def test_resolve_checkpoint_requires_explicit_nondefault_aligned_bypass_contract(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "data", namespace="gemma4")
    config["scene_encoder"].update(ALIGNED_BYPASS_CONTRACT)
    expected = tmp_path / "data" / "checkpoints" / "gemma4" / "best"
    _checkpoint(expected, config, architecture="spatial_coverage_resampler_v2")

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_checkpoint(config)
    assert all(key in str(exc_info.value) for key in ALIGNED_BYPASS_CONTRACT)

    metadata_path = expected / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(ALIGNED_BYPASS_CONTRACT)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert resolve_checkpoint(config) == expected.resolve()


def test_resolve_checkpoint_rejects_aligned_bypass_value_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path / "data", namespace="gemma4")
    config["scene_encoder"].update(ALIGNED_BYPASS_CONTRACT)
    candidate = tmp_path / "data" / "checkpoints" / "gemma4" / "best"
    _checkpoint(
        candidate,
        config,
        architecture="spatial_coverage_resampler_v2",
        metadata_updates={**ALIGNED_BYPASS_CONTRACT, "learned_scene_token_scale": 0.5},
    )

    with pytest.raises(FileNotFoundError, match="learned_scene_token_scale"):
        resolve_checkpoint(config)


def test_behavioral_promotion_is_explicit_and_bound_to_checkpoint_and_config(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "data")
    checkpoint = tmp_path / "data" / "checkpoints" / "prepared" / "best"
    _checkpoint(checkpoint, config, architecture="spatial_coverage_resampler_v2")

    with pytest.raises(FileNotFoundError, match="promotion.json"):
        inspect_promotion(checkpoint, config)

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    promotion = {
        "schema_version": 1,
        "status": "accepted",
        "config_hash": config_hash(config),
        "checkpoint_metadata_sha256": sha256(checkpoint / "metadata.json"),
        "checkpoint_adapter_sha256": sha256(checkpoint / "adapter.safetensors"),
        "evidence": ["reports/metrics/accepted_gate.json"],
    }
    (checkpoint / "promotion.json").write_text(json.dumps(promotion), encoding="utf-8")
    inspected = inspect_promotion(checkpoint, config)
    assert inspected["status"] == "accepted"

    promotion["checkpoint_adapter_sha256"] = "0" * 64
    (checkpoint / "promotion.json").write_text(json.dumps(promotion), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_adapter_sha256"):
        inspect_promotion(checkpoint, config)


def test_inspect_map_reads_shapes_from_headers(tmp_path: Path) -> None:
    path = tmp_path / "voxel_map.npz"
    np.savez_compressed(
        path,
        centers_world=np.zeros((3, 3), dtype=np.float32),
        semantic_features=np.zeros((3, 2048), dtype=np.float16),
        mean_rgb=np.zeros((3, 3), dtype=np.float32),
        normal=np.zeros((3, 3), dtype=np.float32),
        confidence=np.ones(3, dtype=np.float32),
        observation_count=np.ones(3, dtype=np.int32),
    )

    summary = inspect_map(path)

    assert summary["voxel_count"] == 3
    assert summary["semantic_dim"] == 2048
    assert summary["semantic_dtype"] == "float16"


def test_manifest_rejects_path_escape_before_opening_artifact(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene_000001"
    scene_root.mkdir()
    manifest = {
        "scene_id": "scene_000001",
        "frames": [
            {
                "frame_id": "f_000000",
                "intrinsics": np.eye(3).tolist(),
                "camera_to_world": np.eye(4).tolist(),
                "rgb_path": "../outside.png",
                "depth_path": "depth/f_000000.npy",
            }
        ],
    }
    path = scene_root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes the scene directory"):
        inspect_manifest(path, "scene_000001")


def test_manifest_rejects_semantic_metadata_key(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "scene_id": "scene_000001",
                "object_labels": ["prohibited"],
                "frames": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden key: object_labels"):
        inspect_manifest(path, "scene_000001")
