from pathlib import Path

import pytest

from semantic_3d_chat.config import (
    PROJECT_ROOT,
    artifact_root,
    default_checkpoint_path,
    load_config,
    project_path,
    reports_root,
)


def test_project_path_defaults_each_kind_below_data_root() -> None:
    config = {"paths": {"data_root": "data"}}
    assert project_path(config, "rendered", "scene_000001") == (
        PROJECT_ROOT / "data" / "rendered" / "scene_000001"
    )
    assert project_path(config, "features", "scene_000001") == (
        PROJECT_ROOT / "data" / "features" / "scene_000001"
    )


def test_project_path_honors_per_kind_isolation_without_moving_renders() -> None:
    config = {
        "paths": {
            "data_root": "data",
            "features_root": "data_gemma4/features",
            "maps_root": "data_gemma4/maps",
            "checkpoints_root": "data_gemma4/checkpoints",
        }
    }
    assert project_path(config, "rendered", "scene_000001") == (
        PROJECT_ROOT / "data" / "rendered" / "scene_000001"
    )
    assert project_path(config, "oracle", "scene_000001") == (
        PROJECT_ROOT / "data" / "oracle" / "scene_000001"
    )
    assert project_path(config, "qa", "train.jsonl") == PROJECT_ROOT / "data" / "qa" / "train.jsonl"
    assert project_path(config, "features", "scene_000001") == (
        PROJECT_ROOT / "data_gemma4" / "features" / "scene_000001"
    )
    assert artifact_root(config, "maps") == PROJECT_ROOT / "data_gemma4" / "maps"
    assert artifact_root(config, "checkpoints") == PROJECT_ROOT / "data_gemma4" / "checkpoints"


def test_artifact_root_rejects_path_like_kind() -> None:
    with pytest.raises(ValueError, match="plain path component"):
        artifact_root({"paths": {"data_root": "data"}}, "../oracle")


def test_absolute_kind_override_remains_absolute(tmp_path: Path) -> None:
    config = {"paths": {"data_root": "data", "features_root": str(tmp_path)}}
    assert project_path(config, "features", "scene_000001") == tmp_path / "scene_000001"


def test_gemma4_candidate_separates_mps_compute_from_portable_cache_dtype() -> None:
    config = load_config("configs/gemma4_e2b.yaml")
    assert config["vision"]["dtype"] == "bfloat16"
    assert config["vision"]["storage_dtype"] == "float16"
    assert config["language"]["dtype"] == "bfloat16"
    assert artifact_root(config, "qa") == PROJECT_ROOT / "data" / "qa"
    assert artifact_root(config, "rendered") == PROJECT_ROOT / "data" / "rendered"
    assert artifact_root(config, "maps") == PROJECT_ROOT / "data_gemma4" / "maps"
    assert default_checkpoint_path(config) == (
        PROJECT_ROOT / "data_gemma4" / "checkpoints" / "gemma4_e2b" / "best"
    )
    assert reports_root(config) == PROJECT_ROOT / "reports" / "gemma4"
    assert config["scene_encoder"]["language_aligned_tail_dim"] == 1536
    assert config["scene_encoder"]["native_aligned_coverage_scale"] == 1.0
    assert config["scene_encoder"]["learned_scene_token_scale"] == 0.1
    assert config["scene_encoder"]["learned_scene_token_rms_target"] == 0.65
    assert config["training"]["language_decoder_gradient_checkpointing"] is True


def test_default_config_keeps_legacy_qwen_scene_token_path() -> None:
    config = load_config("configs/default.yaml")
    assert config["scene_encoder"]["language_aligned_tail_dim"] == 0
    assert config["scene_encoder"]["native_aligned_coverage_scale"] == 0.0
    assert config["scene_encoder"]["learned_scene_token_scale"] == 1.0
    assert config["scene_encoder"]["learned_scene_token_rms_target"] is None
    assert config["training"]["language_decoder_gradient_checkpointing"] is False
