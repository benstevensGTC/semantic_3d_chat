from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.chat.grounding_sidecar_v78_runtime import (
    V78GroundingSidecarRuntime,
    authenticate_v78_grounding_checkpoint,
)
from semantic_3d_chat.scene_encoder.grounding_sidecar_v78 import GroundingSidecarV78

ROOT = Path(__file__).parents[1]
RELEASE = (
    ROOT
    / "data_gemma4/runtime/checkpoints/gemma4_v78_grounding_diagnostic_release_v1"
)


def _fixture_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = RELEASE
    if not source.is_dir():
        pytest.skip("local V78 runtime diagnostic release is unavailable")
    destination = tmp_path / "checkpoint"
    shutil.copytree(source, destination)
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    return destination, metadata


def _load(checkpoint: Path, metadata: dict[str, object]) -> V78GroundingSidecarRuntime:
    torch.manual_seed(1)
    prefix = torch.randn(1, 258, 1536)
    return V78GroundingSidecarRuntime.load(
        checkpoint,
        scene_prefix=prefix,
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        base_checkpoint_sha256=str(metadata["base_checkpoint_sha256"]),
        base_runtime_config_sha256=str(metadata["base_runtime_config_sha256"]),
        model_id=str(metadata["model_id"]),
        model_revision=str(metadata["model_revision"]),
        device="cpu",
    )


def test_actual_v78_runtime_release_authenticates_without_scene_or_gemma() -> None:
    if not RELEASE.is_dir():
        pytest.skip("local V78 runtime diagnostic release is unavailable")
    metadata = json.loads((RELEASE / "metadata.json").read_text(encoding="utf-8"))

    audit = authenticate_v78_grounding_checkpoint(
        RELEASE,
        base_checkpoint_sha256=str(metadata["base_checkpoint_sha256"]),
        base_runtime_config_sha256=str(metadata["base_runtime_config_sha256"]),
        model_id=str(metadata["model_id"]),
        model_revision=str(metadata["model_revision"]),
    )

    assert audit["passed"] is True
    assert audit["checkpoint_inventory"] == ["grounding.safetensors", "metadata.json"]
    assert audit["gemma_model_loaded"] is False
    assert audit["scene_data_loaded"] is False
    assert audit["official_validation_evidence"] is False
    assert audit["runtime_promotion_authorized"] is False


def test_v78_runtime_prediction_uses_all_tokens_and_preserves_prefix(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture_checkpoint(tmp_path)
    runtime = _load(checkpoint, metadata)
    prefix = runtime._full_prefix.clone()
    before = runtime.full_prefix_sha256

    result = runtime.predict(
        torch.randn(1, 5, 1536),
        scene_prefix=prefix,
        map_xyz=torch.randn(100, 3),
        map_confidence=torch.ones(100),
    )

    assert len(result.xyz_m) == 3
    assert 0.0 <= result.confidence <= 1.0
    assert result.audit["all_scene_tokens_scored"] is True
    assert result.audit["minimum_attention_weight"] > 0.0
    assert result.audit["top_k_selection_used"] is False
    assert runtime.full_prefix_sha256 == before
    runtime.assert_prefix_unchanged(prefix)


def test_v78_runtime_fails_closed_on_base_identity_mismatch(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="different base checkpoint"):
        V78GroundingSidecarRuntime.load(
            checkpoint,
            scene_prefix=torch.zeros(1, 258, 1536),
            room_min=torch.tensor([-3.0, -2.5, 0.0]),
            room_max=torch.tensor([3.0, 2.5, 3.0]),
            base_checkpoint_sha256="0" * 64,
            base_runtime_config_sha256=str(metadata["base_runtime_config_sha256"]),
            model_id=str(metadata["model_id"]),
            model_revision=str(metadata["model_revision"]),
            device="cpu",
        )


def test_v78_runtime_fails_closed_on_inventory_or_weights_change(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture_checkpoint(tmp_path)
    (checkpoint / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly its two"):
        _load(checkpoint, metadata)
    (checkpoint / "extra.json").unlink()

    model = GroundingSidecarV78()
    save_file(model.state_dict(), checkpoint / "grounding.safetensors")
    with pytest.raises(ValueError, match="weights changed"):
        _load(checkpoint, metadata)


def test_v78_runtime_fails_closed_when_prefix_changes(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture_checkpoint(tmp_path)
    runtime = _load(checkpoint, metadata)
    changed = runtime._full_prefix.clone()
    changed[0, 7, 9] += 1.0

    with pytest.raises(RuntimeError, match="prefix changed"):
        runtime.assert_prefix_unchanged(changed)
