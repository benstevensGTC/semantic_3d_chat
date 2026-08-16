from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.evaluation.v28_stage_a_selector import (
    _checkpoint_paths,
    _frozen_tensor_sha256,
    _sidecar_state,
    _validation_nll,
)


def _checkpoint(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    save_file({"value": torch.ones(1)}, path / "adapter.safetensors")
    (path / "metadata.json").write_text("{}", encoding="utf-8")
    return path


def test_checkpoint_inventory_requires_update_zero_and_orders_updates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage_a"
    root.mkdir()
    _checkpoint(root, "update_002")
    _checkpoint(root, "update_000")
    _checkpoint(root, "update_001")
    (root / "best").mkdir()
    assert [path.name for path in _checkpoint_paths(root)] == [
        "update_000",
        "update_001",
        "update_002",
    ]

    missing = tmp_path / "missing_zero"
    missing.mkdir()
    _checkpoint(missing, "update_001")
    with pytest.raises(FileNotFoundError, match="update_000"):
        _checkpoint_paths(missing)


def test_frozen_hash_excludes_only_two_authorized_training_surfaces() -> None:
    base = {
        "scene_model.weight": torch.tensor([1.0]),
        "dense_sidecar_adapter.base_projection.weight": torch.tensor([2.0]),
        "dense_sidecar_adapter.output_projection.weight": torch.tensor([0.0]),
        "dense_sidecar_adapter.channel_gain": torch.tensor([0.0]),
    }
    changed_trainable = {
        **base,
        "dense_sidecar_adapter.output_projection.weight": torch.tensor([3.0]),
        "dense_sidecar_adapter.channel_gain": torch.tensor([4.0]),
    }
    assert _frozen_tensor_sha256(base) == _frozen_tensor_sha256(changed_trainable)
    changed_frozen = {**base, "scene_model.weight": torch.tensor([5.0])}
    assert _frozen_tensor_sha256(base) != _frozen_tensor_sha256(changed_frozen)


def test_sidecar_state_strips_prefix_and_validation_nll_is_finite() -> None:
    state = _sidecar_state(
        {
            "dense_sidecar_adapter.channel_gain": torch.zeros(3),
            "scene_model.weight": torch.ones(2),
        }
    )
    assert set(state) == {"channel_gain"}
    assert _validation_nll(
        {"history": [{"validation_answer_token_nll": 1.25}]}
    ) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="not finite"):
        _validation_nll(
            {"history": [{"validation_answer_token_nll": float("nan")}]}
        )
    with pytest.raises(TypeError, match="numeric"):
        _validation_nll(
            json.loads('{"history": [{"validation_answer_token_nll": "bad"}]}')
        )
