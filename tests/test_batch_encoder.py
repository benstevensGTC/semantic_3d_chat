from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.vision.batch_encoder import (
    extract_batch_features,
    selected_scene_ids,
)
from semantic_3d_chat.vision.encoder import FrameFeatureCache, _resolved_model_revision


def test_batch_encoder_requires_explicit_deferred_unlock() -> None:
    config = load_config("configs/experiments/diverse28.yaml")
    with pytest.raises(ValueError, match="include-deferred-test"):
        selected_scene_ids(config, split="test", include_deferred_test=False)
    assert selected_scene_ids(
        config, split="test", include_deferred_test=True
    ) == tuple(f"scene_{index:06d}" for index in range(25, 31))


def test_batch_encoder_rejects_string_deferred_splits() -> None:
    config = load_config("configs/experiments/diverse28.yaml")
    config["batch"]["deferred_splits"] = "test"
    with pytest.raises(TypeError, match="list or tuple"):
        selected_scene_ids(config, split="test", include_deferred_test=True)


def test_exact_model_revision_cannot_be_redirected_by_auxiliary_report() -> None:
    revision = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    assert _resolved_model_revision("google/gemma-4-E2B-it", revision) == revision


def test_corrupt_feature_cache_is_a_resume_miss(tmp_path) -> None:
    cache = FrameFeatureCache(tmp_path, "sealed-signature")
    cache.path_for("f_000001").write_bytes(b"truncated-npz")
    assert cache.load("f_000001", "a" * 64) is None


def test_batch_encoder_loads_one_model_and_reuses_it_for_every_scene() -> None:
    encoder = SimpleNamespace(name="one shared complete-image encoder")
    loads: list[object] = []
    calls: list[tuple[str, object]] = []

    def loader(config: dict, **_kwargs: object) -> object:
        loads.append(config)
        return encoder

    def extractor(
        _config: dict, scene_id: str, *, encoder: object, **_kwargs: object
    ) -> dict:
        calls.append((scene_id, encoder))
        return {"frames": 24, "extracted": 24, "reused": 0}

    scene_ids = ("scene_000025", "scene_000026", "scene_000027")
    results = extract_batch_features(
        {"vision": {}},
        scene_ids,
        local_files_only=True,
        device=torch.device("cpu"),
        encoder_loader=loader,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
    )

    assert len(loads) == 1
    assert calls == [(scene_id, encoder) for scene_id in scene_ids]
    assert sum(result["frames"] for result in results) == 72
