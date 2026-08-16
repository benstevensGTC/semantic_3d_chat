from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from semantic_3d_chat.vision.encoder import DenseCLIPEncoder, extract_manifest_features
from semantic_3d_chat.vision.model_registry import CLIP_VIT_BASE_PATCH16_224
from semantic_3d_chat.vision.patch_features import DensePatchFeatures


class FakeProcessor:
    def __init__(self) -> None:
        self.image_calls: list[Image.Image] = []
        self.text_calls: list[list[str]] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        if "images" in kwargs:
            image = kwargs["images"]
            assert isinstance(image, Image.Image)
            self.image_calls.append(image)
            return {"pixel_values": torch.ones(1, 3, 224, 224)}
        queries = list(kwargs["text"])
        self.text_calls.append(queries)
        batch = len(queries)
        return {
            "input_ids": torch.arange(5).repeat(batch, 1),
            "attention_mask": torch.ones(batch, 5, dtype=torch.long),
        }


class FakeVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.last_kwargs: dict[str, Any] = {}
        self.post_layernorm = nn.LayerNorm(768)

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        self.call_count += 1
        self.last_kwargs = kwargs
        batch = kwargs["pixel_values"].shape[0]
        token = torch.arange(197, dtype=torch.float32).reshape(1, 197, 1)
        channel = torch.arange(768, dtype=torch.float32).reshape(1, 1, 768)
        # Token-dependent channel modulation survives tokenwise LayerNorm and
        # makes it possible to assert that patches remain spatially distinct.
        base = channel / 768.0 + token * ((channel.remainder(11) - 5.0) / 1000.0)
        base = base.expand(batch, -1, -1)
        hidden_states = tuple(base + layer / 100.0 for layer in range(13))
        return SimpleNamespace(last_hidden_state=hidden_states[-1], hidden_states=hidden_states)


class FakeTextModel(nn.Module):
    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        batch = input_ids.shape[0]
        values = torch.arange(512, dtype=torch.float32).repeat(batch, 1)
        values = values + torch.arange(batch, dtype=torch.float32).reshape(-1, 1)
        return SimpleNamespace(pooler_output=values)


class FakeCLIPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = FakeVisionModel()
        self.text_model = FakeTextModel()
        self.visual_projection = nn.Linear(768, 512, bias=False)
        self.text_projection = nn.Identity()
        with torch.no_grad():
            self.visual_projection.weight.zero_()
            self.visual_projection.weight[:, :512] = torch.eye(512)


def _encoder() -> tuple[DenseCLIPEncoder, FakeCLIPModel, FakeProcessor]:
    model = FakeCLIPModel()
    processor = FakeProcessor()
    encoder = DenseCLIPEncoder(
        model,
        processor,
        CLIP_VIT_BASE_PATCH16_224,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        storage_dtype=torch.float16,
        middle_layer=6,
        late_layer=12,
    )
    return encoder, model, processor


def test_one_complete_image_makes_exactly_one_vision_call_and_many_patches() -> None:
    encoder, model, processor = _encoder()
    complete_image = Image.new("RGB", (224, 224), color=(80, 120, 160))

    features = encoder.encode_image(complete_image)

    assert model.vision_model.call_count == 1
    assert len(processor.image_calls) == 1
    assert processor.image_calls[0].size == (224, 224)
    assert model.vision_model.last_kwargs["output_hidden_states"] is True
    assert "return_dict" not in model.vision_model.last_kwargs
    assert model.vision_model.last_kwargs["pixel_values"].shape == (1, 3, 224, 224)
    assert features.native_middle.shape == (14, 14, 768)
    assert features.native_late.shape == (14, 14, 768)
    assert features.native_middle_late.shape == (14, 14, 1536)
    assert features.clip_aligned.shape == (14, 14, 512)
    assert features.spatial_features.shape == (14, 14, 2048)
    assert features.native_middle.dtype == torch.float16
    assert features.clip_aligned.dtype == torch.float16
    assert not torch.equal(features.native_middle[0, 0], features.native_middle[0, 1])
    assert not torch.equal(features.clip_aligned[0, 0], features.clip_aligned[0, 1])


def test_encoder_rejects_resizing_or_manual_crop_inputs() -> None:
    encoder, model, _ = _encoder()
    image = Image.new("RGB", (448, 224))

    try:
        encoder.encode_image(image)
    except ValueError as exc:
        assert "Re-render rather than crop" in str(exc)
    else:
        raise AssertionError("A non-native render must be rejected")
    assert model.vision_model.call_count == 0


def test_text_query_helper_projects_and_normalizes_queries() -> None:
    encoder, _, processor = _encoder()

    embeddings = encoder.encode_text_queries(["chair", "bowl"])

    assert embeddings.shape == (2, 512)
    assert embeddings.dtype == torch.float32
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=-1), torch.ones(2))
    assert processor.text_calls == [["chair", "bowl"]]


class CountingEncoder:
    def __init__(self) -> None:
        self.calls = 0
        generator = torch.Generator().manual_seed(123)
        self.features = DensePatchFeatures(
            native_middle=torch.randn(14, 14, 768, generator=generator, dtype=torch.float16),
            native_late=torch.randn(14, 14, 768, generator=generator, dtype=torch.float16),
            clip_aligned=torch.randn(14, 14, 512, generator=generator, dtype=torch.float16),
        )

    def encode_image(self, image: Image.Image) -> DensePatchFeatures:
        assert image.size == (224, 224)
        self.calls += 1
        return self.features


def _write_sanitized_manifest(root: Path) -> Path:
    rgb_directory = root / "rgb"
    depth_directory = root / "depth"
    rgb_directory.mkdir(parents=True)
    depth_directory.mkdir()
    frames = []
    for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
        frame_id = f"frame_{index:06d}"
        Image.new("RGB", (224, 224), color=color).save(rgb_directory / f"{frame_id}.png")
        np.save(depth_directory / f"{frame_id}.npy", np.ones((224, 224), dtype=np.float32))
        frames.append(
            {
                "frame_id": frame_id,
                "rgb_path": f"rgb/{frame_id}.png",
                "depth_path": f"depth/{frame_id}.npy",
                "intrinsics": np.eye(3).tolist(),
                "camera_to_world": np.eye(4).tolist(),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"scene_id": "scene_000001", "frames": frames}), encoding="utf-8"
    )
    return manifest


def test_manifest_extraction_writes_fusion_compatible_per_frame_cache_and_reuses_it(
    tmp_path: Path,
) -> None:
    manifest = _write_sanitized_manifest(tmp_path / "rendered" / "scene_000001")
    output = tmp_path / "features" / "scene_000001"
    config = {
        "vision": {
            "model_id": "openai/clip-vit-base-patch16",
            "revision": "main",
            "input_size": 224,
            "middle_layer": 6,
            "late_layer": 12,
            "dtype": "float16",
        }
    }
    encoder = CountingEncoder()

    first = extract_manifest_features(
        config,
        "scene_000001",
        manifest_path=manifest,
        output_root=output,
        encoder=encoder,  # type: ignore[arg-type]
    )

    assert first["extracted"] == 2
    assert first["reused"] == 0
    assert encoder.calls == 2
    assert (output / "frame_000000.npz").is_file()
    assert (output / "frame_000001.npz").is_file()
    with np.load(output / "frame_000000.npz", allow_pickle=False) as archive:
        assert archive["spatial_features"].shape == (14, 14, 2048)
        assert archive["spatial_features"].dtype == np.float16
        assert tuple(archive["component_offsets"].tolist()) == (0, 768, 1536, 2048)
    feature_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert all(frame["vision_encoder_calls"] == 1 for frame in feature_manifest["frames"])
    assert all(frame["complete_image_encoded"] for frame in feature_manifest["frames"])
    assert all(
        frame["manual_crops_or_patch_reencoding"] is False for frame in feature_manifest["frames"]
    )

    second = extract_manifest_features(
        config,
        "scene_000001",
        manifest_path=manifest,
        output_root=output,
        encoder=encoder,  # type: ignore[arg-type]
    )

    assert second["extracted"] == 0
    assert second["reused"] == 2
    assert encoder.calls == 2


def test_manifest_extraction_rejects_nonopaque_frame_id(tmp_path: Path) -> None:
    manifest = _write_sanitized_manifest(tmp_path / "rendered" / "scene_000001")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["frames"][0]["frame_id"] = "chair_view"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    config = {
        "vision": {
            "model_id": "openai/clip-vit-base-patch16",
            "input_size": 224,
            "middle_layer": 6,
            "late_layer": 12,
            "dtype": "float16",
        }
    }

    try:
        extract_manifest_features(
            config,
            "scene_000001",
            manifest_path=manifest,
            output_root=tmp_path / "features",
            encoder=CountingEncoder(),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        assert "frame_id must be opaque" in str(exc)
    else:
        raise AssertionError("Semantic frame filenames must be rejected")
