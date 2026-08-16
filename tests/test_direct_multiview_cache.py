from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.direct_multiview_baseline import (
    LocalMultiViewAnswerer,
    complete_view_paths,
    direct_multiview_scene_cache_contract,
    run_direct_multiview_baseline,
)

_CACHE_CONTRACT = "gemma4_decoder_kv_scene_prefix_v1"


def _write_scene(root: Path, *, view_count: int = 24) -> Path:
    scene = root / "data" / "rendered" / "scene_999001"
    (scene / "rgb").mkdir(parents=True)
    frames: list[dict[str, Any]] = []
    for index in range(view_count):
        frame_id = f"f_{index:06d}"
        relative = f"rgb/{frame_id}.png"
        Image.new("RGB", (8, 6), (index, 0, 0)).save(scene / relative)
        frames.append(
            {
                "camera_id": f"c_{index:06d}",
                "camera_to_world": [[1, 0, 0, 0]] * 4,
                "depth_path": f"depth/{frame_id}.npy",
                "frame_id": frame_id,
                "frame_number": index,
                "intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "rgb_path": relative,
            }
        )
    (scene / "manifest.json").write_text(
        json.dumps(
            {
                "coordinate_system": {
                    "world": "x_right_y_forward_z_up",
                    "camera": "x_right_y_down_z_forward",
                    "depth": "axial_camera_z",
                    "units": "meters",
                },
                "scene_id": "scene_999001",
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return scene


def _write_questions(path: Path) -> None:
    rows = [
        {
            "answer": "yes",
            "question": "Is there a cube?",
            "question_id": "q_000001",
            "scene_id": "scene_999001",
            "target_instance": "must_not_reach_cache",
        },
        {
            "answer": "left",
            "question": "Is it on the left?",
            "question_id": "q_000002",
            "scene_id": "scene_999001",
            "target_instance": "must_not_reach_cache",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_scene_cache_contract_is_explicit_and_gemma_only() -> None:
    assert (
        direct_multiview_scene_cache_contract(
            {
                "backend": "gemma4",
                "model_id": "google/gemma-4-E2B-it",
                "scene_cache": _CACHE_CONTRACT,
            }
        )
        == _CACHE_CONTRACT
    )
    with pytest.raises(ValueError, match="only for the Gemma 4 backend"):
        direct_multiview_scene_cache_contract(
            {
                "backend": "generic_image_text",
                "model_id": "example/generic",
                "scene_cache": _CACHE_CONTRACT,
            }
        )


def test_cache_builder_api_cannot_receive_a_question() -> None:
    parameters = inspect.signature(LocalMultiViewAnswerer.prepare_scene_cache).parameters
    assert list(parameters) == ["self", "images"]


def test_local_answerer_builds_prefix_before_question_and_clones_it_per_answer() -> None:
    trailer = "<turn|>\n<|turn>model\n"

    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def encode(text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            if text == trailer:
                return [90, 91]
            assert text == f" Where?{trailer}"
            return [7, 8, 90, 91]

    class Processor:
        tokenizer = Tokenizer()

        @staticmethod
        def apply_chat_template(conversation, *, tokenize, **_kwargs):
            content = conversation[0]["content"]
            image_count = sum(item["type"] == "image" for item in content)
            prompt = content[-1]["text"]
            rendered_prompt = prompt[:-1] if prompt.endswith(": ") else prompt
            rendered = f"<bos>{'<|image|>' * image_count}{rendered_prompt}{trailer}"
            if not tokenize:
                return rendered
            assert prompt == "Use all views.\nQuestion: "
            return {
                "input_ids": torch.tensor([[2, 3, 4, 90, 91]]),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0]]),
                "pixel_values": torch.zeros(image_count, 2, 3),
                "image_position_ids": torch.zeros(image_count, 2, 2, dtype=torch.long),
            }

        @staticmethod
        def decode(tokens, *, skip_special_tokens):
            assert tokens.tolist() == [42]
            return "yes" if skip_special_tokens else "yes<turn|>"

        @staticmethod
        def parse_response(response, *, prefix):
            assert response == "yes<turn|>"
            assert prefix.shape == (1, 7)
            return {"role": "assistant", "content": "yes"}

    class Cache:
        def __init__(self, length: int) -> None:
            self.length = length

        def get_seq_length(self) -> int:
            return self.length

    class Model:
        def __init__(self) -> None:
            self.source_cache: Cache | None = None
            self.answer_cache: Cache | None = None

        def __call__(self, **kwargs):
            assert kwargs["input_ids"].tolist() == [[2, 3, 4]]
            assert kwargs["pixel_values"].shape[0] == 2
            self.source_cache = Cache(3)
            return SimpleNamespace(past_key_values=self.source_cache)

        def generate(self, **kwargs):
            self.answer_cache = kwargs["past_key_values"]
            assert self.answer_cache is not self.source_cache
            assert kwargs["input_ids"].tolist() == [[7, 8, 90, 91]]
            assert kwargs["attention_mask"].shape == (1, 7)
            return torch.tensor([[7, 8, 90, 91, 42]])

    model = Model()
    answerer = LocalMultiViewAnswerer(
        model=model,
        processor=Processor(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        system_prompt="Use all views.",
        max_answer_tokens=4,
        resize_longest_edge=None,
        backend="gemma4",
        enable_thinking=False,
    )
    images = [Image.new("RGB", (8, 6), "red"), Image.new("RGB", (8, 6), "blue")]
    try:
        scene_cache = answerer.prepare_scene_cache(images)
        assert scene_cache.complete_view_count == 2
        assert scene_cache.past_key_values is model.source_cache
        assert answerer.answer_from_scene_cache(scene_cache, "Where?") == "yes"
    finally:
        for image in images:
            image.close()
    assert model.source_cache is not None
    assert model.source_cache.length == 3


def test_scene_cache_is_built_once_from_all_24_complete_views_and_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_scene(tmp_path)
    references = tmp_path / "questions.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_questions(references)
    config = load_config()
    config["paths"]["data_root"] = "data"
    config["evaluation"]["baselines"]["direct_multiview"].update(
        backend="gemma4",
        max_views=24,
        require_all_manifest_views=True,
        scene_cache=_CACHE_CONTRACT,
    )
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.direct_multiview_baseline.PROJECT_ROOT",
        tmp_path,
    )

    class CacheAnswerer:
        def __init__(self) -> None:
            self.builds: list[list[tuple[int, int, int]]] = []
            self.cache_object_ids: list[int] = []
            self.questions: list[str] = []

        def prepare_scene_cache(self, images: list[Image.Image]) -> SimpleNamespace:
            # Pixel values prove manifest ordering without exposing a filename,
            # category, QA target, or question to cache construction.
            self.builds.append([image.getpixel((0, 0)) for image in images])
            return SimpleNamespace(
                complete_view_count=len(images),
                contract=_CACHE_CONTRACT,
                prefix_token_sha256="a" * 64,
            )

        def answer_from_scene_cache(self, cache: SimpleNamespace, question: str) -> str:
            self.cache_object_ids.append(id(cache))
            self.questions.append(question)
            return "yes" if question.startswith("Is there") else "left"

    answerer = CacheAnswerer()
    report = run_direct_multiview_baseline(
        config,
        references,
        predictions,
        answerer=answerer,  # type: ignore[arg-type]
    )
    assert len(answerer.builds) == 1
    assert answerer.builds[0] == [(index, 0, 0) for index in range(24)]
    assert len(set(answerer.cache_object_ids)) == 1
    assert answerer.questions == ["Is there a cube?", "Is it on the left?"]
    assert report["manifest_view_counts"] == [24]
    assert report["view_counts"] == [24]
    assert report["scene_cache_build_count"] == 1
    assert report["scene_cache_contract"] == _CACHE_CONTRACT
    assert report["scene_cache_question_independent"] is True
    assert list(report["scene_cache_sha256_by_scene"]) == ["scene_999001"]

    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert len({row["scene_cache_sha256"] for row in rows}) == 1
    assert all(row["scene_cache_question_independent"] is True for row in rows)
    serialized = predictions.read_text(encoding="utf-8")
    assert "must_not_reach_cache" not in serialized
    assert "rgb/" not in serialized

    class MustNotRun:
        def prepare_scene_cache(self, _images):
            raise AssertionError("resume rebuilt a valid scene cache")

        def answer_from_scene_cache(self, _cache, _question):
            raise AssertionError("resume regenerated a valid prediction")

    resumed = run_direct_multiview_baseline(
        config,
        references,
        predictions,
        answerer=MustNotRun(),  # type: ignore[arg-type]
    )
    assert resumed["new_prediction_count"] == 0
    assert resumed["scene_cache_build_count"] == 0
    assert resumed["scene_cache_sha256_by_scene"] == report["scene_cache_sha256_by_scene"]


def test_semantic_rgb_filename_is_rejected_before_pixels_are_loaded(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path, view_count=1)
    manifest_path = scene / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = scene / manifest["frames"][0]["rgb_path"]
    destination = scene / "rgb" / "chair.png"
    source.rename(destination)
    manifest["frames"][0]["rgb_path"] = "rgb/chair.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="opaque frame ID|semantically opaque"):
        complete_view_paths(scene)
