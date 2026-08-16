from __future__ import annotations

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
    direct_multiview_backend,
    gemma4_multiview_conversation,
    multiview_conversation,
    run_direct_multiview_baseline,
)
from semantic_3d_chat.evaluation.oracle_text_baseline import (
    LocalOracleAnswerer,
    oracle_scene_text,
    run_oracle_text_baseline,
)


def _reference(path: Path, scene_id: str = "scene_999001") -> None:
    path.write_text(
        json.dumps(
            {
                "answer": "red",
                "answer_type": "attribute",
                "question": "What color is the cube?",
                "question_id": "q_000001",
                "scene_id": scene_id,
                "target_instance": "must_not_reach_answerer",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_oracle_text_contains_exact_scene_facts() -> None:
    oracle = {
        "instances": [
            {
                "instance_id": "i_a",
                "category": "cube",
                "kind": "object",
                "color": {"name": "red"},
                "expected_center_xyz_m": [1, 2, 0.5],
                "dimensions_m": [0.2, 0.2, 0.2],
                "support_surface": "i_b",
            },
            {
                "instance_id": "i_b",
                "category": "table",
                "kind": "object",
                "color": {"name": "brown"},
                "expected_center_xyz_m": [0, 2, 0.4],
                "dimensions_m": [1, 1, 0.8],
                "support_surface": None,
            },
        ],
        "relationships": [
            {
                "subject_instance_id": "i_a",
                "predicate": "right_of",
                "object_instance_id": "i_b",
            }
        ],
    }
    text = oracle_scene_text(oracle)
    assert "category=cube" in text
    assert "cube=1" in text
    assert "color=red" in text
    assert "supported_by=table" in text
    assert "subject=cube; predicate=right; object=table" in text


def test_oracle_answerer_accepts_transformers5_batch_encoding_shape() -> None:
    class Encoded:
        input_ids = torch.tensor([[10, 11, 12]])

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return Encoded()

        @staticmethod
        def decode(tokens, **_kwargs):
            assert tokens.tolist() == [42]
            return "red"

    class Model:
        @staticmethod
        def generate(**kwargs):
            assert kwargs["input_ids"].shape == (1, 3)
            assert kwargs["attention_mask"].shape == (1, 3)
            return torch.tensor([[10, 11, 12, 42]])

    answerer = LocalOracleAnswerer.__new__(LocalOracleAnswerer)
    answerer.system_prompt = "Use exact facts."
    answerer.max_answer_tokens = 4
    answerer.local = SimpleNamespace(
        tokenizer=Tokenizer(),
        model=Model(),
        device=torch.device("cpu"),
    )
    assert answerer("category=cube; color=red", "What color is the cube?") == "red"


def test_oracle_runner_only_passes_question_and_derived_scene_text(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_config()
    scene_id = "scene_999001"
    data_root = tmp_path / "data"
    oracle_dir = data_root / "oracle" / scene_id
    oracle_dir.mkdir(parents=True)
    oracle_dir.joinpath("oracle.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "instances": [
                    {
                        "instance_id": "i_a",
                        "category": "cube",
                        "kind": "object",
                        "color": {"name": "red"},
                        "expected_center_xyz_m": [0, 0, 0],
                        "dimensions_m": [1, 1, 1],
                        "support_surface": None,
                    }
                ],
                "relationships": [],
            }
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "references.jsonl"
    output = tmp_path / "predictions.jsonl"
    _reference(reference, scene_id)
    monkeypatch.setattr("semantic_3d_chat.evaluation.oracle_text_baseline.PROJECT_ROOT", tmp_path)
    config["paths"]["data_root"] = "data"
    observed: dict[str, str] = {}

    def answerer(scene_text: str, question: str) -> str:
        observed.update(scene_text=scene_text, question=question)
        return "red"

    report = run_oracle_text_baseline(config, reference, output, answerer=answerer)
    assert observed["question"] == "What color is the cube?"
    assert "must_not_reach_answerer" not in observed["scene_text"]
    assert report["new_prediction_count"] == 1
    assert json.loads(output.read_text())["answer"] == "red"

    # Reusing an opaque question ID with changed text must invalidate the cache.
    changed = json.loads(reference.read_text(encoding="utf-8"))
    changed["question"] = "Name the cube's color."
    reference.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    observed.clear()
    second = run_oracle_text_baseline(config, reference, output, answerer=answerer)
    assert second["new_prediction_count"] == 1
    assert observed["question"] == "Name the cube's color."


def test_complete_view_paths_and_conversation_use_all_full_frames(tmp_path: Path) -> None:
    scene = tmp_path / "scene_999001"
    (scene / "rgb").mkdir(parents=True)
    frames = []
    for index in [2, 0, 1]:
        relative = f"rgb/f_{index:06d}.png"
        Image.new("RGB", (8, 8), (index, 0, 0)).save(scene / relative)
        frames.append(
            {
                "camera_id": f"c_{index:06d}",
                "camera_to_world": [[1, 0, 0, 0]] * 4,
                "depth_path": f"depth/f_{index:06d}.npy",
                "frame_id": f"f_{index:06d}",
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
    paths = complete_view_paths(scene)
    assert [path.name for path in paths] == ["f_000000.png", "f_000001.png", "f_000002.png"]
    conversation = multiview_conversation("Where?", len(paths), "Use images.")
    content = conversation[0]["content"]
    assert sum(item["type"] == "image" for item in content) == 3
    assert content[-1]["text"].endswith("Question: Where?")


def test_gemma4_conversation_carries_each_complete_frame_in_manifest_order() -> None:
    images = [Image.new("RGB", (8, 6), "red"), Image.new("RGB", (8, 6), "blue")]
    try:
        conversation = gemma4_multiview_conversation("Where?", images, "Use images.")
        content = conversation[0]["content"]
        assert [item["image"] for item in content[:-1]] == images
        assert [item["image"].size for item in content[:-1]] == [(8, 6), (8, 6)]
        assert content[-1]["text"].endswith("Question: Where?")
    finally:
        for image in images:
            image.close()


def test_direct_multiview_backend_resolves_gemma4_explicitly() -> None:
    assert (
        direct_multiview_backend({"backend": "gemma4", "model_id": "google/gemma-4-E2B-it"})
        == "gemma4"
    )
    assert (
        direct_multiview_backend({"backend": "auto", "model_id": "google/gemma-4-E2B-it"})
        == "gemma4"
    )


def test_local_vlm_answerer_batches_complete_images_in_one_processor_call() -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

    class Processor:
        tokenizer = Tokenizer()

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def apply_chat_template(self, conversation, **kwargs):
            assert sum(item["type"] == "image" for item in conversation[0]["content"]) == 2
            return "<image><image>Question"

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "input_ids": torch.tensor([[10, 11, 12]]),
                "pixel_values": torch.zeros(1, 2, 3, 8, 8),
            }

        @staticmethod
        def decode(tokens, **kwargs):
            assert tokens.tolist() == [42]
            return "yes"

    class Model:
        @staticmethod
        def generate(**kwargs):
            assert kwargs["pixel_values"].shape[1] == 2
            return torch.tensor([[10, 11, 12, 42]])

    processor = Processor()
    answerer = LocalMultiViewAnswerer(
        model=Model(),
        processor=processor,
        device=torch.device("cpu"),
        dtype=torch.float32,
        system_prompt="Use images.",
        max_answer_tokens=4,
        resize_longest_edge=512,
    )
    images = [Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")]
    try:
        assert answerer(images, "Is it red?") == "yes"
    finally:
        for image in images:
            image.close()
    assert len(processor.calls) == 1
    assert len(processor.calls[0]["images"]) == 2
    assert processor.calls[0]["images_kwargs"]["do_image_splitting"] is False


def test_gemma4_answerer_uses_official_interleaved_processor_without_crop_kwargs() -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class Processor:
        tokenizer = Tokenizer()

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def apply_chat_template(self, conversation, **kwargs):
            self.calls.append({"conversation": conversation, **kwargs})
            images = [
                item["image"] for item in conversation[0]["content"] if item["type"] == "image"
            ]
            assert [image.size for image in images] == [(8, 6), (8, 6)]
            return {
                "input_ids": torch.tensor([[10, 11, 12]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.zeros(2, 9, 4),
                "image_position_ids": torch.zeros(2, 9, 2, dtype=torch.long),
            }

        @staticmethod
        def decode(tokens, *, skip_special_tokens):
            assert tokens.tolist() == [42]
            return "yes" if skip_special_tokens else "yes<turn|>"

        @staticmethod
        def parse_response(response, *, prefix):
            assert response == "yes<turn|>"
            assert prefix.tolist() == [[10, 11, 12]]
            return {"role": "assistant", "content": "yes"}

    class Model:
        @staticmethod
        def generate(**kwargs):
            assert kwargs["pixel_values"].shape[0] == 2
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 4
            assert "eos_token_id" not in kwargs
            return torch.tensor([[10, 11, 12, 42]])

    processor = Processor()
    answerer = LocalMultiViewAnswerer(
        model=Model(),
        processor=processor,
        device=torch.device("cpu"),
        dtype=torch.float32,
        system_prompt="Use images.",
        max_answer_tokens=4,
        resize_longest_edge=None,
        backend="gemma4",
        enable_thinking=False,
    )
    images = [Image.new("RGB", (8, 6), "red"), Image.new("RGB", (8, 6), "blue")]
    try:
        assert answerer(images, "Is anything red?") == "yes"
    finally:
        for image in images:
            image.close()
    assert len(processor.calls) == 1
    call = processor.calls[0]
    assert call["tokenize"] is True
    assert call["return_dict"] is True
    assert call["enable_thinking"] is False
    assert "images_kwargs" not in call


def test_direct_runner_never_reads_oracle_and_passes_all_views(tmp_path: Path, monkeypatch) -> None:
    config = load_config()
    scene_id = "scene_999001"
    rendered = tmp_path / "data" / "rendered" / scene_id
    (rendered / "rgb").mkdir(parents=True)
    frames: list[dict[str, Any]] = []
    for index in range(2):
        relative = f"rgb/f_{index:06d}.png"
        Image.new("RGB", (8, 8), "red").save(rendered / relative)
        frames.append(
            {
                "camera_id": f"c_{index:06d}",
                "camera_to_world": [[1, 0, 0, 0]] * 4,
                "depth_path": f"depth/f_{index:06d}.npy",
                "frame_id": f"f_{index:06d}",
                "frame_number": index,
                "intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "rgb_path": relative,
            }
        )
    (rendered / "manifest.json").write_text(
        json.dumps(
            {
                "coordinate_system": {
                    "world": "x_right_y_forward_z_up",
                    "camera": "x_right_y_down_z_forward",
                    "depth": "axial_camera_z",
                    "units": "meters",
                },
                "scene_id": scene_id,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "references.jsonl"
    selected_reference = tmp_path / "selected-references.jsonl"
    output = tmp_path / "predictions.jsonl"
    _reference(reference, scene_id)
    monkeypatch.setattr(
        "semantic_3d_chat.evaluation.direct_multiview_baseline.PROJECT_ROOT", tmp_path
    )
    config["paths"]["data_root"] = "data"
    config["evaluation"]["baselines"]["direct_multiview"]["max_views"] = 24
    observed: dict[str, Any] = {}

    def answerer(images: list[Image.Image], question: str) -> str:
        observed.update(count=len(images), question=question)
        return "red"

    report = run_direct_multiview_baseline(
        config,
        reference,
        output,
        answerer=answerer,
        selected_references_path=selected_reference,
    )
    assert observed == {"count": 2, "question": "What color is the cube?"}
    assert report["view_counts"] == [2]
    assert report["manifest_view_counts"] == [2]
    assert report["evaluation_only"] is True
    assert report["primary_path_eligible"] is False
    assert report["prohibited_primary_substitute"] is True
    assert report["excluded_input_modalities"] == [
        "depth",
        "oracle_metadata",
        "segmentation",
        "scene_caption",
    ]
    assert selected_reference.read_text(encoding="utf-8") == reference.read_text(encoding="utf-8")
    assert "oracle" not in output.read_text(encoding="utf-8")

    config["evaluation"]["baselines"]["direct_multiview"].update(
        max_views=1,
        require_all_manifest_views=True,
    )
    with pytest.raises(ValueError, match="requires every manifest view"):
        run_direct_multiview_baseline(
            config,
            reference,
            tmp_path / "truncated.jsonl",
            answerer=answerer,
        )


def test_gemma4_diverse_baseline_is_local_only_all_views_and_validation_smoke() -> None:
    config = load_config("configs/experiments/gemma4_diverse20_direct_multiview_baseline.yaml")
    baseline = config["evaluation"]["baselines"]["direct_multiview"]
    assert baseline["backend"] == "gemma4"
    assert baseline["model_id"] == "google/gemma-4-E2B-it"
    assert baseline["revision"] == "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    assert baseline["max_views"] == 24
    assert baseline["require_all_manifest_views"] is True
    assert baseline["require_local_files_only"] is True
    assert baseline["scene_cache"] == "gemma4_decoder_kv_scene_prefix_v1"
    assert config["experiment"]["primary_path_eligible"] is False
    assert config["experiment"]["final_test_deferred"] is True

    makefile = Path(__file__).parents[1].joinpath("Makefile").read_text(encoding="utf-8")
    target = makefile.split("evaluate-direct-images-gemma4-smoke:", 1)[1].split("\n\n", 1)[0]
    assert ".venv-gemma4/bin/python" in makefile
    assert "GEMMA4_DIRECT_BASELINE_REFERENCES := data_diverse20/qa/validation.jsonl" in makefile
    assert '"$(GEMMA4_DIRECT_BASELINE_REFERENCES)"' in target
    assert "--limit $(GEMMA4_DIRECT_BASELINE_SMOKE_LIMIT)" in target
    assert "scene_000025" not in target


def test_gemma4_direct_baseline_refuses_download_enabled_mode(tmp_path: Path) -> None:
    config = load_config("configs/experiments/gemma4_diverse20_direct_multiview_baseline.yaml")
    reference = tmp_path / "references.jsonl"
    output = tmp_path / "predictions.jsonl"
    _reference(reference)
    with pytest.raises(ValueError, match="local-files-only"):
        run_direct_multiview_baseline(
            config,
            reference,
            output,
            answerer=lambda _images, _question: "red",
            local_files_only=False,
        )


def test_primary_chat_does_not_import_evaluation_baselines() -> None:
    runtime_source = (
        Path(__file__).parents[1] / "src" / "semantic_3d_chat" / "chat" / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "oracle_text_baseline" not in runtime_source
    assert "direct_multiview_baseline" not in runtime_source
