from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.training import train_question_control_v58 as v58
from semantic_3d_chat.training.soft_prompt_teacher_v58 import (
    SoftPromptTarget,
    load_teacher_artifact,
    normalized_prompt_distillation_loss,
    pair_delta_distillation_loss,
    save_teacher_artifact,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64


def test_v58_generation_helper_preserves_literal_no_control_path(monkeypatch) -> None:
    class Backend:
        def prepare(self, _prefix, _ids, **kwargs):
            assert kwargs["control_tokens"] is None
            return object()

        def generate(self, *_args, **_kwargs):
            return torch.tensor([[7]])

    class Tokenizer:
        @staticmethod
        def decode(_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return "base answer"

    language = type(
        "Language",
        (),
        {
            "prefix_backend": Backend(),
            "backend_name": "gemma4",
            "tokenizer": Tokenizer(),
            "device": torch.device("cpu"),
        },
    )()
    runtime = type(
        "Runtime",
        (),
        {
            "language": language,
            "config": {
                "language": {
                    "system_prompt": "system",
                    "max_answer_tokens": 2,
                },
                "scene_encoder": {
                    "scene_prefix_after_bos": True,
                    "scene_boundary_mode": "none",
                },
            },
            "_eos_token_ids": lambda self: (1,),
        },
    )()
    monkeypatch.setattr(v58, "prompt_token_ids", lambda *_args, **_kwargs: torch.ones(1, 1))
    monkeypatch.setattr(v58, "scene_prefix_after_bos_setting", lambda _config: True)
    monkeypatch.setattr(v58, "scene_boundary_mode_setting", lambda _config: "none")

    assert (
        v58._generate_with_control(
            runtime=runtime,
            scene_prefix=torch.zeros(1, 3, 4),
            question="question",
            control_tokens=None,
        )
        == "base answer"
    )


def test_v58_prompt_distillation_is_zero_only_on_matching_tokens() -> None:
    target = torch.randn(3, 4, 8)
    predicted = target.clone().requires_grad_(True)
    loss, diagnostics = normalized_prompt_distillation_loss(predicted, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["normalized_mse"].item() == pytest.approx(0.0)
    assert diagnostics["mean_token_cosine"].item() == pytest.approx(1.0, abs=1e-6)

    collapsed = torch.zeros_like(target, requires_grad=True)
    collapsed_loss, collapsed_diagnostics = normalized_prompt_distillation_loss(
        collapsed, target
    )
    assert collapsed_loss.item() > 1.0
    assert collapsed_diagnostics["normalized_mse"].item() == pytest.approx(1.0)
    collapsed_loss.backward()
    assert collapsed.grad is not None
    assert torch.isfinite(collapsed.grad).all()


def test_v58_pair_delta_loss_detects_collapsed_and_reversed_sides() -> None:
    target = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    exact_loss, exact = pair_delta_distillation_loss(target.clone(), target)
    assert exact_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert exact["delta_cosine"].item() == pytest.approx(1.0, abs=1e-6)

    collapsed = torch.zeros_like(target, requires_grad=True)
    collapsed_loss, collapsed_metrics = pair_delta_distillation_loss(collapsed, target)
    assert collapsed_loss.item() == pytest.approx(2.0)
    assert collapsed_metrics["predicted_delta_rms"].item() == pytest.approx(0.0)

    reversed_loss, reversed_metrics = pair_delta_distillation_loss(
        target.flip(0), target
    )
    assert reversed_loss.item() > collapsed_loss.item()
    assert reversed_metrics["delta_cosine"].item() == pytest.approx(-1.0, abs=1e-6)


def test_v58_teacher_artifact_is_numeric_opaque_and_training_only(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "teachers"
    targets = [
        SoftPromptTarget(
            "scene_000031",
            "q_000001",
            "changed_teacher",
            torch.randn(1, 4, 8),
        ),
        SoftPromptTarget(
            "scene_000032",
            "q_000002",
            "retention_baseline",
            torch.randn(1, 4, 8),
        ),
    ]
    hashes = save_teacher_artifact(
        destination,
        targets=targets,
        base_checkpoint_sha256=_A,
        base_runtime_config_sha256=_B,
        source_control_checkpoint_sha256=_C,
    )
    assert set(hashes) == {"metadata_sha256", "weights_sha256"}
    assert {item.name for item in destination.iterdir()} == {
        "metadata.json",
        "teachers.safetensors",
    }

    loaded, metadata = load_teacher_artifact(destination)
    assert set(loaded) == {target.key for target in targets}
    assert metadata["runtime_load_permitted"] is False
    assert metadata["environmental_text_inputs"] == []
    serialized = json.dumps(metadata, sort_keys=True).casefold()
    assert "question" not in serialized.replace("question_id", "")
    assert "answer" not in serialized
    assert "caption" not in serialized
    assert "oracle" not in serialized

    with pytest.raises(FileExistsError):
        save_teacher_artifact(
            destination,
            targets=targets,
            base_checkpoint_sha256=_A,
            base_runtime_config_sha256=_B,
            source_control_checkpoint_sha256=_C,
        )


def test_v58_teacher_target_rejects_textual_or_invalid_identity() -> None:
    with pytest.raises(ValueError, match="opaque"):
        SoftPromptTarget("room with chair", "q", "changed_teacher", torch.ones(1, 2, 3))
    with pytest.raises(ValueError, match="shape"):
        SoftPromptTarget("scene_000031", "q", "changed_teacher", torch.ones(2, 3))
