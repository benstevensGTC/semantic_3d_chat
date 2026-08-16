from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.scene_encoder.question_control_v2 import (
    BoundedFullSceneQuestionControlV2,
)
from semantic_3d_chat.training.question_control_v2_checkpoint import (
    save_v2_control_checkpoint,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64


def _module() -> BoundedFullSceneQuestionControlV2:
    torch.manual_seed(7)
    return BoundedFullSceneQuestionControlV2(
        16,
        control_tokens=2,
        expected_environment_latents=9,
        moment_count=4,
        interaction_dim=3,
        output_rank=5,
        maximum_control_rms=0.4,
        initial_control_rms=0.15,
    )


def test_v2_caches_environment_only_signature_and_controls_are_bounded() -> None:
    module = _module()
    scene = torch.randn(1, 11, 16)
    question = torch.randn(1, 4, 16)
    with torch.inference_mode():
        signature = module.encode_scene(scene)
        output = module.forward_from_signature(signature, question)
    assert signature.shape == (1, 4, 16)
    # Protocol boundary changes do not affect the environment signature.
    changed_boundaries = scene.clone()
    changed_boundaries[:, 0] += 100.0
    changed_boundaries[:, -1] -= 100.0
    assert torch.equal(signature, module.encode_scene(changed_boundaries))
    assert output.control_rms.max().item() == pytest.approx(0.15, abs=1e-5)
    assert output.gate_probabilities.item() < module.gate_threshold
    audit = module.audit()
    assert audit.scene_token_count == 11
    assert audit.environment_latent_count == 9
    assert audit.scene_moment_count == 4
    assert audit.every_environment_latent_influenced_signature is True
    assert audit.question_dependent_scene_retrieval is False
    assert audit.softmax_scene_attention_used is False
    assert audit.control_used is False


def test_v2_residual_depends_on_scene_and_question_bilinearly() -> None:
    module = _module()
    scene = torch.randn(1, 9, 16)
    scene = torch.cat((torch.randn(1, 1, 16), scene, torch.randn(1, 1, 16)), dim=1)
    question = torch.randn(1, 3, 16)
    first = module(scene, question).control_tokens
    changed_scene = module(scene.flip(1), question).control_tokens
    changed_question = module(scene, question + torch.randn_like(question)).control_tokens
    assert not torch.equal(first, changed_scene)
    assert not torch.equal(first, changed_question)
    assert module.trainable_parameter_count < 1_000_000


def test_v2_checkpoint_round_trips_with_strict_sanitized_inventory(
    tmp_path: Path,
) -> None:
    module = _module()
    checkpoint = tmp_path / "checkpoint"
    save_v2_control_checkpoint(
        checkpoint,
        control=module,
        base_checkpoint_sha256=_A,
        base_runtime_config_sha256=_B,
        source_control_checkpoint_sha256=_C,
    )
    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    metadata = json.loads((checkpoint / "runtime_metadata.json").read_text())
    assert metadata["architecture"] == "bounded_global_scene_question_control_v2"
    assert metadata["environmental_text_inputs"] == []
    assert metadata["exact_no_control_route"] is True
    serialized = json.dumps(metadata).casefold()
    assert "answer" not in serialized
    assert "caption" not in serialized
    assert "oracle" not in serialized

    loaded, loaded_metadata = _load_control_head(
        checkpoint, hidden_size=16, device=torch.device("cpu")
    )
    assert isinstance(loaded, BoundedFullSceneQuestionControlV2)
    assert loaded_metadata == metadata
    assert all(
        torch.equal(module.state_dict()[name], loaded.state_dict()[name])
        for name in module.state_dict()
    )

    with pytest.raises(FileExistsError):
        save_v2_control_checkpoint(
            checkpoint,
            control=module,
            base_checkpoint_sha256=_A,
            base_runtime_config_sha256=_B,
            source_control_checkpoint_sha256=_C,
        )
