from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
    teacher_output_basis,
)
from semantic_3d_chat.training.question_control_v3_checkpoint import (
    save_v3_control_checkpoint,
)

_A = "a" * 64
_B = "b" * 64


def _module() -> TeacherBasisFullSceneQuestionControlV3:
    torch.manual_seed(9)
    basis = torch.linalg.qr(torch.randn(16, 8)).Q.T.contiguous()
    return TeacherBasisFullSceneQuestionControlV3(
        16,
        basis,
        control_tokens=2,
        expected_environment_latents=9,
        moment_count=4,
        interaction_dim=3,
        trunk_dim=6,
        maximum_control_rms=0.4,
        initial_control_rms=0.15,
    )


def test_teacher_basis_is_orthogonal_and_reconstructs_full_rank_targets() -> None:
    torch.manual_seed(11)
    targets = torch.randn(3, 2, 16)
    basis = teacher_output_basis(targets)
    assert basis.shape == (6, 16)
    assert torch.allclose(basis @ basis.T, torch.eye(6), atol=1e-5)
    flattened = targets.reshape(-1, 16)
    reconstructed = flattened @ basis.T @ basis
    cosine = F.cosine_similarity(flattened, reconstructed, dim=-1)
    assert cosine.min().item() > 0.99999


def test_v3_signature_excludes_boundaries_and_control_values_are_scene_bilinear() -> None:
    module = _module()
    scene = torch.randn(1, 11, 16)
    question = torch.randn(1, 5, 16)
    signature = module.encode_scene(scene)
    changed_boundaries = scene.clone()
    changed_boundaries[:, 0] += 100.0
    changed_boundaries[:, -1] -= 100.0
    assert torch.equal(signature, module.encode_scene(changed_boundaries))
    output = module.forward_from_signature(signature, question)
    assert output.control_tokens.shape == (1, 2, 16)
    assert output.coefficient_directions.shape == (1, 2, 8)
    assert output.control_rms.max().item() <= 0.40001
    changed_scene = module(scene[:, [0, *range(9, 0, -1), 10]], question)
    assert not torch.equal(output.control_tokens, changed_scene.control_tokens)
    audit = module.audit()
    assert audit.softmax_scene_attention_used is False
    assert audit.question_dependent_scene_retrieval is False
    assert audit.control_values_scene_question_bilinear is True
    assert audit.every_environment_latent_influenced_signature is True


def test_v3_prototype_route_transfers_nearby_question_features() -> None:
    module = _module()
    positive = F.normalize(torch.randn(3, 16), dim=-1)
    negative = F.normalize(torch.randn(4, 16), dim=-1)
    module.initialize_route_prototypes(positive, negative)
    positive_center = F.normalize(positive.mean(0), dim=0)
    negative_center = F.normalize(negative.mean(0), dim=0)
    assert module.route_logits_from_normalized_question(positive_center[None]).item() > 0
    assert module.route_logits_from_normalized_question(negative_center[None]).item() < 0


def test_v3_rejects_nonorthogonal_basis() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        TeacherBasisFullSceneQuestionControlV3(8, torch.ones(2, 8))


def test_v3_checkpoint_round_trips_as_strict_runtime_artifact(tmp_path) -> None:
    module = _module()
    checkpoint = tmp_path / "checkpoint"
    save_v3_control_checkpoint(
        checkpoint,
        control=module,
        base_checkpoint_sha256=_A,
        base_runtime_config_sha256=_B,
    )
    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    loaded, metadata = _load_control_head(
        checkpoint, hidden_size=16, device=torch.device("cpu")
    )
    assert isinstance(loaded, TeacherBasisFullSceneQuestionControlV3)
    assert metadata["environmental_text_inputs"] == []
    assert metadata["control_values_scene_question_bilinear"] is True
    assert all(
        torch.equal(module.state_dict()[name], loaded.state_dict()[name])
        for name in module.state_dict()
    )
