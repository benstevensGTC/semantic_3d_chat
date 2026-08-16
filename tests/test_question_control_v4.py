from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v4 import (
    SceneConditionedGateTeacherBasisControlV4,
)
from semantic_3d_chat.training.question_control_v4_checkpoint import (
    inherited_value_state_sha256,
    save_v4_control_checkpoint,
)

_BASE_CHECKPOINT_SHA256 = "a" * 64
_RUNTIME_CONFIG_SHA256 = "b" * 64
_SOURCE_V60_SHA256 = "c" * 64


def _v60_source() -> TeacherBasisFullSceneQuestionControlV3:
    torch.manual_seed(604)
    basis = torch.linalg.qr(torch.randn(12, 6)).Q.T.contiguous()
    return TeacherBasisFullSceneQuestionControlV3(
        12,
        basis,
        control_tokens=2,
        expected_environment_latents=5,
        moment_count=3,
        interaction_dim=4,
        trunk_dim=7,
        maximum_control_rms=0.4,
        initial_control_rms=0.1,
    ).eval()


def _v4() -> tuple[
    TeacherBasisFullSceneQuestionControlV3,
    SceneConditionedGateTeacherBasisControlV4,
]:
    source = _v60_source()
    return source, SceneConditionedGateTeacherBasisControlV4.from_v60(
        source, gate_hidden_dim=3
    ).eval()


def _set_two_scene_separator(
    module: SceneConditionedGateTeacherBasisControlV4,
    first_feature: torch.Tensor,
    second_feature: torch.Tensor,
) -> None:
    """Make the tiny gate separate two fixed scene-question features exactly."""

    layer_norm = module.scene_question_gate[0]
    hidden = module.scene_question_gate[1]
    output = module.scene_question_gate[3]
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.zero_()
        first = layer_norm(first_feature).squeeze(0)
        second = layer_norm(second_feature).squeeze(0)
        direction = first - second
        assert direction.norm().item() > 1e-4
        midpoint = 0.5 * (torch.dot(direction, first) + torch.dot(direction, second))
        hidden.weight.zero_()
        hidden.bias.zero_()
        hidden.weight[0].copy_(direction)
        hidden.bias[0].copy_(-midpoint)
        output.weight.zero_()
        output.bias.zero_()
        output.weight[0, 0] = 10.0


def test_v4_exactly_inherits_and_freezes_v60_except_for_new_gate() -> None:
    source, candidate = _v4()
    source_state = {name: value.clone() for name, value in source.state_dict().items()}

    assert set(source_state) == set(candidate.inherited_state_names)
    assert all(
        torch.equal(value, candidate.state_dict()[name])
        for name, value in source_state.items()
    )
    assert candidate.inherited_v60_state_frozen is True
    assert {
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    } == {
        name
        for name, _parameter in candidate.named_parameters()
        if name.startswith("scene_question_gate.")
    }

    inherited_before = inherited_value_state_sha256(candidate)
    optimizer = torch.optim.SGD(candidate.scene_question_gate.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    candidate.scene_question_gate(torch.randn(2, 14)).square().mean().backward()
    optimizer.step()

    assert inherited_value_state_sha256(candidate) == inherited_before
    assert all(
        torch.equal(value, candidate.state_dict()[name])
        for name, value in source_state.items()
    )


def test_v4_values_are_bit_identical_to_v60_while_route_changes_by_scene() -> None:
    source, candidate = _v4()
    torch.manual_seed(614)
    first_scene = torch.randn(1, 7, 12)
    second_scene = torch.randn(1, 7, 12)
    question = torch.randn(1, 4, 12)
    first_signature = source.encode_scene(first_scene)
    second_signature = source.encode_scene(second_scene)
    normalized_question = candidate.normalized_question(question)
    first_feature = candidate._value_trunk(
        first_signature, normalized_question
    ).flatten(1)
    second_feature = candidate._value_trunk(
        second_signature, normalized_question
    ).flatten(1)
    _set_two_scene_separator(candidate, first_feature, second_feature)

    for signature in (first_signature, second_signature):
        inherited = source.forward_from_signature(signature, question)
        observed = candidate.forward_from_signature(signature, question)
        assert torch.equal(observed.control_tokens, inherited.control_tokens)
        assert torch.equal(
            observed.coefficient_directions, inherited.coefficient_directions
        )
        assert torch.equal(observed.control_rms, inherited.control_rms)

    first_route = candidate.forward_from_signature(
        first_signature, question
    ).gate_probabilities.item()
    second_route = candidate.forward_from_signature(
        second_signature, question
    ).gate_probabilities.item()
    assert first_route > candidate.gate_threshold
    assert second_route < candidate.gate_threshold
    audit = candidate.audit()
    assert audit.gate_scene_question_conditioned is True
    assert audit.inherited_v60_state_frozen is True
    assert audit.question_dependent_scene_retrieval is False
    assert audit.softmax_scene_attention_used is False


def test_v4_schema4_checkpoint_round_trips_with_strict_attestations(
    tmp_path: Path,
) -> None:
    _source, candidate = _v4()
    inherited_sha256 = inherited_value_state_sha256(candidate)
    checkpoint = tmp_path / "checkpoint"

    result = save_v4_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256=_BASE_CHECKPOINT_SHA256,
        base_runtime_config_sha256=_RUNTIME_CONFIG_SHA256,
        source_v60_checkpoint_sha256=_SOURCE_V60_SHA256,
        expected_inherited_state_sha256=inherited_sha256,
    )

    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    loaded, metadata = _load_control_head(
        checkpoint, hidden_size=12, device=torch.device("cpu")
    )
    assert isinstance(loaded, SceneConditionedGateTeacherBasisControlV4)
    assert metadata["schema_version"] == 4
    assert metadata["source_v60_checkpoint_sha256"] == _SOURCE_V60_SHA256
    assert metadata["inherited_value_state_sha256"] == inherited_sha256
    assert metadata["only_gate_trainable"] is True
    assert metadata["gate_scene_question_conditioned"] is True
    assert metadata["environmental_text_inputs"] == []
    assert metadata["question_dependent_scene_retrieval"] is False
    assert result["inherited_value_state_sha256"] == inherited_sha256
    assert all(
        torch.equal(candidate.state_dict()[name], loaded.state_dict()[name])
        for name in candidate.state_dict()
    )

    metadata_path = checkpoint / "runtime_metadata.json"
    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered["only_gate_trainable"] = False
    metadata_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="V4 question-control runtime contract mismatch"):
        _load_control_head(checkpoint, hidden_size=12, device=torch.device("cpu"))


def test_v4_checkpoint_refuses_changed_or_unfrozen_inherited_state(tmp_path: Path) -> None:
    _source, candidate = _v4()
    inherited_sha256 = inherited_value_state_sha256(candidate)
    candidate.scene_projection.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="every inherited V60 parameter is frozen"):
        save_v4_control_checkpoint(
            tmp_path / "unfrozen",
            control=candidate,
            base_checkpoint_sha256=_BASE_CHECKPOINT_SHA256,
            base_runtime_config_sha256=_RUNTIME_CONFIG_SHA256,
            source_v60_checkpoint_sha256=_SOURCE_V60_SHA256,
            expected_inherited_state_sha256=inherited_sha256,
        )

    candidate.freeze_inherited_v60_state()
    with torch.no_grad():
        candidate.scene_projection.weight[0, 0].add_(1.0)
    with pytest.raises(ValueError, match="inherited V60 state changed"):
        save_v4_control_checkpoint(
            tmp_path / "changed",
            control=candidate,
            base_checkpoint_sha256=_BASE_CHECKPOINT_SHA256,
            base_runtime_config_sha256=_RUNTIME_CONFIG_SHA256,
            source_v60_checkpoint_sha256=_SOURCE_V60_SHA256,
            expected_inherited_state_sha256=inherited_sha256,
        )
