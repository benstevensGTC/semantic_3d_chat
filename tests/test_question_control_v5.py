from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v5 import (
    FactorizedRouteFeaturesV5,
    NormalizedFactorizedSceneQuestionControlV5,
)
from semantic_3d_chat.training.question_control_v5_checkpoint import (
    inherited_v60_state_sha256,
    save_v5_control_checkpoint,
)

_BASE_CHECKPOINT_SHA256 = "a" * 64
_RUNTIME_CONFIG_SHA256 = "b" * 64
_SOURCE_V60_SHA256 = "c" * 64


def _v60_source() -> TeacherBasisFullSceneQuestionControlV3:
    torch.manual_seed(605)
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


def _v5() -> tuple[
    TeacherBasisFullSceneQuestionControlV3,
    NormalizedFactorizedSceneQuestionControlV5,
]:
    source = _v60_source()
    return source, NormalizedFactorizedSceneQuestionControlV5.from_v60(
        source, route_factor_rank=5
    ).eval()


def _two_scenes_and_question(
    source: TeacherBasisFullSceneQuestionControlV3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(615)
    first_scene = source.encode_scene(torch.randn(1, 7, 12))
    second_scene = source.encode_scene(torch.randn(1, 7, 12))
    question = torch.randn(1, 4, 12)
    return first_scene, second_scene, question


def _set_two_scene_separator(
    module: NormalizedFactorizedSceneQuestionControlV5,
    first_scene: torch.Tensor,
    second_scene: torch.Tensor,
) -> None:
    """Make the scene tower route the first signature on and the second off."""

    head = module.factorized_route
    first = module.encode_route_scene(first_scene).squeeze(0)
    second = module.encode_route_scene(second_scene).squeeze(0)
    direction = first - second
    assert direction.norm().item() > 1e-4
    midpoint = 0.5 * (torch.dot(direction, first) + torch.dot(direction, second))
    with torch.no_grad():
        head.bilinear_diagonal.zero_()
        head.question_calibration.weight.zero_()
        head.scene_calibration.weight.copy_(20.0 * direction.unsqueeze(0))
        head.route_bias.copy_(-20.0 * midpoint)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _saved_checkpoint(tmp_path: Path) -> tuple[
    Path,
    NormalizedFactorizedSceneQuestionControlV5,
    str,
]:
    _source, candidate = _v5()
    inherited_sha256 = inherited_v60_state_sha256(candidate)
    checkpoint = tmp_path / "checkpoint"
    save_v5_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256=_BASE_CHECKPOINT_SHA256,
        base_runtime_config_sha256=_RUNTIME_CONFIG_SHA256,
        source_v60_checkpoint_sha256=_SOURCE_V60_SHA256,
        expected_inherited_state_sha256=inherited_sha256,
    )
    return checkpoint, candidate, inherited_sha256


def test_v5_exactly_copies_and_freezes_every_inherited_v60_tensor() -> None:
    source, candidate = _v5()
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
        if name.startswith("factorized_route.")
    }


def test_v5_gate_only_backward_and_optimizer_preserve_inherited_hash() -> None:
    _source, candidate = _v5()
    first_scene, second_scene, question = _two_scenes_and_question(candidate)
    signatures = torch.cat((first_scene, second_scene), dim=0)
    questions = question.expand(2, -1, -1).clone()
    inherited_before = inherited_v60_state_sha256(candidate)
    gate_before = {
        name: value.detach().clone()
        for name, value in candidate.state_dict().items()
        if name.startswith("factorized_route.")
    }

    optimizer = torch.optim.SGD(candidate.factorized_route.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    logits = candidate.route_logits_from_signature(signatures, questions)
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([1.0, 0.0])
    ).backward()

    assert all(
        parameter.grad is None
        for name, parameter in candidate.named_parameters()
        if not name.startswith("factorized_route.")
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum().item() > 0.0
        for parameter in candidate.factorized_route.parameters()
    )
    optimizer.step()

    assert inherited_v60_state_sha256(candidate) == inherited_before
    assert any(
        not torch.equal(value, candidate.state_dict()[name])
        for name, value in gate_before.items()
    )


def test_v5_route_features_are_separate_normalized_and_cache_equivalent() -> None:
    source, candidate = _v5()
    first_scene, second_scene, question = _two_scenes_and_question(source)
    other_question = question.flip(1)

    question_factor = candidate.encode_route_question(question)
    first_scene_factor = candidate.encode_route_scene(first_scene)
    features = candidate.route_features_from_signature(first_scene, question)
    changed_scene_features = candidate.route_features_from_signature(
        second_scene, question
    )
    changed_question_features = candidate.route_features_from_signature(
        first_scene, other_question
    )

    assert isinstance(features, FactorizedRouteFeaturesV5)
    assert features.question.shape == (1, candidate.route_factor_rank)
    assert features.scene.shape == (1, candidate.route_factor_rank)
    assert torch.equal(features.question, question_factor)
    assert torch.equal(features.scene, first_scene_factor)
    assert torch.allclose(features.question.norm(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(features.scene.norm(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.equal(features.question, changed_scene_features.question)
    assert torch.equal(features.scene, changed_question_features.scene)
    assert not torch.equal(features.scene, changed_scene_features.scene)
    assert not torch.equal(features.question, changed_question_features.question)
    assert torch.equal(
        candidate.route_logits_from_features(features),
        candidate.route_logits_from_signature(first_scene, question),
    )
    with pytest.raises(TypeError, match="wrong type"):
        candidate.route_logits_from_features(  # type: ignore[arg-type]
            (question_factor, first_scene_factor)
        )


def test_v5_values_are_bit_identical_while_same_question_routes_by_scene() -> None:
    source, candidate = _v5()
    first_scene, second_scene, question = _two_scenes_and_question(source)
    _set_two_scene_separator(candidate, first_scene, second_scene)

    probabilities: list[float] = []
    for signature in (first_scene, second_scene):
        inherited = source.forward_from_signature(signature, question)
        observed = candidate.forward_from_signature(signature, question)
        assert torch.equal(observed.control_tokens, inherited.control_tokens)
        assert torch.equal(
            observed.coefficient_directions, inherited.coefficient_directions
        )
        assert torch.equal(observed.control_rms, inherited.control_rms)
        probabilities.append(observed.gate_probabilities.item())

    assert probabilities[0] > candidate.gate_threshold
    assert probabilities[1] < candidate.gate_threshold


def test_v5_batch_one_runtime_audit_attests_factorized_global_route() -> None:
    source, candidate = _v5()
    first_scene, _second_scene, question = _two_scenes_and_question(source)
    output = candidate.forward_from_signature(first_scene, question)
    audit = candidate.audit()

    assert audit.scene_token_count == 7
    assert audit.environment_latent_count == 5
    assert audit.control_token_count == 2
    assert audit.scene_moment_count == 3
    assert audit.output_basis_rank == 6
    assert audit.route_factor_rank == 5
    assert audit.every_environment_latent_influenced_signature is True
    assert audit.control_values_scene_question_bilinear is True
    assert audit.gate_scene_question_conditioned is True
    assert audit.separate_question_scene_route_projections is True
    assert audit.normalized_route_factors is True
    assert audit.low_rank_bilinear_route is True
    assert audit.route_uses_inherited_value_trunk is False
    assert audit.inherited_v60_state_frozen is True
    assert audit.question_dependent_scene_retrieval is False
    assert audit.softmax_scene_attention_used is False
    assert audit.gate_probability == pytest.approx(output.gate_probabilities.item())
    assert audit.control_used is (
        output.gate_probabilities.item() >= candidate.gate_threshold
    )
    assert audit.maximum_control_rms == pytest.approx(
        output.control_rms.max().item()
    )


def test_v5_schema5_checkpoint_round_trips_with_strict_attestations(
    tmp_path: Path,
) -> None:
    checkpoint, candidate, inherited_sha256 = _saved_checkpoint(tmp_path)

    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    loaded, metadata = _load_control_head(
        checkpoint, hidden_size=12, device=torch.device("cpu")
    )
    assert isinstance(loaded, NormalizedFactorizedSceneQuestionControlV5)
    assert metadata["schema_version"] == 5
    assert metadata["architecture"] == "normalized_factorized_scene_question_route_v5"
    assert metadata["source_v60_checkpoint_sha256"] == _SOURCE_V60_SHA256
    assert metadata["inherited_value_state_sha256"] == inherited_sha256
    assert metadata["only_factorized_gate_trainable"] is True
    assert metadata["gate_scene_question_conditioned"] is True
    assert metadata["separate_question_scene_route_projections"] is True
    assert metadata["normalized_route_factors"] is True
    assert metadata["low_rank_bilinear_route"] is True
    assert metadata["route_uses_inherited_value_trunk"] is False
    assert metadata["environmental_text_inputs"] == []
    assert metadata["question_dependent_scene_retrieval"] is False
    assert metadata["route_is_environmental_retrieval"] is False
    assert loaded.inherited_v60_state_frozen is True
    assert all(
        torch.equal(candidate.state_dict()[name], loaded.state_dict()[name])
        for name in candidate.state_dict()
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "only_factorized_gate_trainable",
            False,
            "V5 question-control runtime contract mismatch",
        ),
        (
            "normalized_route_factors",
            False,
            "V5 question-control runtime contract mismatch",
        ),
        (
            "route_uses_inherited_value_trunk",
            True,
            "V5 question-control runtime contract mismatch",
        ),
    ],
)
def test_v5_loader_rejects_tampered_contract_attestations(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    checkpoint, _candidate, _inherited_sha256 = _saved_checkpoint(tmp_path)
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_control_head(checkpoint, hidden_size=12, device=torch.device("cpu"))


def test_v5_loader_rejects_tampered_inherited_state_even_with_new_weights_hash(
    tmp_path: Path,
) -> None:
    checkpoint, _candidate, _inherited_sha256 = _saved_checkpoint(tmp_path)
    weights_path = checkpoint / "control.safetensors"
    metadata_path = checkpoint / "runtime_metadata.json"
    state = load_file(str(weights_path), device="cpu")
    state["scene_projection.weight"][0, 0].add_(1.0)
    save_file(state, weights_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["weights_sha256"] = _sha256(weights_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="V5 inherited V60 value state digest changed"):
        _load_control_head(checkpoint, hidden_size=12, device=torch.device("cpu"))


def test_v5_checkpoint_refuses_changed_or_unfrozen_inherited_state(
    tmp_path: Path,
) -> None:
    _source, candidate = _v5()
    inherited_sha256 = inherited_v60_state_sha256(candidate)
    candidate.scene_projection.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="every inherited V60 parameter is frozen"):
        save_v5_control_checkpoint(
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
        save_v5_control_checkpoint(
            tmp_path / "changed",
            control=candidate,
            base_checkpoint_sha256=_BASE_CHECKPOINT_SHA256,
            base_runtime_config_sha256=_RUNTIME_CONFIG_SHA256,
            source_v60_checkpoint_sha256=_SOURCE_V60_SHA256,
            expected_inherited_state_sha256=inherited_sha256,
        )
