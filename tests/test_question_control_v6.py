from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.question_control_runtime import (
    QuestionControlledChatRuntime,
    _load_control_head,
)
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v6 import (
    MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
)
from semantic_3d_chat.training.question_control_v6_checkpoint import (
    save_v6_control_checkpoint,
    v6_value_state_sha256,
)


def _v65(seed: int = 606) -> TeacherBasisFullSceneQuestionControlV3:
    torch.manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(8, 4)).Q.T.contiguous()
    return TeacherBasisFullSceneQuestionControlV3(
        8,
        basis,
        control_tokens=2,
        expected_environment_latents=4,
        moment_count=2,
        interaction_dim=3,
        trunk_dim=5,
        maximum_control_rms=0.3,
        initial_control_rms=0.1,
    ).eval()


def _candidate(
    threshold: float = 0.01,
) -> MagnitudeGatedTeacherBasisFullSceneQuestionControlV6:
    return MagnitudeGatedTeacherBasisFullSceneQuestionControlV6.from_v65(
        _v65(),
        activation_rms_threshold=threshold,
    ).eval()


def test_v6_preserves_every_v65_value_tensor_and_value_forward_exactly() -> None:
    source = _v65()
    candidate = MagnitudeGatedTeacherBasisFullSceneQuestionControlV6.from_v65(source)
    prefix = torch.randn(3, 6, 8)
    question = torch.randn(3, 5, 8)
    with torch.inference_mode():
        source_output = source(prefix, question)
        candidate_output = candidate(prefix, question)

    assert set(source.state_dict()) == set(candidate.state_dict())
    assert all(
        torch.equal(source.state_dict()[name], candidate.state_dict()[name])
        for name in source.state_dict()
    )
    assert torch.equal(source_output.control_tokens, candidate_output.control_tokens)
    assert torch.equal(
        source_output.coefficient_directions,
        candidate_output.coefficient_directions,
    )
    assert torch.equal(source_output.control_rms, candidate_output.control_rms)


def test_v6_routes_from_maximum_token_rms_with_exact_threshold_boundary() -> None:
    candidate = _candidate(threshold=0.05)
    control_rms = torch.tensor(
        [[0.049, 0.001], [0.05, 0.001], [0.002, 0.08]],
        dtype=torch.float32,
    )

    assert torch.equal(
        candidate.activation_rms(control_rms),
        torch.tensor([0.049, 0.05, 0.08]),
    )
    assert (candidate.activation_rms(control_rms) >= 0.05).tolist() == [
        False,
        True,
        True,
    ]


def test_v6_diagnostic_probability_matches_logit_at_boundary() -> None:
    candidate = _candidate(threshold=0.05)
    with torch.no_grad():
        candidate.magnitude_output.weight.zero_()
        fraction = 0.05 / candidate.maximum_control_rms
        candidate.magnitude_output.bias.fill_(math.log(fraction / (1.0 - fraction)))
    output = candidate(torch.randn(1, 6, 8), torch.randn(1, 3, 8))

    assert torch.allclose(
        output.gate_probabilities,
        torch.sigmoid(output.gate_logits),
    )
    assert float(output.gate_probabilities[0].detach()) == pytest.approx(0.5, abs=1e-6)
    assert candidate.audit().control_used is True


def test_v6_audit_uses_same_production_rule_and_complete_scene_signature() -> None:
    candidate = _candidate(threshold=0.05)
    prefix = torch.randn(1, 6, 8)
    question = torch.randn(1, 3, 8)
    with torch.inference_mode():
        output = candidate(prefix, question)
    audit = candidate.audit()

    expected_activation = float(output.control_rms.max())
    assert audit.activation_rms == pytest.approx(expected_activation)
    assert audit.control_used is (expected_activation >= 0.05)
    assert audit.activation_rms_threshold == 0.05
    assert audit.exact_no_control_below_threshold is True
    assert audit.gate_scene_question_conditioned is True
    assert audit.every_environment_latent_influenced_signature is True
    assert audit.question_dependent_scene_retrieval is False


def test_v6_checkpoint_is_minimal_and_loader_roundtrips_contract(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    value_hash = v6_value_state_sha256(candidate)
    checkpoint = tmp_path / "v65"
    hashes = save_v6_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=value_hash,
        saved_runtime_training_gate_passed=True,
        saved_runtime_training_gate_attestation_sha256="4" * 64,
    )

    loaded, metadata = _load_control_head(
        checkpoint,
        hidden_size=8,
        device=torch.device("cpu"),
    )
    assert type(loaded) is MagnitudeGatedTeacherBasisFullSceneQuestionControlV6
    assert metadata["magnitude_gated_continuous_control"] is True
    assert metadata["exact_no_control_below_threshold"] is True
    assert metadata["activation_rms_aggregation"] == "maximum_over_control_tokens"
    assert metadata["activation_rms_threshold"] == 0.01
    assert metadata["saved_runtime_training_gate_required"] is True
    assert metadata["source_v65_training_fit_state_sha256"] == value_hash
    assert hashes["source_v65_value_state_sha256"] == value_hash
    assert hashes["source_v65_training_fit_state_sha256"] == value_hash
    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    assert all(
        torch.equal(candidate.state_dict()[name], loaded.state_dict()[name])
        for name in candidate.state_dict()
    )


def test_v6_saver_rejects_state_that_differs_from_authenticated_training_fit(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    with pytest.raises(ValueError, match="training-fit state changed"):
        save_v6_control_checkpoint(
            tmp_path / "candidate",
            control=candidate,
            base_checkpoint_sha256="1" * 64,
            base_runtime_config_sha256="2" * 64,
            expected_training_fit_state_sha256="3" * 64,
        )
    assert not (tmp_path / "candidate").exists()


def test_v6_public_loader_rejects_unsealed_training_stage(tmp_path: Path) -> None:
    candidate = _candidate()
    checkpoint = tmp_path / "staged"
    save_v6_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=v6_value_state_sha256(candidate),
    )

    with pytest.raises(ValueError, match="runtime contract mismatch"):
        _load_control_head(checkpoint, hidden_size=8, device=torch.device("cpu"))


def test_v6_loader_requires_saved_state_to_equal_training_fit_state(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    checkpoint = tmp_path / "candidate"
    save_v6_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=v6_value_state_sha256(candidate),
        saved_runtime_training_gate_passed=True,
        saved_runtime_training_gate_attestation_sha256="4" * 64,
    )
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_v65_training_fit_state_sha256"] = "3" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its V65 training fit"):
        _load_control_head(checkpoint, hidden_size=8, device=torch.device("cpu"))


def test_v6_runtime_returns_literal_none_below_magnitude_threshold() -> None:
    candidate = _candidate(threshold=0.05)
    with torch.no_grad():
        candidate.magnitude_output.weight.zero_()
        fraction = 0.01 / candidate.maximum_control_rms
        candidate.magnitude_output.bias.fill_(math.log(fraction / (1.0 - fraction)))
    embedding = torch.nn.Embedding(16, 8)

    class Tokenizer:
        def __call__(self, *_: object, **__: object) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([[4, 5]])}

    base = SimpleNamespace(
        language=SimpleNamespace(
            tokenizer=Tokenizer(),
            model=SimpleNamespace(get_input_embeddings=lambda: embedding),
            device=torch.device("cpu"),
        ),
        scene_prefix=torch.randn(1, 6, 8),
        scene_prefix_hash="d" * 64,
        config={"language": {"max_question_tokens": 8}},
    )
    runtime = QuestionControlledChatRuntime(
        base,
        candidate,
        {"architecture": "magnitude_gated_teacher_basis_full_scene_control_v6"},
    )

    control, audit = runtime._control_tokens("Where is it?")

    assert control is None
    assert audit["control_used"] is False
    assert audit["exact_no_control_route"] is True
    assert audit["exact_no_control_below_threshold"] is True
    assert audit["activation_rms"] == pytest.approx(0.01, abs=1e-6)


def test_v6_loader_rejects_self_consistent_threshold_or_value_tamper(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    checkpoint = tmp_path / "v65"
    save_v6_control_checkpoint(
        checkpoint,
        control=candidate,
        base_checkpoint_sha256="1" * 64,
        base_runtime_config_sha256="2" * 64,
        expected_training_fit_state_sha256=v6_value_state_sha256(candidate),
        saved_runtime_training_gate_passed=True,
        saved_runtime_training_gate_attestation_sha256="4" * 64,
    )
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["activation_rms_threshold"] = metadata["maximum_control_rms"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime contract mismatch"):
        _load_control_head(checkpoint, hidden_size=8, device=torch.device("cpu"))

    metadata["activation_rms_threshold"] = 0.01
    weights = checkpoint / "control.safetensors"
    state = load_file(str(weights), device="cpu")
    state["magnitude_output.bias"].add_(1.0)
    save_file(state, str(weights))
    metadata["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="value state changed"):
        _load_control_head(checkpoint, hidden_size=8, device=torch.device("cpu"))


@pytest.mark.parametrize("threshold", [0.0, -0.1, 0.3, float("inf"), True])
def test_v6_rejects_invalid_magnitude_threshold(threshold: object) -> None:
    source = _v65()
    with pytest.raises(ValueError, match="activation_rms_threshold"):
        MagnitudeGatedTeacherBasisFullSceneQuestionControlV6.from_v65(
            source,
            activation_rms_threshold=threshold,  # type: ignore[arg-type]
        )
