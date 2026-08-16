from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.finetune_v74_gemma_nll import (
    V74_STATE_FIELDS,
    V75_STATE_FIELDS,
    _load_initial_candidate,
    assert_dense_reader_exact_zero_scene,
    assert_exclusive_dense_reader_trainable_surface,
    dense_reader_architecture,
    save_dense_reader_gemma_nll_diagnostic,
)


def _v74() -> DenseFullSceneContinuousControlV74:
    return DenseFullSceneContinuousControlV74(
        16,
        torch.eye(16)[:3],
        environment_latents=4,
        query_count=2,
        model_dimension=4,
    )


def _v75() -> DenseFullSceneContinuousControlV75:
    return DenseFullSceneContinuousControlV75(
        16,
        torch.eye(16)[:3],
        environment_latents=4,
        query_count=2,
        model_dimension=4,
        coefficient_decoder_hidden_dimension=7,
    )


def _source_metadata(architecture: str) -> dict[str, str]:
    return {
        "artifact": f"{architecture}_verified_teacher_dense_reader_candidate_v1",
        "training_pool_only": "true",
        "runtime_promotion_forbidden_until_gemma_gate": "true",
        "numeric_gate_passed": "true",
        "answer_codebook_serialized": "false",
        "environmental_text_inputs": "0",
    }


@pytest.mark.parametrize("architecture", ["v74", "v75"])
def test_source_loader_reconstructs_exact_supported_architecture(
    tmp_path, architecture: str
) -> None:
    source = _v74() if architecture == "v74" else _v75()
    path = tmp_path / f"{architecture}.safetensors"
    save_file(source.state_dict(), path, metadata=_source_metadata(architecture))

    loaded, metadata = _load_initial_candidate(
        path, torch.device("cpu"), hidden_size=16, environment_latents=4
    )

    assert dense_reader_architecture(loaded) == architecture
    assert metadata["artifact"].startswith(architecture)
    assert set(loaded.state_dict()) == (
        set(V74_STATE_FIELDS) if architecture == "v74" else set(V75_STATE_FIELDS)
    )
    assert all(
        torch.equal(source.state_dict()[name], loaded.state_dict()[name])
        for name in source.state_dict()
    )
    audit = assert_dense_reader_exact_zero_scene(loaded)
    assert audit["exact_zero_scene_verified"] is True
    assert audit["question_only_output_path_exists"] is False


def test_source_loader_fails_closed_on_hybrid_layout_or_artifact_mismatch(
    tmp_path,
) -> None:
    model = _v75()
    hybrid = dict(model.state_dict())
    hybrid["unexpected.weight"] = torch.zeros(1)
    hybrid_path = tmp_path / "hybrid.safetensors"
    save_file(hybrid, hybrid_path, metadata=_source_metadata("v75"))
    with pytest.raises(ValueError, match="unsupported state layout"):
        _load_initial_candidate(
            hybrid_path,
            torch.device("cpu"),
            hidden_size=16,
            environment_latents=4,
        )

    mismatch_path = tmp_path / "mismatch.safetensors"
    save_file(
        model.state_dict(), mismatch_path, metadata=_source_metadata("v74")
    )
    with pytest.raises(ValueError, match="quarantine contract"):
        _load_initial_candidate(
            mismatch_path,
            torch.device("cpu"),
            hidden_size=16,
            environment_latents=4,
        )

    leaked_metadata = _source_metadata("v75")
    leaked_metadata["environment_caption"] = "forbidden"
    leaked_path = tmp_path / "leaked-metadata.safetensors"
    save_file(model.state_dict(), leaked_path, metadata=leaked_metadata)
    with pytest.raises(ValueError, match="quarantine contract"):
        _load_initial_candidate(
            leaked_path,
            torch.device("cpu"),
            hidden_size=16,
            environment_latents=4,
        )


def test_v75_trainable_audit_and_diagnostic_remain_exclusive_and_quarantined(
    tmp_path,
) -> None:
    runtime = SimpleNamespace(
        language=SimpleNamespace(model=torch.nn.Linear(3, 3).requires_grad_(False))
    )
    model = _v75()
    trainable = assert_exclusive_dense_reader_trainable_surface(runtime, model)
    assert trainable["architecture_version"] == "v75"
    assert trainable["base_trainable_parameter_count"] == 0
    assert trainable["only_dense_reader_trainable"] is True

    output = tmp_path / "v75-diagnostic.safetensors"
    result = save_dense_reader_gemma_nll_diagnostic(
        output,
        model,
        source_sha256="b" * 64,
        optimizer_steps=18,
        train_behavior_improved=False,
    )
    with safe_open(str(output), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        assert set(handle.keys()) == set(V75_STATE_FIELDS)
    assert metadata["controller_architecture"] == "v75"
    assert metadata["runtime_promotion_forbidden_until_gemma_gate"] == "true"
    assert metadata["runtime_publication_artifact"] == "false"
    assert metadata["numeric_gate_passed"] == "unverified_after_gemma_nll"
    assert metadata["official_validation_loaded"] == "false"
    assert metadata["official_test_loaded"] == "false"
    assert metadata["oracle_loaded"] == "false"
    assert metadata["exact_zero_scene_verified"] == "true"
    assert metadata["answer_codebook_serialized"] == "false"
    assert metadata["environmental_text_inputs"] == "0"
    assert result["zero_scene_audit"]["exact_zero_scene_verified"] is True


def test_generic_path_rejects_unfrozen_base_for_v75() -> None:
    runtime = SimpleNamespace(language=SimpleNamespace(model=torch.nn.Linear(3, 3)))
    with pytest.raises(RuntimeError, match="unexpectedly has trainable"):
        assert_exclusive_dense_reader_trainable_surface(runtime, _v75())
