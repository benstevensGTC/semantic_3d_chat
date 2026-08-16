from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from semantic_3d_chat.evaluation.v81_structured_dense_atlas_sidecar_preregistration import (
    CONFIG,
    EXPECTED_CONFIG_SHA256,
    load_v81_config,
    run_synthetic_cpu_preflight,
    sha256_file,
)
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_MEMORY_TOKENS,
    BASE_ENVIRONMENT_LATENTS,
    BASE_PREFIX_TOKENS,
    CANDIDATE_TENSOR_NAMES,
    FIXED_PREFIX_TOKENS,
    HIDDEN_SIZE,
    INPUT_EMBEDDING_TENSOR_NAME,
    MAXIMUM_CONTROL_RMS,
    MINIMUM_ATLAS_WEIGHT,
    MINIMUM_BASE_WEIGHT,
    MODEL_BLOB_SHA256_IDENTITY,
    PROBE_COUNT,
    RAW_ATLAS_LOGIT_SCALE,
    TRAINABLE_PARAMETER_COUNT,
    VALUES_PER_PROBE,
    StructuredDenseAtlasSidecarV81,
    assert_prefix_binding_v81,
    audit_v75_v2_prefix_v81,
    bind_fixed_prefix_before_question_v81,
    deterministic_atlas_read_v81,
    frozen_lm_head_logits_v81,
    latest_user_question_query_v81,
    reconstruct_base_v54_prefix_v81,
    sanitized_candidate_metadata_v81,
    split_v75_v2_prefix_v81,
)


def _prefix(*, value_scale: float = 0.02) -> torch.Tensor:
    torch.manual_seed(810081)
    prefix = torch.randn(1, FIXED_PREFIX_TOKENS, HIDDEN_SIZE) * 0.02
    banks = split_v75_v2_prefix_v81(prefix)
    banks.atlas_values.copy_(torch.randn_like(banks.atlas_values) * value_scale)
    return prefix


def test_v81_exact_layout_audit_and_base258_reconstruction() -> None:
    prefix = _prefix()
    banks = split_v75_v2_prefix_v81(prefix)
    audit = audit_v75_v2_prefix_v81(prefix)
    base = reconstruct_base_v54_prefix_v81(prefix)

    assert tuple(banks.probe_keys.shape) == (1, PROBE_COUNT, HIDDEN_SIZE)
    assert tuple(banks.atlas_values.shape) == (
        1,
        PROBE_COUNT,
        VALUES_PER_PROBE,
        HIDDEN_SIZE,
    )
    assert tuple(banks.base_latents.shape) == (
        1,
        BASE_ENVIRONMENT_LATENTS,
        HIDDEN_SIZE,
    )
    assert tuple(base.shape) == (1, BASE_PREFIX_TOKENS, HIDDEN_SIZE)
    assert torch.equal(base[:, :1], prefix[:, :1])
    assert torch.equal(base[:, 1:-1], prefix[:, 1 + ATLAS_MEMORY_TOKENS : -1])
    assert torch.equal(base[:, -1:], prefix[:, -1:])
    assert audit.exact_reconstruction
    assert audit.boundary_tokens_retained
    assert not audit.boi_eoi_are_payload
    assert not audit.probe_keys_are_payload
    assert not audit.question_queries_are_payload
    assert audit.stage_a_positive_floor_value_count == 384
    assert audit.stage_a_dense_score_key_count == 96
    assert audit.base_latents_use_native_frozen_gemma_path_in_stage_a
    assert not audit.all_738_tokens_claimed_strict_positive_payload_influence


def test_v81_scale160_dense_direct_control_and_binding_fail_closed() -> None:
    prefix = _prefix(value_scale=0.01)
    query = torch.randn(1, HIDDEN_SIZE)
    binding = bind_fixed_prefix_before_question_v81(prefix)
    output = deterministic_atlas_read_v81(prefix, query, binding=binding)

    assert RAW_ATLAS_LOGIT_SCALE == 160.0
    assert tuple(output.reconstructed_controls.shape) == (
        1,
        VALUES_PER_PROBE,
        HIDDEN_SIZE,
    )
    assert output.finite
    assert output.all_96_groups_positive
    assert output.all_384_values_receive_positive_floor_weight
    assert float(output.atlas_weights.min()) >= MINIMUM_ATLAS_WEIGHT - 1e-9
    assert torch.allclose(
        output.attention_sums,
        torch.ones_like(output.attention_sums),
        atol=1e-6,
        rtol=0.0,
    )
    assert float(output.control_rms.max()) <= MAXIMUM_CONTROL_RMS

    changed = prefix.clone()
    changed[:, 9, 4] += 1.0
    with pytest.raises(ValueError, match="changed after prequestion binding"):
        assert_prefix_binding_v81(changed, binding=binding)
    with pytest.raises(ValueError, match="changed after prequestion binding"):
        deterministic_atlas_read_v81(changed, query, binding=binding)


def test_v81_all_zero_738_short_circuits_to_uniform_exact_zero() -> None:
    prefix = torch.zeros(1, FIXED_PREFIX_TOKENS, HIDDEN_SIZE)
    query = torch.zeros(1, HIDDEN_SIZE)
    binding = bind_fixed_prefix_before_question_v81(prefix)
    output = deterministic_atlas_read_v81(prefix, query, binding=binding)

    assert torch.equal(
        output.atlas_weights,
        torch.full_like(output.atlas_weights, 1.0 / PROBE_COUNT),
    )
    assert int(torch.count_nonzero(output.reconstructed_controls)) == 0
    assert int(torch.count_nonzero(output.control_rms)) == 0


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append((text, kwargs))
        return {"input_ids": torch.tensor([[2, 4, 6]], dtype=torch.long)}


def test_v81_latest_user_only_query_contract() -> None:
    # Bind the scene before the latest user question is tokenized.
    prefix = _prefix()
    binding = bind_fixed_prefix_before_question_v81(prefix)
    tokenizer = _Tokenizer()
    embedding = nn.Embedding(8, HIDDEN_SIZE)
    embedding.requires_grad_(False)
    output = latest_user_question_query_v81(
        tokenizer=tokenizer,
        embedding_layer=embedding,
        latest_user_question="Where is the chair?",
        device=torch.device("cpu"),
        maximum_question_tokens=16,
        model_blob_sha256_identity=MODEL_BLOB_SHA256_IDENTITY,
        embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
    )
    assert_prefix_binding_v81(prefix, binding=binding)

    assert tokenizer.calls == [
        (
            "Where is the chair?",
            {"add_special_tokens": False, "return_tensors": "pt"},
        )
    ]
    assert output.token_count == 3
    assert output.query.dtype == torch.float32
    assert not output.query.requires_grad
    assert not output.add_special_tokens
    assert not output.included_system_prompt
    assert not output.included_history
    assert not output.included_answer
    assert torch.equal(
        output.query,
        embedding(output.token_ids).float().mean(dim=1),
    )

    trainable = nn.Embedding(8, HIDDEN_SIZE)
    with pytest.raises(ValueError, match="frozen embedding"):
        latest_user_question_query_v81(
            tokenizer=tokenizer,
            embedding_layer=trainable,
            latest_user_question="x",
            device=torch.device("cpu"),
            maximum_question_tokens=16,
            model_blob_sha256_identity=MODEL_BLOB_SHA256_IDENTITY,
            embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
        )


def test_v81_stage_b_is_quarantined_zero_safe_and_gemma_detached() -> None:
    prefix = _prefix(value_scale=0.01)
    query = torch.randn(1, HIDDEN_SIZE, requires_grad=True)
    decoder = (torch.randn(1, HIDDEN_SIZE) * 0.01).requires_grad_(True)
    binding = bind_fixed_prefix_before_question_v81(prefix)
    quarantined = StructuredDenseAtlasSidecarV81()
    with pytest.raises(PermissionError, match="quarantined"):
        quarantined(
            prefix,
            query,
            decoder,
            binding=binding,
            enable_stage_b=True,
        )

    sidecar = StructuredDenseAtlasSidecarV81(allow_stage_b=True)
    disabled = sidecar(prefix, query, decoder, binding=binding)
    enabled = sidecar(
        prefix,
        query,
        decoder,
        binding=binding,
        enable_stage_b=True,
    )
    assert not disabled.stage_b_enabled
    assert torch.equal(disabled.fused_hidden, decoder.detach().float())
    assert int(torch.count_nonzero(disabled.residual)) == 0
    assert enabled.stage_b_enabled
    assert int(torch.count_nonzero(enabled.residual)) == 0
    assert float(enabled.learned_atlas_weights.detach().min()) >= MINIMUM_ATLAS_WEIGHT - 1e-9
    assert float(enabled.base_weights.detach().min()) >= MINIMUM_BASE_WEIGHT - 1e-9
    assert all(module.bias is None for module in sidecar.modules() if isinstance(module, nn.Linear))

    with torch.no_grad():
        sidecar.residual_output.weight.normal_(0.0, 0.01)
    zero_payload = prefix.detach().clone()
    banks = split_v75_v2_prefix_v81(zero_payload)
    banks.atlas_values.zero_()
    banks.base_latents.zero_()
    zero_output = sidecar(
        zero_payload,
        query,
        decoder,
        binding=bind_fixed_prefix_before_question_v81(zero_payload),
        enable_stage_b=True,
    )
    assert int(torch.count_nonzero(zero_output.residual)) == 0

    output = sidecar(
        prefix,
        query,
        decoder,
        binding=binding,
        enable_stage_b=True,
    )
    frozen_head = nn.Linear(HIDDEN_SIZE, 17, bias=False)
    frozen_head.requires_grad_(False)
    logits = frozen_lm_head_logits_v81(output.fused_hidden, frozen_lm_head=frozen_head)
    logits.square().mean().backward()
    assert query.grad is None
    assert decoder.grad is None
    assert all(parameter.grad is None for parameter in frozen_head.parameters())
    assert any(
        parameter.grad is not None and float(parameter.grad.norm()) > 0.0
        for parameter in sidecar.parameters()
    )


def test_v81_candidate_inventory_and_metadata_are_numeric_only() -> None:
    sidecar = StructuredDenseAtlasSidecarV81()
    state = sidecar.candidate_state_dict()
    metadata = sanitized_candidate_metadata_v81(weights_sha256="a" * 64)

    assert set(state) == CANDIDATE_TENSOR_NAMES
    assert sum(value.numel() for value in state.values()) == TRAINABLE_PARAMETER_COUNT
    assert all(value.dtype == torch.float32 for value in state.values())
    for field in (
        "probe_bank_serialized",
        "atlas_values_serialized",
        "base_latents_serialized",
        "environmental_prefix_cache_serialized",
        "questions_serialized",
        "answers_serialized",
        "prototypes_serialized",
        "class_ids_serialized",
        "teacher_cache_serialized",
        "prediction_cache_serialized",
        "environmental_text_serialized",
        "runtime_publication_authorized",
    ):
        assert metadata[field] is False


def test_v81_config_is_sealed_and_synthetic_preflight_is_model_free() -> None:
    config = load_v81_config(CONFIG)
    result = run_synthetic_cpu_preflight()
    launcher = Path("scripts/run_v81_strict_fixed_prefix_reader.sh").read_text(encoding="utf-8")

    assert sha256_file(CONFIG) == EXPECTED_CONFIG_SHA256
    assert config["fixed_memory_contract"]["fixed_prefix_tokens"] == 738
    assert config["stage_a_deterministic_reader"]["atlas_logit_scale"] == 160.0
    assert (
        config["stage_a_deterministic_reader"]["development_acceptance_thresholds"][
            "retroactively_preregistered"
        ]
        is False
    )
    assert config["stage_b_optional_postdecoder_fusion"]["authorized_now"] is False
    assert result["passed"]
    assert not result["full_gemma_loaded"]
    assert not result["fit_executed"]
    assert not result["mps_used"]
    assert "--write-preregistration" in launcher
    assert "--write-cpu-preflight" in launcher
    assert "from transformers" not in launcher.casefold()
    assert 'torch.device("mps")' not in launcher.casefold()
