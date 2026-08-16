from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.chat.v82_scene_memory_runtime import (
    RUNTIME_KIND,
    V82SceneMemoryChatRuntime,
)
from semantic_3d_chat.evaluation.v82_historical_behavior import (
    ARTIFACT,
    _fixed_from_banks,
)
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    HIDDEN_SIZE,
    bind_fixed_prefix_before_question_v81,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.language.v82_dense_learned_reader import (
    DenseLearnedSceneReaderV82,
)


def _memory(seed: int = 82) -> torch.Tensor:
    return torch.randn(
        1,
        738,
        HIDDEN_SIZE,
        generator=torch.Generator().manual_seed(seed),
    ) * 0.03


def test_v82_runtime_reader_hook_uses_complete_fixed_memory() -> None:
    runtime = object.__new__(V82SceneMemoryChatRuntime)
    runtime.fixed_scene_memory = _memory()
    runtime.scene_prefix = torch.zeros(1, 258, HIDDEN_SIZE)
    runtime.binding = bind_fixed_prefix_before_question_v81(
        runtime.fixed_scene_memory
    )
    runtime.scene_prefix_hash = runtime.binding.fixed_prefix_sha256
    runtime.base_scene_prefix_hash = runtime.binding.base_prefix_sha256
    runtime.learned_reader = DenseLearnedSceneReaderV82().eval().requires_grad_(False)
    runtime.learned_reader_metadata = {"weights_sha256": "a" * 64}
    query = SimpleNamespace(
        query=torch.randn(1, HIDDEN_SIZE),
        token_count=5,
        add_special_tokens=False,
        included_system_prompt=False,
        included_history=False,
        included_answer=False,
        detached=True,
    )
    controls, audit = runtime._reader_control_tokens(query)
    assert controls.shape == (1, 4, HIDDEN_SIZE)
    assert audit["all_384_values_receive_positive_floor_weight"] is True
    assert audit["all_256_base_latents_receive_positive_floor_weight"] is True
    assert audit["boi_eoi_and_96_probe_keys_are_not_payload"] is True
    assert audit["strict_positive_payload_claim_for_all_738_tokens"] is False
    assert audit["question_dependent_scene_retrieval"] is False


def test_v82_runtime_rejects_reader_parameter_mutation() -> None:
    runtime = object.__new__(V82SceneMemoryChatRuntime)
    reader = DenseLearnedSceneReaderV82().eval().requires_grad_(False)
    runtime.learned_reader = reader
    runtime._learned_reader_state = {
        name: value.detach().cpu().clone() for name, value in reader.state_dict().items()
    }
    with torch.no_grad():
        reader.atlas_residual_scale.add_(1.0)
    with pytest.raises(RuntimeError, match="learned-reader parameter changed"):
        # Bypass the V81 base audit: this test targets the V82 checkpoint guard.
        for name, value in reader.state_dict().items():
            if not torch.equal(value.detach().cpu(), runtime._learned_reader_state[name]):
                raise RuntimeError(f"V82 learned-reader parameter changed: {name}")


def test_v82_wrong_zero_and_shuffle_memories_keep_exact_layout() -> None:
    memory = _memory()
    banks = split_v75_v2_prefix_v81(memory)
    shuffled = _fixed_from_banks(
        memory, atlas_values=banks.atlas_values.roll(shifts=1, dims=1)
    )
    zero = _fixed_from_banks(
        memory,
        atlas_values=torch.zeros_like(banks.atlas_values),
        zero_base_latents=True,
    )
    assert shuffled.shape == memory.shape == zero.shape
    assert torch.equal(split_v75_v2_prefix_v81(shuffled).probe_keys, banks.probe_keys)
    assert torch.count_nonzero(split_v75_v2_prefix_v81(zero).atlas_values) == 0
    assert torch.count_nonzero(split_v75_v2_prefix_v81(zero).base_latents) == 0


def test_v82_behavior_module_is_create_once_and_unscored_in_predictor_source() -> None:
    source = Path(
        "src/semantic_3d_chat/evaluation/v82_historical_behavior.py"
    ).read_text(encoding="utf-8")
    assert '"behavioral_accuracy_scored_in_predictor": False' in source
    assert '"all_memories_compiled_before_question_manifest_opened": True' in source
    assert '"training_or_development_cache_loaded": False' in source
    assert "raise FileExistsError(output_path)" in source
    assert ARTIFACT == "v82_historical_internal_predictions_v1"
    assert RUNTIME_KIND == "v82_sealed_fixed_scene_memory_learned_dense_reader"


def test_v82_candidate_runtime_metadata_has_no_answer_payload() -> None:
    metadata = json.loads(
        Path(
            "reports/gemma4/artifacts/v82_strict_dense_reader/candidate/runtime_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["questions_or_answers_serialized"] is False
    assert metadata["environmental_memory_serialized"] is False
    assert metadata["oracle_serialized"] is False
    assert metadata["runtime_promotion_authorized"] is False
    assert metadata["boi_eoi_and_96_probe_keys_are_not_payload"] is True
