from __future__ import annotations

import json

import pytest
from safetensors.torch import load_file

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v85_strict_multiscene_cli import _parser
from semantic_3d_chat.evaluation.v85_strict_runtime_release import (
    BASE_CHECKPOINT,
    BRIDGE_BANK,
    BRIDGE_STATE_SHA256,
    CANDIDATE_CHECKPOINT,
    CANDIDATE_MEMORY,
    HELD_MEMORY_REPORT,
    RELEASE_CHECKPOINT,
    RUNTIME_CONFIG,
    SMOKE_REPORT,
    SOURCE_MEMORY_TENSOR_FILE_SHA256,
    build_runtime_metadata,
    promote_release,
    sha256_file,
    verify_candidate,
)
from semantic_3d_chat.language.lora import lora_banks_settings, tensor_state_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import MEMORY_FILENAME
from semantic_3d_chat.training.checkpointing import validate_runtime_checkpoint_metadata


def test_v85_runtime_config_is_exactly_seven_frozen_banks() -> None:
    config = load_runtime_config(RUNTIME_CONFIG)
    banks = lora_banks_settings(config)

    assert len(banks.banks) == 7
    assert banks.trainable is False
    assert [bank.name for bank in banks.banks].count(BRIDGE_BANK) == 1
    bridge = banks.bank(BRIDGE_BANK)
    assert bridge.trainable is False
    assert bridge.adapter.rank == 4
    assert bridge.adapter.alpha == 8.0
    assert bridge.adapter.target_modules == (
        "model.language_model.layers.34.mlp.down_proj",
    )
    assert bridge.expected_initial_state_sha256 == BRIDGE_STATE_SHA256


def test_v85_runtime_metadata_binds_evidence_without_environmental_text() -> None:
    metadata = build_runtime_metadata(
        promotion="pending_strict_runtime_leakage", smoke_report_sha256=None
    )
    validate_runtime_checkpoint_metadata(metadata)
    encoded = json.dumps(metadata, sort_keys=True).casefold()
    provenance = metadata["initialization_provenance"]["v85_strict_runtime_release"]

    assert metadata["lora_parameter_count"] == 565_248
    assert metadata["lora_trainable_parameter_count"] == 0
    assert provenance["runtime_promotion_authorized"] is False
    assert provenance["v75_comparator_retained"] is True
    assert all(
        token not in encoded
        for token in (
            "oracle",
            "answer_text",
            "question_text",
            "object_name",
            "scene_graph",
        )
    )


@pytest.mark.skipif(not CANDIDATE_CHECKPOINT.is_dir(), reason="local V85 package absent")
def test_v85_candidate_is_two_file_and_preserves_every_v54_tensor() -> None:
    result = verify_candidate()
    base = load_file(str(BASE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    candidate = load_file(str(CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    retained = {name: candidate[name] for name in base}

    assert result["passed"] is True
    assert tensor_state_sha256(retained) == tensor_state_sha256(base)
    assert set(candidate) - set(base) == {
        f"lora_banks.{BRIDGE_BANK}.adapters.0.lora_a",
        f"lora_banks.{BRIDGE_BANK}.adapters.0.lora_b",
    }
    assert {item.name for item in CANDIDATE_CHECKPOINT.iterdir()} == {
        "adapter.safetensors",
        "runtime_metadata.json",
    }


@pytest.mark.skipif(not CANDIDATE_MEMORY.is_dir(), reason="local V85 memory absent")
def test_v85_scene1_rebinding_changed_metadata_only() -> None:
    assert sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME) == (
        SOURCE_MEMORY_TENSOR_FILE_SHA256
    )
    assert {item.name for item in CANDIDATE_MEMORY.iterdir()} == {
        "memory.safetensors",
        "runtime_metadata.json",
    }


@pytest.mark.skipif(not SMOKE_REPORT.is_file(), reason="local V85 smoke absent")
def test_v85_smoke_passed_leakage_and_invariance_but_failed_behavior() -> None:
    report = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))

    assert report["gates"]["oracle_physically_unavailable"] is True
    assert report["gates"]["file_audit_forbidden_read_count_zero"] is True
    assert report["gates"]["prefix_hash_identical_for_every_question"] is True
    assert report["gates"]["total_environment_conditioned_input_identical"] is True
    assert [row["observed"] for row in report["behavior"]] == ["no", "blue", "left"]
    # The sealed first scorer used the reversed third relation.  Its separate
    # post-hoc oracle correction establishes one true pass (left), still below
    # the predeclared 3/3 promotion gate.
    correction = json.loads(
        (
            SMOKE_REPORT.parent / "v85_strict_runtime_smoke_oracle_correction.json"
        ).read_text(encoding="utf-8")
    )
    assert [row["expected"] for row in report["behavior"]] == ["yes", "red", "right"]
    assert [row["expected"] for row in correction["corrected_behavior"]] == [
        "yes",
        "red",
        "left",
    ]
    assert correction["correct"] == 1
    assert correction["promotion_gate_passed"] is False
    assert report["passed"] is False
    assert report["promotion_authorized"] is False
    assert not RELEASE_CHECKPOINT.exists()
    with pytest.raises(ValueError, match="did not authorize"):
        promote_release()


@pytest.mark.skipif(not HELD_MEMORY_REPORT.is_file(), reason="held export absent")
def test_v85_preselected_scene39_export_is_sanitized_and_not_behavior_selected() -> None:
    report = json.loads(HELD_MEMORY_REPORT.read_text(encoding="utf-8"))

    assert report["scene_id"] == "scene_000039"
    assert report["behavior_used_for_selection"] is False
    assert report["selected_before_runtime_behavior"] is True
    assert report["source_cache_questions_or_answers_serialized"] is False
    assert report["source_cache_environmental_text_serialized"] is False
    assert report["source_cache_oracle_serialized"] is False
    assert report["runtime_inventory"] == ["memory.safetensors", "runtime_metadata.json"]


def test_v85_cli_defaults_to_release_and_candidate_requires_explicit_flag() -> None:
    defaults = _parser().parse_args([])
    candidate = _parser().parse_args(["--allow-candidate"])

    assert defaults.allow_candidate is False
    assert defaults.base_checkpoint.endswith("gemma4_v85_strict_multiscene_release_v1")
    assert defaults.scene_memory.endswith("runtime/scene_memories/v85/scene_000001")
    assert candidate.allow_candidate is True
