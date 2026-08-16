from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v3_preregistration import (
    V2_ABORT_SHA256,
    V2_PREREGISTRATION_SHA256,
    V2_SMOKE_SHA256,
    authenticate_v2_abort,
    build_preregistration,
    v3_implementation_hashes,
    write_preregistration,
)
from semantic_3d_chat.training.train_fixed_prefix_ple_v54_v3 import (
    canonical_tuple_mapping_hash,
)


def test_v3_authenticates_zero_update_v2_abort() -> None:
    abort = authenticate_v2_abort()
    contract = build_preregistration()

    assert abort["failure_scope"]["adapter_update_count"] == 0
    assert abort["failure_scope"]["optimizer_constructed"] is False
    assert abort["checkpoint_absent"] is True
    assert contract["v2_abort"]["preregistration_sha256"] == V2_PREREGISTRATION_SHA256
    assert contract["v2_abort"]["smoke_sha256"] == V2_SMOKE_SHA256
    assert contract["v2_abort"]["abort_sha256"] == V2_ABORT_SHA256
    assert contract["v2_abort"]["adapter_update_count"] == 0


def test_v3_only_changes_diagnostic_tuple_key_hashing() -> None:
    contract = build_preregistration()

    assert contract["only_change"] == {
        "field": "diagnostic_hash_serialization.for_tuple_keyed_mappings",
        "v2": "json_object_with_tuple_keys_raises_type_error",
        "v3": "sorted_records_with_key_list_and_value",
        "affects_model_forward": False,
        "affects_loss": False,
        "affects_gradient": False,
        "affects_optimizer": False,
        "affects_gate_values": False,
    }
    assert contract["smoke_inheritance"]["new_model_forward_required"] is False


def test_v3_inherits_every_v2_model_data_training_gate_and_leakage_contract() -> None:
    v2 = build_preregistration()["unchanged_v2_contract"]
    unchanged = v2["unchanged_v1_contract"]

    assert unchanged["trainable_surface"]["parameter_count"] == 41_984
    assert unchanged["objective"]["same_question_wrong_prefix_hinge_weight"] == 1.0
    assert unchanged["optimization"]["seed"] == 720054
    assert unchanged["optimization"]["learning_rate"] == 0.0003
    assert unchanged["optimization"]["maximum_updates"] == 40
    assert unchanged["selection"]["all_gates_required"] is True
    assert unchanged["runtime_contract"]["question_dependent_retrieval"] is False
    assert unchanged["runtime_contract"]["environmental_text_inputs"] == []


def test_v3_tuple_mapping_hash_is_order_independent_and_json_safe() -> None:
    first = {("scene_000001", "q_2"): 0.2, ("scene_000001", "q_1"): 0.1}
    second = dict(reversed(list(first.items())))

    assert canonical_tuple_mapping_hash(first) == canonical_tuple_mapping_hash(second)
    assert len(canonical_tuple_mapping_hash(first)) == 64


def test_v3_preregistration_is_create_once_and_source_locked(tmp_path: Path) -> None:
    contract = build_preregistration()
    destination = tmp_path / "v3.json"
    path, digest = write_preregistration(destination)

    assert path == destination.resolve()
    assert len(digest) == 64
    assert json.loads(destination.read_text(encoding="utf-8")) == contract
    assert contract["v3_implementation_source_hashes"] == v3_implementation_hashes()
    with pytest.raises(FileExistsError, match="exists"):
        write_preregistration(destination)


def test_v3_wrapper_does_not_define_training_hyperparameters() -> None:
    source = Path(
        "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v3.py"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/run_gemma4_v54_fixed_prefix_ple_reader_v3.sh").read_text(
        encoding="utf-8"
    )

    assert "learning_rate" not in source
    assert "maximum_updates" not in source
    assert "same_question_wrong_prefix" not in source
    assert "preregister|preflight|inherit-smoke|train|authenticate" in launcher
