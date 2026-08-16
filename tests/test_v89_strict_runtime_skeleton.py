from __future__ import annotations

import inspect

import pytest

from semantic_3d_chat.chat.strict_scene1_stack_contract import (
    FrozenRuntimeBankContract,
    StrictScene1StackContract,
)
from semantic_3d_chat.evaluation import v89_strict_runtime_skeleton as v89


def _parent() -> StrictScene1StackContract:
    targets = (
        "model.language_model.layers.34.self_attn.q_proj",
        "model.language_model.layers.30.self_attn.q_proj",
        "model.language_model.layers.13.self_attn.k_proj",
        "model.language_model.layers.28.self_attn.q_proj",
        "model.language_model.layers.13.self_attn.q_proj",
        "model.language_model.layers.18.self_attn.q_proj",
        "model.language_model.layers.34.mlp.down_proj",
        "model.language_model.layers.34.mlp.up_proj",
        "model.language_model.layers.34.mlp.gate_proj",
        "model.language_model.layers.27.self_attn.q_proj",
    )
    counts = (
        45_056,
        229_376,
        30_720,
        36_864,
        36_864,
        131_072,
        55_296,
        110_592,
        110_592,
        57_344,
    )
    ranks = (4, 8, 4, 4, 4, 8, 4, 8, 8, 16)
    alphas = (8.0, 16.0, 8.0, 8.0, 8.0, 16.0, 8.0, 16.0, 16.0, 32.0)
    banks = tuple(
        FrozenRuntimeBankContract(
            name=name,
            target_modules=(target,),
            rank=rank,
            alpha=alpha,
            parameter_count=count,
            state_sha256=f"{ordinal + 1:064x}",
        )
        for ordinal, (name, target, rank, alpha, count) in enumerate(
            zip(v89.V88_PARENT_BANKS, targets, ranks, alphas, counts, strict=True)
        )
    )
    return StrictScene1StackContract(
        banks=banks,
        expected_total_parameter_count=v89.V88_PARENT_PARAMETER_COUNT,
    )


def _surfaces(contract: StrictScene1StackContract) -> tuple[dict, dict]:
    config_banks = {}
    metadata_banks = []
    states = {}
    modules = {}
    counts = {}
    for bank in contract.banks:
        config_banks[bank.name] = {
            "trainable": False,
            "rank": bank.rank,
            "alpha": bank.alpha,
            "dropout": 0.0,
            "expected_initial_state_sha256": bank.state_sha256,
            "target_modules": list(bank.target_modules),
        }
        metadata_banks.append(
            {
                "name": bank.name,
                "trainable": False,
                "rank": bank.rank,
                "alpha": bank.alpha,
                "dropout": 0.0,
                "target_modules": list(bank.target_modules),
                "adapter_parameter_count": bank.parameter_count,
            }
        )
        states[bank.name] = bank.state_sha256
        modules[bank.name] = list(bank.target_modules)
        counts[bank.name] = {bank.target_modules[0]: bank.parameter_count}
    config = {"language": {"lora_banks": config_banks}}
    metadata = {
        "lora": {
            "schema_version": 2,
            "enabled": True,
            "banks": metadata_banks,
            "adapter_parameter_count": contract.expected_total_parameter_count,
            "trainable_adapter_parameter_count": 0,
        },
        "lora_bank_state_sha256": states,
        "lora_bank_wrapped_modules": modules,
        "lora_bank_parameter_counts": counts,
        "lora_parameter_count": contract.expected_total_parameter_count,
        "lora_trainable_parameter_count": 0,
    }
    return config, metadata


def test_v89_builds_exact_eleven_bank_disjoint_scene1_topology() -> None:
    final = v89.build_v89_stack_contract(
        parent=_parent(), final_state_sha256="f" * 64
    )

    assert len(final.banks) == 11
    assert final.bank_order == v89.V89_FINAL_BANKS
    assert final.banks[-1].name == "v89_scene1_retention_bridge"
    assert final.banks[-1].target_modules == (
        "model.language_model.layers.27.self_attn.o_proj",
    )
    assert final.banks[-1].rank == 8
    assert final.banks[-1].alpha == 16.0
    assert final.banks[-1].parameter_count == 28_672
    assert final.expected_total_parameter_count == 872_448


def test_v89_validates_identical_config_checkpoint_inventory() -> None:
    parent = _parent()
    final = v89.build_v89_stack_contract(
        parent=parent, final_state_sha256="f" * 64
    )
    config, metadata = _surfaces(final)

    v89.validate_v89_runtime_stack(
        scene_id="scene_000001",
        runtime_config=config,
        checkpoint_metadata=metadata,
        parent=parent,
        final_state_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="only scene_000001"):
        v89.validate_v89_runtime_stack(
            scene_id="scene_000039",
            runtime_config=config,
            checkpoint_metadata=metadata,
            parent=parent,
            final_state_sha256="f" * 64,
        )


def test_v89_rejects_parent_overlap_and_bank_reordering() -> None:
    parent = _parent()
    changed_banks = list(parent.banks)
    changed_banks[-1] = FrozenRuntimeBankContract(
        name=changed_banks[-1].name,
        target_modules=(v89.V89_TARGET,),
        rank=changed_banks[-1].rank,
        alpha=changed_banks[-1].alpha,
        parameter_count=changed_banks[-1].parameter_count,
        state_sha256=changed_banks[-1].state_sha256,
    )
    overlap = StrictScene1StackContract(
        banks=tuple(changed_banks),
        expected_total_parameter_count=v89.V88_PARENT_PARAMETER_COUNT,
    )
    with pytest.raises(ValueError, match="overlaps"):
        v89.build_v89_stack_contract(
            parent=overlap, final_state_sha256="f" * 64
        )

    reordered = StrictScene1StackContract(
        banks=tuple(reversed(parent.banks)),
        expected_total_parameter_count=v89.V88_PARENT_PARAMETER_COUNT,
    )
    with pytest.raises(ValueError, match="exact frozen ten-bank"):
        v89.build_v89_stack_contract(
            parent=reordered, final_state_sha256="f" * 64
        )


def test_v89_skeleton_has_no_model_eval_training_or_write_surface() -> None:
    source = inspect.getsource(v89).casefold()

    for forbidden in (
        "load_local_language_model",
        "evaluate_v89",
        "train_v89",
        "question",
        "answer",
        "oracle",
        "save_file",
        "write_text",
        "promote",
        "yaml.safe_dump",
    ):
        assert forbidden not in source
