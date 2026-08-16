from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.evaluation.strict_direct_release_core import (
    BridgeSourceContract,
    compose_exact_bank_archive,
    extend_runtime_lora_config,
    extend_runtime_metadata,
    sha256_file,
    validate_runtime_bank_inventory,
)
from semantic_3d_chat.language.lora import tensor_state_sha256

BASE_BANKS = tuple(f"base_bank_{index}" for index in range(7))


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _base_checkpoint(root: Path) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    tensors = {
        f"lora_banks.{name}.adapters.0.lora_a": torch.ones(1, 2) * (index + 1)
        for index, name in enumerate(BASE_BANKS)
    }
    save_file(tensors, str(root / "adapter.safetensors"))
    metadata: dict[str, object] = {
        "lora": {
            "schema_version": 2,
            "enabled": True,
            "banks": [
                {
                    "name": name,
                    "trainable": False,
                    "rank": 1,
                    "alpha": 1.0,
                    "dropout": 0.0,
                    "target_modules": [
                        f"model.language_model.layers.{index}.self_attn.q_proj"
                    ],
                    "expected_initial_state_sha256": f"{index + 1:064x}",
                    "adapter_parameter_count": 2,
                }
                for index, name in enumerate(BASE_BANKS)
            ],
            "adapter_parameter_count": 14,
            "trainable_adapter_parameter_count": 0,
        },
        "lora_bank_state_sha256": {name: f"{index + 1:064x}" for index, name in enumerate(BASE_BANKS)},
        "lora_bank_wrapped_modules": {
            name: [f"model.language_model.layers.{index}.self_attn.q_proj"]
            for index, name in enumerate(BASE_BANKS)
        },
        "lora_bank_parameter_counts": {
            name: {f"model.language_model.layers.{index}.self_attn.q_proj": 2}
            for index, name in enumerate(BASE_BANKS)
        },
        "lora_parameter_count": 14,
        "lora_trainable_parameter_count": 0,
    }
    _json(root / "runtime_metadata.json", metadata)
    return root, metadata


def _bridge(
    root: Path,
    *,
    name: str,
    layer: int,
    target: str,
    rank: int,
    input_dim: int,
    output_dim: int,
) -> BridgeSourceContract:
    root.mkdir()
    raw = {
        "lora_a": torch.arange(rank * input_dim, dtype=torch.float32).reshape(
            rank, input_dim
        ),
        "lora_b": torch.arange(output_dim * rank, dtype=torch.float32).reshape(
            output_dim, rank
        ),
    }
    save_file(
        raw,
        str(root / "bridge.safetensors"),
        metadata={
            "artifact": f"artifact_{name}",
            "environmental_memory_serialized": "false",
            "questions_or_answers_serialized": "false",
            "oracle_serialized": "false",
        },
    )
    state = tensor_state_sha256(
        {
            "adapters.0.lora_a": raw["lora_a"],
            "adapters.0.lora_b": raw["lora_b"],
        }
    )
    weights_hash = sha256_file(root / "bridge.safetensors")
    metadata = {
        "artifact": f"artifact_{name}",
        "status": "fixed_final_awaiting_preregistered_acceptance_gates",
        "bank_name": name,
        "target_module": f"model.language_model.layers.{layer}.{target}",
        "rank": rank,
        "alpha": float(rank * 2),
        "dropout": 0.0,
        "parameter_count": raw["lora_a"].numel() + raw["lora_b"].numel(),
        "state_sha256": state,
        "weights_sha256": weights_hash,
        "environmental_memory_serialized": False,
        "questions_or_answers_serialized": False,
        "oracle_serialized": False,
        "evaluation_scored": False,
        "runtime_promotion_authorized": False,
    }
    _json(root / "runtime_metadata.json", metadata)
    return BridgeSourceContract(
        root=root,
        artifact=str(metadata["artifact"]),
        bank_name=name,
        target_module=str(metadata["target_module"]),
        rank=rank,
        alpha=float(metadata["alpha"]),
        dropout=0.0,
        parameter_count=int(metadata["parameter_count"]),
        state_sha256=state,
        weights_sha256=weights_hash,
        metadata_sha256=sha256_file(root / "runtime_metadata.json"),
    )


def _ten_bank_contract(tmp_path: Path) -> tuple[Path, dict[str, object], tuple[BridgeSourceContract, ...]]:
    base, metadata = _base_checkpoint(tmp_path / "base")
    bridges = (
        _bridge(
            tmp_path / "v86",
            name="v86_scene1_demo_bridge",
            layer=34,
            target="mlp.up_proj",
            rank=8,
            input_dim=2,
            output_dim=4,
        ),
        _bridge(
            tmp_path / "v87",
            name="v87_scene1_balanced_bridge",
            layer=34,
            target="mlp.gate_proj",
            rank=8,
            input_dim=2,
            output_dim=4,
        ),
        _bridge(
            tmp_path / "v88",
            name="v88_scene1_augmented_bridge",
            layer=27,
            target="self_attn.q_proj",
            rank=16,
            input_dim=3,
            output_dim=4,
        ),
    )
    return base, metadata, bridges


def test_release_core_composes_exact_ten_bank_v88_shape(tmp_path: Path) -> None:
    base, _metadata, bridges = _ten_bank_contract(tmp_path)
    final = BASE_BANKS + tuple(bridge.bank_name for bridge in bridges)
    archive, evidence = compose_exact_bank_archive(
        base_checkpoint=base,
        expected_base_banks=BASE_BANKS,
        added_bridges=bridges,
        expected_final_banks=final,
    )

    assert len(final) == 10
    assert evidence["base_bank_order"] == list(BASE_BANKS)
    assert evidence["final_bank_order"] == list(final)
    assert evidence["added_tensor_count"] == 6
    assert evidence["base_tensors_byte_identical"] is True
    assert {
        key
        for key in archive
        if key.startswith("lora_banks.v88_scene1_augmented_bridge.")
    } == {
        "lora_banks.v88_scene1_augmented_bridge.adapters.0.lora_a",
        "lora_banks.v88_scene1_augmented_bridge.adapters.0.lora_b",
    }


def test_release_core_extends_metadata_and_config_to_same_frozen_inventory(
    tmp_path: Path,
) -> None:
    _base, metadata, bridges = _ten_bank_contract(tmp_path)
    final = BASE_BANKS + tuple(bridge.bank_name for bridge in bridges)
    extended = extend_runtime_metadata(
        parent_metadata=metadata,
        added_bridges=bridges,
        expected_final_banks=final,
    )
    config = {
        "language": {
            "lora_banks": {
                name: {
                    "trainable": False,
                    "expected_initial_state_sha256": metadata[
                        "lora_bank_state_sha256"
                    ][name],
                }
                for name in BASE_BANKS
            }
        }
    }
    extended_config = extend_runtime_lora_config(
        parent_runtime_config=config,
        added_bridges=bridges,
        expected_final_banks=final,
    )
    # A frozen parent may have been initialized by an older unseeded path.
    # Its final checkpoint state remains hash-bound even when the historical
    # initialization digest is unavailable in both config and metadata.
    extended_config["language"]["lora_banks"][BASE_BANKS[0]][
        "expected_initial_state_sha256"
    ] = None
    extended["lora"]["banks"][0]["expected_initial_state_sha256"] = None
    states = {
        **metadata["lora_bank_state_sha256"],
        **{bridge.bank_name: bridge.state_sha256 for bridge in bridges},
    }

    validate_runtime_bank_inventory(
        runtime_config=extended_config,
        checkpoint_metadata=extended,
        expected_bank_order=final,
        expected_states=states,
    )
    assert extended["lora"]["trainable_adapter_parameter_count"] == 0
    assert extended["lora_trainable_parameter_count"] == 0
    assert tuple(extended_config["language"]["lora_banks"]) == final


def test_release_core_rejects_nonexact_bank_order_and_tampered_bridge(
    tmp_path: Path,
) -> None:
    base, _metadata, bridges = _ten_bank_contract(tmp_path)
    final = BASE_BANKS + tuple(bridge.bank_name for bridge in bridges)
    with pytest.raises(ValueError, match="exact parent-plus-additions"):
        compose_exact_bank_archive(
            base_checkpoint=base,
            expected_base_banks=BASE_BANKS,
            added_bridges=bridges,
            expected_final_banks=tuple(reversed(final)),
        )

    metadata_path = bridges[-1].root / "runtime_metadata.json"
    metadata_path.write_text(metadata_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source file bytes changed"):
        compose_exact_bank_archive(
            base_checkpoint=base,
            expected_base_banks=BASE_BANKS,
            added_bridges=bridges,
            expected_final_banks=final,
        )


def test_v88_contract_is_scene1_only_and_fresh_target_is_disjoint() -> None:
    parent_targets = {
        "model.language_model.layers.34.mlp.up_proj",
        "model.language_model.layers.34.mlp.gate_proj",
    }
    fresh = "model.language_model.layers.27.self_attn.q_proj"

    assert fresh not in parent_targets
    assert fresh.endswith("layers.27.self_attn.q_proj")
