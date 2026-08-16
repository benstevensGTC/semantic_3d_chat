from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from semantic_3d_chat.chat.runtime import (
    StaticChatRuntime,
    _validate_frozen_block_cross_source_stack,
    validate_checkpoint_contract,
)
from semantic_3d_chat.chat.runtime_config import (
    load_runtime_config,
    validate_runtime_config,
)
from semantic_3d_chat.config import load_config
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    block_cross_residual_settings,
    construct_block_cross_residual,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.scene_encoder.projector import SceneTokenizerOutput
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    load_adapter_checkpoint,
    module_collection_state_sha256,
    runtime_checkpoint_metadata,
    save_adapter_checkpoint,
    validate_runtime_checkpoint_metadata,
)

RUNTIME_CONFIG = Path("configs/runtime/gemma4_primary.yaml")


def _block_cross_metadata() -> dict[str, object]:
    return {
        "schema_version": 3,
        "block_cross_residual": {
            "schema_version": 1,
            "enabled": True,
            "architecture_version": "all_block_fp32_cross_residual_zero_output_v1",
        },
        "block_cross_residual_parameter_count": 983_040,
        "block_cross_residual_initial_state_sha256": "a" * 64,
        "block_cross_residual_state_sha256": "b" * 64,
        "block_cross_residual_zero_output_equivalence": {
            "verified": True,
            "base": "exact_v33_update64_post_sidecar_scene_tokens",
            "application_order": "after_v33_dense_sidecar_before_prefix_composer",
            "all_scene_slots_accounted": True,
            "all_occupied_block_tokens_accounted": True,
            "normalized_block_positions_used": True,
            "all_voxels_covered": True,
            "question_dependent_scene_processing": False,
            "training_only_tensor_diagnostics": {"maximum_error": 0.0},
        },
        "frozen_block_cross_source_stack_state_sha256": "c" * 64,
    }


def test_v35_runtime_metadata_keeps_only_nonsemantic_block_cross_attestation() -> None:
    runtime = runtime_checkpoint_metadata(_block_cross_metadata())

    assert runtime["block_cross_residual_parameter_count"] == 983_040
    assert runtime["block_cross_residual_initial_state_sha256"] == "a" * 64
    assert runtime["block_cross_residual_state_sha256"] == "b" * 64
    assert runtime["frozen_block_cross_source_stack_state_sha256"] == "c" * 64
    assert runtime["block_cross_residual_zero_output_equivalence"] == {
        "verified": True,
        "base": "exact_v33_update64_post_sidecar_scene_tokens",
        "application_order": "after_v33_dense_sidecar_before_prefix_composer",
        "all_scene_slots_accounted": True,
        "all_occupied_block_tokens_accounted": True,
        "normalized_block_positions_used": True,
        "all_voxels_covered": True,
        "question_dependent_scene_processing": False,
    }
    validate_runtime_checkpoint_metadata(runtime)


def test_v35_runtime_metadata_removes_disabled_block_cross_surface() -> None:
    metadata = _block_cross_metadata()
    metadata["block_cross_residual"] = {"schema_version": 1, "enabled": False}

    runtime = runtime_checkpoint_metadata(metadata)

    assert not any(key.startswith("block_cross_residual") for key in runtime)
    assert "frozen_block_cross_source_stack_state_sha256" not in runtime


def test_safe_runtime_config_accepts_only_exact_block_cross_surface() -> None:
    config = copy.deepcopy(load_runtime_config(RUNTIME_CONFIG))
    config.pop("_config_path")
    config.pop("_runtime_safe_config")
    config["scene_encoder"]["block_cross_residual"] = {
        "enabled": False,
        "attention_dim": 256,
        "heads": 4,
        "spatial_temperature": 0.2,
        "residual_scale": 0.25,
        "uniform_floor": 0.01,
        "initialization_seed": 35035,
        "expected_initial_state_sha256": None,
    }

    validated = validate_runtime_config(config)
    assert validated["scene_encoder"]["block_cross_residual"]["enabled"] is False

    unknown = copy.deepcopy(config)
    unknown["scene_encoder"]["block_cross_residual"]["retrieval_limit"] = 8
    with pytest.raises(ValueError, match="forbidden keys"):
        validate_runtime_config(unknown)

    missing = copy.deepcopy(config)
    missing["scene_encoder"]["block_cross_residual"].pop("uniform_floor")
    with pytest.raises(ValueError, match="missing fields"):
        validate_runtime_config(missing)


def test_v35_frozen_source_stack_inventory_hash_detects_tamper() -> None:
    modules = {
        "scene_model": nn.Linear(3, 4),
        "dense_sidecar_adapter": nn.Linear(4, 4),
        "block_cross_residual": nn.Linear(4, 3),
    }
    expected = module_collection_state_sha256(
        {name: module for name, module in modules.items() if name != "block_cross_residual"}
    )

    assert _validate_frozen_block_cross_source_stack(modules, expected) == expected
    with pytest.raises(ValueError, match="mismatch or tamper"):
        _validate_frozen_block_cross_source_stack(modules, "0" * 64)
    with pytest.raises(ValueError, match="missing its residual"):
        _validate_frozen_block_cross_source_stack(
            {"scene_model": modules["scene_model"]}, expected
        )


def test_v35_checkpoint_module_inventory_is_strict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    modules = {
        "scene_model": nn.Linear(3, 4),
        "block_cross_residual": nn.Linear(4, 3),
    }
    metadata = _block_cross_metadata()
    save_adapter_checkpoint(checkpoint, modules, metadata)

    with pytest.raises(RuntimeError, match="unconsumed tensor keys"):
        load_adapter_checkpoint(
            checkpoint,
            {"scene_model": nn.Linear(3, 4)},
            metadata_filename=RUNTIME_METADATA_FILENAME,
        )

    loaded = load_adapter_checkpoint(
        checkpoint,
        {
            "scene_model": nn.Linear(3, 4),
            "block_cross_residual": nn.Linear(4, 3),
        },
        metadata_filename=RUNTIME_METADATA_FILENAME,
    )
    assert loaded == runtime_checkpoint_metadata(metadata)


def test_v35_checkpoint_metadata_is_rejected_when_runtime_branch_is_disabled() -> None:
    config = {
        "scene_encoder": {
            "input_voxel_size_m": 0.15,
            "model_dim": 6,
            "global_latents": 4,
        },
        "language": {
            "model_id": "local/tiny",
            "revision": "revision-1",
        },
    }
    metadata = {
        "schema_version": 3,
        "semantic_dim": 7,
        "language_hidden_dim": 8,
        "language_model_id": "local/tiny",
        "language_revision": "revision-1",
        "scene_latents": 4,
        "scene_model_dim": 6,
        "input_voxel_size_m": 0.15,
        "config_hash": "different",
        **_block_cross_metadata(),
    }

    with pytest.raises(ValueError, match="runtime residual is disabled"):
        validate_checkpoint_contract(
            metadata,
            config,
            semantic_dim=7,
            language_hidden_dim=8,
        )


def test_v35_runtime_attests_every_block_token_and_voxel_before_questions() -> None:
    runtime = object.__new__(StaticChatRuntime)
    runtime.config = {"scene_encoder": {"block_size_m": 0.25, "tokens_per_block": 2}}
    xyz = torch.tensor(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.51, 0.01, 0.01],
            [0.52, 0.02, 0.02],
            [0.53, 0.03, 0.03],
        ]
    )
    runtime.map_data = MapTensorData(
        semantic=torch.zeros(5, 7),
        xyz=xyz,
        rgb=torch.zeros(5, 3),
        normal=torch.zeros(5, 3),
        confidence=torch.ones(5),
        observation_count=torch.ones(5),
        room_min=torch.zeros(3),
        room_max=torch.ones(3),
        source_voxel_count=5,
        input_voxel_size_m=0.15,
    )
    output = SceneTokenizerOutput(
        scene_tokens=torch.zeros(1, 4, 8),
        native_latents=torch.zeros(1, 4, 6),
        block_tokens=torch.zeros(4, 6),
        audit={
            "block_token_positions_normalized": torch.tensor(
                [
                    [-0.75, -0.75, -0.75],
                    [-0.75, -0.75, -0.75],
                    [0.25, -0.75, -0.75],
                    [0.25, -0.75, -0.75],
                ]
            ),
            "block_indices": torch.tensor([[0, 0, 0], [2, 0, 0]]),
            "voxel_counts": torch.tensor([2, 3]),
            "voxel_to_block": torch.tensor([0, 0, 1, 1, 1]),
        },
    )

    attestation = runtime._attest_complete_block_cross_inputs(output)

    assert attestation == {
        "all_scene_slots_accounted": True,
        "all_occupied_block_tokens_accounted": True,
        "occupied_block_indices_match_map": True,
        "voxel_to_block_assignments_match_map": True,
        "block_positions_match_normalized_centers": True,
        "normalized_block_positions_used": True,
        "all_voxels_covered": True,
        "occupied_blocks": 2,
        "block_token_count": 4,
        "voxel_count": 5,
        "question_dependent_scene_processing": False,
    }

    output.block_tokens = output.block_tokens[:3]
    with pytest.raises(RuntimeError, match="one normalized XYZ position"):
        runtime._attest_complete_block_cross_inputs(output)


def test_v35_runtime_accepts_exact_surface_block_centers_outside_unit_cube() -> None:
    runtime = object.__new__(StaticChatRuntime)
    runtime.config = {"scene_encoder": {"block_size_m": 0.25, "tokens_per_block": 2}}
    xyz = torch.tensor(
        [
            [-3.025, -2.525, -0.025],
            [3.025, 2.525, 3.025],
        ],
        dtype=torch.float32,
    )
    runtime.map_data = MapTensorData(
        semantic=torch.zeros(2, 7),
        xyz=xyz,
        rgb=torch.zeros(2, 3),
        normal=torch.zeros(2, 3),
        confidence=torch.ones(2),
        observation_count=torch.ones(2),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=2,
        input_voxel_size_m=0.15,
    )
    block_indices = torch.tensor([[-1, -1, -1], [24, 20, 12]])
    room_min = runtime.map_data.room_min
    extent = runtime.map_data.room_max - room_min
    centers = room_min + (block_indices.float() + 0.5) * 0.25
    positions = (((centers - room_min) / extent) * 2 - 1).repeat_interleave(2, dim=0)
    assert torch.any(positions.abs() > 1.0)
    output = SceneTokenizerOutput(
        scene_tokens=torch.zeros(1, 4, 8),
        native_latents=torch.zeros(1, 4, 6),
        block_tokens=torch.zeros(4, 6),
        audit={
            "block_token_positions_normalized": positions,
            "block_indices": block_indices,
            "voxel_counts": torch.tensor([1, 1]),
            "voxel_to_block": torch.tensor([0, 1]),
        },
    )

    attestation = runtime._attest_complete_block_cross_inputs(output)

    assert attestation["occupied_block_indices_match_map"] is True
    assert attestation["block_positions_match_normalized_centers"] is True


def test_v35_runtime_rejects_block_geometry_not_reconstructed_from_map() -> None:
    runtime = object.__new__(StaticChatRuntime)
    runtime.config = {"scene_encoder": {"block_size_m": 0.25, "tokens_per_block": 2}}
    runtime.map_data = MapTensorData(
        semantic=torch.zeros(1, 7),
        xyz=torch.tensor([[-0.025, 0.025, 0.025]]),
        rgb=torch.zeros(1, 3),
        normal=torch.zeros(1, 3),
        confidence=torch.ones(1),
        observation_count=torch.ones(1),
        room_min=torch.zeros(3),
        room_max=torch.ones(3),
        source_voxel_count=1,
        input_voxel_size_m=0.15,
    )
    output = SceneTokenizerOutput(
        scene_tokens=torch.zeros(1, 4, 8),
        native_latents=torch.zeros(1, 4, 6),
        block_tokens=torch.zeros(2, 6),
        audit={
            "block_token_positions_normalized": torch.tensor(
                [[-1.25, -0.75, -0.75], [-1.25, -0.75, -0.75]]
            ),
            "block_indices": torch.tensor([[0, 0, 0]]),
            "voxel_counts": torch.tensor([1]),
            "voxel_to_block": torch.tensor([0]),
        },
    )

    with pytest.raises(RuntimeError, match="indices do not match the complete map"):
        runtime._attest_complete_block_cross_inputs(output)

    output.audit["block_indices"] = torch.tensor([[-1, 0, 0]])
    output.audit["block_token_positions_normalized"] = torch.tensor(
        [[-1.0, -0.75, -0.75], [-1.0, -0.75, -0.75]]
    )
    with pytest.raises(RuntimeError, match="positions do not match normalized"):
        runtime._attest_complete_block_cross_inputs(output)


def test_v35_enabled_checkpoint_contract_matches_exact_v33_source_when_available() -> None:
    source = Path(
        "data_gemma4/checkpoints/"
        "gemma4_v33_diverse28_environmental_sidecar/update_064/runtime_metadata.json"
    )
    if not source.is_file():
        pytest.skip("Local exact V33 update-64 runtime checkpoint is not materialized")
    config = load_config(
        "configs/experiments/gemma4_diverse28_block_cross_v35.yaml"
    )
    metadata = json.loads(source.read_text(encoding="utf-8"))
    module = construct_block_cross_residual(
        config,
        scene_dim=int(metadata["language_hidden_dim"]),
        block_dim=int(metadata["scene_model_dim"]),
        latent_count=int(metadata["scene_latents"]),
    )
    assert module is not None
    settings = block_cross_residual_settings(config)
    metadata.update(
        {
            "block_cross_residual": settings.contract(),
            "block_cross_residual_parameter_count": module.parameter_count,
            "block_cross_residual_initial_state_sha256": (
                settings.expected_initial_state_sha256
            ),
            "block_cross_residual_state_sha256": module.state_sha256(),
            "block_cross_residual_zero_output_equivalence": {
                "verified": True,
                "base": "exact_v33_update64_post_sidecar_scene_tokens",
                "application_order": (
                    "after_v33_dense_sidecar_before_prefix_composer"
                ),
                "all_scene_slots_accounted": True,
                "all_occupied_block_tokens_accounted": True,
                "normalized_block_positions_used": True,
                "all_voxels_covered": True,
                "question_dependent_scene_processing": False,
            },
            "frozen_block_cross_source_stack_state_sha256": "c" * 64,
        }
    )
    bank_counts = {
        bank: sum(module_counts.values())
        for bank, module_counts in metadata["lora_bank_parameter_counts"].items()
    }

    warnings = validate_checkpoint_contract(
        metadata,
        config,
        semantic_dim=int(metadata["semantic_dim"]),
        language_hidden_dim=int(metadata["language_hidden_dim"]),
        lora_parameter_counts=bank_counts,
        dense_alignment_parameter_count=int(
            metadata["dense_alignment_parameter_count"]
        ),
        dense_sidecar_adapter_parameter_count=int(
            metadata["dense_sidecar_adapter_parameter_count"]
        ),
        block_cross_residual_parameter_count=module.parameter_count,
    )
    assert len(warnings) == 1 and "config hash differs" in warnings[0].casefold()
