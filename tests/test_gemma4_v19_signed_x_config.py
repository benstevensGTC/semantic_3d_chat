from __future__ import annotations

import json
from pathlib import Path

from semantic_3d_chat.config import load_config
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    global_scene_residual_settings,
)
from semantic_3d_chat.scene_encoder.signed_x_residual import (
    SIGNED_X_MOMENT_V1,
    construct_signed_x_scene_residual,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256
from semantic_3d_chat.training.pair_curriculum import (
    pair_curriculum_settings,
    pair_objective_policy_settings,
)
from semantic_3d_chat.training.train_adapter import (
    declared_signed_x_scene_residual_parameter_count,
    file_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_moment_v19.yaml"
SOURCE = (
    PROJECT_ROOT / "data_gemma4/checkpoints/gemma4_color_mirror_centered_content_gate_v18/epoch_004"
)
INITIAL_SIGNED_X_SHA256 = "55b7cb21d0ecbe945cabccfacd5b6aa94693743ceee78443f37a5ca0d1ac68b1"
SOURCE_GLOBAL_RESIDUAL_SHA256 = "ce3bf864eed6dd4a50f1b67296981e144d6a79e9cf192ad9a9230f2ae18208dc"


def test_v19_config_pins_exact_v18_source_and_reflection_odd_surface() -> None:
    config = load_config(CONFIG_PATH)
    training = config["training"]
    global_settings = global_scene_residual_settings(config)
    signed_settings = signed_x_scene_residual_settings(config)
    source_metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))

    assert config["structural_preflight"] is None
    assert config["v18_screen"] is None
    assert config["v19_screen"]["role"] == "signed_x_moment_architecture_screen"
    assert global_settings.architecture_version == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1
    assert signed_settings.architecture_version == SIGNED_X_MOMENT_V1
    assert training["initialize_source_residual_into_frozen_base"] is True
    assert training["train_global_scene_residual_only"] is False
    assert training["train_signed_x_scene_residual_only"] is True
    assert training["freeze_scene_adapter"] is True
    assert training["learning_rate"] == 1.0e-4
    assert training["optimizer"]["learning_rate"] == 1.0e-4
    assert training["initialize_expected_adapter_sha256"] == file_sha256(
        SOURCE / "adapter.safetensors"
    )
    assert training["initialize_expected_metadata_sha256"] == file_sha256(SOURCE / "metadata.json")
    assert (
        training["initialize_expected_global_scene_residual_state_sha256"]
        == SOURCE_GLOBAL_RESIDUAL_SHA256
        == source_metadata["global_scene_residual_state_sha256"]
    )


def test_v19_signed_x_initial_hash_parameter_count_and_pair_policies_are_exact() -> None:
    config = load_config(CONFIG_PATH)
    module = construct_signed_x_scene_residual(
        config,
        scene_dim=1536,
        latent_count=256,
        content_dim=128,
    )
    assert module is not None
    curriculum = pair_curriculum_settings(config)
    policies = pair_objective_policy_settings(config)

    assert module.parameter_count == 196_608
    assert declared_signed_x_scene_residual_parameter_count(config) == 196_608
    assert (
        module_collection_state_sha256({"signed_x_scene_residual": module})
        == INITIAL_SIGNED_X_SHA256
    )
    assert curriculum.enabled is True
    assert curriculum.pair_only is True
    assert policies.pair_ids == ("pair_000001", "pair_000003")
    retention = policies.resolve("pair_000001")
    target = policies.resolve("pair_000003")
    assert retention.language_nll_weight == target.language_nll_weight == 0.0
    assert retention.candidate_hinge_weight == target.candidate_hinge_weight == 8.0
    assert retention.full_vocab_hinge_weight == target.full_vocab_hinge_weight == 2.0
    assert retention.candidate_margin == retention.full_vocab_margin == 0.25
    assert target.candidate_margin == target.full_vocab_margin == 1.0
