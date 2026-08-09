from __future__ import annotations

from semantic_3d_chat.config import load_config
from semantic_3d_chat.scene_encoder import (
    SIGNED_X_LOCAL_FIELD_V2,
    SignedXLocalFieldSceneResidual,
    construct_signed_x_scene_residual,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256
from semantic_3d_chat.training.train_adapter import (
    declared_signed_x_scene_residual_parameter_count,
)

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_local_field_v20.yaml"
INITIAL_STATE_SHA256 = "3f249307901df75ba07a758a7dc5b02c7c6ff9bbb969987741a106b8d8977ce1"


def test_v20_config_restarts_from_exact_v18_and_changes_only_local_field_architecture() -> None:
    config = load_config(CONFIG_PATH)
    training = config["training"]
    screen = config["v20_screen"]
    settings = signed_x_scene_residual_settings(config)

    assert settings.architecture_version == SIGNED_X_LOCAL_FIELD_V2
    assert settings.expected_initial_state_sha256 == INITIAL_STATE_SHA256
    assert training["output_namespace"] == "gemma4_color_mirror_signed_x_local_field_v20"
    assert training["initialize_from"].endswith(
        "gemma4_color_mirror_centered_content_gate_v18/epoch_004"
    )
    assert training["initialize_expected_adapter_sha256"] == (
        "1a7946d2e40aaf4bf66dc570bff19fa8d6ba4425e4e0d59bd52b809bd23dae7a"
    )
    assert training["initialize_expected_metadata_sha256"] == (
        "4853355ef4810f284d9b36eca1f0f1ade71319f4f6f579a5b079ce6178eb2344"
    )
    assert training["initialize_expected_global_scene_residual_state_sha256"] == (
        "ce3bf864eed6dd4a50f1b67296981e144d6a79e9cf192ad9a9230f2ae18208dc"
    )
    assert training["freeze_scene_adapter"] is True
    assert training["train_global_scene_residual_only"] is False
    assert training["train_signed_x_scene_residual_only"] is True
    assert training["epochs"] == 4
    assert training["learning_rate"] == 1e-4
    assert screen["conditional_max_optimizer_updates"] == 8
    assert screen["continuation_requires"] == {
        "mirror_minimum_full_vocab_sides": 8,
        "mirror_minimum_full_vocab_units": 2,
    }


def test_v20_initial_hash_parameter_count_and_pair_objectives_are_exact() -> None:
    config = load_config(CONFIG_PATH)
    module = construct_signed_x_scene_residual(
        config,
        scene_dim=1536,
        latent_count=256,
        content_dim=128,
    )
    assert isinstance(module, SignedXLocalFieldSceneResidual)
    assert declared_signed_x_scene_residual_parameter_count(config) == 196_608
    assert module_collection_state_sha256({"signed_x_scene_residual": module}) == (
        INITIAL_STATE_SHA256
    )

    v19 = load_config("configs/experiments/gemma4_color_mirror_signed_x_moment_v19.yaml")
    assert config["training"]["pair_objectives"] == v19["training"]["pair_objectives"]
    for key in ("learning_rate", "weight_decay", "optimizer"):
        assert config["training"][key] == v19["training"][key]
