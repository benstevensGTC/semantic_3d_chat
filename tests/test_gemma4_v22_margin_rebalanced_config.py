from __future__ import annotations

import copy

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.scene_encoder import (
    SIGNED_X_LOCAL_FIELD_V2,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.training.train_adapter import validate_output_namespace

V21_CONFIG_PATH = (
    "configs/experiments/"
    "gemma4_color_mirror_signed_x_local_field_phase_aware_v21.yaml"
)
V22_CONFIG_PATH = (
    "configs/experiments/"
    "gemma4_color_mirror_signed_x_local_field_margin_rebalanced_v22.yaml"
)
EXPECTED_RESOLVED_CONFIG_HASH = "b336be25fd68"
EXPECTED_RESOLVED_CONFIG_SHA256 = (
    "b336be25fd68127191e86c2337d9b66baf0f5972cc6dade27bfcecfd5368c999"
)
PRIMARY_NAMESPACE = "gemma4_v22_margin_rebalanced_local_field"
EXTENSION_NAMESPACE = "gemma4_v22_margin_rebalanced_local_field_extension_u8"
V22_SCREEN_ROLE = "signed_x_local_field_phase_aware_margin_rebalanced_screen"
V22_EXPERIMENT_ROLE = (
    "exploratory_phase_aware_margin_rebalanced_reflection_odd_local_field_screen_v22"
)


def _without_loader_metadata(config: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(config)
    normalized.pop("_config_path", None)
    return normalized


def test_v22_resolved_hash_namespaces_and_roles_are_pinned() -> None:
    config = load_config(V22_CONFIG_PATH)
    training = config["training"]
    screen = config["v22_screen"]
    experiment = config["experiment"]

    assert config_hash(config) == EXPECTED_RESOLVED_CONFIG_HASH
    assert config_hash(config, length=64) == EXPECTED_RESOLVED_CONFIG_SHA256
    assert training["output_namespace"] == PRIMARY_NAMESPACE
    assert training["extension_output_namespace"] == EXTENSION_NAMESPACE
    assert screen["primary_output_namespace"] == PRIMARY_NAMESPACE
    assert screen["extension_output_namespace"] == EXTENSION_NAMESPACE
    assert PRIMARY_NAMESPACE != EXTENSION_NAMESPACE
    for namespace in (PRIMARY_NAMESPACE, EXTENSION_NAMESPACE):
        assert len(namespace) <= 64
        assert validate_output_namespace(namespace) == namespace

    assert screen["role"] == V22_SCREEN_ROLE
    assert experiment["role"] == V22_EXPERIMENT_ROLE


def test_v22_is_exact_v21_restart_except_versioning_and_rebalanced_margins() -> None:
    v21 = _without_loader_metadata(load_config(V21_CONFIG_PATH))
    observed = _without_loader_metadata(load_config(V22_CONFIG_PATH))

    # Normalize only the preregistered V22 versioning fields and the intended
    # mirror-margin change. Exact equality then fails on any other inherited
    # architecture, dtype, optimizer, order, loss-weight, gate, or source pin.
    training = observed["training"]
    training["output_namespace"] = v21["training"]["output_namespace"]
    training.pop("extension_output_namespace")
    mirror_policy = training["pair_objectives"]["by_pair"]["pair_000003"]
    mirror_policy["candidate_margin"] = 1.0
    mirror_policy["full_vocab_margin"] = 1.0

    screen = observed.pop("v22_screen")
    for key in (
        "primary_output_namespace",
        "extension_output_namespace",
        "target_optimizer_update",
    ):
        screen.pop(key)
    screen["role"] = v21["v21_screen"]["role"]
    observed["v21_screen"] = screen

    experiment = observed["experiment"]
    experiment["role"] = v21["experiment"]["role"]
    experiment.pop("question_dependent_retrieval")
    experiment.pop("runtime_oracle_access")

    assert observed == v21


def test_v22_margin_and_eight_update_contract_are_explicit() -> None:
    config = load_config(V22_CONFIG_PATH)
    training = config["training"]
    policies = training["pair_objectives"]["by_pair"]
    screen = config["v22_screen"]

    assert policies["pair_000001"] == {
        "role": "retention_control",
        "language_nll_weight": 0.0,
        "candidate_hinge_weight": 8.0,
        "candidate_margin": 0.25,
        "full_vocab_hinge_weight": 2.0,
        "full_vocab_margin": 0.25,
    }
    assert policies["pair_000003"] == {
        "role": "signed_target",
        "language_nll_weight": 0.0,
        "candidate_hinge_weight": 8.0,
        "candidate_margin": 0.25,
        "full_vocab_hinge_weight": 2.0,
        "full_vocab_margin": 0.25,
    }
    assert training["initialize_from"].endswith(
        "gemma4_color_mirror_centered_content_gate_v18/epoch_004"
    )
    assert training["epochs"] == 4
    assert training["learning_rate"] == 1.0e-4
    assert training["optimizer"]["learning_rate"] == 1.0e-4
    assert training["optimizer"]["accumulation_divisor"] == 12
    assert screen["screen_optimizer_updates"] == 4
    assert screen["conditional_max_optimizer_updates"] == 8
    assert screen["target_optimizer_update"] == 8
    assert screen["stage_1_optimizer_updates"] == 1
    assert screen["stage_1_stop_required"] is True

    settings = signed_x_scene_residual_settings(config)
    assert settings.enabled is True
    assert settings.architecture_version == SIGNED_X_LOCAL_FIELD_V2
    assert config["language"]["dtype"] == "bfloat16"


def test_v22_runtime_contract_forbids_query_retrieval_and_oracle_access() -> None:
    config = load_config(V22_CONFIG_PATH)
    experiment = config["experiment"]

    assert experiment["question_dependent_scene_processing"] is False
    assert experiment["question_dependent_retrieval"] is False
    assert experiment["runtime_oracle_access"] is False
    assert config["v21_screen"] is None
    assert config["structural_preflight"] is None
    assert config["v18_screen"] is None
