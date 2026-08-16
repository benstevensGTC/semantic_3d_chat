from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from semantic_3d_chat.chat.runtime import validate_checkpoint_contract
from semantic_3d_chat.chat.runtime_config import (
    APPROVED_SYSTEM_PROMPT,
    load_runtime_config,
    validate_runtime_config,
)
from semantic_3d_chat.scene_encoder.dense_alignment import construct_dense_alignment
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    construct_dense_sidecar_adapter,
)

RUNTIME_CONFIG = Path("configs/runtime/gemma4_primary.yaml")


def test_primary_runtime_config_is_standalone_and_contains_no_supervision_vocabulary() -> None:
    source = RUNTIME_CONFIG.read_text(encoding="utf-8").casefold()
    assert "_base_" not in source
    for forbidden in (
        "qa",
        "oracle",
        "experiment",
        "category",
        "change_type",
        "chair",
        "bowl",
        "book",
        "picture",
        "cube",
        "table",
        "lamp",
    ):
        assert forbidden not in source

    config = load_runtime_config(RUNTIME_CONFIG)
    assert config["_runtime_safe_config"] is True
    assert config["language"]["system_prompt"] == APPROVED_SYSTEM_PROMPT
    assert config["runtime"]["reference_viewpoint"] == {
        "position_m": [0.0, 0.0, 1.4],
        "yaw_degrees": 0.0,
        "pitch_degrees": 0.0,
        "scan_view_count": 24,
    }
    assert set(config) == {
        "runtime",
        "paths",
        "scene",
        "vision",
        "scene_encoder",
        "language",
        "training",
        "_runtime_safe_config",
        "_config_path",
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("paths", "qa_root", "data_hidden/qa"),
        ("scene", "variant", "counterfactual"),
        ("training", "dataset", {"split": "test"}),
        ("language", "scene_description", "semantic text"),
    ],
)
def test_runtime_config_schema_rejects_training_or_environmental_fields(
    section: str,
    key: str,
    value: object,
) -> None:
    config = copy.deepcopy(load_runtime_config(RUNTIME_CONFIG))
    config.pop("_config_path")
    config.pop("_runtime_safe_config")
    config[section][key] = value
    with pytest.raises(ValueError, match="forbidden|non-inference"):
        validate_runtime_config(config)


def test_runtime_config_rejects_changed_system_prompt_and_experiment_location(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(load_runtime_config(RUNTIME_CONFIG))
    config.pop("_config_path")
    config.pop("_runtime_safe_config")
    config["language"]["system_prompt"] = "A textual room inventory."
    with pytest.raises(ValueError, match="exact approved"):
        validate_runtime_config(config)

    outside = tmp_path / "runtime.yaml"
    outside.write_text(RUNTIME_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="configs/runtime"):
        load_runtime_config(outside)

    config_alias = tmp_path / "runtime-alias.yaml"
    config_alias.symlink_to(RUNTIME_CONFIG.resolve())
    parent_alias = tmp_path / "runtime-parent"
    parent_alias.symlink_to(RUNTIME_CONFIG.resolve().parent, target_is_directory=True)
    for candidate in (config_alias, parent_alias / RUNTIME_CONFIG.name):
        with pytest.raises(ValueError, match="symbolic-link path components"):
            load_runtime_config(candidate)


def test_runtime_config_rejects_environmental_vocabulary_hidden_in_nested_values() -> None:
    config = copy.deepcopy(load_runtime_config(RUNTIME_CONFIG))
    config.pop("_config_path")
    config.pop("_runtime_safe_config")
    config["language"]["lora_banks"]["chair_hint"] = copy.deepcopy(
        config["language"]["lora_banks"]["inherited_v12"]
    )
    with pytest.raises(ValueError, match="environmental vocabulary"):
        validate_runtime_config(config)


def test_runtime_config_rejects_nonfinite_architecture_values() -> None:
    config = copy.deepcopy(load_runtime_config(RUNTIME_CONFIG))
    config.pop("_config_path")
    config.pop("_runtime_safe_config")
    config["scene_encoder"]["coverage_scale"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        validate_runtime_config(config)


def test_runtime_config_allows_null_lora_constructor_digest_for_checkpoint_overwrite() -> None:
    config = copy.deepcopy(load_runtime_config(RUNTIME_CONFIG))
    config.pop("_config_path")
    config.pop("_runtime_safe_config")
    bank = config["language"]["lora_banks"]["inherited_v12"]
    assert bank["initialization_algorithm"] == "checkpoint_overwrite"
    bank["expected_initial_state_sha256"] = None

    validated = validate_runtime_config(config)

    assert (
        validated["language"]["lora_banks"]["inherited_v12"][
            "expected_initial_state_sha256"
        ]
        is None
    )


def test_runtime_config_still_requires_digest_for_non_overwrite_lora_bank() -> None:
    config = copy.deepcopy(load_runtime_config(RUNTIME_CONFIG))
    config.pop("_config_path")
    config.pop("_runtime_safe_config")
    bank = config["language"]["lora_banks"]["inherited_v12"]
    bank["initialization_algorithm"] = "module_default"
    bank["expected_initial_state_sha256"] = None

    with pytest.raises(ValueError, match="initial digest is invalid"):
        validate_runtime_config(config)


def test_runtime_config_matches_v31_combined_sidecar_and_lora_contract_when_available() -> None:
    checkpoint = Path(
        "data_gemma4/checkpoints/gemma4_v31_diverse28_joint_pair/update_000"
    )
    metadata_path = checkpoint / "runtime_metadata.json"
    if not metadata_path.is_file():
        pytest.skip("Local V31 update-zero checkpoint is not materialized")
    config = load_runtime_config(RUNTIME_CONFIG)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dense = construct_dense_alignment(config, semantic_dim=int(metadata["semantic_dim"]))
    sidecar = construct_dense_sidecar_adapter(
        config,
        scene_dim=int(metadata["language_hidden_dim"]),
        latent_count=int(metadata["scene_latents"]),
    )
    assert dense is not None and sidecar is not None
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
        dense_alignment_parameter_count=dense.parameter_count,
        dense_sidecar_adapter_parameter_count=sidecar.parameter_count,
    )
    assert len(bank_counts) == 6
    assert sidecar.parameter_count == 604_416
    assert dense.parameter_count == 24_576
    assert len(warnings) == 1 and "config hash differs" in warnings[0].casefold()
