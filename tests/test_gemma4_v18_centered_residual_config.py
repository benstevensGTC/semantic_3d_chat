from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.data.dataset import SceneQADataset
from semantic_3d_chat.evaluation.v18_structural_preflight import (
    file_sha256,
    ordered_curriculum_evidence,
    validate_v18_config_contract,
)
from semantic_3d_chat.language.lora import (
    lora_banks_optimizer_settings,
    lora_banks_settings,
    tensor_state_sha256,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    construct_global_scene_residual,
    global_scene_residual_settings,
)
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256
from semantic_3d_chat.training.pair_curriculum import (
    build_epoch_curriculum,
    build_exact_question_pair_units,
    cap_pair_units_per_pair,
    pair_curriculum_settings,
    select_pair_only_records,
)
from semantic_3d_chat.training.train_adapter import (
    build_adapter_optimizer,
    select_training_records,
)

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_centered_content_gate_v18.yaml"
CONFIG_SHA256 = "38b0fd8e679d239f5512c3df20c6b6080031062802c3c6b4d456e359e21d1dfe"
CONTRACT_SHA256 = "cce5899c4ab94096a8424823a5b8df63e9871c4ee9d0f79695238bdf0fb2798a"
IMPLEMENTATION_SHA256 = "10a0e9f5ce69636ea70c0948503f61b8fd966be35634048102a86033841eaa32"
INITIAL_STATE_SHA256 = "f7f6353edb6216029bd155e2baab1b5051c85f297a0e6d6b63210354fe0ff0e0"
POSITION_SHA256 = "43f748ee96bc8a50d061d8bec6faef27821d0e1d6779b38bf77c703f72c58fa2"
ORDERED_UNIT_SHA256 = "1d77157b18636abc6a5dd4a2d63bc62861d7c8147832105d40b87f1470fa3359"


def _config() -> dict:
    return load_config(CONFIG_PATH)


def test_real_v18_config_contract_source_and_evidence_are_pinned() -> None:
    config = _config()
    implementation = PROJECT_ROOT / "src/semantic_3d_chat/scene_encoder/global_residual.py"
    observed_implementation = file_sha256(implementation)
    contract = validate_v18_config_contract(
        config,
        implementation_source_sha256=observed_implementation,
    )

    assert observed_implementation == IMPLEMENTATION_SHA256
    assert config_hash(config, length=64) == CONFIG_SHA256
    assert contract["contract_sha256"] == CONTRACT_SHA256
    assert contract["expected_hashes"]["ordered_unit_sha256"] == ORDERED_UNIT_SHA256
    assert config["sweep"] is None
    assert config["lr_response"] is None
    assert config["experiment"]["question_dependent_scene_processing"] is False
    for name, path in contract["evidence_paths"].items():
        expected = contract["expected_hashes"][f"{name}_sha256"]
        assert file_sha256(PROJECT_ROOT / path) == expected

    source = Path(config["training"]["initialize_from"])
    source = source if source.is_absolute() else PROJECT_ROOT / source
    assert file_sha256(source / "adapter.safetensors") == contract["expected_hashes"][
        "source_adapter_sha256"
    ]
    assert file_sha256(source / "metadata.json") == contract["expected_hashes"][
        "source_metadata_sha256"
    ]


def test_real_v18_residual_hash_structure_and_optimizer_surface_are_exact() -> None:
    config = _config()
    settings = global_scene_residual_settings(config)
    residual = construct_global_scene_residual(config, scene_dim=1536, latent_count=256)
    assert residual is not None

    assert settings.architecture_version == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1
    assert settings.contract()["schema_version"] == 2
    assert residual.parameter_count == config["experiment"]["residual_parameter_count"] == 400_128
    assert module_collection_state_sha256({"global_scene_residual": residual}) == (
        INITIAL_STATE_SHA256
    )
    assert tensor_state_sha256({"position_features": residual.position_features}) == (
        POSITION_SHA256
    )
    assert residual.validate_structural_state() == {
        "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        "parameter_count": 400_128,
        "latent_count": 256,
        "scene_dim": 1536,
        "gate_temperature": 1.0,
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
    }
    assert torch.count_nonzero(residual.output_projection.weight).item() == 0

    banks = lora_banks_settings(config)
    assert banks.trainable is False
    assert lora_banks_optimizer_settings(config, banks) is None
    optimizer, parameters = build_adapter_optimizer(
        config,
        list(residual.parameters()),
        None,
        None,
    )
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in residual.parameters()
    }
    group = optimizer.param_groups[0]
    assert group["lr"] == 1.0e-3
    assert group["weight_decay"] == 0.0
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1.0e-8
    assert group["foreach"] is False
    assert group["fused"] is False
    assert group["capturable"] is False
    assert group["maximize"] is False
    assert group["amsgrad"] is False


def test_real_v18_epoch_one_order_hash_matches_training_scheduler() -> None:
    config = _config()
    settings = pair_curriculum_settings(config)
    records = list(SceneQADataset(artifact_root(config, "qa") / "train.jsonl").records)
    records = select_pair_only_records(records, settings.pair_only_scene_ids)
    records = cap_pair_units_per_pair(
        records,
        settings.max_units_per_pair,
        seed=int(config["seed"]),
    )
    records = select_training_records(
        records,
        max_questions_per_scene=config["training"].get("max_questions_per_scene"),
    )
    by_scene: dict[str, list] = defaultdict(list)
    for record in records:
        by_scene[record.scene_id].append(record)
    curriculum = build_epoch_curriculum(
        by_scene,
        build_exact_question_pair_units(records),
        standard_batch_size=int(config["training"]["batch_size"]),
        pair_units_per_batch=settings.units_per_batch,
        pair_batch_fraction=settings.batch_fraction,
        pair_only=settings.pair_only,
        seed=int(config["seed"]) + 1,
        steps_per_epoch=settings.steps_per_epoch,
    )

    ordered, observed = ordered_curriculum_evidence(curriculum)

    assert len(curriculum) == len(ordered) == 12
    assert observed == ORDERED_UNIT_SHA256
    assert {entry["pair_id"] for entry in ordered} == {"pair_000001", "pair_000003"}
