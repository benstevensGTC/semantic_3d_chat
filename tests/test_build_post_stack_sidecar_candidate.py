from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import load_file

from scripts.build_post_stack_sidecar_candidate import (
    _verify_exact_zero_identity,
    build_candidate,
)
from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.scene_encoder.dense_alignment import dense_alignment_settings
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    construct_dense_sidecar_adapter,
    dense_sidecar_adapter_settings,
)
from semantic_3d_chat.training.checkpointing import validate_runtime_checkpoint_metadata

CONFIG = PROJECT_ROOT / "configs/experiments/gemma4_color_mirror_post_stack_sidecar_v28.yaml"
BASE = PROJECT_ROOT / "data_gemma4/checkpoints/gemma4_v24_shared_query/epoch_001"
BRIDGE = PROJECT_ROOT / "reports/gemma4/artifacts/v26_dense_alignment_bridge.safetensors"


def test_v28_config_pins_zero_routing_and_bounded_stage_a() -> None:
    config = load_config(CONFIG)

    assert config_hash(config) == "ecc1a26b71a8"
    dense = dense_alignment_settings(config)
    assert dense.enabled is True
    assert dense.application_mode == "coverage_sidecar"
    assert dense.sidecar_scale == 0.0
    sidecar = dense_sidecar_adapter_settings(config)
    assert sidecar.enabled is True
    assert sidecar.width == 128
    assert sidecar.fourier_bands == 8
    assert sidecar.max_direct_scale == 0.25
    assert sidecar.initialization_seed == 28028
    assert sidecar.expected_initial_state_sha256 == (
        "fd8abd78bed2d04ac2f83d70564028285c361729c62591a05e3eddcbb8794d02"
    )
    assert config["training"]["post_stack_sidecar_stage_a"] == {
        "enabled": True,
        "max_optimizer_steps": 4,
        "evaluation_interval_steps": 1,
        "batch_size": 1,
        "gradient_accumulation": 12,
        "learning_rate": 0.0001,
        "channel_gain_learning_rate": 0.0001,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "minimum_answer_types": 4,
        "trainable_routes": ["output_projection", "channel_gain"],
    }
    assert config["v28_screen"]["base_adapter_sha256"] == (
        "45e0c5affa9cf556e29bab5de418dffb867817b703c848bc6828255347748d31"
    )
    assert config["v28_screen"]["calibrated_bridge_sha256"] == (
        "3340c453ded5152775e34e6e40ccb7e97dda1d7201e7321ff15c408edf92a83a"
    )
    assert config["v28_screen"]["stage_a_selection_requires"] == {
        "color_full_vocab_sides": 12,
        "mirror_full_vocab_sides": 10,
        "maximum_relative_prefix_rms_drift": 0.10,
        "minimum_prefix_cosine": 0.995,
    }


def test_v28_adapter_is_exact_identity_at_construction() -> None:
    config = load_config(CONFIG)
    adapter = construct_dense_sidecar_adapter(config, scene_dim=1536, latent_count=256)
    assert adapter is not None

    audit = _verify_exact_zero_identity(adapter)

    assert audit["verified"] is True
    assert audit["bit_identical_output"] is True
    assert audit["delta_nonzero_count"] == 0
    assert audit["all_scene_slots_accounted"] is True
    assert audit["all_voxels_covered"] is True


def test_builder_refuses_overwrite_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_candidate(
            config_path=tmp_path / "missing.yaml",
            base_checkpoint=tmp_path / "missing-base",
            bridge_path=tmp_path / "missing-bridge.safetensors",
            output=output,
        )


@pytest.mark.skipif(
    not (BASE / "adapter.safetensors").is_file()
    or not (BASE / "metadata.json").is_file()
    or not BRIDGE.is_file(),
    reason="Pinned local V24/V26 research artifacts are unavailable",
)
def test_builder_combines_pinned_artifacts_and_sanitizes_runtime(
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"
    report = build_candidate(
        config_path=CONFIG,
        base_checkpoint=BASE,
        bridge_path=BRIDGE,
        output=output,
    )

    assert report["adapter_sha256"] == (
        "cf6802c73e689816a5796c920a97c12704e529802c96aa9857087c417ea7a0ac"
    )
    assert report["zero_output_equivalence_verified"] is True
    assert report["base_semantic_path_modified"] is False
    assert report["oracle_loaded"] is False
    assert report["qa_loaded"] is False

    tensors = load_file(output / "adapter.safetensors", device="cpu")
    assert any(name.startswith("scene_model.") for name in tensors)
    assert {name for name in tensors if name.startswith("dense_aligner.")} == {
        "dense_aligner.alignment_a",
        "dense_aligner.alignment_b",
        "dense_aligner.architecture_marker",
        "dense_aligner.scaling",
    }
    assert len(
        [name for name in tensors if name.startswith("dense_sidecar_adapter.")]
    ) == 15

    training = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    runtime = json.loads(
        (output / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    validate_runtime_checkpoint_metadata(runtime)
    assert training["candidate_construction"]["oracle_loaded"] is False
    assert "candidate_construction" not in runtime
    assert "scene_ids" not in runtime
    assert "history" not in runtime
    assert runtime["dense_sidecar_adapter_zero_output_equivalence"] == {
        "verified": True,
        "base": "loaded_v24_post_signed_x_scene_tokens",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
        "all_voxels_covered": True,
        "application_order": "after_global_and_signed_x_before_prefix_composer",
    }

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_candidate(
            config_path=CONFIG,
            base_checkpoint=BASE,
            bridge_path=BRIDGE,
            output=output,
        )
