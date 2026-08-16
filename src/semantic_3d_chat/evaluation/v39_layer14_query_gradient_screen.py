"""No-step V39 gradient screen for Gemma 4's layer-14 producer query.

This module is deliberately diagnostic, not a trainer.  It loads the exact
V38 update-zero K-only hybrid, temporarily enables gradients for the two
already-existing layer-14 ``q_proj`` LoRA tensors, measures gradients with
``torch.autograd.grad``, and restores a completely gradient-free surface.
It never constructs an optimizer, calls ``backward``/``optimizer.step``, saves
a checkpoint, or reads validation, final-scene, oracle, or optimizer inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.lora import (
    LoRAInstallation,
    tensor_state_sha256,
    validate_lora_banks_checkpoint_state,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    BlockCrossResidual,
    validate_block_cross_residual_state,
)
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    load_adapter_checkpoint,
    module_collection_state_sha256,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    build_exact_question_pair_units,
)
from semantic_3d_chat.training.train_adapter import map_forward
from semantic_3d_chat.training.train_block_cross_v35 import (
    V35SceneCache,
    broad_answer_nll,
    current_scene_tokens,
    load_v35_train_qa_records,
    paired_cross_prefix_objective,
    validate_v35_cache_audit,
    validate_v35_scene_cache,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    construct_v36_source_core,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    V30Bundle,
    load_v30_bundle,
    require_approved_v29_source,
)
from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract
from semantic_3d_chat.training.train_query_recovery_v38 import (
    _PRIORITY_KEYS,
    _pair_family,
    build_v38_schedule,
    retag_bundle_for_v38,
    v38_contract,
    v38_loader_config,
)
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    validate_v37_training_cache_boundary,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_query_recovery_v38.yaml")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v39_layer14_query_gradient_screen.json")
DEFAULT_TERMINAL = Path("reports/gemma4/metrics/v38_update8_terminal_gate.json")
SOURCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v38_diverse28_query_recovery/update_000"
)

_V38_TERMINAL_SHA256 = "1015949e802abccd562f7762cc01111818646527f3366aeaf01de3854bbe164a"
_V38_CONFIG_SHA256 = "df884cdebed805fb783d68981011c2a66f1a37dc27aa8ecb529e1b981d25a7c5"
_SOURCE_FILE_SHA256 = {
    "adapter.safetensors": (
        "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0"
    ),
    TRAINING_METADATA_FILENAME: (
        "9a4b03e8fd7f8a6ef50b6d85ae6c07c602f353ecfe104dae28efaa239da5a0ed"
    ),
    RUNTIME_METADATA_FILENAME: (
        "7ec71195b6187524b903f8955af4db375b109c890fbbda9986f179b97dc58d30"
    ),
}
_SOURCE_FULL_STATE_SHA256 = (
    "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
)
_V28_BANK = "extension_v28_stage_b_query"
_V28_PREFIX = f"lora_banks.{_V28_BANK}."
_TARGET_ADAPTER_INDEX = 1
_TARGET_MODULE = "model.language_model.layers.14.self_attn.q_proj"
_TARGET_NAMES = (
    f"{_V28_PREFIX}adapters.1.lora_a",
    f"{_V28_PREFIX}adapters.1.lora_b",
)
_TARGET_SHAPES = ((4, 1536), (4096, 4))
_TARGET_PARAMETER_COUNT = 22_528
_TARGET_STATE_SHA256 = "9ff9d535a094f96328483c46ff8c8ea5fca30edc35878492976c35f8674a9f87"
_V28_STATE_SHA256 = "cc9dfa838bb87f32e2922d675658af4a1085d53a84ccdca6d5bacc6f7097217b"
_FROZEN_STATE_SHA256 = "7f33e541d36de33b10ceeac25e5f40374bffd1cf4b234af7a6b6341198b85360"
_FROZEN_TENSOR_COUNT = 177
_CORE_STATE_SHA256 = "92652fd2dbde2406227503f50717b2031baa1bcbc050902a379ddb9ddb52764f"

_PAIR_CORRECT_NLL_WEIGHT = 0.5
_SIDE_HINGE_WEIGHT = 8.0
_SIDE_HINGE_MARGIN = 0.5
_CROSS_PREFIX_HINGE_WEIGHT = 4.0
_CROSS_PREFIX_MARGIN = 0.10
_BROAD_NLL_WEIGHT = 1.0
_BROAD_ROW_COUNT = 8

# This is declared before any gradient is measured.  It is a causal-screen
# criterion, not permission to train or promote a runtime.
_PASS_CONTRACT: Mapping[str, Any] = {
    "schema_version": 1,
    "target_tensor_count_exact": 2,
    "target_parameter_count_exact": 22_528,
    "require_all_full_priority_target_gradients_finite": True,
    "require_all_full_priority_target_tensors_nonzero": True,
    "require_broad_aggregate_target_tensors_finite": True,
    "require_broad_aggregate_target_tensors_nonzero": True,
    "require_scene_discriminative_aggregate_target_tensors_finite": True,
    "require_scene_discriminative_aggregate_target_tensors_nonzero": True,
    "require_frozen_complement_bit_exact": True,
    "require_frozen_complement_has_no_gradients": True,
    "require_target_state_bit_exact": True,
    "require_full_checkpoint_state_bit_exact": True,
    "require_model_version_counters_unchanged": True,
    "require_no_optimizer_constructed_or_opened": True,
    "require_no_parameter_step_or_checkpoint_write": True,
    "require_temporary_requires_grad_surface_restored_to_frozen": True,
    "directional_pairs": [
        ["proposed_training_aggregate", "book_support_aggregate"],
        ["proposed_training_aggregate", "picture_support_aggregate"],
        ["proposed_training_aggregate", "book_scene_discriminative_aggregate"],
        ["proposed_training_aggregate", "picture_scene_discriminative_aggregate"],
        ["proposed_training_aggregate", "broad_retention_aggregate"],
        ["proposed_training_aggregate", "scene_discriminative_aggregate"],
        ["proposed_training_aggregate", "cross_prefix_maintenance_aggregate"],
    ],
    "directional_dot_product_strictly_positive": True,
    "directional_cosine_minimum_inclusive": 0.0,
    "diagnostic_pass_does_not_authorize_training_or_promotion": True,
}

_EXPECTED_GEMMA_ARCHITECTURE: Mapping[str, Any] = {
    "language_layer_count": 35,
    "num_kv_shared_layers": 20,
    "first_shared_kv_layer": 15,
    "layer_13_attention_type": "sliding_attention",
    "layer_14_attention_type": "full_attention",
    "layer_13_role": "last_nonshared_sliding_kv_producer",
    "layer_14_role": "last_nonshared_full_kv_producer",
    "layers_15_through_34_reuse_shared_kv_states": True,
}


@dataclass(frozen=True)
class V39Source:
    terminal: Mapping[str, Any]
    authorization: Mapping[str, Any]
    metadata: Mapping[str, Any]
    runtime_metadata: Mapping[str, Any]
    tensors: Mapping[str, torch.Tensor]
    audit: Mapping[str, Any]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _target_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = f"{_V28_PREFIX}adapters.{_TARGET_ADAPTER_INDEX}."
    return {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }


def _bank_state(
    tensors: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }


def _frozen_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in tensors.items() if name not in _TARGET_NAMES}


def _terminal_authorization(
    report_path: Path = DEFAULT_TERMINAL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _resolve(report_path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V39 requires a real V38 terminal seal: {path}")
    observed_sha = _sha256(path)
    if observed_sha != _V38_TERMINAL_SHA256:
        raise ValueError(
            "V39 live diagnostic requires exact V38 terminal revision 2: "
            f"observed={observed_sha} expected={_V38_TERMINAL_SHA256}"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("V38 terminal report must be a JSON object")
    authorization = dict(
        _mapping(
            report.get("conditional_successor_authorization"),
            "V38 conditional successor authorization",
        )
    )
    required_top = {
        "artifact": "v38_update8_terminal_gate",
        "passed": True,
        "conditional_v39_v28_layer14_gradient_cosine_screen_authorized": True,
        "v39_training_authorized": False,
        "only_exact_conditional_successor_authorized": (
            "v39_v28_layer14_gradient_cosine_screen"
        ),
        "validation_access_authorized": False,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
        "chat_promotion_authorized": False,
    }
    required_authorization = {
        "authorized": True,
        "successor": "v39_v28_layer14_gradient_cosine_screen",
        "scope": "no_step_no_write_existing_v28_layer14_query_gradient_measurement",
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_adapter_file_sha256": _SOURCE_FILE_SHA256["adapter.safetensors"],
        "source_full_tensor_state_sha256": _SOURCE_FULL_STATE_SHA256,
        "existing_lora_bank": _V28_BANK,
        "existing_adapter_index": _TARGET_ADAPTER_INDEX,
        "target_language_layer": 14,
        "target_module_path": _TARGET_MODULE,
        "target_parameter_names": list(_TARGET_NAMES),
        "target_parameter_shapes": [list(shape) for shape in _TARGET_SHAPES],
        "target_tensor_count": 2,
        "target_parameter_count": _TARGET_PARAMETER_COUNT,
        "target_rank": 4,
        "target_alpha": 8.0,
        "target_dropout": 0.0,
        "target_source_state_sha256": _TARGET_STATE_SHA256,
        "complete_existing_bank_state_sha256": _V28_STATE_SHA256,
        "frozen_excluding_target_state_sha256": _FROZEN_STATE_SHA256,
        "frozen_excluding_target_tensor_count": _FROZEN_TENSOR_COUNT,
        "gradient_computation_authorized": True,
        "backward_or_autograd_grad_for_measurement_authorized": True,
        "temporary_requires_grad_toggle_authorized": True,
        "gradient_accumulation_across_objectives_authorized": False,
        "gradients_must_be_cleared_between_objectives": True,
        "optimizer_construction_authorized": False,
        "optimizer_step_authorized": False,
        "parameter_update_authorized": False,
        "parameter_or_buffer_write_authorized": False,
        "training_authorized": False,
        "source_optimizer_access_authorized": False,
        "update8_optimizer_access_authorized": False,
        "validation_access_authorized": False,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
        "chat_promotion_authorized": False,
        "diagnostic_data_scope": "exact_training_scenes_only",
        "diagnostic_result_may_authorize_training": False,
        "diagnostic_result_may_promote_runtime": False,
        "new_lora_bank_authorized": False,
        "new_scene_encoder_module_authorized": False,
        "new_scene_tokens_authorized": False,
        "question_dependent_retrieval": False,
        "scene_prefixes_must_remain_question_independent": True,
        "all_occupied_blocks_must_be_processed": True,
        "both_existing_target_tensors_nonzero": True,
        "target_state_must_be_bit_exact_after_each_measurement": True,
        "all_parameters_and_buffers_must_be_bit_exact_after_each_measurement": True,
        "separate_terminal_seal_required_for_any_training": True,
    }
    mismatch = {
        key: {"observed": report.get(key), "expected": value}
        for key, value in required_top.items()
        if report.get(key) != value
    }
    mismatch.update(
        {
            f"authorization.{key}": {
                "observed": authorization.get(key),
                "expected": value,
            }
            for key, value in required_authorization.items()
            if authorization.get(key) != value
        }
    )
    required_measurements = set(authorization.get("required_measurements", ()))
    if required_measurements != {
        "book_support_gradient_norm",
        "picture_support_gradient_norm",
        "broad_retention_gradient_norm",
        "cross_prefix_maintenance_gradient_norm",
        "book_picture_gradient_cosine",
        "book_broad_gradient_cosine",
        "picture_broad_gradient_cosine",
        "book_cross_prefix_gradient_cosine",
        "picture_cross_prefix_gradient_cosine",
        "per_tensor_gradient_norms",
    }:
        mismatch["authorization.required_measurements"] = "changed"
    if mismatch:
        raise ValueError(f"V38 terminal does not authorize this exact V39 screen: {mismatch}")
    return report, authorization


def _authenticate_source(config: Mapping[str, Any]) -> V39Source:
    contract = v38_contract(config)
    report, authorization = _terminal_authorization()
    source = _resolve(SOURCE_CHECKPOINT)
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError(f"V39 source must be a real checkpoint directory: {source}")
    inventory = sorted(path.name for path in source.iterdir())
    if inventory != sorted(_SOURCE_FILE_SHA256):
        raise ValueError(f"V39 update-zero source inventory changed: {inventory}")
    observed_files: dict[str, str] = {}
    for name, expected in _SOURCE_FILE_SHA256.items():
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"V39 source member is missing or aliased: {path}")
        observed_files[name] = _sha256(path)
        if observed_files[name] != expected:
            raise ValueError(f"V39 source file changed: {path}")
    tensors = load_file(source / "adapter.safetensors", device="cpu")
    if len(tensors) != 179 or tensor_state_sha256(tensors) != _SOURCE_FULL_STATE_SHA256:
        raise ValueError("V39 source is not exact V38 update-zero hybrid state")
    target = _target_state(tensors)
    v28 = _bank_state(tensors, _V28_PREFIX)
    frozen = _frozen_state(tensors)
    if (
        tuple(target) != ("lora_a", "lora_b")
        or tuple(tuple(value.shape) for value in target.values()) != _TARGET_SHAPES
        or sum(value.numel() for value in target.values()) != _TARGET_PARAMETER_COUNT
        or tensor_state_sha256(target) != _TARGET_STATE_SHA256
        or tensor_state_sha256(v28) != _V28_STATE_SHA256
        or len(frozen) != _FROZEN_TENSOR_COUNT
        or tensor_state_sha256(frozen) != _FROZEN_STATE_SHA256
    ):
        raise ValueError("V39 authenticated target/frozen tensor partition changed")
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("V39 source contains NaN or infinity")
    metadata = json.loads(
        (source / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (source / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V39 source runtime metadata is not exact sanitized metadata")
    stage = _mapping(metadata.get("v38_query_recovery"), "V38 update-zero stage")
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "LoRA bank hashes")
    if (
        metadata.get("optimizer_step") != 0
        or metadata.get("config_hash") != "52df1554e3e5"
        or stage.get("optimizer_step") != 0
        or stage.get("update_zero_hybrid_tensor_state_sha256")
        != _SOURCE_FULL_STATE_SHA256
        or stage.get("source_optimizer_files_opened") is not False
        or stage.get("source_optimizer_states_loaded") is not False
        or bank_hashes.get(_V28_BANK) != _V28_STATE_SHA256
        or metadata.get("block_cross_residual_state_sha256") != _CORE_STATE_SHA256
    ):
        raise ValueError("V39 V38 update-zero metadata contract changed")
    if contract.hybrid_tensor_state_sha256 != _SOURCE_FULL_STATE_SHA256:
        raise ValueError("V39 resolved V38 contract changed")
    return V39Source(
        terminal=report,
        authorization=authorization,
        metadata=metadata,
        runtime_metadata=runtime,
        tensors=tensors,
        audit={
            "source_checkpoint": _relative(source),
            "source_file_sha256": observed_files,
            "source_full_tensor_state_sha256": _SOURCE_FULL_STATE_SHA256,
            "target_source_state_sha256": _TARGET_STATE_SHA256,
            "complete_v28_bank_state_sha256": _V28_STATE_SHA256,
            "frozen_excluding_target_state_sha256": _FROZEN_STATE_SHA256,
            "source_optimizer_file_opened": False,
            "source_optimizer_state_loaded": False,
            "update8_checkpoint_opened": False,
        },
    )


def v39_source_cache_evidence(
    config: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    *,
    scene_ids: Sequence[str],
    manifest_scene_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate cache evidence embedded in authenticated V38-u0 metadata only."""

    requested = tuple(sorted(set(scene_ids)))
    manifest = tuple(sorted(set(manifest_scene_ids)))
    split = v31_contract(config)
    expected_manifest = tuple(
        sorted((*split.train_scene_ids, *split.validation_scene_ids))
    )
    if requested != split.train_scene_ids or manifest != expected_manifest:
        raise ValueError("V39 cache request changed its exact train/manifest boundary")
    v38_stage = _mapping(
        source_metadata.get("v38_query_recovery"), "V38 update-zero stage"
    )
    v38_cache = dict(_mapping(v38_stage.get("scene_cache"), "V38 scene cache"))
    v35_stage = _mapping(
        source_metadata.get("v35_block_cross"), "V38 inherited V35 stage"
    )
    v35_cache = _mapping(v35_stage.get("scene_cache"), "inherited V35 scene cache")
    validate_v35_cache_audit(v38_cache, expected_scene_ids=requested)
    validate_v35_cache_audit(v35_cache, expected_scene_ids=manifest)
    v38_hashes = _mapping(
        v38_cache.get("source_prefix_sha256_by_scene"), "V38 source-prefix hashes"
    )
    v35_hashes = _mapping(
        v35_cache.get("source_prefix_sha256_by_scene"), "V35 source-prefix hashes"
    )
    if any(v38_hashes[scene_id] != v35_hashes[scene_id] for scene_id in requested):
        raise ValueError("V38 train-prefix evidence differs from inherited V35 evidence")
    v38_attestation = _mapping(
        v38_cache.get("post_v33_prefix_manifest_attestation"),
        "V38 prefix attestation",
    )
    v35_attestation = _mapping(
        v35_cache.get("post_v33_prefix_manifest_attestation"),
        "V35 prefix attestation",
    )
    expected_attestation = {
        "attesting_metadata_path": str(
            _resolve(
                "data_gemma4/checkpoints/"
                "gemma4_v34_diverse28_base_surface/update_032/metadata.json"
            )
        ),
        "attesting_metadata_sha256": (
            "14ba328ab9ac1010b75e40123643e3497c59b2bc1c59bfbe307d05a58cea7719"
        ),
        "attesting_optimizer_step": 0,
        "carrier_checkpoint_optimizer_step": 32,
    }
    expected_files = [
        str(
            (
                artifact_root(dict(config), "maps")
                / scene_id
                / "voxel_map.npz"
            ).resolve()
        )
        for scene_id in requested
    ]
    required = {
        "scene_scope": "training_only",
        "authenticated_manifest_scene_count": 22,
        "authenticated_manifest_train_subset_count": 16,
        "validation_scene_ids_loaded": [],
        "validation_environment_maps_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "loaded_environment_files": expected_files,
    }
    mismatch = {
        key: {"observed": v38_cache.get(key), "expected": value}
        for key, value in required.items()
        if v38_cache.get(key) != value
    }
    if dict(v38_attestation) != expected_attestation:
        mismatch["v38_post_v33_prefix_manifest_attestation"] = "changed"
    if dict(v35_attestation) != expected_attestation:
        mismatch["v35_post_v33_prefix_manifest_attestation"] = "changed"
    if mismatch:
        raise ValueError(f"V39 authenticated source-cache evidence changed: {mismatch}")
    return {
        **v38_cache,
        "source": "exact_authenticated_v38_update_zero_metadata",
        "cross_checked_against_inherited_v35_22_scene_cache": True,
        "external_terminal_report_opened": False,
        "v34_recursive_audit_called": False,
        "v33_recursive_audit_called": False,
        "optimizer_file_opened": False,
        "optimizer_state_loaded": False,
    }


def _scene_audit_scalar(audit: Mapping[str, torch.Tensor], field: str) -> int:
    value = audit.get(field)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise TypeError(f"V39 scene audit field {field!r} must be a scalar tensor")
    parsed = float(value.detach().cpu())
    if not parsed.is_integer():
        raise ValueError(f"V39 scene audit field {field!r} must be integral")
    return int(parsed)


def cache_v39_train_scenes(
    *,
    config: Mapping[str, Any],
    bundle: V30Bundle,
    source_metadata: Mapping[str, Any],
    scene_ids: Sequence[str],
    manifest_scene_ids: Sequence[str],
) -> tuple[dict[str, V35SceneCache], dict[str, Any]]:
    """Recompute the exact 16 caches against V38-u0 evidence, never Adam."""

    evidence = v39_source_cache_evidence(
        config,
        source_metadata,
        scene_ids=scene_ids,
        manifest_scene_ids=manifest_scene_ids,
    )
    requested = tuple(sorted(set(scene_ids)))
    expected_hashes = _mapping(
        evidence["source_prefix_sha256_by_scene"], "authenticated V38 prefix hashes"
    )
    expected_coverage = _mapping(
        evidence["coverage_by_scene"], "authenticated V38 map coverage"
    )
    model_dtype = next(bundle.language.model.parameters()).dtype
    tokens_per_block = int(bundle.scene_model.block_encoder.tokens_per_block)
    caches: dict[str, V35SceneCache] = {}
    loaded_files: list[str] = []
    observed_coverage: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    for scene_id in requested:
        map_path = (
            artifact_root(dict(config), "maps") / scene_id / "voxel_map.npz"
        ).resolve()
        data = load_map_tensors(
            map_path,
            config["scene"]["room_size_m"],
            bundle.language.device,
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
        with torch.inference_mode():
            output = map_forward(
                bundle.scene_model,
                data,
                bundle.global_scene_residual,
                bundle.signed_x_scene_residual,
                bundle.dense_aligner,
                None,
            )
            if output.aligned_sidecar_tokens is None:
                raise RuntimeError(f"V39 source lacks dense sidecar tokens: {scene_id}")
            source_tokens = bundle.dense_sidecar_adapter(
                output.scene_tokens, output.aligned_sidecar_tokens
            )
            observed_hash = prefix_sha256(
                bundle.composer.scene_prefix(source_tokens.to(model_dtype))
            )
            if observed_hash != expected_hashes[scene_id]:
                raise RuntimeError(
                    f"V39 recomputed a changed V38-u0 source prefix for {scene_id}"
                )
            processed = _scene_audit_scalar(output.audit, "processed_voxels")
            occupied_blocks = int(output.audit["block_indices"].shape[0])
            cache = V35SceneCache(
                scene_id=scene_id,
                source_scene_tokens=source_tokens.detach().float().cpu().contiguous(),
                block_tokens=output.block_tokens.detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous(),
                block_positions_normalized=output.audit[
                    "block_token_positions_normalized"
                ]
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous(),
                source_prefix_sha256=observed_hash,
                voxel_count=int(data.voxel_count),
                processed_voxels=processed,
                occupied_block_count=occupied_blocks,
                tokens_per_block=tokens_per_block,
            )
            validate_v35_scene_cache(cache)
            replayed = prefix_sha256(
                bundle.composer.scene_prefix(
                    cache.source_scene_tokens.to(bundle.language.device).to(model_dtype)
                )
            )
            if replayed != observed_hash:
                raise RuntimeError(f"V39 CPU cache changed prefix for {scene_id}")
            coverage = {
                "voxel_count": cache.voxel_count,
                "processed_voxels": cache.processed_voxels,
                "occupied_block_count": cache.occupied_block_count,
                "tokens_per_block": cache.tokens_per_block,
                "token_count": int(cache.block_tokens.shape[0]),
            }
            if coverage != dict(
                _mapping(expected_coverage[scene_id], f"coverage.{scene_id}")
            ):
                raise RuntimeError(f"V39 recomputed changed map coverage for {scene_id}")
            caches[scene_id] = cache
            observed_coverage[scene_id] = coverage
        loaded_files.append(str(map_path))
        del data, output
        if bundle.language.device.type == "mps":
            torch.mps.empty_cache()
    audit = {
        key: value
        for key, value in evidence.items()
        if key
        not in {
            "source",
            "cross_checked_against_inherited_v35_22_scene_cache",
            "external_terminal_report_opened",
            "v34_recursive_audit_called",
            "v33_recursive_audit_called",
            "optimizer_file_opened",
            "optimizer_state_loaded",
        }
    }
    audit.update(
        {
            "cache_build_seconds": time.perf_counter() - started,
            "loaded_environment_files": loaded_files,
            "coverage_by_scene": observed_coverage,
            "source_prefix_sha256_by_scene": {
                scene_id: caches[scene_id].source_prefix_sha256
                for scene_id in requested
            },
            "v39_source_cache_evidence": {
                "source": evidence["source"],
                "cross_checked_against_inherited_v35_22_scene_cache": True,
                "external_terminal_report_opened": False,
                "v34_recursive_audit_called": False,
                "v33_recursive_audit_called": False,
                "optimizer_file_opened": False,
                "optimizer_state_loaded": False,
            },
        }
    )
    validate_v35_cache_audit(audit, expected_scene_ids=requested)
    return caches, audit


def _screen_inventory(
    records: Sequence[QARecord], units: Sequence[CounterfactualPairUnit], *, seed: int
) -> tuple[list[CounterfactualPairUnit], list[QARecord], dict[str, Any]]:
    schedule, schedule_audit = build_v38_schedule(records, units, seed=seed)
    priority = [item.pair_unit for item in schedule[:8]]
    broad = [item.broad_record for item in schedule[:_BROAD_ROW_COUNT]]
    keys = [unit.question_key for unit in priority]
    families = [_pair_family(unit) for unit in priority]
    if keys != list(_PRIORITY_KEYS) or families != ["book_support", "picture_support"] * 4:
        raise ValueError("V39 priority gradient inventory changed")
    if len({(record.scene_id, record.question_id) for record in broad}) != 8:
        raise ValueError("V39 broad-retention rows are not eight deterministic unique rows")
    return priority, broad, {
        "priority_question_keys": keys,
        "priority_pair_ids": [unit.pair_id for unit in priority],
        "priority_families": families,
        "broad_rows": [
            {
                "scene_id": record.scene_id,
                "question_id": record.question_id,
                "answer_type": record.answer_type,
            }
            for record in broad
        ],
        "v38_pair_schedule_sha256": schedule_audit["pair_schedule_sha256"],
        "v38_schedule_sha256": schedule_audit["schedule_sha256"],
        "priority_unit_count": 8,
        "broad_row_count": 8,
        "broad_row_scope": "deterministic_first_eight_v38_schedule_rows",
        "broad_row_inventory_is_not_the_48_row_gate": True,
        "questions_or_answers_serialized": False,
    }


def _forbidden_roots(config: Mapping[str, Any]) -> list[Path]:
    loader = v38_loader_config(config)
    split = v31_contract(loader)
    qa_root = artifact_root(loader, "qa").resolve()
    maps_root = artifact_root(loader, "maps").resolve()
    roots = [
        artifact_root(loader, "oracle").resolve(),
        artifact_root(loader, "rendered").resolve(),
        artifact_root(loader, "features").resolve(),
        qa_root / "validation.jsonl",
        qa_root / "test.jsonl",
        _resolve(SOURCE_CHECKPOINT).parent / "update_008",
    ]
    allowed = set(split.train_scene_ids)
    if maps_root.is_dir():
        roots.extend(path for path in maps_root.iterdir() if path.name not in allowed)
    roots.extend(PROJECT_ROOT.rglob("optimizer.pt"))
    return [path.resolve() for path in roots]


def _audit_scope(audit: FileAccessAudit) -> dict[str, Any]:
    forbidden = audit.forbidden_accesses()
    loaded = audit.unique_paths
    environment = [
        path
        for path in loaded
        if any(part in {"data", "data_gemma4", "data_diverse28"} for part in Path(path).parts)
    ]
    return {
        "loaded_file_inventory": loaded,
        "loaded_environment_file_inventory": environment,
        "forbidden_file_accesses": forbidden,
        "forbidden_file_access_count": len(forbidden),
    }


def preflight_v39(
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Authenticate seal/source/train QA without loading Gemma or any scene map."""

    config_path = _resolve(config_path)
    audit = FileAccessAudit(forbidden_component_names={"oracle"}, block_forbidden=True)
    with audit:
        if _sha256(config_path) != _V38_CONFIG_SHA256:
            raise ValueError("V39 requires exact V38 configuration bytes")
        config = load_config(config_path)
        audit.forbidden_roots.extend(_forbidden_roots(config))
        source = _authenticate_source(config)
        loader = v38_loader_config(config)
        assert_deferred_final_scenes_absent(loader)
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
        _priority, _broad, inventory = _screen_inventory(
            records, units, seed=int(config["seed"])
        )
        split = v31_contract(loader)
        cache_evidence = v39_source_cache_evidence(
            loader,
            source.metadata,
            scene_ids=split.train_scene_ids,
            manifest_scene_ids=(*split.train_scene_ids, *split.validation_scene_ids),
        )
    scope = _audit_scope(audit)
    if scope["forbidden_file_accesses"]:
        raise RuntimeError("V39 preflight crossed a forbidden file boundary")
    if any("/maps/" in path for path in scope["loaded_file_inventory"]):
        raise RuntimeError("V39 preflight unexpectedly loaded a scene map")
    return {
        "schema_version": 1,
        "artifact": "v39_v28_layer14_gradient_cosine_screen_preflight",
        "passed": True,
        "config": _relative(config_path),
        "config_sha256": _V38_CONFIG_SHA256,
        "terminal": {
            "path": str(DEFAULT_TERMINAL),
            "sha256": _V38_TERMINAL_SHA256,
            "exact_revision_2_authorization_verified": True,
        },
        "source": dict(source.audit),
        "target_surface": {
            "existing_bank": _V28_BANK,
            "existing_adapter_index": _TARGET_ADAPTER_INDEX,
            "module_path": _TARGET_MODULE,
            "parameter_names": list(_TARGET_NAMES),
            "parameter_shapes": [list(shape) for shape in _TARGET_SHAPES],
            "tensor_count": 2,
            "parameter_count": _TARGET_PARAMETER_COUNT,
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
        },
        "objective": objective_contract(),
        "pass_contract": dict(_PASS_CONTRACT),
        "expected_gemma_architecture": dict(_EXPECTED_GEMMA_ARCHITECTURE),
        "screen_inventory": inventory,
        "source_cache_evidence": {
            "scene_count": cache_evidence["scene_count"],
            "scene_ids": cache_evidence["scene_ids"],
            "source": cache_evidence["source"],
            "cross_checked_against_inherited_v35_22_scene_cache": True,
            "external_terminal_report_opened": False,
            "v34_recursive_audit_called": False,
            "v33_recursive_audit_called": False,
            "optimizer_file_opened": False,
            "optimizer_state_loaded": False,
        },
        "qa_audit": qa_audit,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "optimizer_constructed": False,
        "optimizer_file_opened": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        **scope,
    }


def objective_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "priority_full": (
            "0.5*correct_answer_nll + 8.0*side_hinge + "
            "4.0*cross_prefix_hinge"
        ),
        "answer_nll_weight": _PAIR_CORRECT_NLL_WEIGHT,
        "side_hinge_weight": _SIDE_HINGE_WEIGHT,
        "side_hinge_margin": _SIDE_HINGE_MARGIN,
        "cross_prefix_hinge_weight": _CROSS_PREFIX_HINGE_WEIGHT,
        "cross_prefix_margin": _CROSS_PREFIX_MARGIN,
        "broad_retention_nll_weight": _BROAD_NLL_WEIGHT,
        "broad_retention_scope": "deterministic_first_eight_v38_schedule_rows",
        "broad_retention_is_not_the_48_row_gate": True,
        "proposed_training_gradient_formula": (
            "priority_aggregate + broad_retention_aggregate"
        ),
        "scene_discriminative_components_reported_separately": True,
        "answer_nll_component_reported_separately": True,
        "gradient_accumulation_across_objectives": False,
    }


def _v28_bank(bundle: V30Bundle) -> LoRAInstallation:
    collection = bundle.lora_installation
    if collection is None:
        raise RuntimeError("V39 requires named installed LoRA banks")
    bank = collection.bank(_V28_BANK).installation
    state = bank.state_module.state_dict()
    if (
        bank.target_names
        != (
            "model.language_model.layers.13.self_attn.q_proj",
            _TARGET_MODULE,
        )
        or bank.parameter_count != 36_864
        or tuple(tuple(value.shape) for value in state.values())
        != ((4, 1536), (2048, 4), *_TARGET_SHAPES)
        or bank.settings.rank != 4
        or bank.settings.alpha != 8.0
        or bank.settings.dropout != 0.0
    ):
        raise RuntimeError("V39 existing V28 bank architecture changed")
    return bank


def freeze_for_v39(bundle: V30Bundle) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    bundle.language.model.requires_grad_(False).eval()
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False).eval()
    bank = _v28_bank(bundle)
    bank.eval()
    target = bank.adapters[_TARGET_ADAPTER_INDEX]
    target.lora_a.requires_grad_(True)
    target.lora_b.requires_grad_(True)
    target.lora_a.grad = None
    target.lora_b.grad = None
    return target.lora_a, target.lora_b


def assert_v39_surface(
    bundle: V30Bundle,
    target: Sequence[torch.nn.Parameter],
) -> dict[str, Any]:
    expected_ids = {id(parameter) for parameter in target}
    checkpoint_ids = {
        id(parameter)
        for module in bundle.checkpoint_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    model_ids = {
        id(parameter)
        for parameter in bundle.language.model.parameters()
        if parameter.requires_grad
    }
    shapes = tuple(tuple(parameter.shape) for parameter in target)
    if (
        len(target) != 2
        or sum(parameter.numel() for parameter in target) != _TARGET_PARAMETER_COUNT
        or shapes != _TARGET_SHAPES
        or checkpoint_ids != expected_ids
        or model_ids != expected_ids
    ):
        raise RuntimeError("V39 active gradient surface differs from its exact lock")
    return {
        "target_parameter_names": list(_TARGET_NAMES),
        "target_parameter_shapes": [list(shape) for shape in shapes],
        "trainable_tensor_count": 2,
        "trainable_parameter_count": _TARGET_PARAMETER_COUNT,
        "all_other_checkpoint_parameters_frozen": True,
        "all_other_gemma_parameters_frozen": True,
        "optimizer_constructed": False,
    }


def _checkpoint_tensors(bundle: V30Bundle) -> dict[str, torch.Tensor]:
    return {
        f"{module_name}.{name}": value
        for module_name, module in bundle.checkpoint_modules.items()
        for name, value in module.state_dict().items()
    }


def _state_hashes(bundle: V30Bundle) -> dict[str, str]:
    tensors = _checkpoint_tensors(bundle)
    return {
        "full": tensor_state_sha256(tensors),
        "target": tensor_state_sha256(_target_state(tensors)),
        "v28_bank": tensor_state_sha256(_bank_state(tensors, _V28_PREFIX)),
        "frozen_excluding_target": tensor_state_sha256(_frozen_state(tensors)),
    }


def _model_version_surface(bundle: V30Bundle) -> dict[str, Any]:
    rows = [
        (
            kind,
            name,
            tuple(value.shape),
            str(value.dtype),
            int(value._version),
            bool(getattr(value, "requires_grad", False)),
        )
        for kind, values in (
            ("parameter", bundle.language.model.named_parameters()),
            ("buffer", bundle.language.model.named_buffers()),
        )
        for name, value in values
    ]
    payload = json.dumps(rows, sort_keys=False, separators=(",", ":")).encode()
    return {
        "entry_count": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _assert_exact_state(hashes: Mapping[str, str]) -> None:
    expected = {
        "full": _SOURCE_FULL_STATE_SHA256,
        "target": _TARGET_STATE_SHA256,
        "v28_bank": _V28_STATE_SHA256,
        "frozen_excluding_target": _FROZEN_STATE_SHA256,
    }
    if dict(hashes) != expected:
        raise RuntimeError(f"V39 model/checkpoint state changed: {hashes}")


def _gemma_architecture(bundle: V30Bundle) -> dict[str, Any]:
    root = bundle.language.model.config
    text = getattr(root, "text_config", root)
    layer_types = tuple(str(value) for value in text.layer_types)
    observed = {
        "language_layer_count": int(text.num_hidden_layers),
        "num_kv_shared_layers": int(text.num_kv_shared_layers),
        "first_shared_kv_layer": int(text.num_hidden_layers)
        - int(text.num_kv_shared_layers),
        "layer_13_attention_type": layer_types[13],
        "layer_14_attention_type": layer_types[14],
        "layer_13_role": "last_nonshared_sliding_kv_producer",
        "layer_14_role": "last_nonshared_full_kv_producer",
        "layers_15_through_34_reuse_shared_kv_states": True,
    }
    if observed != dict(_EXPECTED_GEMMA_ARCHITECTURE):
        raise RuntimeError(f"Loaded Gemma 4 architecture changed: {observed}")
    return observed


def load_v39_bundle(
    config: dict[str, Any], source: V39Source
) -> tuple[V30Bundle, BlockCrossResidual, dict[str, Any]]:
    """Load only authenticated V38 update zero; never inspect source Adam."""

    loader = v38_loader_config(config)
    approved = require_approved_v29_source(loader)
    bundle = load_v30_bundle(loader, approved)
    block_core = construct_v36_source_core(loader, device=bundle.language.device)
    bundle.checkpoint_modules["block_cross_residual"] = block_core
    loaded = load_adapter_checkpoint(
        _resolve(SOURCE_CHECKPOINT),
        bundle.checkpoint_modules,
        device="cpu",
        metadata_filename=TRAINING_METADATA_FILENAME,
    )
    if loaded != source.metadata:
        raise RuntimeError("V39 source metadata changed during exact model load")
    transition = retag_bundle_for_v38(bundle, config)
    validate_lora_banks_checkpoint_state(source.runtime_metadata, bundle.lora_installation)
    validate_block_cross_residual_state(
        block_core,
        expected_parameter_count=983_040,
        expected_state_sha256=_CORE_STATE_SHA256,
        context="V39 frozen V38 update-zero block core",
    )
    if module_collection_state_sha256(bundle.checkpoint_modules) != _SOURCE_FULL_STATE_SHA256:
        raise RuntimeError("V39 loaded model is not exact V38 update-zero hybrid")
    if _v28_bank(bundle).state_sha256() != _V28_STATE_SHA256:
        raise RuntimeError("V39 loaded a changed V28 query bank")
    return bundle, block_core, transition


def _gradient_stats(
    gradients: Sequence[torch.Tensor], *, names: Sequence[str] = _TARGET_NAMES
) -> dict[str, Any]:
    if len(gradients) != 2 or len(names) != 2:
        raise ValueError("V39 exact gradient surface requires two tensors")
    cpu = tuple(value.detach().float().cpu().contiguous() for value in gradients)
    finite = [bool(torch.isfinite(value).all()) for value in cpu]
    norms = [float(value.norm()) for value in cpu]
    state = {name: value for name, value in zip(names, cpu, strict=True)}
    vector = torch.cat([value.reshape(-1) for value in cpu])
    return {
        "gradient_state_sha256": tensor_state_sha256(state),
        "total_l2": float(vector.norm()),
        "all_finite": all(finite),
        "all_target_tensors_nonzero": all(value > 0.0 for value in norms),
        "per_tensor": {
            name: {
                "shape": list(value.shape),
                "l2": norm,
                "finite": is_finite,
                "nonzero": norm > 0.0,
                "mean": float(value.mean()),
                "std": float(value.std()),
                "minimum": float(value.min()),
                "maximum": float(value.max()),
            }
            for name, value, norm, is_finite in zip(
                names, cpu, norms, finite, strict=True
            )
        },
    }


def _vector(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [value.detach().float().cpu().reshape(-1) for value in gradients]
    ).contiguous()


def _split_vector(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if vector.numel() != _TARGET_PARAMETER_COUNT:
        raise ValueError("V39 flattened gradient size changed")
    return vector[: 4 * 1536].reshape(4, 1536), vector[4 * 1536 :].reshape(4096, 4)


def _vector_stats(vector: torch.Tensor) -> dict[str, Any]:
    return _gradient_stats(_split_vector(vector))


def gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> float | None:
    if first.shape != second.shape:
        raise ValueError("V39 gradient cosine vectors have different shapes")
    denominator = float(first.norm()) * float(second.norm())
    if denominator == 0.0:
        return None
    value = float(torch.dot(first.double(), second.double()) / denominator)
    return max(-1.0, min(1.0, value))


def gradient_dot(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        raise ValueError("V39 gradient dot vectors have different shapes")
    return float(torch.dot(first.double(), second.double()))


def cosine_conflict_matrix(vectors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    names = list(vectors)
    cosine_rows: list[list[float | None]] = []
    conflict_rows: list[list[bool | None]] = []
    conflicts: list[dict[str, Any]] = []
    for row_name in names:
        cosine_row: list[float | None] = []
        conflict_row: list[bool | None] = []
        for column_name in names:
            cosine = gradient_cosine(vectors[row_name], vectors[column_name])
            conflict = None if cosine is None else cosine < 0.0
            cosine_row.append(cosine)
            conflict_row.append(conflict)
            if names.index(column_name) > names.index(row_name) and conflict is True:
                conflicts.append(
                    {"first": row_name, "second": column_name, "cosine": cosine}
                )
        cosine_rows.append(cosine_row)
        conflict_rows.append(conflict_row)
    return {
        "names": names,
        "cosine": cosine_rows,
        "conflict": conflict_rows,
        "negative_cosine_pairs": conflicts,
        "negative_cosine_pair_count": len(conflicts),
    }


def _mean(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not vectors:
        raise ValueError("V39 cannot aggregate an empty gradient inventory")
    return torch.stack(list(vectors)).mean(dim=0)


def _clear_and_assert_no_accumulated_gradients(
    bundle: V30Bundle, target: Sequence[torch.nn.Parameter]
) -> None:
    for parameter in target:
        parameter.grad = None
    if any(parameter.grad is not None for parameter in bundle.language.model.parameters()):
        raise RuntimeError("V39 found an accumulated model .grad tensor")


def _measure_objective(
    *,
    name: str,
    objective: torch.Tensor,
    bundle: V30Bundle,
    target: Sequence[torch.nn.Parameter],
    retain_graph: bool,
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    _clear_and_assert_no_accumulated_gradients(bundle, target)
    before = _state_hashes(bundle)
    _assert_exact_state(before)
    gradients_raw = torch.autograd.grad(
        objective,
        tuple(target),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    gradients = (gradients_raw[0], gradients_raw[1])
    stats = _gradient_stats(gradients)
    if not stats["all_finite"]:
        raise RuntimeError(f"V39 {name} produced a nonfinite target gradient")
    after = _state_hashes(bundle)
    _assert_exact_state(after)
    if before != after:
        raise RuntimeError(f"V39 {name} mutated model/checkpoint state")
    _clear_and_assert_no_accumulated_gradients(bundle, target)
    return gradients, {
        "objective": name,
        "target_and_frozen_hashes_before": before,
        "target_and_frozen_hashes_after": after,
        "state_bit_exact": True,
        "no_accumulated_gradients": True,
        "gradient": stats,
    }


def _priority_gradients(
    *,
    units: Sequence[CounterfactualPairUnit],
    scene_tokens: Mapping[str, torch.Tensor],
    bundle: V30Bundle,
    target: Sequence[torch.nn.Parameter],
) -> tuple[
    list[dict[str, Any]],
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
    list[dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []
    full_vectors: dict[str, torch.Tensor] = {}
    component_vectors: dict[str, dict[str, torch.Tensor]] = {
        "answer_nll": {},
        "side_scene": {},
        "cross_prefix": {},
    }
    attestations: list[dict[str, Any]] = []
    for unit in units:
        tokens = {scene_id: scene_tokens[scene_id] for scene_id in unit.scene_ids}
        correct, side, cross, diagnostics = paired_cross_prefix_objective(
            unit=unit,
            scene_tokens=tokens,
            bundle=bundle,
            side_margin=_SIDE_HINGE_MARGIN,
            cross_prefix_margin=_CROSS_PREFIX_MARGIN,
        )
        answer_objective = _PAIR_CORRECT_NLL_WEIGHT * correct
        side_objective = _SIDE_HINGE_WEIGHT * side
        cross_objective = _CROSS_PREFIX_HINGE_WEIGHT * cross
        answer_grad, answer_audit = _measure_objective(
            name=f"priority:{unit.question_key}:answer_nll",
            objective=answer_objective,
            bundle=bundle,
            target=target,
            retain_graph=True,
        )
        side_grad, side_audit = _measure_objective(
            name=f"priority:{unit.question_key}:side_hinge",
            objective=side_objective,
            bundle=bundle,
            target=target,
            retain_graph=True,
        )
        cross_grad, cross_audit = _measure_objective(
            name=f"priority:{unit.question_key}:cross_prefix_hinge",
            objective=cross_objective,
            bundle=bundle,
            target=target,
            retain_graph=False,
        )
        answer_vector = _vector(answer_grad)
        side_vector = _vector(side_grad)
        cross_vector = _vector(cross_grad)
        scene_vector = side_vector + cross_vector
        full_vector = answer_vector + scene_vector
        gradient_id = f"priority:{unit.question_key}"
        full_vectors[gradient_id] = full_vector
        component_vectors["answer_nll"][gradient_id] = answer_vector
        component_vectors["side_scene"][gradient_id] = side_vector
        component_vectors["cross_prefix"][gradient_id] = cross_vector
        rows.append(
            {
                "gradient_id": gradient_id,
                "pair_id": unit.pair_id,
                "question_key": unit.question_key,
                "family": _pair_family(unit),
                "scene_ids": list(unit.scene_ids),
                "losses": {
                    "correct_answer_nll": float(correct.detach().float().cpu()),
                    "side_hinge": float(side.detach().float().cpu()),
                    "cross_prefix_hinge": float(cross.detach().float().cpu()),
                    "weighted_full": float(
                        (answer_objective + side_objective + cross_objective)
                        .detach()
                        .float()
                        .cpu()
                    ),
                },
                "side_margins": [
                    float(value)
                    for value in diagnostics["side_margins"].detach().float().cpu()
                ],
                "cross_prefix_margins": [
                    float(value)
                    for value in diagnostics["cross_prefix_margins"]
                    .detach()
                    .float()
                    .cpu()
                ],
                "gradients": {
                    "answer_nll": _vector_stats(answer_vector),
                    "side_scene_discriminative": _vector_stats(side_vector),
                    "cross_prefix_maintenance": _vector_stats(cross_vector),
                    "scene_discriminative_total": _vector_stats(scene_vector),
                    "full_priority": _vector_stats(full_vector),
                },
                "answer_vs_scene_discriminative_cosine": gradient_cosine(
                    answer_vector, scene_vector
                ),
            }
        )
        attestations.extend((answer_audit, side_audit, cross_audit))
        del correct, side, cross, diagnostics
        if bundle.language.device.type == "mps":
            torch.mps.empty_cache()
    return rows, full_vectors, component_vectors, attestations


def _broad_gradients(
    *,
    records: Sequence[QARecord],
    scene_tokens: Mapping[str, torch.Tensor],
    bundle: V30Bundle,
    target: Sequence[torch.nn.Parameter],
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    attestations: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        loss = _BROAD_NLL_WEIGHT * broad_answer_nll(
            scene_tokens=scene_tokens[record.scene_id], record=record, bundle=bundle
        )
        gradients, attestation = _measure_objective(
            name=f"broad:{index}:{record.scene_id}:{record.question_id}",
            objective=loss,
            bundle=bundle,
            target=target,
            retain_graph=False,
        )
        vector = _vector(gradients)
        vectors.append(vector)
        rows.append(
            {
                "gradient_id": f"broad:{index}",
                "scene_id": record.scene_id,
                "question_id": record.question_id,
                "answer_type": record.answer_type,
                "answer_nll": float(loss.detach().float().cpu()),
                "gradient": _vector_stats(vector),
            }
        )
        attestations.append(attestation)
        del loss
        if bundle.language.device.type == "mps":
            torch.mps.empty_cache()
    return rows, vectors, attestations


def _aggregate_gradients(
    priority_rows: Sequence[Mapping[str, Any]],
    priority_vectors: Mapping[str, torch.Tensor],
    broad_vectors: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    by_family: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in priority_rows:
        by_family[str(row["family"])].append(priority_vectors[str(row["gradient_id"])])
    result = {
        "priority_aggregate": _mean(list(priority_vectors.values())),
        "book_support_aggregate": _mean(by_family["book_support"]),
        "picture_support_aggregate": _mean(by_family["picture_support"]),
        "broad_retention_aggregate": _mean(broad_vectors),
    }
    return result


def evaluate_pass_contract(
    *,
    priority_rows: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, torch.Tensor],
    state_exact: bool,
    model_versions_exact: bool,
    surface_exact: bool,
    surface_restored: bool,
    frozen_has_no_gradients: bool,
) -> dict[str, Any]:
    directional: dict[str, dict[str, Any]] = {}
    for first, second in _PASS_CONTRACT["directional_pairs"]:
        dot = gradient_dot(aggregates[first], aggregates[second])
        cosine = gradient_cosine(aggregates[first], aggregates[second])
        directional[f"{first}__{second}"] = {
            "dot_product": dot,
            "cosine": cosine,
            "positive_dot": dot > 0.0,
            "cosine_at_least_zero": cosine is not None and cosine >= 0.0,
            "passed": dot > 0.0 and cosine is not None and cosine >= 0.0,
        }
    priority_finite = all(
        row["gradients"]["full_priority"]["all_finite"] is True
        for row in priority_rows
    )
    priority_nonzero = all(
        row["gradients"]["full_priority"]["all_target_tensors_nonzero"] is True
        for row in priority_rows
    )
    broad_stats = _vector_stats(aggregates["broad_retention_aggregate"])
    scene_stats = _vector_stats(aggregates["scene_discriminative_aggregate"])
    checks = {
        "exact_surface": surface_exact,
        "temporary_requires_grad_surface_restored_to_frozen": surface_restored,
        "all_priority_full_gradients_finite": priority_finite,
        "all_priority_full_gradient_target_tensors_nonzero": priority_nonzero,
        "broad_aggregate_gradient_finite": broad_stats["all_finite"],
        "broad_aggregate_gradient_target_tensors_nonzero": broad_stats[
            "all_target_tensors_nonzero"
        ],
        "scene_discriminative_aggregate_gradient_finite": scene_stats["all_finite"],
        "scene_discriminative_aggregate_gradient_target_tensors_nonzero": scene_stats[
            "all_target_tensors_nonzero"
        ],
        "all_checkpoint_target_and_frozen_state_bit_exact": state_exact,
        "model_version_counters_unchanged": model_versions_exact,
        "frozen_complement_has_no_gradients": frozen_has_no_gradients,
        "no_optimizer_constructed_or_opened": True,
        "no_optimizer_step_or_checkpoint_write": True,
        "all_predeclared_directional_compatibility_checks_passed": all(
            row["passed"] for row in directional.values()
        ),
    }
    return {
        "contract": dict(_PASS_CONTRACT),
        "checks": checks,
        "directional_compatibility": directional,
        "passed": all(checks.values()),
        "passing_this_screen_authorizes_training": False,
        "passing_this_screen_authorizes_runtime_promotion": False,
    }


def run_v39(
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Run the authorized screen.  This function contains no optimizer API."""

    config_path = _resolve(config_path)
    audit = FileAccessAudit(forbidden_component_names={"oracle"}, block_forbidden=True)
    with audit:
        if _sha256(config_path) != _V38_CONFIG_SHA256:
            raise ValueError("V39 requires exact V38 configuration bytes")
        config = load_config(config_path)
        audit.forbidden_roots.extend(_forbidden_roots(config))
        source = _authenticate_source(config)
        loader = v38_loader_config(config)
        assert_deferred_final_scenes_absent(loader)
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
        priority, broad, inventory = _screen_inventory(
            records, units, seed=int(config["seed"])
        )
        bundle, block_core, transition = load_v39_bundle(config, source)
        architecture = _gemma_architecture(bundle)
        target = freeze_for_v39(bundle)
        surface = assert_v39_surface(bundle, target)
        hashes_before = _state_hashes(bundle)
        _assert_exact_state(hashes_before)
        versions_before = _model_version_surface(bundle)

        split = v31_contract(loader)
        all_development_scene_ids = (*split.train_scene_ids, *split.validation_scene_ids)
        caches, cache_audit = cache_v39_train_scenes(
            config=loader,
            bundle=bundle,
            source_metadata=source.metadata,
            scene_ids=split.train_scene_ids,
            manifest_scene_ids=all_development_scene_ids,
        )
        validate_v37_training_cache_boundary(
            cache_audit=cache_audit,
            caches=caches,
            config=loader,
            train_scene_ids=split.train_scene_ids,
            validation_scene_ids=split.validation_scene_ids,
        )
        model_dtype = next(bundle.language.model.parameters()).dtype
        scene_tokens: dict[str, torch.Tensor] = {}
        prefix_before: dict[str, str] = {}
        with torch.inference_mode():
            for scene_id in split.train_scene_ids:
                tokens = current_scene_tokens(
                    caches[scene_id], block_core, device=bundle.language.device
                ).detach()
                scene_tokens[scene_id] = tokens
                prefix_before[scene_id] = prefix_sha256(
                    bundle.composer.scene_prefix(tokens.to(model_dtype))
                )

        (
            priority_rows,
            priority_vectors,
            component_vectors,
            priority_attestations,
        ) = _priority_gradients(
            units=priority,
            scene_tokens=scene_tokens,
            bundle=bundle,
            target=target,
        )
        broad_rows, broad_vectors, broad_attestations = _broad_gradients(
            records=broad,
            scene_tokens=scene_tokens,
            bundle=bundle,
            target=target,
        )
        aggregates = _aggregate_gradients(priority_rows, priority_vectors, broad_vectors)

        answer_vectors = list(component_vectors["answer_nll"].values())
        side_vectors = list(component_vectors["side_scene"].values())
        cross_vectors = list(component_vectors["cross_prefix"].values())
        scene_vectors_by_family: dict[str, list[torch.Tensor]] = defaultdict(list)
        for row in priority_rows:
            gradient_id = str(row["gradient_id"])
            scene_vectors_by_family[str(row["family"])].append(
                component_vectors["side_scene"][gradient_id]
                + component_vectors["cross_prefix"][gradient_id]
            )
        aggregates.update(
            {
                "priority_answer_nll_aggregate": _mean(answer_vectors),
                "priority_side_scene_aggregate": _mean(side_vectors),
                "cross_prefix_maintenance_aggregate": _mean(cross_vectors),
                "scene_discriminative_aggregate": _mean(side_vectors)
                + _mean(cross_vectors),
                "book_scene_discriminative_aggregate": _mean(
                    scene_vectors_by_family["book_support"]
                ),
                "picture_scene_discriminative_aggregate": _mean(
                    scene_vectors_by_family["picture_support"]
                ),
            }
        )
        aggregates["proposed_training_aggregate"] = (
            aggregates["priority_aggregate"]
            + aggregates["broad_retention_aggregate"]
        )
        matrix_vectors = {
            **priority_vectors,
            **aggregates,
        }
        matrix = cosine_conflict_matrix(matrix_vectors)
        tensor_matrices = {
            _TARGET_NAMES[index]: cosine_conflict_matrix(
                {name: _split_vector(vector)[index].reshape(-1) for name, vector in matrix_vectors.items()}
            )
            for index in range(2)
        }

        hashes_after = _state_hashes(bundle)
        _assert_exact_state(hashes_after)
        versions_after = _model_version_surface(bundle)
        _clear_and_assert_no_accumulated_gradients(bundle, target)
        frozen_has_no_gradients = all(
            parameter.grad is None
            for parameter in bundle.language.model.parameters()
            if id(parameter) not in {id(value) for value in target}
        )
        prefix_after = {
            scene_id: prefix_sha256(bundle.composer.scene_prefix(tokens.to(model_dtype)))
            for scene_id, tokens in scene_tokens.items()
        }
        for parameter in target:
            parameter.requires_grad_(False)
            parameter.grad = None
        surface_restored = not any(
            parameter.requires_grad for parameter in bundle.language.model.parameters()
        ) and not any(
            parameter.grad is not None for parameter in bundle.language.model.parameters()
        )
        result_contract = evaluate_pass_contract(
            priority_rows=priority_rows,
            aggregates=aggregates,
            state_exact=hashes_before == hashes_after,
            model_versions_exact=versions_before == versions_after,
            surface_exact=(
                surface["trainable_tensor_count"] == 2
                and surface["trainable_parameter_count"] == _TARGET_PARAMETER_COUNT
            ),
            surface_restored=surface_restored,
            frozen_has_no_gradients=frozen_has_no_gradients,
        )
        safety_attestations_passed = bool(
            hashes_before == hashes_after
            and versions_before == versions_after
            and frozen_has_no_gradients
            and surface_restored
            and prefix_before == prefix_after
        )
        if not safety_attestations_passed:
            raise RuntimeError(
                "V39 refused to emit a report because a no-write/frozen-state "
                "attestation failed"
            )
    scope = _audit_scope(audit)
    if scope["forbidden_file_accesses"]:
        raise RuntimeError("V39 live screen crossed a forbidden file boundary")
    map_files = cache_audit["loaded_environment_files"]
    if len(map_files) != 16 or any(
        Path(path).parent.name not in set(split.train_scene_ids) for path in map_files
    ):
        raise RuntimeError("V39 did not load exactly the 16 authenticated train maps")
    loaded_inventory = scope["loaded_file_inventory"]
    observed_map_reads = sorted(
        path for path in loaded_inventory if "/maps/" in path and path.endswith(".npz")
    )
    observed_qa_jsonl_reads = sorted(
        path for path in loaded_inventory if "/qa/" in path and path.endswith(".jsonl")
    )
    expected_qa_jsonl_reads = [
        str(path)
        for path in qa_audit["loaded_files"]
        if str(path).endswith("train.jsonl")
    ]
    required_source_reads = [
        str((_resolve(SOURCE_CHECKPOINT) / name).resolve())
        for name in _SOURCE_FILE_SHA256
    ]
    if (
        observed_map_reads != sorted(map_files)
        or observed_qa_jsonl_reads != sorted(expected_qa_jsonl_reads)
        or not set(required_source_reads).issubset(loaded_inventory)
        or any(path.endswith("optimizer.pt") for path in loaded_inventory)
        or any(
            Path(path).name
            in {"v34_update32_terminal_gate.json", "v33_update64_terminal_gate.json"}
            for path in loaded_inventory
        )
    ):
        raise RuntimeError("V39 loaded-file boundary differs from its exact declaration")
    declared_data_reads = {
        "v38_update_zero_source_files": required_source_reads,
        "train_qa_jsonl_files": expected_qa_jsonl_reads,
        "train_map_files": sorted(map_files),
        "observed_map_reads_exact": True,
        "observed_train_qa_jsonl_reads_exact": True,
        "required_v38_update_zero_source_reads_present": True,
        "optimizer_file_reads": [],
        "historical_terminal_report_reads": [],
        "validation_or_final_environment_reads": [],
    }
    return {
        "schema_version": 1,
        "artifact": "v39_v28_layer14_query_gradient_cosine_screen",
        "diagnostic_completed": True,
        "passed": result_contract["passed"],
        "terminal": {
            "path": str(DEFAULT_TERMINAL),
            "sha256": _V38_TERMINAL_SHA256,
            "exact_revision_2_authorization_verified": True,
        },
        "source": dict(source.audit),
        "loaded_gemma_architecture": architecture,
        "causal_surface_rationale": (
            "layer 14 is the last nonshared full-attention K/V producer; its existing "
            "query adapter reads the retained layer-14 K-only scene ingress before "
            "layers 15-34 reuse shared K/V states"
        ),
        "target_surface": surface,
        "loader_transition": transition,
        "objective": objective_contract(),
        "screen_inventory": inventory,
        "priority_unit_gradients": priority_rows,
        "broad_row_gradients": broad_rows,
        "aggregate_gradients": {
            name: _vector_stats(vector) for name, vector in aggregates.items()
        },
        "required_terminal_measurements": {
            "book_support_gradient_norm": float(
                aggregates["book_support_aggregate"].norm()
            ),
            "picture_support_gradient_norm": float(
                aggregates["picture_support_aggregate"].norm()
            ),
            "broad_retention_gradient_norm": float(
                aggregates["broad_retention_aggregate"].norm()
            ),
            "cross_prefix_maintenance_gradient_norm": float(
                aggregates["cross_prefix_maintenance_aggregate"].norm()
            ),
            "book_picture_gradient_cosine": gradient_cosine(
                aggregates["book_support_aggregate"],
                aggregates["picture_support_aggregate"],
            ),
            "book_broad_gradient_cosine": gradient_cosine(
                aggregates["book_support_aggregate"],
                aggregates["broad_retention_aggregate"],
            ),
            "picture_broad_gradient_cosine": gradient_cosine(
                aggregates["picture_support_aggregate"],
                aggregates["broad_retention_aggregate"],
            ),
            "book_cross_prefix_gradient_cosine": gradient_cosine(
                aggregates["book_support_aggregate"],
                aggregates["cross_prefix_maintenance_aggregate"],
            ),
            "picture_cross_prefix_gradient_cosine": gradient_cosine(
                aggregates["picture_support_aggregate"],
                aggregates["cross_prefix_maintenance_aggregate"],
            ),
            "per_tensor_gradient_norms": {
                name: stats["per_tensor"]
                for name, stats in (
                    (gradient_name, _vector_stats(vector))
                    for gradient_name, vector in aggregates.items()
                )
            },
        },
        "additional_causal_measurements": {
            "book_scene_discriminative_gradient_norm": float(
                aggregates["book_scene_discriminative_aggregate"].norm()
            ),
            "picture_scene_discriminative_gradient_norm": float(
                aggregates["picture_scene_discriminative_aggregate"].norm()
            ),
            "proposed_training_book_scene_cosine": gradient_cosine(
                aggregates["proposed_training_aggregate"],
                aggregates["book_scene_discriminative_aggregate"],
            ),
            "proposed_training_picture_scene_cosine": gradient_cosine(
                aggregates["proposed_training_aggregate"],
                aggregates["picture_scene_discriminative_aggregate"],
            ),
            "proposed_training_cross_prefix_cosine": gradient_cosine(
                aggregates["proposed_training_aggregate"],
                aggregates["cross_prefix_maintenance_aggregate"],
            ),
            "broad_gradient_scope": "deterministic_first_eight_v38_schedule_rows",
            "broad_gradient_is_not_the_48_row_gate": True,
        },
        "cosine_conflict_matrix": matrix,
        "per_tensor_cosine_conflict_matrices": tensor_matrices,
        "pass_contract_evaluation": result_contract,
        "safety_attestations_passed_before_report_write": safety_attestations_passed,
        "mutation_audit": {
            "checkpoint_state_before": hashes_before,
            "checkpoint_state_after": hashes_after,
            "checkpoint_state_bit_exact": hashes_before == hashes_after,
            "model_version_surface_before": versions_before,
            "model_version_surface_after": versions_after,
            "model_version_counters_unchanged": versions_before == versions_after,
            "objective_measurements": [
                *priority_attestations,
                *broad_attestations,
            ],
            "autograd_grad_used": True,
            "backward_called": False,
            "gradients_accumulated_in_parameter_grad": False,
            "gradients_cleared_between_objectives": True,
            "temporary_requires_grad_surface_restored_to_frozen": surface_restored,
            "optimizer_constructed": False,
            "optimizer_state_opened": False,
            "optimizer_step_called": False,
            "checkpoint_written": False,
            "parameter_or_buffer_write_performed": False,
        },
        "scene_input_audit": {
            "train_scene_count": len(caches),
            "train_scene_ids": list(split.train_scene_ids),
            "map_files": map_files,
            "all_occupied_blocks_processed": cache_audit["all_occupied_blocks_processed"],
            "all_voxels_covered": cache_audit["all_voxels_covered"],
            "scene_prefixes_built_before_questions": True,
            "scene_prefix_sha256_before": prefix_before,
            "scene_prefix_sha256_after": prefix_after,
            "scene_prefixes_question_independent_and_exact": prefix_before == prefix_after,
            "validation_scene_ids_loaded": [],
            "deferred_final_scene_ids_loaded": [],
            "question_dependent_retrieval": False,
            "source_cache_evidence": cache_audit["v39_source_cache_evidence"],
        },
        "qa_audit": qa_audit,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "training_authorized": False,
        "runtime_promotion_authorized": False,
        "declared_data_reads": declared_data_reads,
        **scope,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        report = preflight_v39(args.config)
    else:
        report = run_v39(args.config)
        if report.get("safety_attestations_passed_before_report_write") is not True:
            raise RuntimeError("V39 report write refused before safety attestation")
        _atomic_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "cosine_conflict_matrix",
    "evaluate_pass_contract",
    "gradient_cosine",
    "gradient_dot",
    "objective_contract",
    "preflight_v39",
    "run_v39",
]
