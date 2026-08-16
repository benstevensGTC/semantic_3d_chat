"""Locked preflight for the first true Gemma-4 embodied tool decoder.

The preflight is deliberately read-only.  It authenticates local inputs and
the one preregistered trainable surface without loading the 10.25 GiB model or
starting optimization.  A separate launch authorization is required before a
multi-update MPS job may exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.language.gemma4_tool_decoder_v1 import (
    CLEARANCE_STATE_DIM,
    CLEARANCE_TOKEN_COUNT,
    CONTROL_TOKEN_COUNT,
    HIDDEN_SIZE,
    K_PROJECTION,
    LORA_ALPHA,
    LORA_PARAMETER_COUNT,
    LORA_RANK,
    MODEL_ID,
    MODEL_REVISION,
    PROJECTOR_INITIAL_OUTPUT_SCALE,
    PROJECTOR_INITIALIZATION_SEED,
    PROJECTOR_PARAMETER_COUNT,
    TARGET_STATE_DIM,
    TARGET_TOKEN_COUNT,
    TOTAL_TRAINABLE_PARAMETER_COUNT,
    V_PROJECTION,
    NumericToolContextProjectorV1,
    prepare_tool_decoder_inputs,
    tool_decoder_lora_settings,
)
from semantic_3d_chat.language.lora import (
    initialize_lora_adapter_state,
    install_lora_adapters,
    lora_banks_settings,
    tensor_state_sha256,
)

CONFIG_PATH: Final[str] = "configs/experiments/gemma4_embodied_tool_decoder_v1.yaml"
RUNTIME_CONFIG_PATH: Final[str] = "configs/runtime/embodied_v54.yaml"
BASE_RUNTIME_CONFIG_PATH: Final[str] = "configs/runtime/gemma4_v54.yaml"
BASE_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000"
)
TRACE_ROOT: Final[str] = "data_gemma4/training/navigation_policy_v3"
PREFIX_ROOT: Final[str] = (
    "data_gemma4/scene_tokens/v56_question_control_full_prefixes"
)
ROBOT_STATE_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/robot_state_numeric_v1"
)
PREREGISTRATION_PATH: Final[str] = (
    "reports/gemma4/metrics/gemma4_embodied_tool_decoder_preregistration_v1.json"
)
PREFLIGHT_PATH: Final[str] = (
    "reports/gemma4/metrics/gemma4_embodied_tool_decoder_preflight_v1.json"
)
TOOL_BANK_NAME: Final[str] = "embodied_tool_decoder_v1_kv"
INITIAL_LORA_STATE_SHA256: Final[str] = (
    "00691e1da8959e8f6cefa56aa553749a63bb9b7ec773b77a2619d7b1989fd50d"
)
INITIAL_PROJECTOR_STATE_SHA256: Final[str] = (
    "bbc1db24367b03eabb061db1b504c6c23cce3d9c4151d46f2772dda4ee48daef"
)
PREFIX_INVENTORY_SHA256: Final[str] = (
    "c477fd12bc4104f147f73c2f6d46904e0b83b3c584206cb227fd70e9371d0d63"
)
TRAIN_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(11, 25)
)
VALIDATION_SCENES: Final[tuple[str, ...]] = (
    "scene_000031",
    "scene_000032",
    "scene_000033",
    "scene_000034",
    "scene_000035",
    "scene_000036",
    "scene_000037",
    "scene_000039",
)
EXPECTED_FILE_SHA256: Final[dict[str, str]] = {
    CONFIG_PATH: "5154334157a4741bdc46bfdec889aa149e62e8ca6f06605779242609fee22f0d",
    RUNTIME_CONFIG_PATH: "ba7c47e377d3b1a352e0c8f6348b7c72d39cf372526afea37da3c4dd711212ca",
    f"{TRACE_ROOT}/manifest.json": (
        "005756918c54fbffbb7c6db45e2170174d85f87f278e755e538418d6eb880243"
    ),
    f"{TRACE_ROOT}/traces.jsonl": (
        "72434178ff1cf23c2dfeb98d52cb7b4c443fcc8715c1dd4ee883d87ae127e7ad"
    ),
    f"{BASE_CHECKPOINT}/adapter.safetensors": (
        "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf"
    ),
    f"{BASE_CHECKPOINT}/metadata.json": (
        "db1435f8d38ca587e34dcd55dc4d37532efc0504bfb62bc115838dc0ab7a7ece"
    ),
    f"{BASE_CHECKPOINT}/runtime_metadata.json": (
        "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd"
    ),
    f"{ROBOT_STATE_CHECKPOINT}/state.safetensors": (
        "5d6aa13208264e0a99755d84e8f68b7727249b274c460e9d4e26541cd8e46938"
    ),
    f"{ROBOT_STATE_CHECKPOINT}/runtime_metadata.json": (
        "c48b8748dbde04f2c9294321974b1b13be2d77083970f051ba1c11a9b42d1985"
    ),
}
MODEL_FILE_SHA256: Final[dict[str, str]] = {
    "config.json": "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
    "tokenizer_config.json": (
        "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"
    ),
    "model.safetensors": (
        "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_snapshot() -> Path:
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        hub = Path(configured).expanduser()
    else:
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        hub = home.expanduser() / "hub"
    return (
        hub
        / f"models--{MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / MODEL_REVISION
    )


class _ProjectionPair(nn.Module):
    """Shape-only model used to reproduce the locked LoRA initialization."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [nn.Identity() for _ in range(34)] + [nn.Module()]
        )
        layer = self.model.language_model.layers[34]
        layer.self_attn = nn.Module()
        layer.self_attn.k_proj = nn.Linear(1536, 512, bias=False)
        layer.self_attn.v_proj = nn.Linear(1536, 512, bias=False)


def _expected_initial_states() -> dict[str, Any]:
    pair = _ProjectionPair().requires_grad_(False)
    installation = install_lora_adapters(pair, tool_decoder_lora_settings())
    if installation is None:
        raise RuntimeError("Tool-decoder LoRA unexpectedly disabled")
    initialize_lora_adapter_state(installation, seed=PROJECTOR_INITIALIZATION_SEED)
    projector = NumericToolContextProjectorV1()
    return {
        "lora_state_sha256": installation.state_sha256(),
        "lora_parameter_count": installation.parameter_count,
        "lora_parameter_counts": installation.parameter_counts,
        "projector_state_sha256": tensor_state_sha256(projector.state_dict()),
        "projector_parameter_count": projector.trainable_parameter_count,
    }


def build_tool_decoder_preregistration() -> dict[str, Any]:
    """Return the immutable experiment design; no training is performed."""

    return {
        "schema_version": 1,
        "artifact": "gemma4_embodied_tool_decoder_preregistration_v1",
        "status": "design_locked_preflight_only_training_not_authorized",
        "research_question": (
            "Can local Gemma-4 E2B causally decode exact bounded JSON robot actions "
            "from a complete continuous 3D scene prefix, numeric robot tokens, a "
            "continuously grounded target, and anonymous numeric free-space geometry?"
        ),
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_files_only": True,
            "cloud_inference": False,
            "base_dtype": "bfloat16",
            "hidden_size": HIDDEN_SIZE,
            "layer_count": 35,
            "sliding_window": 512,
            "final_layer_index": 34,
            "final_layer_attention": "full_attention",
            "actual_checkpoint_tensor_shapes": {
                K_PROJECTION: [512, 1536],
                V_PROJECTION: [512, 1536],
            },
            "actual_checkpoint_file_sha256": MODEL_FILE_SHA256,
        },
        "continuous_input_layout": {
            "conceptual_order": [
                "bos_text_embedding",
                "native_boi_embedding",
                "256_complete_question_independent_scene_latents",
                "4_numeric_robot_state_tokens",
                "native_eoi_embedding",
                "environment_free_protocol_and_literal_user_instruction_tokens",
                "2_numeric_grounded_target_tokens",
                "2_numeric_anonymous_free_space_tokens",
                "teacher_forced_json_answer_tokens_training_only",
            ],
            "scene_prefix_tokens": 258,
            "robot_tokens": 4,
            "active_scene_robot_prefix_tokens": 262,
            "target_state_dim": TARGET_STATE_DIM,
            "target_tokens": TARGET_TOKEN_COUNT,
            "clearance_state_dim": CLEARANCE_STATE_DIM,
            "clearance_tokens": CLEARANCE_TOKEN_COUNT,
            "control_tokens": CONTROL_TOKEN_COUNT,
            "scene_prefix_computed_before_instruction": True,
            "static_scene_prefix_question_independent": True,
            "navigation_target_grounding_may_depend_on_user_instruction": True,
            "all_active_map_voxels_scored_for_target_grounding": True,
            "free_space_source": "sanitized_numeric_voxel_geometry_only",
            "gemma4_auxiliary_ple_for_continuous_slots": "native_pad_derived",
        },
        "user_text_policy": {
            "literal_user_navigation_instruction_is_allowed": True,
            "target_names_inside_user_instruction_are_allowed": True,
            "reason": (
                "The no-leakage prohibition applies to environmental information supplied "
                "by the program, not to the user's own question or navigation instruction."
            ),
            "program_supplied_environmental_text": [],
            "runtime_object_inventory": False,
            "runtime_oracle_labels": False,
        },
        "trainable_surface": {
            "adapter_type": "strict_unmerged_fp32_lora_plus_numeric_linear_projector",
            "why_kv_reader": (
                "Continuous slots carry their content in main inputs_embeds while all use "
                "non-semantic PAD-derived PLE. Final full-attention K/V directly turn those "
                "content-bearing hidden states into globally visible memory. V54's already-"
                "trained layer-34 Q/O reader remains frozen, and its exact targets are "
                "disjoint from this new bank."
            ),
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": 0.0,
                "exact_target_modules": [K_PROJECTION, V_PROJECTION],
                "module_shapes": [[512, 1536], [512, 1536]],
                "parameter_count": LORA_PARAMETER_COUNT,
                "initialization_seed": PROJECTOR_INITIALIZATION_SEED,
                "initial_state_sha256": INITIAL_LORA_STATE_SHA256,
                "base_weight_merge": False,
            },
            "numeric_projector": {
                "target_projection": [TARGET_STATE_DIM, TARGET_TOKEN_COUNT * HIDDEN_SIZE],
                "clearance_projection": [
                    CLEARANCE_STATE_DIM,
                    CLEARANCE_TOKEN_COUNT * HIDDEN_SIZE,
                ],
                "bias": True,
                "parameter_count": PROJECTOR_PARAMETER_COUNT,
                "initialization_seed": PROJECTOR_INITIALIZATION_SEED,
                "initial_output_scale": PROJECTOR_INITIAL_OUTPUT_SCALE,
                "initial_state_sha256": INITIAL_PROJECTOR_STATE_SHA256,
            },
            "total_trainable_parameter_count": TOTAL_TRAINABLE_PARAMETER_COUNT,
            "frozen": [
                "gemma4_base",
                "all_v54_lora_banks",
                "scene_encoder_and_prefixes",
                "robot_state_encoder",
                "vision_encoder",
                "semantic_map",
                "target_grounder",
            ],
        },
        "data": {
            "source": TRACE_ROOT,
            "trace_manifest_sha256": EXPECTED_FILE_SHA256[f"{TRACE_ROOT}/manifest.json"],
            "trace_rows_sha256": EXPECTED_FILE_SHA256[f"{TRACE_ROOT}/traces.jsonl"],
            "sample_count": 6468,
            "episode_count": 1370,
            "train_scene_ids": list(TRAIN_SCENES),
            "validation_scene_ids": list(VALIDATION_SCENES),
            "scene_splits_disjoint": True,
            "scene_prefix_root": PREFIX_ROOT,
            "scene_prefix_inventory_sha256": PREFIX_INVENTORY_SHA256,
            "oracle_target_xyz_allowed_training_only": True,
            "oracle_target_xyz_forbidden_runtime": True,
            "clearance_cache": (
                "data_gemma4/training/gemma4_embodied_tool_decoder_v1/clearance.safetensors"
            ),
            "clearance_cache_must_be_derived_before_launch": True,
            "clearance_cache_must_be_hashed_before_launch": True,
            "clearance_cache_source": "sanitized_map_centers_world_and_numeric_pose_only",
        },
        "output_protocol": {
            "format": "one_minified_json_object",
            "key_order": ["arguments", "tool"],
            "tool_vocabulary": ["stop", "scan", "turn", "move_forward", "move_backward"],
            "numeric_precision_decimal_places": 3,
            "schema_validation_before_execution": True,
            "exact_collision_interlock_before_execution": True,
            "prose_or_markdown_allowed": False,
        },
        "objective": {
            "answer_suffix_only_cross_entropy": True,
            "labels_before_json_suffix": -100,
            "token_normalized": True,
            "action_balanced_deterministic_sampler": True,
            "environmental_caption_loss": False,
            "intermediate_text_scene_description": False,
        },
        "optimization": {
            "seed": 2026081217,
            "optimizer": "adamw",
            "projector_learning_rate": 0.0002,
            "lora_learning_rate": 0.0001,
            "weight_decay": 0.0,
            "microbatch_size": 1,
            "gradient_accumulation": 8,
            "maximum_optimizer_updates": 64,
            "validation_every_updates": 8,
            "early_stopping_patience_validations": 3,
            "gradient_clip_l2": 1.0,
            "decoder_gradient_checkpointing": True,
            "bitsandbytes": False,
            "cuda_required": False,
            "preferred_device": "mps",
            "cpu_fallback": True,
            "float32_loss_and_finite_checks": True,
        },
        "hard_promotion_gates": {
            "validation_exact_json_accuracy_minimum": 0.60,
            "validation_valid_schema_rate_minimum": 0.95,
            "validation_tool_accuracy_minimum": 0.80,
            "validation_turn_sign_accuracy_minimum": 0.80,
            "validation_argument_mae_normalized_maximum": 0.25,
            "wrong_target_targeted_tool_accuracy_drop_minimum": 0.10,
            "zero_clearance_obstacle_tool_accuracy_drop_minimum": 0.05,
            "empty_scene_prefix_tool_accuracy_drop_minimum": 0.05,
            "collision_execution_count": 0,
            "oracle_removed_runtime_smoke_required": True,
            "all_gates_must_pass": True,
            "failed_run_publishes_no_runtime_checkpoint": True,
        },
        "required_controls": [
            "frozen_untrained_continuous_prefix_gemma_seam",
            "correct_continuous_context",
            "wrong_scene_prefix_same_instruction",
            "zero_scene_prefix",
            "wrong_grounded_target_same_instruction",
            "zero_grounded_target",
            "zero_free_space_state",
            "shuffled_free_space_state",
            "instruction_only",
            "oracle_directory_removed",
        ],
        "runtime_and_leakage": {
            "runtime_may_load": [
                "sanitized_continuous_scene_prefix",
                "numeric_robot_state_encoder",
                "tool_decoder_lora",
                "numeric_tool_context_projector",
                "sanitized_semantic_voxel_map",
                "frozen_local_gemma4_weights",
                "literal_user_instruction_tokens",
            ],
            "runtime_may_not_load": [
                "oracle",
                "training_traces",
                "training_qa",
                "scene_labels",
                "object_inventory",
                "caption",
                "serialized_scene_graph",
            ],
            "loaded_file_audit_required": True,
            "oracle_directory_removal_test_required": True,
            "raw_model_output_not_executed_before_schema_validation": True,
            "numeric_collision_interlock_authoritative": True,
        },
        "launch_authorization": {
            "preregistration_digest_required": True,
            "preflight_digest_required": True,
            "clearance_cache_digest_required": True,
            "tiny_cpu_backward_smoke_required": True,
            "one_full_model_mps_microbatch_smoke_required": True,
            "parent_release_of_heavy_mps_slot_required": True,
            "multi_update_training_authorized": False,
        },
        "execution": {
            "training_executed": False,
            "full_model_generation_executed": False,
            "runtime_checkpoint_published": False,
        },
    }


def _validate_model_snapshot(*, full_weight_hash: bool) -> dict[str, Any]:
    root = _model_snapshot()
    if not root.is_dir():
        raise FileNotFoundError(f"Pinned local Gemma snapshot is missing: {root}")
    observed_hashes: dict[str, str] = {}
    for name, expected in MODEL_FILE_SHA256.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned local Gemma file is missing: {path}")
        if name == "model.safetensors" and not full_weight_hash:
            resolved_name = path.resolve().name
            observed = resolved_name if len(resolved_name) == 64 else "not_content_addressed"
        else:
            observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"Pinned local Gemma file hash changed: {name}")
        observed_hashes[name] = observed

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise TypeError("Gemma text_config is unavailable")
    layer_types = text.get("layer_types")
    if (
        text.get("hidden_size") != HIDDEN_SIZE
        or text.get("num_hidden_layers") != 35
        or text.get("sliding_window") != 512
        or not isinstance(layer_types, list)
        or len(layer_types) != 35
        or layer_types[34] != "full_attention"
    ):
        raise ValueError("Pinned Gemma text architecture changed")
    weights = root / "model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as archive:
        shapes = {
            name: list(archive.get_slice(f"{name}.weight").get_shape())
            for name in (K_PROJECTION, V_PROJECTION)
        }
    if shapes != {K_PROJECTION: [512, 1536], V_PROJECTION: [512, 1536]}:
        raise ValueError("Pinned Gemma final full-attention K/V shapes changed")
    return {
        "snapshot": str(root.resolve()),
        "file_sha256": observed_hashes,
        "tensor_shapes": shapes,
        "full_weight_hash_recomputed": full_weight_hash,
    }


def _validate_project_files() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_FILE_SHA256.items():
        if expected == "pending_until_preregistration_is_written":
            continue
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Preregistered input is unavailable: {relative}")
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(f"Preregistered input hash changed: {relative}")
        observed[relative] = digest
    return observed


def _validate_traces() -> dict[str, Any]:
    manifest = json.loads((PROJECT_ROOT / TRACE_ROOT / "manifest.json").read_text())
    if (
        manifest.get("sample_count") != 6468
        or manifest.get("episode_count") != 1370
        or manifest.get("train_scene_ids") != list(TRAIN_SCENES)
        or manifest.get("validation_scene_ids") != list(VALIDATION_SCENES)
        or set(TRAIN_SCENES) & set(VALIDATION_SCENES)
        or manifest.get("scene_splits_disjoint") is not True
        or manifest.get("target_coordinates_training_tree_only") is not True
        or manifest.get("runtime_oracle_inputs") is not False
    ):
        raise ValueError("V3 traces no longer satisfy the preregistered split/isolation")
    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    traces = PROJECT_ROOT / TRACE_ROOT / "traces.jsonl"
    with traces.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row.get("sample_id") != f"g_{index:08d}":
                raise ValueError("V3 trace sample order changed")
            action = row.get("action_name")
            split = row.get("split")
            if action not in {"stop", "scan", "turn", "move_forward", "move_backward"}:
                raise ValueError("V3 trace includes an unregistered tool action")
            if split not in {"train", "validation"}:
                raise ValueError("V3 trace includes an invalid split")
            counts[str(action)] += 1
            split_counts[str(split)] += 1
    if sum(counts.values()) != 6468 or any(counts[action] < 1 for action in counts):
        raise ValueError("V3 trace action coverage changed")
    return {
        "sample_count": sum(counts.values()),
        "action_counts": dict(sorted(counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }


def _validate_prefixes() -> dict[str, Any]:
    root = PROJECT_ROOT / PREFIX_ROOT
    inventory: dict[str, str] = {}
    for scene_id in (*TRAIN_SCENES, *VALIDATION_SCENES):
        path = root / f"{scene_id}.safetensors"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Preregistered scene prefix is unavailable: {scene_id}")
        with safe_open(path, framework="pt", device="cpu") as archive:
            if archive.keys() != ["scene_prefix"]:
                raise ValueError(f"Scene prefix tensor inventory changed: {scene_id}")
            tensor = archive.get_tensor("scene_prefix")
        if tensor.shape != (1, 258, HIDDEN_SIZE) or tensor.dtype != torch.bfloat16:
            raise ValueError(f"Scene prefix shape or dtype changed: {scene_id}")
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"Scene prefix contains NaN or infinity: {scene_id}")
        inventory[scene_id] = _sha256(path)
    digest = _canonical_sha256(inventory)
    if digest != PREFIX_INVENTORY_SHA256:
        raise ValueError("Preregistered scene-prefix inventory hash changed")
    return {
        "scene_count": len(inventory),
        "inventory_sha256": digest,
        "shape": [1, 258, HIDDEN_SIZE],
        "dtype": "torch.bfloat16",
    }


def _validate_experiment_config() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    banks = lora_banks_settings(config)
    bank = banks.bank(TOOL_BANK_NAME)
    if (
        tuple(bank.adapter.target_modules) != (K_PROJECTION, V_PROJECTION)
        or bank.adapter.rank != LORA_RANK
        or bank.adapter.alpha != LORA_ALPHA
        or bank.adapter.dropout != 0.0
        or not bank.trainable
        or bank.initialization_algorithm != "cpu_kaiming_uniform_a_exact_zero_b"
        or bank.initialization_seed != PROJECTOR_INITIALIZATION_SEED
        or bank.expected_initial_state_sha256 != INITIAL_LORA_STATE_SHA256
    ):
        raise ValueError("Tool-decoder config differs from the preregistered LoRA arm")
    other_targets = {
        target
        for candidate in banks.banks
        if candidate.name != TOOL_BANK_NAME
        for target in candidate.adapter.target_modules
    }
    if set(bank.adapter.target_modules) & other_targets:
        raise ValueError("Tool-decoder LoRA targets overlap a frozen V54 bank")
    if any(candidate.trainable for candidate in banks.banks if candidate.name != TOOL_BANK_NAME):
        raise ValueError("A frozen V54 LoRA bank became trainable")
    experiment = config.get("gemma4_embodied_tool_decoder_v1")
    if not isinstance(experiment, Mapping):
        raise TypeError("Config has no gemma4_embodied_tool_decoder_v1 mapping")
    expected = build_tool_decoder_preregistration()
    if (
        experiment.get("training_authorized") is not False
        or experiment.get("total_trainable_parameter_count")
        != TOTAL_TRAINABLE_PARAMETER_COUNT
        or experiment.get("preregistration") != PREREGISTRATION_PATH
        or config.get("navigation_policy_v3", {}).get("train_scene_ids")
        != expected["data"]["train_scene_ids"]
        or config.get("navigation_policy_v3", {}).get("validation_scene_ids")
        != expected["data"]["validation_scene_ids"]
    ):
        raise ValueError("Tool-decoder experiment config differs from preregistration")
    return {
        "bank_count": len(banks.banks),
        "new_bank": TOOL_BANK_NAME,
        "new_targets": list(bank.adapter.target_modules),
        "all_older_banks_frozen": True,
        "target_sets_disjoint": True,
        "training_authorized": False,
    }


def run_tool_decoder_preflight(*, full_weight_hash: bool = False) -> dict[str, Any]:
    """Authenticate the design without loading Gemma or starting training."""

    preregistration = build_tool_decoder_preregistration()
    preregistration_path = PROJECT_ROOT / PREREGISTRATION_PATH
    if not preregistration_path.is_file():
        raise FileNotFoundError("Tool-decoder preregistration artifact has not been sealed")
    artifact = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if artifact != preregistration:
        raise ValueError("Tool-decoder preregistration artifact differs from source contract")
    states = _expected_initial_states()
    if states != {
        "lora_state_sha256": INITIAL_LORA_STATE_SHA256,
        "lora_parameter_count": LORA_PARAMETER_COUNT,
        "lora_parameter_counts": {
            K_PROJECTION: 8192,
            V_PROJECTION: 8192,
        },
        "projector_state_sha256": INITIAL_PROJECTOR_STATE_SHA256,
        "projector_parameter_count": PROJECTOR_PARAMETER_COUNT,
    }:
        raise ValueError("Tool-decoder deterministic initial state changed")
    return {
        "schema_version": 1,
        "artifact": "gemma4_embodied_tool_decoder_preflight_v1",
        "status": "passed_design_preflight_training_not_authorized",
        "preregistration_sha256": _sha256(preregistration_path),
        "model": _validate_model_snapshot(full_weight_hash=full_weight_hash),
        "project_files": _validate_project_files(),
        "experiment_config": _validate_experiment_config(),
        "trace_dataset": _validate_traces(),
        "prefixes": _validate_prefixes(),
        "initial_states": states,
        "trainable_parameter_count": TOTAL_TRAINABLE_PARAMETER_COUNT,
        "clearance_cache_materialized": False,
        "tiny_cpu_backward_smoke_passed": False,
        "full_model_mps_microbatch_smoke_executed": False,
        "multi_update_training_authorized": False,
        "gemma_model_loaded": False,
        "mps_memory_allocated": False,
    }


def _tiny_gemma4_config() -> Any:
    """Return a 35-layer, very small CPU-only Gemma-4 structural surrogate."""

    try:
        from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
    except ImportError as exc:  # pragma: no cover - isolated Gemma venv is required
        raise RuntimeError("Tiny Gemma-4 smoke requires the isolated Gemma environment") from exc

    text = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=35,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=64,
        layer_types=(
            [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ]
            * 7
        ),
        sliding_window=32,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        use_double_wide_mlp=False,
        # Match the pinned E2B routing contract so layer 34 owns K/V instead
        # of consuming a shared pair from its predecessor.
        num_kv_shared_layers=20,
    )
    vision = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        pooling_kernel_size=3,
        patch_size=16,
        position_embedding_size=64,
        use_clipped_linears=False,
        standardize=False,
    )
    return Gemma4Config(
        text_config=text,
        vision_config=vision,
        audio_config=None,
        image_token_id=60,
        video_token_id=61,
        audio_token_id=62,
        boi_token_id=58,
        eoi_token_id=59,
    )


def run_tiny_cpu_backward_smoke() -> dict[str, Any]:
    """Run exactly one CPU microbatch through a tiny true Gemma-4 decoder.

    This proves PLE-aware ``inputs_embeds`` assembly and gradient connectivity
    to both authorized trainable surfaces.  It is not evidence about the full
    checkpoint's accuracy, memory use, or training time.
    """

    try:
        from transformers import Gemma4ForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - isolated Gemma venv is required
        raise RuntimeError("Tiny Gemma-4 smoke requires the isolated Gemma environment") from exc

    from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend

    started = time.perf_counter()
    torch.manual_seed(2026081217)
    model = Gemma4ForConditionalGeneration(_tiny_gemma4_config()).cpu().eval()
    model.requires_grad_(False)
    installation = install_lora_adapters(model, tool_decoder_lora_settings())
    if installation is None:
        raise RuntimeError("Tiny tool-decoder LoRA unexpectedly disabled")
    initialize_lora_adapter_state(installation, seed=PROJECTOR_INITIALIZATION_SEED)
    projector = NumericToolContextProjectorV1(hidden_size=32).cpu().train()
    backend = Gemma4PrefixBackend(model, model_revision="tiny-cpu-structural-smoke")
    boi, eoi = backend.native_boundary_embeddings()
    latent_content = torch.randn(1, 8, 32, dtype=torch.float32) * 0.05
    active_prefix = torch.cat((boi, latent_content, eoi), dim=1)
    prompt_ids = torch.tensor([[2, 9, 13]], dtype=torch.long)
    answer_ids = torch.tensor([[17, 18, 1]], dtype=torch.long)
    target_state = torch.tensor(
        [[1.0, 0.2, -0.1, 0.4, 0.1, -0.2, 0.3, 0.5, 0.6, 0.8]],
        dtype=torch.float32,
    )
    clearance_state = torch.linspace(0.1, 1.0, CLEARANCE_STATE_DIM).unsqueeze(0)
    prepared = prepare_tool_decoder_inputs(
        backend,
        active_prefix,
        prompt_ids,
        projector,
        target_state,
        clearance_state,
        answer_ids=answer_ids,
    )
    output = backend.prefill(prepared, use_cache=False)
    loss = output.loss.float()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Tiny Gemma tool-decoder loss is invalid")
    loss.backward()
    projector_gradient_l2 = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in projector.parameters()
                if parameter.grad is not None
            )
        )
    )
    lora_b_gradient_l2 = float(
        torch.sqrt(
            sum(
                adapter.lora_b.grad.detach().float().square().sum()
                for adapter in installation.adapters
                if adapter.lora_b.grad is not None
            )
        )
    )
    lora_a_gradient_l2 = float(
        torch.sqrt(
            sum(
                adapter.lora_a.grad.detach().float().square().sum()
                for adapter in installation.adapters
                if adapter.lora_a.grad is not None
            )
        )
    )
    if projector_gradient_l2 <= 0.0 or lora_b_gradient_l2 <= 0.0:
        raise RuntimeError("Tiny Gemma gradients did not reach both authorized surfaces")
    if lora_a_gradient_l2 != 0.0:
        raise RuntimeError("Zero-output LoRA A gradient should be exactly zero on microbatch one")
    trainable_model_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_model_parameters != installation.parameter_count:
        raise RuntimeError("Tiny Gemma trainable surface contains unexpected parameters")
    ignored = int((prepared.labels == -100).sum()) if prepared.labels is not None else -1
    expected_ignored = prepared.inputs_embeds.shape[1] - answer_ids.shape[1]
    if ignored != expected_ignored:
        raise RuntimeError("Tiny Gemma smoke labels escaped the JSON answer suffix")
    return {
        "schema_version": 1,
        "artifact": "gemma4_embodied_tool_decoder_tiny_cpu_backward_smoke_v1",
        "status": "passed",
        "device": "cpu",
        "microbatches": 1,
        "optimizer_steps": 0,
        "tiny_hidden_size": 32,
        "tiny_layer_count": 35,
        "tiny_final_layer_attention": "full_attention",
        "input_sequence_length": int(prepared.inputs_embeds.shape[1]),
        "answer_token_count": int(answer_ids.shape[1]),
        "ignored_label_count": ignored,
        "control_token_count": CONTROL_TOKEN_COUNT,
        "loss": float(loss.detach()),
        "projector_gradient_l2": projector_gradient_l2,
        "lora_b_gradient_l2": lora_b_gradient_l2,
        "lora_a_gradient_l2_expected_zero": lora_a_gradient_l2,
        "base_model_trainable_parameter_count": 0,
        "tiny_lora_trainable_parameter_count": installation.parameter_count,
        "tiny_projector_trainable_parameter_count": projector.trainable_parameter_count,
        "elapsed_seconds": time.perf_counter() - started,
        "full_checkpoint_loaded": False,
        "mps_used": False,
        "cloud_inference_used": False,
        "training_executed": False,
    }


__all__ = [
    "CONFIG_PATH",
    "INITIAL_LORA_STATE_SHA256",
    "INITIAL_PROJECTOR_STATE_SHA256",
    "PREFIX_INVENTORY_SHA256",
    "PREFLIGHT_PATH",
    "PREREGISTRATION_PATH",
    "TOOL_BANK_NAME",
    "TRAIN_SCENES",
    "VALIDATION_SCENES",
    "build_tool_decoder_preregistration",
    "run_tiny_cpu_backward_smoke",
    "run_tool_decoder_preflight",
]
