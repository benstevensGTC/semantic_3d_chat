"""Select a fully completed V41 projected-gradient development checkpoint.

This is a deliberately fail-closed post-training process.  The exact V41
checkpoint directory envelope, terminal train-only gate, tensor transition,
runtime metadata, and direction-preserving SGD state are authenticated before the first
validation evaluator can be constructed.  Validation remains restricted to
scenes 19--24; oracle and deferred final scenes are never legal inputs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.evaluation.v27_sidecar_screen import _atomic_json
from semantic_3d_chat.evaluation.v28_stage_b_selector import _retention_control_config
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    SelectionRequirements,
    _metadata,
    _selection_requirements,
    _source_v29_evidence,
    _validate_source_against_config,
)
from semantic_3d_chat.evaluation.v35_block_cross_selector import (
    V35GreedyEvidence,
    V35TeacherEvidence,
    _promotion,
    _V35RuntimeEvaluator,
)
from semantic_3d_chat.evaluation.v37_scene_ingress_kv_selector import (
    _approved_v29_envelope,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    module_collection_state_sha256,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.train_block_cross_v35 import validate_v35_cache_audit
from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract
from semantic_3d_chat.training.train_projected_gradient_v41 import (
    OPTIMIZER_AUDIT_FILENAME,
    V41Contract,
    optimizer_step_audit,
    replay_v41_gates,
    require_exact_v41_sources,
    require_v39_terminal_gate,
    retag_bundle_for_v41,
    v41_contract,
    v41_loader_config,
    validate_per_unit_nll_diagnostics,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_projected_gradient_v41.yaml")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v41_diverse28_projected_gradient_l14_query"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v41_projected_gradient_selection.json")

_QUERY_BANK = "extension_v28_stage_b_query"
_QUERY_PREFIX = f"lora_banks.{_QUERY_BANK}."
_V23_BANK = "extension_v23_shared_kv"
_V23_PREFIX = f"lora_banks.{_V23_BANK}."
_CORE_PREFIX = "block_cross_residual."
_QUERY_PARAMETER_NAMES = (f"{_QUERY_PREFIX}adapters.1.lora_b",)
_QUERY_PARAMETER_NAME_SET = frozenset(_QUERY_PARAMETER_NAMES)
_QUERY_SHAPES = ((4096, 4),)
_QUERY_MODULES = ("model.language_model.layers.14.self_attn.q_proj",)
_EXPECTED_STEPS = (0, 8, 16, 24, 32, 40, 41)
_DIAGNOSTIC_STEPS = frozenset({0, 8, 16, 41})
_GREEDY_STEPS = frozenset({16, 24, 32, 40, 41})
_PRIORITY_FAMILIES = ("book_support", "mirror_lr", "picture_support")
_SOURCE_NLL_TOLERANCE = 1e-6
_SOURCE_LOCAL_A_SHA256 = "9f0ee5f9bbb9ec07bd42aaca1e0817be567a11c396c693e6412e5f2b08f37403"
_AUTHORIZED_CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "data_gemma4/checkpoints/gemma4_v41_diverse28_projected_gradient_l14_query"
).resolve()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _prefixed_state(
    tensors: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }


def _query_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {
        name.removeprefix(f"{_QUERY_PREFIX}adapters.1."): value
        for name, value in tensors.items()
        if name in _QUERY_PARAMETER_NAME_SET
    }
    observed = tuple(f"{_QUERY_PREFIX}adapters.1.{name}" for name in state)
    if observed != _QUERY_PARAMETER_NAMES:
        raise ValueError("V41 query-bank tensor order or inventory changed")
    if tuple(tuple(value.shape) for value in state.values()) != _QUERY_SHAPES:
        raise ValueError("V41 query-bank tensor shapes changed")
    return state


def _frozen_excluding_query(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value for name, value in tensors.items() if name not in _QUERY_PARAMETER_NAME_SET
    }


def install_v41_target_b(
    adapter: torch.nn.Module,
    query: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    """Install only layer-14 LoRA-B while proving learned LoRA-A is unchanged."""

    if tuple(query) != ("lora_b",) or tuple(query["lora_b"].shape) != _QUERY_SHAPES[0]:
        raise ValueError("V41 target installer received a changed LoRA-B state")
    lora_a = getattr(adapter, "lora_a", None)
    lora_b = getattr(adapter, "lora_b", None)
    if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor):
        raise TypeError("V41 target adapter lacks LoRA A/B tensors")
    a_before = tensor_state_sha256({"lora_a": lora_a})
    with torch.no_grad():
        lora_b.copy_(query["lora_b"].to(device=lora_b.device, dtype=lora_b.dtype))
    a_after = tensor_state_sha256({"lora_a": lora_a})
    b_after = tensor_state_sha256({"lora_b": lora_b})
    expected_b = tensor_state_sha256(query)
    if a_before != a_after or b_after != expected_b:
        raise RuntimeError("V41 target installation changed LoRA-A or failed to install B")
    return {
        "lora_a_before_sha256": a_before,
        "lora_a_after_sha256": a_after,
        "lora_b_after_sha256": b_after,
    }


def _checkpoint_paths_or_raise(
    checkpoint_root: Path, contract: V41Contract
) -> tuple[Path, ...]:
    """Resolve the exact completed envelope without opening a data artifact."""

    if checkpoint_root.resolve() != _AUTHORIZED_CHECKPOINT_ROOT:
        raise ValueError("V41 selector checkpoint root differs from terminal authorization")
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError("V41 checkpoint root must be a real directory")
    if tuple(contract.saved_optimizer_steps) != _EXPECTED_STEPS:
        raise ValueError("V41 saved-step contract changed")
    checkpoints = tuple(
        checkpoint_root / f"update_{step:03d}" for step in contract.saved_optimizer_steps
    )
    observed = sorted(path.name for path in checkpoint_root.glob("update_*"))
    expected = [path.name for path in checkpoints]
    if observed != expected:
        raise FileNotFoundError(
            "V41 requires the exact completed update-41 envelope: "
            f"observed={observed} expected={expected}"
        )
    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError(f"V41 arm must be a real directory: {checkpoint}")
        required = [
            "adapter.safetensors",
            TRAINING_METADATA_FILENAME,
            RUNTIME_METADATA_FILENAME,
        ]
        if step:
            required.extend(("optimizer.pt", OPTIMIZER_AUDIT_FILENAME))
        if any(
            not (checkpoint / name).is_file() or (checkpoint / name).is_symlink()
            for name in required
        ):
            raise FileNotFoundError(f"V41 arm is incomplete or aliased: {checkpoint.name}")
        if step == 0 and any(
            (checkpoint / name).exists()
            for name in ("optimizer.pt", OPTIMIZER_AUDIT_FILENAME)
        ):
            raise ValueError("V41 update zero must not persist SGD state")
    return checkpoints


def _training_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute expected train-only paths without opening QA or map files."""

    loader = v41_loader_config(config)
    split = v31_contract(loader)
    qa_root = artifact_root(loader, "qa")
    map_root = artifact_root(loader, "maps")
    return {
        "train_scene_ids": list(split.train_scene_ids),
        "validation_scene_ids": list(split.validation_scene_ids),
        "qa_loaded_files": [str(qa_root / "splits.json"), str(qa_root / "train.jsonl")],
        "validation_qa_path": str(qa_root / "validation.jsonl"),
        "map_loaded_files": [
            str((map_root / scene_id / "voxel_map.npz").resolve())
            for scene_id in split.train_scene_ids
        ],
    }


def _validate_train_only_boundary(
    stage: Mapping[str, Any], label: str, provenance: Mapping[str, Any]
) -> None:
    required_false = (
        "validation_qa_loaded",
        "oracle_environment_files_loaded",
        "question_dependent_scene_processing",
        "question_dependent_retrieval",
        "source_optimizer_states_loaded",
        "source_optimizer_files_opened",
    )
    if any(stage.get(key) is not False for key in required_false):
        raise ValueError(f"V41 {label} crossed its train-only boundary")
    if stage.get("deferred_final_scene_ids_loaded") != []:
        raise ValueError(f"V41 {label} touched deferred final scenes")
    qa = _mapping(stage.get("train_qa_dataset"), f"{label} train QA")
    if (
        qa.get("schema_version") != 1
        or qa.get("qa_root") != str(Path(provenance["qa_loaded_files"][0]).parent)
        or qa.get("loaded_files") != provenance["qa_loaded_files"]
        or qa.get("train_question_count") != 384
        or qa.get("train_scene_ids") != provenance["train_scene_ids"]
        or qa.get("train_changed_pair_unit_count") != 25
        or qa.get("validation_scene_ids_from_pinned_contract")
        != provenance["validation_scene_ids"]
        or qa.get("validation_qa_path") != provenance["validation_qa_path"]
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError(f"V41 {label} QA provenance is not exact train-only input")
    cache = _mapping(stage.get("scene_cache"), f"{label} scene cache")
    try:
        validate_v35_cache_audit(cache, expected_scene_ids=provenance["train_scene_ids"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V41 {label} map provenance is not exact train-only input") from exc
    if (
        cache.get("loaded_environment_files") != provenance["map_loaded_files"]
        or cache.get("scene_scope") != "training_only"
        or cache.get("authenticated_manifest_scene_count") != 22
        or cache.get("authenticated_manifest_train_subset_count") != 16
        or cache.get("validation_scene_ids_loaded") != []
        or cache.get("validation_environment_maps_loaded") is not False
        or cache.get("deferred_final_scene_ids_loaded") != []
        or stage.get("independent_selector_required") is not True
    ):
        raise ValueError(f"V41 {label} map provenance is not exact train-only input")


def _validate_prefix_replay(
    stage: Mapping[str, Any], label: str, provenance: Mapping[str, Any]
) -> None:
    replay = _mapping(stage.get("prefix_replay_attestation"), f"{label} prefix replay")
    first = _mapping(replay.get("prefix_sha256_by_scene"), f"{label} first prefix hashes")
    repeated = _mapping(
        replay.get("replayed_prefix_sha256_by_scene"), f"{label} replayed prefix hashes"
    )
    expected_ids = provenance["train_scene_ids"]
    if (
        replay.get("scene_count") != 16
        or replay.get("scene_ids") != expected_ids
        or replay.get("prefixes_replayed_bit_exact") is not True
        or replay.get("scene_prefixes_built_before_questions") is not True
        or replay.get("all_occupied_blocks_processed") is not True
        or replay.get("training_scene_prefixes_question_free") is not True
        or replay.get("validation_environment_maps_loaded") is not False
        or replay.get("validation_qa_loaded") is not False
        or replay.get("question_dependent_scene_processing") is not False
        or replay.get("question_dependent_retrieval") is not False
        or tuple(first) != tuple(expected_ids)
        or dict(first) != dict(repeated)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in first.values()
        )
    ):
        raise ValueError(f"V41 {label} prefix replay is incomplete or non-deterministic")


def _validate_surface(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    contract: V41Contract,
) -> dict[str, Any]:
    stage = _mapping(metadata.get("v41_projected_gradient"), "V41 stage")
    surface = _mapping(stage.get("trainable_surface"), "V41 trainable surface")
    expected_surface = {
        "target_bank": _QUERY_BANK,
        "target_adapter_index": 1,
        "target_adapter_tensor": "lora_b",
        "target_module_paths": list(_QUERY_MODULES),
        "target_parameter_names": list(_QUERY_PARAMETER_NAMES),
        "trainable_tensor_count": 1,
        "trainable_parameter_count": 16_384,
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.0,
        "existing_learned_bank_continued_without_reinitialization": True,
        "gemma_base_frozen": True,
        "v23_k_only_hybrid_frozen": True,
        "v36_learned_block_core_frozen": True,
        "complete_scene_stack_frozen": True,
        "all_other_lora_banks_frozen": True,
        "every_other_tensor_and_buffer_frozen": True,
    }
    if dict(surface) != expected_surface:
        raise ValueError("V41 persisted trainable surface changed")
    query = _query_state(tensors)
    frozen = _frozen_excluding_query(tensors)
    query_hash = tensor_state_sha256(query)
    v28_hash = tensor_state_sha256(_prefixed_state(tensors, _QUERY_PREFIX))
    frozen_hash = tensor_state_sha256(frozen)
    v23_hash = tensor_state_sha256(_prefixed_state(tensors, _V23_PREFIX))
    core_hash = tensor_state_sha256(_prefixed_state(tensors, _CORE_PREFIX))
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "V41 LoRA hashes")
    if (
        sum(value.numel() for value in query.values()) != 16_384
        or any(not torch.isfinite(value).all() for value in query.values())
        or query_hash != stage.get("target_lora_b_state_sha256")
        or v28_hash != stage.get("complete_v28_bank_state_sha256")
        or v28_hash != bank_hashes.get(_QUERY_BANK)
    ):
        raise ValueError("V41 query-bank state or metadata changed")
    if (
        frozen_hash != contract.frozen_state_sha256
        or frozen_hash != stage.get("frozen_excluding_query_state_sha256")
    ):
        raise ValueError("V41 changed a frozen tensor or buffer")
    if (
        v23_hash != contract.hybrid_v23_state_sha256
        or v23_hash != stage.get("hybrid_v23_state_sha256")
        or v23_hash != bank_hashes.get(_V23_BANK)
    ):
        raise ValueError("V41 changed its frozen K-only V23 hybrid")
    if (
        core_hash != contract.core_state_sha256
        or core_hash != stage.get("source_block_core_state_sha256")
        or core_hash != metadata.get("block_cross_residual_state_sha256")
    ):
        raise ValueError("V41 changed its frozen learned block core")
    if metadata.get("lora_trainable_parameter_count") != 0:
        raise ValueError("V41 checkpoint advertises the wrong trainable count")
    return {
        "query_bank_state_sha256": query_hash,
        "complete_v28_bank_state_sha256": v28_hash,
        "frozen_excluding_target_state_sha256": frozen_hash,
        "hybrid_v23_state_sha256": v23_hash,
        "learned_block_core_state_sha256": core_hash,
        "authorized_parameter_count": 16_384,
        "authorized_tensor_count": 1,
    }


def _row_at(history: list[Mapping[str, Any]], step: int) -> Mapping[str, Any]:
    if step >= len(history) or history[step].get("optimizer_update") != step:
        raise ValueError(f"V41 history lacks exact optimizer update {step}")
    return history[step]


def _validate_update_zero_evidence(
    stage: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    contract: V41Contract,
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    attestation = _mapping(stage.get("update_zero_attestation"), "V41 update-zero proof")
    required = {
        "exact_v38_update_zero_source_loaded": True,
        "hybrid_tensor_state_sha256": contract.hybrid_tensor_state_sha256,
        "hybrid_v23_state_sha256": contract.hybrid_v23_state_sha256,
        "existing_learned_query_loaded_without_reinitialization": True,
        "complete_v28_bank_source_state_sha256": (
            "cc9dfa838bb87f32e2922d675658af4a1085d53a84ccdca6d5bacc6f7097217b"
        ),
        "target_lora_b_source_state_sha256": contract.query_source_state_sha256,
        "target_lora_b_nonzero": True,
        "learned_block_core_state_sha256": contract.core_state_sha256,
        "frozen_excluding_query_state_sha256": contract.frozen_state_sha256,
        "source_optimizer_files_opened": False,
        "source_optimizer_states_loaded": False,
        "fresh_momentum_free_sgd_state": True,
        "hybrid_behavior_recomputed_before_optimizer": True,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
    }
    if any(attestation.get(key) != value for key, value in required.items()):
        raise ValueError("V41 update-zero hybrid attestation changed")
    baseline = _mapping(attestation.get("behavioral_baseline"), "V41 behavioral baseline")
    observed = _mapping(baseline.get("observed"), "V41 update-zero observed baseline")
    expected = _mapping(baseline.get("expected"), "V41 update-zero expected baseline")
    tolerance = float(contract.update_zero_expected["floating_absolute_tolerance"])
    floating = {
        "broad_train_nll",
        "priority_book_side_deficit",
        "priority_picture_side_deficit",
        "priority_combined_side_deficit",
        "mean_cross_prefix_margin",
    }
    if (
        baseline.get("passed") is not True
        or baseline.get("training_scenes_only") is not True
        or baseline.get("validation_qa_loaded") is not False
        or baseline.get("recomputed_before_optimizer_step_1") is not True
        or dict(expected) != dict(contract.update_zero_expected)
        or set(observed) != set(contract.update_zero_expected) - {"floating_absolute_tolerance"}
        or any(
            abs(float(observed[key]) - float(contract.update_zero_expected[key])) > tolerance
            if key in floating
            else observed[key] != contract.update_zero_expected[key]
            for key in observed
        )
    ):
        raise ValueError("V41 update-zero behavioral baseline changed")
    if stage.get("source_audit") != dict(source_audit):
        raise ValueError("V41 exact source audit changed")
    row0 = _row_at(history, 0)
    source_pairs = _mapping(stage.get("source_pair_metrics"), "V41 source pair metrics")
    source_diagnostics = stage.get("source_per_unit_nll_diagnostics")
    if not isinstance(source_diagnostics, list):
        raise TypeError("V41 source per-unit NLL diagnostics are absent")
    validate_per_unit_nll_diagnostics(source_diagnostics, source_pairs)
    if (
        row0.get("source_pair_metrics") != source_pairs
        or row0.get("per_unit_nll_diagnostics") != source_diagnostics
        or row0.get("source_broad_train_nll") != stage.get("source_broad_train_nll")
        or row0.get("source_train_greedy_metrics") != stage.get(
            "source_train_greedy_metrics"
        )
        or row0.get("training_residual_diagnostics") != stage.get(
            "source_residual_diagnostics"
        )
        or row0.get("update_zero_attestation") != attestation
        or row0.get("query_bank_state_sha256") != contract.query_source_state_sha256
        or row0.get("frozen_excluding_query_state_sha256") != contract.frozen_state_sha256
        or row0.get("scene_prefix_and_residual_exact") is not True
        or row0.get("saved_checkpoint") is not True
    ):
        raise ValueError("V41 update-zero history/source evidence changed")
    greedy = _mapping(stage.get("source_train_greedy_metrics"), "V41 source greedy")
    if (
        int(greedy.get("broad_exact_correct", -1))
        != int(contract.update_zero_expected["broad_greedy_exact_correct"])
        or int(greedy.get("broad_row_count", -1))
        != int(contract.update_zero_expected["broad_greedy_exact_total"])
    ):
        raise ValueError("V41 source broad-greedy baseline changed")
    return {
        "source_audit": dict(source_audit),
        "update_zero_attestation": dict(attestation),
        "source_pair_metrics": dict(source_pairs),
        "source_per_unit_nll_diagnostics": list(source_diagnostics),
        "source_broad_train_nll": stage.get("source_broad_train_nll"),
        "source_train_greedy_metrics": dict(greedy),
        "source_residual_diagnostics": dict(
            _mapping(stage.get("source_residual_diagnostics"), "V41 source residual")
        ),
    }


def validate_v41_checkpoint_envelope(
    config: Mapping[str, Any],
    checkpoint_root: Path,
    contract: V41Contract,
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    """Authenticate the full V41 run before validation construction is legal."""

    checkpoints = _checkpoint_paths_or_raise(checkpoint_root, contract)
    terminal = require_v39_terminal_gate(config)
    hybrid, _source_metadata, raw_source_audit = require_exact_v41_sources(config)
    source_audit = {
        **raw_source_audit,
        "loader_transition": {
            "construction_used_v30_compatible_copy": True,
            "construction_copy_serialized_to_metadata": False,
            "bank_names_bit_exact": True,
            "target_paths_bit_exact": True,
            "state_hashes_bit_exact": True,
            "v41_trainable_bank": None,
            "v41_manually_trainable_adapter": (
                "extension_v28_stage_b_query.adapters.1.lora_b"
            ),
            "v41_frozen_v23_bank": _V23_BANK,
            "v41_trainable_parameter_count": 16_384,
        },
        "source_optimizer_files_opened": False,
        "source_optimizer_states_loaded": False,
    }
    expected_config_hash = config_hash(dict(config))
    provenance = _training_provenance(config)
    prior_history: list[Mapping[str, Any]] = []
    update0: Mapping[str, torch.Tensor] | None = None
    accepted_gates: dict[int, Mapping[str, Any]] = {}
    common_schedule: Mapping[str, Any] | None = None
    common_prefix_replay: Mapping[str, Any] | None = None
    common_update_zero: Mapping[str, Any] | None = None
    audits: list[dict[str, Any]] = []

    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        metadata = _metadata(checkpoint)
        stage = _mapping(metadata.get("v41_projected_gradient"), "V41 stage")
        if metadata.get("optimizer_step") != step or stage.get("optimizer_step") != step:
            raise ValueError(f"V41 optimizer-step metadata changed: {checkpoint.name}")
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V41 config hash changed: {checkpoint.name}")
        terminal_pin = {"path": terminal["path"], "sha256": terminal["sha256"]}
        if (
            stage.get("conditional_v39_terminal_gate") != terminal_pin
            or stage.get("conditional_authorization") != terminal["authorization"]
        ):
            raise ValueError(f"V41 terminal authorization changed: {checkpoint.name}")
        expected_static = {
            "source_checkpoint": str(contract.source_checkpoint),
            "source_v38_u0_tensor_state_sha256": contract.source_tensor_state_sha256,
            "update_zero_tensor_state_sha256": contract.hybrid_tensor_state_sha256,
            "hybrid_v23_state_sha256": contract.hybrid_v23_state_sha256,
            "source_target_lora_b_state_sha256": contract.query_source_state_sha256,
            "source_block_core_state_sha256": contract.core_state_sha256,
            "source_optimizer_states_loaded": False,
            "source_optimizer_files_opened": False,
            "fresh_direction_preserving_sgd": True,
        }
        if any(stage.get(key) != value for key, value in expected_static.items()):
            raise ValueError(f"V41 source or hybrid provenance changed: {checkpoint.name}")
        _validate_train_only_boundary(stage, checkpoint.name, provenance)
        _validate_prefix_replay(stage, checkpoint.name, provenance)
        prefix_replay = _mapping(stage["prefix_replay_attestation"], "V41 prefix replay")
        if common_prefix_replay is None:
            common_prefix_replay = prefix_replay
        elif dict(prefix_replay) != dict(common_prefix_replay):
            raise ValueError("V41 prefix replay changed across checkpoints")
        schedule = _mapping(stage.get("schedule"), "V41 schedule")
        if (
            schedule.get("schedule_sha256") != contract.schedule_sha256
            or schedule.get("pair_schedule_sha256") != contract.pair_schedule_sha256
            or schedule.get("optimizer_step_count") != 41
            or schedule.get("true_optimizer_step_per_schedule_row") is not True
            or schedule.get("one_unchanged_broad_row_per_update") is not True
            or schedule.get("pair_units_atomic") is not True
            or schedule.get("pair_unit_count") != 25
            or schedule.get("saved_optimizer_steps") != list(_EXPECTED_STEPS)
            or schedule.get("per_unit_nll_diagnostic_steps")
            != sorted(_DIAGNOSTIC_STEPS)
            or schedule.get("questions_or_answers_serialized_to_runtime") is not False
        ):
            raise ValueError(f"V41 schedule proof changed: {checkpoint.name}")
        if common_schedule is None:
            common_schedule = schedule
        elif dict(schedule) != dict(common_schedule):
            raise ValueError("V41 schedule metadata changed across checkpoints")

        raw_history = metadata.get("history")
        if not isinstance(raw_history, list) or len(raw_history) != step + 1:
            raise ValueError(f"V41 history is incomplete: {checkpoint.name}")
        history = [_mapping(row, "V41 history row") for row in raw_history]
        if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
            raise ValueError(f"V41 history is not one row per true update: {checkpoint.name}")
        if prior_history and history[: len(prior_history)] != prior_history:
            raise ValueError("V41 rewrote prior optimizer history in a later checkpoint")
        if any(
            row.get("validation_qa_loaded") is not False
            or row.get("oracle_environment_files_loaded") is not False
            for row in history
        ):
            raise ValueError(f"V41 history crossed its train-only boundary: {checkpoint.name}")
        for row in history[1:]:
            guard = _mapping(row.get("component_gradient_guard"), "V41 gradient guard")
            clip = _mapping(
                row.get("clip_direction_attestation"), "V41 clip attestation"
            )
            if (
                guard.get("gradient_method")
                != "torch.autograd.grad_separate_components"
                or guard.get("raw_guard_passed") is not True
                or clip.get("clip_kind") != "single_global_scalar_l2_clip"
                or clip.get("scalar_clip_direction_preserved") is not True
                or row.get("frozen_excluding_b_hash_before")
                != contract.frozen_state_sha256
                or row.get("frozen_excluding_b_hash_after")
                != contract.frozen_state_sha256
                or row.get("target_hash_after") != row.get("query_bank_state_sha256")
                or row.get("target_hash_before") == row.get("target_hash_after")
            ):
                raise ValueError(
                    f"V41 per-microstep direction/frozen-state proof changed: {checkpoint.name}"
                )
        prior_history = history
        if history[-1].get("saved_checkpoint") is not True:
            raise ValueError(f"V41 saved arm lacks its saved-row proof: {checkpoint.name}")
        update_zero_evidence = _validate_update_zero_evidence(
            stage, history, contract, source_audit
        )
        if common_update_zero is None:
            common_update_zero = update_zero_evidence
        elif dict(update_zero_evidence) != dict(common_update_zero):
            raise ValueError("V41 deterministic update-zero evidence changed across arms")
        if step in _DIAGNOSTIC_STEPS:
            row = _row_at(history, step)
            diagnostics = row.get("per_unit_nll_diagnostics")
            pairs = row.get("source_pair_metrics" if step == 0 else "training_pair_metrics")
            if not isinstance(diagnostics, list) or not isinstance(pairs, Mapping):
                raise ValueError(
                    f"V41 update-{step} lacks all 25 persisted per-unit NLL diagnostics"
                )
            validate_per_unit_nll_diagnostics(diagnostics, pairs)

        runtime = json.loads((checkpoint / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V41 runtime metadata is not freshly sanitized: {checkpoint.name}")
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        surface = _validate_surface(metadata, tensors, contract)
        if update0 is None:
            update0 = tensors
            if (
                set(tensors) != set(hybrid)
                or any(not torch.equal(tensors[name], hybrid[name]) for name in tensors)
                or tensor_state_sha256(tensors) != contract.hybrid_tensor_state_sha256
            ):
                raise ValueError("V41 update zero is not exact authenticated V38 update zero")
            if surface["query_bank_state_sha256"] != contract.query_source_state_sha256:
                raise ValueError("V41 update zero changed inherited layer-14 LoRA-B")
        else:
            if set(tensors) != set(update0):
                raise ValueError(f"V41 tensor inventory changed: {checkpoint.name}")
            if any(
                tuple(tensors[name].shape) != tuple(update0[name].shape)
                or tensors[name].dtype != update0[name].dtype
                for name in tensors
            ):
                raise ValueError(f"V41 tensor shape/dtype changed: {checkpoint.name}")
            changed = {name for name in tensors if not torch.equal(tensors[name], update0[name])}
            if not changed or not changed.issubset(_QUERY_PARAMETER_NAME_SET):
                raise ValueError(f"V41 changed a frozen or no target tensor: {checkpoint.name}")
            if surface["query_bank_state_sha256"] == contract.query_source_state_sha256:
                raise ValueError(f"V41 learned layer-14 LoRA-B did not transition: {checkpoint.name}")

        optimizer_audit = None
        if step:
            optimizer_audit = optimizer_step_audit(
                checkpoint, expected_step=step, tensors=tensors
            )
        gate8, gate16, gate41 = replay_v41_gates(metadata, contract)
        replayed = {8: gate8, 16: gate16, 41: gate41}
        for gate_step, gate in replayed.items():
            if step < gate_step:
                if gate is not None:
                    raise ValueError(f"V41 arm persisted a future update-{gate_step} gate")
                continue
            if (
                gate is None
                or gate.get("passed") is not True
                or gate.get("training_scenes_only") is not True
                or gate.get("validation_qa_loaded") is not False
            ):
                raise ValueError(f"V41 lacks a passed update-{gate_step} train-only gate")
            prior = accepted_gates.get(gate_step)
            if prior is None:
                accepted_gates[gate_step] = gate
            elif dict(gate) != dict(prior):
                raise ValueError(f"V41 update-{gate_step} gate changed across later arms")
        audits.append(
            {
                "checkpoint": str(checkpoint),
                "optimizer_step": step,
                "tensor_and_buffer_inventory_inspected": True,
                "runtime_metadata_inspected": True,
                "optimizer_state": optimizer_audit,
                **surface,
            }
        )

    if set(accepted_gates) != {8, 16, 41}:
        raise ValueError("V41 completed envelope lacks all three passed train-only gates")
    if (
        source_audit.get("source_optimizer_file_opened") is not False
        or source_audit.get("source_optimizer_state_loaded") is not False
    ):
        raise ValueError("V41 source audit crossed the no-source-optimizer boundary")
    return checkpoints, audits


class V41ArmEvaluator(Protocol):
    validation_scene_ids: tuple[str, ...]
    cache_audit: Mapping[str, Any]

    def install(
        self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False
    ) -> None: ...

    def evaluate_teacher(self) -> V35TeacherEvidence: ...

    def evaluate_greedy(self) -> V35GreedyEvidence: ...

    def evaluate_aggregate_exact(self) -> tuple[int, int]: ...

    def attest_prefix_invariance(self) -> Mapping[str, Any]: ...


class _V41RuntimeEvaluator(_V35RuntimeEvaluator):
    """One-Gemma evaluator with exact V41 V23/query/core installation."""

    def __init__(
        self,
        config: dict[str, Any],
        control_config: dict[str, Any],
        checkpoint: Path,
        requirements: SelectionRequirements,
    ) -> None:
        actual = dict(config)
        construction = v41_loader_config(actual)
        super().__init__(construction, control_config, checkpoint, requirements)
        self.loader_transition = retag_bundle_for_v41(self.bundle, actual)
        self.config = actual
        collection = self.bundle.lora_installation
        if collection is None:
            raise RuntimeError("V41 evaluator requires installed LoRA banks")
        self._v41_target_adapter = collection.bank(_QUERY_BANK).installation.adapters[1]
        self._v41_v23 = collection.bank(_V23_BANK).installation.state_module
        self._v41_contract = v41_contract(actual)
        self.bundle.language.model.requires_grad_(False).eval()
        for module in self.bundle.checkpoint_modules.values():
            module.requires_grad_(False).eval()

    def install(
        self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False
    ) -> None:
        super().install(tensors, approved_v29=approved_v29)
        query = _query_state(tensors)
        v23 = _prefixed_state(tensors, _V23_PREFIX)
        core = _prefixed_state(tensors, _CORE_PREFIX)
        target_installation = install_v41_target_b(self._v41_target_adapter, query)
        self._v41_v23.load_state_dict(v23, strict=True)
        self._v41_target_adapter.requires_grad_(False).eval()
        self._v41_v23.requires_grad_(False).eval()
        if (
            tensor_state_sha256({"lora_b": self._v41_target_adapter.lora_b})
            != tensor_state_sha256(query)
            or target_installation["lora_a_before_sha256"] != _SOURCE_LOCAL_A_SHA256
            or target_installation["lora_a_after_sha256"] != _SOURCE_LOCAL_A_SHA256
            or tensor_state_sha256(self._v41_v23.state_dict()) != tensor_state_sha256(v23)
            or tensor_state_sha256(self.block_cross_residual.state_dict())
            != tensor_state_sha256(core)
            or module_collection_state_sha256(self.bundle.checkpoint_modules)
            != tensor_state_sha256(tensors)
        ):
            raise RuntimeError("V41 evaluator did not install the supplied tensor envelope")
        if not approved_v29 and (
            tensor_state_sha256(v23) != self._v41_contract.hybrid_v23_state_sha256
            or tensor_state_sha256(core) != self._v41_contract.core_state_sha256
        ):
            raise RuntimeError("V41 evaluator changed its frozen V23/core state")
        if any(
            parameter.requires_grad
            for module in self.bundle.checkpoint_modules.values()
            for parameter in module.parameters()
        ):
            raise RuntimeError("V41 selector evaluation left a checkpoint tensor trainable")

    def evaluate_teacher(self) -> V35TeacherEvidence:
        evidence = super().evaluate_teacher()
        prefix = dict(evidence.prefix_diagnostics)
        prefix["tensor"] = "composed_v41_question_independent_continuous_scene_prefix"
        return V35TeacherEvidence(
            validation_answer_token_nll=evidence.validation_answer_token_nll,
            pair_margins=evidence.pair_margins,
            family_teacher=evidence.family_teacher,
            prefix_diagnostics=prefix,
            color_full_vocab_sides=evidence.color_full_vocab_sides,
            mirror_full_vocab_sides=evidence.mirror_full_vocab_sides,
            negative_sides=evidence.negative_sides,
            prefix_sha256_by_scene=evidence.prefix_sha256_by_scene,
        )


def _validate_exact_development_evidence(
    teacher: V35TeacherEvidence, greedy: V35GreedyEvidence
) -> None:
    if len(teacher.pair_margins.unit_keys) != 12 or len(teacher.pair_margins.margins) != 12:
        raise ValueError("V41 teacher evidence must score exactly 12 changed validation units")
    if (
        greedy.generation.changed_unit_count != 12
        or greedy.generation.changed_row_count != 24
        or set(greedy.complete_by_family) != set(_PRIORITY_FAMILIES)
        or set(greedy.prediction_changed_by_family) != set(_PRIORITY_FAMILIES)
    ):
        raise ValueError("V41 greedy evidence must score exactly 12 changed validation units")


def _development_checks(
    *,
    teacher: V35TeacherEvidence,
    greedy: V35GreedyEvidence,
    source: V35TeacherEvidence,
    approved: V35TeacherEvidence,
    approved_greedy: V35GreedyEvidence,
) -> tuple[dict[str, bool], list[tuple[str, str]]]:
    _validate_exact_development_evidence(teacher, greedy)
    new_negatives = sorted(teacher.negative_sides - approved.negative_sides)
    checks = {
        "exact_12_changed_teacher_units_scored": len(teacher.pair_margins.unit_keys) == 12,
        "exact_12_changed_greedy_units_scored": greedy.generation.changed_unit_count == 12,
        "greedy_complete_units_at_least_6_of_12": (
            greedy.generation.exact_complete_units_correct >= 6
        ),
        "each_priority_family_greedy_complete": all(
            int(greedy.complete_by_family[family]) >= 1 for family in _PRIORITY_FAMILIES
        ),
        "hybrid_u0_teacher_complete_units_improved_by_at_least_1": (
            teacher.pair_margins.passed_units >= source.pair_margins.passed_units + 1
        ),
        "hybrid_u0_validation_answer_nll_no_worse": (
            teacher.validation_answer_token_nll
            <= source.validation_answer_token_nll + _SOURCE_NLL_TOLERANCE
        ),
        "approved_v29_color_12_sides_retained": teacher.color_full_vocab_sides >= 12,
        "approved_v29_mirror_10_sides_retained": teacher.mirror_full_vocab_sides >= 10,
        "approved_v29_controls_no_new_negatives": not new_negatives,
        "broad_retention_vs_approved_v29": (
            greedy.generation.broad_exact_accuracy
            >= approved_greedy.generation.broad_exact_accuracy
        ),
    }
    return checks, new_negatives


def select_v41(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], V41ArmEvaluator
    ] = _V41RuntimeEvaluator,
) -> dict[str, Any]:
    config = load_config(config_path)
    contract = v41_contract(config)
    # The full filesystem and train-gate envelope is authenticated first.  No
    # validation QA/map/model object may be created above this boundary.
    checkpoints, envelope_audits = validate_v41_checkpoint_envelope(
        config, checkpoint_root, contract
    )

    requirements = _selection_requirements(config)
    control = _retention_control_config(config)
    source_v29 = _source_v29_evidence(_metadata(checkpoints[0]))
    _validate_source_against_config(source_v29, config)
    evaluator = evaluator_factory(config, control, checkpoints[0], requirements)
    expected_validation = tuple(f"scene_{index:06d}" for index in range(19, 25))
    if tuple(evaluator.validation_scene_ids) != expected_validation:
        raise ValueError("V41 evaluator must remain exactly on validation scenes 19--24")

    approved_tensors = load_file(
        Path(str(source_v29["checkpoint"])) / "adapter.safetensors", device="cpu"
    )
    evaluator.install(_approved_v29_envelope(approved_tensors, config=config), approved_v29=True)
    approved_teacher = evaluator.evaluate_teacher()
    approved_greedy = evaluator.evaluate_greedy()
    _validate_exact_development_evidence(approved_teacher, approved_greedy)
    approved_aggregate = evaluator.evaluate_aggregate_exact()

    update0 = load_file(checkpoints[0] / "adapter.safetensors", device="cpu")
    evaluator.install(update0)
    source_teacher = evaluator.evaluate_teacher()
    arms: list[dict[str, Any]] = []
    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        evaluator.install(tensors)
        teacher = evaluator.evaluate_teacher()
        if len(teacher.pair_margins.unit_keys) != 12:
            raise ValueError("V41 teacher scorer omitted a changed validation unit")
        greedy = evaluator.evaluate_greedy() if step in _GREEDY_STEPS else None
        arm: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "optimizer_step": step,
            "validation_answer_token_nll": teacher.validation_answer_token_nll,
            "validation_pair_passed_units": teacher.pair_margins.passed_units,
            "validation_pair_passed_sides": teacher.pair_margins.passed_sides,
            "validation_pair_mean_margin": teacher.pair_margins.mean_margin,
            "validation_pair_minimum_margin": teacher.pair_margins.minimum_margin,
            "hybrid_u0_teacher_complete_units": source_teacher.pair_margins.passed_units,
            "teacher_complete_units_delta_vs_hybrid_u0": (
                teacher.pair_margins.passed_units - source_teacher.pair_margins.passed_units
            ),
            "hybrid_u0_validation_answer_token_nll": (
                source_teacher.validation_answer_token_nll
            ),
            "validation_answer_nll_improvement_vs_hybrid_u0": (
                source_teacher.validation_answer_token_nll
                - teacher.validation_answer_token_nll
            ),
            "validation_family_teacher": dict(teacher.family_teacher),
            "color_full_vocab_sides": teacher.color_full_vocab_sides,
            "mirror_full_vocab_sides": teacher.mirror_full_vocab_sides,
            "new_negative_sides_vs_approved_v29": sorted(
                teacher.negative_sides - approved_teacher.negative_sides
            ),
            "prefix_sha256_by_validation_scene": dict(
                sorted(teacher.prefix_sha256_by_scene.items())
            ),
            "greedy_screen_designated": step in _GREEDY_STEPS,
            "greedy_changed_row_count": None,
            "greedy_changed_unit_count": None,
            "greedy_exact_complete_units_correct": None,
            "greedy_exact_correct_sides": None,
            "greedy_prediction_changed_units": None,
            "greedy_complete_units_by_family": None,
            "greedy_prediction_changed_by_family": None,
            "broad_retention_exact_accuracy": None,
            "checks": {},
            "eligible": False,
        }
        if greedy is not None:
            checks, negatives = _development_checks(
                teacher=teacher,
                greedy=greedy,
                source=source_teacher,
                approved=approved_teacher,
                approved_greedy=approved_greedy,
            )
            arm.update(
                {
                    "new_negative_sides_vs_approved_v29": negatives,
                    "greedy_changed_row_count": greedy.generation.changed_row_count,
                    "greedy_changed_unit_count": greedy.generation.changed_unit_count,
                    "greedy_exact_complete_units_correct": (
                        greedy.generation.exact_complete_units_correct
                    ),
                    "greedy_exact_correct_sides": greedy.generation.exact_correct_sides,
                    "greedy_prediction_changed_units": (
                        greedy.generation.prediction_changed_units
                    ),
                    "greedy_complete_units_by_family": dict(greedy.complete_by_family),
                    "greedy_prediction_changed_by_family": dict(
                        greedy.prediction_changed_by_family
                    ),
                    "broad_retention_exact_accuracy": (
                        greedy.generation.broad_exact_accuracy
                    ),
                    "checks": checks,
                    "eligible": all(checks.values()),
                }
            )
        arms.append(arm)

    candidates = [arm for arm in arms if arm["eligible"]]
    selected = min(
        candidates,
        key=lambda arm: (
            -int(arm["greedy_exact_complete_units_correct"]),
            -sum(
                int(value) > 0
                for value in _mapping(
                    arm["greedy_complete_units_by_family"], "V41 greedy families"
                ).values()
            ),
            -int(arm["validation_pair_passed_units"]),
            float(arm["validation_answer_token_nll"]),
            int(arm["optimizer_step"]),
        ),
        default=None,
    )
    selected_aggregate: tuple[int, int] | None = None
    prefix_attestation: Mapping[str, Any] | None = None
    if selected is not None:
        selected_path = checkpoint_root / f"update_{int(selected['optimizer_step']):03d}"
        evaluator.install(load_file(selected_path / "adapter.safetensors", device="cpu"))
        initial = evaluator.attest_prefix_invariance()
        if initial.get("passed") is not True:
            raise ValueError("V41 selected prefix failed its pre-question invariance replay")
        selected_aggregate = evaluator.evaluate_aggregate_exact()
        prefix_attestation = evaluator.attest_prefix_invariance()

    promotion = _promotion(
        selected,
        approved_v29_aggregate=approved_aggregate,
        selected_aggregate=selected_aggregate,
        prefix_attestation=prefix_attestation,
    )
    split = v31_contract(v41_loader_config(config))
    terminal = require_v39_terminal_gate(config)
    return {
        "schema_version": 1,
        "artifact": "v41_projected_gradient_development_selection",
        "development_validation_model_selection_only": True,
        "training_completed_through_update41_before_validation_loaded": True,
        "validation_used_for_training_continuation": False,
        "final_test_scenes_touched": False,
        "final_evaluation_ran": False,
        "runtime_promotion_written": False,
        "deferred_final_scene_ids": list(split.deferred_final_scene_ids),
        "oracle_loaded": False,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "all_scene_prefixes_built_before_questions": True,
        "gemma_base_frozen": True,
        "only_existing_v28_layer14_lora_b_trained": True,
        "v23_k_only_hybrid_and_v36_block_core_frozen_exact": True,
        "exact_trainable_parameter_count": 16_384,
        "exact_trainable_tensor_count": 1,
        "model_load_count": 1,
        "source_v29": source_v29,
        "retention_and_aggregate_baseline": "approved_v29",
        "improvement_baseline": "exact_v41_k_only_hybrid_update_000",
        "v39_terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
        "train_scene_ids": list(split.train_scene_ids),
        "validation_scene_ids": list(evaluator.validation_scene_ids),
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "all_saved_arms_inspected": True,
        "checkpoint_envelope_audits": envelope_audits,
        "optimizer_state_steps_verified": list(contract.saved_optimizer_steps[1:]),
        "train_only_update8_gate_passed": True,
        "train_only_update16_gate_passed": True,
        "train_only_update41_gate_passed": True,
        "complete_question_independent_block_cache": dict(evaluator.cache_audit),
        "teacher_scored_steps": list(contract.saved_optimizer_steps),
        "greedy_screen_steps": sorted(_GREEDY_STEPS),
        "validation_changed_unit_count": 12,
        "development_requirements": {
            "greedy_complete_units_minimum": 6,
            "greedy_validation_unit_count": 12,
            "one_greedy_complete_per_priority_family": True,
            "hybrid_u0_teacher_complete_units_minimum_delta": 1,
            "hybrid_u0_validation_answer_nll_maximum_regression": _SOURCE_NLL_TOLERANCE,
            "approved_v29_color_sides_minimum": 12,
            "approved_v29_mirror_sides_minimum": 10,
            "approved_v29_no_new_control_negatives": True,
            "approved_v29_broad_accuracy_no_regression": True,
            "approved_v29_aggregate_accuracy_no_regression": True,
        },
        "arms": arms,
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "selected_update": None if selected is None else selected["optimizer_step"],
        "selected_optimizer_step": None if selected is None else selected["optimizer_step"],
        "development_selection_passed": selected is not None,
        "chat_promotion": promotion,
        "chat_promotion_eligible": promotion["eligible"],
        "development_progress_is_not_runtime_promotion": True,
        "passed": selected is not None,
    }


def run_v41_selector(config: Path, checkpoint_root: Path, output: Path) -> dict[str, Any]:
    """Write only after selection returns; all refusal paths leave no report."""

    report = select_v41(config, checkpoint_root)
    _atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_v41_selector(args.config, args.checkpoint_root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V41ArmEvaluator",
    "run_v41_selector",
    "select_v41",
    "validate_v41_checkpoint_envelope",
]
