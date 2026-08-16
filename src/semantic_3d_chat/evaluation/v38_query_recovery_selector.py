"""Select a fully completed V38 query-recovery development checkpoint.

This is a deliberately fail-closed post-training process.  The exact V38
checkpoint directory envelope, terminal train-only gate, tensor transition,
runtime metadata, and fresh AdamW state are authenticated before the first
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

from semantic_3d_chat.config import artifact_root, config_hash, load_config
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
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.train_block_cross_v35 import validate_v35_cache_audit
from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract
from semantic_3d_chat.training.train_query_recovery_v38 import (
    OPTIMIZER_AUDIT_FILENAME,
    V38Contract,
    optimizer_step_audit,
    replay_v38_gates,
    require_exact_v38_sources,
    require_v37_terminal_gate,
    retag_bundle_for_v38,
    v38_contract,
    v38_loader_config,
    validate_per_unit_nll_diagnostics,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_query_recovery_v38.yaml")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v38_diverse28_query_recovery"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v38_query_recovery_selection.json")

_QUERY_BANK = "extension_v30_joint_pair_query"
_QUERY_PREFIX = f"lora_banks.{_QUERY_BANK}."
_V23_BANK = "extension_v23_shared_kv"
_V23_PREFIX = f"lora_banks.{_V23_BANK}."
_CORE_PREFIX = "block_cross_residual."
_QUERY_PARAMETER_NAMES = tuple(
    f"{_QUERY_PREFIX}adapters.{index}.{side}"
    for index in range(4)
    for side in ("lora_a", "lora_b")
)
_QUERY_PARAMETER_NAME_SET = frozenset(_QUERY_PARAMETER_NAMES)
_QUERY_SHAPES = (
    (8, 1536),
    (2048, 8),
    (8, 1536),
    (4096, 8),
    (8, 1536),
    (2048, 8),
    (8, 1536),
    (2048, 8),
)
_QUERY_MODULES = tuple(
    f"model.language_model.layers.{index}.self_attn.q_proj"
    for index in range(18, 22)
)
_EXPECTED_STEPS = (0, 8, 16, 24, 32, 40, 41)
_DIAGNOSTIC_STEPS = frozenset({0, 8, 16, 41})
_GREEDY_STEPS = frozenset({16, 24, 32, 40, 41})
_PRIORITY_FAMILIES = ("book_support", "mirror_lr", "picture_support")
_SOURCE_NLL_TOLERANCE = 1e-6


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
    state = _prefixed_state(tensors, _QUERY_PREFIX)
    observed = tuple(f"{_QUERY_PREFIX}{name}" for name in state)
    if observed != _QUERY_PARAMETER_NAMES:
        raise ValueError("V38 query-bank tensor order or inventory changed")
    if tuple(tuple(value.shape) for value in state.values()) != _QUERY_SHAPES:
        raise ValueError("V38 query-bank tensor shapes changed")
    return state


def _frozen_excluding_query(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value for name, value in tensors.items() if not name.startswith(_QUERY_PREFIX)
    }


def _checkpoint_paths_or_raise(
    checkpoint_root: Path, contract: V38Contract
) -> tuple[Path, ...]:
    """Resolve the exact completed envelope without opening a data artifact."""

    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError("V38 checkpoint root must be a real directory")
    if tuple(contract.saved_optimizer_steps) != _EXPECTED_STEPS:
        raise ValueError("V38 saved-step contract changed")
    checkpoints = tuple(
        checkpoint_root / f"update_{step:03d}" for step in contract.saved_optimizer_steps
    )
    observed = sorted(path.name for path in checkpoint_root.glob("update_*"))
    expected = [path.name for path in checkpoints]
    if observed != expected:
        raise FileNotFoundError(
            "V38 requires the exact completed update-41 envelope: "
            f"observed={observed} expected={expected}"
        )
    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError(f"V38 arm must be a real directory: {checkpoint}")
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
            raise FileNotFoundError(f"V38 arm is incomplete or aliased: {checkpoint.name}")
        if step == 0 and any(
            (checkpoint / name).exists()
            for name in ("optimizer.pt", OPTIMIZER_AUDIT_FILENAME)
        ):
            raise ValueError("V38 update zero must not persist Adam state")
    return checkpoints


def _training_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute expected train-only paths without opening QA or map files."""

    loader = v38_loader_config(config)
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
        raise ValueError(f"V38 {label} crossed its train-only boundary")
    if stage.get("deferred_final_scene_ids_loaded") != []:
        raise ValueError(f"V38 {label} touched deferred final scenes")
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
        raise ValueError(f"V38 {label} QA provenance is not exact train-only input")
    cache = _mapping(stage.get("scene_cache"), f"{label} scene cache")
    try:
        validate_v35_cache_audit(cache, expected_scene_ids=provenance["train_scene_ids"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V38 {label} map provenance is not exact train-only input") from exc
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
        raise ValueError(f"V38 {label} map provenance is not exact train-only input")


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
        raise ValueError(f"V38 {label} prefix replay is incomplete or non-deterministic")


def _validate_surface(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    contract: V38Contract,
) -> dict[str, Any]:
    stage = _mapping(metadata.get("v38_query_recovery"), "V38 stage")
    surface = _mapping(stage.get("trainable_surface"), "V38 trainable surface")
    expected_surface = {
        "target_bank": _QUERY_BANK,
        "target_module_paths": list(_QUERY_MODULES),
        "target_parameter_names": list(_QUERY_PARAMETER_NAMES),
        "trainable_tensor_count": 8,
        "trainable_parameter_count": 131_072,
        "rank": 8,
        "alpha": 16.0,
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
        raise ValueError("V38 persisted trainable surface changed")
    query = _query_state(tensors)
    frozen = _frozen_excluding_query(tensors)
    query_hash = tensor_state_sha256(query)
    frozen_hash = tensor_state_sha256(frozen)
    v23_hash = tensor_state_sha256(_prefixed_state(tensors, _V23_PREFIX))
    core_hash = tensor_state_sha256(_prefixed_state(tensors, _CORE_PREFIX))
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "V38 LoRA hashes")
    if (
        sum(value.numel() for value in query.values()) != 131_072
        or any(not torch.isfinite(value).all() for value in query.values())
        or query_hash != stage.get("query_bank_state_sha256")
        or query_hash != bank_hashes.get(_QUERY_BANK)
    ):
        raise ValueError("V38 query-bank state or metadata changed")
    if (
        frozen_hash != contract.frozen_state_sha256
        or frozen_hash != stage.get("frozen_excluding_query_state_sha256")
    ):
        raise ValueError("V38 changed a frozen tensor or buffer")
    if (
        v23_hash != contract.hybrid_v23_state_sha256
        or v23_hash != stage.get("hybrid_v23_state_sha256")
        or v23_hash != bank_hashes.get(_V23_BANK)
    ):
        raise ValueError("V38 changed its frozen K-only V23 hybrid")
    if (
        core_hash != contract.core_state_sha256
        or core_hash != stage.get("source_block_core_state_sha256")
        or core_hash != metadata.get("block_cross_residual_state_sha256")
    ):
        raise ValueError("V38 changed its frozen learned block core")
    if metadata.get("lora_trainable_parameter_count") != 131_072:
        raise ValueError("V38 checkpoint advertises the wrong trainable count")
    return {
        "query_bank_state_sha256": query_hash,
        "frozen_excluding_target_state_sha256": frozen_hash,
        "hybrid_v23_state_sha256": v23_hash,
        "learned_block_core_state_sha256": core_hash,
        "authorized_parameter_count": 131_072,
        "authorized_tensor_count": 8,
    }


def _row_at(history: list[Mapping[str, Any]], step: int) -> Mapping[str, Any]:
    if step >= len(history) or history[step].get("optimizer_update") != step:
        raise ValueError(f"V38 history lacks exact optimizer update {step}")
    return history[step]


def _validate_update_zero_evidence(
    stage: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    contract: V38Contract,
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    attestation = _mapping(stage.get("update_zero_attestation"), "V38 update-zero proof")
    required = {
        "exact_v37_update16_primary_source_loaded": True,
        "exact_v36_update16_v_projection_donor_loaded": True,
        "hybrid_tensor_state_sha256": contract.hybrid_tensor_state_sha256,
        "hybrid_v23_state_sha256": contract.hybrid_v23_state_sha256,
        "existing_learned_query_loaded_without_reinitialization": True,
        "query_bank_source_state_sha256": contract.query_source_state_sha256,
        "query_bank_all_b_tensors_nonzero": True,
        "learned_block_core_state_sha256": contract.core_state_sha256,
        "frozen_excluding_query_state_sha256": contract.frozen_state_sha256,
        "source_optimizer_files_opened": False,
        "source_optimizer_states_loaded": False,
        "fresh_adam_state": True,
        "hybrid_behavior_recomputed_before_optimizer": True,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
    }
    if any(attestation.get(key) != value for key, value in required.items()):
        raise ValueError("V38 update-zero hybrid attestation changed")
    baseline = _mapping(attestation.get("behavioral_baseline"), "V38 behavioral baseline")
    observed = _mapping(baseline.get("observed"), "V38 update-zero observed baseline")
    expected = _mapping(baseline.get("expected"), "V38 update-zero expected baseline")
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
        raise ValueError("V38 update-zero behavioral baseline changed")
    if stage.get("source_audit") != dict(source_audit):
        raise ValueError("V38 exact source audit changed")
    row0 = _row_at(history, 0)
    source_pairs = _mapping(stage.get("source_pair_metrics"), "V38 source pair metrics")
    source_diagnostics = stage.get("source_per_unit_nll_diagnostics")
    if not isinstance(source_diagnostics, list):
        raise TypeError("V38 source per-unit NLL diagnostics are absent")
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
        raise ValueError("V38 update-zero history/source evidence changed")
    greedy = _mapping(stage.get("source_train_greedy_metrics"), "V38 source greedy")
    if (
        int(greedy.get("broad_exact_correct", -1))
        != int(contract.update_zero_expected["broad_greedy_exact_correct"])
        or int(greedy.get("broad_row_count", -1))
        != int(contract.update_zero_expected["broad_greedy_exact_total"])
    ):
        raise ValueError("V38 source broad-greedy baseline changed")
    return {
        "source_audit": dict(source_audit),
        "update_zero_attestation": dict(attestation),
        "source_pair_metrics": dict(source_pairs),
        "source_per_unit_nll_diagnostics": list(source_diagnostics),
        "source_broad_train_nll": stage.get("source_broad_train_nll"),
        "source_train_greedy_metrics": dict(greedy),
        "source_residual_diagnostics": dict(
            _mapping(stage.get("source_residual_diagnostics"), "V38 source residual")
        ),
    }


def validate_v38_checkpoint_envelope(
    config: Mapping[str, Any],
    checkpoint_root: Path,
    contract: V38Contract,
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    """Authenticate the full V38 run before validation construction is legal."""

    checkpoints = _checkpoint_paths_or_raise(checkpoint_root, contract)
    terminal = require_v37_terminal_gate(config)
    hybrid, _source_metadata, raw_source_audit = require_exact_v38_sources(config)
    source_audit = {
        **raw_source_audit,
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
        stage = _mapping(metadata.get("v38_query_recovery"), "V38 stage")
        if metadata.get("optimizer_step") != step or stage.get("optimizer_step") != step:
            raise ValueError(f"V38 optimizer-step metadata changed: {checkpoint.name}")
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V38 config hash changed: {checkpoint.name}")
        terminal_pin = {"path": terminal["path"], "sha256": terminal["sha256"]}
        if (
            stage.get("conditional_v37_terminal_gate") != terminal_pin
            or stage.get("conditional_authorization") != terminal["authorization"]
        ):
            raise ValueError(f"V38 terminal authorization changed: {checkpoint.name}")
        expected_static = {
            "source_checkpoint": str(contract.source_checkpoint),
            "rollback_checkpoint": str(contract.rollback_checkpoint),
            "source_v37_tensor_state_sha256": contract.source_tensor_state_sha256,
            "rollback_v36_tensor_state_sha256": contract.rollback_tensor_state_sha256,
            "update_zero_hybrid_tensor_state_sha256": contract.hybrid_tensor_state_sha256,
            "hybrid_v23_state_sha256": contract.hybrid_v23_state_sha256,
            "source_query_bank_state_sha256": contract.query_source_state_sha256,
            "source_block_core_state_sha256": contract.core_state_sha256,
            "source_optimizer_states_loaded": False,
            "source_optimizer_files_opened": False,
            "fresh_adam": True,
        }
        if any(stage.get(key) != value for key, value in expected_static.items()):
            raise ValueError(f"V38 source or hybrid provenance changed: {checkpoint.name}")
        _validate_train_only_boundary(stage, checkpoint.name, provenance)
        _validate_prefix_replay(stage, checkpoint.name, provenance)
        prefix_replay = _mapping(stage["prefix_replay_attestation"], "V38 prefix replay")
        if common_prefix_replay is None:
            common_prefix_replay = prefix_replay
        elif dict(prefix_replay) != dict(common_prefix_replay):
            raise ValueError("V38 prefix replay changed across checkpoints")
        schedule = _mapping(stage.get("schedule"), "V38 schedule")
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
            raise ValueError(f"V38 schedule proof changed: {checkpoint.name}")
        if common_schedule is None:
            common_schedule = schedule
        elif dict(schedule) != dict(common_schedule):
            raise ValueError("V38 schedule metadata changed across checkpoints")

        raw_history = metadata.get("history")
        if not isinstance(raw_history, list) or len(raw_history) != step + 1:
            raise ValueError(f"V38 history is incomplete: {checkpoint.name}")
        history = [_mapping(row, "V38 history row") for row in raw_history]
        if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
            raise ValueError(f"V38 history is not one row per true update: {checkpoint.name}")
        if prior_history and history[: len(prior_history)] != prior_history:
            raise ValueError("V38 rewrote prior optimizer history in a later checkpoint")
        if any(
            row.get("validation_qa_loaded") is not False
            or row.get("oracle_environment_files_loaded") is not False
            for row in history
        ):
            raise ValueError(f"V38 history crossed its train-only boundary: {checkpoint.name}")
        prior_history = history
        if history[-1].get("saved_checkpoint") is not True:
            raise ValueError(f"V38 saved arm lacks its saved-row proof: {checkpoint.name}")
        update_zero_evidence = _validate_update_zero_evidence(
            stage, history, contract, source_audit
        )
        if common_update_zero is None:
            common_update_zero = update_zero_evidence
        elif dict(update_zero_evidence) != dict(common_update_zero):
            raise ValueError("V38 deterministic update-zero evidence changed across arms")
        if step in _DIAGNOSTIC_STEPS:
            row = _row_at(history, step)
            diagnostics = row.get("per_unit_nll_diagnostics")
            pairs = row.get("source_pair_metrics" if step == 0 else "training_pair_metrics")
            if not isinstance(diagnostics, list) or not isinstance(pairs, Mapping):
                raise ValueError(
                    f"V38 update-{step} lacks all 25 persisted per-unit NLL diagnostics"
                )
            validate_per_unit_nll_diagnostics(diagnostics, pairs)

        runtime = json.loads((checkpoint / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V38 runtime metadata is not freshly sanitized: {checkpoint.name}")
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        surface = _validate_surface(metadata, tensors, contract)
        if update0 is None:
            update0 = tensors
            if (
                set(tensors) != set(hybrid)
                or any(not torch.equal(tensors[name], hybrid[name]) for name in tensors)
                or tensor_state_sha256(tensors) != contract.hybrid_tensor_state_sha256
            ):
                raise ValueError("V38 update zero is not the exact authenticated K-only hybrid")
            if surface["query_bank_state_sha256"] != contract.query_source_state_sha256:
                raise ValueError("V38 update zero changed the inherited learned query bank")
        else:
            if set(tensors) != set(update0):
                raise ValueError(f"V38 tensor inventory changed: {checkpoint.name}")
            if any(
                tuple(tensors[name].shape) != tuple(update0[name].shape)
                or tensors[name].dtype != update0[name].dtype
                for name in tensors
            ):
                raise ValueError(f"V38 tensor shape/dtype changed: {checkpoint.name}")
            changed = {name for name in tensors if not torch.equal(tensors[name], update0[name])}
            if not changed or not changed.issubset(_QUERY_PARAMETER_NAME_SET):
                raise ValueError(f"V38 changed a frozen or no target tensor: {checkpoint.name}")
            if surface["query_bank_state_sha256"] == contract.query_source_state_sha256:
                raise ValueError(f"V38 learned query bank did not transition: {checkpoint.name}")

        optimizer_audit = None
        if step:
            optimizer_audit = optimizer_step_audit(
                checkpoint, expected_step=step, tensors=tensors
            )
        gate8, gate16, gate41 = replay_v38_gates(metadata, contract)
        replayed = {8: gate8, 16: gate16, 41: gate41}
        for gate_step, gate in replayed.items():
            if step < gate_step:
                if gate is not None:
                    raise ValueError(f"V38 arm persisted a future update-{gate_step} gate")
                continue
            if (
                gate is None
                or gate.get("passed") is not True
                or gate.get("training_scenes_only") is not True
                or gate.get("validation_qa_loaded") is not False
            ):
                raise ValueError(f"V38 lacks a passed update-{gate_step} train-only gate")
            prior = accepted_gates.get(gate_step)
            if prior is None:
                accepted_gates[gate_step] = gate
            elif dict(gate) != dict(prior):
                raise ValueError(f"V38 update-{gate_step} gate changed across later arms")
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
        raise ValueError("V38 completed envelope lacks all three passed train-only gates")
    if source_audit.get("v37_optimizer_file_opened") is not False or source_audit.get(
        "v36_optimizer_file_opened"
    ) is not False:
        raise ValueError("V38 source assembly opened a forbidden source optimizer")
    return checkpoints, audits


class V38ArmEvaluator(Protocol):
    validation_scene_ids: tuple[str, ...]
    cache_audit: Mapping[str, Any]

    def install(
        self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False
    ) -> None: ...

    def evaluate_teacher(self) -> V35TeacherEvidence: ...

    def evaluate_greedy(self) -> V35GreedyEvidence: ...

    def evaluate_aggregate_exact(self) -> tuple[int, int]: ...

    def attest_prefix_invariance(self) -> Mapping[str, Any]: ...


class _V38RuntimeEvaluator(_V35RuntimeEvaluator):
    """One-Gemma evaluator with exact V38 V23/query/core installation."""

    def __init__(
        self,
        config: dict[str, Any],
        control_config: dict[str, Any],
        checkpoint: Path,
        requirements: SelectionRequirements,
    ) -> None:
        actual = dict(config)
        construction = v38_loader_config(actual)
        super().__init__(construction, control_config, checkpoint, requirements)
        self.loader_transition = retag_bundle_for_v38(self.bundle, actual)
        self.config = actual
        collection = self.bundle.lora_installation
        if collection is None:
            raise RuntimeError("V38 evaluator requires installed LoRA banks")
        self._v38_query = collection.bank(_QUERY_BANK).installation.state_module
        self._v38_v23 = collection.bank(_V23_BANK).installation.state_module
        self._v38_contract = v38_contract(actual)
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
        self._v38_query.load_state_dict(query, strict=True)
        self._v38_v23.load_state_dict(v23, strict=True)
        self._v38_query.requires_grad_(False).eval()
        self._v38_v23.requires_grad_(False).eval()
        if (
            tensor_state_sha256(self._v38_query.state_dict()) != tensor_state_sha256(query)
            or tensor_state_sha256(self._v38_v23.state_dict()) != tensor_state_sha256(v23)
            or tensor_state_sha256(self.block_cross_residual.state_dict())
            != tensor_state_sha256(core)
        ):
            raise RuntimeError("V38 evaluator did not install the supplied tensor envelope")
        if not approved_v29 and (
            tensor_state_sha256(v23) != self._v38_contract.hybrid_v23_state_sha256
            or tensor_state_sha256(core) != self._v38_contract.core_state_sha256
        ):
            raise RuntimeError("V38 evaluator changed its frozen V23/core state")
        if any(
            parameter.requires_grad
            for module in self.bundle.checkpoint_modules.values()
            for parameter in module.parameters()
        ):
            raise RuntimeError("V38 selector evaluation left a checkpoint tensor trainable")

    def evaluate_teacher(self) -> V35TeacherEvidence:
        evidence = super().evaluate_teacher()
        prefix = dict(evidence.prefix_diagnostics)
        prefix["tensor"] = "composed_v38_question_independent_continuous_scene_prefix"
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
        raise ValueError("V38 teacher evidence must score exactly 12 changed validation units")
    if (
        greedy.generation.changed_unit_count != 12
        or greedy.generation.changed_row_count != 24
        or set(greedy.complete_by_family) != set(_PRIORITY_FAMILIES)
        or set(greedy.prediction_changed_by_family) != set(_PRIORITY_FAMILIES)
    ):
        raise ValueError("V38 greedy evidence must score exactly 12 changed validation units")


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


def select_v38(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], V38ArmEvaluator
    ] = _V38RuntimeEvaluator,
) -> dict[str, Any]:
    config = load_config(config_path)
    contract = v38_contract(config)
    # The full filesystem and train-gate envelope is authenticated first.  No
    # validation QA/map/model object may be created above this boundary.
    checkpoints, envelope_audits = validate_v38_checkpoint_envelope(
        config, checkpoint_root, contract
    )

    requirements = _selection_requirements(config)
    control = _retention_control_config(config)
    source_v29 = _source_v29_evidence(_metadata(checkpoints[0]))
    _validate_source_against_config(source_v29, config)
    evaluator = evaluator_factory(config, control, checkpoints[0], requirements)
    expected_validation = tuple(f"scene_{index:06d}" for index in range(19, 25))
    if tuple(evaluator.validation_scene_ids) != expected_validation:
        raise ValueError("V38 evaluator must remain exactly on validation scenes 19--24")

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
            raise ValueError("V38 teacher scorer omitted a changed validation unit")
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
                    arm["greedy_complete_units_by_family"], "V38 greedy families"
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
            raise ValueError("V38 selected prefix failed its pre-question invariance replay")
        selected_aggregate = evaluator.evaluate_aggregate_exact()
        prefix_attestation = evaluator.attest_prefix_invariance()

    promotion = _promotion(
        selected,
        approved_v29_aggregate=approved_aggregate,
        selected_aggregate=selected_aggregate,
        prefix_attestation=prefix_attestation,
    )
    split = v31_contract(v38_loader_config(config))
    terminal = require_v37_terminal_gate(config)
    return {
        "schema_version": 1,
        "artifact": "v38_query_recovery_development_selection",
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
        "only_existing_v30_query_lora_trained": True,
        "v23_k_only_hybrid_and_v36_block_core_frozen_exact": True,
        "exact_trainable_parameter_count": 131_072,
        "exact_trainable_tensor_count": 8,
        "model_load_count": 1,
        "source_v29": source_v29,
        "retention_and_aggregate_baseline": "approved_v29",
        "improvement_baseline": "exact_v38_k_only_hybrid_update_000",
        "v37_terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
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


def run_v38_selector(config: Path, checkpoint_root: Path, output: Path) -> dict[str, Any]:
    """Write only after selection returns; all refusal paths leave no report."""

    report = select_v38(config, checkpoint_root)
    _atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_v38_selector(args.config, args.checkpoint_root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V38ArmEvaluator",
    "run_v38_selector",
    "select_v38",
    "validate_v38_checkpoint_envelope",
]
