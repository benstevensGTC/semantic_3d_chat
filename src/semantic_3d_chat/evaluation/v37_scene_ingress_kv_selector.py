"""Independently select a completed V37 shared-K/V development arm.

The validation boundary in this module is deliberate: the complete nine-arm
V37 checkpoint envelope, every persisted tensor, every Adam moment, and all
three train-only continuation gates are authenticated before an evaluator can
be constructed.  Validation is limited to scenes 19--24.  Deferred final
scenes and oracle data are never legal inputs, and this selector writes only a
development report--never a runtime promotion or final-test artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
    _approved_v29_runtime_tensor_envelope,
    _promotion,
    _V35RuntimeEvaluator,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    block_cross_residual_settings,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    OPTIMIZER_AUDIT_FILENAME,
    V37Contract,
    optimizer_step_audit,
    replay_v37_gates,
    require_exact_v36_source,
    require_v36_terminal_gate,
    retag_bundle_for_v37,
    v37_contract,
    v37_loader_config,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_scene_ingress_kv_v37.yaml")
DEFAULT_CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v37_diverse28_scene_ingress_kv")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v37_scene_ingress_kv_selection.json")

_TARGET_BANK = "extension_v23_shared_kv"
_TARGET_PREFIX = f"lora_banks.{_TARGET_BANK}."
_QUERY_BANK = "extension_v30_joint_pair_query"
_QUERY_PREFIX = f"lora_banks.{_QUERY_BANK}."
_CORE_PREFIX = "block_cross_residual."
_TARGET_PARAMETER_NAMES = tuple(
    f"{_TARGET_PREFIX}adapters.{index}.{side}"
    for index in range(4)
    for side in ("lora_a", "lora_b")
)
_TARGET_PARAMETER_NAME_SET = frozenset(_TARGET_PARAMETER_NAMES)
_TARGET_SHAPES = (
    (4, 1536),
    (256, 4),
    (4, 1536),
    (256, 4),
    (4, 1536),
    (512, 4),
    (4, 1536),
    (512, 4),
)
_TARGET_MODULES = (
    "model.language_model.layers.13.self_attn.k_proj",
    "model.language_model.layers.13.self_attn.v_proj",
    "model.language_model.layers.14.self_attn.k_proj",
    "model.language_model.layers.14.self_attn.v_proj",
)
_GREEDY_STEPS = frozenset({16, 32, 64})
_EXPECTED_STEPS = tuple(range(0, 65, 8))
_PRIORITY_FAMILIES = ("book_support", "mirror_lr", "picture_support")
_V35_UPDATE0 = PROJECT_ROOT / (
    "data_gemma4/checkpoints/gemma4_v35_diverse28_block_cross/update_000"
)
_SOURCE_NLL_TOLERANCE = 1e-6


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _prefixed_state(tensors: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }


def _target_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = _prefixed_state(tensors, _TARGET_PREFIX)
    observed = tuple(f"{_TARGET_PREFIX}{name}" for name in state)
    if observed != _TARGET_PARAMETER_NAMES:
        raise ValueError("V37 target-bank tensor order or inventory changed")
    if tuple(tuple(value.shape) for value in state.values()) != _TARGET_SHAPES:
        raise ValueError("V37 target-bank tensor shapes changed")
    return state


def _frozen_complement(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value for name, value in tensors.items() if not name.startswith(_TARGET_PREFIX)}


def _training_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute exact expected train-only paths without opening any data file."""

    loader = v37_loader_config(config)
    split = v31_contract(loader)
    qa_root = artifact_root(loader, "qa")
    map_root = artifact_root(loader, "maps")
    return {
        "train_scene_ids": list(split.train_scene_ids),
        "validation_scene_ids": list(split.validation_scene_ids),
        "qa_loaded_files": [
            str(qa_root / "splits.json"),
            str(qa_root / "train.jsonl"),
        ],
        "validation_qa_path": str(qa_root / "validation.jsonl"),
        "map_loaded_files": [
            str((map_root / scene_id / "voxel_map.npz").resolve())
            for scene_id in split.train_scene_ids
        ],
    }


def _validate_training_boundaries(
    stage: Mapping[str, Any], label: str, provenance: Mapping[str, Any]
) -> None:
    expected = {
        "source_optimizer_state_loaded": False,
        "source_optimizer_file_opened": False,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "independent_selector_required": True,
    }
    if any(stage.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V37 {label} crossed its train-only boundary")
    qa = _mapping(stage.get("train_qa_dataset"), f"{label} train QA")
    expected_qa = {
        "loaded_files": provenance["qa_loaded_files"],
        "train_question_count": 384,
        "train_scene_ids": provenance["train_scene_ids"],
        "train_changed_pair_unit_count": 25,
        "validation_scene_ids_from_pinned_contract": provenance["validation_scene_ids"],
        "validation_qa_path": provenance["validation_qa_path"],
        "validation_qa_loaded": False,
        "deferred_final_qa_loaded": False,
        "oracle_environment_files_loaded": False,
    }
    if any(qa.get(key) != value for key, value in expected_qa.items()):
        raise ValueError(f"V37 {label} QA provenance is not exact train-only input")
    cache = _mapping(stage.get("scene_cache"), f"{label} scene cache")
    expected_cache = {
        "scene_count": 16,
        "scene_ids": provenance["train_scene_ids"],
        "scene_scope": "training_only",
        "authenticated_manifest_scene_count": 22,
        "authenticated_manifest_train_subset_count": 16,
        "validation_environment_maps_loaded": False,
        "validation_scene_ids_loaded": [],
        "deferred_final_scene_ids_loaded": [],
        "loaded_environment_files": provenance["map_loaded_files"],
        "all_voxels_covered": True,
        "all_occupied_blocks_processed": True,
        "all_block_tokens_cached": True,
        "question_inputs_to_scene_cache": False,
        "answer_inputs_to_scene_cache": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
        "validation_qa_loaded": False,
    }
    if any(cache.get(key) != value for key, value in expected_cache.items()):
        raise ValueError(f"V37 {label} map provenance is not exact train-only input")
    if (
        qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError(f"V37 {label} QA provenance is not train-only")
    if (
        cache.get("validation_environment_maps_loaded") is not False
        or cache.get("validation_scene_ids_loaded") != []
        or cache.get("deferred_final_scene_ids_loaded") != []
        or cache.get("question_inputs_to_scene_cache") is not False
        or cache.get("all_voxels_covered") is not True
    ):
        raise ValueError(f"V37 {label} map cache crossed its train-only boundary")


def _validate_prefix_replay(
    stage: Mapping[str, Any], label: str, provenance: Mapping[str, Any]
) -> Mapping[str, Any]:
    replay = _mapping(stage.get("prefix_replay_attestation"), f"{label} prefix replay attestation")
    train_ids = provenance["train_scene_ids"]
    first = _mapping(replay.get("source_prefix_sha256_by_scene"), f"{label} first prefix hashes")
    repeated = _mapping(
        replay.get("replayed_prefix_sha256_by_scene"),
        f"{label} repeated prefix hashes",
    )
    required = {
        "source_prefix_scene_count": 16,
        "source_prefix_scene_ids": train_ids,
        "source_prefixes_replayed_bit_exact": True,
        "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors": True,
        "external_prefix_manifest_used": False,
        "scene_prefixes_built_before_questions": True,
        "training_scene_prefixes_question_free": True,
        "validation_environment_maps_loaded": False,
        "validation_qa_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
    }
    if (
        any(replay.get(key) != value for key, value in required.items())
        or tuple(first) != tuple(train_ids)
        or tuple(repeated) != tuple(train_ids)
        or dict(first) != dict(repeated)
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in first.values()
        )
    ):
        raise ValueError(f"V37 {label} prefix replay is incomplete or non-deterministic")
    return replay


def _validate_surface(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    contract: V37Contract,
) -> dict[str, Any]:
    stage = _mapping(metadata.get("v37_scene_ingress_kv"), "V37 stage")
    surface = _mapping(stage.get("trainable_surface"), "V37 trainable surface")
    expected_surface = {
        "target_bank": _TARGET_BANK,
        "target_module_paths": list(_TARGET_MODULES),
        "target_parameter_names": list(_TARGET_PARAMETER_NAMES),
        "trainable_tensor_count": 8,
        "trainable_parameter_count": 30_720,
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.0,
        "existing_learned_bank_continued_without_reinitialization": True,
        "gemma_base_frozen": True,
        "v36_learned_block_core_frozen": True,
        "v36_learned_query_bank_frozen": True,
        "complete_scene_stack_frozen": True,
        "all_other_lora_banks_frozen": True,
        "every_other_tensor_and_buffer_frozen": True,
    }
    if dict(surface) != expected_surface:
        raise ValueError("V37 persisted trainable surface changed")
    target = _target_state(tensors)
    target_hash = tensor_state_sha256(target)
    frozen_hash = tensor_state_sha256(_frozen_complement(tensors))
    core_hash = tensor_state_sha256(_prefixed_state(tensors, _CORE_PREFIX))
    query_hash = tensor_state_sha256(_prefixed_state(tensors, _QUERY_PREFIX))
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "V37 LoRA hashes")
    if (
        sum(value.numel() for value in target.values()) != 30_720
        or any(not torch.isfinite(value).all() for value in target.values())
        or target_hash != stage.get("target_bank_state_sha256")
        or target_hash != bank_hashes.get(_TARGET_BANK)
    ):
        raise ValueError("V37 target-bank state or metadata changed")
    if frozen_hash != contract.frozen_complement_state_sha256 or frozen_hash != stage.get(
        "frozen_complement_state_sha256"
    ):
        raise ValueError("V37 changed its frozen tensor complement")
    if (
        core_hash != contract.source_core_state_sha256
        or core_hash != stage.get("learned_block_core_state_sha256")
        or core_hash != metadata.get("block_cross_residual_state_sha256")
    ):
        raise ValueError("V37 changed the learned V36 block core")
    if (
        query_hash != contract.source_query_state_sha256
        or query_hash != stage.get("learned_query_bank_state_sha256")
        or query_hash != bank_hashes.get(_QUERY_BANK)
    ):
        raise ValueError("V37 changed the learned V36 query bank")
    if metadata.get("lora_trainable_parameter_count") != 30_720:
        raise ValueError("V37 checkpoint advertises the wrong trainable count")
    return {
        "target_bank_state_sha256": target_hash,
        "frozen_complement_state_sha256": frozen_hash,
        "learned_block_core_state_sha256": core_hash,
        "learned_query_bank_state_sha256": query_hash,
        "authorized_parameter_count": 30_720,
        "authorized_tensor_count": 8,
    }


def _validate_update_zero(
    *,
    tensors: Mapping[str, torch.Tensor],
    source_tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    contract: V37Contract,
) -> None:
    if (
        set(tensors) != set(source_tensors)
        or any(not torch.equal(tensors[name], source_tensors[name]) for name in tensors)
        or tensor_state_sha256(tensors) != contract.source_tensor_state_sha256
    ):
        raise ValueError("V37 update zero is not bit-exact V36 update 16")
    stage = _mapping(metadata.get("v37_scene_ingress_kv"), "V37 update-zero stage")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != 1:
        raise ValueError("V37 update zero lacks its exact one-row history")
    row = _mapping(history[0], "V37 history[0]")
    proof = _mapping(row.get("update_zero_equivalence"), "V37 update-zero proof")
    required = {
        "exact_v36_update16_adapter_loaded": True,
        "source_tensor_state_sha256": contract.source_tensor_state_sha256,
        "existing_learned_target_bank_loaded_without_reinitialization": True,
        "target_bank_source_state_sha256": contract.target_source_state_sha256,
        "target_bank_all_b_tensors_nonzero": True,
        "learned_block_core_state_sha256": contract.source_core_state_sha256,
        "learned_query_bank_state_sha256": contract.source_query_state_sha256,
        "frozen_complement_state_sha256": contract.frozen_complement_state_sha256,
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "fresh_adam_state": True,
        "source_pair_broad_residual_and_prefix_replay_exact": True,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
    }
    if any(proof.get(key) != value for key, value in required.items()):
        raise ValueError("V37 update-zero replay/equivalence proof is incomplete")
    replay = _mapping(stage.get("source_replay_attestation"), "V37 source replay")
    if not all(
        replay.get(key) is True
        for key in (
            "exact_stopped_v36_update16_loaded",
            "fresh_adam_state",
            "source_pair_metrics_bit_exact",
            "source_broad_nll_bit_exact",
            "source_residual_diagnostics_bit_exact",
            "source_prefixes_replayed_bit_exact",
            "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors",
        )
    ) or (
        replay.get("external_prefix_manifest_used") is not False
        or replay.get("validation_qa_loaded") is not False
    ):
        raise ValueError("V37 source replay attestation is incomplete")


def validate_v37_checkpoint_envelope(
    config: Mapping[str, Any],
    checkpoint_root: Path,
    contract: V37Contract,
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    """Authenticate all saved arms before validation construction is legal."""

    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError("V37 checkpoint root must be a real directory")
    if contract.saved_optimizer_steps != _EXPECTED_STEPS:
        raise ValueError("V37 saved-step contract changed")
    checkpoints = tuple(
        checkpoint_root / f"update_{step:03d}" for step in contract.saved_optimizer_steps
    )
    observed = sorted(path.name for path in checkpoint_root.glob("update_*"))
    expected = [path.name for path in checkpoints]
    if observed != expected:
        raise FileNotFoundError(
            f"V37 requires the exact completed update-64 envelope: "
            f"observed={observed} expected={expected}"
        )

    terminal = require_v36_terminal_gate(config)
    source, source_metadata, source_audit = require_exact_v36_source(config)
    source_tensors = load_file(source / "adapter.safetensors", device="cpu")
    if source_audit.get("source_optimizer_file_opened") is not False:
        raise ValueError("V37 source audit opened forbidden V36 optimizer state")
    expected_config_hash = config_hash(dict(config))
    provenance = _training_provenance(config)
    prior_history: list[Mapping[str, Any]] = []
    common_schedule: Mapping[str, Any] | None = None
    common_source_replay: Mapping[str, Any] | None = None
    common_prefix_replay: Mapping[str, Any] | None = None
    update0_tensors: Mapping[str, torch.Tensor] | None = None
    accepted16: Mapping[str, Any] | None = None
    accepted32: Mapping[str, Any] | None = None
    accepted64: Mapping[str, Any] | None = None
    audits: list[dict[str, Any]] = []

    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError(f"V37 arm must be a real directory: {checkpoint}")
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
            raise FileNotFoundError(f"V37 arm is incomplete or aliased: {checkpoint.name}")
        if step == 0 and any(
            (checkpoint / name).exists() for name in ("optimizer.pt", OPTIMIZER_AUDIT_FILENAME)
        ):
            raise ValueError("V37 update zero must not persist Adam state")

        metadata = _metadata(checkpoint)
        stage = _mapping(metadata.get("v37_scene_ingress_kv"), "V37 stage")
        if metadata.get("optimizer_step") != step or stage.get("optimizer_step") != step:
            raise ValueError(f"V37 optimizer-step metadata changed: {checkpoint.name}")
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V37 config hash changed: {checkpoint.name}")
        terminal_pin = {"path": terminal["path"], "sha256": terminal["sha256"]}
        if (
            stage.get("conditional_v36_terminal_gate") != terminal_pin
            or stage.get("conditional_authorization") != terminal["authorization"]
        ):
            raise ValueError(f"V37 terminal authorization changed: {checkpoint.name}")
        static = {
            "source_checkpoint": str(contract.source_checkpoint),
            "source_file_sha256": dict(contract.source_file_sha256),
            "source_v36_tensor_state_sha256": contract.source_tensor_state_sha256,
            "source_block_core_state_sha256": contract.source_core_state_sha256,
            "source_query_bank_state_sha256": contract.source_query_state_sha256,
            "target_bank_source_state_sha256": contract.target_source_state_sha256,
            "source_v36_frozen_nonauthorized_state_sha256": (
                contract.source_v36_frozen_state_sha256
            ),
            "fresh_adam": True,
        }
        if any(stage.get(key) != value for key, value in static.items()):
            raise ValueError(f"V37 source provenance changed: {checkpoint.name}")
        _validate_training_boundaries(stage, checkpoint.name, provenance)

        schedule = _mapping(stage.get("schedule"), "V37 schedule")
        if schedule.get("schedule_sha256") != contract.schedule_sha256:
            raise ValueError(f"V37 schedule hash changed: {checkpoint.name}")
        if common_schedule is None:
            common_schedule = schedule
        elif dict(schedule) != dict(common_schedule):
            raise ValueError("V37 schedule metadata changed across arms")
        replay = _mapping(stage.get("source_replay_attestation"), "V37 source replay")
        if common_source_replay is None:
            common_source_replay = replay
        elif dict(replay) != dict(common_source_replay):
            raise ValueError("V37 source replay changed across arms")
        prefix_replay = _validate_prefix_replay(stage, checkpoint.name, provenance)
        if common_prefix_replay is None:
            common_prefix_replay = prefix_replay
        elif dict(prefix_replay) != dict(common_prefix_replay):
            raise ValueError("V37 prefix replay changed across arms")

        history = metadata.get("history")
        if not isinstance(history, list) or len(history) != step + 1:
            raise ValueError(f"V37 history is incomplete: {checkpoint.name}")
        if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
            raise ValueError(f"V37 history is not one row per true step: {checkpoint.name}")
        if prior_history and history[: len(prior_history)] != prior_history:
            raise ValueError("V37 rewrote prior optimizer history in a later arm")
        if any(
            row.get("validation_qa_loaded") is not False
            or row.get("oracle_environment_files_loaded") is not False
            for row in history
        ):
            raise ValueError(f"V37 history crossed the data boundary: {checkpoint.name}")
        prior_history = history

        runtime = json.loads((checkpoint / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V37 runtime metadata is not freshly sanitized: {checkpoint.name}")

        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        surface = _validate_surface(metadata, tensors, contract)
        if update0_tensors is None:
            update0_tensors = tensors
            _validate_update_zero(
                tensors=tensors,
                source_tensors=source_tensors,
                metadata=metadata,
                contract=contract,
            )
        else:
            if set(tensors) != set(update0_tensors):
                raise ValueError(f"V37 tensor inventory changed: {checkpoint.name}")
            if any(
                tuple(tensors[name].shape) != tuple(update0_tensors[name].shape)
                or tensors[name].dtype != update0_tensors[name].dtype
                for name in tensors
            ):
                raise ValueError(f"V37 tensor shape/dtype changed: {checkpoint.name}")
            changed = {
                name for name in tensors if not torch.equal(tensors[name], update0_tensors[name])
            }
            if not changed or not changed.issubset(_TARGET_PARAMETER_NAME_SET):
                raise ValueError(f"V37 arm changed a non-target tensor: {checkpoint.name}")
            if surface["target_bank_state_sha256"] == contract.target_source_state_sha256:
                raise ValueError(f"V37 target bank did not transition: {checkpoint.name}")

        optimizer_audit = None
        if step:
            optimizer_audit = optimizer_step_audit(checkpoint, expected_step=step, tensors=tensors)
        gate16, gate32, gate64 = replay_v37_gates(metadata, contract)
        if step >= 16:
            if gate16 is None or gate16.get("passed") is not True:
                raise ValueError(f"V37 lacks a passed update-16 gate: {checkpoint.name}")
            if accepted16 is None:
                accepted16 = gate16
            elif gate16 != accepted16:
                raise ValueError("V37 update-16 gate changed across later arms")
        if step >= 32:
            if gate32 is None or gate32.get("passed") is not True:
                raise ValueError(f"V37 lacks a passed update-32 gate: {checkpoint.name}")
            if accepted32 is None:
                accepted32 = gate32
            elif gate32 != accepted32:
                raise ValueError("V37 update-32 gate changed across later arms")
        if step >= 64:
            if gate64 is None or gate64.get("passed") is not True:
                raise ValueError(f"V37 lacks a passed update-64 gate: {checkpoint.name}")
            if accepted64 is None:
                accepted64 = gate64
            elif gate64 != accepted64:
                raise ValueError("V37 update-64 gate changed across later arms")
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

    if accepted16 is None or accepted32 is None or accepted64 is None:
        raise ValueError("V37 completed envelope lacks all three passed train gates")
    if source_metadata.get("optimizer_step") != 16:
        raise ValueError("V37 exact source metadata changed during envelope audit")
    return checkpoints, audits


class V37ArmEvaluator(Protocol):
    validation_scene_ids: tuple[str, ...]
    cache_audit: Mapping[str, Any]

    def install(
        self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False
    ) -> None: ...

    def evaluate_teacher(self) -> V35TeacherEvidence: ...

    def evaluate_greedy(self) -> V35GreedyEvidence: ...

    def evaluate_aggregate_exact(self) -> tuple[int, int]: ...

    def attest_prefix_invariance(self) -> Mapping[str, Any]: ...


class _V37RuntimeEvaluator(_V35RuntimeEvaluator):
    """V35 scene evaluator plus explicit installation of the shared-K/V bank."""

    def __init__(
        self,
        config: dict[str, Any],
        control_config: dict[str, Any],
        checkpoint: Path,
        requirements: SelectionRequirements,
    ) -> None:
        actual_config = dict(config)
        construction_config = v37_loader_config(actual_config)
        super().__init__(construction_config, control_config, checkpoint, requirements)
        self.loader_transition = retag_bundle_for_v37(self.bundle, actual_config)
        self.config = actual_config
        collection = self.bundle.lora_installation
        if collection is None:
            raise RuntimeError("V37 evaluator requires installed LoRA banks")
        self._v37_target = collection.bank(_TARGET_BANK).installation.state_module
        self._v37_query = collection.bank(_QUERY_BANK).installation.state_module
        self._v37_contract = v37_contract(actual_config)
        self.bundle.language.model.requires_grad_(False).eval()
        for module in self.bundle.checkpoint_modules.values():
            module.requires_grad_(False).eval()

    def install(self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False) -> None:
        super().install(tensors, approved_v29=approved_v29)
        target = _target_state(tensors)
        query = _prefixed_state(tensors, _QUERY_PREFIX)
        core = _prefixed_state(tensors, _CORE_PREFIX)
        self._v37_target.load_state_dict(target, strict=True)
        self._v37_target.requires_grad_(False).eval()
        if (
            tensor_state_sha256(self._v37_target.state_dict()) != tensor_state_sha256(target)
            or tensor_state_sha256(self._v37_query.state_dict()) != tensor_state_sha256(query)
            or tensor_state_sha256(self.block_cross_residual.state_dict())
            != tensor_state_sha256(core)
        ):
            raise RuntimeError("V37 evaluator did not install the supplied tensor envelope")
        if not approved_v29 and (
            tensor_state_sha256(query) != self._v37_contract.source_query_state_sha256
            or tensor_state_sha256(core) != self._v37_contract.source_core_state_sha256
        ):
            raise RuntimeError("V37 evaluator changed its frozen learned query/core state")
        if any(
            parameter.requires_grad
            for module in self.bundle.checkpoint_modules.values()
            for parameter in module.parameters()
        ):
            raise RuntimeError("V37 selector evaluation left a checkpoint tensor trainable")

    def evaluate_teacher(self) -> V35TeacherEvidence:
        evidence = super().evaluate_teacher()
        prefix = dict(evidence.prefix_diagnostics)
        prefix["tensor"] = "composed_v37_question_independent_continuous_scene_prefix"
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
    teacher: V35TeacherEvidence,
    greedy: V35GreedyEvidence,
) -> None:
    if len(teacher.pair_margins.unit_keys) != 12 or len(teacher.pair_margins.margins) != 12:
        raise ValueError("V37 teacher evidence must score exactly 12 changed validation units")
    if (
        greedy.generation.changed_unit_count != 12
        or greedy.generation.changed_row_count != 24
        or set(greedy.complete_by_family) != set(_PRIORITY_FAMILIES)
        or set(greedy.prediction_changed_by_family) != set(_PRIORITY_FAMILIES)
    ):
        raise ValueError("V37 greedy evidence must score exactly 12 changed validation units")


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
        "v36_u16_teacher_complete_units_improved_by_at_least_1": (
            teacher.pair_margins.passed_units >= source.pair_margins.passed_units + 1
        ),
        "v36_u16_validation_answer_nll_no_worse": (
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


def _approved_v29_envelope(
    approved_tensors: Mapping[str, torch.Tensor],
    *,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Build the already-audited V29 runtime envelope without V36's learned routes."""

    compatibility_path = _V35_UPDATE0 / "adapter.safetensors"
    if compatibility_path.is_symlink() or not compatibility_path.is_file():
        raise FileNotFoundError("V37 cannot construct the exact V29 compatibility envelope")
    compatibility = load_file(compatibility_path, device="cpu")
    expected_core = block_cross_residual_settings(config).expected_initial_state_sha256
    if expected_core is None:
        raise ValueError("V37 config lacks the exact-zero block-core hash")
    return _approved_v29_runtime_tensor_envelope(
        compatibility,
        approved_tensors,
        expected_core_state_sha256=expected_core,
    )


def select_v37(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], V37ArmEvaluator
    ] = _V37RuntimeEvaluator,
) -> dict[str, Any]:
    config = load_config(config_path)
    contract = v37_contract(config)
    checkpoints, envelope_audits = validate_v37_checkpoint_envelope(
        config, checkpoint_root, contract
    )

    # This is intentionally the first legal validation-QA/map/model boundary.
    requirements = _selection_requirements(config)
    control = _retention_control_config(config)
    source_v29 = _source_v29_evidence(_metadata(checkpoints[0]))
    _validate_source_against_config(source_v29, config)
    evaluator = evaluator_factory(config, control, checkpoints[0], requirements)
    expected_validation = tuple(f"scene_{index:06d}" for index in range(19, 25))
    if tuple(evaluator.validation_scene_ids) != expected_validation:
        raise ValueError("V37 evaluator must remain exactly on validation scenes 19--24")

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
            raise ValueError("V37 teacher scorer omitted a changed validation unit")
        greedy: V35GreedyEvidence | None = None
        if step in _GREEDY_STEPS:
            greedy = evaluator.evaluate_greedy()
        arm: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "optimizer_step": step,
            "update": step,
            "validation_answer_token_nll": teacher.validation_answer_token_nll,
            "validation_pair_passed_units": teacher.pair_margins.passed_units,
            "validation_pair_passed_sides": teacher.pair_margins.passed_sides,
            "validation_pair_mean_margin": teacher.pair_margins.mean_margin,
            "validation_pair_minimum_margin": teacher.pair_margins.minimum_margin,
            "source_v36_u16_teacher_complete_units": (source_teacher.pair_margins.passed_units),
            "teacher_complete_units_delta_vs_v36_u16": (
                teacher.pair_margins.passed_units - source_teacher.pair_margins.passed_units
            ),
            "source_v36_u16_validation_answer_token_nll": (
                source_teacher.validation_answer_token_nll
            ),
            "validation_answer_nll_improvement_vs_v36_u16": (
                source_teacher.validation_answer_token_nll - teacher.validation_answer_token_nll
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
            checks, new_negatives = _development_checks(
                teacher=teacher,
                greedy=greedy,
                source=source_teacher,
                approved=approved_teacher,
                approved_greedy=approved_greedy,
            )
            arm.update(
                {
                    "new_negative_sides_vs_approved_v29": new_negatives,
                    "greedy_changed_row_count": greedy.generation.changed_row_count,
                    "greedy_changed_unit_count": greedy.generation.changed_unit_count,
                    "greedy_exact_complete_units_correct": (
                        greedy.generation.exact_complete_units_correct
                    ),
                    "greedy_exact_correct_sides": greedy.generation.exact_correct_sides,
                    "greedy_prediction_changed_units": (greedy.generation.prediction_changed_units),
                    "greedy_complete_units_by_family": dict(greedy.complete_by_family),
                    "greedy_prediction_changed_by_family": dict(
                        greedy.prediction_changed_by_family
                    ),
                    "broad_retention_exact_accuracy": (greedy.generation.broad_exact_accuracy),
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
                    arm["greedy_complete_units_by_family"], "V37 greedy families"
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
            raise ValueError("V37 selected prefix failed its pre-question invariance replay")
        selected_aggregate = evaluator.evaluate_aggregate_exact()
        prefix_attestation = evaluator.attest_prefix_invariance()

    promotion = _promotion(
        selected,
        approved_v29_aggregate=approved_aggregate,
        selected_aggregate=selected_aggregate,
        prefix_attestation=prefix_attestation,
    )
    split = v31_contract(v37_loader_config(config))
    terminal = require_v36_terminal_gate(config)
    return {
        "schema_version": 1,
        "artifact": "v37_scene_ingress_kv_development_selection",
        "development_validation_model_selection_only": True,
        "training_completed_through_update64_before_validation_loaded": True,
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
        "only_existing_shared_kv_lora_trained": True,
        "learned_v36_block_core_and_query_bank_frozen_exact": True,
        "exact_trainable_parameter_count": 30_720,
        "exact_trainable_tensor_count": 8,
        "model_load_count": 1,
        "source_v29": source_v29,
        "retention_and_aggregate_baseline": "approved_v29",
        "improvement_baseline": "exact_stopped_v36_update_016",
        "v36_terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
        "train_scene_ids": list(split.train_scene_ids),
        "validation_scene_ids": list(evaluator.validation_scene_ids),
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "all_saved_arms_inspected": True,
        "checkpoint_envelope_audits": envelope_audits,
        "optimizer_state_steps_verified": list(contract.saved_optimizer_steps[1:]),
        "train_only_update16_gate_passed": True,
        "train_only_update32_gate_passed": True,
        "train_only_update64_gate_passed": True,
        "complete_question_independent_block_cache": dict(evaluator.cache_audit),
        "teacher_scored_steps": list(contract.saved_optimizer_steps),
        "greedy_screen_steps": sorted(_GREEDY_STEPS),
        "validation_changed_unit_count": 12,
        "development_requirements": {
            "greedy_complete_units_minimum": 6,
            "greedy_validation_unit_count": 12,
            "one_greedy_complete_per_priority_family": True,
            "v36_u16_teacher_complete_units_minimum_delta": 1,
            "v36_u16_validation_answer_nll_maximum_regression": (_SOURCE_NLL_TOLERANCE),
            "approved_v29_color_sides_minimum": 12,
            "approved_v29_mirror_sides_minimum": 10,
            "approved_v29_no_new_control_negatives": True,
            "approved_v29_broad_accuracy_no_regression": True,
            "approved_v29_aggregate_accuracy_no_regression": True,
        },
        "approved_v29_teacher_baseline": {
            "validation_answer_token_nll": approved_teacher.validation_answer_token_nll,
            "color_full_vocab_sides": approved_teacher.color_full_vocab_sides,
            "mirror_full_vocab_sides": approved_teacher.mirror_full_vocab_sides,
            "broad_retention_exact_accuracy": (approved_greedy.generation.broad_exact_accuracy),
        },
        "v36_u16_teacher_baseline": {
            "validation_answer_token_nll": source_teacher.validation_answer_token_nll,
            "validation_pair_passed_units": source_teacher.pair_margins.passed_units,
            "validation_pair_mean_margin": source_teacher.pair_margins.mean_margin,
        },
        "selected_source_relative_evidence": None
        if selected is None
        else {
            "source_teacher_complete_units": source_teacher.pair_margins.passed_units,
            "selected_teacher_complete_units": selected["validation_pair_passed_units"],
            "teacher_complete_units_delta": selected["teacher_complete_units_delta_vs_v36_u16"],
            "source_validation_answer_token_nll": (source_teacher.validation_answer_token_nll),
            "selected_validation_answer_token_nll": selected["validation_answer_token_nll"],
            "validation_answer_nll_improvement": selected[
                "validation_answer_nll_improvement_vs_v36_u16"
            ],
            "nll_tolerance": _SOURCE_NLL_TOLERANCE,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = select_v37(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V37ArmEvaluator",
    "select_v37",
    "validate_v37_checkpoint_envelope",
]
