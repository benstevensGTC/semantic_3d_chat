"""Seal V38's stopped update-8 failure without loading Gemma or scene data.

The audit is intentionally byte- and tensor-oriented.  It opens the recursive
configuration chain, V37's terminal authorization and exact adapter sources,
the two V38 checkpoint arms, V38's own update-8 Adam state, and the protected
V29 selection artifact.  It never opens QA, maps, rendered observations,
oracle metadata, validation inputs, or deferred final-scene artifacts.

Passing this audit means that the stopped envelope is authentic.  It does not
mean that V38 passed its continuation gate.  Revision two binds the completed
query-delta tomography result and authorizes only a no-step/no-write
gradient-cosine measurement on the existing V28 layer-14 query adapter.  It
authorizes no optimizer continuation, training, validation, final evaluation,
or runtime promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation.v38_query_recovery_selector import select_v38
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.train_query_recovery_v38 import (
    OPTIMIZER_AUDIT_FILENAME,
    optimizer_step_audit,
    priority_side_deficit,
    replay_v38_gates,
    require_exact_v38_sources,
    require_v37_terminal_gate,
    v38_contract,
    v38_update8_gate,
    validate_per_unit_nll_diagnostics,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_query_recovery_v38.yaml")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v38_diverse28_query_recovery"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v38_update8_terminal_gate.json")

_CONFIG_SHA256 = "df884cdebed805fb783d68981011c2a66f1a37dc27aa8ecb529e1b981d25a7c5"
_CONFIG_HASH = "52df1554e3e5"
_CONFIG_CHAIN_COUNT = 36
_CONFIG_CHAIN_SHA256 = "809fad1ea042b2863b744b46c73c299c29c8566a4f2f54d3a299661ad2b29751"
_V37_TERMINAL_SHA256 = "8f8d9cfaf2c8cf564794b9f6d03eaa23f63d4fce96427816f5bc7b3fca9b70c2"
_PROTECTED_ARTIFACT = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
_PROTECTED_SHA256 = "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"

_SAVED_STEPS = (0, 8)
_SAVED_FILES = {
    0: {
        "adapter.safetensors": (
            "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0"
        ),
        TRAINING_METADATA_FILENAME: (
            "9a4b03e8fd7f8a6ef50b6d85ae6c07c602f353ecfe104dae28efaa239da5a0ed"
        ),
        RUNTIME_METADATA_FILENAME: (
            "7ec71195b6187524b903f8955af4db375b109c890fbbda9986f179b97dc58d30"
        ),
    },
    8: {
        "adapter.safetensors": (
            "5cb9dafb305fd06c3cf61cdff8affffe2064252dd8c1ed682d820f1afb1b03ea"
        ),
        TRAINING_METADATA_FILENAME: (
            "a222fb4d28b215ded869bced43324c2402d3267b2226ceb84c9ac8cec075fe7e"
        ),
        RUNTIME_METADATA_FILENAME: (
            "bfa10bc9ac00545a49efde1c0b62add1ec66739ee595d93faedab6fc195ebcd4"
        ),
        "optimizer.pt": (
            "6cf931a84157e24ab593ad786733e7ddbd57522ed17985db44fad4d3d0c0d089"
        ),
        OPTIMIZER_AUDIT_FILENAME: (
            "3afd40fdb7ecc37f43a62e173b6d99ee0b4aa270583981888a3db310e63ad79c"
        ),
    },
}
_FULL_TENSOR_SHA256 = {
    0: "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0",
    8: "6af96e291df87ea03f608c5db069e4a535e756fbb94bb52bd1446eb11a3859b6",
}
_QUERY_SHA256 = {
    0: "050706c300e6fb0ac8e4cc02e26c565b54a9a89505104302d4ffcedc02124c64",
    8: "97a02e563648efc52a594df6b6011f8e80e03677bdb5f58a9e9c6733a93dee3a",
}
_FROZEN_SHA256 = "fe39da221505c1968030c67aacb4e99f1a179e05a97d2906d416afe5fef5ed78"
_V23_SHA256 = "5c9233bd96b381e2f63443f8a739a868a21997b28061d90fb407a46d9de2d4cb"
_CORE_SHA256 = "92652fd2dbde2406227503f50717b2031baa1bcbc050902a379ddb9ddb52764f"
_PAIR_SCHEDULE_SHA256 = "4e26160c9a36e20ac894ff5e26604e6fad985ff5c84211282fd9807c9e738b3f"
_FULL_SCHEDULE_SHA256 = "2e7a2a8136c968739a8aaaa1138be87bec322fefa0fb340e6fb9bc6e07278bea"

_QUERY_BANK = "extension_v30_joint_pair_query"
_QUERY_PREFIX = f"lora_banks.{_QUERY_BANK}."
_V23_PREFIX = "lora_banks.extension_v23_shared_kv."
_CORE_PREFIX = "block_cross_residual."
_QUERY_NAMES = tuple(
    f"{_QUERY_PREFIX}adapters.{index}.lora_{side}"
    for index in range(4)
    for side in ("a", "b")
)
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
_QUERY_LAYERS = (18, 19, 20, 21)
_V28_BANK = "extension_v28_stage_b_query"
_V28_PREFIX = f"lora_banks.{_V28_BANK}."
_V28_LAYER14_PREFIX = f"{_V28_PREFIX}adapters.1."
_V28_LAYER14_NAMES = (
    f"{_V28_LAYER14_PREFIX}lora_a",
    f"{_V28_LAYER14_PREFIX}lora_b",
)
_V28_LAYER14_SHAPES = ((4, 1536), (4096, 4))
_V28_LAYER14_STATE_SHA256 = (
    "9ff9d535a094f96328483c46ff8c8ea5fca30edc35878492976c35f8674a9f87"
)
_V28_BANK_STATE_SHA256 = (
    "cc9dfa838bb87f32e2922d675658af4a1085d53a84ccdca6d5bacc6f7097217b"
)
_FROZEN_EXCLUDING_V28_LAYER14_SHA256 = (
    "7f33e541d36de33b10ceeac25e5f40374bffd1cf4b234af7a6b6341198b85360"
)
_PRIOR_TERMINAL_REPORT_SHA256 = (
    "0b637bf6a57ed1a2903e9c58e313fa2539c3dabc636444572c2018c1ee5e6b7f"
)
_TRAIN_SCENES = tuple(
    f"scene_{index:06d}" for index in (*range(11, 19), *range(31, 39))
)

_SOURCE_BROAD_NLL = 2.9013306349515915
_SOURCE_PRIORITY_DEFICIT = 31.113729119300842
_UPDATE8_BROAD_NLL = 2.9220473815997443
_UPDATE8_PRIORITY_DEFICIT = 31.062952995300293
_EXPECTED_UPDATE8_GATE = {
    "all_25_per_unit_nll_diagnostics_persisted": True,
    "broad_nll_delta_from_update_zero": 0.020716746648152817,
    "broad_nll_within_hybrid_update_zero_plus_0_02": False,
    "checks": {
        "all_25_per_unit_nll_diagnostics_persisted": True,
        "broad_nll_within_hybrid_update_zero_plus_0_02": False,
        "frozen_state_exact": True,
        "priority_teacher_deficit_improved_at_least_0_5": False,
        "query_bank_state_changed": True,
        "scene_prefix_and_residual_exact": True,
        "teacher_complete_units_at_least_9": True,
        "teacher_cross_complete_units_at_least_17": False,
        "teacher_positive_sides_at_least_34": False,
    },
    "frozen_state_exact": True,
    "passed": False,
    "priority_teacher_deficit_improved_at_least_0_5": False,
    "priority_teacher_side_deficit": _UPDATE8_PRIORITY_DEFICIT,
    "priority_teacher_side_deficit_improvement": 0.050776124000549316,
    "query_bank_state_changed": True,
    "scene_prefix_and_residual_exact": True,
    "source_priority_teacher_side_deficit": _SOURCE_PRIORITY_DEFICIT,
    "teacher_complete_units_at_least_9": True,
    "teacher_cross_complete_units_at_least_17": False,
    "teacher_positive_sides_at_least_34": False,
    "training_scenes_only": True,
    "validation_qa_loaded": False,
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _real_file(path: Path, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _state(tensors: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
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


def _config_chain(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    current = path
    while True:
        current = current.resolve()
        _real_file(current, "V38 config provenance")
        if current in seen or (current != PROJECT_ROOT and PROJECT_ROOT not in current.parents):
            raise ValueError("V38 config inheritance is cyclic or leaves the project")
        seen.add(current)
        content = current.read_bytes()
        rows.append({"path": _relative(current), "sha256": hashlib.sha256(content).hexdigest()})
        raw = yaml.safe_load(content)
        if not isinstance(raw, Mapping):
            raise TypeError("V38 config-chain member must be a mapping")
        base = raw.get("_base_")
        if base is None:
            break
        current = current.parent / str(base)
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    if (
        len(rows) != _CONFIG_CHAIN_COUNT
        or hashlib.sha256(encoded).hexdigest() != _CONFIG_CHAIN_SHA256
    ):
        raise ValueError("V38 recursive config byte chain changed")
    return rows


def _checkpoint_paths(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V38 checkpoint root must be a real directory: {root}")
    observed = sorted(path.name for path in root.iterdir() if path.name.startswith("update_"))
    expected = [f"update_{step:03d}" for step in _SAVED_STEPS]
    if observed != expected:
        raise ValueError(
            "V38 must be stopped at its contiguous failed update-8 gate: "
            f"observed={observed} expected={expected}"
        )
    paths = tuple(root / name for name in expected)
    for step, path in zip(_SAVED_STEPS, paths, strict=True):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"V38 checkpoint arm is absent or aliased: {path}")
        inventory = {child.name for child in path.iterdir()}
        if inventory != set(_SAVED_FILES[step]):
            raise ValueError(f"V38 update {step} file inventory changed")
        for name, expected_sha in _SAVED_FILES[step].items():
            candidate = path / name
            _real_file(candidate, f"V38 update {step} {name}")
            if _sha256(candidate) != expected_sha:
                raise ValueError(f"V38 update {step} hash changed: {name}")
    return paths


def _validate_prefix_and_cache(stage: Mapping[str, Any]) -> None:
    prefix = _mapping(stage.get("prefix_replay_attestation"), "V38 prefix replay")
    first = _mapping(prefix.get("prefix_sha256_by_scene"), "V38 first prefixes")
    replayed = _mapping(prefix.get("replayed_prefix_sha256_by_scene"), "V38 replayed prefixes")
    if (
        prefix.get("scene_count") != 16
        or tuple(prefix.get("scene_ids", ())) != _TRAIN_SCENES
        or prefix.get("scene_prefixes_built_before_questions") is not True
        or prefix.get("training_scene_prefixes_question_free") is not True
        or prefix.get("prefixes_replayed_bit_exact") is not True
        or prefix.get("all_occupied_blocks_processed") is not True
        or prefix.get("question_dependent_scene_processing") is not False
        or prefix.get("question_dependent_retrieval") is not False
        or prefix.get("validation_qa_loaded") is not False
        or prefix.get("validation_environment_maps_loaded") is not False
        or tuple(first) != _TRAIN_SCENES
        or dict(first) != dict(replayed)
        or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in first.values())
    ):
        raise ValueError("V38 prefix replay or question independence changed")

    cache = _mapping(stage.get("scene_cache"), "V38 scene cache")
    coverage = _mapping(cache.get("coverage_by_scene"), "V38 cache coverage")
    loaded = cache.get("loaded_environment_files")
    if (
        cache.get("scene_scope") != "training_only"
        or cache.get("scene_count") != 16
        or tuple(cache.get("scene_ids", ())) != _TRAIN_SCENES
        or cache.get("authenticated_manifest_scene_count") != 22
        or cache.get("authenticated_manifest_train_subset_count") != 16
        or cache.get("all_voxels_covered") is not True
        or cache.get("all_occupied_blocks_processed") is not True
        or cache.get("question_inputs_to_scene_cache") is not False
        or cache.get("answer_inputs_to_scene_cache") is not False
        or cache.get("validation_scene_ids_loaded") != []
        or cache.get("validation_environment_maps_loaded") is not False
        or cache.get("validation_qa_loaded") is not False
        or cache.get("oracle_environment_files_loaded") is not False
        or cache.get("deferred_final_scene_ids_loaded") != []
        or not isinstance(loaded, list)
        or tuple(Path(str(path)).parent.name for path in loaded) != _TRAIN_SCENES
        or any("oracle" in {part.casefold() for part in Path(str(path)).parts} for path in loaded)
        or tuple(coverage) != _TRAIN_SCENES
    ):
        raise ValueError("V38 scene cache crossed its exact train-only boundary")
    for scene_id in _TRAIN_SCENES:
        row = _mapping(coverage[scene_id], f"V38 coverage {scene_id}")
        if (
            row.get("processed_voxels") != row.get("voxel_count")
            or row.get("token_count") != 2 * int(row.get("occupied_block_count", -1))
            or row.get("tokens_per_block") != 2
        ):
            raise ValueError(f"V38 incomplete full-scene coverage: {scene_id}")


def _validate_stage(
    metadata: Mapping[str, Any],
    *,
    step: int,
    terminal: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    prior_history: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    stage = _mapping(metadata.get("v38_query_recovery"), "V38 stage")
    contract_fields = {
        "optimizer_step": step,
        "source_v37_tensor_state_sha256": (
            "801f80f6fd6a27e7cb815677824fb855c491272d65b81ba464c95200ebd570b9"
        ),
        "rollback_v36_tensor_state_sha256": (
            "e9b6d1362d58f34aede04817b0c8d81320c616dcd4b64e9c0d3bbe56b5835dd7"
        ),
        "update_zero_hybrid_tensor_state_sha256": _FULL_TENSOR_SHA256[0],
        "hybrid_v23_state_sha256": _V23_SHA256,
        "source_query_bank_state_sha256": _QUERY_SHA256[0],
        "source_block_core_state_sha256": _CORE_SHA256,
        "source_optimizer_states_loaded": False,
        "source_optimizer_files_opened": False,
        "fresh_adam": True,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "independent_selector_required": True,
        "frozen_excluding_query_state_sha256": _FROZEN_SHA256,
        "query_bank_state_sha256": _QUERY_SHA256[step],
    }
    terminal_pin = {"path": terminal["path"], "sha256": terminal["sha256"]}
    expected_source_audit = {
        **source_audit,
        "loader_transition": {
            "bank_names_bit_exact": True,
            "construction_copy_serialized_to_metadata": False,
            "construction_used_v30_compatible_copy": True,
            "state_hashes_bit_exact": True,
            "target_paths_bit_exact": True,
            "v38_frozen_v23_bank": "extension_v23_shared_kv",
            "v38_trainable_bank": _QUERY_BANK,
            "v38_trainable_parameter_count": 131_072,
        },
        "source_optimizer_files_opened": False,
        "source_optimizer_states_loaded": False,
    }
    if (
        metadata.get("config_hash") != _CONFIG_HASH
        or metadata.get("optimizer_step") != step
        or stage.get("conditional_v37_terminal_gate") != terminal_pin
        or stage.get("conditional_authorization") != terminal["authorization"]
        or stage.get("source_audit") != expected_source_audit
        or any(stage.get(key) != value for key, value in contract_fields.items())
    ):
        raise ValueError(f"V38 update {step} source/authorization surface changed")

    qa = _mapping(stage.get("train_qa_dataset"), "V38 train QA audit")
    loaded_qa = qa.get("loaded_files")
    if (
        qa.get("train_question_count") != 384
        or qa.get("train_changed_pair_unit_count") != 25
        or tuple(qa.get("train_scene_ids", ())) != _TRAIN_SCENES
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
        or not isinstance(loaded_qa, list)
        or [Path(str(path)).name for path in loaded_qa] != ["splits.json", "train.jsonl"]
    ):
        raise ValueError("V38 QA provenance is not exact train-only input")
    _validate_prefix_and_cache(stage)

    schedule = _mapping(stage.get("schedule"), "V38 schedule")
    if (
        schedule.get("optimizer_step_count") != 41
        or schedule.get("pair_unit_count") != 25
        or schedule.get("pair_schedule_sha256") != _PAIR_SCHEDULE_SHA256
        or schedule.get("schedule_sha256") != _FULL_SCHEDULE_SHA256
        or schedule.get("saved_optimizer_steps") != [0, 8, 16, 24, 32, 40, 41]
        or schedule.get("per_unit_nll_diagnostic_steps") != [0, 8, 16, 41]
        or schedule.get("true_optimizer_step_per_schedule_row") is not True
        or schedule.get("one_unchanged_broad_row_per_update") is not True
        or schedule.get("pair_units_atomic") is not True
        or schedule.get("questions_or_answers_serialized_to_runtime") is not False
    ):
        raise ValueError("V38 persisted deterministic schedule changed")

    raw_history = metadata.get("history")
    if not isinstance(raw_history, list) or len(raw_history) != step + 1:
        raise ValueError(f"V38 update {step} history is incomplete")
    history = [_mapping(row, "V38 history row") for row in raw_history]
    if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
        raise ValueError("V38 history is not one row per true optimizer step")
    if prior_history and history[: len(prior_history)] != list(prior_history):
        raise ValueError("V38 rewrote prior history in update 8")
    for index, row in enumerate(history):
        if (
            row.get("validation_qa_loaded") is not False
            or row.get("oracle_environment_files_loaded") is not False
            or row.get("saved_checkpoint") is not (index in _SAVED_STEPS)
            or (
                row.get("scene_prefix_and_residual_exact")
                is not (True if index in _SAVED_STEPS else None)
            )
            or row.get("frozen_excluding_query_state_sha256") != _FROZEN_SHA256
            or (index and row.get("true_optimizer_step") is not True)
            or (index and row.get("optimizer_stage") != "existing_learned_v30_query_lora_only")
            or (index and row.get("frozen_residual_descriptive_only") is not True)
            or (index and row.get("residual_penalty_contributes_gradient") is not False)
        ):
            raise ValueError("V38 history crossed its train-only true-step boundary")
    expected8: object = None if step == 0 else _EXPECTED_UPDATE8_GATE
    if (
        stage.get("update8_train_only_gate") != expected8
        or stage.get("update16_train_only_gate") is not None
        or stage.get("update41_train_only_gate") is not None
    ):
        raise ValueError("V38 persisted gate boundary changed")
    return history


def _validate_metadata(
    paths: tuple[Path, ...],
    terminal: Mapping[str, Any],
    source_audit: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    metadata_by_step: dict[int, dict[str, Any]] = {}
    prior_history: list[Mapping[str, Any]] = []
    optimizer: dict[str, Any] = {}
    for step, path in zip(_SAVED_STEPS, paths, strict=True):
        metadata = json.loads((path / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
        runtime = json.loads((path / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V38 update {step} runtime metadata is not freshly sanitized")
        forbidden_runtime = ("scene_000", "/qa/", "/maps/", "/oracle/", "validation.jsonl", "final_once")
        runtime_strings = _all_strings(runtime)
        if any(fragment in value.casefold() for value in runtime_strings for fragment in forbidden_runtime):
            raise ValueError(f"V38 update {step} runtime metadata contains environment text")
        prior_history = _validate_stage(
            metadata,
            step=step,
            terminal=terminal,
            source_audit=source_audit,
            prior_history=prior_history,
        )
        metadata_by_step[step] = metadata
        if step:
            tensors = load_file(path / "adapter.safetensors", device="cpu")
            optimizer[str(step)] = optimizer_step_audit(
                path, expected_step=step, tensors=tensors
            )
    return metadata_by_step, optimizer


def _all_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _all_strings(child)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for child in value for item in _all_strings(child)]
    return []


def _validate_tensors(
    paths: tuple[Path, ...], hybrid: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    tensors = {
        step: load_file(path / "adapter.safetensors", device="cpu")
        for step, path in zip(_SAVED_STEPS, paths, strict=True)
    }
    source = tensors[0]
    if (
        set(source) != set(hybrid)
        or any(not torch.equal(source[name], hybrid[name]) for name in source)
    ):
        raise ValueError("V38 update zero is not the exact authenticated hybrid")
    v28_bank = _state(source, _V28_PREFIX)
    v28_layer14 = _state(source, _V28_LAYER14_PREFIX)
    frozen_excluding_v28_layer14 = {
        name: value
        for name, value in source.items()
        if not name.startswith(_V28_LAYER14_PREFIX)
    }
    if (
        tuple(f"{_V28_LAYER14_PREFIX}{name}" for name in v28_layer14)
        != _V28_LAYER14_NAMES
        or tuple(tuple(value.shape) for value in v28_layer14.values())
        != _V28_LAYER14_SHAPES
        or sum(int(value.numel()) for value in v28_layer14.values()) != 22_528
        or tensor_state_sha256(v28_layer14) != _V28_LAYER14_STATE_SHA256
        or tensor_state_sha256(v28_bank) != _V28_BANK_STATE_SHA256
        or tensor_state_sha256(frozen_excluding_v28_layer14)
        != _FROZEN_EXCLUDING_V28_LAYER14_SHA256
        or any(not torch.isfinite(value).all() for value in v28_layer14.values())
        or any(not torch.count_nonzero(value) for value in v28_layer14.values())
    ):
        raise ValueError("V38 update zero V28 layer-14 diagnostic surface changed")
    states: dict[str, Any] = {}
    changes: dict[str, list[str]] = {}
    delta_norms: dict[str, float] = {}
    for step in _SAVED_STEPS:
        current = tensors[step]
        if set(current) != set(source) or len(current) != 179:
            raise ValueError(f"V38 update {step} tensor inventory changed")
        if any(
            current[name].shape != source[name].shape or current[name].dtype != source[name].dtype
            for name in source
        ):
            raise ValueError(f"V38 update {step} tensor shape/dtype changed")
        query = _state(current, _QUERY_PREFIX)
        v23 = _state(current, _V23_PREFIX)
        core = _state(current, _CORE_PREFIX)
        frozen = {
            name: value for name, value in current.items() if not name.startswith(_QUERY_PREFIX)
        }
        full_hash = tensor_state_sha256(current)
        query_hash = tensor_state_sha256(query)
        if (
            full_hash != _FULL_TENSOR_SHA256[step]
            or query_hash != _QUERY_SHA256[step]
            or tensor_state_sha256(v23) != _V23_SHA256
            or tensor_state_sha256(core) != _CORE_SHA256
            or tensor_state_sha256(frozen) != _FROZEN_SHA256
            or tuple(f"{_QUERY_PREFIX}{name}" for name in query) != _QUERY_NAMES
            or tuple(tuple(value.shape) for value in query.values()) != _QUERY_SHAPES
            or sum(int(value.numel()) for value in query.values()) != 131_072
            or any(not torch.isfinite(value).all() for value in current.values())
        ):
            raise ValueError(f"V38 update {step} tensor-state lock changed")
        changed = sorted(name for name in current if not torch.equal(current[name], source[name]))
        if (step == 0 and changed) or (step == 8 and tuple(changed) != tuple(sorted(_QUERY_NAMES))):
            raise ValueError(f"V38 update {step} changed a frozen or no target tensor")
        if step == 8:
            delta_norms = {
                name: float(torch.linalg.vector_norm((current[name] - source[name]).float()))
                for name in _QUERY_NAMES
            }
        changes[str(step)] = changed
        states[str(step)] = {
            "full_tensor_state_sha256": full_hash,
            "query_bank_state_sha256": query_hash,
            "frozen_excluding_query_state_sha256": tensor_state_sha256(frozen),
            "hybrid_v23_state_sha256": tensor_state_sha256(v23),
            "learned_block_core_state_sha256": tensor_state_sha256(core),
        }
    return {
        "adapter_tensor_count": 179,
        "query_tensor_count": 8,
        "query_parameter_count": 131_072,
        "frozen_tensor_count": 171,
        "update_zero_bit_exact_authenticated_hybrid": True,
        "update8_changed_exactly_all_eight_query_tensors": True,
        "all_frozen_tensors_bit_exact": True,
        "all_tensors_finite": True,
        "v28_layer14_gradient_screen_source": {
            "existing_bank": _V28_BANK,
            "target_module": "model.language_model.layers.14.self_attn.q_proj",
            "target_tensor_names": list(_V28_LAYER14_NAMES),
            "target_tensor_shapes": [list(shape) for shape in _V28_LAYER14_SHAPES],
            "target_tensor_count": 2,
            "target_parameter_count": 22_528,
            "target_state_sha256": _V28_LAYER14_STATE_SHA256,
            "complete_existing_bank_state_sha256": _V28_BANK_STATE_SHA256,
            "frozen_excluding_target_tensor_count": 177,
            "frozen_excluding_target_state_sha256": (
                _FROZEN_EXCLUDING_V28_LAYER14_SHA256
            ),
            "both_existing_adapter_tensors_nonzero": True,
        },
        "changed_tensor_names_by_optimizer_step": changes,
        "query_delta_l2_by_tensor": delta_norms,
        "state_sha256_by_optimizer_step": states,
    }


def _replay_failed_gate(
    metadata0: Mapping[str, Any], metadata8: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    stage0 = _mapping(metadata0.get("v38_query_recovery"), "V38 update-zero stage")
    stage8 = _mapping(metadata8.get("v38_query_recovery"), "V38 update-eight stage")
    baseline = _mapping(
        _mapping(stage0.get("update_zero_attestation"), "V38 update-zero attestation").get(
            "behavioral_baseline"
        ),
        "V38 update-zero behavioral baseline",
    )
    observed = _mapping(baseline.get("observed"), "V38 observed update-zero baseline")
    row8 = _mapping(metadata8.get("history", [None])[-1], "V38 update-eight history")
    pairs = _mapping(row8.get("training_pair_metrics"), "V38 update-eight pair metrics")
    diagnostics = row8.get("per_unit_nll_diagnostics")
    if not isinstance(diagnostics, list):
        raise TypeError("V38 update eight lacks per-unit diagnostics")
    validate_per_unit_nll_diagnostics(diagnostics, pairs)
    source_broad = float(stage0.get("source_broad_train_nll"))
    source_deficit = float(observed.get("priority_combined_side_deficit"))
    broad = float(row8.get("training_broad_nll"))
    if (
        not math.isclose(source_broad, _SOURCE_BROAD_NLL, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(source_deficit, _SOURCE_PRIORITY_DEFICIT, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(broad, _UPDATE8_BROAD_NLL, rel_tol=0.0, abs_tol=0.0)
        or pairs.get("complete_units") != 9
        or pairs.get("complete_physical_pair_coverage") != 4
        or pairs.get("cross_prefix_complete_units") != 16
        or pairs.get("positive_sides") != 33
        or pairs.get("mean_cross_prefix_margin") != 1.435127854347229
        or pairs.get("minimum_margin") != -7.6875
        or pairs.get("complete_units_by_family")
        != {"book_support": 0, "mirror_lr": 2, "picture_support": 0}
        or pairs.get("cross_prefix_complete_units_by_family")
        != {"book_support": 0, "mirror_lr": 4, "picture_support": 3}
        or priority_side_deficit(pairs)["combined"] != _UPDATE8_PRIORITY_DEFICIT
    ):
        raise ValueError("V38 update-eight train-only numerics changed")
    contract = v38_contract(config)
    replayed = v38_update8_gate(
        pair_metrics=pairs,
        broad_nll=broad,
        source_broad_nll=source_broad,
        source_priority_deficit=source_deficit,
        query_state_sha256=str(row8.get("query_bank_state_sha256")),
        frozen_state_sha256=str(row8.get("frozen_excluding_query_state_sha256")),
        scene_state_exact=row8.get("scene_prefix_and_residual_exact") is True,
        per_unit_nll_diagnostics=diagnostics,
        contract=contract,
    )
    if (
        replayed != _EXPECTED_UPDATE8_GATE
        or row8.get("update8_train_only_gate") != replayed
        or stage8.get("update8_train_only_gate") != replayed
    ):
        raise ValueError("V38 independently replayed failed update-eight gate changed")
    helper8, helper16, helper41 = replay_v38_gates(metadata8, contract)
    if helper8 != replayed or helper16 is not None or helper41 is not None:
        raise ValueError("V38 public replay helper disagrees with terminal replay")
    source_pairs = _mapping(stage0.get("source_pair_metrics"), "V38 update-zero pairs")
    return {
        "source": {
            "broad_train_nll": source_broad,
            "priority_side_deficit": source_deficit,
            "complete_units": source_pairs.get("complete_units"),
            "complete_physical_pair_coverage": source_pairs.get(
                "complete_physical_pair_coverage"
            ),
            "cross_prefix_complete_units": source_pairs.get("cross_prefix_complete_units"),
            "positive_sides": source_pairs.get("positive_sides"),
            "mean_cross_prefix_margin": source_pairs.get("mean_cross_prefix_margin"),
            "minimum_margin": source_pairs.get("minimum_margin"),
            "complete_units_by_family": source_pairs.get("complete_units_by_family"),
            "cross_prefix_complete_units_by_family": source_pairs.get(
                "cross_prefix_complete_units_by_family"
            ),
        },
        "update8": {
            "broad_train_nll": broad,
            "priority_side_deficit": priority_side_deficit(pairs)["combined"],
            "priority_side_deficit_by_family": priority_side_deficit(pairs),
            "complete_units": pairs.get("complete_units"),
            "complete_physical_pair_coverage": pairs.get(
                "complete_physical_pair_coverage"
            ),
            "cross_prefix_complete_units": pairs.get("cross_prefix_complete_units"),
            "positive_sides": pairs.get("positive_sides"),
            "mean_cross_prefix_margin": pairs.get("mean_cross_prefix_margin"),
            "minimum_margin": pairs.get("minimum_margin"),
            "complete_units_by_family": pairs.get("complete_units_by_family"),
            "cross_prefix_complete_units_by_family": pairs.get(
                "cross_prefix_complete_units_by_family"
            ),
        },
        "failed_requirements": [
            "priority_teacher_deficit_improved_at_least_0_5",
            "teacher_positive_sides_at_least_34",
            "teacher_cross_complete_units_at_least_17",
            "broad_nll_within_hybrid_update_zero_plus_0_02",
        ],
        "checks": replayed,
        "nested_behavioral_baseline_schema_replayed": True,
        "public_replay_helper_regression_passed": True,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "passed": False,
    }


def _selector_refusal(config_path: Path, root: Path) -> dict[str, Any]:
    constructed = False

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("V38 validation evaluator was constructed")

    try:
        select_v38(config_path, root, evaluator_factory=forbidden_factory)  # type: ignore[arg-type]
    except FileNotFoundError as error:
        message = str(error)
    else:
        raise RuntimeError("Incomplete V38 selector did not refuse the stopped envelope")
    if constructed or "exact completed update-41 envelope" not in message:
        raise RuntimeError("V38 selector refusal occurred after the validation boundary")
    return {
        "selector_refused_incomplete_envelope": True,
        "validation_evaluator_constructed": False,
        "refusal_type": "FileNotFoundError",
        "refusal_message": message,
        "selector_output_written": False,
    }


def _tomography_summary() -> dict[str, Any]:
    """Bind the reported train-only/no-write V38 delta tomography outcome.

    The four extrapolation scales and six two-layer masks were outside the
    exact candidate grid in the first terminal report.  They are therefore
    recorded as an unplanned, non-mutating scope extension rather than being
    retroactively described as authorized.
    """

    return {
        "schema_version": 1,
        "artifact": "v38_query_delta_tomography_summary",
        "evidence_scope": "reported_real_local_gemma_exact_training_scenes_only",
        "source_terminal_report_sha256": _PRIOR_TERMINAL_REPORT_SHA256,
        "source_update_zero_full_tensor_state_sha256": _FULL_TENSOR_SHA256[0],
        "source_update_eight_full_tensor_state_sha256": _FULL_TENSOR_SHA256[8],
        "source_update_zero_query_state_sha256": _QUERY_SHA256[0],
        "source_update_eight_query_state_sha256": _QUERY_SHA256[8],
        "candidate_formula": "u0_query + scale * layer_mask(u8_query - u0_query)",
        "source_metrics": {
            "complete_units": 9,
            "complete_physical_pair_coverage": 4,
            "positive_sides": 34,
            "cross_prefix_complete_units": 17,
            "priority_side_deficit": _SOURCE_PRIORITY_DEFICIT,
        },
        "full_bank_scale_screen": {
            "observed_scale_factors": [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            "observed_candidate_count": 9,
            "eligible_candidate_count": 0,
            "no_scale_passed": True,
            "best_priority_deficit_gain": 0.304239,
            "best_priority_deficit_gain_scale": 0.5,
            "best_gain_candidate_metrics": {
                "complete_units": 8,
                "complete_physical_pair_coverage": 3,
                "positive_sides": 33,
            },
            "best_gain_candidate_retention_passed": False,
        },
        "layer_mask_scale_one_screen": {
            "layer_universe": list(_QUERY_LAYERS),
            "observed_nonempty_mask_count": 15,
            "all_nonempty_masks_observed": True,
            "scale_factor": 1.0,
            "eligible_candidate_count": 0,
            "no_mask_reached_priority_deficit_gain_0_5": True,
            "best_retention_shaped_mask": {
                "layers": [19, 20, 21],
                "complete_units": 9,
                "complete_physical_pair_coverage": 4,
                "positive_sides": 34,
                "cross_prefix_complete_units": 20,
                "priority_deficit_gain": 0.101911,
                "eligible": False,
            },
            "other_notable_masks": [
                {
                    "layers": [18],
                    "priority_deficit_gain": 0.103418,
                    "positive_sides": 33,
                    "eligible": False,
                },
                {
                    "layers": [18, 19, 21],
                    "priority_deficit_gain": 0.119285,
                    "positive_sides": 33,
                    "eligible": False,
                },
            ],
        },
        "aggregate_result": {
            "eligible_candidate_count": 0,
            "eligible_guarded_late_query_subset_found": False,
            "late_query_training_authorized_by_result": False,
        },
        "scope_audit": {
            "prior_authorized_scale_factors": [0.0, 0.25, 0.5, 0.75, 1.0],
            "unplanned_scale_extensions": [-1.0, -0.5, 1.5, 2.0],
            "prior_authorized_layer_mask_count": 9,
            "unplanned_layer_masks": [
                [18, 19],
                [18, 20],
                [18, 21],
                [19, 20],
                [19, 21],
                [20, 21],
            ],
            "candidate_grid_fully_within_prior_artifact_authorization": False,
            "unplanned_candidate_grid_extension": True,
            "extension_characterization": "training_free_train_only_no_write",
            "retroactive_authorization_claimed": False,
            "reported_parameter_writes": False,
            "reported_optimizer_step_calls": False,
            "reported_new_bank_construction": False,
            "reported_validation_access": False,
            "reported_final_test_access": False,
            "reported_oracle_access": False,
            "raw_per_candidate_trace_bound": False,
            "summary_bound_from_root_result_attestation": True,
        },
        "passed": True,
    }


def _gradient_screen_authorization(paths: tuple[Path, ...]) -> dict[str, Any]:
    return {
        "authorized": True,
        "successor": "v39_v28_layer14_gradient_cosine_screen",
        "scope": "no_step_no_write_existing_v28_layer14_query_gradient_measurement",
        "source_checkpoint": _relative(paths[0]),
        "source_adapter_file_sha256": _SAVED_FILES[0]["adapter.safetensors"],
        "source_full_tensor_state_sha256": _FULL_TENSOR_SHA256[0],
        "existing_lora_bank": _V28_BANK,
        "existing_adapter_index": 1,
        "target_language_layer": 14,
        "target_module_path": "model.language_model.layers.14.self_attn.q_proj",
        "target_parameter_names": list(_V28_LAYER14_NAMES),
        "target_parameter_shapes": [list(shape) for shape in _V28_LAYER14_SHAPES],
        "target_tensor_count": 2,
        "target_parameter_count": 22_528,
        "target_rank": 4,
        "target_alpha": 8.0,
        "target_dropout": 0.0,
        "target_source_state_sha256": _V28_LAYER14_STATE_SHA256,
        "complete_existing_bank_state_sha256": _V28_BANK_STATE_SHA256,
        "both_existing_target_tensors_nonzero": True,
        "gradient_computation_authorized": True,
        "backward_or_autograd_grad_for_measurement_authorized": True,
        "temporary_requires_grad_toggle_authorized": True,
        "training_authorized": False,
        "optimizer_construction_authorized": False,
        "optimizer_step_authorized": False,
        "parameter_update_authorized": False,
        "parameter_or_buffer_write_authorized": False,
        "gradient_accumulation_across_objectives_authorized": False,
        "source_optimizer_access_authorized": False,
        "update8_optimizer_access_authorized": False,
        "new_lora_bank_authorized": False,
        "new_scene_encoder_module_authorized": False,
        "new_scene_tokens_authorized": False,
        "frozen_excluding_target_tensor_count": 177,
        "frozen_excluding_target_state_sha256": (
            _FROZEN_EXCLUDING_V28_LAYER14_SHA256
        ),
        "all_parameters_and_buffers_must_be_bit_exact_after_each_measurement": True,
        "target_state_must_be_bit_exact_after_each_measurement": True,
        "gradients_must_be_cleared_between_objectives": True,
        "required_measurements": [
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
        ],
        "required_artifact_evidence": [
            "source_file_and_tensor_hashes_before_measurement",
            "target_and_frozen_hashes_before_each_objective",
            "target_and_frozen_hashes_after_each_objective",
            "finite_gradient_checks",
            "exact_training_scene_and_question_inventory",
            "loaded_file_inventory",
            "no_optimizer_constructed_or_opened",
        ],
        "scene_prefixes_must_remain_question_independent": True,
        "all_occupied_blocks_must_be_processed": True,
        "question_dependent_retrieval": False,
        "diagnostic_data_scope": "exact_training_scenes_only",
        "diagnostic_result_may_promote_runtime": False,
        "diagnostic_result_may_authorize_training": False,
        "separate_terminal_seal_required_for_any_training": True,
        "validation_access_authorized": False,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
        "chat_promotion_authorized": False,
    }


def audit_v38_update8(
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    config_path = _resolve(config_path)
    checkpoint_root = _resolve(checkpoint_root)
    _real_file(config_path, "V38 config")
    if _sha256(config_path) != _CONFIG_SHA256:
        raise ValueError("V38 config bytes differ from the terminal lock")
    config_chain = _config_chain(config_path)
    config = load_config(config_path)
    if config_hash(dict(config)) != _CONFIG_HASH:
        raise ValueError("V38 normalized config hash changed")
    contract = v38_contract(config)
    terminal = require_v37_terminal_gate(config)
    if terminal["sha256"] != _V37_TERMINAL_SHA256:
        raise ValueError("V38 exact V37 terminal authorization changed")
    hybrid, _source_metadata, source_audit = require_exact_v38_sources(config)
    paths = _checkpoint_paths(checkpoint_root)
    metadata, optimizer = _validate_metadata(paths, terminal, source_audit)
    tensors = _validate_tensors(paths, hybrid)
    gate = _replay_failed_gate(metadata[0], metadata[8], config)
    selector = _selector_refusal(config_path, checkpoint_root)

    protected = _resolve(_PROTECTED_ARTIFACT)
    _real_file(protected, "protected V29 selection artifact")
    if _sha256(protected) != _PROTECTED_SHA256:
        raise ValueError("Protected V29 selection artifact changed")

    source_files = [
        contract.terminal_report,
        contract.source_checkpoint / "adapter.safetensors",
        contract.source_checkpoint / TRAINING_METADATA_FILENAME,
        contract.source_checkpoint / RUNTIME_METADATA_FILENAME,
        contract.rollback_checkpoint / "adapter.safetensors",
    ]
    loaded = [
        *(PROJECT_ROOT / row["path"] for row in config_chain),
        *source_files,
        protected,
        *(path / name for path in paths for name in _SAVED_FILES[int(path.name[-3:])]),
    ]
    inventory = sorted({_relative(path.resolve()) for path in loaded})
    forbidden = (
        "/qa/",
        "/maps/",
        "/oracle/",
        "validation.jsonl",
        "final_once",
        "scene_000025",
        "scene_000030",
    )
    if any(fragment in path.casefold() for path in inventory for fragment in forbidden):
        raise RuntimeError("V38 terminal audit opened a forbidden environment/data file")

    report = {
        "schema_version": 1,
        "artifact": "v38_update8_terminal_gate",
        "seal_revision": 2,
        "audit_method": (
            "recursive_config_v37_terminal_exact_sources_v38_metadata_runtime_"
            "own_optimizer_manifest_tensor_bytes_failed_gate_and_selector_refusal_only"
        ),
        "passed": True,
        "checkpoint_root": _relative(checkpoint_root),
        "observed_saved_optimizer_steps": list(_SAVED_STEPS),
        "stopped_at_optimizer_step": 8,
        "no_update_016_or_later": True,
        "v38_train_only_continuation_gate_passed": False,
        "v38_development_selector_legal": False,
        "v38_chat_promotion_eligible": False,
        "exact_v37_terminal_and_source": {
            "terminal_report": {
                "path": _relative(contract.terminal_report),
                "sha256": terminal["sha256"],
            },
            "source_checkpoint": _relative(contract.source_checkpoint),
            "rollback_checkpoint": _relative(contract.rollback_checkpoint),
            "source_audit": source_audit,
            "source_optimizer_access": "not_opened_not_hashed_not_deserialized",
            "rollback_optimizer_access": "not_opened_not_hashed_not_deserialized",
        },
        "config_provenance": {
            "root": _relative(config_path),
            "root_sha256": _CONFIG_SHA256,
            "normalized_config_hash": _CONFIG_HASH,
            "recursive_chain_sha256": _CONFIG_CHAIN_SHA256,
            "recursive_chain": config_chain,
        },
        "saved_file_sha256_by_optimizer_step": {
            str(step): dict(_SAVED_FILES[step]) for step in _SAVED_STEPS
        },
        "tensor_transition": tensors,
        "optimizer_transition": {
            "fresh_v38_adam_verified": True,
            "optimizer_integrity_manifest_verified": True,
            "source_or_rollback_optimizer_opened": False,
            "saved_optimizer_states": optimizer,
        },
        "update8_gate_evidence": gate,
        "selector_refusal": selector,
        "replay_bug_regression": {
            "correct_persisted_path": (
                "v38_query_recovery.update_zero_attestation."
                "behavioral_baseline.observed"
            ),
            "nested_schema_used": True,
            "public_replay_helper_matches_independent_replay": True,
        },
        "query_delta_tomography_summary": _tomography_summary(),
        "conditional_successor_authorization": _gradient_screen_authorization(paths),
        "query_delta_tomography_completed": True,
        "conditional_v39_query_delta_tomography_authorized": False,
        "conditional_v39_v28_layer14_gradient_cosine_screen_authorized": True,
        "v39_training_authorized": False,
        "arbitrary_continuation_authorized": False,
        "only_exact_conditional_successor_authorized": (
            "v39_v28_layer14_gradient_cosine_screen"
        ),
        "chat_promotion_authorized": False,
        "validation_access_authorized": False,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
        "validation_qa_loaded": False,
        "validation_model_selection_ran": False,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "loaded_file_inventory": inventory,
        "protected_artifact": {
            "path": _relative(protected),
            "sha256": _PROTECTED_SHA256,
            "access": "bytes_hashed_only",
            "unchanged": True,
        },
    }
    return json.loads(json.dumps(report, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_v38_update8(args.config, args.checkpoint_root)
    _atomic_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_v38_update8"]
