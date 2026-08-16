"""Train V96's exact atomic-pair repair without touching held-out data.

The exact non-promoted V95 fixed-final stack is frozen.  Only one fresh rank-8
LoRA bank on layer 9's full-attention query projection is optimized.  The
fixed schedule combines two full broad-retention passes, four symmetric
changed-pair rounds, and one answer-independent stable-pair subset.  Every
environment remains a precompiled 738-token continuous memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import (
    EXPECTED_BANKS as V94_BANKS,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
    canonical_sha256_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    FRESH_BANK_NAME as V95_BANK_NAME,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    load_config_v95,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    EXPECTED_CHANGED_PAIR_STEPS,
    EXPECTED_FROZEN_BANK_COUNT,
    EXPECTED_FROZEN_PARAMETER_COUNT,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_INVARIANT_PAIR_STEPS,
    EXPECTED_MICRO_STEPS,
    EXPECTED_OPTIMIZER_UPDATES,
    EXPECTED_RETENTION_STEPS,
    EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT,
    EXPECTED_TOTAL_NLL_FORWARDS,
    FRESH_BANK_NAME,
    FRESH_PARAMETER_COUNT,
    TARGET_MODULES,
    PairUnitV96,
    TrainingStepV96,
    assert_deferred_final_absent_v96,
    assert_initial_outputs_absent_v96,
    authenticate_cpu_preflight_v96,
    authenticate_parent_v95_v96,
    authenticate_training_sources_v96,
    balanced_class_weights_v96,
    family_weights_v96,
    forbidden_training_roots_v96,
    invariant_subset_v96,
    load_config_v96,
    load_scene_memories_v96,
    load_training_rows_v96,
    lora_preflight_v96,
    pair_units_v96,
    training_schedule_v96,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_v84_strict_bridge import _prepared_v84
from semantic_3d_chat.training.train_v95_strict_causal_successor import (
    _load_v85_banks_v95,
    _load_v94_bank_v95,
    combined_lora_settings_v95,
    load_fixed_final_bridge_v95,
)

TRAINING_ARTIFACT: Final[str] = "gemma4_v96_atomic_pair_repair_training_v1"
CANDIDATE_ARTIFACT: Final[str] = "gemma4_v96_atomic_pair_repair_fixed_final_v1"
CHECKPOINT_ARTIFACT: Final[str] = "gemma4_v96_atomic_pair_repair_resume_v1"
TOPOLOGY_ARTIFACT: Final[str] = "gemma4_v96_atomic_pair_repair_topology_smoke_v1"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
EXPECTED_V85_BANK_COUNT: Final[int] = 7
EXPECTED_V85_PARAMETER_COUNT: Final[int] = 565_248
EXPECTED_OPTIMIZER_PARAM_GROUPS: Final[list[dict[str, Any]]] = [
    {
        "lr": 0.000075,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
        "params": [0, 1],
    }
]


def _leaf_path_v96(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return Path(os.path.abspath(value))


def strict_json_v96(path: str | Path) -> dict[str, Any]:
    source = _leaf_path_v96(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V96 JSON must contain one object: {source}")
    return value


def _is_sha256_v96(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def combined_lora_settings_v96(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    """Append frozen V95 and fresh V96 banks to the exact prior stack."""

    v95_config = load_config_v95(experiment["sources"]["frozen_v95_config"], allow_draft=False)
    parent = combined_lora_settings_v95(runtime_config, v95_config)
    if (
        len(parent.banks) != EXPECTED_FROZEN_BANK_COUNT
        or tuple(bank.name for bank in parent.banks) != V94_BANKS + (V95_BANK_NAME,)
        or sum(bank.trainable for bank in parent.banks) != 1
    ):
        raise ValueError("V96 requires the exact nine-bank V95 topology")
    parent_v95 = parent.banks[-1]
    frozen_v95 = LoRABankSettings(
        name=parent_v95.name,
        trainable=False,
        adapter=parent_v95.adapter,
        initialization_algorithm="checkpoint_overwrite",
        expected_initial_state_sha256=str(experiment["frozen_stack"]["v95_bank_state_sha256"]),
    )
    bridge = experiment["bridge"]
    fresh = LoRABankSettings(
        name=FRESH_BANK_NAME,
        trainable=True,
        adapter=LoRASettings(
            enabled=True,
            rank=int(bridge["rank"]),
            alpha=float(bridge["alpha"]),
            dropout=float(bridge["dropout"]),
            target_modules=tuple(bridge["target_modules"]),
        ),
        initialization_algorithm=str(bridge["initialization_algorithm"]),
        initialization_seed=int(bridge["initialization_seed"]),
        expected_initial_state_sha256=str(bridge["expected_initial_state_sha256"]),
    )
    result = LoRABanksSettings(parent.banks[:-1] + (frozen_v95, fresh))
    if (
        len(result.banks) != 10
        or sum(bank.trainable for bank in result.banks) != 1
        or result.banks[-1].name != FRESH_BANK_NAME
    ):
        raise RuntimeError("V96 exact ten-bank topology construction failed")
    return result


def load_frozen_parent_v96(
    collection: LoRABankCollection,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load V85, V94, and V95 while leaving only V96 trainable."""

    if (
        collection.bank_names != V94_BANKS + (V95_BANK_NAME, FRESH_BANK_NAME)
        or len([bank for bank in collection.banks if not bank.settings.trainable])
        != EXPECTED_FROZEN_BANK_COUNT
    ):
        raise ValueError("V96 installed bank order changed")
    v95_config = load_config_v95(config["sources"]["frozen_v95_config"], allow_draft=False)
    sources = v95_config["sources"]
    v85 = _load_v85_banks_v95(collection, sources["frozen_v85_checkpoint"])
    v94 = _load_v94_bank_v95(collection, sources["frozen_v94_fixed_final"])
    v95 = load_fixed_final_bridge_v95(collection, config["sources"]["frozen_v95_fixed_final"])
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if (
        fresh.target_names != TARGET_MODULES
        or fresh.parameter_count != FRESH_PARAMETER_COUNT
        or fresh.state_sha256() != EXPECTED_INITIAL_STATE_SHA256
        or any(torch.count_nonzero(adapter.lora_b).item() for adapter in fresh.adapters)
        or collection.parameter_count != EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT
        or collection.trainable_parameter_count != FRESH_PARAMETER_COUNT
    ):
        raise ValueError("V96 fresh bank did not begin at exact zero output")
    collection.validate_state()
    return {
        "parent": "v95_fixed_final_nonpromoted_optimization_parent",
        "v85": v85,
        "v94": v94,
        "v95": {
            "weights_sha256": v95["weights_sha256"],
            "state_sha256": v95["state_sha256"],
        },
        "frozen_bank_count": EXPECTED_FROZEN_BANK_COUNT,
        "frozen_parameter_count": EXPECTED_FROZEN_PARAMETER_COUNT,
        "v95_known_development_gate_passed": False,
        "runtime_release_loaded": False,
    }


def _finite_scalar_v96(value: torch.Tensor, name: str) -> None:
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError(f"V96 {name} must be a finite scalar")


def smoothmax_v96(left: torch.Tensor, right: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """A zero-preserving smooth maximum over two nonnegative penalties."""

    _finite_scalar_v96(left, "smoothmax left")
    _finite_scalar_v96(right, "smoothmax right")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("V96 smoothmax temperature must be positive")
    stacked = torch.stack((left, right))
    return temperature * (torch.logsumexp(stacked / temperature, dim=0) - math.log(2.0))


def symmetric_pair_objective_v96(
    left_correct_nll: torch.Tensor,
    right_correct_nll: torch.Tensor,
    left_alternative_nll: torch.Tensor,
    right_alternative_nll: torch.Tensor,
    *,
    left_class_weight: float,
    right_class_weight: float,
    family_weight: float,
    correct_ce_weight: float,
    answer_margin_weight: float,
    answer_target_margin: float,
    causal_margin_weight: float,
    causal_target_margin: float,
    smoothmax_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Jointly require both exact pair sides to be correct and to switch."""

    values = (
        left_correct_nll,
        right_correct_nll,
        left_alternative_nll,
        right_alternative_nll,
    )
    for name, value in zip(("left correct", "right correct", "left alt", "right alt"), values):
        _finite_scalar_v96(value, name)
    numbers = (
        left_class_weight,
        right_class_weight,
        family_weight,
        correct_ce_weight,
        answer_margin_weight,
        answer_target_margin,
        causal_margin_weight,
        causal_target_margin,
        smoothmax_temperature,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in numbers):
        raise ValueError("V96 pair objective constants must be finite nonnegative")
    correct_ce = 0.5 * (
        float(left_class_weight) * left_correct_nll + float(right_class_weight) * right_correct_nll
    )
    # Within memory A, yA must beat yB; within memory B, yB must beat yA.
    left_answer = torch.relu(left_correct_nll - left_alternative_nll + float(answer_target_margin))
    right_answer = torch.relu(
        right_correct_nll - right_alternative_nll + float(answer_target_margin)
    )
    answer_penalty = smoothmax_v96(left_answer, right_answer, temperature=smoothmax_temperature)
    # For a fixed answer, its true scene must beat the atomic counterpart.
    left_causal = torch.relu(left_correct_nll - right_alternative_nll + float(causal_target_margin))
    right_causal = torch.relu(
        right_correct_nll - left_alternative_nll + float(causal_target_margin)
    )
    causal_penalty = smoothmax_v96(left_causal, right_causal, temperature=smoothmax_temperature)
    objective = float(family_weight) * (
        float(correct_ce_weight) * correct_ce
        + float(answer_margin_weight) * answer_penalty
        + float(causal_margin_weight) * causal_penalty
    )
    return objective, {
        "correct_ce": correct_ce,
        "left_answer_margin_penalty": left_answer,
        "right_answer_margin_penalty": right_answer,
        "answer_smoothmax_penalty": answer_penalty,
        "left_causal_margin_penalty": left_causal,
        "right_causal_margin_penalty": right_causal,
        "causal_smoothmax_penalty": causal_penalty,
        "left_alternative_minus_correct_nll": left_alternative_nll - left_correct_nll,
        "right_alternative_minus_correct_nll": right_alternative_nll - right_correct_nll,
    }


def invariant_pair_objective_v96(
    left_nll: torch.Tensor,
    right_nll: torch.Tensor,
    *,
    left_class_weight: float,
    right_class_weight: float,
    family_weight: float,
    correct_ce_weight: float,
    consistency_weight: float,
    consistency_tolerance: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve facts that must not change across an atomic scene pair."""

    _finite_scalar_v96(left_nll, "invariant left")
    _finite_scalar_v96(right_nll, "invariant right")
    numbers = (
        left_class_weight,
        right_class_weight,
        family_weight,
        correct_ce_weight,
        consistency_weight,
        consistency_tolerance,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in numbers):
        raise ValueError("V96 invariant objective constants must be finite nonnegative")
    correct_ce = 0.5 * (float(left_class_weight) * left_nll + float(right_class_weight) * right_nll)
    gap = torch.abs(left_nll - right_nll)
    penalty = torch.relu(gap - float(consistency_tolerance))
    objective = float(family_weight) * (
        float(correct_ce_weight) * correct_ce + float(consistency_weight) * penalty
    )
    return objective, {
        "correct_ce": correct_ce,
        "absolute_nll_gap": gap,
        "consistency_penalty": penalty,
    }


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _fresh_state_v96(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if len(fresh.adapters) != 1:
        raise ValueError("V96 fresh bank must wrap exactly one module")
    return {
        name: value.detach().cpu().contiguous()
        for name, value in fresh.state_module.state_dict().items()
    }


def _load_fresh_state_v96(
    collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]
) -> None:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    expected = set(fresh.state_module.state_dict())
    if set(archive) != expected:
        raise ValueError("V96 fresh-bank tensor inventory changed")
    fresh.state_module.load_state_dict(dict(archive), strict=True)
    fresh.validate_state()


def _optimizer_tensors_v96(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    state = optimizer.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for parameter_index, values in state["state"].items():
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("V96 AdamW state must be tensor-only")
            tensors[f"optimizer.{parameter_index}.{name}"] = value.detach().cpu().contiguous()
    groups: list[dict[str, Any]] = []
    for group in state["param_groups"]:
        normalized = dict(group)
        normalized["params"] = [int(value) for value in normalized["params"]]
        groups.append(normalized)
    return tensors, groups


def _restore_optimizer_v96(
    optimizer: torch.optim.Optimizer,
    archive: Mapping[str, torch.Tensor],
    groups: Sequence[Mapping[str, Any]],
) -> None:
    state: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in archive.items():
        if key.startswith("optimizer."):
            _prefix, raw_index, name = key.split(".", 2)
            state.setdefault(int(raw_index), {})[name] = value.detach().cpu()
    optimizer.load_state_dict({"state": state, "param_groups": [dict(group) for group in groups]})


def save_resume_checkpoint_v96(
    work_root: str | Path,
    collection: LoRABankCollection,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    row_cursor: int,
    history: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    row_order_sha256: str,
) -> Path:
    root = _leaf_path_v96(work_root)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("V96 resume work root must be an unlinked directory")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"update_{update:06d}"
    if destination.exists() or destination.is_symlink():
        existing = strict_json_v96(destination / "state.json")
        if (
            existing.get("update") == update
            and existing.get("row_cursor") == row_cursor
            and existing.get("fresh_state_sha256")
            == collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        ):
            return destination
        raise FileExistsError(f"V96 checkpoint collision: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=root))
    try:
        optimizer_tensors, groups = _optimizer_tensors_v96(optimizer)
        weights = temporary / "state.safetensors"
        save_file(
            {**_fresh_state_v96(collection), **optimizer_tensors},
            str(weights),
            metadata={
                "artifact": CHECKPOINT_ARTIFACT,
                "environmental_memory_serialized": "false",
                "questions_or_answers_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        state = {
            "artifact": CHECKPOINT_ARTIFACT,
            "schema_version": 96,
            "status": "resumable_training_state",
            "update": update,
            "row_cursor": row_cursor,
            "row_order_sha256": row_order_sha256,
            "fresh_state_sha256": collection.bank(FRESH_BANK_NAME).installation.state_sha256(),
            "tensor_file_sha256": sha256_file_v85(weights),
            "optimizer_param_groups": groups,
            "history": list(history),
            "bindings": dict(bindings),
            "environmental_memory_serialized": False,
            "questions_or_answers_serialized": False,
            "oracle_serialized": False,
        }
        (temporary / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def discover_resume_checkpoint_v96(
    work_root: str | Path,
    *,
    bindings: Mapping[str, Any],
    row_order_sha256: str,
    gradient_accumulation_rows: int,
) -> tuple[Path, dict[str, Any]] | None:
    if gradient_accumulation_rows != 8:
        raise ValueError("V96 resume accumulation changed")
    root = _leaf_path_v96(work_root)
    if root.is_symlink():
        raise ValueError("V96 resume work root may not be a symlink")
    if not root.exists():
        return None
    if not root.is_dir():
        raise ValueError("V96 resume work root must be a directory")
    checkpoint_paths = set(root.glob("update_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if set(root.iterdir()) != checkpoint_paths:
        raise ValueError("V96 resume work-root file inventory changed")
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in checkpoint_paths:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"V96 resume checkpoint must be an unlinked directory: {path}")
        state_path = path / "state.json"
        tensor_path = path / "state.safetensors"
        if state_path.is_symlink() or tensor_path.is_symlink():
            raise ValueError(f"V96 resume checkpoint files may not be symlinks: {path}")
        metadata = strict_json_v96(path / "state.json")
        raw_update = metadata.get("update")
        raw_row_cursor = metadata.get("row_cursor")
        scalar_types_valid = type(raw_update) is int and type(raw_row_cursor) is int
        update = raw_update if type(raw_update) is int else -1
        row_cursor = raw_row_cursor if type(raw_row_cursor) is int else -1
        history = metadata.get("history")
        groups = metadata.get("optimizer_param_groups")
        if {child.name for child in path.iterdir()} != {
            "state.json",
            "state.safetensors",
        }:
            raise ValueError(f"V96 resume checkpoint file inventory changed: {path}")
        archive = load_file(str(tensor_path), device="cpu")
        expected_fresh_keys = {"adapters.0.lora_a", "adapters.0.lora_b"}
        optimizer_keys = {key for key in archive if key.startswith("optimizer.")}
        archive_fresh_keys = set(archive) - optimizer_keys
        expected_optimizer_keys = {
            f"optimizer.{index}.{name}"
            for index in range(2)
            for name in ("step", "exp_avg", "exp_avg_sq")
        }
        fresh_state_hash = tensor_state_sha256(
            {key: archive[key] for key in expected_fresh_keys if key in archive}
        )
        tensor_shapes_valid = (
            all(
                archive[f"optimizer.{index}.step"].numel() == 1
                and archive[f"optimizer.{index}.exp_avg"].shape
                == archive[f"adapters.{index // 2}.lora_{'a' if index % 2 == 0 else 'b'}"].shape
                and archive[f"optimizer.{index}.exp_avg_sq"].shape
                == archive[f"adapters.{index // 2}.lora_{'a' if index % 2 == 0 else 'b'}"].shape
                for index in range(2)
            )
            if optimizer_keys == expected_optimizer_keys
            and archive_fresh_keys == expected_fresh_keys
            else False
        )
        optimizer_steps_valid = (
            all(
                archive[f"optimizer.{index}.step"].numel() == 1
                and float(archive[f"optimizer.{index}.step"].item()) == float(update)
                for index in range(2)
            )
            if optimizer_keys == expected_optimizer_keys
            else False
        )
        if (
            metadata.get("artifact") != CHECKPOINT_ARTIFACT
            or metadata.get("schema_version") != 96
            or metadata.get("status") != "resumable_training_state"
            or not scalar_types_valid
            or path.name != f"update_{update:06d}"
            or not 1 <= update <= EXPECTED_OPTIMIZER_UPDATES
            or update % 15 != 0
            or not 8 <= row_cursor <= EXPECTED_MICRO_STEPS
            or row_cursor != update * gradient_accumulation_rows
            or metadata.get("row_order_sha256") != row_order_sha256
            or metadata.get("bindings") != dict(bindings)
            or metadata.get("environmental_memory_serialized") is not False
            or metadata.get("questions_or_answers_serialized") is not False
            or metadata.get("oracle_serialized") is not False
            or metadata.get("tensor_file_sha256") != sha256_file_v85(tensor_path)
            or not isinstance(history, list)
            or len(history) != update
            or any(
                not isinstance(record, Mapping)
                or type(record.get("update")) is not int
                or type(record.get("row_cursor")) is not int
                or record.get("update") != index
                or record.get("row_cursor") != index * gradient_accumulation_rows
                or not _is_sha256_v96(record.get("state_sha256"))
                for index, record in enumerate(history, 1)
            )
            or history[-1].get("state_sha256") != metadata.get("fresh_state_sha256")
            or groups != EXPECTED_OPTIMIZER_PARAM_GROUPS
            or archive_fresh_keys != expected_fresh_keys
            or optimizer_keys != expected_optimizer_keys
            or fresh_state_hash != metadata.get("fresh_state_sha256")
            or not tensor_shapes_valid
            or not optimizer_steps_valid
        ):
            raise ValueError(f"V96 resume checkpoint authentication failed: {path}")
        candidates.append((update, path, metadata))
    if not candidates:
        return None
    _update, path, metadata = max(candidates, key=lambda value: value[0])
    return path, metadata


def restore_resume_checkpoint_v96(
    checkpoint: Path,
    metadata: Mapping[str, Any],
    collection: LoRABankCollection,
    optimizer: torch.optim.Optimizer,
) -> None:
    archive = load_file(str(checkpoint / "state.safetensors"), device="cpu")
    fresh_keys = set(collection.bank(FRESH_BANK_NAME).installation.state_module.state_dict())
    _load_fresh_state_v96(
        collection, {name: archive[name] for name in fresh_keys if name in archive}
    )
    if (
        collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        != metadata["fresh_state_sha256"]
    ):
        raise ValueError("V96 resumed bridge hash changed")
    groups = metadata.get("optimizer_param_groups")
    if not isinstance(groups, list):
        raise TypeError("V96 resume optimizer groups are missing")
    _restore_optimizer_v96(optimizer, archive, groups)


def publish_fixed_final_candidate_v96(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Create once and publish only V96's two fresh adapter tensors."""

    root = _leaf_path_v96(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V96 create-once fixed-final candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / WEIGHTS_FILENAME
        state = _fresh_state_v96(collection)
        save_file(
            state,
            str(weights),
            metadata={
                "artifact": CANDIDATE_ARTIFACT,
                "environmental_memory_serialized": "false",
                "questions_or_answers_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        fresh = collection.bank(FRESH_BANK_NAME).installation
        if (
            fresh.target_names != TARGET_MODULES
            or fresh.parameter_count != FRESH_PARAMETER_COUNT
            or len(state) != 2
        ):
            raise ValueError("V96 candidate fresh-bank topology changed")
        metadata = {
            "artifact": CANDIDATE_ARTIFACT,
            "schema_version": 96,
            "status": "fixed_final_awaiting_known_development_gate",
            "parent": "v95_fixed_final_nonpromoted_optimization_parent",
            "bank_name": FRESH_BANK_NAME,
            "target_modules": list(TARGET_MODULES),
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "parameter_count": fresh.parameter_count,
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v85(weights),
            "tensor_inventory": sorted(state),
            "environmental_memory_serialized": False,
            "questions_or_answers_serialized": False,
            "oracle_serialized": False,
            "known_development_scored": False,
            "deferred_final_generated": False,
            "runtime_promotion_authorized": False,
            "bindings": dict(bindings),
        }
        (temporary / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, root)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_fixed_final_bridge_v96(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    root = _leaf_path_v96(candidate)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {child.name for child in root.iterdir()} != {WEIGHTS_FILENAME, METADATA_FILENAME}
    ):
        raise ValueError("V96 fixed-final candidate file inventory changed")
    metadata = strict_json_v96(root / METADATA_FILENAME)
    weights = root / WEIGHTS_FILENAME
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("V96 fixed-final candidate weights are absent or linked")
    fresh = collection.bank(FRESH_BANK_NAME).installation
    expected_inventory = sorted(fresh.state_module.state_dict())
    if (
        metadata.get("artifact") != CANDIDATE_ARTIFACT
        or metadata.get("status") != "fixed_final_awaiting_known_development_gate"
        or metadata.get("parent") != "v95_fixed_final_nonpromoted_optimization_parent"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("target_modules") != list(TARGET_MODULES)
        or metadata.get("parameter_count") != FRESH_PARAMETER_COUNT
        or metadata.get("tensor_inventory") != expected_inventory
        or metadata.get("weights_sha256") != sha256_file_v85(weights)
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("known_development_scored") is not False
        or metadata.get("deferred_final_generated") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 fixed-final candidate authentication failed")
    _load_fresh_state_v96(collection, load_file(str(weights), device="cpu"))
    if fresh.state_sha256() != metadata["state_sha256"]:
        raise ValueError("V96 fixed-final candidate state changed")
    return metadata


def finalize_fixed_final_candidate_v96(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Publish once or reuse an exact candidate after an interrupted report write."""

    root = _leaf_path_v96(destination)
    if not root.exists() and not root.is_symlink():
        return (
            publish_fixed_final_candidate_v96(
                root,
                collection,
                bindings=bindings,
            ),
            False,
        )
    expected_state = collection.bank(FRESH_BANK_NAME).installation.state_sha256()
    metadata = load_fixed_final_bridge_v96(collection, root)
    if metadata.get("bindings") != dict(bindings) or metadata.get("state_sha256") != expected_state:
        raise ValueError("V96 existing fixed-final candidate bindings changed")
    return metadata, True


def _schedule_hashes_v96(
    schedule: Sequence[TrainingStepV96],
    stable: Sequence[PairUnitV96],
) -> dict[str, str]:
    return {
        "schedule_sha256": canonical_sha256_v85([step.identity() for step in schedule]),
        "invariant_subset_sha256": canonical_sha256_v85(
            [[unit.pair_id, unit.question_key] for unit in stable]
        ),
    }


def _mean(records: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in records if bool(row.get(name + "_present"))]
    return None if not values else sum(values) / len(values)


def run_training_v96(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v96(config_path, allow_draft=False)
    audit = FileAccessAudit(
        forbidden_training_roots_v96(config),
        forbidden_component_names=frozenset(),
        block_forbidden=True,
    )
    with audit:
        result = _run_training_under_audit_v96(
            config_path=config_path,
            config=config,
            audit=audit,
            started=started,
        )
    audit.assert_clean()
    return result


def _run_training_under_audit_v96(
    *,
    config_path: str | Path,
    config: Mapping[str, Any],
    audit: FileAccessAudit,
    started: float,
) -> dict[str, Any]:
    """Execute the fixed 2,280-step lifecycle under one protected audit."""

    source_hashes = authenticate_training_sources_v96(config)
    preflight = authenticate_topology_smoke_v96(config, config_path=config_path)
    parent_evidence = authenticate_parent_v95_v96(config)
    assert_deferred_final_absent_v96(config)
    rows = load_training_rows_v96(config)
    changed_units, _invariant_units = pair_units_v96(rows)
    invariant_subset = invariant_subset_v96(rows)
    class_weights = balanced_class_weights_v96(config, rows)
    changed_family_weights = family_weights_v96(changed_units)
    invariant_family_weights = family_weights_v96(invariant_subset)
    training = config["training"]
    schedule = training_schedule_v96(rows, seed=int(training["schedule_seed"]))
    schedule_hashes = _schedule_hashes_v96(schedule, invariant_subset)
    if any(training[name] != value for name, value in schedule_hashes.items()):
        raise RuntimeError("V96 fixed schedule changed")
    step_counts = Counter(step.kind for step in schedule)
    if (
        len(schedule) != EXPECTED_MICRO_STEPS
        or step_counts["retention"] != EXPECTED_RETENTION_STEPS
        or step_counts["changed_pair"] != EXPECTED_CHANGED_PAIR_STEPS
        or step_counts["invariant_pair"] != EXPECTED_INVARIANT_PAIR_STEPS
    ):
        raise RuntimeError("V96 fixed schedule inventory changed")

    outputs = config["outputs"]
    report_path = _leaf_path_v96(outputs["training_report"])
    candidate_path = _leaf_path_v96(outputs["fixed_final_candidate"])
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError("V96 create-once fixed-final training report exists")

    # Bind all forty immutable memories before any question is tokenized.
    cpu_memories, memory_hashes_before = load_scene_memories_v96(config, rows)
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    language_config = runtime["language"]
    language = load_local_language_model(
        str(language_config["model_id"]),
        str(language_config["revision"]),
        str(language_config["dtype"]),
        freeze=True,
        local_files_only=True,
        backend="gemma4",
        decoder_gradient_checkpointing=True,
    )
    with torch.enable_grad():
        if language.device.type != "mps":
            raise RuntimeError("V96 full-model training requires local MPS")
        collection = install_lora_banks(language.model, combined_lora_settings_v96(runtime, config))
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V96 LoRA bank installation failed")
        frozen_source = load_frozen_parent_v96(collection, config)
        collection.assert_trainable_surface(language.model)
        fresh = collection.bank(FRESH_BANK_NAME).installation
        initial_state_sha256 = fresh.state_sha256()
        memory_by_scene = {
            scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
            for scene_id, memory in cpu_memories.items()
        }
        system_prompt = str(language_config["system_prompt"])
        parameters = collection.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        _unused_optimizer_tensors, optimizer_groups = _optimizer_tensors_v96(optimizer)
        if json.loads(json.dumps(optimizer_groups)) != EXPECTED_OPTIMIZER_PARAM_GROUPS:
            raise RuntimeError("V96 live AdamW parameter-group contract changed")
        accumulation = int(training["gradient_accumulation_steps"])
        bindings: dict[str, Any] = {
            **preflight,
            "trainer_source_sha256": source_hashes[str(config["sources"]["trainer_source"])],
            **schedule_hashes,
            "v95_parent_evidence_sha256": parent_evidence["v95_evidence_sha256"],
        }
        resumed = discover_resume_checkpoint_v96(
            outputs["work_root"],
            bindings=bindings,
            row_order_sha256=schedule_hashes["schedule_sha256"],
            gradient_accumulation_rows=accumulation,
        )
        history: list[dict[str, Any]] = []
        step_cursor = 0
        optimizer_update = 0
        resumed_from: str | None = None
        if resumed is not None:
            checkpoint, metadata = resumed
            restore_resume_checkpoint_v96(checkpoint, metadata, collection, optimizer)
            step_cursor = int(metadata["row_cursor"])
            optimizer_update = int(metadata["update"])
            history = list(metadata["history"])
            resumed_from = checkpoint.relative_to(PROJECT_ROOT).as_posix()
        if step_cursor > len(schedule) or step_cursor % accumulation:
            raise ValueError("V96 resume cursor is outside the fixed schedule")

        language.decoder_module.train()
        collection.train()
        optimizer.zero_grad(set_to_none=True)
        interval: list[dict[str, Any]] = []
        seen = Counter(step.kind for step in schedule[:step_cursor])
        nll_forwards = sum(
            1 if step.kind == "retention" else 4 if step.kind == "changed_pair" else 2
            for step in schedule[:step_cursor]
        )

        def answer_nll(memory: torch.Tensor, row: Any) -> torch.Tensor:
            prepared, _layout = _prepared_v84(language, system_prompt, memory, row)
            tail = _answer_tail(language, prepared)
            return tail.mean_nll.float()

        for cursor in range(step_cursor, len(schedule)):
            step = schedule[cursor]
            record: dict[str, Any] = {"kind": step.kind}
            if step.kind == "retention":
                if step.row is None:
                    raise RuntimeError("V96 retention step lost its row")
                correct_nll = answer_nll(memory_by_scene[step.row.scene_id], step.row)
                objective = (
                    float(training["retention_balanced_ce_weight"])
                    * float(class_weights[step.row.answer_class])
                    * correct_nll
                )
                record.update(
                    correct_nll=float(correct_nll.detach().cpu()),
                    answer_margin_penalty=0.0,
                    causal_margin_penalty=0.0,
                    invariant_consistency_penalty=0.0,
                )
                nll_forwards += 1
            elif step.kind == "changed_pair":
                if step.unit is None:
                    raise RuntimeError("V96 changed-pair step lost its unit")
                left = step.unit.left
                right = step.unit.right
                left_correct = answer_nll(memory_by_scene[left.scene_id], left)
                right_correct = answer_nll(memory_by_scene[right.scene_id], right)
                # Same question and memory, opposite side's exact canonical answer.
                left_alternative = answer_nll(
                    memory_by_scene[left.scene_id],
                    replace(
                        left,
                        answer=right.answer,
                        answer_class=right.answer_class,
                    ),
                )
                right_alternative = answer_nll(
                    memory_by_scene[right.scene_id],
                    replace(
                        right,
                        answer=left.answer,
                        answer_class=left.answer_class,
                    ),
                )
                objective, components = symmetric_pair_objective_v96(
                    left_correct,
                    right_correct,
                    left_alternative,
                    right_alternative,
                    left_class_weight=float(class_weights[left.answer_class]),
                    right_class_weight=float(class_weights[right.answer_class]),
                    family_weight=float(changed_family_weights[step.unit.change_type]),
                    correct_ce_weight=float(training["pair_correct_ce_weight"]),
                    answer_margin_weight=float(training["within_memory_answer_margin_weight"]),
                    answer_target_margin=float(training["within_memory_answer_target_margin_nll"]),
                    causal_margin_weight=float(training["across_memory_causal_margin_weight"]),
                    causal_target_margin=float(training["across_memory_causal_target_margin_nll"]),
                    smoothmax_temperature=float(training["pair_side_smoothmax_temperature"]),
                )
                record.update(
                    correct_nll=float((0.5 * (left_correct + right_correct)).detach().cpu()),
                    answer_margin_penalty=float(
                        components["answer_smoothmax_penalty"].detach().cpu()
                    ),
                    causal_margin_penalty=float(
                        components["causal_smoothmax_penalty"].detach().cpu()
                    ),
                    invariant_consistency_penalty=0.0,
                    family=step.unit.change_type,
                )
                nll_forwards += 4
            elif step.kind == "invariant_pair":
                if step.unit is None:
                    raise RuntimeError("V96 invariant-pair step lost its unit")
                left = step.unit.left
                right = step.unit.right
                left_nll = answer_nll(memory_by_scene[left.scene_id], left)
                right_nll = answer_nll(memory_by_scene[right.scene_id], right)
                objective, components = invariant_pair_objective_v96(
                    left_nll,
                    right_nll,
                    left_class_weight=float(class_weights[left.answer_class]),
                    right_class_weight=float(class_weights[right.answer_class]),
                    family_weight=float(invariant_family_weights[step.unit.change_type]),
                    correct_ce_weight=float(training["invariant_correct_ce_weight"]),
                    consistency_weight=float(training["invariant_nll_consistency_weight"]),
                    consistency_tolerance=float(training["invariant_nll_consistency_tolerance"]),
                )
                record.update(
                    correct_nll=float((0.5 * (left_nll + right_nll)).detach().cpu()),
                    answer_margin_penalty=0.0,
                    causal_margin_penalty=0.0,
                    invariant_consistency_penalty=float(
                        components["consistency_penalty"].detach().cpu()
                    ),
                    family=step.unit.change_type,
                )
                nll_forwards += 2
            else:
                raise RuntimeError(f"V96 unknown schedule kind: {step.kind}")

            if not torch.isfinite(objective):
                raise RuntimeError("V96 objective is nonfinite")
            record["objective"] = float(objective.detach().cpu())
            interval.append(record)
            (objective / accumulation).backward()
            step_cursor = cursor + 1
            seen[step.kind] += 1
            if step_cursor % accumulation:
                continue
            for parameter in parameters:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("V96 found a missing, NaN, or infinite gradient")
            gradient_l2 = float(collection.gradient_norms()["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V96 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            clip_l2 = float(clipped.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V96 clipped gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            history.append(
                {
                    "update": optimizer_update,
                    "row_cursor": step_cursor,
                    "mean_correct_nll": sum(float(v["correct_nll"]) for v in interval)
                    / len(interval),
                    "mean_objective": sum(float(v["objective"]) for v in interval) / len(interval),
                    "retention_steps": sum(v["kind"] == "retention" for v in interval),
                    "changed_pair_steps": sum(v["kind"] == "changed_pair" for v in interval),
                    "invariant_pair_steps": sum(v["kind"] == "invariant_pair" for v in interval),
                    "mean_answer_margin_penalty": sum(
                        float(v["answer_margin_penalty"]) for v in interval
                    )
                    / len(interval),
                    "mean_causal_margin_penalty": sum(
                        float(v["causal_margin_penalty"]) for v in interval
                    )
                    / len(interval),
                    "mean_invariant_consistency_penalty": sum(
                        float(v["invariant_consistency_penalty"]) for v in interval
                    )
                    / len(interval),
                    "gradient_l2_before_clip": gradient_l2,
                    "clip_return_l2": clip_l2,
                    "state_sha256": fresh.state_sha256(),
                }
            )
            interval.clear()
            if optimizer_update in {1, 71, 142, 213, 285} or optimizer_update % 15 == 0:
                print(
                    json.dumps(
                        {
                            "event": "v96_train_update",
                            "update": optimizer_update,
                            "total_updates": EXPECTED_OPTIMIZER_UPDATES,
                            "step_cursor": step_cursor,
                            "retention_seen": seen["retention"],
                            "changed_pairs_seen": seen["changed_pair"],
                            "invariant_pairs_seen": seen["invariant_pair"],
                            "nll_forwards": nll_forwards,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if (
                optimizer_update % int(training["checkpoint_every_optimizer_updates"]) == 0
                or optimizer_update == EXPECTED_OPTIMIZER_UPDATES
            ):
                save_resume_checkpoint_v96(
                    outputs["work_root"],
                    collection,
                    optimizer,
                    update=optimizer_update,
                    row_cursor=step_cursor,
                    history=history,
                    bindings=bindings,
                    row_order_sha256=schedule_hashes["schedule_sha256"],
                )
            torch.mps.empty_cache()

        if (
            optimizer_update != EXPECTED_OPTIMIZER_UPDATES
            or step_cursor != EXPECTED_MICRO_STEPS
            or seen["retention"] != EXPECTED_RETENTION_STEPS
            or seen["changed_pair"] != EXPECTED_CHANGED_PAIR_STEPS
            or seen["invariant_pair"] != EXPECTED_INVARIANT_PAIR_STEPS
            or nll_forwards != EXPECTED_TOTAL_NLL_FORWARDS
            or len(history) != EXPECTED_OPTIMIZER_UPDATES
        ):
            raise RuntimeError("V96 fixed atomic-pair schedule did not complete exactly")
        language.decoder_module.eval()
        collection.eval()
        memory_hashes_after = {
            scene_id: prefix_sha256(memory.detach().cpu())
            for scene_id, memory in memory_by_scene.items()
        }
        if memory_hashes_after != memory_hashes_before:
            raise RuntimeError("V96 training mutated immutable environmental inputs")
        candidate_bindings = {
            **bindings,
            "fixed_final_optimizer_updates": optimizer_update,
            "class_weight_inventory_sha256": config["training_pool"][
                "balanced_class_weight_inventory_sha256"
            ],
            "changed_family_weight_inventory_sha256": config["training_pool"][
                "changed_family_weight_inventory_sha256"
            ],
            "invariant_family_weight_inventory_sha256": config["training_pool"][
                "invariant_family_weight_inventory_sha256"
            ],
            "known_development_labels_opened": False,
            "known_development_questions_opened": False,
            "deferred_final_generated": False,
        }
        candidate_metadata, candidate_reused_after_interruption = (
            finalize_fixed_final_candidate_v96(
                candidate_path, collection, bindings=candidate_bindings
            )
        )
    audit.assert_clean()

    gates = {
        "all_960_rows_consumed_twice": seen["retention"] == EXPECTED_RETENTION_STEPS,
        "all_66_changed_units_consumed_four_times": seen["changed_pair"]
        == EXPECTED_CHANGED_PAIR_STEPS,
        "all_changed_pair_questions_byte_identical": all(
            unit.left.question.encode("utf-8") == unit.right.question.encode("utf-8")
            for unit in changed_units
        ),
        "all_96_invariant_units_consumed_once": seen["invariant_pair"]
        == EXPECTED_INVARIANT_PAIR_STEPS,
        "all_invariant_subset_questions_byte_identical": all(
            unit.left.question.encode("utf-8") == unit.right.question.encode("utf-8")
            for unit in invariant_subset
        ),
        "exact_3168_nll_forward_schedule": nll_forwards == EXPECTED_TOTAL_NLL_FORWARDS,
        "fixed_final_update_285_reached": optimizer_update == EXPECTED_OPTIMIZER_UPDATES,
        "only_fresh_45056_parameters_trainable": len(parameters) == 2
        and sum(parameter.numel() for parameter in parameters) == FRESH_PARAMETER_COUNT,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(record["gradient_l2_before_clip"]))
            and float(record["gradient_l2_before_clip"]) > 0.0
            for record in history
        ),
        "all_scene_hashes_invariant": memory_hashes_after == memory_hashes_before,
        "exact_canonical_list_answers_retained": sum(row.answer == "book, cube" for row in rows)
        == 22,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
        "deferred_final_still_absent": not assert_deferred_final_absent_v96(config)[
            "physical_artifacts_present"
        ],
    }
    if not all(gates.values()):
        raise RuntimeError(f"V96 training gate failed: {gates}")
    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 96,
        "status": "fixed_final_training_complete_not_promoted",
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "topology_smoke_sha256": preflight["topology_smoke_sha256"],
        "device": "mps",
        "model_id": config["sources"]["model_id"],
        "model_revision": config["sources"]["model_revision"],
        "strict_input_contract": config["strict_input_contract"],
        "source_hashes": source_hashes,
        "frozen_source": frozen_source,
        "trainable_bridge": {
            "bank_name": FRESH_BANK_NAME,
            "target_modules": list(TARGET_MODULES),
            "parameter_count": FRESH_PARAMETER_COUNT,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": candidate_metadata["state_sha256"],
            "parent": "v95_fixed_final_nonpromoted_optimization_parent",
            "unmerged": True,
        },
        "training_protocol": training,
        "schedule_hashes": schedule_hashes,
        "micro_steps_consumed": step_cursor,
        "unique_training_rows": len(rows),
        "training_scene_count": len(cpu_memories),
        "retention_steps_consumed": seen["retention"],
        "changed_pair_steps_consumed": seen["changed_pair"],
        "invariant_pair_steps_consumed": seen["invariant_pair"],
        "total_nll_forwards": nll_forwards,
        "optimizer_updates": optimizer_update,
        "resumed_from": resumed_from,
        "candidate_reused_after_interruption": candidate_reused_after_interruption,
        "training_history": history,
        "scene_memories": {
            "compiled_before_question_tokenization": True,
            "shape_each": [1, 738, 1536],
            "hashes_before": memory_hashes_before,
            "hashes_after": memory_hashes_after,
            "hash_invariant": True,
            "all_memory_slots_retained": True,
            "question_derived_environmental_tokens": 0,
            "question_dependent_retrieval": False,
        },
        "gates": gates,
        "candidate": {
            "path": candidate_path.relative_to(PROJECT_ROOT).as_posix(),
            "weights_sha256": candidate_metadata["weights_sha256"],
            "metadata_canonical_sha256": canonical_sha256_v85(candidate_metadata),
            "fixed_final": True,
            "known_development_scored": False,
            "runtime_promotion_authorized": False,
        },
        "loaded_file_count": len(audit.unique_paths),
        "loaded_file_inventory_sha256": canonical_sha256_v85(audit.unique_paths),
        "protected_read_count": len(audit.forbidden_accesses()),
        "known_development_labels_loaded": False,
        "known_development_questions_loaded": False,
        "deferred_final_generated": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_create_json_v85(report_path, report)
    return report


def authenticate_training_report_v96(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    preflight = authenticate_topology_smoke_v96(config, config_path=config_path)
    path = _leaf_path_v96(config["outputs"]["training_report"])
    report = strict_json_v96(path)
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("topology_smoke_sha256") != preflight["topology_smoke_sha256"]
        or report.get("optimizer_updates") != EXPECTED_OPTIMIZER_UPDATES
        or report.get("micro_steps_consumed") != EXPECTED_MICRO_STEPS
        or report.get("retention_steps_consumed") != EXPECTED_RETENTION_STEPS
        or report.get("changed_pair_steps_consumed") != EXPECTED_CHANGED_PAIR_STEPS
        or report.get("invariant_pair_steps_consumed") != EXPECTED_INVARIANT_PAIR_STEPS
        or report.get("total_nll_forwards") != EXPECTED_TOTAL_NLL_FORWARDS
        or report.get("protected_read_count") != 0
        or report.get("known_development_labels_loaded") is not False
        or report.get("known_development_questions_loaded") is not False
        or report.get("deferred_final_generated") is not False
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 fixed-final training report changed")
    return {**preflight, "training_report_sha256": sha256_file_v85(path)}


def run_topology_smoke_v96(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Prove the sealed topology against Gemma's config without loading weights."""

    config = load_config_v96(config_path, allow_draft=False)
    preflight = authenticate_cpu_preflight_v96(config, config_path=config_path)
    initial_outputs = assert_initial_outputs_absent_v96(config)
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    settings = combined_lora_settings_v96(runtime, config)
    parent = authenticate_parent_v95_v96(config)
    fresh = settings.bank(FRESH_BANK_NAME)
    synthetic = lora_preflight_v96(config)
    from transformers import AutoConfig, Gemma4ForConditionalGeneration

    sources = config["sources"]
    model_config = AutoConfig.from_pretrained(
        str(sources["model_id"]),
        revision=str(sources["model_revision"]),
        local_files_only=True,
    )
    with torch.device("meta"):
        topology_model = Gemma4ForConditionalGeneration(model_config)
    observed_shapes: dict[str, list[int]] = {}
    observed_linear_targets = True
    for target in TARGET_MODULES:
        module = topology_model.get_submodule(target)
        observed_linear_targets = observed_linear_targets and isinstance(module, torch.nn.Linear)
        observed_shapes[target + ".weight"] = list(module.weight.shape)
    del topology_model
    checks = {
        "full_gemma_model_loaded_false": True,
        "pinned_gemma_targets_are_linear": observed_linear_targets,
        "pinned_gemma_target_shapes_exact": observed_shapes
        == config["bridge"]["pinned_weight_shapes"],
        "exact_ten_bank_stack": len(settings.banks) == 10,
        "exact_nine_frozen_banks": sum(not bank.trainable for bank in settings.banks) == 9,
        "sole_fresh_bank_trainable": sum(bank.trainable for bank in settings.banks) == 1,
        "fresh_targets_exact": fresh.adapter.target_modules == TARGET_MODULES,
        "fresh_parameter_count_exact": synthetic["parameter_count"] == FRESH_PARAMETER_COUNT,
        "fresh_initial_state_exact": synthetic["initial_state_sha256"]
        == EXPECTED_INITIAL_STATE_SHA256,
        "fresh_output_starts_zero": synthetic["exact_zero_output_at_initialization"] is True,
        "failed_v95_parent_authenticated": parent["v95_known_development_gate_passed"] is False,
        "candidate_work_and_report_absent": all(
            initial_outputs[key] is True
            for key in (
                "work_root_absent",
                "fixed_final_candidate_absent",
                "training_report_absent",
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V96 CPU topology smoke failed: {checks}")
    report = {
        "artifact": TOPOLOGY_ARTIFACT,
        "schema_version": 96,
        "status": "passed_meta_device_no_weight_load",
        "passed": True,
        "config_sha256": preflight["config_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "checks": checks,
        "initial_output_absence": initial_outputs,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
    }
    output = _leaf_path_v96(config["outputs"]["topology_smoke"])
    atomic_create_json_v85(output, report)
    return {**report, "output": output.relative_to(PROJECT_ROOT).as_posix()}


def authenticate_topology_smoke_v96(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    preflight = authenticate_cpu_preflight_v96(config, config_path=config_path)
    path = _leaf_path_v96(config["outputs"]["topology_smoke"])
    report = strict_json_v96(path)
    checks = report.get("checks")
    absence = report.get("initial_output_absence")
    if (
        report.get("artifact") != TOPOLOGY_ARTIFACT
        or report.get("schema_version") != 96
        or report.get("status") != "passed_meta_device_no_weight_load"
        or report.get("passed") is not True
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(absence, Mapping)
        or any(
            absence.get(key) is not True
            for key in (
                "work_root_absent",
                "fixed_final_candidate_absent",
                "training_report_absent",
            )
        )
        or report.get("full_gemma_model_loaded") is not False
        or report.get("optimizer_constructed") is not False
        or report.get("training_started") is not False
    ):
        raise ValueError("V96 topology-smoke evidence changed")
    return {**preflight, "topology_smoke_sha256": sha256_file_v85(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--topology-smoke", action="store_true")
    parser.add_argument("--authenticate-topology-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.topology_smoke and args.authenticate_topology_smoke:
        parser.error("choose only one topology-smoke action")
    if args.topology_smoke:
        result = run_topology_smoke_v96(args.config)
    elif args.authenticate_topology_smoke:
        config = load_config_v96(args.config, allow_draft=False)
        result = authenticate_topology_smoke_v96(config, config_path=args.config)
    else:
        result = run_training_v96(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_ARTIFACT",
    "CHECKPOINT_ARTIFACT",
    "EXPECTED_CHANGED_PAIR_STEPS",
    "EXPECTED_FROZEN_BANK_COUNT",
    "EXPECTED_INVARIANT_PAIR_STEPS",
    "EXPECTED_MICRO_STEPS",
    "EXPECTED_OPTIMIZER_UPDATES",
    "EXPECTED_RETENTION_STEPS",
    "EXPECTED_TOTAL_NLL_FORWARDS",
    "METADATA_FILENAME",
    "TRAINING_ARTIFACT",
    "WEIGHTS_FILENAME",
    "authenticate_topology_smoke_v96",
    "authenticate_training_report_v96",
    "combined_lora_settings_v96",
    "discover_resume_checkpoint_v96",
    "finalize_fixed_final_candidate_v96",
    "invariant_pair_objective_v96",
    "load_fixed_final_bridge_v96",
    "load_frozen_parent_v96",
    "main",
    "publish_fixed_final_candidate_v96",
    "restore_resume_checkpoint_v96",
    "run_topology_smoke_v96",
    "run_training_v96",
    "save_resume_checkpoint_v96",
    "smoothmax_v96",
    "strict_json_v96",
    "symmetric_pair_objective_v96",
]
