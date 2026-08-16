"""Train V95's causal strict-scene bridge without touching held-out labels.

The exact V85 seven-bank runtime and V94's failed, non-promoted fixed-final
bridge are frozen.  Only a fresh rank-8 bank on the latest disjoint full
attention K/V (layer 9) and final-layer MLP up-proj is optimized.  Every
environment remains the precompiled 738-token continuous memory; neither
training nor checkpoint files serialize environmental text.
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
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import (
    EXPECTED_BANKS as V94_BANKS,
)
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import (
    V94_STATE_SHA256,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
    canonical_sha256_v85,
    resolve_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v94_strict_multiscene_preflight import (
    TARGET_MODULE as V94_TARGET_MODULE,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    CONFIG,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT,
    FRESH_BANK_NAME,
    FRESH_PARAMETER_COUNT,
    TARGET_MODULES,
    assert_deferred_final_absent_v95,
    authenticate_cpu_preflight_v95,
    authenticate_parent_v94_v95,
    authenticate_training_sources_v95,
    balanced_class_weights_v95,
    causal_control_schedule_v95,
    cross_scene_schedule_v95,
    forbidden_training_roots_v95,
    load_config_v95,
    load_scene_memories_v95,
    load_training_rows_v95,
    lora_preflight_v95,
    permuted_payload_memory_v95,
    training_schedule_v95,
    zero_payload_memory_v95,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
    lora_banks_settings,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_v84_strict_bridge import _prepared_v84

TRAINING_ARTIFACT: Final[str] = "gemma4_v95_strict_causal_successor_training_v1"
CANDIDATE_ARTIFACT: Final[str] = "gemma4_v95_strict_causal_successor_fixed_final_v1"
CHECKPOINT_ARTIFACT: Final[str] = "gemma4_v95_strict_causal_successor_resume_v1"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
EXPECTED_V85_BANK_COUNT: Final[int] = 7
EXPECTED_FROZEN_BANK_COUNT: Final[int] = 8
EXPECTED_V85_PARAMETER_COUNT: Final[int] = 565_248
EXPECTED_FROZEN_PARAMETER_COUNT: Final[int] = 675_840
EXPECTED_ROWS_PER_EPOCH: Final[int] = 960
EXPECTED_EPOCHS: Final[int] = 4
EXPECTED_MICRO_ROWS: Final[int] = 3_840
EXPECTED_OPTIMIZER_UPDATES: Final[int] = 480
EXPECTED_WRONG_MEMORY_ROWS: Final[int] = 996
EXPECTED_ZERO_ROWS: Final[int] = 500
EXPECTED_PERMUTATION_ROWS: Final[int] = 500
EXPECTED_TOTAL_NLL_FORWARDS: Final[int] = 5_836


def strict_json_v95(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V95 JSON must contain one object: {source}")
    return value


def combined_lora_settings_v95(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    """Append frozen V94 and fresh V95 banks to the exact V85 stack."""

    base = lora_banks_settings(runtime_config)
    if (
        len(base.banks) != EXPECTED_V85_BANK_COUNT
        or any(bank.trainable for bank in base.banks)
        or tuple(bank.name for bank in base.banks) != V94_BANKS[:-1]
    ):
        raise ValueError("V95 requires the exact seven frozen V85 banks")
    frozen = experiment["frozen_stack"]
    v94 = LoRABankSettings(
        name=str(frozen["v94_bank_name"]),
        trainable=False,
        adapter=LoRASettings(
            enabled=True,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_modules=(V94_TARGET_MODULE,),
        ),
        initialization_algorithm="checkpoint_overwrite",
        expected_initial_state_sha256=str(frozen["v94_bank_state_sha256"]),
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
    result = LoRABanksSettings(base.banks + (v94, fresh))
    if (
        len(result.banks) != 9
        or sum(bank.trainable for bank in result.banks) != 1
        or result.banks[-1].name != FRESH_BANK_NAME
    ):
        raise RuntimeError("V95 exact nine-bank topology construction failed")
    return result


def _load_v85_banks_v95(
    collection: LoRABankCollection,
    checkpoint: str | Path,
) -> dict[str, Any]:
    root = resolve_v85(checkpoint)
    weights = root / "adapter.safetensors"
    metadata_path = root / "runtime_metadata.json"
    metadata = strict_json_v95(metadata_path)
    hashes = metadata.get("lora_bank_state_sha256")
    modules = metadata.get("lora_bank_wrapped_modules")
    counts = metadata.get("lora_bank_parameter_counts")
    if not all(isinstance(value, Mapping) for value in (hashes, modules, counts)):
        raise TypeError("V95 V85 substrate lacks named LoRA metadata")
    v85_banks = collection.banks[:EXPECTED_V85_BANK_COUNT]
    if (
        tuple(bank.settings.name for bank in v85_banks) != V94_BANKS[:-1]
        or set(hashes) != set(V94_BANKS[:-1])
        or metadata.get("lora_parameter_count") != EXPECTED_V85_PARAMETER_COUNT
        or metadata.get("lora_trainable_parameter_count") != 0
    ):
        raise ValueError("V95 V85 substrate contract changed")
    archive = load_file(str(weights), device="cpu")
    expected_keys: set[str] = set()
    for bank in v85_banks:
        name = bank.settings.name
        prefix = f"lora_banks.{name}."
        state_keys = set(bank.installation.state_module.state_dict())
        expected_keys.update(prefix + key for key in state_keys)
        state = {
            key[len(prefix) :]: value for key, value in archive.items() if key.startswith(prefix)
        }
        bank.installation.state_module.load_state_dict(state, strict=True)
        if (
            list(bank.installation.target_names) != modules[name]
            or bank.installation.parameter_counts != counts[name]
            or bank.installation.state_sha256() != hashes[name]
        ):
            raise ValueError(f"V95 frozen V85 bank changed: {name}")
    if {key for key in archive if key.startswith("lora_banks.")} != expected_keys:
        raise ValueError("V95 V85 tensor inventory changed")
    return {
        "adapter_sha256": sha256_file_v85(weights),
        "metadata_sha256": sha256_file_v85(metadata_path),
        "bank_state_sha256": dict(hashes),
    }


def _load_v94_bank_v95(
    collection: LoRABankCollection,
    checkpoint: str | Path,
) -> dict[str, Any]:
    root = resolve_v85(checkpoint)
    weights = root / WEIGHTS_FILENAME
    metadata_path = root / METADATA_FILENAME
    metadata = strict_json_v95(metadata_path)
    archive = load_file(str(weights), device="cpu")
    bank = collection.bank(V94_BANKS[-1])
    state = {f"adapters.0.{name}": value for name, value in archive.items()}
    if (
        metadata.get("artifact") != "gemma4_v94_strict_multiscene_full40_fixed_final_v1"
        or metadata.get("status") != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("runtime_promotion_authorized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("target_module") != V94_TARGET_MODULE
        or metadata.get("state_sha256") != V94_STATE_SHA256
        or metadata.get("parameter_count") != 110_592
        or set(archive) != {"lora_a", "lora_b"}
        or set(state) != set(bank.installation.state_module.state_dict())
    ):
        raise ValueError("V95 frozen V94 bridge metadata or tensors changed")
    bank.installation.state_module.load_state_dict(state, strict=True)
    if bank.installation.state_sha256() != V94_STATE_SHA256:
        raise ValueError("V95 frozen V94 bridge state changed")
    return {
        "weights_sha256": sha256_file_v85(weights),
        "metadata_sha256": sha256_file_v85(metadata_path),
        "state_sha256": bank.installation.state_sha256(),
    }


def load_frozen_parent_v95(
    collection: LoRABankCollection,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load seven V85 banks and the non-promoted V94 fixed-final bank."""

    if (
        collection.bank_names != V94_BANKS + (FRESH_BANK_NAME,)
        or len([bank for bank in collection.banks if not bank.settings.trainable])
        != EXPECTED_FROZEN_BANK_COUNT
    ):
        raise ValueError("V95 installed bank order changed")
    sources = config["sources"]
    v85 = _load_v85_banks_v95(collection, sources["frozen_v85_checkpoint"])
    v94 = _load_v94_bank_v95(collection, sources["frozen_v94_fixed_final"])
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if (
        fresh.target_names != TARGET_MODULES
        or fresh.parameter_count != FRESH_PARAMETER_COUNT
        or fresh.state_sha256() != EXPECTED_INITIAL_STATE_SHA256
        or any(torch.count_nonzero(adapter.lora_b).item() for adapter in fresh.adapters)
        or collection.parameter_count != EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT
        or collection.trainable_parameter_count != FRESH_PARAMETER_COUNT
    ):
        raise ValueError("V95 fresh bank did not begin at exact zero output")
    collection.validate_state()
    return {
        "parent": "fixed_final_nonpromoted_optimization_parent",
        "v85": v85,
        "v94": v94,
        "frozen_bank_count": EXPECTED_FROZEN_BANK_COUNT,
        "frozen_parameter_count": EXPECTED_FROZEN_PARAMETER_COUNT,
        "v94_behavior_gate_passed": False,
        "runtime_release_loaded": False,
    }


def causal_objective_v95(
    correct_nll: torch.Tensor,
    *,
    class_weight: float,
    balanced_ce_weight: float,
    wrong_memory_nll: torch.Tensor | None = None,
    wrong_margin_weight: float = 0.0,
    wrong_target_margin: float = 0.0,
    zero_payload_nll: torch.Tensor | None = None,
    zero_margin_weight: float = 0.0,
    zero_target_margin: float = 0.0,
    permutation_nll: torch.Tensor | None = None,
    permutation_margin_weight: float = 0.0,
    permutation_target_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted answer CE plus three independent causal-memory margins."""

    if correct_nll.ndim != 0 or not torch.isfinite(correct_nll):
        raise ValueError("V95 correct NLL must be a finite scalar")
    numbers = (
        class_weight,
        balanced_ce_weight,
        wrong_margin_weight,
        wrong_target_margin,
        zero_margin_weight,
        zero_target_margin,
        permutation_margin_weight,
        permutation_target_margin,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in numbers):
        raise ValueError("V95 objective weights and margins must be finite nonnegative")
    objective = float(balanced_ce_weight) * float(class_weight) * correct_nll
    zero = correct_nll.detach().new_zeros(())
    records: dict[str, torch.Tensor] = {}
    controls = (
        ("wrong_memory", wrong_memory_nll, wrong_margin_weight, wrong_target_margin),
        ("zero_payload", zero_payload_nll, zero_margin_weight, zero_target_margin),
        (
            "permutation",
            permutation_nll,
            permutation_margin_weight,
            permutation_target_margin,
        ),
    )
    for name, control_nll, weight, target in controls:
        records[f"{name}_minus_correct_nll"] = zero
        records[f"{name}_margin_penalty"] = zero
        if control_nll is None:
            continue
        if control_nll.ndim != 0 or not torch.isfinite(control_nll):
            raise ValueError(f"V95 {name} NLL must be a finite scalar")
        records[f"{name}_minus_correct_nll"] = control_nll - correct_nll
        records[f"{name}_margin_penalty"] = torch.relu(correct_nll - control_nll + float(target))
        objective = objective + float(weight) * records[f"{name}_margin_penalty"]
    return objective, records


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _fresh_state_v95(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if len(fresh.adapters) != 3:
        raise ValueError("V95 fresh bank must wrap exactly three modules")
    return {
        name: value.detach().cpu().contiguous()
        for name, value in fresh.state_module.state_dict().items()
    }


def _load_fresh_state_v95(
    collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]
) -> None:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    expected = set(fresh.state_module.state_dict())
    if set(archive) != expected:
        raise ValueError("V95 fresh-bank tensor inventory changed")
    fresh.state_module.load_state_dict(dict(archive), strict=True)
    fresh.validate_state()


def _optimizer_tensors_v95(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    state = optimizer.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for parameter_index, values in state["state"].items():
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("V95 AdamW state must be tensor-only")
            tensors[f"optimizer.{parameter_index}.{name}"] = value.detach().cpu().contiguous()
    groups: list[dict[str, Any]] = []
    for group in state["param_groups"]:
        normalized = dict(group)
        normalized["params"] = [int(value) for value in normalized["params"]]
        groups.append(normalized)
    return tensors, groups


def _restore_optimizer_v95(
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


def save_resume_checkpoint_v95(
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
    root = resolve_v85(work_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"update_{update:06d}"
    if destination.exists() or destination.is_symlink():
        existing = strict_json_v95(destination / "state.json")
        if (
            existing.get("update") == update
            and existing.get("row_cursor") == row_cursor
            and existing.get("fresh_state_sha256")
            == collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        ):
            return destination
        raise FileExistsError(f"V95 checkpoint collision: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=root))
    try:
        optimizer_tensors, groups = _optimizer_tensors_v95(optimizer)
        weights = temporary / "state.safetensors"
        save_file(
            {**_fresh_state_v95(collection), **optimizer_tensors},
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
            "schema_version": 95,
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


def discover_resume_checkpoint_v95(
    work_root: str | Path,
    *,
    bindings: Mapping[str, Any],
    row_order_sha256: str,
    gradient_accumulation_rows: int,
) -> tuple[Path, dict[str, Any]] | None:
    root = resolve_v85(work_root)
    if not root.exists():
        return None
    checkpoint_paths = set(root.glob("update_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if set(root.iterdir()) != checkpoint_paths:
        raise ValueError("V95 resume work-root file inventory changed")
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in checkpoint_paths:
        if path.is_symlink() or not path.is_dir():
            continue
        metadata = strict_json_v95(path / "state.json")
        update = int(metadata.get("update", -1))
        row_cursor = int(metadata.get("row_cursor", -1))
        history = metadata.get("history")
        groups = metadata.get("optimizer_param_groups")
        tensor_path = path / "state.safetensors"
        if {child.name for child in path.iterdir()} != {
            "state.json",
            "state.safetensors",
        }:
            raise ValueError(f"V95 resume checkpoint file inventory changed: {path}")
        archive = load_file(str(tensor_path), device="cpu")
        expected_fresh_keys = {
            "adapters.0.lora_a",
            "adapters.0.lora_b",
            "adapters.1.lora_a",
            "adapters.1.lora_b",
            "adapters.2.lora_a",
            "adapters.2.lora_b",
        }
        optimizer_keys = {key for key in archive if key.startswith("optimizer.")}
        archive_fresh_keys = set(archive) - optimizer_keys
        expected_optimizer_keys = {
            f"optimizer.{index}.{name}"
            for index in range(6)
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
                for index in range(6)
            )
            if optimizer_keys == expected_optimizer_keys
            and archive_fresh_keys == expected_fresh_keys
            else False
        )
        if (
            metadata.get("artifact") != CHECKPOINT_ARTIFACT
            or update < 1
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
                or record.get("update") != index
                or record.get("row_cursor") != index * gradient_accumulation_rows
                for index, record in enumerate(history, 1)
            )
            or not isinstance(groups, list)
            or len(groups) != 1
            or len(groups[0].get("params", ())) != 6
            or archive_fresh_keys != expected_fresh_keys
            or optimizer_keys != expected_optimizer_keys
            or fresh_state_hash != metadata.get("fresh_state_sha256")
            or not tensor_shapes_valid
        ):
            raise ValueError(f"V95 resume checkpoint authentication failed: {path}")
        candidates.append((update, path, metadata))
    if not candidates:
        return None
    _update, path, metadata = max(candidates, key=lambda value: value[0])
    return path, metadata


def restore_resume_checkpoint_v95(
    checkpoint: Path,
    metadata: Mapping[str, Any],
    collection: LoRABankCollection,
    optimizer: torch.optim.Optimizer,
) -> None:
    archive = load_file(str(checkpoint / "state.safetensors"), device="cpu")
    fresh_keys = set(collection.bank(FRESH_BANK_NAME).installation.state_module.state_dict())
    _load_fresh_state_v95(
        collection, {name: archive[name] for name in fresh_keys if name in archive}
    )
    if (
        collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        != metadata["fresh_state_sha256"]
    ):
        raise ValueError("V95 resumed bridge hash changed")
    groups = metadata.get("optimizer_param_groups")
    if not isinstance(groups, list):
        raise TypeError("V95 resume optimizer groups are missing")
    _restore_optimizer_v95(optimizer, archive, groups)


def publish_fixed_final_candidate_v95(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Create once and publish only V95's six fresh adapter tensors."""

    root = resolve_v85(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V95 create-once fixed-final candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / WEIGHTS_FILENAME
        state = _fresh_state_v95(collection)
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
            or len(state) != 6
        ):
            raise ValueError("V95 candidate fresh-bank topology changed")
        metadata = {
            "artifact": CANDIDATE_ARTIFACT,
            "schema_version": 95,
            "status": "fixed_final_awaiting_known_development_gate",
            "parent": "fixed_final_nonpromoted_optimization_parent",
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


def load_fixed_final_bridge_v95(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    root = resolve_v85(candidate)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {child.name for child in root.iterdir()} != {WEIGHTS_FILENAME, METADATA_FILENAME}
    ):
        raise ValueError("V95 fixed-final candidate file inventory changed")
    metadata = strict_json_v95(root / METADATA_FILENAME)
    weights = root / WEIGHTS_FILENAME
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("V95 fixed-final candidate weights are absent or linked")
    fresh = collection.bank(FRESH_BANK_NAME).installation
    expected_inventory = sorted(fresh.state_module.state_dict())
    if (
        metadata.get("artifact") != CANDIDATE_ARTIFACT
        or metadata.get("status") != "fixed_final_awaiting_known_development_gate"
        or metadata.get("parent") != "fixed_final_nonpromoted_optimization_parent"
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
        raise ValueError("V95 fixed-final candidate authentication failed")
    _load_fresh_state_v95(collection, load_file(str(weights), device="cpu"))
    if fresh.state_sha256() != metadata["state_sha256"]:
        raise ValueError("V95 fixed-final candidate state changed")
    return metadata


def finalize_fixed_final_candidate_v95(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Publish once or reuse an exact candidate after an interrupted report write."""

    root = resolve_v85(destination)
    if not root.exists() and not root.is_symlink():
        return (
            publish_fixed_final_candidate_v95(
                root,
                collection,
                bindings=bindings,
            ),
            False,
        )
    expected_state = collection.bank(FRESH_BANK_NAME).installation.state_sha256()
    metadata = load_fixed_final_bridge_v95(collection, root)
    if metadata.get("bindings") != dict(bindings) or metadata.get("state_sha256") != expected_state:
        raise ValueError("V95 existing fixed-final candidate bindings changed")
    return metadata, True


def _schedule_hashes_v95(
    main: Sequence[tuple[int, Any]],
    wrong: Sequence[tuple[int, Any, Any]],
    zero: Sequence[tuple[int, Any]],
    permutation: Sequence[tuple[int, Any]],
) -> dict[str, str]:
    return {
        "row_order_sha256": canonical_sha256_v85(
            [[epoch, row.scene_id, row.question_id] for epoch, row in main]
        ),
        "cross_scene_schedule_sha256": canonical_sha256_v85(
            [
                [
                    epoch,
                    row.scene_id,
                    row.question_id,
                    wrong_row.scene_id,
                    wrong_row.question_id,
                ]
                for epoch, row, wrong_row in wrong
            ]
        ),
        "zero_payload_schedule_sha256": canonical_sha256_v85(
            [[epoch, row.scene_id, row.question_id] for epoch, row in zero]
        ),
        "permutation_control_schedule_sha256": canonical_sha256_v85(
            [[epoch, row.scene_id, row.question_id] for epoch, row in permutation]
        ),
    }


def _mean(records: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in records if bool(row.get(name + "_present"))]
    return None if not values else sum(values) / len(values)


def run_training_v95(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v95(config_path, allow_draft=False)
    audit = FileAccessAudit(
        forbidden_training_roots_v95(config),
        forbidden_component_names=frozenset(),
        block_forbidden=True,
    )
    with audit:
        result = _run_training_under_audit_v95(
            config_path=config_path,
            config=config,
            audit=audit,
            started=started,
        )
    audit.assert_clean()
    return result


def _run_training_under_audit_v95(
    *,
    config_path: str | Path,
    config: Mapping[str, Any],
    audit: FileAccessAudit,
    started: float,
) -> dict[str, Any]:
    """Execute the complete read/write lifecycle under one protected audit."""

    source_hashes = authenticate_training_sources_v95(config)
    preflight = authenticate_cpu_preflight_v95(config, config_path=config_path)
    assert_deferred_final_absent_v95(config)
    rows = load_training_rows_v95(config)
    class_weights = balanced_class_weights_v95(config, rows)
    training = config["training"]
    main_schedule = training_schedule_v95(
        rows,
        seed=int(training["row_order_seed"]),
        epochs=int(training["epochs"]),
    )
    wrong_schedule = cross_scene_schedule_v95(rows, seed=int(training["row_order_seed"]))
    zero_schedule = causal_control_schedule_v95(
        rows, arm="zero_payload", seed=int(training["row_order_seed"])
    )
    permutation_schedule = causal_control_schedule_v95(
        rows,
        arm="full_interior_permutation",
        seed=int(training["row_order_seed"]),
    )
    schedule_hashes = _schedule_hashes_v95(
        main_schedule, wrong_schedule, zero_schedule, permutation_schedule
    )
    if any(training[name] != value for name, value in schedule_hashes.items()):
        raise RuntimeError("V95 fixed training schedule changed")
    wrong_lookup = {(epoch, row.key): wrong for epoch, row, wrong in wrong_schedule}
    zero_keys = {(epoch, row.key) for epoch, row in zero_schedule}
    permutation_keys = {(epoch, row.key) for epoch, row in permutation_schedule}
    if (
        len(rows) != EXPECTED_ROWS_PER_EPOCH
        or len(main_schedule) != EXPECTED_MICRO_ROWS
        or len(wrong_lookup) != EXPECTED_WRONG_MEMORY_ROWS
        or len(zero_keys) != EXPECTED_ZERO_ROWS
        or len(permutation_keys) != EXPECTED_PERMUTATION_ROWS
    ):
        raise RuntimeError("V95 fixed causal schedule inventory changed")
    outputs = config["outputs"]
    report_path = resolve_v85(outputs["training_report"])
    candidate_path = resolve_v85(outputs["fixed_final_candidate"])
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError("V95 create-once fixed-final training report exists")

    # Bind every immutable scene/control before question tokenization or Gemma.
    cpu_memories, memory_hashes_before = load_scene_memories_v95(config, rows)
    cpu_zero = {
        scene_id: zero_payload_memory_v95(memory) for scene_id, memory in cpu_memories.items()
    }
    cpu_permutation = {
        scene_id: permuted_payload_memory_v95(
            memory, seed=int(training["payload_permutation_seed"])
        )
        for scene_id, memory in cpu_memories.items()
    }
    zero_hashes_before = {scene_id: prefix_sha256(memory) for scene_id, memory in cpu_zero.items()}
    permutation_hashes_before = {
        scene_id: prefix_sha256(memory) for scene_id, memory in cpu_permutation.items()
    }
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
            raise RuntimeError("V95 full-model training requires local MPS")
        collection = install_lora_banks(language.model, combined_lora_settings_v95(runtime, config))
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V95 LoRA bank installation failed")
        frozen_source = load_frozen_parent_v95(collection, config)
        collection.assert_trainable_surface(language.model)
        fresh = collection.bank(FRESH_BANK_NAME).installation
        initial_state_sha256 = fresh.state_sha256()
        memory_by_scene = {
            scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
            for scene_id, memory in cpu_memories.items()
        }
        zero_by_scene = {
            scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
            for scene_id, memory in cpu_zero.items()
        }
        permutation_by_scene = {
            scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
            for scene_id, memory in cpu_permutation.items()
        }
        system_prompt = str(language_config["system_prompt"])
        parameters = collection.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        accumulation = int(training["gradient_accumulation_rows"])
        bindings: dict[str, Any] = {
            **preflight,
            "trainer_source_sha256": source_hashes[str(config["sources"]["trainer_source"])],
            **schedule_hashes,
        }
        resumed = discover_resume_checkpoint_v95(
            outputs["work_root"],
            bindings=bindings,
            row_order_sha256=schedule_hashes["row_order_sha256"],
            gradient_accumulation_rows=accumulation,
        )
        history: list[dict[str, Any]] = []
        row_cursor = 0
        optimizer_update = 0
        resumed_from: str | None = None
        if resumed is not None:
            checkpoint, metadata = resumed
            restore_resume_checkpoint_v95(checkpoint, metadata, collection, optimizer)
            row_cursor = int(metadata["row_cursor"])
            optimizer_update = int(metadata["update"])
            history = list(metadata["history"])
            resumed_from = checkpoint.relative_to(PROJECT_ROOT).as_posix()
        if row_cursor > len(main_schedule) or row_cursor % accumulation:
            raise ValueError("V95 resume cursor is outside the fixed schedule")

        language.decoder_module.train()
        collection.train()
        optimizer.zero_grad(set_to_none=True)
        interval: list[dict[str, float | bool]] = []
        prefix = main_schedule[:row_cursor]
        wrong_seen = sum((epoch, row.key) in wrong_lookup for epoch, row in prefix)
        zero_seen = sum((epoch, row.key) in zero_keys for epoch, row in prefix)
        permutation_seen = sum((epoch, row.key) in permutation_keys for epoch, row in prefix)
        for cursor in range(row_cursor, len(main_schedule)):
            epoch, row = main_schedule[cursor]
            schedule_key = (epoch, row.key)
            prepared, _layout = _prepared_v84(
                language, system_prompt, memory_by_scene[row.scene_id], row
            )
            correct_tail = _answer_tail(language, prepared)
            correct_nll = correct_tail.mean_nll.float()
            wrong_prepared = wrong_tail = None
            zero_prepared = zero_tail = None
            permutation_prepared = permutation_tail = None
            wrong_row = wrong_lookup.get(schedule_key)
            if wrong_row is not None:
                wrong_prepared, _wrong_layout = _prepared_v84(
                    language, system_prompt, memory_by_scene[wrong_row.scene_id], row
                )
                wrong_tail = _answer_tail(language, wrong_prepared)
                wrong_seen += 1
            if schedule_key in zero_keys:
                zero_prepared, _zero_layout = _prepared_v84(
                    language, system_prompt, zero_by_scene[row.scene_id], row
                )
                zero_tail = _answer_tail(language, zero_prepared)
                zero_seen += 1
            if schedule_key in permutation_keys:
                permutation_prepared, _permutation_layout = _prepared_v84(
                    language,
                    system_prompt,
                    permutation_by_scene[row.scene_id],
                    row,
                )
                permutation_tail = _answer_tail(language, permutation_prepared)
                permutation_seen += 1
            objective, components = causal_objective_v95(
                correct_nll,
                class_weight=float(class_weights[row.answer_class]),
                balanced_ce_weight=float(training["balanced_ce_weight"]),
                wrong_memory_nll=(None if wrong_tail is None else wrong_tail.mean_nll.float()),
                wrong_margin_weight=float(training["cross_scene_wrong_memory_margin_weight"]),
                wrong_target_margin=float(training["cross_scene_wrong_memory_target_margin_nll"]),
                zero_payload_nll=(None if zero_tail is None else zero_tail.mean_nll.float()),
                zero_margin_weight=float(training["zero_payload_margin_weight"]),
                zero_target_margin=float(training["zero_payload_target_margin_nll"]),
                permutation_nll=(
                    None if permutation_tail is None else permutation_tail.mean_nll.float()
                ),
                permutation_margin_weight=float(training["permutation_margin_weight"]),
                permutation_target_margin=float(training["permutation_target_margin_nll"]),
            )
            if not torch.isfinite(objective):
                raise RuntimeError("V95 objective is nonfinite")
            interval.append(
                {
                    "correct_nll": float(correct_nll.detach().cpu()),
                    "objective": float(objective.detach().cpu()),
                    "wrong_memory_minus_correct_nll": float(
                        components["wrong_memory_minus_correct_nll"].detach().cpu()
                    ),
                    "wrong_memory_minus_correct_nll_present": wrong_row is not None,
                    "zero_payload_minus_correct_nll": float(
                        components["zero_payload_minus_correct_nll"].detach().cpu()
                    ),
                    "zero_payload_minus_correct_nll_present": schedule_key in zero_keys,
                    "permutation_minus_correct_nll": float(
                        components["permutation_minus_correct_nll"].detach().cpu()
                    ),
                    "permutation_minus_correct_nll_present": schedule_key in permutation_keys,
                }
            )
            (objective / accumulation).backward()
            row_cursor = cursor + 1
            del prepared, correct_tail, correct_nll, objective, components
            del wrong_prepared, wrong_tail, zero_prepared, zero_tail
            del permutation_prepared, permutation_tail
            if row_cursor % accumulation:
                continue
            for parameter in parameters:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("V95 found a missing, NaN, or infinite gradient")
            gradient_l2 = float(collection.gradient_norms()["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V95 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            clip_l2 = float(clipped.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V95 clipped gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            record = {
                "update": optimizer_update,
                "row_cursor": row_cursor,
                "epoch": epoch,
                "mean_correct_nll": sum(float(v["correct_nll"]) for v in interval) / len(interval),
                "mean_objective": sum(float(v["objective"]) for v in interval) / len(interval),
                "wrong_memory_rows": sum(
                    bool(v["wrong_memory_minus_correct_nll_present"]) for v in interval
                ),
                "zero_payload_rows": sum(
                    bool(v["zero_payload_minus_correct_nll_present"]) for v in interval
                ),
                "permutation_rows": sum(
                    bool(v["permutation_minus_correct_nll_present"]) for v in interval
                ),
                "mean_wrong_memory_minus_correct_nll": _mean(
                    interval, "wrong_memory_minus_correct_nll"
                ),
                "mean_zero_payload_minus_correct_nll": _mean(
                    interval, "zero_payload_minus_correct_nll"
                ),
                "mean_permutation_minus_correct_nll": _mean(
                    interval, "permutation_minus_correct_nll"
                ),
                "gradient_l2_before_clip": gradient_l2,
                "clip_return_l2": clip_l2,
                "state_sha256": fresh.state_sha256(),
            }
            history.append(record)
            interval.clear()
            if optimizer_update in {1, 120, 240, 360, 480} or optimizer_update % 15 == 0:
                print(
                    json.dumps(
                        {
                            "event": "v95_train_update",
                            "update": optimizer_update,
                            "total_updates": EXPECTED_OPTIMIZER_UPDATES,
                            "row_cursor": row_cursor,
                            "epoch": epoch,
                            "wrong_memory_seen": wrong_seen,
                            "zero_payload_seen": zero_seen,
                            "permutation_seen": permutation_seen,
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
                save_resume_checkpoint_v95(
                    outputs["work_root"],
                    collection,
                    optimizer,
                    update=optimizer_update,
                    row_cursor=row_cursor,
                    history=history,
                    bindings=bindings,
                    row_order_sha256=schedule_hashes["row_order_sha256"],
                )
            torch.mps.empty_cache()

        if (
            optimizer_update != EXPECTED_OPTIMIZER_UPDATES
            or row_cursor != EXPECTED_MICRO_ROWS
            or wrong_seen != EXPECTED_WRONG_MEMORY_ROWS
            or zero_seen != EXPECTED_ZERO_ROWS
            or permutation_seen != EXPECTED_PERMUTATION_ROWS
            or len(history) != EXPECTED_OPTIMIZER_UPDATES
        ):
            raise RuntimeError("V95 fixed causal schedule did not complete exactly")
        language.decoder_module.eval()
        collection.eval()
        memory_hashes_after = {
            scene_id: prefix_sha256(memory.detach().cpu())
            for scene_id, memory in memory_by_scene.items()
        }
        zero_hashes_after = {
            scene_id: prefix_sha256(memory.detach().cpu())
            for scene_id, memory in zero_by_scene.items()
        }
        permutation_hashes_after = {
            scene_id: prefix_sha256(memory.detach().cpu())
            for scene_id, memory in permutation_by_scene.items()
        }
        if (
            memory_hashes_after != memory_hashes_before
            or zero_hashes_after != zero_hashes_before
            or permutation_hashes_after != permutation_hashes_before
        ):
            raise RuntimeError("V95 training mutated immutable environmental inputs")
        candidate_bindings = {
            **bindings,
            "fixed_final_optimizer_updates": optimizer_update,
            "class_weight_inventory_sha256": config["training_pool"][
                "balanced_class_weight_inventory_sha256"
            ],
            "known_development_labels_opened": False,
            "deferred_final_generated": False,
        }
        candidate_metadata, candidate_reused_after_interruption = (
            finalize_fixed_final_candidate_v95(
                candidate_path, collection, bindings=candidate_bindings
            )
        )
    audit.assert_clean()

    zero_exposures = Counter(row.key for _epoch, row in zero_schedule)
    permutation_exposures = Counter(row.key for _epoch, row in permutation_schedule)
    gates = {
        "all_960_rows_consumed_once_in_each_of_four_epochs": row_cursor == EXPECTED_MICRO_ROWS,
        "all_996_cross_scene_wrong_memory_forwards_consumed": wrong_seen
        == EXPECTED_WRONG_MEMORY_ROWS,
        "zero_arm_covers_all_498_with_only_two_repeats": len(zero_exposures) == 498
        and Counter(zero_exposures.values()) == Counter({1: 496, 2: 2}),
        "permutation_arm_covers_all_498_with_only_two_repeats": len(permutation_exposures) == 498
        and Counter(permutation_exposures.values()) == Counter({1: 496, 2: 2}),
        "exact_5836_nll_forward_schedule": EXPECTED_MICRO_ROWS
        + wrong_seen
        + zero_seen
        + permutation_seen
        == EXPECTED_TOTAL_NLL_FORWARDS,
        "fixed_final_update_480_reached": optimizer_update == EXPECTED_OPTIMIZER_UPDATES,
        "only_fresh_143360_parameters_trainable": len(parameters) == 6
        and sum(parameter.numel() for parameter in parameters) == FRESH_PARAMETER_COUNT,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(record["gradient_l2_before_clip"]))
            and float(record["gradient_l2_before_clip"]) > 0.0
            for record in history
        ),
        "all_scene_and_control_hashes_invariant": memory_hashes_after == memory_hashes_before
        and zero_hashes_after == zero_hashes_before
        and permutation_hashes_after == permutation_hashes_before,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
        "deferred_final_still_absent": not assert_deferred_final_absent_v95(config)[
            "physical_artifacts_present"
        ],
    }
    if not all(gates.values()):
        raise RuntimeError(f"V95 training gate failed: {gates}")
    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 95,
        "status": "fixed_final_training_complete_not_promoted",
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
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
            "parent": "fixed_final_nonpromoted_optimization_parent",
            "unmerged": True,
        },
        "training_protocol": training,
        "schedule_hashes": schedule_hashes,
        "micro_rows_consumed": row_cursor,
        "unique_training_rows": len(rows),
        "training_scene_count": len(cpu_memories),
        "wrong_memory_rows_consumed": wrong_seen,
        "zero_payload_rows_consumed": zero_seen,
        "permutation_rows_consumed": permutation_seen,
        "total_nll_forwards": EXPECTED_TOTAL_NLL_FORWARDS,
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
            "question_conditioned_environmental_readout": False,
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
        "deferred_final_generated": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_create_json_v85(report_path, report)
    return report


def authenticate_training_report_v95(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    preflight = authenticate_cpu_preflight_v95(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["training_report"])
    report = strict_json_v95(path)
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != EXPECTED_OPTIMIZER_UPDATES
        or report.get("micro_rows_consumed") != EXPECTED_MICRO_ROWS
        or report.get("wrong_memory_rows_consumed") != EXPECTED_WRONG_MEMORY_ROWS
        or report.get("zero_payload_rows_consumed") != EXPECTED_ZERO_ROWS
        or report.get("permutation_rows_consumed") != EXPECTED_PERMUTATION_ROWS
        or report.get("total_nll_forwards") != EXPECTED_TOTAL_NLL_FORWARDS
        or report.get("protected_read_count") != 0
        or report.get("known_development_labels_loaded") is not False
        or report.get("deferred_final_generated") is not False
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V95 fixed-final training report changed")
    return {**preflight, "training_report_sha256": sha256_file_v85(path)}


def run_topology_smoke_v95(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Prove the sealed topology against Gemma's config without loading weights."""

    config = load_config_v95(config_path, allow_draft=False)
    preflight = authenticate_cpu_preflight_v95(config, config_path=config_path)
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    settings = combined_lora_settings_v95(runtime, config)
    parent = authenticate_parent_v94_v95(config)
    fresh = settings.bank(FRESH_BANK_NAME)
    synthetic = lora_preflight_v95(config)
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
        observed_linear_targets = observed_linear_targets and isinstance(
            module, torch.nn.Linear
        )
        observed_shapes[target + ".weight"] = list(module.weight.shape)
    del topology_model
    checks = {
        "full_gemma_model_loaded_false": True,
        "pinned_gemma_targets_are_linear": observed_linear_targets,
        "pinned_gemma_target_shapes_exact": observed_shapes
        == config["bridge"]["pinned_weight_shapes"],
        "exact_nine_bank_stack": len(settings.banks) == 9,
        "exact_eight_frozen_banks": sum(not bank.trainable for bank in settings.banks) == 8,
        "sole_fresh_bank_trainable": sum(bank.trainable for bank in settings.banks) == 1,
        "fresh_targets_exact": fresh.adapter.target_modules == TARGET_MODULES,
        "fresh_parameter_count_exact": synthetic["parameter_count"] == FRESH_PARAMETER_COUNT,
        "fresh_initial_state_exact": synthetic["initial_state_sha256"]
        == EXPECTED_INITIAL_STATE_SHA256,
        "fresh_output_starts_zero": synthetic["exact_zero_output_at_initialization"] is True,
        "failed_v94_parent_authenticated": parent["v94_behavior_gate_passed"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V95 CPU topology smoke failed: {checks}")
    return {
        "artifact": "gemma4_v95_cpu_topology_smoke_v1",
        "schema_version": 95,
        "passed": True,
        "config_sha256": preflight["config_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "checks": checks,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--topology-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.topology_smoke:
        result = run_topology_smoke_v95(args.config)
    else:
        result = run_training_v95(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_ARTIFACT",
    "CHECKPOINT_ARTIFACT",
    "EXPECTED_FROZEN_BANK_COUNT",
    "EXPECTED_MICRO_ROWS",
    "EXPECTED_OPTIMIZER_UPDATES",
    "EXPECTED_PERMUTATION_ROWS",
    "EXPECTED_TOTAL_NLL_FORWARDS",
    "EXPECTED_WRONG_MEMORY_ROWS",
    "EXPECTED_ZERO_ROWS",
    "METADATA_FILENAME",
    "TRAINING_ARTIFACT",
    "WEIGHTS_FILENAME",
    "authenticate_training_report_v95",
    "causal_objective_v95",
    "combined_lora_settings_v95",
    "discover_resume_checkpoint_v95",
    "finalize_fixed_final_candidate_v95",
    "load_fixed_final_bridge_v95",
    "load_frozen_parent_v95",
    "main",
    "publish_fixed_final_candidate_v95",
    "restore_resume_checkpoint_v95",
    "run_topology_smoke_v95",
    "run_training_v95",
    "save_resume_checkpoint_v95",
    "strict_json_v95",
]
