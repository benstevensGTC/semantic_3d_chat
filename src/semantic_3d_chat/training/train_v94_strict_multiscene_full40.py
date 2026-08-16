"""Train V94's fixed-final strict full-40-scene Gemma bridge.

V94 is a clean continuation from the exact seven-bank V85 strict runtime; no
V86--V93 adapter is installed or loaded.  All forty immutable scene memories
are compiled before question tokenization and every one of their 738 tokens is
passed to Gemma.  The only trainable state is one fresh rank-8 adapter on the
layer-34 gate projection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
    canonical_sha256_v85,
    resolve_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v94_strict_multiscene_preflight import (
    CONFIG,
    FRESH_BANK_NAME,
    TARGET_MODULE,
    authenticate_cpu_preflight_v94,
    authenticate_sources_v94,
    causal_sides_v94,
    class_weights_v94,
    load_config_v94,
    load_scene_memories_v94,
    load_training_rows_v94,
    training_schedule_v94,
    zero_payload_memory_v94,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
    lora_banks_settings,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_v84_strict_bridge import _prepared_v84

TRAINING_ARTIFACT: Final[str] = "gemma4_v94_strict_multiscene_full40_training_v1"
CANDIDATE_ARTIFACT: Final[str] = "gemma4_v94_strict_multiscene_full40_fixed_final_v1"
CHECKPOINT_ARTIFACT: Final[str] = "gemma4_v94_strict_multiscene_full40_resume_v1"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
EXPECTED_FROZEN_BANK_COUNT: Final[int] = 7
EXPECTED_FROZEN_PARAMETER_COUNT: Final[int] = 565_248
EXPECTED_FRESH_PARAMETER_COUNT: Final[int] = 110_592
EXPECTED_TOTAL_PARAMETER_COUNT: Final[int] = 675_840
EXPECTED_ROWS_PER_EPOCH: Final[int] = 960
EXPECTED_EPOCHS: Final[int] = 3
EXPECTED_MICRO_ROWS: Final[int] = 2_880
EXPECTED_OPTIMIZER_UPDATES: Final[int] = 360
EXPECTED_CHANGED_SIDES_PER_EPOCH: Final[int] = 132
EXPECTED_PAIRED_MARGIN_ROWS: Final[int] = 396
EXPECTED_CAUSAL_SIDES_PER_EPOCH: Final[int] = 18
EXPECTED_CAUSAL_MARGIN_ROWS: Final[int] = 54
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {
        "oracle",
        "validation",
        "test",
        "deferred",
        "v86",
        "v87",
        "v88",
        "v89",
        "v90",
        "v91",
        "v92",
        "v93",
    }
)


def strict_json_v94(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V94 JSON must contain one object: {source}")
    return value


def combined_lora_settings_v94(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    """Retain exactly V85's seven frozen banks and append one fresh bank."""

    base = lora_banks_settings(runtime_config)
    if len(base.banks) != EXPECTED_FROZEN_BANK_COUNT or any(bank.trainable for bank in base.banks):
        raise ValueError("V94 requires exactly seven frozen V85 runtime banks")
    bridge = experiment["bridge"]
    fresh = LoRABankSettings(
        name=FRESH_BANK_NAME,
        trainable=True,
        adapter=LoRASettings(
            enabled=True,
            rank=int(bridge["rank"]),
            alpha=float(bridge["alpha"]),
            dropout=float(bridge["dropout"]),
            target_modules=(str(bridge["target_module"]),),
        ),
        initialization_algorithm=str(bridge["initialization_algorithm"]),
        initialization_seed=int(bridge["initialization_seed"]),
        expected_initial_state_sha256=str(bridge["expected_initial_state_sha256"]),
    )
    return LoRABanksSettings(base.banks + (fresh,))


def _copy_two_tensor_state_v94(
    installation: Any, archive: Mapping[str, torch.Tensor], *, context: str
) -> None:
    if set(archive) != {"lora_a", "lora_b"} or len(installation.adapters) != 1:
        raise ValueError(f"{context} tensor inventory changed")
    adapter = installation.adapters[0]
    with torch.no_grad():
        for name, parameter in (("lora_a", adapter.lora_a), ("lora_b", adapter.lora_b)):
            value = archive[name]
            if value.shape != parameter.shape or value.dtype != torch.float32:
                raise ValueError(f"{context} {name} shape or dtype changed")
            parameter.copy_(value.to(parameter.device))


def load_frozen_v85_stack_v94(
    collection: LoRABankCollection, checkpoint: str | Path
) -> dict[str, Any]:
    """Authenticate and load the exact seven-bank V85 runtime candidate."""

    root = resolve_v85(checkpoint)
    weights = root / "adapter.safetensors"
    metadata_path = root / "runtime_metadata.json"
    metadata = strict_json_v94(metadata_path)
    hashes = metadata.get("lora_bank_state_sha256")
    modules = metadata.get("lora_bank_wrapped_modules")
    counts = metadata.get("lora_bank_parameter_counts")
    if not all(isinstance(value, Mapping) for value in (hashes, modules, counts)):
        raise TypeError("V94 V85 source lacks named LoRA metadata")
    frozen = [bank for bank in collection.banks if not bank.settings.trainable]
    trainable = [bank for bank in collection.banks if bank.settings.trainable]
    if (
        len(frozen) != EXPECTED_FROZEN_BANK_COUNT
        or len(trainable) != 1
        or trainable[0].settings.name != FRESH_BANK_NAME
        or set(hashes) != {bank.settings.name for bank in frozen}
        or metadata.get("lora_parameter_count") != EXPECTED_FROZEN_PARAMETER_COUNT
        or metadata.get("lora_trainable_parameter_count") != 0
    ):
        raise ValueError("V94 parent must be only the exact seven-bank V85 stack")
    archive = load_file(str(weights), device="cpu")
    observed_expected_keys: set[str] = set()
    expected_lora_keys: set[str] = set()
    for bank in frozen:
        name = bank.settings.name
        prefix = f"lora_banks.{name}."
        expected_lora_keys.update(
            prefix + key for key in bank.installation.state_module.state_dict()
        )
        state = {
            key[len(prefix) :]: value for key, value in archive.items() if key.startswith(prefix)
        }
        observed_expected_keys.update(prefix + key for key in state)
        bank.installation.state_module.load_state_dict(state, strict=True)
        if list(bank.installation.target_names) != modules[name]:
            raise ValueError(f"V94 frozen bank targets changed: {name}")
        if bank.installation.parameter_counts != counts[name]:
            raise ValueError(f"V94 frozen bank parameter counts changed: {name}")
        if bank.installation.state_sha256() != hashes[name]:
            raise ValueError(f"V94 frozen bank state changed: {name}")
    observed_lora_keys = {key for key in archive if key.startswith("lora_banks.")}
    if observed_lora_keys != expected_lora_keys or observed_expected_keys != expected_lora_keys:
        raise ValueError("V94 parent named-LoRA tensor inventory changed")

    fresh = trainable[0].installation
    if (
        fresh.target_names != (TARGET_MODULE,)
        or fresh.parameter_count != EXPECTED_FRESH_PARAMETER_COUNT
        or fresh.state_sha256() != trainable[0].settings.expected_initial_state_sha256
        or any(torch.count_nonzero(adapter.lora_b).item() for adapter in fresh.adapters)
        or collection.parameter_count != EXPECTED_TOTAL_PARAMETER_COUNT
        or collection.trainable_parameter_count != EXPECTED_FRESH_PARAMETER_COUNT
    ):
        raise ValueError("V94 fresh bridge did not start at exact zero output")
    collection.validate_state()
    return {
        "adapter_sha256": sha256_file_v85(weights),
        "runtime_metadata_sha256": sha256_file_v85(metadata_path),
        "frozen_bank_count": len(frozen),
        "frozen_parameter_count": EXPECTED_FROZEN_PARAMETER_COUNT,
        "frozen_bank_state_sha256": dict(hashes),
        "fresh_initial_state_sha256": fresh.state_sha256(),
        "total_bank_count": len(collection.banks),
        "total_parameter_count": collection.parameter_count,
        "v86_through_v93_loaded": False,
    }


def multiscene_objective_v94(
    correct_nll: torch.Tensor,
    *,
    class_weight: float,
    answer_ce_weight: float,
    paired_wrong_nll: torch.Tensor | None = None,
    paired_margin_weight: float = 0.0,
    paired_target_margin: float = 0.0,
    zero_payload_nll: torch.Tensor | None = None,
    zero_margin_weight: float = 0.0,
    zero_target_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose weighted CE with independent wrong-scene and zero controls."""

    if correct_nll.ndim != 0 or not torch.isfinite(correct_nll):
        raise ValueError("V94 correct NLL must be a finite scalar")
    numbers = (
        class_weight,
        answer_ce_weight,
        paired_margin_weight,
        paired_target_margin,
        zero_margin_weight,
        zero_target_margin,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in numbers):
        raise ValueError("V94 objective weights and margins must be finite nonnegative")
    objective = float(answer_ce_weight) * float(class_weight) * correct_nll
    zero = correct_nll.detach().new_zeros(())
    records = {
        "paired_wrong_minus_correct_nll": zero,
        "paired_margin_penalty": zero,
        "zero_minus_correct_nll": zero,
        "zero_margin_penalty": zero,
    }
    if paired_wrong_nll is not None:
        if paired_wrong_nll.ndim != 0 or not torch.isfinite(paired_wrong_nll):
            raise ValueError("V94 paired-wrong NLL must be a finite scalar")
        records["paired_wrong_minus_correct_nll"] = paired_wrong_nll - correct_nll
        records["paired_margin_penalty"] = torch.relu(
            correct_nll - paired_wrong_nll + paired_target_margin
        )
        objective = objective + paired_margin_weight * records["paired_margin_penalty"]
    if zero_payload_nll is not None:
        if zero_payload_nll.ndim != 0 or not torch.isfinite(zero_payload_nll):
            raise ValueError("V94 zero-payload NLL must be a finite scalar")
        records["zero_minus_correct_nll"] = zero_payload_nll - correct_nll
        records["zero_margin_penalty"] = torch.relu(
            correct_nll - zero_payload_nll + zero_target_margin
        )
        objective = objective + zero_margin_weight * records["zero_margin_penalty"]
    return objective, records


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _fresh_state_v94(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    installation = collection.bank(FRESH_BANK_NAME).installation
    if len(installation.adapters) != 1:
        raise ValueError("V94 fresh bank must wrap exactly one module")
    adapter = installation.adapters[0]
    return {
        "lora_a": adapter.lora_a.detach().cpu().contiguous(),
        "lora_b": adapter.lora_b.detach().cpu().contiguous(),
    }


def _load_fresh_state_v94(
    collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]
) -> None:
    _copy_two_tensor_state_v94(
        collection.bank(FRESH_BANK_NAME).installation,
        archive,
        context="V94 fresh bridge",
    )
    collection.validate_state()


def _optimizer_tensors_v94(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    state = optimizer.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for parameter_index, values in state["state"].items():
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("V94 AdamW state must be tensor-only")
            tensors[f"optimizer.{parameter_index}.{name}"] = value.detach().cpu().contiguous()
    groups: list[dict[str, Any]] = []
    for group in state["param_groups"]:
        normalized = dict(group)
        normalized["params"] = [int(value) for value in normalized["params"]]
        groups.append(normalized)
    return tensors, groups


def _restore_optimizer_v94(
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


def save_resume_checkpoint_v94(
    work_root: str | Path,
    collection: LoRABankCollection,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    row_cursor: int,
    history: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, str],
    row_order_sha256: str,
) -> Path:
    root = resolve_v85(work_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"update_{update:06d}"
    if destination.exists() or destination.is_symlink():
        existing = strict_json_v94(destination / "state.json")
        if (
            existing.get("update") == update
            and existing.get("row_cursor") == row_cursor
            and existing.get("fresh_state_sha256")
            == collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        ):
            return destination
        raise FileExistsError(f"V94 checkpoint collision: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=root))
    try:
        optimizer_tensors, groups = _optimizer_tensors_v94(optimizer)
        weights = temporary / "state.safetensors"
        save_file(
            {**_fresh_state_v94(collection), **optimizer_tensors},
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
            "schema_version": 94,
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


def discover_resume_checkpoint_v94(
    work_root: str | Path,
    *,
    bindings: Mapping[str, str],
    row_order_sha256: str,
    gradient_accumulation_rows: int,
) -> tuple[Path, dict[str, Any]] | None:
    root = resolve_v85(work_root)
    if not root.exists():
        return None
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in root.glob("update_[0-9][0-9][0-9][0-9][0-9][0-9]"):
        if path.is_symlink() or not path.is_dir():
            continue
        metadata = strict_json_v94(path / "state.json")
        update = int(metadata.get("update", -1))
        if (
            metadata.get("artifact") != CHECKPOINT_ARTIFACT
            or metadata.get("row_cursor") != update * gradient_accumulation_rows
            or metadata.get("row_order_sha256") != row_order_sha256
            or metadata.get("bindings") != dict(bindings)
            or metadata.get("environmental_memory_serialized") is not False
            or metadata.get("questions_or_answers_serialized") is not False
            or metadata.get("oracle_serialized") is not False
            or metadata.get("tensor_file_sha256") != sha256_file_v85(path / "state.safetensors")
        ):
            raise ValueError(f"V94 resume checkpoint authentication failed: {path}")
        candidates.append((update, path, metadata))
    if not candidates:
        return None
    _update, path, metadata = max(candidates, key=lambda value: value[0])
    return path, metadata


def restore_resume_checkpoint_v94(
    checkpoint: Path,
    metadata: Mapping[str, Any],
    collection: LoRABankCollection,
    optimizer: torch.optim.Optimizer,
) -> None:
    archive = load_file(str(checkpoint / "state.safetensors"), device="cpu")
    _load_fresh_state_v94(collection, {name: archive[name] for name in ("lora_a", "lora_b")})
    if (
        collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        != metadata["fresh_state_sha256"]
    ):
        raise ValueError("V94 resumed bridge hash changed")
    groups = metadata.get("optimizer_param_groups")
    if not isinstance(groups, list):
        raise TypeError("V94 resume optimizer groups are missing")
    _restore_optimizer_v94(optimizer, archive, groups)


def publish_fixed_final_candidate_v94(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Create once; publish only the two fresh adapter tensors."""

    root = resolve_v85(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V94 create-once fixed-final candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / WEIGHTS_FILENAME
        state = _fresh_state_v94(collection)
        if set(state) != {"lora_a", "lora_b"}:
            raise RuntimeError("V94 candidate tensor inventory changed")
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
            fresh.target_names != (TARGET_MODULE,)
            or fresh.parameter_count != EXPECTED_FRESH_PARAMETER_COUNT
        ):
            raise ValueError("V94 candidate fresh-bank topology changed")
        metadata = {
            "artifact": CANDIDATE_ARTIFACT,
            "schema_version": 94,
            "status": "fixed_final_awaiting_preregistered_acceptance_gates",
            "parent": "exact_v85_strict_runtime_candidate_only",
            "v86_through_v93_loaded": False,
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "parameter_count": fresh.parameter_count,
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v85(weights),
            "tensor_inventory": ["lora_a", "lora_b"],
            "environmental_memory_serialized": False,
            "questions_or_answers_serialized": False,
            "oracle_serialized": False,
            "evaluation_scored": False,
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


def load_fixed_final_bridge_v94(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    root = resolve_v85(candidate)
    metadata = strict_json_v94(root / METADATA_FILENAME)
    weights = root / WEIGHTS_FILENAME
    if (
        metadata.get("artifact") != CANDIDATE_ARTIFACT
        or metadata.get("status") != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("parent") != "exact_v85_strict_runtime_candidate_only"
        or metadata.get("v86_through_v93_loaded") is not False
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("target_module") != TARGET_MODULE
        or metadata.get("parameter_count") != EXPECTED_FRESH_PARAMETER_COUNT
        or metadata.get("tensor_inventory") != ["lora_a", "lora_b"]
        or metadata.get("weights_sha256") != sha256_file_v85(weights)
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V94 fixed-final candidate authentication failed")
    archive = load_file(str(weights), device="cpu")
    _load_fresh_state_v94(collection, archive)
    if collection.bank(FRESH_BANK_NAME).installation.state_sha256() != metadata["state_sha256"]:
        raise ValueError("V94 fixed-final candidate state changed")
    return metadata


def _value(config: Mapping[str, Any], group: str, name: str, default: Any) -> Any:
    value = config[group].get(name, default)
    return value


def run_training_v94(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v94(config_path, allow_draft=False)
    source_hashes = authenticate_sources_v94(config)
    preflight = authenticate_cpu_preflight_v94(config, config_path=config_path)
    rows = load_training_rows_v94(config)
    class_weights = class_weights_v94(config, rows)
    causal_keys = {row.key for row in causal_sides_v94(config, rows)}
    training = config["training"]
    schedule = training_schedule_v94(
        rows,
        seed=int(training["row_order_seed"]),
        epochs=int(training["epochs"]),
    )
    schedule_hash = canonical_sha256_v85(
        [[epoch, row.scene_id, row.question_id] for epoch, row in schedule]
    )
    if schedule_hash != training["row_order_sha256"]:
        raise RuntimeError("V94 fixed training schedule changed")
    if (
        len(rows) != EXPECTED_ROWS_PER_EPOCH
        or len({row.scene_id for row in rows}) != 40
        or sum(row.expected_change for row in rows) != EXPECTED_CHANGED_SIDES_PER_EPOCH
        or len(causal_keys) != EXPECTED_CAUSAL_SIDES_PER_EPOCH
        or len(schedule) != EXPECTED_MICRO_ROWS
    ):
        raise RuntimeError("V94 fixed full-40 training inventory changed")
    outputs = config["outputs"]
    report_path = resolve_v85(outputs["training_report"])
    candidate_path = resolve_v85(outputs["fixed_final_candidate"])
    if report_path.exists() or candidate_path.exists():
        raise FileExistsError("V94 fixed-final outputs already exist")

    # This immutable environment boundary intentionally precedes Gemma loading
    # and every question-tokenization operation.
    cpu_memories, memory_hashes_before = load_scene_memories_v94(config, rows)
    if len(cpu_memories) != 40 or any(
        tuple(memory.shape) != (1, 738, 1536) or memory.dtype != torch.bfloat16
        for memory in cpu_memories.values()
    ):
        raise RuntimeError("V94 full-scene memory contract changed")
    cpu_zero_memories = {
        scene_id: zero_payload_memory_v94(memory) for scene_id, memory in cpu_memories.items()
    }
    zero_hashes_before = {
        scene_id: prefix_sha256(memory) for scene_id, memory in cpu_zero_memories.items()
    }
    audit = FileAccessAudit(
        forbidden_component_names=_FORBIDDEN_COMPONENTS,
        block_forbidden=True,
    )
    with audit:
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
        if language.device.type != "mps":
            raise RuntimeError("V94 full-model training requires local MPS")
        collection = install_lora_banks(language.model, combined_lora_settings_v94(runtime, config))
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V94 LoRA bank installation failed")
        frozen_source = load_frozen_v85_stack_v94(
            collection, config["sources"]["frozen_v85_checkpoint"]
        )
        collection.assert_trainable_surface(language.model)
        fresh = collection.bank(FRESH_BANK_NAME).installation
        initial_state_sha256 = fresh.state_sha256()
        memory_by_scene = {
            scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
            for scene_id, memory in cpu_memories.items()
        }
        zero_by_scene = {
            scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
            for scene_id, memory in cpu_zero_memories.items()
        }
        system_prompt = str(language_config["system_prompt"])
        parameters = collection.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        accumulation = int(training["gradient_accumulation_rows"])
        bindings = {
            **preflight,
            "frozen_v85_adapter_sha256": frozen_source["adapter_sha256"],
            "row_order_sha256": schedule_hash,
        }
        resumed = discover_resume_checkpoint_v94(
            outputs["work_root"],
            bindings=bindings,
            row_order_sha256=schedule_hash,
            gradient_accumulation_rows=accumulation,
        )
        history: list[dict[str, Any]] = []
        row_cursor = 0
        optimizer_update = 0
        resumed_from: str | None = None
        if resumed is not None:
            checkpoint, metadata = resumed
            restore_resume_checkpoint_v94(checkpoint, metadata, collection, optimizer)
            row_cursor = int(metadata["row_cursor"])
            optimizer_update = int(metadata["update"])
            history = list(metadata["history"])
            resumed_from = checkpoint.relative_to(PROJECT_ROOT).as_posix()
        if row_cursor > len(schedule) or row_cursor % accumulation:
            raise ValueError("V94 resume cursor is outside the fixed schedule")

        language.decoder_module.train()
        collection.train()
        optimizer.zero_grad(set_to_none=True)
        interval: list[dict[str, float | bool]] = []
        changed_seen = sum(row.expected_change for _epoch, row in schedule[:row_cursor])
        causal_seen = sum(row.key in causal_keys for _epoch, row in schedule[:row_cursor])
        for cursor in range(row_cursor, len(schedule)):
            epoch, row = schedule[cursor]
            prepared, _layout = _prepared_v84(
                language, system_prompt, memory_by_scene[row.scene_id], row
            )
            tail = _answer_tail(language, prepared)
            correct_nll = tail.mean_nll.float()
            paired_tail = None
            paired_prepared = None
            zero_tail = None
            zero_prepared = None
            if row.expected_change:
                paired_prepared, _paired_layout = _prepared_v84(
                    language,
                    system_prompt,
                    memory_by_scene[row.paired_scene_id],
                    row,
                )
                paired_tail = _answer_tail(language, paired_prepared)
                changed_seen += 1
            if row.key in causal_keys:
                zero_prepared, _zero_layout = _prepared_v84(
                    language, system_prompt, zero_by_scene[row.scene_id], row
                )
                zero_tail = _answer_tail(language, zero_prepared)
                causal_seen += 1
            objective, components = multiscene_objective_v94(
                correct_nll,
                class_weight=float(class_weights[row.answer_class]),
                answer_ce_weight=float(training["answer_ce_weight"]),
                paired_wrong_nll=(None if paired_tail is None else paired_tail.mean_nll.float()),
                paired_margin_weight=float(training["paired_wrong_scene_margin_weight"]),
                paired_target_margin=float(training["paired_wrong_scene_target_margin_nll"]),
                zero_payload_nll=(None if zero_tail is None else zero_tail.mean_nll.float()),
                zero_margin_weight=float(training["zero_payload_margin_weight"]),
                zero_target_margin=float(training["zero_payload_target_margin_nll"]),
            )
            if not torch.isfinite(objective):
                raise RuntimeError("V94 objective is nonfinite")
            interval.append(
                {
                    "correct_nll": float(correct_nll.detach().cpu()),
                    "objective": float(objective.detach().cpu()),
                    "class_weight": float(class_weights[row.answer_class]),
                    "changed": row.expected_change,
                    "causal": row.key in causal_keys,
                    **{name: float(value.detach().cpu()) for name, value in components.items()},
                }
            )
            (objective / accumulation).backward()
            row_cursor = cursor + 1
            del prepared, tail, correct_nll, objective, components
            del paired_prepared, paired_tail, zero_prepared, zero_tail
            if row_cursor % accumulation:
                continue
            gradient_l2 = float(collection.gradient_norms()["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V94 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            clip_l2 = float(clipped.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V94 clipped gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            changed_interval = [record for record in interval if record["changed"]]
            causal_interval = [record for record in interval if record["causal"]]
            record = {
                "update": optimizer_update,
                "row_cursor": row_cursor,
                "epoch": epoch,
                "mean_correct_nll": sum(float(v["correct_nll"]) for v in interval) / len(interval),
                "mean_objective": sum(float(v["objective"]) for v in interval) / len(interval),
                "changed_sides": len(changed_interval),
                "causal_sides": len(causal_interval),
                "mean_changed_wrong_minus_correct_nll": (
                    sum(float(v["paired_wrong_minus_correct_nll"]) for v in changed_interval)
                    / len(changed_interval)
                    if changed_interval
                    else None
                ),
                "mean_causal_zero_minus_correct_nll": (
                    sum(float(v["zero_minus_correct_nll"]) for v in causal_interval)
                    / len(causal_interval)
                    if causal_interval
                    else None
                ),
                "gradient_l2_before_clip": gradient_l2,
                "clip_return_l2": clip_l2,
                "state_sha256": fresh.state_sha256(),
            }
            history.append(record)
            interval.clear()
            if optimizer_update in {1, 120, 240, 360} or optimizer_update % 15 == 0:
                print(
                    json.dumps(
                        {
                            "event": "v94_train_update",
                            "update": optimizer_update,
                            "total_updates": EXPECTED_OPTIMIZER_UPDATES,
                            "row_cursor": row_cursor,
                            "epoch": epoch,
                            "changed_sides_seen": changed_seen,
                            "causal_sides_seen": causal_seen,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if (
                optimizer_update
                % int(_value(config, "training", "checkpoint_every_optimizer_updates", 30))
                == 0
                or optimizer_update == EXPECTED_OPTIMIZER_UPDATES
            ):
                save_resume_checkpoint_v94(
                    outputs["work_root"],
                    collection,
                    optimizer,
                    update=optimizer_update,
                    row_cursor=row_cursor,
                    history=history,
                    bindings=bindings,
                    row_order_sha256=schedule_hash,
                )
            torch.mps.empty_cache()

        if (
            optimizer_update != EXPECTED_OPTIMIZER_UPDATES
            or row_cursor != EXPECTED_MICRO_ROWS
            or changed_seen != EXPECTED_PAIRED_MARGIN_ROWS
            or causal_seen != EXPECTED_CAUSAL_MARGIN_ROWS
            or len(history) != EXPECTED_OPTIMIZER_UPDATES
        ):
            raise RuntimeError("V94 fixed full-40 schedule did not complete exactly")
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
        if memory_hashes_after != memory_hashes_before or zero_hashes_after != zero_hashes_before:
            raise RuntimeError("V94 training mutated immutable environmental inputs")
        candidate_bindings = {
            **bindings,
            "training_source_sha256": source_hashes[str(config["sources"]["training_source"])],
            "fixed_final_optimizer_updates": optimizer_update,
            "class_weight_inventory_sha256": config["dataset"][
                "inverse_sqrt_class_weight_inventory_sha256"
            ],
        }
        candidate_metadata = publish_fixed_final_candidate_v94(
            candidate_path, collection, bindings=candidate_bindings
        )
    audit.assert_clean()

    gates = {
        "all_960_rows_consumed_once_in_each_of_three_epochs": row_cursor == EXPECTED_MICRO_ROWS,
        "all_396_changed_sides_received_paired_wrong_margin": changed_seen
        == EXPECTED_PAIRED_MARGIN_ROWS,
        "all_54_fixed_causal_sides_received_zero_payload_margin": causal_seen
        == EXPECTED_CAUSAL_MARGIN_ROWS,
        "fixed_final_update_360_reached": optimizer_update == EXPECTED_OPTIMIZER_UPDATES,
        "only_fresh_110592_parameters_trainable": len(parameters) == 2
        and sum(parameter.numel() for parameter in parameters) == EXPECTED_FRESH_PARAMETER_COUNT,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(record["gradient_l2_before_clip"]))
            and float(record["gradient_l2_before_clip"]) > 0.0
            for record in history
        ),
        "all_40_memory_hashes_invariant": memory_hashes_after == memory_hashes_before,
        "all_40_zero_control_hashes_invariant": zero_hashes_after == zero_hashes_before,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"V94 training gate failed: {gates}")
    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 94,
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
            "target_module": TARGET_MODULE,
            "parameter_count": EXPECTED_FRESH_PARAMETER_COUNT,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": candidate_metadata["state_sha256"],
            "starts_from_exact_v85_only": True,
            "v86_through_v93_loaded": False,
            "unmerged": True,
        },
        "training_protocol": training,
        "micro_rows_consumed": row_cursor,
        "unique_training_rows": len(rows),
        "training_scene_count": len(cpu_memories),
        "paired_wrong_margin_rows_consumed": changed_seen,
        "causal_margin_rows_consumed": causal_seen,
        "optimizer_updates": optimizer_update,
        "resumed_from": resumed_from,
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
            "runtime_promotion_authorized": False,
        },
        "loaded_file_count": len(audit.unique_paths),
        "loaded_file_inventory_sha256": canonical_sha256_v85(audit.unique_paths),
        "protected_read_count": len(audit.forbidden_accesses()),
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_create_json_v85(report_path, report)
    return report


def authenticate_training_report_v94(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    preflight = authenticate_cpu_preflight_v94(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["training_report"])
    report = strict_json_v94(path)
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != EXPECTED_OPTIMIZER_UPDATES
        or report.get("micro_rows_consumed") != EXPECTED_MICRO_ROWS
        or report.get("paired_wrong_margin_rows_consumed") != EXPECTED_PAIRED_MARGIN_ROWS
        or report.get("causal_margin_rows_consumed") != EXPECTED_CAUSAL_MARGIN_ROWS
        or report.get("protected_read_count") != 0
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V94 fixed-final training report changed")
    return {**preflight, "training_report_sha256": sha256_file_v85(path)}


def run_topology_smoke_v94(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Load the real local Gemma stack and prove V94's sole trainable surface."""

    config = load_config_v94(config_path, allow_draft=False)
    preflight = authenticate_cpu_preflight_v94(config, config_path=config_path)
    audit = FileAccessAudit(
        forbidden_component_names=_FORBIDDEN_COMPONENTS,
        block_forbidden=True,
    )
    with audit:
        runtime = load_runtime_config(config["sources"]["runtime_config"])
        language_config = runtime["language"]
        language = load_local_language_model(
            str(language_config["model_id"]),
            str(language_config["revision"]),
            str(language_config["dtype"]),
            freeze=True,
            local_files_only=True,
            backend="gemma4",
            decoder_gradient_checkpointing=False,
        )
        collection = install_lora_banks(
            language.model, combined_lora_settings_v94(runtime, config)
        )
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V94 real-model LoRA installation failed")
        frozen = load_frozen_v85_stack_v94(
            collection, config["sources"]["frozen_v85_checkpoint"]
        )
        collection.assert_trainable_surface(language.model)
        fresh = collection.bank(FRESH_BANK_NAME).installation
        adapter = fresh.adapters[0]
        checks = {
            "real_gemma_loaded": True,
            "device_is_mps": language.device.type == "mps",
            "exact_eight_bank_stack": len(collection.banks) == 8,
            "exact_seven_frozen_banks": len(
                [bank for bank in collection.banks if not bank.settings.trainable]
            )
            == 7,
            "sole_fresh_bank_trainable": collection.trainable_parameter_count
            == EXPECTED_FRESH_PARAMETER_COUNT,
            "exact_total_adapter_parameters": collection.parameter_count
            == EXPECTED_TOTAL_PARAMETER_COUNT,
            "target_module_exact": fresh.target_names == (TARGET_MODULE,),
            "lora_a_shape_exact": list(adapter.lora_a.shape) == [8, 1536],
            "lora_b_shape_exact": list(adapter.lora_b.shape) == [12288, 8],
            "fresh_initial_state_exact": fresh.state_sha256()
            == config["bridge"]["expected_initial_state_sha256"],
            "fresh_output_starts_zero": torch.count_nonzero(adapter.lora_b).item()
            == 0,
            "v85_parent_authenticated": frozen["adapter_sha256"]
            == config["sources"]["frozen_v85_adapter_sha256"],
        }
    audit.assert_clean()
    checks["protected_read_count_zero"] = len(audit.forbidden_accesses()) == 0
    if not all(checks.values()):
        raise RuntimeError(f"V94 real-model topology smoke failed: {checks}")
    return {
        "artifact": "gemma4_v94_real_model_topology_smoke_v1",
        "schema_version": 94,
        "passed": True,
        "config_sha256": preflight["config_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "model_id": config["sources"]["model_id"],
        "model_revision": config["sources"]["model_revision"],
        "checks": checks,
        "protected_read_count": len(audit.forbidden_accesses()),
        "oracle_loaded": False,
        "evaluation_labels_loaded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--topology-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.topology_smoke:
        print(
            json.dumps(
                run_topology_smoke_v94(args.config),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    report = run_training_v94(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "optimizer_updates": report["optimizer_updates"],
                "micro_rows_consumed": report["micro_rows_consumed"],
                "candidate": report["candidate"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_ARTIFACT",
    "CHECKPOINT_ARTIFACT",
    "METADATA_FILENAME",
    "TRAINING_ARTIFACT",
    "WEIGHTS_FILENAME",
    "authenticate_training_report_v94",
    "combined_lora_settings_v94",
    "discover_resume_checkpoint_v94",
    "load_fixed_final_bridge_v94",
    "load_frozen_v85_stack_v94",
    "main",
    "multiscene_objective_v94",
    "publish_fixed_final_candidate_v94",
    "restore_resume_checkpoint_v94",
    "run_topology_smoke_v94",
    "run_training_v94",
    "save_resume_checkpoint_v94",
    "strict_json_v94",
]
