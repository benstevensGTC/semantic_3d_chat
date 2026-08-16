"""Train the fixed-final V85 multi-scene Gemma bridge.

The experiment consumes every one of the 576 preregistered training rows once.
Its only trainable parameters are a fresh rank-4 LoRA bank in Gemma's final
full-attention MLP.  Every row receives answer CE; the 80 preregistered changed
sides additionally receive a paired-wrong-scene hinge.  Development is never
opened here and no checkpoint is selected by behavior.
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
    CONFIG,
    FRESH_BANK_NAME,
    TARGET_MODULE,
    _authenticate_sources,
    atomic_create_json_v85,
    authenticate_cpu_preflight_v85,
    canonical_sha256_v85,
    load_config_v85,
    load_scene_memories_v85,
    ordered_training_rows_v85,
    resolve_v85,
    sha256_file_v85,
    split_preflight_v85,
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

TRAINING_ARTIFACT: Final[str] = "gemma4_v85_strict_multiscene_training_v1"
CANDIDATE_ARTIFACT: Final[str] = "gemma4_v85_strict_multiscene_fixed_final_v1"
CHECKPOINT_ARTIFACT: Final[str] = "gemma4_v85_strict_multiscene_resume_v1"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred", "final"}
)


def strict_json_v85(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V85 JSON must be an object: {source}")
    return value


def combined_lora_settings_v85(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    base = lora_banks_settings(runtime_config)
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


def load_frozen_v54_banks_v85(
    collection: LoRABankCollection, checkpoint: str | Path
) -> dict[str, Any]:
    """Load exactly six inherited banks and retain fresh V85 at update zero."""

    root = resolve_v85(checkpoint)
    weights = root / "adapter.safetensors"
    metadata_path = root / "runtime_metadata.json"
    metadata = strict_json_v85(metadata_path)
    hashes = metadata.get("lora_bank_state_sha256")
    modules = metadata.get("lora_bank_wrapped_modules")
    if not isinstance(hashes, Mapping) or not isinstance(modules, Mapping):
        raise TypeError("V85 V54 source lacks named LoRA metadata")
    frozen = [bank for bank in collection.banks if not bank.settings.trainable]
    fresh = [bank for bank in collection.banks if bank.settings.trainable]
    if (
        len(frozen) != 6
        or len(fresh) != 1
        or fresh[0].settings.name != FRESH_BANK_NAME
        or set(hashes) != {bank.settings.name for bank in frozen}
    ):
        raise ValueError("V85 requires six frozen V54 banks and one sole fresh bank")
    archive = load_file(str(weights), device="cpu")
    for bank in frozen:
        name = bank.settings.name
        prefix = f"lora_banks.{name}."
        state = {
            key[len(prefix) :]: value
            for key, value in archive.items()
            if key.startswith(prefix)
        }
        bank.installation.state_module.load_state_dict(state, strict=True)
        if list(bank.installation.target_names) != modules[name]:
            raise ValueError(f"V85 frozen bank targets changed: {name}")
        if bank.installation.state_sha256() != hashes[name]:
            raise ValueError(f"V85 frozen bank state changed: {name}")
    installation = fresh[0].installation
    if (
        installation.target_names != (TARGET_MODULE,)
        or installation.parameter_count != 55_296
        or installation.state_sha256()
        != fresh[0].settings.expected_initial_state_sha256
        or any(
            torch.count_nonzero(adapter.lora_b).item() != 0
            for adapter in installation.adapters
        )
    ):
        raise ValueError("V85 fresh bridge did not start at deterministic zero output")
    collection.validate_state()
    return {
        "adapter_sha256": sha256_file_v85(weights),
        "runtime_metadata_sha256": sha256_file_v85(metadata_path),
        "frozen_bank_state_sha256": dict(hashes),
        "fresh_initial_state_sha256": installation.state_sha256(),
    }


def pair_margin_objective_v85(
    correct_nll: torch.Tensor,
    paired_wrong_nll: torch.Tensor,
    *,
    target_margin: float,
    ce_weight: float,
    margin_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if correct_nll.ndim != 0 or paired_wrong_nll.ndim != 0:
        raise ValueError("V85 NLL inputs must be scalar")
    if not torch.isfinite(correct_nll) or not torch.isfinite(paired_wrong_nll):
        raise ValueError("V85 NLL inputs must be finite")
    observed_margin = paired_wrong_nll - correct_nll
    penalty = torch.relu(
        correct_nll
        - paired_wrong_nll
        + torch.as_tensor(target_margin, device=correct_nll.device)
    )
    objective = ce_weight * correct_nll + margin_weight * penalty
    return objective, observed_margin, penalty


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _fresh_state(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    installation = collection.bank(FRESH_BANK_NAME).installation
    if len(installation.adapters) != 1:
        raise ValueError("V85 fresh bridge must wrap exactly one module")
    adapter = installation.adapters[0]
    return {
        "lora_a": adapter.lora_a.detach().cpu().contiguous(),
        "lora_b": adapter.lora_b.detach().cpu().contiguous(),
    }


def _load_fresh_state(
    collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]
) -> None:
    if set(archive) != {"lora_a", "lora_b"}:
        raise ValueError("V85 fresh state tensor inventory changed")
    installation = collection.bank(FRESH_BANK_NAME).installation
    adapter = installation.adapters[0]
    expected = {"lora_a": adapter.lora_a, "lora_b": adapter.lora_b}
    with torch.no_grad():
        for name, parameter in expected.items():
            value = archive[name]
            if value.shape != parameter.shape or value.dtype != torch.float32:
                raise ValueError(f"V85 resumed {name} shape or dtype changed")
            parameter.copy_(value.to(parameter.device))
    collection.validate_state()


def _optimizer_tensors(optimizer: torch.optim.Optimizer) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    state = optimizer.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for parameter_index, values in state["state"].items():
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("V85 AdamW state must be tensor-only")
            tensors[f"optimizer.{parameter_index}.{name}"] = value.detach().cpu().contiguous()
    groups: list[dict[str, Any]] = []
    for group in state["param_groups"]:
        normalized = dict(group)
        normalized["params"] = [int(value) for value in normalized["params"]]
        groups.append(normalized)
    return tensors, groups


def _restore_optimizer(
    optimizer: torch.optim.Optimizer,
    archive: Mapping[str, torch.Tensor],
    groups: Sequence[Mapping[str, Any]],
) -> None:
    state: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in archive.items():
        if not key.startswith("optimizer."):
            continue
        _prefix, raw_index, name = key.split(".", 2)
        state.setdefault(int(raw_index), {})[name] = value.detach().cpu()
    optimizer.load_state_dict(
        {
            "state": state,
            "param_groups": [dict(group) for group in groups],
        }
    )


def save_resume_checkpoint_v85(
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
        existing = strict_json_v85(destination / "state.json")
        if (
            existing.get("update") == update
            and existing.get("row_cursor") == row_cursor
            and existing.get("fresh_state_sha256")
            == collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        ):
            return destination
        raise FileExistsError(f"V85 checkpoint collision: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=root))
    try:
        fresh = _fresh_state(collection)
        optimizer_tensors, groups = _optimizer_tensors(optimizer)
        tensors = {**fresh, **optimizer_tensors}
        weights = temporary / "state.safetensors"
        save_file(
            tensors,
            str(weights),
            metadata={
                "artifact": CHECKPOINT_ARTIFACT,
                "environmental_memory_serialized": "false",
                "questions_or_answers_serialized": "false",
            },
        )
        state = {
            "artifact": CHECKPOINT_ARTIFACT,
            "schema_version": 85,
            "status": "resumable_training_state",
            "update": update,
            "row_cursor": row_cursor,
            "row_order_sha256": row_order_sha256,
            "fresh_state_sha256": collection.bank(
                FRESH_BANK_NAME
            ).installation.state_sha256(),
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


def discover_resume_checkpoint_v85(
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
        metadata = strict_json_v85(path / "state.json")
        update = int(metadata.get("update", -1))
        if (
            metadata.get("artifact") != CHECKPOINT_ARTIFACT
            or metadata.get("row_cursor") != update * gradient_accumulation_rows
            or metadata.get("row_order_sha256") != row_order_sha256
            or metadata.get("bindings") != dict(bindings)
            or metadata.get("environmental_memory_serialized") is not False
            or metadata.get("questions_or_answers_serialized") is not False
            or metadata.get("oracle_serialized") is not False
            or metadata.get("tensor_file_sha256")
            != sha256_file_v85(path / "state.safetensors")
        ):
            raise ValueError(f"V85 resume checkpoint authentication failed: {path}")
        candidates.append((update, path, metadata))
    if not candidates:
        return None
    _update, path, metadata = max(candidates, key=lambda value: value[0])
    return path, metadata


def restore_resume_checkpoint_v85(
    checkpoint: Path,
    metadata: Mapping[str, Any],
    collection: LoRABankCollection,
    optimizer: torch.optim.Optimizer,
) -> None:
    archive = load_file(str(checkpoint / "state.safetensors"), device="cpu")
    _load_fresh_state(
        collection,
        {name: archive[name] for name in ("lora_a", "lora_b")},
    )
    if (
        collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        != metadata["fresh_state_sha256"]
    ):
        raise ValueError("V85 resumed bridge hash changed")
    groups = metadata.get("optimizer_param_groups")
    if not isinstance(groups, list):
        raise TypeError("V85 resume optimizer groups are missing")
    _restore_optimizer(optimizer, archive, groups)


def publish_fixed_final_candidate_v85(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    root = resolve_v85(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V85 create-once fixed-final candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / WEIGHTS_FILENAME
        save_file(
            _fresh_state(collection),
            str(weights),
            metadata={
                "artifact": CANDIDATE_ARTIFACT,
                "environmental_memory_serialized": "false",
                "questions_or_answers_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        fresh = collection.bank(FRESH_BANK_NAME).installation
        metadata = {
            "artifact": CANDIDATE_ARTIFACT,
            "schema_version": 85,
            "status": "fixed_final_diagnostic_awaiting_development_score",
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "parameter_count": fresh.parameter_count,
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v85(weights),
            "environmental_memory_serialized": False,
            "questions_or_answers_serialized": False,
            "oracle_serialized": False,
            "development_scored": False,
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


def load_fixed_final_bridge_v85(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    root = resolve_v85(candidate)
    metadata = strict_json_v85(root / METADATA_FILENAME)
    weights = root / WEIGHTS_FILENAME
    if (
        metadata.get("artifact") != CANDIDATE_ARTIFACT
        or metadata.get("status")
        != "fixed_final_diagnostic_awaiting_development_score"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("weights_sha256") != sha256_file_v85(weights)
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("development_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V85 fixed-final candidate authentication failed")
    _load_fresh_state(collection, load_file(str(weights), device="cpu"))
    if (
        collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        != metadata["state_sha256"]
    ):
        raise ValueError("V85 fixed-final candidate state hash changed")
    return metadata


def authenticate_training_report_v85(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    preflight = authenticate_cpu_preflight_v85(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["training_report"])
    report = strict_json_v85(path)
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != config["training"]["optimizer_updates"]
        or report.get("rows_consumed") != config["split"]["train_row_count"]
        or report.get("development_behavior_scored") is not False
        or report.get("protected_read_count") != 0
        or report.get("official_validation_loaded") is not False
        or report.get("official_test_loaded") is not False
        or report.get("deferred_final_loaded") is not False
        or report.get("sealed_historical_16_loaded") is not False
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V85 fixed-final training report changed")
    return {
        **preflight,
        "training_report_sha256": sha256_file_v85(path),
    }


def run_training_v85(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v85(config_path)
    source_hashes = _authenticate_sources(config)
    split_report, train_rows, _development_rows = split_preflight_v85(config)
    preflight = authenticate_cpu_preflight_v85(config, config_path=config_path)
    outputs = config["outputs"]
    report_path = resolve_v85(outputs["training_report"])
    candidate_path = resolve_v85(outputs["fixed_final_candidate"])
    if report_path.exists() or candidate_path.exists():
        raise FileExistsError("V85 fixed-final outputs already exist")

    schedule = ordered_training_rows_v85(
        train_rows, seed=int(config["training"]["row_order_seed"])
    )
    if canonical_sha256_v85([[row.scene_id, row.question_id] for row in schedule]) != config[
        "training"
    ]["row_order_sha256"]:
        raise RuntimeError("V85 fixed training schedule changed")

    # This immutable environment boundary intentionally precedes Gemma loading
    # and every question-tokenization operation.
    cpu_memories, memory_hashes_before = load_scene_memories_v85(
        config, train_rows, split_name="train"
    )
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
            raise RuntimeError("V85 full-model training requires local MPS")
        collection = install_lora_banks(
            language.model, combined_lora_settings_v85(runtime, config)
        )
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V85 LoRA bank installation failed")
        frozen_source = load_frozen_v54_banks_v85(
            collection, config["sources"]["base_checkpoint"]
        )
        collection.assert_trainable_surface(language.model)
        fresh = collection.bank(FRESH_BANK_NAME).installation
        initial_state_sha256 = fresh.state_sha256()
        system_prompt = str(language_config["system_prompt"])
        memory_by_scene = {
            scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
            for scene_id, memory in cpu_memories.items()
        }
        parameters = collection.parameters()
        training = config["training"]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        bindings = {
            **preflight,
            "base_adapter_sha256": frozen_source["adapter_sha256"],
        }
        accumulation = int(training["gradient_accumulation_rows"])
        resumed = discover_resume_checkpoint_v85(
            outputs["work_root"],
            bindings=bindings,
            row_order_sha256=str(training["row_order_sha256"]),
            gradient_accumulation_rows=accumulation,
        )
        history: list[dict[str, Any]] = []
        row_cursor = 0
        optimizer_update = 0
        resumed_from: str | None = None
        if resumed is not None:
            checkpoint, metadata = resumed
            restore_resume_checkpoint_v85(checkpoint, metadata, collection, optimizer)
            row_cursor = int(metadata["row_cursor"])
            optimizer_update = int(metadata["update"])
            history = list(metadata["history"])
            resumed_from = checkpoint.relative_to(PROJECT_ROOT).as_posix()
        if row_cursor > len(schedule) or row_cursor % accumulation:
            raise ValueError("V85 resume cursor is outside the fixed schedule")

        language.decoder_module.train()
        collection.train()
        optimizer.zero_grad(set_to_none=True)
        ce_weight = float(training["correct_scene_answer_ce_weight"])
        margin_weight = float(training["changed_side_wrong_scene_margin_weight"])
        target_margin = float(training["changed_side_wrong_scene_target_margin_nll"])
        changed_sides_seen = sum(row.expected_change for row in schedule[:row_cursor])
        interval: list[dict[str, float | bool]] = []
        while row_cursor < len(schedule):
            row = schedule[row_cursor]
            prepared, _layout = _prepared_v84(
                language, system_prompt, memory_by_scene[row.scene_id], row
            )
            correct_tail = _answer_tail(language, prepared)
            correct_nll = correct_tail.mean_nll.float()
            if row.expected_change:
                wrong_prepared, _wrong_layout = _prepared_v84(
                    language,
                    system_prompt,
                    memory_by_scene[row.paired_scene_id],
                    row,
                )
                wrong_tail = _answer_tail(language, wrong_prepared)
                objective, observed_margin, penalty = pair_margin_objective_v85(
                    correct_nll,
                    wrong_tail.mean_nll.float(),
                    target_margin=target_margin,
                    ce_weight=ce_weight,
                    margin_weight=margin_weight,
                )
                changed_sides_seen += 1
                interval.append(
                    {
                        "correct_nll": float(correct_nll.detach().cpu()),
                        "objective": float(objective.detach().cpu()),
                        "changed_side": True,
                        "wrong_minus_correct_nll": float(observed_margin.detach().cpu()),
                        "margin_penalty": float(penalty.detach().cpu()),
                    }
                )
                del wrong_tail, wrong_prepared, observed_margin, penalty
            else:
                objective = ce_weight * correct_nll
                interval.append(
                    {
                        "correct_nll": float(correct_nll.detach().cpu()),
                        "objective": float(objective.detach().cpu()),
                        "changed_side": False,
                        "wrong_minus_correct_nll": 0.0,
                        "margin_penalty": 0.0,
                    }
                )
            if not torch.isfinite(objective):
                raise RuntimeError("V85 training objective is nonfinite")
            (objective / accumulation).backward()
            row_cursor += 1
            del prepared, correct_tail, correct_nll, objective

            if row_cursor % accumulation:
                continue
            gradients = collection.gradient_norms()
            gradient_l2 = float(gradients["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V85 gradient is zero or nonfinite")
            clip_return = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            clip_l2 = float(clip_return.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V85 clipped gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            changed_interval = [value for value in interval if value["changed_side"]]
            record = {
                "update": optimizer_update,
                "row_cursor": row_cursor,
                "mean_correct_nll": sum(float(value["correct_nll"]) for value in interval)
                / len(interval),
                "mean_objective": sum(float(value["objective"]) for value in interval)
                / len(interval),
                "changed_sides": len(changed_interval),
                "mean_changed_wrong_minus_correct_nll": (
                    sum(
                        float(value["wrong_minus_correct_nll"])
                        for value in changed_interval
                    )
                    / len(changed_interval)
                    if changed_interval
                    else None
                ),
                "mean_changed_margin_penalty": (
                    sum(float(value["margin_penalty"]) for value in changed_interval)
                    / len(changed_interval)
                    if changed_interval
                    else None
                ),
                "gradient_l2_before_clip": gradient_l2,
                "clip_return_l2": clip_l2,
                "state_sha256": fresh.state_sha256(),
            }
            history.append(record)
            interval.clear()
            if optimizer_update in {1, 36, 72} or optimizer_update % 6 == 0:
                print(
                    json.dumps(
                        {
                            "event": "v85_train_update",
                            "update": optimizer_update,
                            "total_updates": training["optimizer_updates"],
                            "row_cursor": row_cursor,
                            "mean_correct_nll": record["mean_correct_nll"],
                            "changed_sides_seen": changed_sides_seen,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if (
                optimizer_update % int(training["checkpoint_every_optimizer_updates"])
                == 0
                or optimizer_update == int(training["optimizer_updates"])
            ):
                save_resume_checkpoint_v85(
                    outputs["work_root"],
                    collection,
                    optimizer,
                    update=optimizer_update,
                    row_cursor=row_cursor,
                    history=history,
                    bindings=bindings,
                    row_order_sha256=str(training["row_order_sha256"]),
                )
            torch.mps.empty_cache()

        if (
            optimizer_update != int(training["optimizer_updates"])
            or row_cursor != int(config["split"]["train_row_count"])
            or changed_sides_seen != int(config["split"]["train_changed_side_count"])
            or len(history) != optimizer_update
        ):
            raise RuntimeError("V85 fixed-final schedule did not complete exactly")
        language.decoder_module.eval()
        collection.eval()
        memory_hashes_after = {
            scene_id: prefix_sha256(memory.detach().cpu())
            for scene_id, memory in memory_by_scene.items()
        }
        if memory_hashes_after != memory_hashes_before:
            raise RuntimeError("V85 training mutated immutable scene memory")

        candidate_bindings = {
            **bindings,
            "training_source_sha256": source_hashes[
                str(config["sources"]["training_source"])
            ],
            "fixed_final_optimizer_updates": optimizer_update,
            "row_order_sha256": training["row_order_sha256"],
        }
        candidate_metadata = publish_fixed_final_candidate_v85(
            candidate_path,
            collection,
            bindings=candidate_bindings,
        )
    audit.assert_clean()

    gates = {
        "all_576_rows_consumed_exactly_once": row_cursor == 576,
        "all_80_changed_sides_received_pair_margin": changed_sides_seen == 80,
        "fixed_final_update_72_reached": optimizer_update == 72,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(row["gradient_l2_before_clip"]))
            and float(row["gradient_l2_before_clip"]) > 0.0
            for row in history
        ),
        "memory_hash_invariant": memory_hashes_after == memory_hashes_before,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"V85 fixed-final training gate failed: {gates}")
    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 85,
        "status": "fixed_final_training_complete_not_promoted",
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "device": "mps",
        "model_id": config["sources"]["model_id"],
        "model_revision": config["sources"]["model_revision"],
        "strict_input_contract": config["strict_input_contract"],
        "split_preflight": split_report,
        "source_hashes": source_hashes,
        "frozen_source": frozen_source,
        "trainable_bridge": {
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "parameter_count": 55_296,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": candidate_metadata["state_sha256"],
            "starts_from_v84_candidate": False,
            "unmerged": True,
        },
        "training_protocol": config["training"],
        "rows_consumed": row_cursor,
        "changed_sides_consumed": changed_sides_seen,
        "optimizer_updates": optimizer_update,
        "resumed_from": resumed_from,
        "training_history": history,
        "scene_memories": {
            "compiled_before_question_tokenization": True,
            "shape_each": [1, 738, 1536],
            "hashes_before": memory_hashes_before,
            "hashes_after": memory_hashes_after,
            "hash_invariant": True,
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
        "development_behavior_scored": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "sealed_historical_16_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_create_json_v85(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    report = run_training_v85(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "optimizer_updates": report["optimizer_updates"],
                "rows_consumed": report["rows_consumed"],
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
    "authenticate_training_report_v85",
    "combined_lora_settings_v85",
    "discover_resume_checkpoint_v85",
    "load_fixed_final_bridge_v85",
    "load_frozen_v54_banks_v85",
    "main",
    "pair_margin_objective_v85",
    "publish_fixed_final_candidate_v85",
    "restore_resume_checkpoint_v85",
    "run_training_v85",
    "save_resume_checkpoint_v85",
    "strict_json_v85",
]
