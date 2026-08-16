"""Train V87's fixed-final class-balanced scene-one bridge.

The exact V85 seven-bank runtime stack and V86 up-projection bridge are frozen.
Only a fresh rank-8 bank on the disjoint layer-34 gate projection is optimized.
Every scene-one row appears exactly once in each of eight fixed epochs; opaque
answer classes are round-robin interleaved and inverse-frequency weighted so
all nineteen classes contribute identical aggregate CE mass per epoch.
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
from semantic_3d_chat.evaluation.v86_scene1_demo_preflight import (
    causal_rows_v86,
    load_scene1_memory_v86,
    load_scene1_rows_v86,
    zero_payload_memory_v86,
)
from semantic_3d_chat.evaluation.v87_scene1_balanced_preflight import (
    CONFIG,
    FRESH_BANK_NAME,
    PARENT_BANK_NAME,
    PARENT_TARGET_MODULE,
    TARGET_MODULE,
    answer_class_balance_v87,
    authenticate_cpu_preflight_v87,
    authenticate_sources_v87,
    balanced_schedule_v87,
    load_config_v87,
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
from semantic_3d_chat.training.train_v86_scene1_demo import (
    zero_payload_margin_objective_v86,
)

TRAINING_ARTIFACT: Final[str] = "gemma4_v87_scene1_balanced_training_v1"
CANDIDATE_ARTIFACT: Final[str] = "gemma4_v87_scene1_balanced_fixed_final_v1"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)


def strict_json_v87(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V87 JSON must contain one object: {source}")
    return value


def combined_lora_settings_v87(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    base = lora_banks_settings(runtime_config)
    if len(base.banks) != 7 or any(bank.trainable for bank in base.banks):
        raise ValueError("V87 requires the exact seven frozen V85 runtime banks")
    frozen = experiment["frozen_stack"]
    parent = LoRABankSettings(
        name=PARENT_BANK_NAME,
        trainable=False,
        adapter=LoRASettings(
            enabled=True,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_modules=(str(frozen["v86_bank_target_module"]),),
        ),
        initialization_algorithm="checkpoint_overwrite",
        initialization_seed=None,
        expected_initial_state_sha256=str(frozen["v86_bank_state_sha256"]),
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
            target_modules=(str(bridge["target_module"]),),
        ),
        initialization_algorithm=str(bridge["initialization_algorithm"]),
        initialization_seed=int(bridge["initialization_seed"]),
        expected_initial_state_sha256=str(bridge["expected_initial_state_sha256"]),
    )
    return LoRABanksSettings(base.banks + (parent, fresh))


def _copy_two_tensor_state(
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


def load_frozen_stack_v87(
    collection: LoRABankCollection,
    *,
    v85_checkpoint: str | Path,
    v86_checkpoint: str | Path,
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and load seven V85 banks plus the exact frozen V86 bank."""

    v85_root = resolve_v85(v85_checkpoint)
    v85_weights = v85_root / "adapter.safetensors"
    v85_metadata_path = v85_root / "runtime_metadata.json"
    v85_metadata = strict_json_v87(v85_metadata_path)
    hashes = v85_metadata.get("lora_bank_state_sha256")
    modules = v85_metadata.get("lora_bank_wrapped_modules")
    if not isinstance(hashes, Mapping) or not isinstance(modules, Mapping):
        raise TypeError("V87 V85 source lacks named LoRA metadata")
    v85_banks = [collection.bank(name) for name in hashes]
    if len(v85_banks) != 7 or any(bank.settings.trainable for bank in v85_banks):
        raise ValueError("V87 requires seven frozen V85 source banks")
    v85_archive = load_file(str(v85_weights), device="cpu")
    for bank in v85_banks:
        name = bank.settings.name
        prefix = f"lora_banks.{name}."
        state = {
            key[len(prefix) :]: value
            for key, value in v85_archive.items()
            if key.startswith(prefix)
        }
        bank.installation.state_module.load_state_dict(state, strict=True)
        if list(bank.installation.target_names) != modules[name]:
            raise ValueError(f"V87 frozen V85 bank targets changed: {name}")
        if bank.installation.state_sha256() != hashes[name]:
            raise ValueError(f"V87 frozen V85 bank state changed: {name}")

    v86_root = resolve_v85(v86_checkpoint)
    v86_weights = v86_root / "bridge.safetensors"
    v86_metadata_path = v86_root / "runtime_metadata.json"
    v86_metadata = strict_json_v87(v86_metadata_path)
    parent = collection.bank(PARENT_BANK_NAME)
    if (
        parent.settings.trainable
        or parent.installation.target_names != (PARENT_TARGET_MODULE,)
        or v86_metadata.get("state_sha256") != experiment["frozen_stack"]["v86_bank_state_sha256"]
        or v86_metadata.get("weights_sha256") != sha256_file_v85(v86_weights)
        or v86_metadata.get("questions_or_answers_serialized") is not False
        or v86_metadata.get("environmental_memory_serialized") is not False
    ):
        raise ValueError("V87 frozen V86 bridge metadata changed")
    _copy_two_tensor_state(
        parent.installation,
        load_file(str(v86_weights), device="cpu"),
        context="V87 frozen V86 bridge",
    )
    if parent.installation.state_sha256() != v86_metadata["state_sha256"]:
        raise ValueError("V87 frozen V86 bridge state changed after load")

    fresh = collection.bank(FRESH_BANK_NAME)
    if (
        not fresh.settings.trainable
        or fresh.installation.target_names != (TARGET_MODULE,)
        or fresh.installation.parameter_count != 110592
        or fresh.installation.state_sha256()
        != experiment["bridge"]["expected_initial_state_sha256"]
        or any(
            torch.count_nonzero(adapter.lora_b).item() != 0
            for adapter in fresh.installation.adapters
        )
    ):
        raise ValueError("V87 fresh bridge did not start at exact zero output")
    if len([bank for bank in collection.banks if not bank.settings.trainable]) != 8:
        raise ValueError("V87 frozen bank count changed")
    collection.validate_state()
    return {
        "v85_adapter_sha256": sha256_file_v85(v85_weights),
        "v85_metadata_sha256": sha256_file_v85(v85_metadata_path),
        "v85_bank_state_sha256": dict(hashes),
        "v86_bridge_sha256": sha256_file_v85(v86_weights),
        "v86_metadata_sha256": sha256_file_v85(v86_metadata_path),
        "v86_bridge_state_sha256": parent.installation.state_sha256(),
        "frozen_bank_count": 8,
        "fresh_initial_state_sha256": fresh.installation.state_sha256(),
    }


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _fresh_state(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    installation = collection.bank(FRESH_BANK_NAME).installation
    if len(installation.adapters) != 1:
        raise ValueError("V87 fresh bank must wrap one module")
    adapter = installation.adapters[0]
    return {
        "lora_a": adapter.lora_a.detach().cpu().contiguous(),
        "lora_b": adapter.lora_b.detach().cpu().contiguous(),
    }


def _load_fresh_state(collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]) -> None:
    _copy_two_tensor_state(
        collection.bank(FRESH_BANK_NAME).installation,
        archive,
        context="V87 fresh bridge",
    )
    collection.validate_state()


def publish_fixed_final_candidate_v87(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    root = resolve_v85(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V87 fixed-final candidate exists: {root}")
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
            "schema_version": 87,
            "status": "fixed_final_awaiting_preregistered_acceptance_gates",
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "parameter_count": fresh.parameter_count,
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v85(weights),
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


def load_fixed_final_bridge_v87(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    root = resolve_v85(candidate)
    metadata = strict_json_v87(root / METADATA_FILENAME)
    weights = root / WEIGHTS_FILENAME
    if (
        metadata.get("artifact") != CANDIDATE_ARTIFACT
        or metadata.get("status") != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("weights_sha256") != sha256_file_v85(weights)
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V87 fixed-final candidate authentication failed")
    _load_fresh_state(collection, load_file(str(weights), device="cpu"))
    if collection.bank(FRESH_BANK_NAME).installation.state_sha256() != metadata["state_sha256"]:
        raise ValueError("V87 fixed-final bridge state changed")
    return metadata


def authenticate_training_report_v87(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    preflight = authenticate_cpu_preflight_v87(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["training_report"])
    report = strict_json_v87(path)
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != 184
        or report.get("micro_rows_consumed") != 1104
        or report.get("causal_margin_rows_consumed") != 24
        or report.get("protected_read_count") != 0
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V87 fixed-final training report changed")
    return {**preflight, "training_report_sha256": sha256_file_v85(path)}


def run_training_v87(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v87(config_path)
    source_hashes = authenticate_sources_v87(config)
    preflight = authenticate_cpu_preflight_v87(config, config_path=config_path)
    rows = load_scene1_rows_v86(config)
    _class_counts, class_weights = answer_class_balance_v87(config, rows)
    causal_ids = {row.question_id for row in causal_rows_v86(config, rows)}
    schedule = balanced_schedule_v87(
        rows,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["training"]["epochs"]),
    )
    schedule_hash = canonical_sha256_v85([[epoch, row.question_id] for epoch, row in schedule])
    if schedule_hash != config["training"]["row_order_sha256"]:
        raise RuntimeError("V87 fixed class-balanced schedule changed")
    outputs = config["outputs"]
    report_path = resolve_v85(outputs["training_report"])
    candidate_path = resolve_v85(outputs["fixed_final_candidate"])
    if report_path.exists() or candidate_path.exists():
        raise FileExistsError("V87 fixed-final outputs already exist")

    cpu_memory, memory_hash_before, _metadata = load_scene1_memory_v86(config)
    cpu_zero_memory = zero_payload_memory_v86(cpu_memory)
    zero_hash_before = prefix_sha256(cpu_zero_memory)
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
            raise RuntimeError("V87 full-model training requires local MPS")
        collection = install_lora_banks(language.model, combined_lora_settings_v87(runtime, config))
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V87 LoRA bank installation failed")
        frozen_source = load_frozen_stack_v87(
            collection,
            v85_checkpoint=config["sources"]["frozen_v85_checkpoint"],
            v86_checkpoint=config["sources"]["parent_v86_checkpoint"],
            experiment=config,
        )
        collection.assert_trainable_surface(language.model)
        fresh = collection.bank(FRESH_BANK_NAME).installation
        initial_state_sha256 = fresh.state_sha256()
        memory = cpu_memory.to(device=language.device, dtype=torch.bfloat16)
        zero_memory = cpu_zero_memory.to(device=language.device, dtype=torch.bfloat16)
        system_prompt = str(language_config["system_prompt"])
        training = config["training"]
        parameters = collection.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        language.decoder_module.train()
        collection.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation = int(training["gradient_accumulation_rows"])
        target_margin = float(training["zero_payload_target_margin_nll"])
        ce_weight = float(training["answer_ce_weight"])
        margin_weight = float(training["zero_payload_margin_weight"])
        history: list[dict[str, Any]] = []
        interval: list[dict[str, float | bool]] = []
        optimizer_update = 0
        causal_seen = 0
        for cursor, (epoch, row) in enumerate(schedule, 1):
            prepared, _layout = _prepared_v84(language, system_prompt, memory, row)
            tail = _answer_tail(language, prepared)
            correct_nll = tail.mean_nll.float()
            row_weight = float(class_weights[row.answer_class])
            weighted_ce = ce_weight * row_weight * correct_nll
            if row.question_id in causal_ids:
                zero_prepared, _zero_layout = _prepared_v84(
                    language, system_prompt, zero_memory, row
                )
                zero_tail = _answer_tail(language, zero_prepared)
                _unweighted_objective, observed_margin, penalty = zero_payload_margin_objective_v86(
                    correct_nll,
                    zero_tail.mean_nll.float(),
                    target_margin=target_margin,
                    ce_weight=ce_weight,
                    margin_weight=margin_weight,
                )
                objective = weighted_ce + margin_weight * penalty
                causal_seen += 1
                interval.append(
                    {
                        "correct_nll": float(correct_nll.detach().cpu()),
                        "weighted_ce": float(weighted_ce.detach().cpu()),
                        "objective": float(objective.detach().cpu()),
                        "class_weight": row_weight,
                        "causal": True,
                        "zero_minus_correct_nll": float(observed_margin.detach().cpu()),
                        "margin_penalty": float(penalty.detach().cpu()),
                    }
                )
                del zero_prepared, zero_tail, observed_margin, penalty
            else:
                objective = weighted_ce
                interval.append(
                    {
                        "correct_nll": float(correct_nll.detach().cpu()),
                        "weighted_ce": float(weighted_ce.detach().cpu()),
                        "objective": float(objective.detach().cpu()),
                        "class_weight": row_weight,
                        "causal": False,
                        "zero_minus_correct_nll": 0.0,
                        "margin_penalty": 0.0,
                    }
                )
            if not torch.isfinite(objective):
                raise RuntimeError("V87 objective is nonfinite")
            (objective / accumulation).backward()
            del prepared, tail, correct_nll, weighted_ce, objective
            if cursor % accumulation:
                continue
            gradient_l2 = float(collection.gradient_norms()["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V87 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            clip_l2 = float(clipped.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V87 clipped gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            causal_interval = [record for record in interval if record["causal"]]
            record = {
                "update": optimizer_update,
                "row_cursor": cursor,
                "epoch": epoch,
                "mean_correct_nll": sum(float(value["correct_nll"]) for value in interval)
                / len(interval),
                "mean_weighted_ce": sum(float(value["weighted_ce"]) for value in interval)
                / len(interval),
                "mean_objective": sum(float(value["objective"]) for value in interval)
                / len(interval),
                "causal_rows": len(causal_interval),
                "mean_causal_zero_minus_correct_nll": (
                    sum(float(value["zero_minus_correct_nll"]) for value in causal_interval)
                    / len(causal_interval)
                    if causal_interval
                    else None
                ),
                "mean_causal_margin_penalty": (
                    sum(float(value["margin_penalty"]) for value in causal_interval)
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
            if optimizer_update in {1, 92, 184} or optimizer_update % 12 == 0:
                print(
                    json.dumps(
                        {
                            "event": "v87_train_update",
                            "update": optimizer_update,
                            "total_updates": 184,
                            "row_cursor": cursor,
                            "epoch": epoch,
                            "mean_correct_nll": record["mean_correct_nll"],
                            "mean_weighted_ce": record["mean_weighted_ce"],
                            "causal_rows_seen": causal_seen,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            torch.mps.empty_cache()

        if (
            optimizer_update != 184
            or len(schedule) != 1104
            or causal_seen != 24
            or len(history) != 184
        ):
            raise RuntimeError("V87 fixed class-balanced schedule did not complete")
        language.decoder_module.eval()
        collection.eval()
        memory_hash_after = prefix_sha256(memory.detach().cpu())
        zero_hash_after = prefix_sha256(zero_memory.detach().cpu())
        if memory_hash_after != memory_hash_before or zero_hash_after != zero_hash_before:
            raise RuntimeError("V87 training mutated immutable environmental inputs")
        bindings = {
            **preflight,
            "training_source_sha256": source_hashes[str(config["sources"]["training_source"])],
            "v85_adapter_sha256": frozen_source["v85_adapter_sha256"],
            "v86_bridge_sha256": frozen_source["v86_bridge_sha256"],
            "v86_bridge_state_sha256": frozen_source["v86_bridge_state_sha256"],
            "fixed_final_optimizer_updates": optimizer_update,
            "row_order_sha256": schedule_hash,
            "class_weight_inventory_sha256": config["dataset"]["class_weight_inventory_sha256"],
            "scene_memory_prefix_sha256": memory_hash_before,
        }
        candidate_metadata = publish_fixed_final_candidate_v87(
            candidate_path, collection, bindings=bindings
        )
    audit.assert_clean()

    gates = {
        "all_138_rows_consumed_once_in_each_of_eight_epochs": len(schedule) == 1104,
        "all_19_answer_classes_equal_aggregate_ce_mass": True,
        "all_24_causal_margin_rows_consumed": causal_seen == 24,
        "fixed_final_update_184_reached": optimizer_update == 184,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(row["gradient_l2_before_clip"]))
            and float(row["gradient_l2_before_clip"]) > 0.0
            for row in history
        ),
        "memory_hash_invariant": memory_hash_after == memory_hash_before,
        "zero_payload_hash_invariant": zero_hash_after == zero_hash_before,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"V87 training gate failed: {gates}")
    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 87,
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
            "parameter_count": 110592,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": candidate_metadata["state_sha256"],
            "unmerged": True,
        },
        "training_protocol": config["training"],
        "class_balance": {
            "opaque_class_count": 19,
            "answer_class_inventory_sha256": config["dataset"]["answer_class_inventory_sha256"],
            "class_weight_inventory_sha256": config["dataset"]["class_weight_inventory_sha256"],
            "answer_text_serialized": False,
            "equal_aggregate_ce_mass_per_class": True,
        },
        "micro_rows_consumed": len(schedule),
        "unique_training_rows": 138,
        "causal_margin_rows_consumed": causal_seen,
        "optimizer_updates": optimizer_update,
        "training_history": history,
        "scene_memory": {
            "compiled_before_question_tokenization": True,
            "shape": [1, 738, 1536],
            "prefix_sha256_before": memory_hash_before,
            "prefix_sha256_after": memory_hash_after,
            "zero_payload_prefix_sha256_before": zero_hash_before,
            "zero_payload_prefix_sha256_after": zero_hash_after,
            "native_boi_eoi_retained_in_zero_control": True,
            "zero_payload_tokens": 736,
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
        "held_out_generalization_claim": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_create_json_v85(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    report = run_training_v87(args.config)
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
    "METADATA_FILENAME",
    "TRAINING_ARTIFACT",
    "WEIGHTS_FILENAME",
    "authenticate_training_report_v87",
    "combined_lora_settings_v87",
    "load_fixed_final_bridge_v87",
    "load_frozen_stack_v87",
    "main",
    "publish_fixed_final_candidate_v87",
    "run_training_v87",
    "strict_json_v87",
]
