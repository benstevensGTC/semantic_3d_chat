"""Train V89's fixed-final retention-aware scene-one correction.

The exact V85 seven-bank runtime stack plus the V86, V87, and V88 bridges are
frozen. Only a fresh rank-8 adapter on the disjoint layer-27 attention output
projection is optimized. Training consumes the sealed 310-row deterministic
training-only schedule for three epochs; it never reads oracle, validation,
test, or deferred artifacts. The published two-tensor candidate contains no
questions, answers, replay inventories, or environmental memory.
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
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
    canonical_sha256_v85,
    resolve_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v86_scene1_demo_preflight import (
    load_scene1_memory_v86,
    zero_payload_memory_v86,
)
from semantic_3d_chat.evaluation.v89_scene1_retention_preflight import (
    CONFIG,
    FRESH_BANK_NAME,
    TARGET_MODULE,
    V86_BANK_NAME,
    V86_TARGET_MODULE,
    V87_BANK_NAME,
    V87_TARGET_MODULE,
    V88_BANK_NAME,
    V88_TARGET_MODULE,
    authenticate_cpu_preflight_v89,
    authenticate_sources_v89,
    derive_parent_behavior_v89,
    derive_training_items_v89,
    load_canonical_rows_v89,
    load_config_v89,
    strict_json_v89,
    training_inventory_v89,
    training_schedule_v89,
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

TRAINING_ARTIFACT: Final[str] = "gemma4_v89_scene1_retention_training_v1"
CANDIDATE_ARTIFACT: Final[str] = "gemma4_v89_scene1_retention_fixed_final_v1"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)


def combined_lora_settings_v89(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    base = lora_banks_settings(runtime_config)
    if len(base.banks) != 7 or any(bank.trainable for bank in base.banks):
        raise ValueError("V89 requires the exact seven frozen V85 runtime banks")
    frozen = experiment["frozen_stack"]
    v86 = LoRABankSettings(
        name=V86_BANK_NAME,
        trainable=False,
        adapter=LoRASettings(
            enabled=True,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_modules=(V86_TARGET_MODULE,),
        ),
        initialization_algorithm="checkpoint_overwrite",
        initialization_seed=None,
        expected_initial_state_sha256=str(frozen["v86_bank_state_sha256"]),
    )
    v87 = LoRABankSettings(
        name=V87_BANK_NAME,
        trainable=False,
        adapter=LoRASettings(
            enabled=True,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_modules=(V87_TARGET_MODULE,),
        ),
        initialization_algorithm="checkpoint_overwrite",
        initialization_seed=None,
        expected_initial_state_sha256=str(frozen["v87_bank_state_sha256"]),
    )
    v88 = LoRABankSettings(
        name=V88_BANK_NAME,
        trainable=False,
        adapter=LoRASettings(
            enabled=True,
            rank=16,
            alpha=32.0,
            dropout=0.0,
            target_modules=(V88_TARGET_MODULE,),
        ),
        initialization_algorithm="checkpoint_overwrite",
        initialization_seed=None,
        expected_initial_state_sha256=str(frozen["v88_bank_state_sha256"]),
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
    return LoRABanksSettings(base.banks + (v86, v87, v88, fresh))


def _copy_two_tensor_state_v89(
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


def load_frozen_stack_v89(
    collection: LoRABankCollection,
    *,
    v85_checkpoint: str | Path,
    v86_checkpoint: str | Path,
    v87_checkpoint: str | Path,
    v88_checkpoint: str | Path,
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and load all ten frozen parent banks."""

    v85_root = resolve_v85(v85_checkpoint)
    v85_weights = v85_root / "adapter.safetensors"
    v85_metadata_path = v85_root / "runtime_metadata.json"
    v85_metadata = strict_json_v89(v85_metadata_path)
    hashes = v85_metadata.get("lora_bank_state_sha256")
    modules = v85_metadata.get("lora_bank_wrapped_modules")
    if not isinstance(hashes, Mapping) or not isinstance(modules, Mapping) or len(hashes) != 7:
        raise TypeError("V89 V85 source lacks exact seven-bank metadata")
    v85_archive = load_file(str(v85_weights), device="cpu")
    for name, expected_state in hashes.items():
        bank = collection.bank(str(name))
        if bank.settings.trainable:
            raise ValueError(f"V89 V85 bank unexpectedly trainable: {name}")
        prefix = f"lora_banks.{name}."
        state = {
            key[len(prefix) :]: value
            for key, value in v85_archive.items()
            if key.startswith(prefix)
        }
        bank.installation.state_module.load_state_dict(state, strict=True)
        if list(bank.installation.target_names) != modules[name]:
            raise ValueError(f"V89 V85 bank targets changed: {name}")
        if bank.installation.state_sha256() != expected_state:
            raise ValueError(f"V89 V85 bank state changed: {name}")

    parent_results: dict[str, dict[str, str]] = {}
    for bank_name, target, root_value, expected_state in (
        (
            V86_BANK_NAME,
            V86_TARGET_MODULE,
            v86_checkpoint,
            experiment["frozen_stack"]["v86_bank_state_sha256"],
        ),
        (
            V87_BANK_NAME,
            V87_TARGET_MODULE,
            v87_checkpoint,
            experiment["frozen_stack"]["v87_bank_state_sha256"],
        ),
        (
            V88_BANK_NAME,
            V88_TARGET_MODULE,
            v88_checkpoint,
            experiment["frozen_stack"]["v88_bank_state_sha256"],
        ),
    ):
        root = resolve_v85(root_value)
        weights = root / WEIGHTS_FILENAME
        metadata_path = root / METADATA_FILENAME
        metadata = strict_json_v89(metadata_path)
        bank = collection.bank(bank_name)
        if (
            bank.settings.trainable
            or bank.installation.target_names != (target,)
            or metadata.get("state_sha256") != expected_state
            or metadata.get("weights_sha256") != sha256_file_v85(weights)
            or metadata.get("questions_or_answers_serialized") is not False
            or metadata.get("environmental_memory_serialized") is not False
        ):
            raise ValueError(f"V89 frozen parent bridge metadata changed: {bank_name}")
        _copy_two_tensor_state_v89(
            bank.installation,
            load_file(str(weights), device="cpu"),
            context=f"V89 frozen {bank_name}",
        )
        if bank.installation.state_sha256() != expected_state:
            raise ValueError(f"V89 frozen parent bridge state changed: {bank_name}")
        parent_results[bank_name] = {
            "weights_sha256": sha256_file_v85(weights),
            "metadata_sha256": sha256_file_v85(metadata_path),
            "state_sha256": bank.installation.state_sha256(),
        }

    fresh = collection.bank(FRESH_BANK_NAME)
    if (
        not fresh.settings.trainable
        or fresh.installation.target_names != (TARGET_MODULE,)
        or fresh.installation.parameter_count
        != experiment["bridge"]["trainable_parameter_count"]
        or fresh.installation.state_sha256()
        != experiment["bridge"]["expected_initial_state_sha256"]
        or any(
            torch.count_nonzero(adapter.lora_b).item() != 0
            for adapter in fresh.installation.adapters
        )
    ):
        raise ValueError("V89 fresh bridge did not start at exact zero output")
    if len([bank for bank in collection.banks if not bank.settings.trainable]) != 10:
        raise ValueError("V89 exact frozen bank count changed")
    collection.validate_state()
    return {
        "v85_adapter_sha256": sha256_file_v85(v85_weights),
        "v85_metadata_sha256": sha256_file_v85(v85_metadata_path),
        "v85_bank_state_sha256": dict(hashes),
        "parents": parent_results,
        "frozen_bank_count": 10,
        "fresh_initial_state_sha256": fresh.installation.state_sha256(),
    }


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _fresh_state_v89(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    installation = collection.bank(FRESH_BANK_NAME).installation
    if len(installation.adapters) != 1:
        raise ValueError("V89 fresh bank must wrap exactly one module")
    adapter = installation.adapters[0]
    return {
        "lora_a": adapter.lora_a.detach().cpu().contiguous(),
        "lora_b": adapter.lora_b.detach().cpu().contiguous(),
    }


def _load_fresh_state_v89(
    collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]
) -> None:
    _copy_two_tensor_state_v89(
        collection.bank(FRESH_BANK_NAME).installation,
        archive,
        context="V89 fresh bridge",
    )
    collection.validate_state()


def publish_fixed_final_candidate_v89(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, str | int | bool],
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    root = resolve_v85(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V89 fixed-final candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / WEIGHTS_FILENAME
        save_file(
            _fresh_state_v89(collection),
            str(weights),
            metadata={
                "artifact": CANDIDATE_ARTIFACT,
                "environmental_memory_serialized": "false",
                "questions_or_answers_serialized": "false",
                "training_metadata_serialized": "false",
                "error_inventory_serialized": "false",
                "anchor_inventory_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        fresh = collection.bank(FRESH_BANK_NAME).installation
        bridge = experiment["bridge"]
        metadata = {
            "artifact": CANDIDATE_ARTIFACT,
            "schema_version": 89,
            "status": "fixed_final_awaiting_preregistered_acceptance_gates",
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "rank": int(bridge["rank"]),
            "alpha": float(bridge["alpha"]),
            "dropout": float(bridge["dropout"]),
            "parameter_count": fresh.parameter_count,
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v85(weights),
            "frozen_bank_count": 10,
            "total_bank_count": 11,
            "environmental_memory_serialized": False,
            "questions_or_answers_serialized": False,
            "training_metadata_serialized": False,
            "error_inventory_serialized": False,
            "anchor_inventory_serialized": False,
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


def load_fixed_final_bridge_v89(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    root = resolve_v85(candidate)
    metadata = strict_json_v89(root / METADATA_FILENAME)
    weights = root / WEIGHTS_FILENAME
    if (
        metadata.get("artifact") != CANDIDATE_ARTIFACT
        or metadata.get("status") != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("target_module") != TARGET_MODULE
        or metadata.get("weights_sha256") != sha256_file_v85(weights)
        or metadata.get("schema_version") != 89
        or metadata.get("frozen_bank_count") != 10
        or metadata.get("total_bank_count") != 11
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("training_metadata_serialized") is not False
        or metadata.get("error_inventory_serialized") is not False
        or metadata.get("anchor_inventory_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V89 fixed-final candidate authentication failed")
    _load_fresh_state_v89(collection, load_file(str(weights), device="cpu"))
    if collection.bank(FRESH_BANK_NAME).installation.state_sha256() != metadata["state_sha256"]:
        raise ValueError("V89 fixed-final bridge state changed")
    return metadata


def authenticate_training_report_v89(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    preflight = authenticate_cpu_preflight_v89(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["training_report"])
    report = strict_json_v89(path)
    gates = report.get("gates")
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("schema_version") != 89
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != 155
        or report.get("micro_rows_consumed") != 930
        or report.get("causal_margin_rows_consumed") != 18
        or report.get("protected_read_count") != 0
        or report.get("oracle_loaded") is not False
        or report.get("held_out_generalization_claim") is not False
        or report.get("runtime_promotion_authorized") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("V89 fixed-final training report changed")
    return {**preflight, "training_report_sha256": sha256_file_v85(path)}


def run_training_v89(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v89(config_path)
    source_hashes = authenticate_sources_v89(config)
    preflight = authenticate_cpu_preflight_v89(config, config_path=config_path)
    rows = load_canonical_rows_v89(config)
    _transitions, _errors, _anchors, error_rows, anchor_rows = (
        derive_parent_behavior_v89(config, rows)
    )
    items = derive_training_items_v89(config, rows, error_rows, anchor_rows)
    inventory_hash = canonical_sha256_v85(training_inventory_v89(items))
    schedule = training_schedule_v89(
        items,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["training"]["epochs"]),
    )
    schedule_hash = canonical_sha256_v85(
        [[epoch, item.schedule_id] for epoch, item in schedule]
    )
    if (
        inventory_hash != config["dataset"]["training_row_inventory_sha256"]
        or schedule_hash != config["training"]["row_order_sha256"]
        or len(schedule) != 930
    ):
        raise RuntimeError("V89 fixed training inventory or schedule changed")
    report_path = resolve_v85(config["outputs"]["training_report"])
    candidate_path = resolve_v85(config["outputs"]["fixed_final_candidate"])
    if report_path.exists() or candidate_path.exists():
        raise FileExistsError("V89 fixed-final outputs already exist")

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
            raise RuntimeError("V89 full-model training requires local MPS")
        collection = install_lora_banks(
            language.model, combined_lora_settings_v89(runtime, config)
        )
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V89 LoRA bank installation failed")
        frozen_source = load_frozen_stack_v89(
            collection,
            v85_checkpoint=config["sources"]["frozen_v85_checkpoint"],
            v86_checkpoint=config["sources"]["parent_v86_checkpoint"],
            v87_checkpoint=config["sources"]["parent_v87_checkpoint"],
            v88_checkpoint=config["sources"]["parent_v88_checkpoint"],
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
        kind_seen: Counter[str] = Counter()
        for cursor, (epoch, item) in enumerate(schedule, 1):
            row = item.row
            kind_seen[item.kind] += 1
            prepared, _layout = _prepared_v84(language, system_prompt, memory, row)
            tail = _answer_tail(language, prepared)
            correct_nll = tail.mean_nll.float()
            weighted_ce = ce_weight * correct_nll
            if item.causal_margin:
                zero_prepared, _zero_layout = _prepared_v84(
                    language, system_prompt, zero_memory, row
                )
                zero_tail = _answer_tail(language, zero_prepared)
                _unused, observed_margin, penalty = zero_payload_margin_objective_v86(
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
                        "objective": float(objective.detach().cpu()),
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
                        "objective": float(objective.detach().cpu()),
                        "causal": False,
                        "zero_minus_correct_nll": 0.0,
                        "margin_penalty": 0.0,
                    }
                )
            if not torch.isfinite(objective):
                raise RuntimeError("V89 objective is nonfinite")
            (objective / accumulation).backward()
            del prepared, tail, correct_nll, weighted_ce, objective
            if cursor % accumulation:
                continue
            gradient_l2 = float(collection.gradient_norms()["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V89 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            clip_l2 = float(clipped.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V89 clipped gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            causal_interval = [record for record in interval if record["causal"]]
            record = {
                "update": optimizer_update,
                "row_cursor": cursor,
                "epoch": epoch,
                "mean_correct_nll": sum(
                    float(value["correct_nll"]) for value in interval
                )
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
            if optimizer_update in {1, 78, 155} or optimizer_update % 12 == 0:
                print(
                    json.dumps(
                        {
                            "event": "v89_train_update",
                            "update": optimizer_update,
                            "total_updates": 155,
                            "row_cursor": cursor,
                            "epoch": epoch,
                            "mean_correct_nll": record["mean_correct_nll"],
                            "causal_rows_seen": causal_seen,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            torch.mps.empty_cache()

        if (
            optimizer_update != 155
            or len(schedule) != 930
            or causal_seen != 18
            or len(history) != 155
            or kind_seen
            != Counter(
                {
                    "canonical": 414,
                    "error_replay": 186,
                    "correct_anchor_replay": 321,
                    "development_known_smoke": 9,
                }
            )
        ):
            raise RuntimeError("V89 fixed retention schedule did not complete")
        language.decoder_module.eval()
        collection.eval()
        memory_hash_after = prefix_sha256(memory.detach().cpu())
        zero_hash_after = prefix_sha256(zero_memory.detach().cpu())
        if memory_hash_after != memory_hash_before or zero_hash_after != zero_hash_before:
            raise RuntimeError("V89 training mutated immutable environmental inputs")
        # Candidate bindings are deliberately scalar/hash-only.  In particular
        # they contain no path, question ID, answer, error, or anchor record.
        bindings: dict[str, str | int | bool] = {
            "config_sha256": preflight["config_sha256"],
            "preregistration_sha256": preflight["preregistration_sha256"],
            "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
            "v85_adapter_sha256": frozen_source["v85_adapter_sha256"],
            "v86_bridge_sha256": frozen_source["parents"][V86_BANK_NAME]["weights_sha256"],
            "v87_bridge_sha256": frozen_source["parents"][V87_BANK_NAME]["weights_sha256"],
            "v88_bridge_sha256": frozen_source["parents"][V88_BANK_NAME]["weights_sha256"],
            "fixed_final_optimizer_updates": optimizer_update,
            "row_order_sha256": schedule_hash,
            "training_inventory_sha256": inventory_hash,
            "scene_memory_prefix_sha256": memory_hash_before,
            "development_known_smoke_trained": True,
        }
        candidate_metadata = publish_fixed_final_candidate_v89(
            candidate_path,
            collection,
            bindings=bindings,
            experiment=config,
        )
    audit.assert_clean()

    gates = {
        "all_930_sealed_micro_rows_consumed": len(schedule) == 930,
        "all_138_canonical_rows_consumed_once_each_epoch": kind_seen["canonical"] == 414,
        "all_31_v88_errors_replayed_twice_each_epoch": kind_seen["error_replay"]
        == 186,
        "all_107_v88_correct_anchors_replayed_once_each_epoch": kind_seen[
            "correct_anchor_replay"
        ]
        == 321,
        "all_3_development_smoke_rows_consumed_once_each_epoch": kind_seen[
            "development_known_smoke"
        ]
        == 9,
        "all_18_causal_margin_rows_consumed": causal_seen == 18,
        "fixed_final_update_155_reached": optimizer_update == 155,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(row["gradient_l2_before_clip"]))
            and float(row["gradient_l2_before_clip"]) > 0.0
            for row in history
        ),
        "memory_hash_invariant": memory_hash_after == memory_hash_before,
        "zero_payload_hash_invariant": zero_hash_after == zero_hash_before,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
        "runtime_candidate_contains_no_training_rows_or_answers": (
            candidate_metadata["questions_or_answers_serialized"] is False
            and candidate_metadata["training_metadata_serialized"] is False
            and candidate_metadata["error_inventory_serialized"] is False
            and candidate_metadata["anchor_inventory_serialized"] is False
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"V89 training gate failed: {gates}")
    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 89,
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
            "parameter_count": config["bridge"]["trainable_parameter_count"],
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": candidate_metadata["state_sha256"],
            "unmerged": True,
        },
        "training_protocol": config["training"],
        "training_inventory": {
            "canonical_unique_rows": 138,
            "parent_v88_errors": 31,
            "parent_v88_correct_anchors": 107,
            "error_replay_copies_per_epoch": 2,
            "unique_schedule_items_per_epoch": 310,
            "inventory_sha256": inventory_hash,
            "row_order_sha256": schedule_hash,
            "development_known_smoke_trained": True,
            "held_out_smoke_claim": False,
            "answers_or_questions_serialized_in_candidate": False,
            "inventory_serialized_in_candidate": False,
        },
        "micro_rows_consumed": len(schedule),
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
    report = run_training_v89(args.config)
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
    "authenticate_training_report_v89",
    "combined_lora_settings_v89",
    "load_fixed_final_bridge_v89",
    "load_frozen_stack_v89",
    "main",
    "publish_fixed_final_candidate_v89",
    "run_training_v89",
]
