"""Train V86's fixed-final single-scene direct-memory bridge.

All 138 scene-000001 questions are consumed in each of four fixed epochs.  The
base Gemma model and the seven already-published V54+V85 LoRA banks remain
frozen.  The only trainable state is one fresh rank-8 bank on the final
full-attention MLP up projection.  Three preregistered rows additionally use a
native-BOI/EOI-preserving zero-payload causal margin.
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
    CONFIG,
    FRESH_BANK_NAME,
    TARGET_MODULE,
    authenticate_cpu_preflight_v86,
    authenticate_sources_v86,
    causal_rows_v86,
    load_config_v86,
    load_scene1_memory_v86,
    load_scene1_rows_v86,
    training_schedule_v86,
    zero_payload_memory_v86,
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

TRAINING_ARTIFACT: Final[str] = "gemma4_v86_scene1_demo_training_v1"
CANDIDATE_ARTIFACT: Final[str] = "gemma4_v86_scene1_demo_fixed_final_v1"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)


def strict_json_v86(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V86 JSON must contain one object: {source}")
    return value


def combined_lora_settings_v86(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    base = lora_banks_settings(runtime_config)
    if len(base.banks) != 7 or any(bank.trainable for bank in base.banks):
        raise ValueError("V86 requires exactly seven frozen V54+V85 banks")
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


def load_frozen_v85_stack_v86(
    collection: LoRABankCollection, checkpoint: str | Path
) -> dict[str, Any]:
    """Load all seven inherited banks from the strict V85 runtime package."""

    root = resolve_v85(checkpoint)
    weights = root / "adapter.safetensors"
    metadata_path = root / "runtime_metadata.json"
    metadata = strict_json_v86(metadata_path)
    hashes = metadata.get("lora_bank_state_sha256")
    modules = metadata.get("lora_bank_wrapped_modules")
    if not isinstance(hashes, Mapping) or not isinstance(modules, Mapping):
        raise TypeError("V86 V85 source lacks named LoRA metadata")
    frozen = [bank for bank in collection.banks if not bank.settings.trainable]
    fresh = [bank for bank in collection.banks if bank.settings.trainable]
    if (
        len(frozen) != 7
        or len(fresh) != 1
        or fresh[0].settings.name != FRESH_BANK_NAME
        or set(hashes) != {bank.settings.name for bank in frozen}
    ):
        raise ValueError("V86 requires seven frozen banks plus one fresh bank")
    archive = load_file(str(weights), device="cpu")
    for bank in frozen:
        name = bank.settings.name
        prefix = f"lora_banks.{name}."
        state = {
            key[len(prefix) :]: value for key, value in archive.items() if key.startswith(prefix)
        }
        bank.installation.state_module.load_state_dict(state, strict=True)
        if list(bank.installation.target_names) != modules[name]:
            raise ValueError(f"V86 frozen bank targets changed: {name}")
        if bank.installation.state_sha256() != hashes[name]:
            raise ValueError(f"V86 frozen bank state changed: {name}")
    installation = fresh[0].installation
    if (
        installation.target_names != (TARGET_MODULE,)
        or installation.parameter_count != 110592
        or installation.state_sha256() != fresh[0].settings.expected_initial_state_sha256
        or any(torch.count_nonzero(adapter.lora_b).item() != 0 for adapter in installation.adapters)
    ):
        raise ValueError("V86 fresh bridge did not start at deterministic zero output")
    collection.validate_state()
    return {
        "adapter_sha256": sha256_file_v85(weights),
        "runtime_metadata_sha256": sha256_file_v85(metadata_path),
        "frozen_bank_count": len(frozen),
        "frozen_bank_state_sha256": dict(hashes),
        "fresh_initial_state_sha256": installation.state_sha256(),
    }


def zero_payload_margin_objective_v86(
    correct_nll: torch.Tensor,
    zero_payload_nll: torch.Tensor,
    *,
    target_margin: float,
    ce_weight: float,
    margin_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if correct_nll.ndim != 0 or zero_payload_nll.ndim != 0:
        raise ValueError("V86 NLL inputs must be scalar")
    if not torch.isfinite(correct_nll) or not torch.isfinite(zero_payload_nll):
        raise ValueError("V86 NLL inputs must be finite")
    observed_margin = zero_payload_nll - correct_nll
    penalty = torch.relu(
        correct_nll - zero_payload_nll + torch.as_tensor(target_margin, device=correct_nll.device)
    )
    return (
        ce_weight * correct_nll + margin_weight * penalty,
        observed_margin,
        penalty,
    )


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _fresh_state(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    installation = collection.bank(FRESH_BANK_NAME).installation
    if len(installation.adapters) != 1:
        raise ValueError("V86 fresh bank must wrap one module")
    adapter = installation.adapters[0]
    return {
        "lora_a": adapter.lora_a.detach().cpu().contiguous(),
        "lora_b": adapter.lora_b.detach().cpu().contiguous(),
    }


def _load_fresh_state(collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]) -> None:
    if set(archive) != {"lora_a", "lora_b"}:
        raise ValueError("V86 bridge tensor inventory changed")
    adapter = collection.bank(FRESH_BANK_NAME).installation.adapters[0]
    with torch.no_grad():
        for name, parameter in (("lora_a", adapter.lora_a), ("lora_b", adapter.lora_b)):
            value = archive[name]
            if value.shape != parameter.shape or value.dtype != torch.float32:
                raise ValueError(f"V86 {name} shape or dtype changed")
            parameter.copy_(value.to(parameter.device))
    collection.validate_state()


def publish_fixed_final_candidate_v86(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    root = resolve_v85(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V86 fixed-final candidate exists: {root}")
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
            "schema_version": 86,
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


def load_fixed_final_bridge_v86(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    root = resolve_v85(candidate)
    metadata = strict_json_v86(root / METADATA_FILENAME)
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
        raise ValueError("V86 fixed-final candidate authentication failed")
    _load_fresh_state(collection, load_file(str(weights), device="cpu"))
    if collection.bank(FRESH_BANK_NAME).installation.state_sha256() != metadata["state_sha256"]:
        raise ValueError("V86 fixed-final bridge state changed")
    return metadata


def authenticate_training_report_v86(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    preflight = authenticate_cpu_preflight_v86(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["training_report"])
    report = strict_json_v86(path)
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != 92
        or report.get("micro_rows_consumed") != 552
        or report.get("causal_margin_rows_consumed") != 12
        or report.get("protected_read_count") != 0
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V86 fixed-final training report changed")
    return {**preflight, "training_report_sha256": sha256_file_v85(path)}


def run_training_v86(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v86(config_path)
    source_hashes = authenticate_sources_v86(config)
    preflight = authenticate_cpu_preflight_v86(config, config_path=config_path)
    rows = load_scene1_rows_v86(config)
    causal_ids = {row.question_id for row in causal_rows_v86(config, rows)}
    schedule = training_schedule_v86(
        rows,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["training"]["epochs"]),
    )
    schedule_hash = canonical_sha256_v85([[epoch, row.question_id] for epoch, row in schedule])
    if schedule_hash != config["training"]["row_order_sha256"]:
        raise RuntimeError("V86 fixed schedule changed")
    outputs = config["outputs"]
    report_path = resolve_v85(outputs["training_report"])
    candidate_path = resolve_v85(outputs["fixed_final_candidate"])
    if report_path.exists() or candidate_path.exists():
        raise FileExistsError("V86 fixed-final outputs already exist")

    # The entire environmental input is loaded, hashed, and zero-controlled
    # before Gemma or any user/training question is tokenized.
    cpu_memory, memory_hash_before, _memory_metadata = load_scene1_memory_v86(config)
    cpu_zero_memory = zero_payload_memory_v86(cpu_memory)
    zero_memory_hash_before = prefix_sha256(cpu_zero_memory)

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
            raise RuntimeError("V86 full-model training requires local MPS")
        collection = install_lora_banks(language.model, combined_lora_settings_v86(runtime, config))
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V86 LoRA bank installation failed")
        frozen_source = load_frozen_v85_stack_v86(
            collection, config["sources"]["frozen_checkpoint"]
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
        for cursor, (_epoch, row) in enumerate(schedule, 1):
            prepared, _layout = _prepared_v84(language, system_prompt, memory, row)
            tail = _answer_tail(language, prepared)
            correct_nll = tail.mean_nll.float()
            if row.question_id in causal_ids:
                zero_prepared, _zero_layout = _prepared_v84(
                    language, system_prompt, zero_memory, row
                )
                zero_tail = _answer_tail(language, zero_prepared)
                objective, observed_margin, penalty = zero_payload_margin_objective_v86(
                    correct_nll,
                    zero_tail.mean_nll.float(),
                    target_margin=target_margin,
                    ce_weight=ce_weight,
                    margin_weight=margin_weight,
                )
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
                objective = ce_weight * correct_nll
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
                raise RuntimeError("V86 objective is nonfinite")
            (objective / accumulation).backward()
            del prepared, tail, correct_nll, objective
            if cursor % accumulation:
                continue
            gradients = collection.gradient_norms()
            gradient_l2 = float(gradients["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V86 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            clip_l2 = float(clipped.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V86 clipped gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            causal_interval = [record for record in interval if record["causal"]]
            record = {
                "update": optimizer_update,
                "row_cursor": cursor,
                "mean_correct_nll": sum(float(value["correct_nll"]) for value in interval)
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
            if optimizer_update in {1, 46, 92} or optimizer_update % 8 == 0:
                print(
                    json.dumps(
                        {
                            "event": "v86_train_update",
                            "update": optimizer_update,
                            "total_updates": 92,
                            "row_cursor": cursor,
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
            optimizer_update != 92
            or len(schedule) != 552
            or causal_seen != 12
            or len(history) != 92
        ):
            raise RuntimeError("V86 fixed schedule did not complete exactly")
        language.decoder_module.eval()
        collection.eval()
        memory_hash_after = prefix_sha256(memory.detach().cpu())
        zero_memory_hash_after = prefix_sha256(zero_memory.detach().cpu())
        if (
            memory_hash_after != memory_hash_before
            or zero_memory_hash_after != zero_memory_hash_before
        ):
            raise RuntimeError("V86 training mutated immutable environmental inputs")
        bindings = {
            **preflight,
            "training_source_sha256": source_hashes[str(config["sources"]["training_source"])],
            "frozen_adapter_sha256": frozen_source["adapter_sha256"],
            "fixed_final_optimizer_updates": optimizer_update,
            "row_order_sha256": schedule_hash,
            "scene_memory_prefix_sha256": memory_hash_before,
        }
        candidate_metadata = publish_fixed_final_candidate_v86(
            candidate_path, collection, bindings=bindings
        )
    audit.assert_clean()

    gates = {
        "all_138_rows_consumed_in_each_of_four_epochs": len(schedule) == 552,
        "all_12_causal_margin_rows_consumed": causal_seen == 12,
        "fixed_final_update_92_reached": optimizer_update == 92,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(row["gradient_l2_before_clip"]))
            and float(row["gradient_l2_before_clip"]) > 0.0
            for row in history
        ),
        "memory_hash_invariant": memory_hash_after == memory_hash_before,
        "zero_payload_hash_invariant": zero_memory_hash_after == zero_memory_hash_before,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"V86 training gate failed: {gates}")
    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 86,
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
            "zero_payload_prefix_sha256_before": zero_memory_hash_before,
            "zero_payload_prefix_sha256_after": zero_memory_hash_after,
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
    report = run_training_v86(args.config)
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
    "authenticate_training_report_v86",
    "combined_lora_settings_v86",
    "load_fixed_final_bridge_v86",
    "load_frozen_v85_stack_v86",
    "main",
    "publish_fixed_final_candidate_v86",
    "run_training_v86",
    "strict_json_v86",
    "zero_payload_margin_objective_v86",
]
