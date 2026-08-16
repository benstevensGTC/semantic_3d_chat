"""Train V90's fixed-memory single-scene conversational continuation.

The promoted V89 adapter is the immutable eleven-bank parent.  V90 installs
those exact banks frozen and optimizes only a fresh rank-8 LoRA bank on the
disjoint layer-28 attention output projection.  Every row receives the same
pre-question 738-token continuous scene memory; primary conversational rows
also receive a causal zero-environment margin objective.  The create-once
candidate contains only the fresh two-tensor bridge and hash/numeric metadata.
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
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
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
from semantic_3d_chat.evaluation.v89_scene1_retention_preflight import strict_json_v89
from semantic_3d_chat.evaluation.v90_scene1_conversational_preflight import (
    CONFIG,
    FRESH_BANK_NAME,
    TARGET_MODULE,
    authenticate_cpu_preflight_v90,
    authenticate_sources_v90,
    derive_training_items_v90,
    inventory_v90,
    load_canonical_rows_v90,
    load_config_v90,
    schedule_v90,
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
from semantic_3d_chat.training.train_v86_scene1_demo import (
    zero_payload_margin_objective_v86,
)

TRAINING_ARTIFACT: Final[str] = "gemma4_v90_scene1_conversational_training_v1"
CANDIDATE_ARTIFACT: Final[str] = "gemma4_v90_scene1_conversational_fixed_final_v1"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
PARENT_BANK_COUNT: Final[int] = 11
TOTAL_BANK_COUNT: Final[int] = 12
PARENT_PARAMETER_COUNT: Final[int] = 872_448
FRESH_PARAMETER_COUNT: Final[int] = 28_672
EXPECTED_MICRO_ROWS: Final[int] = 1_032
EXPECTED_OPTIMIZER_UPDATES: Final[int] = 172
EXPECTED_CAUSAL_ROWS: Final[int] = 39
EXPECTED_FRESH_TARGET: Final[str] = "model.language_model.layers.28.self_attn.o_proj"
EXPECTED_PARENT_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    "v85_strict_multiscene_bridge",
    "v86_scene1_demo_bridge",
    "v87_scene1_balanced_bridge",
    "v88_scene1_augmented_bridge",
    "v89_scene1_retention_bridge",
)
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)


def combined_lora_settings_v90(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    """Return the exact frozen V89 stack plus V90's sole trainable bank."""

    parent = lora_banks_settings(runtime_config)
    if (
        parent.legacy_single_bank
        or tuple(bank.name for bank in parent.banks) != EXPECTED_PARENT_BANKS
        or any(bank.trainable for bank in parent.banks)
        or TARGET_MODULE != EXPECTED_FRESH_TARGET
    ):
        raise ValueError("V90 requires the exact promoted frozen V89 bank stack")
    bridge = experiment["bridge"]
    if (
        str(bridge["bank_name"]) != FRESH_BANK_NAME
        or str(bridge["target_module"]) != EXPECTED_FRESH_TARGET
        or int(bridge["rank"]) != 8
        or float(bridge["alpha"]) != 16.0
        or float(bridge["dropout"]) != 0.0
        or int(bridge["trainable_parameter_count"]) != FRESH_PARAMETER_COUNT
    ):
        raise ValueError("V90 fresh bridge topology changed")
    fresh = LoRABankSettings(
        name=FRESH_BANK_NAME,
        trainable=True,
        adapter=LoRASettings(
            enabled=True,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_modules=(EXPECTED_FRESH_TARGET,),
        ),
        initialization_algorithm=str(bridge["initialization_algorithm"]),
        initialization_seed=int(bridge["initialization_seed"]),
        expected_initial_state_sha256=str(bridge["expected_initial_state_sha256"]),
    )
    return LoRABanksSettings(parent.banks + (fresh,))


def _copy_bank_state_v90(
    installation: Any, archive: Mapping[str, torch.Tensor], *, context: str
) -> None:
    expected_keys = set(installation.state_module.state_dict())
    if set(archive) != expected_keys:
        raise ValueError(f"{context} tensor inventory changed")
    if any(value.dtype != torch.float32 for value in archive.values()):
        raise TypeError(f"{context} tensors must be float32")
    installation.state_module.load_state_dict(dict(archive), strict=True)
    installation.validate_state()


def _fresh_state_v90(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if len(fresh.adapters) != 1:
        raise ValueError("V90 fresh bank must wrap exactly one module")
    adapter = fresh.adapters[0]
    return {
        "lora_a": adapter.lora_a.detach().cpu().contiguous(),
        "lora_b": adapter.lora_b.detach().cpu().contiguous(),
    }


def _load_fresh_state_v90(
    collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]
) -> None:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if set(archive) != {"lora_a", "lora_b"} or len(fresh.adapters) != 1:
        raise ValueError("V90 fresh bridge tensor inventory changed")
    adapter = fresh.adapters[0]
    with torch.no_grad():
        for name, parameter in (("lora_a", adapter.lora_a), ("lora_b", adapter.lora_b)):
            value = archive[name]
            if value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError(f"V90 fresh bridge {name} shape or dtype changed")
            parameter.copy_(value.to(parameter.device))
    collection.validate_state()


def load_frozen_parent_v90(
    collection: LoRABankCollection,
    parent_checkpoint: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and load all eleven promoted V89 banks byte-exactly."""

    root = resolve_v85(parent_checkpoint)
    weights = root / "adapter.safetensors"
    metadata_path = root / METADATA_FILENAME
    sources = config["sources"]
    if (
        sha256_file_v85(weights) != sources["parent_adapter_sha256"]
        or sha256_file_v85(metadata_path) != sources["parent_metadata_sha256"]
    ):
        raise ValueError("V90 promoted V89 parent bytes changed")
    metadata = strict_json_v89(metadata_path)
    fingerprint, files = checkpoint_fingerprint(root)
    release = strict_json_v89(sources["parent_release"])
    release_checkpoint = release.get("checkpoint")
    lora = metadata.get("lora")
    metadata_banks = lora.get("banks") if isinstance(lora, Mapping) else None
    states = metadata.get("lora_bank_state_sha256")
    modules = metadata.get("lora_bank_wrapped_modules")
    counts = metadata.get("lora_bank_parameter_counts")
    provenance_root = metadata.get("initialization_provenance")
    provenance = (
        provenance_root.get("v89_strict_runtime_release")
        if isinstance(provenance_root, Mapping)
        else None
    )
    frozen = tuple(bank for bank in collection.banks if not bank.settings.trainable)
    trainable = tuple(bank for bank in collection.banks if bank.settings.trainable)
    if (
        not isinstance(release_checkpoint, Mapping)
        or release.get("promotion_decision") != "strict_scene1_experimental_primary"
        or release.get("all_release_gates_passed") is not True
        or release_checkpoint.get("checkpoint_sha256") != fingerprint
        or release_checkpoint.get("adapter_sha256") != sha256_file_v85(weights)
        or not isinstance(metadata_banks, list)
        or tuple(str(row.get("name")) for row in metadata_banks) != EXPECTED_PARENT_BANKS
        or not isinstance(states, Mapping)
        or set(states) != set(EXPECTED_PARENT_BANKS)
        or not isinstance(modules, Mapping)
        or set(modules) != set(EXPECTED_PARENT_BANKS)
        or not isinstance(counts, Mapping)
        or set(counts) != set(EXPECTED_PARENT_BANKS)
        or lora.get("adapter_parameter_count") != PARENT_PARAMETER_COUNT
        or lora.get("trainable_adapter_parameter_count") != 0
        or metadata.get("lora_parameter_count") != PARENT_PARAMETER_COUNT
        or metadata.get("lora_trainable_parameter_count") != 0
        or not isinstance(provenance, Mapping)
        or provenance.get("schema_version") != 89
        or provenance.get("promotion_decision") != "strict_scene1_experimental_primary"
        or provenance.get("runtime_promotion_authorized") is not True
        or provenance.get("model_acceptance_gate_passed") is not True
        or provenance.get("model_gate_report_authenticated") is not True
        or provenance.get("development_known_smoke_trained") is not True
        or provenance.get("held_out_smoke_claim") is not False
        or provenance.get("held_out_generalization_claim") is not False
        or provenance.get("evaluation_predictions_sha256") != sources["parent_predictions_sha256"]
        or provenance.get("model_gate_report_sha256") != sources["parent_evaluation_sha256"]
        or provenance.get("v86_bridge_state_sha256") != states.get("v86_scene1_demo_bridge")
        or provenance.get("v87_bridge_state_sha256") != states.get("v87_scene1_balanced_bridge")
        or provenance.get("v88_bridge_state_sha256") != states.get("v88_scene1_augmented_bridge")
        or provenance.get("v89_bridge_state_sha256") != states.get("v89_scene1_retention_bridge")
        or len(frozen) != PARENT_BANK_COUNT
        or tuple(bank.settings.name for bank in frozen) != EXPECTED_PARENT_BANKS
        or len(trainable) != 1
        or trainable[0].settings.name != FRESH_BANK_NAME
    ):
        raise ValueError("V90 promoted V89 parent contract changed")

    archive = load_file(str(weights), device="cpu")
    observed_bank_states: dict[str, str] = {}
    for bank in frozen:
        name = bank.settings.name
        prefix = f"lora_banks.{name}."
        state = {
            key[len(prefix) :]: value for key, value in archive.items() if key.startswith(prefix)
        }
        if (
            bank.settings.trainable
            or list(bank.installation.target_names) != modules[name]
            or sum(int(value) for value in counts[name].values())
            != bank.installation.parameter_count
            or tensor_state_sha256(state) != states[name]
        ):
            raise ValueError(f"V90 frozen parent bank changed: {name}")
        _copy_bank_state_v90(
            bank.installation,
            state,
            context=f"V90 frozen parent bank {name}",
        )
        if bank.installation.state_sha256() != states[name]:
            raise ValueError(f"V90 frozen parent state changed after load: {name}")
        observed_bank_states[name] = bank.installation.state_sha256()

    fresh = trainable[0].installation
    bridge = config["bridge"]
    if (
        fresh.target_names != (EXPECTED_FRESH_TARGET,)
        or fresh.parameter_count != FRESH_PARAMETER_COUNT
        or fresh.state_sha256() != bridge["expected_initial_state_sha256"]
        or any(int(torch.count_nonzero(adapter.lora_b).item()) != 0 for adapter in fresh.adapters)
    ):
        raise ValueError("V90 fresh bridge did not start at exact zero output")
    collection.validate_state()
    return {
        "parent_checkpoint_sha256": fingerprint,
        "parent_checkpoint_files": files,
        "parent_adapter_sha256": sha256_file_v85(weights),
        "parent_runtime_metadata_sha256": sha256_file_v85(metadata_path),
        "parent_bank_state_sha256": observed_bank_states,
        "parent_bank_count": PARENT_BANK_COUNT,
        "parent_parameter_count": PARENT_PARAMETER_COUNT,
        "parent_release_provenance_sha256": canonical_sha256_v85(provenance),
        "fresh_initial_state_sha256": fresh.state_sha256(),
        "parent_tensors_loaded_byte_exactly": True,
    }


def _candidate_fingerprint_v90(root: Path) -> tuple[str, list[dict[str, Any]]]:
    files = [root / WEIGHTS_FILENAME, root / METADATA_FILENAME]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("V90 candidate is not an exact two-file artifact")
    if sorted(path.name for path in root.iterdir()) != sorted(
        (WEIGHTS_FILENAME, METADATA_FILENAME)
    ):
        raise ValueError("V90 candidate contains unexpected files")
    entries = [
        {
            "path": path.name,
            "sha256": sha256_file_v85(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    return canonical_sha256_v85(entries), entries


def publish_fixed_final_candidate_v90(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, str | int | bool],
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish only V90's fresh bridge and sanitized metadata."""

    root = resolve_v85(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V90 fixed-final candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / WEIGHTS_FILENAME
        save_file(
            _fresh_state_v90(collection),
            str(weights),
            metadata={
                "artifact": CANDIDATE_ARTIFACT,
                "environmental_memory_serialized": "false",
                "environmental_text_serialized": "false",
                "questions_or_answers_serialized": "false",
                "training_metadata_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        fresh = collection.bank(FRESH_BANK_NAME).installation
        bridge = experiment["bridge"]
        metadata = {
            "artifact": CANDIDATE_ARTIFACT,
            "schema_version": 90,
            "status": "fixed_final_awaiting_preregistered_acceptance_gates",
            "bank_name": FRESH_BANK_NAME,
            "target_module": EXPECTED_FRESH_TARGET,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "parameter_count": fresh.parameter_count,
            "initialization_algorithm": bridge["initialization_algorithm"],
            "initialization_seed": int(bridge["initialization_seed"]),
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v85(weights),
            "frozen_parent_bank_count": PARENT_BANK_COUNT,
            "total_bank_count": TOTAL_BANK_COUNT,
            "environmental_memory_serialized": False,
            "environmental_text_serialized": False,
            "environmental_text_inputs": [],
            "questions_or_answers_serialized": False,
            "training_metadata_serialized": False,
            "training_inventory_serialized": False,
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


def load_fixed_final_bridge_v90(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    """Authenticate and load the two-tensor V90 fixed-final bridge."""

    root = resolve_v85(candidate)
    metadata = strict_json_v89(root / METADATA_FILENAME)
    weights = root / WEIGHTS_FILENAME
    if (
        metadata.get("artifact") != CANDIDATE_ARTIFACT
        or metadata.get("schema_version") != 90
        or metadata.get("status") != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("target_module") != EXPECTED_FRESH_TARGET
        or metadata.get("rank") != 8
        or float(metadata.get("alpha", -1.0)) != 16.0
        or float(metadata.get("dropout", -1.0)) != 0.0
        or metadata.get("parameter_count") != FRESH_PARAMETER_COUNT
        or metadata.get("weights_sha256") != sha256_file_v85(weights)
        or metadata.get("frozen_parent_bank_count") != PARENT_BANK_COUNT
        or metadata.get("total_bank_count") != TOTAL_BANK_COUNT
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("environmental_text_serialized") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("training_metadata_serialized") is not False
        or metadata.get("training_inventory_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V90 fixed-final candidate authentication failed")
    _candidate_fingerprint_v90(root)
    _load_fresh_state_v90(collection, load_file(str(weights), device="cpu"))
    if collection.bank(FRESH_BANK_NAME).installation.state_sha256() != metadata.get("state_sha256"):
        raise ValueError("V90 fixed-final bridge state changed")
    return metadata


def authenticate_training_report_v90(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    """Authenticate V90's fixed training result without loading Gemma."""

    preflight = authenticate_cpu_preflight_v90(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["training_report"])
    report = strict_json_v89(path)
    gates = report.get("gates")
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("schema_version") != 90
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != EXPECTED_OPTIMIZER_UPDATES
        or report.get("micro_rows_consumed") != EXPECTED_MICRO_ROWS
        or report.get("causal_margin_rows_consumed") != EXPECTED_CAUSAL_ROWS
        or report.get("protected_read_count") != 0
        or report.get("oracle_loaded") is not False
        or report.get("held_out_generalization_claim") is not False
        or report.get("runtime_promotion_authorized") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("V90 fixed-final training report changed")
    return {**preflight, "training_report_sha256": sha256_file_v85(path)}


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _empty_mps_cache(device: torch.device) -> None:
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_training_v90(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Execute the sealed 1,032-row, 172-update V90 training schedule."""

    started = time.monotonic()
    config = load_config_v90(config_path)
    report_path = resolve_v85(config["outputs"]["training_report"])
    candidate_path = resolve_v85(config["outputs"]["fixed_final_candidate"])
    if report_path.exists() or candidate_path.exists():
        raise FileExistsError("V90 fixed-final outputs already exist")

    audit = FileAccessAudit(
        forbidden_component_names=_FORBIDDEN_COMPONENTS,
        block_forbidden=True,
    )
    with audit:
        source_hashes = authenticate_sources_v90(config)
        preflight = authenticate_cpu_preflight_v90(config, config_path=config_path)
        canonical = load_canonical_rows_v90(config)
        items = derive_training_items_v90(config, canonical)
        inventory_hash = canonical_sha256_v85(inventory_v90(items))
        schedule = schedule_v90(
            items,
            seed=int(config["training"]["row_order_seed"]),
            epochs=int(config["dataset"]["epochs"]),
        )
        schedule_hash = canonical_sha256_v85(
            [[epoch, item.schedule_id] for epoch, item in schedule]
        )
        if (
            inventory_hash != config["dataset"]["training_inventory_sha256"]
            or schedule_hash != config["dataset"]["training_schedule_sha256"]
            or len(items) != int(config["dataset"]["rows_per_epoch"])
            or len(schedule) != EXPECTED_MICRO_ROWS
        ):
            raise RuntimeError("V90 fixed training inventory or schedule changed")

        cpu_memory, memory_hash_before, _memory_metadata = load_scene1_memory_v86(config)
        cpu_zero_memory = zero_payload_memory_v86(cpu_memory)
        zero_hash_before = prefix_sha256(cpu_zero_memory)
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
        if language.device.type not in {"mps", "cpu"}:
            raise RuntimeError("V90 requires local MPS or CPU execution")
        collection = install_lora_banks(language.model, combined_lora_settings_v90(runtime, config))
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V90 LoRA bank installation failed")
        parent = load_frozen_parent_v90(
            collection,
            config["sources"]["parent_checkpoint"],
            config,
        )
        collection.assert_trainable_surface(language.model)
        if (
            collection.parameter_count != PARENT_PARAMETER_COUNT + FRESH_PARAMETER_COUNT
            or collection.trainable_parameter_count != FRESH_PARAMETER_COUNT
        ):
            raise RuntimeError("V90 trainable parameter surface changed")
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
        if (
            accumulation != 6
            or len(schedule) % accumulation != 0
            or int(training["optimizer_updates"]) != EXPECTED_OPTIMIZER_UPDATES
        ):
            raise RuntimeError("V90 optimizer schedule changed")
        target_margin = float(training["zero_payload_target_margin_nll"])
        ce_weight = float(training["answer_ce_weight"])
        margin_weight = float(training["zero_payload_margin_weight"])
        history: list[dict[str, Any]] = []
        interval: list[dict[str, float | bool]] = []
        optimizer_update = 0
        causal_seen = 0
        kind_seen: Counter[str] = Counter()
        answer_only_seen = 0

        for cursor, (epoch, item) in enumerate(schedule, 1):
            row = item.row
            kind_seen[item.kind] += 1
            prepared, layout = _prepared_v84(language, system_prompt, memory, row)
            if (
                layout.get("answer_only_supervision") is not True
                or layout.get("memory_supplied_directly") is not True
                or layout.get("question_derived_environmental_tokens") != 0
                or layout.get("control_tokens") != 0
            ):
                raise RuntimeError("V90 answer-only direct-memory layout changed")
            answer_only_seen += 1
            tail = _answer_tail(language, prepared)
            correct_nll = tail.mean_nll.float()
            weighted_ce = ce_weight * correct_nll
            if item.causal_margin:
                zero_prepared, zero_layout = _prepared_v84(
                    language, system_prompt, zero_memory, row
                )
                if (
                    zero_layout.get("answer_only_supervision") is not True
                    or zero_layout.get("question_derived_environmental_tokens") != 0
                ):
                    raise RuntimeError("V90 zero-memory answer-only layout changed")
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
                raise RuntimeError("V90 objective is nonfinite")
            (objective / accumulation).backward()
            del prepared, tail, correct_nll, weighted_ce, objective
            if cursor % accumulation:
                continue

            gradient_l2 = float(collection.gradient_norms()["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V90 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            clip_l2 = float(clipped.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V90 clipped gradient is nonfinite")
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
            if optimizer_update in {1, 86, EXPECTED_OPTIMIZER_UPDATES} or (
                optimizer_update % 12 == 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "v90_train_update",
                            "update": optimizer_update,
                            "total_updates": EXPECTED_OPTIMIZER_UPDATES,
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
            _empty_mps_cache(language.device)

        expected_kinds = Counter(
            {
                "canonical": 414,
                "error_replay": 96,
                "correct_anchor_replay": 366,
                "conversational": 156,
            }
        )
        if (
            optimizer_update != EXPECTED_OPTIMIZER_UPDATES
            or len(history) != EXPECTED_OPTIMIZER_UPDATES
            or causal_seen != EXPECTED_CAUSAL_ROWS
            or answer_only_seen != EXPECTED_MICRO_ROWS
            or kind_seen != expected_kinds
        ):
            raise RuntimeError("V90 fixed conversational schedule did not complete")
        language.decoder_module.eval()
        collection.eval()
        memory_hash_after = prefix_sha256(memory.detach().cpu())
        zero_hash_after = prefix_sha256(zero_memory.detach().cpu())
        if memory_hash_after != memory_hash_before or zero_hash_after != zero_hash_before:
            raise RuntimeError("V90 training mutated immutable environmental inputs")

        bindings: dict[str, str | int | bool] = {
            "config_sha256": preflight["config_sha256"],
            "preregistration_sha256": preflight["preregistration_sha256"],
            "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
            "parent_checkpoint_sha256": parent["parent_checkpoint_sha256"],
            "parent_adapter_sha256": parent["parent_adapter_sha256"],
            "parent_runtime_metadata_sha256": parent["parent_runtime_metadata_sha256"],
            "fixed_final_optimizer_updates": optimizer_update,
            "training_schedule_sha256": schedule_hash,
            "training_inventory_sha256": inventory_hash,
            "scene_memory_prefix_sha256": memory_hash_before,
            "development_known_questions_trained": True,
        }
        candidate_metadata = publish_fixed_final_candidate_v90(
            candidate_path,
            collection,
            bindings=bindings,
            experiment=config,
        )
        candidate_fingerprint, candidate_files = _candidate_fingerprint_v90(candidate_path)

    audit.assert_clean()
    gates = {
        "all_1032_sealed_micro_rows_consumed": len(schedule) == EXPECTED_MICRO_ROWS,
        "all_138_canonical_rows_consumed_once_each_epoch": kind_seen["canonical"] == 414,
        "all_16_parent_errors_replayed_twice_each_epoch": kind_seen["error_replay"] == 96,
        "all_122_parent_correct_anchors_replayed_once_each_epoch": kind_seen[
            "correct_anchor_replay"
        ]
        == 366,
        "all_52_conversational_rows_consumed_once_each_epoch": kind_seen["conversational"] == 156,
        "all_39_primary_causal_margin_rows_consumed": causal_seen == EXPECTED_CAUSAL_ROWS,
        "answer_only_ce_on_every_micro_row": answer_only_seen == EXPECTED_MICRO_ROWS,
        "fixed_final_update_172_reached": optimizer_update == EXPECTED_OPTIMIZER_UPDATES,
        "exact_v89_eleven_bank_parent_frozen": (
            parent["parent_bank_count"] == PARENT_BANK_COUNT
            and parent["parent_tensors_loaded_byte_exactly"] is True
        ),
        "sole_trainable_surface_is_v90_bridge": (
            collection.trainable_parameter_count == FRESH_PARAMETER_COUNT
            and collection.bank(FRESH_BANK_NAME).settings.trainable
        ),
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(row["gradient_l2_before_clip"]))
            and float(row["gradient_l2_before_clip"]) > 0.0
            for row in history
        ),
        "memory_hash_invariant": memory_hash_after == memory_hash_before,
        "zero_payload_hash_invariant": zero_hash_after == zero_hash_before,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
        "runtime_candidate_contains_no_supervision_or_environment": (
            candidate_metadata["questions_or_answers_serialized"] is False
            and candidate_metadata["training_metadata_serialized"] is False
            and candidate_metadata["training_inventory_serialized"] is False
            and candidate_metadata["environmental_memory_serialized"] is False
            and candidate_metadata["environmental_text_serialized"] is False
            and candidate_metadata["environmental_text_inputs"] == []
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"V90 training gate failed: {gates}")

    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 90,
        "status": "fixed_final_training_complete_not_promoted",
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "device": language.device.type,
        "model_id": config["sources"]["model_id"],
        "model_revision": config["sources"]["model_revision"],
        "strict_input_contract": config["strict_input_contract"],
        "source_hashes": source_hashes,
        "frozen_parent": parent,
        "trainable_bridge": {
            "bank_name": FRESH_BANK_NAME,
            "target_module": EXPECTED_FRESH_TARGET,
            "parameter_count": FRESH_PARAMETER_COUNT,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": candidate_metadata["state_sha256"],
            "unmerged": True,
        },
        "training_protocol": config["training"],
        "training_inventory": {
            "canonical_unique_rows": 138,
            "parent_errors": 16,
            "parent_correct_anchors": 122,
            "conversational_rows_per_epoch": 52,
            "primary_causal_rows_per_epoch": 13,
            "unique_schedule_items_per_epoch": 344,
            "inventory_sha256": inventory_hash,
            "schedule_sha256": schedule_hash,
            "development_known_questions_trained": True,
            "held_out_generalization_claim": False,
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
            "checkpoint_sha256": candidate_fingerprint,
            "checkpoint_files": candidate_files,
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
    report = run_training_v90(args.config)
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
    "authenticate_training_report_v90",
    "combined_lora_settings_v90",
    "load_fixed_final_bridge_v90",
    "load_frozen_parent_v90",
    "main",
    "publish_fixed_final_candidate_v90",
    "run_training_v90",
]
