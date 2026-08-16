"""Exact local-Gemma training engine for the continuous tool decoder V2.

Importing this module is CPU-safe.  Full Gemma loading occurs only inside the
explicit smoke/train entry points, after a sealed launch-release artifact has
authorized that stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2_2 import (
    ARTIFACT,
    BOUND_SOURCE_PATHS,
    CPU_AUTHORIZATION_PATH,
    build_cpu_authorization_v2_2,
    build_mps_smoke_release_v2_2,
    build_training_release_v2_2,
    load_authorization_payload_v2_2,
)
from semantic_3d_chat.evaluation.gemma4_tool_decoder_v2_evaluation import (
    analyze_canonical_json_vocabulary_v2,
    evaluate_all_heldout_teacher_forced_v2,
    evaluate_causal_controls_v2,
    evaluate_teacher_forced_causal_controls_v2,
    generate_tool_json_v2,
    promotion_gate_results_v2,
    teacher_forced_causal_gate_results_v2,
    teacher_forced_gate_results_v2,
    teacher_forced_row_v2,
)
from semantic_3d_chat.language.gemma4_answer_tail import (
    answer_tail_forward,
    reference_answer_tail_from_full_logits,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_LORA_STATE_SHA256,
    LORA_PARAMETER_COUNT,
    PROJECTOR_PARAMETER_COUNT,
    TARGET_PROJECTION,
    TOTAL_TRAINABLE_PARAMETER_COUNT,
    NumericToolContextProjectorV2,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2_checkpoint import (
    publish_runtime_checkpoint_v2,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    install_lora_banks,
    lora_banks_settings,
)
from semantic_3d_chat.robot.gemma4_tool_decoder_v2_runtime_probe import (
    build_saved_runtime_probe_v2,
)
from semantic_3d_chat.training.gemma4_tool_decoder_v2_data import (
    action_balanced_schedule_v2,
    load_tool_decoder_dataset_v2,
    prepare_microbatch_v2,
)

AUTHORIZATION_PATH: Final[str] = CPU_AUTHORIZATION_PATH
BASE_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000"
)
TOOL_BANK_NAME: Final[str] = "embodied_tool_decoder_v2_final_down"
_BASE_ADAPTER_SHA256: Final[str] = (
    "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf"
)
_BASE_RUNTIME_SHA256: Final[str] = (
    "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def authenticate_training_authorization_v2(
    authorization: str | Path = AUTHORIZATION_PATH,
    *,
    required_stage: str = "cpu_preparation",
) -> tuple[dict[str, Any], str]:
    """Authenticate the immutable authorization and the source it binds."""

    if required_stage not in {
        "cpu_preparation",
        "full_model_mps_microbatch",
        "multi_update_training",
    }:
        raise ValueError("Unknown V2 authorization stage")
    candidate = Path(authorization)
    path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    payload = load_authorization_payload_v2_2(path)
    bindings = payload.get("bound_source_sha256")
    if not isinstance(bindings, Mapping) or set(bindings) != set(BOUND_SOURCE_PATHS):
        raise ValueError("V2 training authorization source inventory changed")
    observed_bindings = {
        source: _sha256(PROJECT_ROOT / source) for source in BOUND_SOURCE_PATHS
    }
    if (
        payload.get("schema_version") != "2.2"
        or payload.get("artifact") != ARTIFACT
        or payload.get("training_source_path")
        != "src/semantic_3d_chat/training/train_gemma4_tool_decoder_v2.py"
        or payload.get("training_source_sha256") != _sha256(Path(__file__))
        or payload.get("clearance_cache_sha256")
        != "658822707389e67481fa59b035a7e7f19c360487b19d3157b80bc23ede1db048"
        or payload.get("clearance_manifest_sha256")
        != "51cf6c0b155e149627f300c17d39369f91f14e415099fe10d9de1682ef8c7e24"
        or payload.get("v2_preregistration_sha256")
        != "0e1e41a6af2830f9b36a8711fb0649246e96254a88cdcc76b97dcb06ee3f82f4"
        or payload.get("v2_cpu_preflight_sha256")
        != "412f1d8bb9804b2d38b0335c985225c9cf1e4226758858cee18d906dc5f742e7"
        or payload.get("trace_rows_sha256")
        != "72434178ff1cf23c2dfeb98d52cb7b4c443fcc8715c1dd4ee883d87ae127e7ad"
        or payload.get("prefix_inventory_sha256")
        != "c477fd12bc4104f147f73c2f6d46904e0b83b3c584206cb227fd70e9371d0d63"
        or payload.get("v1_terminal_failure_sha256")
        != "83939de71e31310b7d523e78c29d3e29add86e2c3dfe916e089b19dfb06decaa"
        or dict(bindings) != observed_bindings
    ):
        raise ValueError("V2.2 authorization or its bound source changed")
    stage = payload.get("authorization_stage")
    if stage != required_stage:
        raise PermissionError(
            f"V2.2 artifact stage {stage!r} cannot authorize {required_stage!r}"
        )
    if required_stage == "cpu_preparation":
        expected = build_cpu_authorization_v2_2()
    elif required_stage == "full_model_mps_microbatch":
        expected = build_mps_smoke_release_v2_2(
            payload.get("parent_authorization_path")
        )
    else:
        expected = build_training_release_v2_2(
            smoke_release=payload.get("parent_authorization_path"),
            smoke_report=payload.get("full_model_mps_microbatch_smoke_path"),
        )
    if payload != expected:
        raise ValueError("V2.2 authorization ancestry, smoke evidence, or bytes changed")
    return payload, _sha256(path)


def _load_frozen_source_lora(
    collection: LoRABankCollection,
    checkpoint: Path,
) -> dict[str, Any]:
    """Load exactly the six V54 banks while leaving fresh V2 at update zero."""

    weights = checkpoint / "adapter.safetensors"
    runtime_path = checkpoint / "runtime_metadata.json"
    if _sha256(weights) != _BASE_ADAPTER_SHA256 or _sha256(runtime_path) != (
        _BASE_RUNTIME_SHA256
    ):
        raise ValueError("V2 frozen V54 source checkpoint changed")
    metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
    source_hashes = metadata.get("lora_bank_state_sha256")
    source_wrapped = metadata.get("lora_bank_wrapped_modules")
    if not isinstance(source_hashes, Mapping) or not isinstance(source_wrapped, Mapping):
        raise TypeError("V2 V54 source has no named LoRA state contract")
    frozen = [bank for bank in collection.banks if not bank.settings.trainable]
    fresh = [bank for bank in collection.banks if bank.settings.trainable]
    if len(frozen) != 6 or len(fresh) != 1 or fresh[0].settings.name != TOOL_BANK_NAME:
        raise ValueError("V2 requires six frozen V54 banks and one fresh tool bank")
    if set(source_hashes) != {bank.settings.name for bank in frozen}:
        raise ValueError("V2 frozen bank inventory differs from V54 source")
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
        if list(bank.installation.target_names) != source_wrapped[name]:
            raise ValueError(f"V2 frozen bank module paths changed: {name}")
        if bank.installation.state_sha256() != source_hashes[name]:
            raise ValueError(f"V2 frozen bank state changed: {name}")
    fresh_installation = fresh[0].installation
    if (
        fresh_installation.target_names != (TARGET_PROJECTION,)
        or fresh_installation.parameter_count != LORA_PARAMETER_COUNT
        or fresh_installation.state_sha256() != INITIAL_LORA_STATE_SHA256
        or any(
            torch.count_nonzero(adapter.lora_b).item() != 0
            for adapter in fresh_installation.adapters
        )
    ):
        raise ValueError("V2 fresh tool bank no longer has its exact zero-output state")
    collection.validate_state()
    return {
        "source_adapter_sha256": _BASE_ADAPTER_SHA256,
        "source_runtime_sha256": _BASE_RUNTIME_SHA256,
        "frozen_bank_state_sha256": dict(source_hashes),
        "fresh_bank_state_sha256": fresh_installation.state_sha256(),
        "fresh_bank_parameter_count": fresh_installation.parameter_count,
    }


def _load_training_bundle(config: dict[str, Any]) -> tuple[Any, LoRABankCollection, Any]:
    """The sole function that loads full Gemma parameters."""

    language = load_local_language_model(
        str(config["language"]["model_id"]),
        str(config["language"]["revision"]),
        str(config["language"]["dtype"]),
        freeze=True,
        local_files_only=True,
        backend="gemma4",
        decoder_gradient_checkpointing=True,
    )
    if language.device.type != "mps":
        raise RuntimeError("V2 full training launch requires the preregistered MPS device")
    collection = install_lora_banks(language.model, lora_banks_settings(config))
    if not isinstance(collection, LoRABankCollection):
        raise TypeError("V2 failed to install named Gemma LoRA banks")
    source = _load_frozen_source_lora(collection, PROJECT_ROOT / BASE_CHECKPOINT)
    collection.assert_trainable_surface(language.model)
    projector = NumericToolContextProjectorV2().to(language.device)
    if projector.trainable_parameter_count != PROJECTOR_PARAMETER_COUNT:
        raise RuntimeError("V2 numeric projector parameter count changed")
    return language, collection, (projector, source)


def run_full_model_mps_microbatch_smoke_v2(
    config: dict[str, Any],
    *,
    authorization: str | Path = AUTHORIZATION_PATH,
) -> dict[str, Any]:
    """Run one real full-model backward microbatch and zero optimizer steps."""

    _payload, authorization_sha = authenticate_training_authorization_v2(
        authorization, required_stage="full_model_mps_microbatch"
    )
    started = time.monotonic()
    language, collection, bundle = _load_training_bundle(config)
    projector, source = bundle
    dataset = load_tool_decoder_dataset_v2(config)
    index = action_balanced_schedule_v2(dataset, microbatch_count=1, seed=2026081218)[0]
    prepared, sample = prepare_microbatch_v2(
        dataset,
        index,
        language=language,
        projector=projector,
        max_turn_degrees=float(config["robot"]["max_turn_degrees"]),
        max_move_m=float(config["robot"]["max_move_m"]),
    )
    # Equivalence runs without gradients first.  The full reference is allowed
    # only in this one-row smoke, never in training or held-out evaluation.
    with torch.inference_mode():
        full = language.prefix_backend.prefill(prepared, use_cache=False)
        reference = reference_answer_tail_from_full_logits(
            full.logits.float(), prepared.labels
        )
        tail_reference = answer_tail_forward(language, prepared)
        equivalence_difference = float(
            (reference.mean_nll - tail_reference.mean_nll).abs().cpu()
        )
    if equivalence_difference > 1e-6:
        raise RuntimeError("V2 real full-vs-tail answer NLL equivalence failed")
    if language.device.type == "mps":
        torch.mps.empty_cache()
    language.model.zero_grad(set_to_none=True)
    projector.zero_grad(set_to_none=True)
    loss = answer_tail_forward(language, prepared).mean_nll.float()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("V2 full-model microbatch loss is invalid")
    loss.backward()
    gradients = collection.gradient_norms()
    projector_gradient = math.sqrt(
        sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in projector.parameters()
            if parameter.grad is not None
        )
    )
    if gradients["total_l2"] <= 0.0 or projector_gradient <= 0.0:
        raise RuntimeError("V2 full-model gradients did not reach both trainable surfaces")
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_full_mps_smoke.v2_2",
        "status": "passed",
        "authorization_sha256": authorization_sha,
        "device": "mps",
        "full_model_loaded": True,
        "mps_used": True,
        "sample_id": sample.sample_id,
        "microbatches": 1,
        "optimizer_steps": 0,
        "loss": float(loss.detach().cpu()),
        "real_full_vs_tail_answer_nll_absolute_difference": equivalence_difference,
        "real_full_vs_tail_answer_nll_tolerance": 1e-6,
        "training_and_evaluation_use_answer_tail_only": True,
        "lora_gradient_l2": gradients["total_l2"],
        "projector_gradient_l2": projector_gradient,
        "trainable_parameter_count": TOTAL_TRAINABLE_PARAMETER_COUNT,
        "source": source,
        "elapsed_seconds": time.monotonic() - started,
        "training_executed": False,
        "checkpoint_published": False,
    }


def train_gemma4_tool_decoder_v2(
    config: dict[str, Any],
    *,
    authorization: str | Path = AUTHORIZATION_PATH,
    report_path: str | Path,
    runtime_checkpoint: str | Path,
    runtime_probe: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train the single sealed arm and publish only after every runtime gate."""

    authorization_payload, authorization_sha = authenticate_training_authorization_v2(
        authorization, required_stage="multi_update_training"
    )
    full_smoke = authorization_payload.get("full_model_mps_microbatch_smoke")
    if (
        not isinstance(full_smoke, Mapping)
        or full_smoke.get("status") != "passed"
        or full_smoke.get("optimizer_steps") != 0
        or full_smoke.get("device") != "mps"
    ):
        raise PermissionError("V2 multi-update training lacks a passed full-model MPS smoke")
    output_report = PROJECT_ROOT / Path(report_path)
    if output_report.exists() or (PROJECT_ROOT / Path(runtime_checkpoint)).exists():
        raise FileExistsError("V2 has one create-once training arm and runtime checkpoint")
    if (
        full_smoke.get("real_full_vs_tail_answer_nll_absolute_difference", math.inf)
        > 1e-6
        or full_smoke.get("training_and_evaluation_use_answer_tail_only") is not True
    ):
        raise PermissionError("V2 multi-update launch lacks real answer-tail equivalence")
    language, collection, bundle = _load_training_bundle(config)
    projector, source = bundle
    dataset = load_tool_decoder_dataset_v2(config)
    vocabulary = analyze_canonical_json_vocabulary_v2(
        language.tokenizer,
        max_turn_degrees=float(config["robot"]["max_turn_degrees"]),
        max_move_m=float(config["robot"]["max_move_m"]),
    )
    trainable_bank = collection.bank(TOOL_BANK_NAME).installation
    optimizer = torch.optim.AdamW(
        [
            {
                "params": trainable_bank.parameters(),
                "lr": 0.0001,
                "weight_decay": 0.0,
            },
            {
                "params": list(projector.parameters()),
                "lr": 0.0002,
                "weight_decay": 0.0,
            },
        ]
    )
    schedule = action_balanced_schedule_v2(
        dataset, microbatch_count=64 * 8, seed=2026081218
    )
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    accumulated = 0.0
    for microbatch, index in enumerate(schedule, start=1):
        prepared, _sample = prepare_microbatch_v2(
            dataset,
            index,
            language=language,
            projector=projector,
            max_turn_degrees=float(config["robot"]["max_turn_degrees"]),
            max_move_m=float(config["robot"]["max_move_m"]),
        )
        loss = answer_tail_forward(language, prepared).mean_nll.float()
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError("V2 training produced a nonfinite loss")
        (loss / 8.0).backward()
        accumulated += float(loss.detach().cpu())
        if microbatch % 8:
            continue
        parameters = [*trainable_bank.parameters(), *projector.parameters()]
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("V2 training produced a nonfinite gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update = microbatch // 8
        mean_loss = accumulated / 8.0
        accumulated = 0.0
        history.append(
            {
                "update": update,
                "training_loss": mean_loss,
                "gradient_l2_before_clip": float(gradient_norm.detach().cpu()),
            }
        )
    if len(history) != 64 or history[-1]["update"] != 64:
        raise RuntimeError("V2 training did not complete its sealed optimizer schedule")
    selected_update = 64
    selected_training_loss = float(history[-1]["training_loss"])
    collection.eval()
    projector.eval()

    teacher_cache: dict[tuple[int, str], Mapping[str, Any]] = {}

    def score_teacher_forced(
        dataset_value: Any, index: int, control: str
    ) -> Mapping[str, Any]:
        key = (index, control)
        if key not in teacher_cache:
            teacher_cache[key] = teacher_forced_row_v2(
                dataset_value,
                index,
                control,
                language=language,
                projector=projector,
                config=config,
                max_turn_degrees=float(config["robot"]["max_turn_degrees"]),
                max_move_m=float(config["robot"]["max_move_m"]),
            )
        return teacher_cache[key]

    teacher_forced = evaluate_all_heldout_teacher_forced_v2(
        dataset, score_teacher_forced
    )
    teacher_gate = teacher_forced_gate_results_v2(teacher_forced)
    if not teacher_gate["passed"]:
        report = {
            "schema": "semantic_3d_chat.gemma4_tool_decoder_training.v2",
            "status": "rejected_before_greedy_generation_no_runtime_checkpoint",
            "authorization_sha256": authorization_sha,
            "source": source,
            "canonical_json_vocabulary": vocabulary,
            "optimizer_updates": 64,
            "gradient_accumulation": 8,
            "microbatch_size": 1,
            "checkpoint_selection": "fixed_final_update_no_posthoc_selection",
            "selected_update": selected_update,
            "selected_training_loss": selected_training_loss,
            "history": history,
            "all_heldout_teacher_forced": teacher_forced,
            "teacher_forced_early_gate": teacher_gate,
            "greedy_generation_executed": False,
            "elapsed_seconds": time.monotonic() - started,
            "runtime_checkpoint_published": False,
        }
        _atomic_json(output_report, report)
        return report
    teacher_causal = evaluate_teacher_forced_causal_controls_v2(
        dataset, score_teacher_forced
    )
    teacher_causal_gate = teacher_forced_causal_gate_results_v2(teacher_causal)
    if not teacher_causal_gate["passed"]:
        report = {
            "schema": "semantic_3d_chat.gemma4_tool_decoder_training.v2",
            "status": "rejected_by_teacher_causal_gate_no_runtime_checkpoint",
            "authorization_sha256": authorization_sha,
            "source": source,
            "canonical_json_vocabulary": vocabulary,
            "optimizer_updates": 64,
            "gradient_accumulation": 8,
            "microbatch_size": 1,
            "checkpoint_selection": "fixed_final_update_no_posthoc_selection",
            "selected_update": selected_update,
            "selected_training_loss": selected_training_loss,
            "history": history,
            "all_heldout_teacher_forced": teacher_forced,
            "teacher_forced_early_gate": teacher_gate,
            "teacher_forced_causal_controls": teacher_causal,
            "teacher_forced_causal_gate": teacher_causal_gate,
            "teacher_forced_unique_forward_count": len(teacher_cache),
            "greedy_generation_executed": False,
            "elapsed_seconds": time.monotonic() - started,
            "runtime_checkpoint_published": False,
        }
        _atomic_json(output_report, report)
        return report

    def generate(dataset_value: Any, index: int, control: str) -> str:
        return generate_tool_json_v2(
            dataset_value,
            index,
            control,
            language=language,
            projector=projector,
            max_turn_degrees=float(config["robot"]["max_turn_degrees"]),
            max_move_m=float(config["robot"]["max_move_m"]),
        )

    evaluation = evaluate_causal_controls_v2(dataset, config, generate)
    evaluation["all_heldout_teacher_forced"] = teacher_forced
    evaluation["teacher_forced_early_gate"] = teacher_gate
    evaluation["teacher_forced_causal_controls"] = teacher_causal
    evaluation["teacher_forced_causal_gate"] = teacher_causal_gate
    gates = promotion_gate_results_v2(evaluation)
    report: dict[str, Any] = {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_training.v2",
        "status": "promotion_pending" if gates["passed"] else "rejected_no_runtime_checkpoint",
        "authorization_sha256": authorization_sha,
        "source": source,
        "canonical_json_vocabulary": vocabulary,
        "optimizer_updates": 64,
        "gradient_accumulation": 8,
        "microbatch_size": 1,
        "checkpoint_selection": "fixed_final_update_no_posthoc_selection",
        "selected_update": selected_update,
        "selected_training_loss": selected_training_loss,
        "history": history,
        "all_heldout_teacher_forced": teacher_forced,
        "teacher_forced_early_gate": teacher_gate,
        "teacher_forced_causal_controls": teacher_causal,
        "teacher_forced_causal_gate": teacher_causal_gate,
        "teacher_forced_unique_forward_count": len(teacher_cache),
        "evaluation": evaluation,
        "promotion_gates": gates,
        "elapsed_seconds": time.monotonic() - started,
        "greedy_generation_executed": True,
        "runtime_checkpoint_published": False,
    }
    if not gates["passed"]:
        _atomic_json(output_report, report)
        return report
    provenance = {
        "base_checkpoint_sha256": _BASE_ADAPTER_SHA256,
        "preregistration_sha256": str(authorization_payload["v2_preregistration_sha256"]),
        "cpu_preflight_sha256": str(authorization_payload["v2_cpu_preflight_sha256"]),
        "training_authorization_sha256": authorization_sha,
        "clearance_cache_sha256": str(authorization_payload["clearance_cache_sha256"]),
        "prefix_inventory_sha256": str(authorization_payload["prefix_inventory_sha256"]),
    }
    probe_callback = runtime_probe or build_saved_runtime_probe_v2(
        language=language,
        installation=trainable_bank,
        projector=projector,
        dataset=dataset,
        provenance=provenance,
        config=config,
    )
    publication = publish_runtime_checkpoint_v2(
        runtime_checkpoint,
        trainable_bank,
        projector,
        provenance=provenance,
        evaluation=evaluation,
        runtime_probe=probe_callback,
        config=config,
    )
    report.update(
        {
            "status": "passed_and_runtime_published",
            "runtime_checkpoint_published": True,
            "publication": publication,
        }
    )
    _atomic_json(output_report, report)
    return report


__all__ = [
    "AUTHORIZATION_PATH",
    "authenticate_training_authorization_v2",
    "run_full_model_mps_microbatch_smoke_v2",
    "train_gemma4_tool_decoder_v2",
]
