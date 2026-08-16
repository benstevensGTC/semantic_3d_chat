"""Bounded V84 wiring run over the immutable 738-token V83 scene memory.

The sole learned surface is one unmerged FP32 LoRA bank inside Gemma.  The
complete scene memory is compiled before question tokenization, supplied
directly through Gemma's native image-prefix protocol, and never retrieved,
summarized, projected, or selected as a function of the question.
"""

from __future__ import annotations

import argparse
import hashlib
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

from semantic_3d_chat.chat.question_control_runtime import sanitize_generated_answer
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    audit_v83_direct_prepared_layout,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.v84_strict_bridge_preflight import (
    CONFIG,
    TARGET_MODULE,
    _authenticate_sources,
    _split_preflight,
    load_config_v84,
    sha256_file_v84,
)
from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward
from semantic_3d_chat.language.generation import generate_from_embeddings
from semantic_3d_chat.language.local_lm import (
    load_local_language_model,
    prompt_token_ids,
)
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
    lora_banks_settings,
)
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    prefix_sha256,
)
from semantic_3d_chat.training.train_adapter import tokenize_answer
from semantic_3d_chat.training.train_question_control_v56 import (
    assert_answer_only_labels,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    RowV73,
    changed_units_v73,
    load_training_rows_v73,
    split_rows_v73,
)
from semantic_3d_chat.training.v82_reader_artifacts import load_v82_cache

FRESH_BANK_NAME: Final[str] = "v84_strict_fixed_memory_bridge"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V84 JSON artifact must be an object: {path}")
    return value


def authenticate_preflight_v84(config: Mapping[str, Any]) -> dict[str, str]:
    """Require the create-once preregistration and passed CPU preflight."""

    outputs = config["outputs"]
    prereg_path = _resolve(outputs["preregistration"])
    preflight_path = _resolve(outputs["cpu_preflight"])
    prereg = _strict_json(prereg_path)
    preflight = _strict_json(preflight_path)
    config_sha = sha256_file_v84(CONFIG)
    if (
        prereg.get("artifact") != "gemma4_v84_strict_bridge_preregistration_v1"
        or prereg.get("status") != "sealed_before_first_full_model_measurement"
        or prereg.get("config_sha256") != config_sha
        or prereg.get("optimizer_updates") != 0
        or prereg.get("sealed_historical_16_loaded") is not False
        or prereg.get("oracle_loaded") is not False
        or preflight.get("artifact") != "gemma4_v84_strict_bridge_cpu_preflight_v1"
        or preflight.get("status") != "passed"
        or preflight.get("passed") is not True
        or preflight.get("config_sha256") != config_sha
        or preflight.get("optimizer_updates") != 0
        or preflight.get("protected_or_sealed_behavior_artifacts_opened") != []
        or preflight.get("oracle_loaded") is not False
        or preflight.get("preregistration_sha256") != sha256_file_v84(prereg_path)
    ):
        raise ValueError("V84 preregistration or CPU preflight changed")
    return {
        "config_sha256": config_sha,
        "preregistration_sha256": sha256_file_v84(prereg_path),
        "cpu_preflight_sha256": sha256_file_v84(preflight_path),
    }


def combined_lora_settings_v84(
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
        initialization_algorithm="cpu_kaiming_uniform_a_exact_zero_b",
        initialization_seed=int(bridge["initialization_seed"]),
        expected_initial_state_sha256=str(bridge["expected_initial_state_sha256"]),
    )
    return LoRABanksSettings(base.banks + (fresh,))


def load_frozen_v54_banks_v84(
    collection: LoRABankCollection,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Load only the six inherited banks; retain V84 at exact update zero."""

    root = _resolve(checkpoint)
    weights = root / "adapter.safetensors"
    runtime_path = root / "runtime_metadata.json"
    metadata = _strict_json(runtime_path)
    source_hashes = metadata.get("lora_bank_state_sha256")
    source_wrapped = metadata.get("lora_bank_wrapped_modules")
    if not isinstance(source_hashes, Mapping) or not isinstance(source_wrapped, Mapping):
        raise TypeError("V84 V54 source lacks named LoRA state metadata")
    frozen = [bank for bank in collection.banks if not bank.settings.trainable]
    fresh = [bank for bank in collection.banks if bank.settings.trainable]
    if (
        len(frozen) != 6
        or len(fresh) != 1
        or fresh[0].settings.name != FRESH_BANK_NAME
        or set(source_hashes) != {bank.settings.name for bank in frozen}
    ):
        raise ValueError("V84 requires six frozen V54 banks and one fresh bridge bank")
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
            raise ValueError(f"V84 frozen bank module paths changed: {name}")
        if bank.installation.state_sha256() != source_hashes[name]:
            raise ValueError(f"V84 frozen bank state changed: {name}")
    fresh_installation = fresh[0].installation
    bridge = fresh[0].settings
    if (
        fresh_installation.target_names != (TARGET_MODULE,)
        or fresh_installation.parameter_count != 55_296
        or fresh_installation.state_sha256()
        != bridge.expected_initial_state_sha256
        or any(
            torch.count_nonzero(adapter.lora_b).item() != 0
            for adapter in fresh_installation.adapters
        )
    ):
        raise ValueError("V84 bridge no longer has its deterministic zero-output state")
    collection.validate_state()
    return {
        "adapter_sha256": sha256_file_v84(weights),
        "runtime_metadata_sha256": sha256_file_v84(runtime_path),
        "frozen_bank_state_sha256": dict(source_hashes),
        "fresh_initial_state_sha256": fresh_installation.state_sha256(),
    }


def select_wiring_rows_v84(config: Mapping[str, Any]) -> tuple[RowV73, RowV73]:
    rows = load_training_rows_v73(config["sources"]["historical_qa"])
    train, _development = split_rows_v73(rows)
    units = sorted(
        changed_units_v73(train),
        key=lambda unit: (unit.change_type, unit.pair_id, unit.question_key),
    )
    unit = units[0]
    wiring = config["wiring"]
    if (
        unit.change_type != wiring["selected_change_type"]
        or unit.pair_id != wiring["selected_pair_id"]
        or unit.question_key != wiring["selected_question_key"]
        or [[unit.left.scene_id, unit.left.question_id], [unit.right.scene_id, unit.right.question_id]]
        != wiring["selected_rows"]
    ):
        raise ValueError("V84 fixed wiring selection changed")
    return unit.left, unit.right


def _scene_memories_v84(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Compile both immutable scene memories before any question is tokenized."""

    cache = load_v82_cache(_resolve(config["sources"]["train_memory_cache"]))
    scene_ids = list(cache.metadata["scene_ids"])
    result: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    for row in rows:
        if row.scene_id in result:
            continue
        memory = cache.tensors["scene_memories"][scene_ids.index(row.scene_id)]
        memory = memory.unsqueeze(0).detach().cpu().contiguous()
        if tuple(memory.shape) != (1, 738, 1536) or memory.dtype != torch.bfloat16:
            raise ValueError("V84 cached scene memory shape or dtype changed")
        result[row.scene_id] = memory
        hashes[row.scene_id] = prefix_sha256(memory)
    if set(result) != {row.scene_id for row in rows}:
        raise RuntimeError("V84 did not compile every wiring scene before questions")
    return result, hashes


def audit_training_layout_v84(
    *, memory: torch.Tensor, prompt_ids: torch.Tensor, answer_ids: torch.Tensor, prepared: Any
) -> dict[str, Any]:
    """Extend the V83 direct-layout proof to an answer-supervised suffix."""

    expected_total = int(memory.shape[1] + prompt_ids.shape[1] + answer_ids.shape[1])
    if (
        tuple(prepared.inputs_embeds.shape) != (1, expected_total, 1536)
        or prepared.scene_prefix_length != 738
        or prepared.labels is None
    ):
        raise RuntimeError("V84 answer-supervised direct layout changed")
    scene_start = 1
    scene_stop = scene_start + 738
    if not torch.equal(
        prepared.inputs_embeds[:, scene_start:scene_stop], memory.to(prepared.inputs_embeds)
    ):
        raise RuntimeError("V84 training did not supply all 738 memory tokens directly")
    assert_answer_only_labels(prepared.labels, answer_ids)
    if (
        bool(torch.any(prepared.mm_token_type_ids[:, 2:738] != 1))
        or bool(torch.any(prepared.mm_token_type_ids[:, :2] != 0))
        or bool(torch.any(prepared.mm_token_type_ids[:, 738:] != 0))
        or not bool(torch.all(prepared.attention_mask == 1))
    ):
        raise RuntimeError("V84 native image modality or visibility contract changed")
    return {
        "memory_tokens": 738,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "answer_tokens": int(answer_ids.shape[1]),
        "prepared_tokens": expected_total,
        "memory_supplied_directly": True,
        "answer_only_supervision": True,
        "control_tokens": 0,
        "question_derived_environmental_tokens": 0,
    }


def _prepared_v84(
    language: Any,
    system_prompt: str,
    memory: torch.Tensor,
    row: RowV73,
) -> tuple[Any, dict[str, Any]]:
    backend = language.prefix_backend
    if backend is None or language.backend_name != "gemma4":
        raise RuntimeError("V84 requires the local Gemma 4 prefix backend")
    model_dtype = next(language.model.parameters()).dtype
    fixed = memory.to(device=language.device, dtype=model_dtype)
    prompt = prompt_token_ids(
        language.tokenizer, system_prompt, row.question, language.device
    )
    answer = tokenize_answer(language.tokenizer, row.answer, language.device)
    prepared = backend.prepare(
        fixed,
        prompt,
        answer,
        scene_prefix_after_bos=True,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        control_tokens=None,
    )
    audit = audit_training_layout_v84(
        memory=fixed,
        prompt_ids=prompt,
        answer_ids=answer,
        prepared=prepared,
    )
    return prepared, audit


@torch.inference_mode()
def _measure_nll_v84(
    language: Any,
    system_prompt: str,
    memory: torch.Tensor,
    row: RowV73,
) -> tuple[dict[str, float], dict[str, Any]]:
    prepared, audit = _prepared_v84(language, system_prompt, memory, row)
    tail = answer_tail_forward(language, prepared)
    top1 = tail.logits[0].float().argmax(dim=-1)
    return (
        {
            "mean_nll": float(tail.mean_nll.cpu()),
            "answer_token_top1_accuracy": float(
                top1.eq(tail.targets).float().mean().cpu()
            ),
            "answer_token_count": float(tail.targets.numel()),
        },
        audit,
    )


def _eos_ids(language: Any) -> int | list[int] | None:
    values: list[int] = []
    for candidate in (
        getattr(language.tokenizer, "eos_token_id", None),
        getattr(getattr(language.model, "generation_config", None), "eos_token_id", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            values.extend(int(value) for value in candidate)
        else:
            values.append(int(candidate))
    unique = sorted(set(values))
    return None if not unique else unique[0] if len(unique) == 1 else unique


@torch.inference_mode()
def _generate_v84(
    language: Any,
    system_prompt: str,
    memory: torch.Tensor,
    row: RowV73,
    *,
    max_new_tokens: int,
) -> str:
    model_dtype = next(language.model.parameters()).dtype
    fixed = memory.to(device=language.device, dtype=model_dtype)
    prompt = prompt_token_ids(
        language.tokenizer, system_prompt, row.question, language.device
    )
    prepared = language.prefix_backend.prepare(
        fixed,
        prompt,
        scene_prefix_after_bos=True,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        control_tokens=None,
    )
    audit_v83_direct_prepared_layout(
        backend=language.prefix_backend,
        fixed_memory=fixed,
        prompt_ids=prompt,
        prepared=prepared,
    )
    generated = language.generate_from_scene_prefix(
        fixed,
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_ids=_eos_ids(language),
        scene_prefix_after_bos=True,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        fallback=generate_from_embeddings,
    )
    decoded = language.tokenizer.decode(
        generated[0].detach().cpu().tolist(), skip_special_tokens=True
    ).strip()
    return sanitize_generated_answer(decoded) or "unknown"


def _publish_candidate_v84(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    root = _resolve(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V84 create-once candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        fresh = collection.bank(FRESH_BANK_NAME).installation
        state = {
            "lora_a": fresh.adapters[0].lora_a.detach().cpu().contiguous(),
            "lora_b": fresh.adapters[0].lora_b.detach().cpu().contiguous(),
        }
        save_file(
            state,
            str(temporary / WEIGHTS_FILENAME),
            metadata={
                "artifact": "gemma4_v84_strict_fixed_memory_bridge_candidate_v1",
                "environmental_memory_serialized": "false",
                "questions_or_answers_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        metadata = {
            "artifact": "gemma4_v84_strict_fixed_memory_bridge_candidate_v1",
            "schema_version": 84,
            "status": "wiring_only_not_runtime_promoted",
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "parameter_count": fresh.parameter_count,
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v84(temporary / WEIGHTS_FILENAME),
            "environmental_memory_serialized": False,
            "questions_or_answers_serialized": False,
            "oracle_serialized": False,
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


def _atomic_create_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V84 create-once wiring report exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run_wiring_v84(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v84(config_path)
    _authenticate_sources(config)
    _split_preflight(config)
    preflight_bindings = authenticate_preflight_v84(config)
    report_path = _resolve(config["outputs"]["wiring_report"])
    candidate_path = _resolve(config["outputs"]["candidate"])
    if report_path.exists() or candidate_path.exists():
        raise FileExistsError("V84 create-once wiring outputs already exist")

    rows = select_wiring_rows_v84(config)
    # This call intentionally precedes local-Gemma loading and all question
    # tokenization. It is the environment-input boundary for the experiment.
    cpu_memories, memory_hashes_before = _scene_memories_v84(config, rows)

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
        raise RuntimeError("V84 bounded full-model wiring requires local MPS")
    settings = combined_lora_settings_v84(runtime, config)
    collection = install_lora_banks(language.model, settings)
    if not isinstance(collection, LoRABankCollection):
        raise TypeError("V84 failed to install its named LoRA bank collection")
    source = load_frozen_v54_banks_v84(
        collection, config["sources"]["base_checkpoint"]
    )
    collection.assert_trainable_surface(language.model)
    fresh = collection.bank(FRESH_BANK_NAME).installation
    initial_state_sha = fresh.state_sha256()
    system_prompt = str(runtime["language"]["system_prompt"])

    memory_by_scene = {
        scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
        for scene_id, memory in cpu_memories.items()
    }
    language.decoder_module.eval()
    collection.eval()
    initial_rows: list[dict[str, Any]] = []
    initial_layouts: list[dict[str, Any]] = []
    for row in rows:
        correct, layout = _measure_nll_v84(
            language, system_prompt, memory_by_scene[row.scene_id], row
        )
        wrong, _ = _measure_nll_v84(
            language, system_prompt, memory_by_scene[row.paired_scene_id], row
        )
        prediction = _generate_v84(
            language,
            system_prompt,
            memory_by_scene[row.scene_id],
            row,
            max_new_tokens=int(runtime["language"]["max_answer_tokens"]),
        )
        initial_layouts.append(layout)
        initial_rows.append(
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "answer": row.answer,
                "correct_scene": correct,
                "paired_wrong_scene": wrong,
                "wrong_minus_correct_nll": wrong["mean_nll"] - correct["mean_nll"],
                "greedy_prediction": prediction,
                "greedy_normalized_exact": normalize_answer(prediction) == row.answer,
            }
        )
        if language.device.type == "mps":
            torch.mps.empty_cache()

    wiring = config["wiring"]
    parameters = collection.parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(wiring["learning_rate"]),
        weight_decay=float(wiring["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    language.decoder_module.train()
    collection.train()
    for update in range(1, int(wiring["optimizer_updates"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        row_losses: list[float] = []
        for row in rows:
            prepared, _layout = _prepared_v84(
                language, system_prompt, memory_by_scene[row.scene_id], row
            )
            tail = answer_tail_forward(language, prepared)
            loss = tail.mean_nll.float()
            if not torch.isfinite(loss):
                raise RuntimeError("V84 wiring loss is nonfinite")
            row_losses.append(float(loss.detach().cpu()))
            (loss / len(rows)).backward()
            del tail, prepared, loss
        gradients = collection.gradient_norms()
        gradient_l2 = float(gradients["total_l2"])
        if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
            raise RuntimeError("V84 wiring gradient is zero or nonfinite")
        clipped = torch.nn.utils.clip_grad_norm_(
            parameters, float(wiring["gradient_clip_norm"])
        )
        clipped_l2 = float(clipped.detach().cpu())
        if not math.isfinite(clipped_l2):
            raise RuntimeError("V84 clipped gradient is nonfinite")
        optimizer.step()
        collection.validate_state()
        history.append(
            {
                "update": update,
                "row_mean_nll": row_losses,
                "mean_nll": sum(row_losses) / len(row_losses),
                "gradient_l2_before_clip": gradient_l2,
                "clip_return_l2": clipped_l2,
                "state_sha256": fresh.state_sha256(),
            }
        )
        if language.device.type == "mps":
            torch.mps.empty_cache()

    language.decoder_module.eval()
    collection.eval()
    final_rows: list[dict[str, Any]] = []
    for row in rows:
        correct, _layout = _measure_nll_v84(
            language, system_prompt, memory_by_scene[row.scene_id], row
        )
        wrong, _ = _measure_nll_v84(
            language, system_prompt, memory_by_scene[row.paired_scene_id], row
        )
        prediction = _generate_v84(
            language,
            system_prompt,
            memory_by_scene[row.scene_id],
            row,
            max_new_tokens=int(runtime["language"]["max_answer_tokens"]),
        )
        final_rows.append(
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "answer": row.answer,
                "correct_scene": correct,
                "paired_wrong_scene": wrong,
                "wrong_minus_correct_nll": wrong["mean_nll"] - correct["mean_nll"],
                "greedy_prediction": prediction,
                "greedy_normalized_exact": normalize_answer(prediction) == row.answer,
            }
        )
        if language.device.type == "mps":
            torch.mps.empty_cache()

    memory_hashes_after = {
        scene_id: prefix_sha256(memory.detach().cpu())
        for scene_id, memory in memory_by_scene.items()
    }
    memory_hash_invariant = memory_hashes_after == memory_hashes_before
    initial_mean = sum(row["correct_scene"]["mean_nll"] for row in initial_rows) / 2
    final_mean = sum(row["correct_scene"]["mean_nll"] for row in final_rows) / 2
    each_improved = all(
        final["correct_scene"]["mean_nll"] < initial["correct_scene"]["mean_nll"]
        for initial, final in zip(initial_rows, final_rows, strict=True)
    )
    gates = {
        "final_mean_nll_below_initial": final_mean < initial_mean,
        "both_rows_nll_improve": each_improved,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(row["gradient_l2_before_clip"])
            and row["gradient_l2_before_clip"] > 0
            for row in history
        ),
        "memory_hash_invariant": memory_hash_invariant,
    }
    passed = all(gates.values())
    candidate_metadata = _publish_candidate_v84(
        candidate_path,
        collection,
        bindings={
            **preflight_bindings,
            "base_adapter_sha256": source["adapter_sha256"],
            "base_runtime_metadata_sha256": source["runtime_metadata_sha256"],
            "fixed_final_optimizer_updates": len(history),
        },
    )
    report = {
        "artifact": "gemma4_v84_strict_fixed_memory_bridge_wiring_v1",
        "schema_version": 84,
        "status": "passed" if passed else "failed_wiring_gate",
        "device": str(language.device),
        "model_id": language_config["model_id"],
        "model_revision": language_config["revision"],
        "strict_input_contract": config["strict_input_contract"],
        "preflight_bindings": preflight_bindings,
        "source_lora": source,
        "trainable_bank": {
            "name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "parameter_count": fresh.parameter_count,
            "initial_state_sha256": initial_state_sha,
            "final_state_sha256": fresh.state_sha256(),
            "unmerged": True,
        },
        "wiring_unit": {
            "change_type": wiring["selected_change_type"],
            "pair_id": wiring["selected_pair_id"],
            "question_key": wiring["selected_question_key"],
            "question_sha256": hashlib.sha256(rows[0].question.encode()).hexdigest(),
            "same_question_both_sides": rows[0].question == rows[1].question,
        },
        "scene_memories": {
            "compiled_before_question_tokenization": True,
            "shape_each": [1, 738, 1536],
            "hashes_before": memory_hashes_before,
            "hashes_after": memory_hashes_after,
            "hash_invariant": memory_hash_invariant,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_retrieval": False,
        },
        "initial_rows": initial_rows,
        "initial_mean_correct_scene_nll": initial_mean,
        "training_history": history,
        "optimizer_updates": len(history),
        "final_rows": final_rows,
        "final_mean_correct_scene_nll": final_mean,
        "mean_correct_scene_nll_delta": final_mean - initial_mean,
        "gates": gates,
        "passed": passed,
        "layout_audits": initial_layouts,
        "candidate": {
            "path": candidate_path.relative_to(PROJECT_ROOT).as_posix(),
            "metadata_sha256": _canonical_sha256(candidate_metadata),
            "runtime_promotion_authorized": False,
        },
        "development_rows_loaded_by_training_run": 0,
        "sealed_historical_16_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_create_report(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    result = run_wiring_v84(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
