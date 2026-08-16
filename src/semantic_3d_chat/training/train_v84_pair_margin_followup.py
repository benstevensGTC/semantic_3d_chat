"""Fixed 32-update train-only V84 pair-margin follow-up.

This follow-up starts from the same exact zero-output LoRA initialization as
the bounded V84 wiring run.  It never consumes the parent candidate.  Its sole
purpose is to test whether a strict, complete, question-independent 738-token
scene input can causally separate one counterfactual ``on``/``under`` pair.
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
import yaml
from safetensors.torch import save_file

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.v84_strict_bridge_preflight import sha256_file_v84
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import LoRABankCollection, install_lora_banks
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import (
    RowV73,
    changed_units_v73,
    load_training_rows_v73,
    split_rows_v73,
)
from semantic_3d_chat.training.train_v84_strict_bridge import (
    FRESH_BANK_NAME,
    _generate_v84,
    _measure_nll_v84,
    _prepared_v84,
    _scene_memories_v84,
    combined_lora_settings_v84,
    load_frozen_v54_banks_v84,
)

CONFIG: Final[Path] = Path(
    "configs/experiments/gemma4_v84_strict_fixed_memory_pair_margin.yaml"
)
PREREG_ARTIFACT: Final[str] = (
    "gemma4_v84_strict_fixed_memory_pair_margin_preregistration_v1"
)
REPORT_ARTIFACT: Final[str] = (
    "gemma4_v84_strict_fixed_memory_pair_margin_wiring_v1"
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _strict_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V84 pair-margin JSON must be an object: {path}")
    return value


def load_pair_margin_config_v84(path: str | Path = CONFIG) -> dict[str, Any]:
    source = _resolve(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v84_pair_margin"}:
        raise ValueError("V84 pair-margin config root changed")
    config = payload["v84_pair_margin"]
    if not isinstance(config, Mapping):
        raise TypeError("V84 pair-margin config payload must be a mapping")
    if (
        config.get("schema_version") != "84.1"
        or config.get("artifact")
        != "gemma4_v84_strict_fixed_memory_pair_margin_followup_v1"
        or config.get("seed") != 840084
    ):
        raise ValueError("V84 pair-margin identity changed")
    strict = config.get("strict_input_contract")
    if not isinstance(strict, Mapping) or strict != {
        "shape_per_scene": [1, 738, 1536],
        "compiled_before_question": True,
        "reused_byte_identically_across_questions": True,
        "supplied_directly_to_native_gemma_image_prefix": True,
        "all_738_memory_slots_retained": True,
        "memory_projector_enabled": False,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "control_tokens": 0,
        "environmental_text_inputs": [],
        "gemma_is_only_question_dependent_consumer": True,
    }:
        raise ValueError("V84 pair-margin strict input contract changed")
    bridge = config.get("bridge")
    if not isinstance(bridge, Mapping) or (
        bridge.get("bank_name") != FRESH_BANK_NAME
        or bridge.get("target_module")
        != "model.language_model.layers.34.mlp.down_proj"
        or bridge.get("rank") != 4
        or bridge.get("alpha") != 8.0
        or bridge.get("dropout") != 0.0
        or bridge.get("trainable_parameter_count") != 55_296
        or bridge.get("initialization_seed") != 840084
        or bridge.get("expected_initial_state_sha256")
        != "1ec186d64cab68a3ea2000968a0ca643e591cc32669c6b1b7138deb365cc5cc1"
        or bridge.get("starts_from_parent_candidate") is not False
        or bridge.get("base_gemma_frozen") is not True
        or bridge.get("inherited_v54_lora_banks_frozen") is not True
        or bridge.get("merged_weights") is not False
    ):
        raise ValueError("V84 pair-margin bridge surface changed")
    training = config.get("training")
    if not isinstance(training, Mapping) or training != {
        "selected_rows": [
            ["scene_000019", "q_000130"],
            ["scene_000020", "q_000001"],
        ],
        "selected_question_key": "cfq_5611adccccae59f1",
        "answers": ["on", "under"],
        "optimizer": "AdamW",
        "optimizer_updates": 32,
        "gradient_accumulation_rows": 2,
        "microbatch_size": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "checkpoint_selection": "fixed_final_update_32",
        "intermediate_selection": False,
        "correct_scene_answer_ce_weight": 1.0,
        "wrong_scene_margin_weight": 1.0,
        "wrong_scene_target_margin_nll": 0.5,
        "margin_definition": (
            "paired_wrong_scene_answer_nll_minus_correct_scene_answer_nll"
        ),
    }:
        raise ValueError("V84 pair-margin fixed training protocol changed")
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or (
        scope.get("optimization_scenes") != ["scene_000019", "scene_000020"]
        or any(
            scope.get(field) is not False
            for field in (
                "historical_pair_scene_disjoint_development_scored",
                "sealed_historical_16_loaded",
                "official_validation_loaded",
                "official_test_loaded",
                "deferred_final_loaded",
                "oracle_loaded",
                "runtime_promotion_authorized",
            )
        )
    ):
        raise ValueError("V84 pair-margin protected scope changed")
    return dict(config)


def authenticate_pair_margin_sources_v84(
    config: Mapping[str, Any],
) -> dict[str, str]:
    sources = config["sources"]
    expected = {
        sources["parent_config"]: sources["parent_config_sha256"],
        sources["parent_wiring_report"]: sources["parent_wiring_report_sha256"],
        sources["runtime_config"]: sources["runtime_config_sha256"],
        sources["historical_qa"]: sources["historical_qa_sha256"],
        str(Path(sources["train_memory_cache"]) / "training_tensors.safetensors"): sources[
            "train_memory_tensor_sha256"
        ],
        str(Path(sources["train_memory_cache"]) / "metadata.json"): sources[
            "train_memory_metadata_sha256"
        ],
        str(Path(sources["base_checkpoint"]) / "adapter.safetensors"): sources[
            "base_adapter_sha256"
        ],
        str(Path(sources["base_checkpoint"]) / "runtime_metadata.json"): sources[
            "base_runtime_metadata_sha256"
        ],
        sources["training_source"]: sources["training_source_sha256"],
    }
    observed: dict[str, str] = {}
    for path, expected_sha in expected.items():
        actual = sha256_file_v84(path)
        if actual != expected_sha:
            raise ValueError(f"V84 pair-margin pinned source changed: {path}")
        observed[str(path)] = actual
    return observed


def authenticate_pair_margin_preregistration_v84(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = _resolve(config["outputs"]["preregistration"])
    prereg = _strict_json(path)
    config_sha = sha256_file_v84(config_path)
    if (
        prereg.get("artifact") != PREREG_ARTIFACT
        or prereg.get("status") != "sealed_before_first_followup_model_measurement"
        or prereg.get("config_sha256") != config_sha
        or prereg.get("full_gemma_model_loaded") is not False
        or prereg.get("optimizer_constructed") is not False
        or prereg.get("optimizer_updates") != 0
        or prereg.get("development_behavior_scored") is not False
        or prereg.get("sealed_historical_16_loaded") is not False
        or prereg.get("official_validation_loaded") is not False
        or prereg.get("official_test_loaded") is not False
        or prereg.get("oracle_loaded") is not False
        or prereg.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V84 pair-margin preregistration changed")
    return {
        "config_sha256": config_sha,
        "preregistration_sha256": sha256_file_v84(path),
    }


def canonical_support_relation_v84(value: str) -> str:
    tokens = normalize_answer(value).split()
    if "under" in tokens or "beneath" in tokens or "below" in tokens:
        return "under"
    if "on" in tokens or "atop" in tokens:
        return "on"
    return "unknown"


def select_pair_margin_rows_v84(
    config: Mapping[str, Any],
) -> tuple[RowV73, RowV73]:
    rows = load_training_rows_v73(config["sources"]["historical_qa"])
    train, _development = split_rows_v73(rows)
    units = sorted(
        changed_units_v73(train),
        key=lambda unit: (unit.change_type, unit.pair_id, unit.question_key),
    )
    unit = units[0]
    selected = [[unit.left.scene_id, unit.left.question_id], [unit.right.scene_id, unit.right.question_id]]
    if (
        unit.question_key != config["training"]["selected_question_key"]
        or selected != config["training"]["selected_rows"]
        or [unit.left.answer, unit.right.answer] != config["training"]["answers"]
        or unit.left.question != unit.right.question
    ):
        raise ValueError("V84 pair-margin fixed unit selection changed")
    return unit.left, unit.right


def pair_margin_objective_v84(
    correct_nll: torch.Tensor,
    paired_wrong_nll: torch.Tensor,
    *,
    target_margin: float,
    ce_weight: float,
    margin_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if correct_nll.ndim != 0 or paired_wrong_nll.ndim != 0:
        raise ValueError("V84 pair-margin NLL inputs must be scalar")
    if not torch.isfinite(correct_nll) or not torch.isfinite(paired_wrong_nll):
        raise ValueError("V84 pair-margin NLL inputs must be finite")
    observed_margin = paired_wrong_nll - correct_nll
    penalty = torch.relu(
        correct_nll
        - paired_wrong_nll
        + torch.as_tensor(target_margin, device=correct_nll.device)
    )
    objective = ce_weight * correct_nll + margin_weight * penalty
    return objective, observed_margin, penalty


def _atomic_create_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V84 pair-margin create-once output exists: {destination}")
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


def _publish_pair_margin_candidate_v84(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    root = _resolve(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V84 pair-margin candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        fresh = collection.bank(FRESH_BANK_NAME).installation
        state = {
            "lora_a": fresh.adapters[0].lora_a.detach().cpu().contiguous(),
            "lora_b": fresh.adapters[0].lora_b.detach().cpu().contiguous(),
        }
        weights = temporary / "bridge.safetensors"
        save_file(
            state,
            str(weights),
            metadata={
                "artifact": "gemma4_v84_pair_margin_candidate_v1",
                "environmental_memory_serialized": "false",
                "questions_or_answers_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        metadata = {
            "artifact": "gemma4_v84_pair_margin_candidate_v1",
            "schema_version": "84.1",
            "status": "train_only_candidate_not_runtime_promoted",
            "bank_name": FRESH_BANK_NAME,
            "target_module": "model.language_model.layers.34.mlp.down_proj",
            "rank": 4,
            "alpha": 8.0,
            "parameter_count": fresh.parameter_count,
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v84(weights),
            "environmental_memory_serialized": False,
            "questions_or_answers_serialized": False,
            "oracle_serialized": False,
            "runtime_promotion_authorized": False,
            "bindings": dict(bindings),
        }
        (temporary / "runtime_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, root)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _evaluate_rows_v84(
    language: Any,
    system_prompt: str,
    memory_by_scene: Mapping[str, torch.Tensor],
    rows: Sequence[Any],
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        correct, layout = _measure_nll_v84(
            language, system_prompt, memory_by_scene[row.scene_id], row
        )
        wrong, _wrong_layout = _measure_nll_v84(
            language, system_prompt, memory_by_scene[row.paired_scene_id], row
        )
        raw_prediction = _generate_v84(
            language,
            system_prompt,
            memory_by_scene[row.scene_id],
            row,
            max_new_tokens=max_new_tokens,
        )
        canonical = canonical_support_relation_v84(raw_prediction)
        result.append(
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "expected_canonical_relation": row.answer,
                "correct_scene": correct,
                "paired_wrong_scene": wrong,
                "wrong_minus_correct_nll": wrong["mean_nll"]
                - correct["mean_nll"],
                "raw_greedy_prediction": raw_prediction,
                "canonical_greedy_prediction": canonical,
                "canonical_greedy_exact": canonical == row.answer,
                "layout_audit": layout,
            }
        )
        if language.device.type == "mps":
            torch.mps.empty_cache()
    return result


def run_pair_margin_followup_v84(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    started = time.monotonic()
    config = load_pair_margin_config_v84(config_path)
    source_hashes = authenticate_pair_margin_sources_v84(config)
    prereg = authenticate_pair_margin_preregistration_v84(
        config, config_path=config_path
    )
    report_path = _resolve(config["outputs"]["report"])
    candidate_path = _resolve(config["outputs"]["candidate"])
    if report_path.exists() or candidate_path.exists():
        raise FileExistsError("V84 pair-margin create-once outputs already exist")

    rows = select_pair_margin_rows_v84(config)
    if [row.answer for row in rows] != config["training"]["answers"]:
        raise ValueError("V84 pair-margin selected answer inventory changed")
    # The complete immutable memories cross the environment boundary here,
    # before Gemma loading and before any question tokenization.
    cpu_memories, memory_hashes_before = _scene_memories_v84(config, rows)

    runtime = load_runtime_config(config["sources"]["runtime_config"])
    language_settings = runtime["language"]
    language = load_local_language_model(
        str(language_settings["model_id"]),
        str(language_settings["revision"]),
        str(language_settings["dtype"]),
        freeze=True,
        local_files_only=True,
        backend="gemma4",
        decoder_gradient_checkpointing=True,
    )
    if language.device.type != "mps":
        raise RuntimeError("V84 pair-margin follow-up requires local MPS")
    collection = install_lora_banks(
        language.model, combined_lora_settings_v84(runtime, config)
    )
    if not isinstance(collection, LoRABankCollection):
        raise TypeError("V84 pair-margin LoRA bank installation failed")
    frozen_source = load_frozen_v54_banks_v84(
        collection, config["sources"]["base_checkpoint"]
    )
    collection.assert_trainable_surface(language.model)
    fresh = collection.bank(FRESH_BANK_NAME).installation
    initial_state_sha256 = fresh.state_sha256()
    memory_by_scene = {
        scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
        for scene_id, memory in cpu_memories.items()
    }
    system_prompt = str(runtime["language"]["system_prompt"])
    max_new_tokens = int(runtime["language"]["max_answer_tokens"])

    language.decoder_module.eval()
    collection.eval()
    initial_rows = _evaluate_rows_v84(
        language,
        system_prompt,
        memory_by_scene,
        rows,
        max_new_tokens=max_new_tokens,
    )

    training = config["training"]
    parameters = collection.parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    target_margin = float(training["wrong_scene_target_margin_nll"])
    margin_weight = float(training["wrong_scene_margin_weight"])
    ce_weight = float(training["correct_scene_answer_ce_weight"])
    history: list[dict[str, Any]] = []
    language.decoder_module.train()
    collection.train()
    for update in range(1, int(training["optimizer_updates"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        update_rows: list[dict[str, float | bool | str]] = []
        for row in rows:
            correct_prepared, _layout = _prepared_v84(
                language, system_prompt, memory_by_scene[row.scene_id], row
            )
            wrong_prepared, _wrong_layout = _prepared_v84(
                language,
                system_prompt,
                memory_by_scene[row.paired_scene_id],
                row,
            )
            correct_tail = _answer_tail(language, correct_prepared)
            wrong_tail = _answer_tail(language, wrong_prepared)
            correct_nll = correct_tail.mean_nll.float()
            wrong_nll = wrong_tail.mean_nll.float()
            row_loss, observed_margin, penalty = pair_margin_objective_v84(
                correct_nll,
                wrong_nll,
                target_margin=target_margin,
                ce_weight=ce_weight,
                margin_weight=margin_weight,
            )
            if not torch.isfinite(row_loss):
                raise RuntimeError("V84 pair-margin row loss is nonfinite")
            update_rows.append(
                {
                    "scene_id": row.scene_id,
                    "correct_scene_nll": float(correct_nll.detach().cpu()),
                    "paired_wrong_scene_nll": float(wrong_nll.detach().cpu()),
                    "wrong_minus_correct_nll": float(observed_margin.detach().cpu()),
                    "margin_penalty": float(penalty.detach().cpu()),
                    "margin_active": bool(float(penalty.detach().cpu()) > 0.0),
                    "row_objective": float(row_loss.detach().cpu()),
                }
            )
            (row_loss / len(rows)).backward()
            del correct_tail, wrong_tail, correct_prepared, wrong_prepared
            del correct_nll, wrong_nll, observed_margin, penalty, row_loss
        gradients = collection.gradient_norms()
        gradient_l2 = float(gradients["total_l2"])
        if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
            raise RuntimeError("V84 pair-margin gradient is zero or nonfinite")
        clip_return = torch.nn.utils.clip_grad_norm_(
            parameters, float(training["gradient_clip_norm"])
        )
        clip_l2 = float(clip_return.detach().cpu())
        if not math.isfinite(clip_l2):
            raise RuntimeError("V84 pair-margin clipped gradient is nonfinite")
        optimizer.step()
        collection.validate_state()
        history.append(
            {
                "update": update,
                "rows": update_rows,
                "mean_objective": sum(
                    float(value["row_objective"]) for value in update_rows
                )
                / len(update_rows),
                "gradient_l2_before_clip": gradient_l2,
                "clip_return_l2": clip_l2,
                "state_sha256": fresh.state_sha256(),
            }
        )
        if language.device.type == "mps":
            torch.mps.empty_cache()

    language.decoder_module.eval()
    collection.eval()
    final_rows = _evaluate_rows_v84(
        language,
        system_prompt,
        memory_by_scene,
        rows,
        max_new_tokens=max_new_tokens,
    )
    memory_hashes_after = {
        scene_id: prefix_sha256(memory.detach().cpu())
        for scene_id, memory in memory_by_scene.items()
    }
    initial_mean_nll = sum(
        row["correct_scene"]["mean_nll"] for row in initial_rows
    ) / len(initial_rows)
    final_mean_nll = sum(
        row["correct_scene"]["mean_nll"] for row in final_rows
    ) / len(final_rows)
    gates = {
        "both_final_canonical_greedy_answers_exact": all(
            row["canonical_greedy_exact"] for row in final_rows
        ),
        "final_canonical_greedy_answers_separate": len(
            {row["canonical_greedy_prediction"] for row in final_rows}
        )
        == len(final_rows),
        "both_final_wrong_minus_correct_nll_strictly_positive": all(
            row["wrong_minus_correct_nll"] > 0.0 for row in final_rows
        ),
        "final_mean_correct_scene_nll_below_initial": final_mean_nll
        < initial_mean_nll,
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(row["gradient_l2_before_clip"])
            and row["gradient_l2_before_clip"] > 0.0
            for row in history
        ),
        "memory_hash_invariant": memory_hashes_after == memory_hashes_before,
    }
    passed = all(gates.values())
    candidate_metadata = _publish_pair_margin_candidate_v84(
        candidate_path,
        collection,
        bindings={
            **prereg,
            "training_source_sha256": source_hashes[
                str(config["sources"]["training_source"])
            ],
            "base_adapter_sha256": frozen_source["adapter_sha256"],
            "fixed_final_optimizer_updates": len(history),
        },
    )
    report = {
        "artifact": REPORT_ARTIFACT,
        "schema_version": "84.1",
        "status": "passed_behavioral_wiring_gate"
        if passed
        else "failed_behavioral_wiring_gate",
        "device": str(language.device),
        "elapsed_seconds": time.monotonic() - started,
        "strict_input_contract": config["strict_input_contract"],
        "source_hashes": source_hashes,
        "preregistration": prereg,
        "trainable_bridge": {
            "bank_name": FRESH_BANK_NAME,
            "target_module": "model.language_model.layers.34.mlp.down_proj",
            "parameter_count": fresh.parameter_count,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": fresh.state_sha256(),
            "starts_from_parent_candidate": False,
            "unmerged": True,
        },
        "objective": {
            "correct_scene_answer_ce_weight": ce_weight,
            "paired_wrong_scene_margin_weight": margin_weight,
            "target_wrong_minus_correct_nll": target_margin,
        },
        "scene_memories": {
            "compiled_before_question_tokenization": True,
            "shape_each": [1, 738, 1536],
            "hashes_before": memory_hashes_before,
            "hashes_after": memory_hashes_after,
            "hash_invariant": memory_hashes_after == memory_hashes_before,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_retrieval": False,
        },
        "initial_rows": initial_rows,
        "initial_mean_correct_scene_nll": initial_mean_nll,
        "training_history": history,
        "optimizer_updates": len(history),
        "checkpoint_selection": "fixed_final_update_32",
        "intermediate_selection": False,
        "final_rows": final_rows,
        "final_mean_correct_scene_nll": final_mean_nll,
        "mean_correct_scene_nll_delta": final_mean_nll - initial_mean_nll,
        "gates": gates,
        "passed": passed,
        "candidate": {
            "path": candidate_path.relative_to(PROJECT_ROOT).as_posix(),
            "weights_sha256": candidate_metadata["weights_sha256"],
            "runtime_promotion_authorized": False,
        },
        "optimization_scene_count": 2,
        "development_behavior_scored": False,
        "sealed_historical_16_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    _atomic_create_json(report_path, report)
    return report


def _answer_tail(language: Any, prepared: Any) -> Any:
    # Local import keeps this module's config/preflight helpers CPU-light.
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    report = run_pair_margin_followup_v84(args.config)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
