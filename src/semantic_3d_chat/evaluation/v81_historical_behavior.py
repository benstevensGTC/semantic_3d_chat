"""No-grad behavioral smoke and memory controls for the V81 dense reader.

All sixteen fixed memories are compiled before the answer-free question file
is opened.  The predictor cannot read the physically separate scorer data.
Scoring is a second model-free process and never loads Gemma or scene memory.
This is historical development evidence, not an untouched final-test claim.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.evaluation.v75_fixed_atlas_behavior import (
    GEMMA_REVISION,
    HIDDEN_SIZE,
    PREFIX_MANIFEST_SHA256,
    ROW_COUNT,
    SCENE_IDS,
    _aggregate_scored,
    _atomic_write_json,
    _disable_decoder_checkpointing,
    _generate,
    _guard_regular,
    _load_base_prefixes,
    _load_predictor_questions,
    _load_probe_bank,
    _load_reference_artifact,
    _prediction_change_units,
    _resolve,
    _runtime_audit,
    _sha256_file,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_MEMORY_TOKENS,
    ATLAS_UNIFORM_FLOOR_MASS,
    INPUT_EMBEDDING_TENSOR_NAME,
    MAXIMUM_CONTROL_RMS,
    MODEL_BLOB_SHA256_IDENTITY,
    RAW_ATLAS_LOGIT_SCALE,
    bind_fixed_prefix_before_question_v81,
    deterministic_atlas_read_v81,
    latest_user_question_query_v81,
    reconstruct_base_v54_prefix_v81,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)

ARTIFACT: Final[str] = "v81_historical_internal_predictions_v1"
SCORE_ARTIFACT: Final[str] = "v81_historical_internal_score_v1"
DEFAULT_RUNTIME_CONFIG: Final[Path] = Path("configs/runtime/gemma4_v54.yaml")
DEFAULT_BASE_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
)
DEFAULT_CONTROL_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
)
DEFAULT_PREFIX_CACHE: Final[Path] = Path(
    "data_gemma4/scene_tokens/v56_question_control_full_prefixes"
)
DEFAULT_PROBE_BANK: Final[Path] = Path(
    "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank"
)
DEFAULT_QUESTIONS: Final[Path] = Path(
    "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/predictor"
)
DEFAULT_REFERENCES: Final[Path] = Path(
    "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/scorer"
)
DEFAULT_PREDICTIONS: Final[Path] = Path(
    "reports/gemma4/predictions/v81_historical_internal.json"
)
DEFAULT_SCORE: Final[Path] = Path(
    "reports/gemma4/metrics/v81_historical_internal_score.json"
)
ARMS: Final[tuple[str, ...]] = (
    "v81",
    "wrong_scene",
    "zero_environment",
    "shuffled_atlas",
    "frozen_v54",
)


def _fixed_from_banks(
    source: torch.Tensor,
    *,
    atlas_values: torch.Tensor | None = None,
    zero_base_latents: bool = False,
    zero_keys: bool = False,
) -> torch.Tensor:
    banks = split_v75_v2_prefix_v81(source)
    keys = torch.zeros_like(banks.probe_keys) if zero_keys else banks.probe_keys
    values = banks.atlas_values if atlas_values is None else atlas_values
    base = torch.zeros_like(banks.base_latents) if zero_base_latents else banks.base_latents
    memory = torch.cat((keys.unsqueeze(2), values), dim=2).reshape(
        source.shape[0], ATLAS_MEMORY_TOKENS, HIDDEN_SIZE
    )
    return torch.cat((banks.boi, memory, base, banks.eoi), dim=1).detach()


def _controls(
    runtime: StaticChatRuntime,
    fixed: torch.Tensor,
    question: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    binding = bind_fixed_prefix_before_question_v81(fixed)
    query = latest_user_question_query_v81(
        tokenizer=runtime.language.tokenizer,
        embedding_layer=runtime.language.model.get_input_embeddings(),
        latest_user_question=question,
        device=runtime.language.device,
        maximum_question_tokens=int(runtime.config["language"]["max_question_tokens"]),
        model_blob_sha256_identity=MODEL_BLOB_SHA256_IDENTITY,
        embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
    )
    output = deterministic_atlas_read_v81(
        fixed,
        query.query,
        binding=binding,
    )
    return output.reconstructed_controls, {
        "minimum_attention_weight": float(output.atlas_weights.min().cpu()),
        "maximum_control_rms": float(output.control_rms.max().cpu()),
        "attention_sum": float(output.attention_sums.max().cpu()),
        "all_96_groups_positive": output.all_96_groups_positive,
        "all_384_values_receive_positive_floor_weight": (
            output.all_384_values_receive_positive_floor_weight
        ),
        "latest_user_token_count": query.token_count,
        "latest_user_only": True,
    }


def _source_provenance(
    *,
    probe_metadata: Mapping[str, Any],
    controller_metadata: Mapping[str, Any],
    prefix_manifest: Mapping[str, Any],
    question_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Name the prefix manifest and its base-checkpoint identities precisely."""

    return {
        "probe_tensor_sha256": probe_metadata["probe_tensor_sha256"],
        "controller_architecture": controller_metadata["architecture"],
        "controller_weights_sha256": controller_metadata["weights_sha256"],
        "prefix_cache_manifest_sha256": PREFIX_MANIFEST_SHA256,
        "prefix_cache_base_checkpoint_sha256": prefix_manifest[
            "base_checkpoint_sha256"
        ],
        "question_file_sha256": question_metadata["questions_file_sha256"],
    }


def predict(
    *,
    runtime_config: str | Path = DEFAULT_RUNTIME_CONFIG,
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    control_checkpoint: str | Path = DEFAULT_CONTROL_CHECKPOINT,
    prefix_cache: str | Path = DEFAULT_PREFIX_CACHE,
    probe_bank: str | Path = DEFAULT_PROBE_BANK,
    questions_root: str | Path = DEFAULT_QUESTIONS,
    references_forbidden_root: str | Path = DEFAULT_REFERENCES,
    output_path: str | Path = DEFAULT_PREDICTIONS,
) -> dict[str, Any]:
    """Run one create-once no-grad V81 smoke plus three memory controls."""

    output = _resolve(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    scorer_root = _resolve(references_forbidden_root)
    audit = _runtime_audit(scorer_root)
    started = time.perf_counter()
    with audit:
        config_path = _guard_regular(_resolve(runtime_config), "V81 runtime config")
        audit.record(config_path)
        config = load_runtime_config(config_path)
        probes, probe_metadata = _load_probe_bank(_resolve(probe_bank), audit)
        base_prefixes, prefix_manifest = _load_base_prefixes(
            _resolve(prefix_cache), SCENE_IDS, audit
        )
        controller, controller_metadata = _load_control_head(
            _resolve(control_checkpoint),
            hidden_size=HIDDEN_SIZE,
            device=torch.device("cpu"),
            audit=audit,
        )
        if type(controller) is not DenseFullSceneContinuousControlV75:
            raise TypeError("V81 behavior compiler requires the sealed V75 controller")

        fixed_memories: dict[str, torch.Tensor] = {}
        base_prefixes_from_memory: dict[str, torch.Tensor] = {}
        shuffled_memories: dict[str, torch.Tensor] = {}
        zero_memories: dict[str, torch.Tensor] = {}
        fixed_tensor_hashes: dict[str, str] = {}
        for scene_id in SCENE_IDS:
            compiled = compile_fixed_scene_atlas_v75_v2(
                base_prefixes[scene_id], controller, probes
            )
            fixed = compiled.scene_prefix.detach().cpu().contiguous()
            banks = split_v75_v2_prefix_v81(fixed)
            fixed_memories[scene_id] = fixed
            base_prefixes_from_memory[scene_id] = reconstruct_base_v54_prefix_v81(fixed)
            shuffled_memories[scene_id] = _fixed_from_banks(
                fixed,
                atlas_values=banks.atlas_values.roll(shifts=1, dims=1),
            )
            zero_memories[scene_id] = _fixed_from_banks(
                fixed,
                atlas_values=torch.zeros_like(banks.atlas_values),
                zero_base_latents=True,
                zero_keys=True,
            )
            fixed_tensor_hashes[scene_id] = tensor_sha256(fixed)
        fixed_hashes_before = {
            scene_id: prefix_sha256(value) for scene_id, value in fixed_memories.items()
        }
        all_fixed_memories_compiled_before_question_manifest = True
        # Compiler objects are no longer necessary after the immutable numeric
        # memories exist.  Deleting them makes the later dependency explicit.
        del controller, probes
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        runtime = StaticChatRuntime.load(
            config,
            "scene_000011",
            checkpoint=_resolve(base_checkpoint),
            audit=audit,
            local_files_only=True,
        )
        _disable_decoder_checkpointing(runtime.language)
        if (
            runtime.language.backend_name != "gemma4"
            or getattr(runtime.language.prefix_backend, "model_revision", None)
            != GEMMA_REVISION
        ):
            raise RuntimeError("V81 behavior smoke loaded an unexpected local model")
        device = runtime.language.device
        model_dtype = next(runtime.language.model.parameters()).dtype

        # The answer-free question projection is intentionally opened only
        # after every environmental memory has been compiled and hash-bound.
        rows, question_metadata = _load_predictor_questions(
            _resolve(questions_root), audit
        )
        wrong_scene = {
            scene_id: SCENE_IDS[(index + 1) % len(SCENE_IDS)]
            for index, scene_id in enumerate(SCENE_IDS)
        }
        records: list[dict[str, Any]] = []
        for ordinal, row in enumerate(rows, 1):
            scene_id = row["scene_id"]
            question = row["question"]
            fixed = fixed_memories[scene_id].to(device=device, dtype=model_dtype)
            base = base_prefixes_from_memory[scene_id].to(
                device=device, dtype=model_dtype
            )
            shuffled = shuffled_memories[scene_id].to(
                device=device, dtype=model_dtype
            )
            zero = zero_memories[scene_id].to(device=device, dtype=model_dtype)
            wrong_id = wrong_scene[scene_id]
            wrong = fixed_memories[wrong_id].to(device=device, dtype=model_dtype)
            row_started = time.perf_counter()
            with torch.inference_mode():
                primary_control, reader_audit = _controls(runtime, fixed, question)
                wrong_control, _wrong_audit = _controls(runtime, wrong, question)
                zero_control, zero_audit = _controls(runtime, zero, question)
                shuffled_control, shuffled_audit = _controls(runtime, shuffled, question)
                primary = _generate(runtime, base, question, primary_control.to(base))
                wrong_prediction = _generate(
                    runtime,
                    reconstruct_base_v54_prefix_v81(wrong),
                    question,
                    wrong_control.to(wrong),
                )
                zero_prediction = _generate(
                    runtime,
                    reconstruct_base_v54_prefix_v81(zero),
                    question,
                    zero_control.to(zero),
                )
                shuffled_prediction = _generate(
                    runtime,
                    reconstruct_base_v54_prefix_v81(shuffled),
                    question,
                    shuffled_control.to(shuffled),
                )
                frozen_prediction = _generate(runtime, base, question, None)
            if float(zero_control.abs().max().cpu()) != 0.0:
                raise RuntimeError("V81 zero-environment control was not exactly zero")
            records.append(
                {
                    "row_id": row["row_id"],
                    "scene_id": scene_id,
                    "wrong_scene_id": wrong_id,
                    "fixed_memory_sha256": fixed_hashes_before[scene_id],
                    "fixed_memory_tensor_sha256": fixed_tensor_hashes[scene_id],
                    "v81_prediction": primary,
                    "wrong_scene_prediction": wrong_prediction,
                    "zero_environment_prediction": zero_prediction,
                    "shuffled_atlas_prediction": shuffled_prediction,
                    "frozen_v54_prediction": frozen_prediction,
                    "reader_audit": reader_audit,
                    "zero_environment_exact_zero_controls": True,
                    "zero_environment_max_control_abs": float(
                        zero_control.abs().max().cpu()
                    ),
                    "zero_reader_audit": zero_audit,
                    "shuffled_reader_audit": shuffled_audit,
                    "elapsed_seconds": time.perf_counter() - row_started,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "v81_historical_behavior_row",
                        "ordinal": ordinal,
                        "total": ROW_COUNT,
                        "row_id": row["row_id"],
                        "scene_id": scene_id,
                        "v81": primary,
                        "wrong_scene": wrong_prediction,
                        "zero_environment": zero_prediction,
                        "shuffled_atlas": shuffled_prediction,
                        "frozen_v54": frozen_prediction,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        fixed_hashes_after = {
            scene_id: prefix_sha256(value) for scene_id, value in fixed_memories.items()
        }
        invariant = fixed_hashes_before == fixed_hashes_after and all(
            record["fixed_memory_sha256"]
            == fixed_hashes_before[str(record["scene_id"])]
            for record in records
        )
        if not invariant:
            raise RuntimeError("V81 fixed memory changed after user questions")
    audit.assert_clean()
    scorer_reads = [
        path for path in audit.unique_paths if Path(path).is_relative_to(scorer_root)
    ]
    if scorer_reads:
        raise RuntimeError("V81 predictor opened answer-bearing scorer data")
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "status": "historical_development_behavior_measured_not_final_test",
        "execution_valid": True,
        "row_count": len(records),
        "scene_count": len(SCENE_IDS),
        "arms": list(ARMS),
        "scope": {
            "historical_training_pool_only": True,
            "pair_disjoint": True,
            "scene_disjoint": True,
            "question_disjoint": False,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
        },
        "reader": {
            "architecture": "normalized_query_probe_cosine_dense_read_v81",
            "logit_scale": RAW_ATLAS_LOGIT_SCALE,
            "uniform_floor_mass": ATLAS_UNIFORM_FLOOR_MASS,
            "maximum_control_rms": MAXIMUM_CONTROL_RMS,
            "question_dependent_scene_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
        },
        "source": _source_provenance(
            probe_metadata=probe_metadata,
            controller_metadata=controller_metadata,
            prefix_manifest=prefix_manifest,
            question_metadata=question_metadata,
        ),
        "memory": {
            "fixed_tokens": 738,
            "base_tokens_supplied_to_gemma": 258,
            "reader_activation_tokens": 4,
            "all_memories_compiled_before_question_manifest_opened": (
                all_fixed_memories_compiled_before_question_manifest
            ),
            "fixed_hashes_before": fixed_hashes_before,
            "fixed_hashes_after": fixed_hashes_after,
            "fixed_tensor_hashes": fixed_tensor_hashes,
            "fixed_memory_invariant": invariant,
            "same_memory_reused_for_every_question": True,
        },
        "controls": {
            "wrong_scene": True,
            "zero_environment": True,
            "shuffled_atlas_values": True,
            "zero_environment_controls_exact_zero": all(
                record["zero_environment_exact_zero_controls"] for record in records
            ),
        },
        "leakage": {
            "loaded_file_count": len(audit.unique_paths),
            "loaded_files": audit.unique_paths,
            "forbidden_access_count": len(audit.forbidden_accesses()),
            "forbidden_accesses": audit.forbidden_accesses(),
            "scorer_reference_files_loaded": False,
            "environmental_text_inputs": [],
        },
        "behavioral_accuracy_scored_in_predictor": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    _atomic_write_json(output, payload)
    return payload


def score(
    predictions_path: str | Path = DEFAULT_PREDICTIONS,
    references_root: str | Path = DEFAULT_REFERENCES,
    output_path: str | Path = DEFAULT_SCORE,
) -> dict[str, Any]:
    """Score sealed V81 predictions in a model-free isolated process."""

    from semantic_3d_chat.evaluation.v55_development_score import (
        canonical_type_specific_match,
    )

    prediction_path = _guard_regular(_resolve(predictions_path), "V81 predictions")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if (
        not isinstance(predictions, Mapping)
        or predictions.get("artifact") != ARTIFACT
        or predictions.get("execution_valid") is not True
        or predictions.get("row_count") != ROW_COUNT
        or predictions.get("behavioral_accuracy_scored_in_predictor") is not False
    ):
        raise ValueError("V81 prediction artifact contract changed")
    memory = predictions.get("memory")
    leakage = predictions.get("leakage")
    controls = predictions.get("controls")
    if (
        not isinstance(memory, Mapping)
        or memory.get("fixed_memory_invariant") is not True
        or memory.get("all_memories_compiled_before_question_manifest_opened") is not True
        or not isinstance(leakage, Mapping)
        or leakage.get("forbidden_access_count") != 0
        or leakage.get("scorer_reference_files_loaded") is not False
        or not isinstance(controls, Mapping)
        or controls.get("zero_environment_controls_exact_zero") is not True
    ):
        raise ValueError("V81 prediction structural/control evidence failed")
    records = predictions.get("records")
    if not isinstance(records, list) or len(records) != ROW_COUNT:
        raise ValueError("V81 prediction rows changed")
    references, reference_metadata = _load_reference_artifact(
        _resolve(references_root)
    )
    if {record.get("row_id") for record in records} != set(references):
        raise ValueError("V81 prediction/reference row IDs differ")
    joined: list[dict[str, Any]] = []
    arm_fields = {
        "v81": "v81_prediction",
        "wrong_scene": "wrong_scene_prediction",
        "zero_environment": "zero_environment_prediction",
        "shuffled_atlas": "shuffled_atlas_prediction",
        "frozen_v54": "frozen_v54_prediction",
    }
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("V81 prediction row must be a mapping")
        reference = references[str(record["row_id"])]
        scored = {**record, **reference}
        for arm, field in arm_fields.items():
            scored[f"{arm}_correct"] = canonical_type_specific_match(
                reference["answer_type"],
                str(record[field]),
                reference["answer"],
            )
        joined.append(scored)
    arm_scores = {
        arm: _aggregate_scored(joined, f"{arm}_correct") for arm in ARMS
    }
    prediction_changes = {
        arm: _prediction_change_units(joined, field)
        for arm, field in arm_fields.items()
    }
    v81_correct = int(arm_scores["v81"]["correct"])
    wrong_correct = int(arm_scores["wrong_scene"]["correct"])
    frozen_correct = int(arm_scores["frozen_v54"]["correct"])
    direct_target = 9
    gates = {
        "candidate_correct_at_least_9": v81_correct >= 9,
        "gain_over_frozen_v54_at_least_3": v81_correct - frozen_correct >= 3,
        "correct_minus_wrong_scene_at_least_2": v81_correct - wrong_correct >= 2,
        "prediction_changing_units_at_least_2": prediction_changes["v81"] >= 2,
        "exact_zero_environment_controls": bool(
            predictions["controls"]["zero_environment_controls_exact_zero"]
        ),
        "fixed_memory_invariant": bool(memory["fixed_memory_invariant"]),
        "predictor_reference_isolation": True,
    }
    result: dict[str, Any] = {
        "artifact": SCORE_ARTIFACT,
        "status": (
            "historical_development_gate_pass"
            if all(gates.values())
            else "historical_development_gate_fail"
        ),
        "execution_valid": True,
        "scope": predictions["scope"],
        "prediction_artifact_sha256": _sha256_file(prediction_path),
        "reference_artifact_sha256": reference_metadata["references_file_sha256"],
        "arms": arm_scores,
        "accuracy_deltas": {
            "v81_minus_frozen_v54": (
                arm_scores["v81"]["accuracy"]
                - arm_scores["frozen_v54"]["accuracy"]
            ),
            "v81_minus_wrong_scene": (
                arm_scores["v81"]["accuracy"]
                - arm_scores["wrong_scene"]["accuracy"]
            ),
            "v81_minus_zero_environment": (
                arm_scores["v81"]["accuracy"]
                - arm_scores["zero_environment"]["accuracy"]
            ),
            "v81_minus_shuffled_atlas": (
                arm_scores["v81"]["accuracy"]
                - arm_scores["shuffled_atlas"]["accuracy"]
            ),
        },
        "direct_v75_historical_comparator": {
            "correct": direct_target,
            "total": ROW_COUNT,
            "source_score_sha256": (
                "224886019172c5080f2bd976de74477d40e37db9a5635aae9c9b7697db53dfd2"
            ),
            "v81_correct_gap": v81_correct - direct_target,
        },
        "prediction_change_units": {**prediction_changes, "total": 8},
        "change_family_counts": dict(
            Counter(str(record["change_type"]) for record in joined)
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "behavioral_accuracy_measured": True,
        "counterfactual_controls_measured": True,
        "protected_evaluation_authorized": False,
        "runtime_promotion_authorized": all(gates.values()),
    }
    _atomic_write_json(_resolve(output_path), result)
    return result


def predict_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_PREDICTIONS))
    args = parser.parse_args(argv)
    value = predict(output_path=args.output)
    print(
        json.dumps(
            {key: item for key, item in value.items() if key != "records"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def score_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--references", default=str(DEFAULT_REFERENCES))
    parser.add_argument("--output", default=str(DEFAULT_SCORE))
    args = parser.parse_args(argv)
    value = score(args.predictions, args.references, args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


__all__ = ["predict", "predict_main", "score", "score_main"]
