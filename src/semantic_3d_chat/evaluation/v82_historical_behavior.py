"""Real local-Gemma behavior smoke for the learned V82 sealed-memory reader.

The predictor opens only answer-free questions.  Every held scene memory is
compiled and hash-bound before that question manifest is opened.  Correct,
paired-wrong, zero-payload, and shuffled-atlas arms all use the same frozen V82
reader and local Gemma generation path.  Answer-bearing references are opened
only by the separate model-free scorer.
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
    INPUT_EMBEDDING_TENSOR_NAME,
    MODEL_BLOB_SHA256_IDENTITY,
    bind_fixed_prefix_before_question_v81,
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
from semantic_3d_chat.training.v82_reader_artifacts import load_v82_candidate

ARTIFACT: Final[str] = "v82_historical_internal_predictions_v1"
SCORE_ARTIFACT: Final[str] = "v82_historical_internal_score_v1"
DEFAULT_RUNTIME_CONFIG: Final[Path] = Path("configs/runtime/gemma4_v54.yaml")
DEFAULT_BASE_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
)
DEFAULT_CONTROL_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
)
DEFAULT_READER_CHECKPOINT: Final[Path] = Path(
    "reports/gemma4/artifacts/v82_strict_dense_reader/candidate"
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
    "reports/gemma4/predictions/v82_historical_internal.json"
)
DEFAULT_SCORE: Final[Path] = Path(
    "reports/gemma4/metrics/v82_historical_internal_score.json"
)
ARMS: Final[tuple[str, ...]] = (
    "v82",
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
) -> torch.Tensor:
    banks = split_v75_v2_prefix_v81(source)
    values = banks.atlas_values if atlas_values is None else atlas_values
    memory = torch.cat((banks.probe_keys.unsqueeze(2), values), dim=2).reshape(
        source.shape[0], ATLAS_MEMORY_TOKENS, HIDDEN_SIZE
    )
    base = torch.zeros_like(banks.base_latents) if zero_base_latents else banks.base_latents
    return torch.cat((banks.boi, memory, base, banks.eoi), dim=1).detach()


def _controls(
    runtime: StaticChatRuntime,
    reader: Any,
    fixed: torch.Tensor,
    question: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    query = latest_user_question_query_v81(
        tokenizer=runtime.language.tokenizer,
        embedding_layer=runtime.language.model.get_input_embeddings(),
        latest_user_question=question,
        device=runtime.language.device,
        maximum_question_tokens=int(runtime.config["language"]["max_question_tokens"]),
        model_blob_sha256_identity=MODEL_BLOB_SHA256_IDENTITY,
        embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
    )
    output = reader(
        fixed,
        query.query,
        binding=bind_fixed_prefix_before_question_v81(fixed),
    )
    return output.controls, {
        "minimum_atlas_attention_weight": float(output.atlas_weights.min().cpu()),
        "minimum_base_attention_weight": float(output.base_weights.min().cpu()),
        "maximum_control_rms": float(output.control_rms.max().cpu()),
        "all_384_atlas_values_positive": output.all_384_atlas_values_positive,
        "all_256_base_latents_positive": output.all_256_base_latents_positive,
        "latest_user_token_count": query.token_count,
        "latest_user_only": True,
    }


def predict(
    *,
    runtime_config: str | Path = DEFAULT_RUNTIME_CONFIG,
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    control_checkpoint: str | Path = DEFAULT_CONTROL_CHECKPOINT,
    reader_checkpoint: str | Path = DEFAULT_READER_CHECKPOINT,
    prefix_cache: str | Path = DEFAULT_PREFIX_CACHE,
    probe_bank: str | Path = DEFAULT_PROBE_BANK,
    questions_root: str | Path = DEFAULT_QUESTIONS,
    references_forbidden_root: str | Path = DEFAULT_REFERENCES,
    output_path: str | Path = DEFAULT_PREDICTIONS,
) -> dict[str, Any]:
    output_path = _resolve(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    scorer_root = _resolve(references_forbidden_root)
    audit = _runtime_audit(scorer_root)
    train_cache = _resolve(
        "reports/gemma4/artifacts/v82_strict_dense_reader/train_cache"
    )
    development_cache = _resolve(
        "reports/gemma4/artifacts/v82_strict_dense_reader/development_cache"
    )
    audit.forbidden_roots.extend([train_cache, development_cache])
    started = time.perf_counter()
    with audit:
        config_path = _guard_regular(_resolve(runtime_config), "V82 runtime config")
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
            raise TypeError("V82 behavior compiler requires the sealed V75 controller")
        loaded_reader = load_v82_candidate(
            _resolve(reader_checkpoint), device="cpu", record_file=audit.record
        )
        reader = loaded_reader.model

        fixed_memories: dict[str, torch.Tensor] = {}
        shuffled_memories: dict[str, torch.Tensor] = {}
        zero_memories: dict[str, torch.Tensor] = {}
        tensor_hashes: dict[str, str] = {}
        for scene_id in SCENE_IDS:
            compiled = compile_fixed_scene_atlas_v75_v2(
                base_prefixes[scene_id], controller, probes
            )
            fixed = compiled.scene_prefix.detach().cpu().contiguous()
            banks = split_v75_v2_prefix_v81(fixed)
            fixed_memories[scene_id] = fixed
            shuffled_memories[scene_id] = _fixed_from_banks(
                fixed, atlas_values=banks.atlas_values.roll(shifts=1, dims=1)
            )
            zero_memories[scene_id] = _fixed_from_banks(
                fixed,
                atlas_values=torch.zeros_like(banks.atlas_values),
                zero_base_latents=True,
            )
            tensor_hashes[scene_id] = tensor_sha256(fixed)
        hashes_before = {
            scene_id: prefix_sha256(value) for scene_id, value in fixed_memories.items()
        }
        del controller, probes

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
            raise RuntimeError("V82 behavior smoke loaded an unexpected local model")
        device = runtime.language.device
        dtype = next(runtime.language.model.parameters()).dtype
        reader.to(device=device, dtype=torch.float32).eval()
        rows, question_metadata = _load_predictor_questions(
            _resolve(questions_root), audit
        )
        row_by_scene = {str(row["scene_id"]): row for row in rows}
        if set(row_by_scene) != set(SCENE_IDS):
            raise ValueError("V82 behavior questions lost one-row-per-scene layout")

        records: list[dict[str, Any]] = []
        for ordinal, row in enumerate(rows, 1):
            scene_id = str(row["scene_id"])
            question = str(row["question"])
            fixed = fixed_memories[scene_id].to(device=device, dtype=dtype)
            wrong_id = next(
                candidate
                for candidate in SCENE_IDS
                if candidate != scene_id
                and row_by_scene[candidate]["question"] == question
            )
            wrong = fixed_memories[wrong_id].to(device=device, dtype=dtype)
            zero = zero_memories[scene_id].to(device=device, dtype=dtype)
            shuffled = shuffled_memories[scene_id].to(device=device, dtype=dtype)
            row_started = time.perf_counter()
            with torch.inference_mode():
                primary_control, reader_audit = _controls(
                    runtime, reader, fixed, question
                )
                wrong_control, _ = _controls(runtime, reader, wrong, question)
                zero_control, zero_audit = _controls(runtime, reader, zero, question)
                shuffled_control, shuffled_audit = _controls(
                    runtime, reader, shuffled, question
                )
                primary = _generate(
                    runtime,
                    reconstruct_base_v54_prefix_v81(fixed),
                    question,
                    primary_control.to(fixed),
                )
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
                frozen_prediction = _generate(
                    runtime,
                    reconstruct_base_v54_prefix_v81(fixed),
                    question,
                    None,
                )
            zero_max = float(zero_control.abs().max().cpu())
            if zero_max != 0.0:
                raise RuntimeError("V82 zero environment did not produce exact-zero controls")
            record = {
                "row_id": row["row_id"],
                "scene_id": scene_id,
                "wrong_scene_id": wrong_id,
                "fixed_memory_sha256": hashes_before[scene_id],
                "fixed_memory_tensor_sha256": tensor_hashes[scene_id],
                "v82_prediction": primary,
                "wrong_scene_prediction": wrong_prediction,
                "zero_environment_prediction": zero_prediction,
                "shuffled_atlas_prediction": shuffled_prediction,
                "frozen_v54_prediction": frozen_prediction,
                "reader_audit": reader_audit,
                "zero_reader_audit": zero_audit,
                "shuffled_reader_audit": shuffled_audit,
                "zero_environment_exact_zero_controls": True,
                "zero_environment_max_control_abs": zero_max,
                "elapsed_seconds": time.perf_counter() - row_started,
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "event": "v82_historical_behavior_row",
                        "ordinal": ordinal,
                        "total": ROW_COUNT,
                        "row_id": row["row_id"],
                        "scene_id": scene_id,
                        "v82": primary,
                        "wrong_scene": wrong_prediction,
                        "zero_environment": zero_prediction,
                        "shuffled_atlas": shuffled_prediction,
                        "frozen_v54": frozen_prediction,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        hashes_after = {
            scene_id: prefix_sha256(value) for scene_id, value in fixed_memories.items()
        }
        invariant = hashes_before == hashes_after
    audit.assert_clean()
    if any(Path(path).is_relative_to(scorer_root) for path in audit.unique_paths):
        raise RuntimeError("V82 predictor opened answer-bearing scorer data")
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
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
        },
        "reader": {
            "architecture": "positive_floor_dual_bank_reader_v82",
            "weights_sha256": loaded_reader.metadata["weights_sha256"],
            "all_384_atlas_values_and_256_base_latents_positive_floor": True,
            "boi_eoi_and_96_probe_keys_are_not_payload": True,
            "question_dependent_scene_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
        },
        "source": {
            "probe_tensor_sha256": probe_metadata["probe_tensor_sha256"],
            "controller_weights_sha256": controller_metadata["weights_sha256"],
            "prefix_cache_manifest_sha256": PREFIX_MANIFEST_SHA256,
            "prefix_cache_base_checkpoint_sha256": prefix_manifest[
                "base_checkpoint_sha256"
            ],
            "question_file_sha256": question_metadata["questions_file_sha256"],
        },
        "memory": {
            "fixed_tokens": 738,
            "base_tokens_supplied_to_gemma": 258,
            "reader_activation_tokens": 4,
            "all_memories_compiled_before_question_manifest_opened": True,
            "fixed_hashes_before": hashes_before,
            "fixed_hashes_after": hashes_after,
            "fixed_tensor_hashes": tensor_hashes,
            "fixed_memory_invariant": invariant,
            "same_memory_reused_for_every_question": True,
        },
        "controls": {
            "paired_wrong_scene": True,
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
            "training_or_development_cache_loaded": False,
            "environmental_text_inputs": [],
        },
        "behavioral_accuracy_scored_in_predictor": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    _atomic_write_json(output_path, payload)
    return payload


def score(
    predictions_path: str | Path = DEFAULT_PREDICTIONS,
    references_root: str | Path = DEFAULT_REFERENCES,
    output_path: str | Path = DEFAULT_SCORE,
) -> dict[str, Any]:
    from semantic_3d_chat.evaluation.v55_development_score import (
        canonical_type_specific_match,
    )

    prediction_path = _guard_regular(_resolve(predictions_path), "V82 predictions")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if (
        not isinstance(predictions, Mapping)
        or predictions.get("artifact") != ARTIFACT
        or predictions.get("execution_valid") is not True
        or predictions.get("row_count") != ROW_COUNT
        or predictions.get("behavioral_accuracy_scored_in_predictor") is not False
    ):
        raise ValueError("V82 prediction artifact contract changed")
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
        or leakage.get("training_or_development_cache_loaded") is not False
        or not isinstance(controls, Mapping)
        or controls.get("zero_environment_controls_exact_zero") is not True
    ):
        raise ValueError("V82 prediction structural/control evidence failed")
    records = predictions.get("records")
    if not isinstance(records, list) or len(records) != ROW_COUNT:
        raise ValueError("V82 prediction rows changed")
    references, reference_metadata = _load_reference_artifact(_resolve(references_root))
    if {record.get("row_id") for record in records} != set(references):
        raise ValueError("V82 prediction/reference row IDs differ")
    arm_fields = {
        "v82": "v82_prediction",
        "wrong_scene": "wrong_scene_prediction",
        "zero_environment": "zero_environment_prediction",
        "shuffled_atlas": "shuffled_atlas_prediction",
        "frozen_v54": "frozen_v54_prediction",
    }
    joined: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("V82 prediction row must be a mapping")
        reference = references[str(record["row_id"])]
        scored = {**record, **reference}
        for arm, field in arm_fields.items():
            scored[f"{arm}_correct"] = canonical_type_specific_match(
                reference["answer_type"], str(record[field]), reference["answer"]
            )
        joined.append(scored)
    scores = {arm: _aggregate_scored(joined, f"{arm}_correct") for arm in ARMS}
    changes = {
        arm: _prediction_change_units(joined, field)
        for arm, field in arm_fields.items()
    }
    candidate_correct = int(scores["v82"]["correct"])
    wrong_correct = int(scores["wrong_scene"]["correct"])
    frozen_correct = int(scores["frozen_v54"]["correct"])
    gates = {
        "candidate_correct_at_least_9": candidate_correct >= 9,
        "gain_over_frozen_v54_at_least_3": candidate_correct - frozen_correct >= 3,
        "correct_minus_wrong_scene_at_least_2": candidate_correct - wrong_correct >= 2,
        "prediction_changing_units_at_least_2": changes["v82"] >= 2,
        "exact_zero_environment_controls": True,
        "fixed_memory_invariant": True,
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
        "arms": scores,
        "accuracy_deltas": {
            "v82_minus_frozen_v54": scores["v82"]["accuracy"]
            - scores["frozen_v54"]["accuracy"],
            "v82_minus_wrong_scene": scores["v82"]["accuracy"]
            - scores["wrong_scene"]["accuracy"],
            "v82_minus_zero_environment": scores["v82"]["accuracy"]
            - scores["zero_environment"]["accuracy"],
            "v82_minus_shuffled_atlas": scores["v82"]["accuracy"]
            - scores["shuffled_atlas"]["accuracy"],
        },
        "direct_v75_historical_comparator": {"correct": 9, "total": ROW_COUNT},
        "v81_historical_comparator": {"correct": 8, "total": ROW_COUNT},
        "prediction_change_units": {**changes, "total": 8},
        "change_family_counts": dict(
            Counter(str(record["change_type"]) for record in joined)
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "behavioral_accuracy_measured": True,
        "protected_evaluation_authorized": False,
        "runtime_promotion_authorized": all(gates.values()),
    }
    _atomic_write_json(_resolve(output_path), result)
    return result


def predict_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", default=str(DEFAULT_READER_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_PREDICTIONS))
    args = parser.parse_args(argv)
    value = predict(reader_checkpoint=args.reader, output_path=args.output)
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


if __name__ == "__main__":
    raise SystemExit(predict_main())


__all__ = ["predict", "predict_main", "score", "score_main"]
