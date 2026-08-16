"""Sealed historical behavior test for direct 738-token V83 memory.

Every correct and control memory is compiled and hash-bound before the
answer-free question manifest is opened.  The predictor never opens scorer
references.  A separate model-free process performs scoring exactly once.
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
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    audit_v83_direct_prepared_layout,
)
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
    bind_fixed_prefix_before_question_v81,
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

ARTIFACT: Final[str] = "v83_direct_historical_internal_predictions_v1"
SCORE_ARTIFACT: Final[str] = "v83_direct_historical_internal_score_v1"
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
    "reports/gemma4/predictions/v83_direct_historical_internal.json"
)
DEFAULT_SCORE: Final[Path] = Path(
    "reports/gemma4/metrics/v83_direct_historical_internal_score.json"
)
ARMS: Final[tuple[str, ...]] = (
    "v83_direct",
    "paired_wrong",
    "zero_payload",
    "shuffled_atlas",
    "frozen_v54",
)
TARGET_CORRECT: Final[int] = 9
TARGET_GAIN_OVER_FROZEN: Final[int] = 3
TARGET_CORRECT_MINUS_WRONG: Final[int] = 2
TARGET_PREDICTION_CHANGING_UNITS: Final[int] = 2


def _zero_payload_memory(source: torch.Tensor) -> torch.Tensor:
    """Keep native boundaries and zero every one of 736 payload tokens."""

    return torch.cat(
        (
            source[:, :1],
            torch.zeros_like(source[:, 1:-1]),
            source[:, -1:],
        ),
        dim=1,
    ).detach()


def _shuffled_atlas_memory(source: torch.Tensor) -> torch.Tensor:
    """Roll atlas values across keys while preserving keys, base, and boundaries."""

    banks = split_v75_v2_prefix_v81(source)
    values = banks.atlas_values.roll(shifts=1, dims=1)
    atlas = torch.cat((banks.probe_keys.unsqueeze(2), values), dim=2).reshape(
        source.shape[0], ATLAS_MEMORY_TOKENS, HIDDEN_SIZE
    )
    return torch.cat((banks.boi, atlas, banks.base_latents, banks.eoi), dim=1).detach()


def _preflight_memory(
    runtime: StaticChatRuntime,
    memory: torch.Tensor,
) -> dict[str, Any]:
    """Run the exact native BOI/EOI/PAD-PLE audit without a user question."""

    backend = runtime.language.prefix_backend
    if backend is None or runtime.language.backend_name != "gemma4":
        raise RuntimeError("V83 historical predictor requires local Gemma 4")
    contract = backend.native_image_contract()
    bos_only = torch.tensor(
        [[int(contract["bos_token_id"])]],
        dtype=torch.long,
        device=runtime.language.device,
    )
    direct = memory.to(
        device=runtime.language.device,
        dtype=next(runtime.language.model.parameters()).dtype,
    )
    prepared = backend.prepare(
        direct,
        bos_only,
        scene_prefix_after_bos=True,
        scene_boundary_mode="gemma4_native_image",
        control_tokens=None,
    )
    return audit_v83_direct_prepared_layout(
        backend=backend,
        fixed_memory=direct,
        prompt_ids=bos_only,
        prepared=prepared,
    )


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
    """Create one immutable V83 prediction artifact; never overwrite it."""

    output = _resolve(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    scorer_root = _resolve(references_forbidden_root)
    audit = _runtime_audit(scorer_root)
    started = time.perf_counter()
    with audit:
        config_path = _guard_regular(_resolve(runtime_config), "V83 runtime config")
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
            raise TypeError("V83 compiler requires the sealed V75 controller")

        # Compile every environment and every control tensor before opening the
        # question manifest. No later operation changes these CPU tensors.
        fixed_memories: dict[str, torch.Tensor] = {}
        zero_memories: dict[str, torch.Tensor] = {}
        shuffled_memories: dict[str, torch.Tensor] = {}
        bindings: dict[str, Any] = {}
        tensor_hashes: dict[str, str] = {}
        for scene_id in SCENE_IDS:
            fixed = (
                compile_fixed_scene_atlas_v75_v2(
                    base_prefixes[scene_id], controller, probes
                )
                .scene_prefix.detach()
                .cpu()
                .contiguous()
            )
            fixed_memories[scene_id] = fixed
            zero_memories[scene_id] = _zero_payload_memory(fixed)
            shuffled_memories[scene_id] = _shuffled_atlas_memory(fixed)
            bindings[scene_id] = bind_fixed_prefix_before_question_v81(fixed)
            tensor_hashes[scene_id] = tensor_sha256(fixed)
        hashes_before = {
            scene_id: prefix_sha256(value) for scene_id, value in fixed_memories.items()
        }
        zero_hashes = {
            scene_id: prefix_sha256(value) for scene_id, value in zero_memories.items()
        }
        shuffled_hashes = {
            scene_id: prefix_sha256(value)
            for scene_id, value in shuffled_memories.items()
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
            raise RuntimeError("V83 loaded an unexpected local Gemma model")

        # Validate the direct 738-token sequence for all sixteen immutable
        # memories while the question file is still unopened.
        layout_audits = {
            scene_id: _preflight_memory(runtime, fixed_memories[scene_id])
            for scene_id in SCENE_IDS
        }
        if len(bindings) != len(SCENE_IDS) or any(
            bindings[scene_id].fixed_prefix_sha256 != hashes_before[scene_id]
            for scene_id in SCENE_IDS
        ):
            raise RuntimeError("V83 pre-question memory binding failed")
        all_memories_compiled_and_bound_before_questions = True

        # Only now may the predictor open answer-free user questions.
        rows, question_metadata = _load_predictor_questions(
            _resolve(questions_root), audit
        )
        row_by_scene = {str(row["scene_id"]): row for row in rows}
        if set(row_by_scene) != set(SCENE_IDS):
            raise ValueError("V83 questions lost the one-row-per-scene layout")

        device = runtime.language.device
        dtype = next(runtime.language.model.parameters()).dtype
        records: list[dict[str, Any]] = []
        for ordinal, row in enumerate(rows, 1):
            scene_id = str(row["scene_id"])
            question = str(row["question"])
            paired = [
                candidate
                for candidate in SCENE_IDS
                if candidate != scene_id
                and str(row_by_scene[candidate]["question"]) == question
            ]
            if len(paired) != 1:
                raise ValueError("V83 paired-wrong scene is not uniquely defined")
            wrong_id = paired[0]
            fixed = fixed_memories[scene_id].to(device=device, dtype=dtype)
            wrong = fixed_memories[wrong_id].to(device=device, dtype=dtype)
            zero = zero_memories[scene_id].to(device=device, dtype=dtype)
            shuffled = shuffled_memories[scene_id].to(device=device, dtype=dtype)
            frozen = reconstruct_base_v54_prefix_v81(fixed)
            row_started = time.perf_counter()
            with torch.inference_mode():
                direct_prediction = _generate(runtime, fixed, question, None)
                wrong_prediction = _generate(runtime, wrong, question, None)
                zero_prediction = _generate(runtime, zero, question, None)
                shuffled_prediction = _generate(runtime, shuffled, question, None)
                frozen_prediction = _generate(runtime, frozen, question, None)
            records.append(
                {
                    "row_id": row["row_id"],
                    "scene_id": scene_id,
                    "paired_wrong_scene_id": wrong_id,
                    "fixed_memory_sha256": hashes_before[scene_id],
                    "fixed_memory_tensor_sha256": tensor_hashes[scene_id],
                    "v83_direct_prediction": direct_prediction,
                    "paired_wrong_prediction": wrong_prediction,
                    "zero_payload_prediction": zero_prediction,
                    "shuffled_atlas_prediction": shuffled_prediction,
                    "frozen_v54_prediction": frozen_prediction,
                    "control_activation_tokens": 0,
                    "question_derived_environmental_tokens": 0,
                    "elapsed_seconds": time.perf_counter() - row_started,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "v83_direct_historical_row",
                        "ordinal": ordinal,
                        "total": ROW_COUNT,
                        "row_id": row["row_id"],
                        "scene_id": scene_id,
                        "v83_direct": direct_prediction,
                        "paired_wrong": wrong_prediction,
                        "zero_payload": zero_prediction,
                        "shuffled_atlas": shuffled_prediction,
                        "frozen_v54": frozen_prediction,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        hashes_after = {
            scene_id: prefix_sha256(value) for scene_id, value in fixed_memories.items()
        }
        invariant = hashes_before == hashes_after and all(
            record["fixed_memory_sha256"]
            == hashes_before[str(record["scene_id"])]
            for record in records
        )
        if not invariant:
            raise RuntimeError("V83 fixed scene memory changed after user questions")
    audit.assert_clean()
    if any(Path(path).is_relative_to(scorer_root) for path in audit.unique_paths):
        raise RuntimeError("V83 predictor opened answer-bearing scorer data")

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
        "architecture": {
            "name": RUNTIME_ARCHITECTURE,
            "fixed_scene_memory_tokens": 738,
            "continuous_environment_payload_tokens": 736,
            "native_boundary_tokens": 2,
            "tokens_supplied_directly_to_gemma": 738,
            "control_activation_tokens": 0,
            "reader_enabled": False,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_scene_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "native_boi_eoi_exact": True,
            "interior_pad_ple_exact": True,
            "interior_image_modality_exact": True,
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
            "all_memories_compiled_and_bound_before_question_manifest_opened": (
                all_memories_compiled_and_bound_before_questions
            ),
            "fixed_hashes_before": hashes_before,
            "fixed_hashes_after": hashes_after,
            "fixed_tensor_hashes": tensor_hashes,
            "zero_payload_hashes": zero_hashes,
            "shuffled_atlas_hashes": shuffled_hashes,
            "fixed_memory_invariant": invariant,
            "same_memory_reused_for_every_question": True,
            "layout_audited_scene_count_before_questions": len(layout_audits),
        },
        "controls": {
            "paired_wrong_scene": True,
            "zero_all_736_payload_tokens": True,
            "shuffled_atlas_values": True,
            "frozen_v54_base_prefix": True,
        },
        "leakage": {
            "loaded_file_count": len(audit.unique_paths),
            "loaded_files": audit.unique_paths,
            "forbidden_access_count": len(audit.forbidden_accesses()),
            "forbidden_accesses": audit.forbidden_accesses(),
            "scorer_reference_files_loaded": False,
            "environmental_text_inputs": [],
        },
        "targets_preregistered_before_score": {
            "candidate_correct_at_least": TARGET_CORRECT,
            "gain_over_frozen_v54_at_least": TARGET_GAIN_OVER_FROZEN,
            "correct_minus_paired_wrong_at_least": TARGET_CORRECT_MINUS_WRONG,
            "prediction_changing_units_at_least": TARGET_PREDICTION_CHANGING_UNITS,
        },
        "behavioral_accuracy_scored_in_predictor": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    _atomic_write_json(output, payload)
    return payload


RUNTIME_ARCHITECTURE: Final[str] = "exact_738_token_direct_native_gemma_prefix_v83"


def score(
    predictions_path: str | Path = DEFAULT_PREDICTIONS,
    references_root: str | Path = DEFAULT_REFERENCES,
    output_path: str | Path = DEFAULT_SCORE,
) -> dict[str, Any]:
    """Score sealed predictions without loading Gemma or any scene memory."""

    from semantic_3d_chat.evaluation.v55_development_score import (
        canonical_type_specific_match,
    )

    prediction_path = _guard_regular(_resolve(predictions_path), "V83 predictions")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if (
        not isinstance(predictions, Mapping)
        or predictions.get("artifact") != ARTIFACT
        or predictions.get("execution_valid") is not True
        or predictions.get("row_count") != ROW_COUNT
        or predictions.get("behavioral_accuracy_scored_in_predictor") is not False
        or predictions.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V83 prediction artifact contract changed")
    memory = predictions.get("memory")
    leakage = predictions.get("leakage")
    architecture = predictions.get("architecture")
    if (
        not isinstance(memory, Mapping)
        or memory.get("fixed_memory_invariant") is not True
        or memory.get(
            "all_memories_compiled_and_bound_before_question_manifest_opened"
        )
        is not True
        or memory.get("layout_audited_scene_count_before_questions")
        != len(SCENE_IDS)
        or not isinstance(leakage, Mapping)
        or leakage.get("forbidden_access_count") != 0
        or leakage.get("scorer_reference_files_loaded") is not False
        or not isinstance(architecture, Mapping)
        or architecture.get("tokens_supplied_directly_to_gemma") != 738
        or architecture.get("question_derived_environmental_tokens") != 0
        or architecture.get("reader_enabled") is not False
    ):
        raise ValueError("V83 structural, sequence, or isolation evidence failed")
    records = predictions.get("records")
    if not isinstance(records, list) or len(records) != ROW_COUNT:
        raise ValueError("V83 prediction rows changed")
    references, reference_metadata = _load_reference_artifact(_resolve(references_root))
    if {record.get("row_id") for record in records} != set(references):
        raise ValueError("V83 prediction/reference row IDs differ")

    arm_fields = {
        "v83_direct": "v83_direct_prediction",
        "paired_wrong": "paired_wrong_prediction",
        "zero_payload": "zero_payload_prediction",
        "shuffled_atlas": "shuffled_atlas_prediction",
        "frozen_v54": "frozen_v54_prediction",
    }
    joined: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("V83 prediction row must be a mapping")
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
    direct_correct = int(scores["v83_direct"]["correct"])
    wrong_correct = int(scores["paired_wrong"]["correct"])
    frozen_correct = int(scores["frozen_v54"]["correct"])
    gates = {
        "candidate_correct_at_least_9": direct_correct >= TARGET_CORRECT,
        "gain_over_frozen_v54_at_least_3": (
            direct_correct - frozen_correct >= TARGET_GAIN_OVER_FROZEN
        ),
        "correct_minus_paired_wrong_at_least_2": (
            direct_correct - wrong_correct >= TARGET_CORRECT_MINUS_WRONG
        ),
        "prediction_changing_units_at_least_2": (
            changes["v83_direct"] >= TARGET_PREDICTION_CHANGING_UNITS
        ),
        "exact_738_token_direct_prefix": True,
        "zero_question_derived_environmental_tokens": True,
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
            "v83_direct_minus_frozen_v54": (
                scores["v83_direct"]["accuracy"] - scores["frozen_v54"]["accuracy"]
            ),
            "v83_direct_minus_paired_wrong": (
                scores["v83_direct"]["accuracy"] - scores["paired_wrong"]["accuracy"]
            ),
            "v83_direct_minus_zero_payload": (
                scores["v83_direct"]["accuracy"] - scores["zero_payload"]["accuracy"]
            ),
            "v83_direct_minus_shuffled_atlas": (
                scores["v83_direct"]["accuracy"]
                - scores["shuffled_atlas"]["accuracy"]
            ),
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
        "counterfactual_controls_measured": True,
        "no_tuning_after_this_score": True,
        "protected_evaluation_authorized": False,
        # Historical development evidence alone never promotes a runtime.
        "runtime_promotion_authorized": False,
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


if __name__ == "__main__":
    raise SystemExit(predict_main())


__all__ = [
    "ARTIFACT",
    "SCORE_ARTIFACT",
    "predict",
    "predict_main",
    "score",
    "score_main",
]
