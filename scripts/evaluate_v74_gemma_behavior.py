#!/usr/bin/env python3
"""Run a training-pool-only V74 bridge through real local Gemma generation.

This is the behavioral gate that the cheap continuous-prototype screens cannot
replace.  It opens only the locked 40-scene training pool, uses the pair- and
scene-disjoint V73 development fold, and loads Gemma once from the pinned local
snapshot.  Official validation, test, deferred-final, and oracle paths are not
accepted by any input argument.

The candidate is deliberately a numeric training artifact.  This command does
not publish or seal it for chat inference, regardless of its result.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import (
    LIST_ANSWER_TYPES,
    list_order_insensitive_match,
    normalize_answer,
)
from semantic_3d_chat.evaluation.v55_development_score import (
    canonical_type_specific_match,
)
from semantic_3d_chat.language.local_lm import question_token_ids
from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)
from semantic_3d_chat.training.soft_prompt_teacher_v62 import load_v62_teacher_cache
from semantic_3d_chat.training.soft_prompt_teacher_v66 import (
    load_v66_answer_class_teacher_cache,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    freeze_base_runtime,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _disable_decoder_checkpointing,
    _generate_with_control,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    EXPECTED_HIDDEN_SIZE,
    RowV73,
    _sha256_file,
    changed_units_v73,
    load_config_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)

_FORBIDDEN_INPUT_PARTS = frozenset(
    {"oracle", "validation", "validate", "test", "deferred", "final"}
)
_EXPECTED_STATE_FIELDS = frozenset(
    {
        "output_basis",
        "key.weight",
        "value.weight",
        "query.weight",
        "coefficient_output.weight",
    }
)
_NATIVE_STATE_FIELDS = frozenset(
    {
        "output_basis",
        "core.key.weight",
        "core.value.weight",
        "core.query.weight",
        "coefficient_output.weight",
    }
)
_V75_STATE_FIELDS = frozenset(
    {
        "output_basis",
        "key.weight",
        "value.weight",
        "query.weight",
        "coefficient_hidden.weight",
        "coefficient_output.weight",
    }
)
_OPAQUE_ID = re.compile(r"(?:scene|pair|q|cfq)_[0-9a-f]+")


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _guard_training_input(path: str | Path, purpose: str) -> Path:
    source = _resolve(path)
    scoped = source.relative_to(PROJECT_ROOT) if source.is_relative_to(PROJECT_ROOT) else source
    tokens = {
        token
        for part in scoped.parts
        for token in re.split(r"[^a-z0-9]+", part.casefold())
        if token
    }
    forbidden = sorted(tokens & _FORBIDDEN_INPUT_PARTS)
    if forbidden:
        raise ValueError(f"V74 {purpose} crosses forbidden split tokens: {forbidden}")
    if not source.exists() or source.is_symlink():
        raise FileNotFoundError(f"V74 {purpose} is unavailable or symlinked: {source}")
    return source


def select_smoke_rows_v74(held_rows: Sequence[RowV73]) -> tuple[RowV73, ...]:
    """Select both sides of the first sorted changed unit in every family."""

    first_by_family: dict[str, Any] = {}
    for unit in changed_units_v73(held_rows):
        first_by_family.setdefault(unit.change_type, unit)
    if len(first_by_family) != 8:
        raise ValueError("V74 smoke requires exactly eight held change families")
    rows: list[RowV73] = []
    for family in sorted(first_by_family):
        unit = first_by_family[family]
        rows.extend((unit.left, unit.right))
    if len(rows) != 16 or len({row.key for row in rows}) != 16:
        raise RuntimeError("V74 smoke row inventory changed")
    return tuple(rows)


def shard_rows_v74(
    rows: Sequence[RowV73], *, shard_count: int, shard_index: int
) -> tuple[RowV73, ...]:
    """Return a stable modulo shard without changing the locked row order."""

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("V74 behavior shard parameters are invalid")
    shard = tuple(
        row
        for ordinal, row in enumerate(rows)
        if ordinal % shard_count == shard_index
    )
    if not shard:
        raise ValueError("V74 behavior shard contains no rows")
    return shard


def _answer_matches(
    row: RowV73,
    prediction: str,
    reference_items: Sequence[str] | None = None,
) -> bool:
    if row.answer_type in LIST_ANSWER_TYPES:
        reference: str | Sequence[str] = (
            row.answer if reference_items is None else reference_items
        )
        return list_order_insensitive_match(prediction, reference)
    return canonical_type_specific_match(row.answer_type, prediction, row.answer)


def _load_answer_items(path: str | Path) -> dict[tuple[str, str], tuple[str, ...]]:
    """Load list boundaries for scoring only from the locked training-pool QA."""

    source = _guard_training_input(path, "training QA answer items")
    result: dict[tuple[str, str], tuple[str, ...]] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        key = (value.get("scene_id"), value.get("question_id"))
        if not all(isinstance(item, str) and item for item in key):
            raise ValueError(f"V74 scoring key changed at line {line_number}")
        items = value.get("answer_items")
        if items is None:
            continue
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item for item in items)
        ):
            raise ValueError(f"V74 answer-items field changed at line {line_number}")
        result[key] = tuple(items)
    return result


def _candidate_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        fields = frozenset(handle.keys())
    if fields not in {_EXPECTED_STATE_FIELDS, _NATIVE_STATE_FIELDS, _V75_STATE_FIELDS}:
        raise ValueError(f"V74 candidate numeric state fields changed: {sorted(fields)}")
    quarantined = (
        metadata.get("runtime_promotion_forbidden_until_gemma_gate") == "true"
        or metadata.get("runtime_promotion_forbidden") == "true"
    )
    if not quarantined:
        raise ValueError("V74 candidate lacks its behavioral-gate quarantine")
    if metadata.get("answer_codebook_serialized", "false") != "false":
        raise ValueError("V74 candidate answer-codebook contract changed")
    environmental_text_inputs = metadata.get(
        "environmental_text_inputs",
        metadata.get("environmental_text_inputs_at_inference"),
    )
    if environmental_text_inputs != "0":
        raise ValueError("V74 candidate environmental-input contract changed")
    state = load_file(str(path), device="cpu")
    basis = state["output_basis"]
    if fields == _EXPECTED_STATE_FIELDS:
        model: torch.nn.Module = DenseFullSceneContinuousControlV74(
            EXPECTED_HIDDEN_SIZE, basis
        )
    elif fields == _V75_STATE_FIELDS:
        from semantic_3d_chat.scene_encoder.question_control_v75 import (
            DenseFullSceneContinuousControlV75,
        )

        model = DenseFullSceneContinuousControlV75(
            EXPECTED_HIDDEN_SIZE,
            basis,
            coefficient_decoder_hidden_dimension=int(
                state["coefficient_hidden.weight"].shape[0]
            ),
        )
    else:
        from scripts.diagnose_v73_failure import DenseNativePrototypeReader

        model = DenseNativePrototypeReader(basis)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("V74 candidate contains nonfinite tensors")
    return model, metadata


def _question_embeddings(runtime: Any, question: str) -> torch.Tensor:
    ids = question_token_ids(
        runtime.language.tokenizer, question, runtime.language.device
    )
    with torch.inference_mode():
        value = runtime.language.model.get_input_embeddings()(ids).detach().float()
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("V74 Gemma question embedding shape changed")
    return value


def _generate_row(
    *,
    runtime: Any,
    model: torch.nn.Module,
    row: RowV73,
    prefix: torch.Tensor,
    use_control: bool,
    control_override: torch.Tensor | None = None,
) -> tuple[str, float, float]:
    model_dtype = next(runtime.language.model.parameters()).dtype
    scene = prefix.to(device=runtime.language.device, dtype=model_dtype)
    started = time.perf_counter()
    control: torch.Tensor | None = None
    control_rms = 0.0
    if control_override is not None:
        control = control_override.to(device=runtime.language.device).float()
        control_rms = float(control.square().mean().sqrt().cpu())
    elif use_control:
        with torch.inference_mode():
            control = model(
                scene.float(), _question_embeddings(runtime, row.question)
            ).control_tokens
        control_rms = float(control.float().square().mean().sqrt().cpu())
    prediction = _generate_with_control(
        runtime=runtime,
        scene_prefix=scene,
        question=row.question,
        control_tokens=control,
    )
    return prediction, control_rms, time.perf_counter() - started


def _aggregate(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    selected = [record for record in records if record.get(field) is not None]
    correct = sum(bool(record[field]) for record in selected)
    by_family: dict[str, list[bool]] = {}
    for record in selected:
        by_family.setdefault(str(record["change_type"]), []).append(bool(record[field]))
    return {
        "correct": correct,
        "total": len(selected),
        "accuracy": correct / max(len(selected), 1),
        "by_change_type": {
            family: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
            }
            for family, values in sorted(by_family.items())
        },
    }


def _pair_changes(records: Sequence[Mapping[str, Any]], prediction_field: str) -> int:
    grouped: dict[tuple[str, str], list[str]] = {}
    for record in records:
        grouped.setdefault(
            (str(record["pair_id"]), str(record["question_key"])), []
        ).append(normalize_answer(record[prediction_field]))
    return sum(len(values) == 2 and values[0] != values[1] for values in grouped.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        required=True,
        help="Quarantined numeric V74 safetensors candidate",
    )
    parser.add_argument(
        "--runtime-config", default="configs/runtime/gemma4_v54.yaml"
    )
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
    )
    parser.add_argument(
        "--v73-config",
        default="configs/experiments/gemma4_v73_fullscene_controller.yaml",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--scene-arm",
        choices=("correct", "paired"),
        default="correct",
        help=(
            "Use each row's correct scene or its paired counterfactual as the "
            "primary continuous environment input"
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Deterministically divide the selected rows into this many shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to evaluate; intended for resumable full screens",
    )
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--with-wrong-scene", action="store_true")
    parser.add_argument(
        "--with-teacher-diagnostics",
        action="store_true",
        help=(
            "Evaluation-only: compare nearest and oracle training-teacher prompts; "
            "these label/codebook paths can never be used by chat inference"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-count must be positive and --shard-index must be in range")
    if args.scene_arm == "paired" and args.with_wrong_scene:
        parser.error("--scene-arm paired cannot be combined with --with-wrong-scene")

    candidate_path = _guard_training_input(args.candidate, "candidate")
    runtime_config_path = _guard_training_input(args.runtime_config, "runtime config")
    base_checkpoint = _guard_training_input(args.base_checkpoint, "base checkpoint")
    v73_config_path = _guard_training_input(args.v73_config, "screen config")
    output_path = _resolve(args.output)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    v73 = load_config_v73(v73_config_path)
    all_rows = load_training_rows_v73(v73["training_qa"])
    train_rows, held_rows = split_rows_v73(all_rows)
    answer_items = _load_answer_items(v73["training_qa"])
    selected_rows = (
        select_smoke_rows_v74(held_rows) if args.mode == "smoke" else held_rows
    )
    rows = shard_rows_v74(
        selected_rows,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    prefixes, prefix_manifest = load_prefixes_v73(
        v73["prefix_cache"], {row.scene_id for row in rows} | {row.paired_scene_id for row in rows}
    )
    runtime_config, _ = _load_sanitized_runtime_config(runtime_config_path)
    runtime = StaticRuntimePrefixFactory(
        runtime_config, base_checkpoint, "scene_000011"
    ).bootstrap
    freeze_base_runtime(runtime)
    _disable_decoder_checkpointing(runtime.language)
    device = torch.device(runtime.language.device)
    model, candidate_metadata = _candidate_model(candidate_path, device)
    training_classes = {row.answer_class for row in train_rows}
    teacher_bank = None
    teacher_prototypes = None
    if args.with_teacher_diagnostics:
        # Import lazily so the primary behavioral path has no teacher-cache edge.
        from scripts.train_v74_teacher_reader import _teacher_bank

        primary, _primary_metadata = load_v62_teacher_cache(
            "data_gemma4/training/v62_changed_teachers"
        )
        supplemental, _supplemental_metadata = load_v66_answer_class_teacher_cache(
            "data_gemma4/training/v66_answer_class_teachers"
        )
        teacher_bank = _teacher_bank(train_rows, {**primary, **supplemental})
        teacher_prototypes = teacher_bank.prototypes.to(device)

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        primary_scene_id = (
            row.scene_id if args.scene_arm == "correct" else row.paired_scene_id
        )
        prediction, control_rms, elapsed = _generate_row(
            runtime=runtime,
            model=model,
            row=row,
            prefix=prefixes[primary_scene_id],
            use_control=True,
        )
        record: dict[str, Any] = {
            "scene_id": row.scene_id,
            "environment_scene_id": primary_scene_id,
            "question_id": row.question_id,
            "pair_id": row.pair_id,
            "question_key": row.question_key,
            "change_type": row.change_type,
            "answer_type": row.answer_type,
            "answer_class_supported": row.answer_class in training_classes,
            "reference": row.answer,
            "prediction": prediction,
            "correct": _answer_matches(row, prediction, answer_items.get(row.key)),
            "control_rms": control_rms,
            "elapsed_seconds": elapsed,
        }
        if teacher_bank is not None and teacher_prototypes is not None:
            # Both arms below are explicitly prohibited runtime controls.  The
            # nearest arm quantifies whether vector regression, rather than
            # class recognition, is the failure.  The oracle arm uses the held
            # answer only as an evaluation upper bound on teacher universality.
            scene = prefixes[primary_scene_id].to(
                device=runtime.language.device,
                dtype=next(runtime.language.model.parameters()).dtype,
            )
            with torch.inference_mode():
                predicted_control = model(
                    scene.float(), _question_embeddings(runtime, row.question)
                ).control_tokens
            similarities = F.normalize(predicted_control.flatten(1), dim=-1) @ F.normalize(
                teacher_prototypes.flatten(1), dim=-1
            ).T
            nearest_index = int(similarities.argmax(dim=-1).item())
            nearest, _nearest_rms, nearest_elapsed = _generate_row(
                runtime=runtime,
                model=model,
                row=row,
                prefix=prefixes[primary_scene_id],
                use_control=False,
                control_override=teacher_prototypes[nearest_index : nearest_index + 1],
            )
            record.update(
                nearest_teacher_prediction=nearest,
                nearest_teacher_correct=_answer_matches(
                    row, nearest, answer_items.get(row.key)
                ),
                nearest_teacher_class_id=teacher_bank.class_ids[nearest_index],
                nearest_teacher_elapsed_seconds=nearest_elapsed,
            )
            oracle_index = teacher_bank.class_index.get(row.answer_class)
            if oracle_index is not None:
                oracle_prediction, _oracle_rms, oracle_elapsed = _generate_row(
                    runtime=runtime,
                    model=model,
                    row=row,
                    prefix=prefixes[primary_scene_id],
                    use_control=False,
                    control_override=teacher_prototypes[
                        oracle_index : oracle_index + 1
                    ],
                )
                record.update(
                    oracle_teacher_prediction=oracle_prediction,
                    oracle_teacher_correct=_answer_matches(
                        row, oracle_prediction, answer_items.get(row.key)
                    ),
                    oracle_teacher_elapsed_seconds=oracle_elapsed,
                )
        if args.with_baseline:
            baseline, _unused_rms, baseline_elapsed = _generate_row(
                runtime=runtime,
                model=model,
                row=row,
                prefix=prefixes[primary_scene_id],
                use_control=False,
            )
            record.update(
                baseline_prediction=baseline,
                baseline_correct=_answer_matches(
                    row, baseline, answer_items.get(row.key)
                ),
                baseline_elapsed_seconds=baseline_elapsed,
            )
        if args.with_wrong_scene:
            wrong, wrong_rms, wrong_elapsed = _generate_row(
                runtime=runtime,
                model=model,
                row=row,
                prefix=prefixes[row.paired_scene_id],
                use_control=True,
            )
            record.update(
                wrong_scene_prediction=wrong,
                wrong_scene_correct=_answer_matches(
                    row, wrong, answer_items.get(row.key)
                ),
                wrong_scene_control_rms=wrong_rms,
                wrong_scene_elapsed_seconds=wrong_elapsed,
            )
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "v74_gemma_behavior_row",
                    "index": index,
                    "total": len(rows),
                    "scene_id": row.scene_id,
                    "question_id": row.question_id,
                    "prediction": prediction,
                    "reference": row.answer,
                    "correct": record["correct"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    candidate_metrics = _aggregate(records, "correct")
    result: dict[str, Any] = {
        "artifact": "v74_training_pool_pair_disjoint_real_gemma_behavior_v1",
        "mode": args.mode,
        "scene_arm": args.scene_arm,
        "environment_scene_source": (
            "row_scene" if args.scene_arm == "correct" else "paired_counterfactual_scene"
        ),
        "selected_row_count_before_sharding": len(selected_rows),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
        "candidate_sha256": _sha256_file(candidate_path),
        "candidate_metadata": candidate_metadata,
        "base_checkpoint_sha256": prefix_manifest["base_checkpoint_sha256"],
        "device": str(device),
        "row_count": len(rows),
        "change_family_counts": dict(Counter(row.change_type for row in rows)),
        "candidate": candidate_metrics,
        "candidate_prediction_change_units": _pair_changes(records, "prediction"),
        "mean_control_rms": sum(record["control_rms"] for record in records)
        / len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "checkpoint_published": False,
        "runtime_promotion_authorized": False,
        "records": records,
    }
    if args.with_baseline:
        result["frozen_v54_baseline"] = _aggregate(records, "baseline_correct")
        result["baseline_prediction_change_units"] = _pair_changes(
            records, "baseline_prediction"
        )
        result["candidate_accuracy_gain"] = (
            candidate_metrics["accuracy"]
            - result["frozen_v54_baseline"]["accuracy"]
        )
    if args.with_wrong_scene:
        result["wrong_scene"] = _aggregate(records, "wrong_scene_correct")
        result["correct_over_wrong_scene_accuracy"] = (
            candidate_metrics["accuracy"] - result["wrong_scene"]["accuracy"]
        )
    if args.with_teacher_diagnostics:
        result["teacher_diagnostics_runtime_permitted"] = False
        result["nearest_teacher"] = _aggregate(
            records, "nearest_teacher_correct"
        )
        result["nearest_teacher_prediction_change_units"] = _pair_changes(
            records, "nearest_teacher_prediction"
        )
        result["oracle_teacher_upper_bound"] = _aggregate(
            records, "oracle_teacher_correct"
        )
        result["oracle_teacher_prediction_change_units"] = _pair_changes(
            records, "oracle_teacher_prediction"
        )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
