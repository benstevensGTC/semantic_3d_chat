"""Scene-disjoint behavioral gates for the preregistered V79 diagnostic.

The screen evaluates every changed spatial-relation side in the 16-scene V73
internal held fold.  Exact V75, exact V77, and V79 receive both the correct
continuous scene and the paired wrong continuous scene in one frozen-Gemma
process.  V79 must strictly beat both baselines on correct-scene answers while
matching their best causal gap and changed-unit response before the full held
evaluation is allowed.

The conditional full stage evaluates V79 on all 384 internal held rows with a
matched wrong-scene arm and reports answer-type, change-type, and changed/stable
breakdowns.  Neither stage accepts official validation, test, deferred-final,
or oracle paths, and neither publishes a runtime checkpoint.
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
from safetensors import safe_open

from scripts.evaluate_v74_gemma_behavior import (
    _answer_matches,
    _candidate_model,
    _generate_row,
    _load_answer_items,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.training.finetune_v79_relation_counterfactual import (
    V79_PREREGISTRATION,
    V79_PREREGISTRATION_SHA256,
    _atomic_create_v79,
    candidate_metadata_v79,
    guard_input_v79,
    guard_output_v79,
    load_preregistration_v79,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    StaticRuntimePrefixFactory,
    _load_sanitized_runtime_config,
    freeze_base_runtime,
)
from semantic_3d_chat.training.train_question_control_v58 import (
    _disable_decoder_checkpointing,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    HELD_PAIR_IDS,
    RowV73,
    _sha256_file,
    load_config_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)

V79_CANDIDATE: Final[str] = (
    "reports/gemma4/artifacts/v79_v75_relation_counterfactual_diagnostic.safetensors"
)
V79_SCREEN_OUTPUT: Final[str] = "reports/gemma4/metrics/v79_relation_counterfactual_screen.json"
V79_FULL_OUTPUT: Final[str] = "reports/gemma4/metrics/v79_relation_counterfactual_full.json"
V75_FULL_CORRECT_REPORT: Final[str] = "reports/gemma4/metrics/v75_gemma_nll_balanced_held_full.json"
V75_FULL_WRONG_REPORT: Final[str] = (
    "reports/gemma4/metrics/v75_gemma_nll_balanced_wrong_scene_full.json"
)
V77_FULL_CORRECT_REPORT: Final[str] = (
    "reports/gemma4/metrics/v77_v75_r72_historical_repair_held_full.json"
)
EXPECTED_SCREEN_ROWS: Final[int] = 28
EXPECTED_SCREEN_UNITS: Final[int] = 14


def select_screen_rows_v79(held_rows: Sequence[RowV73]) -> tuple[RowV73, ...]:
    """Select all changed held relation sides, with no cherry-picked family."""

    if {row.pair_id for row in held_rows} != set(HELD_PAIR_IDS):
        raise ValueError("V79 held pair inventory changed")
    selected = tuple(
        row for row in held_rows if row.expected_change and row.answer_type == "spatial_relation"
    )
    unit_keys = {(row.pair_id, row.question_key) for row in selected}
    if (
        len(selected) != EXPECTED_SCREEN_ROWS
        or len({row.key for row in selected}) != EXPECTED_SCREEN_ROWS
        or len(unit_keys) != EXPECTED_SCREEN_UNITS
        or any(
            sum(
                member.pair_id == pair_id and member.question_key == question_key
                for member in selected
            )
            != 2
            for pair_id, question_key in unit_keys
        )
    ):
        raise ValueError("V79 relation screen inventory changed")
    return selected


def _load_candidate_v79(
    path: Path,
    *,
    device: torch.device,
    expected_sha256: str | None,
    expected_artifact: str,
) -> tuple[torch.nn.Module, dict[str, str], str]:
    digest = _sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"V79 baseline candidate hash changed: {path.name}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    if metadata.get("artifact") != expected_artifact:
        raise ValueError(f"V79 candidate artifact contract changed: {path.name}")
    if (
        expected_artifact == candidate_metadata_v79()["artifact"]
        and metadata != candidate_metadata_v79()
    ):
        raise ValueError("V79 diagnostic metadata changed")
    model, loaded_metadata = _candidate_model(path, device)
    if loaded_metadata != metadata:
        raise RuntimeError("V79 candidate metadata reload changed")
    return model, metadata, digest


def _accuracy(values: Sequence[bool]) -> dict[str, int | float]:
    if not values:
        raise ValueError("V79 accuracy group cannot be empty")
    correct = sum(values)
    return {
        "correct": correct,
        "total": len(values),
        "accuracy": correct / len(values),
    }


def _breakdown(
    records: Sequence[Mapping[str, Any]], field: str, key: str
) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[bool]] = {}
    for record in records:
        grouped.setdefault(str(record[key]), []).append(bool(record[field]))
    return {value: _accuracy(outcomes) for value, outcomes in sorted(grouped.items())}


def _prediction_change_units(records: Sequence[Mapping[str, Any]], field: str) -> int:
    grouped: dict[tuple[str, str], list[str]] = {}
    for record in records:
        grouped.setdefault((str(record["pair_id"]), str(record["question_key"])), []).append(
            normalize_answer(str(record[field]))
        )
    return sum(len(values) == 2 and values[0] != values[1] for values in grouped.values())


def _complete_changed_units(records: Sequence[Mapping[str, Any]], field: str) -> int:
    grouped: dict[tuple[str, str], list[bool]] = {}
    for record in records:
        if bool(record["expected_change"]):
            grouped.setdefault((str(record["pair_id"]), str(record["question_key"])), []).append(
                bool(record[field])
            )
    return sum(len(values) == 2 and all(values) for values in grouped.values())


def summarize_records_v79(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("V79 summary requires records")
    correct_values = [bool(record["correct_scene_correct"]) for record in records]
    wrong_values = [bool(record["wrong_scene_correct"]) for record in records]
    correct = _accuracy(correct_values)
    wrong = _accuracy(wrong_values)
    return {
        "correct_scene": correct,
        "wrong_scene": wrong,
        "correct_minus_wrong_count": int(correct["correct"]) - int(wrong["correct"]),
        "correct_over_wrong_accuracy": float(correct["accuracy"]) - float(wrong["accuracy"]),
        "correct_scene_prediction_changing_units": _prediction_change_units(
            records, "correct_scene_prediction"
        ),
        "wrong_scene_prediction_changing_units": _prediction_change_units(
            records, "wrong_scene_prediction"
        ),
        "matched_scene_input_prediction_changes": sum(
            normalize_answer(str(record["correct_scene_prediction"]))
            != normalize_answer(str(record["wrong_scene_prediction"]))
            for record in records
        ),
        "correct_scene_complete_changed_units": _complete_changed_units(
            records, "correct_scene_correct"
        ),
        "wrong_scene_complete_changed_units": _complete_changed_units(
            records, "wrong_scene_correct"
        ),
        "by_answer_type": {
            "correct_scene": _breakdown(records, "correct_scene_correct", "answer_type"),
            "wrong_scene": _breakdown(records, "wrong_scene_correct", "answer_type"),
        },
        "by_change_type": {
            "correct_scene": _breakdown(records, "correct_scene_correct", "change_type"),
            "wrong_scene": _breakdown(records, "wrong_scene_correct", "change_type"),
        },
        "by_expected_change": {
            "correct_scene": _breakdown(records, "correct_scene_correct", "expected_change_label"),
            "wrong_scene": _breakdown(records, "wrong_scene_correct", "expected_change_label"),
        },
    }


def screen_decision_v79(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(summaries) != {"v75", "v77", "v79"}:
        raise ValueError("V79 screen requires exact V75/V77/V79 summaries")
    correct_counts = {
        name: int(summary["correct_scene"]["correct"]) for name, summary in summaries.items()
    }
    causal_gaps = {
        name: int(summary["correct_minus_wrong_count"]) for name, summary in summaries.items()
    }
    change_units = {
        name: int(summary["correct_scene_prediction_changing_units"])
        for name, summary in summaries.items()
    }
    conditions = {
        "correct_count_strictly_above_v75": correct_counts["v79"] > correct_counts["v75"],
        "correct_count_strictly_above_v77": correct_counts["v79"] > correct_counts["v77"],
        "causal_gap_at_least_best_baseline": causal_gaps["v79"]
        >= max(causal_gaps["v75"], causal_gaps["v77"]),
        "prediction_changing_units_at_least_best_baseline": change_units["v79"]
        >= max(change_units["v75"], change_units["v77"]),
    }
    return {
        "conditions": conditions,
        "screen_passed": all(conditions.values()),
        "correct_counts": correct_counts,
        "correct_minus_wrong_counts": causal_gaps,
        "correct_scene_prediction_changing_units": change_units,
        "full_evaluation_authorized": all(conditions.values()),
        "runtime_promotion_authorized": False,
    }


def _evaluate_model(
    *,
    runtime: Any,
    model: torch.nn.Module,
    rows: Sequence[RowV73],
    prefixes: Mapping[str, torch.Tensor],
    answer_items: Mapping[tuple[str, str], Sequence[str]],
    model_name: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        prediction, control_rms, correct_elapsed = _generate_row(
            runtime=runtime,
            model=model,
            row=row,
            prefix=prefixes[row.scene_id],
            use_control=True,
        )
        wrong_prediction, wrong_rms, wrong_elapsed = _generate_row(
            runtime=runtime,
            model=model,
            row=row,
            prefix=prefixes[row.paired_scene_id],
            use_control=True,
        )
        record = {
            "scene_id": row.scene_id,
            "paired_scene_id": row.paired_scene_id,
            "question_id": row.question_id,
            "pair_id": row.pair_id,
            "question_key": row.question_key,
            "answer_type": row.answer_type,
            "change_type": row.change_type,
            "expected_change": row.expected_change,
            "expected_change_label": ("changed" if row.expected_change else "stable"),
            "reference": row.answer,
            "correct_scene_prediction": prediction,
            "correct_scene_correct": _answer_matches(row, prediction, answer_items.get(row.key)),
            "correct_scene_control_rms": control_rms,
            "correct_scene_elapsed_seconds": correct_elapsed,
            "wrong_scene_prediction": wrong_prediction,
            "wrong_scene_correct": _answer_matches(
                row, wrong_prediction, answer_items.get(row.key)
            ),
            "wrong_scene_control_rms": wrong_rms,
            "wrong_scene_elapsed_seconds": wrong_elapsed,
        }
        result.append(record)
        print(
            json.dumps(
                {
                    "event": "v79_behavior_row",
                    "model": model_name,
                    "index": index,
                    "total": len(rows),
                    "scene_id": row.scene_id,
                    "question_id": row.question_id,
                    "correct_scene_correct": record["correct_scene_correct"],
                    "wrong_scene_correct": record["wrong_scene_correct"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return result


def _bootstrap_v79(
    prereg: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[Any, dict[str, torch.Tensor], dict[tuple[str, str], tuple[str, ...]]]:
    sources = prereg["sources"]
    v73_config_path = guard_input_v79(sources["v73_split_config"]["path"], "V73 split config")
    runtime_config_path = guard_input_v79(sources["runtime_config"]["path"], "runtime config")
    base_checkpoint = guard_input_v79(
        "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
        "base checkpoint",
    )
    v73 = load_config_v73(v73_config_path)
    qa_path = guard_input_v79(v73["training_qa"], "historical training QA")
    prefix_cache = guard_input_v79(v73["prefix_cache"], "immutable prefix cache")
    prefixes, manifest = load_prefixes_v73(
        prefix_cache,
        {row.scene_id for row in rows} | {row.paired_scene_id for row in rows},
    )
    if manifest["base_checkpoint_sha256"] != sources["base_checkpoint_sha256"]:
        raise ValueError("V79 evaluation base checkpoint hash changed")
    runtime_config, _ = _load_sanitized_runtime_config(runtime_config_path)
    runtime = StaticRuntimePrefixFactory(
        runtime_config, base_checkpoint, rows[0].scene_id
    ).bootstrap
    freeze_base_runtime(runtime)
    _disable_decoder_checkpointing(runtime.language)
    return runtime, prefixes, _load_answer_items(qa_path)


def _load_rows_v79(
    prereg: Mapping[str, Any], *, screen: bool
) -> tuple[tuple[RowV73, ...], tuple[RowV73, ...]]:
    v73 = load_config_v73(prereg["sources"]["v73_split_config"]["path"])
    all_rows = load_training_rows_v73(v73["training_qa"])
    train_rows, held_rows = split_rows_v73(all_rows)
    if {row.scene_id for row in train_rows} & {row.scene_id for row in held_rows}:
        raise RuntimeError("V79 evaluation split is not scene disjoint")
    return held_rows, select_screen_rows_v79(held_rows) if screen else held_rows


def run_screen_v79(
    preregistration: str | Path = V79_PREREGISTRATION,
) -> dict[str, Any]:
    prereg_path, prereg = load_preregistration_v79(preregistration)
    output = guard_output_v79(prereg["screen"]["output"], suffix=".json")
    held_rows, rows = _load_rows_v79(prereg, screen=True)
    runtime, prefixes, answer_items = _bootstrap_v79(prereg, rows)
    device = torch.device(runtime.language.device)
    sources = prereg["sources"]
    candidate_specs = {
        "v75": (
            sources["v75_initial_candidate"]["path"],
            sources["v75_initial_candidate"]["sha256"],
            "v75_historical_train_gemma_nll_diagnostic_v1",
        ),
        "v77": (
            sources["v77_screen_baseline"]["path"],
            sources["v77_screen_baseline"]["sha256"],
            "v77_all_historical_answer_repair_diagnostic_v1",
        ),
        "v79": (
            V79_CANDIDATE,
            None,
            "v79_historical_relation_counterfactual_diagnostic_v1",
        ),
    }
    summaries: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for name, (path, digest, artifact) in candidate_specs.items():
        candidate_path = guard_input_v79(path, f"{name} candidate")
        model, metadata, actual_digest = _load_candidate_v79(
            candidate_path,
            device=device,
            expected_sha256=digest,
            expected_artifact=artifact,
        )
        model_records = _evaluate_model(
            runtime=runtime,
            model=model,
            rows=rows,
            prefixes=prefixes,
            answer_items=answer_items,
            model_name=name,
        )
        records[name] = model_records
        summaries[name] = summarize_records_v79(model_records)
        candidates[name] = {
            "path": str(candidate_path.relative_to(PROJECT_ROOT)),
            "sha256": actual_digest,
            "metadata": metadata,
        }
        del model
    decision = screen_decision_v79(summaries)
    report = {
        "artifact": "v79_relation_counterfactual_scene_disjoint_screen_v1",
        "preregistration": {
            "path": str(prereg_path.relative_to(PROJECT_ROOT)),
            "sha256": V79_PREREGISTRATION_SHA256,
            "authenticated_before_screen": True,
        },
        "scope": {
            "historical_training_pool_only": True,
            "optimization_scene_count": 24,
            "internal_held_scene_count": len({row.scene_id for row in held_rows}),
            "optimization_held_scene_overlap": 0,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
            "runtime_promotion_authorized": False,
            "checkpoint_published": False,
        },
        "selection": {
            "row_count": len(rows),
            "unit_count": len({(row.pair_id, row.question_key) for row in rows}),
            "answer_type_count": len({row.answer_type for row in rows}),
            "change_type_counts": dict(sorted(Counter(row.change_type for row in rows).items())),
            "all_changed_relation_sides_selected": True,
        },
        "candidates": candidates,
        "summaries": summaries,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    _atomic_create_v79(
        output,
        (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "event": "v79_screen_complete",
                "output": str(output.relative_to(PROJECT_ROOT)),
                "screen_passed": decision["screen_passed"],
                "correct_counts": decision["correct_counts"],
                "causal_gaps": decision["correct_minus_wrong_counts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def _authenticated_screen_v79(prereg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = guard_input_v79(prereg["screen"]["output"], "V79 screen report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("artifact") != "v79_relation_counterfactual_scene_disjoint_screen_v1"
        or report.get("preregistration", {}).get("sha256") != V79_PREREGISTRATION_SHA256
        or report.get("decision", {}).get("screen_passed") is not True
    ):
        raise ValueError("V79 full evaluation is not authorized by its screen")
    candidate_path = guard_input_v79(V79_CANDIDATE, "V79 candidate")
    if report.get("candidates", {}).get("v79", {}).get("sha256") != _sha256_file(candidate_path):
        raise ValueError("V79 candidate changed after the screen")
    return path, report


def _historical_baseline_types_v79(
    prereg: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    hashes = prereg["sources"]["historical_internal_reports"]
    specs = {
        "v75_correct": (
            V75_FULL_CORRECT_REPORT,
            hashes["v75_correct_full_sha256"],
        ),
        "v75_wrong": (
            V75_FULL_WRONG_REPORT,
            hashes["v75_wrong_full_sha256"],
        ),
        "v77_correct": (
            V77_FULL_CORRECT_REPORT,
            hashes["v77_correct_full_sha256"],
        ),
    }
    reports: dict[str, Any] = {}
    for name, (path, expected_hash) in specs.items():
        source = guard_input_v79(path, f"historical {name} report")
        if _sha256_file(source) != expected_hash:
            raise ValueError(f"V79 historical baseline report changed: {name}")
        reports[name] = json.loads(source.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for record in reports["v75_correct"]["records"]:
        counts.setdefault(str(record["answer_type"]), 0)
        counts[str(record["answer_type"])] += bool(record["correct"])
    return counts, {
        name: {
            "path": path,
            "sha256": expected_hash,
            "correct": report["candidate"]["correct"],
            "row_count": report["row_count"],
        }
        for (name, (path, expected_hash)), report in zip(
            specs.items(), reports.values(), strict=True
        )
    }


def full_decision_v79(
    summary: Mapping[str, Any], v75_type_counts: Mapping[str, int]
) -> dict[str, Any]:
    v79_type_counts = {
        answer_type: int(metrics["correct"])
        for answer_type, metrics in summary["by_answer_type"]["correct_scene"].items()
    }
    drops = {
        answer_type: int(v75_type_counts[answer_type]) - int(v79_type_counts[answer_type])
        for answer_type in sorted(v75_type_counts)
    }
    conditions = {
        "correct_count_at_least_300": int(summary["correct_scene"]["correct"]) >= 300,
        "spatial_relation_correct_at_least_53": v79_type_counts["spatial_relation"] >= 53,
        "correct_minus_wrong_count_at_least_18": int(summary["correct_minus_wrong_count"]) >= 18,
        "prediction_changing_units_at_least_35": int(
            summary["correct_scene_prediction_changing_units"]
        )
        >= 35,
        "maximum_answer_type_correct_drop_from_v75_at_most_2": max(drops.values()) <= 2,
    }
    passed = all(conditions.values())
    return {
        "conditions": conditions,
        "answer_type_correct_count_deltas_vs_v75": {key: -value for key, value in drops.items()},
        "maximum_answer_type_correct_drop_from_v75": max(drops.values()),
        "advancement_gate_passed": passed,
        "clearly_better_than_v75_v77_under_preregistered_gate": passed,
        "quarantine_retained": True,
        "runtime_promotion_authorized": False,
    }


def run_full_v79(
    preregistration: str | Path = V79_PREREGISTRATION,
) -> dict[str, Any]:
    prereg_path, prereg = load_preregistration_v79(preregistration)
    screen_path, screen = _authenticated_screen_v79(prereg)
    output = guard_output_v79(prereg["conditional_full_evaluation"]["output"], suffix=".json")
    _held_rows, rows = _load_rows_v79(prereg, screen=False)
    runtime, prefixes, answer_items = _bootstrap_v79(prereg, rows)
    device = torch.device(runtime.language.device)
    candidate_path = guard_input_v79(V79_CANDIDATE, "V79 candidate")
    model, metadata, digest = _load_candidate_v79(
        candidate_path,
        device=device,
        expected_sha256=screen["candidates"]["v79"]["sha256"],
        expected_artifact="v79_historical_relation_counterfactual_diagnostic_v1",
    )
    started = time.perf_counter()
    records = _evaluate_model(
        runtime=runtime,
        model=model,
        rows=rows,
        prefixes=prefixes,
        answer_items=answer_items,
        model_name="v79",
    )
    summary = summarize_records_v79(records)
    v75_types, baselines = _historical_baseline_types_v79(prereg)
    decision = full_decision_v79(summary, v75_types)
    report = {
        "artifact": "v79_relation_counterfactual_matched_full_v1",
        "preregistration": {
            "path": str(prereg_path.relative_to(PROJECT_ROOT)),
            "sha256": V79_PREREGISTRATION_SHA256,
            "authenticated_before_full_evaluation": True,
        },
        "screen_authorization": {
            "path": str(screen_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256_file(screen_path),
            "screen_passed": True,
            "candidate_sha256": screen["candidates"]["v79"]["sha256"],
        },
        "scope": {
            "historical_training_pool_only": True,
            "row_count": len(rows),
            "internal_held_scene_count": len({row.scene_id for row in rows}),
            "optimization_held_scene_overlap": 0,
            "correct_and_wrong_arms_matched_per_row": True,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
            "runtime_promotion_authorized": False,
            "checkpoint_published": False,
        },
        "candidate": {
            "path": str(candidate_path.relative_to(PROJECT_ROOT)),
            "sha256": digest,
            "metadata": metadata,
        },
        "authenticated_historical_baselines": baselines,
        "summary": summary,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    _atomic_create_v79(
        output,
        (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "event": "v79_full_complete",
                "output": str(output.relative_to(PROJECT_ROOT)),
                "correct": summary["correct_scene"]["correct"],
                "wrong": summary["wrong_scene"]["correct"],
                "advancement_gate_passed": decision["advancement_gate_passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "full"), required=True)
    parser.add_argument("--preregistration", default=V79_PREREGISTRATION)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stage == "screen":
        run_screen_v79(args.preregistration)
    else:
        run_full_v79(args.preregistration)
    return 0


__all__ = [
    "EXPECTED_SCREEN_ROWS",
    "EXPECTED_SCREEN_UNITS",
    "V79_CANDIDATE",
    "V79_FULL_OUTPUT",
    "V79_SCREEN_OUTPUT",
    "full_decision_v79",
    "run_full_v79",
    "run_screen_v79",
    "screen_decision_v79",
    "select_screen_rows_v79",
    "summarize_records_v79",
]


if __name__ == "__main__":
    raise SystemExit(main())
