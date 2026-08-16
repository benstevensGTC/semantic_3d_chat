"""One hard pair-disjoint V6.4 confirmation of the V6.3 attention reader.

This is a train-only screening fold, never a runtime-promotion experiment.
Three complete physical counterfactual pairs (six scenes, twelve QA units) are
held from gradient updates.  The remaining 28 units receive exactly three
balanced exposures.  All losses use full Hugging Face Gemma forwards and the
same symmetric two-scene by two-answer objective proven in V6.3.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import fixed_prefix_attention_reader_v6_3_evidence as v63e
from semantic_3d_chat.training import train_fixed_prefix_attention_reader_v6_3 as v63

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_attention_reader_v6_4"
CONFIG: Final[str] = (
    "configs/experiments/gemma4_v54_fixed_prefix_attention_reader_v6_4.yaml"
)
RESULT_REPORT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_attention_reader_v6_4_pair_disjoint_screen.json"
)
PROHIBITED_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_attention_reader_v6_4"
)
HELD_PAIR_IDS: Final[tuple[str, ...]] = (
    "pair_000010",
    "pair_000015",
    "pair_000017",
)
HELD_SCENE_IDS: Final[tuple[str, ...]] = (
    "scene_000021",
    "scene_000022",
    "scene_000031",
    "scene_000032",
    "scene_000035",
    "scene_000036",
)
EPOCHS: Final[int] = 3
UPDATES: Final[int] = 12
UNITS_PER_UPDATE: Final[int] = 7
TRAIN_UNIT_COUNT: Final[int] = 28
HELD_UNIT_COUNT: Final[int] = 12
LEARNING_RATE: Final[float] = 2e-5
RETENTION_WEIGHT: Final[float] = 0.25
GRADIENT_CLIP: Final[float] = 0.5
HARD_RUNTIME_SECONDS: Final[float] = 480.0
MAXIMUM_MPS_DRIVER_BYTES: Final[int] = 23_000_000_000


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _write_report(value: Mapping[str, Any]) -> Path:
    destination = _resolve(RESULT_REPORT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def split_pair_units(
    units: Sequence[v63.PairUnit],
) -> tuple[list[v63.PairUnit], list[v63.PairUnit]]:
    train = [unit for unit in units if unit.pair_id not in HELD_PAIR_IDS]
    held = [unit for unit in units if unit.pair_id in HELD_PAIR_IDS]
    train_scenes = {
        scene for unit in train for scene in (unit.first.scene_id, unit.second.scene_id)
    }
    held_scenes = {
        scene for unit in held for scene in (unit.first.scene_id, unit.second.scene_id)
    }
    if (
        len(train) != TRAIN_UNIT_COUNT
        or len(held) != HELD_UNIT_COUNT
        or {unit.pair_id for unit in held} != set(HELD_PAIR_IDS)
        or held_scenes != set(HELD_SCENE_IDS)
        or len(train_scenes) != 18
        or train_scenes.intersection(held_scenes)
        or {unit.key for unit in train}.intersection(unit.key for unit in held)
    ):
        raise ValueError("V6.4 hard pair-disjoint split changed")
    return train, held


def build_schedule(units: Sequence[v63.PairUnit]) -> list[tuple[v63.PairUnit, ...]]:
    if len(units) != TRAIN_UNIT_COUNT:
        raise ValueError("V6.4 schedule requires exactly 28 training units")
    schedule: list[tuple[v63.PairUnit, ...]] = []
    for epoch in range(EPOCHS):
        ordered = list(units)
        random.Random(v63.INITIALIZATION_SEED + 1000 + epoch).shuffle(ordered)
        schedule.extend(
            tuple(ordered[index : index + UNITS_PER_UPDATE])
            for index in range(0, TRAIN_UNIT_COUNT, UNITS_PER_UPDATE)
        )
    if len(schedule) != UPDATES or any(len(update) != UNITS_PER_UPDATE for update in schedule):
        raise ValueError("V6.4 12x7 schedule changed")
    counts = Counter(unit.key for update in schedule for unit in update)
    if set(counts) != {unit.key for unit in units} or set(counts.values()) != {EPOCHS}:
        raise ValueError("V6.4 schedule does not expose every training unit once per epoch")
    if any(unit.pair_id in HELD_PAIR_IDS for update in schedule for unit in update):
        raise ValueError("V6.4 held pair leaked into the optimizer schedule")
    return schedule


def schedule_diagnostics(schedule: Sequence[Sequence[v63.PairUnit]]) -> dict[str, Any]:
    rows = [
        {
            "update": update_index,
            "epoch": (update_index - 1) // 4 + 1,
            "within_epoch_update": (update_index - 1) % 4 + 1,
            "unit_keys": [list(unit.key) for unit in update],
            "answer_types": [unit.first.answer_type for unit in update],
        }
        for update_index, update in enumerate(schedule, start=1)
    ]
    counts = Counter(tuple(key) for row in rows for key in row["unit_keys"])
    return {
        "updates": len(rows),
        "epochs": EPOCHS,
        "units_per_update": UNITS_PER_UPDATE,
        "total_unit_exposures": sum(len(row["unit_keys"]) for row in rows),
        "unique_training_units": len(counts),
        "exposures_per_unit_distribution": dict(sorted(Counter(counts.values()).items())),
        "held_pair_ids_in_schedule": sorted(
            {key[0] for key in counts if key[0] in HELD_PAIR_IDS}
        ),
        "records": rows,
        "records_sha256": v63._canonical_hash(rows),
    }


def _family_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    records = metrics.get("records")
    if not isinstance(records, list):
        raise TypeError("V6.4 pair metrics lack raw records")
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("answer_type"), str):
            raise TypeError("V6.4 pair metric record lacks answer type")
        grouped[str(record["answer_type"])].append(record)
    result: dict[str, Any] = {}
    for family, members in sorted(grouped.items()):
        margins = [
            float(side["wrong_minus_correct_margin"])
            for member in members
            for side in member["sides"]
        ]
        softplus = [
            float(torch.nn.functional.softplus(torch.tensor(v63.MARGIN_TARGET - margin)))
            for margin in margins
        ]
        result[family] = {
            "unit_count": len(members),
            "side_count": len(margins),
            "positive_margin_sides": sum(value > 0.0 for value in margins),
            "complete_units": sum(bool(member["complete_unit"]) for member in members),
            "mean_margin": sum(margins) / len(margins),
            "mean_margin_softplus": sum(softplus) / len(softplus),
        }
    return result


def _family_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    before = _family_metrics(baseline)
    after = _family_metrics(candidate)
    if set(before) != set(after):
        raise ValueError("V6.4 answer-family inventory changed")
    deltas = {
        family: {
            "positive_margin_sides": after[family]["positive_margin_sides"]
            - before[family]["positive_margin_sides"],
            "complete_units": after[family]["complete_units"]
            - before[family]["complete_units"],
            "mean_margin": after[family]["mean_margin"] - before[family]["mean_margin"],
            "mean_margin_softplus": after[family]["mean_margin_softplus"]
            - before[family]["mean_margin_softplus"],
        }
        for family in before
    }
    return {"baseline": before, "candidate": after, "delta": deltas}


def _pair_delta(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "positive_margin_sides": int(candidate["positive_margin_sides"])
        - int(baseline["positive_margin_sides"]),
        "complete_units": int(candidate["complete_units"]) - int(baseline["complete_units"]),
        "mean_margin": float(candidate["mean_margin"]) - float(baseline["mean_margin"]),
        "mean_margin_softplus": float(candidate["mean_margin_softplus"])
        - float(baseline["mean_margin_softplus"]),
        "mean_correct_nll": float(candidate["mean_correct_nll"])
        - float(baseline["mean_correct_nll"]),
    }


def _check_deadline(started: float, *, phase: str) -> None:
    elapsed = time.perf_counter() - started
    if elapsed >= HARD_RUNTIME_SECONDS:
        raise TimeoutError(f"V6.4 exceeded hard 480-second ceiling during {phase}")


def _memory_metrics() -> dict[str, Any]:
    metrics = v63.v1.memory_metrics()
    driver = metrics.get("mps_driver_allocated_bytes")
    if driver is not None and int(driver) > MAXIMUM_MPS_DRIVER_BYTES:
        raise MemoryError("V6.4 exceeded the 23 GB MPS-driver safety ceiling")
    return metrics


def run_screen() -> dict[str, Any]:
    started = time.perf_counter()
    parent = v63e.authenticate_terminal_marker()
    if parent.get("continuation") != "v6_4_pair_disjoint_train_only_confirmation":
        raise ValueError("V6.4 lacks authenticated parent continuation authority")
    if _resolve(RESULT_REPORT).exists() or _resolve(PROHIBITED_CHECKPOINT).exists():
        raise FileExistsError("V6.4 single screen already ran or a prohibited checkpoint exists")

    audit = FileAccessAudit(
        v63.training_forbidden_roots(),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    failure: BaseException | None = None
    report: dict[str, Any] | None = None
    with audit:
        try:
            _check_deadline(started, phase="load")
            torch.manual_seed(v63.INITIALIZATION_SEED)
            bundle = v63.load_base_bundle(audit)
            units = v63.build_pair_units(v63.v1.load_training_records())
            train_units, held_units = split_pair_units(units)
            schedule = build_schedule(train_units)
            order = schedule_diagnostics(schedule)
            installation = v63.install_outer_residuals(bundle.language.model)
            bundle.installation = installation
            initial_hash = installation.state_sha256()
            retention = v63.v1.load_retention_corpus()[:UPDATES]
            teachers = v63._retention_teachers(bundle, retention)

            _check_deadline(started, phase="baseline_train")
            baseline_train = v63.evaluate_pair_units(bundle, train_units)
            _check_deadline(started, phase="baseline_held")
            baseline_held = v63.evaluate_pair_units(bundle, held_units)
            baseline_retention = v63.evaluate_retention(bundle, retention, teachers)
            optimizer = torch.optim.AdamW(
                installation.parameters(),
                lr=LEARNING_RATE,
                weight_decay=0.0,
                betas=(0.9, 0.999),
                eps=1e-8,
                foreach=False,
                fused=False,
            )
            trace: list[dict[str, Any]] = []
            for update_index, update_units in enumerate(schedule, start=1):
                _check_deadline(started, phase=f"update_{update_index}")
                optimizer.zero_grad(set_to_none=True)
                margins: list[float] = []
                objectives: list[float] = []
                scale = 1.0 / (2.0 * len(update_units))
                for unit in update_units:
                    for side in (0, 1):
                        correct, wrong = v63._side_tensors(bundle, unit, side)
                        side_loss, margin = v63.softplus_margin_side(correct, wrong)
                        if not torch.isfinite(side_loss) or not torch.isfinite(margin):
                            raise FloatingPointError("V6.4 encountered nonfinite pair loss")
                        (scale * side_loss).backward()
                        objectives.append(float(side_loss.detach().cpu()))
                        margins.append(float(margin.detach().cpu()))
                retention_kl = v63.v1.retention_kl_loss(
                    bundle,
                    retention[update_index - 1],
                    teachers[update_index - 1],
                ).clamp_min(0.0)
                if not torch.isfinite(retention_kl):
                    raise FloatingPointError("V6.4 retention KL is nonfinite")
                (RETENTION_WEIGHT * retention_kl).backward()
                preclip = float(
                    torch.nn.utils.clip_grad_norm_(installation.parameters(), GRADIENT_CLIP)
                    .detach()
                    .cpu()
                )
                if not math.isfinite(preclip) or preclip <= 0.0:
                    raise FloatingPointError("V6.4 gradient is nonfinite or zero")
                optimizer.step()
                installation.validate_state()
                installation.assert_only_outer_trainable(bundle.language.model)
                memory = _memory_metrics()
                item = {
                    "update": update_index,
                    "epoch": (update_index - 1) // 4 + 1,
                    "within_epoch_update": (update_index - 1) % 4 + 1,
                    "unit_keys": [list(unit.key) for unit in update_units],
                    "mean_preupdate_margin": sum(margins) / len(margins),
                    "mean_preupdate_side_objective": sum(objectives) / len(objectives),
                    "retention_index": update_index - 1,
                    "retention_kl": float(retention_kl.detach().cpu()),
                    "preclip_gradient_l2": preclip,
                    "adapter_state_sha256": installation.state_sha256(),
                    "mps_driver_allocated_bytes": memory.get("mps_driver_allocated_bytes"),
                }
                trace.append(item)
                print(
                    json.dumps(
                        {
                            "phase": "v6_4_pair_disjoint_attention_reader",
                            "update": update_index,
                            "updates": UPDATES,
                            "epoch": item["epoch"],
                            "mean_margin": item["mean_preupdate_margin"],
                            "preclip_gradient_l2": preclip,
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            _check_deadline(started, phase="candidate_train")
            candidate_train = v63.evaluate_pair_units(bundle, train_units)
            _check_deadline(started, phase="candidate_held")
            candidate_held = v63.evaluate_pair_units(bundle, held_units)
            candidate_retention = v63.evaluate_retention(bundle, retention, teachers)
            _check_deadline(started, phase="finalization")
            memory = _memory_metrics()
            train_delta = _pair_delta(baseline_train, candidate_train)
            held_delta = _pair_delta(baseline_held, candidate_held)
            checks = {
                "held_mean_margin_softplus_delta_at_most_minus_0_001": held_delta[
                    "mean_margin_softplus"
                ]
                <= -0.001,
                "held_mean_margin_delta_at_least_0_002": held_delta["mean_margin"] >= 0.002,
                "held_positive_margin_sides_nonworse": held_delta["positive_margin_sides"] >= 0,
                "held_complete_units_nonworse": held_delta["complete_units"] >= 0,
                "train_mean_margin_softplus_improved": train_delta["mean_margin_softplus"] < 0,
                "retention_mean_kl_at_most_0_005": candidate_retention["mean_kl_nats"] <= 0.005,
                "retention_maximum_kl_at_most_0_02": candidate_retention[
                    "maximum_kl_nats"
                ]
                <= 0.02,
                "retention_top1_exact": candidate_retention["top1_agreement"] == 1.0,
            }
            report = {
                "schema_version": 1,
                "artifact": f"{ARTIFACT}_pair_disjoint_screen",
                "status": "screen_pass" if all(checks.values()) else "screen_fail",
                "screen_pass": all(checks.values()),
                "sufficient_for_runtime_promotion": False,
                "promotion_authorized": False,
                "runtime_checkpoint_published": False,
                "parent_v6_3_terminal_sha256": parent["terminal_sha256"],
                "full_huggingface_forward": True,
                "target_modules": list(v63.TARGET_MODULES),
                "trainable_parameter_count": installation.parameter_count,
                "split": {
                    "held_physical_pair_ids": list(HELD_PAIR_IDS),
                    "held_scene_ids": list(HELD_SCENE_IDS),
                    "train_unit_count": len(train_units),
                    "held_unit_count": len(held_units),
                    "train_scene_count": len(
                        {
                            scene
                            for unit in train_units
                            for scene in (unit.first.scene_id, unit.second.scene_id)
                        }
                    ),
                    "held_scene_count": 6,
                    "physical_pair_disjoint": True,
                    "scene_disjoint": True,
                },
                "optimization": {
                    "optimizer": "AdamW",
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "updates": UPDATES,
                    "units_per_update": UNITS_PER_UPDATE,
                    "pair_unit_exposures": TRAIN_UNIT_COUNT * EPOCHS,
                    "gradient_clip_l2": GRADIENT_CLIP,
                    "initial_state_sha256": initial_hash,
                    "final_state_sha256": installation.state_sha256(),
                    "hard_runtime_seconds": HARD_RUNTIME_SECONDS,
                    "maximum_mps_driver_bytes": MAXIMUM_MPS_DRIVER_BYTES,
                },
                "train_order_diagnostics": order,
                "baseline_train": baseline_train,
                "candidate_train": candidate_train,
                "train_delta": train_delta,
                "train_answer_family_metrics": _family_deltas(
                    baseline_train, candidate_train
                ),
                "baseline_held": baseline_held,
                "candidate_held": candidate_held,
                "held_delta": held_delta,
                "held_answer_family_metrics": _family_deltas(
                    baseline_held, candidate_held
                ),
                "baseline_retention": baseline_retention,
                "candidate_retention": candidate_retention,
                "checks_before_audit": checks,
                "trace": trace,
                "trace_sha256": v63._canonical_hash(trace),
                "memory": memory,
                "elapsed_seconds": time.perf_counter() - started,
                "v6_2_down_projection_installed": False,
                "internal_validation_inputs_loaded": False,
                "deferred_or_final_inputs_loaded": False,
                "oracle_inputs_loaded": False,
            }
        except BaseException as exc:  # noqa: BLE001 - emit fail-closed terminal evidence
            failure = exc

    audit_result = v63._audit_summary(audit)
    if failure is not None:
        failure_report = {
            "schema_version": 1,
            "artifact": f"{ARTIFACT}_pair_disjoint_screen",
            "status": "hard_stop_failure",
            "screen_pass": False,
            "sufficient_for_runtime_promotion": False,
            "promotion_authorized": False,
            "runtime_checkpoint_published": False,
            "error_type": type(failure).__name__,
            "error": str(failure),
            "audit": audit_result,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_report(failure_report)
        raise failure
    if report is None:
        raise RuntimeError("V6.4 ended without a terminal report")
    report["audit"] = audit_result
    report["checks"] = {
        **report.pop("checks_before_audit"),
        "audit_clean": audit_result["passed"],
        "checkpoint_absent": not _resolve(PROHIBITED_CHECKPOINT).exists(),
        "completed_under_480_seconds": report["elapsed_seconds"] < HARD_RUNTIME_SECONDS,
    }
    report["screen_pass"] = all(report["checks"].values())
    report["status"] = "screen_pass" if report["screen_pass"] else "screen_fail"
    _write_report(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = run_screen()
    summary = {
        "artifact": report["artifact"],
        "status": report["status"],
        "screen_pass": report["screen_pass"],
        "promotion_authorized": False,
        "held_delta": report["held_delta"],
        "train_delta": report["train_delta"],
        "candidate_retention": {
            key: report["candidate_retention"][key]
            for key in ("mean_kl_nats", "maximum_kl_nats", "top1_agreement")
        },
        "checks": report["checks"],
        "audit": report["audit"],
        "elapsed_seconds": report["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["screen_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
