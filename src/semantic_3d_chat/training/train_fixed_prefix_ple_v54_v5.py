"""Run the single preregistered V5 scene-selective fixed-prefix reader arm."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v5_preregistration import (
    ARTIFACT,
    CONFIG,
    OUTPUT_CHECKPOINT,
    PREREGISTRATION,
    RESULT_REPORT,
    SMOKE_REPORT,
    V4_SMOKE_SHA256,
    authenticate_preregistration,
)
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1
from semantic_3d_chat.training import train_fixed_prefix_ple_v54_v4 as v4

_SEED: Final[int] = 720054
_UPDATES: Final[int] = 80
_PAIR_CYCLES: Final[int] = 2
_PAIR_MARGIN: Final[float] = 0.5
_PAIR_CE_WEIGHT: Final[float] = 0.5
_PAIR_HINGE_WEIGHT: Final[float] = 4.0
_BROAD_CE_WEIGHT: Final[float] = 0.5
_RETENTION_KL_WEIGHT: Final[float] = 0.5
_WARMUP_UPDATES: Final[int] = 8
_PEAK_LR: Final[float] = 1e-4
_MIN_LR: Final[float] = 1e-5


@dataclass(frozen=True)
class V5Update:
    pair: tuple[v1.ReaderRecord, v1.ReaderRecord]
    broad: tuple[v1.ReaderRecord, ...]


def symmetric_pair_objective(
    correct_nll: torch.Tensor,
    wrong_nll: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the exact side-swap-symmetric V5 pair loss."""

    if correct_nll.shape != (2,) or wrong_nll.shape != (2,):
        raise ValueError("PLE-V54 V5 pair objective requires exactly two sides")
    if not torch.isfinite(correct_nll).all() or not torch.isfinite(wrong_nll).all():
        raise ValueError("PLE-V54 V5 pair objective received NaN or infinity")
    margins = wrong_nll - correct_nll
    hinge = F.relu(_PAIR_MARGIN - margins)
    answer_ce = correct_nll.mean()
    loss = _PAIR_CE_WEIGHT * answer_ce + _PAIR_HINGE_WEIGHT * hinge.mean()
    return loss, {
        "correct_answer_ce": answer_ce,
        "wrong_prefix_hinge": hinge.mean(),
        "wrong_prefix_margins": margins,
    }


def _side_pair_loss(
    bundle: v1.ReaderBundle,
    row: v1.ReaderRecord,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one side's additive half of the symmetric objective."""

    if not row.changed or row.paired_scene_id is None:
        raise ValueError("PLE-V54 V5 pair side requires a paired changed row")
    nll = v4.streamed_answer_nlls(
        bundle,
        (
            (bundle.prefixes[row.scene_id], row),
            (bundle.prefixes[row.paired_scene_id], row),
        ),
    )
    correct, wrong = nll[0], nll[1]
    margin = wrong - correct
    hinge = F.relu(_PAIR_MARGIN - margin)
    # Summing this expression over both sides equals
    # 0.5*mean(corrects) + 4.0*mean(hinges) exactly.
    loss = 0.25 * correct + 2.0 * hinge
    return loss, {
        "correct_nll": float(correct.detach().cpu()),
        "wrong_nll": float(wrong.detach().cpu()),
        "margin": float(margin.detach().cpu()),
        "hinge": float(hinge.detach().cpu()),
    }


def _pair_units(
    records: Sequence[v1.ReaderRecord],
) -> tuple[list[tuple[v1.ReaderRecord, v1.ReaderRecord]], list[v1.ReaderRecord]]:
    grouped: defaultdict[tuple[str, str], list[v1.ReaderRecord]] = defaultdict(list)
    broad: list[v1.ReaderRecord] = []
    for row in records:
        if row.changed:
            if row.pair_id is None or row.pair_question_key is None:
                raise ValueError("PLE-V54 V5 changed row lacks pair identity")
            grouped[(row.pair_id, row.pair_question_key)].append(row)
        else:
            broad.append(row)
    units: list[tuple[v1.ReaderRecord, v1.ReaderRecord]] = []
    for key in sorted(grouped):
        sides = sorted(grouped[key], key=lambda row: (row.role or "", row.scene_id))
        if (
            len(sides) != 2
            or len({row.scene_id for row in sides}) != 2
            or len({row.answer for row in sides}) != 2
            or len({row.question for row in sides}) != 1
            or sides[0].paired_scene_id != sides[1].scene_id
            or sides[1].paired_scene_id != sides[0].scene_id
        ):
            raise ValueError(f"PLE-V54 V5 pair unit is invalid: {key}")
        units.append((sides[0], sides[1]))
    if len(units) != 40 or len(broad) != 496:
        raise ValueError("PLE-V54 V5 training inventory changed")
    return units, broad


def build_v5_schedule(records: Sequence[v1.ReaderRecord]) -> list[V5Update]:
    """Build two full pair passes and one complete broad-row pass."""

    units, broad = _pair_units(records)
    rng = random.Random(_SEED)
    pair_schedule: list[tuple[v1.ReaderRecord, v1.ReaderRecord]] = []
    for _cycle in range(_PAIR_CYCLES):
        cycle = list(units)
        rng.shuffle(cycle)
        pair_schedule.extend(cycle)
    rng.shuffle(broad)
    counts = [6] * 64 + [7] * 16
    if sum(counts) != len(broad) or len(pair_schedule) != _UPDATES:
        raise AssertionError("PLE-V54 V5 schedule arithmetic changed")
    schedule: list[V5Update] = []
    offset = 0
    for pair, count in zip(pair_schedule, counts, strict=True):
        schedule.append(V5Update(pair=pair, broad=tuple(broad[offset : offset + count])))
        offset += count
    pair_counts = Counter(
        (row.pair_id, row.pair_question_key) for update in schedule for row in update.pair[:1]
    )
    broad_keys = [(row.scene_id, row.question_id) for update in schedule for row in update.broad]
    if (
        len(schedule) != _UPDATES
        or set(pair_counts.values()) != {_PAIR_CYCLES}
        or len(pair_counts) != 40
        or len(broad_keys) != 496
        or len(set(broad_keys)) != 496
    ):
        raise ValueError("PLE-V54 V5 schedule coverage changed")
    return schedule


def learning_rate(update: int) -> float:
    if isinstance(update, bool) or not isinstance(update, int) or not 1 <= update <= _UPDATES:
        raise ValueError("PLE-V54 V5 update must be in [1, 80]")
    if update <= _WARMUP_UPDATES:
        return _PEAK_LR * update / _WARMUP_UPDATES
    progress = (update - _WARMUP_UPDATES) / (_UPDATES - _WARMUP_UPDATES)
    return _MIN_LR + 0.5 * (_PEAK_LR - _MIN_LR) * (1.0 + math.cos(math.pi * progress))


def _activate_v5() -> None:
    v1.ARTIFACT = ARTIFACT
    v1.CONFIG = CONFIG
    v1.PREREGISTRATION = PREREGISTRATION
    v1.SMOKE_REPORT = SMOKE_REPORT
    v1.RESULT_REPORT = RESULT_REPORT
    v1.OUTPUT_CHECKPOINT = OUTPUT_CHECKPOINT
    v1.authenticate_preregistration = authenticate_preregistration
    v1.answer_nlls = v4.streamed_answer_nlls
    v1.evaluate_teacher_forcing = v4.evaluate_teacher_forcing_streamed
    v1.retention_logits = v4.bounded_retention_logits


def _validate_config() -> dict[str, Any]:
    raw = yaml.safe_load(v1._resolve(CONFIG).read_text(encoding="utf-8"))
    expected = {
        "artifact": ARTIFACT,
        "rank": 4,
        "alpha": 8.0,
        "updates": _UPDATES,
        "pair_cycles": _PAIR_CYCLES,
        "pair_margin": _PAIR_MARGIN,
        "pair_hinge_weight": _PAIR_HINGE_WEIGHT,
        "broad_weight": _BROAD_CE_WEIGHT,
        "retention_weight": _RETENTION_KL_WEIGHT,
        "checkpoint_selection": "final_state_after_update_80_only",
        "best_loss_selection": False,
        "post_hoc_state_selection": False,
    }
    observed = {
        "artifact": raw["experiment"]["artifact"],
        "rank": raw["adapter"]["rank"],
        "alpha": raw["adapter"]["alpha"],
        "updates": raw["optimization"]["maximum_updates"],
        "pair_cycles": raw["optimization"]["pair_cycles"],
        "pair_margin": raw["objective"]["wrong_prefix_margin_nats_per_token"],
        "pair_hinge_weight": raw["objective"]["symmetric_wrong_prefix_hinge_weight"],
        "broad_weight": raw["objective"]["broad_answer_ce_weight"],
        "retention_weight": raw["objective"]["retention_kl_weight"],
        "checkpoint_selection": raw["schedule"]["checkpoint_selection"],
        "best_loss_selection": raw["schedule"]["best_loss_selection"],
        "post_hoc_state_selection": raw["schedule"]["post_hoc_state_selection"],
    }
    if observed != expected:
        raise ValueError(f"PLE-V54 V5 config changed: {observed} != {expected}")
    return raw


def structural_preflight() -> dict[str, Any]:
    _activate_v5()
    _validate_config()
    result = v1.structural_preflight()
    schedule = build_v5_schedule(v1.load_training_records())
    result.update(
        {
            "artifact": ARTIFACT,
            "updates": len(schedule),
            "pair_cycles": _PAIR_CYCLES,
            "broad_rows_consumed": sum(len(update.broad) for update in schedule),
            "internal_gates_unchanged_from_v4": True,
            "deferred_holdout_accessed": False,
            "final_scenes_accessed": False,
            "passed": bool(result["passed"] and len(schedule) == 80),
        }
    )
    return result


def objective_gradient_smoke() -> dict[str, Any]:
    _activate_v5()
    if v1._resolve(SMOKE_REPORT).exists():
        raise FileExistsError("PLE-V54 V5 smoke report already exists")
    preflight = structural_preflight()
    if preflight["passed"] is not True:
        raise RuntimeError("PLE-V54 V5 structural preflight failed")
    started = time.perf_counter()
    bundle = v1._load_bundle(gradient_checkpointing=True)
    schedule = build_v5_schedule(v1.load_training_records())
    corpus = v1.load_retention_corpus()
    teachers = v1.retention_baseline(bundle, corpus[:1])
    update = schedule[0]
    bundle.installation.train()
    bundle.language.model.zero_grad(set_to_none=True)
    side_losses: list[float] = []
    margins: list[float] = []
    for row in update.pair:
        loss, diagnostics = _side_pair_loss(bundle, row)
        loss.backward()
        side_losses.append(float(loss.detach().cpu()))
        margins.append(diagnostics["margin"])
    broad = v4.streamed_answer_nlls(
        bundle, ((bundle.prefixes[update.broad[0].scene_id], update.broad[0]),)
    ).mean()
    (_BROAD_CE_WEIGHT * broad).backward()
    retention = v1.retention_kl_loss(bundle, corpus[0], teachers[0])
    (_RETENTION_KL_WEIGHT * retention).backward()
    gradients = bundle.installation.gradient_norms()
    bundle.installation.validate_state()
    memory = v1.memory_metrics()
    driver = memory["mps_driver_allocated_bytes"]
    retention_value = float(retention.detach().cpu())
    passed = (
        all(math.isfinite(value) for value in (*side_losses, *margins))
        and math.isfinite(float(broad.detach().cpu()))
        and abs(retention_value) <= 1e-5
        and math.isfinite(float(gradients["total_l2"]))
        and float(gradients["total_l2"]) > 0.0
        and (driver is None or driver <= 25_000_000_000)
    )
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_objective_gradient_smoke",
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "pair_side_losses": side_losses,
        "pair_margins": margins,
        "broad_nll": float(broad.detach().cpu()),
        "retention_self_kl": retention_value,
        "gradient_l2": gradients["total_l2"],
        "gradient_by_module": gradients["by_module"],
        "trainable_parameter_count": bundle.installation.parameter_count,
        "memory": memory,
        "elapsed_seconds": time.perf_counter() - started,
        "inherited_v4_tail_equivalence_smoke_sha256": V4_SMOKE_SHA256,
        "fixed_prefix_shape": [1, 258, 1536],
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "deferred_holdout_accessed": False,
        "final_scenes_accessed": False,
        "preregistration_sha256": v1.sha256_file(PREREGISTRATION),
    }
    v1._atomic_create_json(SMOKE_REPORT, report)
    return report


def _teacher_checks(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    retention: Mapping[str, Any],
) -> dict[str, bool]:
    """The V4 gates, repeated literally and never weakened."""

    return {
        "validation_answer_nll_improvement": (
            baseline["answer_nll_mean"] - candidate["answer_nll_mean"] >= 0.03
        ),
        "changed_wrong_prefix_positive_margin_rate": (
            candidate["changed_positive_margin_rate"] >= 0.65
        ),
        "changed_wrong_prefix_positive_margin_rate_delta": (
            candidate["changed_positive_margin_rate"] - baseline["changed_positive_margin_rate"]
            >= 0.10
        ),
        "changed_pair_complete_unit_delta": (
            candidate["changed_complete_units"] - baseline["changed_complete_units"] >= 3
        ),
        "retention_mean_ce_increase": retention["mean_ce_increase_nats"] <= 0.03,
        "retention_mean_kl": retention["mean_kl_nats"] <= 0.02,
        "retention_next_token_top1_agreement": (retention["next_token_top1_agreement"] >= 0.98),
    }


def train_and_gate() -> dict[str, Any]:
    _activate_v5()
    if v1._resolve(RESULT_REPORT).exists() or v1._resolve(OUTPUT_CHECKPOINT).exists():
        raise FileExistsError("PLE-V54 V5 terminal result or checkpoint already exists")
    smoke = v1._read_json(SMOKE_REPORT)
    if smoke.get("passed") is not True or smoke.get("preregistration_sha256") != v1.sha256_file(
        PREREGISTRATION
    ):
        raise ValueError("PLE-V54 V5 requires its matching passing gradient smoke")
    if structural_preflight()["passed"] is not True:
        raise RuntimeError("PLE-V54 V5 structural preflight failed")

    started = time.perf_counter()
    torch.manual_seed(_SEED)
    random.seed(_SEED)
    bundle = v1._load_bundle(gradient_checkpointing=True)
    train_rows = v1.load_training_records()
    validation_rows = v1.load_validation_records()
    corpus = v1.load_retention_corpus()
    teachers = v1.retention_baseline(bundle, corpus)
    baseline_teacher = v4.evaluate_teacher_forcing_streamed(bundle, validation_rows)
    baseline_retention = v1.evaluate_retention(bundle, corpus, teachers)
    schedule = build_v5_schedule(train_rows)
    optimizer = torch.optim.AdamW(
        bundle.installation.parameters(), lr=learning_rate(1), weight_decay=0.0
    )
    trace: list[dict[str, Any]] = []
    maximum_gradient = 0.0
    bundle.installation.train()

    for update_index, update in enumerate(schedule, start=1):
        current_lr = learning_rate(update_index)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        side_losses: list[float] = []
        margins: list[float] = []
        for row in update.pair:
            side_loss, diagnostics = _side_pair_loss(bundle, row)
            side_loss.backward()
            side_losses.append(float(side_loss.detach().cpu()))
            margins.append(diagnostics["margin"])
        broad_nlls: list[float] = []
        for row in update.broad:
            broad = v4.streamed_answer_nlls(bundle, ((bundle.prefixes[row.scene_id], row),)).mean()
            (_BROAD_CE_WEIGHT * broad / len(update.broad)).backward()
            broad_nlls.append(float(broad.detach().cpu()))
        retention_index = (update_index - 1) % len(corpus)
        retention = v1.retention_kl_loss(bundle, corpus[retention_index], teachers[retention_index])
        (_RETENTION_KL_WEIGHT * retention).backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(bundle.installation.parameters(), 1.0).detach().cpu()
        )
        if not math.isfinite(gradient) or gradient <= 0.0:
            raise RuntimeError("PLE-V54 V5 gradient norm is invalid")
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        bundle.installation.validate_state()
        item = {
            "update": update_index,
            "learning_rate": current_lr,
            "mean_pair_side_loss": sum(side_losses) / len(side_losses),
            "mean_preupdate_changed_margin": sum(margins) / len(margins),
            "mean_broad_nll": sum(broad_nlls) / len(broad_nlls),
            "broad_row_count": len(broad_nlls),
            "retention_kl": float(retention.detach().cpu()),
            "preclip_gradient_l2": gradient,
            "adapter_state_sha256": bundle.installation.state_sha256(),
        }
        trace.append(item)
        print(
            json.dumps(
                {"phase": "ple_v54_v5_train", "updates": len(schedule), **item},
                sort_keys=True,
            ),
            flush=True,
        )

    bundle.installation.eval()
    candidate_teacher = v4.evaluate_teacher_forcing_streamed(bundle, validation_rows)
    candidate_retention = v1.evaluate_retention(bundle, corpus, teachers)
    checks = _teacher_checks(baseline_teacher, candidate_teacher, candidate_retention)
    greedy = None
    if all(checks.values()):
        greedy = v1.evaluate_greedy(bundle, validation_rows)
    checks["greedy_exact_accuracy_delta"] = bool(
        greedy is not None and greedy["exact_accuracy_delta"] >= 0.02
    )
    passed = all(checks.values())
    selection = {
        "baseline_teacher": baseline_teacher,
        "candidate_teacher": candidate_teacher,
        "baseline_retention": baseline_retention,
        "candidate_retention": candidate_retention,
        "greedy": greedy,
        "checks": checks,
        "passed": passed,
    }
    checkpoint = v1._publish_checkpoint(bundle, selection) if passed else None
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_terminal_result",
        "status": "passed_checkpoint_published" if passed else "failed_no_checkpoint",
        "passed": passed,
        "promotion_eligible": passed,
        "checkpoint_published": checkpoint is not None,
        "checkpoint": checkpoint,
        "preregistration_sha256": v1.sha256_file(PREREGISTRATION),
        "smoke_report_sha256": v1.sha256_file(SMOKE_REPORT),
        "training": {
            "updates": len(schedule),
            "pair_cycles": _PAIR_CYCLES,
            "broad_rows_consumed_exactly_once": 496,
            "trainable_parameter_count": bundle.installation.parameter_count,
            "maximum_preclip_gradient_l2": maximum_gradient,
            "initial_trace": trace[:3],
            "milestone_trace": [trace[index - 1] for index in (20, 40, 60, 80)],
            "final_trace": trace[-3:],
            "trace_sha256": v1._canonical_hash(trace),
            "final_adapter_state_sha256": bundle.installation.state_sha256(),
        },
        "selection": selection,
        "fixed_prefix": {
            "shape": [1, 258, 1536],
            "computed_before_question": True,
            "same_prefix_for_unchanged_scene": True,
            "question_dependent_retrieval": False,
            "all_scene_latents_present": True,
        },
        "runtime_leakage": {
            "environmental_text_inputs": [],
            "oracle_runtime_access": False,
            "training_qa_not_in_runtime_checkpoint": True,
            "validation_answers_not_in_runtime_checkpoint": True,
        },
        "deferred_holdout": {
            "scene_ids": [f"scene_{index:06d}" for index in range(57, 63)],
            "accessed": False,
            "required_next_only_if_internal_passed": True,
        },
        "final_scenes_000025_through_000030_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "memory": v1.memory_metrics(),
    }
    v1._atomic_create_json(RESULT_REPORT, report)
    if not passed and v1._resolve(OUTPUT_CHECKPOINT).exists():
        raise RuntimeError("PLE-V54 V5 failed run unexpectedly published a checkpoint")
    return report


def authenticate_result() -> dict[str, Any]:
    _activate_v5()
    return v1.authenticate_result()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "train", "authenticate"))
    mode = parser.parse_args(argv).mode
    result = {
        "preflight": structural_preflight,
        "smoke": objective_gradient_smoke,
        "train": train_and_gate,
        "authenticate": authenticate_result,
    }[mode]()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
