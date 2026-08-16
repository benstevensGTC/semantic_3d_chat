"""Authenticate and seal the negative V6.4 pair-disjoint screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import fixed_prefix_attention_reader_v6_3_evidence as v63e

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_attention_reader_v6_4"
RESULT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_attention_reader_v6_4_pair_disjoint_screen.json"
)
TERMINAL: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_attention_reader_v6_4_terminal.json"
)
SOURCE: Final[str] = (
    "src/semantic_3d_chat/evaluation/fixed_prefix_attention_reader_v6_4_evidence.py"
)
TEST: Final[str] = "tests/test_fixed_prefix_attention_reader_v6_4_evidence.py"
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
PINNED_SHA256: Final[dict[str, str]] = {
    "configs/experiments/gemma4_v54_fixed_prefix_attention_reader_v6_4.yaml": (
        "1241f01e587365c2f93e252e7fa9e65bc9df1a52b25dae81fa9cac11b5b823f6"
    ),
    "scripts/run_gemma4_v54_fixed_prefix_attention_reader_v6_4.sh": (
        "3bbbe648193a9c25b90f1fe33d13e229d1cfaee5bf48a788626bee48c08441b2"
    ),
    "src/semantic_3d_chat/training/train_fixed_prefix_attention_reader_v6_4.py": (
        "9c53f142c684cff0e99174624c28a8dcd0c06c6428f38000d6cfd1e8d1deddc3"
    ),
    "tests/test_fixed_prefix_attention_reader_v6_4.py": (
        "8b4e94cc3fef5e360d4d00fbf40171c59ccc615068ec356915aca7fd0540559a"
    ),
    RESULT: "a909c71e10c2cca5757556dd462132a499b09f05576bb11119bf1b7f424f0414",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"V6.4 duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V6.4 evidence missing or unsafe: {source}")
    value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError("V6.4 evidence must contain an object")
    return value


def _finite(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def authenticate_pinned_bytes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, expected in PINNED_SHA256.items():
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V6.4 pinned artifact missing or linked: {relative}")
        observed = _sha256(source)
        if observed != expected:
            raise ValueError(f"V6.4 pinned bytes changed: {relative}")
        result[relative] = observed
    return result


def _recompute_pair_metrics(metrics: object, *, expected_units: int) -> dict[str, Any]:
    records = metrics.get("records") if isinstance(metrics, Mapping) else None
    if not isinstance(records, list) or len(records) != expected_units:
        raise ValueError("V6.4 pair-record inventory changed")
    identities: set[tuple[str, str]] = set()
    scenes: set[str] = set()
    margins: list[float] = []
    correct_nlls: list[float] = []
    complete = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("V6.4 pair record is invalid")
        identity = (record.get("pair_id"), record.get("pair_question_key"))
        scene_ids = record.get("scene_ids")
        sides = record.get("sides")
        if (
            identity in identities
            or not all(isinstance(value, str) and value for value in identity)
            or not isinstance(scene_ids, list)
            or len(scene_ids) != 2
            or not all(isinstance(value, str) for value in scene_ids)
            or not isinstance(sides, list)
            or len(sides) != 2
        ):
            raise ValueError("V6.4 pair identity or side inventory changed")
        identities.add(identity)
        scenes.update(scene_ids)
        positives: list[bool] = []
        for side in sides:
            if not isinstance(side, Mapping) or not all(
                _finite(side.get(key))
                for key in ("correct_nll", "wrong_nll", "wrong_minus_correct_margin")
            ):
                raise ValueError("V6.4 pair side contains invalid metrics")
            correct = float(side["correct_nll"])
            wrong = float(side["wrong_nll"])
            margin = float(side["wrong_minus_correct_margin"])
            if correct < 0 or wrong < 0 or abs((wrong - correct) - margin) > 1e-6:
                raise ValueError("V6.4 raw pair margin changed")
            correct_nlls.append(correct)
            margins.append(margin)
            positives.append(margin > 0)
        expected_complete = all(positives)
        if record.get("complete_unit") is not expected_complete:
            raise ValueError("V6.4 complete-unit flag changed")
        complete += expected_complete
    import torch
    import torch.nn.functional as F

    softplus = [float(F.softplus(torch.tensor(0.5 - margin))) for margin in margins]
    derived = {
        "unit_count": expected_units,
        "side_count": expected_units * 2,
        "positive_margin_sides": sum(margin > 0 for margin in margins),
        "complete_units": complete,
        "mean_margin": sum(margins) / len(margins),
        "mean_margin_softplus": sum(softplus) / len(softplus),
        "mean_correct_nll": sum(correct_nlls) / len(correct_nlls),
        "records_sha256": _canonical_hash(records),
    }
    for key, expected in derived.items():
        observed = metrics.get(key)
        if isinstance(expected, float):
            if not _finite(observed) or abs(float(observed) - expected) > 1e-7:
                raise ValueError(f"V6.4 pair aggregate changed: {key}")
        elif observed != expected:
            raise ValueError(f"V6.4 pair aggregate changed: {key}")
    return {**derived, "identities": identities, "scenes": scenes}


def _validate_delta(
    stored: object, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float | int]:
    if not isinstance(stored, Mapping):
        raise TypeError("V6.4 delta is not a mapping")
    expected: dict[str, float | int] = {
        "positive_margin_sides": int(candidate["positive_margin_sides"])
        - int(baseline["positive_margin_sides"]),
        "complete_units": int(candidate["complete_units"]) - int(baseline["complete_units"]),
        "mean_margin": float(candidate["mean_margin"]) - float(baseline["mean_margin"]),
        "mean_margin_softplus": float(candidate["mean_margin_softplus"])
        - float(baseline["mean_margin_softplus"]),
        "mean_correct_nll": float(candidate["mean_correct_nll"])
        - float(baseline["mean_correct_nll"]),
    }
    for key, value in expected.items():
        observed = stored.get(key)
        if isinstance(value, float):
            if not _finite(observed) or abs(float(observed) - value) > 1e-7:
                raise ValueError(f"V6.4 stored delta changed: {key}")
        elif observed != value:
            raise ValueError(f"V6.4 stored delta changed: {key}")
    return expected


def authenticate_result() -> dict[str, Any]:
    parent = v63e.authenticate_terminal_marker()
    pinned = authenticate_pinned_bytes()
    result = _read_json(RESULT)
    if not (
        result.get("artifact") == f"{ARTIFACT}_pair_disjoint_screen"
        and result.get("status") == "screen_fail"
        and result.get("screen_pass") is False
        and result.get("sufficient_for_runtime_promotion") is False
        and result.get("promotion_authorized") is False
        and result.get("runtime_checkpoint_published") is False
        and result.get("parent_v6_3_terminal_sha256") == parent["terminal_sha256"]
        and result.get("full_huggingface_forward") is True
        and result.get("trainable_parameter_count") == 30_720
        and result.get("v6_2_down_projection_installed") is False
        and result.get("internal_validation_inputs_loaded") is False
        and result.get("deferred_or_final_inputs_loaded") is False
        and result.get("oracle_inputs_loaded") is False
        and _finite(result.get("elapsed_seconds"))
        and float(result["elapsed_seconds"]) < 480.0
    ):
        raise ValueError("V6.4 top-level result contract changed")
    split = result.get("split")
    if not (
        isinstance(split, Mapping)
        and split.get("held_physical_pair_ids") == list(HELD_PAIR_IDS)
        and split.get("held_scene_ids") == list(HELD_SCENE_IDS)
        and split.get("train_unit_count") == 28
        and split.get("held_unit_count") == 12
        and split.get("train_scene_count") == 18
        and split.get("held_scene_count") == 6
        and split.get("physical_pair_disjoint") is True
        and split.get("scene_disjoint") is True
    ):
        raise ValueError("V6.4 split contract changed")
    baseline_train = _recompute_pair_metrics(result.get("baseline_train"), expected_units=28)
    candidate_train = _recompute_pair_metrics(result.get("candidate_train"), expected_units=28)
    baseline_held = _recompute_pair_metrics(result.get("baseline_held"), expected_units=12)
    candidate_held = _recompute_pair_metrics(result.get("candidate_held"), expected_units=12)
    if (
        baseline_train["identities"] != candidate_train["identities"]
        or baseline_held["identities"] != candidate_held["identities"]
        or baseline_train["scenes"].intersection(baseline_held["scenes"])
        or baseline_held["scenes"] != set(HELD_SCENE_IDS)
    ):
        raise ValueError("V6.4 train/held raw-record separation changed")
    train_delta = _validate_delta(result.get("train_delta"), baseline_train, candidate_train)
    held_delta = _validate_delta(result.get("held_delta"), baseline_held, candidate_held)
    trace = result.get("trace")
    order = result.get("train_order_diagnostics")
    if not (
        isinstance(trace, list)
        and len(trace) == 12
        and result.get("trace_sha256") == _canonical_hash(trace)
        and isinstance(order, Mapping)
        and order.get("updates") == 12
        and order.get("epochs") == 3
        and order.get("units_per_update") == 7
        and order.get("total_unit_exposures") == 84
        and order.get("unique_training_units") == 28
        and order.get("exposures_per_unit_distribution") == {"3": 28}
        and order.get("held_pair_ids_in_schedule") == []
    ):
        raise ValueError("V6.4 train-order evidence changed")
    seen: dict[tuple[str, str], int] = {}
    for index, item in enumerate(trace, start=1):
        keys = item.get("unit_keys") if isinstance(item, Mapping) else None
        if (
            item.get("update") != index
            or not isinstance(keys, list)
            or len(keys) != 7
            or not _finite(item.get("preclip_gradient_l2"))
            or float(item["preclip_gradient_l2"]) <= 0.0
        ):
            raise ValueError("V6.4 optimizer trace changed")
        for raw in keys:
            key = tuple(raw) if isinstance(raw, list) else ()
            if len(key) != 2 or key[0] in HELD_PAIR_IDS:
                raise ValueError("V6.4 held or malformed unit entered optimizer trace")
            seen[key] = seen.get(key, 0) + 1
    if len(seen) != 28 or set(seen.values()) != {3}:
        raise ValueError("V6.4 optimizer unit exposure counts changed")
    retention = result.get("candidate_retention")
    audit = result.get("audit")
    if not (
        isinstance(retention, Mapping)
        and retention.get("example_count") == 12
        and _finite(retention.get("mean_kl_nats"))
        and float(retention["mean_kl_nats"]) <= 0.005
        and _finite(retention.get("maximum_kl_nats"))
        and float(retention["maximum_kl_nats"]) <= 0.02
        and retention.get("top1_agreement") == 1.0
        and isinstance(audit, Mapping)
        and audit.get("passed") is True
        and audit.get("forbidden_accesses") == []
        and audit.get("oracle_accessed") is False
        and audit.get("validation_deferred_or_final_accessed") is False
    ):
        raise ValueError("V6.4 retention or file audit changed")
    checks = {
        "held_mean_margin_softplus_delta_at_most_minus_0_001": held_delta[
            "mean_margin_softplus"
        ]
        <= -0.001,
        "held_mean_margin_delta_at_least_0_002": held_delta["mean_margin"] >= 0.002,
        "held_positive_margin_sides_nonworse": held_delta["positive_margin_sides"] >= 0,
        "held_complete_units_nonworse": held_delta["complete_units"] >= 0,
        "train_mean_margin_softplus_improved": train_delta["mean_margin_softplus"] < 0,
        "retention_mean_kl_at_most_0_005": retention["mean_kl_nats"] <= 0.005,
        "retention_maximum_kl_at_most_0_02": retention["maximum_kl_nats"] <= 0.02,
        "retention_top1_exact": retention["top1_agreement"] == 1.0,
        "audit_clean": True,
        "checkpoint_absent": not _resolve(PROHIBITED_CHECKPOINT).exists(),
        "completed_under_480_seconds": result["elapsed_seconds"] < 480.0,
    }
    if result.get("checks") != checks or all(checks.values()):
        raise ValueError("V6.4 negative gate decision changed")
    if _resolve(PROHIBITED_CHECKPOINT).exists():
        raise ValueError("V6.4 prohibited checkpoint exists")
    return {
        "passed": True,
        "status": "authenticated_negative_pair_disjoint_screen",
        "result_sha256": pinned[RESULT],
        "train_delta": train_delta,
        "held_delta": held_delta,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "audit_clean": True,
        "checkpoint_absent": True,
    }


def build_terminal() -> dict[str, Any]:
    result = authenticate_result()
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_terminal",
        "status": "failed_pair_disjoint_generalization_no_checkpoint_no_promotion",
        "result": result,
        "authenticator_source_sha256": _sha256(SOURCE),
        "authenticator_test_sha256": _sha256(TEST),
        "exact_attention_surface_continuation_authorized": False,
        "runtime_checkpoint_promotion_authorized": False,
        "runtime_checkpoint_exists": False,
        "internal_validation_consumed": False,
        "deferred_or_final_consumed": False,
        "oracle_consumed": False,
    }


def seal_terminal() -> tuple[Path, str]:
    destination = _resolve(TERMINAL)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("V6.4 terminal marker already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(build_terminal(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(payload).hexdigest()


def authenticate_terminal() -> dict[str, Any]:
    observed = _read_json(TERMINAL)
    expected = build_terminal()
    if observed != expected:
        raise ValueError("V6.4 terminal marker changed")
    return {
        "passed": True,
        "status": observed["status"],
        "terminal_sha256": _sha256(TERMINAL),
        "result_sha256": observed["result"]["result_sha256"],
        "runtime_checkpoint_promotion_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "authenticate"))
    args = parser.parse_args(argv)
    if args.command == "seal":
        path, digest = seal_terminal()
        result = {"passed": True, "path": str(path), "sha256": digest}
    else:
        result = authenticate_terminal()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
