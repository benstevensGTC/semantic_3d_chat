"""Fail-closed authentication for the completed V6.3 train-only pilot."""

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

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_attention_reader_v6_3"
GRADIENT_REPORT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_attention_reader_v6_3_gradient_screen.json"
)
PILOT_REPORT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_attention_reader_v6_3_pilot.json"
)
TERMINAL_MARKER: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_attention_reader_v6_3_terminal.json"
)
AUTHENTICATOR_SOURCE: Final[str] = (
    "src/semantic_3d_chat/evaluation/fixed_prefix_attention_reader_v6_3_evidence.py"
)
AUTHENTICATOR_TEST: Final[str] = "tests/test_fixed_prefix_attention_reader_v6_3_evidence.py"
PROHIBITED_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_attention_reader_v6_3"
)

TARGET_MODULES: Final[tuple[str, ...]] = (
    "model.language_model.layers.13.self_attn.k_proj",
    "model.language_model.layers.13.self_attn.v_proj",
    "model.language_model.layers.14.self_attn.k_proj",
    "model.language_model.layers.14.self_attn.v_proj",
)

PINNED_V6_3_SHA256: Final[dict[str, str]] = {
    "configs/experiments/gemma4_v54_fixed_prefix_attention_reader_v6_3.yaml": (
        "fd2d76f7742d9429f73c564c2ec5fa9d3e6e665f3b15e764bb15b0292c617399"
    ),
    "scripts/run_gemma4_v54_fixed_prefix_attention_reader_v6_3.sh": (
        "106d7e61f0bc00fa4df7d5371aca6c893e0ee7844869cc4300f6e404d897e5c1"
    ),
    "src/semantic_3d_chat/training/train_fixed_prefix_attention_reader_v6_3.py": (
        "1375d47453b4d4609d68837bcdabedca934fcb2f66e51382465ec19875c7a488"
    ),
    "tests/test_fixed_prefix_attention_reader_v6_3.py": (
        "b838551c798d716b49c269f0e4b64b1e77b55b21b390e5730a6ab046c83add22"
    ),
    GRADIENT_REPORT: "93f4fb452e51eae425ca412f92b150a40ef59cb255cba30da531e271f9a45354",
    PILOT_REPORT: "43fbce25b0b1566ef73bc0ba8e0440f218f388497605359e26f97af306b3dc67",
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
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"V6.3 evidence contains duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V6.3 evidence missing or unsafe: {source}")
    value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"V6.3 evidence must be an object: {source}")
    return value


def _finite(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def authenticate_pinned_bytes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in PINNED_V6_3_SHA256.items():
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V6.3 pinned artifact missing or linked: {relative}")
        digest = _sha256(source)
        if digest != expected:
            raise ValueError(f"V6.3 pinned bytes changed: {relative}: {digest} != {expected}")
        observed[relative] = digest
    return observed


def authenticate_gradient_report() -> dict[str, Any]:
    report = _read_json(GRADIENT_REPORT)
    gradients = report.get("gradients")
    by_module = gradients.get("by_module") if isinstance(gradients, Mapping) else None
    identity = report.get("exact_zero_initialization")
    audit = report.get("audit")
    if not (
        report.get("artifact") == f"{ARTIFACT}_gradient_screen"
        and report.get("status") == "passed"
        and report.get("passed") is True
        and report.get("full_huggingface_forward") is True
        and report.get("optimizer_constructed") is False
        and report.get("optimizer_steps") == 0
        and report.get("target_modules") == list(TARGET_MODULES)
        and report.get("trainable_parameter_count") == 30_720
        and report.get("all_output_factor_gradients_nonzero") is True
        and report.get("all_input_factor_gradients_exact_zero_at_zero_output_init") is True
        and report.get("v6_2_down_projection_installed") is False
        and report.get("runtime_checkpoint_published") is False
        and isinstance(identity, Mapping)
        and identity.get("answer_logits_bit_exact") is True
        and identity.get("answer_nll_bit_exact") is True
        and identity.get("state_unchanged") is True
        and identity.get("state_sha256_before_backward")
        == identity.get("state_sha256_after_backward")
        and isinstance(by_module, Mapping)
        and set(by_module) == set(TARGET_MODULES)
        and _finite(gradients.get("total_l2"))
        and float(gradients["total_l2"]) > 0.0
        and isinstance(audit, Mapping)
        and audit.get("passed") is True
        and audit.get("forbidden_accesses") == []
        and audit.get("oracle_accessed") is False
        and audit.get("validation_deferred_or_final_accessed") is False
    ):
        raise ValueError("V6.3 gradient-screen contract changed")
    for name in TARGET_MODULES:
        values = by_module[name]
        if not (
            isinstance(values, Mapping)
            and values.get("residual_a") == 0.0
            and _finite(values.get("residual_b"))
            and float(values["residual_b"]) > 0.0
        ):
            raise ValueError(f"V6.3 gradient evidence changed: {name}")
    return {
        "passed": True,
        "sha256": _sha256(GRADIENT_REPORT),
        "trainable_parameter_count": 30_720,
        "gradient_l2": float(gradients["total_l2"]),
    }


def _derive_pair_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    records = metrics.get("records")
    if not isinstance(records, list) or len(records) != 40:
        raise ValueError("V6.3 pair records changed")
    identities: set[tuple[str, str]] = set()
    margins: list[float] = []
    correct_nlls: list[float] = []
    complete = 0
    for row in records:
        if not isinstance(row, Mapping):
            raise TypeError("V6.3 pair record is invalid")
        identity = (row.get("pair_id"), row.get("pair_question_key"))
        sides = row.get("sides")
        if (
            not all(isinstance(value, str) and value for value in identity)
            or identity in identities
            or not isinstance(sides, list)
            or len(sides) != 2
        ):
            raise ValueError("V6.3 pair-unit identity or side count changed")
        identities.add(identity)
        side_positive: list[bool] = []
        for side in sides:
            if not isinstance(side, Mapping) or not all(
                _finite(side.get(key))
                for key in ("correct_nll", "wrong_nll", "wrong_minus_correct_margin")
            ):
                raise ValueError("V6.3 pair side contains a nonfinite value")
            correct = float(side["correct_nll"])
            wrong = float(side["wrong_nll"])
            margin = float(side["wrong_minus_correct_margin"])
            if correct < 0.0 or wrong < 0.0 or abs((wrong - correct) - margin) > 1e-6:
                raise ValueError("V6.3 pair-side derived margin changed")
            margins.append(margin)
            correct_nlls.append(correct)
            side_positive.append(margin > 0.0)
        expected_complete = all(side_positive)
        if row.get("complete_unit") is not expected_complete:
            raise ValueError("V6.3 complete-unit flag changed")
        complete += expected_complete
    import torch
    import torch.nn.functional as F

    softplus = [float(F.softplus(torch.tensor(0.5 - margin))) for margin in margins]
    return {
        "unit_count": 40,
        "side_count": 80,
        "positive_margin_sides": sum(value > 0.0 for value in margins),
        "complete_units": complete,
        "mean_margin": sum(margins) / 80,
        "mean_margin_softplus": sum(softplus) / 80,
        "mean_correct_nll": sum(correct_nlls) / 80,
        "records_sha256": _canonical_hash(records),
    }


def _same(left: object, right: object, *, atol: float = 1e-9) -> bool:
    return _finite(left) and _finite(right) and abs(float(left) - float(right)) <= atol


def authenticate_pilot_report() -> dict[str, Any]:
    report = _read_json(PILOT_REPORT)
    audit = report.get("audit")
    optimization = report.get("optimization")
    trace = report.get("trace")
    baseline = report.get("baseline_pair_metrics")
    candidate = report.get("candidate_pair_metrics")
    if not (
        report.get("artifact") == f"{ARTIFACT}_pilot"
        and report.get("status") == "diagnostic_pass"
        and report.get("diagnostic_pass") is True
        and report.get("promotion_authorized") is False
        and report.get("runtime_checkpoint_published") is False
        and report.get("full_huggingface_forward") is True
        and report.get("target_modules") == list(TARGET_MODULES)
        and report.get("trainable_parameter_count") == 30_720
        and report.get("v6_2_down_projection_installed") is False
        and report.get("validation_inputs_loaded") is False
        and report.get("deferred_or_final_inputs_loaded") is False
        and report.get("oracle_inputs_loaded") is False
        and isinstance(audit, Mapping)
        and audit.get("passed") is True
        and audit.get("forbidden_accesses") == []
        and audit.get("oracle_accessed") is False
        and audit.get("validation_deferred_or_final_accessed") is False
        and isinstance(optimization, Mapping)
        and optimization.get("optimizer") == "AdamW"
        and optimization.get("updates") == 8
        and optimization.get("pair_units_per_update") == 5
        and optimization.get("total_pair_units") == 40
        and optimization.get("all_pair_units_consumed_exactly_once") is True
        and isinstance(trace, list)
        and len(trace) == 8
        and report.get("trace_sha256") == _canonical_hash(trace)
        and isinstance(baseline, Mapping)
        and isinstance(candidate, Mapping)
    ):
        raise ValueError("V6.3 pilot top-level contract changed")
    seen: set[tuple[str, str]] = set()
    for update_index, item in enumerate(trace, start=1):
        units = item.get("pair_units") if isinstance(item, Mapping) else None
        if (
            item.get("update") != update_index
            or item.get("pair_unit_count") != 5
            or not isinstance(units, list)
            or len(units) != 5
            or not _finite(item.get("preclip_gradient_l2"))
            or float(item["preclip_gradient_l2"]) <= 0.0
        ):
            raise ValueError("V6.3 pilot trace changed")
        for unit in units:
            identity = (unit.get("pair_id"), unit.get("pair_question_key"))
            if identity in seen or not all(isinstance(value, str) for value in identity):
                raise ValueError("V6.3 pilot did not consume each unit once")
            seen.add(identity)
    if len(seen) != 40:
        raise ValueError("V6.3 pilot unit coverage changed")
    for stored in (baseline, candidate):
        derived = _derive_pair_metrics(stored)
        for key, expected in derived.items():
            observed = stored.get(key)
            if isinstance(expected, float):
                if not _same(observed, expected, atol=1e-7):
                    raise ValueError(f"V6.3 pair aggregate changed: {key}")
            elif observed != expected:
                raise ValueError(f"V6.3 pair aggregate changed: {key}")
    delta = report.get("pair_metric_delta")
    if not isinstance(delta, Mapping):
        raise ValueError("V6.3 pair delta is missing")
    for key in (
        "positive_margin_sides",
        "complete_units",
        "mean_margin",
        "mean_margin_softplus",
        "mean_correct_nll",
    ):
        expected = float(candidate[key]) - float(baseline[key])
        if not _same(delta.get(key), expected, atol=1e-7):
            raise ValueError(f"V6.3 pair delta changed: {key}")
    retention = report.get("candidate_retention")
    if not (
        isinstance(retention, Mapping)
        and retention.get("example_count") == 8
        and _finite(retention.get("mean_kl_nats"))
        and _finite(retention.get("maximum_kl_nats"))
        and float(retention["mean_kl_nats"]) <= 0.02
        and retention.get("top1_agreement") == 1.0
    ):
        raise ValueError("V6.3 retention evidence changed")
    expected_checks = {
        "mean_margin_softplus_improved": candidate["mean_margin_softplus"]
        < baseline["mean_margin_softplus"],
        "positive_margin_sides_not_worse": candidate["positive_margin_sides"]
        >= baseline["positive_margin_sides"],
        "complete_units_not_worse": candidate["complete_units"]
        >= baseline["complete_units"],
        "retention_mean_kl_at_most_0_02": retention["mean_kl_nats"] <= 0.02,
        "retention_top1_exact": retention["top1_agreement"] == 1.0,
        "audit_clean": True,
    }
    if report.get("checks") != expected_checks or not all(expected_checks.values()):
        raise ValueError("V6.3 diagnostic decision changed")
    if _resolve(PROHIBITED_CHECKPOINT).exists():
        raise ValueError("V6.3 prohibited runtime checkpoint exists")
    return {
        "passed": True,
        "sha256": _sha256(PILOT_REPORT),
        "status": report["status"],
        "promotion_authorized": False,
        "positive_margin_side_delta": int(delta["positive_margin_sides"]),
        "complete_unit_delta": int(delta["complete_units"]),
        "mean_margin_delta": float(delta["mean_margin"]),
        "mean_softplus_delta": float(delta["mean_margin_softplus"]),
        "checkpoint_absent": True,
    }


def build_terminal_marker() -> dict[str, Any]:
    pinned = authenticate_pinned_bytes()
    gradient = authenticate_gradient_report()
    pilot = authenticate_pilot_report()
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_terminal",
        "status": "positive_train_only_pilot_continuation_authorized_no_runtime_promotion",
        "v6_3_pinned_sha256": pinned,
        "gradient_evidence": gradient,
        "pilot_evidence": pilot,
        "authenticator_source_sha256": _sha256(AUTHENTICATOR_SOURCE),
        "authenticator_test_sha256": _sha256(AUTHENTICATOR_TEST),
        "runtime_checkpoint_promotion_authorized": False,
        "runtime_checkpoint_exists": False,
        "continuation": "v6_4_pair_disjoint_train_only_confirmation",
        "continuation_may_read_internal_validation": False,
        "continuation_may_read_deferred_or_final": False,
        "continuation_may_read_oracle": False,
    }


def seal_terminal_marker() -> tuple[Path, str]:
    destination = _resolve(TERMINAL_MARKER)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V6.3 terminal marker already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(build_terminal_marker(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(payload).hexdigest()


def authenticate_terminal_marker() -> dict[str, Any]:
    observed = _read_json(TERMINAL_MARKER)
    expected = build_terminal_marker()
    if observed != expected:
        raise ValueError("V6.3 terminal/continuation marker changed")
    return {
        "passed": True,
        "status": observed["status"],
        "terminal_sha256": _sha256(TERMINAL_MARKER),
        "pilot_sha256": observed["pilot_evidence"]["sha256"],
        "continuation": observed["continuation"],
        "runtime_checkpoint_promotion_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "authenticate"))
    args = parser.parse_args(argv)
    if args.command == "seal":
        path, digest = seal_terminal_marker()
        result = {"passed": True, "path": str(path), "sha256": digest}
    else:
        result = authenticate_terminal_marker()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
