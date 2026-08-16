"""Seal V46's unique candidate and authorize one bounded train-only V47 run.

The V46 diagnostic is deliberately report-only and does not select a
candidate.  This module records the candidate policy that was fixed before
the diagnostic result existed.  It can authenticate and review an explicitly
hashed V46 report.  The unique candidate and stable V47 config/trainer/test
bytes are pinned below.  The sole successor is a four-update V47 continuation
that reconstructs the candidate in memory; standalone candidate persistence,
validation, selection, and promotion remain forbidden.

The expected V46 report hash is always supplied by the caller.  It is not
embedded here because the eventual terminal may pin this implementation.
Passing :data:`REPORT_SHA256_PLACEHOLDER` builds the pre-result scaffold
without opening the V46 report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT

V45_TERMINAL = Path(
    "reports/gemma4/metrics/v45_retention_repair_terminal_gate.json"
)
V46_SCREEN = Path(
    "src/semantic_3d_chat/evaluation/v46_v45_u4_lost_side_screen.py"
)
V46_SCREEN_TEST = Path("tests/test_v46_v45_u4_lost_side_screen.py")
V46_REPORT = Path(
    "reports/gemma4/metrics/v46_v45_u4_lost_side_no_step_diagnostic.json"
)
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v46_v45_u4_lost_side_terminal_gate.json"
)
V47_CONFIG = Path(
    "configs/experiments/gemma4_diverse28_book_continuation_v47.yaml"
)
V47_TRAINER = Path(
    "src/semantic_3d_chat/training/train_book_continuation_v47.py"
)
V47_TEST = Path("tests/test_train_book_continuation_v47.py")
V47_OUTPUT = Path(
    "data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query"
)

REPORT_SHA256_PLACEHOLDER = "PENDING_V46_REPORT_SHA256"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_V45_TERMINAL_SHA256 = (
    "c069fb29d6729dfa3c2c2df3ef4854c3f037172870cfbd0ae2414be306f7e9d5"
)
_V46_SCREEN_SHA256 = (
    "acda1857fd8e1c8673d313dabca07dd1834df5fd17d1eb54021f46a1ae451926"
)
_V46_SCREEN_TEST_SHA256 = (
    "300d58d1f412f5f94e89193bd091f9e15665960aac12035c896f61c2c04d3547"
)
_V47_CONFIG_SHA256 = (
    "6b15813237b217d8daad446c799127215bbb1366f2b442b61e975787efa4f6b7"
)
_V47_TRAINER_SHA256 = (
    "971fdbaf2f6e6b22dc27b83cfa0f6604c2c1145d92509c06bc98410f6927ea22"
)
_V47_TEST_SHA256 = (
    "86c3e0e49b0c42b1161227cdc42ff577afca6b37bd0db183ca1648f23125aedd"
)
_PROTECTED_REPORT_SHA256 = (
    "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
)
_SOURCE_FULL_SHA256 = (
    "468f493a746c6125f8ebc62d57ca8ae0419160f6e13ce903dd9f40c64aa772c2"
)
_SOURCE_AUTHORIZED_SHA256 = (
    "e4165bb1c2a4664eeb146a48107aead3e69bb576c1604bea39b3b7474d17c696"
)
_SOURCE_FROZEN_SHA256 = (
    "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"
)
_V47_AUTHORIZATION_ID = "v47_exact_book_support_continuation"

_DIRECTION_IDS = ("g5_scene_sign", "g5_query_sign", "g5_both_sign")
_ALPHA_GRID = (0.125, 0.25, 0.5, 1.0, 2.0)
_Q5 = "cfq_5c84a2c27d2be251"
_Q699 = "cfq_699675ceeaf65406"
_ORIGINAL_V41_PRIORITY_DEFICIT = 31.113729119300842
_BROAD_NLL_MAXIMUM = 2.9213306349515915
_EXPECTED_SELECTION: dict[str, Any] = {
    "candidate_id": "g5_both_sign_alpha_1p0",
    "direction_id": "g5_both_sign",
    "alpha": 1.0,
    "inventory_index": 13,
    "authorized_surface_state_sha256": (
        "d60b665d9a970433b2ed59e6769b9114468bef608b98eae828268101d39db56c"
    ),
    "full_tensor_state_sha256": (
        "1d5adc1fb0d7a895056b77d38c8a12aba95c9997ec8a94edf68673f9c58fb954"
    ),
    "complete_units": 9,
    "positive_sides": 34,
    "cross_prefix_complete_units": 18,
    "complete_physical_pair_coverage": 4,
    "q5_margin": 0.0625,
    "q699_margin": 0.3125,
    "robustness_tier": 2,
    "minimum_integer_surplus": 0,
    "priority_deficit_improvement": 0.8047257661819458,
    "broad_nll": 2.91504575808843,
    "minimum_continuous_headroom": 0.0062848768631615926,
}
_FIXED_THRESHOLDS = {
    "complete_units": 9,
    "positive_sides": 34,
    "cross_prefix_complete_units": 17,
    "complete_physical_pair_coverage": 4,
    "priority_deficit_improvement": 0.5,
    "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
    "both_lost_sides_strictly_positive": True,
}
_CHECK_NAMES = (
    "complete_units_at_least_9",
    "positive_sides_at_least_34",
    "cross_prefix_complete_units_at_least_17",
    "complete_physical_pair_coverage_at_least_4",
    "priority_deficit_improvement_at_least_0_5_vs_original_v41_u0",
    "broad_nll_at_most_v45_maximum",
    "both_lost_sides_strictly_positive",
)

# This literal records the exact V46 result and stable V47 implementation pins.
# It remains non-authorizing until the caller explicitly authenticates the V46
# report.  A caller cannot override these fields with a function argument.
_SUCCESSOR_REVIEW_PLACEHOLDER: dict[str, Any] = {
    "status": "pending_explicit_v46_report_authentication",
    "v46_result_authenticated": False,
    "unique_eligible_candidate_count": 1,
    "selected_candidate_id": _EXPECTED_SELECTION["candidate_id"],
    "selected_authorized_surface_state_sha256": _EXPECTED_SELECTION[
        "authorized_surface_state_sha256"
    ],
    "selected_full_tensor_state_sha256": _EXPECTED_SELECTION[
        "full_tensor_state_sha256"
    ],
    "intended_successor_action": (
        "one_bounded_train_only_v47_four_step_book_support_continuation"
    ),
    "v47_maximum_optimizer_updates": 4,
    "v47_focus": "book_support_continuation",
    "v47_config_sha256": _V47_CONFIG_SHA256,
    "v47_trainer_sha256": _V47_TRAINER_SHA256,
    "v47_test_sha256": _V47_TEST_SHA256,
    "v47_implementation_hashes_complete": True,
    "exact_successor_action": None,
    "candidate_checkpoint_write_authorized": False,
    "validation_access_authorized": False,
    "oracle_access_authorized": False,
    "final_test_access_authorized": False,
    "selector_execution_authorized": False,
    "runtime_promotion_authorized": False,
    "chat_promotion_authorized": False,
    "embodied_promotion_authorized": False,
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _lower_hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal digits")
    return value


def _locked_file(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{field} bytes changed: expected {expected}, observed {observed}"
        )


def _authenticate_pre_result_inputs() -> dict[str, Any]:
    """Authenticate V45 and V46 implementation without touching V46 output."""

    pins = {
        V45_TERMINAL: _V45_TERMINAL_SHA256,
        V46_SCREEN: _V46_SCREEN_SHA256,
        V46_SCREEN_TEST: _V46_SCREEN_TEST_SHA256,
        V47_CONFIG: _V47_CONFIG_SHA256,
        V47_TRAINER: _V47_TRAINER_SHA256,
        V47_TEST: _V47_TEST_SHA256,
    }
    for relative, expected in pins.items():
        _locked_file(_resolve(relative), expected, str(relative))
    terminal = _mapping(
        json.loads(_resolve(V45_TERMINAL).read_text(encoding="utf-8")),
        "V45 terminal",
    )
    authorization = _mapping(
        terminal.get("conditional_successor_authorization"),
        "V45 successor authorization",
    )
    integrity = _mapping(
        authorization.get("implementation_integrity"),
        "V45 V46 implementation integrity",
    )
    scope = _mapping(authorization.get("scope"), "V45 V46 scope")
    checks = {
        "v45_artifact": terminal.get("artifact") == "v45_retention_repair_terminal_gate",
        "v45_passed": terminal.get("passed") is True,
        "only_successor": terminal.get("only_exact_successor_authorized")
        == "v46_train_only_checkpoint_gradient_diagnostic",
        "authorization_id": authorization.get("authorization_id")
        == "v46_train_only_checkpoint_gradient_diagnostic",
        "authorized_report": authorization.get("authorized_report") == str(V46_REPORT),
        "screen_path": authorization.get("authorized_script") == str(V46_SCREEN),
        "screen_test_path": authorization.get("authorized_test")
        == str(V46_SCREEN_TEST),
        "screen_sha256": integrity.get("script_sha256") == _V46_SCREEN_SHA256,
        "screen_test_sha256": integrity.get("test_sha256")
        == _V46_SCREEN_TEST_SHA256,
        "v46_report_only": scope.get("report_only_output") is True,
        "v46_no_candidate": scope.get("no_candidate_is_authorized_by_this_diagnostic")
        is True,
        "v46_requires_new_terminal": scope.get("new_terminal_seal_required_for_any_successor")
        is True,
        "v46_no_validation": scope.get("validation_access_authorized") is False,
        "v46_no_oracle": scope.get("oracle_access_authorized") is False,
        "v46_no_final": scope.get("final_test_access_authorized") is False,
        "v46_no_selector": scope.get("selector_execution_authorized") is False,
        "v46_no_promotion": scope.get("runtime_promotion_authorized") is False,
        "terminal_no_checkpoint": terminal.get("candidate_checkpoint_write_authorized")
        is False,
        "terminal_no_validation": terminal.get("validation_access_authorized") is False,
        "terminal_no_oracle": terminal.get("oracle_access_authorized") is False,
        "terminal_no_final": terminal.get("final_test_access_authorized") is False,
        "terminal_no_selector": terminal.get("selector_execution_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"V46 scaffold V45 authorization changed: {checks}")
    return {
        "file_sha256": {str(path): expected for path, expected in pins.items()},
        "authorization_checks": checks,
        "v46_report_opened": False,
    }


def _report_hash_reference(value: str) -> dict[str, Any]:
    if value == REPORT_SHA256_PLACEHOLDER:
        return {
            "status": "pending_explicit_sha256",
            "expected_sha256": REPORT_SHA256_PLACEHOLDER,
            "report_opened": False,
            "report_authenticated": False,
        }
    digest = _lower_hex64(value, "expected V46 report SHA256")
    return {
        "status": "explicit_sha256_supplied",
        "expected_sha256": digest,
        "report_opened": False,
        "report_authenticated": False,
    }


def _focus_margin(
    focus: Mapping[str, Any],
    *,
    question_key: str,
    expected_pair_id: str,
    side_index: int,
) -> float:
    row = _mapping(focus.get(question_key), f"V46 focus row {question_key}")
    margins = _sequence(row.get("side_margins"), f"V46 side margins {question_key}")
    if row.get("pair_id") != expected_pair_id or len(margins) != 2:
        raise ValueError(f"V46 focus row identity changed: {question_key}")
    return _finite(margins[side_index], f"V46 focus margin {question_key}")


def robustness_tier(minimum_lost_side_margin: float) -> int:
    """Map the weaker q5/q699 margin to the precommitted robustness tier."""

    margin = _finite(minimum_lost_side_margin, "minimum lost-side margin")
    if margin >= 0.125:
        return 3
    if margin >= 0.0625:
        return 2
    if margin > 0.0:
        return 1
    return 0


def _optional_l2(row: Mapping[str, Any]) -> float | None:
    values = [
        row.get("authorized_surface_l2_perturbation"),
        _mapping(row.get("candidate_state_before_forward"), "V46 candidate state").get(
            "authorized_surface_l2_perturbation"
        ),
    ]
    present = [value for value in values if value is not None]
    if len(present) > 1 and _finite(present[0], "V46 authorized-surface L2") != _finite(
        present[1], "V46 authorized-surface L2"
    ):
        raise ValueError("V46 candidate reports conflicting authorized-surface L2 values")
    if not present:
        return None
    result = _finite(present[0], "V46 authorized-surface L2")
    if result < 0.0:
        raise ValueError("V46 authorized-surface L2 cannot be negative")
    return result


def candidate_eligibility(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all fixed gates; persisted booleans are never trusted alone."""

    metrics = _mapping(row.get("pair_metrics"), "V46 candidate pair metrics")
    threshold = _mapping(row.get("threshold_diagnostic"), "V46 threshold diagnostic")
    focus = _mapping(row.get("focus_units"), "V46 focus units")
    retention = _mapping(
        threshold.get("retention_diagnostics"),
        "V46 retention diagnostics",
    )
    q5 = _focus_margin(
        focus,
        question_key=_Q5,
        expected_pair_id="pair_000006",
        side_index=0,
    )
    q699 = _focus_margin(
        focus,
        question_key=_Q699,
        expected_pair_id="pair_000016",
        side_index=1,
    )
    complete = _integer(metrics.get("complete_units"), "V46 complete units")
    positive = _integer(metrics.get("positive_sides"), "V46 positive sides")
    cross = _integer(
        metrics.get("cross_prefix_complete_units"),
        "V46 cross-prefix complete units",
    )
    physical = _integer(
        metrics.get("complete_physical_pair_coverage"),
        "V46 complete physical-pair coverage",
    )
    if _integer(metrics.get("unit_count"), "V46 unit count") != 25:
        raise ValueError("V46 candidate does not contain all 25 changed-pair units")
    broad = _finite(row.get("broad_nll"), "V46 broad NLL")
    deficit = _finite(threshold.get("priority_side_deficit"), "V46 priority deficit")
    improvement = _ORIGINAL_V41_PRIORITY_DEFICIT - deficit
    persisted_improvement = _finite(
        threshold.get("priority_deficit_improvement_vs_original_v41_u0"),
        "V46 persisted priority improvement",
    )
    if not math.isclose(improvement, persisted_improvement, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("V46 persisted priority improvement is inconsistent")
    if not math.isclose(
        broad,
        _finite(threshold.get("broad_nll"), "V46 threshold broad NLL"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("V46 candidate broad NLL is inconsistent")
    lost_positive = q5 > 0.0 and q699 > 0.0
    if retention.get("both_lost_sides_strictly_positive") is not lost_positive:
        raise ValueError("V46 lost-side retention diagnostic is inconsistent")
    checks = {
        _CHECK_NAMES[0]: complete >= _FIXED_THRESHOLDS["complete_units"],
        _CHECK_NAMES[1]: positive >= _FIXED_THRESHOLDS["positive_sides"],
        _CHECK_NAMES[2]: cross >= _FIXED_THRESHOLDS["cross_prefix_complete_units"],
        _CHECK_NAMES[3]: physical
        >= _FIXED_THRESHOLDS["complete_physical_pair_coverage"],
        _CHECK_NAMES[4]: improvement
        >= _FIXED_THRESHOLDS["priority_deficit_improvement"],
        _CHECK_NAMES[5]: broad <= _FIXED_THRESHOLDS["broad_nll_maximum"],
        _CHECK_NAMES[6]: lost_positive,
    }
    persisted_checks = _mapping(threshold.get("checks"), "V46 persisted checks")
    # The diagnostic report is serialized with ``sort_keys=True``.  Key order
    # therefore cannot carry policy meaning; the exact key set and values do.
    if set(persisted_checks) != set(_CHECK_NAMES) or dict(persisted_checks) != checks:
        raise ValueError("V46 persisted threshold checks differ from recomputation")
    eligible = all(checks.values())
    if threshold.get("all_numeric_thresholds_met") is not eligible:
        raise ValueError("V46 persisted aggregate eligibility is inconsistent")
    if threshold.get("diagnostic_only_no_candidate_authorization") is not True:
        raise ValueError("V46 diagnostic attempted to authorize a candidate")
    integer_surplus = min(
        complete - _FIXED_THRESHOLDS["complete_units"],
        positive - _FIXED_THRESHOLDS["positive_sides"],
        cross - _FIXED_THRESHOLDS["cross_prefix_complete_units"],
        physical - _FIXED_THRESHOLDS["complete_physical_pair_coverage"],
    )
    continuous_headroom = min(
        improvement - _FIXED_THRESHOLDS["priority_deficit_improvement"],
        _FIXED_THRESHOLDS["broad_nll_maximum"] - broad,
    )
    minimum_margin = min(q5, q699)
    return {
        "eligible": eligible,
        "checks": checks,
        "complete_units": complete,
        "positive_sides": positive,
        "cross_prefix_complete_units": cross,
        "complete_physical_pair_coverage": physical,
        "q5_margin": q5,
        "q699_margin": q699,
        "minimum_lost_side_margin": minimum_margin,
        "robustness_tier": robustness_tier(minimum_margin),
        "minimum_integer_surplus": integer_surplus,
        "priority_deficit_improvement": improvement,
        "broad_nll": broad,
        "minimum_continuous_headroom": continuous_headroom,
        "authorized_surface_l2_perturbation": _optional_l2(row),
    }


def _candidate_identity(
    row: Mapping[str, Any],
    inventory_row: Mapping[str, Any],
    inventory_index: int,
) -> dict[str, Any]:
    direction = row.get("direction_id")
    alpha = _finite(row.get("alpha"), "V46 candidate alpha")
    if direction not in _DIRECTION_IDS or alpha not in _ALPHA_GRID:
        raise ValueError("V46 candidate direction or alpha is outside the fixed grid")
    fields = (
        "candidate_id",
        "direction_id",
        "alpha",
        "authorized_surface_state_sha256",
        "full_tensor_state_sha256",
    )
    if any(row.get(field) != inventory_row.get(field) for field in fields):
        raise ValueError("V46 candidate result differs from its prehashed inventory row")
    authorized = _lower_hex64(
        row.get("authorized_surface_state_sha256"),
        "V46 authorized-surface hash",
    )
    full = _lower_hex64(row.get("full_tensor_state_sha256"), "V46 full-state hash")
    state = _mapping(row.get("candidate_state_before_forward"), "V46 candidate state")
    if (
        state.get("passed") is not True
        or state.get("authorized_surface_state_sha256") != authorized
        or state.get("full_tensor_state_sha256") != full
        or state.get("frozen_state_sha256") != _SOURCE_FROZEN_SHA256
        or state.get("all_parameter_gradients_absent") is not True
        or row.get("candidate_checkpoint_written") is not False
        or row.get("candidate_authorized") is not False
    ):
        raise ValueError("V46 candidate state or non-authorization evidence changed")
    return {
        "candidate_id": str(row["candidate_id"]),
        "direction_id": str(direction),
        "alpha": alpha,
        "authorized_surface_state_sha256": authorized,
        "full_tensor_state_sha256": full,
        "inventory_index": inventory_index,
    }


def _rank_key(
    row: Mapping[str, Any],
    *,
    l2_available: bool,
) -> tuple[Any, ...]:
    evidence = _mapping(row.get("eligibility"), "V46 eligibility evidence")
    direction = str(row["direction_id"])
    l2_value = evidence.get("authorized_surface_l2_perturbation")
    if (l2_value is not None) is not l2_available:
        raise ValueError("V46 eligible candidates have mixed L2 availability")
    l2_rank = _finite(l2_value, "V46 authorized-surface L2") if l2_available else 0.0
    return (
        -_integer(evidence.get("robustness_tier"), "V46 robustness tier"),
        -_integer(
            evidence.get("minimum_integer_surplus"),
            "V46 minimum integer surplus",
        ),
        -_finite(
            evidence.get("minimum_continuous_headroom"),
            "V46 minimum continuous headroom",
        ),
        l2_rank,
        _finite(row.get("alpha"), "V46 alpha"),
        _DIRECTION_IDS.index(direction),
        _integer(row.get("inventory_index"), "V46 inventory index"),
        _lower_hex64(
            row.get("authorized_surface_state_sha256"),
            "V46 authorized-surface hash",
        ),
    )


def rank_eligible_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the immutable lexicographic policy to already validated rows."""

    eligible = [dict(row) for row in candidates if row.get("eligibility", {}).get("eligible")]
    l2_presence = {
        _mapping(row["eligibility"], "V46 eligibility").get(
            "authorized_surface_l2_perturbation"
        )
        is not None
        for row in eligible
    }
    if len(l2_presence) > 1:
        raise ValueError("V46 eligible candidates have mixed L2 availability")
    l2_available = l2_presence == {True}
    ranked = sorted(
        eligible,
        key=lambda row: _rank_key(row, l2_available=l2_available),
    )
    return [
        {
            **row,
            "deterministic_rank": index,
            "authorized_surface_l2_criterion_available": l2_available,
        }
        for index, row in enumerate(ranked, start=1)
    ]


def _authenticate_expected_unique_selection(
    ranked: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate the exact result observed by the independent review."""

    if len(ranked) != 1:
        raise ValueError(
            "V46 reviewed result must contain exactly one eligible candidate; "
            f"observed {len(ranked)}"
        )
    selected = _mapping(ranked[0], "V46 uniquely selected candidate")
    evidence = _mapping(selected.get("eligibility"), "V46 selected eligibility")
    observed = {
        "candidate_id": selected.get("candidate_id"),
        "direction_id": selected.get("direction_id"),
        "alpha": _finite(selected.get("alpha"), "V46 selected alpha"),
        "inventory_index": _integer(
            selected.get("inventory_index"), "V46 selected inventory index"
        ),
        "authorized_surface_state_sha256": selected.get(
            "authorized_surface_state_sha256"
        ),
        "full_tensor_state_sha256": selected.get("full_tensor_state_sha256"),
        "complete_units": _integer(
            evidence.get("complete_units"), "V46 selected complete units"
        ),
        "positive_sides": _integer(
            evidence.get("positive_sides"), "V46 selected positive sides"
        ),
        "cross_prefix_complete_units": _integer(
            evidence.get("cross_prefix_complete_units"),
            "V46 selected cross-prefix complete units",
        ),
        "complete_physical_pair_coverage": _integer(
            evidence.get("complete_physical_pair_coverage"),
            "V46 selected physical-pair coverage",
        ),
        "q5_margin": _finite(evidence.get("q5_margin"), "V46 selected q5 margin"),
        "q699_margin": _finite(
            evidence.get("q699_margin"), "V46 selected q699 margin"
        ),
        "robustness_tier": _integer(
            evidence.get("robustness_tier"), "V46 selected robustness tier"
        ),
        "minimum_integer_surplus": _integer(
            evidence.get("minimum_integer_surplus"),
            "V46 selected minimum integer surplus",
        ),
        "priority_deficit_improvement": _finite(
            evidence.get("priority_deficit_improvement"),
            "V46 selected priority improvement",
        ),
        "broad_nll": _finite(evidence.get("broad_nll"), "V46 selected broad NLL"),
        "minimum_continuous_headroom": _finite(
            evidence.get("minimum_continuous_headroom"),
            "V46 selected continuous headroom",
        ),
    }
    exact_fields = {
        key: observed[key] == _EXPECTED_SELECTION[key]
        for key in (
            "candidate_id",
            "direction_id",
            "alpha",
            "inventory_index",
            "authorized_surface_state_sha256",
            "full_tensor_state_sha256",
            "complete_units",
            "positive_sides",
            "cross_prefix_complete_units",
            "complete_physical_pair_coverage",
            "q5_margin",
            "q699_margin",
            "robustness_tier",
            "minimum_integer_surplus",
            "priority_deficit_improvement",
            "broad_nll",
            "minimum_continuous_headroom",
        )
    }
    if selected.get("deterministic_rank") != 1:
        exact_fields["deterministic_rank"] = False
    else:
        exact_fields["deterministic_rank"] = True
    if evidence.get("eligible") is not True:
        exact_fields["all_fixed_thresholds_true"] = False
    else:
        exact_fields["all_fixed_thresholds_true"] = all(
            _mapping(evidence.get("checks"), "V46 selected threshold checks").values()
        )
    if not all(exact_fields.values()):
        raise ValueError(
            "V46 unique eligible candidate differs from the independently reviewed result: "
            f"{exact_fields}"
        )
    return {
        "passed": True,
        "unique_eligible_candidate_count": 1,
        "exact_field_checks": exact_fields,
        "selected_candidate": dict(selected),
        "candidate_checkpoint_write_authorized": False,
        "validation_access_authorized": False,
    }


def review_screen_payload(screen: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a V46 payload and rank eligible rows without authorizing one."""

    terminal = _mapping(screen.get("terminal"), "V46 terminal attestation")
    inventory = _mapping(screen.get("candidate_inventory"), "V46 inventory")
    inventory_rows = _sequence(inventory.get("candidates"), "V46 inventory rows")
    candidate_rows = _sequence(screen.get("candidate_results"), "V46 candidate results")
    expected_order = [
        (direction, alpha) for direction in _DIRECTION_IDS for alpha in _ALPHA_GRID
    ]
    observed_inventory_order = [
        (
            _mapping(row, "V46 inventory row").get("direction_id"),
            _mapping(row, "V46 inventory row").get("alpha"),
        )
        for row in inventory_rows
    ]
    fixed = {
        "artifact": screen.get("artifact")
        == "v46_v45_u4_lost_side_no_step_diagnostic",
        "integrity": screen.get("screen_integrity_passed") is True,
        "terminal_sha256": terminal.get("sha256") == _V45_TERMINAL_SHA256,
        "terminal_authorization": terminal.get("authorization_id")
        == "v46_train_only_checkpoint_gradient_diagnostic",
        "formula": inventory.get("formula")
        == "float32_P0-alpha*lr_group*sign(g5)",
        "directions": inventory.get("direction_ids") == list(_DIRECTION_IDS),
        "alphas": inventory.get("alpha_grid") == list(_ALPHA_GRID),
        "candidate_count": inventory.get("candidate_count") == 15,
        "inventory_order": observed_inventory_order == expected_order,
        "inventory_prehash": inventory.get(
            "candidate_hashes_fixed_before_candidate_forward_evaluation"
        )
        is True,
        "inventory_hash": inventory.get("candidate_inventory_sha256")
        == _canonical_sha256(list(inventory_rows)),
        "result_count": len(candidate_rows) == 15,
        "full_pairs": screen.get("all_15_candidates_received_full_25_unit_metrics")
        is True,
        "full_broad": screen.get("all_15_candidates_received_fixed_48_row_broad_nll")
        is True,
        "no_selection": screen.get("candidate_selection_performed") is False,
        "nonadaptive": screen.get("adaptive_direction_or_scalar_selection") is False,
        "no_authorization": screen.get("candidate_authorization_granted") is False,
        "no_checkpoint": screen.get("candidate_checkpoint_written") is False,
        "no_optimizer": screen.get("optimizer_constructed_or_loaded") is False,
        "no_optimizer_file": screen.get("optimizer_state_file_opened") is False,
        "no_optimizer_step": screen.get("optimizer_step_executed") is False,
        "no_parameter_persist": screen.get("parameter_state_persisted") is False,
        "train_maps_only": screen.get("all_16_training_maps_loaded") is True,
        "no_validation_qa": screen.get("validation_qa_loaded") is False,
        "no_validation_maps": screen.get("validation_environment_maps_loaded") is False,
        "no_oracle": screen.get("oracle_loaded") is False,
        "no_final": screen.get("final_test_scenes_touched") is False,
        "no_selector": screen.get("selector_executed") is False,
        "no_runtime": screen.get("runtime_promotion_executed") is False,
        "no_chat": screen.get("chat_promotion_executed") is False,
        "no_embodied": screen.get("embodied_promotion_executed") is False,
        "protected_unchanged": screen.get("protected_report_sha256_before_and_after")
        == _PROTECTED_REPORT_SHA256,
        "forbidden_empty": screen.get("forbidden_file_accesses") == [],
    }
    if not all(fixed.values()):
        raise ValueError(f"V46 screen fixed envelope changed: {fixed}")
    source = _mapping(screen.get("source_audit"), "V46 source audit")
    source_replay = _mapping(screen.get("source_replay"), "V46 source replay")
    gradient = _mapping(screen.get("gradient_audit"), "V46 gradient audit")
    gradient_source = _mapping(
        gradient.get("source_state_after_gradient_measurement"),
        "V46 gradient source state",
    )
    final = _mapping(screen.get("final_state"), "V46 final state")
    for field, value in (
        ("source full", source.get("full_tensor_state_sha256")),
        ("gradient full", gradient_source.get("full_tensor_state_sha256")),
        ("final full", final.get("full_tensor_state_sha256")),
    ):
        if value != _SOURCE_FULL_SHA256:
            raise ValueError(f"V46 {field} hash changed")
    for field, value in (
        ("source authorized", source.get("authorized_surface_state_sha256")),
        ("gradient authorized", gradient_source.get("authorized_surface_state_sha256")),
        ("final authorized", final.get("authorized_surface_state_sha256")),
    ):
        if value != _SOURCE_AUTHORIZED_SHA256:
            raise ValueError(f"V46 {field} hash changed")
    if (
        source.get("frozen_state_sha256") != _SOURCE_FROZEN_SHA256
        or gradient_source.get("frozen_state_sha256") != _SOURCE_FROZEN_SHA256
        or final.get("frozen_state_sha256") != _SOURCE_FROZEN_SHA256
        or source.get("optimizer_file_opened") is not False
        or source.get("optimizer_state_deserialized") is not False
        or source.get("optimizer_state_loaded") is not False
        or source_replay.get("passed") is not True
        or gradient.get("source_state_unchanged") is not True
        or gradient.get("optimizer_constructed_or_loaded") is not False
        or final.get("passed") is not True
        or final.get("all_15_before_after_restorations_passed") is not True
    ):
        raise ValueError("V46 source replay, gradient, or final restoration changed")
    restorations = _sequence(screen.get("restoration_audit"), "V46 restorations")
    if len(restorations) != 30 or any(
        _mapping(row, "V46 restoration").get("passed") is not True
        or _mapping(row, "V46 restoration").get("full_tensor_state_sha256")
        != _SOURCE_FULL_SHA256
        for row in restorations
    ):
        raise ValueError("V46 did not restore exact update four around all candidates")
    reviewed: list[dict[str, Any]] = []
    for index, (inventory_value, candidate_value) in enumerate(
        zip(inventory_rows, candidate_rows, strict=True)
    ):
        inventory_row = _mapping(inventory_value, "V46 inventory row")
        candidate = _mapping(candidate_value, "V46 candidate row")
        identity = _candidate_identity(candidate, inventory_row, index)
        per_unit = _sequence(
            candidate.get("per_unit_nll_diagnostics"),
            "V46 per-unit NLL diagnostics",
        )
        if len(per_unit) != 25:
            raise ValueError("V46 candidate did not receive all 25 per-unit NLL rows")
        reviewed.append({**identity, "eligibility": candidate_eligibility(candidate)})
    ranked = rank_eligible_candidates(reviewed)
    result_authentication = _authenticate_expected_unique_selection(ranked)
    return {
        "fixed_envelope_checks": fixed,
        "candidate_count": len(reviewed),
        "eligible_candidate_count": len(ranked),
        "eligible_candidates_ranked": ranked,
        "result_authentication": result_authentication,
        "recommended_candidate_for_future_review": ranked[0],
        "ranking_is_advisory_only": True,
        "candidate_authorization_granted": False,
        "candidate_checkpoint_write_authorized": False,
        "validation_access_authorized": False,
    }


def load_and_review_screen(expected_v46_report_sha256: str) -> dict[str, Any]:
    """Authenticate the explicitly hashed report, then run the pure review."""

    digest = _lower_hex64(expected_v46_report_sha256, "expected V46 report SHA256")
    _authenticate_pre_result_inputs()
    path = _resolve(V46_REPORT)
    _locked_file(path, digest, "V46 report")
    screen = _mapping(json.loads(path.read_text(encoding="utf-8")), "V46 report")
    return {
        "path": str(V46_REPORT),
        "sha256": digest,
        "review": review_screen_payload(screen),
    }


def _v47_authorization(expected_v46_report_sha256: str) -> dict[str, Any]:
    """Return the sole exact successor accepted by the pinned V47 trainer."""

    digest = _lower_hex64(expected_v46_report_sha256, "expected V46 report SHA256")
    return {
        "authorization_id": _V47_AUTHORIZATION_ID,
        "authorized": True,
        "only_exact_action": "one_bounded_four_step_v47_book_support_continuation",
        "authorized_config": str(V47_CONFIG),
        "authorized_trainer": str(V47_TRAINER),
        "authorized_test": str(V47_TEST),
        "authorized_output": str(V47_OUTPUT),
        "explicit_terminal_sha256_cli_required": True,
        "implementation_integrity": {
            "config_sha256": _V47_CONFIG_SHA256,
            "trainer_sha256": _V47_TRAINER_SHA256,
            "test_sha256": _V47_TEST_SHA256,
        },
        "source": {
            "v46_report_sha256": digest,
            "base_checkpoint": (
                "data_gemma4/checkpoints/"
                "gemma4_v45_retention_repair_l14_query/update_004"
            ),
            "candidate_id": _EXPECTED_SELECTION["candidate_id"],
            "candidate_full_tensor_state_sha256": _EXPECTED_SELECTION[
                "full_tensor_state_sha256"
            ],
            "candidate_authorized_surface_sha256": _EXPECTED_SELECTION[
                "authorized_surface_state_sha256"
            ],
            "candidate_frozen_state_sha256": _SOURCE_FROZEN_SHA256,
        },
        "training": {
            "optimizer_steps": 4,
            "checkpoint_steps": [0, 2, 4],
            "target_question_keys": ["cfq_163eb92339ad35a5"] * 4,
            "broad_question_ids": [
                "q_000099",
                "q_000138",
                "q_000053",
                "q_000089",
            ],
            "fresh_adamw": True,
            "same_v45_objective": True,
            "update2_integrity_only": True,
            "update4_original_v45_final_gate": True,
        },
        "scope": {
            "train_only": True,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
        },
    }


def build_terminal_scaffold(expected_v46_report_sha256: str) -> dict[str, Any]:
    """Build the review scaffold used immediately before materialization.

    The placeholder mode never opens the V46 report.  Supplying a real hash
    authenticates and ranks it.  Restricted-data access and standalone
    candidate persistence remain forbidden in either mode.
    """

    inputs = _authenticate_pre_result_inputs()
    reference = _report_hash_reference(expected_v46_report_sha256)
    review = None
    if expected_v46_report_sha256 != REPORT_SHA256_PLACEHOLDER:
        reviewed = load_and_review_screen(expected_v46_report_sha256)
        reference.update(
            {
                "status": "authenticated_for_advisory_review_only",
                "report_opened": True,
                "report_authenticated": True,
                "path": reviewed["path"],
            }
        )
        review = reviewed["review"]
    successor_review = dict(_SUCCESSOR_REVIEW_PLACEHOLDER)
    if review is not None:
        result_authentication = _mapping(
            review.get("result_authentication"),
            "V46 result authentication",
        )
        if result_authentication.get("passed") is not True:
            raise ValueError("V46 exact result authentication did not pass")
        successor_review.update(
            {
                "status": "v46_result_and_v47_implementation_authenticated",
                "v46_result_authenticated": True,
                "exact_successor_action": _V47_AUTHORIZATION_ID,
            }
        )
    ready = review is not None
    return {
        "schema_version": 1,
        "artifact": "v46_v45_u4_lost_side_terminal_gate_scaffold",
        "pre_result_policy_fixed": True,
        "terminal_materialization_authorized": ready,
        "input_integrity": inputs,
        "v46_report_reference": reference,
        "fixed_eligibility_thresholds": dict(_FIXED_THRESHOLDS),
        "fixed_candidate_ranking": [
            "highest_min_q5_q699_robustness_tier_ge_0p125_else_ge_0p0625_else_gt_0",
            "maximize_minimum_integer_surplus_complete9_positive34_cross17_physical4",
            "maximize_minimum_continuous_headroom_priority_improvement0p5_and_broad_cap",
            "minimize_authorized_surface_l2_perturbation_if_uniformly_present",
            "lower_alpha",
            "direction_order_scene_query_both",
            "inventory_order_then_authorized_surface_hash",
        ],
        "advisory_result_review": review,
        "successor_review": successor_review,
        "only_exact_successor_authorized": _V47_AUTHORIZATION_ID if ready else None,
        "candidate_checkpoint_write_authorized": False,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
        "chat_promotion_authorized": False,
        "embodied_promotion_authorized": False,
    }


def build_terminal_report(expected_v46_report_sha256: str) -> dict[str, Any]:
    """Build the exact V46 terminal after result and V47 integrity review."""

    scaffold = build_terminal_scaffold(expected_v46_report_sha256)
    if scaffold.get("terminal_materialization_authorized") is not True:
        raise ValueError("V46 terminal requires an explicit authenticated report hash")
    review = _mapping(
        scaffold.get("advisory_result_review"),
        "V46 terminal result review",
    )
    result = _mapping(review.get("result_authentication"), "V46 result authentication")
    successor = _mapping(scaffold.get("successor_review"), "V46 successor review")
    if (
        result.get("passed") is not True
        or successor.get("v46_result_authenticated") is not True
        or successor.get("v47_implementation_hashes_complete") is not True
        or successor.get("exact_successor_action") != _V47_AUTHORIZATION_ID
    ):
        raise ValueError("V46 result or V47 implementation review is incomplete")
    return {
        "schema_version": 1,
        "artifact": "v46_v45_u4_lost_side_terminal_gate",
        "passed": True,
        "terminal_conclusion": "unique_train_eligible_candidate_authenticated",
        "screen_sha256": expected_v46_report_sha256,
        "input_integrity": scaffold["input_integrity"],
        "pre_result_policy_fixed": True,
        "fixed_eligibility_thresholds": scaffold["fixed_eligibility_thresholds"],
        "fixed_candidate_ranking": scaffold["fixed_candidate_ranking"],
        "screen_review": review,
        "result_authentication": result,
        "conditional_successor_authorization": _v47_authorization(
            expected_v46_report_sha256
        ),
        "only_exact_successor_authorized": _V47_AUTHORIZATION_ID,
        "v47_exact_book_support_continuation_authorized": True,
        "standalone_v46_candidate_checkpoint_write_authorized": False,
        "arbitrary_training_authorized": False,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
        "chat_promotion_authorized": False,
        "embodied_promotion_authorized": False,
        "terminal_process_access_audit": {
            "gemma_loaded": False,
            "qa_loaded": False,
            "maps_loaded": False,
            "optimizer_state_loaded": False,
            "validation_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
            "selector_executed": False,
            "candidate_checkpoint_written": False,
            "v46_report_read_only": True,
            "v47_config_trainer_and_test_bytes_hashed_only": True,
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_report(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    expected_v46_report_sha256: str,
) -> dict[str, Any]:
    """Materialize the exact terminal once, atomically, at its pinned path."""

    path = _resolve(output)
    if path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V46 terminal output path is pinned")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V46 terminal is one-shot and will not overwrite {path}")
    report = build_terminal_report(expected_v46_report_sha256)
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-v46-report-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            write_report(
                args.output,
                expected_v46_report_sha256=args.expected_v46_report_sha256,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "REPORT_SHA256_PLACEHOLDER",
    "build_terminal_report",
    "build_terminal_scaffold",
    "candidate_eligibility",
    "load_and_review_screen",
    "rank_eligible_candidates",
    "review_screen_payload",
    "robustness_tier",
    "write_report",
]
