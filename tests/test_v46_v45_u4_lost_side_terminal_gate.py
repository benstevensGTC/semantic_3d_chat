from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v46_v45_u4_lost_side_terminal_gate as gate

_V46_REPORT_SHA256 = (
    "ce48a1fd484fa5dab71c76a2dd3e39194dd6964e068d6762925a02fb73f6aee6"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checks(
    *,
    complete: int,
    positive: int,
    cross: int,
    physical: int,
    improvement: float,
    broad: float,
    q5: float,
    q699: float,
) -> dict[str, bool]:
    return {
        gate._CHECK_NAMES[0]: complete >= 9,
        gate._CHECK_NAMES[1]: positive >= 34,
        gate._CHECK_NAMES[2]: cross >= 17,
        gate._CHECK_NAMES[3]: physical >= 4,
        gate._CHECK_NAMES[4]: improvement >= 0.5,
        gate._CHECK_NAMES[5]: broad <= gate._BROAD_NLL_MAXIMUM,
        gate._CHECK_NAMES[6]: q5 > 0.0 and q699 > 0.0,
    }


def _candidate(
    *,
    candidate_id: str = "g5_scene_sign_alpha_0p125",
    direction: str = "g5_scene_sign",
    alpha: float = 0.125,
    complete: int = 9,
    positive: int = 34,
    cross: int = 17,
    physical: int = 4,
    improvement: float = 0.75,
    broad: float = 2.90,
    q5: float = 0.125,
    q699: float = 0.125,
    l2: float | None = None,
    authorized_hash: str | None = None,
    full_hash: str | None = None,
) -> dict[str, Any]:
    checks = _checks(
        complete=complete,
        positive=positive,
        cross=cross,
        physical=physical,
        improvement=improvement,
        broad=broad,
        q5=q5,
        q699=q699,
    )
    authorized = authorized_hash or _digest(f"authorized:{candidate_id}")
    full = full_hash or _digest(f"full:{candidate_id}")
    state: dict[str, Any] = {
        "passed": True,
        "authorized_surface_state_sha256": authorized,
        "full_tensor_state_sha256": full,
        "frozen_state_sha256": gate._SOURCE_FROZEN_SHA256,
        "all_parameter_gradients_absent": True,
    }
    if l2 is not None:
        state["authorized_surface_l2_perturbation"] = l2
    return {
        "candidate_id": candidate_id,
        "direction_id": direction,
        "alpha": alpha,
        "authorized_surface_state_sha256": authorized,
        "full_tensor_state_sha256": full,
        "candidate_state_before_forward": state,
        "pair_metrics": {
            "unit_count": 25,
            "complete_units": complete,
            "positive_sides": positive,
            "cross_prefix_complete_units": cross,
            "complete_physical_pair_coverage": physical,
        },
        "per_unit_nll_diagnostics": [{"index": index} for index in range(25)],
        "broad_nll": broad,
        "focus_units": {
            gate._Q5: {
                "pair_id": "pair_000006",
                "side_margins": [q5, 0.5],
            },
            gate._Q699: {
                "pair_id": "pair_000016",
                "side_margins": [0.5, q699],
            },
        },
        "threshold_diagnostic": {
            "checks": checks,
            "all_numeric_thresholds_met": all(checks.values()),
            "diagnostic_only_no_candidate_authorization": True,
            "priority_side_deficit": gate._ORIGINAL_V41_PRIORITY_DEFICIT
            - improvement,
            "priority_deficit_improvement_vs_original_v41_u0": improvement,
            "broad_nll": broad,
            "retention_diagnostics": {
                "both_lost_sides_strictly_positive": q5 > 0.0 and q699 > 0.0,
            },
        },
        "candidate_checkpoint_written": False,
        "candidate_authorized": False,
    }


def _reviewed(
    candidate_id: str,
    *,
    direction: str = "g5_scene_sign",
    alpha: float = 0.125,
    inventory_index: int = 0,
    tier: int = 3,
    integer_surplus: int = 0,
    continuous_headroom: float = 0.02,
    l2: float | None = None,
    authorized_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "direction_id": direction,
        "alpha": alpha,
        "authorized_surface_state_sha256": authorized_hash or _digest(candidate_id),
        "full_tensor_state_sha256": _digest(f"full:{candidate_id}"),
        "inventory_index": inventory_index,
        "eligibility": {
            "eligible": True,
            "robustness_tier": tier,
            "minimum_integer_surplus": integer_surplus,
            "minimum_continuous_headroom": continuous_headroom,
            "authorized_surface_l2_perturbation": l2,
        },
    }


def _screen() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for direction in gate._DIRECTION_IDS:
        for alpha in gate._ALPHA_GRID:
            candidate_id = f"{direction}_alpha_{str(alpha).replace('.', 'p')}"
            selected = candidate_id == gate._EXPECTED_SELECTION["candidate_id"]
            row = _candidate(
                candidate_id=candidate_id,
                direction=direction,
                alpha=alpha,
                complete=9,
                positive=34,
                cross=18 if selected else 17,
                physical=4,
                improvement=(
                    gate._EXPECTED_SELECTION["priority_deficit_improvement"]
                    if selected
                    else 0.75
                ),
                broad=(gate._EXPECTED_SELECTION["broad_nll"] if selected else 2.90),
                q5=gate._EXPECTED_SELECTION["q5_margin"] if selected else 0.0,
                q699=gate._EXPECTED_SELECTION["q699_margin"] if selected else 0.125,
                authorized_hash=(
                    gate._EXPECTED_SELECTION["authorized_surface_state_sha256"]
                    if selected
                    else None
                ),
                full_hash=(
                    gate._EXPECTED_SELECTION["full_tensor_state_sha256"]
                    if selected
                    else None
                ),
            )
            candidates.append(row)
            inventory.append(
                {
                    field: row[field]
                    for field in (
                        "candidate_id",
                        "direction_id",
                        "alpha",
                        "authorized_surface_state_sha256",
                        "full_tensor_state_sha256",
                    )
                }
            )
    restorations = [
        {
            "candidate_id": row["candidate_id"],
            "phase": phase,
            "passed": True,
            "full_tensor_state_sha256": gate._SOURCE_FULL_SHA256,
        }
        for row in candidates
        for phase in ("before", "after")
    ]
    return {
        "schema_version": 1,
        "artifact": "v46_v45_u4_lost_side_no_step_diagnostic",
        "screen_integrity_passed": True,
        "terminal": {
            "sha256": gate._V45_TERMINAL_SHA256,
            "authorization_id": "v46_train_only_checkpoint_gradient_diagnostic",
        },
        "source_audit": {
            "full_tensor_state_sha256": gate._SOURCE_FULL_SHA256,
            "authorized_surface_state_sha256": gate._SOURCE_AUTHORIZED_SHA256,
            "frozen_state_sha256": gate._SOURCE_FROZEN_SHA256,
            "optimizer_file_opened": False,
            "optimizer_state_deserialized": False,
            "optimizer_state_loaded": False,
        },
        "source_replay": {"passed": True},
        "gradient_audit": {
            "source_state_unchanged": True,
            "optimizer_constructed_or_loaded": False,
            "source_state_after_gradient_measurement": {
                "full_tensor_state_sha256": gate._SOURCE_FULL_SHA256,
                "authorized_surface_state_sha256": gate._SOURCE_AUTHORIZED_SHA256,
                "frozen_state_sha256": gate._SOURCE_FROZEN_SHA256,
            },
        },
        "candidate_inventory": {
            "formula": "float32_P0-alpha*lr_group*sign(g5)",
            "direction_ids": list(gate._DIRECTION_IDS),
            "alpha_grid": list(gate._ALPHA_GRID),
            "candidate_count": 15,
            "candidate_hashes_fixed_before_candidate_forward_evaluation": True,
            "candidate_inventory_sha256": gate._canonical_sha256(inventory),
            "candidates": inventory,
        },
        "candidate_results": candidates,
        "all_15_candidates_received_full_25_unit_metrics": True,
        "all_15_candidates_received_fixed_48_row_broad_nll": True,
        "candidate_selection_performed": False,
        "adaptive_direction_or_scalar_selection": False,
        "candidate_authorization_granted": False,
        "candidate_checkpoint_written": False,
        "restoration_audit": restorations,
        "final_state": {
            "passed": True,
            "all_15_before_after_restorations_passed": True,
            "full_tensor_state_sha256": gate._SOURCE_FULL_SHA256,
            "authorized_surface_state_sha256": gate._SOURCE_AUTHORIZED_SHA256,
            "frozen_state_sha256": gate._SOURCE_FROZEN_SHA256,
        },
        "optimizer_constructed_or_loaded": False,
        "optimizer_state_file_opened": False,
        "optimizer_step_executed": False,
        "parameter_state_persisted": False,
        "all_16_training_maps_loaded": True,
        "validation_qa_loaded": False,
        "validation_environment_maps_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_executed": False,
        "runtime_promotion_executed": False,
        "chat_promotion_executed": False,
        "embodied_promotion_executed": False,
        "protected_report_sha256_before_and_after": gate._PROTECTED_REPORT_SHA256,
        "forbidden_file_accesses": [],
    }


def test_pre_result_scaffold_authenticates_pins_without_opening_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_value: str) -> dict[str, Any]:
        raise AssertionError("placeholder scaffold must not open the V46 report")

    monkeypatch.setattr(gate, "load_and_review_screen", forbidden)
    report = gate.build_terminal_scaffold(gate.REPORT_SHA256_PLACEHOLDER)
    assert report["pre_result_policy_fixed"] is True
    assert report["v46_report_reference"] == {
        "status": "pending_explicit_sha256",
        "expected_sha256": gate.REPORT_SHA256_PLACEHOLDER,
        "report_opened": False,
        "report_authenticated": False,
    }
    assert report["advisory_result_review"] is None
    assert report["successor_review"] == gate._SUCCESSOR_REVIEW_PLACEHOLDER
    assert report["only_exact_successor_authorized"] is None
    assert report["terminal_materialization_authorized"] is False
    assert report["candidate_checkpoint_write_authorized"] is False
    assert report["validation_access_authorized"] is False


def test_report_hash_must_be_explicit_lowercase_hex_or_placeholder() -> None:
    with pytest.raises(ValueError, match="64 lowercase"):
        gate._report_hash_reference("not-a-hash")
    with pytest.raises(ValueError, match="64 lowercase"):
        gate._report_hash_reference("A" * 64)
    explicit = gate._report_hash_reference("a" * 64)
    assert explicit["expected_sha256"] == "a" * 64
    assert explicit["report_opened"] is False


@pytest.mark.parametrize(
    ("margin", "tier"),
    [
        (0.125, 3),
        (0.124999, 2),
        (0.0625, 2),
        (0.062499, 1),
        (1e-9, 1),
        (0.0, 0),
        (-0.1, 0),
    ],
)
def test_robustness_tiers_are_exact(margin: float, tier: int) -> None:
    assert gate.robustness_tier(margin) == tier


def test_candidate_eligibility_recomputes_every_fixed_threshold() -> None:
    row = _candidate()
    # The real report is emitted with sorted JSON keys, so serialized mapping
    # order must not affect the fixed gate.
    row["threshold_diagnostic"]["checks"] = dict(
        sorted(row["threshold_diagnostic"]["checks"].items())
    )
    evidence = gate.candidate_eligibility(row)
    assert evidence["eligible"] is True
    assert all(evidence["checks"].values())
    assert evidence["robustness_tier"] == 3
    assert evidence["minimum_integer_surplus"] == 0
    assert evidence["minimum_continuous_headroom"] == pytest.approx(
        min(0.25, gate._BROAD_NLL_MAXIMUM - 2.90)
    )


@pytest.mark.parametrize(
    "override",
    [
        {"complete": 8},
        {"positive": 33},
        {"cross": 16},
        {"physical": 3},
        {"improvement": 0.499},
        {"broad": gate._BROAD_NLL_MAXIMUM + 1e-6},
        {"q5": 0.0},
        {"q699": 0.0},
    ],
)
def test_candidate_is_ineligible_if_any_one_fixed_threshold_fails(
    override: dict[str, Any],
) -> None:
    assert gate.candidate_eligibility(_candidate(**override))["eligible"] is False


def test_persisted_gate_boolean_cannot_override_recomputation() -> None:
    row = _candidate(complete=8)
    row["threshold_diagnostic"]["checks"][gate._CHECK_NAMES[0]] = True
    row["threshold_diagnostic"]["all_numeric_thresholds_met"] = True
    with pytest.raises(ValueError, match="differ from recomputation"):
        gate.candidate_eligibility(row)


def test_ranking_policy_is_lexicographic_in_precommitted_order() -> None:
    rows = [
        _reviewed(
            "lower_tier_better_elsewhere",
            tier=2,
            integer_surplus=20,
            continuous_headroom=10.0,
            inventory_index=0,
        ),
        _reviewed(
            "tier3_low_surplus",
            tier=3,
            integer_surplus=0,
            continuous_headroom=0.01,
            inventory_index=1,
        ),
        _reviewed(
            "tier3_high_surplus_low_headroom",
            tier=3,
            integer_surplus=1,
            continuous_headroom=0.02,
            inventory_index=2,
        ),
        _reviewed(
            "tier3_high_surplus_high_headroom",
            tier=3,
            integer_surplus=1,
            continuous_headroom=0.03,
            inventory_index=3,
        ),
    ]
    ranked = gate.rank_eligible_candidates(rows)
    assert [row["candidate_id"] for row in ranked] == [
        "tier3_high_surplus_high_headroom",
        "tier3_high_surplus_low_headroom",
        "tier3_low_surplus",
        "lower_tier_better_elsewhere",
    ]


def test_l2_alpha_direction_inventory_and_hash_are_ordered_after_headroom() -> None:
    common = {
        "tier": 3,
        "integer_surplus": 1,
        "continuous_headroom": 0.03,
    }
    rows = [
        _reviewed(
            "larger_l2",
            l2=2.0,
            alpha=0.125,
            inventory_index=0,
            **common,
        ),
        _reviewed(
            "smaller_l2_larger_alpha",
            l2=1.0,
            alpha=0.25,
            inventory_index=1,
            **common,
        ),
        _reviewed(
            "smaller_l2_alpha_scene_late_inventory",
            l2=1.0,
            alpha=0.125,
            direction="g5_scene_sign",
            inventory_index=4,
            **common,
        ),
        _reviewed(
            "smaller_l2_alpha_query_early_inventory",
            l2=1.0,
            alpha=0.125,
            direction="g5_query_sign",
            inventory_index=2,
            **common,
        ),
        _reviewed(
            "smaller_l2_alpha_scene_early_inventory",
            l2=1.0,
            alpha=0.125,
            direction="g5_scene_sign",
            inventory_index=3,
            **common,
        ),
    ]
    ranked = gate.rank_eligible_candidates(rows)
    assert [row["candidate_id"] for row in ranked] == [
        "smaller_l2_alpha_scene_early_inventory",
        "smaller_l2_alpha_scene_late_inventory",
        "smaller_l2_alpha_query_early_inventory",
        "smaller_l2_larger_alpha",
        "larger_l2",
    ]
    assert all(row["authorized_surface_l2_criterion_available"] for row in ranked)


def test_missing_l2_is_skipped_uniformly_but_mixed_availability_fails_closed() -> None:
    no_l2 = [
        _reviewed("later", inventory_index=2),
        _reviewed("earlier", inventory_index=1),
    ]
    ranked = gate.rank_eligible_candidates(no_l2)
    assert [row["candidate_id"] for row in ranked] == ["earlier", "later"]
    assert not any(row["authorized_surface_l2_criterion_available"] for row in ranked)
    mixed = [_reviewed("missing", inventory_index=0), _reviewed("present", l2=1.0)]
    with pytest.raises(ValueError, match="mixed L2"):
        gate.rank_eligible_candidates(mixed)


def test_full_synthetic_screen_is_validated_and_ranked_without_authorization() -> None:
    result = gate.review_screen_payload(_screen())
    assert result["candidate_count"] == 15
    assert result["eligible_candidate_count"] == 1
    assert result["recommended_candidate_for_future_review"]["candidate_id"] == (
        "g5_both_sign_alpha_1p0"
    )
    assert result["result_authentication"]["passed"] is True
    assert result["result_authentication"]["selected_candidate"][
        "full_tensor_state_sha256"
    ] == gate._EXPECTED_SELECTION["full_tensor_state_sha256"]
    assert result["ranking_is_advisory_only"] is True
    assert result["candidate_authorization_granted"] is False
    assert result["candidate_checkpoint_write_authorized"] is False
    assert result["validation_access_authorized"] is False


def test_materialized_v46_report_has_exact_unique_reviewed_candidate() -> None:
    result = gate.load_and_review_screen(_V46_REPORT_SHA256)
    assert result["sha256"] == _V46_REPORT_SHA256
    review = result["review"]
    assert review["candidate_count"] == 15
    assert review["eligible_candidate_count"] == 1
    authentication = review["result_authentication"]
    assert authentication["passed"] is True
    selected = authentication["selected_candidate"]
    assert selected["candidate_id"] == gate._EXPECTED_SELECTION["candidate_id"]
    assert selected["authorized_surface_state_sha256"] == gate._EXPECTED_SELECTION[
        "authorized_surface_state_sha256"
    ]
    assert selected["full_tensor_state_sha256"] == gate._EXPECTED_SELECTION[
        "full_tensor_state_sha256"
    ]


def test_authenticated_scaffold_has_exact_v47_pins_and_is_ready() -> None:
    report = gate.build_terminal_scaffold(_V46_REPORT_SHA256)
    assert report["v46_report_reference"]["report_authenticated"] is True
    successor = report["successor_review"]
    assert successor["status"] == "v46_result_and_v47_implementation_authenticated"
    assert successor["v46_result_authenticated"] is True
    assert successor["selected_candidate_id"] == "g5_both_sign_alpha_1p0"
    assert successor["intended_successor_action"] == (
        "one_bounded_train_only_v47_four_step_book_support_continuation"
    )
    assert successor["v47_maximum_optimizer_updates"] == 4
    assert successor["v47_config_sha256"] == gate._V47_CONFIG_SHA256
    assert successor["v47_trainer_sha256"] == gate._V47_TRAINER_SHA256
    assert successor["v47_test_sha256"] == gate._V47_TEST_SHA256
    assert successor["v47_implementation_hashes_complete"] is True
    assert successor["exact_successor_action"] == gate._V47_AUTHORIZATION_ID
    assert report["terminal_materialization_authorized"] is True
    assert report["candidate_checkpoint_write_authorized"] is False
    assert report["validation_access_authorized"] is False


def test_screen_fails_closed_on_selection_write_or_restricted_access() -> None:
    for key in (
        "candidate_selection_performed",
        "candidate_authorization_granted",
        "candidate_checkpoint_written",
        "validation_qa_loaded",
        "oracle_loaded",
        "final_test_scenes_touched",
        "selector_executed",
        "runtime_promotion_executed",
    ):
        screen = _screen()
        screen[key] = True
        with pytest.raises(ValueError, match="fixed envelope changed"):
            gate.review_screen_payload(screen)


def test_screen_requires_the_exact_unique_candidate_and_tensor_hashes() -> None:
    screen = _screen()
    selected = next(
        row
        for row in screen["candidate_results"]
        if row["candidate_id"] == gate._EXPECTED_SELECTION["candidate_id"]
    )
    selected["full_tensor_state_sha256"] = "0" * 64
    selected["candidate_state_before_forward"]["full_tensor_state_sha256"] = "0" * 64
    inventory = screen["candidate_inventory"]["candidates"]
    inventory[gate._EXPECTED_SELECTION["inventory_index"]][
        "full_tensor_state_sha256"
    ] = "0" * 64
    screen["candidate_inventory"]["candidate_inventory_sha256"] = gate._canonical_sha256(
        inventory
    )
    with pytest.raises(ValueError, match="independently reviewed result"):
        gate.review_screen_payload(screen)


def test_screen_rejects_more_than_one_eligible_candidate() -> None:
    screen = _screen()
    first = screen["candidate_results"][0]
    first["focus_units"][gate._Q5]["side_margins"][0] = 0.125
    first["threshold_diagnostic"]["retention_diagnostics"][
        "both_lost_sides_strictly_positive"
    ] = True
    first["threshold_diagnostic"]["checks"][gate._CHECK_NAMES[6]] = True
    first["threshold_diagnostic"]["all_numeric_thresholds_met"] = True
    with pytest.raises(ValueError, match="exactly one eligible candidate"):
        gate.review_screen_payload(screen)


def test_terminal_authorizes_only_exact_v47_and_keeps_restricted_actions_closed() -> None:
    assert all(
        gate._SUCCESSOR_REVIEW_PLACEHOLDER[field] is False
        for field in (
            "candidate_checkpoint_write_authorized",
            "validation_access_authorized",
            "oracle_access_authorized",
            "final_test_access_authorized",
            "selector_execution_authorized",
            "runtime_promotion_authorized",
            "chat_promotion_authorized",
            "embodied_promotion_authorized",
        )
    )
    report = gate.build_terminal_report(_V46_REPORT_SHA256)
    assert report["passed"] is True
    assert report["only_exact_successor_authorized"] == gate._V47_AUTHORIZATION_ID
    authorization = report["conditional_successor_authorization"]
    assert authorization == gate._v47_authorization(_V46_REPORT_SHA256)
    assert authorization["implementation_integrity"] == {
        "config_sha256": gate._V47_CONFIG_SHA256,
        "trainer_sha256": gate._V47_TRAINER_SHA256,
        "test_sha256": gate._V47_TEST_SHA256,
    }
    assert authorization["source"]["candidate_full_tensor_state_sha256"] == (
        gate._EXPECTED_SELECTION["full_tensor_state_sha256"]
    )
    assert authorization["training"]["optimizer_steps"] == 4
    assert authorization["training"]["checkpoint_steps"] == [0, 2, 4]
    assert authorization["scope"]["validation_access_authorized"] is False
    assert report["standalone_v46_candidate_checkpoint_write_authorized"] is False
    assert report["validation_access_authorized"] is False


def test_terminal_writer_is_pinned_atomic_and_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "terminal.json"
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    report = gate.write_report(
        output,
        expected_v46_report_sha256=_V46_REPORT_SHA256,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="one-shot"):
        gate.write_report(
            output,
            expected_v46_report_sha256=_V46_REPORT_SHA256,
        )
    with pytest.raises(ValueError, match="pinned"):
        gate.write_report(
            tmp_path / "other.json",
            expected_v46_report_sha256=_V46_REPORT_SHA256,
        )


def test_materialized_terminal_replays_exactly_if_present() -> None:
    path = gate._resolve(gate.DEFAULT_OUTPUT)
    if not path.is_file():
        pytest.skip("V46 terminal is materialized after its implementation tests pass")
    assert json.loads(path.read_text(encoding="utf-8")) == gate.build_terminal_report(
        _V46_REPORT_SHA256
    )


def test_tampered_v46_implementation_pin_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_V46_SCREEN_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="bytes changed"):
        gate.build_terminal_scaffold(gate.REPORT_SHA256_PLACEHOLDER)


def test_tampered_v47_implementation_pin_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_V47_TRAINER_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="bytes changed"):
        gate.build_terminal_report(_V46_REPORT_SHA256)


def test_review_does_not_mutate_input() -> None:
    screen = _screen()
    before = copy.deepcopy(screen)
    gate.review_screen_payload(screen)
    assert screen == before
