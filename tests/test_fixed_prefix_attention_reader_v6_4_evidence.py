from __future__ import annotations

from semantic_3d_chat.evaluation import fixed_prefix_attention_reader_v6_4_evidence as evidence


def test_v6_4_negative_result_reauthenticates_from_raw_records() -> None:
    result = evidence.authenticate_result()
    assert result["passed"] is True
    assert result["status"] == "authenticated_negative_pair_disjoint_screen"
    assert result["train_delta"]["mean_margin"] > 0.0
    assert result["held_delta"]["mean_margin"] < 0.0
    assert result["held_delta"]["mean_margin_softplus"] > 0.0
    assert result["failed_checks"] == [
        "held_mean_margin_delta_at_least_0_002",
        "held_mean_margin_softplus_delta_at_most_minus_0_001",
    ]
    assert result["audit_clean"] is True
    assert result["checkpoint_absent"] is True


def test_v6_4_terminal_forbids_continuing_or_promoting_exact_surface() -> None:
    terminal = evidence.build_terminal()
    assert terminal["status"] == (
        "failed_pair_disjoint_generalization_no_checkpoint_no_promotion"
    )
    assert terminal["exact_attention_surface_continuation_authorized"] is False
    assert terminal["runtime_checkpoint_promotion_authorized"] is False
    assert terminal["runtime_checkpoint_exists"] is False
    assert terminal["internal_validation_consumed"] is False
    assert terminal["deferred_or_final_consumed"] is False
    assert terminal["oracle_consumed"] is False
