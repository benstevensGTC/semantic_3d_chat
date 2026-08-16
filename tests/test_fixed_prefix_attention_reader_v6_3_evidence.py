from __future__ import annotations

from semantic_3d_chat.evaluation import fixed_prefix_attention_reader_v6_3_evidence as evidence


def test_v6_3_pinned_bytes_and_reports_authenticate() -> None:
    assert len(evidence.authenticate_pinned_bytes()) == 6
    assert evidence.authenticate_gradient_report()["passed"] is True
    pilot = evidence.authenticate_pilot_report()
    assert pilot["passed"] is True
    assert pilot["promotion_authorized"] is False
    assert pilot["positive_margin_side_delta"] == 1
    assert pilot["complete_unit_delta"] == 2


def test_v6_3_terminal_decision_is_positive_but_nonpromotable() -> None:
    marker = evidence.build_terminal_marker()
    assert marker["status"] == (
        "positive_train_only_pilot_continuation_authorized_no_runtime_promotion"
    )
    assert marker["runtime_checkpoint_promotion_authorized"] is False
    assert marker["runtime_checkpoint_exists"] is False
    assert marker["continuation"] == "v6_4_pair_disjoint_train_only_confirmation"
    assert marker["continuation_may_read_internal_validation"] is False
    assert marker["continuation_may_read_deferred_or_final"] is False
    assert marker["continuation_may_read_oracle"] is False
