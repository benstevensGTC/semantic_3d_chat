from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import fixed_prefix_decoder_reader_v6_1_release as release


def _objective_metrics() -> dict[str, object]:
    digest = "a" * 64
    prepared_fields = {
        name: {
            "reference_sha256": digest,
            "selected_sha256": digest,
            "shape": [1, 2],
            "dtype": "torch.float32",
            "exact": True,
        }
        for name in (
            "inputs_embeds",
            "attention_mask",
            "per_layer_inputs",
            "mm_token_type_ids",
            "labels",
        )
    }
    return {
        "contract_version": release.OBJECTIVE_EQUIVALENCE_THRESHOLDS[
            "contract_version"
        ],
        "thresholds": release.OBJECTIVE_EQUIVALENCE_THRESHOLDS,
        "token_count": 2,
        "vocabulary_size": 20,
        "prepared_identity": {"fields": prepared_fields, "all_exact": True},
        "index_identity": {
            "target_token_ids": [3, 4],
            "label_positions": [10, 11],
            "causal_positions": [9, 10],
            "targets_exact": True,
            "label_positions_exact": True,
            "causal_positions_exact": True,
            "reference_targets_sha256": digest,
            "selected_targets_sha256": digest,
            "reference_label_positions_sha256": digest,
            "selected_label_positions_sha256": digest,
            "reference_causal_positions_sha256": digest,
            "selected_causal_positions_sha256": digest,
        },
        "hidden_identity": {
            "hook_module": "model.language_model.norm",
            "accessible": True,
            "entire_exact": True,
            "selected_exact": True,
            "entire_shape": [1, 12, 8],
            "selected_shape": [1, 2, 8],
            "reference_entire_sha256": digest,
            "selected_entire_sha256": digest,
            "reference_selected_sha256": digest,
            "selected_selected_sha256": digest,
        },
        "common_shape_reprojection": {
            "shape": [1, 2, 20],
            "logits_exact": True,
            "nll_exact": True,
            "reference_logits_sha256": digest,
            "selected_logits_sha256": digest,
            "reference_per_token_nll": [1.0, 2.0],
            "selected_per_token_nll": [1.0, 2.0],
        },
        "hf_loss_manual_ce": {
            "hf_batch1_loss": 1.5,
            "manual_full_fp32_ce": 1.5,
            "absolute_difference": 0.0,
            "passed": True,
        },
        "raw_postsoftcap_logits": {
            "reference_shape": [1, 2, 20],
            "selected_shape": [1, 2, 20],
            "sufficient_statistics": {
                "reference_logits_sha256": "b" * 64,
                "selected_logits_sha256": "c" * 64,
                "per_token": [
                    {
                        "vocabulary_count": 20,
                        "difference_sum_abs": 0.02,
                        "difference_sum_squares": 0.0004,
                        "difference_max_abs": 0.02,
                        "reference_selected_dot": 99.9998,
                        "reference_sum_squares": 100.0,
                        "selected_sum_squares": 100.0,
                    },
                    {
                        "vocabulary_count": 20,
                        "difference_sum_abs": 0.01,
                        "difference_sum_squares": 0.0001,
                        "difference_max_abs": 0.01,
                        "reference_selected_dot": 99.99995,
                        "reference_sum_squares": 100.0,
                        "selected_sum_squares": 100.0,
                    },
                ],
            },
            "byte_exact": False,
            "max_abs_difference": 0.02,
            "rms_difference": 0.0035355339059327377,
            "mean_abs_difference": 0.00075,
            "per_token_cosine_similarity": [0.999998, 0.9999995],
            "minimum_per_token_cosine_similarity": 0.999998,
        },
        "nll": {
            "reference_per_token": [1.0, 2.0],
            "selected_per_token": [1.0000005, 2.0000005],
            "max_abs_difference": 0.000000500000000069889,
            "reference_mean": 1.5,
            "selected_mean": 1.5000005,
            "mean_absolute_difference": 0.000000500000000069889,
        },
        "distribution": {
            "js_divergence_by_token": [1e-8, 2e-8],
            "maximum_js_divergence": 2e-8,
            "softmax_ce_gradient_max_abs_difference": 1e-5,
            "softmax_ce_gradient_cosine_similarity": 0.999999,
        },
        "predictions_and_ranks": {
            "reference_top1_token_ids": [1, 2],
            "selected_top1_token_ids": [1, 2],
            "top1_exact": True,
            "top5_overlap_fraction_by_token": [1.0, 0.8],
            "minimum_top5_overlap_fraction": 0.8,
            "reference_target_top10_membership": [True, False],
            "selected_target_top10_membership": [True, False],
            "target_top10_membership_exact": True,
            "reference_target_ranks": [3, 12],
            "selected_target_ranks": [4, 12],
            "per_token_max_vocabulary_abs_logit_difference": [0.02, 0.01],
            "per_token_rank_tie_bands": [0.04, 0.02],
            "maximum_crossed_reference_target_gap_by_token": [0.03, 0.0],
            "target_rank_changes_confined_to_tie_band": True,
            "reference_strict_above_band_ranks": [2, 11],
            "selected_strict_above_band_ranks": [2, 11],
            "strict_above_band_rank_exact": True,
        },
        "passed": True,
    }


def _gradient_metrics() -> dict[str, object]:
    comparisons = {}
    for branch in ("correct", "wrong", "broad", "aggregate"):
        comparisons[branch] = {
            "full_norm": 2.0,
            "tail_norm": 2.0,
            "cosine_similarity": 0.999999995,
            "relative_l2": 0.0001,
            "norm_ratio": 1.0,
            "full_lora_b_gradient_l2_by_target": {
                target: 2.0**0.5 for target in release.TARGET_MODULES
            },
            "tail_lora_b_gradient_l2_by_target": {
                target: 2.0**0.5 for target in release.TARGET_MODULES
            },
            "full_lora_a_gradient_l2_by_target": {
                target: 0.0 for target in release.TARGET_MODULES
            },
            "tail_lora_a_gradient_l2_by_target": {
                target: 0.0 for target in release.TARGET_MODULES
            },
            "full_lora_a_exact_zero": True,
            "tail_lora_a_exact_zero": True,
            "full_coverage": list(release.TARGET_MODULES),
            "tail_coverage": list(release.TARGET_MODULES),
            "coverage_exact": True,
            "sufficient_statistics": {
                "element_count": 8,
                "full_vector_sha256": "d" * 64,
                "tail_vector_sha256": "e" * 64,
                "full_sum_squares": 4.0,
                "tail_sum_squares": 4.0,
                "full_tail_dot": 3.99999998,
                "difference_sum_squares": 0.00000004,
            },
            "passed": True,
        }
    return {
        "contract_version": release.GRADIENT_EQUIVALENCE_THRESHOLDS[
            "contract_version"
        ],
        "thresholds": release.GRADIENT_EQUIVALENCE_THRESHOLDS,
        "objective_values": {
            "full_correct_nll": 1.0,
            "tail_correct_nll": 1.0,
            "correct_nll_abs_difference": 0.0,
            "full_wrong_nll": 1.2,
            "tail_wrong_nll": 1.2,
            "wrong_nll_abs_difference": 0.0,
            "full_broad_nll": 1.1,
            "tail_broad_nll": 1.1,
            "broad_nll_abs_difference": 0.0,
            "full_margin": 0.2,
            "tail_margin": 0.2,
            "margin_abs_difference": 0.0,
            "full_composite": 2.25,
            "tail_composite": 2.25,
            "composite_abs_difference": 0.0,
            "full_hinge_active": True,
            "tail_hinge_active": True,
        },
        "gradient_comparisons": comparisons,
        "retention_self_kl": 0.0,
        "retention_gradient": {
            "measured_from_freshly_zeroed_gradients": True,
            "lora_a_exact_zero": False,
            "lora_b_gradient_l2_by_target": {
                target: 0.0 for target in release.TARGET_MODULES
            },
        },
        "passed": True,
    }


def _set_path(value: dict[str, object], path: tuple[str, ...], replacement: object) -> None:
    cursor: object = value
    for component in path[:-1]:
        assert isinstance(cursor, dict)
        cursor = cursor[component]
    assert isinstance(cursor, dict)
    cursor[path[-1]] = replacement


def _gradient_terminal_summary() -> dict[str, object]:
    gradient = _gradient_metrics()
    a_values = {target: 0.0 for target in release.TARGET_MODULES}
    b_values = {target: 2.0**0.5 for target in release.TARGET_MODULES}
    return {
        "v6_lora_a_gradient_l2_expected_zero_by_target": a_values,
        "v6_lora_b_gradient_l2_by_target": b_values,
        "v6_gradient_by_module": {
            target: {
                "lora_a": 0.0,
                "lora_b": 2.0**0.5,
                "total_l2": 2.0**0.5,
            }
            for target in release.TARGET_MODULES
        },
        "v6_gradient_l2": 2.0,
        "gradient_equivalence": gradient,
        "contrastive_correct_nll": 1.0,
        "contrastive_wrong_nll": 1.2,
        "contrastive_margin": 0.2,
        "broad_nll": 1.1,
        "retention_self_kl": 0.0,
    }


def test_v6_1_binds_exact_consumed_v6_failure() -> None:
    assert release.sha256_file(release.V6_PREREGISTRATION) == (
        release.V6_PREREGISTRATION_SHA256
    )
    assert release.sha256_file(release.V6_RELEASE) == release.V6_RELEASE_SHA256
    assert release.sha256_file(release.V6_ATTEMPT) == release.V6_ATTEMPT_SHA256
    assert release.sha256_file(release.V6_TERMINAL_FAILURE) == (
        release.V6_TERMINAL_FAILURE_SHA256
    )
    terminal = json.loads(Path(release.V6_TERMINAL_FAILURE).read_text())
    assert terminal["status"] == release.V6_TERMINAL_FAILURE_STATUS
    assert terminal["failure_message"] == release.V6_TERMINAL_FAILURE_MESSAGE


def test_v6_1_preregistered_thresholds_match_independent_audit() -> None:
    objective = release.OBJECTIVE_EQUIVALENCE_THRESHOLDS
    gradient = release.GRADIENT_EQUIVALENCE_THRESHOLDS
    assert objective["raw_logits_max_abs"] == 0.25
    assert objective["raw_logits_rms"] == 0.01
    assert objective["raw_logits_mean_abs"] == 0.002
    assert objective["raw_logits_per_token_cosine_min"] == 0.99999
    assert objective["per_token_nll_max_abs"] == 0.00002
    assert objective["mean_nll_abs"] == 0.000001
    assert objective["js_divergence_max"] == 0.000001
    assert objective["top_k"] == 5
    assert objective["top_k_minimum_overlap_fraction"] == 0.8
    assert objective["rank_tie_band_multiplier"] == 2.0
    assert gradient["branch_nll_abs"] == 0.000001
    assert gradient["margin_abs"] == 0.000002
    assert gradient["composite_abs"] == 0.000005
    assert gradient["gradient_cosine_min"] == 0.99999
    assert gradient["gradient_relative_l2_max"] == 0.005
    assert (gradient["gradient_norm_ratio_min"], gradient["gradient_norm_ratio_max"]) == (
        0.995,
        1.005,
    )


def test_v6_1_installed_transformers_sources_are_exact() -> None:
    observed = release._installed_transformers_sources()
    assert {
        name: (value["sha256"], value["size_bytes"])
        for name, value in observed.items()
    } == release.INSTALLED_TRANSFORMERS_SOURCE_BINDINGS


def test_v6_1_create_once_claim_refuses_second_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempt = tmp_path / "attempt.json"
    terminal = tmp_path / "terminal.json"
    monkeypatch.setattr(release, "MPS_SMOKE_ATTEMPT", str(attempt))
    monkeypatch.setattr(release, "MPS_SMOKE_REPORT", str(terminal))
    monkeypatch.setattr(
        release,
        "authenticate_v6_1_mps_smoke_release",
        lambda: ({"status": "released"}, "a" * 64),
    )
    first, _digest = release.claim_v6_1_mps_smoke_attempt()
    assert first == attempt
    with pytest.raises(FileExistsError, match="create-once"):
        release.claim_v6_1_mps_smoke_attempt()


def test_v6_1_objective_authenticator_rejects_each_critical_gate() -> None:
    metrics = _objective_metrics()
    assert release.objective_equivalence_passes(metrics)
    metrics["hidden_identity"]["entire_exact"] = False  # type: ignore[index]
    assert not release.objective_equivalence_passes(metrics)
    metrics = _objective_metrics()
    metrics["predictions_and_ranks"][  # type: ignore[index]
        "per_token_rank_tie_bands"
    ][0] = 0.5
    assert not release.objective_equivalence_passes(metrics)
    metrics = _objective_metrics()
    metrics["nll"]["max_abs_difference"] = 0.1  # type: ignore[index]
    assert not release.objective_equivalence_passes(metrics)


def test_v6_1_gradient_authenticator_rejects_coverage_and_drift() -> None:
    metrics = _gradient_metrics()
    assert release.gradient_equivalence_passes(metrics)
    metrics["gradient_comparisons"]["wrong"][  # type: ignore[index]
        "relative_l2"
    ] = 0.02
    assert not release.gradient_equivalence_passes(metrics)
    metrics = _gradient_metrics()
    metrics["gradient_comparisons"]["aggregate"][  # type: ignore[index]
        "tail_coverage"
    ] = []
    assert not release.gradient_equivalence_passes(metrics)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("objective_values", "correct_nll_abs_difference"), 1e-7),
        (("objective_values", "wrong_nll_abs_difference"), 1e-7),
        (("objective_values", "broad_nll_abs_difference"), 1e-7),
        (("objective_values", "full_margin"), 0.3),
        (("objective_values", "tail_margin"), 0.3),
        (("objective_values", "margin_abs_difference"), 1e-7),
        (("objective_values", "full_hinge_active"), False),
        (("objective_values", "tail_hinge_active"), False),
        (("objective_values", "full_composite"), 2.3),
        (("objective_values", "tail_composite"), 2.3),
        (("objective_values", "composite_abs_difference"), 1e-7),
        (("objective_values", "full_correct_nll"), -1.0),
        (("objective_values", "tail_wrong_nll"), float("nan")),
    ),
)
def test_v6_1_gradient_objective_rejects_derived_scalar_tampering(
    path: tuple[str, ...], replacement: object
) -> None:
    metrics = _gradient_metrics()
    assert release.gradient_equivalence_passes(metrics)
    _set_path(metrics, path, replacement)
    assert not release.gradient_equivalence_passes(metrics)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("raw_postsoftcap_logits", "max_abs_difference"), 0.01),
        (("raw_postsoftcap_logits", "rms_difference"), 0.0001),
        (("raw_postsoftcap_logits", "mean_abs_difference"), 0.0001),
        (("raw_postsoftcap_logits", "minimum_per_token_cosine_similarity"), 1.0),
        (("raw_postsoftcap_logits", "byte_exact"), True),
        (
            (
                "raw_postsoftcap_logits",
                "sufficient_statistics",
                "reference_logits_sha256",
            ),
            "c" * 64,
        ),
        (("nll", "reference_mean"), 1.6),
        (("nll", "selected_mean"), 1.6),
        (("nll", "mean_absolute_difference"), 0.0),
        (("hf_loss_manual_ce", "absolute_difference"), 1e-7),
        (
            (
                "predictions_and_ranks",
                "per_token_max_vocabulary_abs_logit_difference",
            ),
            [0.01, 0.01],
        ),
    ),
)
def test_v6_1_objective_rejects_sufficient_evidence_tampering(
    path: tuple[str, ...], replacement: object
) -> None:
    metrics = _objective_metrics()
    assert release.objective_equivalence_passes(metrics)
    _set_path(metrics, path, replacement)
    assert not release.objective_equivalence_passes(metrics)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("vocabulary_count", 19),
        ("difference_sum_abs", 0.03),
        ("difference_sum_squares", 0.0003),
        ("difference_max_abs", 0.019),
        ("reference_selected_dot", 99.0),
        ("reference_sum_squares", 101.0),
        ("selected_sum_squares", 101.0),
    ),
)
def test_v6_1_objective_rejects_raw_statistic_tampering(
    field: str, replacement: object
) -> None:
    metrics = _objective_metrics()
    assert release.objective_equivalence_passes(metrics)
    raw = metrics["raw_postsoftcap_logits"]  # type: ignore[index]
    raw["sufficient_statistics"]["per_token"][0][field] = replacement
    assert not release.objective_equivalence_passes(metrics)


def test_v6_1_gradient_rejects_vector_statistic_tampering() -> None:
    metrics = _gradient_metrics()
    assert release.gradient_equivalence_passes(metrics)
    aggregate = metrics["gradient_comparisons"]["aggregate"]  # type: ignore[index]
    aggregate["sufficient_statistics"]["full_tail_dot"] = 3.0
    assert not release.gradient_equivalence_passes(metrics)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (
            (
                "v6_gradient_by_module",
                release.TARGET_MODULES[0],
                "total_l2",
            ),
            9.0,
        ),
        (("v6_gradient_l2",), 9.0),
        (
            (
                "gradient_equivalence",
                "gradient_comparisons",
                "aggregate",
                "tail_norm",
            ),
            9.0,
        ),
        (("contrastive_correct_nll",), 9.0),
    ),
)
def test_v6_1_terminal_gradient_summary_rejects_tampering(
    path: tuple[str, ...], replacement: object
) -> None:
    summary = _gradient_terminal_summary()
    assert release._gradient_report_consistency(summary)
    _set_path(summary, path, replacement)
    assert not release._gradient_report_consistency(summary)


def test_v6_1_release_authorizes_no_optimizer_training_or_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release, "_authenticate_frozen_lineage", lambda: {"ok": True})
    monkeypatch.setattr(release, "_authenticate_model_blob", lambda: {"ok": True})
    monkeypatch.setattr(release, "_software_versions", lambda: release.EXPECTED_SOFTWARE_VERSIONS)
    monkeypatch.setattr(release, "_installed_transformers_sources", lambda: {"ok": True})
    monkeypatch.setattr(release, "_bound_hashes", lambda _paths: {"source": "a" * 64})
    payload = release.build_v6_1_mps_smoke_release()
    assert payload["authorized"]["optimizer_construction"] is False
    assert payload["authorized"]["optimizer_steps"] == 0
    assert payload["authorized"]["multi_update_training"] is False
    assert payload["authorized"]["checkpoint_write"] is False
