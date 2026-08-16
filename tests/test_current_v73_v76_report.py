from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_current_summary_authenticates_v73_through_v76_state() -> None:
    summary = _summary()
    evidence = summary["v73_v75_static_readout"]

    assert evidence["status"] == (
        "authenticated_v73_v74_rejected_v75_promoted_runtime_leakage_"
        "passed_official_validation_spatial_gate_failed_v76_rejected_"
        "v77_full_internal_screen_positive_not_promoted"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["checkpoint_published"] is True
    assert evidence["runtime_promotion_authorized"] is True
    assert evidence["official_validation_measured"] is True
    assert evidence["official_validation_question_manifest_used"] is True
    assert evidence["official_validation_answers_loaded_by_inference"] is False
    assert evidence["official_test_loaded"] is False
    assert evidence["oracle_loaded"] is False

    v73 = evidence["v73"]
    assert v73["status"] == "numeric_screen_failed"
    assert v73["full_scene_supported_accuracy"] == pytest.approx(0.1618798955613577)
    assert v73["dct40_supported_accuracy"] == pytest.approx(0.2193211488250653)
    assert v73["full_scene_prediction_change_units"] == 0
    assert v73["dct40_prediction_change_units"] == 18

    v74 = evidence["v74"]
    assert v74["status"] == "numeric_proxy_passed_real_gemma_failed_quarantined"
    assert v74["numeric"]["all_numeric_gates_passed"] is True
    assert v74["real_gemma"]["candidate_correct"] == 2
    assert v74["real_gemma"]["baseline_correct"] == 6
    assert v74["nll_train_only"]["answer_nll_before"] == pytest.approx(6.610632373227014)
    assert v74["nll_train_only"]["answer_nll_after"] == pytest.approx(3.855855609222393)
    assert v74["nll_train_only"]["held_optimization_rows"] == 0
    assert v74["nll_held_real_gemma"]["candidate_correct"] == 6
    assert v74["nll_held_real_gemma"]["baseline_correct"] == 6

    v75 = evidence["v75"]
    assert v75["status"] == (
        "promoted_runtime_leakage_passed_official_validation_completed_spatial_relation_gate_failed"
    )
    assert v75["numeric"]["all_numeric_gates_passed"] is True
    assert v75["real_gemma"]["candidate_correct"] == 4
    assert v75["real_gemma"]["wrong_scene_correct"] == 5
    assert v75["nll_train_only"]["answer_nll_before"] == pytest.approx(7.562225860440069)
    assert v75["nll_train_only"]["answer_nll_after"] == pytest.approx(2.506601880500562)
    assert v75["nll_train_only"]["held_optimization_rows"] == 0
    assert v75["nll_train_only"]["exact_zero_scene_verified"] is True
    held = v75["nll_held_real_gemma"]
    assert held["candidate_correct"] == 9
    assert held["baseline_correct"] == 6
    assert held["wrong_scene_correct"] == 6
    assert held["candidate_accuracy_gain"] == pytest.approx(0.1875)
    assert held["candidate_prediction_change_units"] == 2
    assert held["prediction_change_unit_total"] == 8
    assert held["accuracy_improved"] is True
    assert held["prediction_change_gate_passed"] is False
    full = v75["full_internal_development"]
    assert full["scope"] == "training_pool_pair_and_scene_disjoint_internal_development"
    assert full["untouched_official_validation"] is False
    assert full["row_count"] == 384
    assert full["scene_count"] == 16
    assert full["candidate_correct"] == 295
    assert full["candidate_accuracy"] == pytest.approx(295 / 384)
    assert full["v54_baseline_correct"] == 148
    assert full["v54_baseline_accuracy"] == pytest.approx(148 / 384)
    assert full["candidate_accuracy_gain"] == pytest.approx(147 / 384)
    assert full["wrong_paired_scene_original_target_correct"] == 278
    assert full["wrong_paired_scene_original_target_accuracy"] == pytest.approx(278 / 384)
    assert full["correct_over_wrong_scene_accuracy"] == pytest.approx(17 / 384)

    paired = v75["full_changed_pair_controls"]
    assert paired["changed_side_count"] == 52
    assert paired["changed_unit_count"] == 26
    assert paired["correct_scene_original_target_correct_sides"] == 31
    assert paired["v54_baseline_original_target_correct_sides"] == 18
    assert paired["wrong_scene_original_target_correct_sides"] == 14
    assert paired["wrong_scene_paired_target_correct_sides"] == 31
    assert paired["correct_scene_original_target_complete_units"] == 6
    assert paired["v54_baseline_original_target_complete_units"] == 0
    assert paired["wrong_scene_original_target_complete_units"] == 0
    assert paired["wrong_scene_paired_target_complete_units"] == 6
    assert paired["correct_scene_prediction_change_units"] == 12
    assert paired["v54_baseline_prediction_change_units"] == 6
    assert paired["correct_vs_wrong_scene_outputs_changed"] == 24
    assert paired["causal_paired_scene_target_following_demonstrated"] is True
    assert v75["promotion"] == {
        "internal_development_positive": True,
        "official_validation_measured": True,
        "official_validation_passed": False,
        "official_validation_failed_gates": ["spatial_relation_accuracy"],
        "runtime_leakage_suite_passed": True,
        "runtime_leakage_suite_status": "passed",
        "runtime_promotion_authorized": True,
        "checkpoint_published": True,
        "behavioral_acceptance_claimed": False,
    }

    release = v75["runtime_release"]
    assert release["status"] == "authenticated_sealed_two_file_v75_runtime_release"
    assert release["file_count"] == 2
    assert release["inventory"] == ["control.safetensors", "runtime_metadata.json"]
    assert release["control_checkpoint_sha256"] == (
        "5c2a79d47ddae145d7a3e459a0c91d4685f43475035dc3fa6d88364309a4b307"
    )
    assert release["environmental_text_inputs"] == []
    assert release["question_dependent_scene_retrieval"] is False
    assert release["all_256_environment_latents_attended"] is True

    leakage = v75["live_runtime_leakage"]
    assert leakage["passed"] is True
    assert leakage["loaded_file_count"] == 4198
    assert leakage["forbidden_access_count"] == 0
    assert leakage["qa_or_oracle_loaded"] is False
    assert leakage["training_artifact_loaded"] is False
    assert leakage["oracle_directory_renamed_and_restored"] is True
    assert leakage["training_directory_renamed_and_restored"] is True
    assert leakage["teacher_directory_renamed_and_restored"] is True
    assert leakage["prefix_computed_before_first_question"] is True
    assert leakage["prefix_invariant"] is True
    assert leakage["question_conditioned_scene_readout_tokens"] is True
    assert leakage["environment_conditioned_input_invariant"] is False
    assert leakage["strict_fixed_environment_embedding_input"] is False

    official = v75["official_validation"]
    assert official["status"] == ("completed_one_shot_failed_only_spatial_relation_gate")
    assert official["passed"] is False
    assert official["candidate_count"] == 1
    assert official["scene_count"] == 6
    assert official["question_count"] == 216
    assert official["canonical_correct"] == 167
    assert official["canonical_accuracy"] == pytest.approx(167 / 216)
    assert official["normalized_exact_accuracy"] == pytest.approx(165 / 216)
    assert official["per_type"]["attribute"]["correct"] == 30
    assert official["per_type"]["count"]["correct"] == 40
    assert official["per_type"]["metric"]["correct"] == 6
    assert official["per_type"]["orientation"]["correct"] == 6
    assert official["per_type"]["presence"]["correct"] == 38
    assert official["per_type"]["spatial_relation"]["correct"] == 28
    assert official["per_type"]["support"]["correct"] == 19
    assert official["changed_counterfactual"] == {
        "atomic_scene_pair_count": 3,
        "unit_count": 12,
        "side_count": 24,
        "complete_units": 7,
        "correct_sides": 17,
        "prediction_changed_units": 8,
        "successful_change_families": 3,
        "change_family_count": 3,
    }
    assert official["gates"]["spatial_relation_accuracy"] is False
    assert sum(not value for value in official["gates"].values()) == 1
    assert official["failed_gates"] == ["spatial_relation_accuracy"]
    assert official["grounding"]["mean_coordinate_error_m"] == pytest.approx(2.1355699299299333)
    assert official["grounding"]["within_1m_accuracy"] == 0.0
    assert official["complete_scene_prefix_built_before_questions"] is True
    assert official["prefix_invariant_within_scene"] is True
    assert official["all_256_environment_latents_attended"] is True
    assert official["prediction_process_accepts_answer_references"] is False

    demo = v75["live_demo"]
    assert demo["status"] == (
        "authenticated_live_v75_chat_with_vocabulary_free_broad_answer_fail_closed_guard"
    )
    assert demo["default_make_demo_uses_v75"] is False
    assert demo["default_make_demo_uses_v89"] is True
    assert demo["operator_role"] == "historical_question_conditioned_comparator"
    assert demo["scene_latents"] == 256
    assert demo["prefix_shape"] == [1, 258, 1536]
    assert demo["prefix_computed_before_first_question"] is True
    assert demo["prefix_invariant"] is True
    assert demo["loaded_file_count"] == 5197
    assert demo["forbidden_access_count"] == 0
    assert [row["answer"] for row in demo["successful_examples"]] == [
        "yes",
        "red",
    ]
    assert demo["successful_example_count"] == 2
    assert demo["well_formed_example_count"] == 3
    assert demo["historical_known_incorrect_example_count"] == 1
    assert demo["known_incorrect_examples"] == [
        {
            "question": "Is the bowl left or right of the chair?",
            "answer": "right",
            "expected": "left",
            "reason": (
                "The historical evaluator reversed subject and reference; world +X "
                "is right, and the bowl is left of the chair."
            ),
        }
    ]
    assert demo["scene1_bowl_to_chair_correct_relation"] == "left"
    assert demo["posthoc_relation_correction"]["sealed_v75_transcript_mutated"] is False
    assert demo["historical_malformed_example_count"] == 2
    assert demo["current_malformed_example_count"] == 0
    assert demo["guarded_unknown_example_count"] == 2
    assert [row["answer"] for row in demo["guarded_unknown_examples"]] == [
        "unknown",
        "unknown",
    ]
    assert demo["vocabulary_free_output_guard"] is True
    assert demo["answer_mapping_or_codebook_used"] is False
    assert demo["broad_list_qa_supported"] is False
    assert demo["broad_guard_prefix_invariant"] is True
    assert demo["broad_guard_loaded_file_count"] == 5197
    assert demo["broad_guard_forbidden_access_count"] == 0

    v76 = evidence["v76"]
    assert v76["status"] == "terminal_pair_contrast_held_smoke_no_gain_rejected"
    assert v76["measured_result_available"] is True
    assert v76["runtime_promotion_authorized"] is False
    assert v76["checkpoint_published"] is False
    assert v76["source_was_raw_pre_nll_v75"] is True
    assert v76["source_was_promoted_nll_v75"] is False
    assert v76["superseded_by_promoted_v75"] is True
    assert v76["training"]["changed_units"] == 40
    assert v76["training"]["changed_sides"] == 80
    assert v76["training"]["answer_nll_before"] == pytest.approx(7.900907715177164)
    assert v76["training"]["answer_nll_after"] == pytest.approx(4.371633912948892)
    assert v76["training"]["paired_margin_before"] == pytest.approx(4.154925424838439)
    assert v76["training"]["paired_margin_after"] == pytest.approx(5.632363477023318)
    assert v76["training"]["elapsed_seconds"] == pytest.approx(471.25590812484734)
    assert v76["held_smoke"] == {
        "row_count": 16,
        "candidate_correct": 6,
        "v54_baseline_correct": 6,
        "wrong_scene_correct": 6,
        "candidate_accuracy": 0.375,
        "candidate_accuracy_gain": 0.0,
        "correct_over_wrong_scene_accuracy": 0.0,
        "prediction_change_units": 4,
        "prediction_change_unit_total": 8,
    }
    v77 = evidence["v77"]
    assert v77["status"] == ("training_pool_historical_repair_full_internal_positive_not_promoted")
    assert v77["measured_result_available"] is True
    assert v77["runtime_promotion_authorized"] is False
    assert v77["checkpoint_published"] is False
    assert v77["source_candidate_sha256"] == (
        "d01275538489b3493a8e1ff080109d1db46832be6ca2a26f6d89d161c597188a"
    )
    assert v77["source_was_promoted_nll_v75"] is True
    assert v77["candidate_sha256"] == (
        "64cecd2e900ffd1763ce04b3ed51b20da8daa147830f402c5d5a561d50437256"
    )
    assert v77["training"]["selected_historical_rows"] == 72
    assert v77["training"]["distinct_answer_classes"] == 28
    assert v77["training"]["distinct_question_templates"] == 47
    assert v77["training"]["optimizer_steps"] == 9
    assert v77["training"]["correct_answer_nll_before"] == pytest.approx(2.5182164262887454)
    assert v77["training"]["correct_answer_nll_after"] == pytest.approx(1.604625632111644)
    assert v77["training"]["negative_answer_margin_before"] == pytest.approx(6.7364180881622815)
    assert v77["training"]["negative_answer_margin_after"] == pytest.approx(7.248878702775225)
    assert v77["held_smoke"] == {
        "row_count": 16,
        "candidate_correct": 9,
        "v54_baseline_correct": 6,
        "wrong_scene_correct": 6,
        "candidate_accuracy": 0.5625,
        "candidate_accuracy_gain": 0.1875,
        "correct_over_wrong_scene_accuracy": 0.1875,
        "prediction_change_units": 2,
        "prediction_change_unit_total": 8,
    }
    assert v77["full_internal_screen"] == {
        "measured": True,
        "scope": "training_pool_pair_and_scene_disjoint_internal_development",
        "row_count": 384,
        "scene_count": 16,
        "candidate_correct": 299,
        "candidate_accuracy": 299 / 384,
        "v75_correct": 295,
        "v75_accuracy": 295 / 384,
        "gain_correct": 4,
        "accuracy_gain": 4 / 384,
        "candidate_prediction_change_units": 35,
        "v75_prediction_change_units": 33,
        "wrong_scene_arm_measured": False,
        "elapsed_seconds": pytest.approx(144.92468141717836),
    }
    for path in (
        "reports/gemma4/metrics/v75_gemma_nll_balanced_held_full.json",
        "reports/gemma4/metrics/v75_gemma_nll_balanced_wrong_scene_full.json",
        "reports/gemma4/metrics/v62_v54_no_control_baseline_lock.json",
        "reports/gemma4/predictions/v62_v54_no_control_internal_validation.jsonl",
    ):
        assert summary["source_artifacts"][path] == BUILDER["V73_V75_EVIDENCE_SHA256"][Path(path)]
    for path, digest in {
        **BUILDER["V75_RUNTIME_EVIDENCE_SHA256"],
        **BUILDER["V75_OFFICIAL_VALIDATION_EVIDENCE_SHA256"],
        **BUILDER["V75_DEMO_EVIDENCE_SHA256"],
        **BUILDER["V75_BROAD_ANSWER_GUARD_EVIDENCE_SHA256"],
        **BUILDER["V76_EVIDENCE_SHA256"],
        **BUILDER["V77_EVIDENCE_SHA256"],
    }.items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_summary_authenticates_minimal_demo_release() -> None:
    summary = _summary()
    release = summary["demo_runtime_release"]

    assert release["status"] == "authenticated_minimal_two_file_v54_demo_release"
    assert release["evidence_authenticated"] is True
    assert release["inference_file_count"] == 2
    assert release["inference_inventory"] == [
        "adapter.safetensors",
        "runtime_metadata.json",
    ]
    assert release["environmental_text_inputs"] == []
    assert release["training_metadata_included"] is False
    assert release["model_promotion_claimed"] is False
    assert release["behavioral_acceptance_claimed"] is False
    for name, record in release["files"].items():
        path = Path(release["runtime_checkpoint"]) / name
        assert summary["source_artifacts"][path.as_posix()] == record["sha256"]


def test_current_summary_bounds_auto_scan_and_mcp_claim() -> None:
    evidence = _summary()["embodied_auto_scan_mcp"]

    assert evidence["status"] == (
        "live_semantic_mcp_and_embodied_conversation_scan_turn_refresh_passed_v75_controller_active"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["auto_scan_after_motion_enabled"] is True
    assert evidence["successful_motion_actions_covered"] == [
        "look",
        "turn",
        "move_forward",
        "move_backward",
        "move_to",
    ]
    assert evidence["rejected_motion_does_not_scan_tested"] is True
    assert evidence["map_and_prefix_refresh_after_motion_tested"] is True
    assert evidence["semantic_mcp_preflight_passed"] is True
    assert evidence["semantic_mcp_preflight_loaded_file_count"] == 4085
    assert evidence["semantic_mcp_preflight_forbidden_access_count"] == 0
    assert evidence["numeric_stdio_transport_passed"] is True
    assert evidence["mcp_sdk_version"] == "2.0.0"
    assert evidence["mcp_tool_count"] == 9
    assert evidence["live_semantic_motion_refresh_via_mcp_measured"] is True
    live = evidence["live_auto_motion_refresh"]
    assert live["passed"] is True
    assert live["turn_degrees"] == 15.0
    assert live["complete_image_encoder_calls_total"] == 2
    assert live["initial_observation_id"] == "o_000001"
    assert live["motion_observation_id"] == "o_000002"
    assert live["scan_count_total"] == 2
    assert live["scene_version_after_motion"] == 2
    assert live["map_version_after_motion"] == 2
    assert live["map_changed_after_motion"] is True
    assert live["prefix_changed_after_motion"] is True
    assert live["loaded_file_count"] == 125
    assert live["forbidden_access_count"] == 0
    assert live["oracle_or_qa_loaded"] is False
    assert live["transport"] == "direct_runtime_function"
    live_mcp = evidence["live_semantic_mcp"]
    assert live_mcp["passed"] is True
    assert live_mcp["transport"] == "stdio"
    assert live_mcp["protocol_version"] == "2025-11-25"
    assert live_mcp["base_checkpoint"] == "gemma4_v54_release_v1"
    assert live_mcp["control_checkpoint"] == "gemma4_v75_nll_control_release_v1"
    assert live_mcp["continuous_controller_active"] is True
    assert [
        live_mcp["initial_map_version"],
        live_mcp["scan_map_version"],
        live_mcp["turn_map_version"],
    ] == [0, 1, 2]
    assert [
        live_mcp["initial_source_voxels"],
        live_mcp["scan_source_voxels"],
        live_mcp["turn_source_voxels"],
    ] == [74699, 74897, 75594]
    assert live_mcp["explicit_scan_valid_depth_pixels"] == 50176
    assert live_mcp["turn_auto_scan_valid_depth_pixels"] == 50176
    assert live_mcp["full_image_encoder_calls_total"] == 2
    assert live_mcp["map_prefix_controller_robot_hashes_changed_after_scan"] is True
    assert live_mcp["map_prefix_controller_robot_hashes_changed_after_turn"] is True
    assert live_mcp["robot_state_encoder_identity_invariant"] is True
    assert live_mcp["loaded_file_count"] == 4178
    assert live_mcp["forbidden_access_count"] == 0
    assert live_mcp["environmental_text_in_tool_results"] is False
    assert live_mcp["semantic_result_leaks"] == 0
    conversation = evidence["live_embodied_conversation"]
    assert conversation["passed"] is True
    assert conversation["record_count"] == 4
    assert conversation["map_versions"] == [0, 1, 2, 2]
    assert conversation["prequestion_scene_key_value_cache"] is True
    assert conversation["full_image_encoder_calls_total"] == 2
    assert conversation["valid_depth_pixels_per_observation"] == 50176
    assert conversation["turn_degrees"] == 15.0
    assert conversation["answer"] == "yes"
    assert conversation["answer_map_version"] == 2
    assert conversation["answer_uses_refreshed_active_prefix"] is True
    assert conversation["binding_unchanged_from_turn_to_answer"] is True
    assert conversation["environmental_text_inputs"] == []
    assert conversation["loaded_file_count"] == 123
    assert conversation["forbidden_access_count"] == 0


def test_current_markdown_reports_promoted_v75_and_failed_official_gate() -> None:
    markdown = BUILDER["render_markdown"](_summary())
    collapsed = " ".join(markdown.split())

    assert (
        "two terminal negatives and one promoted V75 candidate with completed one-shot "
        "official validation" in collapsed
    )
    assert "reduced train NLL from 7.562226 to 2.506602" in collapsed
    assert "answered 295/384 (76.82%) correctly versus 148/384 (38.54%)" in collapsed
    assert "31/52 original-target answers" in collapsed
    assert "14/52 original-target but 31/52 paired-target answers" in collapsed
    assert "24/52 outputs changed between scene arms" in collapsed
    assert "**internal development** result" in collapsed
    assert "all 4,198 audited reads produced zero forbidden accesses" in collapsed
    assert "167/216 (77.31%) canonical" in collapsed
    assert "spatial relations reached 58.33% against the 60% minimum" in collapsed
    assert "mean coordinate error 2.136 m and 0/132 targets within 1 m" in collapsed
    assert "overall official gate is **failed**, not passed" in collapsed
    assert "four control tokens are question-conditioned" in collapsed
    assert (
        "V76 then trained a pair-contrast objective over all 40 historical changed units"
        in collapsed
    )
    assert "the held Gemma smoke was 6/16, tied with both V54 and wrong-scene" in collapsed
    assert "V77 started from promoted NLL V75" in collapsed
    assert "72 of 576 available rows" in collapsed
    assert "correct-answer NLL fell from 2.518216 to 1.604626" in collapsed
    assert "bounded 16-row pair-disjoint Gemma smoke reached 9/16" in collapsed
    assert "changed across only 2/8 paired units" in collapsed
    assert "384-row internal screen reached 299/384" in collapsed
    assert "versus V75's 295/384" in collapsed
    assert "no matched full wrong-scene arm was run" in collapsed
    assert "not a promotion" in collapsed
    assert "At V75's promotion, `make demo` launched this path" in collapsed
    assert "current operator default is strict V89" in collapsed
    assert "answered the first two bounded examples `yes` and `red` correctly" in collapsed
    assert "third output, `right`, is physically incorrect" in collapsed
    assert "corrected bowl-to-chair relation is `left`" in collapsed
    assert "historically produced malformed short strings" in collapsed
    assert "vocabulary-free output guard" in collapsed
    assert "fails closed as `unknown` for both" in collapsed
    assert "does not make broad list QA supported" in collapsed
    assert "15-degree turn captured `o_000002`" in collapsed
    assert "scene/map version 2" in collapsed
    assert "actual official-SDK MCP stdio boundary" in collapsed
    assert "version 0 to 1 to 2" in collapsed
    assert "audited 4,178 reads with zero forbidden accesses" in collapsed
    assert "returned no environmental text or semantic labels" in collapsed
    assert "not evidence of conversational navigation success" in collapsed
    assert "separately sealed direct embodied conversation" in collapsed
    assert "generated `yes` at map version 2" in collapsed
    assert "answer is bound to newly observed continuous scene state" in collapsed
    assert "motion sent through live MCP transport is still unmeasured" not in collapsed
    assert "safe packaging of the existing below-acceptance V54 mechanism demo" in collapsed
    assert "V75 is rejected and quarantined" not in collapsed
    assert "V75 runtime leakage/oracle-removal clearance" not in collapsed
    assert "No V73--V75 runtime checkpoint was promoted" not in collapsed


def test_v75_report_fails_closed_on_nll_evidence_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_v73_v75_static_readout_evidence"]
    original = ROOT / BUILDER["V75_NLL_HELD_SMOKE"]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    hashes = dict(inspector.__globals__["V73_V75_EVIDENCE_SHA256"])
    original_key = next(path for path in hashes if path.name == original.name)
    hashes[tampered] = hashes.pop(original_key)
    monkeypatch.setitem(inspector.__globals__, "V75_NLL_HELD_SMOKE", tampered)
    monkeypatch.setitem(inspector.__globals__, "V73_V75_EVIDENCE_SHA256", hashes)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["checkpoint_published"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "digest differs" in result["measurement_evidence_error"]


@pytest.mark.parametrize(
    ("path_name", "digest_mapping"),
    [
        (
            "V75_RUNTIME_STRICT_LEAKAGE",
            "V75_RUNTIME_EVIDENCE_SHA256",
        ),
        (
            "V75_OFFICIAL_VALIDATION_SCORE",
            "V75_OFFICIAL_VALIDATION_EVIDENCE_SHA256",
        ),
        (
            "V75_BROAD_ANSWER_GUARD_CHAT",
            "V75_BROAD_ANSWER_GUARD_EVIDENCE_SHA256",
        ),
        (
            "V75_BROAD_ANSWER_GUARD_ACCESS",
            "V75_BROAD_ANSWER_GUARD_EVIDENCE_SHA256",
        ),
        (
            "V76_PAIR_CONTRAST_HELD",
            "V76_EVIDENCE_SHA256",
        ),
        (
            "V77_HISTORICAL_REPAIR_TRAIN",
            "V77_EVIDENCE_SHA256",
        ),
        (
            "V77_HISTORICAL_REPAIR_HELD",
            "V77_EVIDENCE_SHA256",
        ),
        (
            "V77_HISTORICAL_REPAIR_FULL",
            "V77_EVIDENCE_SHA256",
        ),
        (
            "V77_HISTORICAL_REPAIR_CANDIDATE",
            "V77_EVIDENCE_SHA256",
        ),
    ],
)
def test_v75_report_fails_closed_on_promotion_or_validation_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_name: str,
    digest_mapping: str,
) -> None:
    inspector = BUILDER["_inspect_v73_v75_static_readout_evidence"]
    original = ROOT / BUILDER[path_name]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    hashes = dict(inspector.__globals__[digest_mapping])
    original_key = next(path for path in hashes if path.name == original.name)
    hashes[tampered] = hashes.pop(original_key)
    monkeypatch.setitem(inspector.__globals__, path_name, tampered)
    monkeypatch.setitem(inspector.__globals__, digest_mapping, hashes)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["checkpoint_published"] is False
    assert result["runtime_promotion_authorized"] is False
    assert "digest differs" in result["measurement_evidence_error"]


def test_live_motion_report_fails_closed_on_evidence_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_embodied_auto_scan_mcp"]
    original = ROOT / BUILDER["EMBODIED_AUTO_MOTION_LIVE"]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    monkeypatch.setitem(inspector.__globals__, "EMBODIED_AUTO_MOTION_LIVE", tampered)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["live_semantic_motion_refresh_via_mcp_measured"] is False
    assert "digest differs" in result["measurement_evidence_error"]


@pytest.mark.parametrize(
    "path_name",
    ["V75_SEMANTIC_MCP_LIVE", "V75_SEMANTIC_MCP_LIVE_ACCESS"],
)
def test_live_semantic_mcp_report_fails_closed_on_evidence_tamper(
    path_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_embodied_auto_scan_mcp"]
    original = ROOT / BUILDER[path_name]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    hashes = dict(inspector.__globals__["V75_SEMANTIC_MCP_LIVE_EVIDENCE_SHA256"])
    original_key = next(path for path in hashes if path.name == original.name)
    hashes[tampered] = hashes.pop(original_key)
    monkeypatch.setitem(inspector.__globals__, path_name, tampered)
    monkeypatch.setitem(inspector.__globals__, "V75_SEMANTIC_MCP_LIVE_EVIDENCE_SHA256", hashes)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["live_semantic_motion_refresh_via_mcp_measured"] is False
    assert "digest differs" in result["measurement_evidence_error"]


@pytest.mark.parametrize(
    "path_name",
    [
        "V75_EMBODIED_CONVERSATION_TRANSCRIPT",
        "V75_EMBODIED_CONVERSATION_SEAL",
        "V75_EMBODIED_CONVERSATION_ACCESS",
    ],
)
def test_live_embodied_conversation_fails_closed_on_evidence_tamper(
    path_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_embodied_auto_scan_mcp"]
    original = ROOT / BUILDER[path_name]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b" ")
    hashes = dict(inspector.__globals__["V75_EMBODIED_CONVERSATION_EVIDENCE_SHA256"])
    original_key = next(path for path in hashes if path.name == original.name)
    hashes[tampered] = hashes.pop(original_key)
    monkeypatch.setitem(inspector.__globals__, path_name, tampered)
    monkeypatch.setitem(
        inspector.__globals__,
        "V75_EMBODIED_CONVERSATION_EVIDENCE_SHA256",
        hashes,
    )

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert result["live_semantic_motion_refresh_via_mcp_measured"] is False
    assert "digest differs" in result["measurement_evidence_error"]
