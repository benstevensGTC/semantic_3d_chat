from __future__ import annotations

import json
from pathlib import Path

from semantic_3d_chat.evaluation.reporting import build_report, collect_report_inputs
from semantic_3d_chat.mapping.voxel_map import PERSISTED_MAP_CONTENT_HASH_DOMAIN


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config() -> dict:
    return {
        "paths": {"data_root": "data", "reports_root": "reports"},
        "scene": {"scene_id": "scene_000001"},
        "render": {"resolution": [64, 64]},
        "vision": {
            "model_id": "test/vision",
            "revision": "vision-revision",
            "input_size": 224,
            "aligned_method": "maskclip_value",
        },
        "mapping": {"voxel_size_m": 0.05},
        "scene_encoder": {
            "block_size_m": 0.25,
            "global_latents": 256,
            "model_dim": 384,
        },
        "language": {"model_id": "test/language", "revision": "lm-revision"},
    }


def _populate_core_artifacts(root: Path) -> None:
    metrics = root / "reports" / "metrics"
    _write_json(
        metrics / "machine_report.json",
        {
            "architecture": "arm64",
            "memory_bytes": 24 * 2**30,
            "disk_free_bytes": 100 * 2**30,
            "python_version": "3.12",
            "torch_mps_available": True,
        },
    )
    _write_json(
        metrics / "map_scene_000001.json",
        {
            "voxel_size_m": 0.05,
            "occupied_voxels": 1000,
            "feature_dim": 2048,
            "semantic_dtype_on_disk": "float16",
            "total_observations": 5000,
            "frame_count": 24,
            "content_hash": "map-a",
        },
    )
    _write_json(
        metrics / "semantic_sanity_scene_000001.json",
        {
            "voxel_count": 1000,
            "map_content_hash": "map-a",
            "top_k": 100,
            "feature_layout": {"aligned_method": "maskclip_value"},
            "aggregate": {
                "top1_localization_accuracy": 0.75,
                "mean_random_top1_probability": 0.1,
                "top1_accuracy_minus_random": 0.65,
                "top_k_localization_accuracy": 1.0,
                "mean_random_hit_at_k_probability": 0.2,
                "top_k_accuracy_minus_random": 0.8,
                "mean_precision_at_k": 0.5,
                "mean_random_precision_at_k": 0.1,
                "precision_at_k_minus_random": 0.4,
            },
            "same_voxel_consistency": {
                "same_voxel_similarity": {"mean": 0.8},
                "different_voxel_similarity": {"mean": 0.6},
                "same_minus_different_mean": 0.2,
                "same_voxel_pair_count": 50,
            },
            "queries": [
                {
                    "category": "opaque evaluation category",
                    "precision_at_k": 0.5,
                    "random_precision_at_k": 0.1,
                    "hit_at_k": True,
                }
            ],
        },
    )
    _write_json(
        metrics / "leakage.json",
        {
            "passed": True,
            "oracle_unavailable_during_inference": True,
            "oracle_restored": True,
            "forbidden_accesses": [],
            "prefix_computed_before_first_question": True,
            "prefix_invariant": True,
            "prefix_hash": "abc123",
            "question_count": 3,
            "loaded_files": ["runtime-map.npz"],
            "startup": {
                "processed_voxels": 800,
                "occupied_blocks": 300,
                "prefix_shape": [1, 258, 896],
            },
        },
    )
    for epoch, loss in ((1, 1.5), (2, 0.7)):
        _write_json(
            root / "data" / "checkpoints" / f"epoch_{epoch:03d}" / "metadata.json",
            {
                "epoch": epoch,
                "train_loss": loss,
                "scene_ids": ["scene_000001"],
                "scene_latents": 256,
                "scene_model_dim": 384,
                "language_hidden_dim": 896,
            },
        )
    _write_json(
        root / "data" / "checkpoints" / "best" / "metadata.json",
        {
            "epoch": 2,
            "train_loss": 0.7,
            "scene_ids": ["scene_000001"],
            "scene_latents": 256,
            "scene_model_dim": 384,
            "language_hidden_dim": 896,
        },
    )
    qa_root = root / "data" / "qa"
    qa_root.mkdir(parents=True)
    (qa_root / "train.jsonl").write_text('{"question_id":"q1"}\n', encoding="utf-8")
    (qa_root / "validation.jsonl").write_text("", encoding="utf-8")
    (qa_root / "test.jsonl").write_text("", encoding="utf-8")
    _write_json(qa_root / "splits.json", {"splits": {"train": ["scene_000001"]}})


def test_report_marks_absent_experiments_without_inventing_results(tmp_path: Path) -> None:
    _populate_core_artifacts(tmp_path)

    result = build_report(tmp_path, _config())
    report = (tmp_path / "reports" / "final_report.md").read_text(encoding="utf-8")

    assert result["scene_id"] == "scene_000001"
    assert "No held-out scene result can be reported" in report
    assert "**Not measured.** No held-out prediction metrics JSON is present" in report
    assert "75.0%" in report
    assert (tmp_path / "reports" / "figures" / "architecture.png").is_file()
    assert (tmp_path / "reports" / "figures" / "training_loss.png").is_file()
    assert (tmp_path / "reports" / "figures" / "semantic_localization_by_category.png").is_file()
    assert not (tmp_path / "reports" / "figures" / "accuracy_by_question_type.png").exists()


def test_report_does_not_compare_legacy_semantic_hash_from_unknown_domain(
    tmp_path: Path,
) -> None:
    _populate_core_artifacts(tmp_path)
    semantic_path = tmp_path / "reports" / "metrics" / "semantic_sanity_scene_000001.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["map_content_hash"] = "reloaded-map-b"
    _write_json(semantic_path, semantic)

    build_report(tmp_path, _config())
    report = (tmp_path / "reports" / "final_report.md").read_text(encoding="utf-8")

    assert "Artifact-version warning" not in report


def test_report_warns_only_for_comparable_persisted_hashes(tmp_path: Path) -> None:
    _populate_core_artifacts(tmp_path)
    metrics = tmp_path / "reports" / "metrics"
    map_path = metrics / "map_scene_000001.json"
    semantic_path = metrics / "semantic_sanity_scene_000001.json"
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    mapping["content_hash_domain"] = PERSISTED_MAP_CONTENT_HASH_DOMAIN
    _write_json(map_path, mapping)
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic.update(
        {
            "map_content_hash": "reloaded-map-b",
            "map_content_hash_domain": (
                "semantic_3d_chat.voxel_map.materialized_numeric_arrays.v1"
            ),
            "map_persisted_content_hash": "persisted-map-b",
            "map_persisted_content_hash_domain": PERSISTED_MAP_CONTENT_HASH_DOMAIN,
        }
    )
    _write_json(semantic_path, semantic)

    build_report(tmp_path, _config())
    report = (tmp_path / "reports" / "final_report.md").read_text(encoding="utf-8")

    assert "Artifact-version warning" in report
    assert "persisted numeric-array hash" in report
    assert "MaskCLIP map rebuild" not in report

    semantic["map_persisted_content_hash"] = "map-a"
    _write_json(semantic_path, semantic)
    build_report(tmp_path, _config())
    synchronized_report = (tmp_path / "reports" / "final_report.md").read_text(encoding="utf-8")
    assert "Artifact-version warning" not in synchronized_report


def test_report_collection_honors_isolated_candidate_roots(tmp_path: Path) -> None:
    config = _config()
    config["paths"].update(
        {
            "reports_root": "reports/gemma4",
            "checkpoints_root": "data_gemma4/checkpoints",
            "qa_root": "data/qa",
            "rendered_root": "data/rendered",
        }
    )
    config["training"] = {"output_namespace": "gemma4_e2b"}
    metrics = tmp_path / "reports" / "gemma4" / "metrics"
    _write_json(metrics / "training_gemma4_e2b.json", {"history": []})
    checkpoint = (
        tmp_path
        / "data_gemma4"
        / "checkpoints"
        / "gemma4_e2b"
        / "best"
        / "metadata.json"
    )
    _write_json(checkpoint, {"epoch": 3, "language_hidden_dim": 1536})
    qa_root = tmp_path / "data" / "qa"
    qa_root.mkdir(parents=True)
    (qa_root / "train.jsonl").write_text('{"question_id":"q1"}\n', encoding="utf-8")
    (qa_root / "validation.jsonl").write_text("", encoding="utf-8")
    (qa_root / "test.jsonl").write_text("", encoding="utf-8")

    inputs = collect_report_inputs(tmp_path, config)

    assert inputs.reports_root == tmp_path / "reports" / "gemma4"
    assert inputs.training_namespace == "gemma4_e2b"
    assert inputs.best_checkpoint is not None
    assert inputs.best_checkpoint["epoch"] == 3
    assert inputs.qa_counts == {"train": 1, "validation": 0, "test": 0}


def test_report_adds_evaluation_and_ablation_charts_when_metrics_appear(tmp_path: Path) -> None:
    _populate_core_artifacts(tmp_path)
    metrics = tmp_path / "reports" / "metrics"
    _write_json(
        metrics / "metrics.json",
        {
            "normalized_exact_accuracy": 0.8,
            "per_type": {
                "presence": {"normalized_exact_accuracy": 0.9},
                "spatial_relation": {"normalized_exact_accuracy": 0.7},
            },
            "counterfactual": {
                "eligible_pairs": 5,
                "pair_accuracy": 0.6,
                "changed_when_expected_rate": 0.8,
                "invariant_when_expected_rate": 1.0,
            },
            "grounding": {"target_count": 4, "mean_coordinate_error_m": 0.3},
        },
    )
    _write_json(
        metrics / "ablations.json",
        {
            "results": {
                "primary": {"normalized_exact_accuracy": 0.8},
                "shuffled_geometry": {"normalized_exact_accuracy": 0.3},
            }
        },
    )

    result = build_report(tmp_path, _config())
    report = (tmp_path / "reports" / "final_report.md").read_text(encoding="utf-8")

    assert "80.0%" in report
    assert result["figures"]["accuracy_by_type"] == "figures/accuracy_by_question_type.png"
    assert result["figures"]["counterfactual"] == "figures/counterfactual_consistency.png"
    assert result["figures"]["ablations"] == "figures/ablation_accuracy.png"
    for relative_path in result["figures"].values():
        assert (tmp_path / "reports" / relative_path).is_file()


def test_report_does_not_call_robot_benchmark_unmeasured_when_artifact_exists(
    tmp_path: Path,
) -> None:
    _populate_core_artifacts(tmp_path)
    _write_json(
        tmp_path / "reports" / "metrics" / "robot_navigation.json",
        {
            "benchmark_scope": "bounded_numeric_actions_collision_scan_and_reset",
            "passed": 11,
            "total": 11,
            "pass_rate": 1.0,
            "mcp_sdk_version": "2.0.0",
            "mcp_tool_count": 9,
            "semantic_target_navigation_evaluated": False,
        },
    )

    result = build_report(tmp_path, _config())
    report = (tmp_path / "reports" / "final_report.md").read_text(encoding="utf-8")

    assert "robot" not in result["missing_metric_groups"]
    assert '"passed": 11' in report
    assert "Robot navigation and MCP benchmarks are not yet measured" not in report
    assert "language-conditioned semantic target navigation remains unmeasured" in report
    assert "Train and evaluate language-conditioned target-facing" in report


def test_report_separates_collapsed_v1_from_structural_v2(tmp_path: Path) -> None:
    _populate_core_artifacts(tmp_path)
    config = _config()
    config["scene_encoder"]["architecture_version"] = "spatial_coverage_resampler_v2"
    qa_root = tmp_path / "data" / "qa"
    (qa_root / "validation.jsonl").write_text('{"question_id":"v1"}\n', encoding="utf-8")
    (qa_root / "test.jsonl").write_text('{"question_id":"t1"}\n', encoding="utf-8")
    _write_json(
        qa_root / "splits.json",
        {
            "splits": {
                "train": ["scene_000001"],
                "validation": ["scene_000009"],
                "test": ["scene_000005"],
            }
        },
    )
    metrics = tmp_path / "reports" / "metrics"
    _write_json(
        metrics / "metrics.json",
        {
            "reference_count": 100,
            "predictions_path": "reports/predictions/multiscene_test.jsonl",
            "normalized_exact_accuracy": 0.75,
            "counterfactual": {
                "eligible_pairs": 10,
                "expected_change_pairs": 2,
                "changed_when_expected_rate": 0.0,
                "pair_accuracy": 0.7,
            },
            "grounding": {},
        },
    )
    _write_json(
        metrics / "ablations.json",
        {
            "results": {
                "primary": {"normalized_exact_accuracy": 0.75},
                "wrong_scene_prefix": {"normalized_exact_accuracy": 0.76},
            },
            "interpretation": {
                "empty_prefix_collapse": True,
                "warning": (
                    "A nonzero prefix is necessary, but the first multiscene checkpoint "
                    "is insensitive to its scene-specific content."
                ),
            },
        },
    )
    _write_json(
        metrics / "scene_signal_audit.json",
        {
            "corroborating_control_results": {"primary_changed_when_expected_rate": 0.0},
            "summary_findings": {
                "diagnosis": "The legacy global latents collapsed.",
                "raw_to_projected_attenuation_factor_range": [100.0, 200.0],
            },
        },
    )
    _write_json(
        metrics / "resampler_fix_diagnostic.json",
        {
            "pairs": [
                {
                    "change_type": "mirror_lr",
                    "improvement_factor": {
                        "native_scene_change": 30.0,
                        "projected_scene_change": 300.0,
                    },
                    "after": {
                        "native_mean_off_diagonal_cosine": 0.87,
                        "projected_mean_off_diagonal_cosine": 0.93,
                    },
                }
            ]
        },
    )
    _write_json(
        tmp_path / "data" / "checkpoints" / "multiscene_anticollapse" / "best" / "metadata.json",
        {
            "epoch": 1,
            "train_loss": 1.0,
            "scene_ids": ["scene_000001"],
            "scene_latents": 256,
            "scene_model_dim": 384,
            "language_hidden_dim": 896,
            "scene_encoder_architecture_version": "spatial_coverage_resampler_v2",
        },
    )

    result = build_report(tmp_path, config)
    report = (tmp_path / "reports" / "final_report.md").read_text(encoding="utf-8")

    assert result["selected_training_namespace"] == "multiscene_anticollapse"
    assert result["qa_lineage"] == "v1"
    assert result["v1_collapse_evidence_present"] is True
    assert "Held-out QA is measured on 100 test records" in report
    assert "v1 controls invalidate that raw score" in report
    assert "structural evidence only, not a QA result" in report
    assert "Validation and test splits contain zero records" not in report
    assert "Only one scene is represented in the current QA split" not in report
