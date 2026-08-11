from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_TOKEN_EMBEDDING_KEY,
)
from semantic_3d_chat.scene_encoder.dense_alignment import DenseAlignmentResidual
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.training.dense_alignment_calibration import (
    require_dense_alignment_calibration_authorized,
    run_dense_alignment_calibration_warmup,
    summarize_dense_alignment_regions,
)
from semantic_3d_chat.training.dense_alignment_supervision import (
    DenseAlignmentRegionTargets,
)


def _module() -> DenseAlignmentResidual:
    return DenseAlignmentResidual(
        semantic_dim=6,
        dense_dim=3,
        aligned_dim=3,
        rank=2,
        alpha=4.0,
        initialization_seed=25025,
    )


def _config(module: DenseAlignmentResidual) -> dict[str, Any]:
    return {
        "paths": {
            "data_root": "data",
            "maps_root": "data/maps",
            "checkpoints_root": "data/checkpoints",
            "reports_root": "reports",
        },
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "scene_encoder": {
            "input_voxel_size_m": None,
            "dense_alignment": {
                "enabled": True,
                "dense_dim": 3,
                "aligned_dim": 3,
                "rank": 2,
                "alpha": 4.0,
                "initialization_seed": 25025,
                "expected_initial_state_sha256": module.state_sha256(),
            },
        },
        "language": {"model_id": "local/tiny", "revision": "revision"},
        "training": {
            "output_namespace": "tiny_v25",
            "dense_alignment_warmup": {
                "enabled": True,
                "training_only": True,
                "max_optimizer_steps": 20,
                "evaluation_interval_steps": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0001,
                "delta_rms_cap": 1.0,
                "delta_abs_max_cap": 3.5,
                "delta_rms_regularization_weight": 0.01,
                "early_stop_top1_accuracy": 1.0,
                "early_stop_minimum_margin": 0.10,
                "held_out_scene_gradient_access": False,
                "reset_pair_optimizer_after_warmup": True,
            },
            "dense_alignment_supervision": {
                "enabled": True,
                "training_only": True,
                "oracle_access_process": "training_and_evaluation_only",
                "runtime_oracle_access": False,
                "runtime_serializes_category_strings": False,
                "runtime_serializes_text_embeddings": False,
                "question_dependent_scene_processing": False,
                "all_voxels_transformed": True,
                "calibration_scene_ids": [
                    "scene_000001",
                    "scene_000002",
                    "scene_000003",
                    "scene_000004",
                    "scene_000005",
                    "scene_000006",
                    "scene_000009",
                    "scene_000010",
                ],
                "held_out_scene_ids": ["scene_000007", "scene_000008"],
                "dense_dim": 3,
                "aligned_start": 3,
                "aligned_dim": 3,
                "temperature": 0.07,
                "bbox_padding_m": 0.0,
                "minimum_voxels_per_region": 8,
                "loss_weight": 1.0,
            },
        },
        "v25_screen": {
            "held_out_localization_requires": {
                "scene_count": 2,
                "target_region_count": 4,
                "minimum_precision_at_k": 0.10,
                "maximum_mirror_centroid_error_m": 0.15,
            }
        },
    }


def _points(center_x: float, *, mirrored_offsets: bool = False) -> torch.Tensor:
    offsets = torch.linspace(-0.09, 0.09, 120)
    if mirrored_offsets:
        offsets = -offsets
    return torch.stack(
        (
            torch.full_like(offsets, center_x) + offsets,
            torch.linspace(-0.04, 0.04, 120),
            torch.linspace(0.15, 0.25, 120),
        ),
        dim=1,
    )


def _map(scene_id: str, *, break_mirror: bool = False) -> MapTensorData:
    if scene_id == "scene_000008":
        bowl_center = 1.0 + (0.4 if break_mirror else 0.0)
        cabinet_center = -1.0
        bowl_xyz = _points(bowl_center, mirrored_offsets=True)
        cabinet_xyz = _points(cabinet_center, mirrored_offsets=True)
    else:
        bowl_xyz = _points(-1.0)
        cabinet_xyz = _points(1.0)
    xyz = torch.cat((bowl_xyz, cabinet_xyz))
    bowl_semantic = torch.tensor([1.0, 0.0, -1.0, 0.01, 0.0, 0.0]).repeat(120, 1)
    cabinet_semantic = torch.tensor([-1.0, 0.0, 1.0, 0.01, 0.0, 0.0]).repeat(120, 1)
    semantic = torch.cat((bowl_semantic, cabinet_semantic)).float()
    count = semantic.shape[0]
    return MapTensorData(
        semantic=semantic,
        xyz=xyz.float(),
        rgb=torch.zeros(count, 3),
        normal=torch.zeros(count, 3),
        confidence=torch.ones(count),
        observation_count=torch.ones(count),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=count,
        input_voxel_size_m=None,
    )


def _oracle(scene_id: str, *, break_mirror: bool = False) -> dict[str, Any]:
    data = _map(scene_id, break_mirror=break_mirror)
    instances = []
    for index, (category, selected) in enumerate(
        (("bowl", data.xyz[:120]), ("cabinet", data.xyz[120:])), start=1
    ):
        instances.append(
            {
                "instance_id": f"i_{index:06d}",
                "kind": "object",
                "category": category,
                "visible_from_center_scan": True,
                "bbox": {
                    "min_xyz_m": selected.min(dim=0).values.tolist(),
                    "max_xyz_m": selected.max(dim=0).values.tolist(),
                },
            }
        )
    return {"scene_id": scene_id, "instances": instances}


def _embedding_loader(_snapshot: Path, categories, _tokenizer):
    assert tuple(categories) == ("bowl", "cabinet")
    return np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32), {
        "loaded_parameter_keys": [GEMMA4_TOKEN_EMBEDDING_KEY],
        "selective_row_read": True,
        "unique_token_rows_read": 2,
    }


def _assert_numeric_hash_only(value: Any) -> None:
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        assert re.fullmatch(r"[0-9a-f]{64}", value)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _assert_numeric_hash_only(nested)
        return
    if isinstance(value, list):
        for nested in value:
            _assert_numeric_hash_only(nested)
        return
    raise AssertionError(f"Forbidden report value: {type(value).__name__}")


def test_region_summary_is_exact_mean_of_per_voxel_layernorm() -> None:
    module = _module()
    semantic = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.2, 0.3, 0.4],
            [0.5, 0.1, -0.4, 0.4, 0.5, 0.6],
            [-1.0, 0.0, 1.0, -0.2, -0.3, -0.4],
            [-0.5, -0.1, 0.4, -0.4, -0.5, -0.6],
        ]
    )
    membership = torch.tensor([[True, True, False, False], [False, False, True, True]])
    targets = DenseAlignmentRegionTargets(
        region_membership=membership,
        category_indices=torch.tensor([0, 1]),
        voxel_counts=torch.tensor([2, 2]),
        input_voxel_count=4,
    )

    observed = summarize_dense_alignment_regions(semantic, targets, module)
    normalized = F.layer_norm(semantic[:, :3], (3,), eps=module.layer_norm_eps)

    assert torch.equal(observed.category_indices, torch.tensor([0, 1]))
    assert torch.allclose(observed.mean_layernorm_dense[0], normalized[:2].mean(dim=0))
    assert torch.allclose(observed.mean_layernorm_dense[1], normalized[2:].mean(dim=0))
    assert torch.equal(observed.mean_aligned_tail[0], semantic[:2, 3:].mean(dim=0))
    assert torch.equal(observed.mean_aligned_tail[1], semantic[2:, 3:].mean(dim=0))

    # A scene may legitimately omit one globally known category (the real
    # scene_000010 omits book).  Per-scene summaries therefore allow a single,
    # non-zero global category index; completeness is enforced after scenes
    # are concatenated.
    one_region = DenseAlignmentRegionTargets(
        region_membership=membership[1:].clone(),
        category_indices=torch.tensor([1]),
        voxel_counts=torch.tensor([2]),
        input_voxel_count=4,
    )
    missing_category_summary = summarize_dense_alignment_regions(semantic, one_region, module)
    assert missing_category_summary.region_count == 1
    assert missing_category_summary.category_indices.tolist() == [1]


def test_full_warmup_is_deterministic_authorized_and_serializes_no_semantics(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    bridge = tmp_path / "bridge.safetensors"
    report_path = tmp_path / "report.json"

    def execute(module: DenseAlignmentResidual, *, write: bool) -> dict[str, Any]:
        return run_dense_alignment_calibration_warmup(
            _config(module),
            module,
            model_snapshot=snapshot,
            map_loader=_map,
            oracle_loader=_oracle,
            embedding_loader=_embedding_loader,
            bridge_output=bridge if write else None,
            report_output=report_path if write else None,
        )

    first_module = _module()
    first = execute(first_module, write=True)
    second_module = _module()
    second = execute(second_module, write=False)

    assert first == second | {
        "bridge_written": True,
        "bridge_sha256": first["bridge_sha256"],
    }
    assert first["qa_update_authorized"] is True
    assert first["training"]["calibration_passed"] is True
    assert first["training"]["optimizer_steps"] <= 20
    assert first["held_out_localization"]["passed"] is True
    assert first["held_out_localization"]["minimum_precision_at_k"] >= 0.10
    assert first["held_out_localization"]["minimum_region_margin"] > 0.0
    assert first["held_out_localization"]["minimum_correct_vs_distractor_margin"] > 0.0
    assert first["held_out_localization"]["maximum_mirror_centroid_error_m"] <= 0.15
    assert first_module.state_sha256() == second_module.state_sha256()
    require_dense_alignment_calibration_authorized(first)
    _assert_numeric_hash_only(first)
    serialized = report_path.read_text(encoding="utf-8")
    assert json.loads(serialized) == first
    assert "bowl" not in serialized and "cabinet" not in serialized

    tensors = load_file(bridge)
    assert set(tensors) == {
        "dense_aligner.alignment_a",
        "dense_aligner.alignment_b",
        "dense_aligner.architecture_marker",
        "dense_aligner.scaling",
    }
    assert b"bowl" not in bridge.read_bytes()
    assert b"cabinet" not in bridge.read_bytes()


def test_failed_heldout_mirror_gate_writes_no_bridge(tmp_path: Path) -> None:
    module = _module()
    config = _config(module)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    bridge = tmp_path / "forbidden.safetensors"

    audit = run_dense_alignment_calibration_warmup(
        config,
        module,
        model_snapshot=snapshot,
        map_loader=lambda scene_id: _map(scene_id, break_mirror=True),
        oracle_loader=lambda scene_id: _oracle(scene_id, break_mirror=True),
        embedding_loader=_embedding_loader,
        bridge_output=bridge,
    )

    assert audit["training"]["calibration_passed"] is True
    assert audit["held_out_localization"]["passed"] is False
    assert audit["qa_update_authorized"] is False
    assert audit["bridge_written"] is False
    assert not bridge.exists()
    with pytest.raises(RuntimeError, match="did not authorize"):
        require_dense_alignment_calibration_authorized(audit)


def test_runner_fails_closed_on_optimizer_or_split_drift(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    module = _module()
    wrong_weight_decay = _config(module)
    wrong_weight_decay["training"]["dense_alignment_warmup"]["weight_decay"] = 0.0
    with pytest.raises(ValueError, match="weight_decay"):
        run_dense_alignment_calibration_warmup(
            wrong_weight_decay,
            module,
            model_snapshot=snapshot,
            map_loader=_map,
            oracle_loader=_oracle,
            embedding_loader=_embedding_loader,
        )

    overlap = deepcopy(_config(_module()))
    overlap["training"]["dense_alignment_supervision"]["calibration_scene_ids"].append(
        "scene_000007"
    )
    with pytest.raises(ValueError, match="disjoint"):
        run_dense_alignment_calibration_warmup(
            overlap,
            _module(),
            model_snapshot=snapshot,
            map_loader=_map,
            oracle_loader=_oracle,
            embedding_loader=_embedding_loader,
        )


def test_archived_real_v25_calibration_is_bit_pinned_and_fail_closed() -> None:
    report_path = (
        Path(__file__).resolve().parents[1]
        / "reports/gemma4/metrics/v25_dense_alignment_calibration.json"
    )
    raw = report_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "642f16599892a8d7b9a2f21a7d74c1dba6d5f2dbb64c9d37333ffac43dae8637"
    )
    report = json.loads(raw)

    _assert_numeric_hash_only(report)
    assert report["final_state_sha256"] == (
        "9a3a71fe4d7894cae694c00fa5eec5adcaff75fc27751067df7d1bf795c3566e"
    )
    assert report["calibration_summary_sha256"] == (
        "4169b611fca30fbc0631c174a7a61d6fb7d88a7235c7fafd8de07b9572b28cc4"
    )
    assert report["training"]["optimizer_steps"] == 20
    assert report["training"]["calibration_passed"] is False
    assert report["held_out_localization"]["passed"] is True
    assert report["qa_update_authorized"] is False
    assert report["bridge_written"] is False
    assert report["bridge_sha256"] is None
    assert report["skipped_underfilled_region_count"] == 3
    assert b"bowl" not in raw and b"cabinet" not in raw
