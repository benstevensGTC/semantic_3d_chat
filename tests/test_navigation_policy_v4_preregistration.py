from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation import navigation_policy_v41_preregistration as v41
from semantic_3d_chat.evaluation.navigation_policy_v4_preregistration import (
    SCHEMA,
    V3_DATASET_SHA256,
    V4_DATASET_SHA256,
    PreregistrationError,
    authenticate_navigation_policy_v4_preregistration,
    build_navigation_policy_v4_preregistration,
    file_sha256,
    write_navigation_policy_v4_preregistration,
)
from semantic_3d_chat.evaluation.navigation_policy_v41_preregistration import (
    INCIDENT_SHA256,
    ORIGINAL_PREREGISTRATION_SHA256,
    authenticate_navigation_policy_v41_preregistration,
    build_navigation_policy_v41_preregistration,
    write_navigation_policy_v41_preregistration,
)


def _inputs(tmp_path: Path):
    config = copy.deepcopy(load_config("configs/experiments/navigation_policy_v4.yaml"))
    settings = config["navigation_policy_v4"]
    settings["checkpoint_output"] = str(tmp_path / "checkpoint")
    scene_ids = [*settings["train_scene_ids"], *settings["validation_scene_ids"]]
    maps = {
        scene_id: file_sha256(f"data_gemma4/maps/{scene_id}/voxel_map.npz")
        for scene_id in scene_ids
    }
    return config, maps


def test_v4_preregistration_is_single_arm_source_locked_and_create_once(
    tmp_path: Path,
) -> None:
    config, maps = _inputs(tmp_path)
    payload = build_navigation_policy_v4_preregistration(
        config,
        source_v3_dataset_sha256=V3_DATASET_SHA256,
        v4_dataset_sha256=V4_DATASET_SHA256,
        map_sha256=maps,
    )
    assert payload["schema"] == SCHEMA
    assert payload["single_arm"]["one_arm_only"] is True
    assert payload["single_arm"]["hyperparameter_search"] is False
    assert payload["data"]["train_scene_count"] == 14
    assert payload["data"]["validation_scene_count"] == 8
    assert payload["data"]["live_benchmark_used_for_training_or_selection"] is False
    assert payload["architecture_contract"]["clearance_ray_count"] == 24
    assert payload["architecture_contract"][
        "unsafe_motion_fallback"
    ] == "highest_safe_nonterminal_action"
    assert payload["runtime_separation"]["environmental_text_inputs"] == []
    assert payload["runtime_separation"]["oracle_inputs"] is False

    destination = tmp_path / "preregistration.json"
    path, digest = write_navigation_policy_v4_preregistration(
        destination,
        config,
        source_v3_dataset_sha256=V3_DATASET_SHA256,
        v4_dataset_sha256=V4_DATASET_SHA256,
        map_sha256=maps,
        training_report=tmp_path / "training.json",
    )
    assert path == destination.resolve()
    assert digest == file_sha256(destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    authenticated = authenticate_navigation_policy_v4_preregistration(
        destination,
        config,
        source_v3_dataset_sha256=V3_DATASET_SHA256,
        v4_dataset_sha256=V4_DATASET_SHA256,
        map_sha256=maps,
    )
    assert authenticated["authenticated"] is True
    assert authenticated["sha256"] == digest
    with pytest.raises(FileExistsError, match="requires absent"):
        write_navigation_policy_v4_preregistration(
            destination,
            config,
            source_v3_dataset_sha256=V3_DATASET_SHA256,
            v4_dataset_sha256=V4_DATASET_SHA256,
            map_sha256=maps,
            training_report=tmp_path / "training.json",
        )


def test_v4_preregistration_authentication_fails_closed_on_tamper(
    tmp_path: Path,
) -> None:
    config, maps = _inputs(tmp_path)
    destination, _digest = write_navigation_policy_v4_preregistration(
        tmp_path / "preregistration.json",
        config,
        source_v3_dataset_sha256=V3_DATASET_SHA256,
        v4_dataset_sha256=V4_DATASET_SHA256,
        map_sha256=maps,
        training_report=tmp_path / "training.json",
    )
    value = json.loads(destination.read_text(encoding="utf-8"))
    value["acceptance_gates"]["minimum_validation_action_accuracy"] = 0.0
    destination.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PreregistrationError, match="differs"):
        authenticate_navigation_policy_v4_preregistration(
            destination,
            config,
            source_v3_dataset_sha256=V3_DATASET_SHA256,
            v4_dataset_sha256=V4_DATASET_SHA256,
            map_sha256=maps,
        )


def test_v4_preregistration_rejects_unsealed_dataset_identity(tmp_path: Path) -> None:
    config, maps = _inputs(tmp_path)
    with pytest.raises(PreregistrationError, match="dataset identity"):
        build_navigation_policy_v4_preregistration(
            config,
            source_v3_dataset_sha256="0" * 64,
            v4_dataset_sha256=V4_DATASET_SHA256,
            map_sha256=maps,
        )


def test_v4_1_amendment_preserves_arm_and_locks_only_mechanical_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(
        load_config("configs/experiments/navigation_policy_v4_1.yaml")
    )
    settings = config["navigation_policy_v4"]
    settings["checkpoint_output"] = str(tmp_path / "checkpoint_v4_1")
    scene_ids = [*settings["train_scene_ids"], *settings["validation_scene_ids"]]
    maps = {
        scene_id: file_sha256(f"data_gemma4/maps/{scene_id}/voxel_map.npz")
        for scene_id in scene_ids
    }
    original = v41._load_sealed(
        v41.ORIGINAL_PREREGISTRATION, v41.ORIGINAL_PREREGISTRATION_SHA256
    )["implementation_source_hashes"]
    original_sha256 = v41._sha256
    monkeypatch.setattr(
        v41,
        "_sha256",
        lambda path: (
            original[str(path)]
            if str(path) in original
            and str(path)
            != "src/semantic_3d_chat/training/train_navigation_policy_v4.py"
            else original_sha256(path)
        ),
    )
    payload = build_navigation_policy_v41_preregistration(
        config,
        source_v3_dataset_sha256=V3_DATASET_SHA256,
        v4_dataset_sha256=V4_DATASET_SHA256,
        map_sha256=maps,
    )
    assert payload["original_failed_attempt"][
        "preregistration_sha256"
    ] == ORIGINAL_PREREGISTRATION_SHA256
    assert payload["original_failed_attempt"]["incident_sha256"] == INCIDENT_SHA256
    assert payload["mechanical_amendment"]["optimizer_math_changed"] is False
    assert payload["mechanical_amendment"]["training_loss_changed"] is False
    assert payload["source_audit"]["changed_original_source_paths"] == [
        "src/semantic_3d_chat/training/train_navigation_policy_v4.py"
    ]
    assert payload["preserved_single_arm"]["one_arm_only"] is True
    assert payload["preserved_single_arm"]["exact_deterministic_rerun"] is True

    destination = tmp_path / "v4_1_preregistration.json"
    path, digest = write_navigation_policy_v41_preregistration(
        destination,
        config,
        source_v3_dataset_sha256=V3_DATASET_SHA256,
        v4_dataset_sha256=V4_DATASET_SHA256,
        map_sha256=maps,
        training_report=tmp_path / "training_v4_1.json",
    )
    authenticated = authenticate_navigation_policy_v41_preregistration(
        path,
        config,
        source_v3_dataset_sha256=V3_DATASET_SHA256,
        v4_dataset_sha256=V4_DATASET_SHA256,
        map_sha256=maps,
    )
    assert authenticated["sha256"] == digest
    assert authenticated["protocol_version"] == "v4.1"
