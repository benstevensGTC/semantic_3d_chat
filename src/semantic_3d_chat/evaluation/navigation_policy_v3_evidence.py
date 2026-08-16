"""Fail-closed authentication for the measured V3 learned navigator.

This inspector is intentionally independent of the large report builder.  It
pins the exact V3 implementation, training evidence, two-file checkpoint, and
sealed live journal.  It does not open the oracle scoring specification or a
scene oracle; their digests are consumed only from the already separated score.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
TRAINING_STATUS: Final[str] = "supervised_continuous_semantic_grounded_navigation_policy_v3"
DATASET_SHA256: Final[str] = "d8d97ac248a5821eb971301efb742c25c996627bae22d6417c02755e61d50f9d"
WEIGHTS_SHA256: Final[str] = "975c7c6c5e103dd1bb055feb2eceff6cc7fe9ab82a3f7f492a8fbdb5a26cc87c"
CHECKPOINT_TREE_SHA256: Final[str] = (
    "1998d7313d023e65a5bef4c52eee00d867b4006344de51c6830e35c32f7ed1e0"
)
JOURNAL_ROOT_SHA256: Final[str] = "865e829fdcd6cf0bd0bb05c7f18f30fa57269d139649abe00104d0d983c55aa6"
SCORING_SPEC_SHA256: Final[str] = "586e57cdadddd05287816d55090b0a20da615ddfe45b7e90079565ccc25a9196"
SCENE_ORACLE_SHA256: Final[str] = "d3b7f56a810ac606964e8218b3150ad0f652edd4c0f139d1d7b8e3cf1882d78c"

# Keys are stable names used by tamper tests and machine-readable output.
PINNED_FILES: Final[dict[str, tuple[str, str]]] = {
    "historical_source_manifest": (
        "reports/gemma4/evidence/navigation_policy_v3_sources/manifest.json",
        "a4e81b3f2679070b8ae65f33e920dd1912bb6b8b96855888bceb38fb62ee0ab1",
    ),
    "config": (
        "configs/experiments/navigation_policy_v3.yaml",
        "9daf6bdbf2d059064a5b447c984b0cd394cb647c2994499110060dd90ebaea3a",
    ),
    "trace_manifest": (
        "data_gemma4/training/navigation_policy_v3/manifest.json",
        "005756918c54fbffbb7c6db45e2170174d85f87f278e755e538418d6eb880243",
    ),
    "trace_rows": (
        "data_gemma4/training/navigation_policy_v3/traces.jsonl",
        "72434178ff1cf23c2dfeb98d52cb7b4c443fcc8715c1dd4ee883d87ae127e7ad",
    ),
    "checkpoint_weights": (
        "data_gemma4/checkpoints/navigation_policy_v3/policy.safetensors",
        WEIGHTS_SHA256,
    ),
    "checkpoint_metadata": (
        "data_gemma4/checkpoints/navigation_policy_v3/runtime_metadata.json",
        "1350fb92e667a793088c7f4e4a3063f3aad0b218f50fff10c418ae850e5cc6d8",
    ),
    "training": (
        "reports/gemma4/metrics/navigation_policy_v3_training.json",
        "63fa9468cfd08423c48747e13b7d013ea1ac1b8867152c9dd0d57031ffb4c7a9",
    ),
    "offline_evaluation": (
        "reports/gemma4/metrics/navigation_policy_v3_offline_evaluation.json",
        "a00c3f99a30ac9dd98c3603e9f5d5e54b03fd2638904344405992dcb51681264",
    ),
    "controls": (
        "reports/gemma4/metrics/navigation_policy_v3_controls.json",
        "760bd61ebbeda69a42b18fecc653c8763acf89747ba6050f870718e2f95ae951",
    ),
    "runtime_audit": (
        "reports/gemma4/metrics/navigation_policy_v3_runtime_audit.json",
        "dd155e4dec2b1df05699e3bac2db741417d99efc71872d7c480c229d5b5a68d5",
    ),
    "task_manifest": (
        "configs/benchmarks/llm_navigation_v2_scene_000001.json",
        "29bd12966f28b0b9ecc4ba444af25bde712b98512d7c74e322cbc7019e4f5e07",
    ),
    "runtime_config": (
        "configs/runtime/embodied_navigation_v2.yaml",
        "5ee4610104f4d8058a5fd739678f0e400c2181d6875e0e223727bbbef40ccb13",
    ),
    "journal": (
        "reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3.json",
        "d5768be296ecc26bae02162b9dd564064fa62247966f98a132051068e1a6eed3",
    ),
    "inference_audit": (
        "reports/gemma4/metrics/llm_navigation_inference_access_learned_v3.json",
        "33fe6fd2356f9d4363c3aab57ac83acadb2df72e50950c46b6affe69e4d0c4a6",
    ),
    "score": (
        "reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3.json",
        "e62e8375cdc9d7e1a80e43fed5b02861e0e9d3f08325af287cee2bb957d6126e",
    ),
    "trajectory_data": (
        "reports/gemma4/examples/navigation_policy_v3_trajectories.json",
        "6a3399bc2b157af88da1fcc12dea8c0bc3f6fce9ad4ce37fa4f535cc20321ac1",
    ),
    "trajectory_figure": (
        "reports/gemma4/figures/navigation_policy_v3_trajectory.png",
        "0934e011787e1306431e98805a8b485c280fb5aa0ed658d26c4a47e3ee598bd3",
    ),
}

PINNED_SOURCES: Final[dict[str, tuple[str, str]]] = {
    "policy": (
        "reports/gemma4/evidence/navigation_policy_v3_sources/navigation_policy_v3.py",
        "4e687161f6174192a2e44de160c847a70c6dbbab09f7f3277373f6bceed5fcc2",
    ),
    "trace_builder": (
        "src/semantic_3d_chat/training/navigation_target_trace_v3.py",
        "d9ee0d92b049f5bc6e43f5a3b4212baf40d844bc4d6149a4b8fd33298ac71733",
    ),
    "trainer": (
        "src/semantic_3d_chat/training/train_navigation_policy_v3.py",
        "25e714eb350024f01ad6dd8bee7e64623a1b99b37942cb83ce2726ecb0d1dfce",
    ),
    "trace_cli": (
        "scripts/generate_navigation_policy_v3_traces.py",
        "6a53b705344c2b97053c78db3dc267924628456449c155d361d7bf664d4621ea",
    ),
    "train_cli": (
        "scripts/train_navigation_policy_v3.py",
        "b022185d9abbc0313c445fd6c785d736250b85f1718d56badd26f2ebd14e0067",
    ),
    "evaluate_cli": (
        "scripts/evaluate_navigation_policy_v3.py",
        "dc44823ede63fed6ea5bf1f2debedfa22ada3db164a0bafe5c065f96479b9bec",
    ),
    "controls_cli": (
        "scripts/evaluate_navigation_policy_v3_controls.py",
        "f75ab764c62ab24c925f4f0e426c7ea447096f5417a107f6e3b22e77ae607226",
    ),
    "audit_cli": (
        "scripts/audit_navigation_policy_v3_runtime.py",
        "e133782e8229f089c3bffcc4f0ea093d1803e4f9921e5987bd34be96f009c388",
    ),
    "inference_cli": (
        "reports/gemma4/evidence/navigation_policy_v3_sources/run_llm_navigation_inference.py",
        "df19394a1add0ace5dd4aa542989233ad2dea0ecdde1c115819828a71f44a31c",
    ),
    "tool_policy": (
        "reports/gemma4/evidence/navigation_policy_v3_sources/llm_tool_policy.py",
        "93100da22d93124a8928285bfba90f67dd1dfc38094de1d18ead9ae688140e25",
    ),
    "benchmark": (
        "reports/gemma4/evidence/navigation_policy_v3_sources/llm_navigation_benchmark.py",
        "0fcd836c56d15a96d905c8da078c1cf56dd2959eb94d1e0684cddd5ad2f20fa4",
    ),
    "scorer_cli": (
        "scripts/score_llm_navigation.py",
        "34a57dace3121e4d77f7e848c14c185ae351c8ff85789578608db8230184edd2",
    ),
}

_CHECKPOINT_METADATA_KEYS: Final[set[str]] = {
    "action_names",
    "all_map_voxels_scored_for_grounding",
    "architecture",
    "collision_interlock_required",
    "complete_scene_prefix_required",
    "continuous_semantic_grounding_required",
    "environmental_text_inputs",
    "every_scene_token_processed",
    "grounding_feature_dim",
    "grounding_feature_start",
    "hidden_size",
    "max_move_m",
    "max_turn_degrees",
    "model_dim",
    "model_id",
    "model_revision",
    "numeric_robot_tokens_required",
    "oracle_inputs_at_runtime",
    "query_dependent_grounding_navigation_only",
    "question_independent_static_scene_prefix_required",
    "robot_token_count",
    "room_size_m",
    "runtime_required_files",
    "scene_splits_disjoint",
    "scene_token_count",
    "schema_version",
    "target_state_dim",
    "task_trained",
    "train_scene_count",
    "training_dataset_sha256",
    "validation_scene_count",
    "weights_sha256",
}


class EvidenceAuthenticationError(RuntimeError):
    """Raised when any pinned V3 evidence invariant differs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceAuthenticationError(f"Expected JSON object: {path}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceAuthenticationError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceAuthenticationError(f"{name} is not finite")
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceAuthenticationError(message)


def _resolved_paths(
    root: Path,
    pins: Mapping[str, tuple[str, str]],
    overrides: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    unknown = set(overrides or ()) - set(pins)
    if unknown:
        raise EvidenceAuthenticationError(f"Unknown evidence overrides: {sorted(unknown)}")
    result: dict[str, Path] = {}
    for key, (relative, _digest) in pins.items():
        override = None if overrides is None else overrides.get(key)
        path = root / relative if override is None else Path(override).expanduser()
        path = Path(os.path.abspath(path))
        if path.is_symlink() or not path.is_file():
            raise EvidenceAuthenticationError(f"Evidence is not a regular file: {key}")
        result[key] = path
    return result


def _authenticate_pins(
    paths: Mapping[str, Path], pins: Mapping[str, tuple[str, str]], kind: str
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for key, path in paths.items():
        digest = _sha256_file(path)
        if digest != pins[key][1]:
            raise EvidenceAuthenticationError(f"Pinned V3 {kind} digest differs: {key}")
        observed[key] = digest
    return observed


def _validate_trace_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    train_ids = manifest.get("train_scene_ids")
    validation_ids = manifest.get("validation_scene_ids")
    _require(
        manifest.get("schema") == "semantic_3d_chat.navigation_target_trace_dataset.v3",
        "V3 trace schema differs",
    )
    _require(manifest.get("dataset_sha256") == DATASET_SHA256, "V3 dataset differs")
    _require(
        manifest.get("traces_sha256") == PINNED_FILES["trace_rows"][1],
        "V3 trace rows are not manifest-bound",
    )
    _require(manifest.get("sample_count") == 6468, "V3 trace sample count differs")
    _require(manifest.get("episode_count") == 1370, "V3 episode count differs")
    _require(
        isinstance(train_ids, list)
        and isinstance(validation_ids, list)
        and len(train_ids) == 14
        and len(validation_ids) == 8
        and set(train_ids).isdisjoint(validation_ids)
        and manifest.get("scene_splits_disjoint") is True,
        "V3 train/validation scene split differs",
    )
    _require(
        manifest.get("target_coordinates_oracle_derived") is True
        and manifest.get("target_coordinates_training_tree_only") is True
        and manifest.get("runtime_oracle_inputs") is False
        and manifest.get("checkpoint_contains_object_labels") is False
        and manifest.get("checkpoint_contains_trace_rows") is False,
        "V3 trace isolation contract differs",
    )
    _require(
        manifest.get("bounded_action_targets") is True
        and manifest.get("collision_checked_movement_targets") is True,
        "V3 trace safety contract differs",
    )
    return {
        "dataset_sha256": DATASET_SHA256,
        "sample_count": 6468,
        "episode_count": 1370,
        "train_scene_count": len(train_ids),
        "validation_scene_count": len(validation_ids),
        "scene_splits_disjoint": True,
    }


def _validate_checkpoint(checkpoint: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        checkpoint.is_dir()
        and not checkpoint.is_symlink()
        and sorted(path.name for path in checkpoint.iterdir())
        == ["policy.safetensors", "runtime_metadata.json"],
        "V3 checkpoint is not the exact two-file tree",
    )
    _require(_tree_sha256(checkpoint) == CHECKPOINT_TREE_SHA256, "V3 checkpoint tree differs")
    _require(set(metadata) == _CHECKPOINT_METADATA_KEYS, "V3 checkpoint metadata keys differ")
    _require(
        metadata.get("schema_version") == 3
        and metadata.get("architecture") == "continuous_semantic_grounded_navigation_controller_v3"
        and metadata.get("weights_sha256") == WEIGHTS_SHA256
        and metadata.get("training_dataset_sha256") == DATASET_SHA256
        and metadata.get("target_state_dim") == 10,
        "V3 checkpoint identity differs",
    )
    for key in (
        "task_trained",
        "scene_splits_disjoint",
        "complete_scene_prefix_required",
        "question_independent_static_scene_prefix_required",
        "every_scene_token_processed",
        "numeric_robot_tokens_required",
        "continuous_semantic_grounding_required",
        "all_map_voxels_scored_for_grounding",
        "query_dependent_grounding_navigation_only",
        "collision_interlock_required",
    ):
        _require(metadata.get(key) is True, f"V3 checkpoint contract differs: {key}")
    _require(metadata.get("oracle_inputs_at_runtime") is False, "V3 checkpoint permits oracle")
    _require(metadata.get("environmental_text_inputs") == [], "V3 checkpoint contains text")
    _require(
        metadata.get("runtime_required_files") == ["policy.safetensors", "runtime_metadata.json"],
        "V3 checkpoint runtime inventory differs",
    )
    return {
        "tree_sha256": CHECKPOINT_TREE_SHA256,
        "weights_sha256": WEIGHTS_SHA256,
        "file_count": 2,
        "target_state_dim": 10,
        "all_map_voxels_scored_for_grounding": True,
    }


def _validate_training_reports(
    training: Mapping[str, Any],
    offline: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    validation = offline.get("validation")
    _require(
        training.get("schema") == "semantic_3d_chat.navigation_policy_v3_training_result.v3"
        and training.get("status") == "accepted"
        and training.get("checkpoint_written") is True
        and training.get("dataset_sha256") == DATASET_SHA256
        and training.get("oracle_inputs_at_runtime") is False
        and training.get("environmental_text_inputs_at_runtime") == []
        and training.get("continuous_semantic_target_runtime") is True
        and training.get("query_dependent_grounding_navigation_only") is True
        and isinstance(training.get("gates"), Mapping)
        and all(training["gates"].values()),
        "V3 accepted training report differs",
    )
    _require(
        offline.get("schema") == "semantic_3d_chat.navigation_policy_v3_offline_evaluation.v3"
        and offline.get("dataset_sha256") == DATASET_SHA256
        and offline.get("checkpoint_weights_sha256") == WEIGHTS_SHA256
        and offline.get("oracle_inputs_used_by_runtime") is False
        and offline.get("environmental_text_inputs_at_runtime") == []
        and isinstance(validation, Mapping)
        and validation == training.get("validation"),
        "V3 offline evaluation differs",
    )
    expected_validation = {
        "action_accuracy": 0.9144620895385742,
        "targeted_action_accuracy": 0.9022177457809448,
        "stop_recall": 0.9028339982032776,
        "turn_sign_accuracy": 0.9398148059844971,
        "argument_mae": 0.13924801349639893,
    }
    for key, expected in expected_validation.items():
        _require(
            abs(_finite(validation.get(key), f"validation {key}") - expected) <= 1e-12,
            f"V3 validation metric differs: {key}",
        )
    conditions = controls.get("conditions")
    action_deltas = controls.get("action_accuracy_deltas_from_primary")
    turn_deltas = controls.get("turn_sign_accuracy_deltas_from_primary")
    target_output = controls.get("wrong_target_output_change")
    _require(
        controls.get("schema") == "semantic_3d_chat.navigation_policy_v3_causal_controls.v3"
        and controls.get("dataset_sha256") == DATASET_SHA256
        and controls.get("checkpoint_weights_sha256") == WEIGHTS_SHA256
        and controls.get("held_out_scenes_only") is True
        and controls.get("oracle_inputs_used_by_runtime") is False
        and controls.get("environmental_text_inputs_at_runtime") == []
        and isinstance(conditions, Mapping)
        and set(conditions)
        == {
            "primary",
            "wrong_scene_prefix",
            "zero_scene_prefix",
            "wrong_target_state",
            "zero_target_state",
        }
        and conditions.get("primary") == validation
        and isinstance(action_deltas, Mapping)
        and isinstance(turn_deltas, Mapping)
        and isinstance(target_output, Mapping),
        "V3 causal-control contract differs",
    )
    expected_action_deltas = {
        "wrong_scene_prefix": 0.003527343273162842,
        "zero_scene_prefix": -0.00044089555740356445,
        "wrong_target_state": 0.003527343273162842,
        "zero_target_state": 0.6481481492519379,
    }
    for key, expected in expected_action_deltas.items():
        observed = _finite(action_deltas.get(key), f"control {key}")
        recomputed = _finite(validation.get("action_accuracy"), "primary accuracy") - _finite(
            conditions[key].get("action_accuracy"), f"{key} accuracy"
        )
        _require(
            abs(observed - expected) <= 1e-12 and abs(observed - recomputed) <= 1e-12,
            f"V3 causal-control delta differs: {key}",
        )
    _require(
        abs(
            _finite(turn_deltas.get("wrong_target_state"), "wrong-target turn delta")
            - 0.09837961196899414
        )
        <= 1e-12
        and target_output.get("changed_turn_argument_sign_count") == 89
        and target_output.get("changed_targeted_action_count") == 26,
        "V3 wrong-target response differs",
    )
    return {
        "validation": {key: validation[key] for key in expected_validation},
        "action_accuracy_deltas": dict(action_deltas),
        "turn_sign_accuracy_deltas": dict(turn_deltas),
        "wrong_target_output_change": dict(target_output),
        "weak_direct_scene_prefix_controls": True,
        "target_state_materially_causal": True,
    }


def _validate_runtime_audit(audit: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    expected_files = {
        str((checkpoint / "policy.safetensors").resolve()),
        str((checkpoint / "runtime_metadata.json").resolve()),
    }
    _require(
        audit.get("schema") == "semantic_3d_chat.navigation_policy_v3_runtime_audit.v3"
        and audit.get("passed") is True
        and audit.get("oracle_directory_unavailable_during_load") is True
        and audit.get("oracle_directory_restored") is True
        and audit.get("forbidden_accesses") == []
        and audit.get("oracle_inputs_at_runtime") is False
        and audit.get("environmental_text_inputs_at_runtime") == []
        and audit.get("runtime_required_files") == ["policy.safetensors", "runtime_metadata.json"]
        and set(audit.get("loaded_files", [])) == expected_files
        and audit.get("weights_sha256") == WEIGHTS_SHA256,
        "V3 oracle-removal runtime audit differs",
    )
    return {
        "passed": True,
        "oracle_directory_unavailable": True,
        "forbidden_accesses": 0,
        "loaded_file_count": 2,
    }


def _validate_journal(
    journal: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    inference_audit: Mapping[str, Any],
) -> dict[str, Any]:
    body = {key: value for key, value in journal.items() if key != "journal_sha256"}
    header = journal.get("header")
    episodes = journal.get("episodes")
    runtime_audit = journal.get("runtime_file_audit")
    _require(
        journal.get("schema") == "semantic_3d_chat.llm_navigation_journal.v1"
        and journal.get("status") == "complete"
        and journal.get("journal_sha256") == JOURNAL_ROOT_SHA256
        and _canonical_sha256(body) == JOURNAL_ROOT_SHA256
        and isinstance(header, Mapping)
        and isinstance(episodes, list)
        and len(episodes) == 6
        and isinstance(runtime_audit, Mapping),
        "V3 live journal seal differs",
    )
    _require(
        header.get("tool_policy_training_status") == TRAINING_STATUS
        and header.get("oracle_or_labels_available") is False
        and header.get("claimed_learned_navigation_success") is False
        and header.get("task_manifest_sha256") == _canonical_sha256(task_manifest),
        "V3 live journal header differs",
    )
    contract = header.get("run_contract")
    _require(isinstance(contract, Mapping), "V3 live run contract is missing")
    source_hash = PINNED_SOURCES
    _require(
        contract.get("tool_policy_training_status") == TRAINING_STATUS
        and contract.get("navigation_policy_checkpoint_tree_sha256") == CHECKPOINT_TREE_SHA256
        and contract.get("navigation_policy_source_sha256") == source_hash["policy"][1]
        and contract.get("inference_source_sha256") == source_hash["inference_cli"][1]
        and contract.get("tool_policy_source_sha256") == source_hash["tool_policy"][1]
        and contract.get("fallback_policy") == "fail_closed"
        and contract.get("strict_fixed_environment_embedding_input") is True
        and contract.get("question_conditioned_scene_readout_tokens") is False
        and contract.get("continuous_semantic_grounding_required") is True
        and contract.get("all_map_voxels_scored_for_grounding") is True
        and contract.get("query_dependent_grounding_navigation_only") is True
        and contract.get("oracle_inputs_at_runtime") is False
        and contract.get("environmental_text_inputs_at_runtime") == [],
        "V3 live run provenance differs",
    )
    previous_episode_ids: set[str] = set()
    for episode in episodes:
        _require(isinstance(episode, Mapping), "V3 journal episode is invalid")
        episode_body = {key: value for key, value in episode.items() if key != "episode_sha256"}
        task_id = episode.get("task_id")
        _require(
            isinstance(task_id, str)
            and task_id not in previous_episode_ids
            and episode.get("episode_sha256") == _canonical_sha256(episode_body)
            and episode.get("tool_policy_training_status") == TRAINING_STATUS
            and episode.get("oracle_or_labels_available") is False
            and episode.get("environmental_text_inputs") == [],
            "V3 journal episode binding differs",
        )
        previous_episode_ids.add(task_id)
        chain = "0" * 64
        for step in episode.get("steps", []):
            _require(isinstance(step, Mapping), "V3 journal step is invalid")
            step_body = {
                key: value for key, value in step.items() if key != "transcript_chain_sha256"
            }
            chain = hashlib.sha256(
                bytes.fromhex(chain)
                + json.dumps(
                    step_body,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            _require(
                step.get("transcript_chain_sha256") == chain,
                "V3 transcript hash chain differs",
            )
        _require(
            episode.get("transcript_chain_sha256") == chain,
            "V3 episode transcript root differs",
        )
    _require(
        inference_audit.get("passed") is True
        and inference_audit.get("block_forbidden") is True
        and inference_audit.get("forbidden_accesses") == []
        and runtime_audit.get("passed") is True
        and runtime_audit.get("blocking_enabled") is True
        and runtime_audit.get("forbidden_accesses") == []
        and runtime_audit.get("audit_report_sha256") == PINNED_FILES["inference_audit"][1],
        "V3 live inference audit differs",
    )
    return {
        "journal_sha256": JOURNAL_ROOT_SHA256,
        "episode_count": 6,
        "inference_forbidden_access_count": 0,
        "continuous_semantic_grounding": True,
        "all_voxels_scored_per_grounding": True,
    }


def _validate_score(score: Mapping[str, Any], journal: Mapping[str, Any]) -> dict[str, Any]:
    metrics = score.get("metrics")
    by_family = score.get("by_family")
    tasks = score.get("tasks")
    separation = score.get("separation")
    feasibility = score.get("benchmark_feasibility")
    _require(
        score.get("schema") == "semantic_3d_chat.llm_navigation_score.v1"
        and score.get("scene_id") == "scene_000001"
        and score.get("policy_status") == TRAINING_STATUS
        and score.get("claimed_trained_navigation_policy") is True
        and score.get("passed") is False
        and score.get("navigation_policy_checkpoint_tree_sha256") == CHECKPOINT_TREE_SHA256
        and isinstance(metrics, Mapping)
        and isinstance(by_family, Mapping)
        and isinstance(tasks, list)
        and isinstance(separation, Mapping)
        and isinstance(feasibility, Mapping),
        "V3 live score contract differs",
    )
    _require(
        metrics.get("success_count") == 5
        and metrics.get("task_count") == 6
        and abs(_finite(metrics.get("success_rate"), "live success") - 5.0 / 6.0) <= 1e-12
        and metrics.get("executed_action_count") == 23
        and metrics.get("collision_count") == 0
        and metrics.get("action_failure_count") == 0
        and metrics.get("policy_rejection_count") == 0,
        "V3 live aggregate result differs",
    )
    expected_family = {
        "face": 1,
        "approach": 1,
        "stop": 1,
        "obstacle": 1,
        "left_right": 1,
        "update_after_scan": 0,
    }
    _require(
        set(by_family) == set(expected_family)
        and all(
            by_family[name].get("success_count") == success
            and by_family[name].get("task_count") == 1
            for name, success in expected_family.items()
        ),
        "V3 family results differ",
    )
    _require(
        separation.get("prediction_journal_sha256") == journal.get("journal_sha256")
        and separation.get("scoring_spec_sha256") == SCORING_SPEC_SHA256
        and separation.get("scene_oracle_sha256") == SCENE_ORACLE_SHA256
        and separation.get("inference_journal_validated_before_oracle_open") is True
        and separation.get("inference_received_oracle_or_labels") is False
        and separation.get("oracle_used_only_by_post_inference_scorer") is True,
        "V3 scoring separation differs",
    )
    _require(
        feasibility.get("benchmark_version") == 2
        and feasibility.get("preregistered_numeric_start") is True
        and feasibility.get("criteria_changed_from_v1") is False
        and feasibility.get("all_progress_criteria_feasible") is True,
        "V3 preregistered benchmark feasibility differs",
    )
    failed = [row for row in tasks if row.get("passed") is not True]
    _require(
        len(failed) == 1
        and failed[0].get("task_id") == "nav_005"
        and failed[0].get("checks", {}).get("target_standoff") is False
        and failed[0].get("checks", {}).get("successful_scan") is True
        and failed[0].get("checks", {}).get("updated_prefix_consumed") is True
        and failed[0].get("checks", {}).get("post_scan_motion") is True,
        "V3 live failure characterization differs",
    )
    return {
        "passed": False,
        "metrics": dict(metrics),
        "by_family": {key: dict(value) for key, value in by_family.items()},
        "failed_task_id": "nav_005",
        "failed_check": "target_standoff",
        "failed_final_target_standoff_m": failed[0]["metrics"]["final_target_standoff_m"],
        "failed_target_progress_m": failed[0]["metrics"]["target_progress_m"],
        "scan_update_consumed": True,
        "benchmark_feasibility": dict(feasibility),
    }


def _validate_trajectory_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    trajectories = value.get("trajectories")
    _require(
        value.get("schema") == "semantic_3d_chat.navigation_trajectories.v1"
        and value.get("scene_id") == "scene_000001"
        and value.get("source_journal_root_sha256") == JOURNAL_ROOT_SHA256
        and value.get("source_journal_sha256") == PINNED_FILES["journal"][1]
        and isinstance(trajectories, list)
        and len(trajectories) == 6
        and {row.get("task_id") for row in trajectories if isinstance(row, Mapping)}
        == {f"nav_{index:03d}" for index in range(6)},
        "V3 trajectory visualization data differs",
    )
    return {
        "trajectory_count": 6,
        "data_path": PINNED_FILES["trajectory_data"][0],
        "figure_path": PINNED_FILES["trajectory_figure"][0],
    }


def _validate_historical_source_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    sources = value.get("sources")
    expected_paths = {
        name: PINNED_SOURCES[name][0]
        for name in ("inference_cli", "tool_policy", "benchmark")
    }
    expected_hashes = {
        name: PINNED_SOURCES[name][1]
        for name in ("inference_cli", "tool_policy", "benchmark")
    }
    _require(
        value.get("schema")
        == "semantic_3d_chat.navigation_policy_v3_source_snapshot.v1"
        and value.get("status") == "historical_shared_source_bytes_materialized"
        and value.get("scope") == "sealed_v3_run_only"
        and value.get("current_runtime_source_claimed") is False
        and value.get("historical_journal_sha256") == JOURNAL_ROOT_SHA256
        and isinstance(sources, Mapping)
        and set(sources) == set(expected_paths),
        "V3 historical source-snapshot manifest differs",
    )
    for name, expected_path in expected_paths.items():
        entry = sources.get(name)
        _require(
            isinstance(entry, Mapping)
            and entry.get("historical_path") == expected_path
            and entry.get("sealed_sha256") == expected_hashes[name]
            and isinstance(entry.get("current_successor_path"), str)
            and isinstance(entry.get("current_successor_sha256"), str),
            f"V3 historical source-snapshot entry differs: {name}",
        )
    return {
        "scope": "historical_sealed_v3_run",
        "manifest_sha256": PINNED_FILES["historical_source_manifest"][1],
        "snapshot_source_count": 3,
        "exact_original_bytes_available": True,
        "current_runtime_source_claimed": False,
    }


def authenticate_navigation_policy_v3(
    *,
    root: str | Path = PROJECT_ROOT,
    file_overrides: Mapping[str, str | Path] | None = None,
    source_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Authenticate every pinned V3 artifact, raising on any mismatch."""

    project_root = Path(os.path.abspath(Path(root).expanduser()))
    file_paths = _resolved_paths(project_root, PINNED_FILES, file_overrides)
    source_paths = _resolved_paths(project_root, PINNED_SOURCES, source_overrides)
    file_hashes = _authenticate_pins(file_paths, PINNED_FILES, "artifact")
    source_hashes = _authenticate_pins(source_paths, PINNED_SOURCES, "source")

    trace = _validate_trace_manifest(_read_object(file_paths["trace_manifest"]))
    checkpoint_root = file_paths["checkpoint_weights"].parent
    metadata = _read_object(file_paths["checkpoint_metadata"])
    checkpoint = _validate_checkpoint(checkpoint_root, metadata)
    training = _validate_training_reports(
        _read_object(file_paths["training"]),
        _read_object(file_paths["offline_evaluation"]),
        _read_object(file_paths["controls"]),
    )
    runtime = _validate_runtime_audit(_read_object(file_paths["runtime_audit"]), checkpoint_root)
    task_manifest = _read_object(file_paths["task_manifest"])
    journal_value = _read_object(file_paths["journal"])
    journal = _validate_journal(
        journal_value,
        task_manifest,
        _read_object(file_paths["inference_audit"]),
    )
    live = _validate_score(_read_object(file_paths["score"]), journal_value)
    visualization = _validate_trajectory_artifact(_read_object(file_paths["trajectory_data"]))
    historical_sources = _validate_historical_source_snapshot(
        _read_object(file_paths["historical_source_manifest"])
    )
    return {
        "schema": "semantic_3d_chat.navigation_policy_v3_evidence.v3",
        "status": "authenticated_historical_partial_success",
        "measurement_authenticated": True,
        "evidence_version": "v3",
        "current_version": "v3_historical",
        "evidence_scope": "historical_sealed_run",
        "current_runtime_compatibility_claimed": False,
        "claimed_trained_navigation_policy": True,
        "complete_success_claimed": False,
        "scene_id": "scene_000001",
        "trace_dataset": trace,
        "checkpoint": checkpoint,
        "offline_training": training,
        "runtime_checkpoint_audit": runtime,
        "live_inference": journal,
        "live_benchmark": live,
        "trajectory_visualization": visualization,
        "historical_source_snapshot": historical_sources,
        "artifact_sha256": file_hashes,
        "implementation_source_sha256": source_hashes,
        "artifact_paths": {key: relative for key, (relative, _digest) in PINNED_FILES.items()},
        "implementation_source_paths": {
            key: relative for key, (relative, _digest) in PINNED_SOURCES.items()
        },
        "evidence_paths": [
            *(relative for relative, _digest in PINNED_FILES.values()),
            *(relative for relative, _digest in PINNED_SOURCES.values()),
        ],
        "scope_warning": (
            "This authenticates the exact historical V3 source bytes and sealed run, "
            "not compatibility with the current successor runtime. Held-out offline "
            "traces span eight scene-disjoint validation scenes, but "
            "the live Blender benchmark contains one unseen development scene. The "
            "grounded target state is strongly causal; direct raw scene-prefix controls "
            "are weak. Overall live benchmark pass remains false at 5/6."
        ),
    }


def inspect_navigation_policy_v3(
    *,
    root: str | Path = PROJECT_ROOT,
    file_overrides: Mapping[str, str | Path] | None = None,
    source_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Return a claim-bounded result instead of propagating authentication errors."""

    try:
        return authenticate_navigation_policy_v3(
            root=root,
            file_overrides=file_overrides,
            source_overrides=source_overrides,
        )
    except (EvidenceAuthenticationError, OSError, ValueError, TypeError, KeyError) as error:
        return {
            "schema": "semantic_3d_chat.navigation_policy_v3_evidence.v3",
            "status": "artifact_present_authentication_failed",
            "measurement_authenticated": False,
            "current_version": None,
            "claimed_trained_navigation_policy": False,
            "complete_success_claimed": False,
            "evidence_error": f"{type(error).__name__}: {error}",
        }


def main() -> int:
    """Print the read-only packaged-evidence check for Make/CI users."""

    result = inspect_navigation_policy_v3()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("measurement_authenticated") is True else 2


__all__ = [
    "CHECKPOINT_TREE_SHA256",
    "DATASET_SHA256",
    "JOURNAL_ROOT_SHA256",
    "PINNED_FILES",
    "PINNED_SOURCES",
    "WEIGHTS_SHA256",
    "EvidenceAuthenticationError",
    "authenticate_navigation_policy_v3",
    "inspect_navigation_policy_v3",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
