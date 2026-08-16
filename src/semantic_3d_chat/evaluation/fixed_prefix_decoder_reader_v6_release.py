"""Create-once release chain for the audited fixed-prefix V6 reader.

The mutable proposal was independently audited before this module existed.  A
sealed preregistration binds those exact four files and the complete local model
blob.  A separate release then authorizes exactly one zero-update MPS smoke.
Only a passing, byte-authenticated smoke can authorize the later fixed trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    INITIAL_STATE_SHA256,
    MODEL_WEIGHTS_BLOB_SHA256,
    MODEL_WEIGHTS_SIZE_BYTES,
    TARGET_MODULES,
    _canonical_sha256,
    _model_snapshot,
    build_preregistration_draft,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_LORA_STATE_SHA256 as TOOL_INITIAL_LORA_STATE_SHA256,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_PROJECTOR_STATE_SHA256,
)

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_decoder_reader_v6"
PROPOSAL_SHA256: Final[str] = (
    "fd2a85ae8f29e56a710fe6c3c63970c5c1593ea4d7aa12b9a815bbf0a9edfa14"
)
PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_preregistration.json"
)
MPS_SMOKE_RELEASE: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke_release.json"
)
MPS_SMOKE_ATTEMPT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke_attempt.json"
)
MPS_SMOKE_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke.json"
)
TRAINING_RELEASE: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_training_release.json"
)
RESULT_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_result.json"
)
OUTPUT_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_decoder_reader_v6"
)

FROZEN_PROPOSAL_HASHES: Final[dict[str, str]] = {
    "configs/experiments/gemma4_v54_fixed_prefix_decoder_reader_v6.yaml": (
        "cad5f0af664021b6e5c2bacb2ad1261d3222862e916b320effffe75ae6ab5cf0"
    ),
    "src/semantic_3d_chat/language/fixed_prefix_decoder_reader_v6.py": (
        "4214846092f1fe0ca16a2c4e8cc369fad47598453e12c62c429fcc6e698a9627"
    ),
    "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_preregistration.py": (
        "b347df8a9cd52c50a76b9a4455fae04be0ffc4fc250256f0f9cff5bf3cc5ef56"
    ),
    "tests/test_fixed_prefix_decoder_reader_v6.py": (
        "62195e460be667bdf3fad83ab203eec7e0d4cac77cbb24e5c9b92bf9a6560a3b"
    ),
}
ROBOT_STATE_HASHES: Final[dict[str, str]] = {
    "data_gemma4/checkpoints/robot_state_numeric_v1/state.safetensors": (
        "5d6aa13208264e0a99755d84e8f68b7727249b274c460e9d4e26541cd8e46938"
    ),
    "data_gemma4/checkpoints/robot_state_numeric_v1/runtime_metadata.json": (
        "c48b8748dbde04f2c9294321974b1b13be2d77083970f051ba1c11a9b42d1985"
    ),
}
SMOKE_BOUND_PATHS: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "uv.lock",
    "requirements-gemma4-probe.txt",
    "src/semantic_3d_chat/__init__.py",
    "src/semantic_3d_chat/chat/__init__.py",
    "src/semantic_3d_chat/chat/file_audit.py",
    "src/semantic_3d_chat/chat/runtime.py",
    "src/semantic_3d_chat/chat/runtime_config.py",
    "src/semantic_3d_chat/config.py",
    "src/semantic_3d_chat/coordinates.py",
    "src/semantic_3d_chat/data/__init__.py",
    "src/semantic_3d_chat/data/dataset.py",
    "src/semantic_3d_chat/device.py",
    "src/semantic_3d_chat/evaluation/__init__.py",
    "src/semantic_3d_chat/evaluation/candidate_gate_detail.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_preregistration.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_release.py",
    "src/semantic_3d_chat/evaluation/gemma4_semantic_sanity.py",
    "src/semantic_3d_chat/evaluation/gemma4_tool_decoder_preregistration_v2.py",
    "src/semantic_3d_chat/evaluation/semantic_sanity.py",
    "src/semantic_3d_chat/language/__init__.py",
    "src/semantic_3d_chat/language/fixed_prefix_decoder_reader_v6.py",
    "src/semantic_3d_chat/language/gemma4_answer_tail.py",
    "src/semantic_3d_chat/language/gemma4_backend.py",
    "src/semantic_3d_chat/language/gemma4_tool_decoder_v1.py",
    "src/semantic_3d_chat/language/gemma4_tool_decoder_v2.py",
    "src/semantic_3d_chat/language/generation.py",
    "src/semantic_3d_chat/language/local_lm.py",
    "src/semantic_3d_chat/language/lora.py",
    "src/semantic_3d_chat/language/prefix_injection.py",
    "src/semantic_3d_chat/mapping/__init__.py",
    "src/semantic_3d_chat/mapping/depth_projection.py",
    "src/semantic_3d_chat/mapping/fusion.py",
    "src/semantic_3d_chat/mapping/semantic_codec.py",
    "src/semantic_3d_chat/mapping/voxel_map.py",
    "src/semantic_3d_chat/rendering_io.py",
    "src/semantic_3d_chat/robot/__init__.py",
    "src/semantic_3d_chat/robot/collision.py",
    "src/semantic_3d_chat/robot/llm_tool_policy.py",
    "src/semantic_3d_chat/robot/navigation_policy.py",
    "src/semantic_3d_chat/robot/runtime_refresh.py",
    "src/semantic_3d_chat/robot/semantic_mapping.py",
    "src/semantic_3d_chat/robot/simulator.py",
    "src/semantic_3d_chat/robot/state_checkpoint.py",
    "src/semantic_3d_chat/robot/state_encoder.py",
    "src/semantic_3d_chat/robot/tools.py",
    "src/semantic_3d_chat/scene_encoder/__init__.py",
    "src/semantic_3d_chat/scene_encoder/block_cross_residual.py",
    "src/semantic_3d_chat/scene_encoder/dense_alignment.py",
    "src/semantic_3d_chat/scene_encoder/dense_sidecar_adapter.py",
    "src/semantic_3d_chat/scene_encoder/global_residual.py",
    "src/semantic_3d_chat/scene_encoder/map_io.py",
    "src/semantic_3d_chat/scene_encoder/perceiver.py",
    "src/semantic_3d_chat/scene_encoder/point_tokens.py",
    "src/semantic_3d_chat/scene_encoder/projector.py",
    "src/semantic_3d_chat/scene_encoder/signed_x_dispatch.py",
    "src/semantic_3d_chat/scene_encoder/spatial_blocks.py",
    "src/semantic_3d_chat/training/__init__.py",
    "src/semantic_3d_chat/training/checkpointing.py",
    "src/semantic_3d_chat/training/dense_alignment_calibration.py",
    "src/semantic_3d_chat/training/dense_alignment_supervision.py",
    "src/semantic_3d_chat/training/losses.py",
    "src/semantic_3d_chat/training/pair_curriculum.py",
    "src/semantic_3d_chat/training/smoke_fixed_prefix_decoder_reader_v6.py",
    "src/semantic_3d_chat/training/source_provenance.py",
    "src/semantic_3d_chat/training/train_adapter.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v4.py",
    "src/semantic_3d_chat/vision/__init__.py",
    "src/semantic_3d_chat/vision/encoder.py",
    "src/semantic_3d_chat/vision/gemma4_encoder.py",
    "src/semantic_3d_chat/vision/gemma4_probe.py",
    "src/semantic_3d_chat/vision/model_registry.py",
    "src/semantic_3d_chat/vision/patch_features.py",
    "scripts/run_gemma4_v54_fixed_prefix_decoder_reader_v6.sh",
    "tests/test_fixed_prefix_decoder_reader_v6_release.py",
    "tests/test_smoke_fixed_prefix_decoder_reader_v6.py",
)
TRAINING_BOUND_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_release.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_decoder_reader_v6.py",
    "scripts/run_gemma4_v54_fixed_prefix_decoder_reader_v6.sh",
    "tests/test_train_fixed_prefix_decoder_reader_v6.py",
)
_MPS_MEMORY_PHASES: Final[frozenset[str]] = frozenset(
    {
        "before_model_load",
        "after_model_load_and_prefix_cache",
        "after_full_vs_tail_equivalence",
        "after_retention_teacher",
        "after_full_logit_cache_clear",
        "after_v6_reader_install",
        "after_v6_zero_output_forward",
        "after_contrastive_forwards",
        "after_contrastive_backward",
        "after_broad_forward",
        "after_broad_backward",
        "after_retention_forward",
        "after_retention_backward",
        "after_v6_gradient_validation",
        "after_numeric_robot_tool_inputs",
        "after_reader_only_tool_forward",
        "after_zero_output_tool_install",
        "after_joint_zero_output_tool_forward",
        "after_joint_state_roundtrip",
    }
)
_DEFERRED_FINAL_SCENES: Final[frozenset[str]] = frozenset(
    f"scene_{index:06d}" for index in (*range(25, 31), *range(57, 63))
)
_EXPECTED_SOFTWARE_VERSIONS: Final[dict[str, str]] = {
    "python": "3.12.13",
    "numpy": "2.5.1",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "transformers": "5.14.1",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V6 released source is missing or unsafe: {relative}")
        result[relative] = sha256_file(source)
    return result


def _authenticate_frozen_proposal() -> dict[str, Any]:
    observed = _bound_hashes(tuple(FROZEN_PROPOSAL_HASHES))
    if observed != FROZEN_PROPOSAL_HASHES:
        raise ValueError("V6 frozen proposal source hashes changed")
    proposal = build_preregistration_draft()
    digest = _canonical_sha256(proposal)
    if digest != PROPOSAL_SHA256:
        raise ValueError(f"V6 canonical proposal changed: {digest} != {PROPOSAL_SHA256}")
    weights = _model_snapshot() / "model.safetensors"
    resolved = weights.resolve()
    if resolved.stat().st_size != MODEL_WEIGHTS_SIZE_BYTES:
        raise ValueError("V6 local Gemma weight size changed")
    # Do not trust only Hugging Face's content-addressed filename.  This streams
    # and verifies the actual 10.25 GB blob at every release authentication.
    actual_model_hash = sha256_file(weights)
    if actual_model_hash != MODEL_WEIGHTS_BLOB_SHA256:
        raise ValueError("V6 local Gemma weight bytes changed")
    return {
        "source_sha256": observed,
        "canonical_proposal_sha256": digest,
        "local_model_weights_sha256": actual_model_hash,
        "local_model_weights_size_bytes": MODEL_WEIGHTS_SIZE_BYTES,
        "actual_model_bytes_streamed": True,
    }


def build_sealed_preregistration() -> dict[str, Any]:
    frozen = _authenticate_frozen_proposal()
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_sealed_preregistration",
        "status": "sealed_before_real_mps_smokes_training_not_authorized",
        "frozen_proposal": frozen,
        "independent_audit": {
            "status": "passed",
            "full_model_blob_independently_streamed": True,
            "all_40_prefix_files_authenticated": True,
            "train_rows": 576,
            "validation_rows": 384,
            "train_answer_varying_rows": 288,
            "validation_answer_varying_rows": 170,
            "schedule_updates": 96,
            "deferred_or_final_qa_accessed": False,
        },
        "authorization": {
            "sealed": True,
            "full_model_mps_smoke_authorized": False,
            "joint_runtime_smoke_authorized": False,
            "optimizer_construction_authorized": False,
            "multi_update_training_authorized": False,
            "checkpoint_write_authorized": False,
        },
        "required_next_stage": "separate_create_once_mps_smoke_release",
    }


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def _create_once(path: str | Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V6 create-once artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialized(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(payload).hexdigest()


def write_sealed_preregistration() -> tuple[Path, str]:
    return _create_once(PREREGISTRATION, build_sealed_preregistration())


def authenticate_sealed_preregistration() -> tuple[dict[str, Any], str]:
    source = _resolve(PREREGISTRATION)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V6 sealed preregistration is missing or unsafe")
    observed = json.loads(source.read_text(encoding="utf-8"))
    expected = build_sealed_preregistration()
    if observed != expected:
        raise ValueError("V6 sealed preregistration differs from frozen sources")
    return observed, sha256_file(source)


def _build_mps_smoke_release(*, require_output_absent: bool) -> dict[str, Any]:
    preregistration, preregistration_sha = authenticate_sealed_preregistration()
    if require_output_absent and any(
        _resolve(path).exists() for path in (MPS_SMOKE_ATTEMPT, MPS_SMOKE_REPORT)
    ):
        raise FileExistsError("V6 MPS smoke already has an attempt or terminal report")
    robot_hashes = _bound_hashes(tuple(ROBOT_STATE_HASHES))
    if robot_hashes != ROBOT_STATE_HASHES:
        raise ValueError("V6 robot-state runtime bytes changed")
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_mps_smoke_release",
        "status": "released_exactly_one_zero_update_full_model_mps_smoke",
        "parent_preregistration": PREREGISTRATION,
        "parent_preregistration_sha256": preregistration_sha,
        "parent_status": preregistration["status"],
        "bound_source_sha256": _bound_hashes(SMOKE_BOUND_PATHS),
        "robot_state_runtime_sha256": robot_hashes,
        "authorized": {
            "full_model_mps_answer_tail_equivalence": True,
            "full_model_mps_v6_gradient": True,
            "full_model_joint_v6_tool_v2_zero_output_structural_coexistence": True,
            "joint_nonzero_semantic_or_tool_behavior": False,
            "maximum_smoke_runs": 1,
            "optimizer_construction": False,
            "optimizer_steps": 0,
            "multi_update_training": False,
            "checkpoint_write": False,
            "deferred_or_final_qa_access": False,
        },
        "required_software_versions": _EXPECTED_SOFTWARE_VERSIONS,
        "attempt_journal": MPS_SMOKE_ATTEMPT,
        "terminal_output": MPS_SMOKE_REPORT,
    }


def build_mps_smoke_release() -> dict[str, Any]:
    return _build_mps_smoke_release(require_output_absent=True)


def write_mps_smoke_release() -> tuple[Path, str]:
    return _create_once(MPS_SMOKE_RELEASE, build_mps_smoke_release())


def authenticate_mps_smoke_release() -> tuple[dict[str, Any], str]:
    source = _resolve(MPS_SMOKE_RELEASE)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V6 MPS smoke release is missing or unsafe")
    observed = json.loads(source.read_text(encoding="utf-8"))
    expected = _build_mps_smoke_release(require_output_absent=False)
    if observed != expected:
        raise ValueError("V6 MPS smoke release or bound source changed")
    return observed, sha256_file(source)


def _smoke_attempt_payload(release_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_mps_smoke_attempt",
        "status": "claimed_before_model_loading",
        "authorization": MPS_SMOKE_RELEASE,
        "authorization_sha256": release_sha256,
        "maximum_attempts": 1,
        "optimizer_construction_authorized": False,
        "optimizer_steps_authorized": 0,
        "checkpoint_write_authorized": False,
    }


def claim_mps_smoke_attempt() -> tuple[Path, str]:
    _release, release_sha = authenticate_mps_smoke_release()
    if _resolve(MPS_SMOKE_REPORT).exists():
        raise FileExistsError("V6 MPS smoke already has a terminal report")
    return _create_once(MPS_SMOKE_ATTEMPT, _smoke_attempt_payload(release_sha))


def authenticate_mps_smoke_attempt() -> tuple[dict[str, Any], str]:
    source = _resolve(MPS_SMOKE_ATTEMPT)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V6 MPS smoke attempt journal is missing or unsafe")
    release_sha = sha256_file(MPS_SMOKE_RELEASE)
    observed = json.loads(source.read_text(encoding="utf-8"))
    expected = _smoke_attempt_payload(release_sha)
    if observed != expected:
        raise ValueError("V6 MPS smoke attempt journal differs from its release")
    return observed, sha256_file(source)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float("-inf") < float(value) < float("inf")
    )


def _path_inventory_sha256(paths: list[str]) -> str:
    payload = json.dumps(
        paths, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _forbidden_evaluation_path(path: str) -> bool:
    candidate = Path(path)
    components = set(candidate.parts)
    lowered = path.casefold()
    return bool(
        components & _DEFERRED_FINAL_SCENES
        or "v56_fresh_development_validation" in lowered
        or lowered.endswith(
            (
                "/questions/test.json",
                "/qa/test.jsonl",
                "/data_diverse52/qa/validation.jsonl",
            )
        )
    )


def _authenticate_passing_smoke() -> tuple[dict[str, Any], str]:
    source = _resolve(MPS_SMOKE_REPORT)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V6 real MPS smoke report is missing or unsafe")
    report = json.loads(source.read_text(encoding="utf-8"))
    _attempt, attempt_sha = authenticate_mps_smoke_attempt()
    expected_fields = {
        "schema_version",
        "artifact",
        "status",
        "passed",
        "authorization_sha256",
        "attempt_sha256",
        "device",
        "software_versions",
        "full_model_loaded",
        "mps_used",
        "optimizer_constructed",
        "optimizer_steps",
        "training_executed",
        "checkpoint_published",
        "answer_tail_equivalence_passed",
        "full_vs_tail_selected_logits_exact",
        "full_vs_tail_selected_logits_max_abs_difference",
        "full_vs_tail_per_token_nll_max_abs_difference",
        "full_vs_tail_mean_nll_absolute_difference",
        "full_vs_tail_targets_exact",
        "full_vs_tail_label_positions_exact",
        "full_vs_tail_causal_positions_exact",
        "v6_zero_output_exact_noop",
        "v6_initial_state_sha256",
        "v6_gradient_l2",
        "v6_gradient_by_module",
        "v6_lora_b_gradient_l2_by_target",
        "v6_lora_a_gradient_l2_expected_zero_by_target",
        "both_v6_adapter_gradients_nonzero",
        "contrastive_correct_nll",
        "contrastive_wrong_nll",
        "contrastive_margin",
        "broad_nll",
        "retention_self_kl",
        "joint_zero_output_structural_runtime_coexistence_passed",
        "joint_nonzero_semantic_or_tool_behavior_proven",
        "joint_zero_output_exact_noop",
        "tool_numeric_projector_state_sha256",
        "joint_state_roundtrip",
        "scene_prefix_shape",
        "question_dependent_scene_retrieval",
        "environmental_text_inputs",
        "file_access_audit_active_for_entire_execution",
        "loaded_files",
        "loaded_file_count",
        "loaded_file_inventory_sha256",
        "forbidden_file_accesses",
        "deferred_or_final_qa_accessed",
        "memory",
        "elapsed_seconds",
    }
    if not isinstance(report, dict) or set(report) != expected_fields:
        raise ValueError("V6 real MPS smoke report schema changed")
    release_sha = sha256_file(MPS_SMOKE_RELEASE)
    b_gradients = report.get("v6_lora_b_gradient_l2_by_target")
    a_gradients = report.get("v6_lora_a_gradient_l2_expected_zero_by_target")
    roundtrip = report.get("joint_state_roundtrip")
    memory = report.get("memory")
    loaded_files = report.get("loaded_files")
    gradient_by_module = report.get("v6_gradient_by_module")
    numeric_losses = (
        report.get("v6_gradient_l2"),
        report.get("contrastive_correct_nll"),
        report.get("contrastive_wrong_nll"),
        report.get("contrastive_margin"),
        report.get("broad_nll"),
        report.get("retention_self_kl"),
        report.get("elapsed_seconds"),
    )
    required = (
        report.get("schema_version") == 1
        and report.get("artifact")
        == "gemma4_v54_fixed_prefix_decoder_reader_v6_real_mps_smoke"
        and report.get("status") == "passed"
        and report.get("passed") is True
        and report.get("authorization_sha256") == release_sha
        and report.get("attempt_sha256") == attempt_sha
        and report.get("device") == "mps"
        and report.get("software_versions") == _EXPECTED_SOFTWARE_VERSIONS
        and report.get("full_model_loaded") is True
        and report.get("mps_used") is True
        and report.get("optimizer_constructed") is False
        and report.get("optimizer_steps") == 0
        and report.get("training_executed") is False
        and report.get("checkpoint_published") is False
        and report.get("answer_tail_equivalence_passed") is True
        and report.get("full_vs_tail_selected_logits_exact") is True
        and report.get("full_vs_tail_selected_logits_max_abs_difference") == 0.0
        and report.get("full_vs_tail_per_token_nll_max_abs_difference") == 0.0
        and report.get("full_vs_tail_mean_nll_absolute_difference") == 0.0
        and report.get("full_vs_tail_targets_exact") is True
        and report.get("full_vs_tail_label_positions_exact") is True
        and report.get("full_vs_tail_causal_positions_exact") is True
        and report.get("v6_zero_output_exact_noop") is True
        and report.get("v6_initial_state_sha256") == INITIAL_STATE_SHA256
        and report.get("both_v6_adapter_gradients_nonzero") is True
        and isinstance(b_gradients, dict)
        and set(b_gradients) == set(TARGET_MODULES)
        and all(_finite_number(value) and float(value) > 0.0 for value in b_gradients.values())
        and isinstance(a_gradients, dict)
        and set(a_gradients) == set(TARGET_MODULES)
        and all(value == 0.0 for value in a_gradients.values())
        and isinstance(gradient_by_module, dict)
        and set(gradient_by_module) == set(TARGET_MODULES)
        and all(
            isinstance(value, dict)
            and value.get("lora_a") == a_gradients[target]
            and value.get("lora_b") == b_gradients[target]
            and _finite_number(value.get("total_l2"))
            for target, value in gradient_by_module.items()
        )
        and all(_finite_number(value) for value in numeric_losses)
        and float(report.get("v6_gradient_l2")) > 0.0
        and float(report.get("contrastive_correct_nll")) > 0.0
        and float(report.get("contrastive_wrong_nll")) > 0.0
        and float(report.get("broad_nll")) > 0.0
        and abs(float(report.get("retention_self_kl"))) <= 1e-5
        and abs(
            float(report.get("contrastive_wrong_nll"))
            - float(report.get("contrastive_correct_nll"))
            - float(report.get("contrastive_margin"))
        )
        <= 1e-6
        and report.get("joint_zero_output_structural_runtime_coexistence_passed") is True
        and report.get("joint_nonzero_semantic_or_tool_behavior_proven") is False
        and report.get("joint_zero_output_exact_noop") is True
        and report.get("tool_numeric_projector_state_sha256")
        == INITIAL_PROJECTOR_STATE_SHA256
        and isinstance(roundtrip, dict)
        and set(roundtrip)
        == {
            "reader_state_sha256",
            "tool_state_sha256",
            "serialized_bytes",
            "strict_state_roundtrip",
        }
        and roundtrip.get("reader_state_sha256") == INITIAL_STATE_SHA256
        and roundtrip.get("tool_state_sha256") == TOOL_INITIAL_LORA_STATE_SHA256
        and type(roundtrip.get("serialized_bytes")) is int
        and roundtrip.get("serialized_bytes", 0) > 0
        and roundtrip.get("strict_state_roundtrip") is True
        and report.get("scene_prefix_shape") == [1, 258, 1536]
        and report.get("question_dependent_scene_retrieval") is False
        and report.get("environmental_text_inputs") == []
        and report.get("file_access_audit_active_for_entire_execution") is True
        and isinstance(loaded_files, list)
        and all(isinstance(path, str) and Path(path).is_absolute() for path in loaded_files)
        and loaded_files == sorted(set(loaded_files))
        and not any(_forbidden_evaluation_path(path) for path in loaded_files)
        and type(report.get("loaded_file_count")) is int
        and report.get("loaded_file_count", 0) > 0
        and report.get("loaded_file_count") == len(loaded_files)
        and isinstance(report.get("loaded_file_inventory_sha256"), str)
        and report.get("loaded_file_inventory_sha256")
        == _path_inventory_sha256(loaded_files)
        and report.get("forbidden_file_accesses") == []
        and report.get("deferred_or_final_qa_accessed") is False
        and isinstance(memory, dict)
        and set(memory)
        == {
            "peak_process_rss_bytes",
            "mps_current_allocated_bytes",
            "mps_driver_allocated_bytes",
            "mps_driver_allocated_bytes_sampled_peak",
            "mps_driver_sample_count",
            "mps_driver_samples_by_phase",
        }
        and type(memory.get("mps_driver_allocated_bytes_sampled_peak")) is int
        and 0 < memory.get("mps_driver_allocated_bytes_sampled_peak", 0)
        <= 25_000_000_000
        and type(memory.get("mps_driver_sample_count")) is int
        and memory.get("mps_driver_sample_count") == len(_MPS_MEMORY_PHASES)
        and isinstance(memory.get("mps_driver_samples_by_phase"), dict)
        and set(memory.get("mps_driver_samples_by_phase", {})) == _MPS_MEMORY_PHASES
        and all(
            type(value) is int and value >= 0
            for value in memory.get("mps_driver_samples_by_phase", {}).values()
        )
        and memory.get("mps_driver_allocated_bytes_sampled_peak")
        == max(memory.get("mps_driver_samples_by_phase", {}).values())
    )
    if not required:
        raise ValueError("V6 real MPS smoke did not pass every locked condition")
    return report, sha256_file(source)


def _build_training_release(*, require_outputs_absent: bool) -> dict[str, Any]:
    authenticate_mps_smoke_release()
    smoke, smoke_sha = _authenticate_passing_smoke()
    if require_outputs_absent and (
        _resolve(RESULT_REPORT).exists() or _resolve(OUTPUT_CHECKPOINT).exists()
    ):
        raise FileExistsError("V6 already has a terminal result or checkpoint")
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_training_release",
        "status": "released_single_fixed_96_update_training_arm",
        "parent_smoke": MPS_SMOKE_REPORT,
        "parent_smoke_sha256": smoke_sha,
        "parent_smoke_authorization_sha256": smoke["authorization_sha256"],
        "bound_source_sha256": _bound_hashes(TRAINING_BOUND_PATHS),
        "authorized": {
            "optimizer": "adamw_exact_pinned_parameters",
            "optimizer_updates": 96,
            "final_state_only": True,
            "internal_validation": True,
            "greedy_only_after_all_teacher_and_retention_gates": True,
            "checkpoint_only_if_every_gate_passes": True,
            "deferred_holdout_only_after_internal_passes": True,
            "final_split_access": False,
        },
        "result": RESULT_REPORT,
        "checkpoint": OUTPUT_CHECKPOINT,
    }


def build_training_release() -> dict[str, Any]:
    return _build_training_release(require_outputs_absent=True)


def write_training_release() -> tuple[Path, str]:
    return _create_once(TRAINING_RELEASE, build_training_release())


def authenticate_training_release() -> tuple[dict[str, Any], str]:
    source = _resolve(TRAINING_RELEASE)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V6 training release is missing or unsafe")
    observed = json.loads(source.read_text(encoding="utf-8"))
    expected = _build_training_release(require_outputs_absent=False)
    if observed != expected:
        raise ValueError("V6 training release or bound source changed")
    return observed, sha256_file(source)


__all__ = [
    "ARTIFACT",
    "MPS_SMOKE_ATTEMPT",
    "MPS_SMOKE_RELEASE",
    "MPS_SMOKE_REPORT",
    "OUTPUT_CHECKPOINT",
    "PREREGISTRATION",
    "PROPOSAL_SHA256",
    "RESULT_REPORT",
    "TRAINING_RELEASE",
    "authenticate_mps_smoke_attempt",
    "authenticate_mps_smoke_release",
    "authenticate_sealed_preregistration",
    "authenticate_training_release",
    "build_mps_smoke_release",
    "build_sealed_preregistration",
    "build_training_release",
    "claim_mps_smoke_attempt",
    "sha256_file",
    "write_mps_smoke_release",
    "write_sealed_preregistration",
    "write_training_release",
]
