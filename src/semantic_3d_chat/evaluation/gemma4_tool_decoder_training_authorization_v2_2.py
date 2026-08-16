"""Immutable staged authorization chain for executable Gemma decoder V2.2.

V2.2 supersedes (without modifying) the sealed V2.1 CPU artifact.  Its only
scope expansion is the bound default training runner and the in-process saved
runtime execution probe.  The CPU artifact still denies both full-model MPS
loading and optimizer construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

CPU_AUTHORIZATION_PATH: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_embodied_tool_decoder_training_authorization_v2_2.json"
)
MPS_SMOKE_RELEASE_PATH: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_embodied_tool_decoder_mps_smoke_release_v2_2.json"
)
MPS_SMOKE_REPORT_PATH: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_embodied_tool_decoder_full_mps_smoke_v2_2.json"
)
TRAINING_RELEASE_PATH: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_embodied_tool_decoder_multi_update_release_v2_2.json"
)
SUPERSEDED_AUTHORIZATION_PATH: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_embodied_tool_decoder_training_authorization_v2_1.json"
)
SUPERSEDED_AUTHORIZATION_SHA256: Final[str] = (
    "3fd44b1055974044c55959be4ba81e9fbb8e8569cc82819f76d0bf57644f5e0b"
)
EVALUATION_V2_1_SHA256: Final[str] = (
    "7b322d57ed46d920f7253383e75254b4157ac5397afc172bf7ffd0141310e007"
)
ARTIFACT: Final[str] = "gemma4_embodied_tool_decoder_training_authorization_v2_2"
BOUND_SOURCE_PATHS: Final[tuple[str, ...]] = (
    "configs/experiments/gemma4_embodied_tool_decoder_v2.yaml",
    "src/semantic_3d_chat/language/gemma4_answer_tail.py",
    "src/semantic_3d_chat/language/gemma4_tool_decoder_v2.py",
    "src/semantic_3d_chat/language/gemma4_tool_decoder_v2_checkpoint.py",
    "src/semantic_3d_chat/training/gemma4_tool_decoder_v2_clearance.py",
    "src/semantic_3d_chat/training/gemma4_tool_decoder_v2_data.py",
    "src/semantic_3d_chat/training/train_gemma4_tool_decoder_v2.py",
    "src/semantic_3d_chat/evaluation/gemma4_tool_decoder_v2_evaluation.py",
    (
        "src/semantic_3d_chat/evaluation/"
        "gemma4_tool_decoder_evaluation_preregistration_v2_1.py"
    ),
    (
        "src/semantic_3d_chat/evaluation/"
        "gemma4_tool_decoder_training_authorization_v2_2.py"
    ),
    "src/semantic_3d_chat/robot/gemma4_tool_decoder_v2_backend.py",
    "src/semantic_3d_chat/robot/gemma4_tool_decoder_v2_runtime_probe.py",
    "scripts/materialize_gemma4_tool_decoder_v2_clearance.py",
    "scripts/run_gemma4_tool_decoder_v2_preflight.py",
    "scripts/preregister_gemma4_tool_decoder_v2_1_evaluation.py",
    "scripts/authorize_gemma4_tool_decoder_v2_2_cpu.py",
    "scripts/release_gemma4_tool_decoder_v2_2_mps_smoke.py",
    "scripts/run_gemma4_tool_decoder_v2_2_mps_smoke.py",
    "scripts/release_gemma4_tool_decoder_v2_2_training.py",
    "scripts/run_gemma4_tool_decoder_v2_2_training.py",
    "tests/test_gemma4_tool_decoder_v2.py",
    "tests/test_gemma4_tool_decoder_v2_pipeline.py",
    "tests/test_gemma4_tool_decoder_v2_checkpoint.py",
    "tests/test_gemma4_tool_decoder_v2_authorization.py",
    "tests/test_gemma4_tool_decoder_v2_runtime_probe.py",
)
AUTHORIZATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "authorization_stage",
        "parent_authorization_path",
        "parent_authorization_sha256",
        "supersedes_authorization_path",
        "supersedes_authorization_sha256",
        "training_source_path",
        "training_source_sha256",
        "bound_source_sha256",
        "v1_terminal_failure_sha256",
        "v2_preregistration_sha256",
        "v2_cpu_preflight_sha256",
        "v2_1_evaluation_preregistration_sha256",
        "clearance_cache_sha256",
        "clearance_manifest_sha256",
        "trace_rows_sha256",
        "prefix_inventory_sha256",
        "cpu_preparation_authorized",
        "full_model_mps_microbatch_authorized",
        "multi_update_training_authorized",
        "parent_heavy_mps_release",
        "full_model_mps_microbatch_smoke_path",
        "full_model_mps_microbatch_smoke_sha256",
        "full_model_mps_microbatch_smoke",
        "resource_contract",
        "execution",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def source_hashes_v2_2() -> dict[str, str]:
    return {path: sha256_file(PROJECT_ROOT / path) for path in BOUND_SOURCE_PATHS}


def _strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate V2.2 authorization field: {key}")
            output[key] = value
        return output

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != set(AUTHORIZATION_FIELDS):
        raise ValueError("V2.2 authorization field inventory changed")
    return value


def _strict_report_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate V2.2 smoke report field: {key}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite V2.2 smoke report constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError("V2.2 MPS smoke report must be a mapping")
    return value


def load_authorization_payload_v2_2(path: str | Path) -> dict[str, Any]:
    return _strict_json(_rooted(path))


def _base_payload() -> dict[str, Any]:
    superseded = PROJECT_ROOT / SUPERSEDED_AUTHORIZATION_PATH
    if sha256_file(superseded) != SUPERSEDED_AUTHORIZATION_SHA256:
        raise ValueError("Superseded V2.1 CPU authorization evidence changed")
    return {
        "schema_version": "2.2",
        "artifact": ARTIFACT,
        "supersedes_authorization_path": SUPERSEDED_AUTHORIZATION_PATH,
        "supersedes_authorization_sha256": SUPERSEDED_AUTHORIZATION_SHA256,
        "training_source_path": (
            "src/semantic_3d_chat/training/train_gemma4_tool_decoder_v2.py"
        ),
        "training_source_sha256": sha256_file(
            PROJECT_ROOT
            / "src/semantic_3d_chat/training/train_gemma4_tool_decoder_v2.py"
        ),
        "bound_source_sha256": source_hashes_v2_2(),
        "v1_terminal_failure_sha256": (
            "83939de71e31310b7d523e78c29d3e29add86e2c3dfe916e089b19dfb06decaa"
        ),
        "v2_preregistration_sha256": (
            "0e1e41a6af2830f9b36a8711fb0649246e96254a88cdcc76b97dcb06ee3f82f4"
        ),
        "v2_cpu_preflight_sha256": (
            "412f1d8bb9804b2d38b0335c985225c9cf1e4226758858cee18d906dc5f742e7"
        ),
        "v2_1_evaluation_preregistration_sha256": EVALUATION_V2_1_SHA256,
        "clearance_cache_sha256": (
            "658822707389e67481fa59b035a7e7f19c360487b19d3157b80bc23ede1db048"
        ),
        "clearance_manifest_sha256": (
            "51cf6c0b155e149627f300c17d39369f91f14e415099fe10d9de1682ef8c7e24"
        ),
        "trace_rows_sha256": (
            "72434178ff1cf23c2dfeb98d52cb7b4c443fcc8715c1dd4ee883d87ae127e7ad"
        ),
        "prefix_inventory_sha256": (
            "c477fd12bc4104f147f73c2f6d46904e0b83b3c584206cb227fd70e9371d0d63"
        ),
        "resource_contract": {
            "answer_tail_only_training_and_evaluation": True,
            "full_vs_tail_real_equivalence_required_before_optimizer": True,
            "full_vs_tail_nll_tolerance": 1e-6,
            "training_microbatches": 512,
            "optimizer_updates": 64,
            "checkpoint_selection": "fixed_final_update_64_no_posthoc_selection",
            "teacher_forced_unique_evaluation_forwards": 5852,
            "greedy_unique_sequences": 896,
            "greedy_unique_sequence_maximum": 1024,
            "greedy_maximum_new_tokens": 24,
            "default_training_runner_bound": True,
            "saved_checkpoint_strict_reload_in_resident_model": True,
            "real_decoder_validator_simulator_probe_before_promotion": True,
            "probe_sample_id": "g_00004208",
            "probe_scene_split": "validation",
        },
    }


def _execution() -> dict[str, Any]:
    return {
        "full_model_loaded": False,
        "mps_used": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_executed": False,
        "greedy_generations": 0,
        "runtime_checkpoint_published": False,
    }


def build_cpu_authorization_v2_2() -> dict[str, Any]:
    return {
        **_base_payload(),
        "status": "cpu_inputs_sealed_heavy_mps_not_released",
        "authorization_stage": "cpu_preparation",
        "parent_authorization_path": SUPERSEDED_AUTHORIZATION_PATH,
        "parent_authorization_sha256": SUPERSEDED_AUTHORIZATION_SHA256,
        "cpu_preparation_authorized": True,
        "full_model_mps_microbatch_authorized": False,
        "multi_update_training_authorized": False,
        "parent_heavy_mps_release": False,
        "full_model_mps_microbatch_smoke_path": None,
        "full_model_mps_microbatch_smoke_sha256": None,
        "full_model_mps_microbatch_smoke": None,
        "execution": _execution(),
    }


def _authenticate_cpu_parent(path: str | Path) -> tuple[dict[str, Any], str]:
    source = _rooted(path)
    payload = _strict_json(source)
    digest = sha256_file(source)
    if (
        payload != build_cpu_authorization_v2_2()
        or payload.get("authorization_stage") != "cpu_preparation"
        or payload.get("full_model_mps_microbatch_authorized") is not False
        or payload.get("multi_update_training_authorized") is not False
    ):
        raise ValueError("V2.2 CPU parent authorization changed")
    return payload, digest


def build_mps_smoke_release_v2_2(
    cpu_authorization: str | Path = CPU_AUTHORIZATION_PATH,
) -> dict[str, Any]:
    _cpu, cpu_sha = _authenticate_cpu_parent(cpu_authorization)
    return {
        **_base_payload(),
        "status": "heavy_mps_microbatch_released",
        "authorization_stage": "full_model_mps_microbatch",
        "parent_authorization_path": str(Path(cpu_authorization)),
        "parent_authorization_sha256": cpu_sha,
        "cpu_preparation_authorized": True,
        "full_model_mps_microbatch_authorized": True,
        "multi_update_training_authorized": False,
        "parent_heavy_mps_release": True,
        "full_model_mps_microbatch_smoke_path": None,
        "full_model_mps_microbatch_smoke_sha256": None,
        "full_model_mps_microbatch_smoke": None,
        "execution": _execution(),
    }


def validate_mps_smoke_report_v2_2(
    report: Mapping[str, Any], *, release_sha256: str
) -> dict[str, Any]:
    difference = report.get("real_full_vs_tail_answer_nll_absolute_difference")
    gradients = (report.get("lora_gradient_l2"), report.get("projector_gradient_l2"))
    if (
        report.get("schema")
        != "semantic_3d_chat.gemma4_tool_decoder_full_mps_smoke.v2_2"
        or report.get("status") != "passed"
        or report.get("authorization_sha256") != release_sha256
        or report.get("device") != "mps"
        or report.get("microbatches") != 1
        or report.get("optimizer_steps") != 0
        or report.get("training_executed") is not False
        or report.get("checkpoint_published") is not False
        or report.get("full_model_loaded") is not True
        or report.get("mps_used") is not True
        or report.get("training_and_evaluation_use_answer_tail_only") is not True
        or isinstance(difference, bool)
        or not isinstance(difference, (int, float))
        or not math.isfinite(float(difference))
        or not 0.0 <= float(difference) <= 1e-6
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in gradients
        )
    ):
        raise ValueError("V2.2 full-model MPS smoke report did not pass exact gates")
    return {
        "status": "passed",
        "device": "mps",
        "sample_id": report.get("sample_id"),
        "microbatches": 1,
        "optimizer_steps": 0,
        "real_full_vs_tail_answer_nll_absolute_difference": float(difference),
        "lora_gradient_l2": float(gradients[0]),
        "projector_gradient_l2": float(gradients[1]),
        "training_and_evaluation_use_answer_tail_only": True,
        "full_model_loaded": True,
        "mps_used": True,
        "training_executed": False,
        "checkpoint_published": False,
    }


def build_training_release_v2_2(
    *,
    smoke_release: str | Path = MPS_SMOKE_RELEASE_PATH,
    smoke_report: str | Path = MPS_SMOKE_REPORT_PATH,
) -> dict[str, Any]:
    release_path = _rooted(smoke_release)
    release = _strict_json(release_path)
    release_sha = sha256_file(release_path)
    if release != build_mps_smoke_release_v2_2(
        release["parent_authorization_path"]
    ):
        raise ValueError("V2.2 MPS smoke release ancestry or bytes changed")
    report_path = _rooted(smoke_report)
    report_sha = sha256_file(report_path)
    report = _strict_report_json(report_path)
    summary = validate_mps_smoke_report_v2_2(report, release_sha256=release_sha)
    return {
        **_base_payload(),
        "status": "multi_update_training_released",
        "authorization_stage": "multi_update_training",
        "parent_authorization_path": str(Path(smoke_release)),
        "parent_authorization_sha256": release_sha,
        "cpu_preparation_authorized": True,
        "full_model_mps_microbatch_authorized": True,
        "multi_update_training_authorized": True,
        "parent_heavy_mps_release": True,
        "full_model_mps_microbatch_smoke_path": str(Path(smoke_report)),
        "full_model_mps_microbatch_smoke_sha256": report_sha,
        "full_model_mps_microbatch_smoke": summary,
        "execution": _execution(),
    }


def _atomic_create(path: str | Path, payload: Mapping[str, Any]) -> tuple[Path, str]:
    destination = _rooted(path)
    if destination.exists():
        raise FileExistsError(f"V2.2 authorization is create-once: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return destination, sha256_file(destination)


def write_cpu_authorization_v2_2(
    output: str | Path = CPU_AUTHORIZATION_PATH,
) -> tuple[Path, str]:
    return _atomic_create(output, build_cpu_authorization_v2_2())


def write_mps_smoke_release_v2_2(
    output: str | Path = MPS_SMOKE_RELEASE_PATH,
    *,
    cpu_authorization: str | Path = CPU_AUTHORIZATION_PATH,
) -> tuple[Path, str]:
    return _atomic_create(output, build_mps_smoke_release_v2_2(cpu_authorization))


def write_training_release_v2_2(
    output: str | Path = TRAINING_RELEASE_PATH,
    *,
    smoke_release: str | Path = MPS_SMOKE_RELEASE_PATH,
    smoke_report: str | Path = MPS_SMOKE_REPORT_PATH,
) -> tuple[Path, str]:
    return _atomic_create(
        output,
        build_training_release_v2_2(
            smoke_release=smoke_release, smoke_report=smoke_report
        ),
    )


__all__ = [
    "ARTIFACT",
    "AUTHORIZATION_FIELDS",
    "BOUND_SOURCE_PATHS",
    "CPU_AUTHORIZATION_PATH",
    "MPS_SMOKE_RELEASE_PATH",
    "MPS_SMOKE_REPORT_PATH",
    "TRAINING_RELEASE_PATH",
    "build_cpu_authorization_v2_2",
    "build_mps_smoke_release_v2_2",
    "build_training_release_v2_2",
    "load_authorization_payload_v2_2",
    "sha256_file",
    "source_hashes_v2_2",
    "validate_mps_smoke_report_v2_2",
    "write_cpu_authorization_v2_2",
    "write_mps_smoke_release_v2_2",
    "write_training_release_v2_2",
]
