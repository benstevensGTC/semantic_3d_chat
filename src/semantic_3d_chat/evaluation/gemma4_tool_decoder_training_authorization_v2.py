"""Create-once CPU preparation authorization for Gemma tool-decoder V2.1."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

DEFAULT_OUTPUT: Final[Path] = Path(
    "reports/gemma4/metrics/"
    "gemma4_embodied_tool_decoder_training_authorization_v2.json"
)
EVALUATION_V2_1_SHA256: Final[str] = (
    "7b322d57ed46d920f7253383e75254b4157ac5397afc172bf7ffd0141310e007"
)
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
        "gemma4_tool_decoder_training_authorization_v2.py"
    ),
    "src/semantic_3d_chat/robot/gemma4_tool_decoder_v2_backend.py",
    "scripts/materialize_gemma4_tool_decoder_v2_clearance.py",
    "scripts/run_gemma4_tool_decoder_v2_preflight.py",
    "scripts/preregister_gemma4_tool_decoder_v2_1_evaluation.py",
    "tests/test_gemma4_tool_decoder_v2.py",
    "tests/test_gemma4_tool_decoder_v2_pipeline.py",
    "tests/test_gemma4_tool_decoder_v2_checkpoint.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_authorization_v2() -> dict[str, Any]:
    evaluation = (
        PROJECT_ROOT
        / "reports/gemma4/metrics/"
        "gemma4_embodied_tool_decoder_evaluation_preregistration_v2_1.json"
    )
    if _sha256(evaluation) != EVALUATION_V2_1_SHA256:
        raise ValueError("V2.1 evaluation preregistration changed")
    return {
        "schema_version": 2,
        "artifact": "gemma4_embodied_tool_decoder_training_authorization_v2",
        "status": "cpu_inputs_sealed_heavy_mps_not_released",
        "training_source_path": (
            "src/semantic_3d_chat/training/train_gemma4_tool_decoder_v2.py"
        ),
        "training_source_sha256": _sha256(
            PROJECT_ROOT
            / "src/semantic_3d_chat/training/train_gemma4_tool_decoder_v2.py"
        ),
        "bound_source_sha256": {
            path: _sha256(PROJECT_ROOT / path) for path in BOUND_SOURCE_PATHS
        },
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
        "cpu_preparation_authorized": True,
        "full_model_mps_microbatch_authorized": False,
        "multi_update_training_authorized": False,
        "parent_heavy_mps_release": False,
        "full_model_mps_microbatch_smoke": None,
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
        },
        "execution": {
            "full_model_loaded": False,
            "mps_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_executed": False,
            "greedy_generations": 0,
            "runtime_checkpoint_published": False,
        },
    }


def write_training_authorization_v2(
    output: str | Path = DEFAULT_OUTPUT,
) -> tuple[Path, str]:
    destination = Path(output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError("V2 training authorization is create-once")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                build_training_authorization_v2(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return destination, _sha256(destination)


__all__ = [
    "BOUND_SOURCE_PATHS",
    "DEFAULT_OUTPUT",
    "EVALUATION_V2_1_SHA256",
    "build_training_authorization_v2",
    "write_training_authorization_v2",
]
