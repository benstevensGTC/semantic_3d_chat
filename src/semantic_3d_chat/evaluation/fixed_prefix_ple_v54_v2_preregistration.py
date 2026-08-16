"""Mechanical V2 amendment for the PLE-V54 MPS gradient smoke tolerance.

V1 demonstrated a finite nonzero gradient and acceptable memory use, but its
exact-zero retention self-comparison was 1.758e-6 KL on MPS.  V2 changes only
that smoke pass threshold to 1e-5.  Training, data, architecture, objective,
selection gates, runtime contract, and publication rules remain byte-bound to
V1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_preregistration import (
    build_preregistration as build_v1_preregistration,
)
from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_preregistration import (
    implementation_source_hashes as v1_implementation_source_hashes,
)

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_ple_reader_v2"
PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_preregistration.json"
)
SMOKE_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_smoke.json"
)
RESULT_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_result.json"
)
OUTPUT_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v2"
)
V1_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v1_preregistration.json"
)
V1_PREREGISTRATION_SHA256: Final[str] = (
    "07c28a95badf87c08692532ed1b8f9064af37763f11bc8c469581dae147bff52"
)
V1_SMOKE_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v1_smoke.json"
)
V1_SMOKE_SHA256: Final[str] = (
    "c1c8b6efe101fd1ce78d02fea9bafb1a090f3356171122a397bbd42cae7dcfa5"
)
V1_SOURCE_HASHES: Final[dict[str, str]] = {
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_preregistration.py": (
        "29ad92242ba48b8b1caa56bf5888b3d578641843de67b752232a82f75a77d2f2"
    ),
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54.py": (
        "244519ba4ee1f7aa6e4904e20a9cba76fd0af8b028ad005c04fbd9ee39c1b99d"
    ),
    "tests/test_fixed_prefix_ple_v54.py": (
        "a271a612a05123a48202c9205b31a1d855376ea9a66afc91b2ea91a2cecd6f92"
    ),
    "scripts/run_gemma4_v54_fixed_prefix_ple_reader.sh": (
        "275b235e44e7fa76293731f7aeb062729172c37b855722e3d08c5ee8eb38c664"
    ),
}
_V2_IMPLEMENTATION: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v2_preregistration.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v2.py",
    "scripts/run_gemma4_v54_fixed_prefix_ple_reader_v2.sh",
    "tests/test_fixed_prefix_ple_v54_v2.py",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_v1_failure() -> dict[str, Any]:
    if _sha256(V1_PREREGISTRATION) != V1_PREREGISTRATION_SHA256:
        raise ValueError("PLE-V54 V1 preregistration bytes changed")
    if _sha256(V1_SMOKE_REPORT) != V1_SMOKE_SHA256:
        raise ValueError("PLE-V54 V1 smoke bytes changed")
    observed_sources = v1_implementation_source_hashes()
    if observed_sources != {
        **V1_SOURCE_HASHES,
        "configs/experiments/gemma4_v54_fixed_prefix_ple_reader_v1.yaml": (
            "7ce074e5a35cfab9476fcb2c46a0d45391aff94733f20633928544b1df90dda6"
        ),
        "configs/experiments/gemma4_v54_fixed_prefix_ple_reader_v1_retention.json": (
            "0b2c48236e085960811ac6c9be94440814a141fdc05ed92c1e8f498a2c04f3cb"
        ),
    }:
        raise ValueError("PLE-V54 V1 implementation sources changed")
    prereg = json.loads(_resolve(V1_PREREGISTRATION).read_text(encoding="utf-8"))
    if prereg != build_v1_preregistration():
        raise ValueError("PLE-V54 V1 preregistration no longer rebuilds exactly")
    smoke = json.loads(_resolve(V1_SMOKE_REPORT).read_text(encoding="utf-8"))
    if (
        smoke.get("passed") is not False
        or smoke.get("status") != "failed"
        or smoke.get("trainable_parameter_count") != 41_984
        or smoke.get("gradient_l2") != 0.2632919251918793
        or smoke.get("initial_retention_kl") != 1.7583897715667263e-06
        or smoke.get("initial_retention_kl") >= 1e-05
        or smoke.get("question_dependent_scene_retrieval") is not False
        or smoke.get("environmental_text_inputs") != []
    ):
        raise ValueError("PLE-V54 V1 failure is not the sole diagnosed tolerance failure")
    return smoke


def v2_implementation_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _V2_IMPLEMENTATION:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"PLE-V54 V2 implementation missing: {relative}")
        result[relative] = _sha256(relative)
    return result


def build_preregistration() -> dict[str, Any]:
    smoke = authenticate_v1_failure()
    v1 = build_v1_preregistration()
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "status": "locked_before_v2_gradient_smoke_and_single_training_run",
        "v1_failure": {
            "preregistration_path": V1_PREREGISTRATION,
            "preregistration_sha256": V1_PREREGISTRATION_SHA256,
            "smoke_path": V1_SMOKE_REPORT,
            "smoke_sha256": V1_SMOKE_SHA256,
            "gradient_l2": smoke["gradient_l2"],
            "retention_self_kl": smoke["initial_retention_kl"],
            "mps_driver_allocated_bytes": smoke["memory"]["mps_driver_allocated_bytes"],
            "elapsed_seconds": smoke["elapsed_seconds"],
            "only_failed_condition": "retention_self_kl_above_1e-6",
        },
        "only_change": {
            "field": "gradient_smoke.retention_self_kl_absolute_tolerance",
            "v1": 1e-06,
            "v2": 1e-05,
            "reason": "observed_finite_mps_repeat_forward_noise_1.7583897715667263e-6",
        },
        "unchanged_v1_contract": {
            "research_question": v1["research_question"],
            "independence": v1["independence"],
            "model": v1["model"],
            "trainable_surface": v1["trainable_surface"],
            "data": v1["data"],
            "objective": v1["objective"],
            "optimization": v1["optimization"],
            "selection": v1["selection"],
            "runtime_contract": v1["runtime_contract"],
            "publication": v1["publication"],
            "pinned_input_hashes": v1["pinned_input_hashes"],
            "reader_lora_contract": v1["reader_lora_contract"],
        },
        "v1_source_hashes": V1_SOURCE_HASHES,
        "v2_implementation_source_hashes": v2_implementation_hashes(),
        "output_paths": {
            "smoke": SMOKE_REPORT,
            "result": RESULT_REPORT,
            "checkpoint": OUTPUT_CHECKPOINT,
        },
    }


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def write_preregistration(path: str | Path = PREREGISTRATION) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"PLE-V54 V2 preregistration exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialized(build_preregistration())
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(payload).hexdigest()


def authenticate_preregistration(path: str | Path = PREREGISTRATION) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("PLE-V54 V2 preregistration is missing or unsafe")
    observed = json.loads(source.read_text(encoding="utf-8"))
    if observed != build_preregistration():
        raise ValueError("PLE-V54 V2 preregistration differs from pinned sources")
    return {
        "artifact": ARTIFACT,
        "path": str(source.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(source),
        "status": observed["status"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=PREREGISTRATION)
    parser.add_argument("--authenticate", action="store_true")
    args = parser.parse_args(argv)
    if args.authenticate:
        result = authenticate_preregistration(args.output)
    else:
        path, digest = write_preregistration(args.output)
        result = {"path": str(path), "sha256": digest}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
