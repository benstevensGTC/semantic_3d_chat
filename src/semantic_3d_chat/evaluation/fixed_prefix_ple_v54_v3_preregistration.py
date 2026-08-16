"""Mechanical V3 amendment for tuple-key diagnostic hashing in PLE-V54.

V2 completed its frozen baseline forwards and then aborted before optimizer
construction because a diagnostics-only dictionary used tuple keys with
``json.dumps``.  V3 changes only that hash serialization to sorted explicit
``{"key": [...], "value": ...}`` records.  It inherits every model, data,
training, gating, leakage, and publication contract from V2.
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
from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v2_preregistration import (
    build_preregistration as build_v2_preregistration,
)
from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v2_preregistration import (
    v2_implementation_hashes,
)

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_ple_reader_v3"
PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_preregistration.json"
)
SMOKE_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_smoke.json"
)
RESULT_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_result.json"
)
OUTPUT_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v3"
)
V2_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_preregistration.json"
)
V2_PREREGISTRATION_SHA256: Final[str] = (
    "b82163a0e3fe030f84403e10822944a52a8c3c99ef215d7a67a90dcb88d6d8fd"
)
V2_SMOKE: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_smoke.json"
)
V2_SMOKE_SHA256: Final[str] = (
    "f7daaf2df2f052d0dad45fdf5cacff3c21652c99b953adc12f0361072ba189f0"
)
V2_ABORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_abort.json"
)
V2_ABORT_SHA256: Final[str] = (
    "97ec2de77484fcf5b014478c8701ed20b5ab8b0e7f394cdd2158a452c2005210"
)
V2_SOURCE_HASHES: Final[dict[str, str]] = {
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v2_preregistration.py": (
        "9d026b5d41848809ab223d064a1b212ca99744453040d584e42674da3b7aa339"
    ),
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v2.py": (
        "95728dc82b0577dae2c6262b53f3267312a487c61bc544688b27e6852cef042f"
    ),
    "scripts/run_gemma4_v54_fixed_prefix_ple_reader_v2.sh": (
        "ecce536d3a886e2d681066e0c895df0d2922ba843354ef07f684e3fdd5b9e1bc"
    ),
    "tests/test_fixed_prefix_ple_v54_v2.py": (
        "fc3cdd24d1e50c0ecc228a52b0abc8e2558c758abed77a2fd87f73890f1c1203"
    ),
}
_V3_IMPLEMENTATION: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v3_preregistration.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v3.py",
    "scripts/run_gemma4_v54_fixed_prefix_ple_reader_v3.sh",
    "tests/test_fixed_prefix_ple_v54_v3.py",
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


def authenticate_v2_abort() -> dict[str, Any]:
    for relative, expected in (
        (V2_PREREGISTRATION, V2_PREREGISTRATION_SHA256),
        (V2_SMOKE, V2_SMOKE_SHA256),
        (V2_ABORT, V2_ABORT_SHA256),
    ):
        if _sha256(relative) != expected:
            raise ValueError(f"PLE-V54 V2 evidence changed: {relative}")
    if v2_implementation_hashes() != V2_SOURCE_HASHES:
        raise ValueError("PLE-V54 V2 implementation sources changed")
    prereg = json.loads(_resolve(V2_PREREGISTRATION).read_text(encoding="utf-8"))
    if prereg != build_v2_preregistration():
        raise ValueError("PLE-V54 V2 preregistration no longer rebuilds exactly")
    abort = json.loads(_resolve(V2_ABORT).read_text(encoding="utf-8"))
    failure = abort.get("failure_scope")
    if (
        abort.get("status") != "aborted_before_optimizer_construction_no_checkpoint"
        or abort.get("checkpoint_absent") is not True
        or abort.get("checkpoint_published") is not False
        or not isinstance(failure, Mapping)
        or failure.get("adapter_update_count") != 0
        or failure.get("optimizer_constructed") is not False
        or failure.get("training_started") is not False
        or failure.get("terminal_result_written") is not False
    ):
        raise ValueError("PLE-V54 V2 is not the authenticated pretraining hash abort")
    if _resolve("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_result.json").exists():
        raise ValueError("PLE-V54 V2 terminal result unexpectedly exists")
    if _resolve("data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v2").exists():
        raise ValueError("PLE-V54 V2 checkpoint unexpectedly exists")
    return abort


def v3_implementation_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _V3_IMPLEMENTATION:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"PLE-V54 V3 implementation missing: {relative}")
        result[relative] = _sha256(relative)
    return result


def build_preregistration() -> dict[str, Any]:
    abort = authenticate_v2_abort()
    v2 = build_v2_preregistration()
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "status": "locked_before_single_v3_training_run",
        "v2_abort": {
            "preregistration_path": V2_PREREGISTRATION,
            "preregistration_sha256": V2_PREREGISTRATION_SHA256,
            "smoke_path": V2_SMOKE,
            "smoke_sha256": V2_SMOKE_SHA256,
            "abort_path": V2_ABORT,
            "abort_sha256": V2_ABORT_SHA256,
            "status": abort["status"],
            "adapter_update_count": 0,
            "optimizer_constructed": False,
            "checkpoint_absent": True,
        },
        "only_change": {
            "field": "diagnostic_hash_serialization.for_tuple_keyed_mappings",
            "v2": "json_object_with_tuple_keys_raises_type_error",
            "v3": "sorted_records_with_key_list_and_value",
            "affects_model_forward": False,
            "affects_loss": False,
            "affects_gradient": False,
            "affects_optimizer": False,
            "affects_gate_values": False,
        },
        "unchanged_v2_contract": v2,
        "v2_source_hashes": V2_SOURCE_HASHES,
        "v3_implementation_source_hashes": v3_implementation_hashes(),
        "smoke_inheritance": {
            "source": V2_SMOKE,
            "source_sha256": V2_SMOKE_SHA256,
            "new_model_forward_required": False,
            "reason": "V3 changes only post-forward diagnostic key serialization",
        },
        "output_paths": {
            "smoke_attestation": SMOKE_REPORT,
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
        raise FileExistsError(f"PLE-V54 V3 preregistration exists: {destination}")
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
        raise FileNotFoundError("PLE-V54 V3 preregistration is missing or unsafe")
    observed = json.loads(source.read_text(encoding="utf-8"))
    if observed != build_preregistration():
        raise ValueError("PLE-V54 V3 preregistration differs from pinned sources")
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
