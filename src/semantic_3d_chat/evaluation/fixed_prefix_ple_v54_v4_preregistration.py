"""Resource-only V4 amendment for streamed answer-tail NLL in PLE-V54.

V3 aborted before optimizer construction because it materialized vocabulary
logits at every prefix/prompt position for a two-row validation microbatch,
although the loss supervises only the answer suffix.  V4 keeps the exact causal
answer-token-normalized NLL while forwarding one example at a time and asking
Gemma for only ``answer_token_count + 1`` tail logits.  No data, model, prefix,
seed, schedule, objective, optimizer, threshold, or runtime contract changes.
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
from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v3_preregistration import (
    build_preregistration as build_v3_preregistration,
)
from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v3_preregistration import (
    v3_implementation_hashes,
)

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_ple_reader_v4"
PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_preregistration.json"
)
SMOKE_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_smoke.json"
)
RESULT_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_result.json"
)
OUTPUT_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v4"
)
V3_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_preregistration.json"
)
V3_PREREGISTRATION_SHA256: Final[str] = (
    "eff55d288be9bb6337e2a9d9a086359aba7c9c181b105d7188dfa6dbefcea614"
)
V3_SMOKE: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_smoke.json"
)
V3_SMOKE_SHA256: Final[str] = (
    "3f2fbb71e7fa69491d606b19984d5207ba5642945bb26e4876b94ead350d12e9"
)
V3_ABORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_abort.json"
)
V3_ABORT_SHA256: Final[str] = (
    "12461df8dffa9304646a97c33dcad1855496fa65cf9d9d0050663362fc01dcdd"
)
V3_SOURCE_HASHES: Final[dict[str, str]] = {
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v3_preregistration.py": (
        "682e4ae3cac8d9881c94e8e56d7763ccc350aee8941c25f0fd22bbbd5faec71b"
    ),
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v3.py": (
        "23fe6025a3a479aad06b6e99957f59e4b2fd72ef80f87422c4a786dfc9a88770"
    ),
    "scripts/run_gemma4_v54_fixed_prefix_ple_reader_v3.sh": (
        "b0673af6d284b8ccc227c85b4824f1a78920fc30f1f224d19a7cf9b13e47bb69"
    ),
    "tests/test_fixed_prefix_ple_v54_v3.py": (
        "37c7d45b8c0a7bfc40497100e18990bcd4ec0585f577efb9f09b26a2de973385"
    ),
}
_V4_IMPLEMENTATION: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v4_preregistration.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v4.py",
    "scripts/run_gemma4_v54_fixed_prefix_ple_reader_v4.sh",
    "tests/test_fixed_prefix_ple_v54_v4.py",
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


def authenticate_v3_abort() -> dict[str, Any]:
    for relative, expected in (
        (V3_PREREGISTRATION, V3_PREREGISTRATION_SHA256),
        (V3_SMOKE, V3_SMOKE_SHA256),
        (V3_ABORT, V3_ABORT_SHA256),
    ):
        if _sha256(relative) != expected:
            raise ValueError(f"PLE-V54 V3 evidence changed: {relative}")
    if v3_implementation_hashes() != V3_SOURCE_HASHES:
        raise ValueError("PLE-V54 V3 implementation sources changed")
    prereg = json.loads(_resolve(V3_PREREGISTRATION).read_text(encoding="utf-8"))
    if prereg != build_v3_preregistration():
        raise ValueError("PLE-V54 V3 preregistration no longer rebuilds exactly")
    abort = json.loads(_resolve(V3_ABORT).read_text(encoding="utf-8"))
    failure = abort.get("failure_scope")
    memory = abort.get("error", {}).get("memory")
    if (
        abort.get("status")
        != "aborted_before_optimizer_construction_mps_oom_no_checkpoint"
        or abort.get("checkpoint_absent") is not True
        or abort.get("checkpoint_published") is not False
        or not isinstance(failure, Mapping)
        or failure.get("adapter_update_count") != 0
        or failure.get("optimizer_constructed") is not False
        or failure.get("training_started") is not False
        or failure.get("terminal_result_written") is not False
        or not isinstance(memory, Mapping)
        or memory.get("mps_allocated_gib") != 20.32
        or memory.get("other_allocations_gib") != 9.51
        or memory.get("attempted_allocation_mib") != 642.0
    ):
        raise ValueError("PLE-V54 V3 is not the authenticated zero-update MPS OOM")
    if _resolve("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_result.json").exists():
        raise ValueError("PLE-V54 V3 terminal result unexpectedly exists")
    if _resolve("data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v3").exists():
        raise ValueError("PLE-V54 V3 checkpoint unexpectedly exists")
    return abort


def v4_implementation_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _V4_IMPLEMENTATION:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"PLE-V54 V4 implementation missing: {relative}")
        result[relative] = _sha256(relative)
    return result


def build_preregistration() -> dict[str, Any]:
    abort = authenticate_v3_abort()
    v3 = build_v3_preregistration()
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "status": "locked_before_v4_equivalence_smoke_and_single_training_run",
        "v3_abort": {
            "preregistration_path": V3_PREREGISTRATION,
            "preregistration_sha256": V3_PREREGISTRATION_SHA256,
            "smoke_path": V3_SMOKE,
            "smoke_sha256": V3_SMOKE_SHA256,
            "abort_path": V3_ABORT,
            "abort_sha256": V3_ABORT_SHA256,
            "status": abort["status"],
            "adapter_update_count": 0,
            "optimizer_constructed": False,
            "checkpoint_absent": True,
            "memory_failure": abort["error"]["memory"],
        },
        "resource_only_changes": {
            "teacher_forcing_microbatch_size": {"v3": 2, "v4": 1},
            "logit_positions": {
                "v3": "all_prefix_prompt_and_answer_positions",
                "v4": "answer_token_count_plus_one_causal_tail_positions_only",
            },
            "examples_streamed_without_batch_padding": True,
            "retention_next_token_logits_to_keep": 1,
            "each_example_token_normalized_before_any_mean": True,
            "same_answer_labels": True,
            "same_fp32_cross_entropy": True,
            "same_per_example_answer_token_normalization": True,
            "same_correct_wrong_prefix_objective": True,
            "same_gradient_accumulation_divisor": True,
        },
        "equivalence_requirements_before_training": {
            "deterministic_synthetic_full_vs_tail_nll_exact": True,
            "real_one_row_frozen_gemma_full_vs_tail_nll_absolute_tolerance": 1e-06,
            "tail_reader_gradient_finite_and_nonzero": True,
            "mps_driver_allocated_bytes_maximum": 25_000_000_000,
        },
        "unchanged_v3_contract": v3,
        "locked_unchanged_training_fields": {
            "data": v3["unchanged_v2_contract"]["unchanged_v1_contract"]["data"],
            "objective": v3["unchanged_v2_contract"]["unchanged_v1_contract"]["objective"],
            "optimization": v3["unchanged_v2_contract"]["unchanged_v1_contract"]["optimization"],
            "selection_except_resource_microbatch": v3["unchanged_v2_contract"][
                "unchanged_v1_contract"
            ]["selection"],
            "runtime_contract": v3["unchanged_v2_contract"]["unchanged_v1_contract"][
                "runtime_contract"
            ],
            "publication": v3["unchanged_v2_contract"]["unchanged_v1_contract"][
                "publication"
            ],
        },
        "v3_source_hashes": V3_SOURCE_HASHES,
        "v4_implementation_source_hashes": v4_implementation_hashes(),
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
        raise FileExistsError(f"PLE-V54 V4 preregistration exists: {destination}")
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
        raise FileNotFoundError("PLE-V54 V4 preregistration is missing or unsafe")
    observed = json.loads(source.read_text(encoding="utf-8"))
    if observed != build_preregistration():
        raise ValueError("PLE-V54 V4 preregistration differs from pinned sources")
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
