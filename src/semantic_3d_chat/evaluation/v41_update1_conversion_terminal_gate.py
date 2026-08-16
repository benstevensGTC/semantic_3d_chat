"""Seal the failed V41 MPS conversion attempt and authorize one isolated retry.

This gate is train-only.  It authenticates the immutable failed V41 root,
proves that update 1 never reached clipping or ``optimizer.step()``, binds the
CPU-first MPS conversion repair, and authorizes only the sibling retry1 root.
It never opens validation, oracle, or final-scene data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT

DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v41_update1_conversion_terminal_gate.json"
)
MATERIALIZED_REPORT_SHA256 = (
    "cefe759791e1d97557f0a230d6605a1ae079d5f32be7fac3e2b806adaf82eef8"
)
REV3_REPORT = Path("reports/gemma4/metrics/v40_update3_terminal_gate.json")
REV3_SHA256 = "d4c30be9e4f685697478b6e5a37f4f55d6e99962484b1cbae5c3c3214c24b35e"
FAILED_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v41_diverse28_projected_gradient_l14_query"
)
RETRY_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v41_retry1_diverse28_projected_gradient_l14_query"
)
FAILED_FILES = {
    "guard_failure_update_001.json": (
        "b416bd598832c7fcd07a1d098e03e29a50648489b486591158867e4dc586c53d"
    ),
    "update_000/adapter.safetensors": (
        "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0"
    ),
    "update_000/metadata.json": (
        "b1e66cec1aba693a3ffa6d5fd91dea78da4eb925db14c37b79171fd5bf94e4d8"
    ),
    "update_000/runtime_metadata.json": (
        "0037ac30ce329dc7041f0b18369ee56b947ff67780421cbaa4d72cbf01a4f1e2"
    ),
}
FAILED_MANIFEST_SHA256 = (
    "44117c29921f3cb2d9a454d8a470d3dd29bfaaf1f90dab42ee4d38a4fa29bac9"
)
AUDITED_MANIFEST_STREAM_SHA256 = (
    "1b2bc5ef099325fa7f09715ad9ec614446bf928ac1f5cbbd6e4c42c4d2accc49"
)
FIXED_TRAINER = Path(
    "src/semantic_3d_chat/training/train_projected_gradient_v41.py"
)
FIXED_TRAINER_SHA256 = (
    "f7f9ce057e90ec063d7b49ecd966222aaeb4179ea66b38db63700d3540c5a6da"
)
FIXED_TRAINING_TEST = Path("tests/test_v41_projected_gradient_training.py")
FIXED_TRAINING_TEST_SHA256 = (
    "ba3b4864e4691ac51383e4934e5b25b7654c1a362debf8ad47bdd117a2841203"
)
PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
PROTECTED_SHA256 = (
    "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
)
SOURCE_TARGET_SHA256 = (
    "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
)
FROZEN_SHA256 = (
    "cec01bc088bb87c6bb44e0659eb03aa766f951ddeee706ca9a70edaa080dea5e"
)


def _resolve(path: Path) -> Path:
    return (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _manifest_sha256(root: Path) -> str:
    rows = [f"{name}\t{FAILED_FILES[name]}" for name in sorted(FAILED_FILES)]
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


def _authenticate_failed_root() -> tuple[dict[str, Any], dict[str, Any]]:
    root = _resolve(FAILED_ROOT)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("V41 failed root is absent or aliased")
    if sorted(path.name for path in root.iterdir()) != [
        "guard_failure_update_001.json",
        "update_000",
    ]:
        raise ValueError("V41 failed-root inventory changed")
    update0 = root / "update_000"
    if update0.is_symlink() or not update0.is_dir():
        raise ValueError("V41 failed update zero is absent or aliased")
    if sorted(path.name for path in update0.iterdir()) != [
        "adapter.safetensors",
        "metadata.json",
        "runtime_metadata.json",
    ]:
        raise ValueError("V41 failed update-zero inventory changed")
    observed: dict[str, str] = {}
    for relative, expected in FAILED_FILES.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"V41 failed artifact is absent or aliased: {relative}")
        observed[relative] = _sha256(path)
        if observed[relative] != expected:
            raise ValueError(f"V41 failed artifact changed: {relative}")
    if _manifest_sha256(root) != FAILED_MANIFEST_SHA256:
        raise ValueError("V41 failed manifest digest changed")

    failure = _mapping(
        json.loads((root / "guard_failure_update_001.json").read_text()),
        "V41 failure",
    )
    audit = _mapping(failure.get("audit"), "V41 failure audit")
    raw = _mapping(
        audit.get("raw_component_gradient_diagnostic"), "raw diagnostic"
    )
    projection = _mapping(
        audit.get("projected_gradient_attestation"), "failed projection"
    )
    raw_finite = _mapping(raw.get("component_finite"), "raw finite flags")
    projected_finite = _mapping(
        projection.get("component_finite"), "projection finite flags"
    )
    if (
        failure.get("artifact") != "v41_pre_step_gradient_guard_failure"
        or failure.get("optimizer_step_not_executed") != 1
        or failure.get("optimizer_step_executed") is not False
        or failure.get("checkpoint_written") is not False
        or failure.get("validation_qa_loaded") is not False
        or failure.get("oracle_environment_files_loaded") is not False
        or audit.get("failed_guard_stage") != "projection_input"
        or audit.get("clip_direction_attestation") is not None
        or audit.get("target_hash_before") != SOURCE_TARGET_SHA256
        or audit.get("target_hash_after") != SOURCE_TARGET_SHA256
        or audit.get("frozen_excluding_b_hash_before") != FROZEN_SHA256
        or audit.get("frozen_excluding_b_hash_after") != FROZEN_SHA256
        or any(value is not True for value in raw_finite.values())
        or any(value is not False for value in projected_finite.values())
    ):
        raise ValueError("V41 failure semantics changed")
    return failure, {
        "root": str(FAILED_ROOT),
        "root_entries": ["guard_failure_update_001.json", "update_000"],
        "file_sha256": observed,
        "manifest_sha256": FAILED_MANIFEST_SHA256,
        "audited_manifest_stream_sha256": AUDITED_MANIFEST_STREAM_SHA256,
        "failed_before_optimizer_step": 1,
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "target_and_frozen_state_unchanged": True,
        "raw_cpu_first_diagnostic_finite": True,
        "combined_mps_to_cpu_float64_projection_nonfinite": True,
    }


def load_materialized_report() -> dict[str, Any]:
    """Load the immutable pre-retry seal after its one-shot successor has run.

    ``build_report`` intentionally refuses once ``RETRY_ROOT`` is populated;
    replaying the already-issued authorization must therefore authenticate the
    materialized seal instead of minting a fresh one.
    """

    path = _resolve(DEFAULT_OUTPUT)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Materialized V41 retry terminal is absent or aliased")
    if _sha256(path) != MATERIALIZED_REPORT_SHA256:
        raise ValueError("Materialized V41 retry terminal changed")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), "V41 retry terminal")


def build_report() -> dict[str, Any]:
    rev3 = _resolve(REV3_REPORT)
    if rev3.is_symlink() or not rev3.is_file() or _sha256(rev3) != REV3_SHA256:
        raise ValueError("V41 retry requires the exact revision-3 terminal seal")
    failure, predecessor = _authenticate_failed_root()
    fixed_files = {
        str(FIXED_TRAINER): _sha256(_resolve(FIXED_TRAINER)),
        str(FIXED_TRAINING_TEST): _sha256(_resolve(FIXED_TRAINING_TEST)),
    }
    if fixed_files != {
        str(FIXED_TRAINER): FIXED_TRAINER_SHA256,
        str(FIXED_TRAINING_TEST): FIXED_TRAINING_TEST_SHA256,
    }:
        raise ValueError("V41 CPU-first repair source or regression changed")
    protected = _resolve(PROTECTED_REPORT)
    if protected.is_symlink() or not protected.is_file() or _sha256(protected) != PROTECTED_SHA256:
        raise ValueError("Protected training-selection artifact changed")
    retry_root = _resolve(RETRY_ROOT)
    if retry_root.is_symlink() or (retry_root.exists() and any(retry_root.iterdir())):
        raise ValueError("V41 retry1 output is aliased or nonempty")

    authorization = {
        "authorization_revision": 1,
        "authorization_id": "v41_retry1_cpu_first_projected_gradient_l14_lora_b",
        "authorized": True,
        "authorized_output_root": str(RETRY_ROOT),
        "predecessor_guard_failure_sha256": FAILED_FILES[
            "guard_failure_update_001.json"
        ],
        "predecessor_manifest_sha256": FAILED_MANIFEST_SHA256,
        "cpu_first_mps_conversion_required": True,
        "conversion_protocol": (
            "detach_then_cpu_then_float64; never combine MPS transfer and float64 cast"
        ),
        "fixed_repair_file_sha256": fixed_files,
        "mps_regression": {
            "full_suite_collected_and_passed": 19,
            "non_source_retry_regression_subset_passed": 18,
            "raw_feasible_mask_zero_bit_exact_tested": True,
            "conflicting_nonzero_mask_cpu_projection_mps_cast_clip_tested": True,
            "live_shape_4096_by_4_tested": True,
        },
        "source": {
            "checkpoint": (
                "data_gemma4/checkpoints/"
                "gemma4_v40_diverse28_cross_preserving_l14_query/update_000"
            ),
            "optimizer_step": 0,
            "target_lora_b_state_sha256": SOURCE_TARGET_SHA256,
            "frozen_excluding_target_state_sha256": FROZEN_SHA256,
            "source_optimizer_access_authorized": False,
        },
        "experiment_unchanged_from_revision3": True,
        "objective_schedule_gates_unchanged": True,
        "training_qa_and_maps_only": True,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "chat_or_runtime_promotion_authorized": False,
        "new_post_training_terminal_required": True,
    }
    return {
        "schema_version": 1,
        "artifact": "v41_update1_conversion_terminal_gate",
        "seal_revision": 4,
        "passed": True,
        "failure_kind": "mps_combined_device_and_float64_conversion",
        "predecessor_failure": predecessor,
        "failure_payload_replayed": failure,
        "conditional_successor_authorization": authorization,
        "only_exact_successor_authorized": (
            "v41_retry1_train_only_projected_gradient_continuation"
        ),
        "v41_retry1_train_only_projected_gradient_continuation_authorized": True,
        "arbitrary_training_authorized": False,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "chat_or_runtime_promotion_authorized": False,
        "protected_artifact_sha256": PROTECTED_SHA256,
    }


def write_report(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = build_report()
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = write_report(args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
