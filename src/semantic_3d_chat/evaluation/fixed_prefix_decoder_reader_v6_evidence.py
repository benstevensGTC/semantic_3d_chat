"""Authenticate the immutable V6 decoder-reader smoke failure without a model.

The authenticator opens exactly four allowlisted JSON artifacts.  It does not
import the sealed V6 implementation, load model weights, inspect QA/oracle data,
or touch the V6.1 successor.  A terminal failure is evidence, not promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

PREREGISTRATION: Final[Path] = Path(
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_preregistration.json"
)
SMOKE_RELEASE: Final[Path] = Path(
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke_release.json"
)
SMOKE_ATTEMPT: Final[Path] = Path(
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke_attempt.json"
)
TERMINAL_SMOKE: Final[Path] = Path(
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke.json"
)
EVIDENCE_SHA256: Final[dict[Path, str]] = {
    PREREGISTRATION: "4f0e3b0da793fc0f6baeb5d032579a50d81e05c8014d738ddfb05756ec0a90cf",
    SMOKE_RELEASE: "65af03ab1259201d824fbd44e1ce3e69def10a47007a14c5f5151e772257e6c3",
    SMOKE_ATTEMPT: "c4d08911a69db7d0f97b7ca53def5b023a771045941a076754ff53affefa1c15",
    TERMINAL_SMOKE: "a78e38e9e5112f757927a9590cecb854c9c99f7881d929b531b83b9db305f2fa",
}
TRAINING_RELEASE: Final[Path] = Path(
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_training_release.json"
)
TRAINING_RESULT: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_result.json"
)
CHECKPOINT: Final[Path] = Path(
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_decoder_reader_v6"
)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads(_resolve(root, path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V6 evidence must be a JSON object: {path}")
    return value


def _inventory_sha256(paths: list[str]) -> str:
    payload = json.dumps(
        sorted(set(paths)), separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def authenticate_v6_terminal_smoke_failure(
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Authenticate one consumed smoke attempt that failed before gradients."""

    resolved_root = Path(root).expanduser().resolve()
    reports: dict[Path, dict[str, Any]] = {}
    observed_sha256: dict[str, str] = {}
    for path, expected in EVIDENCE_SHA256.items():
        resolved = _resolve(resolved_root, path)
        if not resolved.is_file() or resolved.is_symlink():
            raise FileNotFoundError(f"V6 evidence is absent or linked: {path}")
        observed = _sha256(resolved)
        if observed != expected:
            raise ValueError(f"V6 evidence digest changed: {path}: {observed} != {expected}")
        observed_sha256[path.as_posix()] = observed
        reports[path] = _read(resolved_root, path)

    preregistration = reports[PREREGISTRATION]
    release = reports[SMOKE_RELEASE]
    attempt = reports[SMOKE_ATTEMPT]
    terminal = reports[TERMINAL_SMOKE]
    absent_outputs = {
        TRAINING_RELEASE.as_posix(): not _resolve(resolved_root, TRAINING_RELEASE).exists(),
        TRAINING_RESULT.as_posix(): not _resolve(resolved_root, TRAINING_RESULT).exists(),
        CHECKPOINT.as_posix(): not _resolve(resolved_root, CHECKPOINT).exists(),
    }

    expected_preregistration_authorization = {
        "sealed": True,
        "full_model_mps_smoke_authorized": False,
        "joint_runtime_smoke_authorized": False,
        "optimizer_construction_authorized": False,
        "multi_update_training_authorized": False,
        "checkpoint_write_authorized": False,
    }
    independent_audit = preregistration.get("independent_audit")
    preregistration_valid = (
        preregistration.get("schema_version") == 1
        and preregistration.get("artifact")
        == "gemma4_v54_fixed_prefix_decoder_reader_v6_sealed_preregistration"
        and preregistration.get("status")
        == "sealed_before_real_mps_smokes_training_not_authorized"
        and preregistration.get("authorization") == expected_preregistration_authorization
        and preregistration.get("required_next_stage")
        == "separate_create_once_mps_smoke_release"
        and isinstance(independent_audit, dict)
        and independent_audit
        == {
            "status": "passed",
            "full_model_blob_independently_streamed": True,
            "all_40_prefix_files_authenticated": True,
            "train_rows": 576,
            "validation_rows": 384,
            "train_answer_varying_rows": 288,
            "validation_answer_varying_rows": 170,
            "schedule_updates": 96,
            "deferred_or_final_qa_accessed": False,
        }
    )

    source_hashes = release.get("bound_source_sha256")
    historical_source_inventory_valid = (
        isinstance(source_hashes, dict)
        and len(source_hashes) == 77
        and source_hashes.get(
            "src/semantic_3d_chat/training/smoke_fixed_prefix_decoder_reader_v6.py"
        )
        == "f0d1b8a8c6e85b92acacb197e779357b7130e89c324b0ffc4f3ae7d96746e80a"
        and all(
            isinstance(path, str)
            and path
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for path, digest in source_hashes.items()
        )
    )
    release_authorization = release.get("authorized")
    release_valid = (
        release.get("schema_version") == 1
        and release.get("artifact")
        == "gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke_release"
        and release.get("status")
        == "released_exactly_one_zero_update_full_model_mps_smoke"
        and release.get("parent_preregistration") == PREREGISTRATION.as_posix()
        and release.get("parent_preregistration_sha256")
        == EVIDENCE_SHA256[PREREGISTRATION]
        and release.get("parent_status")
        == "sealed_before_real_mps_smokes_training_not_authorized"
        and release.get("attempt_journal") == SMOKE_ATTEMPT.as_posix()
        and release.get("terminal_output") == TERMINAL_SMOKE.as_posix()
        and release_authorization
        == {
            "checkpoint_write": False,
            "deferred_or_final_qa_access": False,
            "full_model_joint_v6_tool_v2_zero_output_structural_coexistence": True,
            "full_model_mps_answer_tail_equivalence": True,
            "full_model_mps_v6_gradient": True,
            "joint_nonzero_semantic_or_tool_behavior": False,
            "maximum_smoke_runs": 1,
            "multi_update_training": False,
            "optimizer_construction": False,
            "optimizer_steps": 0,
        }
    )
    attempt_valid = (
        attempt
        == {
            "schema_version": 1,
            "artifact": "gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke_attempt",
            "status": "claimed_before_model_loading",
            "authorization": SMOKE_RELEASE.as_posix(),
            "authorization_sha256": EVIDENCE_SHA256[SMOKE_RELEASE],
            "maximum_attempts": 1,
            "optimizer_construction_authorized": False,
            "optimizer_steps_authorized": 0,
            "checkpoint_write_authorized": False,
        }
    )

    loaded_files = terminal.get("loaded_files")
    audit_valid = (
        terminal.get("file_access_audit_active_for_entire_execution") is True
        and terminal.get("forbidden_file_accesses") == []
        and terminal.get("deferred_or_final_qa_accessed") is False
        and isinstance(loaded_files, list)
        and len(loaded_files) == 233
        and len(set(loaded_files)) == 233
        and all(isinstance(path, str) and Path(path).is_absolute() for path in loaded_files)
        and terminal.get("loaded_file_count") == len(loaded_files)
        and terminal.get("loaded_file_inventory_sha256")
        == _inventory_sha256(loaded_files)
        == "10f99a6be87cdd3ea1807c3a3fb254f933a1f997f4e83062b45a4c26d34c4c52"
    )
    forbidden_post_equivalence_fields = {
        "answer_tail_equivalence_passed",
        "v6_zero_output_exact_noop",
        "v6_gradient_l2",
        "v6_gradient_by_module",
        "contrastive_correct_nll",
        "contrastive_wrong_nll",
        "broad_nll",
        "retention_self_kl",
        "joint_zero_output_structural_runtime_coexistence_passed",
    }
    terminal_valid = (
        terminal.get("schema_version") == 1
        and terminal.get("artifact")
        == "gemma4_v54_fixed_prefix_decoder_reader_v6_real_mps_smoke"
        and terminal.get("status") == "failed_terminal_attempt_consumed"
        and terminal.get("passed") is False
        and terminal.get("authorization_sha256") == EVIDENCE_SHA256[SMOKE_RELEASE]
        and terminal.get("attempt_sha256") == EVIDENCE_SHA256[SMOKE_ATTEMPT]
        and terminal.get("failure_type") == "RuntimeError"
        and terminal.get("failure_message")
        == "V6 real full-vs-tail answer-logit equivalence failed"
        and terminal.get("optimizer_constructed") is False
        and terminal.get("optimizer_steps") == 0
        and terminal.get("training_executed") is False
        and terminal.get("checkpoint_published") is False
        and math.isfinite(float(terminal.get("elapsed_seconds")))
        and float(terminal["elapsed_seconds"]) > 0.0
        and forbidden_post_equivalence_fields.isdisjoint(terminal)
    )
    checks = {
        "all_four_evidence_hashes_match": len(observed_sha256) == 4,
        "sealed_preregistration_matches": preregistration_valid,
        "historical_bound_source_inventory_valid": historical_source_inventory_valid,
        "single_zero_update_release_matches": release_valid,
        "attempt_claimed_before_model_load": attempt_valid,
        "terminal_failure_at_full_vs_tail_equivalence": terminal_valid,
        "no_post_equivalence_gradient_fields": forbidden_post_equivalence_fields.isdisjoint(
            terminal
        ),
        "full_execution_file_audit_matches": audit_valid,
        "zero_forbidden_file_reads": terminal.get("forbidden_file_accesses") == [],
        "deferred_and_final_qa_untouched": (
            terminal.get("deferred_or_final_qa_accessed") is False
        ),
        "training_release_result_and_checkpoint_absent": all(absent_outputs.values()),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V6 terminal smoke evidence failed authentication: {failed}")

    return {
        "schema_version": 1,
        "artifact": "gemma4_v54_fixed_prefix_decoder_reader_v6_terminal_evidence",
        "status": "authenticated_terminal_smoke_failure_no_training_no_checkpoint",
        "evidence_authenticated": True,
        "passed": False,
        "promotion_eligible": False,
        "failure_stage": "byte_exact_full_vs_tail_answer_logit_equivalence",
        "failure_type": terminal["failure_type"],
        "failure_message": terminal["failure_message"],
        "single_smoke_attempt_consumed": True,
        "maximum_smoke_attempts": 1,
        "gradient_computation_executed": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_executed": False,
        "checkpoint_published": False,
        "checkpoint_absent": True,
        "training_release_absent": True,
        "training_result_absent": True,
        "greedy_generation_executed": False,
        "file_access_audit_active_for_entire_execution": True,
        "loaded_file_count": int(terminal["loaded_file_count"]),
        "loaded_file_inventory_sha256": terminal["loaded_file_inventory_sha256"],
        "forbidden_file_read_count": 0,
        "deferred_or_final_qa_accessed": False,
        "elapsed_seconds": float(terminal["elapsed_seconds"]),
        "planned_train_rows": int(independent_audit["train_rows"]),
        "planned_validation_rows": int(independent_audit["validation_rows"]),
        "planned_updates": int(independent_audit["schedule_updates"]),
        "historical_source_inventory_authenticated": True,
        "current_runtime_compatibility_claimed": False,
        "checks": checks,
        "absent_outputs": absent_outputs,
        "evidence_sha256": observed_sha256,
        "measurement_evidence_paths": sorted(observed_sha256),
        "scientific_conclusion": (
            "The proposed upper-decoder reader did not reach its gradient smoke: the "
            "real full-sequence and answer-tail paths failed the preregistered byte-exact "
            "selected-logit equivalence gate. No optimization or behavioral inference "
            "about the reader is supported by this attempt."
        ),
    }


def main() -> None:
    print(
        json.dumps(
            authenticate_v6_terminal_smoke_failure(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT",
    "EVIDENCE_SHA256",
    "TERMINAL_SMOKE",
    "TRAINING_RELEASE",
    "TRAINING_RESULT",
    "authenticate_v6_terminal_smoke_failure",
]
