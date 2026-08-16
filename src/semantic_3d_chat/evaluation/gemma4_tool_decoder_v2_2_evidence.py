"""Authenticate the terminal Gemma tool-decoder V2.2 rejection without a model.

Only five immutable, explicitly allowlisted JSON artifacts are opened.  Their
byte hashes bind the staged CPU authorization, MPS smoke release, real-model
smoke, multi-update release, and terminal training result.  No trace, oracle,
scene-map, model, or held-out answer file is read by this authenticator.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

CPU_AUTHORIZATION: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_embodied_tool_decoder_training_authorization_v2_2.json"
)
MPS_SMOKE_RELEASE: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_embodied_tool_decoder_mps_smoke_release_v2_2.json"
)
MPS_SMOKE_REPORT: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_embodied_tool_decoder_full_mps_smoke_v2_2.json"
)
TRAINING_RELEASE: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_embodied_tool_decoder_multi_update_release_v2_2.json"
)
TERMINAL_RESULT: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_embodied_tool_decoder_training_v2_2.json"
)
EVIDENCE_SHA256: Final[dict[Path, str]] = {
    CPU_AUTHORIZATION: "ee96f44d1e6e3cde74d2264750bfa3f5b238e45c3628379784bba3fa7d65f27d",
    MPS_SMOKE_RELEASE: "679e58b3644a340de94b8bd53bc7e40ceb6d788e57e96b17b4f022413a9a5766",
    MPS_SMOKE_REPORT: "48d7ba9ca2e0d3a1b8f68a490475c1d433fa449cc01907ac59df5e0f73e1fd48",
    TRAINING_RELEASE: "ba9d5523bb34a6d5e406f13baea821090a0d1714b6ecd76cf851616c6f349f77",
    TERMINAL_RESULT: "fc6cb4a829e8a69aa94c03c13d79c270b1829bb8cddde500e6b6b0fe10cbfc01",
}
CHECKPOINT: Final[Path] = Path(
    "data_gemma4/checkpoints/gemma4_embodied_tool_decoder_v2/final"
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
        raise TypeError(f"Tool-decoder evidence must be a JSON object: {path}")
    return value


def authenticate_tool_decoder_v2_2_negative_result(
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return the exact terminal rejection or fail closed on any discrepancy."""

    resolved_root = Path(root).expanduser().resolve()
    reports: dict[Path, dict[str, Any]] = {}
    observed_sha256: dict[str, str] = {}
    for path, expected in EVIDENCE_SHA256.items():
        resolved = _resolve(resolved_root, path)
        if not resolved.is_file() or resolved.is_symlink():
            raise FileNotFoundError(f"Tool-decoder evidence is absent or linked: {path}")
        observed = _sha256(resolved)
        if observed != expected:
            raise ValueError(
                f"Tool-decoder evidence digest changed: {path}: {observed} != {expected}"
            )
        observed_sha256[path.as_posix()] = observed
        reports[path] = _read(resolved_root, path)

    cpu = reports[CPU_AUTHORIZATION]
    smoke_release = reports[MPS_SMOKE_RELEASE]
    smoke = reports[MPS_SMOKE_REPORT]
    training_release = reports[TRAINING_RELEASE]
    result = reports[TERMINAL_RESULT]
    checkpoint_absent = not _resolve(resolved_root, CHECKPOINT).exists()

    cpu_sources = cpu.get("bound_source_sha256")
    shared_authorization = {
        "artifact": "gemma4_embodied_tool_decoder_training_authorization_v2_2",
        "schema_version": "2.2",
        "trace_rows_sha256": "72434178ff1cf23c2dfeb98d52cb7b4c443fcc8715c1dd4ee883d87ae127e7ad",
        "prefix_inventory_sha256": "c477fd12bc4104f147f73c2f6d46904e0b83b3c584206cb227fd70e9371d0d63",
        "clearance_cache_sha256": "658822707389e67481fa59b035a7e7f19c360487b19d3157b80bc23ede1db048",
        "clearance_manifest_sha256": "51cf6c0b155e149627f300c17d39369f91f14e415099fe10d9de1682ef8c7e24",
    }
    shared_authorization_matches = all(
        all(artifact.get(key) == value for key, value in shared_authorization.items())
        and artifact.get("bound_source_sha256") == cpu_sources
        for artifact in (cpu, smoke_release, training_release)
    )
    resource_contract = cpu.get("resource_contract")
    resource_contract_valid = (
        isinstance(resource_contract, dict)
        and all(
            artifact.get("resource_contract") == resource_contract
            for artifact in (smoke_release, training_release)
        )
        and resource_contract.get("optimizer_updates") == 64
        and resource_contract.get("training_microbatches") == 512
        and resource_contract.get("teacher_forced_unique_evaluation_forwards") == 5852
        and resource_contract.get("greedy_unique_sequences") == 896
        and resource_contract.get("greedy_maximum_new_tokens") == 24
        and resource_contract.get("checkpoint_selection")
        == "fixed_final_update_64_no_posthoc_selection"
        and resource_contract.get("answer_tail_only_training_and_evaluation") is True
        and resource_contract.get("real_decoder_validator_simulator_probe_before_promotion")
        is True
    )
    bound_source_inventory_valid = (
        isinstance(cpu_sources, dict)
        and len(cpu_sources) == 25
        and all(
            isinstance(path, str)
            and path
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for path, digest in cpu_sources.items()
        )
    )
    authorization_chain_matches = (
        cpu.get("authorization_stage") == "cpu_preparation"
        and cpu.get("status") == "cpu_inputs_sealed_heavy_mps_not_released"
        and cpu.get("cpu_preparation_authorized") is True
        and cpu.get("full_model_mps_microbatch_authorized") is False
        and cpu.get("multi_update_training_authorized") is False
        and smoke_release.get("authorization_stage") == "full_model_mps_microbatch"
        and smoke_release.get("status") == "heavy_mps_microbatch_released"
        and smoke_release.get("parent_authorization_sha256")
        == EVIDENCE_SHA256[CPU_AUTHORIZATION]
        and smoke_release.get("full_model_mps_microbatch_authorized") is True
        and smoke_release.get("multi_update_training_authorized") is False
        and smoke.get("authorization_sha256") == EVIDENCE_SHA256[MPS_SMOKE_RELEASE]
        and training_release.get("authorization_stage") == "multi_update_training"
        and training_release.get("status") == "multi_update_training_released"
        and training_release.get("parent_authorization_sha256")
        == EVIDENCE_SHA256[MPS_SMOKE_RELEASE]
        and training_release.get("full_model_mps_microbatch_smoke_sha256")
        == EVIDENCE_SHA256[MPS_SMOKE_REPORT]
        and training_release.get("multi_update_training_authorized") is True
        and result.get("authorization_sha256") == EVIDENCE_SHA256[TRAINING_RELEASE]
    )

    smoke_valid = (
        smoke.get("schema")
        == "semantic_3d_chat.gemma4_tool_decoder_full_mps_smoke.v2_2"
        and smoke.get("status") == "passed"
        and smoke.get("device") == "mps"
        and smoke.get("full_model_loaded") is True
        and smoke.get("mps_used") is True
        and smoke.get("training_executed") is False
        and smoke.get("optimizer_steps") == 0
        and smoke.get("checkpoint_published") is False
        and smoke.get("microbatches") == 1
        and smoke.get("trainable_parameter_count") == 165_888
        and float(smoke["real_full_vs_tail_answer_nll_absolute_difference"]) == 0.0
        and float(smoke["lora_gradient_l2"]) > 0.0
        and float(smoke["projector_gradient_l2"]) > 0.0
    )

    history = result.get("history")
    heldout = result.get("all_heldout_teacher_forced")
    gate = result.get("teacher_forced_early_gate")
    history_valid = (
        isinstance(history, list)
        and len(history) == 64
        and [row.get("update") for row in history if isinstance(row, dict)]
        == list(range(1, 65))
        and all(
            isinstance(row, dict)
            and math.isfinite(float(row["training_loss"]))
            and math.isfinite(float(row["gradient_l2_before_clip"]))
            and float(row["gradient_l2_before_clip"]) > 0.0
            for row in history
        )
        and float(history[0]["training_loss"]) == 2.414295881986618
        and float(history[-1]["training_loss"]) == 0.2341814790852368
    )
    heldout_valid = (
        isinstance(heldout, dict)
        and heldout.get("schema")
        == "semantic_3d_chat.gemma4_tool_decoder_teacher_forced.v2"
        and heldout.get("sample_count") == 2268
        and heldout.get("expected_sample_count") == 2268
        and heldout.get("scene_count") == 8
        and heldout.get("held_out_scenes_only") is True
        and heldout.get("all_heldout_rows_scored") is True
        and heldout.get("environmental_text_inputs") == []
        and heldout.get("oracle_inputs_at_runtime") is False
        and heldout.get("answer_token_count") == 38_054
        and float(heldout["answer_token_nll"]) == 0.37775762747489017
        and float(heldout["answer_token_accuracy"]) == 0.8712881694434225
        and float(heldout["exact_sequence_accuracy"]) == 0.17416225749559083
        and float(heldout["teacher_forced_argmax_valid_schema_rate"])
        == 0.2641093474426808
        and float(heldout["teacher_forced_argmax_tool_accuracy"])
        == 0.24118165784832452
    )
    expected_failed = [
        "exact_sequence_accuracy",
        "teacher_forced_argmax_valid_schema_rate",
        "teacher_forced_argmax_tool_accuracy",
    ]
    expected_gate_checks = {
        "all_heldout_rows_scored": True,
        "answer_token_accuracy": True,
        "answer_token_nll": True,
        "exact_sequence_accuracy": False,
        "sample_count": True,
        "scene_count": True,
        "teacher_forced_argmax_tool_accuracy": False,
        "teacher_forced_argmax_valid_schema_rate": False,
    }
    terminal_rejection_valid = (
        result.get("schema") == "semantic_3d_chat.gemma4_tool_decoder_training.v2"
        and result.get("status")
        == "rejected_before_greedy_generation_no_runtime_checkpoint"
        and result.get("optimizer_updates") == 64
        and result.get("microbatch_size") == 1
        and result.get("gradient_accumulation") == 8
        and result.get("selected_update") == 64
        and float(result["selected_training_loss"]) == 0.2341814790852368
        and math.isfinite(float(result["elapsed_seconds"]))
        and float(result["elapsed_seconds"]) > 0.0
        and result.get("checkpoint_selection") == "fixed_final_update_no_posthoc_selection"
        and result.get("greedy_generation_executed") is False
        and result.get("runtime_checkpoint_published") is False
        and isinstance(gate, dict)
        and gate.get("schema")
        == "semantic_3d_chat.gemma4_tool_decoder_teacher_forced_gate.v2"
        and gate.get("evaluated_before_greedy_generation") is True
        and gate.get("passed") is False
        and gate.get("failed") == expected_failed
        and gate.get("checks") == expected_gate_checks
        and checkpoint_absent
    )
    checks = {
        "all_evidence_hashes_match": len(observed_sha256) == len(EVIDENCE_SHA256),
        "historical_bound_source_inventory_valid": bound_source_inventory_valid,
        "shared_authorization_contract_matches": shared_authorization_matches,
        "resource_contract_matches": resource_contract_valid,
        "authorization_chain_matches": authorization_chain_matches,
        "real_full_model_smoke_passed": smoke_valid,
        "all_64_updates_present": history_valid,
        "all_heldout_teacher_forcing_present": heldout_valid,
        "terminal_rejection_and_gate_match": terminal_rejection_valid,
        "runtime_checkpoint_absent": checkpoint_absent,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Tool-decoder V2.2 evidence failed authentication: {failed}")

    assert isinstance(history, list)
    assert isinstance(heldout, dict)
    return {
        "schema_version": 1,
        "artifact": "gemma4_tool_decoder_v2_2_terminal_evidence",
        "status": "authenticated_terminal_negative_no_runtime_checkpoint",
        "evidence_authenticated": True,
        "passed": False,
        "promotion_eligible": False,
        "runtime_checkpoint_published": False,
        "runtime_checkpoint_absent": True,
        "greedy_generation_executed": False,
        "model_loaded_by_authenticator": False,
        "mps_used_by_authenticator": False,
        "historical_source_inventory_authenticated": True,
        "current_runtime_compatibility_claimed": False,
        "optimizer_updates": 64,
        "training_microbatches": 512,
        "trainable_parameter_count": 165_888,
        "training_loss_first": float(history[0]["training_loss"]),
        "training_loss_final": float(history[-1]["training_loss"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "heldout_scene_count": int(heldout["scene_count"]),
        "heldout_sample_count": int(heldout["sample_count"]),
        "heldout_answer_token_count": int(heldout["answer_token_count"]),
        "heldout_answer_token_nll": float(heldout["answer_token_nll"]),
        "heldout_answer_token_accuracy": float(heldout["answer_token_accuracy"]),
        "heldout_exact_sequence_accuracy": float(heldout["exact_sequence_accuracy"]),
        "heldout_valid_schema_rate": float(
            heldout["teacher_forced_argmax_valid_schema_rate"]
        ),
        "heldout_tool_accuracy": float(
            heldout["teacher_forced_argmax_tool_accuracy"]
        ),
        "failed_early_gates": expected_failed,
        "terminal_result_sha256": EVIDENCE_SHA256[TERMINAL_RESULT],
        "training_release_sha256": EVIDENCE_SHA256[TRAINING_RELEASE],
        "checks": checks,
        "evidence_sha256": observed_sha256,
        "measurement_evidence_paths": sorted(observed_sha256),
        "scientific_conclusion": (
            "The decoder learned token likelihood but did not learn sufficiently exact, "
            "schema-valid tool sequences; its locked teacher-forced early gate stopped "
            "greedy evaluation and checkpoint publication."
        ),
    }


def main() -> None:
    print(
        json.dumps(
            authenticate_tool_decoder_v2_2_negative_result(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT",
    "EVIDENCE_SHA256",
    "TERMINAL_RESULT",
    "authenticate_tool_decoder_v2_2_negative_result",
]
