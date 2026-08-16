"""Create-once V6.1 release correcting only V6 numerical equivalence.

V6 consumed its sole MPS attempt because two mathematically equivalent BF16
language-head shapes were required to be byte-identical.  This successor binds
that terminal failure and every frozen ancestor, preregisters objective-level
tolerances before another real run, and authorizes exactly one zero-update MPS
smoke.  It does not authorize an optimizer, training, or a checkpoint write.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    INITIAL_STATE_SHA256,
    TARGET_MODULES,
    _model_snapshot,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_LORA_STATE_SHA256 as TOOL_INITIAL_LORA_STATE_SHA256,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_PROJECTOR_STATE_SHA256,
)

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_decoder_reader_v6_1"
V6_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_preregistration.json"
)
V6_RELEASE: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke_release.json"
)
V6_ATTEMPT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke_attempt.json"
)
V6_TERMINAL_FAILURE: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke.json"
)
MPS_SMOKE_RELEASE: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_1_mps_smoke_release.json"
)
MPS_SMOKE_ATTEMPT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_1_mps_smoke_attempt.json"
)
MPS_SMOKE_REPORT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_1_mps_smoke.json"
)

V6_PREREGISTRATION_SHA256: Final[str] = (
    "4f0e3b0da793fc0f6baeb5d032579a50d81e05c8014d738ddfb05756ec0a90cf"
)
V6_RELEASE_SHA256: Final[str] = (
    "65af03ab1259201d824fbd44e1ce3e69def10a47007a14c5f5151e772257e6c3"
)
V6_ATTEMPT_SHA256: Final[str] = (
    "c4d08911a69db7d0f97b7ca53def5b023a771045941a076754ff53affefa1c15"
)
V6_TERMINAL_FAILURE_SHA256: Final[str] = (
    "a78e38e9e5112f757927a9590cecb854c9c99f7881d929b531b83b9db305f2fa"
)
V6_TERMINAL_FAILURE_STATUS: Final[str] = "failed_terminal_attempt_consumed"
V6_TERMINAL_FAILURE_TYPE: Final[str] = "RuntimeError"
V6_TERMINAL_FAILURE_MESSAGE: Final[str] = (
    "V6 real full-vs-tail answer-logit equivalence failed"
)
V6_PROPOSAL_SHA256: Final[str] = (
    "fd2a85ae8f29e56a710fe6c3c63970c5c1593ea4d7aa12b9a815bbf0a9edfa14"
)
MODEL_WEIGHTS_BLOB_SHA256: Final[str] = (
    "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
)
MODEL_WEIGHTS_SIZE_BYTES: Final[int] = 10_246_621_918

# These values are fixed before the V6.1 MPS attempt.  The raw-logit limits
# permit bounded BF16 language-head shape rounding only after the entire final
# decoder hidden state and a same-shape reprojection have been proved exact.
# The substantially tighter NLL/distribution/gradient gates are what define
# numerical objective equivalence; the raw maximum alone can never pass it.
OBJECTIVE_EQUIVALENCE_THRESHOLDS: Final[dict[str, Any]] = {
    "contract_version": "v6.1_bounded_numerical_objective_gradient_equivalence_1",
    "bf16_unit_roundoff": 0.0078125,
    "raw_logits_max_abs": 0.25,
    "raw_logits_rms": 0.01,
    "raw_logits_mean_abs": 0.002,
    "raw_logits_per_token_cosine_min": 0.99999,
    "per_token_nll_max_abs": 0.00002,
    "mean_nll_abs": 0.000001,
    "js_divergence_max": 0.000001,
    "softmax_ce_gradient_max_abs": 0.001,
    "softmax_ce_gradient_cosine_min": 0.99999,
    "hf_loss_vs_manual_full_fp32_ce_abs": 0.000001,
    "top1_predictions_exact": True,
    "top_k": 5,
    "top_k_minimum_overlap_fraction": 0.8,
    "target_membership_top_k": 10,
    "rank_delta_logit_bound": 0.25,
    "rank_tie_band_multiplier": 2.0,
    "targets_exact": True,
    "label_positions_exact": True,
    "causal_positions_exact": True,
    "prepared_inputs_exact": True,
    "entire_final_hidden_states_exact": True,
    "selected_hidden_states_exact_when_accessible": True,
    "common_shape_reprojection_logits_exact": True,
    "common_shape_reprojection_nll_exact": True,
}
GRADIENT_EQUIVALENCE_THRESHOLDS: Final[dict[str, Any]] = {
    "contract_version": "v6.1_first_schedule_clean_backward_equivalence_1",
    "branch_nll_abs": 0.000001,
    "margin_abs": 0.000002,
    "composite_abs": 0.000005,
    "gradient_cosine_min": 0.99999,
    "gradient_relative_l2_max": 0.005,
    "gradient_norm_ratio_min": 0.995,
    "gradient_norm_ratio_max": 1.005,
    "lora_a_exact_zero": True,
    "gradient_coverage_exact": True,
    "retention_self_kl_abs_max": 0.00001,
}

EXPECTED_SOFTWARE_VERSIONS: Final[dict[str, str]] = {
    "python": "3.12.13",
    "numpy": "2.5.1",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "transformers": "5.14.1",
}
V6_1_BOUND_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_1_release.py",
    "src/semantic_3d_chat/training/smoke_fixed_prefix_decoder_reader_v6_1.py",
    "scripts/run_gemma4_v54_fixed_prefix_decoder_reader_v6_1.sh",
    "tests/test_fixed_prefix_decoder_reader_v6_1_release.py",
    "tests/test_smoke_fixed_prefix_decoder_reader_v6_1.py",
)
MPS_MEMORY_PHASES: Final[frozenset[str]] = frozenset(
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

INSTALLED_TRANSFORMERS_SOURCE_BINDINGS: Final[dict[str, tuple[str, int]]] = {
    "transformers.models.gemma4.modeling_gemma4": (
        "ccab8e2dd80b71e9ca34e2c87291e17c40a27c755006e554da2ebf70d6616916",
        125_730,
    ),
    "transformers.loss.loss_utils": (
        "83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe",
        8_540,
    ),
}

_EQUIVALENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "contract_version",
        "thresholds",
        "token_count",
        "vocabulary_size",
        "prepared_identity",
        "index_identity",
        "hidden_identity",
        "common_shape_reprojection",
        "hf_loss_manual_ce",
        "raw_postsoftcap_logits",
        "nll",
        "distribution",
        "predictions_and_ranks",
        "passed",
    }
)


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
            raise FileNotFoundError(f"V6.1 bound source is missing or unsafe: {relative}")
        result[relative] = sha256_file(source)
    return result


def _read_exact_json(path: str, expected_sha256: str, label: str) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V6.1 {label} is missing or unsafe")
    digest = sha256_file(source)
    if digest != expected_sha256:
        raise ValueError(f"V6.1 {label} bytes changed: {digest} != {expected_sha256}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V6.1 {label} is not a JSON object")
    return value


def _software_versions() -> dict[str, str]:
    observed = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "safetensors", "torch", "transformers")
        },
    }
    if observed != EXPECTED_SOFTWARE_VERSIONS:
        raise RuntimeError(f"V6.1 software versions changed: {observed}")
    return observed


def _installed_transformers_sources() -> dict[str, dict[str, Any]]:
    from transformers.loss import loss_utils
    from transformers.models.gemma4 import modeling_gemma4

    modules = {
        "transformers.models.gemma4.modeling_gemma4": modeling_gemma4,
        "transformers.loss.loss_utils": loss_utils,
    }
    observed: dict[str, dict[str, Any]] = {}
    for name, module in modules.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise FileNotFoundError(f"V6.1 installed source has no file: {name}")
        path = Path(raw_path).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"V6.1 installed source is missing or unsafe: {name}")
        expected_sha, expected_size = INSTALLED_TRANSFORMERS_SOURCE_BINDINGS[name]
        digest = sha256_file(path)
        size = path.stat().st_size
        if (digest, size) != (expected_sha, expected_size):
            raise ValueError(f"V6.1 installed Transformers source changed: {name}")
        observed[name] = {
            "sha256": digest,
            "size_bytes": size,
            "basename": path.name,
        }
    return observed


def _authenticate_model_blob() -> dict[str, Any]:
    weights = _model_snapshot() / "model.safetensors"
    resolved = weights.resolve()
    if not resolved.is_file() or resolved.stat().st_size != MODEL_WEIGHTS_SIZE_BYTES:
        raise ValueError("V6.1 local Gemma model weight size changed")
    digest = sha256_file(resolved)
    if digest != MODEL_WEIGHTS_BLOB_SHA256:
        raise ValueError("V6.1 local Gemma model weight bytes changed")
    return {
        "model_weights_blob_sha256": digest,
        "model_weights_size_bytes": MODEL_WEIGHTS_SIZE_BYTES,
        "actual_model_bytes_streamed": True,
    }


def _authenticate_frozen_lineage() -> dict[str, Any]:
    preregistration = _read_exact_json(
        V6_PREREGISTRATION, V6_PREREGISTRATION_SHA256, "V6 sealed preregistration"
    )
    release = _read_exact_json(V6_RELEASE, V6_RELEASE_SHA256, "V6 smoke release")
    attempt = _read_exact_json(V6_ATTEMPT, V6_ATTEMPT_SHA256, "V6 attempt journal")
    terminal = _read_exact_json(
        V6_TERMINAL_FAILURE,
        V6_TERMINAL_FAILURE_SHA256,
        "V6 terminal failure",
    )
    frozen = preregistration.get("frozen_proposal")
    v6_sources = release.get("bound_source_sha256")
    required = (
        preregistration.get("status")
        == "sealed_before_real_mps_smokes_training_not_authorized"
        and isinstance(frozen, dict)
        and frozen.get("canonical_proposal_sha256") == V6_PROPOSAL_SHA256
        and frozen.get("local_model_weights_sha256") == MODEL_WEIGHTS_BLOB_SHA256
        and frozen.get("local_model_weights_size_bytes") == MODEL_WEIGHTS_SIZE_BYTES
        and release.get("status")
        == "released_exactly_one_zero_update_full_model_mps_smoke"
        and release.get("parent_preregistration_sha256")
        == V6_PREREGISTRATION_SHA256
        and release.get("required_software_versions")
        == EXPECTED_SOFTWARE_VERSIONS
        and isinstance(v6_sources, dict)
        and bool(v6_sources)
        and attempt.get("status") == "claimed_before_model_loading"
        and attempt.get("authorization_sha256") == V6_RELEASE_SHA256
        and terminal.get("status") == V6_TERMINAL_FAILURE_STATUS
        and terminal.get("passed") is False
        and terminal.get("failure_type") == V6_TERMINAL_FAILURE_TYPE
        and terminal.get("failure_message") == V6_TERMINAL_FAILURE_MESSAGE
        and terminal.get("authorization_sha256") == V6_RELEASE_SHA256
        and terminal.get("attempt_sha256") == V6_ATTEMPT_SHA256
        and terminal.get("optimizer_constructed") is False
        and terminal.get("optimizer_steps") == 0
        and terminal.get("training_executed") is False
        and terminal.get("checkpoint_published") is False
        and terminal.get("deferred_or_final_qa_accessed") is False
    )
    if not required:
        raise ValueError("V6.1 frozen V6 lineage fields changed")
    observed_v6_sources = _bound_hashes(tuple(v6_sources))
    if observed_v6_sources != v6_sources:
        raise ValueError("V6.1 frozen V6 source bytes changed")
    return {
        "v6_preregistration_sha256": V6_PREREGISTRATION_SHA256,
        "v6_release_sha256": V6_RELEASE_SHA256,
        "v6_attempt_sha256": V6_ATTEMPT_SHA256,
        "v6_terminal_failure_sha256": V6_TERMINAL_FAILURE_SHA256,
        "v6_terminal_failure_status": V6_TERMINAL_FAILURE_STATUS,
        "v6_terminal_failure_type": V6_TERMINAL_FAILURE_TYPE,
        "v6_terminal_failure_message": V6_TERMINAL_FAILURE_MESSAGE,
        "frozen_proposal": frozen,
        "v6_bound_source_sha256": v6_sources,
    }


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def _create_once(path: str | Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V6.1 create-once artifact exists: {destination}")
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


def _build_v6_1_mps_smoke_release(*, require_output_absent: bool) -> dict[str, Any]:
    if require_output_absent and any(
        _resolve(path).exists() for path in (MPS_SMOKE_ATTEMPT, MPS_SMOKE_REPORT)
    ):
        raise FileExistsError("V6.1 MPS smoke already has an attempt or terminal report")
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_mps_smoke_release",
        "status": (
            "released_exactly_one_zero_update_full_model_mps_smoke_"
            "with_preregistered_objective_equivalence"
        ),
        "frozen_v6_lineage": _authenticate_frozen_lineage(),
        "local_model": _authenticate_model_blob(),
        "required_software_versions": _software_versions(),
        "installed_transformers_source": _installed_transformers_sources(),
        "v6_1_bound_source_sha256": _bound_hashes(V6_1_BOUND_PATHS),
        "objective_equivalence_thresholds": OBJECTIVE_EQUIVALENCE_THRESHOLDS,
        "authorized": {
            "full_model_mps_objective_equivalence": True,
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
        "attempt_journal": MPS_SMOKE_ATTEMPT,
        "terminal_output": MPS_SMOKE_REPORT,
    }


def build_v6_1_mps_smoke_release() -> dict[str, Any]:
    """Build the release after authenticating all frozen bytes."""

    return _build_v6_1_mps_smoke_release(require_output_absent=True)


def write_v6_1_mps_smoke_release() -> tuple[Path, str]:
    return _create_once(MPS_SMOKE_RELEASE, build_v6_1_mps_smoke_release())


def authenticate_v6_1_mps_smoke_release() -> tuple[dict[str, Any], str]:
    source = _resolve(MPS_SMOKE_RELEASE)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V6.1 MPS smoke release is missing or unsafe")
    observed = json.loads(source.read_text(encoding="utf-8"))
    expected = _build_v6_1_mps_smoke_release(require_output_absent=False)
    if observed != expected:
        raise ValueError("V6.1 MPS smoke release or bound source changed")
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


def claim_v6_1_mps_smoke_attempt() -> tuple[Path, str]:
    _release, release_sha = authenticate_v6_1_mps_smoke_release()
    if _resolve(MPS_SMOKE_REPORT).exists():
        raise FileExistsError("V6.1 MPS smoke already has a terminal report")
    return _create_once(MPS_SMOKE_ATTEMPT, _smoke_attempt_payload(release_sha))


def authenticate_v6_1_mps_smoke_attempt() -> tuple[dict[str, Any], str]:
    source = _resolve(MPS_SMOKE_ATTEMPT)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V6.1 MPS smoke attempt journal is missing or unsafe")
    release_sha = sha256_file(MPS_SMOKE_RELEASE)
    observed = json.loads(source.read_text(encoding="utf-8"))
    expected = _smoke_attempt_payload(release_sha)
    if observed != expected:
        raise ValueError("V6.1 MPS smoke attempt differs from its release")
    return observed, sha256_file(source)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _same_number(claimed: object, derived: float, *, atol: float = 1e-12) -> bool:
    return _finite_number(claimed) and math.isclose(
        float(claimed), derived, rel_tol=1e-12, abs_tol=atol
    )


def _sha256_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def objective_equivalence_passes(metrics: object) -> bool:
    """Recompute every preregistered objective-equivalence condition."""

    if not isinstance(metrics, dict) or set(metrics) != _EQUIVALENCE_FIELDS:
        return False
    thresholds = OBJECTIVE_EQUIVALENCE_THRESHOLDS
    if (
        metrics.get("contract_version") != thresholds["contract_version"]
        or metrics.get("thresholds") != thresholds
    ):
        return False
    count = metrics.get("token_count")
    vocabulary = metrics.get("vocabulary_size")
    if type(count) is not int or count < 1 or type(vocabulary) is not int or vocabulary < 2:
        return False

    prepared = metrics.get("prepared_identity")
    if not isinstance(prepared, dict) or set(prepared) != {"fields", "all_exact"}:
        return False
    fields = prepared.get("fields")
    required_prepared = {
        "inputs_embeds",
        "attention_mask",
        "per_layer_inputs",
        "mm_token_type_ids",
        "labels",
    }
    if not isinstance(fields, dict) or set(fields) != required_prepared:
        return False
    for value in fields.values():
        if (
            not isinstance(value, dict)
            or set(value)
            != {"reference_sha256", "selected_sha256", "shape", "dtype", "exact"}
            or value.get("exact") is not True
            or value.get("reference_sha256") != value.get("selected_sha256")
            or not isinstance(value.get("shape"), list)
            or not isinstance(value.get("dtype"), str)
        ):
            return False
    if prepared.get("all_exact") is not True:
        return False

    indices = metrics.get("index_identity")
    index_fields = {
        "target_token_ids",
        "label_positions",
        "causal_positions",
        "targets_exact",
        "label_positions_exact",
        "causal_positions_exact",
        "reference_targets_sha256",
        "selected_targets_sha256",
        "reference_label_positions_sha256",
        "selected_label_positions_sha256",
        "reference_causal_positions_sha256",
        "selected_causal_positions_sha256",
    }
    if not isinstance(indices, dict) or set(indices) != index_fields:
        return False
    for field in ("target_token_ids", "label_positions", "causal_positions"):
        if not isinstance(indices.get(field), list) or len(indices[field]) != count:
            return False
    if not (
        indices.get("targets_exact") is True
        and indices.get("label_positions_exact") is True
        and indices.get("causal_positions_exact") is True
        and indices.get("reference_targets_sha256")
        == indices.get("selected_targets_sha256")
        and indices.get("reference_label_positions_sha256")
        == indices.get("selected_label_positions_sha256")
        and indices.get("reference_causal_positions_sha256")
        == indices.get("selected_causal_positions_sha256")
    ):
        return False

    hidden = metrics.get("hidden_identity")
    hidden_fields = {
        "hook_module",
        "accessible",
        "entire_exact",
        "selected_exact",
        "entire_shape",
        "selected_shape",
        "reference_entire_sha256",
        "selected_entire_sha256",
        "reference_selected_sha256",
        "selected_selected_sha256",
    }
    if not isinstance(hidden, dict) or set(hidden) != hidden_fields:
        return False
    if not (
        hidden.get("hook_module") == "model.language_model.norm"
        and hidden.get("accessible") is True
        and hidden.get("entire_exact") is True
        and hidden.get("selected_exact") is True
        and isinstance(hidden.get("entire_shape"), list)
        and isinstance(hidden.get("selected_shape"), list)
        and hidden.get("reference_entire_sha256")
        == hidden.get("selected_entire_sha256")
        and hidden.get("reference_selected_sha256")
        == hidden.get("selected_selected_sha256")
    ):
        return False

    common = metrics.get("common_shape_reprojection")
    common_fields = {
        "shape",
        "logits_exact",
        "nll_exact",
        "reference_logits_sha256",
        "selected_logits_sha256",
        "reference_per_token_nll",
        "selected_per_token_nll",
    }
    if not isinstance(common, dict) or set(common) != common_fields:
        return False
    if not (
        common.get("logits_exact") is True
        and common.get("nll_exact") is True
        and isinstance(common.get("shape"), list)
        and common.get("reference_logits_sha256")
        == common.get("selected_logits_sha256")
        and common.get("reference_per_token_nll")
        == common.get("selected_per_token_nll")
        and isinstance(common.get("reference_per_token_nll"), list)
        and len(common["reference_per_token_nll"]) == count
    ):
        return False

    hf_ce = metrics.get("hf_loss_manual_ce")
    if not isinstance(hf_ce, dict) or set(hf_ce) != {
        "hf_batch1_loss",
        "manual_full_fp32_ce",
        "absolute_difference",
        "passed",
    }:
        return False
    if any(
        not _finite_number(hf_ce.get(field))
        for field in ("hf_batch1_loss", "manual_full_fp32_ce", "absolute_difference")
    ):
        return False
    derived_hf_ce_difference = abs(
        float(hf_ce["hf_batch1_loss"]) - float(hf_ce["manual_full_fp32_ce"])
    )
    if not (
        hf_ce.get("passed") is True
        and _same_number(
            hf_ce.get("absolute_difference"), derived_hf_ce_difference
        )
        and derived_hf_ce_difference
        <= float(thresholds["hf_loss_vs_manual_full_fp32_ce_abs"])
    ):
        return False

    raw = metrics.get("raw_postsoftcap_logits")
    raw_fields = {
        "reference_shape",
        "selected_shape",
        "sufficient_statistics",
        "byte_exact",
        "max_abs_difference",
        "rms_difference",
        "mean_abs_difference",
        "per_token_cosine_similarity",
        "minimum_per_token_cosine_similarity",
    }
    if not isinstance(raw, dict) or set(raw) != raw_fields:
        return False
    sufficient = raw.get("sufficient_statistics")
    if not isinstance(sufficient, dict) or set(sufficient) != {
        "reference_logits_sha256",
        "selected_logits_sha256",
        "per_token",
    }:
        return False
    reference_logits_sha = sufficient.get("reference_logits_sha256")
    selected_logits_sha = sufficient.get("selected_logits_sha256")
    statistic_rows = sufficient.get("per_token")
    statistic_fields = {
        "vocabulary_count",
        "difference_sum_abs",
        "difference_sum_squares",
        "difference_max_abs",
        "reference_selected_dot",
        "reference_sum_squares",
        "selected_sum_squares",
    }
    if (
        not _sha256_string(reference_logits_sha)
        or not _sha256_string(selected_logits_sha)
        or not isinstance(statistic_rows, list)
        or len(statistic_rows) != count
    ):
        return False
    for row in statistic_rows:
        if not isinstance(row, dict) or set(row) != statistic_fields:
            return False
        if type(row.get("vocabulary_count")) is not int or row.get(
            "vocabulary_count"
        ) != vocabulary:
            return False
        if any(
            not _finite_number(row.get(field)) or float(row[field]) < 0.0
            for field in (
                "difference_sum_abs",
                "difference_sum_squares",
                "difference_max_abs",
                "reference_sum_squares",
                "selected_sum_squares",
            )
        ) or not _finite_number(row.get("reference_selected_dot")):
            return False
        if (
            float(row["difference_max_abs"]) * vocabulary
            + 1e-12
            < float(row["difference_sum_abs"])
            or float(row["difference_max_abs"])
            > float(row["difference_sum_abs"]) + 1e-12
            or float(row["difference_sum_abs"]) ** 2
            > vocabulary * float(row["difference_sum_squares"]) + 1e-9
            or float(row["difference_max_abs"]) ** 2
            > float(row["difference_sum_squares"]) + 1e-12
            or float(row["reference_sum_squares"]) <= 0.0
            or float(row["selected_sum_squares"]) <= 0.0
            or abs(float(row["reference_selected_dot"]))
            > math.sqrt(
                float(row["reference_sum_squares"])
                * float(row["selected_sum_squares"])
            )
            + 1e-8
            or not _same_number(
                row["difference_sum_squares"],
                float(row["reference_sum_squares"])
                + float(row["selected_sum_squares"])
                - 2.0 * float(row["reference_selected_dot"]),
                atol=1e-8,
            )
        ):
            return False
    element_count = count * vocabulary
    sum_abs = sum(float(row["difference_sum_abs"]) for row in statistic_rows)
    sum_squares = sum(
        float(row["difference_sum_squares"]) for row in statistic_rows
    )
    derived_max_abs = max(
        float(row["difference_max_abs"]) for row in statistic_rows
    )
    derived_rms = math.sqrt(sum_squares / element_count)
    derived_mean_abs = sum_abs / element_count
    derived_cosines = [
        float(row["reference_selected_dot"])
        / math.sqrt(
            float(row["reference_sum_squares"])
            * float(row["selected_sum_squares"])
        )
        for row in statistic_rows
    ]
    all_difference_statistics_zero = all(
        float(row["difference_sum_abs"]) == 0.0
        and float(row["difference_sum_squares"]) == 0.0
        and float(row["difference_max_abs"]) == 0.0
        for row in statistic_rows
    )
    hashes_equal = reference_logits_sha == selected_logits_sha
    cosines = raw.get("per_token_cosine_similarity")
    if not isinstance(cosines, list) or len(cosines) != count:
        return False
    raw_scalars = (
        "max_abs_difference",
        "rms_difference",
        "mean_abs_difference",
        "minimum_per_token_cosine_similarity",
    )
    if any(not _finite_number(raw.get(field)) for field in raw_scalars):
        return False
    if not all(_finite_number(value) for value in cosines):
        return False
    if not all(-1.0 - 1e-12 <= value <= 1.0 + 1e-12 for value in derived_cosines):
        return False
    if not (
        raw.get("reference_shape") == raw.get("selected_shape")
        and raw.get("reference_shape") == [1, count, vocabulary]
        and hashes_equal is all_difference_statistics_zero
        and raw.get("byte_exact") is hashes_equal
        and _same_number(raw.get("max_abs_difference"), derived_max_abs)
        and _same_number(raw.get("rms_difference"), derived_rms)
        and _same_number(raw.get("mean_abs_difference"), derived_mean_abs)
        and all(
            _same_number(claimed, derived)
            for claimed, derived in zip(cosines, derived_cosines, strict=True)
        )
        and _same_number(
            raw.get("minimum_per_token_cosine_similarity"), min(derived_cosines)
        )
        and derived_max_abs <= float(thresholds["raw_logits_max_abs"])
        and derived_rms <= float(thresholds["raw_logits_rms"])
        and derived_mean_abs <= float(thresholds["raw_logits_mean_abs"])
        and min(derived_cosines)
        >= float(thresholds["raw_logits_per_token_cosine_min"])
    ):
        return False

    nll = metrics.get("nll")
    nll_fields = {
        "reference_per_token",
        "selected_per_token",
        "max_abs_difference",
        "reference_mean",
        "selected_mean",
        "mean_absolute_difference",
    }
    if not isinstance(nll, dict) or set(nll) != nll_fields:
        return False
    reference_nll = nll.get("reference_per_token")
    selected_nll = nll.get("selected_per_token")
    if (
        not isinstance(reference_nll, list)
        or not isinstance(selected_nll, list)
        or len(reference_nll) != count
        or len(selected_nll) != count
        or not all(_finite_number(value) and float(value) >= 0 for value in reference_nll)
        or not all(_finite_number(value) and float(value) >= 0 for value in selected_nll)
    ):
        return False
    computed_nll_max = max(
        abs(float(left) - float(right))
        for left, right in zip(reference_nll, selected_nll, strict=True)
    )
    computed_reference_mean = sum(float(value) for value in reference_nll) / count
    computed_selected_mean = sum(float(value) for value in selected_nll) / count
    computed_mean_difference = abs(
        computed_reference_mean - computed_selected_mean
    )
    if not (
        _finite_number(nll.get("max_abs_difference"))
        and math.isclose(
            float(nll["max_abs_difference"]), computed_nll_max, abs_tol=1e-12
        )
        and computed_nll_max <= float(thresholds["per_token_nll_max_abs"])
        and _same_number(nll.get("reference_mean"), computed_reference_mean)
        and _same_number(
            hf_ce.get("manual_full_fp32_ce"), computed_reference_mean
        )
        and _same_number(nll.get("selected_mean"), computed_selected_mean)
        and _same_number(
            nll.get("mean_absolute_difference"), computed_mean_difference
        )
        and computed_mean_difference <= float(thresholds["mean_nll_abs"])
    ):
        return False

    distribution = metrics.get("distribution")
    distribution_fields = {
        "js_divergence_by_token",
        "maximum_js_divergence",
        "softmax_ce_gradient_max_abs_difference",
        "softmax_ce_gradient_cosine_similarity",
    }
    if not isinstance(distribution, dict) or set(distribution) != distribution_fields:
        return False
    divergences = distribution.get("js_divergence_by_token")
    if (
        not isinstance(divergences, list)
        or len(divergences) != count
        or not all(_finite_number(value) and float(value) >= 0 for value in divergences)
    ):
        return False
    if not (
        _finite_number(distribution.get("maximum_js_divergence"))
        and math.isclose(
            float(distribution["maximum_js_divergence"]),
            max(divergences),
            abs_tol=1e-12,
        )
        and max(divergences) <= float(thresholds["js_divergence_max"])
        and _finite_number(distribution.get("softmax_ce_gradient_max_abs_difference"))
        and float(distribution["softmax_ce_gradient_max_abs_difference"])
        <= float(thresholds["softmax_ce_gradient_max_abs"])
        and _finite_number(distribution.get("softmax_ce_gradient_cosine_similarity"))
        and float(distribution["softmax_ce_gradient_cosine_similarity"])
        >= float(thresholds["softmax_ce_gradient_cosine_min"])
    ):
        return False

    predictions = metrics.get("predictions_and_ranks")
    prediction_fields = {
        "reference_top1_token_ids",
        "selected_top1_token_ids",
        "top1_exact",
        "top5_overlap_fraction_by_token",
        "minimum_top5_overlap_fraction",
        "reference_target_top10_membership",
        "selected_target_top10_membership",
        "target_top10_membership_exact",
        "reference_target_ranks",
        "selected_target_ranks",
        "per_token_max_vocabulary_abs_logit_difference",
        "per_token_rank_tie_bands",
        "maximum_crossed_reference_target_gap_by_token",
        "target_rank_changes_confined_to_tie_band",
        "reference_strict_above_band_ranks",
        "selected_strict_above_band_ranks",
        "strict_above_band_rank_exact",
    }
    if not isinstance(predictions, dict) or set(predictions) != prediction_fields:
        return False
    length_fields = prediction_fields - {
        "top1_exact",
        "minimum_top5_overlap_fraction",
        "target_rank_changes_confined_to_tie_band",
        "target_top10_membership_exact",
        "strict_above_band_rank_exact",
    }
    if any(
        not isinstance(predictions.get(field), list)
        or len(predictions[field]) != count
        for field in length_fields
    ):
        return False
    overlaps = predictions["top5_overlap_fraction_by_token"]
    deltas = predictions["per_token_max_vocabulary_abs_logit_difference"]
    tie_bands = predictions["per_token_rank_tie_bands"]
    crossed_gaps = predictions["maximum_crossed_reference_target_gap_by_token"]
    if not all(_finite_number(value) and 0 <= float(value) <= 1 for value in overlaps):
        return False
    if not all(
        _finite_number(value)
        and 0 <= float(value) <= float(thresholds["rank_delta_logit_bound"])
        for value in deltas
    ):
        return False
    if not all(
        _same_number(delta, float(row["difference_max_abs"]))
        for delta, row in zip(deltas, statistic_rows, strict=True)
    ):
        return False
    if not all(
        _finite_number(gap) and 0 <= float(gap) <= float(band)
        for gap, band in zip(crossed_gaps, tie_bands, strict=True)
    ):
        return False
    if not all(
        _finite_number(band)
        and math.isclose(
            float(band),
            float(thresholds["rank_tie_band_multiplier"]) * float(delta),
            abs_tol=1e-12,
        )
        for band, delta in zip(tie_bands, deltas, strict=True)
    ):
        return False
    computed = (
        predictions.get("top1_exact") is True
        and predictions["reference_top1_token_ids"]
        == predictions["selected_top1_token_ids"]
        and _finite_number(predictions.get("minimum_top5_overlap_fraction"))
        and float(predictions["minimum_top5_overlap_fraction"]) == min(overlaps)
        and min(overlaps) >= float(thresholds["top_k_minimum_overlap_fraction"])
        and math.isclose(
            max(deltas),
            derived_max_abs,
            abs_tol=1e-12,
        )
        and predictions.get("target_top10_membership_exact") is True
        and predictions["reference_target_top10_membership"]
        == predictions["selected_target_top10_membership"]
        and predictions.get("target_rank_changes_confined_to_tie_band") is True
        and predictions.get("strict_above_band_rank_exact") is True
        and predictions["reference_strict_above_band_ranks"]
        == predictions["selected_strict_above_band_ranks"]
    )
    return bool(computed and metrics.get("passed") is True)


def gradient_equivalence_passes(metrics: object) -> bool:
    """Recompute clean first-schedule objective and LoRA-gradient gates."""

    if not isinstance(metrics, dict) or set(metrics) != {
        "contract_version",
        "thresholds",
        "objective_values",
        "gradient_comparisons",
        "retention_self_kl",
        "retention_gradient",
        "passed",
    }:
        return False
    thresholds = GRADIENT_EQUIVALENCE_THRESHOLDS
    if (
        metrics.get("contract_version") != thresholds["contract_version"]
        or metrics.get("thresholds") != thresholds
    ):
        return False
    objective = metrics.get("objective_values")
    objective_fields = {
        "full_correct_nll",
        "tail_correct_nll",
        "correct_nll_abs_difference",
        "full_wrong_nll",
        "tail_wrong_nll",
        "wrong_nll_abs_difference",
        "full_broad_nll",
        "tail_broad_nll",
        "broad_nll_abs_difference",
        "full_margin",
        "tail_margin",
        "margin_abs_difference",
        "full_composite",
        "tail_composite",
        "composite_abs_difference",
        "full_hinge_active",
        "tail_hinge_active",
    }
    if not isinstance(objective, dict) or set(objective) != objective_fields:
        return False
    numeric_objectives = objective_fields - {"full_hinge_active", "tail_hinge_active"}
    if any(not _finite_number(objective.get(field)) for field in numeric_objectives):
        return False
    raw_nll_fields = (
        "full_correct_nll",
        "tail_correct_nll",
        "full_wrong_nll",
        "tail_wrong_nll",
        "full_broad_nll",
        "tail_broad_nll",
    )
    if any(float(objective[field]) < 0.0 for field in raw_nll_fields):
        return False
    derived_nonnegative_fields = (
        "correct_nll_abs_difference",
        "wrong_nll_abs_difference",
        "broad_nll_abs_difference",
        "margin_abs_difference",
        "composite_abs_difference",
    )
    if any(float(objective[field]) < 0.0 for field in derived_nonnegative_fields):
        return False
    full_correct = float(objective["full_correct_nll"])
    tail_correct = float(objective["tail_correct_nll"])
    full_wrong = float(objective["full_wrong_nll"])
    tail_wrong = float(objective["tail_wrong_nll"])
    full_broad = float(objective["full_broad_nll"])
    tail_broad = float(objective["tail_broad_nll"])
    correct_difference = abs(full_correct - tail_correct)
    wrong_difference = abs(full_wrong - tail_wrong)
    broad_difference = abs(full_broad - tail_broad)
    full_margin = full_wrong - full_correct
    tail_margin = tail_wrong - tail_correct
    margin_difference = abs(full_margin - tail_margin)
    full_hinge = full_margin < 0.5
    tail_hinge = tail_margin < 0.5
    full_composite = (
        0.5 * full_correct
        + 4.0 * max(0.0, 0.5 - full_margin)
        + 0.5 * full_broad
    )
    tail_composite = (
        0.5 * tail_correct
        + 4.0 * max(0.0, 0.5 - tail_margin)
        + 0.5 * tail_broad
    )
    composite_difference = abs(full_composite - tail_composite)
    if not (
        _same_number(objective.get("correct_nll_abs_difference"), correct_difference)
        and _same_number(objective.get("wrong_nll_abs_difference"), wrong_difference)
        and _same_number(objective.get("broad_nll_abs_difference"), broad_difference)
        and _same_number(objective.get("full_margin"), full_margin)
        and _same_number(objective.get("tail_margin"), tail_margin)
        and _same_number(objective.get("margin_abs_difference"), margin_difference)
        and type(objective.get("full_hinge_active")) is bool
        and objective.get("full_hinge_active") is full_hinge
        and type(objective.get("tail_hinge_active")) is bool
        and objective.get("tail_hinge_active") is tail_hinge
        and full_hinge is tail_hinge
        and _same_number(objective.get("full_composite"), full_composite)
        and _same_number(objective.get("tail_composite"), tail_composite)
        and _same_number(
            objective.get("composite_abs_difference"), composite_difference
        )
        and correct_difference <= float(thresholds["branch_nll_abs"])
        and wrong_difference <= float(thresholds["branch_nll_abs"])
        and broad_difference <= float(thresholds["branch_nll_abs"])
        and margin_difference <= float(thresholds["margin_abs"])
        and composite_difference <= float(thresholds["composite_abs"])
    ):
        return False
    comparisons = metrics.get("gradient_comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != {
        "correct",
        "wrong",
        "broad",
        "aggregate",
    }:
        return False
    expected_coverage = list(TARGET_MODULES)
    comparison_fields = {
        "full_norm",
        "tail_norm",
        "cosine_similarity",
        "relative_l2",
        "norm_ratio",
        "full_lora_b_gradient_l2_by_target",
        "tail_lora_b_gradient_l2_by_target",
        "full_lora_a_gradient_l2_by_target",
        "tail_lora_a_gradient_l2_by_target",
        "full_lora_a_exact_zero",
        "tail_lora_a_exact_zero",
        "full_coverage",
        "tail_coverage",
        "coverage_exact",
        "sufficient_statistics",
        "passed",
    }
    for comparison in comparisons.values():
        if not isinstance(comparison, dict) or set(comparison) != comparison_fields:
            return False
        if any(
            not _finite_number(comparison.get(field))
            for field in (
                "full_norm",
                "tail_norm",
                "cosine_similarity",
                "relative_l2",
                "norm_ratio",
            )
        ):
            return False
        full_by_target = comparison.get("full_lora_b_gradient_l2_by_target")
        tail_by_target = comparison.get("tail_lora_b_gradient_l2_by_target")
        full_a_by_target = comparison.get("full_lora_a_gradient_l2_by_target")
        tail_a_by_target = comparison.get("tail_lora_a_gradient_l2_by_target")
        if (
            not isinstance(full_by_target, dict)
            or not isinstance(tail_by_target, dict)
            or set(full_by_target) != set(TARGET_MODULES)
            or set(tail_by_target) != set(TARGET_MODULES)
            or not all(_finite_number(value) and float(value) > 0 for value in full_by_target.values())
            or not all(_finite_number(value) and float(value) > 0 for value in tail_by_target.values())
            or not isinstance(full_a_by_target, dict)
            or not isinstance(tail_a_by_target, dict)
            or set(full_a_by_target) != set(TARGET_MODULES)
            or set(tail_a_by_target) != set(TARGET_MODULES)
            or any(value != 0.0 for value in full_a_by_target.values())
            or any(value != 0.0 for value in tail_a_by_target.values())
        ):
            return False
        evidence = comparison.get("sufficient_statistics")
        if not isinstance(evidence, dict) or set(evidence) != {
            "element_count",
            "full_vector_sha256",
            "tail_vector_sha256",
            "full_sum_squares",
            "tail_sum_squares",
            "full_tail_dot",
            "difference_sum_squares",
        }:
            return False
        if (
            type(evidence.get("element_count")) is not int
            or evidence.get("element_count", 0) < 1
            or not _sha256_string(evidence.get("full_vector_sha256"))
            or not _sha256_string(evidence.get("tail_vector_sha256"))
            or any(
                not _finite_number(evidence.get(field))
                for field in (
                    "full_sum_squares",
                    "tail_sum_squares",
                    "full_tail_dot",
                    "difference_sum_squares",
                )
            )
            or float(evidence["full_sum_squares"]) <= 0.0
            or float(evidence["tail_sum_squares"]) <= 0.0
            or float(evidence["difference_sum_squares"]) < 0.0
            or abs(float(evidence["full_tail_dot"]))
            > math.sqrt(
                float(evidence["full_sum_squares"])
                * float(evidence["tail_sum_squares"])
            )
            + 1e-8
            or not _same_number(
                evidence["difference_sum_squares"],
                float(evidence["full_sum_squares"])
                + float(evidence["tail_sum_squares"])
                - 2.0 * float(evidence["full_tail_dot"]),
                atol=1e-8,
            )
        ):
            return False
        derived_full_norm = math.sqrt(float(evidence["full_sum_squares"]))
        derived_tail_norm = math.sqrt(float(evidence["tail_sum_squares"]))
        derived_cosine = float(evidence["full_tail_dot"]) / (
            derived_full_norm * derived_tail_norm
        )
        derived_relative = math.sqrt(float(evidence["difference_sum_squares"])) / max(
            derived_full_norm, derived_tail_norm
        )
        derived_ratio = derived_tail_norm / derived_full_norm
        b_full_norm = math.sqrt(
            sum(float(value) ** 2 for value in full_by_target.values())
        )
        b_tail_norm = math.sqrt(
            sum(float(value) ** 2 for value in tail_by_target.values())
        )
        passed = (
            _same_number(comparison.get("full_norm"), derived_full_norm)
            and _same_number(comparison.get("tail_norm"), derived_tail_norm)
            and _same_number(derived_full_norm, b_full_norm)
            and _same_number(derived_tail_norm, b_tail_norm)
            and _same_number(comparison.get("cosine_similarity"), derived_cosine)
            and _same_number(comparison.get("relative_l2"), derived_relative)
            and _same_number(comparison.get("norm_ratio"), derived_ratio)
            and derived_cosine >= float(thresholds["gradient_cosine_min"])
            and derived_relative <= float(thresholds["gradient_relative_l2_max"])
            and float(thresholds["gradient_norm_ratio_min"])
            <= derived_ratio
            <= float(thresholds["gradient_norm_ratio_max"])
            and comparison.get("full_lora_a_exact_zero") is True
            and comparison.get("tail_lora_a_exact_zero") is True
            and comparison.get("full_coverage") == expected_coverage
            and comparison.get("tail_coverage") == expected_coverage
            and comparison.get("coverage_exact") is True
        )
        if not passed or comparison.get("passed") is not True:
            return False
    retention_gradient = metrics.get("retention_gradient")
    if not isinstance(retention_gradient, dict) or set(retention_gradient) != {
        "measured_from_freshly_zeroed_gradients",
        "lora_a_exact_zero",
        "lora_b_gradient_l2_by_target",
    }:
        return False
    retention_b = retention_gradient.get("lora_b_gradient_l2_by_target")
    return bool(
        retention_gradient.get("measured_from_freshly_zeroed_gradients") is True
        and type(retention_gradient.get("lora_a_exact_zero")) is bool
        and isinstance(retention_b, dict)
        and set(retention_b) == set(TARGET_MODULES)
        and all(
            _finite_number(value) and float(value) >= 0.0
            for value in retention_b.values()
        )
        and _finite_number(metrics.get("retention_self_kl"))
        and float(metrics["retention_self_kl"]) >= -1e-12
        and abs(float(metrics["retention_self_kl"]))
        <= float(thresholds["retention_self_kl_abs_max"])
        and metrics.get("passed") is True
    )


def _path_inventory_sha256(paths: list[str]) -> str:
    payload = json.dumps(paths, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _forbidden_evaluation_path(path: str) -> bool:
    components = set(Path(path).parts)
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


def _gradient_report_consistency(report: dict[str, Any]) -> bool:
    """Recompute all duplicated terminal gradient and objective summaries."""

    b_gradients = report.get("v6_lora_b_gradient_l2_by_target")
    a_gradients = report.get("v6_lora_a_gradient_l2_expected_zero_by_target")
    modules = report.get("v6_gradient_by_module")
    if (
        not isinstance(b_gradients, dict)
        or not isinstance(a_gradients, dict)
        or not isinstance(modules, dict)
        or set(b_gradients) != set(TARGET_MODULES)
        or set(a_gradients) != set(TARGET_MODULES)
        or set(modules) != set(TARGET_MODULES)
    ):
        return False
    global_squared = 0.0
    for target in TARGET_MODULES:
        module = modules[target]
        a_value = a_gradients[target]
        b_value = b_gradients[target]
        if (
            not isinstance(module, dict)
            or set(module) != {"lora_a", "lora_b", "total_l2"}
            or not _finite_number(a_value)
            or not _finite_number(b_value)
            or float(a_value) < 0.0
            or float(b_value) < 0.0
            or module.get("lora_a") != a_value
            or module.get("lora_b") != b_value
            or not _same_number(
                module.get("total_l2"),
                math.hypot(float(a_value), float(b_value)),
            )
        ):
            return False
        global_squared += float(a_value) ** 2 + float(b_value) ** 2
    global_gradient_l2 = math.sqrt(global_squared)
    gate = report.get("gradient_equivalence")
    if not isinstance(gate, dict):
        return False
    comparisons = gate.get("gradient_comparisons")
    aggregate = comparisons.get("aggregate") if isinstance(comparisons, dict) else None
    objective = gate.get("objective_values")
    return bool(
        _same_number(report.get("v6_gradient_l2"), global_gradient_l2)
        and isinstance(aggregate, dict)
        and aggregate.get("tail_lora_b_gradient_l2_by_target") == b_gradients
        and aggregate.get("tail_lora_a_gradient_l2_by_target") == a_gradients
        and _same_number(aggregate.get("tail_norm"), global_gradient_l2)
        and isinstance(objective, dict)
        and report.get("contrastive_correct_nll")
        == objective.get("tail_correct_nll")
        and report.get("contrastive_wrong_nll") == objective.get("tail_wrong_nll")
        and report.get("contrastive_margin") == objective.get("tail_margin")
        and report.get("broad_nll") == objective.get("tail_broad_nll")
        and report.get("retention_self_kl") == gate.get("retention_self_kl")
    )


def authenticate_v6_1_passing_smoke() -> tuple[dict[str, Any], str]:
    """Authenticate the only V6.1 artifact that may unlock a later trainer."""

    source = _resolve(MPS_SMOKE_REPORT)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V6.1 real MPS smoke report is missing or unsafe")
    report = json.loads(source.read_text(encoding="utf-8"))
    _attempt, attempt_sha = authenticate_v6_1_mps_smoke_attempt()
    expected_fields = {
        "schema_version",
        "artifact",
        "status",
        "passed",
        "authorization_sha256",
        "attempt_sha256",
        "v6_parent_preregistration_sha256",
        "v6_parent_release_sha256",
        "v6_parent_terminal_failure_sha256",
        "device",
        "software_versions",
        "full_model_loaded",
        "mps_used",
        "optimizer_constructed",
        "optimizer_steps",
        "training_executed",
        "checkpoint_published",
        "objective_equivalence",
        "gradient_equivalence",
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
        raise ValueError("V6.1 real MPS smoke report schema changed")
    release_sha = sha256_file(MPS_SMOKE_RELEASE)
    b_gradients = report.get("v6_lora_b_gradient_l2_by_target")
    a_gradients = report.get("v6_lora_a_gradient_l2_expected_zero_by_target")
    roundtrip = report.get("joint_state_roundtrip")
    loaded_files = report.get("loaded_files")
    memory = report.get("memory")
    numeric_losses = (
        report.get("v6_gradient_l2"),
        report.get("contrastive_correct_nll"),
        report.get("contrastive_wrong_nll"),
        report.get("contrastive_margin"),
        report.get("broad_nll"),
        report.get("retention_self_kl"),
        report.get("elapsed_seconds"),
    )
    gradient_summary_consistent = _gradient_report_consistency(report)
    required = (
        report.get("schema_version") == 1
        and report.get("artifact") == f"{ARTIFACT}_real_mps_smoke"
        and report.get("status") == "passed"
        and report.get("passed") is True
        and report.get("authorization_sha256") == release_sha
        and report.get("attempt_sha256") == attempt_sha
        and report.get("v6_parent_preregistration_sha256")
        == V6_PREREGISTRATION_SHA256
        and report.get("v6_parent_release_sha256") == V6_RELEASE_SHA256
        and report.get("v6_parent_terminal_failure_sha256")
        == V6_TERMINAL_FAILURE_SHA256
        and report.get("device") == "mps"
        and report.get("software_versions") == EXPECTED_SOFTWARE_VERSIONS
        and report.get("full_model_loaded") is True
        and report.get("mps_used") is True
        and report.get("optimizer_constructed") is False
        and report.get("optimizer_steps") == 0
        and report.get("training_executed") is False
        and report.get("checkpoint_published") is False
        and objective_equivalence_passes(report.get("objective_equivalence"))
        and gradient_equivalence_passes(report.get("gradient_equivalence"))
        and report.get("v6_zero_output_exact_noop") is True
        and report.get("v6_initial_state_sha256") == INITIAL_STATE_SHA256
        and report.get("both_v6_adapter_gradients_nonzero") is True
        and gradient_summary_consistent
        and isinstance(b_gradients, dict)
        and set(b_gradients) == set(TARGET_MODULES)
        and all(
            _finite_number(value) and float(value) > 0.0
            for value in b_gradients.values()
        )
        and isinstance(a_gradients, dict)
        and set(a_gradients) == set(TARGET_MODULES)
        and all(value == 0.0 for value in a_gradients.values())
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
        and report.get("joint_zero_output_structural_runtime_coexistence_passed")
        is True
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
        and loaded_files == sorted(set(loaded_files))
        and all(isinstance(path, str) and Path(path).is_absolute() for path in loaded_files)
        and not any(_forbidden_evaluation_path(path) for path in loaded_files)
        and report.get("loaded_file_count") == len(loaded_files)
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
        and memory.get("mps_driver_sample_count") == len(MPS_MEMORY_PHASES)
        and isinstance(memory.get("mps_driver_samples_by_phase"), dict)
        and set(memory.get("mps_driver_samples_by_phase", {})) == MPS_MEMORY_PHASES
        and all(
            type(value) is int and value >= 0
            for value in memory.get("mps_driver_samples_by_phase", {}).values()
        )
        and memory.get("mps_driver_allocated_bytes_sampled_peak")
        == max(memory.get("mps_driver_samples_by_phase", {}).values())
    )
    if not required:
        raise ValueError("V6.1 real MPS smoke did not pass every locked condition")
    return report, sha256_file(source)


__all__ = [
    "ARTIFACT",
    "EXPECTED_SOFTWARE_VERSIONS",
    "GRADIENT_EQUIVALENCE_THRESHOLDS",
    "INSTALLED_TRANSFORMERS_SOURCE_BINDINGS",
    "MODEL_WEIGHTS_BLOB_SHA256",
    "MODEL_WEIGHTS_SIZE_BYTES",
    "MPS_MEMORY_PHASES",
    "MPS_SMOKE_ATTEMPT",
    "MPS_SMOKE_RELEASE",
    "MPS_SMOKE_REPORT",
    "OBJECTIVE_EQUIVALENCE_THRESHOLDS",
    "V6_ATTEMPT_SHA256",
    "V6_PREREGISTRATION_SHA256",
    "V6_RELEASE_SHA256",
    "V6_TERMINAL_FAILURE_MESSAGE",
    "V6_TERMINAL_FAILURE_SHA256",
    "V6_TERMINAL_FAILURE_STATUS",
    "authenticate_v6_1_mps_smoke_attempt",
    "authenticate_v6_1_mps_smoke_release",
    "authenticate_v6_1_passing_smoke",
    "build_v6_1_mps_smoke_release",
    "claim_v6_1_mps_smoke_attempt",
    "gradient_equivalence_passes",
    "objective_equivalence_passes",
    "sha256_file",
    "write_v6_1_mps_smoke_release",
]
