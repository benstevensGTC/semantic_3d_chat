"""Create-once correction and launch authenticator for the V80 CPU preflight.

The sealed v1 CPU artifact contains a poorly named boolean check whose literal
``true`` can be misread as saying that the real model received optimizer
updates.  Its explicit ``real_model.optimizer_updates`` value is zero.  This
addendum preserves the original bytes, corrects the ambiguous fact to false,
and content-addresses every model-free input needed before a real MPS smoke.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v80_atlas_attention_reader_preregistration import (
    CONFIG,
    EXPECTED_CONFIG_SHA256,
    atomic_create_json,
    load_v80_config,
    sha256_file,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256

CORRECTION: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v80_v75_atlas_attention_reader_cpu_preflight_correction_v2.json"
)
SUPERSEDED_CORRECTION: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v80_v75_atlas_attention_reader_cpu_preflight_correction_v1.json"
)
SUPERSEDED_CORRECTION_SHA256: Final[str] = (
    "3e5fb52561015de2a6c74e826f66d9c3c414c567f186f9e2a9a7748181a1523f"
)
PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v80_v75_atlas_attention_reader_preregistration.json"
)
CPU_PREFLIGHT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v80_v75_atlas_attention_reader_cpu_preflight.json"
)
PREREGISTRATION_SHA256: Final[str] = (
    "e44dc9aed1176cdfc30befe56d50e49a31f1638223a529a01bb086f5b3ea5894"
)
CPU_PREFLIGHT_SHA256: Final[str] = (
    "e56e4a8a4e9dc450988eb6d8e3788b2469f51433ab9c862d1f973c3608066c70"
)
V54_METADATA_SHA256: Final[str] = (
    "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd"
)
V75_METADATA_SHA256: Final[str] = (
    "a45a192d27336329580612524d43f71f08e3f472e5fe833747ffc1395e2aa2be"
)
PROBE_METADATA_SHA256: Final[str] = (
    "3e736940f4c83b55e96aa5e36f6774fd007454508722f5b25ddc44f298c2518d"
)
PREFIX_MANIFEST_SHA256: Final[str] = (
    "5a288a7fef65a957ba7b20132c63380cfadc7edbc37b32c1885037f939b9db61"
)

# The training module is deliberately not self-pinned here: it pins this
# addendum, records its own hash in the real smoke, and the bounded phase then
# requires that exact smoke-recorded hash.  This avoids a circular self-hash.
SOURCE_DEPENDENCIES: Final[tuple[str, ...]] = (
    "scripts/preflight_v80_v75_atlas_attention_reader.py",
    "scripts/preflight_v80_v75_atlas_attention_reader_correction.py",
    "scripts/run_v80_v75_atlas_attention_reader.py",
    "src/semantic_3d_chat/chat/file_audit.py",
    "src/semantic_3d_chat/chat/question_control_runtime.py",
    "src/semantic_3d_chat/chat/runtime.py",
    "src/semantic_3d_chat/chat/runtime_config.py",
    "src/semantic_3d_chat/evaluation/v75_fixed_atlas_behavior.py",
    "src/semantic_3d_chat/evaluation/v80_atlas_attention_reader_preregistration.py",
    "src/semantic_3d_chat/evaluation/v80_cpu_preflight_correction.py",
    "src/semantic_3d_chat/language/gemma4_answer_tail.py",
    "src/semantic_3d_chat/language/local_lm.py",
    "src/semantic_3d_chat/language/lora.py",
    "src/semantic_3d_chat/language/prefix_injection.py",
    "src/semantic_3d_chat/language/v80_atlas_attention_reader.py",
    "src/semantic_3d_chat/scene_encoder/fixed_prefix_atlas.py",
    "src/semantic_3d_chat/scene_encoder/fixed_prefix_atlas_v75.py",
    "src/semantic_3d_chat/scene_encoder/question_control_v75.py",
    "src/semantic_3d_chat/training/train_adapter.py",
    "src/semantic_3d_chat/training/train_question_control_v56.py",
    "src/semantic_3d_chat/training/train_question_control_v73.py",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"V80 correction input is unavailable: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V80 correction JSON must be an object: {source}")
    return value


def _runtime_authentication(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    v54_root = _resolve(inputs["base_checkpoint"])
    v75_root = _resolve(inputs["atlas_controller"])
    if {path.name for path in v54_root.iterdir()} != {
        "adapter.safetensors",
        "runtime_metadata.json",
    }:
        raise ValueError("V80 V54 runtime inventory changed")
    if {path.name for path in v75_root.iterdir()} != {
        "control.safetensors",
        "runtime_metadata.json",
    }:
        raise ValueError("V80 V75 runtime inventory changed")

    v54_metadata_path = v54_root / "runtime_metadata.json"
    v75_metadata_path = v75_root / "runtime_metadata.json"
    v54 = _read_json(v54_metadata_path)
    v75 = _read_json(v75_metadata_path)
    if sha256_file(v54_metadata_path) != V54_METADATA_SHA256:
        raise ValueError("V80 V54 runtime metadata changed")
    if sha256_file(v75_metadata_path) != V75_METADATA_SHA256:
        raise ValueError("V80 V75 runtime metadata changed")
    expected_v54 = {
        "schema_version": 1,
        "language_backend": "gemma4",
        "language_model_id": inputs["model_id"],
        "language_revision": inputs["model_revision"],
        "language_hidden_dim": 1536,
        "language_aligned_tail_dim": 1536,
        "scene_latents": 256,
        "freeze_scene_adapter": True,
        "question_dependent_scene_processing": False,
        "lora_trainable_parameter_count": 0,
    }
    if any(v54.get(key) != value for key, value in expected_v54.items()):
        raise ValueError("V80 V54 runtime contract changed")
    expected_v75 = {
        "schema_version": 75,
        "architecture": "dense_full_scene_continuous_control_v75",
        "weights_sha256": inputs["atlas_controller_weights_sha256"],
        "hidden_size": 1536,
        "environment_latents": 256,
        "query_count": 4,
        "control_tokens": 4,
        "complete_scene_prefix_required": True,
        "all_environment_latents_attended": True,
        "latent_selection_or_top_k_used": False,
        "question_dependent_scene_retrieval": False,
        "question_only_output_path_exists": False,
        "saved_runtime_training_gate_passed": True,
        "oracle_runtime_loaded": False,
        "answer_text_runtime_loaded": False,
        "training_answers_runtime_loaded": False,
    }
    if any(v75.get(key) != value for key, value in expected_v75.items()):
        raise ValueError("V80 V75 runtime contract changed")
    adapter = v54_root / "adapter.safetensors"
    control = v75_root / "control.safetensors"
    if sha256_file(adapter) != inputs["base_adapter_sha256"]:
        raise ValueError("V80 V54 adapter changed")
    if sha256_file(control) != inputs["atlas_controller_weights_sha256"]:
        raise ValueError("V80 V75 controller weights changed")
    return {
        "v54": {
            "metadata_sha256": V54_METADATA_SHA256,
            "adapter_sha256": inputs["base_adapter_sha256"],
            "critical_contract": expected_v54,
        },
        "v75": {
            "metadata_sha256": V75_METADATA_SHA256,
            "control_sha256": inputs["atlas_controller_weights_sha256"],
            "critical_contract": expected_v75,
        },
    }


def _probe_authentication(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _resolve(config["inputs"]["numeric_probe_bank"])
    if {path.name for path in root.iterdir()} != {
        "probes.safetensors",
        "runtime_metadata.json",
    }:
        raise ValueError("V80 probe-bank inventory changed")
    metadata_path = root / "runtime_metadata.json"
    probes_path = root / "probes.safetensors"
    metadata = _read_json(metadata_path)
    if sha256_file(metadata_path) != PROBE_METADATA_SHA256:
        raise ValueError("V80 probe metadata changed")
    if sha256_file(probes_path) != config["inputs"]["numeric_probe_file_sha256"]:
        raise ValueError("V80 probe tensor file changed")
    exact = {
        "artifact": "v75_fixed_atlas_numeric_probe_bank_v1",
        "schema_version": 1,
        "status": "historical_internal_diagnostic_not_promoted",
        "dtype": "torch.float32",
        "hidden_size": 1536,
        "probe_count": 96,
        "source_scope": "v73_historical_optimization_fold_only",
        "source_train_pair_count": 12,
        "source_train_scene_count": 24,
        "source_train_row_count": 576,
        "source_qa_sha256": config["inputs"]["historical_training_qa_sha256"],
        "source_v73_config_sha256": config["inputs"]["source_v73_config_sha256"],
        "model_revision": config["inputs"]["model_revision"],
        "model_file_sha256": config["inputs"]["model_file_sha256"],
        "questions_or_answers_serialized": False,
        "answer_codebook_serialized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    if any(metadata.get(key) != value for key, value in exact.items()):
        raise ValueError("V80 probe metadata contract changed")
    with safe_open(str(probes_path), framework="pt", device="cpu") as archive:
        if list(archive.keys()) != ["probe_embeddings"]:
            raise ValueError("V80 probe tensor inventory changed")
        probe_slice = archive.get_slice("probe_embeddings")
        if probe_slice.get_shape() != [96, 1536] or probe_slice.get_dtype() != "F32":
            raise ValueError("V80 probe tensor shape/dtype changed")
    probes = load_file(str(probes_path), device="cpu")["probe_embeddings"]
    if (
        probes.dtype != torch.float32
        or tuple(probes.shape) != (96, 1536)
        or not bool(torch.isfinite(probes).all())
        or tensor_sha256(probes) != metadata.get("probe_tensor_sha256")
    ):
        raise ValueError("V80 probe tensor content changed")
    return {
        "metadata_sha256": PROBE_METADATA_SHA256,
        "file_sha256": config["inputs"]["numeric_probe_file_sha256"],
        "tensor_sha256": metadata["probe_tensor_sha256"],
        "shape": [96, 1536],
        "dtype": "float32",
        "critical_contract": exact,
    }


def _prefix_authentication(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _resolve(config["inputs"]["base_prefix_cache"])
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if sha256_file(manifest_path) != PREFIX_MANIFEST_SHA256:
        raise ValueError("V80 base-prefix manifest changed")
    entries = manifest.get("scenes")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact") != "question_independent_scene_prefix_cache_v1"
        or manifest.get("scene_count") != 40
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_scene_retrieval") is not False
        or manifest.get("complete_scene_prefixes") is not True
        or manifest.get("environmental_text_inputs") != []
        or not isinstance(entries, Mapping)
        or len(entries) != 40
    ):
        raise ValueError("V80 base-prefix manifest contract changed")
    expected_inventory = {"manifest.json"} | {
        f"{scene_id}.safetensors" for scene_id in entries
    }
    if {path.name for path in root.iterdir()} != expected_inventory:
        raise ValueError("V80 base-prefix file inventory changed")

    authenticated: dict[str, dict[str, Any]] = {}
    entry_fields = {
        "dtype",
        "file_sha256",
        "file_size_bytes",
        "filename",
        "prefix_sha256",
        "shape",
    }
    for scene_id in sorted(entries):
        entry = entries[scene_id]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != entry_fields
            or entry.get("filename") != f"{scene_id}.safetensors"
            or entry.get("shape") != [1, 258, 1536]
            or entry.get("dtype") != "bfloat16"
        ):
            raise ValueError(f"V80 base-prefix manifest entry changed: {scene_id}")
        path = root / str(entry["filename"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("file_size_bytes")
            or sha256_file(path) != entry.get("file_sha256")
        ):
            raise ValueError(f"V80 base-prefix file changed: {scene_id}")
        with safe_open(str(path), framework="pt", device="cpu") as archive:
            if list(archive.keys()) != ["scene_prefix"]:
                raise ValueError(f"V80 base-prefix tensor inventory changed: {scene_id}")
            tensor_slice = archive.get_slice("scene_prefix")
            if (
                tensor_slice.get_shape() != [1, 258, 1536]
                or tensor_slice.get_dtype() != "BF16"
            ):
                raise ValueError(f"V80 base-prefix tensor shape/dtype changed: {scene_id}")
        prefix = load_file(str(path), device="cpu")["scene_prefix"]
        if (
            prefix.dtype != torch.bfloat16
            or tuple(prefix.shape) != (1, 258, 1536)
            or not bool(torch.isfinite(prefix).all())
            or prefix_sha256(prefix) != entry.get("prefix_sha256")
        ):
            raise ValueError(f"V80 base-prefix tensor content changed: {scene_id}")
        authenticated[scene_id] = dict(entry)
    return {
        "manifest_sha256": PREFIX_MANIFEST_SHA256,
        "scene_count": 40,
        "shape": [1, 258, 1536],
        "dtype": "bfloat16",
        "entries": authenticated,
    }


def build_correction() -> dict[str, Any]:
    """Rebuild the deterministic correction against current local bytes."""

    config = load_v80_config(CONFIG)
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("V80 sealed preregistration changed")
    if sha256_file(CPU_PREFLIGHT) != CPU_PREFLIGHT_SHA256:
        raise ValueError("V80 sealed CPU preflight changed")
    if sha256_file(SUPERSEDED_CORRECTION) != SUPERSEDED_CORRECTION_SHA256:
        raise ValueError("V80 superseded correction changed")
    preregistration = _read_json(PREREGISTRATION)
    preflight = _read_json(CPU_PREFLIGHT)
    authoritative = {
        "loaded": False,
        "gradient_smoke_run": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
    }
    if (
        preregistration.get("training_executed") is not False
        or preregistration.get("optimizer_constructed") is not False
        or preflight.get("passed") is not True
        or preflight.get("checks", {}).get("optimizer_updates_on_real_model") is not True
        or preflight.get("real_model") != authoritative
    ):
        raise ValueError("V80 original preflight evidence does not match the correction")
    runtime = _runtime_authentication(config)
    probes = _probe_authentication(config)
    prefixes = _prefix_authentication(config)
    source_hashes = {path: sha256_file(path) for path in SOURCE_DEPENDENCIES}
    checks = {
        "original_artifacts_authenticated": True,
        "ambiguous_boolean_detected": True,
        "corrected_real_model_update_fact_is_false": True,
        "authoritative_real_model_update_count_is_zero": True,
        "canonical_config_authenticated": True,
        "v54_runtime_metadata_and_weights_authenticated": True,
        "v75_runtime_metadata_and_weights_authenticated": True,
        "probe_metadata_and_tensor_authenticated": True,
        "all_40_prefix_manifest_entries_authenticated": len(prefixes["entries"]) == 40,
        "all_dependency_sources_content_addressed": len(source_hashes)
        == len(SOURCE_DEPENDENCIES),
        "superseded_correction_authenticated": True,
        "v75_held_subset_loader_compatibility_preserved": True,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
    }
    # The final three values above are facts rather than pass booleans, so do
    # not feed the two intentional false values into the aggregate pass bit.
    passed = all(
        bool(value)
        for key, value in checks.items()
        if key not in {"model_loaded", "optimizer_constructed", "optimizer_updates"}
    ) and checks["optimizer_updates"] == 0
    return {
        "schema_version": 1,
        "artifact": "gemma4_v80_v75_atlas_attention_reader_cpu_preflight_correction_v2",
        "status": "sealed_superseding_correction_and_launch_input_authenticator_model_still_not_loaded",
        "passed": passed,
        "authentication_method": "sha256_content_addressed_addendum_pinned_by_real_runner",
        "config": {"path": CONFIG, "sha256": EXPECTED_CONFIG_SHA256},
        "original_artifacts": {
            "preregistration": {
                "path": PREREGISTRATION,
                "sha256": PREREGISTRATION_SHA256,
            },
            "superseded_correction_v1": {
                "path": SUPERSEDED_CORRECTION,
                "sha256": SUPERSEDED_CORRECTION_SHA256,
                "reason": "superseded_to_restore_v75_16_scene_subset_loader_compatibility",
            },
            "cpu_preflight": {
                "path": CPU_PREFLIGHT,
                "sha256": CPU_PREFLIGHT_SHA256,
            },
        },
        "correction": {
            "field": "checks.optimizer_updates_on_real_model",
            "original_value": True,
            "corrected_value": False,
            "classification": "misnamed_unconditional_pass_boolean_not_an_update_fact",
            "authoritative_field": "real_model.optimizer_updates",
            "authoritative_value": 0,
            "original_artifact_overwritten": False,
        },
        "authoritative_real_model_state": authoritative,
        "runtime_authentication": runtime,
        "probe_authentication": probes,
        "prefix_authentication": prefixes,
        "source_dependency_sha256": source_hashes,
        "source_hash_chain": {
            "trainer_self_hashing": "recorded_by_gradient_smoke",
            "bounded_phase_trainer_authentication": "must_equal_gradient_smoke_hash",
            "circular_self_hash_avoided": True,
        },
        "checks": checks,
        "checkpoint_published": False,
        "runtime_promotion_authorized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
    }


def validate_correction(
    path: str | Path = CORRECTION, *, expected_sha256: str
) -> dict[str, Any]:
    source = _resolve(path)
    if source != _resolve(CORRECTION):
        raise ValueError("V80 refuses a noncanonical correction artifact path")
    if sha256_file(source) != expected_sha256:
        raise ValueError("V80 correction artifact bytes changed")
    observed = _read_json(source)
    expected = build_correction()
    if observed != expected or observed.get("passed") is not True:
        raise ValueError("V80 correction/authentication evidence failed")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    payload = build_correction()
    if args.write:
        path, digest = atomic_create_json(CORRECTION, payload)
        print(json.dumps({"path": str(path), "sha256": digest}, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CORRECTION",
    "CPU_PREFLIGHT_SHA256",
    "PREREGISTRATION_SHA256",
    "SOURCE_DEPENDENCIES",
    "SUPERSEDED_CORRECTION",
    "SUPERSEDED_CORRECTION_SHA256",
    "build_correction",
    "main",
    "validate_correction",
]
