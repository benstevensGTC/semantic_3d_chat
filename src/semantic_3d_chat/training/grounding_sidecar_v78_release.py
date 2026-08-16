"""Materialize a sanitized, base-bound optional V78 runtime release."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.scene_encoder.grounding_sidecar_v78 import (
    EXPECTED_CHECKPOINT_FILES,
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
)
from semantic_3d_chat.training.grounding_sidecar_v78 import sha256_file, validate_candidate

RUNTIME_ARTIFACT = "continuous_full_scene_grounding_sidecar_v78_runtime_diagnostic_v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def materialize_v78_runtime_release(
    source_candidate: str | Path,
    destination: str | Path,
    *,
    base_checkpoint: str | Path,
    runtime_config: str | Path,
) -> dict[str, Any]:
    """Copy only numeric weights and sanitized metadata into an exact release."""

    source_audit = validate_candidate(source_candidate)
    source = Path(source_candidate).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output.iterdir()} - EXPECTED_CHECKPOINT_FILES
    if unexpected:
        raise ValueError(f"V78 runtime release contains unexpected files: {sorted(unexpected)}")
    base_sha256, _ = checkpoint_fingerprint(base_checkpoint)
    config = load_runtime_config(runtime_config)
    config_sha256 = effective_runtime_config_sha256(config)
    source_metadata = source_audit["metadata"]
    weights_source = source / WEIGHTS_FILENAME
    weights_destination = output / WEIGHTS_FILENAME
    temporary_weights = weights_destination.with_name(
        f".{weights_destination.name}.tmp-{os.getpid()}"
    )
    shutil.copyfile(weights_source, temporary_weights)
    temporary_weights.replace(weights_destination)
    if sha256_file(weights_destination) != source_metadata["weights_sha256"]:
        raise RuntimeError("V78 runtime weights changed while materializing")
    prefix_manifest = (
        PROJECT_ROOT
        / "data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes/manifest.json"
    )
    manifest = json.loads(prefix_manifest.read_text(encoding="utf-8"))
    if sha256_file(prefix_manifest) != source_metadata["source_prefix_manifest_sha256"]:
        raise ValueError("V78 source prefix manifest identity changed")
    if manifest.get("base_runtime_config_sha256") != config_sha256:
        raise ValueError("V78 source prefixes used a different runtime config")
    metadata = {
        **source_metadata,
        "artifact": RUNTIME_ARTIFACT,
        "source_candidate_artifact": source_metadata["artifact"],
        "source_candidate_weights_sha256": source_metadata["weights_sha256"],
        "source_candidate_metadata_sha256": source_audit["metadata_sha256"],
        "source_prefix_base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        "base_checkpoint_sha256": base_sha256,
        "base_runtime_config_sha256": config_sha256,
        "optional_runtime_demo_authorized": True,
        "official_validation_evidence": False,
        "runtime_promotion_authorized": False,
    }
    _atomic_json(output / METADATA_FILENAME, metadata)
    if {path.name for path in output.iterdir()} != EXPECTED_CHECKPOINT_FILES:
        raise RuntimeError("V78 runtime release is not an exact two-file inventory")
    return {
        "directory": str(output),
        "files": sorted(EXPECTED_CHECKPOINT_FILES),
        "weights_sha256": sha256_file(weights_destination),
        "metadata_sha256": sha256_file(output / METADATA_FILENAME),
        "source_candidate_metadata_sha256": source_audit["metadata_sha256"],
        "base_checkpoint_sha256": base_sha256,
        "base_runtime_config_sha256": config_sha256,
    }


__all__ = ["RUNTIME_ARTIFACT", "materialize_v78_runtime_release"]
