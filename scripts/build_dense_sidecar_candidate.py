"""Build a runtime-only dense-sidecar candidate from immutable artifacts.

The builder combines an already trained scene/decoder adapter with a separately
calibrated numerical dense bridge.  It reads training metadata only in this
offline construction process; chat receives the sanitized runtime sidecar.
No oracle, QA, labels, captions, or text prototypes enter the output adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.language.lora import (
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
)
from semantic_3d_chat.scene_encoder.dense_alignment import (
    construct_dense_alignment,
    dense_alignment_settings,
    validate_dense_alignment_state,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(directory: Path, name: str, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, directory / name)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def build_candidate(
    *,
    config_path: Path,
    base_checkpoint: Path,
    bridge_path: Path,
    output: Path,
) -> dict:
    config = load_config(config_path)
    screen = config.get("v27_screen")
    if not isinstance(screen, dict):
        raise TypeError("V27 config requires a v27_screen contract")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate directory: {output}")

    base_adapter_path = base_checkpoint / "adapter.safetensors"
    base_metadata_path = base_checkpoint / TRAINING_METADATA_FILENAME
    if not base_adapter_path.is_file() or not base_metadata_path.is_file():
        raise FileNotFoundError("Base checkpoint is incomplete")
    if sha256_file(base_adapter_path) != screen["base_adapter_sha256"]:
        raise ValueError("Base adapter hash does not match the V27 contract")
    if sha256_file(bridge_path) != screen["calibrated_bridge_sha256"]:
        raise ValueError("Calibrated bridge hash does not match the V27 contract")

    base_tensors = load_file(base_adapter_path, device="cpu")
    bridge_tensors = load_file(bridge_path, device="cpu")
    overlap = sorted(set(base_tensors) & set(bridge_tensors))
    if overlap:
        raise ValueError(f"Base and bridge tensor names overlap: {overlap}")
    if set(bridge_tensors) != {
        "dense_aligner.alignment_a",
        "dense_aligner.alignment_b",
        "dense_aligner.architecture_marker",
        "dense_aligner.scaling",
    }:
        raise ValueError("Calibrated bridge has an unexpected tensor inventory")

    semantic_dim = 3072
    dense_aligner = construct_dense_alignment(config, semantic_dim=semantic_dim)
    if dense_aligner is None:
        raise ValueError("V27 requires an enabled dense alignment sidecar")
    dense_aligner.load_state_dict(
        {name.removeprefix("dense_aligner."): value for name, value in bridge_tensors.items()},
        strict=True,
    )
    dense_audit = validate_dense_alignment_state(
        dense_aligner,
        expected_parameter_count=24_576,
        context="V27 calibrated sidecar",
    )
    if dense_audit["state_sha256"] != screen["calibrated_state_sha256"]:
        raise ValueError("Calibrated dense state hash does not match the V27 contract")
    if dense_aligner.application_mode != "coverage_sidecar":
        raise ValueError("V27 candidate must use coverage_sidecar application mode")
    if dense_aligner.sidecar_scale != float(screen["selected_sidecar_scale"]):
        raise ValueError("Configured sidecar scale does not match the V27 contract")

    output.mkdir(parents=True)
    combined = {**base_tensors, **bridge_tensors}
    save_file(combined, output / "adapter.safetensors")

    metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    lora_parameter_counts = {
        name: sum(module_counts.values())
        for name, module_counts in metadata["lora_bank_parameter_counts"].items()
    }
    lora_settings = lora_banks_settings(config)
    lora_contract = lora_banks_checkpoint_contract(
        lora_settings,
        lora_banks_optimizer_settings(config, lora_settings),
        lora_parameter_counts,
    )
    metadata.update(
        {
            "config_hash": config_hash(config),
            "output_namespace": config["training"]["output_namespace"],
            "dense_alignment": dense_alignment_settings(config).contract(),
            "dense_alignment_parameter_count": dense_aligner.parameter_count,
            "dense_alignment_initial_state_sha256": dense_alignment_settings(
                config
            ).expected_initial_state_sha256,
            "dense_alignment_state_sha256": dense_aligner.state_sha256(),
            "all_voxels_transformed": True,
            "question_dependent_scene_processing": False,
            "lora": lora_contract,
            "lora_trainable_parameter_count": lora_contract[
                "trainable_adapter_parameter_count"
            ],
            "candidate_construction": {
                "schema_version": 1,
                "base_adapter_sha256": sha256_file(base_adapter_path),
                "calibrated_bridge_sha256": sha256_file(bridge_path),
                "base_semantic_path_modified": False,
                "application_mode": dense_aligner.application_mode,
                "sidecar_scale": dense_aligner.sidecar_scale,
                "oracle_loaded": False,
                "qa_loaded": False,
                "category_text_prototypes_serialized": False,
            },
        }
    )
    runtime_metadata = runtime_checkpoint_metadata(metadata)
    validate_runtime_checkpoint_metadata(runtime_metadata)
    _atomic_json(output, TRAINING_METADATA_FILENAME, metadata)
    _atomic_json(output, RUNTIME_METADATA_FILENAME, runtime_metadata)

    return {
        "schema_version": 1,
        "artifact": "v27_dense_sidecar_candidate",
        "config_hash": config_hash(config),
        "output": str(output.resolve()),
        "adapter_sha256": sha256_file(output / "adapter.safetensors"),
        "runtime_metadata_sha256": sha256_file(output / RUNTIME_METADATA_FILENAME),
        "tensor_count": len(combined),
        "dense_state_sha256": dense_aligner.state_sha256(),
        "application_mode": dense_aligner.application_mode,
        "sidecar_scale": dense_aligner.sidecar_scale,
        "base_semantic_path_modified": False,
        "oracle_loaded": False,
        "qa_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/gemma4_color_mirror_dense_sidecar_v27.yaml"),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("data_gemma4/checkpoints/gemma4_v24_shared_query/epoch_001"),
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=Path("reports/gemma4/artifacts/v26_dense_alignment_bridge.safetensors"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_gemma4/checkpoints/gemma4_v27_dense_sidecar/candidate_beta_010"),
    )
    args = parser.parse_args()
    report = build_candidate(
        config_path=args.config,
        base_checkpoint=args.base_checkpoint,
        bridge_path=args.bridge,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
