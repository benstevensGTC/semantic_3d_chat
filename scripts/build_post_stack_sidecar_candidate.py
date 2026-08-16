"""Build the V28 zero-output post-stack dense-sidecar candidate.

This offline builder combines three hash-pinned numerical artifacts:

* the immutable V24 scene/decoder adapter,
* the calibrated V26 all-voxel dense bridge, and
* a deterministically initialized, exact-zero post-stack sidecar adapter.

The chat runtime receives only the combined tensors and sanitized runtime
metadata.  Oracle files, QA records, labels, captions, category prototypes,
and training selection reports are neither opened nor serialized here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.runtime import validate_checkpoint_contract
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
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    DenseSidecarAdapter,
    construct_dense_sidecar_adapter,
    dense_sidecar_adapter_settings,
    validate_dense_sidecar_adapter_state,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)

_BRIDGE_TENSOR_NAMES = frozenset(
    {
        "dense_aligner.alignment_a",
        "dense_aligner.alignment_b",
        "dense_aligner.architecture_marker",
        "dense_aligner.scaling",
    }
)
_ZERO_OUTPUT_EQUIVALENCE = {
    "verified": True,
    "base": "loaded_v24_post_signed_x_scene_tokens",
    "question_dependent_scene_processing": False,
    "all_scene_slots_accounted": True,
    "all_voxels_covered": True,
    "application_order": "after_global_and_signed_x_before_prefix_composer",
}
_BASE_STATE_BINDINGS = {
    "base_scene_state_sha256": "frozen_scene_state_sha256",
    "base_global_scene_residual_state_sha256": "global_scene_residual_state_sha256",
    "base_signed_x_scene_residual_state_sha256": "signed_x_scene_residual_state_sha256",
}


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of one local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(directory: Path, name: str, payload: Mapping[str, Any]) -> None:
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


def _require_v28_screen(config: Mapping[str, Any]) -> dict[str, Any]:
    screen = config.get("v28_screen")
    if not isinstance(screen, dict):
        raise TypeError("V28 config requires a v28_screen contract")
    if screen.get("schema_version") != 1:
        raise ValueError("Unsupported v28_screen schema")
    if screen.get("zero_output_equivalence") != _ZERO_OUTPUT_EQUIVALENCE:
        raise ValueError("V28 zero-output equivalence contract is not exact")
    return screen


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _validate_base_metadata(
    metadata: Mapping[str, Any], screen: Mapping[str, Any]
) -> None:
    if metadata.get("schema_version") != 3:
        raise ValueError("V28 requires a schema-3 V24 base checkpoint")
    if metadata.get("semantic_dim") != 3072:
        raise ValueError("V28 requires the 3072D high-fidelity V24 semantic map")
    if metadata.get("language_hidden_dim") != 1536:
        raise ValueError("V28 requires the 1536D Gemma 4 language interface")
    if metadata.get("scene_latents") != 256:
        raise ValueError("V28 requires all 256 V24 scene slots")
    if "dense_alignment" in metadata or "dense_sidecar_adapter" in metadata:
        raise ValueError("V24 base metadata unexpectedly contains a dense sidecar")
    if metadata.get("question_dependent_scene_processing") is not False:
        raise ValueError("V24 base is not attested question-independent")
    for screen_key, metadata_key in _BASE_STATE_BINDINGS.items():
        if metadata.get(metadata_key) != screen.get(screen_key):
            raise ValueError(
                f"V24 base state binding mismatch: {metadata_key} does not match "
                f"v28_screen.{screen_key}"
            )


def _verify_exact_zero_identity(adapter: DenseSidecarAdapter) -> dict[str, Any]:
    """Prove the installed module is a bit-identical no-op at construction."""

    latent_count = int(adapter.latent_count)
    scene_dim = int(adapter.scene_dim)
    values = torch.linspace(
        -1.0,
        1.0,
        steps=latent_count * scene_dim,
        dtype=torch.float32,
    ).reshape(1, latent_count, scene_dim)
    sidecar = torch.flip(values, dims=(1, 2)).contiguous()
    with torch.inference_mode():
        delta = adapter.residual_delta(values, sidecar)
        output = adapter(values, sidecar)
    verified = bool(torch.count_nonzero(delta).item() == 0 and torch.equal(output, values))
    if not verified:
        raise ValueError("Dense sidecar adapter failed exact-zero identity verification")
    return {
        **_ZERO_OUTPUT_EQUIVALENCE,
        "representative_tensor_shape": list(values.shape),
        "delta_nonzero_count": int(torch.count_nonzero(delta).item()),
        "bit_identical_output": True,
    }


def build_candidate(
    *,
    config_path: Path,
    base_checkpoint: Path,
    bridge_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Construct one immutable V28 candidate and return its audit report."""

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate directory: {output}")

    config = load_config(config_path)
    screen = _require_v28_screen(config)
    base_adapter_path = base_checkpoint / "adapter.safetensors"
    base_metadata_path = base_checkpoint / TRAINING_METADATA_FILENAME
    if not base_adapter_path.is_file() or not base_metadata_path.is_file():
        raise FileNotFoundError("Base checkpoint is incomplete")
    if not bridge_path.is_file():
        raise FileNotFoundError(f"Calibrated bridge does not exist: {bridge_path}")

    pinned_files = {
        "base adapter": (base_adapter_path, screen.get("base_adapter_sha256")),
        "base metadata": (base_metadata_path, screen.get("base_metadata_sha256")),
        "calibrated bridge": (bridge_path, screen.get("calibrated_bridge_sha256")),
    }
    for label, (path, expected_hash) in pinned_files.items():
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"{label.capitalize()} hash does not match the V28 contract: "
                f"expected={expected_hash} observed={observed_hash}"
            )

    base_metadata = _read_json_object(base_metadata_path)
    _validate_base_metadata(base_metadata, screen)
    base_tensors = load_file(base_adapter_path, device="cpu")
    if any(
        name.startswith(("dense_aligner.", "dense_sidecar_adapter."))
        for name in base_tensors
    ):
        raise ValueError("Immutable V24 adapter unexpectedly contains V26/V28 tensors")

    bridge_tensors = load_file(bridge_path, device="cpu")
    if set(bridge_tensors) != _BRIDGE_TENSOR_NAMES:
        raise ValueError(
            "Calibrated bridge has an unexpected tensor inventory: "
            f"{sorted(bridge_tensors)}"
        )
    if set(base_tensors) & set(bridge_tensors):
        raise ValueError("V24 and V26 tensor inventories overlap")

    semantic_dim = int(base_metadata["semantic_dim"])
    language_hidden_dim = int(base_metadata["language_hidden_dim"])
    latent_count = int(base_metadata["scene_latents"])
    dense_aligner = construct_dense_alignment(config, semantic_dim=semantic_dim)
    if dense_aligner is None:
        raise ValueError("V28 requires an enabled calibrated dense alignment bridge")
    dense_aligner.load_state_dict(
        {name.removeprefix("dense_aligner."): value for name, value in bridge_tensors.items()},
        strict=True,
    )
    dense_audit = validate_dense_alignment_state(
        dense_aligner,
        expected_parameter_count=int(screen["dense_alignment_parameter_count"]),
        context="V28 frozen calibrated dense bridge",
    )
    if dense_audit["state_sha256"] != screen.get("calibrated_state_sha256"):
        raise ValueError("Calibrated dense state hash does not match the V28 contract")
    if dense_aligner.application_mode != screen.get("dense_alignment_application_mode"):
        raise ValueError("Dense alignment application mode does not match V28")
    if dense_aligner.sidecar_scale != float(screen["dense_alignment_sidecar_scale"]):
        raise ValueError("Dense alignment sidecar scale does not match V28")
    if dense_aligner.application_mode != "coverage_sidecar" or dense_aligner.sidecar_scale != 0.0:
        raise ValueError("V28 requires coverage_sidecar routing with exact scale 0.0")

    sidecar_adapter = construct_dense_sidecar_adapter(
        config,
        scene_dim=language_hidden_dim,
        latent_count=latent_count,
    )
    if sidecar_adapter is None:
        raise ValueError("V28 requires an enabled post-stack dense sidecar adapter")
    sidecar_audit = validate_dense_sidecar_adapter_state(
        sidecar_adapter,
        expected_parameter_count=int(screen["dense_sidecar_adapter_parameter_count"]),
        expected_state_sha256=str(screen["dense_sidecar_adapter_initial_state_sha256"]),
        context="V28 zero-output post-stack adapter",
    )
    sidecar_settings = dense_sidecar_adapter_settings(config)
    if sidecar_audit["state_sha256"] != sidecar_settings.expected_initial_state_sha256:
        raise ValueError("Configured V28 initial sidecar hash is not the constructed state")
    if not sidecar_audit["output_projection_exact_zero"]:
        raise ValueError("V28 learned output route does not initialize to exact zero")
    if not sidecar_audit["channel_gain_exact_zero"]:
        raise ValueError("V28 direct sidecar route does not initialize to exact zero")
    zero_output_audit = _verify_exact_zero_identity(sidecar_adapter)

    sidecar_tensors = {
        f"dense_sidecar_adapter.{name}": value.detach().cpu().contiguous()
        for name, value in sidecar_adapter.state_dict().items()
    }
    inventories = (set(base_tensors), set(bridge_tensors), set(sidecar_tensors))
    if any(inventories[index] & inventories[other] for index in range(3) for other in range(index)):
        raise ValueError("Candidate tensor inventories overlap")
    combined = {
        **{name: value.contiguous() for name, value in base_tensors.items()},
        **{name: value.contiguous() for name, value in bridge_tensors.items()},
        **sidecar_tensors,
    }

    lora_parameter_counts = {
        name: sum(module_counts.values())
        for name, module_counts in base_metadata["lora_bank_parameter_counts"].items()
    }
    lora_settings = lora_banks_settings(config)
    lora_contract = lora_banks_checkpoint_contract(
        lora_settings,
        lora_banks_optimizer_settings(config, lora_settings),
        lora_parameter_counts,
    )
    dense_settings = dense_alignment_settings(config)
    sidecar_contract = sidecar_settings.contract()
    metadata = dict(base_metadata)
    metadata.update(
        {
            "config_hash": config_hash(config),
            "output_namespace": config["training"]["output_namespace"],
            "dense_alignment": dense_settings.contract(),
            "dense_alignment_parameter_count": dense_aligner.parameter_count,
            "dense_alignment_initial_state_sha256": (
                dense_settings.expected_initial_state_sha256
            ),
            "dense_alignment_state_sha256": dense_aligner.state_sha256(),
            "all_voxels_transformed": True,
            "dense_sidecar_adapter": sidecar_contract,
            "dense_sidecar_adapter_parameter_count": sidecar_adapter.parameter_count,
            "dense_sidecar_adapter_initial_state_sha256": (
                sidecar_settings.expected_initial_state_sha256
            ),
            "dense_sidecar_adapter_state_sha256": sidecar_adapter.state_sha256(),
            "dense_sidecar_adapter_zero_output_equivalence": zero_output_audit,
            "frozen_dense_alignment_state_sha256": dense_aligner.state_sha256(),
            "question_dependent_scene_processing": False,
            "lora": lora_contract,
            "lora_trainable_parameter_count": lora_contract[
                "trainable_adapter_parameter_count"
            ],
            "candidate_construction": {
                "schema_version": 1,
                "artifact": "v28_post_stack_sidecar_candidate",
                "base_adapter_sha256": sha256_file(base_adapter_path),
                "base_metadata_sha256": sha256_file(base_metadata_path),
                "calibrated_bridge_sha256": sha256_file(bridge_path),
                "calibrated_dense_state_sha256": dense_aligner.state_sha256(),
                "dense_sidecar_adapter_initial_state_sha256": (
                    sidecar_adapter.state_sha256()
                ),
                "base_semantic_path_modified": False,
                "application_order": _ZERO_OUTPUT_EQUIVALENCE["application_order"],
                "dense_alignment_application_mode": dense_aligner.application_mode,
                "dense_alignment_sidecar_scale": dense_aligner.sidecar_scale,
                "oracle_loaded": False,
                "qa_loaded": False,
                "category_text_prototypes_serialized": False,
            },
        }
    )
    runtime_metadata = runtime_checkpoint_metadata(metadata)
    validate_runtime_checkpoint_metadata(runtime_metadata)
    validate_checkpoint_contract(
        runtime_metadata,
        config,
        semantic_dim=semantic_dim,
        language_hidden_dim=language_hidden_dim,
        lora_parameter_counts=lora_parameter_counts,
        dense_alignment_parameter_count=dense_aligner.parameter_count,
        dense_sidecar_adapter_parameter_count=sidecar_adapter.parameter_count,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.build.", dir=output.parent))
    try:
        save_file(combined, temporary / "adapter.safetensors")
        _atomic_json(temporary, TRAINING_METADATA_FILENAME, metadata)
        _atomic_json(temporary, RUNTIME_METADATA_FILENAME, runtime_metadata)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite candidate directory: {output}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "schema_version": 1,
        "artifact": "v28_post_stack_sidecar_candidate",
        "config_hash": config_hash(config),
        "output": str(output.resolve()),
        "adapter_sha256": sha256_file(output / "adapter.safetensors"),
        "runtime_metadata_sha256": sha256_file(output / RUNTIME_METADATA_FILENAME),
        "tensor_count": len(combined),
        "base_tensor_count": len(base_tensors),
        "dense_tensor_count": len(bridge_tensors),
        "sidecar_tensor_count": len(sidecar_tensors),
        "dense_state_sha256": dense_aligner.state_sha256(),
        "dense_sidecar_adapter_state_sha256": sidecar_adapter.state_sha256(),
        "dense_sidecar_adapter_parameter_count": sidecar_adapter.parameter_count,
        "application_order": _ZERO_OUTPUT_EQUIVALENCE["application_order"],
        "zero_output_equivalence_verified": True,
        "base_semantic_path_modified": False,
        "question_dependent_scene_processing": False,
        "all_voxels_covered": True,
        "oracle_loaded": False,
        "qa_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/gemma4_color_mirror_post_stack_sidecar_v28.yaml"
        ),
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
        default=Path(
            "data_gemma4/checkpoints/gemma4_v28_post_stack_sidecar/candidate_zero"
        ),
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
