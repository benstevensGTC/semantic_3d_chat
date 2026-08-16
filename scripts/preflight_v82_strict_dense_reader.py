#!/usr/bin/env python3
"""Model-free structural, gradient, and artifact preflight for V82."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_MEMORY_TOKENS,
    HIDDEN_SIZE,
    bind_fixed_prefix_before_question_v81,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.language.v82_dense_learned_reader import (
    ATLAS_DIRECT_MIX,
    BASE_DIRECT_MIX,
    TRAINABLE_PARAMETER_COUNT,
    DenseLearnedSceneReaderV82,
    wrong_scene_contrast_loss_v82,
)
from semantic_3d_chat.training.v82_reader_artifacts import (
    CANDIDATE_METADATA_FILENAME,
    load_v82_candidate,
    save_v82_candidate,
    sha256_file_v82,
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v82"}:
        raise ValueError("V82 preflight config must contain exactly v82")
    value = payload["v82"]
    if not isinstance(value, Mapping):
        raise TypeError("V82 preflight config payload changed")
    expected = {
        "schema_version": 82,
        "artifact": "gemma4_v82_strict_dense_learned_reader_v1",
        "status": "implemented_model_free_preflight_only_fit_not_executed",
        "seed": 820082,
    }
    if any(value.get(field) != item for field, item in expected.items()):
        raise ValueError("V82 preflight identity changed")
    memory = value.get("immutable_memory")
    reader = value.get("reader")
    split = value.get("split")
    if not all(isinstance(item, Mapping) for item in (memory, reader, split)):
        raise ValueError("V82 preflight contract sections are missing")
    required_memory = {
        "shape": [1, 738, 1536],
        "compiled_before_question": True,
        "reused_byte_identically_across_questions": True,
        "atlas_values": 384,
        "base_environment_latents": 256,
        "boi_eoi_are_payload": False,
        "probe_keys_are_payload": False,
        "strict_positive_payload_claim_for_all_738_tokens": False,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "environmental_text_inputs": [],
    }
    if any(memory.get(field) != item for field, item in required_memory.items()):
        raise ValueError("V82 immutable-memory contract changed")
    required_reader = {
        "architecture": "positive_floor_dual_bank_reader_v82",
        "internal_dimension": 64,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "atlas_uniform_floor_mass": 0.05,
        "base_uniform_floor_mass": 0.10,
        "atlas_direct_mix": ATLAS_DIRECT_MIX,
        "base_direct_mix": BASE_DIRECT_MIX,
        "bias_free": True,
        "zero_environment_produces_exact_zero_controls": True,
        "every_atlas_value_participates_with_positive_coefficient": True,
        "every_base_latent_participates_with_positive_coefficient": True,
    }
    if any(reader.get(field) != item for field, item in required_reader.items()):
        raise ValueError("V82 reader contract changed")
    required_split = {
        "train_pair_count": 12,
        "train_scene_count": 24,
        "train_row_count": 576,
        "development_pair_count": 8,
        "development_scene_count": 16,
        "development_row_count": 384,
        "pair_disjoint": True,
        "scene_disjoint": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
    }
    if any(split.get(field) != item for field, item in required_split.items()):
        raise ValueError("V82 split contract changed")
    for field, digest in {
        "v73_config": "v73_config_sha256",
        "historical_qa": "historical_qa_sha256",
    }.items():
        path_value = _resolve(value["sources"][field])
        if sha256_file_v82(path_value) != value["sources"][digest]:
            raise ValueError(f"V82 source changed: {field}")
    controller = _resolve(value["sources"]["v75_controller"])
    if (
        sha256_file_v82(controller / "control.safetensors")
        != value["sources"]["v75_controller_weights_sha256"]
        or sha256_file_v82(controller / "runtime_metadata.json")
        != value["sources"]["v75_controller_metadata_sha256"]
    ):
        raise ValueError("V82 source changed: v75_controller")
    prefix_manifest = _resolve(value["sources"]["prefix_cache"]) / "manifest.json"
    if sha256_file_v82(prefix_manifest) != value["sources"]["prefix_manifest_sha256"]:
        raise ValueError("V82 prefix manifest changed")
    return source, dict(value)


def _synthetic_memory(seed: int, *, batch_size: int = 2) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    memory = torch.randn(
        batch_size, 738, HIDDEN_SIZE, generator=generator, dtype=torch.float32
    ) * 0.035
    banks = split_v75_v2_prefix_v81(memory)
    if bool(torch.any(banks.probe_keys.norm(dim=-1) <= 1e-8)):
        raise RuntimeError("V82 synthetic probe unexpectedly contains a zero row")
    return memory.contiguous()


def _zero_payload_memory(memory: torch.Tensor) -> torch.Tensor:
    banks = split_v75_v2_prefix_v81(memory)
    atlas = torch.cat(
        (banks.probe_keys.unsqueeze(2), torch.zeros_like(banks.atlas_values)), dim=2
    ).reshape(memory.shape[0], ATLAS_MEMORY_TOKENS, HIDDEN_SIZE)
    return torch.cat(
        (banks.boi, atlas, torch.zeros_like(banks.base_latents), banks.eoi), dim=1
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_preflight(config_path: str | Path, *, output: str | Path | None) -> dict[str, Any]:
    started = time.perf_counter()
    config_file, config = _config(config_path)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    model = DenseLearnedSceneReaderV82(initialization_seed=seed)
    memory = _synthetic_memory(seed)
    query = torch.randn(2, HIDDEN_SIZE, generator=torch.Generator().manual_seed(seed + 1))
    binding = bind_fixed_prefix_before_question_v81(memory)
    first = model(memory, query, binding=binding)
    second = model(memory, -query, binding=binding)
    if binding.fixed_prefix_sha256 != bind_fixed_prefix_before_question_v81(memory).fixed_prefix_sha256:
        raise RuntimeError("V82 fixed memory hash changed across questions")
    atlas_floor = 0.05 / 96
    base_floor = 0.10 / 256
    if (
        float(first.atlas_weights.detach().min()) < atlas_floor - 1e-8
        or float(first.base_weights.detach().min()) < base_floor - 1e-8
        or not first.all_384_atlas_values_positive
        or not first.all_256_base_latents_positive
        or first.zero_environmental_payload
    ):
        raise RuntimeError("V82 positive-floor dense participation failed")

    zero_memory = _zero_payload_memory(memory)
    zero = model(
        zero_memory,
        query,
        binding=bind_fixed_prefix_before_question_v81(zero_memory),
    )
    zero_max = float(zero.controls.abs().max())
    if zero_max != 0.0 or not zero.zero_environmental_payload:
        raise RuntimeError("V82 zero-environment identity failed")

    # Force the hinge active and prove the scene-contrast objective reaches a
    # trainable reader parameter.  Targets are numeric tensors only.
    own_target = torch.randn(
        first.controls.shape,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    wrong_target = torch.randn(
        first.controls.shape,
        generator=torch.Generator().manual_seed(seed + 3),
    )
    contrast, preference = wrong_scene_contrast_loss_v82(
        first.controls, own_target, wrong_target, margin=3.0
    )
    model.zero_grad(set_to_none=True)
    contrast.backward()
    gradient_norms = {
        name: 0.0 if parameter.grad is None else float(parameter.grad.norm())
        for name, parameter in model.named_parameters()
    }
    nonzero_gradient_parameters = sorted(
        name for name, value in gradient_norms.items() if value > 0.0
    )
    if not bool(torch.isfinite(contrast)) or not nonzero_gradient_parameters:
        raise RuntimeError("V82 wrong-scene contrast produced no reader gradient")
    if any(not math.isfinite(value) for value in gradient_norms.values()):
        raise RuntimeError("V82 wrong-scene contrast gradient became nonfinite")

    # Round-trip a create-once candidate and repeat the exact-zero check after
    # strict runtime loading.  The temporary root lives under the real project
    # tree to avoid platform /var symlink ambiguity.
    scratch_parent = PROJECT_ROOT / "reports" / "gemma4" / "artifacts"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v82-preflight-", dir=scratch_parent) as raw:
        candidate_root = Path(raw) / "candidate"
        metadata = save_v82_candidate(
            candidate_root,
            model,
            training_cache_sha256="0" * 64,
            training_cache_metadata_sha256="1" * 64,
            fit_summary={
                "optimizer_updates": 1,
                "zero_environment_maximum_absolute_control": 0.0,
                "training_fold_only": True,
            },
        )
        try:
            save_v82_candidate(
                candidate_root,
                model,
                training_cache_sha256="0" * 64,
                training_cache_metadata_sha256="1" * 64,
                fit_summary={"optimizer_updates": 1},
            )
        except FileExistsError:
            create_once_rejected_overwrite = True
        else:
            create_once_rejected_overwrite = False
        loaded = load_v82_candidate(candidate_root)
        loaded_zero = loaded.model(
            zero_memory,
            query,
            binding=bind_fixed_prefix_before_question_v81(zero_memory),
        )
        loaded_zero_max = float(loaded_zero.controls.abs().max())
        candidate_metadata_fields = sorted(metadata)
        candidate_metadata_sha256 = sha256_file_v82(
            candidate_root / CANDIDATE_METADATA_FILENAME
        )
    if not create_once_rejected_overwrite or loaded_zero_max != 0.0:
        raise RuntimeError("V82 sealed-candidate create-once/zero contract failed")

    report: dict[str, Any] = {
        "schema_version": 82,
        "artifact": "gemma4_v82_strict_dense_reader_model_free_preflight_v1",
        "status": "passed_model_free_no_fit_no_gemma_load",
        "config_sha256": sha256_file_v82(config_file),
        "trainable_parameter_count": model.trainable_parameter_count,
        "fixed_memory_shape": list(memory.shape),
        "fixed_memory_hash_identical_across_questions": True,
        "question_changes_memory": False,
        "question_changes_reader_output": not torch.equal(first.controls, second.controls),
        "minimum_atlas_attention_weight": float(first.atlas_weights.detach().min()),
        "required_minimum_atlas_attention_weight": atlas_floor,
        "minimum_base_attention_weight": float(first.base_weights.detach().min()),
        "required_minimum_base_attention_weight": base_floor,
        "all_384_atlas_values_positive_floor": True,
        "all_256_base_latents_positive_floor": True,
        "boi_eoi_and_96_probe_keys_are_not_payload": True,
        "strict_positive_payload_claim_for_all_738_tokens": False,
        "zero_environment_maximum_absolute_control": zero_max,
        "loaded_candidate_zero_environment_maximum_absolute_control": loaded_zero_max,
        "wrong_scene_contrast_loss": float(contrast.detach()),
        "wrong_scene_preference_mean": float(preference.detach().mean()),
        "wrong_scene_contrast_nonzero_gradient_parameters": nonzero_gradient_parameters,
        "wrong_scene_contrast_gradient_norms": gradient_norms,
        "candidate_create_once_rejected_overwrite": create_once_rejected_overwrite,
        "candidate_metadata_fields": candidate_metadata_fields,
        "candidate_metadata_sha256": candidate_metadata_sha256,
        "questions_or_answers_serialized": False,
        "environmental_text_serialized": False,
        "oracle_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "gemma_loaded": False,
        "fit_executed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if output is not None:
        destination = _resolve(output)
        _atomic_json(destination, report)
        report["output"] = str(destination)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v82_strict_dense_learned_reader.yaml",
    )
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/gemma4_v82_strict_dense_reader_preflight.json",
    )
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_preflight(
        args.config, output=None if args.no_write else args.output
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
