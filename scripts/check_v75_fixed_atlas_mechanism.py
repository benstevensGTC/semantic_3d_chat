#!/usr/bin/env python3
"""Run the V75 fixed-prefix compiler on one cached numeric scene prefix.

This finite structural check loads no Gemma model, question, answer, oracle, or
protected split. Random probes validate the complete compile/layout mechanism;
they are not a behavioral probe bank and therefore authorize no runtime claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75,
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)


def _rooted(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v75_fixed_prefix_atlas_diagnostic.yaml",
    )
    args = parser.parse_args(argv)
    source = _rooted(args.config)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    settings = raw.get("v75_fixed_prefix_atlas") if isinstance(raw, dict) else None
    if not isinstance(settings, dict):
        raise TypeError("V75 fixed-atlas config is invalid")
    prefix_root = _rooted(settings["source_prefix_cache"])
    scene_id = str(settings["structural_scene"])
    prefix_path = prefix_root / f"{scene_id}.safetensors"
    prefix = load_file(str(prefix_path), device="cpu")["scene_prefix"]
    control, metadata = _load_control_head(
        _rooted(settings["source_controller"]),
        hidden_size=int(settings["language_hidden_size"]),
        device=torch.device("cpu"),
    )
    if type(control) is not DenseFullSceneContinuousControlV75:
        raise TypeError("Configured controller is not the exact V75 architecture")
    generator = torch.Generator().manual_seed(int(settings["probe_seed"]))
    probes = torch.randn(
        (int(settings["probe_count"]), int(settings["language_hidden_size"])),
        generator=generator,
    )
    probes = torch.nn.functional.normalize(probes, dim=-1) * 18.0
    v1 = compile_fixed_scene_atlas_v75(prefix, control, probes)
    v2 = compile_fixed_scene_atlas_v75_v2(prefix, control, probes)
    expected_shape = (
        1,
        int(settings["fixed_prefix_tokens"]),
        int(settings["language_hidden_size"]),
    )
    passed = bool(
        tuple(v1.scene_prefix.shape) == expected_shape
        and tuple(v2.scene_prefix.shape) == expected_shape
        and v1.audit.environment_latent_count
        == int(settings["base_environment_latents"])
        and v1.audit.atlas_memory_token_count
        == int(settings["atlas_memory_tokens"])
        and v1.audit.every_probe_processed
        and v2.audit.base_environment_tokens_preserved_exactly
        and v2.audit.atlas_key_value_tokens_preserved_exactly
        and not v2.audit.user_question_inputs_used_for_compilation
        and not v2.audit.question_dependent_scene_processing
        and not v2.audit.question_dependent_retrieval
    )
    result = {
        "artifact": "v75_fixed_prefix_atlas_structural_mechanism_check_v1",
        "schema_version": 1,
        "passed": passed,
        "status": settings["status"],
        "controller_architecture": metadata["architecture"],
        "controller_weights_sha256": metadata["weights_sha256"],
        "source_v75_candidate_sha256": metadata["source_v75_candidate_sha256"],
        "scene_id": scene_id,
        "base_prefix_shape": list(prefix.shape),
        "base_prefix_sha256": v1.audit.base_scene_prefix_sha256,
        "compiled_prefix_shape": list(v2.scene_prefix.shape),
        "base_environment_latents": v1.audit.environment_latent_count,
        "probe_count": v1.audit.probe_count,
        "values_per_probe": v1.audit.values_per_probe,
        "atlas_memory_tokens": v1.audit.atlas_memory_token_count,
        "all_base_latents_preserved": (
            v2.audit.base_environment_tokens_preserved_exactly
        ),
        "all_atlas_tokens_preserved": (
            v2.audit.atlas_key_value_tokens_preserved_exactly
        ),
        "every_probe_processed": v1.audit.every_probe_processed,
        "compiled_before_user_question": True,
        "user_question_inputs_used_for_compilation": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
        "v1_prefix_sha256": v1.audit.fixed_scene_prefix_sha256,
        "v2_prefix_sha256": v2.audit.fixed_scene_prefix_sha256,
        "gemma_model_loaded": False,
        "questions_or_answers_loaded": False,
        "oracle_loaded": False,
        "protected_split_loaded": False,
        "behavioral_accuracy_measured": False,
        "runtime_promotion_authorized": False,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = _rooted(settings["output_report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
