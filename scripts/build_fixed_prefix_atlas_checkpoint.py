#!/usr/bin/env python3
"""Convert a sealed V7 controller into a strict fixed-prefix atlas checkpoint.

This is an offline preparation command.  Source text is read only to create a
deterministic continuous probe bank; the output contains tensors and sanitized
numeric provenance only.  No source record or text is copied into the runtime
artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.local_lm import (
    load_local_language_model,
    question_token_ids,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training.fixed_prefix_atlas_checkpoint import (
    deterministic_spherical_probe_bank,
    save_fixed_prefix_atlas_checkpoint,
    sha256_file,
    two_file_checkpoint_fingerprint,
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _source_questions(path: Path) -> tuple[list[str], str]:
    """Read a training-only JSONL and return sorted unique question strings."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Training source is unavailable: {path}")
    digest = sha256_file(path)
    questions: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid training JSONL at line {line_number}") from error
        if not isinstance(row, dict):
            raise TypeError(f"Training row {line_number} must be an object")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Training row {line_number} has no question")
        questions.add(question.strip())
    if not questions:
        raise ValueError("Training source contains no questions")
    return sorted(questions), digest


def _continuous_vectors(language, questions: list[str]) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    layer = language.model.get_input_embeddings()
    with torch.inference_mode():
        for question in questions:
            ids = question_token_ids(language.tokenizer, question, language.device)
            value = layer(ids).detach().float().mean(dim=1).cpu()
            if tuple(value.shape) != (1, language.hidden_size):
                raise RuntimeError("Unexpected pooled source-vector shape")
            embeddings.append(value[0])
    result = torch.stack(embeddings).contiguous()
    if not torch.isfinite(result).all():
        raise RuntimeError("Source-vector extraction produced NaN or infinity")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime/gemma4_v54.yaml")
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
    )
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument(
        "--training-source",
        default="data_gemma4/training/v62_pair_disjoint/train.jsonl",
    )
    parser.add_argument("--probe-count", type=int, default=96)
    parser.add_argument("--cluster-iterations", type=int, default=12)
    parser.add_argument(
        "--output",
        default="data_gemma4/checkpoints/gemma4_strict_fixed_prefix_atlas_v1",
    )
    args = parser.parse_args(argv)

    config = load_runtime_config(args.config)
    config_sha256 = effective_runtime_config_sha256(config)
    base_checkpoint = _rooted(args.base_checkpoint)
    base_sha256, _base_files = checkpoint_fingerprint(base_checkpoint)
    control_checkpoint = _rooted(args.control_checkpoint)
    control_sha256, control_files = two_file_checkpoint_fingerprint(control_checkpoint)

    language_config = config["language"]
    language = load_local_language_model(
        str(language_config["model_id"]),
        revision=str(language_config["revision"]),
        requested_dtype=str(language_config["dtype"]),
        freeze=True,
        local_files_only=True,
        backend=str(language_config["backend"]),
    )
    control, control_metadata = _load_control_head(
        control_checkpoint,
        hidden_size=language.hidden_size,
        device=language.device,
    )
    if type(control) is not AlwaysOnTeacherBasisFullSceneQuestionControlV7:
        raise TypeError("Fixed-prefix conversion requires an exact sealed V7 checkpoint")
    if (
        control_metadata.get("saved_runtime_training_gate_passed") is not True
        or control_metadata.get("base_checkpoint_sha256") != base_sha256
        or control_metadata.get("base_runtime_config_sha256") != config_sha256
    ):
        raise ValueError("Sealed V7 controller is not bound to the requested base runtime")

    source_path = _rooted(args.training_source)
    questions, source_file_sha256 = _source_questions(source_path)
    vectors = _continuous_vectors(language, questions)
    probes, probe_audit = deterministic_spherical_probe_bank(
        vectors,
        probe_count=args.probe_count,
        iterations=args.cluster_iterations,
    )
    # Record only hashes and counts.  Source path, question strings, and the
    # record schema are intentionally absent from the runtime artifact.
    probe_audit = {
        **probe_audit,
        "source_file_sha256": source_file_sha256,
        "unique_source_vector_count": len(questions),
        "source_text_retained": False,
    }
    result = save_fixed_prefix_atlas_checkpoint(
        _rooted(args.output),
        controller=control.cpu(),
        probe_embeddings=probes,
        source_controller_checkpoint_sha256=control_sha256,
        source_controller_files=control_files,
        base_checkpoint_sha256=base_sha256,
        base_runtime_config_sha256=config_sha256,
        probe_audit=probe_audit,
    )
    output = {
        "phase": "fixed_prefix_atlas_checkpoint_built",
        "output": str(_rooted(args.output)),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "probe_count": args.probe_count,
        "source_vector_count": len(questions),
        "hidden_size": language.hidden_size,
        "question_dependent_scene_processing": False,
        "runtime_source_text_retained": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
