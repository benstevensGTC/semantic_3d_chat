#!/usr/bin/env python3
"""Measure V74 verified-teacher geometry without optimization or held-out sets."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)
from semantic_3d_chat.training.soft_prompt_teacher_v62 import load_v62_teacher_cache
from semantic_3d_chat.training.soft_prompt_teacher_v66 import (
    load_v66_answer_class_teacher_cache,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    _predict_v73,
    load_config_v73,
    load_embedding_assets_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)
from train_v74_teacher_reader import _teacher_bank


def stats(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().flatten()
    return {
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "mean": float(flat.mean()),
        "median": float(flat.median()),
    }


def main() -> int:
    config = load_config_v73(
        "configs/experiments/gemma4_v73_fullscene_controller.yaml"
    )
    rows = load_training_rows_v73(config["training_qa"])
    train, _held = split_rows_v73(rows)
    primary, _primary_metadata = load_v62_teacher_cache(
        "data_gemma4/training/v62_changed_teachers"
    )
    supplemental, _supplemental_metadata = load_v66_answer_class_teacher_cache(
        "data_gemma4/training/v66_answer_class_teachers"
    )
    bank = _teacher_bank(train, {**primary, **supplemental})
    target = bank.prototypes.float()
    coefficient = torch.einsum("bch,rh->bcr", target, bank.output_basis)
    reconstructed = torch.einsum("bcr,rh->bch", coefficient, bank.output_basis)
    residual = reconstructed - target
    target_rms = target.square().mean(dim=-1).sqrt()
    payload: dict[str, object] = {
        "artifact": "v74_verified_teacher_geometry_diagnostic_v1",
        "training_pool_only": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "oracle_loaded": False,
        "teacher_class_count": len(bank.class_ids),
        "teacher_token_count": int(target.shape[0] * target.shape[1]),
        "basis_rank": int(bank.output_basis.shape[0]),
        "teacher_token_rms": stats(target_rms),
        "fraction_teacher_tokens_above_runtime_cap_0_25": float(
            (target_rms > 0.25).float().mean()
        ),
        "basis_reconstruction_relative_mse": float(
            residual.square().mean() / target.square().mean()
        ),
        "basis_reconstruction_maximum_absolute_error": float(residual.abs().max()),
    }

    candidate = (
        PROJECT_ROOT
        / "reports/gemma4/artifacts/v74_verified_teacher_reader_cpu_c05v05_diagnostic.safetensors"
    )
    if candidate.exists():
        prefixes, _manifest = load_prefixes_v73(
            config["prefix_cache"], {row.scene_id for row in train}
        )
        assets = load_embedding_assets_v73(
            config["gemma_snapshot"],
            {row.question for row in train},
            {row.answer_class: row.answer for row in train},
        )
        state = load_file(str(candidate), device="cpu")
        model = DenseFullSceneContinuousControlV74(1536, state["output_basis"])
        model.load_state_dict(state, strict=True)
        output = _predict_v73(
            model,
            train,
            prefixes=prefixes,
            questions=assets.questions,
            batch_size=48,
            device=torch.device("cpu"),
        )
        classes = torch.tensor([bank.class_index[row.answer_class] for row in train])
        row_target = target[classes]
        output_rms = output.square().mean(dim=-1).sqrt()
        normalized_mse = (
            (output - row_target).square().mean(dim=(1, 2))
            / row_target.square().mean(dim=(1, 2)).clamp_min(1e-8)
        )
        payload["diagnostic_candidate"] = {
            "output_token_rms": stats(output_rms),
            "fraction_output_tokens_at_cap": float(
                (output_rms >= 0.25 - 1e-6).float().mean()
            ),
            "normalized_teacher_mse": stats(normalized_mse),
            "mean_flat_teacher_cosine": float(
                F.cosine_similarity(output.flatten(1), row_target.flatten(1)).mean()
            ),
            "candidate_path": str(candidate.relative_to(PROJECT_ROOT)),
        }
    output_path = (
        PROJECT_ROOT
        / "reports/gemma4/metrics/v74_verified_teacher_geometry_diagnostic.json"
    )
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
