from __future__ import annotations

import torch
import torch.nn.functional as F

from semantic_3d_chat.training.train_question_control_v60 import (
    _basis_targets,
    _parser,
    _prompt_cosines,
    _validate_args,
)


def _arguments() -> list[str]:
    return [
        "--base-runtime-config",
        "configs/runtime/gemma4_v56_question_control.yaml",
        "--base-checkpoint",
        "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
        "--anchor-teacher-artifact",
        "data_gemma4/training/v58_teachers_pair31_32",
        "--expansion-teacher-cache",
        "data_gemma4/training/v59_expansion_teachers",
        "--train-qa",
        "data_diverse52/qa/train.jsonl",
        *(value for scene in (31, 32, 33, 34, 37, 38) for value in ("--scene-id", f"scene_{scene:06d}")),
        "--prefix-cache",
        "data_gemma4/scene_tokens/v59_locked_six_prefixes",
        "--output-checkpoint",
        "data_gemma4/checkpoints/v60_candidate",
        "--training-report",
        "reports/gemma4/metrics/v60_training.json",
    ]


def test_v60_defaults_lock_cheap_cached_teacher_recipe() -> None:
    args = _parser().parse_args(_arguments())
    _validate_args(args)
    assert args.basis_rank == 80
    assert args.epochs == 160
    assert args.learning_rate == 3e-4
    assert args.maximum_control_rms == 0.2


def test_v60_basis_targets_reconstruct_direction_and_rms() -> None:
    torch.manual_seed(17)
    target = torch.randn(1, 2, 8)
    basis = torch.linalg.qr(torch.randn(8, 8)).Q.T.contiguous()
    coefficients, rms, metrics = _basis_targets({("scene_000031", "q"): target}, basis)
    key = "scene_000031", "q"
    reconstructed = torch.einsum("bcr,rh->bch", coefficients[key], basis)
    cosine = F.cosine_similarity(target, reconstructed, dim=-1)
    assert cosine.min().item() > 0.99999
    assert metrics["minimum_cosine"] > 0.99999
    assert torch.allclose(rms[key], target.square().mean(dim=-1).sqrt())


def test_v60_prompt_cosine_moves_cached_target_to_prediction_device() -> None:
    predicted = torch.randn(1, 2, 8)
    assert min(_prompt_cosines(predicted, predicted.cpu())) > 0.99999


def test_v60_seed_deterministically_initializes_v3_module() -> None:
    from semantic_3d_chat.scene_encoder.question_control_v3 import (
        TeacherBasisFullSceneQuestionControlV3,
    )

    basis = torch.eye(8)
    states = []
    for _ in range(2):
        torch.manual_seed(60060)
        module = TeacherBasisFullSceneQuestionControlV3(
            8,
            basis,
            control_tokens=2,
            expected_environment_latents=4,
            moment_count=2,
            interaction_dim=3,
            trunk_dim=4,
        )
        states.append({key: value.clone() for key, value in module.state_dict().items()})
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
