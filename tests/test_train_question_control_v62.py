from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.training import train_question_control_v62 as v62
from semantic_3d_chat.training.question_control_v5_checkpoint import (
    inherited_v60_state_sha256,
)


def _row(
    pair: int,
    scene: int,
    question: str,
    *,
    changed: bool,
    side: int,
    ordinal: int,
) -> dict[str, Any]:
    return {
        "scene_id": f"scene_{scene:06d}",
        "question_id": f"q_{ordinal:06d}",
        "question": question,
        "counterfactual_expected_change": changed,
        "counterfactual_pair_id": f"pair_{pair:06d}",
        "counterfactual_question_key": f"cfq_{pair:06x}{ordinal:010x}",
        "counterfactual_role": "reference" if side == 0 else "counterfactual",
    }


def _pair_rows(pair: int, first_scene: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side, scene in enumerate((first_scene, first_scene + 1)):
        rows.extend(
            (
                _row(
                    pair,
                    scene,
                    f"changed question {pair}",
                    changed=True,
                    side=side,
                    ordinal=pair * 100 + side * 2,
                ),
                _row(
                    pair,
                    scene,
                    f"retention question {pair}",
                    changed=False,
                    side=side,
                    ordinal=pair * 100 + side * 2 + 1,
                ),
            )
        )
    # Both sides of a paired question require one shared opaque unit key.
    rows[1]["counterfactual_question_key"] = rows[3]["counterfactual_question_key"]
    rows[0]["counterfactual_question_key"] = rows[2]["counterfactual_question_key"]
    return rows


def _source() -> TeacherBasisFullSceneQuestionControlV3:
    torch.manual_seed(620)
    basis = torch.linalg.qr(torch.randn(8, 4)).Q.T.contiguous()
    return TeacherBasisFullSceneQuestionControlV3(
        8,
        basis,
        control_tokens=2,
        expected_environment_latents=4,
        moment_count=2,
        interaction_dim=3,
        trunk_dim=5,
        maximum_control_rms=0.3,
        initial_control_rms=0.1,
    ).eval()


def test_parser_has_one_training_data_path_and_no_held_out_or_scorer_path() -> None:
    destinations = {
        action.dest for action in v62._parser()._actions if action.dest != "help"
    }
    assert "filtered_train_qa" in destinations
    assert "baseline_lock" in destinations
    assert destinations.isdisjoint(v62.V62_PROHIBITED_TRAINER_DATA_ARGUMENTS)
    assert not {
        "preregistration",
        "questions_manifest",
        "question_manifest",
        "gate_path",
        "validation_qa",
    } & destinations
    assert v62._parser().get_default("device") == "auto"
    with pytest.raises(SystemExit):
        v62._parser().parse_args(["--device", "cuda"])


def test_dense_cartesian_uses_all_scenes_and_only_exact_changed_cells() -> None:
    rows = [*_pair_rows(1, 11), *_pair_rows(2, 13)]
    scenes = tuple(f"scene_{value:06d}" for value in (11, 12, 13, 14))
    dense = v62.dense_cartesian_route_examples(rows, scene_ids=scenes)

    assert len(dense) == 2 * 4
    assert {item.question for item in dense} == {
        "changed question 1",
        "changed question 2",
    }
    assert {item.scene_id for item in dense} == set(scenes)
    positives = {(item.scene_id, item.question) for item in dense if item.label}
    assert positives == {
        ("scene_000011", "changed question 1"),
        ("scene_000012", "changed question 1"),
        ("scene_000013", "changed question 2"),
        ("scene_000014", "changed question 2"),
    }
    assert sum(not item.label for item in dense) == 4


def test_leave_one_pair_out_never_places_held_pair_or_scenes_in_training() -> None:
    rows = [*_pair_rows(1, 11), *_pair_rows(2, 13), *_pair_rows(3, 15)]
    pair_ids = tuple(f"pair_{value:06d}" for value in (1, 2, 3))
    scenes = tuple(f"scene_{value:06d}" for value in range(11, 17))
    folds = v62.leave_one_pair_out_folds(
        rows, pair_ids=pair_ids, all_scene_ids=scenes
    )

    assert len(folds) == 3
    for fold in folds:
        held_scenes = {item.scene_id for item in fold.held_natural}
        train_scenes = {item.scene_id for item in fold.training_natural}
        assert set(fold.training_pair_ids).isdisjoint(fold.held_pair_ids)
        assert held_scenes.isdisjoint(train_scenes)
        assert {item.pair_id for item in fold.training_natural} == set(
            fold.training_pair_ids
        )
        assert {item.pair_id for item in fold.held_natural} == set(
            fold.held_pair_ids
        )
        # Held Cartesian evaluation intentionally covers the full authorized
        # training-scene set, while its changed text comes only from the fold.
        assert {item.scene_id for item in fold.held_cartesian} == set(scenes)


def test_route_only_fit_is_deterministic_and_preserves_every_v60_byte() -> None:
    source = _source()
    source_state = {name: value.clone() for name, value in source.state_dict().items()}
    natural = (
        v62.RouteExampleV62("scene_000011", "alpha", True, "p", "q1", "u1", "natural"),
        v62.RouteExampleV62("scene_000012", "alpha", False, "p", "q2", "u1", "natural"),
        v62.RouteExampleV62("scene_000011", "beta", False, "p", "q3", "u2", "natural"),
        v62.RouteExampleV62("scene_000012", "beta", True, "p", "q4", "u2", "natural"),
    )
    dense = tuple(
        v62.RouteExampleV62(
            item.scene_id,
            item.question,
            item.label,
            "cartesian",
            None,
            None,
            "dense",
        )
        for item in natural
    )
    torch.manual_seed(621)
    questions = {
        "alpha": torch.nn.functional.normalize(torch.randn(8), dim=0),
        "beta": torch.nn.functional.normalize(torch.randn(8), dim=0),
    }
    scenes = {
        "scene_000011": source.encode_scene(torch.randn(1, 6, 8)),
        "scene_000012": source.encode_scene(torch.randn(1, 6, 8)),
    }
    config = v62.RouteFitConfigV62(
        epochs=5,
        minimum_epochs=5,
        success_patience=99,
        learning_rate=1e-2,
        weight_decay=0.0,
        gradient_clip_norm=2.0,
        minimum_signed_logit_margin=0.0,
        margin_weight=0.0,
        log_every=100,
    )
    first = v62.fit_route_only(
        source,
        natural_examples=natural,
        dense_examples=dense,
        question_inputs=questions,
        scene_inputs=scenes,
        route_factor_rank=3,
        seed=622,
        config=config,
    )
    second = v62.fit_route_only(
        source,
        natural_examples=natural,
        dense_examples=dense,
        question_inputs=questions,
        scene_inputs=scenes,
        route_factor_rank=3,
        seed=622,
        config=config,
    )

    assert all(
        torch.equal(first.control.state_dict()[name], second.control.state_dict()[name])
        for name in first.control.state_dict()
    )
    assert all(torch.equal(source.state_dict()[name], value) for name, value in source_state.items())
    assert set(source_state) == set(first.control.inherited_state_names)
    assert all(
        torch.equal(first.control.state_dict()[name], value)
        for name, value in source_state.items()
    )
    assert inherited_v60_state_sha256(first.control) == inherited_v60_state_sha256(
        second.control
    )
    assert inherited_v60_state_sha256(first.control) == v62._tensor_state_sha256(
        source_state
    )
    assert first.route_device == second.route_device == "cpu"
    assert {
        parameter.device.type for parameter in first.control.factorized_route.parameters()
    } == {"cpu"}
    assert all(parameter.grad is None for parameter in source.parameters())


def test_route_fit_restores_exact_best_epoch_after_late_oscillatory_regression() -> None:
    """High-LR synthetic XOR is exact at epoch 3 and regresses by epoch 5."""

    source = _source()
    source_hash = v62._tensor_state_sha256(source.state_dict())
    examples = (
        v62.RouteExampleV62("scene_000011", "alpha", True, "p", "q1", "u1", "natural"),
        v62.RouteExampleV62("scene_000012", "alpha", False, "p", "q2", "u1", "natural"),
        v62.RouteExampleV62("scene_000011", "beta", False, "p", "q3", "u2", "natural"),
        v62.RouteExampleV62("scene_000012", "beta", True, "p", "q4", "u2", "natural"),
    )
    torch.manual_seed(621)
    questions = {
        "alpha": torch.nn.functional.normalize(torch.randn(8), dim=0),
        "beta": torch.nn.functional.normalize(torch.randn(8), dim=0),
    }
    scenes = {
        "scene_000011": source.encode_scene(torch.randn(1, 6, 8)),
        "scene_000012": source.encode_scene(torch.randn(1, 6, 8)),
    }
    fit = v62.fit_route_only(
        source,
        natural_examples=examples,
        dense_examples=examples,
        question_inputs=questions,
        scene_inputs=scenes,
        route_factor_rank=3,
        seed=622,
        config=v62.RouteFitConfigV62(
            epochs=5,
            minimum_epochs=5,
            success_patience=99,
            learning_rate=0.2,
            weight_decay=0.0,
            gradient_clip_norm=100.0,
            minimum_signed_logit_margin=0.0,
            margin_weight=0.0,
            log_every=99,
        ),
    )

    assert fit.completed_epochs == 5
    assert fit.best_epoch == 3
    assert fit.best_epoch < fit.completed_epochs
    assert fit.natural_metrics["exact"] is True
    assert fit.dense_metrics["exact"] is True
    assert fit.natural_metrics["minimum_signed_logit_margin"] == pytest.approx(
        fit.best_minimum_signed_logit_margin
    )
    assert inherited_v60_state_sha256(fit.control) == source_hash
    assert all(parameter.grad is None for parameter in source.parameters())
    assert {
        name for name, parameter in fit.control.named_parameters() if parameter.requires_grad
    } == {
        name
        for name, _parameter in fit.control.named_parameters()
        if name.startswith("factorized_route.")
    }


def _passing_fold_report() -> dict[str, Any]:
    return {
        "training_natural": {"exact": True},
        "training_dense": {"exact": True},
        "held_natural": {
            "correct": 48,
            "total": 48,
            "accuracy": 1.0,
            "positive_correct": 8,
            "positive_total": 8,
            "negative_correct": 40,
            "negative_total": 40,
        },
        "held_cartesian": {
            "correct": 96,
            "total": 96,
            "accuracy": 1.0,
            "negative_correct": 88,
            "negative_total": 88,
        },
        "changed_units_complete": 4,
        "changed_units_total": 4,
    }


def test_cv_gate_requires_train_exactness_and_conservative_held_metrics() -> None:
    passing = [_passing_fold_report() for _ in range(12)]
    result = v62.assess_route_cv(passing)
    assert result["passed"] is True
    assert len(result["thresholds_sha256"]) == 64
    assert result["checks"]["all_fold_training_dense_exact"] is True

    failing = json.loads(json.dumps(passing))
    for fold in failing:
        fold["held_natural"]["positive_correct"] = 0
    failed = v62.assess_route_cv(failing)
    assert failed["passed"] is False
    assert failed["checks"]["held_changed_recall"] is False


def test_baseline_lock_is_validated_before_filtered_training_is_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = v62._parser()
    args = parser.parse_args(
        [
            "--base-runtime-config",
            "configs/runtime/x.yaml",
            "--base-checkpoint",
            "base",
            "--source-v60-checkpoint",
            "v60",
            "--source-v60-report",
            "v60.json",
            "--filtered-train-qa",
            "train.jsonl",
            "--prefix-cache",
            "prefixes",
            "--baseline-lock",
            "lock.json",
            "--output-checkpoint",
            "candidate",
            "--training-report",
            "report.json",
        ]
    )
    opened_training = False

    def forbidden_training(_path: str | Path) -> tuple[dict[str, Any], ...]:
        nonlocal opened_training
        opened_training = True
        raise AssertionError("training data opened before authorization")

    monkeypatch.setattr(v62, "load_filtered_training_qa", forbidden_training)
    monkeypatch.setattr(
        v62,
        "load_v62_baseline_authorization",
        lambda _path: (_ for _ in ()).throw(RuntimeError("authorization refused")),
    )
    with pytest.raises(RuntimeError, match="authorization refused"):
        v62.train_v62(args)
    assert opened_training is False


def test_authorization_wrapper_uses_boundary_validator_and_hashes_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "lock.json"
    raw = b'{"hash_only":true}\n'
    path.write_bytes(raw)
    payload = {
        "preregistration_sha256": "a" * 64,
        "v54_checkpoint_sha256": "b" * 64,
    }
    calls: list[Path] = []

    def validate(value: str | Path) -> dict[str, Any]:
        calls.append(Path(value))
        return payload

    monkeypatch.setattr(v62, "validate_baseline_lock", validate)
    observed = v62.load_v62_baseline_authorization(path)
    assert calls == [path.resolve()]
    assert observed.payload == payload
    assert observed.sha256 == hashlib.sha256(raw).hexdigest()


def test_training_module_does_not_name_held_out_or_scorer_inputs() -> None:
    source = Path(v62.__file__).read_text(encoding="utf-8")
    assert "--preregistration" not in source
    assert "--validation-questions" not in source
    assert "--scorer-references" not in source
    assert "--questions-manifest" not in source
    assert "--gate-path" not in source
    assert "generate_with" not in source
    assert ".backward()" in source  # route head only; guarded by tests and code.


def test_fit_config_rejects_a_disabled_route_objective() -> None:
    config = v62.RouteFitConfigV62(
        natural_loss_weight=0.0,
        dense_loss_weight=0.0,
    )
    with pytest.raises(ValueError, match="no enabled population"):
        config.validate()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable on this host"
)
def test_route_fit_accepts_cpu_caches_on_mps_and_returns_cpu_metrics() -> None:
    source = _source()
    examples = (
        v62.RouteExampleV62("scene_000011", "alpha", True, "p", "q1", "u1", "natural"),
        v62.RouteExampleV62("scene_000012", "alpha", False, "p", "q2", "u1", "natural"),
        v62.RouteExampleV62("scene_000011", "beta", False, "p", "q3", "u2", "natural"),
        v62.RouteExampleV62("scene_000012", "beta", True, "p", "q4", "u2", "natural"),
    )
    torch.manual_seed(629)
    questions = {
        "alpha": torch.nn.functional.normalize(torch.randn(8), dim=0),
        "beta": torch.nn.functional.normalize(torch.randn(8), dim=0),
    }
    scenes = {
        "scene_000011": source.encode_scene(torch.randn(1, 6, 8)),
        "scene_000012": source.encode_scene(torch.randn(1, 6, 8)),
    }
    fit = v62.fit_route_only(
        source,
        natural_examples=examples,
        dense_examples=examples,
        question_inputs=questions,
        scene_inputs=scenes,
        route_factor_rank=3,
        seed=630,
        config=v62.RouteFitConfigV62(
            epochs=1,
            minimum_epochs=1,
            success_patience=2,
            learning_rate=1e-3,
            minimum_signed_logit_margin=0.0,
            margin_weight=0.0,
            log_every=2,
        ),
        device="mps",
    )
    assert fit.route_device == "mps"
    assert {parameter.device.type for parameter in fit.control.parameters()} == {"mps"}
    assert all(
        isinstance(value, (bool, int, float)) for value in fit.natural_metrics.values()
    )
    assert inherited_v60_state_sha256(fit.control) == v62._tensor_state_sha256(
        source.state_dict()
    )
    assert all(parameter.grad is None for parameter in source.parameters())


def test_parser_returns_namespace_for_hash_only_boundary() -> None:
    parser = v62._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert isinstance(parser, argparse.ArgumentParser)
