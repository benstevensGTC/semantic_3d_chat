from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation import v31_joint_pair_selector as selector
from semantic_3d_chat.training import train_joint_pair_v31 as trainer
from semantic_3d_chat.training.train_joint_pair_v31 import (
    v31_contract,
)

V31_CONFIG = Path("configs/experiments/gemma4_diverse28_joint_pair_v31.yaml")


def test_v31_locks_expanded_train_unchanged_validation_and_deferred_final() -> None:
    config = load_config(V31_CONFIG)
    contract = v31_contract(config)

    assert contract.train_scene_ids == tuple(
        [f"scene_{index:06d}" for index in range(11, 19)]
        + [f"scene_{index:06d}" for index in range(31, 39)]
    )
    assert contract.validation_scene_ids == tuple(
        f"scene_{index:06d}" for index in range(19, 25)
    )
    assert contract.deferred_final_scene_ids == tuple(
        f"scene_{index:06d}" for index in range(25, 31)
    )
    assert contract.train_question_count == 384
    assert contract.validation_question_count == 216
    assert contract.train_changed_pair_unit_count == 25
    assert contract.optimizer_cycles == 8
    assert contract.development_changed_complete_pairs_minimum == 1
    assert contract.chat_promotion_changed_complete_pairs_minimum == 6


def test_v31_is_fresh_from_approved_v29_and_uses_stronger_bounded_rates() -> None:
    config = load_config(V31_CONFIG)
    source = config["v30_joint_pair"]
    settings = config["training"]["v30_joint_pair"]

    assert source["source_selected_update"] == 4
    assert source["source_checkpoint_root"].endswith(
        "gemma4_v29_diverse20_post_stack_decoder_stage_b"
    )
    assert "v30_diverse20" not in source["source_checkpoint_root"]
    assert settings["max_optimizer_steps"] == 8
    assert settings["evaluation_interval_steps"] == 1
    assert settings["sidecar_learning_rate"] == 1.0e-4
    assert settings["decoder_learning_rate"] == 2.0e-4
    assert settings["gradient_clip_norm"] == 1.0
    assert settings["weight_decay"] == 0.0


def test_v31_allows_only_new_training_scenes_to_derive_source_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_run_v30(**kwargs):
        observed.update(kwargs)
        return {
            "artifact": "v30_joint_pair_training",
            "optimizer_updates": 8,
            "final_test_scene_ids_loaded": [],
        }

    monkeypatch.setattr(trainer, "run_v30", fake_run_v30)
    report = trainer.run_v31(
        config=load_config(V31_CONFIG),
        output=tmp_path / "candidate",
    )

    assert observed["allow_unpinned_source_scene_ids"] == tuple(
        f"scene_{index:06d}" for index in range(31, 39)
    )
    assert report["optimizer_updates"] == 8


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("v31_expanded_pair", "validation_scene_ids"), ["scene_000025"]),
        (("v31_expanded_pair", "optimizer_cycles"), 7),
        (("v31_expanded_pair", "strict_validation_nll_improvement"), False),
        (
            ("v30_joint_pair", "selection_requires", "minimum_greedy_complete_units_correct"),
            0,
        ),
        (
            (
                "v30_joint_pair",
                "promotion_requires",
                "validation_changed_complete_pairs_minimum",
            ),
            1,
        ),
    ],
)
def test_v31_contract_fails_closed_when_data_or_gates_are_relaxed(
    path: tuple[str, ...], value: object
) -> None:
    config = copy.deepcopy(load_config(V31_CONFIG))
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises((TypeError, ValueError)):
        v31_contract(config)


def _write_envelope(root: Path, config: dict) -> None:
    contract = v31_contract(config)
    for update in range(9):
        checkpoint = root / f"update_{update:03d}"
        checkpoint.mkdir(parents=True)
        metadata = {
            "optimizer_step": update,
            "config_hash": config_hash(config),
            "v30_joint_pair": {
                "train_scene_ids": list(contract.train_scene_ids),
                "validation_scene_ids": list(contract.validation_scene_ids),
                "train_question_count": contract.train_question_count,
                "validation_question_count": contract.validation_question_count,
                "final_test_scene_ids_loaded": [],
                "qa_dataset": {
                    "qa_root": str(contract.qa_root),
                    "split_fingerprint": contract.split_fingerprint,
                    "deferred_test_scene_ids_loaded": [],
                },
            },
        }
        (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_v31_checkpoint_envelope_requires_all_nine_expanded_data_arms(tmp_path: Path) -> None:
    config = load_config(V31_CONFIG)
    contract = v31_contract(config)
    _write_envelope(tmp_path, config)
    paths = selector.validate_v31_checkpoint_envelope(config, tmp_path, contract)
    assert len(paths) == 9

    (tmp_path / "update_004" / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError):
        selector.validate_v31_checkpoint_envelope(config, tmp_path, contract)


def test_v31_checkpoint_envelope_rejects_final_scene_or_wrong_train_split(
    tmp_path: Path,
) -> None:
    config = load_config(V31_CONFIG)
    contract = v31_contract(config)
    _write_envelope(tmp_path, config)
    path = tmp_path / "update_008" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["v30_joint_pair"]["train_scene_ids"][-1] = "scene_000025"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="train split mismatch"):
        selector.validate_v31_checkpoint_envelope(config, tmp_path, contract)


def test_v31_selector_wraps_all_intermediates_and_keeps_promotion_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(V31_CONFIG)
    contract = v31_contract(config)
    _write_envelope(tmp_path, config)
    required_checks = {
        "color_retained": True,
        "mirror_retained": True,
        "no_new_negative_sides": True,
        "below_selected_v29_source_nll": True,
        "greedy_changed_units_demonstrated": True,
        "broad_exact_accuracy_retained": True,
    }
    fake = {
        "artifact": "v30_joint_pair_development_selection",
        "arms": [
            {"update": update, "checks": ({} if update == 0 else required_checks)}
            for update in range(9)
        ],
        "validation_scene_ids": list(contract.validation_scene_ids),
        "requirements": {
            "selected_v29_source_nll_must_improve": True,
            "color_full_vocab_sides": 12,
            "mirror_full_vocab_sides": 10,
            "no_new_negative_sides": True,
            "minimum_greedy_complete_units_correct": 1,
            "chat_promotion_changed_complete_pairs_minimum": 6,
            "chat_promotion_aggregate_validation_exact_accuracy_no_regression": True,
        },
        "chat_promotion": {
            "checks": {
                "development_checkpoint_selected": True,
                "changed_complete_pair_threshold_met": False,
                "aggregate_validation_exact_accuracy_retained": True,
            }
        },
        "passed": True,
    }
    monkeypatch.setattr(selector, "select_joint_pair", lambda *args, **kwargs: fake)
    report = selector.select_v31(V31_CONFIG, tmp_path)
    assert report["all_intermediate_checkpoints_inspected"] is True
    assert report["development_progress_is_not_chat_promotion"] is True
    assert report["artifact"] == "v31_diverse28_joint_pair_development_selection"
    assert report["engine_artifact"] == "v30_joint_pair_development_selection"


def test_v31_make_targets_and_docs_never_offer_final_test() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "gemma4-v31-preflight-expanded-pair" in makefile
    assert "gemma4-v31-train-expanded-pair" in makefile
    assert "gemma4-v31-select-expanded-pair" in makefile
    assert "gemma4-v31-evaluate-final" not in makefile
    region = makefile[
        makefile.index("gemma4-v31-preflight-expanded-pair:") : makefile.index(
            "# Development-set measurement only."
        )
    ]
    assert "scene_000025" not in region
    assert "include-deferred-test" not in region
    assert "There is intentionally no V31 final-test" in readme
