from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v67_strict_atlas_preregistration as contract

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "configs/experiments/v67_strict_atlas_internal_validation_preregistration.json"
)


def test_v67_strict_atlas_preregistration_matches_public_pin() -> None:
    value = contract.validate_v67_strict_atlas_preregistration(PREREGISTRATION)

    assert hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest() == (
        contract.PINNED_V67_STRICT_ATLAS_PREREGISTRATION_SHA256
    )
    assert value["population"] == contract.POPULATION
    assert value["thresholds"] == contract.TERMINAL_THRESHOLDS
    assert value["split"]["scene_ids"] == list(contract.SCENE_IDS)
    assert value["candidate_contract"]["compiled_fixed_prefix_tokens"] == 738
    assert value["candidate_contract"]["environmental_text_inputs"] == []
    assert value["source_boundary"]["atlas_prediction_source_sha256"] == (
        contract.PINNED_ATLAS_PREDICTION_SOURCE_SHA256
    )
    assert value["source_boundary"]["terminal_gate_source_sha256"] == (
        contract.PINNED_TERMINAL_GATE_SOURCE_SHA256
    )


def test_v67_strict_atlas_executable_sources_match_preregistered_hashes() -> None:
    value = contract.validate_v67_strict_atlas_preregistration(PREREGISTRATION)
    sources = {
        "atlas_compiler_config_sha256": (
            ROOT / "configs/experiments/gemma4_strict_fixed_prefix_atlas_v1.yaml"
        ),
        "atlas_compiler_source_sha256": (
            ROOT / "src/semantic_3d_chat/training/fixed_prefix_atlas_checkpoint.py"
        ),
        "atlas_prediction_source_sha256": (
            ROOT / "src/semantic_3d_chat/evaluation/predict_fixed_prefix_atlas.py"
        ),
        "atlas_runtime_source_sha256": (
            ROOT / "src/semantic_3d_chat/scene_encoder/fixed_prefix_atlas.py"
        ),
        "terminal_gate_source_sha256": (
            ROOT / "src/semantic_3d_chat/evaluation/fixed_prefix_atlas_gate.py"
        ),
    }

    for field, source in sources.items():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == value["source_boundary"][field]


def test_v67_strict_atlas_preregistration_fails_closed_on_mutation(
    tmp_path: Path,
) -> None:
    value = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    value["thresholds"]["changed_side_exact"]["minimum"] -= 1
    changed = tmp_path / "preregistration.json"
    changed.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its pin"):
        contract.validate_v67_strict_atlas_preregistration(changed)


def test_v67_strict_atlas_validator_exposes_no_evaluation_input() -> None:
    module_source = Path(contract.__file__).read_text(encoding="utf-8")

    assert "scorer_only/" not in module_source
    assert "data_gemma4/maps/" not in module_source
    assert "data_gemma4/qa/" not in module_source
    assert "data_gemma4/oracle/" not in module_source
    assert "predicted_answer" not in module_source
