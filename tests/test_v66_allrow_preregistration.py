from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v66_allrow_preregistration import (
    build_v66_preregistration,
    write_v66_preregistration,
)


def test_v66_preregistration_pins_baselines_inventory_controls_and_scope() -> None:
    payload = build_v66_preregistration()

    assert payload["status"] == "locked_before_v66_controller_training_or_generation"
    assert payload["frozen_v54_training_baseline"]["canonical_type_specific_exact"] == 227
    assert payload["frozen_v54_training_baseline"]["changed_side_exact"] == 35
    assert payload["pair_heldout_inventory"]["vocabulary_supported_rows"] == 571
    assert payload["pair_heldout_inventory"]["vocabulary_unsupported_singleton_rows"] == 5
    assert payload["thresholds"]["held_supported_exact_minimum"] == 300
    assert payload["thresholds"]["final_exact_minimum"] == 520
    assert payload["controls"]["cyclic_wrong_complete_scene_prefix_and_signature"] is True
    assert payload["controls"]["unverified_native_answer_embedding_fallback_permitted"] is False
    assert all(
        payload["scope"][field] is False
        for field in (
            "validation_inputs_used",
            "scorer_inputs_used",
            "oracle_loaded",
            "fresh_development_loaded",
            "deferred_final_loaded",
        )
    )


def test_v66_preregistration_is_create_once_and_canonical_json(tmp_path: Path) -> None:
    output = tmp_path / "prereg.json"

    path, digest = write_v66_preregistration(output)

    assert path == output
    assert len(digest) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == build_v66_preregistration()
    with pytest.raises(FileExistsError):
        write_v66_preregistration(output)
