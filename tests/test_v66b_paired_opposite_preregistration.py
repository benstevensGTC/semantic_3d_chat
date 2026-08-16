from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v66b_paired_opposite_preregistration import (
    build_v66b_preregistration,
    write_v66b_preregistration,
)

_OLD_SHA256 = "974f7049d2cf96670c77e6c19808a53fbca8b7c68e7cba7f9f5b184d0fc6ac4c"
_NEW_SHA256 = "9c47e43e85b66bcf07794ccc206783db6a40b18af8ad29407475f081e60930bf"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v66b_preregistration_locks_identified_paired_scene_control() -> None:
    payload = build_v66b_preregistration()

    assert payload["artifact"] == "v66b_allrow_paired_opposite_training_preregistration"
    assert payload["status"] == "locked_before_v66b_controller_training_or_generation"
    assert payload["invalidated_predecessor"]["sha256"] == _OLD_SHA256
    assert payload["invalidated_predecessor"]["preserved_answer_mappings"] == 340
    assert payload["invalidated_predecessor"]["exact_text_cyclic_mappings"] == 409
    assert payload["invalidated_predecessor"][
        "predecessor_artifact_bytes_modified"
    ] is False
    control = payload["paired_opposite_control"]
    assert control["changed_sides"] == 80
    assert control["counterfactual_units"] == 40
    assert control["same_question_bytes_on_both_sides"] is True
    assert control["exact_paired_opposite_scene_prefix_injected"] is True
    assert control["exact_paired_opposite_scene_signature_injected"] is True
    assert payload["thresholds"]["paired_opposite_follows_side_minimum"] == 60
    assert payload["thresholds"]["paired_opposite_follows_complete_minimum"] == 25
    assert payload["thresholds"]["paired_opposite_original_exact_maximum"] == 20
    assert payload["thresholds"]["paired_opposite_original_complete_maximum"] == 5
    assert payload["controls"][
        "cyclic_wrong_complete_scene_prefix_and_signature"
    ] is False
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


def test_v66b_preregistration_is_create_once_and_matches_locked_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prereg.json"

    path, digest = write_v66b_preregistration(output)

    assert path == output
    assert len(digest) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == build_v66b_preregistration()
    with pytest.raises(FileExistsError):
        write_v66b_preregistration(output)

    locked = Path("reports/gemma4/metrics/v66b_paired_opposite_preregistration.json")
    predecessor = Path("reports/gemma4/metrics/v66_allrow_preregistration.json")
    assert _file_sha256(locked) == _NEW_SHA256
    assert _file_sha256(predecessor) == _OLD_SHA256
    assert json.loads(locked.read_text(encoding="utf-8")) == build_v66b_preregistration()
