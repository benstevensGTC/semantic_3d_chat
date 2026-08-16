from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v95_deferred_final as deferred
from semantic_3d_chat.evaluation import (
    v95_deferred_final_materialization as materialization,
)
from semantic_3d_chat.evaluation import v95_deferred_final_qa as final_qa


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _synthetic_candidate_pool() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair_id, scenes in final_qa.PAIR_SCENES.items():
        for answer_type, quota in final_qa.PAIR_UNIT_QUOTAS.items():
            # Full stable and changed inventories make every deterministic
            # four-unit allocation feasible without using answer values.
            for expected_change in (False, True):
                for index in range(quota):
                    question_key = (
                        f"{pair_id}_{answer_type}_{int(expected_change)}_{index:03d}"
                    )
                    for side, scene_id in enumerate(scenes):
                        records.append(
                            {
                                "scene_id": scene_id,
                                "question_id": f"q_{answer_type}_{side}_{index:03d}",
                                "question": f"opaque template {question_key}",
                                "answer": f"answer-{side}-{index}",
                                "answer_type": answer_type,
                                "target_xyz": [float(side), 0.0, 0.0],
                                "target_instance": f"i_{index:06d}",
                                "counterfactual_pair_id": pair_id,
                                "counterfactual_paired_scene_id": scenes[1 - side],
                                "counterfactual_change_type": "synthetic",
                                "counterfactual_role": (
                                    "reference" if side == 0 else "counterfactual"
                                ),
                                "counterfactual_question_key": question_key,
                                "counterfactual_expected_change": expected_change,
                            }
                        )
    return records


def _fake_toolchain() -> dict[str, Any]:
    return {
        "support_python": {
            "path": "/fixed/support-python",
            "executable_sha256": _digest("support"),
            "implementation": "CPython",
            "version": "3.12.13",
            "packages": {"PyYAML": "fixed", "numpy": "fixed"},
        },
        "gemma_python": {
            "path": "/fixed/gemma-python",
            "executable_sha256": _digest("gemma"),
            "implementation": "CPython",
            "version": "3.12.13",
            "packages": {
                "torch": "fixed",
                "transformers": "fixed",
                "safetensors": "fixed",
                "numpy": "fixed",
                "Pillow": "fixed",
            },
        },
        "blender": {
            "path": "/fixed/blender",
            "executable_sha256": _digest("blender"),
            "version_output": ["Blender fixed"],
        },
    }


def test_recipe_is_six_new_opaque_scenes_from_public_default_only() -> None:
    result = materialization.validate_recipe_v95()

    assert result["default_seed"] == 20260808
    assert [item["scene_id"] for item in result["scene_plans"]] == list(
        final_qa.SCENE_IDS
    )
    assert result["pair_scenes"] == {
        key: list(value) for key, value in final_qa.PAIR_SCENES.items()
    }
    assert [
        (item["seed"], item["change_type"]) for item in result["scene_plans"]
    ] == [
        (20285024, "color_swap"),
        (20285024, "color_swap"),
        (20287042, "cube_support"),
        (20287042, "cube_support"),
        (20289060, "mirror_lr"),
        (20289060, "mirror_lr"),
    ]
    assert result["legacy_plan_files_opened"] == []
    assert result["qa_contract"]["answer_type_totals"] == final_qa.ANSWER_TYPE_TOTALS


def test_exact_selector_produces_216_pair_atomic_rows_and_fixed_totals() -> None:
    selected, manifest = final_qa.select_exact_final_records_v95(
        _synthetic_candidate_pool()
    )

    assert len(selected) == 216
    assert manifest["row_count"] == 216
    assert manifest["rows_per_scene"] == 36
    assert manifest["answer_type_totals"] == final_qa.ANSWER_TYPE_TOTALS
    assert manifest["changed_unit_count"] == 12
    assert manifest["changed_side_count"] == 24
    assert manifest["selection_uses_question_or_answer_text"] is False
    assert manifest["selection_uses_answer_values"] is False


def test_exact_selector_is_answer_value_independent() -> None:
    records = _synthetic_candidate_pool()
    selected_a, manifest_a = final_qa.select_exact_final_records_v95(records)
    mutated = [{**record, "answer": "deliberately-different"} for record in records]
    selected_b, manifest_b = final_qa.select_exact_final_records_v95(mutated)

    identity_a = [
        (
            item["scene_id"],
            item["counterfactual_pair_id"],
            item["counterfactual_question_key"],
        )
        for item in selected_a
    ]
    identity_b = [
        (
            item["scene_id"],
            item["counterfactual_pair_id"],
            item["counterfactual_question_key"],
        )
        for item in selected_b
    ]
    assert identity_a == identity_b
    assert (
        manifest_a["selected_unit_inventory_sha256"]
        == manifest_b["selected_unit_inventory_sha256"]
    )


def test_exact_selector_fails_closed_when_a_quota_is_not_feasible() -> None:
    records = [
        record
        for record in _synthetic_candidate_pool()
        if not (
            record["counterfactual_pair_id"] == "pair_000013"
            and record["answer_type"] == "orientation"
        )
    ]
    with pytest.raises(ValueError, match="cannot satisfy the preregistered exact quotas"):
        final_qa.select_exact_final_records_v95(records)


def test_preregistration_binds_commands_sources_tools_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        materialization,
        "_file_identities",
        lambda paths: {str(path): _digest(str(path)) for path in paths},
    )
    result = materialization.build_materialization_preregistration_v95(
        toolchain=_fake_toolchain()
    )

    assert result["status"] == materialization.STATUS
    assert result["scene_count"] == 6
    assert result["pair_count"] == 3
    assert result["intended_row_count"] == 216
    assert result["intended_changed_unit_count"] == 12
    assert result["answer_type_totals"] == final_qa.ANSWER_TYPE_TOTALS
    assert result["source_sha256"]
    assert result["numeric_compiler_source_sha256"]
    assert result["stage_order"] == [
        "generate",
        "render",
        "features",
        "maps",
        "memory",
        "qa_raw",
        "qa_select",
        "questions",
    ]
    assert len(result["stages"]["render"]["expected_outputs"]) == 6 * 50
    assert len(result["stages"]["features"]["expected_outputs"]) == 6 * 25
    assert len(result["stages"]["memory"]["expected_outputs"]) == 6 * 3
    assert result["generation_requires_authenticated_unlock"] is True
    assert result["deferred_artifacts_generated"] is False
    assert result["protected_read_count"] == 0


def test_preregistration_is_create_once_and_authenticates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "preregistration.json"
    payload = {
        "artifact": materialization.ARTIFACT,
        "schema_version": 95,
        "status": materialization.STATUS,
        "preregistration_identity_sha256": _digest("identity"),
    }
    monkeypatch.setattr(materialization, "PREREGISTRATION", output)
    monkeypatch.setattr(
        materialization, "build_materialization_preregistration_v95", lambda: payload
    )

    created = materialization.preregister_materialization_v95()
    reused = materialization.preregister_materialization_v95()

    assert created["authenticated"] is True
    assert created["reused_authenticated_preregistration"] is False
    assert reused["reused_authenticated_preregistration"] is True
    assert created["preregistration_file_sha256"] == reused[
        "preregistration_file_sha256"
    ]


def test_run_stage_authenticates_prereg_and_unlock_before_child_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    stages = {
        stage: {
            "child_argv": [["fixed-child", stage]],
            "expected_outputs": [f"fixed/{stage}"],
        }
        for stage in materialization._STAGE_ORDER
    }
    prereg = {
        "preregistration_file_sha256": _digest("prereg"),
        "stage_order": list(materialization._STAGE_ORDER),
        "stages": stages,
    }

    def authenticate_prereg() -> dict[str, Any]:
        events.append("prereg")
        return prereg

    def authenticate_unlock(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("unlock")
        return {
            "unlock_file_sha256": _digest("unlock-file"),
            "unlock_identity_sha256": _digest("unlock-identity"),
        }

    def run(*_args: Any, **_kwargs: Any) -> None:
        events.append("child")

    monkeypatch.setattr(
        materialization,
        "authenticate_materialization_preregistration_v95",
        authenticate_prereg,
    )
    monkeypatch.setattr(
        deferred, "authenticate_deferred_final_unlock_v95", authenticate_unlock
    )
    monkeypatch.setattr(materialization, "_authenticate_predecessor_receipts", lambda *_a: None)
    monkeypatch.setattr(
        materialization,
        "_existing_output_identity",
        lambda paths: {str(path): _digest(str(path)) for path in paths},
    )
    monkeypatch.setattr(materialization.subprocess, "run", run)
    monkeypatch.setattr(materialization, "RECEIPT_ROOT", tmp_path / "receipts")

    result = materialization.run_materialization_stage_v95("generate")

    assert events == ["prereg", "unlock", "child"]
    assert result["status"] == "completed_after_authenticated_unlock"
    assert result["automatic_runtime_promotion"] is False


def test_no_model_or_generation_runs_on_preregistration_import_or_recipe_validation() -> None:
    source = inspect.getsource(materialization)
    run_stage = source[source.index("def run_materialization_stage_v95") :]
    assert run_stage.index("authenticate_deferred_final_unlock_v95") < run_stage.index(
        "subprocess.run"
    )
    assert "load_local_language_model" not in source
    assert "Gemma4ForConditionalGeneration" not in source
    assert "legacy_plan_files_opened\": []" in source


def test_fixed_paths_do_not_create_deferred_artifacts_during_tests() -> None:
    for scene_id in final_qa.SCENE_IDS:
        for root in ("data/oracle", "data/rendered", "data_gemma4/features", "data_gemma4/maps"):
            assert not (materialization.PROJECT_ROOT / root / scene_id).exists()
