from __future__ import annotations

import ast
import inspect
from collections import Counter
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.evaluation import v94_strong_causal_ablations as causal
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    QuestionRecord,
    questions_sha256,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData


def _map(count: int = 8) -> MapTensorData:
    return MapTensorData(
        semantic=torch.arange(count * 5, dtype=torch.float32).reshape(count, 5),
        xyz=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10,
        rgb=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 24,
        normal=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 7,
        confidence=torch.linspace(0.1, 0.8, count),
        observation_count=torch.arange(1, count + 1, dtype=torch.float32),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=count,
        input_voxel_size_m=0.05,
    )


def _rows(value: torch.Tensor) -> Counter[tuple[float, ...]]:
    return Counter(tuple(float(item) for item in row) for row in value.tolist())


@pytest.mark.parametrize(
    ("condition", "changed"),
    [
        (causal.SEMANTIC_PAYLOAD_SHUFFLE, "semantic"),
        (causal.POSITION_SPATIAL_SHUFFLE, "xyz"),
        (causal.REMOVE_RGB, "rgb"),
    ],
)
def test_targeted_map_controls_change_exactly_one_channel(
    condition: str, changed: str
) -> None:
    source = _map()
    controlled, receipt = causal.apply_strong_map_control(
        source,
        condition,
        seed=causal.SEED,
        scene_id=causal.SCENE_IDS[0],
    )

    fields = ("semantic", "xyz", "rgb", "normal", "confidence", "observation_count")
    assert receipt["changed_channels"] == [changed]
    assert receipt["all_voxels_processed"] is True
    assert receipt["question_inputs_used"] is False
    for field in fields:
        equal = torch.equal(getattr(source, field), getattr(controlled, field))
        assert equal is (field != changed)


def test_semantic_and_position_shuffles_preserve_row_multisets() -> None:
    source = _map()
    semantic, semantic_receipt = causal.apply_strong_map_control(
        source,
        causal.SEMANTIC_PAYLOAD_SHUFFLE,
        seed=causal.SEED,
        scene_id=causal.SCENE_IDS[0],
    )
    positioned, position_receipt = causal.apply_strong_map_control(
        source,
        causal.POSITION_SPATIAL_SHUFFLE,
        seed=causal.SEED,
        scene_id=causal.SCENE_IDS[0],
    )

    assert _rows(semantic.semantic) == _rows(source.semantic)
    assert _rows(positioned.xyz) == _rows(source.xyz)
    assert semantic_receipt["semantic_multiset_preserved"] is True
    assert position_receipt["semantic_rows_retained_exactly"] is True
    assert semantic_receipt["permutation_sha256"] != position_receipt["permutation_sha256"]


def test_zero_and_full_permutation_cover_all_736_environment_tokens() -> None:
    memory = (
        torch.arange(causal.MEMORY_SHAPE[1], dtype=torch.float32)
        .reshape(1, causal.MEMORY_SHAPE[1], 1)
        .expand(causal.MEMORY_SHAPE)
        .clone()
    )
    zero = causal.zero_full_scene_memory(memory)
    shuffled, receipt = causal.permute_full_scene_memory(
        memory, scene_id=causal.SCENE_IDS[0]
    )
    shuffled_again, repeated = causal.permute_full_scene_memory(
        memory, scene_id=causal.SCENE_IDS[0]
    )

    assert torch.equal(zero[:, :1], memory[:, :1])
    assert torch.equal(zero[:, -1:], memory[:, -1:])
    assert torch.count_nonzero(zero[:, 1:-1]).item() == 0
    assert torch.equal(shuffled[:, :1], memory[:, :1])
    assert torch.equal(shuffled[:, -1:], memory[:, -1:])
    assert not torch.equal(shuffled[:, 1:-1], memory[:, 1:-1])
    assert sorted(shuffled[0, 1:-1, 0].tolist()) == sorted(
        memory[0, 1:-1, 0].tolist()
    )
    assert torch.equal(shuffled, shuffled_again)
    assert receipt == repeated
    assert receipt["interior_token_count"] == 736
    assert receipt["scope"] == "all_736_continuous_environment_tokens"


def _manifest() -> QuestionManifest:
    rows = tuple(
        QuestionRecord(
            scene_id=scene_id,
            question_id=f"q_{scene_index * 36 + ordinal:06d}",
            question=f"Question {ordinal}?",
        )
        for scene_index, scene_id in enumerate(causal.SCENE_IDS)
        for ordinal in range(36)
    )
    return QuestionManifest(
        questions=rows,
        questions_sha256=questions_sha256(rows),
        source_qa_sha256="a" * 64,
        manifest_path=Path("questions.json"),
        manifest_sha256="b" * 64,
    )


def test_representative_profile_is_deterministic_balanced_and_label_blind() -> None:
    profile = causal.evaluation_profile("representative-core")
    first = causal.select_profile_questions(_manifest(), profile)
    second = causal.select_profile_questions(_manifest(), profile)

    assert first.questions == second.questions
    assert first.question_count == 36
    assert Counter(row.scene_id for row in first.questions) == {
        scene_id: 6 for scene_id in causal.SCENE_IDS
    }
    selector_tree = ast.parse(inspect.getsource(causal.select_profile_questions))
    referenced_names = {
        node.id.casefold()
        for node in ast.walk(selector_tree)
        if isinstance(node, ast.Name)
    } | {
        node.attr.casefold()
        for node in ast.walk(selector_tree)
        if isinstance(node, ast.Attribute)
    }
    assert not referenced_names & {"answer", "answers", "label", "labels", "oracle"}


def test_profiles_have_separate_namespaces_and_core_needs_no_compiled_cache() -> None:
    core = causal.evaluation_profile("representative-core")
    full = causal.evaluation_profile("full")
    core_paths = causal.evaluation_paths(core)
    full_paths = causal.evaluation_paths(full)

    assert core.conditions == causal.CORE_CONDITIONS
    assert not set(core.conditions) & set(causal.COMPILED_CONDITIONS)
    assert causal.FULL_INTERIOR_TOKEN_PERMUTATION in full.conditions
    assert causal.REMOVE_NORMALS not in full.conditions
    assert causal.REMOVE_NORMALS not in causal.COMPILED_CONDITIONS
    assert core_paths.predictions != full_paths.predictions
    assert core_paths.score != full_paths.score


def test_unavailable_normal_and_viewpoint_controls_are_unsupported_not_faked() -> None:
    contract = causal.channel_identifiability_contract()

    assert contract["normal"]["available_in_sealed_evaluation_artifact"] is False
    assert contract["normal"]["included_control"] is None
    assert contract["normal"]["excluded_control"] == causal.REMOVE_NORMALS
    assert contract["normal"]["observed_nonzero_value_count_across_six_scenes"] == 0
    assert contract["viewpoint"]["independently_identifiable_before_scene_tokenization"] is False
    assert contract["viewpoint"]["included_control"] is None
    assert "does not consume" in contract["viewpoint"]["reason"]
    assert "viewpoint" not in causal.CONDITIONS


def test_sealed_normal_noop_contract_accepts_exact_zero_and_rejects_nonzero() -> None:
    scene_id = causal.SCENE_IDS[0]
    source = _map(8608)
    source.normal.zero_()

    record = causal._normal_availability_record(source, scene_id)

    assert record == causal._expected_normal_availability()[scene_id]
    assert record["nonzero_value_count"] == 0
    assert record["remove_normals_memory_would_be_identical"] is True
    source.normal[0, 0] = 1.0
    with pytest.raises(RuntimeError, match="normal-channel availability changed"):
        causal._normal_availability_record(source, scene_id)


def test_remove_normals_is_rejected_as_noncompiled_for_this_artifact() -> None:
    with pytest.raises(ValueError, match="not map-compiled"):
        causal.apply_strong_map_control(
            _map(),
            causal.REMOVE_NORMALS,
            seed=causal.SEED,
            scene_id=causal.SCENE_IDS[0],
        )


def test_synthetic_cache_round_trip_covers_every_canonical_compiled_condition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "evaluation_cache"
    cache_root.mkdir()
    monkeypatch.setattr(causal, "CONTROL_CACHE", cache_root)
    monkeypatch.setattr(causal, "MEMORY_SHAPE", (1, 5, 3))
    bindings = {
        "source_chain_sha256": "1" * 64,
        "candidate_weights_sha256": "2" * 64,
        "candidate_state_sha256": "3" * 64,
        "correct_cache_manifest_sha256": "4" * 64,
        "correct_memory_sha256": {
            scene_id: "5" * 64 for scene_id in causal.SCENE_IDS
        },
        "source_maps": {},
    }
    controls: dict[str, dict[str, object]] = {
        condition: {} for condition in causal.COMPILED_CONDITIONS
    }
    for condition_index, condition in enumerate(causal.COMPILED_CONDITIONS, 1):
        for scene_index, scene_id in enumerate(causal.SCENE_IDS, 1):
            memory = torch.full(
                causal.MEMORY_SHAPE,
                float(condition_index * 10 + scene_index),
                dtype=torch.bfloat16,
            )
            row = causal._save_control_memory(
                cache_root / f"{scene_id}__{condition}.safetensors",
                memory,
                condition=condition,
                scene_id=scene_id,
            )
            controls[condition][scene_id] = {
                **row,
                "transform": {
                    "condition": condition,
                    "all_voxels_processed": True,
                    "question_inputs_used": False,
                    "question_dependent_selection": False,
                },
            }
    manifest = {
        "artifact": causal.CACHE_ARTIFACT,
        "schema_version": causal.SCHEMA_VERSION,
        "status": "terminal_posthoc_diagnostic_non_promotable",
        "terminal_diagnostic_only": True,
        "seed": causal.SEED,
        "scene_ids": list(causal.SCENE_IDS),
        "scene_count": 6,
        "compiled_conditions": list(causal.COMPILED_CONDITIONS),
        "shape_each": list(causal.MEMORY_SHAPE),
        "dtype": "bfloat16",
        "compiled_before_questions": True,
        "question_inputs_used": False,
        "question_dependent_retrieval": False,
        "all_memory_slots_retained": True,
        "environmental_text_inputs": [],
        "source_chain_sha256": bindings["source_chain_sha256"],
        "candidate_weights_sha256": bindings["candidate_weights_sha256"],
        "candidate_state_sha256": bindings["candidate_state_sha256"],
        "correct_cache_manifest_sha256": bindings["correct_cache_manifest_sha256"],
        "correct_memory_sha256": bindings["correct_memory_sha256"],
        "source_maps": bindings["source_maps"],
        "channel_identifiability": causal.channel_identifiability_contract(),
        "unsupported_channel_availability": {
            "normal": causal._expected_normal_availability()
        },
        "controls": controls,
        "cannot_alter_v94_gates": True,
        "runtime_promotion_authorized": False,
    }
    causal._write_json_create_once(cache_root / "manifest.json", manifest)

    authenticated = causal.authenticate_control_cache(
        bindings=bindings, require_compile_receipt=False
    )

    assert causal.COMPILED_CONDITIONS == tuple(sorted(causal.COMPILED_CONDITIONS))
    assert tuple(authenticated["manifest"]["controls"]) == causal.COMPILED_CONDITIONS
    assert tuple(authenticated["memories"]) == causal.COMPILED_CONDITIONS
    assert all(
        set(authenticated["memories"][condition]) == set(causal.SCENE_IDS)
        for condition in causal.COMPILED_CONDITIONS
    )


def test_prediction_provenance_binds_candidate_memory_question_and_access_hashes() -> None:
    profile = causal.evaluation_profile("representative-core")
    questions = causal.select_profile_questions(_manifest(), profile)
    source = {
        "source_chain_sha256": "1" * 64,
        "candidate_weights_sha256": "2" * 64,
        "candidate_metadata_sha256": "3" * 64,
        "candidate_state_sha256": "4" * 64,
        "correct_cache_manifest_sha256": "5" * 64,
    }
    hashes = {
        condition: {scene_id: "6" * 64 for scene_id in causal.SCENE_IDS}
        for condition in profile.conditions
    }
    receipts = {scene_id: {"permutation_sha256": "7" * 64} for scene_id in causal.SCENE_IDS}

    provenance = causal._prediction_provenance(
        source, None, questions, profile, hashes, receipts
    )

    assert provenance["candidate_weights_sha256"] == "2" * 64
    assert provenance["bound_memory_sha256"] == hashes
    assert provenance["questions_sha256"] == questions.questions_sha256
    assert provenance["compile_access_sha256"] is None
    assert provenance["terminal_diagnostic_only"] is True
    assert provenance["runtime_promotion_authorized"] is False


def test_compile_predict_and_score_keep_hard_process_ordering() -> None:
    compile_tree = ast.parse(inspect.getsource(causal.compile_control_memory_cache))
    compile_calls = {
        node.func.id
        for node in ast.walk(compile_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    predictor = inspect.getsource(causal.predict_question_only)
    scorer = inspect.getsource(causal.score_label_isolated)

    assert "_questions" not in compile_calls
    assert predictor.index("_bind_profile_memories") < predictor.index("_profile_questions")
    assert "_load_references" not in predictor
    assert scorer.index("authenticate_prediction_bundle") < scorer.index("_load_references")


def test_structured_scoring_reports_drop_and_change_without_promotion() -> None:
    profile = causal.EvaluationProfile(
        name="unit",
        question_count=2,
        questions_per_scene=1,
        conditions=causal.CORE_CONDITIONS,
        output_stem="unit",
    )
    references = [
        {"scene_id": causal.SCENE_IDS[0], "question_id": "q_000000", "answer_type": "presence", "answer": "yes"},
        {"scene_id": causal.SCENE_IDS[1], "question_id": "q_000001", "answer_type": "presence", "answer": "no"},
    ]
    rows = [
        {
            "scene_id": causal.SCENE_IDS[0],
            "question_id": "q_000000",
            "primary_prediction": "yes",
            "zero_full_scene_prediction": "no",
            "wrong_scene_swap_prediction": "no",
            "full_interior_token_permutation_prediction": "yes",
        },
        {
            "scene_id": causal.SCENE_IDS[1],
            "question_id": "q_000001",
            "primary_prediction": "no",
            "zero_full_scene_prediction": "no",
            "wrong_scene_swap_prediction": "yes",
            "full_interior_token_permutation_prediction": "yes",
        },
    ]

    result = causal.score_records(references, rows, profile)

    assert result["arms"][causal.PRIMARY]["accuracy"] == 1.0
    assert result["comparisons"][causal.WRONG_SCENE_SWAP]["accuracy_drop_from_primary"] == 1.0
    assert result["comparisons"][causal.ZERO_FULL_SCENE]["prediction_change_count"] == 1
    assert result["runtime_promotion_authorized"] is False


def test_cli_exposes_only_diagnostic_actions_and_profiles() -> None:
    parser = causal._parser()

    assert parser.parse_args(["predict", "--profile", "representative-core"]).profile == (
        "representative-core"
    )
    assert parser.parse_args(["score", "--profile", "full"]).profile == "full"
    with pytest.raises(SystemExit):
        parser.parse_args(["promote"])
