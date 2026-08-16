from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v62_pair_disjoint_preregistration as v62

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_QA = _PROJECT_ROOT / "data_diverse52" / "qa" / "train.jsonl"
_V61_TERMINAL = (
    _PROJECT_ROOT
    / "reports"
    / "gemma4"
    / "metrics"
    / "v61_scene_conditioned_route_generalization_gate.json"
)


def _destinations(tmp_path: Path) -> dict[str, Path]:
    return {
        "filtered_train_output": tmp_path / "training-only" / "train.jsonl",
        "validation_questions_output": tmp_path / "inference-only" / "questions.json",
        "scorer_references_output": tmp_path / "scorer-only" / "references.json",
        "preregistration_output": tmp_path / "metrics" / "preregistration.json",
    }


def _prepare(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    destinations = _destinations(tmp_path)
    result = v62.prepare(
        source_train_qa=_SOURCE_QA,
        v61_terminal=_V61_TERMINAL,
        **destinations,
    )
    return result, destinations


def test_prepare_emits_pair_disjoint_create_once_artifacts(tmp_path: Path) -> None:
    preregistration, paths = _prepare(tmp_path)

    training_rows = tuple(
        json.loads(line)
        for line in paths["filtered_train_output"].read_text(encoding="utf-8").splitlines()
    )
    assert len(training_rows) == 576
    assert {row["counterfactual_pair_id"] for row in training_rows} == set(v62.TRAIN_PAIR_IDS)
    assert all("answer" in row and "counterfactual_expected_change" in row for row in training_rows)
    assert not {
        int(row["scene_id"].removeprefix("scene_")) for row in training_rows
    } & (set(range(25, 31)) | set(range(57, 63)))
    assert hashlib.sha256(paths["filtered_train_output"].read_bytes()).hexdigest() == (
        "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1"
    )
    assert len(v62.load_filtered_training_qa(paths["filtered_train_output"])) == 576

    questions = json.loads(paths["validation_questions_output"].read_text(encoding="utf-8"))
    assert questions["question_count"] == 384
    assert questions["scene_count"] == 16
    assert all(
        set(record) == {"scene_id", "question_id", "question"}
        for record in questions["questions"]
    )
    prohibited_question_fields = {
        "answer",
        "answer_type",
        "route_label",
        "counterfactual_pair_id",
        "counterfactual_expected_change",
    }
    assert all(not (set(record) & prohibited_question_fields) for record in questions["questions"])

    sidecar = json.loads(paths["scorer_references_output"].read_text(encoding="utf-8"))
    assert sidecar["question_count"] == 384
    assert sidecar["paired_unit_count"] == 192
    assert sidecar["contains_question_text"] is False
    assert sidecar["runtime_access_permitted"] is False
    assert all("question" not in record for record in sidecar["records"])
    assert all(type(record["route_label"]) is bool for record in sidecar["records"])
    assert sum(record["route_label"] for record in sidecar["records"]) == 52

    assert preregistration["preserved_v61_terminal"] == {
        "artifact": "v61_scene_conditioned_route_generalization_gate",
        "sha256": "cb7ce887c03dc156693cca489e7638d32adc1ad11cbd0f33464bcfcc4ae5db38",
        "passed": False,
        "may_be_replaced_or_reinterpreted": False,
    }
    split = preregistration["split"]
    assert set(split["training_pair_ids"]).isdisjoint(split["internal_validation_pair_ids"])
    assert set(split["training_scene_ids"]).isdisjoint(split["internal_validation_scene_ids"])
    assert split["protected_scene_numbers_never_loaded"] == [
        *range(25, 31),
        *range(57, 63),
    ]
    assert preregistration["source"]["read_count_during_prepare"] == 1
    assert preregistration["data_boundaries"]["held_out_qa_or_oracle_loaded"] is False

    artifacts = preregistration["artifacts"]
    assert artifacts["filtered_training"]["sha256"] == hashlib.sha256(
        paths["filtered_train_output"].read_bytes()
    ).hexdigest()
    assert artifacts["internal_validation_questions"]["sha256"] == hashlib.sha256(
        paths["validation_questions_output"].read_bytes()
    ).hexdigest()
    assert artifacts["scorer_references"]["sha256"] == hashlib.sha256(
        paths["scorer_references_output"].read_bytes()
    ).hexdigest()
    assert json.loads(paths["preregistration_output"].read_text(encoding="utf-8")) == (
        preregistration
    )


def test_preregistration_locks_natural_metrics_pair_completeness_and_controls(
    tmp_path: Path,
) -> None:
    preregistration, _paths = _prepare(tmp_path)

    natural = preregistration["natural_population"]
    assert natural["training"] | {}  # Mapping-like and JSON serializable.
    assert {
        key: natural["training"][key]
        for key in (
            "pair_count",
            "scene_count",
            "row_count",
            "paired_unit_count",
            "changed_side_count",
            "retention_side_count",
            "changed_unit_count",
            "retention_unit_count",
        )
    } == {
        "pair_count": 12,
        "scene_count": 24,
        "row_count": 576,
        "paired_unit_count": 288,
        "changed_side_count": 80,
        "retention_side_count": 496,
        "changed_unit_count": 40,
        "retention_unit_count": 248,
    }
    assert {
        key: natural["internal_validation"][key]
        for key in (
            "pair_count",
            "scene_count",
            "row_count",
            "paired_unit_count",
            "changed_side_count",
            "retention_side_count",
            "changed_unit_count",
            "retention_unit_count",
        )
    } == {
        "pair_count": 8,
        "scene_count": 16,
        "row_count": 384,
        "paired_unit_count": 192,
        "changed_side_count": 52,
        "retention_side_count": 332,
        "changed_unit_count": 26,
        "retention_unit_count": 166,
    }
    assert natural["primary_reporting_uses_all_384_sides_without_rebalancing"] is True
    assert natural["changed_unit_completeness_requires_both_sides_correct"] is True

    thresholds = preregistration["thresholds"]
    internal = thresholds["internal_validation"]
    assert internal["changed_side_exact"] == {"minimum": 42, "total": 52}
    assert internal["changed_paired_unit_complete"] == {"minimum": 19, "total": 26}
    assert internal["changed_paired_unit_correct_direction"] == {
        "minimum": 23,
        "total": 26,
    }
    assert internal["retention_exact_no_control_output_identity"] == {
        "minimum": 332,
        "total": 332,
        "comparison": "exact_utf8_output_bytes_sha256",
    }
    same_prefix = thresholds["same_question_different_prefix_control"]
    assert same_prefix["question_text_identity"] == {"minimum": 26, "total": 26}
    assert same_prefix["distinct_scene_prefix_hashes"] == {"minimum": 26, "total": 26}
    scene_swap = thresholds["scene_swap_control"]
    assert scene_swap["swapped_side_coverage"] == {"minimum": 52, "total": 52}
    assert scene_swap["answer_follows_injected_scene"] == {"minimum": 42, "total": 52}


def test_prepare_reads_the_complete_source_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _SOURCE_QA.resolve()
    original = Path.read_bytes
    source_reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal source_reads
        if path.resolve() == source:
            source_reads += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    _prepare(tmp_path)
    assert source_reads == 1


def test_refuses_overwrite_before_opening_either_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destinations = _destinations(tmp_path)
    destinations["preregistration_output"].parent.mkdir(parents=True)
    destinations["preregistration_output"].write_text("existing\n", encoding="utf-8")

    def forbidden_read(_path: Path) -> bytes:
        raise AssertionError("input opened before create-once refusal")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(FileExistsError, match="create-once output already exists"):
        v62.prepare(
            source_train_qa=_SOURCE_QA,
            v61_terminal=_V61_TERMINAL,
            **destinations,
        )
    assert destinations["preregistration_output"].read_text(encoding="utf-8") == "existing\n"


def test_scorer_sidecar_requires_physical_directory_separation(tmp_path: Path) -> None:
    destinations = _destinations(tmp_path)
    destinations["scorer_references_output"] = (
        destinations["validation_questions_output"].parent / "references.json"
    )
    with pytest.raises(ValueError, match="separate directories"):
        v62.prepare(
            source_train_qa=_SOURCE_QA,
            v61_terminal=_V61_TERMINAL,
            **destinations,
        )
    assert not any(path.exists() for path in destinations.values())


def test_pinned_sha_is_followed_by_complete_inventory_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [json.loads(line) for line in _SOURCE_QA.read_text(encoding="utf-8").splitlines()]
    rows[0]["counterfactual_pair_id"] = "pair_999999"
    tampered_raw = b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    tampered = tmp_path / "tampered-source.jsonl"
    tampered.write_bytes(tampered_raw)
    monkeypatch.setattr(v62, "_PINNED_SOURCE_QA_SHA256", hashlib.sha256(tampered_raw).hexdigest())
    monkeypatch.setattr(v62, "_PINNED_SOURCE_QA_SIZE_BYTES", len(tampered_raw))
    destinations = _destinations(tmp_path / "outputs")

    with pytest.raises(ValueError, match="unregistered pair"):
        v62.prepare(
            source_train_qa=tampered,
            v61_terminal=_V61_TERMINAL,
            **destinations,
        )
    assert not any(path.exists() for path in destinations.values())


def test_v61_failed_terminal_is_preserved_not_reinterpreted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal_payload = json.loads(_V61_TERMINAL.read_text(encoding="utf-8"))
    terminal_payload["passed"] = True
    raw = (json.dumps(terminal_payload, sort_keys=True) + "\n").encode()
    terminal = tmp_path / "changed-terminal.json"
    terminal.write_bytes(raw)
    monkeypatch.setattr(v62, "_PINNED_V61_TERMINAL_SHA256", hashlib.sha256(raw).hexdigest())
    destinations = _destinations(tmp_path / "outputs")

    with pytest.raises(ValueError, match="failed terminal must be preserved"):
        v62.prepare(
            source_train_qa=_SOURCE_QA,
            v61_terminal=terminal,
            **destinations,
        )
    assert not any(path.exists() for path in destinations.values())


def test_future_trainer_data_contract_accepts_only_filtered_training_path() -> None:
    parser = argparse.ArgumentParser()
    v62.add_filtered_training_data_argument(parser)
    destinations = {action.dest for action in parser._actions if action.dest != "help"}
    assert destinations == set(v62.V62_TRAINER_DATA_ARGUMENTS) == {"filtered_train_qa"}
    assert destinations.isdisjoint(v62.V62_PROHIBITED_TRAINER_DATA_ARGUMENTS)
    assert parser.parse_args(["--filtered-train-qa", "filtered.jsonl"]).filtered_train_qa == (
        "filtered.jsonl"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--scorer-references", "references.json"])
    with pytest.raises(ValueError, match="Filtered V62 training QA SHA-256"):
        v62.load_filtered_training_qa(_SOURCE_QA)


def test_group_publish_rolls_back_when_a_later_create_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destinations = _destinations(tmp_path)
    blocked = destinations["scorer_references_output"].resolve()
    original_open = Path.open

    def fail_scorer_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if path.resolve() == blocked and mode == "xb":
            raise OSError("synthetic scorer write failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_scorer_open)
    with pytest.raises(OSError, match="synthetic scorer write failure"):
        v62.prepare(
            source_train_qa=_SOURCE_QA,
            v61_terminal=_V61_TERMINAL,
            **destinations,
        )
    assert not any(path.exists() for path in destinations.values())


def _write_baseline_predictions(path: Path, questions_path: Path) -> None:
    manifest = json.loads(questions_path.read_text(encoding="utf-8"))
    with path.open("x", encoding="utf-8") as handle:
        for row in manifest["questions"]:
            scene_id = row["scene_id"]
            handle.write(
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "question_id": row["question_id"],
                        "predicted_answer": (
                            f"private-v54-output::{scene_id}::{row['question_id']}"
                        ),
                        "prefix_hash": hashlib.sha256(scene_id.encode()).hexdigest(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    scene_ids = sorted({row["scene_id"] for row in manifest["questions"]})
    path.with_suffix(path.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "references_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
                "checkpoint_sha256": (
                    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
                ),
                "scene_map_manifest": {scene_id: {} for scene_id in scene_ids},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_lock_baseline_is_hash_only_and_binds_all_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preregistration, paths = _prepare(tmp_path / "prepared")
    predictions = tmp_path / "baseline-predictions.jsonl"
    _write_baseline_predictions(predictions, paths["validation_questions_output"])
    monkeypatch.setattr(
        v62,
        "checkpoint_fingerprint",
        lambda _path: (
            "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8",
            [
                {
                    "path": "adapter.safetensors",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                }
            ],
        ),
    )
    output = tmp_path / "authorization" / "baseline-lock.json"
    lock = v62.lock_baseline(
        predictions=predictions,
        preregistration=paths["preregistration_output"],
        v54_checkpoint=tmp_path / "checkpoint-not-read",
        output=output,
    )

    assert lock["schema"] == "semantic_3d_chat.v62.v54_no_control_baseline_lock.v1"
    assert lock["artifact"] == "v62_v54_no_control_baseline_lock"
    assert lock["question_count"] == 384
    assert lock["scene_count"] == 16
    assert lock["one_invariant_prefix_per_scene"] is True
    assert lock["distinct_prefix_per_scene"] is True
    assert lock["preregistration_sha256"] == hashlib.sha256(
        paths["preregistration_output"].read_bytes()
    ).hexdigest()
    assert lock["questions_manifest_sha256"] == preregistration["artifacts"][
        "internal_validation_questions"
    ]["sha256"]
    assert lock["question_key_inventory_sha256"] == (
        "f36885e43100a5b7a3682ca38f7a06187c1f9b204095f5dc89b2e597e227ba27"
    )
    assert len(lock["required_output_hashes"]) == 384
    assert all(
        set(record) == {"scene_id", "question_id", "raw_output_sha256"}
        for record in lock["required_output_hashes"]
    )
    serialized = output.read_text(encoding="utf-8")
    assert "private-v54-output" not in serialized
    assert lock["environmental_answer_text_stored"] is False
    assert lock["question_text_stored"] is False
    assert lock["filtered_training_qa_loaded"] is False
    assert lock["scorer_references_loaded"] is False
    assert v62.validate_baseline_lock(
        output,
        preregistration=paths["preregistration_output"],
    ) == lock
    assert v62.validate_baseline_lock(output) == lock


def test_lock_baseline_rejects_prefix_drift_inventory_drift_and_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _preregistration, paths = _prepare(tmp_path / "prepared")
    predictions = tmp_path / "baseline-predictions.jsonl"
    _write_baseline_predictions(predictions, paths["validation_questions_output"])
    monkeypatch.setattr(
        v62,
        "checkpoint_fingerprint",
        lambda _path: (
            "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8",
            [],
        ),
    )
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    rows[1]["prefix_hash"] = "f" * 64
    drifted = tmp_path / "drifted.jsonl"
    drifted.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    drifted.with_suffix(drifted.suffix + ".provenance.json").write_bytes(
        predictions.with_suffix(predictions.suffix + ".provenance.json").read_bytes()
    )
    with pytest.raises(ValueError, match="one invariant prefix"):
        v62.lock_baseline(
            predictions=drifted,
            preregistration=paths["preregistration_output"],
            v54_checkpoint=tmp_path / "checkpoint-not-read",
            output=tmp_path / "drifted-lock.json",
        )

    missing = tmp_path / "missing.jsonl"
    missing.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[:-1]),
        encoding="utf-8",
    )
    missing.with_suffix(missing.suffix + ".provenance.json").write_bytes(
        predictions.with_suffix(predictions.suffix + ".provenance.json").read_bytes()
    )
    with pytest.raises(ValueError, match="inventory differs"):
        v62.lock_baseline(
            predictions=missing,
            preregistration=paths["preregistration_output"],
            v54_checkpoint=tmp_path / "checkpoint-not-read",
            output=tmp_path / "missing-lock.json",
        )

    existing = tmp_path / "existing-lock.json"
    existing.write_text("do-not-overwrite\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        v62.lock_baseline(
            predictions=predictions,
            preregistration=tmp_path / "missing-preregistration.json",
            v54_checkpoint=tmp_path / "checkpoint-not-read",
            output=existing,
        )
    assert existing.read_text(encoding="utf-8") == "do-not-overwrite\n"


def test_validate_baseline_lock_rejects_answer_text_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _preregistration, paths = _prepare(tmp_path / "prepared")
    predictions = tmp_path / "predictions.jsonl"
    _write_baseline_predictions(predictions, paths["validation_questions_output"])
    monkeypatch.setattr(
        v62,
        "checkpoint_fingerprint",
        lambda _path: (
            "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8",
            [],
        ),
    )
    lock_path = tmp_path / "lock.json"
    v62.lock_baseline(
        predictions=predictions,
        preregistration=paths["preregistration_output"],
        v54_checkpoint=tmp_path / "checkpoint-not-read",
        output=lock_path,
    )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["answer"] = "environment text"
    tampered = tmp_path / "tampered-lock.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="strict schema"):
        v62.validate_baseline_lock(
            tampered,
            preregistration=paths["preregistration_output"],
        )
