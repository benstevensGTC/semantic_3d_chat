from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.training import train_fixed_prefix_decoder_reader_v6_2 as train
from semantic_3d_chat.training.train_fixed_prefix_ple_v54 import load_training_records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v6_2_authenticates_exact_consumed_v6_1_failure() -> None:
    evidence = train.authenticate_v6_1_terminal_failure()

    assert evidence == {
        "passed": True,
        "release_sha256": train.V6_1_RELEASE_SHA256,
        "attempt_sha256": train.V6_1_ATTEMPT_SHA256,
        "terminal_failure_sha256": train.V6_1_FAILURE_SHA256,
        "objective_equivalence_recomputed_passed": True,
        "full_reference_gradient_branches": ["aggregate", "broad", "correct", "wrong"],
        "full_reference_gradient_coverage": list(train.TARGET_MODULES),
        "full_reference_gradients_finite_nonzero": True,
        "full_reference_lora_a_exact_zero": True,
        "sole_failed_comparison": "tail_vs_full_aggregate",
        "failed_aggregate_gates": ["cosine", "relative_l2"],
    }


def test_v6_2_rejects_any_v6_1_byte_change(monkeypatch: pytest.MonkeyPatch) -> None:
    original = train._sha256_file

    def changed(path: str | Path) -> str:
        if str(path) == train.V6_1_ATTEMPT:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(train, "_sha256_file", changed)
    with pytest.raises(ValueError, match="artifact bytes changed"):
        train.authenticate_v6_1_terminal_failure()


def test_v6_2_qa_loss_is_full_hf_forward_token_normalized_ce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = torch.tensor([[-100, -100, 2, 3]], dtype=torch.long)
    logits = torch.zeros((1, 4, 5), dtype=torch.float32, requires_grad=True)
    logits.data[0, 1, 2] = 3.0
    logits.data[0, 2, 3] = 2.0
    expected = train.v1.token_normalized_nll(logits, labels)[0]
    seen: dict[str, object] = {}

    class Model:
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            seen.update(kwargs)
            return SimpleNamespace(logits=logits, loss=expected)

    prepared = SimpleNamespace(
        inputs_embeds=torch.zeros((1, 4, 3)),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        labels=labels,
        per_layer_inputs=torch.zeros((1, 4, 2, 2)),
        mm_token_type_ids=torch.zeros((1, 4), dtype=torch.long),
    )
    monkeypatch.setattr(train.v1, "_prepared_batch", lambda *_args: prepared)
    bundle = SimpleNamespace(language=SimpleNamespace(model=Model()))

    observed = train.answer_nll(bundle, torch.zeros(1), SimpleNamespace())

    assert torch.equal(observed, expected)
    assert seen["labels"] is labels
    assert seen["use_cache"] is False
    assert seen["return_dict"] is True
    assert "logits_to_keep" not in seen
    observed.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_v6_2_source_never_uses_shape_specialized_qa_training() -> None:
    source = Path(
        "src/semantic_3d_chat/training/train_fixed_prefix_decoder_reader_v6_2.py"
    ).read_text(encoding="utf-8")

    assert "gemma4_answer_tail" not in source
    assert "answer_tail" not in source
    assert '"logits_to_keep"' not in source
    assert "labels=prepared.labels" in source
    assert "v1.token_normalized_nll(logits, prepared.labels)" in source
    assert train._QA_FORWARD_PATH == "full_huggingface_forward_token_normalized_ce"


def test_v6_2_release_binds_required_dependency_closure_and_inputs() -> None:
    required = {
        "src/semantic_3d_chat/evaluation/baseline_io.py",
        "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_preregistration.py",
        "src/semantic_3d_chat/evaluation/metrics.py",
        "src/semantic_3d_chat/evaluation/v55_development_score.py",
        "src/semantic_3d_chat/training/train_question_control_v56.py",
        "src/semantic_3d_chat/training/train_fixed_prefix_decoder_reader_v6_1.py",
        "src/semantic_3d_chat/training/train_fixed_prefix_decoder_reader_v6_2.py",
    }
    assert required <= set(train.TRAINING_BOUND_PATHS)
    assert set(train.v6_release.SMOKE_BOUND_PATHS) <= set(train.TRAINING_BOUND_PATHS)
    assert "src/semantic_3d_chat/language/fixed_prefix_decoder_reader_v6.py" in (
        train.TRAINING_BOUND_PATHS
    )
    sources = train._source_hashes()
    assert set(sources) == set(train.TRAINING_BOUND_PATHS)
    assert all(len(value) == 64 for value in sources.values())
    assets = train._training_asset_hashes()
    assert train.BASE_CHECKPOINT in {
        str(Path(key).parent) for key in assets if key.endswith("adapter.safetensors")
    }
    assert train.BASE_RUNTIME_CONFIG in assets
    assert train.v1.TRAIN_QA in assets
    assert train.v1.VALIDATION_QUESTIONS in assets
    assert train.v1.VALIDATION_REFERENCES in assets
    assert train.v1.RETENTION in assets
    assert train.v1.BASELINE_PREDICTIONS in assets
    assert len([key for key in assets if key.startswith(f"{train.v1.PREFIX_CACHE}/")]) == 41


def test_v6_2_bound_sources_cover_recursive_local_import_closure() -> None:
    source_root = Path("src")
    module_paths: dict[str, Path] = {}
    path_modules: dict[str, str] = {}
    for path in source_root.rglob("*.py"):
        parts = list(path.relative_to(source_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        module_paths[module] = path
        path_modules[path.as_posix()] = module
    bound = {
        Path(path).as_posix()
        for path in train.TRAINING_BOUND_PATHS
        if path.startswith("src/") and path.endswith(".py")
    }
    closure: set[str] = set()
    pending = list(bound)
    while pending:
        raw = pending.pop()
        if raw in closure or not Path(raw).is_file():
            continue
        closure.add(raw)
        path = Path(raw)
        module = path_modules[raw]
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package.split(".") if package else []
                    keep = len(package_parts) - (node.level - 1)
                    base_parts = package_parts[:keep]
                    if node.module:
                        base_parts.append(node.module)
                    base = ".".join(base_parts)
                else:
                    base = node.module or ""
                candidates = [base, *(
                    f"{base}.{alias.name}" if base else alias.name
                    for alias in node.names
                    if alias.name != "*"
                )]
            for candidate in candidates:
                dependency = module_paths.get(candidate)
                if dependency is not None:
                    pending.append(dependency.as_posix())
    assert closure <= bound, f"unbound local imports: {sorted(closure - bound)}"


def test_v6_2_model_binding_calls_real_v6_1_authenticators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = {
        "model_weights_blob_sha256": "a" * 64,
        "model_weights_size_bytes": 123,
        "actual_model_bytes_streamed": True,
    }
    transformers_sources = {
        "transformers.models.gemma4.modeling_gemma4": {
            "sha256": "b" * 64,
            "size_bytes": 456,
            "basename": "modeling_gemma4.py",
        }
    }
    snapshot = {
        "model.safetensors": {
            "resolved_blob": "blob",
            "sha256": "a" * 64,
            "size_bytes": 123,
        }
    }
    monkeypatch.setattr(
        train.v61_release, "_authenticate_model_blob", lambda: weights
    )
    monkeypatch.setattr(
        train.v61_release,
        "_installed_transformers_sources",
        lambda: transformers_sources,
    )
    monkeypatch.setattr(train, "_local_model_snapshot_inventory", lambda: snapshot)

    assert train._current_model_binding() == {
        "weights": weights,
        "snapshot_files": snapshot,
        "installed_transformers_sources": transformers_sources,
    }


def test_v6_2_build_training_release_exercises_complete_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = {"passed": True, "terminal_failure_sha256": train.V6_1_FAILURE_SHA256}
    sources = {"source.py": "a" * 64}
    assets = {"asset.bin": {"sha256": "b" * 64, "size_bytes": 1}}
    model = {
        "weights": {"model_weights_blob_sha256": "c" * 64},
        "snapshot_files": {"config.json": {"sha256": "d" * 64}},
        "installed_transformers_sources": {"module": {"sha256": "e" * 64}},
    }
    monkeypatch.setattr(train, "authenticate_v6_1_terminal_failure", lambda: lineage)
    monkeypatch.setattr(train, "_source_hashes", lambda: sources)
    monkeypatch.setattr(train, "_training_asset_hashes", lambda: assets)
    monkeypatch.setattr(train, "_current_model_binding", lambda: model)

    release = train.build_training_release()

    assert release["sealed_v6_1_terminal_failure"] is lineage
    assert release["bound_source_sha256"] is sources
    assert release["bound_training_asset_sha256"] is assets
    assert release["local_model_binding"] is model
    assert release["authorized"]["exact_optimizer_updates"] == 96


def test_v6_2_training_release_is_create_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "release.json"
    payload = {
        "schema_version": 1,
        "artifact": "test",
        "status": "released_exactly_one_full_reference_96_update_training_run",
    }
    monkeypatch.setattr(train, "build_training_release", lambda: payload)

    path, digest = train.write_training_release(destination)

    assert path == destination.resolve()
    assert digest == _sha256(destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="create-once"):
        train.write_training_release(destination)


def _retention_metrics() -> dict[str, object]:
    records = [
        {
            "index": index,
            "target_token_id": 10 + index,
            "baseline_ce_nats": 1.0 + index / 100,
            "current_ce_nats": 1.01 + index / 100,
            "ce_increase_nats": 0.010000000000000009,
            "kl_nats": 0.001 + index / 10000,
            "baseline_top1_token_id": 20 + index,
            "current_top1_token_id": 20 + index,
            "top1_agreement": True,
        }
        for index in range(16)
    ]
    increases = [float(row["ce_increase_nats"]) for row in records]
    kls = [float(row["kl_nats"]) for row in records]
    return {
        "example_count": 16,
        "records": records,
        "mean_ce_increase_nats": sum(increases) / 16,
        "maximum_ce_increase_nats": max(increases),
        "mean_kl_nats": sum(kls) / 16,
        "maximum_kl_nats": max(kls),
        "next_token_top1_agreement": 1.0,
        "metrics_sha256": train._canonical_hash(records),
    }


def test_v6_2_retention_auth_recomputes_all_raw_measurements() -> None:
    metrics = _retention_metrics()
    train._authenticate_retention_evidence(metrics)

    for field, value in (
        ("mean_kl_nats", 0.5),
        ("metrics_sha256", "0" * 64),
    ):
        tampered = copy.deepcopy(metrics)
        tampered[field] = value
        with pytest.raises(ValueError, match="derived metric"):
            train._authenticate_retention_evidence(tampered)
    negative = copy.deepcopy(metrics)
    negative["records"][0]["kl_nats"] = -0.1
    with pytest.raises(ValueError, match="raw record"):
        train._authenticate_retention_evidence(negative)
    negative = copy.deepcopy(metrics)
    negative["records"][0]["current_ce_nats"] = -0.1
    with pytest.raises(ValueError, match="raw record"):
        train._authenticate_retention_evidence(negative)


def test_v6_2_training_retention_clamps_epsilon_but_rejects_material_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train.v1, "retention_kl_loss", lambda *_args: torch.tensor(-1e-8)
    )
    clamped = train.retention_kl_loss_v6_2(None, {}, torch.zeros(1))
    assert clamped.item() == 0.0
    monkeypatch.setattr(
        train.v1, "retention_kl_loss", lambda *_args: torch.tensor(-2e-6)
    )
    with pytest.raises(RuntimeError, match="materially negative"):
        train.retention_kl_loss_v6_2(None, {}, torch.zeros(1))


def _greedy_metrics() -> dict[str, object]:
    validation = train.v1.load_validation_records()
    selected = train.v1._greedy_subset(validation)
    baseline_index = train.v1._baseline_prediction_index()
    manifest = train._read_json(Path(train.v1.PREFIX_CACHE) / "manifest.json")
    records = []
    for row in selected:
        key = (row.scene_id, row.question_id)
        baseline = train.v1.normalize_answer(baseline_index[key])
        candidate = train.v1.normalize_answer(row.answer)
        records.append(
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "baseline_correct": bool(
                    train.v1.canonical_type_specific_match(
                        row.answer_type, baseline, row.answer
                    )
                ),
                "candidate_correct": bool(
                    train.v1.canonical_type_specific_match(
                        row.answer_type, candidate, row.answer
                    )
                ),
                "normalized_baseline_prediction": baseline,
                "normalized_candidate_prediction": candidate,
                "normalized_baseline_prediction_sha256": hashlib.sha256(
                    baseline.encode()
                ).hexdigest(),
                "normalized_candidate_prediction_sha256": hashlib.sha256(
                    candidate.encode()
                ).hexdigest(),
                "prefix_sha256": manifest["scenes"][row.scene_id]["prefix_sha256"],
            }
        )
    baseline_correct = sum(bool(row["baseline_correct"]) for row in records)
    candidate_correct = sum(bool(row["candidate_correct"]) for row in records)
    return {
        "row_count": 96,
        "records": records,
        "baseline_exact_correct": baseline_correct,
        "baseline_exact_accuracy": baseline_correct / 96,
        "candidate_exact_correct": candidate_correct,
        "candidate_exact_accuracy": candidate_correct / 96,
        "exact_accuracy_delta": (candidate_correct - baseline_correct) / 96,
        "prediction_records_sha256": train._canonical_hash(records),
        "question_dependent_scene_retrieval": False,
    }


def test_v6_2_greedy_auth_recomputes_96_raw_rows() -> None:
    metrics = _greedy_metrics()
    validation = train.v1.load_validation_records()
    assert train._authenticate_greedy_evidence(metrics, validation) is True

    tampered = copy.deepcopy(metrics)
    tampered["candidate_exact_correct"] = 49
    with pytest.raises(ValueError, match="derived metric"):
        train._authenticate_greedy_evidence(tampered, validation)
    tampered = copy.deepcopy(metrics)
    tampered["records"][0]["candidate_correct"] = False
    with pytest.raises(ValueError, match="raw record"):
        train._authenticate_greedy_evidence(tampered, validation)


def _valid_training_evidence() -> dict[str, object]:
    rows = load_training_records()
    schedule = train.build_v6_schedule(rows)
    wrong = train.answer_varying_wrong_prefixes(rows)
    trace: list[dict[str, object]] = []
    for update_index, update in enumerate(schedule, start=1):
        contrastive = []
        for row in update.contrastive:
            correct = 1.0
            wrong_nll = 1.2
            margin = wrong_nll - correct
            hinge = max(0.0, 0.5 - margin)
            contrastive.append(
                {
                    "scene_id": row.scene_id,
                    "question_id": row.question_id,
                    "wrong_scene_id": wrong[(row.scene_id, row.question_id)],
                    "correct_nll": correct,
                    "wrong_nll": wrong_nll,
                    "margin": margin,
                    "hinge": hinge,
                    "weighted_objective": (0.5 / 3.0) * correct
                    + (4.0 / 3.0) * hinge,
                }
            )
        broad = [
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "nll": 2.0,
                "weighted_objective": (0.5 / 3.0) * 2.0,
            }
            for row in update.broad
        ]
        trace.append(
            train._trace_item(
                update=update_index,
                learning_rate=train.learning_rate_v6(update_index),
                contrastive=contrastive,
                broad=broad,
                retention_index=(update_index - 1) % 16,
                retention_kl=0.001,
                gradient=0.5,
                adapter_hash=f"{update_index:064x}",
            )
        )
    return {
        "qa_forward_path": train._QA_FORWARD_PATH,
        "full_sequence_logits": True,
        "optimizer": "AdamW",
        "optimizer_kwargs": json.loads(json.dumps(train.optimizer_kwargs())),
        "updates": 96,
        "contrastive_rows_consumed_exactly_once": 288,
        "broad_rows_consumed_exactly_once": 288,
        "retention_examples": 16,
        "retention_exposures_per_example": 6,
        "trainable_parameter_count": train.LORA_PARAMETER_COUNT,
        "maximum_preclip_gradient_l2": 0.5,
        "initial_trace": trace[:3],
        "milestone_trace": [trace[index - 1] for index in (24, 48, 72, 96)],
        "final_trace": trace[-3:],
        "trace": trace,
        "trace_sha256": train._canonical_hash(trace),
        "final_adapter_state_sha256": trace[-1]["adapter_state_sha256"],
        "intermediate_selection_or_checkpoint": False,
        "gradient_checkpointing": False,
    }


def test_v6_2_trace_auth_recomputes_schedule_objectives_and_rotation() -> None:
    rows = load_training_records()
    evidence = _valid_training_evidence()
    train._authenticate_training_trace(evidence, rows)

    negative = copy.deepcopy(evidence)
    negative["trace"][0]["broad_components"][0]["nll"] = -1.0
    negative["trace_sha256"] = train._canonical_hash(negative["trace"])
    with pytest.raises(ValueError, match="broad trace"):
        train._authenticate_training_trace(negative, rows)
    rotated = copy.deepcopy(evidence)
    rotated["trace"][5]["retention_index"] = 9
    rotated["trace_sha256"] = train._canonical_hash(rotated["trace"])
    with pytest.raises(ValueError, match="rotation"):
        train._authenticate_training_trace(rotated, rows)
    changed_row = copy.deepcopy(evidence)
    changed_row["trace"][0]["contrastive_components"][0]["question_id"] = "wrong"
    changed_row["trace_sha256"] = train._canonical_hash(changed_row["trace"])
    with pytest.raises(ValueError, match="contrastive trace"):
        train._authenticate_training_trace(changed_row, rows)


def _audit_payload(required: set[str], forbidden_roots: list[Path]) -> dict[str, object]:
    loaded = sorted(required)
    return {
        "schema_version": 1,
        "artifact": f"{train.ARTIFACT}_file_audit",
        "loaded_files": loaded,
        "loaded_file_count": len(loaded),
        "loaded_file_inventory_sha256": train._canonical_hash(loaded),
        "forbidden_roots": [str(path) for path in forbidden_roots],
        "forbidden_component_names": ["oracle"],
        "block_forbidden": True,
        "forbidden_accesses": [],
        "passed": True,
    }


def test_v6_2_strict_json_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        train._read_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        train._read_json(nonfinite)


def test_v6_2_audit_auth_requires_exact_roots_files_and_recomputes_forbidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    required = {str((tmp_path / "runtime.yaml").resolve()), str((tmp_path / "model").resolve())}
    forbidden_roots = [(tmp_path / "held_out").resolve()]
    monkeypatch.setattr(train, "_required_loaded_paths", lambda: required)
    monkeypatch.setattr(train, "training_forbidden_roots", lambda: forbidden_roots)
    audit = _audit_payload(required, forbidden_roots)
    assert train._authenticate_audit(audit) is True

    missing = copy.deepcopy(audit)
    missing["loaded_files"] = missing["loaded_files"][:-1]
    missing["loaded_file_count"] -= 1
    missing["loaded_file_inventory_sha256"] = train._canonical_hash(missing["loaded_files"])
    with pytest.raises(ValueError, match="omitted required"):
        train._authenticate_audit(missing)
    dirty = copy.deepcopy(audit)
    forbidden = str((forbidden_roots[0] / "qa.json").resolve())
    dirty["loaded_files"].append(forbidden)
    dirty["loaded_files"].sort()
    dirty["loaded_file_count"] += 1
    dirty["loaded_file_inventory_sha256"] = train._canonical_hash(dirty["loaded_files"])
    with pytest.raises(ValueError, match="were not recomputed"):
        train._authenticate_audit(dirty)


def _patch_publication_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(train, "PUBLICATION_ROOT", str(root))
    monkeypatch.setattr(train, "RESULT_REPORT", str(root / "terminal_result.json"))
    monkeypatch.setattr(train, "FILE_AUDIT_REPORT", str(root / "file_audit.json"))
    monkeypatch.setattr(train, "PUBLICATION_MANIFEST", str(root / "publication_manifest.json"))
    monkeypatch.setattr(train, "OUTPUT_CHECKPOINT", str(root / "checkpoint"))


def _staged_checkpoint(tmp_path: Path, output: str) -> train.StagedCheckpoint:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "adapter.safetensors").write_bytes(b"weights")
    (staged / "runtime_metadata.json").write_text("{}\n", encoding="utf-8")
    return train.StagedCheckpoint(
        staged,
        {
            "path": output,
            "adapter_file_sha256": _sha256(staged / "adapter.safetensors"),
            "runtime_metadata_sha256": _sha256(staged / "runtime_metadata.json"),
            "adapter_state_sha256": "a" * 64,
            "tensor_keys": ["x"],
        },
    )


def test_v6_2_atomic_publication_contains_result_audit_marker_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "published"
    _patch_publication_paths(monkeypatch, root)
    staged = _staged_checkpoint(tmp_path, str(root / "checkpoint"))

    published = train._commit_publication(
        {"passed": True, "status": "passed_checkpoint_published"},
        {"loaded_file_count": 3, "passed": True},
        staged,
    )

    assert published["checkpoint_published"] is True
    assert (root / "terminal_result.json").is_file()
    assert (root / "file_audit.json").is_file()
    assert (root / "publication_manifest.json").is_file()
    assert (root / "checkpoint/adapter.safetensors").is_file()
    manifest = json.loads((root / "publication_manifest.json").read_text())
    train._authenticate_publication_manifest(manifest)


def test_v6_2_publication_auth_rejects_extra_content_even_with_rehashed_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "published"
    _patch_publication_paths(monkeypatch, root)
    staged = _staged_checkpoint(tmp_path, str(root / "checkpoint"))
    train._commit_publication(
        {"passed": True, "status": "passed_checkpoint_published"},
        {"loaded_file_count": 3, "passed": True},
        staged,
    )
    extra = root / "checkpoint/extra.bin"
    extra.write_bytes(b"unbound")
    manifest_path = root / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files_sha256"]["checkpoint/extra.bin"] = _sha256(extra)
    manifest["file_inventory_sha256"] = train._canonical_hash(manifest["files_sha256"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="file inventory"):
        train._authenticate_publication_manifest(train._read_json(manifest_path))


def test_v6_2_publication_auth_rejects_symlinked_checkpoint_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "published"
    _patch_publication_paths(monkeypatch, root)
    staged = _staged_checkpoint(tmp_path, str(root / "checkpoint"))
    train._commit_publication(
        {"passed": True, "status": "passed_checkpoint_published"},
        {"loaded_file_count": 3, "passed": True},
        staged,
    )
    external = tmp_path / "replacement.safetensors"
    external.write_bytes(b"replacement")
    weights = root / "checkpoint/adapter.safetensors"
    weights.unlink()
    weights.symlink_to(external)
    manifest = train._read_json(root / "publication_manifest.json")

    with pytest.raises(ValueError, match="symlink"):
        train._authenticate_publication_manifest(manifest)


def test_v6_2_atomic_publication_rolls_back_everything_on_final_rename_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "published"
    _patch_publication_paths(monkeypatch, root)
    staged = _staged_checkpoint(tmp_path, str(root / "checkpoint"))
    original_rename = os.rename

    def fail_final(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == root:
            raise OSError("injected final rename failure")
        original_rename(source, destination)

    monkeypatch.setattr(train.os, "rename", fail_final)
    with pytest.raises(OSError, match="injected"):
        train._commit_publication(
            {"passed": True}, {"loaded_file_count": 1, "passed": True}, staged
        )

    assert not root.exists()
    assert not staged.directory.exists()
    assert not list(tmp_path.glob(".published.publication.*"))


def test_v6_2_attempt_status_authenticates_unterminalized_hard_crash_without_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = {"sha256": "d" * 64}
    attempt = tmp_path / "attempt.json"
    publication = tmp_path / "publication"
    monkeypatch.setattr(train, "TRAINING_ATTEMPT", str(attempt))
    monkeypatch.setattr(train, "PUBLICATION_ROOT", str(publication))
    monkeypatch.setattr(train, "authenticate_training_release", lambda: release)
    attempt.write_text(
        json.dumps(train._expected_attempt_payload(release), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status = train.authenticate_attempt_state()

    assert status["passed"] is False
    assert status["status"] == "claimed_unterminalized_requires_successor_release"
    assert status["attempt_consumed"] is True
    assert status["training_resume_authorized"] is False
    assert status["publication_exists"] is False


def test_v6_2_runner_is_syntax_valid_and_has_no_implicit_run() -> None:
    runner = Path("scripts/run_gemma4_v54_fixed_prefix_decoder_reader_v6_2.sh")
    source = runner.read_text(encoding="utf-8")
    assert "preflight|release|authenticate-release|attempt-status|train|authenticate" in source
    assert "semantic_3d_chat.training.train_fixed_prefix_decoder_reader_v6_2" in source
