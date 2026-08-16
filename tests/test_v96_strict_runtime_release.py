from __future__ import annotations

import ast
import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.chat import v96_strict_multiscene_cli as cli
from semantic_3d_chat.chat import v96_strict_multiscene_runtime as runtime
from semantic_3d_chat.chat.runtime_config import load_runtime_config, validate_runtime_config
from semantic_3d_chat.chat.v96_explicit_candidate_runtime import (
    TOTAL_PARAMETER_COUNT,
    V96CandidateAuthorization,
)
from semantic_3d_chat.evaluation import v96_strict_runtime_release as release
from semantic_3d_chat.language.lora import tensor_state_sha256


def _digest(character: str = "a") -> str:
    return character * 64


def _authorization() -> V96CandidateAuthorization:
    root = Path.cwd().resolve()
    runtime_config = root / "configs/runtime/gemma4_v85_strict_multiscene.yaml"
    v85 = root / "reports/gemma4/artifacts/v85_strict_runtime_candidate"
    v94 = root / "reports/gemma4/artifacts/v94_strict_multiscene_full40_final"
    v95 = root / "reports/gemma4/artifacts/v95_strict_causal_successor_final"
    return V96CandidateAuthorization.from_payload(
        {
            "artifact": "gemma4_v96_explicit_candidate_authorization_v1",
            "schema_version": 96,
            "status": "authenticated_pass_unpromoted_explicit_use_only",
            "authorization_config_path": str(
                root / "configs/experiments/gemma4_v96_atomic_pair_repair.yaml"
            ),
            "authorization_config_sha256": _digest("1"),
            "runtime_config_path": str(runtime_config),
            "runtime_config_file_sha256": release.sha256_file(runtime_config),
            "runtime_config_effective_sha256": _digest("2"),
            "v85_checkpoint_path": str(v85),
            "v85_adapter_sha256": release.sha256_file(v85 / "adapter.safetensors"),
            "v85_metadata_sha256": release.sha256_file(
                v85 / "runtime_metadata.json"
            ),
            "v94_bridge_path": str(v94),
            "v94_weights_sha256": release.sha256_file(v94 / "bridge.safetensors"),
            "v94_metadata_sha256": release.sha256_file(
                v94 / "runtime_metadata.json"
            ),
            "v94_state_sha256": runtime.V94_STATE_SHA256,
            "v95_bridge_path": str(v95),
            "v95_weights_sha256": release.sha256_file(v95 / "bridge.safetensors"),
            "v95_metadata_sha256": release.sha256_file(
                v95 / "runtime_metadata.json"
            ),
            "v95_state_sha256": (
                "53404c733586ebd25caa440f822a4d4af6cc3dbb71bf4f6b6f94af23f3a2492a"
            ),
            "v96_candidate_path": str(
                root / "reports/gemma4/artifacts/v96_atomic_pair_repair_final"
            ),
            "v96_weights_sha256": _digest("3"),
            "v96_metadata_file_sha256": _digest("4"),
            "v96_metadata_canonical_sha256": _digest("5"),
            "v96_state_sha256": _digest("6"),
            "candidate_fingerprint_sha256": _digest("7"),
            "config_sha256": _digest("8"),
            "preregistration_sha256": _digest("9"),
            "cpu_preflight_sha256": _digest("a"),
            "training_report_sha256": _digest("b"),
            "final_score_path": str(root / "reports/gemma4/metrics/v96-known.json"),
            "final_score_sha256": _digest("c"),
            "evidence_path": str(root / "reports/gemma4/metrics/v96-evidence.json"),
            "evidence_sha256": _digest("d"),
            "implementation_seal_sha256": _digest("e"),
            "implementation_source_inventory_sha256": _digest("f"),
            "v1_implementation_seal_sha256": _digest("1"),
            "v2_implementation_seal_sha256": _digest("e"),
            "candidate_attestation_file_sha256": _digest("2"),
            "candidate_attestation_identity_sha256": _digest("3"),
            "candidate_attestation_immutable": True,
            "gate_results_sha256": _digest("1"),
            "gate_count": 10,
            "all_gate_results_passed": True,
            "candidate_authenticated": True,
            "pass_evidence_authenticated": True,
            "known_development_gate_passed": True,
            "scene_prefix_question_independent": True,
            "row_level_content_serialized": False,
            "environmental_text_inputs": [],
            "deferred_final_unlock_eligible": True,
            "automatic_runtime_promotion": False,
            "runtime_promotion_authorized": False,
            "explicit_candidate_flag_required": True,
        }
    )


def _runtime_contract() -> tuple[dict, dict]:
    config = load_runtime_config("configs/runtime/gemma4_v85_strict_multiscene.yaml")
    config.pop("_runtime_safe_config", None)
    config.pop("_config_path", None)
    metadata = json.loads(
        Path(
            "reports/gemma4/artifacts/v85_strict_runtime_candidate/runtime_metadata.json"
        ).read_text(encoding="utf-8")
    )
    states = metadata["lora_bank_state_sha256"]
    for name, row in config["language"]["lora_banks"].items():
        row["expected_initial_state_sha256"] = states[name]
    for row in metadata["lora"]["banks"]:
        row["expected_initial_state_sha256"] = states[row["name"]]

    extension_states = {
        runtime.V94_BANK: runtime.V94_STATE_SHA256,
        runtime.V95_BANK: _digest("5"),
        runtime.V96_BANK: _digest("6"),
    }
    known_counts = {
        runtime.V94_BANK: (110_592,),
        runtime.V95_BANK: (16_384, 16_384, 110_592),
        runtime.V96_BANK: (45_056,),
    }
    for spec in runtime._BANK_SPECS[len(runtime.EXPECTED_BANKS) - 3 :]:
        state = extension_states[spec.name]
        config["language"]["lora_banks"][spec.name] = {
            "trainable": False,
            "rank": spec.rank,
            "alpha": spec.alpha,
            "dropout": 0.0,
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": state,
            "target_modules": list(spec.targets),
        }
        metadata["lora"]["banks"].append(
            {
                "name": spec.name,
                "trainable": False,
                "rank": spec.rank,
                "alpha": spec.alpha,
                "dropout": 0.0,
                "target_modules": list(spec.targets),
                "initialization_algorithm": "checkpoint_overwrite",
                "initialization_seed": None,
                "expected_initial_state_sha256": state,
                "adapter_parameter_count": spec.parameter_count,
            }
        )
        metadata["lora_bank_state_sha256"][spec.name] = state
        metadata["lora_bank_wrapped_modules"][spec.name] = list(spec.targets)
        metadata["lora_bank_parameter_counts"][spec.name] = dict(
            zip(spec.targets, known_counts[spec.name], strict=True)
        )
    metadata["lora"]["adapter_parameter_count"] = TOTAL_PARAMETER_COUNT
    metadata["lora"]["trainable_adapter_parameter_count"] = 0
    metadata["lora_parameter_count"] = TOTAL_PARAMETER_COUNT
    metadata["lora_trainable_parameter_count"] = 0
    metadata["initialization_provenance"]["v96_strict_runtime_release"] = {
        "schema_version": 96,
        "candidate_fingerprint_sha256": _digest("7"),
        "candidate_attestation_file_sha256": _digest("2"),
        "candidate_attestation_identity_sha256": _digest("3"),
        "v1_implementation_seal_sha256": _digest("1"),
        "v2_implementation_seal_sha256": _digest("e"),
        "deferred_final_evidence_sha256": _digest("8"),
        "deferred_final_score_sha256": _digest("9"),
        "deferred_final_gate_results_sha256": _digest("a"),
        "runtime_implementation_inventory_sha256": runtime.runtime_implementation_inventory_v96()[
            "inventory_sha256"
        ],
        "known_development_gate_passed": True,
        "deferred_final_gate_passed": True,
        "deferred_final_evidence_authenticated": True,
        "supervision_isolation_proven": True,
        "prefix_hash_invariant_in_evaluation": True,
        "v94_state_sha256": runtime.V94_STATE_SHA256,
        "v95_state_sha256": extension_states[runtime.V95_BANK],
        "v96_state_sha256": extension_states[runtime.V96_BANK],
        "promotion_decision": runtime.PENDING_DECISION,
        "runtime_promotion_authorized": False,
        "smoke_report_sha256": None,
        "held_out_generalization_claim": True,
        "environmental_text_inputs": [],
    }
    return config, metadata


def test_v96_release_runtime_accepts_exact_ten_bank_candidate_contract() -> None:
    config, metadata = _runtime_contract()

    result = runtime.validate_v96_release_runtime_contract(
        runtime_config=config,
        checkpoint_metadata=metadata,
    )

    assert result["runtime_package_mode"] == "candidate"
    assert result["runtime_promotion_authorized"] is False
    assert len(config["language"]["lora_banks"]) == 10


@pytest.mark.parametrize(
    "mutation",
    (
        "failed_final",
        "failed_prefix_invariance",
        "trainable_bank",
        "wrong_v96_state",
        "implementation_drift",
        "fake_promotion",
    ),
)
def test_v96_release_runtime_fails_closed_on_gate_or_stack_drift(mutation: str) -> None:
    config, metadata = _runtime_contract()
    provenance = metadata["initialization_provenance"]["v96_strict_runtime_release"]
    if mutation == "failed_final":
        provenance["deferred_final_gate_passed"] = False
    elif mutation == "failed_prefix_invariance":
        provenance["prefix_hash_invariant_in_evaluation"] = False
    elif mutation == "trainable_bank":
        metadata["lora"]["banks"][-1]["trainable"] = True
    elif mutation == "wrong_v96_state":
        provenance["v96_state_sha256"] = _digest("0")
    elif mutation == "implementation_drift":
        provenance["runtime_implementation_inventory_sha256"] = _digest("0")
    else:
        provenance["promotion_decision"] = runtime.PROMOTED_DECISION
        provenance["runtime_promotion_authorized"] = True

    with pytest.raises(ValueError):
        runtime.validate_v96_release_runtime_contract(
            runtime_config=config,
            checkpoint_metadata=metadata,
        )


def test_v96_release_runtime_imports_no_evaluation_or_training_module() -> None:
    tree = ast.parse(inspect.getsource(runtime))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        module.startswith(("semantic_3d_chat.evaluation", "semantic_3d_chat.training"))
        for module in imported
    )


def test_v96_release_config_extends_exact_real_v85_parent_without_writes() -> None:
    authorization = _authorization()

    config = release.build_runtime_config_payload({"authorization": authorization})

    banks = config["language"]["lora_banks"]
    assert tuple(banks) == runtime.EXPECTED_BANKS
    assert len(banks) == 10
    assert all(row["trainable"] is False for row in banks.values())
    assert all(row["expected_initial_state_sha256"] for row in banks.values())
    assert config["paths"]["maps_root"] == "data_gemma4/runtime/maps/v96"
    assert "_runtime_safe_config" not in config
    assert "_config_path" not in config
    assert validate_runtime_config(config)["_runtime_safe_config"] is True


def test_v96_release_authenticates_real_v94_and_v95_bridge_bytes_model_free() -> None:
    specs = release._bridge_specs(_authorization())

    v94, v94_counts = release._load_bridge(specs[0])
    v95, v95_counts = release._load_bridge(specs[1])

    assert set(v94) == {"adapters.0.lora_a", "adapters.0.lora_b"}
    assert sum(v94_counts.values()) == 110_592
    assert len(v95) == 6
    assert sum(v95_counts.values()) == 143_360


def test_v96_release_composes_exact_ten_bank_archive_and_metadata_model_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authorization = _authorization()
    root = tmp_path / "v96_bridge"
    root.mkdir()
    state = {
        "adapters.0.lora_a": torch.zeros((8, 1536), dtype=torch.float32),
        "adapters.0.lora_b": torch.zeros((4096, 8), dtype=torch.float32),
    }
    weights = root / "bridge.safetensors"
    save_file(
        state,
        str(weights),
        metadata={
            "artifact": "gemma4_v96_atomic_pair_repair_fixed_final_v1",
            **release._SAFE_TENSOR_METADATA,
        },
    )
    state_sha = tensor_state_sha256(state)
    weights_sha = release.sha256_file(weights)
    metadata = {
        "artifact": "gemma4_v96_atomic_pair_repair_fixed_final_v1",
        "schema_version": 96,
        "status": "fixed_final_awaiting_known_development_gate",
        "bank_name": runtime.V96_BANK,
        "target_modules": ["model.language_model.layers.9.self_attn.q_proj"],
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "parameter_count": 45_056,
        "state_sha256": state_sha,
        "weights_sha256": weights_sha,
        "environmental_memory_serialized": False,
        "questions_or_answers_serialized": False,
        "oracle_serialized": False,
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
    }
    metadata_path = root / "runtime_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    authorization = replace(
        authorization,
        v96_candidate_path=str(root),
        v96_weights_sha256=weights_sha,
        v96_metadata_file_sha256=release.sha256_file(metadata_path),
        v96_state_sha256=state_sha,
    )
    specs = (*release._bridge_specs(authorization)[:2], release._BridgeSpec(
        root=root,
        artifact="gemma4_v96_atomic_pair_repair_fixed_final_v1",
        status="fixed_final_awaiting_known_development_gate",
        schema_version=96,
        bank_name=runtime.V96_BANK,
        targets=("model.language_model.layers.9.self_attn.q_proj",),
        rank=8,
        alpha=16.0,
        parameter_count=45_056,
        state_sha256=state_sha,
        weights_sha256=weights_sha,
        metadata_sha256=release.sha256_file(metadata_path),
    ))
    monkeypatch.setattr(release, "_bridge_specs", lambda _authorization: specs)
    gate = {
        "authorization": authorization,
        "final": {
            "question_label_isolation_proven": True,
            "prefix_hash_invariant": True,
        },
        "deferred_final_evidence_sha256": _digest("d"),
        "final_score_sha256": _digest("e"),
        "gate_results_sha256": _digest("f"),
    }

    tensors, counts = release._composed_adapter(authorization)
    runtime_metadata = release.build_runtime_metadata(
        gate,
        promotion=runtime.PENDING_DECISION,
        smoke_report_sha256=None,
    )

    assert len(tensors) == 191
    assert tuple(counts) == (runtime.V94_BANK, runtime.V95_BANK, runtime.V96_BANK)
    assert tuple(row["name"] for row in runtime_metadata["lora"]["banks"]) == (
        runtime.EXPECTED_BANKS
    )
    assert runtime_metadata["lora"]["adapter_parameter_count"] == 864_256
    assert runtime_metadata["lora"]["trainable_adapter_parameter_count"] == 0


def test_v96_release_gate_requires_all_deferred_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authorization = _authorization()
    score_path = tmp_path / "score.json"
    evidence_path = tmp_path / "evidence.json"
    score = {
        "artifact": "gemma4_v96_deferred_final_gate_v1",
        "status": "passed_deferred_final_not_runtime_promoted",
        "passed": True,
        "candidate_fingerprint_sha256": authorization.candidate_fingerprint_sha256,
        "candidate_attestation_file_sha256": (
            authorization.candidate_attestation_file_sha256
        ),
        "candidate_attestation_identity_sha256": (
            authorization.candidate_attestation_identity_sha256
        ),
        "v1_implementation_seal_sha256": authorization.v1_implementation_seal_sha256,
        "v2_implementation_seal_sha256": authorization.v2_implementation_seal_sha256,
        "eligible_for_separate_runtime_leakage_evaluation": True,
        "runtime_promotion_authorized": False,
        "gate_results": {"held_out_accuracy": True, "prefix_invariant": True},
    }
    score_path.write_text(json.dumps(score), encoding="utf-8")
    evidence_path.write_text("{}", encoding="utf-8")
    final = {
        "artifact": "gemma4_v96_deferred_final_evidence_v1",
        "schema_version": 96,
        "status": "passed_deferred_final_not_runtime_promoted",
        "deferred_final_gate_passed": True,
        "candidate_fingerprint_sha256": authorization.candidate_fingerprint_sha256,
        "candidate_attestation_file_sha256": (
            authorization.candidate_attestation_file_sha256
        ),
        "candidate_attestation_identity_sha256": (
            authorization.candidate_attestation_identity_sha256
        ),
        "v1_implementation_seal_sha256": authorization.v1_implementation_seal_sha256,
        "v2_implementation_seal_sha256": authorization.v2_implementation_seal_sha256,
        "question_label_isolation_proven": True,
        "prefix_hash_invariant": True,
        "protected_read_count": 0,
        "row_level_content_serialized": False,
        "runtime_packaging_requires_separate_leakage_gate": True,
        "runtime_promotion_authorized": False,
        "automatic_runtime_promotion": False,
        "authenticated": True,
        "gate_results_sha256": _digest("a"),
        "final_score_sha256": release.sha256_file(score_path),
        "evidence_file_sha256": release.sha256_file(evidence_path),
    }
    monkeypatch.setattr(release, "authorize_v96_explicit_candidate", lambda: authorization)
    monkeypatch.setattr(
        release, "authenticate_deferred_final_evidence_v96", lambda: final
    )
    monkeypatch.setattr(
        release,
        "output_paths_v96_final",
        lambda: {"final_score": score_path, "evidence": evidence_path},
    )

    result = release.authenticate_v96_release_gate()
    assert result["deferred_final_gate_passed"] is True

    failed = copy.deepcopy(score)
    failed["gate_results"]["held_out_accuracy"] = False
    score_path.write_text(json.dumps(failed), encoding="utf-8")
    final["final_score_sha256"] = release.sha256_file(score_path)
    with pytest.raises(ValueError, match="held-out release gate"):
        release.authenticate_v96_release_gate()


def test_v96_smoke_protocol_has_no_expectation_channel_and_keeps_default() -> None:
    command = release._smoke_command(
        "scene_000025",
        audit_path=Path("audit.json"),
        chat_path=Path("chat.jsonl"),
    )
    assert "--allow-candidate" in command
    assert all(flag not in command for flag in ("--expected", "--answer", "--reference"))
    assert release.SCENE_IDS == tuple(f"scene_{index:06d}" for index in range(25, 31))

    args = cli._parser().parse_args([])
    assert args.scene == "scene_000025"
    assert args.allow_candidate is False
    assert cli.DEFAULT_CONFIG.endswith("gemma4_v96_strict_multiscene.yaml")
    forbidden = cli._forbidden_roots()
    source_maps = (Path.cwd() / "data_gemma4/maps").resolve()
    packaged_maps = (Path.cwd() / "data_gemma4/runtime/maps/v96").resolve()
    assert source_maps in forbidden
    assert all(root != packaged_maps and root not in packaged_maps.parents for root in forbidden)


def test_v96_release_prepare_authenticates_before_any_materialization() -> None:
    source = inspect.getsource(release.prepare_candidate)
    assert source.index("authenticate_v96_release_gate") < source.index(
        "materialize_runtime_config"
    )
    assert source.index("authenticate_v96_release_gate") < source.index(
        "_atomic_checkpoint"
    )
    assert source.index("authenticate_v96_release_gate") < source.index(
        "_package_runtime_maps"
    )
    assert source.index("authenticate_v96_release_gate") < source.index(
        "_package_memories"
    )


def test_v96_runtime_implementation_inventory_is_physical_and_stable() -> None:
    first = runtime.runtime_implementation_inventory_v96()
    second = runtime.runtime_implementation_inventory_v96()

    assert first == second
    assert first["inventory_sha256"] == release._require_sha256(
        first["inventory_sha256"], "runtime implementation"
    )
    assert len(first["files"]) == len(runtime.RUNTIME_IMPLEMENTATION_FILES)
    assert all(row["size_bytes"] > 0 for row in first["files"])


def test_v96_oracle_recovery_journal_restores_interrupted_move(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path.resolve()
    source = project / "data_test/oracle"
    hidden = project / "data_test/.oracle-unavailable-v96-123-0"
    journal = project / "smoke/oracle_move_journal.json"
    source.mkdir(parents=True)
    monkeypatch.setattr(release, "PROJECT_ROOT", project)
    monkeypatch.setattr(release, "ORACLE_JOURNAL", journal)
    release._write_json_atomic(
        journal,
        release._oracle_journal_payload(
            ((source, hidden),), "prepared_for_physical_oracle_isolation"
        ),
    )
    source.rename(hidden)

    result = release.recover_oracle_roots()

    assert result["passed"] is True
    assert result["recovered"] is True
    assert source.is_dir()
    assert not hidden.exists()
    assert release._read_json(journal)["status"] == "recovered_after_interrupted_smoke"


def test_v96_runtime_maps_are_copied_exactly_into_sanitized_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path.resolve()
    sources: dict[str, dict] = {}
    for index, scene_id in enumerate(release.SCENE_IDS):
        source = project / f"source/{scene_id}/voxel_map.npz"
        source.parent.mkdir(parents=True)
        source.write_bytes(f"numeric-map-{index}".encode())
        sources[scene_id] = {
            "source_path": source.relative_to(project).as_posix(),
            "source_sha256": release.sha256_file(source),
            "size_bytes": source.stat().st_size,
            "maps_receipt_sha256": _digest("c"),
        }
    destination = project / "data_gemma4/runtime/maps/v96"
    monkeypatch.setattr(release, "PROJECT_ROOT", project)
    monkeypatch.setattr(release, "RUNTIME_MAP_ROOT", destination)
    monkeypatch.setattr(release, "_authenticated_source_maps", lambda: sources)
    monkeypatch.setattr(release, "validate_runtime_map_sidecars", lambda _path: None)

    packaged = release._package_runtime_maps(destination)
    verified = release._verify_runtime_maps()

    assert packaged == verified
    assert tuple(sorted(packaged)) == release.SCENE_IDS
    assert all(row["numeric_bytes_reused_exactly"] is True for row in packaged.values())


def test_v96_partial_release_cleanup_is_exact_and_refuses_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint"
    memories = tmp_path / "memories"
    report = tmp_path / "report.json"
    checkpoint.mkdir()
    memories.mkdir()
    monkeypatch.setattr(release, "RELEASE_CHECKPOINT", checkpoint)
    monkeypatch.setattr(release, "RELEASE_MEMORY_ROOT", memories)
    monkeypatch.setattr(release, "RELEASE_REPORT", report)

    release.cleanup_partial_release()
    assert not checkpoint.exists()
    assert not memories.exists()

    checkpoint.mkdir()
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="release report"):
        release.cleanup_partial_release()
    assert checkpoint.is_dir()


def test_v96_runtime_package_inventory_rejects_extras_and_link_roots(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
    (checkpoint / "runtime_metadata.json").write_text("{}", encoding="utf-8")
    release._require_exact_checkpoint_package(checkpoint)

    extra = checkpoint / "unaccounted.txt"
    extra.write_text("must be rejected", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        release._require_exact_checkpoint_package(checkpoint)
    extra.unlink()

    checkpoint_link = tmp_path / "checkpoint-link"
    checkpoint_link.symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(FileNotFoundError, match="physical directory"):
        release._require_exact_checkpoint_package(checkpoint_link)

    memories = tmp_path / "memories"
    memories.mkdir()
    for scene_id in release.SCENE_IDS:
        (memories / scene_id).mkdir()
    release._require_exact_scene_bundle(memories, label="test memories")
    (memories / "unexpected").mkdir()
    with pytest.raises(ValueError, match="exactly six scenes"):
        release._require_exact_scene_bundle(memories, label="test memories")


def test_v96_release_module_does_not_change_default_demo_target() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    default_body = makefile.split("\ndemo:", 1)[1].split("\n\n", 1)[0]
    assert "run_v89_strict_scene1_demo.sh" in default_body
    assert "v96" not in default_body.casefold()
    snapshot = release._default_runtime_snapshot()
    assert snapshot["default_demo_uses_v89"] is True
    assert snapshot["default_demo_uses_v96"] is False
