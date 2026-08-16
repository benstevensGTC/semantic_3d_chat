"""Execute exactly one crash-safe V55 development selection.

The selector has a deliberately staged trust boundary:

1. authenticate only the explicit terminal bytes;
2. create or exactly resume a permanent launch claim;
3. only then open development QA, development maps, or model bytes;
4. prepare a strict questions-only manifest in a separate process;
5. run inference in a process that never opens QA supervision;
6. score answers in a separate, model-free process;
7. seal one aggregate selector report, whether the gate passes or fails.

There is one candidate and one precommitted gate.  A crash may resume the same
artifacts, but no output is cleared, overwritten, retuned, or replaced.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.model_snapshot import local_model_snapshot_identity
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.prediction_artifacts import (
    build_prediction_provenance,
    provenance_path_for,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest
from semantic_3d_chat.evaluation.run import load_jsonl
from semantic_3d_chat.evaluation.v55_development_terminal import (
    ARTIFACT as TERMINAL_ARTIFACT,
)
from semantic_3d_chat.evaluation.v55_development_terminal import (
    AUTHORIZATION_ID,
    BOUND_SOURCES,
    CLAIM_PATH,
    EXPECTED_SCENES,
    MODEL_SNAPSHOT_PATH,
    PREDICTION_PROVENANCE_PATH,
    PREDICTIONS_PATH,
    PROTECTED_REPORT,
    PROTECTED_REPORT_SHA256,
    QUESTION_MANIFEST_SHA256,
    QUESTIONS_PATH,
    QUESTIONS_SHA256,
    REFERENCE_PATH,
    REFERENCE_SHA256,
    RUNTIME_CONFIG,
    RUNTIME_CONFIG_SHA256,
    SCORE_PATH,
    SELECTOR_REPORT_PATH,
    V29_ARTIFACT_HASHES,
    V29_METRICS,
    V29_PREDICTIONS,
    V47_CONFIG,
    V47_CONFIG_SHA256,
    V54_CHECKPOINT,
    V54_CHECKPOINT_FILES,
    V54_REPORT,
    V54_REPORT_SHA256,
    V54_TENSOR_HASHES,
)
from semantic_3d_chat.evaluation.v55_development_terminal import (
    DEFAULT_OUTPUT as DEFAULT_TERMINAL,
)

ARTIFACT: Final[str] = "v55_one_shot_development_selector"
SCORE_ARTIFACT: Final[str] = "v55_one_shot_development_score"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_PREDICTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "scene_id",
        "question_id",
        "predicted_answer",
        "grounding_xyz",
        "grounding_confidence",
        "prefix_hash",
        "generated_tokens",
        "elapsed_seconds",
        "provenance_sha256",
    }
)
CommandRunner = Callable[[Sequence[str]], None]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    combined = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(combined))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V55 {label} must be a mapping")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    source = _resolve(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"V55 {label} is unavailable or unsafe: {source}")
    return _mapping(json.loads(source.read_text(encoding="utf-8")), label)


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_create(path: Path, value: Mapping[str, Any]) -> None:
    destination = _resolve(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialized(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def require_terminal(
    expected_sha256: str,
    path: str | Path = DEFAULT_TERMINAL,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate terminal bytes without opening any claimed evaluation input."""

    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V55 explicit terminal SHA256 must be lowercase hexadecimal")
    source = _resolve(path)
    if source != _resolve(DEFAULT_TERMINAL):
        raise ValueError("V55 terminal path is pinned")
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"V55 terminal is unavailable or unsafe: {source}")
    payload = source.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            "V55 terminal differs from the explicit invocation digest: "
            f"expected={expected_sha256} observed={observed}"
        )
    terminal = _mapping(json.loads(payload), "terminal")
    authorization = _mapping(terminal.get("authorization"), "terminal authorization")
    if (
        terminal.get("schema_version") != 1
        or terminal.get("artifact") != TERMINAL_ARTIFACT
        or terminal.get("passed") is not True
        or terminal.get("terminal_materialization_authorized") is not True
        or authorization.get("authorization_id") != AUTHORIZATION_ID
        or authorization.get("only_exact_action")
        != "one_candidate_one_shot_development_selection"
        or authorization.get("explicit_terminal_sha256_required") is not True
    ):
        raise ValueError("V55 terminal authorization is not exact")
    return terminal, observed


def _claim_payload(terminal_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "v55_permanent_development_launch_claim",
        "authorization_id": AUTHORIZATION_ID,
        "terminal_path": str(DEFAULT_TERMINAL),
        "terminal_sha256": terminal_sha256,
        "model_snapshot_path": str(MODEL_SNAPSHOT_PATH),
        "candidate_checkpoint": str(V54_CHECKPOINT),
        "runtime_config": str(RUNTIME_CONFIG),
        "split": "validation",
        "scene_ids": list(EXPECTED_SCENES),
        "reference_path": str(REFERENCE_PATH),
        "reference_sha256": REFERENCE_SHA256,
        "question_manifest_path": str(QUESTIONS_PATH),
        "question_manifest_sha256": QUESTION_MANIFEST_SHA256,
        "questions_sha256": QUESTIONS_SHA256,
        "prediction_path": str(PREDICTIONS_PATH),
        "prediction_provenance_path": str(PREDICTION_PROVENANCE_PATH),
        "score_path": str(SCORE_PATH),
        "selector_report_path": str(SELECTOR_REPORT_PATH),
        "one_candidate_only": True,
        "crash_resume_same_artifacts_only": True,
        "outputs_may_not_be_cleared_or_overwritten": True,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
        "training_or_checkpoint_write_authorized": False,
    }


def create_or_resume_claim(terminal_sha256: str) -> dict[str, Any]:
    """Create the permanent claim before any development/model/map open."""

    expected = _claim_payload(terminal_sha256)
    destination = _resolve(CLAIM_PATH)
    if destination.exists() or destination.is_symlink():
        observed = _read_json(CLAIM_PATH, "launch claim")
        if dict(observed) != expected:
            raise RuntimeError("V55 existing launch claim differs; resume is forbidden")
        return {"created": False, "sha256": _sha256(destination), **expected}

    forbidden_preexisting = (
        QUESTIONS_PATH,
        MODEL_SNAPSHOT_PATH,
        PREDICTIONS_PATH,
        PREDICTION_PROVENANCE_PATH,
        SCORE_PATH,
        SELECTOR_REPORT_PATH,
    )
    existing = [str(path) for path in forbidden_preexisting if _resolve(path).exists()]
    if existing:
        raise RuntimeError(
            "V55 outputs exist without a permanent claim; refusing unclaimed resume: "
            f"{existing}"
        )
    _atomic_create(CLAIM_PATH, expected)
    return {"created": True, "sha256": _sha256(destination), **expected}


@contextmanager
def _execution_lock() -> Any:
    """Hold an OS-released exclusive lock on the immutable launch claim."""

    claim = _resolve(CLAIM_PATH)
    if claim.is_symlink() or not claim.is_file():
        raise FileNotFoundError("V55 launch claim disappeared before execution")
    with claim.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another V55 process already holds the one-shot execution lease"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _locked_hash(path: str | Path, expected: str, label: str) -> None:
    source = _resolve(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"V55 {label} is unavailable or unsafe: {source}")
    observed = _sha256(source)
    if observed != expected:
        raise ValueError(
            f"V55 {label} changed: expected={expected} observed={observed}"
        )


def authenticate_claimed_inputs(terminal: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate every bound byte only after the permanent claim exists."""

    authorization = _mapping(terminal.get("authorization"), "authorization")
    outputs = _mapping(authorization.get("outputs"), "authorized outputs")
    expected_outputs = {
        "claim": str(CLAIM_PATH),
        "model_snapshot": str(MODEL_SNAPSHOT_PATH),
        "questions": str(QUESTIONS_PATH),
        "predictions": str(PREDICTIONS_PATH),
        "prediction_provenance": str(PREDICTION_PROVENANCE_PATH),
        "score": str(SCORE_PATH),
        "selector_report": str(SELECTOR_REPORT_PATH),
    }
    if dict(outputs) != expected_outputs:
        raise ValueError("V55 terminal output contract changed")

    immutable_expected = {
        str(V54_REPORT): V54_REPORT_SHA256,
        str(V47_CONFIG): V47_CONFIG_SHA256,
        str(RUNTIME_CONFIG): RUNTIME_CONFIG_SHA256,
        str(PROTECTED_REPORT): PROTECTED_REPORT_SHA256,
    }
    historical_expected = {
        str(path): digest for path, digest in V29_ARTIFACT_HASHES.items()
    }
    bound_source_paths = {str(path) for path in BOUND_SOURCES}
    groups = {
        "immutable_inputs": immutable_expected,
        "historical_comparator": historical_expected,
    }
    for group_name, expected_group in groups.items():
        group = _mapping(authorization.get(group_name), group_name)
        if dict(group) != expected_group:
            raise ValueError(f"V55 terminal {group_name} contract changed")
        for raw_path, raw_digest in group.items():
            if not isinstance(raw_path, str) or not isinstance(raw_digest, str):
                raise TypeError(f"V55 terminal {group_name} entries must be strings")
            _locked_hash(raw_path, raw_digest, f"{group_name}:{raw_path}")
    sources = _mapping(authorization.get("bound_sources"), "bound_sources")
    if set(sources) != bound_source_paths:
        raise ValueError("V55 terminal bound source inventory changed")
    for raw_path, raw_digest in sources.items():
        if not isinstance(raw_digest, str) or _HEX64.fullmatch(raw_digest) is None:
            raise ValueError(f"V55 bound source digest is invalid: {raw_path}")
        _locked_hash(raw_path, raw_digest, f"bound_sources:{raw_path}")

    candidate = _mapping(authorization.get("candidate"), "candidate")
    if (
        candidate.get("checkpoint") != str(V54_CHECKPOINT)
        or candidate.get("selected_update") != 0
        or candidate.get("selected_optimizer_step") != 0
        or candidate.get("file_sha256") != V54_CHECKPOINT_FILES
        or candidate.get("tensor_state_sha256") != V54_TENSOR_HASHES
    ):
        raise ValueError("V55 candidate contract changed")
    checkpoint = _resolve(V54_CHECKPOINT)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise FileNotFoundError("V55 candidate checkpoint is unavailable or unsafe")
    inventory = sorted(item.name for item in checkpoint.iterdir())
    if inventory != sorted(V54_CHECKPOINT_FILES):
        raise ValueError(f"V55 candidate checkpoint inventory changed: {inventory}")
    for name, digest in V54_CHECKPOINT_FILES.items():
        _locked_hash(checkpoint / name, digest, f"candidate:{name}")

    development = _mapping(authorization.get("development"), "development")
    runtime = _mapping(authorization.get("runtime"), "runtime")
    scope = _mapping(authorization.get("scope"), "scope")
    thresholds = _mapping(authorization.get("thresholds"), "thresholds")
    expected_thresholds = {
        "normalized_exact_accuracy_minimum": 0.375,
        "spatial_relation_accuracy_minimum": 0.55,
        "count_accuracy_minimum": 0.80,
        "presence_f1_minimum": 0.15,
        "canonical_complete_units_minimum": 2,
        "canonical_correct_sides_minimum": 12,
        "canonical_prediction_changed_units_minimum": 2,
        "physical_change_families_minimum": 2,
        "canonical_aggregate_correct_minimum": 91,
    }
    expected_scope = {
        "development_access_authorized_after_launch_claim": True,
        "exactly_one_candidate": True,
        "question_dependent_retrieval_authorized": False,
        "training_access_authorized": False,
        "optimizer_construction_authorized": False,
        "optimizer_state_access_authorized": False,
        "backward_authorized": False,
        "checkpoint_write_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "runtime_promotion_authorized": False,
        "chat_promotion_authorized": False,
        "embodied_promotion_authorized": False,
    }
    if (
        development.get("split") != "validation"
        or development.get("scene_ids") != list(EXPECTED_SCENES)
        or development.get("reference_path") != str(REFERENCE_PATH)
        or development.get("reference_sha256") != REFERENCE_SHA256
        or development.get("question_count") != 216
        or development.get("changed_unit_count") != 12
        or development.get("changed_side_count") != 24
        or development.get("question_manifest_sha256") != QUESTION_MANIFEST_SHA256
        or development.get("questions_sha256") != QUESTIONS_SHA256
        or runtime.get("config") != str(RUNTIME_CONFIG)
        or runtime.get("config_sha256") != immutable_expected[str(RUNTIME_CONFIG)]
        or dict(thresholds) != expected_thresholds
        or dict(scope) != expected_scope
    ):
        raise ValueError("V55 development or scope contract changed")
    _locked_hash(REFERENCE_PATH, REFERENCE_SHA256, "claimed development references")
    return {
        "terminal_inputs_authenticated": True,
        "candidate_file_count": len(V54_CHECKPOINT_FILES),
        "candidate_optimizer_file_absent": True,
        "scene_count": len(EXPECTED_SCENES),
        "question_count": 216,
        "final_test_scenes_touched": False,
        "oracle_loaded": False,
    }


def _authenticate_or_create_model_snapshot(terminal_sha256: str) -> dict[str, Any]:
    """Bind exact local Gemma bytes after the permanent launch claim."""

    runtime = load_runtime_config(_resolve(RUNTIME_CONFIG))
    snapshot = local_model_snapshot_identity(runtime)
    artifact = {
        "schema_version": 1,
        "artifact": "v55_exact_local_model_snapshot",
        "terminal_sha256": terminal_sha256,
        "snapshot": snapshot,
    }
    destination = _resolve(MODEL_SNAPSHOT_PATH)
    if destination.exists() or destination.is_symlink():
        stored = _read_json(MODEL_SNAPSHOT_PATH, "model snapshot seal")
        if dict(stored) != artifact:
            raise RuntimeError(
                "V55 local model snapshot changed after the permanent claim"
            )
    else:
        _atomic_create(MODEL_SNAPSHOT_PATH, artifact)
    return {
        "path": str(MODEL_SNAPSHOT_PATH),
        "sha256": _sha256(destination),
        "tree_sha256": snapshot["tree_sha256"],
        "file_count": snapshot["file_count"],
        "total_size_bytes": snapshot["total_size_bytes"],
        "revision": snapshot["revision"],
    }


def _default_command_runner(command: Sequence[str]) -> None:
    environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not current else f"{source_root}{os.pathsep}{current}"
    subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def _question_evidence() -> dict[str, Any]:
    destination = _resolve(QUESTIONS_PATH)
    manifest = load_question_manifest(destination)
    scenes = tuple(sorted(manifest.by_scene()))
    if (
        manifest.manifest_sha256 != QUESTION_MANIFEST_SHA256
        or manifest.questions_sha256 != QUESTIONS_SHA256
        or manifest.source_qa_sha256 != REFERENCE_SHA256
        or manifest.question_count != 216
        or manifest.scene_count != 6
        or scenes != EXPECTED_SCENES
    ):
        raise ValueError("V55 sanitized question manifest differs from the sealed projection")
    return {
        "path": str(QUESTIONS_PATH),
        "sha256": manifest.manifest_sha256,
        "questions_sha256": manifest.questions_sha256,
        "question_count": manifest.question_count,
        "scene_count": manifest.scene_count,
    }


def _prepare_questions(runner: CommandRunner) -> dict[str, Any]:
    destination = _resolve(QUESTIONS_PATH)
    if not destination.exists():
        runner(
            (
                sys.executable,
                "-m",
                "semantic_3d_chat.evaluation.prepare_questions",
                "--config",
                str(RUNTIME_CONFIG),
                "--split",
                "validation",
                "--qa",
                str(REFERENCE_PATH),
                "--output",
                str(QUESTIONS_PATH),
            )
        )
    return _question_evidence()


def _run_predictions(runner: CommandRunner) -> None:
    runner(
        (
            sys.executable,
            "-m",
            "semantic_3d_chat.evaluation.predict",
            "--config",
            str(RUNTIME_CONFIG),
            "--split",
            "validation",
            "--questions-manifest",
            str(QUESTIONS_PATH),
            "--checkpoint",
            str(V54_CHECKPOINT),
            "--output",
            str(PREDICTIONS_PATH),
        )
    )


def _validate_predictions() -> dict[str, Any]:
    predictions_source = _resolve(PREDICTIONS_PATH)
    provenance_source = provenance_path_for(predictions_source)
    if provenance_source != _resolve(PREDICTION_PROVENANCE_PATH):
        raise ValueError("V55 prediction provenance path derivation changed")
    if predictions_source.is_symlink() or not predictions_source.is_file():
        raise FileNotFoundError("V55 predictions are unavailable after inference")
    records = load_jsonl(predictions_source)
    manifest = load_question_manifest(_resolve(QUESTIONS_PATH))
    expected_keys = {
        (record.scene_id, record.question_id) for record in manifest.questions
    }
    observed_keys: set[tuple[str, str]] = set()
    prefix_hashes: dict[str, set[str]] = {scene_id: set() for scene_id in EXPECTED_SCENES}
    for row in records:
        if set(row) != _PREDICTION_FIELDS:
            raise ValueError("V55 prediction record fields changed")
        key = (str(row.get("scene_id")), str(row.get("question_id")))
        if key in observed_keys:
            raise ValueError(f"V55 duplicate prediction key: {key}")
        observed_keys.add(key)
        prefix = row.get("prefix_hash")
        if not isinstance(prefix, str) or _HEX64.fullmatch(prefix) is None:
            raise ValueError(f"V55 prediction has invalid prefix hash: {key}")
        if key[0] not in prefix_hashes:
            raise ValueError(f"V55 prediction uses an unauthorized scene: {key[0]}")
        prefix_hashes[key[0]].add(prefix)
    if len(records) != 216 or observed_keys != expected_keys:
        raise ValueError("V55 predictions do not exactly cover all 216 questions")
    if any(len(values) != 1 for values in prefix_hashes.values()):
        raise ValueError("V55 scene prefix changed between questions")

    config = load_config(_resolve(RUNTIME_CONFIG))
    expected_provenance = build_prediction_provenance(
        config,
        config_path=_resolve(RUNTIME_CONFIG),
        checkpoint_path=_resolve(V54_CHECKPOINT),
        references_path=_resolve(QUESTIONS_PATH),
        scene_ids=list(EXPECTED_SCENES),
        split="validation",
        run_kind="continuous_scene_static",
        condition="all_questions",
    )
    stored_provenance = _read_json(PREDICTION_PROVENANCE_PATH, "prediction provenance")
    if dict(stored_provenance) != expected_provenance.as_dict():
        raise ValueError("V55 prediction provenance differs from exact claimed inputs")
    if any(
        row.get("provenance_sha256") != expected_provenance.sha256 for row in records
    ):
        raise ValueError("V55 prediction rows have stale provenance")
    return {
        "path": str(PREDICTIONS_PATH),
        "sha256": _sha256(predictions_source),
        "provenance_path": str(PREDICTION_PROVENANCE_PATH),
        "provenance_sha256": _sha256(provenance_source),
        "prediction_provenance_sha256": expected_provenance.sha256,
        "prediction_count": len(records),
        "prefix_sha256_by_scene": {
            scene_id: next(iter(prefix_hashes[scene_id]))
            for scene_id in EXPECTED_SCENES
        },
        "scene_map_manifest_sha256": expected_provenance.scene_map_manifest_sha256,
        "scene_map_manifest": expected_provenance.scene_map_manifest,
    }


def _run_score(runner: CommandRunner) -> None:
    if _resolve(SCORE_PATH).exists():
        return
    runner(
        (
            sys.executable,
            "-m",
            "semantic_3d_chat.evaluation.v55_development_score",
            "--references",
            str(REFERENCE_PATH),
            "--predictions",
            str(PREDICTIONS_PATH),
            "--baseline",
            str(V29_METRICS),
            "--baseline-predictions",
            str(V29_PREDICTIONS),
            "--output",
            str(SCORE_PATH),
        )
    )


def _validate_score(prediction_evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    score = _read_json(SCORE_PATH, "development score")
    scope = _mapping(score.get("scope"), "score scope")
    inputs = _mapping(score.get("inputs"), "score inputs")
    gates = _mapping(score.get("gates"), "score gates")
    if (
        score.get("schema_version") != 1
        or score.get("artifact") != SCORE_ARTIFACT
        or not isinstance(score.get("passed"), bool)
        or scope.get("split") != "validation"
        or scope.get("scene_ids") != list(EXPECTED_SCENES)
        or scope.get("final_test_scenes_touched") is not False
        or scope.get("oracle_loaded") is not False
        or scope.get("model_loaded") is not False
        or scope.get("map_loaded") is not False
        or scope.get("question_or_answer_text_serialized") is not False
        or inputs.get("references_sha256") != REFERENCE_SHA256
        or inputs.get("predictions_sha256") != prediction_evidence.get("sha256")
        or not gates
        or score.get("passed") is not all(value is True for value in gates.values())
    ):
        raise ValueError("V55 development score contract changed")
    serialized = json.dumps(score, sort_keys=True, allow_nan=False)
    if any(token in serialized for token in ('"question":', '"answer":', '"predicted_answer":')):
        raise ValueError("V55 score report serialized question or answer text")
    return score


def build_selector_report(
    terminal_sha256: str,
    claim: Mapping[str, Any],
    model_snapshot_evidence: Mapping[str, Any],
    question_evidence: Mapping[str, Any],
    prediction_evidence: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, Any]:
    gates = _mapping(score.get("gates"), "score gates")
    passed = score.get("passed") is True
    changed_gate = gates.get(
        "canonical_changed_complete_units_at_least_2_of_12"
    ) is True
    aggregate_gate = (
        gates.get("normalized_exact_accuracy_at_least_0_375") is True
        and gates.get("canonical_aggregate_correct_at_least_v29_91_of_216") is True
    )
    standard = _mapping(score.get("standard_metrics"), "standard metrics")
    canonical = _mapping(score.get("canonical_type_specific"), "canonical metrics")
    changed = _mapping(score.get("changed_counterfactual"), "changed metrics")
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "passed": passed,
        "development_selection_passed": passed,
        "chat_promotion_eligible": passed,
        "selected_checkpoint": str(V54_CHECKPOINT) if passed else None,
        "selected_update": 0 if passed else None,
        "selected_optimizer_step": 0 if passed else None,
        "final_test_scenes_touched": False,
        "chat_promotion": {
            "evaluated": True,
            "eligible": passed,
            "checks": {
                "development_checkpoint_selected": True,
                "changed_complete_pair_threshold_met": changed_gate,
                "aggregate_validation_exact_accuracy_retained": aggregate_gate,
            },
        },
        "scope": {
            "split": "validation",
            "scene_ids": list(EXPECTED_SCENES),
            "candidate_count": 1,
            "question_dependent_retrieval": False,
            "scene_prefix_built_before_first_question_for_each_scene": True,
            "scene_prefix_reused_for_every_question_within_scene": True,
            "oracle_loaded": False,
            "final_test_scenes_touched": False,
            "training_executed": False,
            "optimizer_constructed_or_loaded": False,
            "checkpoint_written": False,
            "promotion_executed": False,
            "question_or_answer_text_serialized": False,
        },
        "evidence": {
            "terminal_path": str(DEFAULT_TERMINAL),
            "terminal_sha256": terminal_sha256,
            "claim_path": str(CLAIM_PATH),
            "claim_sha256": claim["sha256"],
            "model_snapshot": dict(model_snapshot_evidence),
            "questions": dict(question_evidence),
            "predictions": dict(prediction_evidence),
            "score_path": str(SCORE_PATH),
            "score_sha256": _sha256(_resolve(SCORE_PATH)),
        },
        "development_metrics": {
            "normalized_exact_accuracy": standard.get("normalized_exact_accuracy"),
            "spatial_relation_accuracy": standard.get("spatial_relation_accuracy"),
            "count": standard.get("count"),
            "presence": standard.get("presence"),
            "canonical_type_specific": dict(canonical),
            "changed_counterfactual": dict(changed),
            "gates": dict(gates),
        },
    }


def run_selector(
    expected_terminal_sha256: str,
    *,
    runner: CommandRunner = _default_command_runner,
) -> dict[str, Any]:
    terminal, observed_terminal_sha = require_terminal(expected_terminal_sha256)
    claim = create_or_resume_claim(observed_terminal_sha)
    with _execution_lock():
        authenticate_claimed_inputs(terminal)
        model_snapshot_evidence = _authenticate_or_create_model_snapshot(
            observed_terminal_sha
        )

        existing_report = _resolve(SELECTOR_REPORT_PATH)
        if existing_report.exists():
            if not all(
                _resolve(path).is_file()
                for path in (
                    QUESTIONS_PATH,
                    PREDICTIONS_PATH,
                    PREDICTION_PROVENANCE_PATH,
                    SCORE_PATH,
                )
            ):
                raise RuntimeError(
                    "V55 completed report is missing immutable supporting evidence"
                )
            question_evidence = _question_evidence()
            prediction_evidence = _validate_predictions()
            score = _validate_score(prediction_evidence)
            expected_report = build_selector_report(
                observed_terminal_sha,
                claim,
                model_snapshot_evidence,
                question_evidence,
                prediction_evidence,
                score,
            )
            report = _read_json(SELECTOR_REPORT_PATH, "selector report")
            if dict(report) != expected_report:
                raise RuntimeError(
                    "V55 completed selector report differs from recomputed evidence"
                )
            return dict(report)

        question_evidence = _prepare_questions(runner)
        _run_predictions(runner)
        post_inference_snapshot = _authenticate_or_create_model_snapshot(
            observed_terminal_sha
        )
        if post_inference_snapshot != model_snapshot_evidence:
            raise RuntimeError("V55 model snapshot changed during inference")
        prediction_evidence = _validate_predictions()
        _run_score(runner)
        score = _validate_score(prediction_evidence)
        report = build_selector_report(
            observed_terminal_sha,
            claim,
            model_snapshot_evidence,
            question_evidence,
            prediction_evidence,
            score,
        )
        _atomic_create(SELECTOR_REPORT_PATH, report)
        return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-terminal-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    terminal, digest = require_terminal(args.expected_terminal_sha256)
    if args.preflight:
        if _resolve(CLAIM_PATH).exists():
            raise RuntimeError("V55 claim already exists; preflight is no longer pristine")
        result = {
            "artifact": ARTIFACT,
            "preflight_passed": True,
            "terminal_sha256": digest,
            "claim_created": False,
            "development_qa_loaded": False,
            "development_maps_loaded": False,
            "model_loaded": False,
            "authorization_id": _mapping(
                terminal.get("authorization"), "authorization"
            ).get("authorization_id"),
        }
        return_code = 0
    else:
        result = run_selector(args.expected_terminal_sha256)
        return_code = 0 if result.get("passed") is True else 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return return_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "authenticate_claimed_inputs",
    "build_selector_report",
    "create_or_resume_claim",
    "main",
    "require_terminal",
    "run_selector",
]
