"""Execute exactly one crash-safe V56 fresh-development evaluation.

The permanent claim and exclusive lease are established before reference,
scene-map, model, or candidate bytes are authenticated.  Crashes may resume
the same content-addressed artifacts, but nothing may be cleared, replaced,
retuned, or evaluated as a second candidate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
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
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
    runtime_config_file_sha256,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.predict_question_control import (
    _prediction_condition,
)
from semantic_3d_chat.evaluation.prediction_artifacts import (
    build_prediction_provenance,
    checkpoint_fingerprint,
    provenance_path_for,
    scene_map_manifest_sha256,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest
from semantic_3d_chat.evaluation.run import load_jsonl
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    ARTIFACT as SCORE_ARTIFACT,
)
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    EXPECTED_REFERENCE_COUNT,
    EXPECTED_SCENE_IDS,
    GATE_KEYS,
    threshold_contract,
)
from semantic_3d_chat.evaluation.v56_fresh_development_terminal import (
    ARTIFACT as TERMINAL_ARTIFACT,
)
from semantic_3d_chat.evaluation.v56_fresh_development_terminal import (
    AUTHORIZATION_ID,
    BOUND_SOURCES,
    CLAIM_PATH,
    MODEL_SNAPSHOT_PATH,
    PREDICTION_PROVENANCE_PATH,
    PREDICTIONS_PATH,
    QUESTIONS_PATH,
    REFERENCE_PATH,
    RUNTIME_CONFIG,
    RUNTIME_CONFIG_EFFECTIVE_SHA256,
    RUNTIME_CONFIG_FILE_SHA256,
    SCORE_PATH,
    SELECTOR_REPORT_PATH,
    V54_CHECKPOINT,
    V54_CHECKPOINT_FILES,
    V54_CHECKPOINT_SHA256,
    _authenticate_static_predecessors,
    _control_checkpoint_identity,
    _question_identity,
    _reject_symlink_components,
    _scene_map_identity,
    _training_report_identity,
    software_identity,
)
from semantic_3d_chat.evaluation.v56_fresh_development_terminal import (
    DEFAULT_OUTPUT as DEFAULT_TERMINAL,
)

ARTIFACT: Final[str] = "v56_sealed_fresh_development_selector"
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
        "control_checkpoint_sha256",
        "provenance_sha256",
    }
)
CommandRunner = Callable[[Sequence[str]], None]


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V56 {field} must be a mapping")
    return value


def _read_json(path: str | Path, field: str) -> Mapping[str, Any]:
    source = _resolve(path)
    _reject_symlink_components(source, field)
    if not source.is_file():
        raise FileNotFoundError(f"V56 {field} is unavailable or unsafe: {source}")
    return _mapping(json.loads(source.read_text(encoding="utf-8")), field)


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_create(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = _resolve(path)
    _reject_symlink_components(destination, "atomic output")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V56 output is immutable: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_serialized(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def require_terminal(
    expected_sha256: str,
    path: str | Path = DEFAULT_TERMINAL,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate only terminal bytes before creating the permanent claim."""

    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V56 explicit terminal SHA-256 must be lowercase hexadecimal")
    source = _resolve(path)
    if source != _resolve(DEFAULT_TERMINAL):
        raise ValueError("V56 terminal path is pinned")
    _reject_symlink_components(source, "terminal")
    if not source.is_file():
        raise FileNotFoundError(f"V56 terminal is unavailable or unsafe: {source}")
    payload = source.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError("V56 terminal differs from the explicit invocation digest")
    terminal = _mapping(json.loads(payload), "terminal")
    authorization = _mapping(terminal.get("authorization"), "terminal authorization")
    if (
        terminal.get("schema_version") != 1
        or terminal.get("artifact") != TERMINAL_ARTIFACT
        or terminal.get("passed") is not True
        or terminal.get("terminal_materialization_authorized") is not True
        or authorization.get("authorization_id") != AUTHORIZATION_ID
        or authorization.get("only_exact_action")
        != "one_control_one_shot_fresh_development"
        or authorization.get("explicit_terminal_sha256_required") is not True
    ):
        raise ValueError("V56 terminal authorization is not exact")
    return terminal, observed


def _claim_payload(
    terminal: Mapping[str, Any],
    terminal_sha256: str,
) -> dict[str, Any]:
    authorization = _mapping(terminal.get("authorization"), "authorization")
    control = _mapping(authorization.get("control_checkpoint"), "control checkpoint")
    training = _mapping(authorization.get("training_report"), "training report")
    development = _mapping(authorization.get("development"), "development")
    return {
        "schema_version": 1,
        "artifact": "v56_permanent_fresh_development_launch_claim",
        "authorization_id": AUTHORIZATION_ID,
        "terminal_path": str(DEFAULT_TERMINAL),
        "terminal_sha256": terminal_sha256,
        "base_checkpoint": str(V54_CHECKPOINT),
        "control_checkpoint": control.get("path"),
        "control_checkpoint_sha256": control.get("sha256"),
        "training_report": training.get("path"),
        "training_report_sha256": training.get("sha256"),
        "runtime_config": str(RUNTIME_CONFIG),
        "questions_manifest": str(QUESTIONS_PATH),
        "questions_manifest_sha256": _mapping(
            development.get("questions"), "questions"
        ).get("manifest_sha256"),
        "reference_path": str(REFERENCE_PATH),
        "reference_sha256": development.get("reference_sha256"),
        "scene_ids": list(EXPECTED_SCENE_IDS),
        "prediction_path": str(PREDICTIONS_PATH),
        "prediction_provenance_path": str(PREDICTION_PROVENANCE_PATH),
        "score_path": str(SCORE_PATH),
        "selector_report_path": str(SELECTOR_REPORT_PATH),
        "one_candidate_only": True,
        "crash_resume_same_artifacts_only": True,
        "outputs_may_not_be_cleared_or_overwritten": True,
        "deferred_final_access_authorized": False,
        "simulator_oracle_access_authorized": False,
        "training_or_checkpoint_write_authorized": False,
    }


def create_or_resume_claim(
    terminal: Mapping[str, Any],
    terminal_sha256: str,
) -> dict[str, Any]:
    expected = _claim_payload(terminal, terminal_sha256)
    destination = _resolve(CLAIM_PATH)
    if destination.exists() or destination.is_symlink():
        observed = _read_json(CLAIM_PATH, "launch claim")
        if dict(observed) != expected:
            raise RuntimeError("V56 launch claim differs; resume is forbidden")
        return {"created": False, "sha256": _sha256(destination), **expected}
    forbidden_preexisting = (
        MODEL_SNAPSHOT_PATH,
        PREDICTIONS_PATH,
        PREDICTION_PROVENANCE_PATH,
        SCORE_PATH,
        SELECTOR_REPORT_PATH,
    )
    existing = [
        str(path)
        for path in forbidden_preexisting
        if _resolve(path).exists() or _resolve(path).is_symlink()
    ]
    if existing:
        raise RuntimeError(
            "V56 outputs exist without a permanent claim; refusing resume: "
            f"{existing}"
        )
    _atomic_create(CLAIM_PATH, expected)
    return {"created": True, "sha256": _sha256(destination), **expected}


@contextmanager
def _execution_lock() -> Any:
    claim = _resolve(CLAIM_PATH)
    _reject_symlink_components(claim, "launch claim")
    if not claim.is_file():
        raise FileNotFoundError("V56 launch claim disappeared before execution")
    with claim.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another V56 process holds the one-shot execution lease"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _locked_hash(path: str | Path, expected: object, field: str) -> None:
    if not isinstance(expected, str) or _HEX64.fullmatch(expected) is None:
        raise ValueError(f"V56 {field} expected digest is invalid")
    source = _resolve(path)
    _reject_symlink_components(source, field)
    if not source.is_file():
        raise FileNotFoundError(f"V56 {field} is unavailable or unsafe: {source}")
    if _sha256(source) != expected:
        raise ValueError(f"V56 {field} changed after terminal sealing")


def authenticate_claimed_inputs(terminal: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every terminal-bound input only after claim creation."""

    authorization = _mapping(terminal.get("authorization"), "authorization")
    expected_authorization_fields = {
        "authorization_id",
        "only_exact_action",
        "explicit_terminal_sha256_required",
        "base_checkpoint",
        "control_checkpoint",
        "training_report",
        "runtime",
        "development",
        "pre_inference_model_snapshot",
        "software",
        "predecessor_authentication",
        "bound_sources",
        "outputs",
        "thresholds",
        "scope",
    }
    if set(authorization) != expected_authorization_fields:
        raise ValueError("V56 terminal authorization inventory changed")
    outputs = _mapping(authorization.get("outputs"), "outputs")
    if dict(outputs) != {
        "claim": str(CLAIM_PATH),
        "model_snapshot": str(MODEL_SNAPSHOT_PATH),
        "predictions": str(PREDICTIONS_PATH),
        "prediction_provenance": str(PREDICTION_PROVENANCE_PATH),
        "score": str(SCORE_PATH),
        "selector_report": str(SELECTOR_REPORT_PATH),
    }:
        raise ValueError("V56 terminal output contract changed")
    if dict(_mapping(authorization.get("thresholds"), "thresholds")) != (
        threshold_contract()
    ):
        raise ValueError("V56 preregistered thresholds changed")
    if authorization.get("software") != software_identity():
        raise ValueError("V56 Python dependency environment changed")
    if authorization.get("predecessor_authentication") != (
        _authenticate_static_predecessors()
    ):
        raise ValueError("V56 predecessor authentication changed")

    sources = _mapping(authorization.get("bound_sources"), "bound sources")
    if set(sources) != {str(path) for path in BOUND_SOURCES}:
        raise ValueError("V56 bound source inventory changed")
    for raw_path, digest in sources.items():
        _locked_hash(raw_path, digest, f"bound source {raw_path}")

    runtime_contract = _mapping(authorization.get("runtime"), "runtime")
    if dict(runtime_contract) != {
        "config": str(RUNTIME_CONFIG),
        "file_sha256": RUNTIME_CONFIG_FILE_SHA256,
        "effective_sha256": RUNTIME_CONFIG_EFFECTIVE_SHA256,
    }:
        raise ValueError("V56 runtime config contract changed")
    if runtime_config_file_sha256(RUNTIME_CONFIG) != RUNTIME_CONFIG_FILE_SHA256:
        raise ValueError("V56 runtime config bytes changed")
    runtime = load_runtime_config(RUNTIME_CONFIG)
    if effective_runtime_config_sha256(runtime) != RUNTIME_CONFIG_EFFECTIVE_SHA256:
        raise ValueError("V56 runtime effective config changed")

    base = _mapping(authorization.get("base_checkpoint"), "base checkpoint")
    checkpoint_root = _resolve(V54_CHECKPOINT)
    _reject_symlink_components(checkpoint_root, "base checkpoint")
    if not checkpoint_root.is_dir():
        raise FileNotFoundError("V56 base checkpoint is unavailable")
    inventory = sorted(item.name for item in checkpoint_root.iterdir())
    if inventory != sorted(V54_CHECKPOINT_FILES):
        raise ValueError(f"V56 base checkpoint inventory changed: {inventory}")
    for name in V54_CHECKPOINT_FILES:
        item = checkpoint_root / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"V56 base checkpoint entry is unsafe: {item}")
    base_sha256, base_files = checkpoint_fingerprint(V54_CHECKPOINT)
    if (
        base.get("path") != str(V54_CHECKPOINT)
        or base.get("sha256") != V54_CHECKPOINT_SHA256
        or base.get("file_sha256") != V54_CHECKPOINT_FILES
        or base_sha256 != V54_CHECKPOINT_SHA256
        or {entry["path"]: entry["sha256"] for entry in base_files}
        != V54_CHECKPOINT_FILES
    ):
        raise ValueError("V56 base checkpoint changed")

    control = _mapping(authorization.get("control_checkpoint"), "control checkpoint")
    observed_control = _control_checkpoint_identity(str(control.get("path")))
    if observed_control != dict(control):
        raise ValueError("V56 control checkpoint changed")
    training = _mapping(authorization.get("training_report"), "training report")
    observed_training = _training_report_identity(
        str(training.get("path")), observed_control
    )
    if observed_training != dict(training):
        raise ValueError("V56 training report changed")

    development = _mapping(authorization.get("development"), "development")
    if set(development) != {
        "split",
        "scene_ids",
        "scene_count",
        "atomic_pair_count",
        "question_count",
        "reference_path",
        "reference_sha256",
        "questions",
        "scene_map_manifest",
        "scene_map_manifest_sha256",
    }:
        raise ValueError("V56 fresh-development contract inventory changed")
    questions = _mapping(development.get("questions"), "questions")
    scope = _mapping(authorization.get("scope"), "scope")
    expected_scope = {
        "exactly_one_control_checkpoint": True,
        "fresh_development_access_authorized_after_launch_claim": True,
        "question_dependent_scene_retrieval_authorized": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "backward_authorized": False,
        "checkpoint_write_authorized": False,
        "simulator_oracle_access_authorized": False,
        "deferred_final_access_authorized": False,
        "runtime_promotion_authorized": False,
    }
    if (
        development.get("split") != "validation"
        or development.get("scene_ids") != list(EXPECTED_SCENE_IDS)
        or development.get("scene_count") != len(EXPECTED_SCENE_IDS)
        or development.get("atomic_pair_count") != 3
        or development.get("question_count") != EXPECTED_REFERENCE_COUNT
        or development.get("reference_path") != str(REFERENCE_PATH)
        or not isinstance(development.get("reference_sha256"), str)
        or _HEX64.fullmatch(str(development.get("reference_sha256"))) is None
        or dict(scope) != expected_scope
    ):
        raise ValueError("V56 fresh-development or scope contract changed")
    if _question_identity(QUESTIONS_PATH) != dict(questions):
        raise ValueError("V56 sanitized questions changed")
    if questions.get("reference_sha256") != development.get("reference_sha256"):
        raise ValueError("V56 question manifest and reference digest disagree")
    maps = _scene_map_identity(runtime)
    if maps != development.get("scene_map_manifest"):
        raise ValueError("V56 fresh scene maps changed after terminal sealing")
    if scene_map_manifest_sha256(maps) != development.get(
        "scene_map_manifest_sha256"
    ):
        raise ValueError("V56 fresh scene-map manifest digest changed")
    _locked_hash(
        REFERENCE_PATH,
        development.get("reference_sha256"),
        "fresh-development references",
    )
    return {
        "authenticated": True,
        "scene_count": len(EXPECTED_SCENE_IDS),
        "question_count": EXPECTED_REFERENCE_COUNT,
        "control_checkpoint_sha256": control["sha256"],
        "reference_sha256": development["reference_sha256"],
        "deferred_final_scenes_touched": False,
        "simulator_oracle_loaded": False,
    }


def _authenticate_or_create_model_snapshot(
    terminal_sha256: str,
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = _mapping(terminal.get("authorization"), "authorization")
    expected = _mapping(
        authorization.get("pre_inference_model_snapshot"), "model snapshot"
    )
    runtime = load_runtime_config(RUNTIME_CONFIG)
    observed = local_model_snapshot_identity(runtime)
    if observed != dict(expected):
        raise RuntimeError("V56 local model snapshot changed from the terminal")
    artifact = {
        "schema_version": 1,
        "artifact": "v56_exact_local_model_snapshot",
        "terminal_sha256": terminal_sha256,
        "snapshot": observed,
    }
    destination = _resolve(MODEL_SNAPSHOT_PATH)
    if destination.exists() or destination.is_symlink():
        stored = _read_json(MODEL_SNAPSHOT_PATH, "model snapshot seal")
        if dict(stored) != artifact:
            raise RuntimeError("V56 model snapshot seal changed")
    else:
        _atomic_create(MODEL_SNAPSHOT_PATH, artifact)
    return {
        "path": str(MODEL_SNAPSHOT_PATH),
        "sha256": _sha256(destination),
        "tree_sha256": observed["tree_sha256"],
        "file_count": observed["file_count"],
        "total_size_bytes": observed["total_size_bytes"],
        "revision": observed["revision"],
    }


def _default_command_runner(command: Sequence[str]) -> None:
    environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not current else f"{source_root}{os.pathsep}{current}"
    )
    subprocess.run(list(command), cwd=PROJECT_ROOT, env=environment, check=True)


def _run_predictions(terminal: Mapping[str, Any], runner: CommandRunner) -> None:
    authorization = _mapping(terminal.get("authorization"), "authorization")
    control = _mapping(authorization.get("control_checkpoint"), "control checkpoint")
    runner(
        (
            sys.executable,
            "-m",
            "semantic_3d_chat.evaluation.predict_question_control",
            "--config",
            str(RUNTIME_CONFIG),
            "--questions-manifest",
            str(QUESTIONS_PATH),
            "--base-checkpoint",
            str(V54_CHECKPOINT),
            "--control-checkpoint",
            str(control["path"]),
            "--output",
            str(PREDICTIONS_PATH),
            "--split",
            "validation",
        )
    )


def _validate_predictions(terminal: Mapping[str, Any]) -> dict[str, Any]:
    authorization = _mapping(terminal.get("authorization"), "authorization")
    development = _mapping(authorization.get("development"), "development")
    control = _mapping(authorization.get("control_checkpoint"), "control checkpoint")
    source = _resolve(PREDICTIONS_PATH)
    provenance_source = provenance_path_for(source)
    if provenance_source != _resolve(PREDICTION_PROVENANCE_PATH):
        raise ValueError("V56 prediction provenance path derivation changed")
    _reject_symlink_components(source, "predictions")
    _reject_symlink_components(provenance_source, "prediction provenance")
    if not source.is_file():
        raise FileNotFoundError("V56 predictions are unavailable after inference")
    records = load_jsonl(source)
    manifest = load_question_manifest(QUESTIONS_PATH)
    expected_keys = {
        (record.scene_id, record.question_id) for record in manifest.questions
    }
    observed_keys: set[tuple[str, str]] = set()
    prefix_hashes: dict[str, set[str]] = {
        scene_id: set() for scene_id in EXPECTED_SCENE_IDS
    }
    for row in records:
        if set(row) != _PREDICTION_FIELDS:
            raise ValueError("V56 prediction record fields changed")
        scene_id = row.get("scene_id")
        question_id = row.get("question_id")
        if not isinstance(scene_id, str) or not isinstance(question_id, str):
            raise TypeError("V56 prediction identifiers must be strings")
        key = (scene_id, question_id)
        if key in observed_keys:
            raise ValueError(f"V56 duplicate prediction key: {key}")
        observed_keys.add(key)
        prefix = row.get("prefix_hash")
        if (
            key[0] not in prefix_hashes
            or not isinstance(prefix, str)
            or _HEX64.fullmatch(prefix) is None
        ):
            raise ValueError(f"V56 prediction prefix identity is invalid: {key}")
        prefix_hashes[key[0]].add(prefix)
        if row.get("control_checkpoint_sha256") != control.get("sha256"):
            raise ValueError("V56 prediction used a different control checkpoint")
        predicted_answer = row.get("predicted_answer")
        generated_tokens = row.get("generated_tokens")
        elapsed_seconds = row.get("elapsed_seconds")
        confidence = row.get("grounding_confidence")
        provenance_sha256 = row.get("provenance_sha256")
        if not isinstance(predicted_answer, str) or not predicted_answer.strip():
            raise ValueError("V56 prediction answer must be nonempty text")
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens < 0
        ):
            raise ValueError("V56 prediction generated-token count is invalid")
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or float(elapsed_seconds) < 0.0
        ):
            raise ValueError("V56 prediction elapsed time is invalid")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("V56 prediction grounding confidence is invalid")
        if (
            not isinstance(provenance_sha256, str)
            or _HEX64.fullmatch(provenance_sha256) is None
        ):
            raise ValueError("V56 prediction provenance identity is invalid")
        xyz = row.get("grounding_xyz")
        if (
            not isinstance(xyz, list)
            or len(xyz) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in xyz
            )
        ):
            raise ValueError("V56 prediction has invalid grounding coordinates")
    if len(records) != EXPECTED_REFERENCE_COUNT or observed_keys != expected_keys:
        raise ValueError("V56 predictions do not exactly cover all 216 questions")
    if any(len(values) != 1 for values in prefix_hashes.values()):
        raise ValueError("V56 scene prefix changed between questions")

    runtime = load_runtime_config(RUNTIME_CONFIG)
    expected_provenance = build_prediction_provenance(
        runtime,
        config_path=RUNTIME_CONFIG,
        checkpoint_path=V54_CHECKPOINT,
        references_path=QUESTIONS_PATH,
        scene_ids=list(EXPECTED_SCENE_IDS),
        split="validation",
        run_kind="continuous_scene_question_control_v1",
        condition=_prediction_condition(None, str(control["sha256"])),
    )
    stored = _read_json(PREDICTION_PROVENANCE_PATH, "prediction provenance")
    if dict(stored) != expected_provenance.as_dict():
        raise ValueError("V56 prediction provenance differs from sealed inputs")
    if expected_provenance.scene_map_manifest != development.get(
        "scene_map_manifest"
    ):
        raise ValueError("V56 prediction provenance used different scene maps")
    if any(
        row.get("provenance_sha256") != expected_provenance.sha256 for row in records
    ):
        raise ValueError("V56 prediction row has stale provenance")
    return {
        "path": str(PREDICTIONS_PATH),
        "sha256": _sha256(source),
        "provenance_path": str(PREDICTION_PROVENANCE_PATH),
        "provenance_sha256": _sha256(provenance_source),
        "prediction_provenance_sha256": expected_provenance.sha256,
        "prediction_count": len(records),
        "prefix_sha256_by_scene": {
            scene_id: next(iter(prefix_hashes[scene_id]))
            for scene_id in EXPECTED_SCENE_IDS
        },
        "scene_map_manifest_sha256": (
            expected_provenance.scene_map_manifest_sha256
        ),
        "scene_map_manifest": expected_provenance.scene_map_manifest,
        "control_checkpoint_sha256": control["sha256"],
    }


def _run_score(
    expected_terminal_sha256: str,
    runner: CommandRunner,
) -> None:
    if _resolve(SCORE_PATH).exists():
        return
    runner(
        (
            sys.executable,
            "-m",
            "semantic_3d_chat.evaluation.v56_fresh_development_score",
            "--terminal",
            str(DEFAULT_TERMINAL),
            "--expected-terminal-sha256",
            expected_terminal_sha256,
            "--references",
            str(REFERENCE_PATH),
            "--predictions",
            str(PREDICTIONS_PATH),
            "--output",
            str(SCORE_PATH),
        )
    )


def _validate_score(
    terminal_sha256: str,
    prediction_evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    score = _read_json(SCORE_PATH, "fresh-development score")
    scope = _mapping(score.get("scope"), "score scope")
    inputs = _mapping(score.get("inputs"), "score inputs")
    gates = _mapping(score.get("gates"), "score gates")
    if (
        score.get("schema_version") != 1
        or score.get("artifact") != SCORE_ARTIFACT
        or not isinstance(score.get("passed"), bool)
        or score.get("thresholds") != threshold_contract()
        or scope.get("scene_ids") != list(EXPECTED_SCENE_IDS)
        or scope.get("deferred_final_scenes_touched") is not False
        or scope.get("simulator_oracle_loaded") is not False
        or scope.get("model_loaded") is not False
        or scope.get("map_loaded") is not False
        or scope.get("question_or_answer_text_serialized") is not False
        or inputs.get("terminal_sha256") != terminal_sha256
        or inputs.get("predictions_sha256") != prediction_evidence.get("sha256")
        or set(gates) != GATE_KEYS
        or not all(isinstance(value, bool) for value in gates.values())
        or score.get("passed") is not all(value is True for value in gates.values())
    ):
        raise ValueError("V56 fresh-development score contract changed")
    serialized = json.dumps(score, sort_keys=True, allow_nan=False)
    if any(token in serialized for token in ('"question":', '"answer":', '"predicted_answer":')):
        raise ValueError("V56 score serialized question or answer text")
    return score


def build_selector_report(
    terminal: Mapping[str, Any],
    terminal_sha256: str,
    claim: Mapping[str, Any],
    model_snapshot: Mapping[str, Any],
    prediction_evidence: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = _mapping(terminal.get("authorization"), "authorization")
    control = _mapping(authorization.get("control_checkpoint"), "control checkpoint")
    standard = _mapping(score.get("standard_metrics"), "standard metrics")
    canonical = _mapping(score.get("canonical_type_specific"), "canonical metrics")
    changed = _mapping(score.get("changed_counterfactual"), "changed metrics")
    grounding = _mapping(score.get("grounding"), "grounding metrics")
    gates = _mapping(score.get("gates"), "score gates")
    passed = score.get("passed") is True
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "passed": passed,
        "fresh_development_gate_passed": passed,
        "selected_control_checkpoint": control.get("path") if passed else None,
        "selected_control_checkpoint_sha256": (
            control.get("sha256") if passed else None
        ),
        "final_evaluation_authorized": passed,
        "chat_promotion_eligible": False,
        "deferred_final_scenes_touched": False,
        "scope": {
            "split": "validation",
            "scene_ids": list(EXPECTED_SCENE_IDS),
            "candidate_count": 1,
            "question_dependent_scene_retrieval": False,
            "complete_scene_prefix_built_before_question": True,
            "scene_prefix_reused_for_every_question": True,
            "answer_reference_loaded_only_by_isolated_scorer": True,
            "simulator_oracle_loaded": False,
            "deferred_final_scenes_touched": False,
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
            "model_snapshot": dict(model_snapshot),
            "predictions": dict(prediction_evidence),
            "score_path": str(SCORE_PATH),
            "score_sha256": _sha256(_resolve(SCORE_PATH)),
        },
        "development_metrics": {
            "normalized_exact_accuracy": standard.get("normalized_exact_accuracy"),
            "canonical_type_specific": dict(canonical),
            "spatial_relation_accuracy": standard.get("spatial_relation_accuracy"),
            "count": standard.get("count"),
            "presence": standard.get("presence"),
            "changed_counterfactual": dict(changed),
            "grounding": dict(grounding),
            "gates": dict(gates),
        },
    }


def run_selector(
    expected_terminal_sha256: str,
    *,
    runner: CommandRunner = _default_command_runner,
) -> dict[str, Any]:
    terminal, observed_sha256 = require_terminal(expected_terminal_sha256)
    claim = create_or_resume_claim(terminal, observed_sha256)
    with _execution_lock():
        authenticate_claimed_inputs(terminal)
        model_snapshot = _authenticate_or_create_model_snapshot(
            observed_sha256, terminal
        )
        completed = _resolve(SELECTOR_REPORT_PATH)
        if completed.exists():
            if not all(
                _resolve(path).is_file()
                for path in (PREDICTIONS_PATH, PREDICTION_PROVENANCE_PATH, SCORE_PATH)
            ):
                raise RuntimeError(
                    "V56 completed selector is missing immutable supporting evidence"
                )
            prediction_evidence = _validate_predictions(terminal)
            score = _validate_score(observed_sha256, prediction_evidence)
            expected = build_selector_report(
                terminal,
                observed_sha256,
                claim,
                model_snapshot,
                prediction_evidence,
                score,
            )
            report = _read_json(SELECTOR_REPORT_PATH, "selector report")
            if dict(report) != expected:
                raise RuntimeError("V56 selector report differs from recomputed evidence")
            return dict(report)

        _run_predictions(terminal, runner)
        post_snapshot = _authenticate_or_create_model_snapshot(
            observed_sha256, terminal
        )
        if post_snapshot != model_snapshot:
            raise RuntimeError("V56 model snapshot changed during inference")
        # Reauthenticate the checkpoint, control head, questions, maps, and
        # references after inference so the score cannot certify a mid-run swap.
        authenticate_claimed_inputs(terminal)
        prediction_evidence = _validate_predictions(terminal)
        _run_score(observed_sha256, runner)
        score = _validate_score(observed_sha256, prediction_evidence)
        report = build_selector_report(
            terminal,
            observed_sha256,
            claim,
            model_snapshot,
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
            raise RuntimeError("V56 claim already exists; preflight is no longer pristine")
        result = {
            "artifact": ARTIFACT,
            "preflight_passed": True,
            "terminal_sha256": digest,
            "claim_created": False,
            "fresh_reference_loaded": False,
            "fresh_maps_loaded": False,
            "model_loaded": False,
            "deferred_final_scenes_touched": False,
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
