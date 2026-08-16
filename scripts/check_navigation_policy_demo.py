"""Read-only authentication check for a sealed learned-navigation demo."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.llm_navigation_benchmark import (
    file_sha256,
    load_task_manifest,
    tree_sha256,
    validate_navigation_journal,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_FILES = frozenset({"policy.safetensors", "runtime_metadata.json"})
_FORBIDDEN_COMPONENTS = frozenset({"oracle", "qa", "scorer_only", "training"})


def _rooted(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _regular_file(value: str | Path, *, name: str) -> Path:
    path = _rooted(value)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{name} is unavailable or unsafe: {path}")
    relative = path.relative_to(PROJECT_ROOT)
    if _FORBIDDEN_COMPONENTS.intersection(relative.parts):
        raise ValueError(f"{name} is inside a forbidden runtime data tree: {relative}")
    return path


def _strict_json(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _expected_hash(value: str, *, name: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_hash(path: Path, expected: str, *, name: str) -> str:
    observed = file_sha256(path)
    if observed != _expected_hash(expected, name=name):
        raise ValueError(f"{name} changed: expected {expected}, observed {observed}")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--expected-runtime-config-sha256", required=True)
    parser.add_argument("--expected-tasks-sha256", required=True)
    parser.add_argument("--expected-journal-file-sha256", required=True)
    parser.add_argument("--expected-journal-sha256", required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--expected-score-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint = _rooted(args.checkpoint)
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise FileNotFoundError(f"Navigation checkpoint is unavailable or unsafe: {checkpoint}")
    inventory = {item.name for item in checkpoint.iterdir()}
    if inventory != _CHECKPOINT_FILES:
        raise ValueError(f"Navigation checkpoint inventory changed: {sorted(inventory)}")
    for name in _CHECKPOINT_FILES:
        if not (checkpoint / name).is_file() or (checkpoint / name).is_symlink():
            raise ValueError(f"Navigation checkpoint member is unsafe: {checkpoint / name}")

    runtime_config = _regular_file(args.runtime_config, name="runtime config")
    tasks_path = _regular_file(args.tasks, name="task manifest")
    journal_path = _regular_file(args.journal, name="navigation journal")
    audit_path = _regular_file(args.audit, name="runtime audit")
    score_path = _regular_file(args.score, name="navigation score")

    checkpoint_hash = tree_sha256(checkpoint)
    expected_checkpoint = _expected_hash(
        args.expected_checkpoint_tree_sha256,
        name="expected checkpoint tree SHA-256",
    )
    if checkpoint_hash != expected_checkpoint:
        raise ValueError(
            "Navigation checkpoint tree changed: "
            f"expected {expected_checkpoint}, observed {checkpoint_hash}"
        )
    runtime_config_hash = _require_hash(
        runtime_config,
        args.expected_runtime_config_sha256,
        name="runtime config SHA-256",
    )
    _require_hash(tasks_path, args.expected_tasks_sha256, name="task manifest SHA-256")
    _require_hash(
        journal_path,
        args.expected_journal_file_sha256,
        name="journal file SHA-256",
    )
    audit_hash = _require_hash(
        audit_path,
        args.expected_audit_sha256,
        name="runtime audit SHA-256",
    )
    _require_hash(score_path, args.expected_score_sha256, name="score SHA-256")

    metadata = _strict_json(checkpoint / "runtime_metadata.json")
    weights_hash = file_sha256(checkpoint / "policy.safetensors")
    required_metadata = {
        "task_trained": True,
        "complete_scene_prefix_required": True,
        "question_independent_scene_prefix_required": True,
        "every_scene_token_processed": True,
        "numeric_robot_tokens_required": True,
        "collision_interlock_required": True,
        "oracle_inputs_at_runtime": False,
        "environmental_text_inputs": [],
        "weights_sha256": weights_hash,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Navigation checkpoint runtime contract changed at {key}")

    manifest = load_task_manifest(tasks_path)
    journal = validate_navigation_journal(
        _strict_json(journal_path),
        require_complete=True,
    )
    expected_journal = _expected_hash(
        args.expected_journal_sha256,
        name="expected canonical journal SHA-256",
    )
    if journal["journal_sha256"] != expected_journal:
        raise ValueError("Canonical navigation journal digest changed")
    header = journal.get("header")
    if not isinstance(header, dict):
        raise TypeError("Navigation journal header is invalid")
    if (
        header.get("scene_id") != manifest.scene_id
        or header.get("seed") != manifest.seed
        or header.get("task_count") != len(manifest.tasks)
        or header.get("task_manifest_sha256") != manifest.canonical_sha256
        or header.get("local_inference") is not True
        or header.get("oracle_or_labels_available") is not False
        or header.get("environment_input")
        != "continuous_scene_and_numeric_robot_prefix"
    ):
        raise ValueError("Navigation journal differs from the user-authored task manifest")
    contract = header.get("run_contract")
    if not isinstance(contract, dict):
        raise TypeError("Navigation journal run contract is invalid")
    required_contract = {
        "config_sha256": runtime_config_hash,
        "navigation_policy_checkpoint_tree_sha256": checkpoint_hash,
        "strict_fixed_environment_embedding_input": True,
        "question_conditioned_scene_readout_tokens": False,
        "fallback_policy": "fail_closed",
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"Navigation journal run contract changed at {key}")

    audit = _strict_json(audit_path)
    if (
        audit.get("passed") is not True
        or audit.get("block_forbidden") is not True
        or audit.get("forbidden_accesses") != []
    ):
        raise ValueError("Navigation runtime audit is not a clean blocking audit")
    journal_audit = journal.get("runtime_file_audit")
    if not isinstance(journal_audit, dict) or (
        journal_audit.get("passed") is not True
        or journal_audit.get("blocking_enabled") is not True
        or journal_audit.get("forbidden_accesses") != []
        or journal_audit.get("audit_report_sha256") != audit_hash
    ):
        raise ValueError("Navigation journal is not bound to the clean runtime audit")

    score = _strict_json(score_path)
    separation = score.get("separation")
    metrics = score.get("metrics")
    if not isinstance(separation, dict) or not isinstance(metrics, dict):
        raise TypeError("Navigation score is incomplete")
    if (
        score.get("schema") != "semantic_3d_chat.llm_navigation_score.v1"
        or score.get("scene_id") != manifest.scene_id
        or score.get("claimed_trained_navigation_policy") is not True
        or score.get("navigation_policy_checkpoint_tree_sha256") != checkpoint_hash
        or separation.get("prediction_journal_sha256") != journal["journal_sha256"]
        or separation.get("inference_journal_validated_before_oracle_open") is not True
        or separation.get("inference_received_oracle_or_labels") is not False
        or metrics.get("task_count") != len(manifest.tasks)
    ):
        raise ValueError("Navigation score is not bound to the sealed V2 inference run")

    payload = {
        "phase": "navigation_policy_v2_demo_preflight",
        "passed": True,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_tree_sha256": checkpoint_hash,
        "scene_id": manifest.scene_id,
        "task_count": metrics["task_count"],
        "success_count": metrics.get("success_count"),
        "success_rate": metrics.get("success_rate"),
        "collision_count": metrics.get("collision_count"),
        "policy_rejection_count": metrics.get("policy_rejection_count"),
        "benchmark_passed": score.get("passed"),
        "oracle_or_labels_available_to_inference": False,
        "environment_input": "continuous_scene_and_numeric_robot_prefix",
        "scope": "authenticated one-scene V2 partial-success evidence",
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
