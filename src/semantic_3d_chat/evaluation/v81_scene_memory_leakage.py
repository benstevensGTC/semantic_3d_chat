"""Oracle-deletion and file-boundary audit for sealed V81 scene-memory chat."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root
from semantic_3d_chat.evaluation.leakage import (
    DEFAULT_QUESTIONS,
    _atomic_json,
    oracle_temporarily_unavailable,
)


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _within(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def run_v81_scene_memory_leakage(
    *,
    config_path: str | Path,
    scene_id: str,
    base_checkpoint: str | Path,
    scene_memory: str | Path,
    questions: Sequence[str] = DEFAULT_QUESTIONS,
    report_path: str | Path,
    compiler_checkpoint: str | Path = (
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
    ),
    probe_bank: str | Path = (
        "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank"
    ),
) -> dict[str, Any]:
    """Run real local inference while oracle and compiler sources are forbidden."""

    if not questions:
        raise ValueError("V81 leakage audit requires at least one question")
    config_source = _rooted(config_path)
    checkpoint = _rooted(base_checkpoint)
    memory = _rooted(scene_memory)
    compiler = _rooted(compiler_checkpoint)
    probes = _rooted(probe_bank)
    output = _rooted(report_path)
    default_data = PROJECT_ROOT / "data"
    audit = FileAccessAudit(
        [
            *(default_data / name for name in ("oracle", "qa", "rendered", "features")),
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
            compiler,
            probes,
        ],
        forbidden_component_names=frozenset(
            {"oracle", "qa", "rendered", "features", "scorer"}
        ),
        block_forbidden=True,
    )
    runtime: Any | None = None
    startup: dict[str, Any] | None = None
    isolation: Any | None = None
    answers: list[dict[str, Any]] = []
    fixed_hashes: list[str] = []
    base_hashes: list[str] = []
    reader_audits: list[dict[str, Any]] = []
    layout_audits: list[dict[str, Any]] = []
    failure: BaseException | None = None
    oracle_unavailable = False
    oracle_path: Path | None = None
    try:
        with audit:
            from semantic_3d_chat.chat.runtime_config import load_runtime_config
            from semantic_3d_chat.chat.v81_scene_memory_runtime import (
                V81SceneMemoryChatRuntime,
            )

            config = load_runtime_config(config_source, record_file=audit.record)
            oracle = artifact_root(config, "oracle").resolve()
            oracle_path = oracle
            for kind in ("oracle", "qa", "rendered", "features"):
                root = artifact_root(config, kind).resolve()
                if root not in audit.forbidden_roots:
                    audit.forbidden_roots.append(root)
            with oracle_temporarily_unavailable(oracle) as state:
                isolation = state
                if state.hidden is not None:
                    audit.forbidden_roots.append(state.hidden)
                oracle_unavailable = not oracle.exists()
                runtime = V81SceneMemoryChatRuntime.load(
                    config,
                    scene_id,
                    base_checkpoint=checkpoint,
                    scene_memory=memory,
                    audit=audit,
                    local_files_only=True,
                )
                startup = runtime.startup_summary()
                fixed_hashes.append(runtime.current_prefix_hash())
                base_hashes.append(runtime.base_scene_prefix_hash)
                if runtime.questions_answered != 0:
                    raise RuntimeError("V81 runtime accepted a question before memory binding")
                for question in questions:
                    result = runtime.answer(str(question))
                    answers.append(result.to_dict())
                    fixed_hashes.append(runtime.current_prefix_hash())
                    base_hashes.append(runtime.base_scene_prefix_hash)
                    reader_audits.append(dict(runtime.last_reader_audit or {}))
                    layout_audits.append(dict(runtime.last_prepared_layout_audit or {}))
                runtime.assert_prefix_unchanged()
            audit.assert_clean()
    except BaseException as error:  # noqa: BLE001 - persist evidence after safe restore
        failure = error

    if oracle_path is None:
        oracle_path = _rooted("data/oracle")
    oracle_restored = bool(
        isolation is None or not isolation.renamed or oracle_path.is_dir()
    )
    loaded_files = audit.unique_paths
    compiler_reads = [path for path in loaded_files if _within(path, compiler)]
    probe_reads = [path for path in loaded_files if _within(path, probes)]
    training_reads = [
        path
        for path in loaded_files
        if _within(path, PROJECT_ROOT / "data_gemma4" / "training")
    ]
    prefix_invariant = bool(fixed_hashes) and len(set(fixed_hashes)) == 1
    base_prefix_invariant = bool(base_hashes) and len(set(base_hashes)) == 1
    reader_contract_passed = bool(
        len(reader_audits) == len(questions)
        and all(
            item.get("all_96_groups_positive") is True
            and item.get("all_384_values_receive_positive_floor_weight") is True
            and item.get("question_dependent_scene_retrieval") is False
            and item.get("semantic_or_spatial_top_k_selection") is False
            and item.get("environmental_text_inputs") == []
            for item in reader_audits
        )
    )
    layout_contract_passed = bool(
        len(layout_audits) == len(questions)
        and all(
            item.get("base_scene_prefix_tokens") == 258
            and item.get("control_activation_tokens") == 4
            and item.get("control_pad_ple") is True
            and item.get("control_text_modality_zero") is True
            for item in layout_audits
        )
    )
    passed = bool(
        failure is None
        and oracle_unavailable
        and oracle_restored
        and prefix_invariant
        and base_prefix_invariant
        and len(answers) == len(questions)
        and reader_contract_passed
        and layout_contract_passed
        and not compiler_reads
        and not probe_reads
        and not training_reads
        and not audit.forbidden_accesses()
        and startup is not None
        and startup.get("scene_prefix_computed_before_question") is True
        and startup.get("compiler_or_probe_bank_loaded_by_chat") is False
    )
    report: dict[str, Any] = {
        "schema": "semantic_3d_chat.v81_scene_memory_leakage.v1",
        "passed": passed,
        "scene_id": scene_id,
        "runtime_config": str(config_source),
        "base_checkpoint": str(checkpoint),
        "scene_memory": str(memory),
        "oracle_directory": str(oracle_path),
        "oracle_was_renamed": bool(isolation and isolation.renamed),
        "oracle_unavailable_during_inference": oracle_unavailable,
        "oracle_restored": oracle_restored,
        "prefix_computed_before_first_question": bool(
            startup and startup.get("scene_prefix_computed_before_question") is True
        ),
        "fixed_738_memory_invariant": prefix_invariant,
        "base_258_prefix_invariant": base_prefix_invariant,
        "fixed_738_memory_hashes": fixed_hashes,
        "base_258_prefix_hashes": base_hashes,
        "question_count": len(questions),
        "answers": answers,
        "reader_audits": reader_audits,
        "prepared_layout_audits": layout_audits,
        "reader_contract_passed": reader_contract_passed,
        "prepared_layout_contract_passed": layout_contract_passed,
        "compiler_checkpoint_forbidden": str(compiler),
        "compiler_checkpoint_loaded_paths": compiler_reads,
        "probe_bank_forbidden": str(probes),
        "probe_bank_loaded_paths": probe_reads,
        "training_artifact_loaded_paths": training_reads,
        "qa_or_oracle_environmental_text_loaded": False,
        "question_dependent_scene_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "startup": startup,
        "loaded_files": loaded_files,
        "forbidden_accesses": audit.forbidden_accesses(),
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
    }
    _atomic_json(output, report)
    if failure is not None:
        raise RuntimeError(f"V81 leakage audit failed; report written to {output}") from failure
    if not passed:
        raise RuntimeError(f"V81 leakage audit did not pass; report written to {output}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--scene-memory", required=True)
    parser.add_argument("--compiler-checkpoint", default=(
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
    ))
    parser.add_argument("--probe-bank", default=(
        "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank"
    ))
    parser.add_argument("--question", action="append")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_v81_scene_memory_leakage(
        config_path=args.config,
        scene_id=args.scene,
        base_checkpoint=args.base_checkpoint,
        scene_memory=args.scene_memory,
        compiler_checkpoint=args.compiler_checkpoint,
        probe_bank=args.probe_bank,
        questions=args.question or DEFAULT_QUESTIONS,
        report_path=args.output,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "oracle_unavailable_during_inference": report[
                    "oracle_unavailable_during_inference"
                ],
                "oracle_restored": report["oracle_restored"],
                "fixed_738_memory_invariant": report["fixed_738_memory_invariant"],
                "base_258_prefix_invariant": report["base_258_prefix_invariant"],
                "compiler_or_probe_reads": len(
                    report["compiler_checkpoint_loaded_paths"]
                    + report["probe_bank_loaded_paths"]
                ),
                "loaded_file_count": len(report["loaded_files"]),
                "forbidden_access_count": len(report["forbidden_accesses"]),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_v81_scene_memory_leakage"]
