"""Compile one learned 3D scene into a sealed, question-free V81 memory."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import (
    _load_control_head,
    block_question_control_training_artifacts,
)
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v75_fixed_atlas_behavior import _load_probe_bank
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    FIXED_MEMORY_TOKENS,
    HIDDEN_SIZE,
    load_v81_scene_memory,
    save_v81_scene_memory,
)


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--probe-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-report")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Recompute and authenticate an existing sealed memory without writing or replacing it."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = _rooted(args.output)
    default_data = PROJECT_ROOT / "data"
    audit = FileAccessAudit(
        [
            *(default_data / name for name in ("oracle", "qa", "rendered", "features")),
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        ],
        forbidden_component_names=frozenset({"oracle", "qa", "rendered", "features", "scorer"}),
        block_forbidden=True,
    )
    audit_report = _rooted(
        args.audit_report or f"reports/gemma4/metrics/v81_compile_access_{args.scene}.json"
    )
    with audit:
        config = load_runtime_config(args.config, record_file=audit.record)
        for kind in ("oracle", "qa", "rendered", "features"):
            root = artifact_root(config, kind).resolve()
            if root not in audit.forbidden_roots:
                audit.forbidden_roots.append(root)
        block_question_control_training_artifacts(audit, config)
        base_checkpoint = _rooted(args.base_checkpoint)
        control_checkpoint = _rooted(args.control_checkpoint)
        probe_bank = _rooted(args.probe_bank)
        runtime = StaticChatRuntime.load(
            config,
            args.scene,
            checkpoint=base_checkpoint,
            audit=audit,
            local_files_only=True,
        )
        controller, controller_metadata = _load_control_head(
            control_checkpoint,
            hidden_size=runtime.language.hidden_size,
            device=runtime.language.device,
            audit=audit,
        )
        if type(controller) is not DenseFullSceneContinuousControlV75:
            raise TypeError("V81 memory compilation requires the sealed V75 controller")
        probes, probe_metadata = _load_probe_bank(probe_bank, audit)
        with torch.inference_mode():
            compiled = compile_fixed_scene_atlas_v75_v2(
                runtime.scene_prefix,
                controller,
                probes,
            )
        if (
            tuple(compiled.scene_prefix.shape) != (1, FIXED_MEMORY_TOKENS, HIDDEN_SIZE)
            or not compiled.audit.base_environment_tokens_preserved_exactly
            or not compiled.audit.atlas_key_value_tokens_preserved_exactly
            or not compiled.audit.every_probe_processed
            or not compiled.audit.complete_atlas_included
            or compiled.audit.user_question_inputs_used_for_compilation
            or compiled.audit.question_dependent_scene_processing
            or compiled.audit.question_dependent_retrieval
            or compiled.audit.semantic_or_spatial_top_k_selection
        ):
            raise RuntimeError("V81 source compiler did not produce a complete fixed memory")
        base_sha256, _base_files = checkpoint_fingerprint(base_checkpoint)
        runtime_sha256 = effective_runtime_config_sha256(config)
        control_sha256 = _control_checkpoint_sha256(control_checkpoint)
        if controller_metadata.get("base_checkpoint_sha256") != base_sha256:
            raise ValueError("V81 controller was trained against another base checkpoint")
        if controller_metadata.get("base_runtime_config_sha256") != runtime_sha256:
            raise ValueError("V81 controller was trained against another runtime config")
        probe_sha256 = str(probe_metadata["probe_tensor_sha256"])
        if args.verify_existing:
            loaded = load_v81_scene_memory(
                output,
                expected_scene_id=args.scene,
                expected_base_checkpoint_sha256=base_sha256,
                expected_runtime_config_sha256=runtime_sha256,
                expected_model_device=runtime.language.device,
                record_file=audit.record,
            )
            if loaded.metadata["source_control_checkpoint_sha256"] != control_sha256:
                raise ValueError("V81 existing memory uses another control checkpoint")
            if loaded.metadata["source_probe_tensor_sha256"] != probe_sha256:
                raise ValueError("V81 existing memory uses another probe bank")
            if not torch.equal(loaded.memory, compiled.scene_prefix.to(loaded.memory)):
                raise ValueError(
                    "V81 existing scene memory differs from deterministic recompilation"
                )
            metadata = loaded.metadata
            phase = "v81_scene_memory_verified_existing"
        else:
            metadata = save_v81_scene_memory(
                output,
                compiled.scene_prefix,
                scene_id=args.scene,
                source_base_checkpoint_sha256=base_sha256,
                runtime_config_sha256=runtime_sha256,
                source_control_checkpoint_sha256=control_sha256,
                source_probe_tensor_sha256=probe_sha256,
            )
            phase = "v81_scene_memory_compiled"
        runtime.assert_prefix_unchanged()
    audit.assert_clean()
    audit.save(audit_report)
    print(
        json.dumps(
            {
                "phase": phase,
                "scene_id": args.scene,
                "output": str(output),
                "shape": metadata["shape"],
                "dtype": metadata["dtype"],
                "fixed_memory_sha256": metadata["canonical_prefix_sha256"],
                "fixed_memory_tensor_sha256": metadata["tensor_sha256"],
                "base_prefix_sha256": metadata["base_prefix_sha256"],
                "compiled_before_user_question": True,
                "existing_memory_verified_without_write": bool(args.verify_existing),
                "question_inputs_used_for_compilation": False,
                "environmental_text_inputs": [],
                "forbidden_access_count": len(audit.forbidden_accesses()),
                "loaded_file_count": len(audit.unique_paths),
                "audit_report": str(audit_report),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
