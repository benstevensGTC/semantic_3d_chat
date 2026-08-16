"""One released full-Gemma MPS smoke for the fixed-prefix V6 reader."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    BASE_CHECKPOINT,
    BASE_RUNTIME_CONFIG,
    CONFIG,
    INITIAL_STATE_SHA256,
    INITIALIZATION_SEED,
    TARGET_MODULES,
    answer_varying_wrong_prefixes,
    build_v6_schedule,
    decoder_reader_lora_settings_v6,
    structural_preflight,
    validate_decoder_reader_surface_v6,
)
from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_release import (
    MPS_SMOKE_ATTEMPT,
    MPS_SMOKE_RELEASE,
    MPS_SMOKE_REPORT,
    claim_mps_smoke_attempt,
    sha256_file,
)
from semantic_3d_chat.language.gemma4_answer_tail import (
    answer_tail_forward,
    reference_answer_tail_from_full_logits,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_LORA_STATE_SHA256 as TOOL_INITIAL_LORA_STATE_SHA256,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_PROJECTOR_STATE_SHA256,
    PROJECTOR_INITIALIZATION_SEED,
    NumericToolContextProjectorV2,
    canonical_answer_token_ids,
    prepare_tool_decoder_inputs,
    tool_decoder_lora_settings_v2,
    tool_decoder_system_prompt,
    validate_decoder_surface_v2,
)
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.lora import (
    initialize_lora_adapter_state,
    install_lora_adapters,
    tensor_state_sha256,
)
from semantic_3d_chat.robot.state_checkpoint import load_robot_state_checkpoint
from semantic_3d_chat.robot.state_encoder import (
    ROBOT_STATE_FEATURE_DIM,
    insert_robot_state_tokens,
)
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1
from semantic_3d_chat.training import train_fixed_prefix_ple_v54_v4 as v4

_SCENE = "scene_000011"
_ROBOT_STATE_CHECKPOINT = "data_gemma4/checkpoints/robot_state_numeric_v1"
_DEFERRED_SCENES = tuple(f"scene_{index:06d}" for index in range(57, 63))
_FINAL_SCENES = tuple(f"scene_{index:06d}" for index in range(25, 31))
_EXPECTED_SOFTWARE_VERSIONS = {
    "python": "3.12.13",
    "numpy": "2.5.1",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "transformers": "5.14.1",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _forbidden_evaluation_roots() -> list[Path]:
    roots = [
        _resolve("data_diverse52/qa/validation.jsonl"),
        _resolve("data_diverse52/qa/test.jsonl"),
        _resolve("data_diverse28/qa/test.jsonl"),
        _resolve("data/qa/test.jsonl"),
        _resolve("reports/gemma4/questions/v56_fresh_development_validation.json"),
        _resolve("reports/gemma4/questions/test.json"),
        _resolve("reports/gemma4/predictions/v56_fresh_development_validation.jsonl"),
    ]
    for scene_id in (*_DEFERRED_SCENES, *_FINAL_SCENES):
        roots.extend(
            _resolve(root) / scene_id
            for root in (
                "data/oracle",
                "data/rendered",
                "data/features",
                "data/maps",
                "data_gemma4/features",
                "data_gemma4/maps",
                "data_gemma4/rendered",
                "data_gemma4/scene_tokens",
            )
        )
    return roots


def _path_inventory_sha256(paths: list[str]) -> str:
    payload = json.dumps(
        sorted(set(paths)), separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _software_versions() -> dict[str, str]:
    versions = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "safetensors", "torch", "transformers")
        },
    }
    if versions != _EXPECTED_SOFTWARE_VERSIONS:
        raise RuntimeError(
            f"V6 MPS smoke software versions changed: {versions}"
        )
    return versions


class _MPSMemorySampler:
    def __init__(self) -> None:
        self.samples: dict[str, int] = {}

    def sample(self, phase: str) -> None:
        if phase in self.samples:
            raise ValueError(f"Duplicate V6 MPS memory phase: {phase}")
        method = getattr(torch.mps, "driver_allocated_memory", None)
        if method is None:
            raise RuntimeError("V6 MPS driver memory sampling is unavailable")
        value = int(method())
        if value < 0:
            raise RuntimeError("V6 MPS driver memory sample is invalid")
        self.samples[phase] = value

    def report(self) -> dict[str, Any]:
        if len(self.samples) < 10:
            raise RuntimeError("V6 MPS smoke did not sample every heavy phase")
        base = v1.memory_metrics()
        return {
            **base,
            "mps_driver_allocated_bytes_sampled_peak": max(self.samples.values()),
            "mps_driver_sample_count": len(self.samples),
            "mps_driver_samples_by_phase": dict(self.samples),
        }


def _atomic_create_report(value: dict[str, Any]) -> None:
    destination = _resolve(MPS_SMOKE_REPORT)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("V6 MPS smoke report already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
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


def _load_base_bundle() -> v1.ReaderBundle:
    experiment = yaml.safe_load(_resolve(CONFIG).read_text(encoding="utf-8"))
    runtime_config = load_runtime_config(_resolve(BASE_RUNTIME_CONFIG))
    runtime = StaticChatRuntime.load(
        runtime_config,
        _SCENE,
        checkpoint=_resolve(BASE_CHECKPOINT),
        local_files_only=True,
    )
    if runtime.language.device.type != "mps":
        raise RuntimeError("V6 full-model smoke requires MPS")
    prefixes = v1.load_prefixes()
    if runtime.scene_prefix_hash != v1.prefix_sha256(prefixes[_SCENE]):
        raise ValueError("V6 cached prefix differs from the loaded V54 runtime")
    runtime.language.model.requires_grad_(False)
    validate_decoder_reader_surface_v6(runtime.language.model)
    validate_decoder_surface_v2(runtime.language.model)
    # The placeholder is replaced before any helper inspects installation.
    return v1.ReaderBundle(runtime, None, prefixes, experiment)  # type: ignore[arg-type]


def _full_vs_tail(
    bundle: v1.ReaderBundle, row: v1.ReaderRecord
) -> tuple[dict[str, float | bool], torch.Tensor]:
    prepared = v1._prepared_batch(bundle, bundle.prefixes[row.scene_id], row)
    with torch.inference_mode():
        full = bundle.language.prefix_backend.prefill(prepared, use_cache=False)
        reference = reference_answer_tail_from_full_logits(
            full.logits.float(), prepared.labels
        )
        tail = answer_tail_forward(bundle.language, prepared)
        reference_logits = reference.logits.detach().float().cpu().contiguous()
        selected = tail.logits.detach().float().cpu().contiguous()
        reference_nll = reference.per_token_nll.detach().float().cpu()
        selected_nll = tail.per_token_nll.detach().float().cpu()
        metrics: dict[str, float | bool] = {
            "selected_logits_exact": torch.equal(reference_logits, selected),
            "selected_logits_max_abs_difference": float(
                (reference_logits - selected).abs().max()
            ),
            "per_token_nll_max_abs_difference": float(
                (reference_nll - selected_nll).abs().max()
            ),
            "mean_nll_absolute_difference": float(
                (reference.mean_nll - tail.mean_nll).abs().cpu()
            ),
            "targets_exact": torch.equal(
                reference.targets.detach().cpu(), tail.targets.detach().cpu()
            ),
            "label_positions_exact": torch.equal(
                reference.label_positions.detach().cpu(),
                tail.label_positions.detach().cpu(),
            ),
            "causal_positions_exact": torch.equal(
                reference.causal_positions.detach().cpu(),
                tail.causal_positions.detach().cpu(),
            ),
        }
    return metrics, selected


def _selected_logits(bundle: v1.ReaderBundle, row: v1.ReaderRecord) -> torch.Tensor:
    prepared = v1._prepared_batch(bundle, bundle.prefixes[row.scene_id], row)
    with torch.inference_mode():
        return (
            answer_tail_forward(bundle.language, prepared)
            .logits.detach()
            .float()
            .cpu()
            .contiguous()
        )


def _tool_runtime_inputs(bundle: v1.ReaderBundle) -> tuple[Any, NumericToolContextProjectorV2]:
    language = bundle.language
    dtype = next(language.model.parameters()).dtype
    prefix = bundle.prefixes[_SCENE].to(language.device, dtype=dtype)
    robot_encoder, _digest, metadata = load_robot_state_checkpoint(
        _ROBOT_STATE_CHECKPOINT,
        expected_output_dim=1536,
        device="cpu",
    )
    if metadata.get("numeric_inputs_only") is not True:
        raise ValueError("V6 joint smoke robot encoder is not numeric-only")
    with torch.inference_mode():
        robot_tokens = robot_encoder(torch.zeros(ROBOT_STATE_FEATURE_DIM)).to(
            language.device, dtype=dtype
        )
    active_prefix = insert_robot_state_tokens(prefix, robot_tokens)
    projector = NumericToolContextProjectorV2()
    if tensor_state_sha256(projector.state_dict()) != INITIAL_PROJECTOR_STATE_SHA256:
        raise ValueError("V6 joint smoke tool projector initialization changed")
    projector = projector.to(language.device).eval().requires_grad_(False)
    system = tool_decoder_system_prompt(max_turn_degrees=45.0, max_move_m=0.5)
    prompt = prompt_token_ids(
        language.tokenizer,
        system,
        "Turn right by ten degrees.",
        language.device,
    )
    answer = canonical_answer_token_ids(
        language.tokenizer,
        '{"arguments":{"angle_degrees":10.0},"tool":"turn"}',
        device=language.device,
    )
    prepared = prepare_tool_decoder_inputs(
        language.prefix_backend,
        active_prefix,
        prompt,
        projector,
        torch.zeros((1, 10), device=language.device),
        torch.full((1, 24), 0.5, device=language.device),
        answer_ids=answer,
        scene_prefix_after_bos=True,
        scene_boundary_mode="gemma4_native_image",
    )
    return prepared, projector


def _joint_state_roundtrip(reader: Any, tool: Any) -> dict[str, Any]:
    reader_hash = reader.state_sha256()
    tool_hash = tool.state_sha256()
    state = {
        **{
            f"reader.{key}": value.detach().cpu().contiguous()
            for key, value in reader.state_module.state_dict().items()
        },
        **{
            f"tool.{key}": value.detach().cpu().contiguous()
            for key, value in tool.state_module.state_dict().items()
        },
    }
    serialized = save_safetensors(state)
    with torch.no_grad():
        for installation in (reader, tool):
            for parameter in installation.parameters():
                parameter.add_(1.0)
    restored = load_safetensors(serialized)
    reader.state_module.load_state_dict(
        {
            key.removeprefix("reader."): value
            for key, value in restored.items()
            if key.startswith("reader.")
        },
        strict=True,
    )
    tool.state_module.load_state_dict(
        {
            key.removeprefix("tool."): value
            for key, value in restored.items()
            if key.startswith("tool.")
        },
        strict=True,
    )
    if reader.state_sha256() != reader_hash or tool.state_sha256() != tool_hash:
        raise RuntimeError("V6 real joint adapter state did not round-trip")
    return {
        "reader_state_sha256": reader_hash,
        "tool_state_sha256": tool_hash,
        "serialized_bytes": len(serialized),
        "strict_state_roundtrip": True,
    }


def _execute_released_smoke(
    *, release_sha: str, attempt_sha: str, started: float
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("V6 released smoke requires available PyTorch MPS")
    if structural_preflight()["passed"] is not True:
        raise RuntimeError("V6 structural preflight failed after release")
    software_versions = _software_versions()

    memory_sampler = _MPSMemorySampler()
    memory_sampler.sample("before_model_load")
    torch.manual_seed(INITIALIZATION_SEED)
    bundle = _load_base_bundle()
    memory_sampler.sample("after_model_load_and_prefix_cache")
    train = v1.load_training_records()
    schedule = build_v6_schedule(train)
    varying = schedule[0].contrastive[0]
    broad_row = schedule[0].broad[0]
    wrong_scene = answer_varying_wrong_prefixes(train)[
        (varying.scene_id, varying.question_id)
    ]
    retention_row = v1.load_retention_corpus()[0]

    equivalence, frozen_logits = _full_vs_tail(bundle, varying)
    memory_sampler.sample("after_full_vs_tail_equivalence")
    with torch.inference_mode():
        retention_teacher = v4.bounded_retention_logits(
            bundle, retention_row["prompt"]
        ).detach().cpu()
    memory_sampler.sample("after_retention_teacher")
    if not (
        equivalence["selected_logits_exact"] is True
        and equivalence["selected_logits_max_abs_difference"] == 0.0
        and equivalence["per_token_nll_max_abs_difference"] == 0.0
        and equivalence["mean_nll_absolute_difference"] == 0.0
        and equivalence["targets_exact"] is True
        and equivalence["label_positions_exact"] is True
        and equivalence["causal_positions_exact"] is True
    ):
        raise RuntimeError("V6 real full-vs-tail answer-logit equivalence failed")
    torch.mps.empty_cache()
    memory_sampler.sample("after_full_logit_cache_clear")

    reader = install_lora_adapters(
        bundle.language.model, decoder_reader_lora_settings_v6()
    )
    if reader is None:
        raise RuntimeError("V6 full-model reader adapter was not installed")
    initialize_lora_adapter_state(reader, seed=INITIALIZATION_SEED)
    if reader.state_sha256() != INITIAL_STATE_SHA256:
        raise ValueError("V6 full-model initial adapter state changed")
    reader.assert_only_lora_trainable(bundle.language.model)
    bundle.installation = reader
    memory_sampler.sample("after_v6_reader_install")
    zero_logits = _selected_logits(bundle, varying)
    memory_sampler.sample("after_v6_zero_output_forward")
    zero_noop = torch.equal(frozen_logits, zero_logits)
    if not zero_noop:
        raise RuntimeError("V6 zero-output adapter changed real answer logits")

    bundle.language.model.zero_grad(set_to_none=True)
    correct, wrong = v4.streamed_answer_nlls(
        bundle,
        (
            (bundle.prefixes[varying.scene_id], varying),
            (bundle.prefixes[wrong_scene], varying),
        ),
    )
    memory_sampler.sample("after_contrastive_forwards")
    margin = wrong - correct
    contrastive_loss = 0.5 * correct + 4.0 * F.relu(0.5 - margin)
    contrastive_loss.backward()
    memory_sampler.sample("after_contrastive_backward")
    broad = v4.streamed_answer_nlls(
        bundle, ((bundle.prefixes[broad_row.scene_id], broad_row),)
    )[0]
    memory_sampler.sample("after_broad_forward")
    (0.5 * broad).backward()
    memory_sampler.sample("after_broad_backward")
    retention = v1.retention_kl_loss(bundle, retention_row, retention_teacher)
    memory_sampler.sample("after_retention_forward")
    (0.5 * retention).backward()
    memory_sampler.sample("after_retention_backward")
    gradients = reader.gradient_norms()
    b_gradients = {
        target: float(adapter.lora_b.grad.detach().float().norm().cpu())
        for target, adapter in zip(TARGET_MODULES, reader.adapters, strict=True)
    }
    a_gradients = {
        target: float(adapter.lora_a.grad.detach().float().norm().cpu())
        for target, adapter in zip(TARGET_MODULES, reader.adapters, strict=True)
    }
    both_nonzero = all(math.isfinite(value) and value > 0.0 for value in b_gradients.values())
    if not both_nonzero or any(value != 0.0 for value in a_gradients.values()):
        raise RuntimeError("V6 real gradient did not reach both exact-zero-B adapters")
    reader.validate_state()
    memory_sampler.sample("after_v6_gradient_validation")

    bundle.language.model.zero_grad(set_to_none=True)
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    validate_decoder_surface_v2(bundle.language.model)
    prepared_tool, projector = _tool_runtime_inputs(bundle)
    memory_sampler.sample("after_numeric_robot_tool_inputs")
    with torch.inference_mode():
        reader_only_tool = answer_tail_forward(
            bundle.language, prepared_tool
        ).logits.detach().cpu().contiguous()
    memory_sampler.sample("after_reader_only_tool_forward")
    tool = install_lora_adapters(
        bundle.language.model, tool_decoder_lora_settings_v2()
    )
    if tool is None:
        raise RuntimeError("V6 joint smoke tool adapter was not installed")
    initialize_lora_adapter_state(tool, seed=PROJECTOR_INITIALIZATION_SEED)
    if tool.state_sha256() != TOOL_INITIAL_LORA_STATE_SHA256:
        raise ValueError("V6 joint smoke tool adapter state changed")
    tool.assert_only_lora_trainable(bundle.language.model)
    memory_sampler.sample("after_zero_output_tool_install")
    with torch.inference_mode():
        joint_tool = answer_tail_forward(
            bundle.language, prepared_tool
        ).logits.detach().cpu().contiguous()
    memory_sampler.sample("after_joint_zero_output_tool_forward")
    joint_noop = torch.equal(reader_only_tool, joint_tool)
    if not joint_noop:
        raise RuntimeError("V6 plus zero-output tool adapter changed real tool logits")
    for parameter in tool.parameters():
        parameter.requires_grad_(False)
    roundtrip = _joint_state_roundtrip(reader, tool)
    memory_sampler.sample("after_joint_state_roundtrip")
    if any(parameter.requires_grad for parameter in bundle.language.model.parameters()):
        raise RuntimeError("V6 joint runtime smoke did not finish fully frozen")
    if any(parameter.requires_grad for parameter in projector.parameters()):
        raise RuntimeError("V6 joint runtime numeric projector did not finish frozen")

    memory = memory_sampler.report()
    if memory["mps_driver_allocated_bytes_sampled_peak"] > 25_000_000_000:
        raise RuntimeError("V6 real MPS smoke exceeded its driver-memory gate")
    return {
        "schema_version": 1,
        "artifact": "gemma4_v54_fixed_prefix_decoder_reader_v6_real_mps_smoke",
        "status": "passed",
        "passed": True,
        "authorization_sha256": release_sha,
        "attempt_sha256": attempt_sha,
        "device": "mps",
        "software_versions": software_versions,
        "full_model_loaded": True,
        "mps_used": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_executed": False,
        "checkpoint_published": False,
        "answer_tail_equivalence_passed": True,
        "full_vs_tail_selected_logits_exact": equivalence["selected_logits_exact"],
        "full_vs_tail_selected_logits_max_abs_difference": equivalence[
            "selected_logits_max_abs_difference"
        ],
        "full_vs_tail_per_token_nll_max_abs_difference": equivalence[
            "per_token_nll_max_abs_difference"
        ],
        "full_vs_tail_mean_nll_absolute_difference": equivalence[
            "mean_nll_absolute_difference"
        ],
        "full_vs_tail_targets_exact": equivalence["targets_exact"],
        "full_vs_tail_label_positions_exact": equivalence["label_positions_exact"],
        "full_vs_tail_causal_positions_exact": equivalence["causal_positions_exact"],
        "v6_zero_output_exact_noop": zero_noop,
        "v6_initial_state_sha256": reader.state_sha256(),
        "v6_gradient_l2": gradients["total_l2"],
        "v6_gradient_by_module": gradients["by_module"],
        "v6_lora_b_gradient_l2_by_target": b_gradients,
        "v6_lora_a_gradient_l2_expected_zero_by_target": a_gradients,
        "both_v6_adapter_gradients_nonzero": both_nonzero,
        "contrastive_correct_nll": float(correct.detach().cpu()),
        "contrastive_wrong_nll": float(wrong.detach().cpu()),
        "contrastive_margin": float(margin.detach().cpu()),
        "broad_nll": float(broad.detach().cpu()),
        "retention_self_kl": float(retention.detach().cpu()),
        "joint_zero_output_structural_runtime_coexistence_passed": True,
        "joint_nonzero_semantic_or_tool_behavior_proven": False,
        "joint_zero_output_exact_noop": joint_noop,
        "tool_numeric_projector_state_sha256": tensor_state_sha256(
            projector.state_dict()
        ),
        "joint_state_roundtrip": roundtrip,
        "scene_prefix_shape": list(bundle.prefixes[_SCENE].shape),
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "memory": memory,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_released_full_model_mps_smoke() -> dict[str, Any]:
    """Claim and consume one full-model MPS smoke attempt, without an optimizer."""

    if any(_resolve(path).exists() for path in (MPS_SMOKE_ATTEMPT, MPS_SMOKE_REPORT)):
        raise FileExistsError("V6 MPS smoke attempt was already consumed")
    audit = FileAccessAudit(_forbidden_evaluation_roots(), block_forbidden=True)
    started = time.perf_counter()
    attempt_claimed = False
    attempt_sha: str | None = None
    release_sha: str | None = None
    core: dict[str, Any] | None = None
    failure: BaseException | None = None
    with audit:
        try:
            _attempt_path, attempt_sha = claim_mps_smoke_attempt()
            attempt_claimed = True
            release_sha = sha256_file(MPS_SMOKE_RELEASE)
            core = _execute_released_smoke(
                release_sha=release_sha,
                attempt_sha=attempt_sha,
                started=started,
            )
            audit.assert_clean()
        except Exception as error:  # noqa: BLE001 - terminalize the consumed attempt
            failure = error

    loaded = audit.unique_paths
    audit_summary = {
        "file_access_audit_active_for_entire_execution": True,
        "loaded_files": loaded,
        "loaded_file_count": len(loaded),
        "loaded_file_inventory_sha256": _path_inventory_sha256(loaded),
        "forbidden_file_accesses": audit.forbidden_accesses(),
        "deferred_or_final_qa_accessed": bool(audit.forbidden_accesses()),
    }
    if failure is not None:
        if attempt_claimed:
            _atomic_create_report(
                {
                    "schema_version": 1,
                    "artifact": (
                        "gemma4_v54_fixed_prefix_decoder_reader_v6_real_mps_smoke"
                    ),
                    "status": "failed_terminal_attempt_consumed",
                    "passed": False,
                    "authorization_sha256": release_sha,
                    "attempt_sha256": attempt_sha,
                    "optimizer_constructed": False,
                    "optimizer_steps": 0,
                    "training_executed": False,
                    "checkpoint_published": False,
                    "failure_type": type(failure).__name__,
                    "failure_message": str(failure),
                    **audit_summary,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        raise failure
    if core is None or attempt_sha is None or release_sha is None:
        raise RuntimeError("V6 MPS smoke ended without a result or a captured failure")
    report = {**core, **audit_summary}
    _atomic_create_report(report)
    return report


__all__ = ["run_released_full_model_mps_smoke"]
