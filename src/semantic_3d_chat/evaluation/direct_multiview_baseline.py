"""Evaluation-only direct multi-view local VLM baseline.

Only complete RGB frames from the sanitized manifest and the user's question
are supplied.  This baseline consumes neither oracle metadata nor depth and is
not a substitute for the project's primary persistent 3D scene-memory path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.device import safe_dtype, select_device
from semantic_3d_chat.evaluation.baseline_io import (
    atomic_write_jsonl,
    indexed_records,
    read_jsonl,
    sha256_file,
    text_fingerprint,
)
from semantic_3d_chat.rendering_io import load_manifest

ImageAnswerer = Callable[[list[Image.Image], str], str]

_GENERIC_BACKEND = "generic_image_text"
_GEMMA4_BACKEND = "gemma4"
_GEMMA4_SCENE_CACHE_CONTRACT = "gemma4_decoder_kv_scene_prefix_v1"
_DISABLED_SCENE_CACHE = "disabled"

# A sanitized render manifest uses opaque frame IDs.  These tokens are rejected
# in relative RGB paths so an evaluation-only VLM cannot accidentally benefit
# from a file named after its contents.  The pixels remain the sole source of
# environmental semantics supplied to this baseline.
_SEMANTIC_FILENAME_TOKENS = frozenset(
    {
        "book",
        "bowl",
        "cabinet",
        "chair",
        "cube",
        "door",
        "floor",
        "frame",
        "lamp",
        "picture",
        "plant",
        "table",
        "wall",
        "window",
    }
)


def direct_multiview_scene_cache_contract(baseline: dict[str, Any]) -> str:
    """Resolve the versioned, question-independent scene-cache contract."""

    configured = str(baseline.get("scene_cache", _DISABLED_SCENE_CACHE)).casefold()
    aliases = {
        "none": _DISABLED_SCENE_CACHE,
        "off": _DISABLED_SCENE_CACHE,
        _DISABLED_SCENE_CACHE: _DISABLED_SCENE_CACHE,
        _GEMMA4_SCENE_CACHE_CONTRACT: _GEMMA4_SCENE_CACHE_CONTRACT,
    }
    try:
        contract = aliases[configured]
    except KeyError as exc:
        raise ValueError(
            "direct_multiview scene_cache must be disabled or "
            f"{_GEMMA4_SCENE_CACHE_CONTRACT}"
        ) from exc
    if (
        contract != _DISABLED_SCENE_CACHE
        and direct_multiview_backend(baseline) != _GEMMA4_BACKEND
    ):
        raise ValueError("The decoder-KV scene cache is implemented only for the Gemma 4 backend")
    return contract


def _validate_sanitized_rgb_path(relative_path: str) -> None:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"RGB path must be a safe relative path: {relative_path!r}")
    tokens = {
        token
        for part in path.parts
        for token in re.findall(r"[a-z]+", part.casefold())
    }
    leaked = sorted(tokens & _SEMANTIC_FILENAME_TOKENS)
    if leaked:
        raise ValueError(
            "Sanitized RGB filenames must be semantically opaque; found " + ", ".join(leaked)
        )


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def direct_multiview_backend(baseline: dict[str, Any]) -> str:
    """Resolve the explicit processor/model contract for the image control."""

    configured = str(baseline.get("backend", "auto")).casefold()
    if configured == "auto":
        model_id = str(baseline["model_id"]).casefold()
        configured = _GEMMA4_BACKEND if "gemma-4" in model_id else _GENERIC_BACKEND
    aliases = {
        "generic": _GENERIC_BACKEND,
        "image_text": _GENERIC_BACKEND,
        _GENERIC_BACKEND: _GENERIC_BACKEND,
        _GEMMA4_BACKEND: _GEMMA4_BACKEND,
    }
    try:
        return aliases[configured]
    except KeyError as exc:
        raise ValueError(
            "direct_multiview backend must be auto, generic_image_text, or gemma4"
        ) from exc


def complete_view_paths(
    rendered_scene: str | Path,
    *,
    max_views: int | None = None,
) -> list[Path]:
    """Return deterministic complete-frame paths from a sanitized manifest."""
    scene_root = Path(rendered_scene).resolve()
    manifest = load_manifest(scene_root / "manifest.json")
    frames = sorted(manifest["frames"], key=lambda frame: int(frame["frame_number"]))
    if max_views is not None:
        if max_views <= 0:
            raise ValueError("max_views must be positive")
        frames = frames[:max_views]
    relative_paths = [str(frame["rgb_path"]) for frame in frames]
    for relative_path in relative_paths:
        _validate_sanitized_rgb_path(relative_path)
    paths = [(scene_root / relative_path).resolve() for relative_path in relative_paths]
    if len(set(paths)) != len(paths):
        raise ValueError(f"Sanitized manifest repeats an RGB frame below {scene_root}")
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"Complete RGB frames are missing below {scene_root}")
    return paths


def multiview_conversation(
    question: str, image_count: int, system_prompt: str
) -> list[dict[str, Any]]:
    if image_count <= 0:
        raise ValueError("At least one complete image is required")
    content: list[dict[str, str]] = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": f"{system_prompt}\nQuestion: {question}"})
    return [{"role": "user", "content": content}]


def gemma4_multiview_conversation(
    question: str,
    images: Sequence[Image.Image],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Build Gemma 4's official interleaved chat payload from complete frames."""

    if not images:
        raise ValueError("At least one complete image is required")
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": f"{system_prompt}\nQuestion: {question}"})
    return [{"role": "user", "content": content}]


def observation_fingerprint(scene_root: Path, image_paths: list[Path]) -> str:
    manifest_hash = sha256_file(scene_root / "manifest.json")
    image_hashes = [sha256_file(path) for path in image_paths]
    return text_fingerprint(manifest_hash, *image_hashes)


@dataclass
class Gemma4ScenePrefixCache:
    """Question-independent causal state for one complete multi-view scan.

    ``past_key_values`` is a continuous decoder state produced from all images
    and the fixed instruction through ``Question:``.  No user question is an
    input to cache construction.  A private clone is extended for each answer,
    leaving this source state identical across questions.
    """

    past_key_values: Any
    prefix_input_ids_cpu: torch.Tensor
    prefix_attention_mask: torch.Tensor
    prefix_token_sha256: str
    rendered_prefix_text: str
    complete_view_count: int
    contract: str = _GEMMA4_SCENE_CACHE_CONTRACT


def _gemma4_empty_question_prefill(
    processor: Any,
    images: Sequence[Image.Image],
    system_prompt: str,
    *,
    enable_thinking: bool,
) -> tuple[dict[str, Any], str, str]:
    """Process all complete images once and trim only the response trailer."""

    conversation = gemma4_multiview_conversation("", images, system_prompt)
    rendered = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("Gemma 4 chat template must render to text before tokenization")
    anchor = f"{system_prompt}\nQuestion:"
    anchor_offset = rendered.rfind(anchor)
    if anchor_offset < 0:
        raise ValueError("Gemma 4 chat template did not preserve the fixed question anchor")
    rendered_prefix = rendered[: anchor_offset + len(anchor)]
    response_trailer = rendered[len(rendered_prefix) :]
    if not response_trailer:
        raise ValueError("Gemma 4 chat template omitted the assistant response trailer")

    encoded = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = dict(encoded)
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Gemma 4 scene prefill requires one tokenized conversation")
    trailer_ids = processor.tokenizer.encode(response_trailer, add_special_tokens=False)
    if not trailer_ids or input_ids.shape[1] <= len(trailer_ids):
        raise ValueError("Gemma 4 response trailer cannot consume the scene prefix")
    expected = torch.tensor(trailer_ids, dtype=input_ids.dtype)
    if not torch.equal(input_ids[0, -len(trailer_ids) :].cpu(), expected):
        raise ValueError("Gemma 4 tokenized response trailer changed unexpectedly")
    prefix_length = int(input_ids.shape[1] - len(trailer_ids))
    for key in (
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "position_ids",
        "token_type_ids",
    ):
        value = inputs.get(key)
        if isinstance(value, torch.Tensor):
            if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] != input_ids.shape[1]:
                raise ValueError(f"Unexpected sequence-aligned Gemma 4 tensor shape for {key}")
            inputs[key] = value[:, :prefix_length].contiguous()
    return inputs, response_trailer, rendered_prefix


@dataclass
class LocalMultiViewAnswerer:
    model: Any
    processor: Any
    device: torch.device
    dtype: torch.dtype
    system_prompt: str
    max_answer_tokens: int
    resize_longest_edge: int | None
    backend: str = _GENERIC_BACKEND
    enable_thinking: bool = False

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        *,
        local_files_only: bool = True,
        prefer_mps: bool = True,
    ) -> LocalMultiViewAnswerer:
        baseline = config["evaluation"]["baselines"]["direct_multiview"]
        device = select_device(prefer_mps=prefer_mps)
        dtype = safe_dtype(device, str(baseline.get("dtype", "float16")))
        model_id = str(baseline["model_id"])
        revision = str(baseline["revision"])
        backend = direct_multiview_backend(baseline)
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        if backend == _GEMMA4_BACKEND:
            try:
                from transformers import Gemma4ForConditionalGeneration
            except ImportError as exc:  # pragma: no cover - legacy venv only
                raise RuntimeError(
                    "The Gemma 4 direct-image control requires the isolated "
                    "Transformers 5 environment; run it with .venv-gemma4."
                ) from exc
            load_kwargs: dict[str, Any] = {
                "revision": revision,
                "dtype": dtype,
                "local_files_only": local_files_only,
                "low_cpu_mem_usage": True,
            }
            if device.type == "mps":
                # Stream the single cached checkpoint directly to unified memory;
                # loading a second CPU copy first can exceed a 24 GB Mac's budget.
                load_kwargs["device_map"] = {"": device}
            model = Gemma4ForConditionalGeneration.from_pretrained(
                model_id,
                **load_kwargs,
            )
            if device.type != "mps":
                model = model.to(device)
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
                local_files_only=local_files_only,
            ).to(device)
        model.requires_grad_(False)
        model.eval()
        resize = baseline.get("resize_longest_edge")
        resize_longest_edge = int(resize) if resize is not None else None
        if backend == _GENERIC_BACKEND and resize_longest_edge is None:
            raise ValueError("Generic direct-image baselines require resize_longest_edge")
        return cls(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            system_prompt=str(baseline["system_prompt"]),
            max_answer_tokens=int(baseline["max_answer_tokens"]),
            resize_longest_edge=resize_longest_edge,
            backend=backend,
            enable_thinking=bool(baseline.get("enable_thinking", False)),
        )

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
            elif torch.is_floating_point(value):
                moved[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

    def _decode_gemma4_continuation(
        self,
        continuation: torch.Tensor,
        *,
        prefix: torch.Tensor,
    ) -> str:
        decoded = self.processor.decode(continuation, skip_special_tokens=False)
        try:
            parsed = self.processor.parse_response(decoded, prefix=prefix)
        except (AttributeError, TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
            return str(parsed["content"]).strip()
        return self.processor.decode(continuation, skip_special_tokens=True).strip()

    def prepare_scene_cache(self, images: list[Image.Image]) -> Gemma4ScenePrefixCache:
        """Build one immutable causal prefix from every complete scene image.

        This method deliberately has no question argument.  It runs the Gemma 4
        processor, vision tower, and long multimodal decoder prefill exactly
        once.  Per-question generation clones rather than mutates this state.
        """

        if self.backend != _GEMMA4_BACKEND:
            raise ValueError("Scene-prefix caching is implemented only for Gemma 4")
        if not images:
            raise ValueError("At least one complete image is required")
        inputs, _response_trailer, rendered_prefix = _gemma4_empty_question_prefill(
            self.processor,
            images,
            self.system_prompt,
            enable_thinking=self.enable_thinking,
        )
        prefix_input_ids = inputs["input_ids"].detach().cpu().contiguous()
        moved = self._move_inputs(inputs)
        with torch.inference_mode():
            outputs = self.model(
                **moved,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None or not hasattr(past_key_values, "get_seq_length"):
            raise RuntimeError("Gemma 4 did not return a reusable causal decoder cache")
        prefix_token_count = int(prefix_input_ids.shape[1])
        if int(past_key_values.get_seq_length()) != prefix_token_count:
            raise RuntimeError(
                "Gemma 4 decoder cache length does not match the question-independent prefix"
            )
        prefix_attention_mask = moved.get("attention_mask")
        if not isinstance(prefix_attention_mask, torch.Tensor):
            prefix_attention_mask = torch.ones(
                (1, prefix_token_count),
                dtype=torch.long,
                device=self.device,
            )
        return Gemma4ScenePrefixCache(
            past_key_values=past_key_values,
            prefix_input_ids_cpu=prefix_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            prefix_token_sha256=_tensor_sha256(prefix_input_ids),
            rendered_prefix_text=rendered_prefix,
            complete_view_count=len(images),
        )

    def answer_from_scene_cache(
        self,
        scene_cache: Gemma4ScenePrefixCache,
        question: str,
    ) -> str:
        """Greedily answer by extending a private clone of a fixed scene cache."""

        if self.backend != _GEMMA4_BACKEND:
            raise ValueError("Scene-prefix caching is implemented only for Gemma 4")
        if scene_cache.contract != _GEMMA4_SCENE_CACHE_CONTRACT:
            raise ValueError("Unsupported direct-image scene-cache contract")
        if not question.strip():
            raise ValueError("Question must not be empty")
        placeholder_conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in range(scene_cache.complete_view_count)],
                    {
                        "type": "text",
                        "text": f"{self.system_prompt}\nQuestion: {question}",
                    },
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            placeholder_conversation,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            tokenize=False,
        )
        if not isinstance(rendered, str) or not rendered.startswith(
            scene_cache.rendered_prefix_text
        ):
            raise ValueError("Gemma 4 question changed the fixed scene-prefix template")
        suffix_text = rendered[len(scene_cache.rendered_prefix_text) :]
        suffix_token_ids = self.processor.tokenizer.encode(
            suffix_text,
            add_special_tokens=False,
        )
        if not suffix_token_ids:
            raise ValueError("Question produced no Gemma 4 suffix tokens")
        suffix = torch.tensor(
            [suffix_token_ids],
            dtype=torch.long,
            device=self.device,
        )
        suffix_attention = torch.ones_like(suffix)
        attention_mask = torch.cat(
            [scene_cache.prefix_attention_mask, suffix_attention],
            dim=1,
        )
        # DynamicSlidingWindowLayer cannot be cropped after a long prefill.
        # Deep-copying the approximately KV-sized state is both safe and much
        # cheaper than recomputing the 24-image, ~6k-token prefix.
        question_cache = copy.deepcopy(scene_cache.past_key_values)
        generation_kwargs: dict[str, Any] = {
            "input_ids": suffix,
            "attention_mask": attention_mask,
            "past_key_values": question_cache,
            "use_cache": True,
            "do_sample": False,
            "max_new_tokens": self.max_answer_tokens,
        }
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        with torch.inference_mode():
            output = self.model.generate(**generation_kwargs)
        continuation = output[0, suffix.shape[1] :].detach().cpu()
        parser_prefix = torch.cat(
            [scene_cache.prefix_input_ids_cpu, suffix.detach().cpu()],
            dim=1,
        )
        return self._decode_gemma4_continuation(continuation, prefix=parser_prefix)

    def __call__(self, images: list[Image.Image], question: str) -> str:
        if self.backend == _GEMMA4_BACKEND:
            conversation = gemma4_multiview_conversation(
                question,
                images,
                self.system_prompt,
            )
            # Gemma 4's official processor consumes the PIL frames directly from
            # the interleaved conversation. No resize/crop options from another
            # VLM family are forwarded.
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            conversation = multiview_conversation(
                question,
                len(images),
                self.system_prompt,
            )
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(
                images=images,
                text=prompt,
                return_tensors="pt",
                images_kwargs={
                    "size": {"longest_edge": self.resize_longest_edge},
                    "max_image_size": {"longest_edge": self.resize_longest_edge},
                    "do_image_splitting": False,
                },
            )
        moved = self._move_inputs(dict(inputs))
        prompt_tokens = int(moved["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            **moved,
            "do_sample": False,
            "max_new_tokens": self.max_answer_tokens,
        }
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if self.backend != _GEMMA4_BACKEND:
            generation_kwargs["eos_token_id"] = self.processor.tokenizer.eos_token_id
        with torch.inference_mode():
            output = self.model.generate(**generation_kwargs)
        continuation = output[0, prompt_tokens:].detach().cpu()
        if self.backend == _GEMMA4_BACKEND:
            prefix = moved["input_ids"].detach().cpu()
            return self._decode_gemma4_continuation(continuation, prefix=prefix)
        return self.processor.decode(continuation, skip_special_tokens=True).strip()


def run_direct_multiview_baseline(
    config: dict[str, Any],
    references_path: str | Path,
    output_path: str | Path,
    *,
    answerer: ImageAnswerer | None = None,
    limit: int | None = None,
    resume: bool = True,
    local_files_only: bool = True,
    prefer_mps: bool = True,
    selected_references_path: str | Path | None = None,
    use_scene_cache: bool | None = None,
) -> dict[str, Any]:
    """Generate cached predictions from every complete view of each scene."""
    references_source = Path(references_path).resolve()
    destination = Path(output_path).resolve()
    baseline = config["evaluation"]["baselines"]["direct_multiview"]
    if bool(baseline.get("require_local_files_only", False)) and not local_files_only:
        raise ValueError("This direct-image baseline is pinned to local-files-only loading")
    backend = direct_multiview_backend(baseline)
    configured_scene_cache = direct_multiview_scene_cache_contract(baseline)
    scene_cache_contract = (
        configured_scene_cache if use_scene_cache is not False else _DISABLED_SCENE_CACHE
    )
    references = read_jsonl(references_source)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        references = references[:limit]
    selected_references_destination = (
        Path(selected_references_path).resolve() if selected_references_path is not None else None
    )
    if selected_references_destination is not None:
        # Evaluation targets remain physically separate and are never forwarded
        # to the image answerer; this exact slice permits honest smoke scoring.
        atomic_write_jsonl(selected_references_destination, references)
    existing = indexed_records(destination) if resume else {}
    model_answerer = answerer
    model_revision = str(baseline["revision"])
    model_id = str(baseline["model_id"])
    inference_sha256 = text_fingerprint(
        backend,
        model_id,
        model_revision,
        str(baseline["system_prompt"]),
        str(baseline["max_answer_tokens"]),
        str(baseline.get("max_views")),
        str(bool(baseline.get("require_all_manifest_views", False))),
        str(bool(baseline.get("enable_thinking", False))),
        scene_cache_contract,
    )
    data_root = PROJECT_ROOT / str(config["paths"]["data_root"])
    rendered_root = data_root / "rendered"
    path_cache: dict[str, list[Path]] = {}
    observation_hashes: dict[str, str] = {}
    timings: list[float] = []
    scene_cache_build_timings: list[float] = []
    generated: list[dict[str, Any]] = []
    view_counts: set[int] = set()
    manifest_view_counts: set[int] = set()
    scene_cache_hashes: dict[str, str] = {}
    active_scene_id: str | None = None
    active_scene_cache: Any | None = None
    for reference in references:
        scene_id = str(reference["scene_id"])
        question_id = str(reference["question_id"])
        question = str(reference["question"])
        question_sha256 = text_fingerprint(scene_id, question_id, question)
        if scene_id not in path_cache:
            all_paths = complete_view_paths(rendered_root / scene_id)
            manifest_view_counts.add(len(all_paths))
            configured_max_views = baseline.get("max_views")
            max_views = int(configured_max_views) if configured_max_views is not None else None
            if (
                bool(baseline.get("require_all_manifest_views", False))
                and max_views is not None
                and max_views < len(all_paths)
            ):
                raise ValueError(
                    f"{scene_id} has {len(all_paths)} complete views, but max_views={max_views}; "
                    "the configured direct-image control requires every manifest view"
                )
            path_cache[scene_id] = all_paths if max_views is None else all_paths[:max_views]
            observation_hashes[scene_id] = observation_fingerprint(
                rendered_root / scene_id, path_cache[scene_id]
            )
        image_paths = path_cache[scene_id]
        view_counts.add(len(image_paths))
        cached = existing.get((scene_id, question_id))
        if (
            cached is not None
            and cached.get("baseline") == "direct_multiview_images"
            and cached.get("question_sha256") == question_sha256
            and cached.get("observation_sha256") == observation_hashes[scene_id]
            and cached.get("model_revision") == model_revision
            and cached.get("inference_sha256") == inference_sha256
            and int(cached.get("complete_view_count", -1)) == len(image_paths)
            and cached.get("scene_cache_contract") == scene_cache_contract
        ):
            cached_scene_hash = cached.get("scene_cache_sha256")
            if isinstance(cached_scene_hash, str):
                scene_cache_hashes.setdefault(scene_id, cached_scene_hash)
            generated.append(cached)
            continue
        if model_answerer is None:
            model_answerer = LocalMultiViewAnswerer.load(
                config,
                local_files_only=local_files_only,
                prefer_mps=prefer_mps,
            )
        scene_cache_sha256: str | None = None
        if scene_cache_contract != _DISABLED_SCENE_CACHE:
            prepare_scene_cache = getattr(model_answerer, "prepare_scene_cache", None)
            answer_from_scene_cache = getattr(model_answerer, "answer_from_scene_cache", None)
            if not callable(prepare_scene_cache) or not callable(answer_from_scene_cache):
                raise TypeError(
                    "Configured direct-image scene caching requires prepare_scene_cache and "
                    "answer_from_scene_cache methods"
                )
            if active_scene_id != scene_id or active_scene_cache is None:
                # Validation references are scene-major (36 questions per
                # scene), so only one immutable scene cache must reside on the
                # accelerator. A non-contiguous repeat is safely recomputed.
                active_scene_cache = None
                active_scene_id = scene_id
                images: list[Image.Image] = []
                try:
                    for path in image_paths:
                        with Image.open(path) as source:
                            images.append(source.convert("RGB"))
                    cache_started = time.perf_counter()
                    active_scene_cache = prepare_scene_cache(images)
                    scene_cache_build_timings.append(time.perf_counter() - cache_started)
                finally:
                    for image in images:
                        image.close()
                if getattr(active_scene_cache, "contract", None) != scene_cache_contract:
                    raise ValueError("Answerer returned the wrong direct-image scene-cache contract")
                if int(getattr(active_scene_cache, "complete_view_count", -1)) != len(image_paths):
                    raise ValueError("Scene cache did not consume every selected complete image")
                prefix_token_sha256 = getattr(active_scene_cache, "prefix_token_sha256", None)
                if not isinstance(prefix_token_sha256, str) or len(prefix_token_sha256) != 64:
                    raise ValueError("Scene cache omitted its question-independent prefix-token hash")
                scene_cache_hashes[scene_id] = text_fingerprint(
                    scene_cache_contract,
                    inference_sha256,
                    observation_hashes[scene_id],
                    prefix_token_sha256,
                )
            scene_cache_sha256 = scene_cache_hashes[scene_id]
            started = time.perf_counter()
            answer = answer_from_scene_cache(active_scene_cache, question)
            timings.append(time.perf_counter() - started)
        else:
            images = []
            try:
                for path in image_paths:
                    with Image.open(path) as source:
                        images.append(source.convert("RGB"))
                started = time.perf_counter()
                answer = model_answerer(images, question)
                timings.append(time.perf_counter() - started)
            finally:
                for image in images:
                    image.close()
        generated.append(
            {
                "answer": answer,
                "baseline": "direct_multiview_images",
                "complete_view_count": len(image_paths),
                "evaluation_only": True,
                "inference_sha256": inference_sha256,
                "model_id": model_id,
                "model_revision": model_revision,
                "observation_sha256": observation_hashes[scene_id],
                "primary_path_eligible": False,
                "prohibited_primary_substitute": True,
                "question_id": question_id,
                "question_sha256": question_sha256,
                "scene_cache_contract": scene_cache_contract,
                "scene_cache_question_independent": (
                    scene_cache_contract != _DISABLED_SCENE_CACHE
                ),
                "scene_cache_sha256": scene_cache_sha256,
                "scene_id": scene_id,
            }
        )
        atomic_write_jsonl(destination, generated)
    # Also removes stale records when every requested prediction was cached.
    atomic_write_jsonl(destination, generated)
    return {
        "baseline": "direct_multiview_images",
        "evaluation_only": True,
        "primary_path_eligible": False,
        "prohibited_primary_substitute": True,
        "input_modalities": ["complete_rgb_frames", "user_question"],
        "excluded_input_modalities": [
            "depth",
            "oracle_metadata",
            "segmentation",
            "scene_caption",
        ],
        "all_manifest_views_required": bool(baseline.get("require_all_manifest_views", False)),
        "backend": backend,
        "generated_count": len(generated),
        "new_prediction_count": len(timings),
        "mean_seconds_per_new_prediction": (sum(timings) / len(timings) if timings else 0.0),
        "scene_cache_build_count": len(scene_cache_build_timings),
        "scene_cache_build_seconds": sum(scene_cache_build_timings),
        "scene_cache_contract": scene_cache_contract,
        "scene_cache_question_independent": scene_cache_contract != _DISABLED_SCENE_CACHE,
        "scene_cache_sha256_by_scene": dict(sorted(scene_cache_hashes.items())),
        "inference_sha256": inference_sha256,
        "local_files_only": local_files_only,
        "model_id": model_id,
        "model_revision": str(baseline["revision"]),
        "model_license": str(baseline["license"]),
        "output_path": str(destination),
        "output_sha256": sha256_file(destination),
        "reference_path": str(references_source),
        "selected_references_path": (
            str(selected_references_destination)
            if selected_references_destination is not None
            else None
        ),
        "selected_references_sha256": (
            sha256_file(selected_references_destination)
            if selected_references_destination is not None
            else None
        ),
        "manifest_view_counts": sorted(manifest_view_counts),
        "view_counts": sorted(view_counts),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--no-scene-cache",
        action="store_true",
        help="Disable the versioned Gemma 4 scene-level decoder-KV cache for parity debugging.",
    )
    parser.add_argument("--selected-references-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    data_root = PROJECT_ROOT / str(config["paths"]["data_root"])
    reports_root = PROJECT_ROOT / str(config["paths"]["reports_root"])
    configured_qa_root = config["paths"].get("qa_root")
    qa_root = (
        PROJECT_ROOT / str(configured_qa_root)
        if configured_qa_root is not None
        else data_root / "qa"
    )
    references = args.references or qa_root / "test.jsonl"
    output = args.output or reports_root / "predictions" / "direct_multiview.jsonl"
    report = run_direct_multiview_baseline(
        config,
        references,
        output,
        limit=args.limit,
        resume=not args.no_resume,
        local_files_only=not args.allow_download,
        prefer_mps=not args.cpu,
        selected_references_path=args.selected_references_output,
        use_scene_cache=not args.no_scene_cache,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
