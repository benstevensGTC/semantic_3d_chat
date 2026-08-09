"""Evaluation-only direct multi-view local VLM baseline.

Only complete RGB frames from the sanitized manifest and the user's question
are supplied.  This baseline consumes neither oracle metadata nor depth and is
not a substitute for the project's primary persistent 3D scene-memory path.
"""

from __future__ import annotations

import argparse
import json
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
    paths = [(scene_root / str(frame["rgb_path"])).resolve() for frame in frames]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"Complete RGB frames are missing below {scene_root}")
    return paths


def multiview_conversation(question: str, image_count: int, system_prompt: str) -> list[dict[str, Any]]:
    if image_count <= 0:
        raise ValueError("At least one complete image is required")
    content: list[dict[str, str]] = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": f"{system_prompt}\nQuestion: {question}"})
    return [{"role": "user", "content": content}]


def observation_fingerprint(scene_root: Path, image_paths: list[Path]) -> str:
    manifest_hash = sha256_file(scene_root / "manifest.json")
    image_hashes = [sha256_file(path) for path in image_paths]
    return text_fingerprint(manifest_hash, *image_hashes)


@dataclass
class LocalMultiViewAnswerer:
    model: Any
    processor: Any
    device: torch.device
    dtype: torch.dtype
    system_prompt: str
    max_answer_tokens: int
    resize_longest_edge: int

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
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
        model.requires_grad_(False)
        model.eval()
        return cls(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            system_prompt=str(baseline["system_prompt"]),
            max_answer_tokens=int(baseline["max_answer_tokens"]),
            resize_longest_edge=int(baseline["resize_longest_edge"]),
        )

    def __call__(self, images: list[Image.Image], question: str) -> str:
        conversation = multiview_conversation(question, len(images), self.system_prompt)
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
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
            elif torch.is_floating_point(value):
                moved[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        prompt_tokens = int(moved["input_ids"].shape[1])
        with torch.inference_mode():
            output = self.model.generate(
                **moved,
                do_sample=False,
                max_new_tokens=self.max_answer_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        continuation = output[0, prompt_tokens:].detach().cpu()
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
) -> dict[str, Any]:
    """Generate cached predictions from every complete view of each scene."""
    references_source = Path(references_path).resolve()
    destination = Path(output_path).resolve()
    references = read_jsonl(references_source)
    if limit is not None:
        references = references[:limit]
    existing = indexed_records(destination) if resume else {}
    baseline = config["evaluation"]["baselines"]["direct_multiview"]
    model_answerer = answerer
    model_revision = str(baseline["revision"])
    data_root = PROJECT_ROOT / str(config["paths"]["data_root"])
    rendered_root = data_root / "rendered"
    path_cache: dict[str, list[Path]] = {}
    observation_hashes: dict[str, str] = {}
    timings: list[float] = []
    generated: list[dict[str, Any]] = []
    view_counts: set[int] = set()
    for reference in references:
        scene_id = str(reference["scene_id"])
        question_id = str(reference["question_id"])
        question = str(reference["question"])
        question_sha256 = text_fingerprint(scene_id, question_id, question)
        if scene_id not in path_cache:
            path_cache[scene_id] = complete_view_paths(
                rendered_root / scene_id,
                max_views=int(baseline["max_views"]),
            )
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
            and int(cached.get("complete_view_count", -1)) == len(image_paths)
        ):
            generated.append(cached)
            continue
        if model_answerer is None:
            model_answerer = LocalMultiViewAnswerer.load(
                config,
                local_files_only=local_files_only,
                prefer_mps=prefer_mps,
            )
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
                "model_revision": model_revision,
                "observation_sha256": observation_hashes[scene_id],
                "question_id": question_id,
                "question_sha256": question_sha256,
                "scene_id": scene_id,
            }
        )
        atomic_write_jsonl(destination, generated)
    # Also removes stale records when every requested prediction was cached.
    atomic_write_jsonl(destination, generated)
    return {
        "baseline": "direct_multiview_images",
        "evaluation_only": True,
        "generated_count": len(generated),
        "new_prediction_count": len(timings),
        "mean_seconds_per_new_prediction": (
            sum(timings) / len(timings) if timings else 0.0
        ),
        "model_id": str(baseline["model_id"]),
        "model_revision": str(baseline["revision"]),
        "model_license": str(baseline["license"]),
        "output_path": str(destination),
        "output_sha256": sha256_file(destination),
        "reference_path": str(references_source),
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    data_root = PROJECT_ROOT / str(config["paths"]["data_root"])
    reports_root = PROJECT_ROOT / str(config["paths"]["reports_root"])
    references = args.references or data_root / "qa" / "test.jsonl"
    output = args.output or reports_root / "predictions" / "direct_multiview.jsonl"
    report = run_direct_multiview_baseline(
        config,
        references,
        output,
        limit=args.limit,
        resume=not args.no_resume,
        local_files_only=not args.allow_download,
        prefer_mps=not args.cpu,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
