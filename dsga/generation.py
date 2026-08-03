from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .progress import tqdm


DEFAULT_PROMPT = "a Chest X-ray photo of a {class_name} disease"


def _dtype(name: str) -> torch.dtype:
    values = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in values:
        raise ValueError(f"Unsupported dtype {name!r}")
    return values[name]


def load_pipeline(backend: str, model: str, dtype: str, device: str, cpu_offload: bool):
    try:
        from diffusers import FluxPipeline, StableDiffusionXLPipeline
    except ImportError as error:
        raise RuntimeError("Generation dependencies are missing; install requirements-generation.txt") from error
    torch_dtype = _dtype(dtype)
    if backend == "flux":
        pipeline = FluxPipeline.from_pretrained(model, torch_dtype=torch_dtype)
    elif backend == "sdxl":
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model, torch_dtype=torch_dtype, use_safetensors=True
        )
    else:
        raise ValueError("backend must be 'flux' or 'sdxl'")
    if cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    return pipeline


def generate_support_images(
    output_root: str | Path,
    class_names: list[str],
    backend: str,
    model: str,
    images_per_class: int = 8,
    prompt_template: str = DEFAULT_PROMPT,
    negative_prompt: str = "text, watermark, photograph of a person, non-medical object",
    steps: int = 30,
    guidance_scale: float = 3.5,
    width: int = 1024,
    height: int = 1024,
    seed: int = 1,
    dtype: str = "bfloat16",
    device: str = "cuda",
    cpu_offload: bool = False,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(backend, model, dtype, device, cpu_offload)
    metadata_path = output_root / "generation.jsonl"
    records = []
    for class_index, class_name in enumerate(class_names):
        class_dir = output_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        prompt = prompt_template.format(class_name=class_name)
        for image_index in tqdm(range(images_per_class), desc=f"Generating {class_name}"):
            image_seed = seed + class_index * 100_000 + image_index
            generator_device = "cpu" if cpu_offload else device
            generator = torch.Generator(device=generator_device).manual_seed(image_seed)
            arguments: dict[str, Any] = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "generator": generator,
            }
            if backend == "sdxl":
                arguments["negative_prompt"] = negative_prompt
            image = pipeline(**arguments).images[0]
            destination = class_dir / f"{image_index:03d}.png"
            image.save(destination)
            records.append(
                {
                    "path": str(destination.relative_to(output_root)),
                    "class_name": class_name,
                    "prompt": prompt,
                    "backend": backend,
                    "model": model,
                    "seed": image_seed,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                }
            )
    with metadata_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metadata_path
