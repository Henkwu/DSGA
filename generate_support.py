from __future__ import annotations

import argparse
from pathlib import Path

from dsga.generation import DEFAULT_PROMPT, generate_support_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate class-conditioned auxiliary support images")
    parser.add_argument("--classes", "--txt_list", default="ChestX.txt", help="One disease class per line")
    parser.add_argument("--output", "--out", default="data/generated")
    parser.add_argument("--backend", choices=["flux", "sdxl"], default="flux")
    parser.add_argument("--model", "--base_model_path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--images-per-class", "--num-images-per-view", type=int, default=8)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="text, watermark, person, non-medical object")
    parser.add_argument("--steps", "--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()
    classes = [line.strip() for line in Path(args.classes).read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = generate_support_images(
        args.output, classes, args.backend, args.model, args.images_per_class,
        args.prompt_template, args.negative_prompt, args.steps, args.guidance_scale,
        args.width, args.height, args.seed, args.dtype, args.device, args.cpu_offload,
    )
    print(f"Generation metadata: {metadata}")


if __name__ == "__main__":
    main()
