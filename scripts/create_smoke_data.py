from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_image(class_index: int, image_index: int, size: int = 80) -> Image.Image:
    rng = np.random.default_rng(class_index * 10_000 + image_index)
    base = rng.normal(80 + 45 * class_index, 20, (size, size, 3)).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    margin = 8 + image_index % 8
    draw.ellipse((margin, margin, size - margin, size - margin), outline=(220, 220, 220), width=3)
    draw.line((0, class_index * 12 + 10, size, class_index * 12 + 10), fill="white", width=2)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic data for the CPU smoke test")
    parser.add_argument("--output", default="smoke_data")
    args = parser.parse_args()
    root = Path(args.output)
    classes = ["class_a", "class_b", "class_c"]
    for domain, count in (("source", 6), ("target", 6), ("generated", 2)):
        for class_index, class_name in enumerate(classes):
            directory = root / domain / class_name
            directory.mkdir(parents=True, exist_ok=True)
            for image_index in range(count):
                make_image(class_index, image_index + (100 if domain == "target" else 200 if domain == "generated" else 0)).save(
                    directory / f"{image_index:03d}.png"
                )
    print(f"Created deterministic smoke data under {root}")


if __name__ == "__main__":
    main()
