from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


class ChestXStyleTransform:
    """Paper transformation: grayscale -> CLAHE contrast -> Gaussian blur."""

    def __init__(self, clip_limit: float = 4.0, tile_grid: int = 8, blur_kernel: int = 13):
        if blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd")
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid
        self.blur_kernel = blur_kernel

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid, self.tile_grid),
        )
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (self.blur_kernel, self.blur_kernel), 0)
        return Image.fromarray(blurred, mode="L").convert("RGB")


def materialize_chestx_style(
    input_root: str | Path,
    output_root: str | Path,
    clip_limit: float = 4.0,
    tile_grid: int = 8,
    blur_kernel: int = 13,
) -> int:
    input_root, output_root = Path(input_root), Path(output_root)
    transform = ChestXStyleTransform(clip_limit, tile_grid, blur_kernel)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    count = 0
    for source in sorted(p for p in input_root.rglob("*") if p.suffix.lower() in extensions):
        relative = source.relative_to(input_root).with_suffix(".png")
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            transform(image).save(destination)
        count += 1
    return count


class RandomChestXStyle:
    def __init__(self, transform: ChestXStyleTransform, probability: float = 1.0):
        self.transform = transform
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.transform(image) if random.random() < self.probability else image.convert("RGB")

