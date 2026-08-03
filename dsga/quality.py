from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .data import IMAGE_EXTENSIONS


def _entropy(gray: np.ndarray) -> float:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def inspect_image(path: Path, min_size: int = 224) -> dict:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    channel_difference = float(np.abs(rgb.astype(np.float32) - gray[..., None]).mean())
    metrics = {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "entropy": _entropy(gray),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "channel_difference": channel_difference,
    }
    reasons = []
    if min(rgb.shape[:2]) < min_size:
        reasons.append("resolution")
    if metrics["mean"] < 8 or metrics["mean"] > 247:
        reasons.append("exposure")
    if metrics["std"] < 8:
        reasons.append("low_contrast")
    if metrics["entropy"] < 3.0:
        reasons.append("low_entropy")
    if metrics["channel_difference"] > 15:
        reasons.append("not_grayscale_like")
    return {**metrics, "auto_pass": not reasons, "reasons": ";".join(reasons)}


def build_qc_manifest(root: str | Path, output: str | Path, accept_auto: bool = False) -> Path:
    root, output = Path(root).resolve(), Path(output)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    rows = []
    for path in paths:
        result = inspect_image(path)
        rows.append(
            {
                "path": str(path),
                "class_name": path.parent.name,
                **result,
                "human_anatomy_ok": "",
                "human_pathology_ok": "",
                "review_notes": "",
                "accepted": bool(result["auto_pass"] and accept_auto),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["path", "class_name", "accepted"]
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output

