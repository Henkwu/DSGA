from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .data import IMAGE_EXTENSIONS, build_transform
from .models import build_model, load_encoder_weights
from .utils import load_checkpoint


def prototype_gradcam(
    checkpoint: str | Path,
    query_path: str | Path,
    support_dir: str | Path,
    output: str | Path,
    device: torch.device,
    image_size: int = 224,
) -> Path:
    state = load_checkpoint(checkpoint, device)
    model = build_model(max(len(state.get("class_names", [])), 1)).to(device)
    load_encoder_weights(model, state, strict=False)
    model.eval()
    transform = build_transform(image_size, train=False)
    support_paths = sorted(
        p for p in Path(support_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not support_paths:
        raise ValueError("support_dir contains no images")
    support_tensors = []
    for path in support_paths:
        with Image.open(path) as image:
            support_tensors.append(transform(image.convert("RGB")))
    with torch.no_grad():
        prototype = model(torch.stack(support_tensors).to(device), return_embeddings=True)[-1].mean(0)
    with Image.open(query_path) as image:
        original = image.convert("RGB").resize((image_size, image_size))
        query = transform(image.convert("RGB")).unsqueeze(0).to(device)
    feature_map = model.feature_maps(query)[-1]
    feature_map.retain_grad()
    embedding = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)[0]
    score = F.cosine_similarity(embedding[None], prototype[None]).sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    gradients = feature_map.grad[0]
    weights = gradients.mean(dim=(1, 2), keepdim=True)
    heatmap = torch.relu((weights * feature_map[0]).sum(0)).detach().cpu().numpy()
    heatmap -= heatmap.min()
    heatmap /= max(float(heatmap.max()), 1e-8)
    heatmap = cv2.resize(heatmap, (image_size, image_size))
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    original_array = np.asarray(original)
    overlay = np.uint8(np.clip(0.55 * original_array + 0.45 * colored, 0, 255))
    panels = [original, Image.fromarray(colored), Image.fromarray(overlay)]
    labels = ["Query image", "Prototype Grad-CAM", "Overlay"]
    header = 28
    canvas = Image.new("RGB", (image_size * 3, image_size + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * image_size
        canvas.paste(panel, (x, header))
        draw.text((x + 8, 7), label, fill="black")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output

