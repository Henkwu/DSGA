from __future__ import annotations

import argparse
import torch

from dsga.visualize import prototype_gradcam


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DSGA prototype evidence with Grad-CAM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--support-dir", required=True, help="Real/generated support images for one disease")
    parser.add_argument("--output", default="outputs/gradcam.png")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    output = prototype_gradcam(
        args.checkpoint, args.query, args.support_dir, args.output, torch.device(args.device), args.image_size
    )
    print(f"Saved visualization: {output}")


if __name__ == "__main__":
    main()

