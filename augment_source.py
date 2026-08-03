from __future__ import annotations

import argparse

from dsga.augmentation import materialize_chestx_style


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize DSGA chest-X-ray-style source images")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-limit", type=float, default=4.0)
    parser.add_argument("--tile-grid", type=int, default=8)
    parser.add_argument("--blur-kernel", type=int, default=13)
    args = parser.parse_args()
    count = materialize_chestx_style(args.input, args.output, args.clip_limit, args.tile_grid, args.blur_kernel)
    print(f"Wrote {count} transformed images to {args.output}")


if __name__ == "__main__":
    main()

