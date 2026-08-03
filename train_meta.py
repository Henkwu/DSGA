from __future__ import annotations

import argparse

from dsga.config import get, load_config, merge_overrides
from dsga.pipeline import meta_train
from dsga.utils import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="DSGA stage 2: multi-branch episodic meta-training")
    parser.add_argument("--config", default="configs/dsga.yaml")
    parser.add_argument("--checkpoint", help="Stage-1 checkpoint; defaults to output_dir/collaborative.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = merge_overrides(load_config(args.config), args.set)
    seed_everything(int(get(config, "seed", 1)))
    output = meta_train(config, resolve_device(str(get(config, "device", "cuda"))), args.checkpoint)
    print(f"Saved meta-trained checkpoint: {output}")


if __name__ == "__main__":
    main()

