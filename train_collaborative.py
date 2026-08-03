from __future__ import annotations

import argparse

from dsga.config import get, load_config, merge_overrides
from dsga.pipeline import collaborative_train
from dsga.utils import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="DSGA stage 1: cross-domain collaborative training")
    parser.add_argument("--config", default="configs/dsga.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = merge_overrides(load_config(args.config), args.set)
    seed_everything(int(get(config, "seed", 1)))
    output = collaborative_train(config, resolve_device(str(get(config, "device", "cuda"))))
    print(f"Saved collaborative checkpoint: {output}")


if __name__ == "__main__":
    main()

