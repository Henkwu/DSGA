from __future__ import annotations

import argparse

from dsga.config import get, load_config, merge_overrides
from dsga.pipeline import evaluate
from dsga.utils import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="DSGA target-domain episodic meta-testing")
    parser.add_argument("--config", default="configs/dsga.yaml")
    parser.add_argument("--checkpoint", help="Meta checkpoint; defaults to output_dir/meta.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = merge_overrides(load_config(args.config), args.set)
    seed_everything(int(get(config, "seed", 1)))
    output = evaluate(config, resolve_device(str(get(config, "device", "cuda"))), args.checkpoint)
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

