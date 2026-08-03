from __future__ import annotations

import argparse

from dsga.quality import build_qc_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a review manifest for synthetic support images")
    parser.add_argument("--input", default="data/generated")
    parser.add_argument("--output", default="data/generated/qc_manifest.csv")
    parser.add_argument("--accept-auto", action="store_true", help="Accept technical passes; manual review remains recommended")
    args = parser.parse_args()
    output = build_qc_manifest(args.input, args.output, args.accept_auto)
    print(f"Review manifest: {output}")


if __name__ == "__main__":
    main()

