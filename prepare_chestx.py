from __future__ import annotations

import argparse
import csv
from pathlib import Path


CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumothorax"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a target manifest from NIH ChestX-ray14 metadata")
    parser.add_argument("--metadata", required=True, help="Data_Entry_2017.csv")
    parser.add_argument("--images", required=True, help="Directory containing PNG files, recursively")
    parser.add_argument("--output", default="data/manifests/chestx.csv")
    parser.add_argument("--single-label-only", action="store_true")
    args = parser.parse_args()
    image_index = {p.name: p.resolve() for p in Path(args.images).rglob("*.png")}
    rows, missing = [], []
    with Path(args.metadata).open("r", encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            name = record.get("Image Index", "")
            findings = [x.strip() for x in record.get("Finding Labels", "").split("|")]
            if args.single_label_only and len(findings) != 1:
                continue
            path = image_index.get(name)
            if path is None:
                missing.append(name)
                continue
            for disease in CLASSES:
                if disease in findings:
                    rows.append({"path": str(path), "label": disease, "split": "test"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {output}; {len(missing)} metadata rows had no image")


if __name__ == "__main__":
    main()

