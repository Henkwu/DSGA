#!/usr/bin/env bash
set -euo pipefail

python train_collaborative.py --config configs/dsga.yaml
python train_meta.py --config configs/dsga.yaml
python evaluate.py --config configs/dsga.yaml

