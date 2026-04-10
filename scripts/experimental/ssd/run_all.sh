#!/bin/bash
# Experimental SSD training pipeline (self-supervised distillation).
# See scripts/experimental/ssd/README.md for status and prerequisites.
# Run from repo root so the data/ssd_* paths resolve.
set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== 1. Starting Data Generation (Target: 2000 pairs) ==="
$PYTHON_BIN scripts/experimental/ssd/build_dataset.py -o data/ssd_pairs -c 2000 -w 8

echo "=== 2. Starting Pseudo Label Extraction ==="
$PYTHON_BIN scripts/experimental/ssd/extract_pseudo_labels.py -d data/ssd_pairs -o data/ssd_labels

echo "=== 3. Starting SSD Fine-Tuning ==="
$PYTHON_BIN scripts/experimental/ssd/finetune.py -d data/ssd_labels

echo "=== All Steps Completed Successfully ==="