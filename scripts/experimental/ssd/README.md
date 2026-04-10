# Self-Supervised Distillation (SSD) — Experimental

**Status: in-progress research. Downstream alignment improvement has not been demonstrated. Do not enable for production runs without validating against the v66 Bahrain baseline (score 60.1, patch median 9m).**

See `hypothesis.md` Contribution 7 for the full research framing.

## What this is

A teacher-student distillation framework that fine-tunes dense feature matchers (RoMa, SEA-RAFT) on the domain of historical film-to-modern-digital satellite registration:

1. **Data generation** — fetch historical KH imagery from the CORONA Atlas WMS and paired Sentinel-2 L2A, then apply synthetic geometric perturbations (affine + TPS + radial distortion) to the historical image.
2. **Pseudo-label extraction** — run frozen pretrained RoMa + SEA-RAFT as the *teacher* on the clean (unperturbed) pair, capturing its correspondence/flow predictions.
3. **Student training** — train the same architecture on the perturbed pair, supervised by the teacher's pseudo-labels. RoMa backbone (DINOv3) is frozen; only the matcher head and refiners are trained.

## Enabling at inference time

Weight loading is gated behind an environment variable. Default is **off** — the pipeline loads the base RoMa and SEA-RAFT weights.

```bash
# Default (base weights, production-validated)
python3 auto-align.py input.tif -r reference.tif

# Opt in to SSD-finetuned weights (experimental)
DECLASS_EXPERIMENTAL_SSD=1 python3 auto-align.py input.tif -r reference.tif
```

The gate is implemented in `align/models.py::use_ssd_weights()`. When on, the override logic in `align/models.py` (RoMa) and `align/flow_refine.py` (SEA-RAFT) loads `align/weights/roma_ssd.pth` / `searaft_ssd.pth` in place of the base checkpoints.

## Training pipeline

Run from the repo root:

```bash
# 1. Generate training data (target: 2000 historical-modern pairs)
python3 scripts/experimental/ssd/build_dataset.py -o data/ssd_pairs -c 2000 -w 8

# 2. Extract teacher pseudo-labels on clean pairs
python3 scripts/experimental/ssd/extract_pseudo_labels.py -d data/ssd_pairs -o data/ssd_labels

# 3. Fine-tune the student
python3 scripts/experimental/ssd/finetune.py -d data/ssd_labels

# Or run all three in sequence
bash scripts/experimental/ssd/run_all.sh
```

Hyperparameter search via Optuna:

```bash
python3 scripts/experimental/ssd/tune.py -d data/ssd_labels --n-trials 20
```

## Known gaps

- **Dataset is incomplete** — last run reached ~935/2000 pairs before hitting Sentinel-2 S3 rate-limiting. Resume logic is in place but the full 2000-pair target has not been met.
- **No downstream validation** — fine-tuned weights have not been shown to beat base weights on the Bahrain v66 test case. Until they do, the pipeline should stay on base weights.
- **`download_weights.py` URLs are stubs** — the fine-tuned weights are not hosted publicly. Train locally or copy `roma_ssd.pth` / `searaft_ssd.pth` into `align/weights/` manually.

## Deleting the thread if it turns out to be unviable

The entire SSD thread is isolated under `scripts/experimental/ssd/` plus the two override blocks in `align/models.py` and `align/flow_refine.py` (both guarded by `use_ssd_weights()`). Removing the thread is:

1. `rm -rf scripts/experimental/ssd/`
2. `rm -f align/weights/roma_ssd.pth align/weights/searaft_ssd.pth`
3. Remove `use_ssd_weights()` function from `align/models.py` and its two call sites in `align/models.py` and `align/flow_refine.py`.
4. Remove Contribution 7 from `hypothesis.md`.
