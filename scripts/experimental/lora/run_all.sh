#!/usr/bin/env bash
# End-to-end LoRA fine-tune: G1 → G3 → G3a → G4 → G5 → G6 → G7 → G8.
#
# Set environment variables to skip earlier gates if you have already
# produced their outputs (idempotent re-runs).
#
#   SKIP_FEATURE_GAP=1     skip G1 if diagnostics/lora/feature_gap.json exists
#   SKIP_PAIR_BUILD=1      skip G3 if data/lora_pairs has ≥ 1000 pairs
#   SKIP_LABEL_EXTRACT=1   skip G3a if labels/ is populated
#   SKIP_TRAIN=1           skip G5 if align/weights/roma_ssd.pth exists
#
# All gates produce JSON outputs under diagnostics/ — failures are visible
# without re-reading the gate's stdout.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PAIRS_DIR="${PAIRS_DIR:-data/lora_pairs}"
TARGET_PAIRS="${TARGET_PAIRS:-1500}"
WEIGHTS_OUT="${WEIGHTS_OUT:-align/weights/roma_ssd.pth}"
RUN_ID="${RUN_ID:-lora_r8_$(date +%Y%m%d_%H%M%S)}"

log() { printf "\n[run_all] %s\n" "$1"; }

# G1 — feature gap diagnostic
if [[ -z "${SKIP_FEATURE_GAP:-}" ]] || [[ ! -f diagnostics/lora/feature_gap.json ]]; then
    log "G1 — feature gap diagnostic"
    poetry run python scripts/experimental/lora/measure_feature_gap.py
fi

# Inspect the decision band (will fall through if PROCEED, exit 1 if STOP)
band=$(poetry run python -c "import json; print(json.load(open('diagnostics/lora/feature_gap.json'))['decision_band'])")
log "G1 result: $band"
if [[ "$band" == "STOP" ]]; then
    log "Features already cross-temporally invariant; aborting"
    exit 0
fi
RANK=8
if [[ "$band" == "PROCEED_r16" ]]; then
    RANK=16
fi

# G2 — mapbh client unit tests
log "G2 — mapbh client tests"
poetry run pytest tests/test_mapbh_client.py -v

# G3 — mapbh pair build
if [[ -z "${SKIP_PAIR_BUILD:-}" ]] || [[ ! -d "$PAIRS_DIR" ]]; then
    log "G3 — mapbh pair build (~1-2 hours)"
    poetry run python scripts/experimental/lora/build_mapbh_pairs.py \
        --output "$PAIRS_DIR" --target-pairs "$TARGET_PAIRS"
fi

# G3a — teacher pseudo-label extraction
if [[ -z "${SKIP_LABEL_EXTRACT:-}" ]] || [[ ! -d "$PAIRS_DIR/labels" ]]; then
    log "G3a — teacher pseudo-label extraction (~30-45 min)"
    poetry run python scripts/experimental/lora/extract_mapbh_pseudo_labels.py \
        --pairs-dir "$PAIRS_DIR"
fi

# G4 — LoRA inject unit tests
log "G4 — LoRA inject tests"
poetry run pytest tests/test_lora_inject.py -v

# G5 — training
if [[ -z "${SKIP_TRAIN:-}" ]] || [[ ! -f "$WEIGHTS_OUT" ]]; then
    log "G5 — LoRA training (4-8 hours on MPS) rank=$RANK run-id=$RUN_ID"
    poetry run python scripts/experimental/lora/finetune.py \
        --rank "$RANK" --alpha "$RANK" \
        --mapbh-dir "$PAIRS_DIR" \
        --run-id "$RUN_ID" \
        --output-weights "$WEIGHTS_OUT"
fi

# G6 — synthetic eval
log "G6 — synthetic A/B eval"
poetry run python scripts/experimental/lora/eval_synthetic.py \
    --weights-a base --weights-b "$WEIGHTS_OUT"

log "G6 results in diagnostics/lora/eval_synthetic.json"
gate_promote=$(poetry run python -c "import json; d = json.load(open('diagnostics/lora/eval_synthetic.json')); print(d.get('promotion_candidate'))")
log "promotion candidate: $gate_promote"

if [[ "$gate_promote" != "True" ]]; then
    log "G6 gate did NOT pass — investigate before proceeding to G7/G8"
    exit 1
fi

log "G7 (production iterate_phase) and G8 (KH-9 regression) must be run manually:"
log ""
log "  DECLASS_ROMA_WEIGHTS=$WEIGHTS_OUT \\"
log "    poetry run python scripts/test/iterate_phase.py \\"
log "      --manifest <kh4b_manifest> --scene <scene_id> --from-phase post_scale_rotation"
log ""
log "  DECLASS_ROMA_WEIGHTS=$WEIGHTS_OUT \\"
log "    poetry run python scripts/test/run_test.py \\"
log "      --config scripts/test/e2e_configs/bahrain_kh9_pc_1977.yaml"
