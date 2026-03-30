#!/bin/bash
# Rebuild checkpoint with tuned kh9 profile, build grid cache, then tune flow + normalization
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
TUNE_SCRIPT="$SCRIPT_DIR/tune.py"
DIAG_DIR="$PROJECT_ROOT/diagnostics/tune_kh9"
CHECKPOINT_DIR="$DIAG_DIR/checkpoints"

echo "=========================================="
echo "  Step 1: Rebuild checkpoint with tuned kh9 profile"
echo "  $(date)"
echo "=========================================="

rm -f "$CHECKPOINT_DIR"/post_*.json "$CHECKPOINT_DIR"/post_*.npz
echo "  Cleared old checkpoints"

python3 "$TUNE_SCRIPT" \
    --phase flow \
    --profile kh9 \
    --n-trials 0 \
    --timeout 60 \
    --build-checkpoint \
    --force-rebuild \
    --build-grid-cache \
    || true

echo ""
echo "  Checkpoint + grid cache done at $(date)"
ls -la "$CHECKPOINT_DIR"/grid_optim_cache.* 2>/dev/null || echo "  WARNING: Grid cache not found!"

echo ""
echo "=========================================="
echo "  Step 2: Clear old flow/normalization studies"
echo "=========================================="

python3 -c "
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
db = 'sqlite:///$DIAG_DIR/tune_kh9.db'
for name in ['tune_flow_kh9', 'tune_normalization_kh9']:
    try:
        optuna.delete_study(study_name=name, storage=db)
        print(f'  Deleted: {name}')
    except: pass
"

echo ""
echo "=========================================="
echo "  Step 3: Tune flow (20 trials, fast via grid cache)"
echo "  $(date)"
echo "=========================================="

python3 "$TUNE_SCRIPT" \
    --phase flow \
    --profile kh9 \
    --n-trials 20 \
    --timeout 14400 \
    --fast-proxy \
    || echo "WARNING: flow phase failed or timed out"

echo ""
echo "=========================================="
echo "  Step 4: Tune normalization (20 trials, fast via grid cache)"
echo "  $(date)"
echo "=========================================="

python3 "$TUNE_SCRIPT" \
    --phase normalization \
    --profile kh9 \
    --n-trials 20 \
    --timeout 14400 \
    --fast-proxy \
    || echo "WARNING: normalization phase failed or timed out"

echo ""
echo "=========================================="
echo "  All done: $(date)"
echo "=========================================="
