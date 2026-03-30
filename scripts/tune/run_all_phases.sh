#!/bin/bash
# Full tuning sweep: 25 trials per phase, kh9 profile, fast proxy
# Phases run in dependency order

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TUNE_SCRIPT="$SCRIPT_DIR/tune.py"

PHASES=(coarse scale_rotation matching validation grid_optim flow normalization)
N_TRIALS=25
PROFILE=kh9
TIMEOUT=14400  # 4 hours per phase max

for phase in "${PHASES[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Phase: $phase ($N_TRIALS trials)"
    echo "  $(date)"
    echo "=========================================="
    echo ""

    python3 "$TUNE_SCRIPT" \
        --phase "$phase" \
        --profile "$PROFILE" \
        --n-trials "$N_TRIALS" \
        --timeout "$TIMEOUT" \
        --fast-proxy \
        --build-checkpoint \
        || echo "WARNING: Phase $phase failed or timed out"

    echo ""
    echo "Phase $phase completed at $(date)"
done

echo ""
echo "=========================================="
echo "  All phases complete: $(date)"
echo "=========================================="
