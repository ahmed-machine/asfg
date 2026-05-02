#!/usr/bin/env bash
# Re-run kh7 / kh4b / kh9 with the downstream-path fixes shipped in
# this session (AlreadyAlignedError pass-through, footprint mask CRS
# reproject, KH-7 pinhole fallback, DZB dispatch fix).
set -u
cd "$(dirname "$0")/../.."
STATUS=scripts/test/verify_profiles_remaining.status
PYTHON=.venv/bin/python
RUNNER="$PYTHON scripts/test/run_e2e_test.py"
TIMEOUT=7200  # 2h each

configs=(
    bahrain_kh4b_1968
    bahrain_kh9_pc_1977
    bahrain_kh7_1967
)

: > "$STATUS"
echo "=== verify_profiles_remaining.sh: starting $(date -u +%FT%TZ) ===" | tee -a "$STATUS"

for cfg in "${configs[@]}"; do
    echo "--- $cfg --- $(date -u +%FT%TZ)" | tee -a "$STATUS"
    start=$(date +%s)
    log="scripts/test/verify_${cfg}_v2.log"
    if $RUNNER --config "$cfg" --timeout "$TIMEOUT" --skip-download > "$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    end=$(date +%s)
    echo "    exit=$rc  duration_s=$((end-start))  log=$log" | tee -a "$STATUS"
done
echo "=== verify_profiles_remaining.sh: finished $(date -u +%FT%TZ) ===" | tee -a "$STATUS"
