#!/usr/bin/env bash
# Sequentially run e2e tests for each active KH profile — verification of
# the new per-camera-system profile set (kh4a/kh4b/kh7/kh9). The stub
# profiles (kh8, kh9_mc) are excluded because their camera models aren't
# wired into the pipeline yet.
#
# Writes per-config status to scripts/test/verify_profiles.status.

set -u
cd "$(dirname "$0")/../.."

STATUS=scripts/test/verify_profiles.status
PYTHON=.venv/bin/python
RUNNER="$PYTHON scripts/test/run_e2e_test.py"
# 60-min budget per test.
TIMEOUT=3600

# Order: lightest first, OOM-prone KH-9 PC per-segment path last.
# See memory/machine_crash_2026_04_21.md for context.
configs=(
    bahrain_kh7_1967
    bahrain_kh4a_1965
    bahrain_kh4b_1968
    bahrain_kh9_pc_1977
)

: > "$STATUS"
echo "=== verify_profiles.sh: starting $(date -u +%FT%TZ) ===" | tee -a "$STATUS"

for cfg in "${configs[@]}"; do
    echo "--- $cfg --- $(date -u +%FT%TZ)" | tee -a "$STATUS"
    start=$(date +%s)
    # Pipe runner output to a per-config log; tee overall status.
    log="scripts/test/verify_${cfg}.log"
    if $RUNNER --config "$cfg" --timeout "$TIMEOUT" --cleanup > "$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    end=$(date +%s)
    echo "    exit=$rc  duration_s=$((end-start))  log=$log" | tee -a "$STATUS"
done

echo "=== verify_profiles.sh: finished $(date -u +%FT%TZ) ===" | tee -a "$STATUS"
