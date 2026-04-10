# Fast Piecewise Tests

These tests are designed to verify the KH-4 pipeline changes without running a
multi-hour end-to-end alignment.

Stages:

- `process`: cached-scene preprocessing and ASP-ortho backfill
- `manifest`: alignment input preference and crop invalidation
- `qa`: post-warp regression checks and geometry-only QA behavior
- `selection`: coarse-localization tagging, affine-vs-grid selection, and
  synthetic image-accuracy checks for scale/rotation/match refinement

Run all fast tests:

```bash
/Users/mish/Code/declass-process/.venv-hb/bin/python -m pytest tests -q
```

Run one subset:

```bash
/Users/mish/Code/declass-process/.venv-hb/bin/python scripts/test/run_piecewise.py qa
/Users/mish/Code/declass-process/.venv-hb/bin/python scripts/test/run_piecewise.py process manifest
```

Harness artifacts:

- default run dir: `diagnostics/test_runs/piecewise_latest/`
- persistent files: `run.log`, `junit.xml`, `summary.json`, `piecewise_results.json`
- per-test artifacts: `artifacts/<stage>/<sanitized-nodeid>/case_summary.json`

Useful runner flags:

```bash
/Users/mish/Code/declass-process/.venv-hb/bin/python scripts/test/run_piecewise.py \
  --run-id piecewise_qa \
  --pytest-args -q
```
