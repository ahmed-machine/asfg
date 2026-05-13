# LoRA Fine-Tune of RoMa v2 (Experimental)

End-to-end pipeline for adapting the DINOv3 ViT-L/16 backbone inside
RoMa v2 to cross-temporal KH ↔ modern satellite-imagery matching. See
`/Users/mish/Code/declass-process/lora-plan.md` for the architectural
analysis and `/Users/mish/.claude/plans/ultrathink-i-d-like-to-rustling-nebula.md`
for the implementation plan + verification gates.

## Status

- **Phase 1 — feature gap diagnostic**: complete. Decision: PROCEED with r=8.
  See `diagnostics/lora/feature_gap.json`. Median cos sim at layer 11 = 0.738
  (KH-4B = 0.718, the dominant mission has the largest gap).
- **Phases 2-4 — code complete**, awaiting end-to-end run.
- **Phase 5 — eval harness ready** (synthetic + iterate_phase + KH-9 regression).

## KH-9 layers are IN training (per user direction)

The `1976-KH9-DZB1212` (KH-9 MC) and `1982-D3C1217` (KH-9 PC) mapbh layers
are deliberately included in training. The KH-9 production-scene check
(`D3C1213-200346A003`, 1977 Bahrain) is therefore a **no-regression check**
rather than a held-out distribution check. See plan §R4.

## Verification chain

```bash
# G1 — feature gap diagnostic (~7 min on MPS for 200 pairs)
poetry run python scripts/experimental/lora/measure_feature_gap.py
# → diagnostics/lora/feature_gap.json
# Decision: median ∈ [0.5, 0.8) → r=8;  ∈ [0, 0.5) → r=16;  ≥ 0.8 → STOP

# G2 — mapbh client smoke (offline, ~5 s)
poetry run pytest tests/test_mapbh_client.py -v

# G3 — mapbh pair build (~1-2 hours overnight; mostly bandwidth-bound)
poetry run python scripts/experimental/lora/build_mapbh_pairs.py \
    --output data/lora_pairs --target-pairs 1500

# G3a — teacher pseudo-label extraction (~30-45 min on MPS for 1500 pairs)
poetry run python scripts/experimental/lora/extract_mapbh_pseudo_labels.py \
    --pairs-dir data/lora_pairs

# G4 — LoRA inject unit tests (offline, ~2 s)
poetry run pytest tests/test_lora_inject.py -v

# G5 — training run (4-8 h on MPS, 2-4 h on CUDA)
poetry run python scripts/experimental/lora/finetune.py \
    --rank 8 --alpha 8 --epochs 4
# → align/weights/roma_ssd.pth (LoRA-merged)
# → diagnostics/lora_train/<run-id>/metrics.jsonl

# G6 — synthetic eval A/B (~5-10 min)
poetry run python scripts/experimental/lora/eval_synthetic.py \
    --weights-a base \
    --weights-b align/weights/roma_ssd.pth

# G7 — production iterate_phase A/B on KH-4B (~15-25 min)
DECLASS_ROMA_WEIGHTS=align/weights/roma_ssd.pth \
  poetry run python scripts/test/iterate_phase.py \
    --manifest diagnostics/kh4b_ds1104_da023_to_1976_kh9/manifests/alignment_manifest.json \
    --scene DS1104-1057DA023 --from-phase post_scale_rotation --label lora_r8_v1

# G8 — KH-9 no-regression check (~40 min)
DECLASS_ROMA_WEIGHTS=align/weights/roma_ssd.pth \
  poetry run python scripts/test/run_test.py \
    --config scripts/test/e2e_configs/bahrain_kh9_pc_1977.yaml

# G9 — promote (only if G6 + G7 + G8 pass): set
#   matching.roma_weights_override: align/weights/roma_ssd.pth
# in data/profiles/{kh4,kh4a,kh4b,kh7}.yaml. Leave kh9.yaml unset.
```

## Files

- `measure_feature_gap.py` — Phase 1 cosine-similarity diagnostic at satast resolution
- `build_mapbh_pairs.py` — Phase 2.2 cross-temporal pair generator from mapbh.org
- `extract_mapbh_pseudo_labels.py` — Phase 2.4 teacher-label extractor
- `finetune.py` — Phase 4 trainer (LoRA injection + curriculum + bidirectional loss)
- `eval_synthetic.py` — Phase 5.1 fast A/B harness on held-out CORONA pairs
- `run_all.sh` — orchestrator for the full chain

## Module dependencies

- `align/romav2/lora.py` — LoRA injection / merge / I/O
- `align/romav2/lora_io.py` — merge + save helpers
- `preprocess/mapbh/{client,layers}.py` — TileServer GL XYZ fetcher
- `align/models.py:140-156` — `DECLASS_ROMA_WEIGHTS` env var override

## Resolution mismatch (R9)

Training runs at satast (800×800, bidirectional). Production uses fast
(512×512, unidirectional). LoRA params are nominally resolution-agnostic
but were optimised against the 800px feature distribution. If G7 shows
no lift but G6 does, retrain with `apply_setting("fast")` or accept the
fidelity cost.
