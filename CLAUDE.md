# Declass-Process — Satellite Image Alignment Pipeline

General-purpose pipeline for aligning declassified KH satellite imagery (1965–1984, global coverage) to modern georeferenced basemaps. The pipeline handles coarse offset estimation, scale/rotation correction, neural feature matching, grid-based optimization, optical flow refinement, and QA scoring.

Bahrain is one test case. The pipeline must work worldwide across all KH camera systems (KH-4, KH-7, KH-9) and geographies.

## Autonomous Iteration Loop

When asked to improve alignment quality, follow this loop:

1. **Run the test:** `python3 scripts/test/run_test.py` — run in background (~20-40 min)
2. **Read results:** When notified, read `diagnostics/run_vN/summary.json`
3. **Compare versions:** `python3 scripts/test/compare.py --format json` for cross-version analysis
4. **Analyze:** Identify what improved, what regressed, and what the biggest bottlenecks are (check regional errors, IoU metrics, patch_med).
5. **Change code:** Make targeted changes in `align/*.py`
6. **Repeat**

## Generalisability Mandate

Every code change MUST work across:
- **Geographies:** coastal, desert, mountainous, urban, rural
- **Eras:** 1965–1984 — feature detectors must handle era-specific film grain, contrast, and artifacts
- **Camera systems:** KH-4, KH-7, KH-9 — different resolutions, distortion profiles, scan patterns
- **No test-case-specific hardcoding.** Bahrain is one test image, not the product.

## Literature Search Protocol

When metrics plateau across 2+ iterations:
- Web search for SOTA in the relevant sub-problem (e.g. "cross-temporal satellite image registration", "optical flow for historical imagery", "learned feature matching under radiometric change")
- Check for recent papers (2016–2026) on the specific bottleneck
- Consider techniques from adjacent fields (medical image registration, structure-from-motion, GIS, image reconstruction, etc.)
- Document findings and rationale in the summary before implementing

## GPU Optimization Principles

- Always check if a CPU-bound step can move to MPS/CUDA
- Batch operations where possible
- Profile before optimizing — don't sacrifice accuracy for speed without evidence
- Use `torch.no_grad()` for inference, `torch.cuda.amp` / mixed precision where applicable
- Prefer vectorized NumPy/PyTorch over Python loops for geometric transforms

## Key Architecture

- Entry: `auto-align.py` → `align/pipeline.py` → step functions
- Grid optimizer: `align/grid_optim.py` — affine baseline + learnable residual, hierarchical levels [(8,200), (24,200), (64,200)]
- Flow refinement is critical — never disable it (patch_med 18m vs 66m without)
- Chamfer loss uses reclamation-aware masking (XOR land masks, dilated 200m)
- Cross-validation threshold is the main blocker for `accepted=True`
- Reference image has ~30m internal distortion — hard floor on achievable accuracy

## What NOT to Do

- Don't add test-image-specific special cases (Bahrain or otherwise)
- Don't disable flow refinement
- Don't skip the grid optimizer in favor of TPS-only warping
- Don't make changes that increase wall clock >50% without proportional accuracy gain
