#!/usr/bin/env python3
"""Fast iteration harness for DIS / SEA-RAFT / WAFT flow parameter tuning.

Loads saved flow inputs (from SAVE_FLOW_INPUTS=1 pipeline run) and re-runs
flow estimation with configurable parameters.  ~30-60s per DIS test,
~2-5 min per SEA-RAFT test.

Usage:
    # DIS (default) with current MEDIUM settings
    python3 scripts/debug/debug_flow.py

    # DIS with custom params
    python3 scripts/debug/debug_flow.py --finest-scale 0 --var-refine-iter 25

    # SEA-RAFT
    python3 scripts/debug/debug_flow.py --method sea-raft --iters 20 --tile-size 1024

    # With preprocessing
    python3 scripts/debug/debug_flow.py --preblur 1.0
    python3 scripts/debug/debug_flow.py --rank-filter 31
    python3 scripts/debug/debug_flow.py --histogram-match
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from align.constants import FB_CONSISTENCY_PX
from align.image import to_u8_percentile


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def apply_preblur(img_u8: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian pre-blur to suppress film grain."""
    return cv2.GaussianBlur(img_u8, (0, 0), sigma)


def apply_rank_filter(img_u8: np.ndarray, window: int) -> np.ndarray:
    """Replace pixels with local median rank (invariant to monotonic transforms)."""
    from scipy.ndimage import median_filter
    return median_filter(img_u8, size=window).astype(np.uint8)


def apply_histogram_match(source_u8: np.ndarray, reference_u8: np.ndarray) -> np.ndarray:
    """Match histogram of source to reference."""
    from skimage.exposure import match_histograms
    return match_histograms(source_u8, reference_u8).astype(np.uint8)


def apply_clahe(img_u8: np.ndarray, clip_limit: float = 2.0,
                grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))
    return clahe.apply(img_u8)


def preprocess(img_u8: np.ndarray, ref_u8: np.ndarray, args) -> np.ndarray:
    """Apply configured preprocessing to a u8 image."""
    out = img_u8.copy()
    if args.clahe:
        out = apply_clahe(out)
    if args.histogram_match:
        out = apply_histogram_match(out, ref_u8)
    if args.rank_filter > 0:
        out = apply_rank_filter(out, args.rank_filter)
    if args.preblur > 0:
        out = apply_preblur(out, args.preblur)
    return out


# ---------------------------------------------------------------------------
# Flow estimators
# ---------------------------------------------------------------------------

def estimate_flow_dis(ref_gray: np.ndarray, warped_gray: np.ndarray,
                      finest_scale: int = 1, var_refine_iter: int = 5,
                      patch_size: int = 8, patch_stride: int = 3,
                      grad_descent_iter: int = 25,
                      alpha: float = 20.0) -> np.ndarray:
    """DIS optical flow with configurable parameters."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(finest_scale)
    dis.setVariationalRefinementIterations(var_refine_iter)
    dis.setPatchSize(patch_size)
    dis.setPatchStride(patch_stride)
    dis.setGradientDescentIterations(grad_descent_iter)
    dis.setVariationalRefinementAlpha(alpha)
    return dis.calc(ref_gray, warped_gray, None)


def estimate_flow_sea_raft(ref_gray: np.ndarray, warped_gray: np.ndarray,
                           iters: int = 20, tile_size: int = 1024,
                           overlap: int = 128) -> np.ndarray:
    """SEA-RAFT tiled flow estimation."""
    from align.flow_refine import (
        _get_raft_model, _raft_forward_single, _raft_forward_batch,
    )

    model, device = _get_raft_model()
    if model is None:
        raise RuntimeError("SEA-RAFT not available (no GPU?)")

    # Override iteration count
    if hasattr(model, '_ITERS'):
        model._ITERS = iters

    h, w = ref_gray.shape[:2]

    if h <= tile_size and w <= tile_size:
        return _raft_forward_single(model, device, ref_gray, warped_gray)

    # Tiled inference (same logic as _estimate_flow_raft)
    stride = tile_size - overlap
    flow_accum = np.zeros((h, w, 2), dtype=np.float64)
    weight_accum = np.zeros((h, w), dtype=np.float64)

    def _blend_weight_1d(n, ovlp):
        wt = np.ones(n, dtype=np.float64)
        ramp = np.linspace(0, 1, ovlp)
        wt[:ovlp] = ramp
        wt[-ovlp:] = np.minimum(wt[-ovlp:], ramp[::-1])
        return wt

    tiles_y = max(1, (h - overlap + stride - 1) // stride)
    tiles_x = max(1, (w - overlap + stride - 1) // stride)

    tile_specs = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = min(ty * stride, max(0, h - tile_size))
            x0 = min(tx * stride, max(0, w - tile_size))
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile_specs.append((y0, x0, y1, x1))

    batch_refs, batch_warps, batch_specs = [], [], []
    batch_size = 4

    def _flush():
        nonlocal batch_refs, batch_warps, batch_specs
        if not batch_refs:
            return
        flows = _raft_forward_batch(model, device, batch_refs, batch_warps)
        for (y0, x0, y1, x1), tile_flow in zip(batch_specs, flows):
            th, tw = y1 - y0, x1 - x0
            wy = _blend_weight_1d(th, min(overlap, th // 2))
            wx = _blend_weight_1d(tw, min(overlap, tw // 2))
            tile_weight = wy[:, None] * wx[None, :]
            flow_accum[y0:y1, x0:x1, 0] += tile_flow[:, :, 0] * tile_weight
            flow_accum[y0:y1, x0:x1, 1] += tile_flow[:, :, 1] * tile_weight
            weight_accum[y0:y1, x0:x1] += tile_weight
        batch_refs.clear()
        batch_warps.clear()
        batch_specs.clear()

    for y0, x0, y1, x1 in tile_specs:
        th, tw = y1 - y0, x1 - x0
        batch_refs.append(ref_gray[y0:y1, x0:x1])
        batch_warps.append(warped_gray[y0:y1, x0:x1])
        batch_specs.append((y0, x0, y1, x1))
        if len(batch_refs) >= batch_size:
            _flush()
    _flush()

    mask = weight_accum > 0
    flow_accum[mask, 0] /= weight_accum[mask]
    flow_accum[mask, 1] /= weight_accum[mask]
    return flow_accum.astype(np.float32)


# ---------------------------------------------------------------------------
# FB consistency
# ---------------------------------------------------------------------------

def forward_backward_error(ref_gray, warped_gray, flow_fwd, flow_fn, flow_kwargs):
    """Compute FB error. Returns (flow_bwd, err)."""
    flow_bwd = flow_fn(warped_gray, ref_gray, **flow_kwargs)
    h, w = flow_fwd.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    x_fwd = x_coords + flow_fwd[:, :, 0]
    y_fwd = y_coords + flow_fwd[:, :, 1]
    bwd_at_fwd = cv2.remap(flow_bwd, x_fwd, y_fwd,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    err = np.sqrt((flow_fwd[:, :, 0] + bwd_at_fwd[:, :, 0])**2 +
                  (flow_fwd[:, :, 1] + bwd_at_fwd[:, :, 1])**2)
    return flow_bwd, err


def clamp_flow_magnitude(flow, max_m, res_m):
    """Clamp flow magnitude. Returns magnitude array."""
    mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    max_px = max_m / res_m
    too_large = mag > max_px
    if np.any(too_large):
        clip = np.where(too_large, max_px / (mag + 1e-8), 1.0)
        flow[:, :, 0] *= clip
        flow[:, :, 1] *= clip
        mag = np.clip(mag, 0, max_px)
    return mag


# ---------------------------------------------------------------------------
# Main pipeline: coarse + fine
# ---------------------------------------------------------------------------

def run_flow_test(data, args):
    """Run the full coarse+fine flow pipeline with given parameters."""
    ref_arr = data['ref_arr_fr']
    warped_arr = data['warped_arr_fr']
    fr_work_res = float(data['fr_work_res'])
    coarse_res = float(data['coarse_res'])
    fr_w = int(data['fr_w'])
    fr_h = int(data['fr_h'])
    do_two_pass = bool(data['do_two_pass'])
    output_res = float(data['output_res'])

    # Build flow function + kwargs
    if args.method == 'dis':
        flow_fn = estimate_flow_dis
        flow_kwargs = dict(
            finest_scale=args.finest_scale,
            var_refine_iter=args.var_refine_iter,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            grad_descent_iter=args.grad_descent_iter,
            alpha=args.alpha,
        )
        method_label = (f"DIS (finestScale={args.finest_scale}, "
                        f"varRefine={args.var_refine_iter}, "
                        f"patchSize={args.patch_size}, alpha={args.alpha})")
    elif args.method == 'sea-raft':
        flow_fn = lambda r, w, **kw: estimate_flow_sea_raft(
            r, w, iters=args.iters, tile_size=args.tile_size, overlap=args.overlap)
        flow_kwargs = {}
        method_label = (f"SEA-RAFT (iters={args.iters}, "
                        f"tile={args.tile_size}, overlap={args.overlap})")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    t0 = time.time()

    # Convert to u8
    ref_u8 = to_u8_percentile(ref_arr)
    warped_u8 = to_u8_percentile(warped_arr)

    # Preprocessing
    preproc_parts = []
    if args.clahe:
        preproc_parts.append("clahe")
    if args.histogram_match:
        preproc_parts.append("hist_match")
    if args.rank_filter > 0:
        preproc_parts.append(f"rank_{args.rank_filter}")
    if args.preblur > 0:
        preproc_parts.append(f"blur_{args.preblur}")

    if preproc_parts:
        ref_u8 = preprocess(ref_u8, ref_u8, args)  # ref preprocessed against itself
        warped_u8 = preprocess(warped_u8, ref_u8, args)
        method_label += f" + {'+'.join(preproc_parts)}"

    coarse_w = max(64, int(fr_w * fr_work_res / coarse_res)) if do_two_pass else fr_w
    coarse_h = max(64, int(fr_h * fr_work_res / coarse_res)) if do_two_pass else fr_h

    # ---- Coarse pass ----
    flow_coarse_fine = np.zeros((fr_h, fr_w, 2), dtype=np.float32)
    coarse_reliable_pct = 0.0
    coarse_mean_m = 0.0
    coarse_max_m = 0.0
    t_coarse = 0.0

    if do_two_pass:
        t_c0 = time.time()
        ref_c = cv2.resize(ref_u8, (coarse_w, coarse_h), interpolation=cv2.INTER_AREA)
        warped_c = cv2.resize(warped_u8, (coarse_w, coarse_h), interpolation=cv2.INTER_AREA)

        flow_c = flow_fn(ref_c, warped_c, **flow_kwargs)

        # FB check
        _, fb_err_c = forward_backward_error(ref_c, warped_c, flow_c, flow_fn, flow_kwargs)
        fb_mask_c = fb_err_c < FB_CONSISTENCY_PX

        flow_c[:, :, 0] = cv2.medianBlur(flow_c[:, :, 0], 5)
        flow_c[:, :, 1] = cv2.medianBlur(flow_c[:, :, 1], 5)
        flow_c[~fb_mask_c] = 0

        valid_c = (cv2.resize(ref_arr, (coarse_w, coarse_h),
                              interpolation=cv2.INTER_AREA) > 0) & \
                  (cv2.resize(warped_arr, (coarse_w, coarse_h),
                              interpolation=cv2.INTER_AREA) > 0)
        flow_c[~valid_c] = 0

        flow_mag_c = clamp_flow_magnitude(flow_c, 75.0, coarse_res)

        coarse_reliable_pct = float(fb_mask_c.sum()) / max(1, fb_mask_c.size) * 100
        coarse_mean_m = (float(np.mean(flow_mag_c[fb_mask_c])) * coarse_res
                         if fb_mask_c.any() else 0.0)
        coarse_max_m = float(flow_mag_c.max()) * coarse_res

        # Upsample coarse to fine
        px_ratio = coarse_res / fr_work_res
        flow_coarse_fine[:, :, 0] = cv2.resize(
            flow_c[:, :, 0], (fr_w, fr_h), interpolation=cv2.INTER_LINEAR) * px_ratio
        flow_coarse_fine[:, :, 1] = cv2.resize(
            flow_c[:, :, 1], (fr_w, fr_h), interpolation=cv2.INTER_LINEAR) * px_ratio

        # Apply coarse correction to warped for fine pass
        ys_grid = np.arange(fr_h, dtype=np.float32)
        xs_grid = np.arange(fr_w, dtype=np.float32)
        xg, yg = np.meshgrid(xs_grid, ys_grid)
        map_x = (xg + flow_coarse_fine[:, :, 0]).astype(np.float32)
        map_y = (yg + flow_coarse_fine[:, :, 1]).astype(np.float32)
        warped_u8 = cv2.remap(warped_u8, map_x, map_y,
                              cv2.INTER_LINEAR, borderValue=0)

        t_coarse = time.time() - t_c0

    # ---- Fine pass ----
    t_f0 = time.time()
    flow_fine = flow_fn(ref_u8, warped_u8, **flow_kwargs)

    _, fb_err = forward_backward_error(ref_u8, warped_u8, flow_fine, flow_fn, flow_kwargs)
    fb_mask = fb_err < FB_CONSISTENCY_PX

    flow_fine[:, :, 0] = cv2.medianBlur(flow_fine[:, :, 0], 5)
    flow_fine[:, :, 1] = cv2.medianBlur(flow_fine[:, :, 1], 5)
    flow_fine[~fb_mask] = 0

    valid_fine = (ref_arr > 0) & (warped_arr > 0)
    flow_fine[~valid_fine] = 0

    fine_clamp_m = 30.0 if do_two_pass else 75.0
    flow_mag = clamp_flow_magnitude(flow_fine, fine_clamp_m, fr_work_res)

    fine_reliable_pct = float(fb_mask.sum()) / max(1, fb_mask.size) * 100
    fine_mean_m = (float(np.mean(flow_mag[fb_mask])) * fr_work_res
                   if fb_mask.any() else 0.0)
    fine_max_m = float(flow_mag.max()) * fr_work_res
    t_fine = time.time() - t_f0

    # Combined
    flow_total_x = flow_coarse_fine[:, :, 0] + flow_fine[:, :, 0]
    flow_total_y = flow_coarse_fine[:, :, 1] + flow_fine[:, :, 1]
    combined_clamp_m = 100.0 if do_two_pass else 75.0
    flow_mag_total = np.sqrt(flow_total_x**2 + flow_total_y**2)
    max_px_total = combined_clamp_m / fr_work_res
    too_large = flow_mag_total > max_px_total
    if np.any(too_large):
        clip_t = np.where(too_large, max_px_total / (flow_mag_total + 1e-8), 1.0)
        flow_total_x *= clip_t
        flow_total_y *= clip_t
        flow_mag_total = np.clip(flow_mag_total, 0, max_px_total)

    combined_mean_m = (float(np.mean(flow_mag_total[flow_mag_total > 0])) * fr_work_res
                       if np.any(flow_mag_total > 0) else 0.0)
    combined_max_m = float(flow_mag_total.max()) * fr_work_res

    total_time = time.time() - t0

    # Report
    print(f"\nMethod: {method_label}")
    print(f"Image: {fr_w}x{fr_h} @ {fr_work_res}m/px"
          f" (coarse {coarse_w}x{coarse_h} @ {coarse_res}m/px)")
    if do_two_pass:
        print(f"Coarse:   {coarse_reliable_pct:5.1f}% reliable | "
              f"mean {coarse_mean_m:6.1f}m | max {coarse_max_m:6.1f}m  "
              f"({t_coarse:.1f}s)")
    print(f"Fine:     {fine_reliable_pct:5.1f}% reliable | "
          f"mean {fine_mean_m:6.1f}m | max {fine_max_m:6.1f}m  "
          f"({t_fine:.1f}s)")
    print(f"Combined: mean {combined_mean_m:6.1f}m | max {combined_max_m:6.1f}m")
    print(f"Time: {total_time:.1f}s")

    return {
        'method': method_label,
        'coarse_reliable_pct': coarse_reliable_pct,
        'coarse_mean_m': coarse_mean_m,
        'coarse_max_m': coarse_max_m,
        'fine_reliable_pct': fine_reliable_pct,
        'fine_mean_m': fine_mean_m,
        'fine_max_m': fine_max_m,
        'combined_mean_m': combined_mean_m,
        'combined_max_m': combined_max_m,
        'time_s': total_time,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fast flow parameter tuning harness")
    parser.add_argument('--input', default='diagnostics/flow_inputs.npz',
                        help='Path to saved flow inputs')
    parser.add_argument('--method', choices=['dis', 'sea-raft'],
                        default='dis', help='Flow method')

    # DIS parameters
    parser.add_argument('--finest-scale', type=int, default=1)
    parser.add_argument('--var-refine-iter', type=int, default=5)
    parser.add_argument('--patch-size', type=int, default=8)
    parser.add_argument('--patch-stride', type=int, default=3)
    parser.add_argument('--grad-descent-iter', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=20.0)

    # SEA-RAFT parameters
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=128)

    # Preprocessing
    parser.add_argument('--preblur', type=float, default=0.0,
                        help='Gaussian blur sigma (0=off)')
    parser.add_argument('--rank-filter', type=int, default=0,
                        help='Rank filter window size (0=off)')
    parser.add_argument('--histogram-match', action='store_true',
                        help='Match warped histogram to reference')
    parser.add_argument('--clahe', action='store_true',
                        help='Apply CLAHE to both images')

    # Batch mode: run all DIS tests from Step 3
    parser.add_argument('--batch-dis', action='store_true',
                        help='Run all DIS parameter sweep tests')
    parser.add_argument('--batch-sea-raft', action='store_true',
                        help='Run all SEA-RAFT parameter sweep tests')

    args = parser.parse_args()

    # Load saved data
    print(f"Loading {args.input}...")
    data = np.load(args.input, allow_pickle=True)
    print(f"  ref: {data['ref_arr_fr'].shape}, warped: {data['warped_arr_fr'].shape}")
    print(f"  work_res={float(data['fr_work_res'])}m, "
          f"coarse_res={float(data['coarse_res'])}m, "
          f"two_pass={bool(data['do_two_pass'])}")

    if args.batch_dis:
        run_batch_dis(data, args)
    elif args.batch_sea_raft:
        run_batch_sea_raft(data, args)
    else:
        run_flow_test(data, args)


def run_batch_dis(data, base_args):
    """Run all DIS parameter sweep tests from Step 3."""
    tests = [
        ('baseline',  dict(finest_scale=1, var_refine_iter=5,  patch_size=8,  alpha=20.0, preblur=0.0)),
        ('A',         dict(finest_scale=0, var_refine_iter=5,  patch_size=8,  alpha=20.0, preblur=0.0)),
        ('B',         dict(finest_scale=1, var_refine_iter=25, patch_size=8,  alpha=20.0, preblur=0.0)),
        ('C',         dict(finest_scale=1, var_refine_iter=5,  patch_size=12, alpha=20.0, preblur=0.0)),
        ('D',         dict(finest_scale=0, var_refine_iter=25, patch_size=8,  alpha=20.0, preblur=0.0)),
        ('E',         dict(finest_scale=0, var_refine_iter=25, patch_size=12, alpha=40.0, preblur=0.0)),
        ('F',         dict(finest_scale=1, var_refine_iter=5,  patch_size=8,  alpha=20.0, preblur=1.0)),
    ]
    results = []
    for name, params in tests:
        print(f"\n{'='*60}")
        print(f"Test {name}: {params}")
        print(f"{'='*60}")
        args = argparse.Namespace(
            method='dis',
            finest_scale=params['finest_scale'],
            var_refine_iter=params['var_refine_iter'],
            patch_size=params['patch_size'],
            patch_stride=3,
            grad_descent_iter=25,
            alpha=params['alpha'],
            preblur=params['preblur'],
            rank_filter=0,
            histogram_match=False,
            clahe=False,
            iters=20, tile_size=1024, overlap=128,
        )
        result = run_flow_test(data, args)
        result['test'] = name
        results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("DIS Parameter Sweep Summary")
    print(f"{'='*80}")
    print(f"{'Test':<10} {'Coarse %':>9} {'Coarse m':>9} "
          f"{'Fine %':>7} {'Fine m':>7} "
          f"{'Combined m':>11} {'Time':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['test']:<10} {r['coarse_reliable_pct']:>8.1f}% "
              f"{r['coarse_mean_m']:>8.1f} "
              f"{r['fine_reliable_pct']:>6.1f}% {r['fine_mean_m']:>6.1f} "
              f"{r['combined_mean_m']:>10.1f} {r['time_s']:>5.1f}s")


def run_batch_sea_raft(data, base_args):
    """Run all SEA-RAFT parameter sweep tests from Step 4."""
    tests = [
        ('SR-base', dict(iters=20, tile_size=1024, overlap=128,
                         preblur=0.0, rank_filter=0, histogram_match=False)),
        ('SR-A',    dict(iters=12, tile_size=1024, overlap=128,
                         preblur=0.0, rank_filter=0, histogram_match=False)),
        ('SR-B',    dict(iters=20, tile_size=512,  overlap=128,
                         preblur=0.0, rank_filter=0, histogram_match=False)),
        ('SR-C',    dict(iters=32, tile_size=1024, overlap=128,
                         preblur=0.0, rank_filter=0, histogram_match=False)),
        ('SR-D',    dict(iters=20, tile_size=1024, overlap=128,
                         preblur=1.0, rank_filter=0, histogram_match=False)),
        ('SR-E',    dict(iters=20, tile_size=1024, overlap=128,
                         preblur=0.0, rank_filter=31, histogram_match=False)),
        ('SR-F',    dict(iters=20, tile_size=1024, overlap=128,
                         preblur=0.0, rank_filter=0, histogram_match=True)),
    ]
    results = []
    for name, params in tests:
        print(f"\n{'='*60}")
        print(f"Test {name}: {params}")
        print(f"{'='*60}")
        args = argparse.Namespace(
            method='sea-raft',
            finest_scale=1, var_refine_iter=5, patch_size=8,
            patch_stride=3, grad_descent_iter=25, alpha=20.0,
            iters=params['iters'],
            tile_size=params['tile_size'],
            overlap=params['overlap'],
            preblur=params['preblur'],
            rank_filter=params['rank_filter'],
            histogram_match=params['histogram_match'],
            clahe=False,
        )
        result = run_flow_test(data, args)
        result['test'] = name
        results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("SEA-RAFT Parameter Sweep Summary")
    print(f"{'='*80}")
    print(f"{'Test':<10} {'Coarse %':>9} {'Coarse m':>9} "
          f"{'Fine %':>7} {'Fine m':>7} "
          f"{'Combined m':>11} {'Time':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['test']:<10} {r['coarse_reliable_pct']:>8.1f}% "
              f"{r['coarse_mean_m']:>8.1f} "
              f"{r['fine_reliable_pct']:>6.1f}% {r['fine_mean_m']:>6.1f} "
              f"{r['combined_mean_m']:>10.1f} {r['time_s']:>5.1f}s")


if __name__ == '__main__':
    main()
