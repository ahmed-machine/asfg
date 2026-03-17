#!/usr/bin/env python3
"""Standalone ELoFTR integration test using real Bahrain satellite data.

Computes a coarse offset first (land-mask template matching), shifts the offset
image, then runs ELoFTR.  Saves a match visualization to diagnostics/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import rasterio
import torch
import cv2

TARGET = "/Users/mish/Dropbox/mapbh/Maps/Declassified Sats/1977-08-27 - Bahrain - D3C1213-200346A003.tif"
REFERENCE = "/Users/mish/Code/openmaps/public/maps/1976-KH9-DZB1212.warped.tif"
VIS_DIR = Path(__file__).resolve().parent.parent / "diagnostics" / "eloftr_test"


def load_overlap():
    """Load real overlap region at detection_res=5.0 m/px and compute coarse offset."""
    from align.geo import get_metric_crs, compute_overlap, read_overlap_region
    from align.coarse import detect_offset_at_resolution

    src_off = rasterio.open(TARGET)
    src_ref = rasterio.open(REFERENCE)
    work_crs = get_metric_crs(src_off, src_ref)
    overlap = compute_overlap(src_off, src_ref, work_crs)

    detection_res = 5.0
    arr_ref, _ = read_overlap_region(src_ref, overlap, work_crs, detection_res)
    arr_off, _ = read_overlap_region(src_off, overlap, work_crs, detection_res)

    print(f"  Overlap: {arr_ref.shape[1]}x{arr_ref.shape[0]} px at {detection_res} m/px")

    # Two-pass coarse offset (same as pipeline: 15m scan → 5m refinement)
    print("  Computing coarse offset (15m pass)...")
    dx_c, dy_c, corr_c = detect_offset_at_resolution(
        src_off, src_ref, overlap, work_crs, 15.0,
        template_radius_m=6000)

    dx_m, dy_m = 0.0, 0.0
    if dx_c is not None:
        print(f"    15m: dx={dx_c:.1f}m dy={dy_c:.1f}m (corr={corr_c:.3f})")
        print("  Refining at 5m...")
        dx_r, dy_r, corr_r = detect_offset_at_resolution(
            src_off, src_ref, overlap, work_crs, detection_res,
            template_radius_m=6000, coarse_offset=(dx_c, dy_c),
            search_margin_m=300)
        if dx_r is not None:
            dx_m, dy_m = dx_r, dy_r
            print(f"    5m: dx={dx_m:.1f}m dy={dy_m:.1f}m (corr={corr_r:.3f})")
        else:
            dx_m, dy_m = dx_c, dy_c
            print(f"    5m refinement failed — using 15m estimate")
    else:
        print("  Coarse offset failed — using zero shift")

    # Shift offset image to align with reference
    shift_px_x = int(round(dx_m / detection_res))
    shift_px_y = int(round(dy_m / detection_res))
    h, w = arr_off.shape
    M_shift = np.array([[1.0, 0.0, float(-shift_px_x)],
                         [0.0, 1.0, float(-shift_px_y)]], dtype=np.float32)
    arr_off_shifted = cv2.warpAffine(arr_off, M_shift, (w, h),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    print(f"  Applied shift: {-shift_px_x}, {-shift_px_y} px")

    src_off.close()
    src_ref.close()
    return arr_ref, arr_off_shifted


def save_match_vis(ref_u8, off_u8, kp0, kp1, conf, name, n_per_tier=5):
    """Save a side-by-side visualization showing 5 top, 5 medium, 5 weak matches."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    h, w = ref_u8.shape
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
    canvas[:, w:] = cv2.cvtColor(off_u8, cv2.COLOR_GRAY2BGR)

    if len(conf) == 0:
        out_path = VIS_DIR / f"{name}.jpg"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Saved: {out_path} (no matches)")
        return

    # Sort all matches by confidence
    order = np.argsort(conf)
    kp0_s, kp1_s, conf_s = kp0[order], kp1[order], conf[order]
    n = len(conf_s)

    # Pick 5 from each tier: top, middle, weak
    top_idx = np.linspace(n - 1, max(n - n * 0.1, n - n_per_tier), n_per_tier, dtype=int)
    mid_start = max(0, n // 2 - n_per_tier // 2)
    mid_idx = np.arange(mid_start, min(n, mid_start + n_per_tier))
    weak_idx = np.linspace(0, min(n * 0.1, n_per_tier), n_per_tier, dtype=int)

    # BGR colors: green=top, yellow=medium, red=weak
    tiers = [
        (weak_idx, (0, 0, 220), "weak"),
        (mid_idx,  (0, 180, 220), "medium"),
        (top_idx,  (0, 220, 0), "top"),
    ]

    drawn = 0
    for indices, color, label in tiers:
        for i in np.unique(indices):
            if i >= n:
                continue
            pt0 = (int(kp0_s[i, 0]), int(kp0_s[i, 1]))
            pt1 = (int(kp1_s[i, 0]) + w, int(kp1_s[i, 1]))
            cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt0, 4, color, -1)
            cv2.circle(canvas, pt1, 4, color, -1)
            # Label confidence near the left keypoint
            cv2.putText(canvas, f"{conf_s[i]:.2f}", (pt0[0] + 6, pt0[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            drawn += 1

    # Legend
    y0 = 20
    for color, label in [((0, 220, 0), "top 5"), ((0, 180, 220), "mid 5"), ((0, 0, 220), "weak 5")]:
        cv2.line(canvas, (10, y0), (30, y0), color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (35, y0 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y0 += 18

    out_path = VIS_DIR / f"{name}.jpg"
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Saved: {out_path} ({drawn} matches drawn)")


def save_side_by_side(ref_u8, off_u8, name):
    """Save a plain side-by-side comparison (no matches drawn)."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    h, w = ref_u8.shape
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
    canvas[:, w:] = cv2.cvtColor(off_u8, cv2.COLOR_GRAY2BGR)

    # Draw thin separator line
    cv2.line(canvas, (w, 0), (w, h - 1), (80, 80, 80), 1)

    out_path = VIS_DIR / f"{name}.jpg"
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Saved: {out_path}")


def test_model_load(cache):
    """Test 1: Model loads and is in eval mode."""
    print("Test 1: Model load ...", end=" ", flush=True)
    t0 = time.time()
    model = cache.eloftr
    elapsed = time.time() - t0
    assert model is not None, "model is None"
    assert not model.training, "model not in eval mode"
    print(f"OK ({elapsed:.1f}s)")
    return model


def test_forward_pass(model, arr_ref, arr_off, device):
    """Test 2: Forward pass on coarsely-aligned data produces valid matches."""
    print("Test 2: Forward pass on real data ...", end=" ", flush=True)
    from align.image import clahe_normalize

    ref_u8 = clahe_normalize(arr_ref)
    off_u8 = clahe_normalize(arr_off)

    # Centre crop to 800x800
    h, w = ref_u8.shape
    ch, cw = min(800, h), min(800, w)
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    ref_crop = ref_u8[r0:r0+ch, c0:c0+cw]
    off_crop = off_u8[r0:r0+ch, c0:c0+cw]

    ref_t = torch.from_numpy(ref_crop).float()[None, None] / 255.0
    off_t = torch.from_numpy(off_crop).float()[None, None] / 255.0

    out = model({"image0": ref_t.to(device), "image1": off_t.to(device)})

    for key in ("mkpts0_f", "mkpts1_f", "mconf", "m_bids"):
        assert key in out, f"missing key: {key}"

    n_matches = len(out["mkpts0_f"])
    assert n_matches > 0, "no matches on real satellite data"

    kp0 = out["mkpts0_f"].cpu().numpy()
    kp1 = out["mkpts1_f"].cpu().numpy()
    conf = out["mconf"].cpu().numpy()
    assert kp0[:, 0].max() <= cw + 1, "kpts0 x out of bounds"
    assert kp0[:, 1].max() <= ch + 1, "kpts0 y out of bounds"
    assert kp1[:, 0].max() <= cw + 1, "kpts1 x out of bounds"
    assert kp1[:, 1].max() <= ch + 1, "kpts1 y out of bounds"

    high_conf = (conf > 0.5).sum()
    print(f"OK ({n_matches} matches, {high_conf} high-conf)")

    # Save visualizations
    save_side_by_side(ref_crop, off_crop, "center_crop_side_by_side")
    save_match_vis(ref_crop, off_crop, kp0, kp1, conf, "center_crop_matches")


def test_batch_forward(model, arr_ref, arr_off, device):
    """Test 3: Batch forward pass (B=2) with correct m_bids and consistent matches."""
    print("Test 3: Batch forward pass ...", end=" ", flush=True)
    from align.image import clahe_normalize

    ref_u8 = clahe_normalize(arr_ref)
    off_u8 = clahe_normalize(arr_off)

    h, w = ref_u8.shape
    ch, cw = min(800, h), min(800, w)
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    ref_crop = ref_u8[r0:r0+ch, c0:c0+cw]
    off_crop = off_u8[r0:r0+ch, c0:c0+cw]

    ref_t = torch.from_numpy(ref_crop).float()[None, None] / 255.0
    off_t = torch.from_numpy(off_crop).float()[None, None] / 255.0

    # Duplicate to B=2
    ref_batch = torch.cat([ref_t, ref_t], dim=0).to(device)
    off_batch = torch.cat([off_t, off_t], dim=0).to(device)

    out = model({"image0": ref_batch, "image1": off_batch})

    bids = out["m_bids"].cpu().numpy()
    unique_bids = set(bids.tolist())
    assert 0 in unique_bids, "batch index 0 missing"
    assert 1 in unique_bids, "batch index 1 missing"

    n0 = (bids == 0).sum()
    n1 = (bids == 1).sum()
    assert n0 == n1, f"identical inputs produced different match counts: {n0} vs {n1}"

    kp0_b0 = out["mkpts0_f"][bids == 0].cpu().numpy()
    kp0_b1 = out["mkpts0_f"][bids == 1].cpu().numpy()
    assert np.allclose(kp0_b0, kp0_b1, atol=0.1), "identical inputs produced different keypoints"

    print("OK")


def test_detect_eloftr_scale(arr_ref, arr_off, model):
    """Test 4: detect_eloftr_scale on coarsely-aligned images."""
    print("Test 4: detect_eloftr_scale ...", end=" ", flush=True)
    from align.scale import detect_eloftr_scale, _gate_neural

    result = detect_eloftr_scale(arr_ref, arr_off, model=model)

    if result is not None:
        sx, sy, rot, n_inliers = result
        gated = _gate_neural(result)
        print(f"OK (scale={((sx+sy)/2):.4f}, rot={rot:.2f}\u00b0, "
              f"{n_inliers} inliers, gated={gated})")
    else:
        # Still OK — verify model produced matches
        from align.image import clahe_normalize
        from align.scale import _get_matcher_crops, _run_eloftr_batch
        from align.models import get_torch_device

        ref_u8 = clahe_normalize(arr_ref)
        off_u8 = clahe_normalize(arr_off)
        crops = _get_matcher_crops(ref_u8, off_u8)
        device = get_torch_device()
        results = _run_eloftr_batch(crops, model, device, batch_size=4)
        total_matches = sum(len(kp0) for kp0, _, _ in results)
        assert total_matches > 0, "no matches produced from any crop"
        print(f"OK (affine=None, but {total_matches} raw matches)")


def test_run_eloftr_batch_crops(arr_ref, arr_off, model, device):
    """Test 5: _run_eloftr_batch on real 3x3 crop grid with visualization."""
    print("Test 5: _run_eloftr_batch crops ...", end=" ", flush=True)
    from align.image import clahe_normalize
    from align.scale import _get_matcher_crops, _run_eloftr_batch

    ref_u8 = clahe_normalize(arr_ref)
    off_u8 = clahe_normalize(arr_off)

    crops = _get_matcher_crops(ref_u8, off_u8)
    assert len(crops) > 0, "no crops generated"

    results = _run_eloftr_batch(crops, model, device, batch_size=4)
    assert len(results) == len(crops), "result count mismatch"

    nonempty = sum(1 for kp0, kp1, conf in results if len(kp0) > 0)
    total_matches = sum(len(kp0) for kp0, kp1, conf in results)
    assert nonempty > 0, "all crops returned empty matches"

    for kp0, kp1, conf in results:
        if len(kp0) > 0:
            assert kp0.shape[1] == 2, f"kp0 shape wrong: {kp0.shape}"
            assert kp1.shape[1] == 2, f"kp1 shape wrong: {kp1.shape}"
            assert len(kp0) == len(kp1) == len(conf), "length mismatch"

    print(f"OK ({total_matches} matches from {nonempty}/{len(crops)} crops)")

    # Save per-crop visualizations for crops with matches
    for i, (kp0, kp1, conf) in enumerate(results):
        if len(kp0) > 0:
            save_match_vis(crops[i]['ref'], crops[i]['off'],
                           kp0, kp1, conf, f"crop_{i}_matches")


def main():
    from align.models import ModelCache, get_torch_device

    print("Loading test images...")
    arr_ref, arr_off = load_overlap()

    device = get_torch_device()
    cache = ModelCache(device)

    try:
        model = test_model_load(cache)
        test_forward_pass(model, arr_ref, arr_off, device)
        test_batch_forward(model, arr_ref, arr_off, device)
        test_detect_eloftr_scale(arr_ref, arr_off, model)
        test_run_eloftr_batch_crops(arr_ref, arr_off, model, device)
    finally:
        cache.close()

    print(f"\nAll tests passed. Visualizations in {VIS_DIR}")


if __name__ == "__main__":
    main()
