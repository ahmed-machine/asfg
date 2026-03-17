"""Dense optical flow post-refinement using RAFT or DIS.

After grid-based warping brings images within ~50-100m, dense optical flow
gives pixel-level displacement corrections.  Forward-backward consistency
checks reject unreliable flow in changed areas (water, clouds, new buildings).

RAFT (torchvision) is preferred over DIS for cross-temporal satellite imagery:
- Learned features handle radiometric/semantic differences between eras
- Better accuracy on large displacements and texture-poor regions
- Forward-backward consistency still used for reliability masking
- Falls back to DIS if RAFT unavailable (no GPU, import failure, etc.)
"""

import gc

import cv2
import numpy as np
import rasterio
import rasterio.transform
import torch
from rasterio.warp import reproject, Resampling

from .constants import MAX_FLOW_BIAS_M as _DEFAULT_MAX_BIAS_M
from .image import chunked_remap, to_u8_percentile

# ---------------------------------------------------------------------------
# RAFT (torchvision) — preferred flow estimator
# ---------------------------------------------------------------------------
_RAFT_MODEL = None
_RAFT_DEVICE = None
_RAFT_AVAILABLE = None  # None = not yet checked


def _get_raft_model():
    """Load RAFT model singleton. Returns (model, device) or (None, None).

    Tries CUDA first, then MPS, skips CPU (too slow for production images).
    On MPS, does a small forward pass to verify operator support.
    """
    global _RAFT_MODEL, _RAFT_DEVICE, _RAFT_AVAILABLE
    if _RAFT_AVAILABLE is False:
        return None, None
    if _RAFT_MODEL is not None:
        return _RAFT_MODEL, _RAFT_DEVICE
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            print("  [RAFT] Skipped (no GPU available, DIS fallback)", flush=True)
            _RAFT_AVAILABLE = False
            return None, None

        # Force garbage collection to free GPU memory from previous models
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f"  [RAFT] Loading RAFT-Large model on {device}...", flush=True)
        model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device)
        model.eval()

        # Smoke test on MPS — some ops may not be supported
        if device.type == 'mps':
            try:
                dummy = torch.randn(1, 3, 128, 128, device=device)
                with torch.no_grad():
                    model(dummy, dummy)
                print(f"  [RAFT] MPS smoke test passed", flush=True)
            except Exception as e:
                print(f"  [RAFT] MPS not supported ({e}), falling back to DIS", flush=True)
                del model
                _RAFT_AVAILABLE = False
                return None, None

        _RAFT_MODEL = model
        _RAFT_DEVICE = device
        _RAFT_AVAILABLE = True
        print(f"  [RAFT] Model ready on {device}", flush=True)
        return model, device
    except Exception as e:
        print(f"  [RAFT] Not available ({e}), falling back to DIS", flush=True)
        _RAFT_AVAILABLE = False
        return None, None


def _raft_forward_single(model, device, ref_gray: np.ndarray, warped_gray: np.ndarray) -> np.ndarray:
    """Run RAFT on a single (small enough) image pair. Returns flow (H, W, 2)."""
    h, w = ref_gray.shape[:2]

    def _prepare(gray):
        rgb = np.stack([gray, gray, gray], axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).unsqueeze(0).to(device)

    # Pad to multiple of 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        ref_padded = np.pad(ref_gray, ((0, pad_h), (0, pad_w)), mode='reflect')
        warped_padded = np.pad(warped_gray, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        ref_padded = ref_gray
        warped_padded = warped_gray

    ref_t = _prepare(ref_padded)
    warped_t = _prepare(warped_padded)

    with torch.no_grad():
        flow_preds = model(ref_t, warped_t)
        flow = flow_preds[-1]  # final iteration, (1, 2, H, W)

    flow_np = flow.squeeze(0).cpu().numpy()  # (2, H, W)
    flow_np = np.transpose(flow_np, (1, 2, 0))  # (H, W, 2)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        flow_np = flow_np[:h, :w]

    return flow_np


def _raft_forward_batch(
    model, device,
    ref_tiles: list[np.ndarray],
    warped_tiles: list[np.ndarray],
) -> list[np.ndarray]:
    """Run RAFT on a batch of same-sized tile pairs.

    All tiles must have the same dimensions. Returns list of flow arrays (H, W, 2).
    """
    if not ref_tiles:
        return []

    batch_size = len(ref_tiles)
    h, w = ref_tiles[0].shape[:2]

    # Pad to multiple of 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    def _prepare_batch(tiles):
        tensors = []
        for gray in tiles:
            if pad_h > 0 or pad_w > 0:
                gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='reflect')
            rgb = np.stack([gray, gray, gray], axis=0).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(rgb))
        return torch.stack(tensors).to(device)

    ref_batch = _prepare_batch(ref_tiles)
    warped_batch = _prepare_batch(warped_tiles)

    with torch.no_grad():
        flow_preds = model(ref_batch, warped_batch)
        flow = flow_preds[-1]  # (B, 2, H, W)

    results = []
    for i in range(batch_size):
        flow_np = flow[i].cpu().numpy()  # (2, H, W)
        flow_np = np.transpose(flow_np, (1, 2, 0))  # (H, W, 2)
        if pad_h > 0 or pad_w > 0:
            flow_np = flow_np[:ref_tiles[i].shape[0], :ref_tiles[i].shape[1]]
        results.append(flow_np)

    return results


# Max tile size for RAFT — keeps MPS/CUDA memory under ~4 GiB per tile
_RAFT_TILE_SIZE = 1024
_RAFT_TILE_OVERLAP = 128
_RAFT_BATCH_SIZE = 4


def _estimate_flow_raft(ref_gray: np.ndarray, warped_gray: np.ndarray,
                        batch_size: int = _RAFT_BATCH_SIZE) -> np.ndarray:
    """Compute dense optical flow using torchvision RAFT.

    Returns flow (H, W, 2) in pixels: flow[y, x] = (dx, dy).
    For large images, processes in overlapping tiles to manage GPU memory.
    Tiles are batched for throughput (OOM falls back to batch_size=1, then DIS).
    """
    model, device = _get_raft_model()
    if model is None:
        return _estimate_flow_dis(ref_gray, warped_gray)

    h, w = ref_gray.shape[:2]

    if h < 128 or w < 128:
        return _estimate_flow_dis(ref_gray, warped_gray)

    # Small enough for a single forward pass
    if h <= _RAFT_TILE_SIZE and w <= _RAFT_TILE_SIZE:
        try:
            return _raft_forward_single(model, device, ref_gray, warped_gray)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [RAFT] OOM on {w}x{h}, falling back to DIS", flush=True)
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
                return _estimate_flow_dis(ref_gray, warped_gray)
            raise

    # Tile-based processing for large images
    tile_size = _RAFT_TILE_SIZE
    overlap = _RAFT_TILE_OVERLAP
    stride = tile_size - overlap

    flow_accum = np.zeros((h, w, 2), dtype=np.float64)
    weight_accum = np.zeros((h, w), dtype=np.float64)

    # Build 1D cosine blending ramp
    def _blend_weight_1d(n, ovlp):
        w = np.ones(n, dtype=np.float64)
        ramp = np.linspace(0, 1, ovlp)
        w[:ovlp] = ramp
        w[-ovlp:] = np.minimum(w[-ovlp:], ramp[::-1])
        return w

    tiles_y = max(1, (h - overlap + stride - 1) // stride)
    tiles_x = max(1, (w - overlap + stride - 1) // stride)
    total_tiles = tiles_y * tiles_x

    # Collect tile coordinates
    tile_specs = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = min(ty * stride, max(0, h - tile_size))
            x0 = min(tx * stride, max(0, w - tile_size))
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile_specs.append((y0, x0, y1, x1))

    # Process in batches
    done = 0
    raft_batch_refs = []
    raft_batch_warps = []
    raft_batch_specs = []
    effective_batch = batch_size

    def _flush_batch():
        nonlocal done, raft_batch_refs, raft_batch_warps, raft_batch_specs, effective_batch
        if not raft_batch_refs:
            return
        try:
            flows = _raft_forward_batch(model, device, raft_batch_refs, raft_batch_warps)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and effective_batch > 1:
                print(f"  [RAFT] OOM on batch of {len(raft_batch_refs)}, retrying individually", flush=True)
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
                effective_batch = 1
                flows = [_raft_forward_single(model, device, r, w)
                         for r, w in zip(raft_batch_refs, raft_batch_warps)]
            elif "out of memory" in str(e).lower():
                print(f"  [RAFT] OOM on single tile, falling back to DIS", flush=True)
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
                flows = [_estimate_flow_dis(r, w)
                         for r, w in zip(raft_batch_refs, raft_batch_warps)]
            else:
                raise

        for (y0, x0, y1, x1), tile_flow in zip(raft_batch_specs, flows):
            th, tw = y1 - y0, x1 - x0
            wy = _blend_weight_1d(th, min(overlap, th // 2))
            wx = _blend_weight_1d(tw, min(overlap, tw // 2))
            tile_weight = wy[:, None] * wx[None, :]
            flow_accum[y0:y1, x0:x1, 0] += tile_flow[:, :, 0] * tile_weight
            flow_accum[y0:y1, x0:x1, 1] += tile_flow[:, :, 1] * tile_weight
            weight_accum[y0:y1, x0:x1] += tile_weight
            done += 1

        raft_batch_refs.clear()
        raft_batch_warps.clear()
        raft_batch_specs.clear()

    for y0, x0, y1, x1 in tile_specs:
        th, tw = y1 - y0, x1 - x0
        if th < 128 or tw < 128:
            tile_flow = _estimate_flow_dis(ref_gray[y0:y1, x0:x1],
                                            warped_gray[y0:y1, x0:x1])
            wy = _blend_weight_1d(th, min(overlap, th // 2))
            wx = _blend_weight_1d(tw, min(overlap, tw // 2))
            tile_weight = wy[:, None] * wx[None, :]
            flow_accum[y0:y1, x0:x1, 0] += tile_flow[:, :, 0] * tile_weight
            flow_accum[y0:y1, x0:x1, 1] += tile_flow[:, :, 1] * tile_weight
            weight_accum[y0:y1, x0:x1] += tile_weight
            done += 1
        else:
            raft_batch_refs.append(ref_gray[y0:y1, x0:x1])
            raft_batch_warps.append(warped_gray[y0:y1, x0:x1])
            raft_batch_specs.append((y0, x0, y1, x1))
            if len(raft_batch_refs) >= effective_batch:
                _flush_batch()

    _flush_batch()

    if done > 1:
        print(f"  [RAFT] Processed {done} tiles ({tiles_x}x{tiles_y})", flush=True)

    # Normalize by accumulated weights
    mask = weight_accum > 0
    flow_accum[mask, 0] /= weight_accum[mask]
    flow_accum[mask, 1] /= weight_accum[mask]

    return flow_accum.astype(np.float32)


def _estimate_flow_dis(ref_gray: np.ndarray, warped_gray: np.ndarray) -> np.ndarray:
    """Compute dense optical flow using OpenCV DIS (fast, no GPU needed).

    Returns flow (H, W, 2) in pixels: flow[y, x] = (dx, dy).
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setVariationalRefinementIterations(5)
    flow = dis.calc(ref_gray, warped_gray, None)
    return flow


def _estimate_flow(ref_gray: np.ndarray, warped_gray: np.ndarray) -> np.ndarray:
    """Compute dense optical flow using best available method (RAFT > DIS)."""
    model, _ = _get_raft_model()
    if model is not None:
        return _estimate_flow_raft(ref_gray, warped_gray)
    return _estimate_flow_dis(ref_gray, warped_gray)


def _forward_backward_mask(
    ref_gray: np.ndarray,
    warped_gray: np.ndarray,
    flow_fwd: np.ndarray,
    threshold_px: float = 3.0,
) -> np.ndarray:
    """Compute forward-backward consistency mask.

    Returns bool mask where True = reliable flow.
    Uses the same flow estimator as the forward pass (RAFT or DIS).
    """
    _, _, err = _forward_backward_error(ref_gray, warped_gray, flow_fwd)
    return err < threshold_px


def _forward_backward_error(
    ref_gray: np.ndarray,
    warped_gray: np.ndarray,
    flow_fwd: np.ndarray,
) -> tuple:
    """Compute forward-backward consistency error.

    Returns (flow_bwd, bwd_at_fwd, err) where err is per-pixel FB error in pixels.
    """
    flow_bwd = _estimate_flow(warped_gray, ref_gray)

    h, w = flow_fwd.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    x_fwd = x_coords + flow_fwd[:, :, 0]
    y_fwd = y_coords + flow_fwd[:, :, 1]

    bwd_at_fwd = cv2.remap(
        flow_bwd, x_fwd, y_fwd,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    err = np.sqrt(
        (flow_fwd[:, :, 0] + bwd_at_fwd[:, :, 0])**2 +
        (flow_fwd[:, :, 1] + bwd_at_fwd[:, :, 1])**2
    )
    return flow_bwd, bwd_at_fwd, err


# ---------------------------------------------------------------------------
# Shared flow post-processing helpers
# ---------------------------------------------------------------------------

def clamp_flow_magnitude(flow: np.ndarray, max_m: float, res_m: float) -> np.ndarray:
    """Clamp flow vectors whose magnitude exceeds *max_m* metres.

    *flow* is (H, W, 2) in pixel units; *res_m* converts pixels to metres.
    Returns the (H, W) magnitude array (post-clamping).
    """
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    max_px = max_m / res_m
    too_large = mag > max_px
    if np.any(too_large):
        clip = np.where(too_large, max_px / (mag + 1e-8), 1.0)
        flow[:, :, 0] *= clip
        flow[:, :, 1] *= clip
        mag = np.clip(mag, 0, max_px)
    return mag


def subtract_flow_bias(
    flow: np.ndarray,
    mask: np.ndarray,
    res_m: float,
    max_bias_m: float = _DEFAULT_MAX_BIAS_M,
    label: str = "FlowRefine",
) -> None:
    """Subtract systematic median bias from *flow* using reliable *mask* pixels.

    Modifies *flow* in-place.  Only subtracts if per-axis bias exceeds 5 m,
    and caps the subtraction at *max_bias_m*.
    """
    if not mask.any():
        return
    reliable_dx = flow[mask, 0]
    reliable_dy = flow[mask, 1]
    median_dx = float(np.median(reliable_dx))
    median_dy = float(np.median(reliable_dy))
    bias_m_dx = abs(median_dx) * res_m
    bias_m_dy = abs(median_dy) * res_m
    if bias_m_dx > 5.0 or bias_m_dy > 5.0:
        max_bias_px = max_bias_m / res_m
        capped_dx = float(np.clip(median_dx, -max_bias_px, max_bias_px))
        capped_dy = float(np.clip(median_dy, -max_bias_px, max_bias_px))
        flow[:, :, 0] -= capped_dx
        flow[:, :, 1] -= capped_dy
        print(
            f"  [{label}] Subtracted median bias: "
            f"dx={median_dx * res_m:+.1f}m, dy={median_dy * res_m:+.1f}m"
            f" (capped to dx={capped_dx * res_m:+.1f}m, dy={capped_dy * res_m:+.1f}m)",
            flush=True,
        )


def dense_flow_refinement(
    ref_arr: np.ndarray,
    warped_arr: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    output_res_m: float = 1.0,
    work_res_m: float = 2.0,
    max_correction_m: float = 50.0,
    fb_threshold_px: float = 3.0,
    median_kernel: int = 5,
) -> tuple:
    """Apply dense optical flow refinement to an existing remap field.

    Parameters
    ----------
    ref_arr : (H, W) float32 reference image
    warped_arr : (H, W) float32 already-warped source image
    map_x, map_y : (H, W) float32 existing remap coordinates (source pixel coords)
    output_res_m : pixel size of the output grid in metres
    work_res_m : resolution for flow computation (lower = faster)
    max_correction_m : clamp corrections beyond this (metres)
    fb_threshold_px : forward-backward consistency threshold
    median_kernel : median filter kernel for flow smoothing

    Returns
    -------
    (map_x_refined, map_y_refined) : refined remap fields
    """
    h, w = ref_arr.shape[:2]

    # Work at reduced resolution for speed
    scale = output_res_m / work_res_m
    if scale < 1.0:
        work_h = max(64, int(h * scale))
        work_w = max(64, int(w * scale))
    else:
        work_h, work_w = h, w
        scale = 1.0

    # Convert to uint8 for optical flow
    ref_u8 = to_u8_percentile(ref_arr)
    warped_u8 = to_u8_percentile(warped_arr)

    if scale < 1.0:
        ref_small = cv2.resize(ref_u8, (work_w, work_h), interpolation=cv2.INTER_AREA)
        warped_small = cv2.resize(warped_u8, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        ref_small = ref_u8
        warped_small = warped_u8

    # Compute forward flow: ref → warped (tells us how to shift warped to match ref)
    flow_fwd = _estimate_flow(ref_small, warped_small)

    # Forward-backward consistency check
    fb_mask = _forward_backward_mask(ref_small, warped_small, flow_fwd, fb_threshold_px)

    # Median filter to smooth noise
    if median_kernel > 1:
        flow_fwd[:, :, 0] = cv2.medianBlur(flow_fwd[:, :, 0], median_kernel)
        flow_fwd[:, :, 1] = cv2.medianBlur(flow_fwd[:, :, 1], median_kernel)

    subtract_flow_bias(flow_fwd, fb_mask, work_res_m)

    # Zero out unreliable flow
    flow_fwd[~fb_mask] = 0

    # Also zero where either image has no data
    if scale < 1.0:
        ref_valid = cv2.resize((ref_arr > 0).astype(np.uint8), (work_w, work_h),
                               interpolation=cv2.INTER_NEAREST) > 0
        warped_valid = cv2.resize((warped_arr > 0).astype(np.uint8), (work_w, work_h),
                                  interpolation=cv2.INTER_NEAREST) > 0
    else:
        ref_valid = ref_arr > 0
        warped_valid = warped_arr > 0
    flow_fwd[~(ref_valid & warped_valid)] = 0

    # Clamp corrections
    flow_mag = clamp_flow_magnitude(flow_fwd, max_correction_m, work_res_m)

    # Upsample flow to full resolution if needed
    if scale < 1.0:
        flow_full_x = cv2.resize(flow_fwd[:, :, 0], (w, h),
                                  interpolation=cv2.INTER_LINEAR) / scale
        flow_full_y = cv2.resize(flow_fwd[:, :, 1], (w, h),
                                  interpolation=cv2.INTER_LINEAR) / scale
    else:
        flow_full_x = flow_fwd[:, :, 0]
        flow_full_y = flow_fwd[:, :, 1]

    # The flow tells us the shift from ref to warped. To correct the warped
    # image, we adjust the source remap coordinates in the opposite direction.
    # flow_fwd[y,x] = (dx, dy) means ref pixel (x,y) matches warped pixel (x+dx, y+dy).
    # So the warped image needs to be shifted by -flow to align with ref.
    # In remap space: adjust source coordinates by +flow (since we're pulling from source).
    # But the scale differs: flow is in output pixels, map_x/map_y are in source pixels.
    # The ratio is output_res_m / source_res_m ≈ 1 for same-resolution grids.
    # Since the grid optimizer already matched resolutions, we apply flow directly.
    map_x_refined = map_x + flow_full_x.astype(np.float32)
    map_y_refined = map_y + flow_full_y.astype(np.float32)

    reliable_pct = float(fb_mask.sum()) / max(1, fb_mask.size) * 100
    mean_correction = float(np.mean(flow_mag[fb_mask])) * work_res_m if fb_mask.any() else 0.0
    print(f"  [FlowRefine] {reliable_pct:.0f}% reliable pixels, "
          f"mean correction {mean_correction:.1f}m, "
          f"max correction {float(flow_mag.max()) * work_res_m:.1f}m", flush=True)

    return map_x_refined, map_y_refined


def apply_flow_refinement_to_file(
    warped_path: str,
    reference_path: str,
    work_crs,
    output_bounds: tuple,
    output_res_m: float,
    max_correction_m: float = 50.0,
    fb_threshold_px: float = 3.0,
    median_kernel: int = 5,
) -> bool:
    """Apply dense flow refinement to an already-warped GeoTIFF file in-place.

    This enables flow refinement for TPS (or any other warp method) that
    produces a GeoTIFF directly, without going through grid-based remap fields.

    Returns True if flow was applied, False if skipped (insufficient reliability).
    """
    # Open warped file to get dimensions and transform
    with rasterio.open(warped_path) as warped_ds:
        out_w = warped_ds.width
        out_h = warped_ds.height
        out_transform = warped_ds.transform
        out_crs = warped_ds.crs
        warped_profile = warped_ds.profile.copy()

    # Work resolution: ~4m/px (or output_res if coarser)
    fr_work_res = max(4.0, output_res_m)
    fr_scale = output_res_m / fr_work_res
    fr_w = max(64, int(out_w * fr_scale))
    fr_h = max(64, int(out_h * fr_scale))

    # Compute bounds from output_bounds (left, bottom, right, top)
    left, bottom, right, top = output_bounds

    fr_transform = rasterio.transform.from_bounds(left, bottom, right, top, fr_w, fr_h)

    # Read reference at reduced resolution
    src_ref = rasterio.open(reference_path)
    ref_arr_fr = np.zeros((fr_h, fr_w), dtype=np.float32)
    ref_data = src_ref.read(1).astype(np.float32)
    reproject(
        source=ref_data,
        destination=ref_arr_fr,
        src_transform=src_ref.transform,
        src_crs=src_ref.crs,
        dst_transform=fr_transform,
        dst_crs=work_crs,
        resampling=Resampling.bilinear,
    )
    src_ref.close()
    del ref_data

    # Read warped image at reduced resolution
    with rasterio.open(warped_path) as warped_ds:
        warped_full = warped_ds.read(1).astype(np.float32)
    warped_arr_fr = cv2.resize(warped_full, (fr_w, fr_h), interpolation=cv2.INTER_AREA)

    # Convert to uint8
    ref_u8_fr = to_u8_percentile(ref_arr_fr)
    warped_u8_fr = to_u8_percentile(warped_arr_fr)

    # Compute flow
    flow_fwd = _estimate_flow(ref_u8_fr, warped_u8_fr)
    fb_mask = _forward_backward_mask(ref_u8_fr, warped_u8_fr, flow_fwd, fb_threshold_px)

    # Median filter
    if median_kernel > 1:
        flow_fwd[:, :, 0] = cv2.medianBlur(flow_fwd[:, :, 0], median_kernel)
        flow_fwd[:, :, 1] = cv2.medianBlur(flow_fwd[:, :, 1], median_kernel)

    subtract_flow_bias(flow_fwd, fb_mask, fr_work_res, label="FlowRefine-TPS")

    flow_fwd[~fb_mask] = 0

    # Zero where no data
    valid_mask_fr = (ref_arr_fr > 0) & (warped_arr_fr > 0)
    flow_fwd[~valid_mask_fr] = 0

    # Clamp corrections
    flow_mag = clamp_flow_magnitude(flow_fwd, max_correction_m, fr_work_res)

    reliable_pct = float(fb_mask.sum()) / max(1, fb_mask.size) * 100
    mean_corr = float(np.mean(flow_mag[fb_mask])) * fr_work_res if fb_mask.any() else 0.0
    print(
        f"  [FlowRefine-TPS] {reliable_pct:.0f}% reliable, "
        f"mean correction {mean_corr:.1f}m, "
        f"max {float(flow_mag.max()) * fr_work_res:.1f}m",
        flush=True,
    )

    # Reliability gate: same thresholds as grid flow refinement
    if reliable_pct < 10.0:
        print(
            f"  [FlowRefine-TPS] Only {reliable_pct:.0f}% reliable "
            f"(mean corr {mean_corr:.1f}m) — skipping",
            flush=True,
        )
        del ref_arr_fr, warped_arr_fr, ref_u8_fr, warped_u8_fr, fb_mask, valid_mask_fr
        gc.collect()
        return False

    del ref_arr_fr, warped_arr_fr, ref_u8_fr, warped_u8_fr, fb_mask, valid_mask_fr

    # Upsample flow to full output resolution
    flow_x_full = cv2.resize(
        flow_fwd[:, :, 0], (out_w, out_h), interpolation=cv2.INTER_LINEAR
    ) / fr_scale
    flow_y_full = cv2.resize(
        flow_fwd[:, :, 1], (out_w, out_h), interpolation=cv2.INTER_LINEAR
    ) / fr_scale
    del flow_fwd
    gc.collect()

    # Remap the warped image strip-by-strip using identity + flow correction.
    # flow_fwd is ref→warped, so to correct warped toward ref we remap with
    # map_x = x + flow_x, map_y = y + flow_y (pulling from offset locations).
    #
    # cv2.remap requires src and dst dimensions < SHRT_MAX (32767).  For large
    # images we chunk across columns, cropping the source per-chunk so both
    # stay within limits.
    _REMAP_MAX = 30000
    needs_chunking = out_w > _REMAP_MAX or out_h > _REMAP_MAX or warped_full.shape[0] > _REMAP_MAX or warped_full.shape[1] > _REMAP_MAX
    if needs_chunking:
        print(f"  [FlowRefine-TPS] Image {warped_full.shape[1]}x{warped_full.shape[0]} exceeds SHRT_MAX, using chunked remap", flush=True)

    print("  [FlowRefine-TPS] Re-mapping with flow corrections...", flush=True)
    strip_h = min(2048, out_h)
    n_strips = (out_h + strip_h - 1) // strip_h

    with rasterio.open(warped_path, "r+") as dst:
        for si in range(n_strips):
            rs = si * strip_h
            re = min(rs + strip_h, out_h)
            sh = re - rs

            # Identity remap + flow correction
            yc = np.arange(rs, re, dtype=np.float32)
            xc = np.arange(out_w, dtype=np.float32)
            xg, yg = np.meshgrid(xc, yc)

            map_x = xg + flow_x_full[rs:re].astype(np.float32)
            map_y = yg + flow_y_full[rs:re].astype(np.float32)

            # Remap from full image (flow can pull from outside the strip)
            strip_out = chunked_remap(warped_full, map_x, map_y)

            # Preserve nodata where remap went out of bounds
            oob = (map_x < 0) | (map_x >= out_w - 1) | (map_y < 0) | (map_y >= out_h - 1)
            strip_out[oob] = 0

            dst.write(strip_out.astype(warped_profile["dtype"]), indexes=1, window=((rs, re), (0, out_w)))

            if (si + 1) % 5 == 0 or si == n_strips - 1:
                print(f"    FlowRefine-TPS strip {si + 1}/{n_strips} complete", flush=True)

    del warped_full, flow_x_full, flow_y_full
    gc.collect()
    return True
