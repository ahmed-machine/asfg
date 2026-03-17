"""Image utilities: shifting, normalization, enhancement, and masks."""

import cv2
import numpy as np


def shift_array(arr, dx_px, dy_px):
    """Shift a 2D array by integer pixels using cv2.warpAffine or slicing.

    Uses fast cv2.warpAffine if dimensions are under OpenCV's SHRT_MAX limit,
    otherwise falls back to slicing.
    """
    dx_px = int(round(dx_px))
    dy_px = int(round(dy_px))
    h, w = arr.shape
    
    if h < 32767 and w < 32767:
        M = np.array([[1.0, 0.0, float(dx_px)], [0.0, 1.0, float(dy_px)]], dtype=np.float32)
        return cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
    result = np.zeros_like(arr)
    sx0 = max(0, -dx_px); sx1 = min(w, w - dx_px)
    sy0 = max(0, -dy_px); sy1 = min(h, h - dy_px)
    dx0 = max(0, dx_px);  dx1 = min(w, w + dx_px)
    dy0 = max(0, dy_px);  dy1 = min(h, h + dy_px)
    if sx1 > sx0 and sy1 > sy0:
        result[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return result


def to_u8(arr):
    """Normalize array to uint8 range."""
    valid = arr > 0
    out = np.zeros_like(arr, dtype=np.uint8)
    if not valid.any():
        return out
    mn, mx = arr[valid].min(), arr[valid].max()
    if mx > mn:
        out[valid] = ((arr[valid] - mn) / (mx - mn) * 255).astype(np.uint8)
    return out


def to_u8_percentile(arr, lo_pct=1, hi_pct=99):
    """Normalize array to uint8 using percentile clipping on valid (>0) pixels."""
    valid = arr[arr > 0]
    if len(valid) == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def clahe_normalize(img, clip_limit=3.0, grid=(8, 8)):
    """Apply CLAHE contrast enhancement to a uint8 image.

    If *img* is not uint8, converts via :func:`to_u8` first.
    Returns a uint8 image.
    """
    if img.dtype != np.uint8:
        img = to_u8(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    return clahe.apply(img)


def is_cloudy_patch(arr, threshold_bright=0.85, threshold_low_texture=0.15):
    """Detect if a patch is likely cloudy or featureless.

    Clouds in B&W satellite imagery appear as:
    1. Very bright (>85th percentile)
    2. Low texture/variation (std dev < 15% of range)

    Returns True if patch appears cloudy.
    """
    valid = arr > 0
    if not valid.any() or valid.sum() < arr.size * 0.3:
        return True

    valid_vals = arr[valid]
    mean_val = valid_vals.mean()
    std_val = valid_vals.std()
    val_range = valid_vals.max() - valid_vals.min()

    brightness = mean_val / 255.0 if arr.dtype == np.uint8 else (
        mean_val / valid_vals.max() if valid_vals.max() > 0 else 0
    )
    texture = std_val / val_range if val_range > 0 else 0

    return brightness > threshold_bright and texture < threshold_low_texture


def build_semantic_masks(arr, mode="coastal_obia"):
    """Return semantic-ish masks via the configured provider."""

    from .semantic_masking import build_semantic_masks as _build_semantic_masks

    return _build_semantic_masks(arr, mode=mode)


def make_land_mask(arr, mode="coastal_obia"):
    """Return the provider-backed land mask."""

    return build_semantic_masks(arr, mode=mode).land


def stable_feature_mask(arr, mode="coastal_obia"):
    """Return a stable feature mask."""

    return build_semantic_masks(arr, mode=mode).stable


def shoreline_mask(arr, mode="coastal_obia"):
    """Return a shoreline/intertidal mask."""

    return build_semantic_masks(arr, mode=mode).shoreline


def class_weight_map(arr, mode="coastal_obia"):
    """Return a soft class-aware weight map."""

    from .semantic_masking import class_weight_map as _class_weight_map

    return _class_weight_map(arr, mode=mode)


def sobel_gradient(img_u8):
    """Compute Sobel gradient magnitude from a uint8 image.

    Returns float64 gradient magnitude array.
    """
    f = img_u8.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


# ---------------------------------------------------------------------------
# Large-image remap with automatic chunking
# ---------------------------------------------------------------------------

_REMAP_MAX = 30000  # stay under OpenCV's SHRT_MAX (32767) with margin


def chunked_remap(src_data: np.ndarray, map_x: np.ndarray,
                  map_y: np.ndarray) -> np.ndarray:
    """cv2.remap with automatic column-chunking for images exceeding SHRT_MAX.

    Splits the destination into column chunks and crops the source per-chunk
    so both src and dst stay under the 32767 pixel limit in each dimension.
    Falls back to scipy.ndimage.map_coordinates when even the cropped source
    exceeds the limit.

    Parameters
    ----------
    src_data : 2-D float32 array (single band)
    map_x, map_y : 2-D float32 remap coordinate arrays (destination size)

    Returns
    -------
    2-D float32 result array of shape (map_x.shape)
    """
    src_h, src_w = src_data.shape
    dst_h, dst_w = map_x.shape

    if (src_h <= _REMAP_MAX and src_w <= _REMAP_MAX
            and dst_h <= _REMAP_MAX and dst_w <= _REMAP_MAX):
        return cv2.remap(src_data, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    result = np.zeros((dst_h, dst_w), dtype=np.float32)
    n_col_chunks = (dst_w + _REMAP_MAX - 1) // _REMAP_MAX

    for ci in range(n_col_chunks):
        c0 = ci * _REMAP_MAX
        c1 = min(c0 + _REMAP_MAX, dst_w)
        chunk_mx = map_x[:, c0:c1]
        chunk_my = map_y[:, c0:c1]

        valid = (chunk_mx >= 0) & (chunk_my >= 0)
        if not np.any(valid):
            continue

        sx_min = max(0, int(np.floor(chunk_mx[valid].min())) - 2)
        sx_max = min(src_w, int(np.ceil(chunk_mx[valid].max())) + 3)
        sy_min = max(0, int(np.floor(chunk_my[valid].min())) - 2)
        sy_max = min(src_h, int(np.ceil(chunk_my[valid].max())) + 3)

        if sx_max <= sx_min or sy_max <= sy_min:
            continue

        src_crop = src_data[sy_min:sy_max, sx_min:sx_max]
        adj_mx = chunk_mx - sx_min
        adj_my = chunk_my - sy_min
        crop_h, crop_w = src_crop.shape

        if crop_h <= _REMAP_MAX and crop_w <= _REMAP_MAX:
            result[:, c0:c1] = cv2.remap(
                src_crop, adj_mx, adj_my,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            from scipy.ndimage import map_coordinates
            coords = np.array([adj_my.astype(np.float64),
                               adj_mx.astype(np.float64)])
            result[:, c0:c1] = map_coordinates(
                src_crop, coords, order=1, mode='constant',
                cval=0.0, prefilter=False).astype(np.float32)

    return result
