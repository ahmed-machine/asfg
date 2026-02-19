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


def clahe_normalize(img, clip_limit=3.0, grid=(8, 8)):
    """Apply CLAHE contrast enhancement to a uint8 image.

    If *img* is not uint8, converts via :func:`to_u8` first.
    Returns a uint8 image.
    """
    if img.dtype != np.uint8:
        img = to_u8(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    return clahe.apply(img)


def prepare_match_pair(ref, off, clip_limit=3.0, grid=(8, 8)):
    """Normalise both images to CLAHE-enhanced uint8.

    Convenience wrapper used by scale detection and NCC matching.
    Returns (ref_u8, off_u8).
    """
    return clahe_normalize(ref, clip_limit, grid), clahe_normalize(off, clip_limit, grid)


def build_preprocessed_stack(arr):
    """Build a small multi-channel preprocessing stack for QA and masking."""

    u8 = to_u8(arr)
    clahe = clahe_normalize(u8)
    grad = sobel_gradient(clahe)
    lap = cv2.Laplacian(clahe.astype(np.float32), cv2.CV_32F, ksize=3)
    texture = cv2.GaussianBlur(np.abs(lap), (7, 7), 0)

    if np.any(u8 > 0):
        grad = grad / max(float(np.max(grad)), 1.0)
        texture = texture / max(float(np.max(texture)), 1.0)

    return {
        "raw": u8,
        "clahe": clahe,
        "gradient": grad.astype(np.float32),
        "texture": texture.astype(np.float32),
    }


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
