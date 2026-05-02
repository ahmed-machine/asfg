"""Detect and inpaint white diagonal scratches on declassified film scans.

Scanned KH-9 (and KH-4) imagery often shows long thin bright streaks at
the strip edges — clear emulsion scratches that print as bright lines
on the positive scan, plus occasional static-discharge "lightning" tracks
and light leaks. They co-occur most heavily near the film rebate where
the transport mechanism touched the emulsion repeatedly.

The matchers (RoMa, ELoFTR, NCC) treat these streaks as high-contrast
features and produce strong but spatially-wrong correspondences in the
contaminated bands. On Bahrain KH-4B alignment the western edge of the
1976 KH-9 reference is heavily scratched; matchers picked up bright
streaks instead of real coastline features and the western GCP coverage
collapsed.

Detection follows a three-step pipeline:

1. **Directional top-hat** highlights bright thin structures along the
   suspected scratch axis. White top-hat = ``arr - opening(arr, kernel)``
   where the kernel is a long thin line at the scratch angle. Rotated
   variants cover ±20° around horizontal/diagonal/vertical.
2. **Hough line transform** picks out long straight ridges in the top-
   hat response. Length and angle gates reject curved geographic
   features (rivers, roads, coastlines) which are rarely both straight
   AND >1 km AND high-contrast.
3. **Mask dilation** widens detected lines by ``max_width_px / 2`` so
   the full scratch is covered, not just its bright spine.

Inpainting uses ``cv2.INPAINT_TELEA`` (Telea 2004) — fast, derived from
Bertalmio's algorithm, good for thin elongated structures.

References:
  - Bruni et al. 2010, "Multidirectional Scratch Detection and
    Restoration in Digitized Old Images", EURASIP J. Image Video Proc.
  - Ko & Kim 2007, "Improvement of Film Scratch Inpainting Algorithm
    Using Sobel-Based Isophote Computation over Hilbert Scan Line".
  - Telea 2004, "An Image Inpainting Technique Based on the Fast
    Marching Method".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class ScratchDetectorParams:
    """Tunable thresholds for ``detect_scratches``.

    Defaults are calibrated for KH-9 panoramic scans at native resolution.
    Each angle is given in degrees, length and width in pixels of the
    array passed to the detector.
    """

    # Minimum line length (in pixels) for a Hough segment to count as a
    # scratch. Tuned to reject roads/rivers on Bahrain at ~50 m/px.
    min_length_px: int = 200
    # Maximum scratch width in pixels. Drives the dilation kernel size
    # so the inpaint mask covers the full bright strip, not just the
    # spine.
    max_width_px: int = 8
    # Brightness percentile used to threshold the top-hat response.
    # Higher = stricter = fewer false positives + fewer detected scratches.
    brightness_percentile: float = 97.0
    # Allowed scratch angles, in degrees from horizontal (0 = horizontal,
    # ±90 = vertical). Default is near-vertical only: scanner roller
    # transport along the strip direction is the dominant scratch axis
    # on KH-9 (vertical after warping to north-up), and the tight
    # ±75–90° band avoids real geographic features. Diagonal ranges
    # (±10–45°) were tried but caught Bahrain shorelines (which trend
    # ~NW-SE / NE-SW around ±20–40° in the warped reference) wholesale.
    # Callers can opt back into diagonals via the ``--angle-range`` CLI
    # flag when the scene's geography won't conflict.
    angle_ranges_deg: tuple = (
        (-90.0, -75.0),
        (75.0, 90.0),
    )
    # Hough accumulator threshold (min votes for a candidate line).
    hough_threshold: int = 80
    # Top-hat kernel length. Each rotation is searched at this length —
    # longer = better selectivity but slower. ``min_length_px / 2`` is a
    # reasonable default.
    tophat_kernel_length: int = 100
    # Number of angle samples for the directional top-hat. More samples
    # = finer angular resolution but slower.
    tophat_n_angles: int = 8
    # Robust angle clustering (RANSAC/MAGSAC analogue). After the angle-
    # range prefilter, fit a single dominant scratch direction by taking
    # the length-weighted mean angle in doubled-angle space (handles ±90°
    # wraparound) and reject lines whose angle deviates by more than
    # ``cluster_mad_k`` MADs from the cluster centre. Real scanner-roller
    # scratches form a tight (≲2°) cluster; spurious shoreline / road
    # detections lie outside it. Set ``cluster_mad_k <= 0`` to disable.
    cluster_mad_k: float = 3.0
    # Hard floor on cluster tolerance — prevents a near-perfect inlier
    # cluster from having zero MAD and rejecting everything. Tuned so a
    # tight scanner cluster (MAD ~0.3°) keeps lines within ±2°.
    cluster_mad_floor_deg: float = 2.0
    # Skip clustering when fewer than this many lines survive the angle-
    # range prefilter — too noisy to identify a dominant direction.
    cluster_min_lines: int = 3
    # Maximum perpendicular extent (minor axis of the rotated bounding
    # rectangle) for a connected component in the rendered mask. Real
    # scanner scratches are thin and spaced apart, so each component is
    # only as wide as ``max_width_px`` plus a little dilation. Heavy
    # overlap of fragmented detections (e.g. a building edge sliced into
    # many parallel Hough segments) produces a wide blob — those get
    # rejected here. Set to 0 to disable.
    cluster_max_short_axis_px: int = 25

    def hash(self) -> str:
        """Stable hash for provenance keying. Used by the cleaned-
        reference sidecar to detect parameter changes that should
        invalidate a cached cleaned image."""
        import hashlib
        payload = json.dumps(
            {
                "min_length_px": self.min_length_px,
                "max_width_px": self.max_width_px,
                "brightness_percentile": self.brightness_percentile,
                "angle_ranges_deg": list(self.angle_ranges_deg),
                "hough_threshold": self.hough_threshold,
                "tophat_kernel_length": self.tophat_kernel_length,
                "tophat_n_angles": self.tophat_n_angles,
                "cluster_mad_k": self.cluster_mad_k,
                "cluster_mad_floor_deg": self.cluster_mad_floor_deg,
                "cluster_min_lines": self.cluster_min_lines,
                "cluster_max_short_axis_px": self.cluster_max_short_axis_px,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


_DEFAULT_PARAMS = ScratchDetectorParams()


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype in (np.float32, np.float64):
        valid = arr[arr > 0]
        if valid.size < 100:
            return np.zeros(arr.shape, dtype=np.uint8)
        lo = float(np.percentile(valid, 1))
        hi = float(np.percentile(valid, 99))
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return out
    return arr.astype(np.uint8)


def _directional_tophat(arr_u8: np.ndarray, length: int, n_angles: int) -> np.ndarray:
    """Maximum top-hat response across rotated linear structuring
    elements. Highlights bright thin structures aligned with any of the
    sampled angles. Diagonal scratches show up; circular features
    (round structures) and short bright spots are suppressed."""
    if min(arr_u8.shape) < length:
        length = max(11, min(arr_u8.shape) // 2)
        if length % 2 == 0:
            length -= 1
    response = np.zeros(arr_u8.shape, dtype=np.uint8)
    for i in range(n_angles):
        angle_deg = -90.0 + (180.0 / n_angles) * i
        # Build a horizontal line then rotate
        kernel = np.zeros((length, length), dtype=np.uint8)
        cv2.line(kernel, (0, length // 2), (length - 1, length // 2), 1, 1)
        if abs(angle_deg) > 1e-3:
            M = cv2.getRotationMatrix2D((length / 2, length / 2), angle_deg, 1.0)
            kernel = cv2.warpAffine(kernel, M, (length, length))
            kernel = (kernel > 0).astype(np.uint8)
        if int(kernel.sum()) < 3:
            continue
        opened = cv2.morphologyEx(arr_u8, cv2.MORPH_OPEN, kernel)
        # White top-hat
        diff = cv2.subtract(arr_u8, opened)
        np.maximum(response, diff, out=response)
    return response


def _normalise_angle_deg(angle: float) -> float:
    """Wrap an angle into ``(-90, 90]`` (lines have period 180°)."""
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    return angle


def _filter_to_dominant_angle_cluster(
    lines: list,
    angles: np.ndarray,
    lengths: np.ndarray,
    params: ScratchDetectorParams,
) -> tuple[list, dict]:
    """Robust outlier rejection on the per-line angle distribution.

    Real scanner-roller scratches share one tight direction (the strip
    transport axis after warping); spurious detections from shorelines
    or roads scatter across other angles. We fit a single dominant
    direction and discard lines whose angular distance to it exceeds a
    MAD-based tolerance — analogous to a 1D MAGSAC on line angles.

    The dominant direction is the length-weighted mean in doubled-angle
    space (cos(2θ), sin(2θ)) so that ±89° fold to the same direction —
    important because near-vertical scratches frequently land on both
    sides of the ±90° wraparound after Hough discretisation.

    Returns ``(kept_lines, info)`` where ``info`` carries the cluster
    centre and tolerance for diagnostics.
    """
    info = {"dominant_angle_deg": None, "mad_deg": None,
            "tolerance_deg": None, "n_in": int(len(lines)),
            "n_out": int(len(lines))}
    if params.cluster_mad_k <= 0 or len(lines) < params.cluster_min_lines:
        return lines, info
    rad2 = np.deg2rad(angles) * 2.0
    w = lengths.astype(np.float64)
    if w.sum() <= 0:
        return lines, info
    w = w / w.sum()
    mean_x = float(np.sum(w * np.cos(rad2)))
    mean_y = float(np.sum(w * np.sin(rad2)))
    # Resultant length R near 0 means angles are evenly spread (no
    # dominant direction): bail out and keep prefilter-only behaviour.
    if (mean_x * mean_x + mean_y * mean_y) < 1e-6:
        return lines, info
    dom_angle = _normalise_angle_deg(
        0.5 * float(np.degrees(np.arctan2(mean_y, mean_x))))
    # Signed angular distance with period-180 wrap → values in (-90, 90].
    deltas = (angles - dom_angle + 90.0) % 180.0 - 90.0
    mad = float(np.median(np.abs(deltas - float(np.median(deltas)))))
    sigma = max(mad * 1.4826, params.cluster_mad_floor_deg)
    tol = params.cluster_mad_k * sigma
    keep = np.abs(deltas - float(np.median(deltas))) <= tol
    info.update({
        "dominant_angle_deg": dom_angle,
        "mad_deg": mad,
        "tolerance_deg": float(tol),
        "n_out": int(keep.sum()),
    })
    return [lines[i] for i in range(len(lines)) if bool(keep[i])], info


def _hough_lines_filtered(
    top_hat: np.ndarray, params: ScratchDetectorParams,
) -> tuple[list, dict]:
    """Run probabilistic Hough on a thresholded top-hat response and
    filter by length, angle range, and dominant-angle cluster. Returns
    ``(lines, info)`` with diagnostics from the cluster filter."""
    empty_info = {"dominant_angle_deg": None, "mad_deg": None,
                  "tolerance_deg": None, "n_in": 0, "n_out": 0}
    if top_hat.size == 0:
        return [], empty_info
    # Threshold the top-hat response at the brightness percentile of the
    # top-hat distribution itself (not the input image): this picks out
    # the strongest ridges regardless of overall brightness.
    nonzero = top_hat[top_hat > 0]
    if nonzero.size < 100:
        return [], empty_info
    threshold_value = float(np.percentile(nonzero, params.brightness_percentile))
    binary = (top_hat >= threshold_value).astype(np.uint8) * 255
    raw_lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 360,
        threshold=params.hough_threshold,
        minLineLength=params.min_length_px,
        maxLineGap=20,
    )
    if raw_lines is None:
        return [], empty_info
    kept = []
    angles = []
    lengths = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < params.min_length_px:
            continue
        angle = _normalise_angle_deg(
            float(np.degrees(np.arctan2(dy, dx))))
        if not any(lo <= angle <= hi for (lo, hi) in params.angle_ranges_deg):
            continue
        kept.append((int(x1), int(y1), int(x2), int(y2)))
        angles.append(angle)
        lengths.append(length)
    if not kept:
        return [], empty_info
    return _filter_to_dominant_angle_cluster(
        kept, np.asarray(angles, dtype=np.float64),
        np.asarray(lengths, dtype=np.float64), params)


def detect_scratches(arr: np.ndarray,
                     params: Optional[ScratchDetectorParams] = None,
                     ) -> np.ndarray:
    """Detect bright thin diagonal scratches on a film image.

    Returns a ``uint8`` mask (255 where scratch, 0 elsewhere) with the
    same shape as ``arr``. ``arr`` may be float (treated as raw
    intensity) or uint8.

    The detector is conservative — it requires a structure to be:
      * thin (width within ``max_width_px``),
      * long (length ≥ ``min_length_px``),
      * bright (top-hat response above the configured percentile),
      * diagonal-ish (slope within the configured angle ranges).

    Geographic features rarely satisfy all four. False-positive risk is
    highest for: ruler-straight Roman roads, modern highway segments,
    breakwaters / harbour walls. None of these are typical of declassified
    1960s–70s civilian terrain captures.
    """
    mask, _info = detect_scratches_with_info(arr, params)
    return mask


def _filter_dense_components(
    mask: np.ndarray,
    params: ScratchDetectorParams,
) -> tuple[np.ndarray, dict]:
    """Reject mask components whose minor axis exceeds the threshold.

    Real scanner-roller scratches are thin and spaced apart — each
    appears as a long, narrow component (minor axis ≈ ``max_width_px``).
    A heavy overlap of fragmented Hough segments — e.g. a building edge
    sliced into many parallel detections, or a printing-drum artifact —
    fuses into a wide blob with a much larger minor axis. We use the
    minor axis of ``cv2.minAreaRect`` (rotated bounding rectangle, so
    the threshold is angle-agnostic) and discard components above the
    cap.

    Returns ``(filtered_mask, info)``.
    """
    info = {"n_components_in": 0, "n_components_out": 0,
            "max_short_axis_px": int(params.cluster_max_short_axis_px)}
    if params.cluster_max_short_axis_px <= 0:
        return mask, info
    n_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if n_labels <= 1:
        return mask, info
    info["n_components_in"] = int(n_labels - 1)  # exclude background
    out = np.zeros_like(mask)
    n_kept = 0
    for lbl in range(1, n_labels):
        comp = (labels == lbl)
        rows, cols = np.nonzero(comp)
        if rows.size < 5:
            continue
        pts_xy = np.stack([cols, rows], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts_xy)
        # rect[1] is (width, height); minor axis is the smaller of the two.
        short_axis = float(min(rect[1]))
        if short_axis <= float(params.cluster_max_short_axis_px):
            out[comp] = 255
            n_kept += 1
    info["n_components_out"] = n_kept
    return out, info


def detect_scratches_with_info(
    arr: np.ndarray,
    params: Optional[ScratchDetectorParams] = None,
) -> tuple[np.ndarray, dict]:
    """Variant of :func:`detect_scratches` that also returns cluster
    diagnostics from the dominant-angle MAD filter and the dense-
    component (heavy-overlap) filter. Used by the cleaning CLI to
    surface dominant angle, MAD tolerance, and component counts per
    run."""
    p = params or _DEFAULT_PARAMS
    arr_u8 = _ensure_uint8(arr)
    if min(arr_u8.shape) < 32:
        return (np.zeros(arr_u8.shape, dtype=np.uint8),
                {"dominant_angle_deg": None, "mad_deg": None,
                 "tolerance_deg": None, "n_in": 0, "n_out": 0,
                 "n_components_in": 0, "n_components_out": 0,
                 "max_short_axis_px": int(p.cluster_max_short_axis_px)})
    top_hat = _directional_tophat(arr_u8, p.tophat_kernel_length, p.tophat_n_angles)
    lines, info = _hough_lines_filtered(top_hat, p)
    mask = np.zeros(arr_u8.shape, dtype=np.uint8)
    if not lines:
        info.update(n_components_in=0, n_components_out=0,
                    max_short_axis_px=int(p.cluster_max_short_axis_px))
        return mask, info
    width = max(1, int(p.max_width_px))
    for x1, y1, x2, y2 in lines:
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=width)
    # Density-based component filter: rejects fat blobs that arise from
    # heavy local overlap of parallel detections. Runs BEFORE the final
    # 3x3 dilation so the component widths reflect the rendered Hough
    # output, not a dilation-thickened version of it.
    mask, comp_info = _filter_dense_components(mask, p)
    info.update(comp_info)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask, info


def inpaint_scratches(arr: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      params: Optional[ScratchDetectorParams] = None,
                      ) -> np.ndarray:
    """Inpaint bright diagonal scratches with surrounding texture.

    Uses ``cv2.INPAINT_TELEA`` — fast marching method, well-suited to
    thin elongated structures. Returns an array of the same dtype and
    shape as ``arr``. If ``mask`` is None the scratches are detected
    automatically via :func:`detect_scratches`.
    """
    p = params or _DEFAULT_PARAMS
    if mask is None:
        mask = detect_scratches(arr, p)
    if int(mask.sum()) == 0:
        return arr.copy()
    arr_u8 = _ensure_uint8(arr)
    inpainted = cv2.inpaint(arr_u8, mask, 3, cv2.INPAINT_TELEA)
    if arr.dtype == np.uint8:
        return inpainted
    # Restore original dynamic range for float arrays. We inpaint in
    # uint8 (the matchers consume uint8 anyway) but callers reading the
    # raw float buffer expect their original range — scale back.
    valid = arr[arr > 0]
    if valid.size < 100:
        return arr.copy()
    lo = float(np.percentile(valid, 1))
    hi = float(np.percentile(valid, 99))
    if hi <= lo:
        return arr.copy()
    out = inpainted.astype(np.float32) / 255.0 * (hi - lo) + lo
    # Preserve no-data zeros from the original
    out[arr <= 0] = 0
    return out.astype(arr.dtype)


# ---------------------------------------------------------------------------
# Cached cleaned-reference sidecar
# ---------------------------------------------------------------------------


def write_provenance(path: str, ref_path: str,
                     params: ScratchDetectorParams) -> None:
    """Write a provenance JSON next to a cleaned reference TIF."""
    stat = os.stat(ref_path)
    payload = {
        "reference_basename": os.path.basename(ref_path),
        "reference_mtime_ns": stat.st_mtime_ns,
        "reference_size": stat.st_size,
        "params_hash": params.hash(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def provenance_matches(provenance_path: str, ref_path: str,
                       params: ScratchDetectorParams) -> bool:
    """True iff the cleaned-reference sidecar at ``provenance_path`` was
    produced from the same reference + params as the current call."""
    if not os.path.exists(provenance_path):
        return False
    try:
        with open(provenance_path) as f:
            prov = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    try:
        if prov.get("reference_basename") != os.path.basename(ref_path):
            return False
        if prov.get("reference_size") != os.path.getsize(ref_path):
            return False
        if prov.get("reference_mtime_ns") != os.stat(ref_path).st_mtime_ns:
            return False
        if prov.get("params_hash") != params.hash():
            return False
    except OSError:
        return False
    return True


def generate_scratch_cleaned_reference(ref_path: str, output_path: str,
                                       provenance_path: str,
                                       params: Optional[ScratchDetectorParams] = None,
                                       ) -> str | None:
    """Read the full reference, detect+inpaint scratches at native
    resolution, and write a cleaned sidecar TIF + provenance JSON.

    Returns the output path on success, ``None`` on failure. Skips the
    work when the provenance JSON already matches the current
    (reference, params) pair.
    """
    p = params or _DEFAULT_PARAMS
    if (os.path.exists(output_path)
            and provenance_matches(provenance_path, ref_path, p)):
        print(f"  [scratch_clean] sidecar up-to-date: {output_path}", flush=True)
        return output_path

    import rasterio
    print(f"  [scratch_clean] processing {os.path.basename(ref_path)}...", flush=True)
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
        # Read at native resolution; reference rasters are typically
        # under 2 GB so an in-memory pass is fine.
        arr = src.read(1)
    mask, info = detect_scratches_with_info(arr, p)
    if info.get("dominant_angle_deg") is not None:
        print(f"  [scratch_clean] cluster: dominant_angle="
              f"{info['dominant_angle_deg']:+.2f}° ± "
              f"{info['tolerance_deg']:.2f}° (MAD={info['mad_deg']:.2f}°), "
              f"kept {info['n_out']}/{info['n_in']} lines", flush=True)
    elif info.get("n_in"):
        print(f"  [scratch_clean] cluster: skipped "
              f"(n_in={info['n_in']} below cluster_min_lines or k≤0); "
              f"prefilter alone gates", flush=True)
    if info.get("n_components_in"):
        n_in = info["n_components_in"]
        n_out = info["n_components_out"]
        rejected = n_in - n_out
        print(f"  [scratch_clean] density: kept {n_out}/{n_in} "
              f"components (rejected {rejected} fat blobs over "
              f"{info['max_short_axis_px']} px short-axis)", flush=True)
    n_masked = int(mask.sum() > 0)  # any-pixel check
    if n_masked == 0:
        print(f"  [scratch_clean] no scratches detected; copying reference",
              flush=True)
        cleaned = arr
    else:
        scratch_pct = float(np.count_nonzero(mask)) / mask.size * 100
        print(f"  [scratch_clean] masked {scratch_pct:.2f}% of pixels; "
              f"inpainting...", flush=True)
        cleaned = inpaint_scratches(arr, mask=mask, params=p)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Tile + LZW compress for production-size references; rasterio
    # requires tile dimensions to be multiples of 16, so disable tiling
    # on small images (test fixtures, anything < 256 on either axis).
    profile.update(compress="LZW", bigtiff="IF_SAFER")
    h_full = profile.get("height", cleaned.shape[0])
    w_full = profile.get("width", cleaned.shape[1])
    if h_full >= 256 and w_full >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cleaned, 1)
    write_provenance(provenance_path, ref_path, p)
    print(f"  [scratch_clean] wrote cleaned sidecar: {output_path}", flush=True)
    return output_path
