"""GeoTIFF I/O, overlap computation, CRS helpers, and affine fitting."""

from contextlib import contextmanager

import cv2
import numpy as np
import rasterio
import rasterio.transform
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform, transform_bounds

from . import constants as _C
from .models import ModelCache, get_torch_device

# Cached CRS singletons — avoids repeated parsing on every call.
WGS84 = CRS.from_epsg(4326)
WEB_MERCATOR = CRS.from_epsg(3857)


def get_native_resolution_m(src, priors=None):
    """Compute the ground resolution of a rasterio dataset in meters.

    For geographic CRS (EPSG:4326), converts degrees to meters using
    cosine approximation at centre latitude.
    For projected CRS, returns pixel spacing directly.
    For no-CRS datasets (raw scans), estimates from metadata priors.
    """
    if src.crs is None:
        # Raw scan — estimate from metadata prior bounding box + image dims
        if priors:
            for prior in priors:
                if (getattr(prior, "west", None) is not None
                        and getattr(prior, "east", None) is not None
                        and getattr(prior, "south", None) is not None
                        and getattr(prior, "north", None) is not None):
                    center_lat = (prior.south + prior.north) / 2.0
                    m_per_deg_lon = 111320 * np.cos(np.radians(center_lat))
                    m_per_deg_lat = 110540
                    extent_x_m = (prior.east - prior.west) * m_per_deg_lon
                    extent_y_m = (prior.north - prior.south) * m_per_deg_lat
                    res_x = extent_x_m / max(src.width, 1)
                    res_y = extent_y_m / max(src.height, 1)
                    return (res_x + res_y) / 2.0
        # No priors — return 1.0 as sentinel (global search will handle)
        return 1.0

    transform = src.transform
    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)

    if src.crs.is_geographic:
        center_lat = (src.bounds.top + src.bounds.bottom) / 2
        meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
        meters_per_degree_lat = 110540
        res_x = pixel_x * meters_per_degree_lon
        res_y = pixel_y * meters_per_degree_lat
    else:
        res_x = pixel_x
        res_y = pixel_y

    return (res_x + res_y) / 2


def transform_point(src_crs, dst_crs, x, y):
    """Transform a single point between CRS objects."""

    xx, yy = transform(src_crs, dst_crs, [x], [y])
    return float(xx[0]), float(yy[0])


def dataset_bounds_in_crs(src, target_crs):
    """Return dataset bounds in *target_crs*."""

    if src.crs is None:
        return None
    return transform_bounds(src.crs, target_crs, *src.bounds)


def compute_overlap_or_none(src_offset, src_ref, work_crs):
    """Return overlap bounds or ``None`` when the rasters do not overlap."""

    offset_bounds = dataset_bounds_in_crs(src_offset, work_crs)
    ref_bounds = dataset_bounds_in_crs(src_ref, work_crs)
    if offset_bounds is None or ref_bounds is None:
        return None
    overlap = (
        max(offset_bounds[0], ref_bounds[0]),
        max(offset_bounds[1], ref_bounds[1]),
        min(offset_bounds[2], ref_bounds[2]),
        min(offset_bounds[3], ref_bounds[3]),
    )
    if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
        return None
    return overlap


def get_utm_crs_from_lonlat(lon, lat):
    """Return the UTM CRS covering a lon/lat point."""

    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def get_dataset_center_lonlat(src):
    """Return dataset center in EPSG:4326."""

    left, bottom, right, top = transform_bounds(src.crs, WGS84, *src.bounds)
    return (float((left + right) / 2.0), float((bottom + top) / 2.0))


def get_metric_crs(src_offset, src_ref, priors=None):
    """Choose a local metric CRS without requiring overlap."""

    priors = priors or []
    for prior in priors:
        center_lon = getattr(prior, "center_lon", None)
        center_lat = getattr(prior, "center_lat", None)
        if center_lon is not None and center_lat is not None:
            utm_crs = get_utm_crs_from_lonlat(center_lon, center_lat)
            print(
                "  Auto-detected work CRS from metadata priors: "
                f"EPSG:{utm_crs.to_epsg()}"
            )
            return utm_crs

    try:
        overlap_crs = get_utm_crs(src_offset, src_ref)
        return overlap_crs
    except ValueError:
        ref_lon, ref_lat = get_dataset_center_lonlat(src_ref)
        utm_crs = get_utm_crs_from_lonlat(ref_lon, ref_lat)
        print(
            "  Auto-detected work CRS from reference center: "
            f"EPSG:{utm_crs.to_epsg()}"
        )
        return utm_crs


def work_shift_to_dataset_shift(src, work_crs, dx_m, dy_m):
    """Approximate a shift in *work_crs* meters as a shift in ``src.crs``."""

    center_x = (src.bounds.left + src.bounds.right) / 2.0
    center_y = (src.bounds.bottom + src.bounds.top) / 2.0
    work_x, work_y = transform_point(src.crs, work_crs, center_x, center_y)
    shifted_x, shifted_y = transform_point(work_crs, src.crs, work_x - dx_m, work_y + dy_m)
    return float(shifted_x - center_x), float(shifted_y - center_y)


_overlap_cache = {}
_OVERLAP_CACHE_MAX = 8


def clear_overlap_cache():
    """Clear cached overlap arrays (call after pre-correction changes current_input)."""
    _overlap_cache.clear()


def read_overlap_region(src, overlap_bounds, target_crs, target_res):
    """Read and reproject a region of *src* into *target_crs* at *target_res*.

    Returns (array, transform) where transform maps pixel coords to CRS coords.
    Results are cached by (source path, overlap bounds, resolution).
    """
    cache_key = (src.name, overlap_bounds, target_res)
    if cache_key in _overlap_cache:
        return _overlap_cache[cache_key]

    left, bottom, right, top = overlap_bounds
    width = int(round((right - left) / target_res))
    height = int(round((top - bottom) / target_res))
    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    dst_array = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=dst_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )

    # LRU eviction: remove oldest entry if cache is full
    if len(_overlap_cache) >= _OVERLAP_CACHE_MAX:
        oldest_key = next(iter(_overlap_cache))
        del _overlap_cache[oldest_key]

    _overlap_cache[cache_key] = (dst_array, dst_transform)
    return dst_array, dst_transform


def get_utm_crs(src_offset, src_ref):
    """Auto-detect the UTM CRS from the center of the overlap region."""
    off_bounds = transform_bounds(src_offset.crs, WGS84, *src_offset.bounds)
    ref_bounds = transform_bounds(src_ref.crs, WGS84, *src_ref.bounds)

    left = max(off_bounds[0], ref_bounds[0])
    bottom = max(off_bounds[1], ref_bounds[1])
    right = min(off_bounds[2], ref_bounds[2])
    top = min(off_bounds[3], ref_bounds[3])

    if left >= right or bottom >= top:
        raise ValueError("No overlap between the two images")

    center_lon = (left + right) / 2
    center_lat = (bottom + top) / 2
    utm_crs = get_utm_crs_from_lonlat(center_lon, center_lat)
    print(f"  Auto-detected UTM CRS: EPSG:{utm_crs.to_epsg()}")
    return utm_crs


def compute_overlap(src_offset, src_ref, work_crs):
    """Compute the overlap bounding box between two datasets in *work_crs*.

    Returns (left, bottom, right, top) or raises ValueError if no overlap.
    """
    overlap = compute_overlap_or_none(src_offset, src_ref, work_crs)
    if overlap is None:
        raise ValueError("No overlap between the two images")
    return overlap


# ---------------------------------------------------------------------------
# Affine fitting
# ---------------------------------------------------------------------------

def fit_affine_from_gcps(src_points, dst_points, weights=None):
    """Fit a 6-parameter affine transformation from matched point pairs.

    Returns the 2x3 affine matrix M and the per-point residuals in metres.
    """
    n = len(src_points)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        sx, sy = src_points[i]
        dx, dy = dst_points[i]
        A[2 * i] = [sx, sy, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, sx, sy, 1]
        b[2 * i] = dx
        b[2 * i + 1] = dy

    if weights is not None:
        W = np.zeros(2 * n)
        for i in range(n):
            W[2 * i] = weights[i]
            W[2 * i + 1] = weights[i]
        W_sqrt = np.sqrt(W)
        A = A * W_sqrt[:, np.newaxis]
        b = b * W_sqrt

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, bv, tx, c, d, ty = result
    M = np.array([[a, bv, tx], [c, d, ty]])

    residuals = compute_affine_residuals(M, src_points, dst_points)

    return M, residuals


def compute_affine_residuals(M, src_points, dst_points):
    """Compute per-point residuals (metres) for an affine transform."""
    src_arr = np.asarray(src_points)
    dst_arr = np.asarray(dst_points)
    pred_x = M[0, 0] * src_arr[:, 0] + M[0, 1] * src_arr[:, 1] + M[0, 2]
    pred_y = M[1, 0] * src_arr[:, 0] + M[1, 1] * src_arr[:, 1] + M[1, 2]
    return list(np.sqrt((pred_x - dst_arr[:, 0]) ** 2 + (pred_y - dst_arr[:, 1]) ** 2))


def ransac_affine(src_pts, dst_pts, threshold=None):
    """RANSAC affine estimation wrapping cv2.estimateAffine2D.

    Returns (M, inlier_mask) or (None, None) on failure.
    """
    if threshold is None:
        threshold = _C.RANSAC_REPROJ_THRESHOLD
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold)
    if M is None or inliers is None:
        return None, None
    return M, inliers.ravel().astype(bool)


# ---------------------------------------------------------------------------
# Rasterio I/O helpers
# ---------------------------------------------------------------------------

@contextmanager
def open_pair(offset_path: str, reference_path: str):
    """Context manager that opens offset + reference rasterio datasets."""
    src_off = rasterio.open(offset_path)
    src_ref = rasterio.open(reference_path)
    try:
        yield src_off, src_ref
    finally:
        src_off.close()
        src_ref.close()


def read_overlap_pair(src_offset, src_ref, overlap, work_crs, resolution,
                      coarse_dx=0.0, coarse_dy=0.0):
    """Read overlap regions for both datasets and apply coarse shift.

    Returns (arr_ref, ref_transform, arr_off_shifted, off_transform,
             shift_px_x, shift_py_y).
    """
    from .image import shift_array

    arr_ref, ref_transform = read_overlap_region(
        src_ref, overlap, work_crs, resolution)
    arr_off, off_transform = read_overlap_region(
        src_offset, overlap, work_crs, resolution)

    shift_px_x = int(round(coarse_dx / resolution))
    shift_py_y = int(round(coarse_dy / resolution))

    if shift_px_x == 0 and shift_py_y == 0:
        arr_off_shifted = arr_off
    else:
        arr_off_shifted = shift_array(arr_off, -shift_px_x, -shift_py_y)

    return arr_ref, ref_transform, arr_off_shifted, off_transform, shift_px_x, shift_py_y


# ---------------------------------------------------------------------------
# Boundary GCP generation
# ---------------------------------------------------------------------------

def generate_boundary_gcps(gcps, M_geo, img_width, img_height, spacing_px=500):
    """Generate synthetic GCPs along the image boundary for warp stability."""
    from .types import GCP

    if M_geo is None or len(gcps) < 3:
        return []

    px_coords = np.array([(g.col, g.row) for g in gcps])
    geo_coords = np.array([(g.gx, g.gy) for g in gcps])
    n = len(gcps)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = px_coords[i]
        A[2 * i] = [x, y, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i] = geo_coords[i, 0]
        b[2 * i + 1] = geo_coords[i, 1]
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M_px2geo = np.array([[result[0], result[1], result[2]],
                          [result[3], result[4], result[5]]])

    def px_to_geo(px, py):
        gx = M_px2geo[0, 0] * px + M_px2geo[0, 1] * py + M_px2geo[0, 2]
        gy = M_px2geo[1, 0] * px + M_px2geo[1, 1] * py + M_px2geo[1, 2]
        return gx, gy

    edge_points = []
    # Top edge
    for x in range(0, img_width, spacing_px): edge_points.append((float(x), 0.0))
    if edge_points[-1][0] < img_width - 1:
        edge_points.append((float(img_width - 1), 0.0))
    # Bottom edge
    bottom_start = len(edge_points)
    for x in range(0, img_width, spacing_px): edge_points.append((float(x), float(img_height - 1)))
    if edge_points[-1][0] < img_width - 1:
        edge_points.append((float(img_width - 1), float(img_height - 1)))
    # Left edge
    for y in range(spacing_px, img_height - spacing_px, spacing_px): edge_points.append((0.0, float(y)))
    if len(edge_points) > 0:
        left_ys = [p[1] for p in edge_points if p[0] == 0.0]
        if left_ys and max(left_ys) < img_height - 1 - spacing_px:
            edge_points.append((0.0, float(img_height - 1)))
    # Right edge
    for y in range(spacing_px, img_height - spacing_px, spacing_px): edge_points.append((float(img_width - 1), float(y)))
    if len(edge_points) > 0:
        right_ys = [p[1] for p in edge_points if p[0] == float(img_width - 1)]
        if right_ys and max(right_ys) < img_height - 1 - spacing_px:
            edge_points.append((float(img_width - 1), float(img_height - 1)))
    # Ensure all four corners are included
    corners = [(0.0, 0.0), (float(img_width - 1), 0.0), (0.0, float(img_height - 1)), (float(img_width - 1), float(img_height - 1))]
    for c in corners:
        if c not in edge_points: edge_points.append(c)

    real_px = np.array([(g.col, g.row) for g in gcps])
    boundary_gcps = []
    min_dist = spacing_px * 0.5
    for px, py in edge_points:
        dists = np.sqrt((real_px[:, 0] - px) ** 2 + (real_px[:, 1] - py) ** 2)
        if np.min(dists) > min_dist:
            gx, gy = px_to_geo(px, py)
            boundary_gcps.append(GCP(col=px, row=py, gx=gx, gy=gy, synthetic=True, source="boundary"))
    return boundary_gcps
