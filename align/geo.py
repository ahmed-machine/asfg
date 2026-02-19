"""GeoTIFF I/O, overlap computation, and CRS helpers."""

import numpy as np
import rasterio
import rasterio.transform
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform, transform_bounds

from .models import ModelCache, get_torch_device


def get_native_resolution_m(src):
    """Compute the ground resolution of a rasterio dataset in meters.

    For geographic CRS (EPSG:4326), converts degrees to meters using
    cosine approximation at centre latitude.
    For projected CRS, returns pixel spacing directly.
    """
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

    return transform_bounds(src.crs, target_crs, *src.bounds)


def compute_overlap_or_none(src_offset, src_ref, work_crs):
    """Return overlap bounds or ``None`` when the rasters do not overlap."""

    offset_bounds = dataset_bounds_in_crs(src_offset, work_crs)
    ref_bounds = dataset_bounds_in_crs(src_ref, work_crs)
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

    wgs84 = CRS.from_epsg(4326)
    left, bottom, right, top = transform_bounds(src.crs, wgs84, *src.bounds)
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
    """Auto-detect the UTM CRS from the center of the overlap region.

    Computes overlap in EPSG:4326, finds the center lon/lat, and returns
    the corresponding UTM zone CRS.
    """
    wgs84 = CRS.from_epsg(4326)
    off_bounds = transform_bounds(src_offset.crs, wgs84, *src_offset.bounds)
    ref_bounds = transform_bounds(src_ref.crs, wgs84, *src_ref.bounds)

    # Overlap in lon/lat
    left = max(off_bounds[0], ref_bounds[0])
    bottom = max(off_bounds[1], ref_bounds[1])
    right = min(off_bounds[2], ref_bounds[2])
    top = min(off_bounds[3], ref_bounds[3])

    if left >= right or bottom >= top:
        raise ValueError("No overlap between the two images")

    center_lon = (left + right) / 2
    center_lat = (bottom + top) / 2

    # UTM zone number
    zone = int((center_lon + 180) / 6) + 1
    # North or South hemisphere
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone

    utm_crs = CRS.from_epsg(epsg)
    print(f"  Auto-detected UTM CRS: EPSG:{epsg} (zone {zone}{'N' if center_lat >= 0 else 'S'})")
    return utm_crs


def compute_overlap(src_offset, src_ref, work_crs):
    """Compute the overlap bounding box between two datasets in *work_crs*.

    Returns (left, bottom, right, top) or raises ValueError if no overlap.
    """
    overlap = compute_overlap_or_none(src_offset, src_ref, work_crs)
    if overlap is None:
        raise ValueError("No overlap between the two images")
    return overlap
