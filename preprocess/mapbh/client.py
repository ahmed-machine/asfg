"""HTTP client for the map.mapbh.org XYZ tile server.

Fetches PNG tiles for a layer + bbox + zoom, stitches them into a single
image array, and returns the array along with a rasterio Affine transform
in EPSG:3857. Handles polite rate limiting (4 concurrent / 200 ms throttle),
exponential backoff on 429 / 5xx, and a persistent on-disk cache so re-runs
are free.
"""

from __future__ import annotations

import io
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests


log = logging.getLogger(__name__)


TILE_SIZE = 256
EARTH_CIRCUMFERENCE_M = 2 * math.pi * 6_378_137.0
# map.mapbh.org runs TileServer GL; tiles live under /data/{slug}/{z}/{x}/{y}.{ext}
# Most layers are .png; the 2020 modern mosaic is .jpg.
TILE_URL_BASE = "https://map.mapbh.org/data"
DEFAULT_TILE_EXT = "png"


# ---------------------------------------------------------------------------
# Tile coordinate math
# ---------------------------------------------------------------------------


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Slippy-map XYZ tile index for a lon/lat at a zoom level."""
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lonlat(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Top-left corner lon/lat of the given XYZ tile."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Project lon/lat (degrees, WGS84) to EPSG:3857 metres."""
    R = 6_378_137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def tile_bounds_mercator(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) in EPSG:3857 of the given XYZ tile."""
    half = EARTH_CIRCUMFERENCE_M / 2
    n = 2 ** zoom
    minx = -half + x * (EARTH_CIRCUMFERENCE_M / n)
    maxx = -half + (x + 1) * (EARTH_CIRCUMFERENCE_M / n)
    maxy = half - y * (EARTH_CIRCUMFERENCE_M / n)
    miny = half - (y + 1) * (EARTH_CIRCUMFERENCE_M / n)
    return minx, miny, maxx, maxy


def covering_tiles(
    bbox_lonlat: tuple[float, float, float, float], zoom: int
) -> tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) tile range covering the bbox."""
    lon_w, lat_s, lon_e, lat_n = bbox_lonlat
    x_min, y_min = lonlat_to_tile(lon_w, lat_n, zoom)  # NW corner
    x_max, y_max = lonlat_to_tile(lon_e, lat_s, zoom)  # SE corner
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class FetchManifest:
    """Per-call manifest summarising tile fetches."""
    n_total: int
    n_cached: int
    n_fetched: int
    n_failed: int
    failed_tiles: list[tuple[int, int]]


class MapbhClient:
    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        max_concurrent: int = 4,
        per_request_throttle_s: float = 0.2,
        max_retries: int = 3,
        timeout_s: float = 30.0,
        user_agent: str = "declass-process/lora",
    ) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "declass-process" / "mapbh"
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.per_request_throttle_s = per_request_throttle_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.user_agent = user_agent
        self._throttle_lock = threading.Lock()
        self._last_request_t = 0.0

    # ------------------------------------------------------------------
    # Tile-level helpers
    # ------------------------------------------------------------------

    def _tile_extension(self, slug: str) -> str:
        """Resolve a layer's tile file extension via the layer table."""
        try:
            from .layers import get_layer
            layer = get_layer(slug)
            if layer is not None and layer.tile_ext:
                return layer.tile_ext
        except Exception:
            pass
        return DEFAULT_TILE_EXT

    def _tile_url(self, slug: str, zoom: int, x: int, y: int) -> str:
        ext = self._tile_extension(slug)
        return f"{TILE_URL_BASE}/{slug}/{zoom}/{x}/{y}.{ext}"

    def _cache_path(self, slug: str, zoom: int, x: int, y: int) -> Path:
        ext = self._tile_extension(slug)
        return self.cache_dir / slug / str(zoom) / str(x) / f"{y}.{ext}"

    def _wait_throttle(self) -> None:
        with self._throttle_lock:
            now = time.monotonic()
            wait = self.per_request_throttle_s - (now - self._last_request_t)
            if wait > 0:
                time.sleep(wait)
            self._last_request_t = time.monotonic()

    def _fetch_tile_bytes(self, slug: str, zoom: int, x: int, y: int) -> bytes | None:
        """Fetch a single tile, with retry/backoff. Returns ``None`` on persistent failure."""
        url = self._tile_url(slug, zoom, x, y)
        backoff = [1.0, 4.0, 16.0]
        for attempt in range(self.max_retries):
            self._wait_throttle()
            try:
                resp = requests.get(
                    url,
                    timeout=self.timeout_s,
                    headers={"User-Agent": self.user_agent},
                )
            except requests.exceptions.RequestException as e:
                log.debug("tile %s/%d/%d/%d request error: %s", slug, zoom, x, y, e)
                if attempt < self.max_retries - 1:
                    time.sleep(backoff[min(attempt, len(backoff) - 1)])
                continue
            if resp.status_code == 200:
                return resp.content
            if resp.status_code in (204, 404):
                # 204 No Content / 404 Not Found — coverage gap, not retry-worthy
                log.debug("tile %s/%d/%d/%d: %d (no coverage)", slug, zoom, x, y, resp.status_code)
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < self.max_retries - 1:
                    time.sleep(backoff[min(attempt, len(backoff) - 1)])
                    continue
            log.warning(
                "tile %s/%d/%d/%d HTTP %d on attempt %d",
                slug, zoom, x, y, resp.status_code, attempt + 1,
            )
        return None

    def get_tile(self, slug: str, zoom: int, x: int, y: int) -> Optional[np.ndarray]:
        """Return a uint8 (TILE_SIZE, TILE_SIZE, 3) array, or ``None`` if missing."""
        cache_p = self._cache_path(slug, zoom, x, y)
        if cache_p.exists():
            try:
                return _decode_png(cache_p.read_bytes())
            except Exception as e:
                log.warning("cache decode failed for %s: %s; refetching", cache_p, e)
                cache_p.unlink(missing_ok=True)

        data = self._fetch_tile_bytes(slug, zoom, x, y)
        if data is None:
            return None
        try:
            arr = _decode_png(data)
        except Exception as e:
            log.warning("decode failed for %s/%d/%d/%d: %s", slug, zoom, x, y, e)
            return None

        # Persist
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_bytes(data)
        return arr

    def probe_tile(self, slug: str, zoom: int, x: int, y: int) -> bool:
        """Quick coverage check via cache hit or HEAD request. Returns True if a tile exists."""
        cache_p = self._cache_path(slug, zoom, x, y)
        if cache_p.exists():
            return True
        url = self._tile_url(slug, zoom, x, y)
        try:
            self._wait_throttle()
            resp = requests.head(
                url, timeout=self.timeout_s, headers={"User-Agent": self.user_agent}
            )
        except requests.exceptions.RequestException:
            return False
        return resp.status_code == 200  # 204/404/etc → no coverage

    # ------------------------------------------------------------------
    # bbox-level interface
    # ------------------------------------------------------------------

    def fetch_bbox(
        self,
        layer: str,
        bbox: tuple[float, float, float, float],
        zoom: int,
        out_size: tuple[int, int] = (1024, 1024),
        *,
        return_manifest: bool = False,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]] | tuple[np.ndarray, tuple[float, float, float, float], FetchManifest]:
        """Fetch a stitched image for the given bbox at zoom level.

        Returns ``(arr, bbox_3857)``: arr is uint8 (out_size_y, out_size_x, 3)
        in EPSG:3857; bbox_3857 is the actual mercator bbox of the returned
        array (snapped to tile boundaries; may extend beyond the requested
        bbox by up to one tile width).

        With ``return_manifest=True`` a ``FetchManifest`` is also returned
        for diagnostics.
        """
        x_min, y_min, x_max, y_max = covering_tiles(bbox, zoom)
        n_x = x_max - x_min + 1
        n_y = y_max - y_min + 1
        canvas = np.zeros((n_y * TILE_SIZE, n_x * TILE_SIZE, 3), dtype=np.uint8)

        n_total = n_x * n_y
        n_cached = 0
        n_fetched = 0
        n_failed = 0
        failed: list[tuple[int, int]] = []

        # Track which tiles need a network fetch
        targets = []
        for ix in range(n_x):
            for iy in range(n_y):
                xt = x_min + ix
                yt = y_min + iy
                if self._cache_path(layer, zoom, xt, yt).exists():
                    n_cached += 1
                targets.append((xt, yt, ix, iy))

        # Fetch (uses cache where possible)
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futs = {
                pool.submit(self.get_tile, layer, zoom, xt, yt): (xt, yt, ix, iy)
                for xt, yt, ix, iy in targets
            }
            for fut in as_completed(futs):
                xt, yt, ix, iy = futs[fut]
                arr = fut.result()
                if arr is None:
                    n_failed += 1
                    failed.append((xt, yt))
                    continue
                n_fetched += 1 if not self._cache_path(layer, zoom, xt, yt).exists() else 0
                # Paint the tile into the canvas; canvas indexing is
                # row-major (y in pixels) with origin at top-left of NW tile.
                top = iy * TILE_SIZE
                left = ix * TILE_SIZE
                canvas[top:top + TILE_SIZE, left:left + TILE_SIZE, :] = arr

        n_fetched = n_total - n_cached - n_failed

        # Compute the canvas's bbox in EPSG:3857 from tile bounds
        nw_minx, _, _, nw_maxy = tile_bounds_mercator(x_min, y_min, zoom)
        _, se_miny, se_maxx, _ = tile_bounds_mercator(x_max, y_max, zoom)
        canvas_bbox_3857 = (nw_minx, se_miny, se_maxx, nw_maxy)

        # Crop to the requested bbox + resample to out_size
        out_w, out_h = out_size
        target_bbox_3857 = (
            *lonlat_to_mercator(bbox[0], bbox[3]),  # NW: minx, maxy
            *lonlat_to_mercator(bbox[2], bbox[1]),  # SE: maxx, miny
        )
        # target_bbox_3857 = (minx, maxy, maxx, miny) at this point — reorder
        target_minx = target_bbox_3857[0]
        target_maxx = target_bbox_3857[2]
        target_miny = target_bbox_3857[3]
        target_maxy = target_bbox_3857[1]

        # Map canvas pixel coords → EPSG:3857
        canvas_h, canvas_w = canvas.shape[:2]
        x_per_px = (canvas_bbox_3857[2] - canvas_bbox_3857[0]) / canvas_w
        y_per_px = (canvas_bbox_3857[3] - canvas_bbox_3857[1]) / canvas_h  # NB: maxy - miny / h

        col_lo = int(round((target_minx - canvas_bbox_3857[0]) / x_per_px))
        col_hi = int(round((target_maxx - canvas_bbox_3857[0]) / x_per_px))
        row_lo = int(round((canvas_bbox_3857[3] - target_maxy) / y_per_px))
        row_hi = int(round((canvas_bbox_3857[3] - target_miny) / y_per_px))

        col_lo = max(0, col_lo)
        row_lo = max(0, row_lo)
        col_hi = min(canvas_w, col_hi)
        row_hi = min(canvas_h, row_hi)
        if col_hi <= col_lo or row_hi <= row_lo:
            raise ValueError(
                f"requested bbox does not intersect fetched tile coverage; "
                f"canvas={canvas_bbox_3857} target_3857=({target_minx},{target_miny},{target_maxx},{target_maxy})"
            )
        cropped = canvas[row_lo:row_hi, col_lo:col_hi, :]

        if cropped.shape[0] != out_h or cropped.shape[1] != out_w:
            try:
                import cv2
                resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
            except ImportError:
                # PIL fallback — slower
                from PIL import Image
                resized = np.asarray(
                    Image.fromarray(cropped).resize((out_w, out_h), Image.LANCZOS)
                )
        else:
            resized = cropped

        out_bbox_3857 = (target_minx, target_miny, target_maxx, target_maxy)
        manifest = FetchManifest(
            n_total=n_total,
            n_cached=n_cached,
            n_fetched=n_fetched,
            n_failed=n_failed,
            failed_tiles=failed,
        )
        if return_manifest:
            return resized, out_bbox_3857, manifest
        return resized, out_bbox_3857

    def has_coverage(
        self,
        layer: str,
        bbox: tuple[float, float, float, float],
        zoom: int,
        *,
        sample_n: int = 4,
    ) -> bool:
        """Heuristic coverage check: probe a few tiles in the bbox; require all to exist."""
        x_min, y_min, x_max, y_max = covering_tiles(bbox, zoom)
        # Sample tiles uniformly across the covering region
        xs = np.linspace(x_min, x_max, max(1, int(math.sqrt(sample_n)))).round().astype(int)
        ys = np.linspace(y_min, y_max, max(1, int(math.sqrt(sample_n)))).round().astype(int)
        for xi in xs:
            for yi in ys:
                if not self.probe_tile(layer, zoom, int(xi), int(yi)):
                    return False
        return True


def _decode_png(data: bytes) -> np.ndarray:
    """Decode PNG bytes to uint8 (H, W, 3) numpy array. Strips alpha if present."""
    try:
        import cv2
        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("cv2 returned None on PNG decode")
        # cv2 returns BGR; convert to RGB for downstream consistency
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img)
