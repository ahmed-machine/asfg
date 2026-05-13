"""Unit tests for the mapbh.org tile client.

Uses ``responses`` if available, falls back to direct monkeypatching of
``requests.get``/``requests.head`` so tests are deterministic and offline.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from preprocess.mapbh.client import (
    MapbhClient,
    TILE_SIZE,
    covering_tiles,
    lonlat_to_tile,
    tile_to_lonlat,
    tile_bounds_mercator,
)
from preprocess.mapbh.layers import (
    LAYER_TABLE,
    cross_temporal_pairs,
    get_layer,
    kh_layers,
    modern_layers,
    _infer_mission_from_slug,
)


# ---------------------------------------------------------------------------
# Tile coordinate math
# ---------------------------------------------------------------------------


def test_lonlat_tile_round_trip():
    # Bahrain ~ (50.55, 26.07); zoom 14 should round-trip into the same tile bin
    lon, lat = 50.55, 26.07
    z = 14
    x, y = lonlat_to_tile(lon, lat, z)
    nw_lon, nw_lat = tile_to_lonlat(x, y, z)
    se_lon, se_lat = tile_to_lonlat(x + 1, y + 1, z)
    assert nw_lon <= lon <= se_lon
    assert se_lat <= lat <= nw_lat


def test_covering_tiles_bahrain_z14():
    bbox = (50.4, 26.0, 50.7, 26.3)  # roughly Bahrain
    x_min, y_min, x_max, y_max = covering_tiles(bbox, zoom=14)
    assert x_max >= x_min
    assert y_max >= y_min
    # At z=14 a 0.3° span at 26°N is ~14 tiles wide (each tile ~ 2.4 km)
    assert 8 <= (x_max - x_min + 1) <= 24
    assert 8 <= (y_max - y_min + 1) <= 24


def test_tile_bounds_mercator_monotonic():
    minx, miny, maxx, maxy = tile_bounds_mercator(0, 0, zoom=1)
    assert maxx > minx
    assert maxy > miny


# ---------------------------------------------------------------------------
# Layer table
# ---------------------------------------------------------------------------


def test_known_layers_classified():
    assert get_layer("1965-DS1022-1024DA").is_kh
    assert get_layer("20141112").is_modern
    assert get_layer("20201216").is_modern


def test_mission_inference():
    assert _infer_mission_from_slug("1965-DS1022-1024DA", 1965) == "kh-4a"
    assert _infer_mission_from_slug("1968-DS1104-1057DA", 1968) == "kh-4b"
    assert _infer_mission_from_slug("1976-KH9-DZB1212", 1976) == "kh-9-mc"
    assert _infer_mission_from_slug("1982-D3C1217", 1982) == "kh-9-pc"
    assert _infer_mission_from_slug("20141112", 2014) == "modern"
    assert _infer_mission_from_slug("1967-Bahrain", 1967) == "unknown"


def test_kh_layers_exclude_modern():
    kh = kh_layers()
    assert all(l.kind == "kh" for l in kh)
    assert all(l.year < 1984 for l in kh)
    assert len(kh) >= 5


def test_modern_layers_only_modern():
    mods = modern_layers()
    assert all(l.kind == "modern" for l in mods)
    assert len(mods) >= 2


def test_cross_temporal_pairs_includes_kh9():
    pairs = cross_temporal_pairs()
    pair_slugs = [(a.slug, b.slug) for a, b in pairs]
    assert any("KH9" in a or "D3C" in a for a, b in pair_slugs), \
        "expected at least one KH-9 layer to appear in cross-temporal pairs"


def test_cross_temporal_pairs_can_exclude():
    pairs = cross_temporal_pairs(exclude_missions=("kh-9-mc", "kh-9-pc"))
    for a, b in pairs:
        assert a.mission not in ("kh-9-mc", "kh-9-pc")
        assert b.mission not in ("kh-9-mc", "kh-9-pc")


# ---------------------------------------------------------------------------
# Client cache + retry behaviour
# ---------------------------------------------------------------------------


def _png_bytes(rgb: tuple[int, int, int]) -> bytes:
    """Encode a TILE_SIZE x TILE_SIZE solid-colour PNG into bytes."""
    import cv2
    arr = np.full((TILE_SIZE, TILE_SIZE, 3), rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return bytes(buf)


def _mock_response(status: int, content: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    return resp


def test_cache_hit_avoids_http(tmp_path: Path):
    client = MapbhClient(cache_dir=tmp_path, per_request_throttle_s=0.0)
    cache_p = client._cache_path("test", 1, 0, 0)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache_p.write_bytes(_png_bytes((10, 20, 30)))

    with patch("preprocess.mapbh.client.requests.get") as mocked:
        arr = client.get_tile("test", 1, 0, 0)
        mocked.assert_not_called()
    assert arr is not None
    assert arr.shape == (TILE_SIZE, TILE_SIZE, 3)
    # cv2 reads BGR but client converts to RGB; verify the colour matches what we wrote
    assert tuple(arr[0, 0].tolist()) == (10, 20, 30)


def test_cache_miss_fetches_and_caches(tmp_path: Path):
    client = MapbhClient(cache_dir=tmp_path, per_request_throttle_s=0.0)
    payload = _png_bytes((100, 50, 25))

    with patch("preprocess.mapbh.client.requests.get") as mocked:
        mocked.return_value = _mock_response(200, payload)
        arr = client.get_tile("test", 1, 0, 0)
        assert mocked.call_count == 1
    assert arr.shape == (TILE_SIZE, TILE_SIZE, 3)
    # Second call hits cache
    with patch("preprocess.mapbh.client.requests.get") as mocked2:
        client.get_tile("test", 1, 0, 0)
        mocked2.assert_not_called()


def test_404_returns_none_no_cache(tmp_path: Path):
    client = MapbhClient(cache_dir=tmp_path, per_request_throttle_s=0.0)
    with patch("preprocess.mapbh.client.requests.get") as mocked:
        mocked.return_value = _mock_response(404, b"")
        arr = client.get_tile("test", 1, 99, 99)
        assert arr is None
    # 404 must not have been cached
    assert not client._cache_path("test", 1, 99, 99).exists()


def test_429_retries_then_succeeds(tmp_path: Path):
    client = MapbhClient(cache_dir=tmp_path, per_request_throttle_s=0.0, max_retries=3)
    payload = _png_bytes((1, 2, 3))
    with patch("preprocess.mapbh.client.requests.get") as mocked, \
            patch("preprocess.mapbh.client.time.sleep") as sleep_mock:
        mocked.side_effect = [
            _mock_response(429, b""),
            _mock_response(429, b""),
            _mock_response(200, payload),
        ]
        arr = client.get_tile("test", 1, 0, 0)
    assert mocked.call_count == 3
    assert arr is not None
    # Backoff sleeps were called
    assert sleep_mock.called


def test_5xx_exhausts_retries_returns_none(tmp_path: Path):
    client = MapbhClient(cache_dir=tmp_path, per_request_throttle_s=0.0, max_retries=2)
    with patch("preprocess.mapbh.client.requests.get") as mocked, \
            patch("preprocess.mapbh.client.time.sleep"):
        mocked.return_value = _mock_response(503, b"")
        arr = client.get_tile("test", 1, 0, 0)
    assert arr is None
    assert mocked.call_count == 2


def test_fetch_bbox_stitches_tiles(tmp_path: Path):
    """Mock 4 tiles covering a bbox; fetch_bbox should produce a single stitched image."""
    client = MapbhClient(cache_dir=tmp_path, per_request_throttle_s=0.0)

    # Use a tiny bbox that maps to a 2x2 tile grid at z=14 in Bahrain
    bbox = (50.4, 26.0, 50.5, 26.1)
    x_min, y_min, x_max, y_max = covering_tiles(bbox, zoom=14)
    n_x = x_max - x_min + 1
    n_y = y_max - y_min + 1

    # Pre-populate cache with distinct colours per tile
    for ix in range(n_x):
        for iy in range(n_y):
            colour = ((ix + 1) * 60, (iy + 1) * 60, 100)
            cache_p = client._cache_path("test", 14, x_min + ix, y_min + iy)
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            cache_p.write_bytes(_png_bytes(colour))

    arr, bbox_3857 = client.fetch_bbox("test", bbox, zoom=14, out_size=(512, 512))
    assert arr.shape == (512, 512, 3)
    assert arr.dtype == np.uint8
    # Image should not be all zeros
    assert arr.mean() > 0
    # bbox_3857 should be a 4-tuple of ordered metres
    minx, miny, maxx, maxy = bbox_3857
    assert maxx > minx
    assert maxy > miny


def test_probe_tile_uses_head(tmp_path: Path):
    client = MapbhClient(cache_dir=tmp_path, per_request_throttle_s=0.0)
    with patch("preprocess.mapbh.client.requests.head") as mocked:
        mocked.return_value = _mock_response(200, b"")
        assert client.probe_tile("test", 1, 0, 0) is True
        mocked.return_value = _mock_response(404, b"")
        assert client.probe_tile("test", 1, 1, 1) is False
