"""Tests for the anchor-file preflight used to gate the coarse-offset skip.

A missing / clustered / undersized anchor file must NOT short-circuit the
coarse stage. Only a file that clears both the count and spatial-spread
floors should trigger the skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from align.pipeline import _anchor_file_preflight


def _write_anchors(path: Path, gcps) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gcps": list(gcps)}))
    return path


def _grid_gcps(rows: int, cols: int, *, lon0=50.5, lat0=26.0, step=0.05):
    return [
        {
            "name": f"g_{r}_{c}",
            "lon": lon0 + c * step,
            "lat": lat0 + r * step,
            "feature_type": "auto_match",
            "confidence": "medium",
        }
        for r in range(rows)
        for c in range(cols)
    ]


def test_missing_file_fails(tmp_path):
    ok, reason = _anchor_file_preflight(str(tmp_path / "nope.json"))
    assert not ok
    assert reason == "missing"


def test_empty_file_fails(tmp_path):
    path = _write_anchors(tmp_path / "a.json", [])
    ok, reason = _anchor_file_preflight(str(path), min_anchors=6)
    assert not ok
    assert "count=0" in reason


def test_clustered_anchors_fail(tmp_path):
    # 8 anchors covering 2×4 but spread wide apart via one outlier, so the
    # bbox is non-degenerate yet all but one anchor share the same cell.
    gcps = [
        {"name": f"n_{i}", "lon": 50.5 + i * 1e-4, "lat": 26.0 + (i % 2) * 1e-4}
        for i in range(7)
    ]
    gcps.append({"name": "far", "lon": 50.5 + 0.30, "lat": 26.0 + 0.30})
    path = _write_anchors(tmp_path / "cluster.json", gcps)
    ok, reason = _anchor_file_preflight(str(path), min_anchors=6,
                                        min_rows=2, min_cols=2, grid_cells=4)
    assert not ok
    assert "clustered" in reason


def test_degenerate_bbox_flagged(tmp_path):
    # All anchors in a single row → lat_hi == lat_lo → degenerate
    gcps = _grid_gcps(1, 8, step=1e-5)
    path = _write_anchors(tmp_path / "degenerate.json", gcps)
    ok, reason = _anchor_file_preflight(str(path), min_anchors=6)
    assert not ok
    assert reason == "degenerate_bbox"


def test_spread_anchors_pass(tmp_path):
    # 8 anchors spanning 4 rows × 2 cols over ~0.3° — all different cells.
    gcps = _grid_gcps(4, 2, step=0.05)
    path = _write_anchors(tmp_path / "spread.json", gcps)
    ok, reason = _anchor_file_preflight(str(path), min_anchors=6,
                                        min_rows=2, min_cols=2, min_cells=4,
                                        grid_cells=4)
    assert ok, f"expected pass, got: {reason}"
    assert reason.startswith("ok:")


def test_malformed_file_fails(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    ok, reason = _anchor_file_preflight(str(path))
    assert not ok
    assert reason.startswith("unreadable:")


def test_missing_coords_fail(tmp_path):
    # Correct count but no lon/lat keys → degenerate
    gcps = [{"name": f"n_{i}"} for i in range(8)]
    path = _write_anchors(tmp_path / "nokeys.json", gcps)
    ok, reason = _anchor_file_preflight(str(path), min_anchors=6)
    assert not ok
    assert reason == "missing_coords"
