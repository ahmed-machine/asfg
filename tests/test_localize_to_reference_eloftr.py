"""Regression tests for the ELoFTR-backed ``localize_to_reference``.

The function tiles the reference image into ELoFTR-friendly windows and
runs (target, ref_tile) pairs through ``_run_eloftr_batch``. Tiles whose
matches pass the agreement floor become hypotheses, ranked by
``agreement * sqrt(matches)``. We mock ``_run_eloftr_batch`` so the test
doesn't load real weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from align import coarse


class _FakeSrc:
    """Minimal rasterio-like dataset for the localize_to_reference path.
    The real function only calls dataset_bounds_in_crs / _read_bounds /
    _read_raw_resized on the src, all of which we monkeypatch.
    """

    def __init__(self, bounds=(0.0, 0.0, 1500.0, 1500.0)):
        self._bounds = bounds
        self.crs = SimpleNamespace(is_geographic=False, to_string=lambda: "EPSG:3857")

    def close(self):
        pass


def _stub_helpers(monkeypatch, ref_arr, tgt_arr,
                  ref_bounds=(0.0, 0.0, 1500.0, 1500.0),
                  target_bounds=(100.0, 100.0, 700.0, 700.0)):
    """Wire dataset_bounds_in_crs, _read_bounds, _read_raw_resized,
    _adaptive_res, and _prepare_search_bounds to deterministic stubs."""
    monkeypatch.setattr(coarse, "dataset_bounds_in_crs", lambda src, crs: src._bounds)
    monkeypatch.setattr(coarse, "_adaptive_res", lambda *a, **kw: 5.0)
    monkeypatch.setattr(coarse, "_prepare_search_bounds",
                        lambda rb, pri, wc, explicit_bounds=None: explicit_bounds or rb)

    seq = iter([(ref_arr, None), (tgt_arr, None)])

    def _read_bounds(src, bounds, crs, res):
        return next(seq)

    monkeypatch.setattr(coarse, "_read_bounds", _read_bounds)
    monkeypatch.setattr(coarse, "_read_raw_resized",
                        lambda src, bounds, res: tgt_arr)


def _patch_eloftr_runner(monkeypatch, results_per_tile):
    def _runner(tiles, model, device, batch_size=4):
        # Pad with empty results if caller produced more tiles than fixtures.
        out = list(results_per_tile)
        while len(out) < len(tiles):
            out.append((np.zeros((0, 2), dtype=np.float32),
                        np.zeros((0, 2), dtype=np.float32),
                        np.zeros((0,), dtype=np.float32)))
        return out

    monkeypatch.setattr("align.scale._run_eloftr_batch", _runner)


def _stub_cache():
    """Stand-in for ModelCache — only `eloftr`, `device`, `close` are touched."""
    return SimpleNamespace(eloftr="stub-model", device="cpu", close=lambda: None)


def _make_pairs(n=60, dx_px=0.0, dy_px=0.0, mad_px=0.0, conf=0.8, seed=0):
    rng = np.random.default_rng(seed)
    kp_ref = rng.uniform(50, 600, size=(n, 2)).astype(np.float32)
    kp_tgt = kp_ref.copy()
    kp_tgt[:, 0] -= dx_px
    kp_tgt[:, 1] += dy_px
    if mad_px > 0:
        noise = rng.laplace(0, mad_px / np.log(2), size=kp_ref.shape).astype(np.float32)
        kp_ref += noise
    return kp_ref, kp_tgt, np.full(n, conf, dtype=np.float32)


def test_returns_empty_when_no_tiles_match(monkeypatch):
    """All tiles produce empty match sets → empty hypothesis list."""
    ref_arr = np.full((1500, 1500), 200, dtype=np.uint8)
    tgt_arr = np.full((400, 400), 200, dtype=np.uint8)
    _stub_helpers(monkeypatch, ref_arr, tgt_arr)
    _patch_eloftr_runner(monkeypatch, [])

    out = coarse.localize_to_reference(
        _FakeSrc(), _FakeSrc(), work_crs="EPSG:3857",
        search_bounds=(0.0, 0.0, 1500.0, 1500.0),
        model_cache=_stub_cache(),
    )
    assert out == []


def test_returns_hypothesis_with_eloftr_matches(monkeypatch):
    """A single tile produces 60 confident matches at agreement≈1.0 →
    one GlobalHypothesis with that score."""
    ref_arr = np.tile(np.arange(0, 256, dtype=np.uint8), (1500, 6))[:1500, :1500]
    tgt_arr = np.tile(np.arange(0, 256, dtype=np.uint8), (400, 2))[:400, :400]
    _stub_helpers(monkeypatch, ref_arr, tgt_arr)

    pairs = _make_pairs(n=60, dx_px=0.0, dy_px=0.0, mad_px=0.0)
    _patch_eloftr_runner(monkeypatch, [pairs])

    out = coarse.localize_to_reference(
        _FakeSrc(), _FakeSrc(), work_crs="EPSG:3857",
        top_k=1,
        search_bounds=(0.0, 0.0, 1500.0, 1500.0),
        model_cache=_stub_cache(),
    )
    assert len(out) == 1
    hyp = out[0]
    assert hyp.source == "global_template_search"
    assert hyp.scale_hint == 1.0
    assert hyp.rotation_hint_deg == 0.0
    assert hyp.diagnostics["n_matches"] == 60
    assert hyp.diagnostics["agreement"] > 0.9


def test_drops_low_agreement_tile(monkeypatch):
    """A tile with high MAD → agreement below floor → dropped."""
    ref_arr = np.tile(np.arange(0, 256, dtype=np.uint8), (1500, 6))[:1500, :1500]
    tgt_arr = np.tile(np.arange(0, 256, dtype=np.uint8), (400, 2))[:400, :400]
    _stub_helpers(monkeypatch, ref_arr, tgt_arr)

    # Wide dispersion → agreement = 1/(1 + ~10) ≈ 0.09 < 0.30 floor
    pairs = _make_pairs(n=60, dx_px=0.0, dy_px=0.0, mad_px=5.0)
    _patch_eloftr_runner(monkeypatch, [pairs])

    out = coarse.localize_to_reference(
        _FakeSrc(), _FakeSrc(), work_crs="EPSG:3857",
        search_bounds=(0.0, 0.0, 1500.0, 1500.0),
        model_cache=_stub_cache(),
    )
    assert out == []


def test_top_k_limits_returned_hypotheses(monkeypatch):
    """Multiple high-scoring tiles → only top_k returned, sorted by
    score (agreement * sqrt(matches)) descending."""
    ref_arr = np.tile(np.arange(0, 256, dtype=np.uint8), (1500, 6))[:1500, :1500]
    tgt_arr = np.tile(np.arange(0, 256, dtype=np.uint8), (400, 2))[:400, :400]
    _stub_helpers(monkeypatch, ref_arr, tgt_arr)

    # Provide identical match sets across many tiles; the function
    # should rank them and cap at top_k=2.
    pairs = _make_pairs(n=60, dx_px=0.0, dy_px=0.0, mad_px=0.0)
    _patch_eloftr_runner(monkeypatch, [pairs] * 16)

    out = coarse.localize_to_reference(
        _FakeSrc(), _FakeSrc(), work_crs="EPSG:3857",
        top_k=2,
        search_bounds=(0.0, 0.0, 1500.0, 1500.0),
        model_cache=_stub_cache(),
    )
    assert len(out) == 2
    assert out[0].score >= out[1].score
