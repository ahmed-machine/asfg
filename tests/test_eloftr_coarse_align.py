"""Tests for the ELoFTR-based coarse-align translation estimator.

We don't load real ELoFTR weights here; instead we monkeypatch
``align.scale._run_eloftr_batch`` to return synthetic keypoint pairs and
verify the helper's translation math + abstain gates.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from preprocess import georef


COARSE_RES = 50.0  # metres per pixel — matches default in coarse_align_and_crop


def _stub_cache() -> SimpleNamespace:
    """A ModelCache-shaped stand-in: only `eloftr`, `device`, and `close`
    are touched by the helper."""
    return SimpleNamespace(eloftr="stub-model", device="cpu", close=lambda: None)


def _make_canvas(shape=(120, 120), nonzero_frac=0.6, seed=0) -> np.ndarray:
    """A mostly-non-zero array so the validity gate (>5% nonzero) passes."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(50, 256, size=shape, dtype=np.int32).astype(np.float32)
    n_zero = int(arr.size * (1.0 - nonzero_frac))
    if n_zero > 0:
        flat_idx = rng.choice(arr.size, n_zero, replace=False)
        arr.flat[flat_idx] = 0
    return arr


def _patch_eloftr_runner(monkeypatch, batch_results):
    """Replace align.scale._run_eloftr_batch with a stub returning the
    given results. Order in `batch_results` must match the order of tiles
    the helper produces."""
    def _stub(tiles, model, device, batch_size=4):
        # Return one (kp_ref, kp_tgt, conf) per tile; pad with empty if
        # the caller produced more tiles than fixtures.
        results = list(batch_results)
        while len(results) < len(tiles):
            results.append((np.zeros((0, 2), dtype=np.float32),
                            np.zeros((0, 2), dtype=np.float32),
                            np.zeros((0,), dtype=np.float32)))
        return results
    monkeypatch.setattr("align.scale._run_eloftr_batch", _stub)


def _make_pairs(n: int, dx_px: float, dy_px: float, mad_px: float = 0.0,
                conf: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate n synthetic match pairs with the given pixel translation
    and optional MAD jitter. Per-match: kp_tgt = kp_ref - (dx_px, -dy_px)
    so that the helper's median (kp_ref - kp_tgt) recovers dx_px in
    columns and (kp_tgt - kp_ref) recovers dy_px in rows.
    """
    rng = np.random.default_rng(0)
    kp_ref = rng.uniform(50, 100, size=(n, 2)).astype(np.float32)
    kp_tgt = kp_ref.copy()
    kp_tgt[:, 0] -= dx_px        # ref - tgt = +dx_px
    kp_tgt[:, 1] += dy_px        # tgt - ref = +dy_px (rows down)
    if mad_px > 0:
        # Add Laplace noise so MAD ≈ mad_px / ln(2)
        noise = rng.laplace(0, mad_px / np.log(2), size=kp_ref.shape).astype(np.float32)
        kp_ref += noise
    confs = np.full(n, conf, dtype=np.float32)
    return kp_ref.astype(np.float32), kp_tgt.astype(np.float32), confs


def test_recovers_perfect_translation(monkeypatch):
    """Synthetic +10 px east, +6 px north translation → dx_m = +500 m,
    dy_m = +300 m at coarse_res=50."""
    arr_ref = _make_canvas()
    arr_tgt = _make_canvas(seed=1)
    pairs = _make_pairs(n=60, dx_px=10.0, dy_px=6.0, mad_px=0.0)
    _patch_eloftr_runner(monkeypatch, [pairs])

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is not None
    dx_m, dy_m, n, agreement = result
    assert dx_m == pytest.approx(500.0, abs=1.0)
    assert dy_m == pytest.approx(300.0, abs=1.0)
    assert n == 60
    assert agreement > 0.9


def test_abstains_when_too_few_matches(monkeypatch):
    """Fewer than _ELOFTR_MIN_MATCHES confident pairs → None."""
    arr_ref = _make_canvas()
    arr_tgt = _make_canvas(seed=1)
    pairs = _make_pairs(n=5, dx_px=10.0, dy_px=0.0)
    _patch_eloftr_runner(monkeypatch, [pairs])

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is None


def test_high_mad_returns_low_agreement(monkeypatch):
    """Wide dispersion → agreement below the 0.30 floor.
    Under the N-corrected metric, large MAD is forgiven for high N (the
    median is still precise). To trigger abstain we need SE_px > 2.3,
    i.e. (MAD_x + MAD_y) * 1.4826 / sqrt(N) > 2.3. With N=60 that means
    each axis MAD > ~6 px; this test uses MAD ≈ 15 px so the floor
    fires regardless of count."""
    arr_ref = _make_canvas()
    arr_tgt = _make_canvas(seed=1)
    pairs = _make_pairs(n=60, dx_px=10.0, dy_px=0.0, mad_px=15.0)
    _patch_eloftr_runner(monkeypatch, [pairs])

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is not None
    _, _, _, agreement = result
    assert agreement < georef._ELOFTR_MIN_AGREEMENT


def test_high_n_overcomes_moderate_mad(monkeypatch):
    """The N-correction credits high match counts: 500 matches with
    MAD ≈ 5 px (raw 1/(1+10)=0.09) clear the floor because SE ≈ 0.66 px
    → agreement ≈ 0.60."""
    arr_ref = _make_canvas()
    arr_tgt = _make_canvas(seed=1)
    pairs = _make_pairs(n=500, dx_px=10.0, dy_px=0.0, mad_px=5.0)
    _patch_eloftr_runner(monkeypatch, [pairs])

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is not None
    _, _, _, agreement = result
    assert agreement >= georef._ELOFTR_MIN_AGREEMENT


def test_filters_low_confidence_matches(monkeypatch):
    """Confidence <= conf_min are dropped before the count gate."""
    arr_ref = _make_canvas()
    arr_tgt = _make_canvas(seed=1)
    # 50 pairs at conf=0.05 (below the 0.20 default floor)
    pairs = _make_pairs(n=50, dx_px=10.0, dy_px=0.0, conf=0.05)
    _patch_eloftr_runner(monkeypatch, [pairs])

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is None


def test_shape_mismatch_returns_none(monkeypatch):
    """Helper requires a shared canvas; mismatched shapes → None."""
    arr_ref = _make_canvas(shape=(120, 120))
    arr_tgt = _make_canvas(shape=(150, 110))
    _patch_eloftr_runner(monkeypatch, [])

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is None


def test_insufficient_content_returns_none(monkeypatch):
    """Mostly-zero canvas → bypasses the content gate before reaching
    the matcher."""
    arr_ref = _make_canvas(nonzero_frac=0.01)
    arr_tgt = _make_canvas(nonzero_frac=0.01, seed=2)
    _patch_eloftr_runner(monkeypatch, [])

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is None


def test_canvas_too_small_returns_none(monkeypatch):
    """Canvases below the 64 px floor cannot host a meaningful tile."""
    arr_ref = _make_canvas(shape=(50, 50))
    arr_tgt = _make_canvas(shape=(50, 50), seed=1)
    _patch_eloftr_runner(monkeypatch, [])

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is None


def test_tiles_a_large_canvas(monkeypatch):
    """Large canvas (>tile_px) is split into multiple tiles; matches
    aggregate across tiles. We assert the helper survives multiple-tile
    inputs and recovers the same translation."""
    arr_ref = _make_canvas(shape=(1500, 1500))
    arr_tgt = _make_canvas(shape=(1500, 1500), seed=1)
    pairs = _make_pairs(n=40, dx_px=10.0, dy_px=0.0)
    # Reuse the same pair set for every tile the helper produces
    _patch_eloftr_runner(monkeypatch, [pairs] * 64)

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is not None
    dx_m, dy_m, n, _ = result
    assert dx_m == pytest.approx(500.0, abs=1.0)
    assert n >= 40  # at least one tile contributed (in practice many)


def test_inference_failure_returns_none(monkeypatch):
    """If _run_eloftr_batch raises, the helper abstains rather than
    crashing the preprocess pipeline."""
    arr_ref = _make_canvas()
    arr_tgt = _make_canvas(seed=1)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated MPS OOM")

    monkeypatch.setattr("align.scale._run_eloftr_batch", _boom)

    result = georef._eloftr_translation_estimate(
        arr_ref, arr_tgt, COARSE_RES, model_cache=_stub_cache(),
    )
    assert result is None
