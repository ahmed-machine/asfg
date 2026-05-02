"""Regression tests for the ELoFTR-backed ``detect_offset_at_resolution``.

The shared ``align.scale.eloftr_translation_estimate`` is exercised by
``tests/test_eloftr_coarse_align.py``. These tests cover the thin coarse
wrapper specifically: sign-convention negation, agreement floor, and
the ``(None, None, 0)`` abstain path.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from align import coarse


COARSE_RES = 15.0


class _FakeSrc:
    """Minimal rasterio-like object — coarse wrapper only calls
    read_overlap_region on it, which we monkeypatch."""

    def close(self):
        pass


def _stub_read_overlap_region(monkeypatch, ref_arr, off_arr):
    """Patch read_overlap_region so the first call returns ``off_arr``
    (matches src_offset) and the second returns ``ref_arr`` (src_ref).
    Wrapper calls in that order."""
    seq = [(off_arr, None), (ref_arr, None)]
    iterator = iter(seq)

    def _read(*args, **kwargs):
        return next(iterator)

    monkeypatch.setattr(coarse, "read_overlap_region", _read)


def _stub_eloftr(monkeypatch, return_value):
    """Make the shared eloftr_translation_estimate import a stub."""
    import align.scale as scale
    monkeypatch.setattr(scale, "eloftr_translation_estimate",
                        lambda *args, **kwargs: return_value)


def test_negates_sign_to_match_historic_ncc_convention(monkeypatch):
    """Helper returns dx_m = +500 (shift target east); wrapper must
    return -500 (target is east of reference, downstream translator
    shifts west)."""
    arr = np.ones((100, 100), dtype=np.float32)
    _stub_read_overlap_region(monkeypatch, arr, arr)
    _stub_eloftr(monkeypatch, (500.0, 300.0, 60, 0.95))

    dx, dy, agreement = coarse.detect_offset_at_resolution(
        _FakeSrc(), _FakeSrc(), (0, 0, 1500, 1500), "EPSG:3857", COARSE_RES,
    )

    assert dx == pytest.approx(-500.0)
    assert dy == pytest.approx(-300.0)
    assert agreement == pytest.approx(0.95)


def test_returns_none_when_helper_abstains(monkeypatch):
    arr = np.ones((100, 100), dtype=np.float32)
    _stub_read_overlap_region(monkeypatch, arr, arr)
    _stub_eloftr(monkeypatch, None)

    result = coarse.detect_offset_at_resolution(
        _FakeSrc(), _FakeSrc(), (0, 0, 1500, 1500), "EPSG:3857", COARSE_RES,
    )

    assert result == (None, None, 0)


def test_abstains_when_agreement_below_floor(monkeypatch):
    """ELoFTR returned a translation but with low agreement — the
    wrapper must reject it and return (None, None, 0)."""
    arr = np.ones((100, 100), dtype=np.float32)
    _stub_read_overlap_region(monkeypatch, arr, arr)
    _stub_eloftr(monkeypatch, (1200.0, -800.0, 161, 0.05))

    result = coarse.detect_offset_at_resolution(
        _FakeSrc(), _FakeSrc(), (0, 0, 1500, 1500), "EPSG:3857", COARSE_RES,
    )

    assert result == (None, None, 0)


def test_uses_min_ncc_as_agreement_floor(monkeypatch):
    """A caller-supplied min_ncc tighter than the ELoFTR default lifts
    the floor — agreement above the helper default but below caller's
    floor must still abstain."""
    arr = np.ones((100, 100), dtype=np.float32)
    _stub_read_overlap_region(monkeypatch, arr, arr)
    _stub_eloftr(monkeypatch, (200.0, 100.0, 60, 0.40))

    result = coarse.detect_offset_at_resolution(
        _FakeSrc(), _FakeSrc(), (0, 0, 1500, 1500), "EPSG:3857", COARSE_RES,
        min_ncc=0.50,
    )
    assert result == (None, None, 0)
