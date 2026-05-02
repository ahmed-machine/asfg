"""Tests for the land / texture / spread gates in auto_anchors.

RoMa matching is expensive and not unit-testable here, so we exercise the
pure gating helpers directly. The full `generate_auto_anchors` entry point
is covered by integration / regression runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocess import auto_anchors


def _make_textured_patch(shape=(80, 80), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _make_flat_patch(shape=(80, 80), value=128) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_laplacian_variance_flat_is_zero():
    assert auto_anchors._laplacian_variance_u8(_make_flat_patch()) == pytest.approx(0.0)


def test_laplacian_variance_textured_is_large():
    variance = auto_anchors._laplacian_variance_u8(_make_textured_patch())
    assert variance > 100.0


def test_patch_is_valid_rejects_flat_target():
    flat = _make_flat_patch()
    textured = _make_textured_patch()
    stable = np.ones_like(flat, dtype=np.float32)
    cx = cy = 40
    assert not auto_anchors._anchor_patch_is_valid(
        cx, cy, cx, cy,
        tgt_u8=flat, ref_u8=textured,
        tgt_stable=stable, ref_stable=stable,
        patch_half=20, min_stable_frac=0.25, min_texture_var=50.0,
    )


def test_patch_is_valid_rejects_flat_reference():
    flat = _make_flat_patch()
    textured = _make_textured_patch()
    stable = np.ones_like(flat, dtype=np.float32)
    cx = cy = 40
    assert not auto_anchors._anchor_patch_is_valid(
        cx, cy, cx, cy,
        tgt_u8=textured, ref_u8=flat,
        tgt_stable=stable, ref_stable=stable,
        patch_half=20, min_stable_frac=0.25, min_texture_var=50.0,
    )


def test_patch_is_valid_rejects_water_stable_mask():
    textured = _make_textured_patch()
    water_mask = np.zeros_like(textured, dtype=np.float32)
    cx = cy = 40
    assert not auto_anchors._anchor_patch_is_valid(
        cx, cy, cx, cy,
        tgt_u8=textured, ref_u8=textured,
        tgt_stable=water_mask, ref_stable=np.ones_like(water_mask),
        patch_half=20, min_stable_frac=0.25, min_texture_var=50.0,
    )


def test_patch_is_valid_passes_textured_land():
    textured_t = _make_textured_patch(seed=1)
    textured_r = _make_textured_patch(seed=2)
    land_mask = np.ones_like(textured_t, dtype=np.float32)
    cx = cy = 40
    assert auto_anchors._anchor_patch_is_valid(
        cx, cy, cx, cy,
        tgt_u8=textured_t, ref_u8=textured_r,
        tgt_stable=land_mask, ref_stable=land_mask,
        patch_half=20, min_stable_frac=0.25, min_texture_var=50.0,
    )


def test_patch_is_valid_without_stable_mask_uses_texture_only():
    textured_t = _make_textured_patch(seed=3)
    textured_r = _make_textured_patch(seed=4)
    cx = cy = 40
    assert auto_anchors._anchor_patch_is_valid(
        cx, cy, cx, cy,
        tgt_u8=textured_t, ref_u8=textured_r,
        tgt_stable=None, ref_stable=None,
        patch_half=20, min_stable_frac=0.25, min_texture_var=50.0,
    )


def test_patch_is_valid_rejects_patch_outside_raster():
    textured = _make_textured_patch()
    land = np.ones_like(textured, dtype=np.float32)
    # A center far outside the raster → patch window too small.
    assert not auto_anchors._anchor_patch_is_valid(
        9999, 9999, 9999, 9999,
        tgt_u8=textured, ref_u8=textured,
        tgt_stable=land, ref_stable=land,
        patch_half=20, min_stable_frac=0.25, min_texture_var=50.0,
    )
