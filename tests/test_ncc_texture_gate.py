"""Tests for the Laplacian-variance gate on NCC fallback patches.

Featureless patches (open sea, flat haze) must not pass the NCC worker
even when the land-mask fraction check is permissive.
"""

from __future__ import annotations

import numpy as np
import pytest

from align import pipeline


def _flat(shape=(64, 64), value=128) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def _textured(shape=(64, 64), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_texture_var_flat_patch_zero():
    assert pipeline._ncc_patch_texture_var(_flat()) == pytest.approx(0.0)


def test_texture_var_textured_patch_exceeds_floor():
    assert pipeline._ncc_patch_texture_var(_textured()) > \
        pipeline._NCC_MIN_SEARCH_TEXTURE_VAR


def test_texture_var_tiny_patch_returns_zero():
    assert pipeline._ncc_patch_texture_var(_textured(shape=(4, 4))) == 0.0


def test_template_floor_is_stricter_than_search_floor():
    # The template floor should be stricter than the search floor:
    # the reference side is what we're correlating against, so a weak
    # reference is more dangerous than a weak search.
    assert pipeline._NCC_MIN_TEMPLATE_TEXTURE_VAR > pipeline._NCC_MIN_SEARCH_TEXTURE_VAR
