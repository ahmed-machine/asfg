"""Shared rasterio I/O utilities for the alignment pipeline."""

from contextlib import contextmanager

import numpy as np
import rasterio

from .geo import read_overlap_region
from .image import shift_array


@contextmanager
def open_pair(offset_path: str, reference_path: str):
    """Context manager that opens offset + reference rasterio datasets.

    Guarantees both datasets are closed on exit, even on exception.

    Usage::

        with open_pair(state.current_input, state.reference_path) as (src_off, src_ref):
            ...
    """
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
