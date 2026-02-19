"""Debug visualization: side-by-side GCP overlay image."""

import os

import cv2
import numpy as np
import rasterio
import rasterio.transform

from .geo import read_overlap_region
from .image import to_u8


def generate_debug_image(src_ref, src_offset, overlap, work_crs,
                         matched_pairs, geo_residuals, mean_residual,
                         output_path):
    """Generate a diagnostic side-by-side JPEG showing GCP positions.

    Anchors are drawn as yellow crosses, good matches as green dots,
    higher-residual matches as orange dots.
    """
    try:
        vis_res = 15.0
        arr_ref_vis, ref_vis_transform = read_overlap_region(
            src_ref, overlap, work_crs, vis_res)
        arr_off_vis, off_vis_transform = read_overlap_region(
            src_offset, overlap, work_crs, vis_res)

        ref_rgb = cv2.cvtColor(to_u8(arr_ref_vis), cv2.COLOR_GRAY2BGR)
        off_rgb = cv2.cvtColor(to_u8(arr_off_vis), cv2.COLOR_GRAY2BGR)

        for i, pair in enumerate(matched_pairs):
            rgx, rgy, ogx, ogy = pair[0], pair[1], pair[2], pair[3]
            name = pair[5]
            residual = geo_residuals[i]

            ref_row, ref_col = rasterio.transform.rowcol(ref_vis_transform, rgx, rgy)
            off_row, off_col = rasterio.transform.rowcol(off_vis_transform, ogx, ogy)
            ref_row, ref_col = int(ref_row), int(ref_col)
            off_row, off_col = int(off_row), int(off_col)

            if not (0 <= ref_row < ref_rgb.shape[0] and
                    0 <= ref_col < ref_rgb.shape[1] and
                    0 <= off_row < off_rgb.shape[0] and
                    0 <= off_col < off_rgb.shape[1]):
                continue

            if name.startswith("anchor:"):
                color = (0, 255, 255)
                radius = 5
            elif residual < mean_residual:
                color = (0, 255, 0)
                radius = 3
            else:
                color = (0, 165, 255)
                radius = 3

            cv2.circle(ref_rgb, (ref_col, ref_row), radius, color, -1)
            cv2.circle(ref_rgb, (ref_col, ref_row), radius + 2, (255, 255, 255), 1)
            cv2.circle(off_rgb, (off_col, off_row), radius, color, -1)
            cv2.circle(off_rgb, (off_col, off_row), radius + 2, (255, 255, 255), 1)

            if name.startswith("anchor:"):
                cv2.line(ref_rgb, (ref_col - 8, ref_row), (ref_col + 8, ref_row), (0, 255, 255), 2)
                cv2.line(ref_rgb, (ref_col, ref_row - 8), (ref_col, ref_row + 8), (0, 255, 255), 2)
                cv2.line(off_rgb, (off_col - 8, off_row), (off_col + 8, off_row), (0, 255, 255), 2)
                cv2.line(off_rgb, (off_col, off_row - 8), (off_col, off_row + 8), (0, 255, 255), 2)

        h_ref, w_ref = ref_rgb.shape[:2]
        h_off, w_off = off_rgb.shape[:2]
        h_max = max(h_ref, h_off)

        if h_ref != h_max:
            scale = h_max / h_ref
            ref_rgb = cv2.resize(ref_rgb, (int(w_ref * scale), h_max))
        if h_off != h_max:
            scale = h_max / h_off
            off_rgb = cv2.resize(off_rgb, (int(w_off * scale), h_max))

        combined = np.hstack([ref_rgb, off_rgb])

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Reference", (10, 30), font, 1.0, (255, 255, 255), 2)
        cv2.putText(combined, "Target (offset)", (ref_rgb.shape[1] + 10, 30),
                    font, 1.0, (255, 255, 255), 2)
        cv2.putText(combined,
                    f"Matches: {len(matched_pairs)} | Mean residual: {mean_residual:.1f}m",
                    (10, combined.shape[0] - 20), font, 0.7, (255, 255, 255), 2)

        legend_y = 60
        cv2.circle(combined, (15, legend_y), 5, (0, 255, 255), -1)
        cv2.putText(combined, "Anchor GCP", (25, legend_y + 5), font, 0.5, (255, 255, 255), 1)
        cv2.circle(combined, (15, legend_y + 25), 3, (0, 255, 0), -1)
        cv2.putText(combined, "Good match", (25, legend_y + 30), font, 0.5, (255, 255, 255), 1)
        cv2.circle(combined, (15, legend_y + 50), 3, (0, 165, 255), -1)
        cv2.putText(combined, "Higher residual", (25, legend_y + 55), font, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  Diagnostic image saved: {output_path}")
    except Exception as e:
        print(f"  WARNING: Could not create diagnostic visualization: {e}")
