"""Semantic-ish masking utilities for historical grayscale imagery."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.morphology import reconstruction
from skimage.segmentation import felzenszwalb

from .image import clahe_normalize, sobel_gradient, to_u8


@dataclass(slots=True)
class MaskBundle:
    land: np.ndarray
    water: np.ndarray
    shoreline: np.ndarray
    stable: np.ndarray
    shallow_water: np.ndarray
    dark_farmland: np.ndarray
    texture: np.ndarray
    brightness: np.ndarray


def _zero_bundle(shape) -> MaskBundle:
    zero = np.zeros(shape, dtype=np.float32)
    return MaskBundle(
        land=zero,
        water=zero,
        shoreline=zero,
        stable=zero,
        shallow_water=zero,
        dark_farmland=zero,
        texture=zero,
        brightness=zero,
    )


def _resize_field(field: np.ndarray, size, interpolation) -> np.ndarray:
    width, height = size
    return cv2.resize(field.astype(np.float32), (width, height), interpolation=interpolation)


def _resize_bundle(bundle: MaskBundle, shape) -> MaskBundle:
    height, width = shape
    return MaskBundle(
        land=_resize_field(bundle.land, (width, height), cv2.INTER_NEAREST),
        water=_resize_field(bundle.water, (width, height), cv2.INTER_NEAREST),
        shoreline=_resize_field(bundle.shoreline, (width, height), cv2.INTER_NEAREST),
        stable=_resize_field(bundle.stable, (width, height), cv2.INTER_NEAREST),
        shallow_water=_resize_field(bundle.shallow_water, (width, height), cv2.INTER_NEAREST),
        dark_farmland=_resize_field(bundle.dark_farmland, (width, height), cv2.INTER_NEAREST),
        texture=_resize_field(bundle.texture, (width, height), cv2.INTER_LINEAR),
        brightness=_resize_field(bundle.brightness, (width, height), cv2.INTER_LINEAR),
    )


def _postprocess_land(land: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    land_u8 = (land.astype(np.uint8) * 255)
    land_u8 = cv2.morphologyEx(land_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    land_u8 = cv2.morphologyEx(land_u8, cv2.MORPH_OPEN, kernel, iterations=2)
    return land_u8 > 0


def _shoreline_and_stable(land: np.ndarray, grad_norm: np.ndarray, std_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((3, 3), np.uint8)
    shoreline = cv2.morphologyEx(land.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    shoreline = cv2.dilate(shoreline.astype(np.uint8), kernel, iterations=1) > 0
    stable = land & ((grad_norm > 0.22) | (std_norm > 0.14) | shoreline)
    return shoreline, stable


class HeuristicMaskProvider:
    """Texture-aware heuristic masks for historical grayscale frames."""

    name = "heuristic"

    def build_masks(self, arr) -> MaskBundle:
        valid = arr > 0
        u8 = to_u8(arr)
        if not np.any(valid):
            return _zero_bundle(u8.shape)

        blurred = cv2.GaussianBlur(u8, (5, 5), 0)
        try:
            thresholds = threshold_multiotsu(blurred[valid], classes=3)
            class_0 = blurred <= thresholds[0]
            class_1 = (blurred > thresholds[0]) & (blurred <= thresholds[1])
            class_2 = blurred > thresholds[1]
        except ValueError:
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            class_2 = mask > 0
            class_1 = np.zeros_like(class_2)
            class_0 = ~class_2

        grad = sobel_gradient(u8)
        grad_energy = cv2.GaussianBlur(grad, (7, 7), 0)
        avg_energy = float(np.mean(grad_energy[valid])) if np.any(valid) else 0.0
        high_texture = grad_energy > max(avg_energy * 0.8, 3.0)
        very_high_texture = grad_energy > max(avg_energy * 1.5, 5.0)
        smooth = grad_energy < max(avg_energy * 0.3, 1.0)

        local_mean = cv2.GaussianBlur(u8.astype(np.float32), (11, 11), 0)
        brightness = np.zeros_like(local_mean, dtype=np.float32)
        brightness[valid] = local_mean[valid] / 255.0

        water_seed = (class_0 | (class_1 & ~high_texture)) & valid
        near_water = cv2.dilate(water_seed.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        near_water_bright = cv2.dilate(water_seed.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2) > 0

        class2_core = class_2 & ~near_water & ~smooth & valid
        near_class2_core = cv2.dilate(
            class2_core.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=2,
        ) > 0

        land = class_2.copy()
        dark_farmland = class_1 & very_high_texture & near_class2_core & valid
        land = land | dark_farmland
        # Only remove smooth pixels near water (shoals/shallow water).
        # Preserve smooth inland/coastal land (sandy areas, fort walls, etc).
        land = land & ~(smooth & near_water_bright)
        land = land & valid

        bright_shoal = class_2 & near_water_bright & ~very_high_texture & valid
        # Also catch bright, smooth class_2 pixels adjacent to water that
        # lack any strong texture — these are shallow water reflecting light,
        # not actual land.  The original rule required ~very_high_texture which
        # lets moderately-textured shoals through.
        bright_smooth_shoal = (
            class_2 & near_water & smooth & (brightness >= 0.45) & valid
        )
        bright_shoal = bright_shoal | bright_smooth_shoal
        land = land & ~bright_shoal

        shallow_water = (
            (class_1 & ((~high_texture) | near_water))
            | bright_shoal
        ) & ~land & valid
        water = (class_0 | shallow_water | (smooth & ~land)) & valid

        texture = np.zeros_like(grad_energy, dtype=np.float32)
        texture[valid] = grad_energy[valid]
        max_tex = float(texture[valid].max()) if np.any(valid) else 1.0
        if max_tex > 0:
            texture[valid] = texture[valid] / max_tex

        land = _postprocess_land(land)

        shoreline, stable = _shoreline_and_stable(land, texture, texture)
        stable = stable | (land & high_texture)

        return MaskBundle(
            land=land.astype(np.float32),
            water=water.astype(np.float32),
            shoreline=shoreline.astype(np.float32),
            stable=stable.astype(np.float32),
            shallow_water=shallow_water.astype(np.float32),
            dark_farmland=dark_farmland.astype(np.float32),
            texture=texture.astype(np.float32),
            brightness=brightness.astype(np.float32),
        )

    def make_land_mask(self, arr):
        return self.build_masks(arr).land


class CoastalObiaMaskProvider:
    """Hybrid suppression: heuristic base + sea-connected OBIA demotion."""

    name = "coastal_obia"
    max_dim = 2400

    def build_masks(self, arr, _debug_dir: str | None = None) -> MaskBundle:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("CoastalObiaMaskProvider expects a 2D array")

        valid = arr > 0
        if not np.any(valid):
            return _zero_bundle(arr.shape)

        # Step 1: run heuristic at ORIGINAL resolution
        heuristic = HeuristicMaskProvider().build_masks(arr)

        if _debug_dir:
            self._dump_heuristic(heuristic, arr, _debug_dir)

        # Step 2: compute demotion mask (possibly at reduced resolution)
        height, width = arr.shape
        ds_scale = min(1.0, self.max_dim / max(height, width))
        if ds_scale < 1.0:
            new_w = max(32, int(round(width * ds_scale)))
            new_h = max(32, int(round(height * ds_scale)))
            arr_small = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            land_small = cv2.resize(
                heuristic.land, (new_w, new_h), interpolation=cv2.INTER_NEAREST,
            ) > 0.5
            demote_small = self._compute_demotion(arr_small, land_small, scale=ds_scale, _debug_dir=_debug_dir)
            if demote_small is None:
                return heuristic
            # Upsample demotion mask back to original resolution
            demoted = cv2.resize(
                demote_small.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        else:
            land = heuristic.land > 0.5
            demoted = self._compute_demotion(arr, land, scale=1.0, _debug_dir=_debug_dir)
            if demoted is None:
                return heuristic

        # Step 3: apply demotions to the full-res heuristic
        result = self._apply_demotion(heuristic, demoted, arr)

        if _debug_dir:
            self._dump_final_overlay(result, arr, _debug_dir)

        return result

    def _compute_demotion(self, arr: np.ndarray, land: np.ndarray, scale: float = 1.0,
                           _debug_dir: str | None = None):
        """Compute a boolean demotion mask. Returns None if fallback needed.

        Parameters
        ----------
        arr : 2-D float32 array (possibly downscaled).
        land : boolean land mask at the same resolution as *arr*.
        scale : downscale factor applied to reach *arr* (1.0 = full resolution).
            Used to adapt kernel sizes, distance thresholds, and felzenszwalb
            parameters so behaviour is consistent regardless of resolution.
        _debug_dir : if set, dump intermediate arrays as images.
        """
        valid = arr > 0
        u8 = to_u8(arr)

        def _kern(base_k: int) -> int:
            k = max(3, int(round(base_k * scale)))
            return k if k % 2 == 1 else k + 1

        blur_k = _kern(5)
        stat_k = _kern(11)

        blurred = cv2.GaussianBlur(u8, (blur_k, blur_k), 0)
        try:
            thresholds = threshold_multiotsu(blurred[valid], classes=3)
            class_0 = blurred <= thresholds[0]
            class_1 = (blurred > thresholds[0]) & (blurred <= thresholds[1])
        except ValueError:
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            class_0 = mask == 0
            class_1 = np.zeros_like(class_0)

        grad = sobel_gradient(u8)
        grad_energy = cv2.GaussianBlur(grad, (blur_k, blur_k), 0).astype(np.float32)
        grad_norm = np.zeros_like(arr, dtype=np.float32)
        if np.any(valid):
            grad_norm[valid] = grad_energy[valid] / max(float(grad_energy[valid].max()), 1.0)

        local_mean = cv2.GaussianBlur(u8.astype(np.float32), (stat_k, stat_k), 0)
        local_sq_mean = cv2.GaussianBlur((u8.astype(np.float32) ** 2), (stat_k, stat_k), 0)
        local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0.0, None)).astype(np.float32)
        std_norm = np.zeros_like(arr, dtype=np.float32)
        if np.any(valid):
            std_norm[valid] = local_std[valid] / max(float(local_std[valid].max()), 1.0)

        brightness = np.zeros_like(local_mean, dtype=np.float32)
        brightness[valid] = local_mean[valid] / 255.0

        wl_grad_mid = min(0.28, 0.28 * scale)   
        wl_std_mid  = min(0.18, 0.18 * scale)   
        wl_grad_low = min(0.10, 0.10 * scale)   
        wl_std_low  = min(0.08, 0.08 * scale)   

        waterlike = (
            class_0
            | (class_1 & (grad_norm < wl_grad_mid) & (std_norm < wl_std_mid))
            | ((grad_norm < wl_grad_low) & (std_norm < wl_std_low) & (brightness < 0.55))
        ) & valid

        if _debug_dir:
            self._dump_waterlike(arr, waterlike, grad_norm, std_norm, brightness, land, _debug_dir)

        border = np.zeros_like(valid, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        water_seed = waterlike & border

        if int(np.count_nonzero(water_seed)) < max(8, int(np.count_nonzero(valid) * 0.0002)):
            return None

        sea_connected = reconstruction(
            water_seed.astype(np.uint8),
            waterlike.astype(np.uint8),
            method="dilation",
        ) > 0

        if float(np.mean(sea_connected[valid])) < 0.01:
            return None

        dist_to_sea = ndi.distance_transform_edt(~sea_connected)

        if _debug_dir:
            self._dump_sea_distance(arr, sea_connected, dist_to_sea, land, _debug_dir)

        inland_thresh = max(4.0, 20.0 * scale)
        coastal_band_thresh = max(8.0, 40.0 * scale)
        near_sea_thresh = max(5.0, 24.0 * scale)
        close_sea_thresh = max(3.0, 15.0 * scale)
        cc_dist_thresh = max(4.0, 18.0 * scale)

        inland_land = land & (dist_to_sea >= inland_thresh)
        morph_k = max(3, _kern(5))
        reach_per_iter = max(1, morph_k // 2)
        morph_iters = max(1, int(np.ceil((inland_thresh + 2) / reach_per_iter)))
        
        land_support = cv2.dilate(
            inland_land.astype(np.uint8),
            np.ones((morph_k, morph_k), np.uint8),
            iterations=morph_iters,
        ) > 0

        sea_margin_reach = max(3.0, 9.0 * scale)
        sea_margin_iters = max(1, int(np.ceil(sea_margin_reach / reach_per_iter)))
        sea_margin = cv2.dilate(
            sea_connected.astype(np.uint8),
            np.ones((morph_k, morph_k), np.uint8),
            iterations=sea_margin_iters,
        ) > 0

        coastal_band = ((dist_to_sea <= coastal_band_thresh) | sea_margin) & valid
        coastal_area = int(np.count_nonzero(coastal_band))
        min_coastal = max(100, int(round(500 * scale * scale)))
        if coastal_area < min_coastal:
            return None

        seg_features = np.dstack([
            blurred.astype(np.float32) / 255.0,
            grad_norm,
            std_norm,
        ])

        fz_scale = max(20.0, 100.0 * scale)
        fz_sigma = max(0.3, 0.8 * scale)
        min_size = max(8, min(256, int(round(coastal_area * scale / 10000.0 * scale))))
        min_size = max(min_size, int(round(32 * scale * scale)))
        
        segments = felzenszwalb(
            seg_features, scale=fz_scale, sigma=fz_sigma, min_size=min_size,
            channel_axis=-1,
        )

        # Iterate coastal-band land segments and decide demotions
        demoted = np.zeros_like(valid, dtype=bool)
        segment_ids = np.unique(segments[coastal_band & land])

        # Minimum segment size scales quadratically
        min_seg_px = max(2, int(round(4 * scale * scale)))

        # Collect per-segment stats for debug dump
        _seg_stats = [] if _debug_dir else None

        for seg_id in segment_ids:
            seg = (segments == seg_id) & valid
            if int(np.count_nonzero(seg)) < min_seg_px:
                continue
            if float(np.mean(land[seg])) < 0.15:
                continue

            mean_grad = float(np.mean(grad_norm[seg]))
            mean_std = float(np.mean(std_norm[seg]))
            mean_brightness = float(np.mean(brightness[seg]))
            mean_dist = float(np.mean(dist_to_sea[seg]))
            p90_dist = float(np.percentile(dist_to_sea[seg], 90))
            near_sea_frac = float(np.mean(dist_to_sea[seg] <= near_sea_thresh))
            sea_touch = float(np.mean(sea_margin[seg]))
            supported = float(np.mean(land_support[seg]))

            # Rule A: touches sea margin, weak support, low gradient -> shoal
            demote_a = (
                sea_touch >= 0.15
                and supported <= 0.25
                and mean_grad <= 0.35
                and mean_std <= 0.20
            )
            # Rule B: mostly near-sea, unsupported, low texture
            demote_b = (
                near_sea_frac >= 0.50
                and supported <= 0.20
                and mean_dist <= (20.0 * scale)
                and mean_grad <= 0.45
            )
            # Rule C: close to sea, touches sea, unsupported -> reef/shoal
            demote_c = (
                sea_touch >= 0.25
                and mean_dist <= close_sea_thresh
                and supported <= 0.08
            )
            # Rule D: narrow, bright, low-texture coastal ribbons can still be
            # shoals even when they are attached to inland land and therefore
            # appear "supported".  This catches the remaining northwest strip.
            demote_d = (
                sea_touch >= 0.18
                and near_sea_frac >= 0.60
                and p90_dist <= max(6.0, 20.0 * scale)
                and mean_grad <= 0.16
                and mean_std <= 0.18
                and mean_brightness >= 0.50
            )
            # Rule E: bright, ultra-smooth near-sea segments — shoal/shallow water
            # that may appear "supported" due to proximity to mainland
            demote_e = (
                near_sea_frac >= 0.50
                and mean_grad <= 0.10
                and mean_std <= 0.10
                and mean_brightness >= 0.55
            )

            any_demoted = demote_a or demote_b or demote_c or demote_d or demote_e
            if any_demoted:
                demoted[seg] = True

            if _seg_stats is not None:
                # Compute segment centroid for location
                seg_rows, seg_cols = np.where(seg)
                centroid_r = float(np.mean(seg_rows))
                centroid_c = float(np.mean(seg_cols))
                rules = []
                if demote_a: rules.append("A")
                if demote_b: rules.append("B")
                if demote_c: rules.append("C")
                if demote_d: rules.append("D")
                if demote_e: rules.append("E")
                _seg_stats.append(dict(
                    seg_id=int(seg_id),
                    area=int(np.count_nonzero(seg)),
                    centroid_r=centroid_r, centroid_c=centroid_c,
                    quadrant=("NW" if centroid_r < arr.shape[0] / 2 and centroid_c < arr.shape[1] / 2 else
                              "NE" if centroid_r < arr.shape[0] / 2 else
                              "SW" if centroid_c < arr.shape[1] / 2 else "SE"),
                    sea_touch=sea_touch, supported=supported,
                    near_sea_frac=near_sea_frac, mean_dist=mean_dist,
                    p90_dist=p90_dist,
                    mean_grad=mean_grad, mean_std=mean_std,
                    mean_brightness=mean_brightness,
                    demoted=any_demoted,
                    rules=",".join(rules) if rules else "-",
                ))

        if _debug_dir and _seg_stats is not None:
            self._dump_segments_and_stats(
                arr, segments, coastal_band, land, demoted, _seg_stats, _debug_dir)

        # Pixel-level cleanup for attached coastal ribbons that remain because
        # their segment still has enough inland support.  We look for narrow
        # land strips right against the sea with low texture and moderate/high
        # brightness, then demote only that fringe.
        land_remaining = land & ~demoted
        land_thickness = ndi.distance_transform_edt(land_remaining)
        ribbon = (
            land_remaining
            & (dist_to_sea <= max(6.0, 24.0 * scale))
            & (land_thickness <= max(4.0, 14.0 * scale))
            & (grad_norm <= 0.15)
            & (std_norm <= 0.14)
            & (brightness >= 0.50)
        )
        if np.any(ribbon):
            ribbon = cv2.morphologyEx(
                ribbon.astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8),
                iterations=1,
            ) > 0
            demoted |= ribbon

        # Also remove small unsupported coastal connected components
        # Scale area thresholds quadratically
        cc_max_area = max(100, int(round(800 * scale * scale)))
        cc_small_area = max(50, int(round(400 * scale * scale)))
        land_after = (land & ~demoted).astype(np.uint8) * 255
        n_labels, labels = cv2.connectedComponents(land_after, connectivity=8)
        for lbl in range(1, n_labels):
            cc = labels == lbl
            cc_area = int(np.count_nonzero(cc))
            if cc_area > cc_max_area:
                continue
            if (float(np.mean(dist_to_sea[cc])) <= cc_dist_thresh
                    and float(np.mean(land_support[cc])) <= 0.10
                    and cc_area <= cc_small_area):
                demoted[cc] = True

        return demoted

    # ------------------------------------------------------------------
    # Diagnostic dump helpers (only called when _debug_dir is set)
    # ------------------------------------------------------------------

    @staticmethod
    def _dump_heuristic(heuristic: MaskBundle, arr: np.ndarray, debug_dir: str):
        os.makedirs(debug_dir, exist_ok=True)
        u8 = to_u8(arr)
        # Land mask overlay: green = land, blue = shallow_water
        vis = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        land = heuristic.land > 0.5
        shal = heuristic.shallow_water > 0.5
        vis[land, 1] = np.clip(vis[land, 1].astype(int) + 100, 0, 255).astype(np.uint8)
        vis[shal, 0] = np.clip(vis[shal, 0].astype(int) + 120, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "01_heuristic_land.png"), vis)
        print(f"  [mask_debug] saved 01_heuristic_land.png")

    @staticmethod
    def _dump_waterlike(arr, waterlike, grad_norm, std_norm, brightness, land, debug_dir):
        os.makedirs(debug_dir, exist_ok=True)
        u8 = to_u8(arr)
        # Waterlike overlay: red = waterlike, green = land
        vis = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        vis[waterlike, 2] = np.clip(vis[waterlike, 2].astype(int) + 120, 0, 255).astype(np.uint8)
        vis[land, 1] = np.clip(vis[land, 1].astype(int) + 80, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "02_waterlike.png"), vis)
        # Also save gradient and std heatmaps
        grad_vis = (np.clip(grad_norm, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "02b_grad_norm.png"),
                    cv2.applyColorMap(grad_vis, cv2.COLORMAP_VIRIDIS))
        std_vis = (np.clip(std_norm, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "02c_std_norm.png"),
                    cv2.applyColorMap(std_vis, cv2.COLORMAP_VIRIDIS))
        brt_vis = (np.clip(brightness, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "02d_brightness.png"),
                    cv2.applyColorMap(brt_vis, cv2.COLORMAP_VIRIDIS))
        print(f"  [mask_debug] saved 02_waterlike.png + gradient/std/brightness heatmaps")

    @staticmethod
    def _dump_sea_distance(arr, sea_connected, dist_to_sea, land, debug_dir):
        os.makedirs(debug_dir, exist_ok=True)
        u8 = to_u8(arr)
        # Sea connected overlay
        vis = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        vis[sea_connected, 0] = np.clip(vis[sea_connected, 0].astype(int) + 120, 0, 255).astype(np.uint8)
        vis[land, 1] = np.clip(vis[land, 1].astype(int) + 80, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "03_sea_connected.png"), vis)
        # Distance heatmap (clamp to 60px for vis)
        dist_clamp = np.clip(dist_to_sea, 0, 60)
        dist_norm = (dist_clamp / 60.0 * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "03b_dist_to_sea.png"),
                    cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET))
        print(f"  [mask_debug] saved 03_sea_connected.png + dist_to_sea heatmap")

    @staticmethod
    def _dump_segments_and_stats(arr, segments, coastal_band, land, demoted, seg_stats, debug_dir):
        os.makedirs(debug_dir, exist_ok=True)
        u8 = to_u8(arr)
        h, w = arr.shape
        # Colored segment map (only coastal-band land segments)
        seg_vis = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        coastal_land = coastal_band & land
        if np.any(coastal_land):
            seg_ids = np.unique(segments[coastal_land])
            rng = np.random.RandomState(42)
            colors = rng.randint(60, 255, size=(int(seg_ids.max()) + 1, 3))
            for sid in seg_ids:
                mask = (segments == sid) & coastal_land
                c = colors[sid % len(colors)]
                seg_vis[mask] = c
        # Mark demoted segments with red X
        for stat in seg_stats:
            if stat["demoted"]:
                cr, cc = int(stat["centroid_r"]), int(stat["centroid_c"])
                cv2.drawMarker(seg_vis, (cc, cr), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 2)
        cv2.imwrite(os.path.join(debug_dir, "04_segments.png"), seg_vis)

        # Labeled segment map — large segments (≥150px) only, to stay readable
        seg_labeled = seg_vis.copy()
        for stat in sorted(seg_stats, key=lambda x: x["area"]):
            if stat["area"] < 150:
                continue
            cr, cc = int(stat["centroid_r"]), int(stat["centroid_c"])
            sid = stat["seg_id"]
            is_demoted = stat["demoted"]
            color = (0, 0, 255) if is_demoted else (255, 255, 255)
            scale_f = 0.4 if stat["area"] < 500 else (0.5 if stat["area"] < 2000 else 0.6)
            thickness = 1
            cv2.putText(seg_labeled, str(sid), (cc + 3, cr - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, scale_f, (0, 0, 0), thickness + 2)
            cv2.putText(seg_labeled, str(sid), (cc + 3, cr - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, scale_f, color, thickness)
            if is_demoted:
                cv2.drawMarker(seg_labeled, (cc, cr), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.imwrite(os.path.join(debug_dir, "04b_segments_labeled.png"), seg_labeled)
        # Quadrant crops of the labeled image
        for qname, r_sl, c_sl in [
            ("nw", slice(0, h // 2), slice(0, w // 2)),
            ("ne", slice(0, h // 2), slice(w // 2, w)),
            ("sw", slice(h // 2, h), slice(0, w // 2)),
            ("se", slice(h // 2, h), slice(w // 2, w)),
        ]:
            cv2.imwrite(os.path.join(debug_dir, f"04c_segments_{qname}.png"),
                        seg_labeled[r_sl, c_sl])

        # Print NW-quadrant stats table and all-segment summary
        print(f"\n  [mask_debug] === Segment Stats ({len(seg_stats)} land segments in coastal band) ===")
        print(f"  {'seg_id':>6} {'area':>5} {'quad':>3} {'sea_touch':>9} {'supported':>9} "
              f"{'near_sea':>8} {'mean_dist':>9} {'p90_dist':>8} "
              f"{'mean_grad':>9} {'mean_std':>8} {'mean_brt':>8} {'rules':>6}")
        nw_stats = [s for s in seg_stats if s["quadrant"] == "NW"]
        other_stats = [s for s in seg_stats if s["quadrant"] != "NW"]
        for label, stats_list in [("NW QUADRANT", nw_stats), ("OTHER", other_stats)]:
            if not stats_list:
                continue
            print(f"  --- {label} ({len(stats_list)} segments) ---")
            for s in sorted(stats_list, key=lambda x: -x["area"]):
                print(f"  {s['seg_id']:>6} {s['area']:>5} {s['quadrant']:>3} "
                      f"{s['sea_touch']:>9.3f} {s['supported']:>9.3f} "
                      f"{s['near_sea_frac']:>8.3f} {s['mean_dist']:>9.2f} {s['p90_dist']:>8.2f} "
                      f"{s['mean_grad']:>9.3f} {s['mean_std']:>8.3f} {s['mean_brightness']:>8.3f} "
                      f"{s['rules']:>6}")

        # Also write stats to CSV
        import csv
        csv_path = os.path.join(debug_dir, "segment_stats.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seg_id", "area", "quadrant", "centroid_r", "centroid_c",
                "sea_touch", "supported", "near_sea_frac", "mean_dist", "p90_dist",
                "mean_grad", "mean_std", "mean_brightness", "demoted", "rules",
            ])
            writer.writeheader()
            for s in seg_stats:
                writer.writerow(s)
        print(f"  [mask_debug] saved 04_segments.png + segment_stats.csv")

    @staticmethod
    def _dump_final_overlay(bundle: MaskBundle, arr: np.ndarray, debug_dir: str):
        os.makedirs(debug_dir, exist_ok=True)
        u8 = to_u8(arr)
        vis = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        land = bundle.land > 0.5
        shal = bundle.shallow_water > 0.5
        shore = bundle.shoreline > 0.5
        # green = land, blue = shallow_water, red = shoreline
        vis[land, 1] = np.clip(vis[land, 1].astype(int) + 100, 0, 255).astype(np.uint8)
        vis[shal, 0] = np.clip(vis[shal, 0].astype(int) + 120, 0, 255).astype(np.uint8)
        vis[shore, 2] = np.clip(vis[shore, 2].astype(int) + 100, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "05_final_mask.png"), vis)
        # Also save the raw binary land mask
        cv2.imwrite(os.path.join(debug_dir, "05b_final_land_binary.png"),
                    (land.astype(np.uint8) * 255))
        print(f"  [mask_debug] saved 05_final_mask.png + 05b_final_land_binary.png")

    @staticmethod
    def _apply_demotion(heuristic: MaskBundle, demoted: np.ndarray, arr: np.ndarray) -> MaskBundle:
        """Apply demotion mask to a full-res heuristic bundle."""
        valid = arr > 0
        land = (heuristic.land > 0.5) & ~demoted
        dark_farmland = (heuristic.dark_farmland > 0.5) & ~demoted
        land = _postprocess_land(land & valid)

        # Recompute derived masks
        grad = sobel_gradient(to_u8(arr))
        grad_energy = cv2.GaussianBlur(grad, (5, 5), 0).astype(np.float32)
        grad_norm = np.zeros_like(arr, dtype=np.float32)
        if np.any(valid):
            grad_norm[valid] = grad_energy[valid] / max(float(grad_energy[valid].max()), 1.0)

        shallow_water = (heuristic.shallow_water > 0.5) | (demoted & ~land)
        shallow_water &= ~land & valid
        water = (heuristic.water > 0.5) | shallow_water | (demoted & ~land)
        water &= valid

        shoreline, stable = _shoreline_and_stable(land, grad_norm, heuristic.texture)
        stable |= land & (heuristic.texture > 0.22)
        stable &= ~shallow_water

        return MaskBundle(
            land=land.astype(np.float32),
            water=water.astype(np.float32),
            shoreline=shoreline.astype(np.float32),
            stable=stable.astype(np.float32),
            shallow_water=shallow_water.astype(np.float32),
            dark_farmland=dark_farmland.astype(np.float32),
            texture=heuristic.texture,
            brightness=heuristic.brightness,
        )

    def make_land_mask(self, arr):
        return self.build_masks(arr).land


_PROVIDERS = {
    HeuristicMaskProvider.name: HeuristicMaskProvider(),
    CoastalObiaMaskProvider.name: CoastalObiaMaskProvider(),
}


def get_mask_provider(mode: str = "coastal_obia"):
    """Return a registered mask provider."""

    if mode not in _PROVIDERS:
        raise ValueError(f"Unknown mask provider: {mode}")
    return _PROVIDERS[mode]


def build_semantic_masks(arr, mode: str = "coastal_obia") -> MaskBundle:
    """Return semantic-ish masks for an array."""

    return get_mask_provider(mode).build_masks(arr)


def stable_feature_mask(arr, mode: str = "coastal_obia"):
    """Return a stability-weighted feature mask."""

    bundle = build_semantic_masks(arr, mode=mode)
    return bundle.stable


def shoreline_mask(arr, mode: str = "coastal_obia"):
    """Return the shoreline/intertidal mask."""

    bundle = build_semantic_masks(arr, mode=mode)
    return bundle.shoreline


def class_weight_map(arr, mode: str = "coastal_obia"):
    """Return a soft weight map for matching and QA."""

    bundle = build_semantic_masks(arr, mode=mode)
    weights = np.zeros_like(bundle.land, dtype=np.float32)
    weights += 1.25 * bundle.stable
    weights += 0.75 * bundle.shoreline
    weights += 0.35 * bundle.dark_farmland
    weights -= 0.80 * bundle.shallow_water
    weights = np.clip(weights, 0.0, 1.5)
    return weights
