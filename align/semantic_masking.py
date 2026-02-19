"""Semantic-ish masking utilities for historical grayscale imagery."""

from __future__ import annotations

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

    def build_masks(self, arr) -> MaskBundle:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("CoastalObiaMaskProvider expects a 2D array")

        valid = arr > 0
        if not np.any(valid):
            return _zero_bundle(arr.shape)

        # Step 1: run heuristic at ORIGINAL resolution
        heuristic = HeuristicMaskProvider().build_masks(arr)

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
            demote_small = self._compute_demotion(arr_small, land_small, scale=ds_scale)
            if demote_small is None:
                return heuristic
            # Upsample demotion mask back to original resolution
            demoted = cv2.resize(
                demote_small.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        else:
            land = heuristic.land > 0.5
            demoted = self._compute_demotion(arr, land, scale=1.0)
            if demoted is None:
                return heuristic

        # Step 3: apply demotions to the full-res heuristic
        return self._apply_demotion(heuristic, demoted, arr)

    def _compute_demotion(self, arr: np.ndarray, land: np.ndarray, scale: float = 1.0):
        """Compute a boolean demotion mask. Returns None if fallback needed.

        Parameters
        ----------
        arr : 2-D float32 array (possibly downscaled).
        land : boolean land mask at the same resolution as *arr*.
        scale : downscale factor applied to reach *arr* (1.0 = full resolution).
            Used to adapt kernel sizes, distance thresholds, and felzenszwalb
            parameters so behaviour is consistent regardless of resolution.
        """
        valid = arr > 0
        u8 = to_u8(arr)

        # Adapt kernel sizes to effective resolution.  At full resolution
        # (scale=1.0) we use 5x5 / 11x11.  When downscaled we shrink the
        # kernels proportionally but clamp to a 3x3 minimum (must be odd).
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

        # Broad water-like definition for morphological reconstruction.
        # At reduced resolution downsampling acts as a low-pass filter,
        # suppressing both gradient and local-std signals.  We compensate
        # by tightening the thresholds proportionally so textured land
        # does not leak into the waterlike mask.
        wl_grad_mid = min(0.28, 0.28 * scale)   # class-1 gradient gate
        wl_std_mid  = min(0.18, 0.18 * scale)   # class-1 std gate
        wl_grad_low = min(0.10, 0.10 * scale)   # catch-all gradient
        wl_std_low  = min(0.08, 0.08 * scale)   # catch-all std

        waterlike = (
            class_0
            | (class_1 & (grad_norm < wl_grad_mid) & (std_norm < wl_std_mid))
            | ((grad_norm < wl_grad_low) & (std_norm < wl_std_low) & (brightness < 0.55))
        ) & valid

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

        # Distance to sea and support fields.
        # All pixel-distance thresholds are expressed in "full-res equivalent
        # pixels" so they stay constant regardless of the downscale factor.
        dist_to_sea = ndi.distance_transform_edt(~sea_connected)

        # Scale distance thresholds: at full res we use 20px for inland_land;
        # at half res the same ground distance is 10px, etc.
        inland_thresh = max(4.0, 20.0 * scale)
        coastal_band_thresh = max(8.0, 40.0 * scale)
        near_sea_thresh = max(5.0, 24.0 * scale)
        close_sea_thresh = max(3.0, 15.0 * scale)
        cc_dist_thresh = max(4.0, 18.0 * scale)

        inland_land = land & (dist_to_sea >= inland_thresh)
        # The land_support dilation must reach at least `inland_thresh`
        # pixels so that coastal land immediately seaward of inland_land
        # is marked as supported.  At full resolution the 5x5 kernel with
        # 3 iterations reaches ~9px (> 20px not guaranteed, but the inland
        # threshold already carves a wide buffer).  We keep the same
        # *ground distance* reach at every scale.
        morph_k = max(3, _kern(5))
        # Each dilation iteration extends by ~(morph_k//2) pixels.
        # We want total reach >= inland_thresh + small margin.
        reach_per_iter = max(1, morph_k // 2)
        morph_iters = max(1, int(np.ceil((inland_thresh + 2) / reach_per_iter)))
        land_support = cv2.dilate(
            inland_land.astype(np.uint8),
            np.ones((morph_k, morph_k), np.uint8),
            iterations=morph_iters,
        ) > 0

        # Sea margin dilation — same ground-distance reach as full-res (3
        # iterations of 5x5 ≈ 9px at full res → 9*scale px at downscaled).
        sea_margin_reach = max(3.0, 9.0 * scale)
        sea_margin_iters = max(1, int(np.ceil(sea_margin_reach / reach_per_iter)))
        sea_margin = cv2.dilate(
            sea_connected.astype(np.uint8),
            np.ones((morph_k, morph_k), np.uint8),
            iterations=sea_margin_iters,
        ) > 0

        # Segment the coastal band
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

        # felzenszwalb parameter adaptation for downscaled images.
        # - `scale` (felzenszwalb sense): controls the observation-vs-boundary
        #   cost tradeoff.  At lower resolution each pixel covers more ground,
        #   so we reduce it proportionally to avoid over-merging disparate
        #   coastal features (shoal + land) into single segments.
        # - `sigma`: pre-smoothing before graph construction.  Downsampling
        #   already acts as a low-pass filter, so we reduce sigma to preserve
        #   the fine edges that distinguish shoals from land.
        # - `min_size`: post-merge threshold.  Scale quadratically with the
        #   downscale factor since area shrinks as scale^2.
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
                and supported <= 0.15
                and mean_grad <= 0.35
                and mean_std <= 0.20
            )
            # Rule B: mostly near-sea, unsupported, low texture
            demote_b = (
                near_sea_frac >= 0.50
                and supported <= 0.10
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
                and p90_dist <= max(6.0, 16.0 * scale)
                and mean_grad <= 0.16
                and mean_std <= 0.12
                and mean_brightness >= 0.42
            )

            if demote_a or demote_b or demote_c or demote_d:
                demoted[seg] = True

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
