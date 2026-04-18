"""Unit tests for the 14-parameter 2OC panoramic camera model.

Exercises forward_project → fit_panoramic round-trips on synthetic data
so we catch sign errors, rotation-convention errors, and bounds-handling
errors before running against real KH-9 data.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from preprocess.experimental.match_ip import PreprocessMatcherRuntime, run_preprocess_matcher
from preprocess.kh_panoramic import (
    _gcp_distribution_ok,
    PanoramicParams,
    ecef_to_local_xyz,
    extract_ortho_tie_point_gcps,
    fit_panoramic,
    forward_project,
    mapproject,
)


# Camera geometry constants for a typical KH-9 PC sub-frame.
PIXEL_PITCH = 7e-6          # 7 μm scan resolution
IMG_W = 37226               # raw rotated sub-frame width (px)
IMG_H = 25069               # raw rotated sub-frame height (px)
NOMINAL_F = 1.524           # KH-9 PC focal length (m)
SCAN_ARC_DEG = 70.0         # full scan arc
L = IMG_W * PIXEL_PITCH     # film length along scan axis (m)


def _nadir_params(cx: float = 5_620_000.0, cy: float = 3_035_000.0) -> PanoramicParams:
    """Perfect nadir-pointing camera centred over (cx, cy)."""
    return PanoramicParams(
        Xs0=cx,
        Ys0=cy,
        Zs0=170_000.0,
        omega0=0.0,
        phi0=0.0,
        kappa0=0.0,
        Xs1=0.0,
        Ys1=0.0,
        Zs1=0.0,
        omega1=0.0,
        phi1=0.0,
        kappa1=0.0,
        P=0.0,
        f=NOMINAL_F,
    )


def _dense_ground_grid(
    cx: float, cy: float,
    half_km_x: float = 35.0, half_km_y: float = 25.0,
    n_x: int = 12, n_y: int = 10,
    z_range: tuple = (0.0, 200.0),
    seed: int = 42,
) -> np.ndarray:
    """Return (N, 3) ground points (X, Y, Z) in the local planar CRS."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(cx - half_km_x * 1000, cx + half_km_x * 1000, n_x)
    ys = np.linspace(cy - half_km_y * 1000, cy + half_km_y * 1000, n_y)
    X, Y = np.meshgrid(xs, ys)
    Z = rng.uniform(z_range[0], z_range[1], size=X.shape)
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def _project_to_pixels(
    params: PanoramicParams,
    ground_xyz: np.ndarray,
) -> np.ndarray:
    """Project known ground points through ``forward_project`` to (col, row).

    Uses a naive Newton iteration to find the observed ``x_p`` such that
    ``forward_project(x_p_obs) == x_p_obs`` for the given ground point
    (since the observed x_p parameterises both t and α, we need a
    fixed-point solve to project FORWARD).
    """
    import torch
    params_t = params.to_tensor()
    X = torch.from_numpy(ground_xyz[:, 0].astype(np.float64))
    Y = torch.from_numpy(ground_xyz[:, 1].astype(np.float64))
    Z = torch.from_numpy(ground_xyz[:, 2].astype(np.float64))
    n = ground_xyz.shape[0]

    x_p = torch.zeros(n, dtype=torch.float64)
    for _ in range(30):
        xp_new, _ = forward_project(params_t, X, Y, Z, x_p, L)
        if torch.max(torch.abs(xp_new - x_p)) < 1e-8:
            x_p = xp_new
            break
        x_p = xp_new
    _, y_p = forward_project(params_t, X, Y, Z, x_p, L)

    cols = (x_p.detach().cpu().numpy() / PIXEL_PITCH) + IMG_W / 2.0
    # Film y-axis is UP-positive, raster row index is DOWN-positive,
    # so row = -y_p/pitch + H/2 (matches fit_panoramic's inverse).
    rows = (-y_p.detach().cpu().numpy() / PIXEL_PITCH) + IMG_H / 2.0
    return np.column_stack([cols, rows])


def test_params_tensor_roundtrip():
    p = _nadir_params()
    t = p.to_tensor()
    assert t.shape == (14,)
    p2 = PanoramicParams.from_tensor(t)
    assert p2.Xs0 == p.Xs0
    assert p2.f == p.f
    assert p2.omega0 == p.omega0


def test_bounds_contain_initial():
    p = _nadir_params()
    lo, hi = p.bounds()
    assert lo.shape == (14,) and hi.shape == (14,)
    vec = p.to_tensor().detach().cpu().numpy()
    assert np.all(lo <= vec + 1e-9)
    assert np.all(vec <= hi + 1e-9)


def test_forward_project_nadir_center_pixel_is_zero_xp():
    """A ground point directly below a nadir camera should project to
    x_p ≈ 0 (the scan centre)."""
    import torch
    params = _nadir_params()
    params_t = params.to_tensor()
    X = torch.tensor([params.Xs0], dtype=torch.float64)
    Y = torch.tensor([params.Ys0], dtype=torch.float64)
    Z = torch.tensor([0.0], dtype=torch.float64)
    # Start from x_p_obs = 0 (consistent with nadir target) and check
    # the modelled x_p is also ≈ 0.
    x_p_obs = torch.tensor([0.0], dtype=torch.float64)
    xp_mod, yp_mod = forward_project(params_t, X, Y, Z, x_p_obs, L)
    assert abs(float(xp_mod.item())) < 1e-6
    assert abs(float(yp_mod.item())) < 1e-6


def test_forward_project_east_ground_point_projects_east_on_film():
    """A ground point 1 km east of a nadir camera should project to a
    small positive x_p (≈ f · atan(1000/170000) ≈ 8.96 mm)."""
    import torch
    params = _nadir_params()
    params_t = params.to_tensor()
    dx_m = 1000.0
    expected_alpha = math.atan(dx_m / params.Zs0)
    expected_xp = params.f * expected_alpha

    X = torch.tensor([params.Xs0 + dx_m], dtype=torch.float64)
    Y = torch.tensor([params.Ys0], dtype=torch.float64)
    Z = torch.tensor([0.0], dtype=torch.float64)
    # Observed x_p = expected_xp (simulating the fit case where observation
    # gives t/α directly). The model should predict the same value.
    x_p_obs = torch.tensor([expected_xp], dtype=torch.float64)
    xp_mod, yp_mod = forward_project(params_t, X, Y, Z, x_p_obs, L)
    assert abs(float(xp_mod.item()) - expected_xp) < 1e-6
    assert abs(float(yp_mod.item())) < 1e-4


def test_phase4_pano_params_to_opticalbar_tsai_structure(tmp_path):
    """Phase 4: ``pano_params_to_opticalbar_tsai`` writes a valid
    ASP OpticalBar .tsai. Check the structure (required headers,
    required fields, rotation matrix orthonormality), the ECEF round-
    trip, and that the rotation matrix is a valid SO(3) element."""
    import math
    from pyproj import Transformer
    from preprocess.kh_panoramic import (
        pano_params_to_opticalbar_tsai, PanoramicParams,
    )

    # Bahrain UTM zone 39N, mid-scan position near (50.4°E, 26.2°N) at
    # 197 km altitude with zero attitude.
    local_crs = "EPSG:32639"
    tr_ll = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    xs, ys = tr_ll.transform(50.4, 26.2)
    params = PanoramicParams(
        Xs0=xs, Ys0=ys, Zs0=197_000.0,
        omega0=0.0, phi0=0.0, kappa0=0.0,
        Xs1=0.0, Ys1=0.0, Zs1=0.0,
        omega1=0.0, phi1=0.0, kappa1=0.0,
        P=0.0, f=1.1685,
    )

    out_path = tmp_path / "seed.tsai"
    pano_params_to_opticalbar_tsai(
        params,
        pixel_pitch=7e-6,
        image_width_px=37226,
        image_height_px=25069,
        local_crs=local_crs,
        camera_params={
            "scan_time": 0.5,
            "forward_tilt": 0.1745,
            "speed": 7800.0,
            "scan_dir": "right",
            "motion_compensation_factor": 1.0,
        },
        out_path=str(out_path),
        at_time=0.5,
    )
    content = out_path.read_text()
    assert content.startswith("VERSION_4\nOPTICAL_BAR\n")
    required_fields = (
        "image_size", "image_center", "pitch", "f", "scan_time",
        "forward_tilt", "iC", "iR", "speed", "mean_earth_radius",
        "mean_surface_elevation", "motion_compensation_factor", "scan_dir",
    )
    for key in required_fields:
        assert key in content, f"missing required field {key!r} in .tsai"

    # Parse iC → check distance-from-origin ≈ R_earth + altitude.
    for line in content.splitlines():
        if line.startswith("iC = "):
            parts = line.split("=", 1)[1].split()
            iC = [float(p) for p in parts]
            break
    else:
        raise AssertionError("iC line not found")
    r_ecef = math.sqrt(sum(v * v for v in iC))
    # At 26.2°N lat + 197 km altitude on the WGS84 ellipsoid the
    # geocentric radius is ~6571 km (ellipsoid is flatter at higher
    # lats; N_radius of curvature slightly > equatorial a).
    assert 6_565_000 < r_ecef < 6_575_000, (
        f"ECEF magnitude {r_ecef:.0f} m not consistent with "
        f"6.565–6.575 Mm expected at 26.2°N + 197 km"
    )

    # Parse iR → check orthonormal (R · R^T ≈ I, det ≈ 1).
    import numpy as np
    for line in content.splitlines():
        if line.startswith("iR = "):
            parts = line.split("=", 1)[1].split()
            iR = np.array([float(p) for p in parts]).reshape(3, 3)
            break
    else:
        raise AssertionError("iR line not found")
    should_be_I = iR @ iR.T
    assert np.allclose(should_be_I, np.eye(3), atol=1e-9), (
        f"iR should be orthonormal; R·R^T =\n{should_be_I}"
    )
    assert abs(np.linalg.det(iR) - 1.0) < 1e-9


def test_phase4_pano_params_to_opticalbar_f_matches(tmp_path):
    """Phase 4: the written .tsai ``f = ...`` line must equal the
    14-parameter ``params.f`` (the whole point of the shared-f seed
    is to push the Phase 3d value into ASP's initial conditions)."""
    from pyproj import Transformer
    from preprocess.kh_panoramic import (
        pano_params_to_opticalbar_tsai, PanoramicParams,
    )
    tr_ll = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)
    xs, ys = tr_ll.transform(50.4, 26.2)
    custom_f = 1.1685429834016876
    params = PanoramicParams(
        Xs0=xs, Ys0=ys, Zs0=197_000.0,
        omega0=0.0, phi0=0.0, kappa0=0.0,
        Xs1=0.0, Ys1=0.0, Zs1=0.0,
        omega1=0.0, phi1=0.0, kappa1=0.0,
        P=0.0, f=custom_f,
    )
    out_path = tmp_path / "seed.tsai"
    pano_params_to_opticalbar_tsai(
        params, 7e-6, 37226, 25069, "EPSG:32639",
        {"scan_time": 0.5, "forward_tilt": 0.1745, "speed": 7800.0,
         "scan_dir": "right", "motion_compensation_factor": 1.0},
        str(out_path),
    )
    # The written f should match to double precision.
    for line in out_path.read_text().splitlines():
        if line.startswith("f = "):
            written = float(line.split("=", 1)[1].strip())
            break
    else:
        raise AssertionError("f = line not found")
    assert written == pytest.approx(custom_f, rel=1e-12)


def test_phase9_bounds_respect_f_frac_range_override():
    """Phase 9: tightening ``f_frac_range`` via the new kwarg must
    narrow the focal-length bound around ``nominal_f``. Exercises the
    profile-driven override without running a full fit."""
    params = PanoramicParams(
        Xs0=5_600_000.0, Ys0=3_040_000.0, Zs0=170_000.0,
        omega0=0.0, phi0=0.0, kappa0=0.0,
        Xs1=0.0, Ys1=0.0, Zs1=0.0,
        omega1=0.0, phi1=0.0, kappa1=0.0,
        P=0.0, f=1.524,
    )
    # Default (None) → legacy ±30 %.
    lo_default, hi_default = params.bounds(nominal_f=1.524)
    assert hi_default[13] == pytest.approx(1.524 * 1.30, rel=1e-9)
    assert lo_default[13] == pytest.approx(1.524 * 0.70, rel=1e-9)
    # Tightened to ±5 %.
    lo_tight, hi_tight = params.bounds(nominal_f=1.524, f_frac_range=0.05)
    assert hi_tight[13] == pytest.approx(1.524 * 1.05, rel=1e-9)
    assert lo_tight[13] == pytest.approx(1.524 * 0.95, rel=1e-9)
    # The rest of the 14-vec is unaffected — only the f bound moves.
    assert np.allclose(lo_default[:13], lo_tight[:13])
    assert np.allclose(hi_default[:13], hi_tight[:13])


def test_fit_recovers_kh4b_geometry_phase2_sanity():
    """Phase 2 sanity check: the 14-parameter fit must reproduce KH-4B
    geometry on noise-free synthetic GCPs.

    The 2OC paper (Hou et al. 2023) is framed around KH-4B (f ≈ 0.6096 m,
    shorter focal length than KH-9 PC's 1.524 m). Table 6 reports fitted
    focal lengths of 0.60025, 0.6028, 0.60602, 0.6029 m across sub-images
    a/b/c/d of one strip — a ~0.44 % spread around a ~0.6030 m mean.

    If the fit core cannot recover a KH-4B focal length from noise-free
    synthetic inputs, Phase 3+ of the recovery plan is wasted effort.
    A passing run here confirms the port is sound and failures on real
    data must be upstream (altitude, GCP coverage, bbox policy).
    """
    # KH-4B geometry — rough USGS sub-frame dimensions.
    kh4b_f = 0.6096
    kh4b_pixel_pitch = 7e-6
    kh4b_img_w = 34_000
    kh4b_img_h = 24_000
    kh4b_L = kh4b_img_w * kh4b_pixel_pitch

    cx, cy = 5_620_000.0, 3_035_000.0
    truth = PanoramicParams(
        Xs0=cx,
        Ys0=cy,
        Zs0=170_000.0,
        omega0=0.0,
        phi0=0.0,
        kappa0=0.0,
        Xs1=0.0, Ys1=0.0, Zs1=0.0,
        omega1=0.0, phi1=0.0, kappa1=0.0,
        P=0.0,
        f=kh4b_f,
    )

    # Dense grid, similar coverage to KH-4B sub-frame footprint.
    rng = np.random.default_rng(42)
    xs = np.linspace(cx - 45_000, cx + 45_000, 14)
    ys = np.linspace(cy - 25_000, cy + 25_000, 10)
    X, Y = np.meshgrid(xs, ys)
    Z = rng.uniform(0.0, 200.0, size=X.shape)
    ground_xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Project with explicit KH-4B L so film-coord geometry is self-consistent.
    import torch
    params_t = truth.to_tensor()
    Xt = torch.from_numpy(ground_xyz[:, 0].astype(np.float64))
    Yt = torch.from_numpy(ground_xyz[:, 1].astype(np.float64))
    Zt = torch.from_numpy(ground_xyz[:, 2].astype(np.float64))
    n = ground_xyz.shape[0]
    x_p = torch.zeros(n, dtype=torch.float64)
    for _ in range(30):
        xp_new, _ = forward_project(params_t, Xt, Yt, Zt, x_p, kh4b_L)
        if torch.max(torch.abs(xp_new - x_p)) < 1e-8:
            x_p = xp_new
            break
        x_p = xp_new
    _, y_p = forward_project(params_t, Xt, Yt, Zt, x_p, kh4b_L)
    cols = (x_p.detach().cpu().numpy() / kh4b_pixel_pitch) + kh4b_img_w / 2.0
    rows = (-y_p.detach().cpu().numpy() / kh4b_pixel_pitch) + kh4b_img_h / 2.0
    pixels = np.column_stack([cols, rows])
    gcps = np.column_stack([pixels, ground_xyz])

    perturbed = PanoramicParams(
        Xs0=truth.Xs0 + 200.0,
        Ys0=truth.Ys0 - 150.0,
        Zs0=truth.Zs0,
        omega0=0.01, phi0=-0.01, kappa0=0.005,
        Xs1=0.0, Ys1=0.0, Zs1=0.0,
        omega1=0.0, phi1=0.0, kappa1=0.0,
        P=0.0,
        f=truth.f * 1.005,
    )
    result = fit_panoramic(
        sub_frame_gcps=gcps,
        initial=perturbed,
        pixel_pitch=kh4b_pixel_pitch,
        image_width_px=kh4b_img_w,
        image_height_px=kh4b_img_h,
        nominal_f=kh4b_f,
        max_iter=200,
        f_scale_px=1.0,
        verbose=False,
    )
    assert result.success, f"KH-4B fit failed: {result.message}"
    assert result.reprojection_rms_px < 1.0, (
        f"KH-4B RMS too high: {result.reprojection_rms_px:.3f} px "
        f"({result.message})"
    )
    # f must stay within 0.5 % of KH-4B nominal — comfortably inside the
    # 0.44 % observed-spread upper bound reported in paper Table 6.
    f_dev_frac = abs(result.params.f - truth.f) / truth.f
    assert f_dev_frac < 0.005, (
        f"KH-4B f deviation {f_dev_frac * 100:.3f}% exceeds 0.5 % "
        f"gate (fit f={result.params.f:.5f}, truth f={truth.f:.5f})"
    )


def test_fit_recovers_nadir_params_from_noise_free_synth():
    """Synthesise many ground points under a known nadir camera, project,
    and check fit_panoramic recovers the parameters within tight tolerance."""
    truth = _nadir_params()
    ground_xyz = _dense_ground_grid(truth.Xs0, truth.Ys0)
    pixels = _project_to_pixels(truth, ground_xyz)
    gcps = np.column_stack([pixels, ground_xyz])  # (N, 5)

    # Start the fit from a slightly perturbed initial. Zs0 matches truth
    # exactly because ``fix_zs0=True`` (the default) pins Zs0 to the
    # initial value — perturbing Zs0 here would leave the fit stuck at
    # the wrong altitude, and f would shift to absorb the mismatch.
    perturbed = PanoramicParams(
        Xs0=truth.Xs0 + 200.0,
        Ys0=truth.Ys0 - 150.0,
        Zs0=truth.Zs0,
        omega0=0.01,
        phi0=-0.01,
        kappa0=0.005,
        Xs1=0.0, Ys1=0.0, Zs1=0.0,
        omega1=0.0, phi1=0.0, kappa1=0.0,
        P=0.0,
        f=truth.f * 1.001,
    )
    result = fit_panoramic(
        sub_frame_gcps=gcps,
        initial=perturbed,
        pixel_pitch=PIXEL_PITCH,
        image_width_px=IMG_W,
        image_height_px=IMG_H,
        nominal_f=NOMINAL_F,
        max_iter=100,
        f_scale_px=1.0,
        verbose=False,
    )
    assert result.success, f"fit failed: {result.message}"
    assert result.reprojection_rms_px < 1.0, (
        f"RMS too high: {result.reprojection_rms_px:.3f} px "
        f"(message: {result.message})"
    )
    # Recovered parameters should be close to truth
    assert abs(result.params.Xs0 - truth.Xs0) < 100.0
    assert abs(result.params.Ys0 - truth.Ys0) < 100.0
    assert abs(result.params.Zs0 - truth.Zs0) < 1000.0
    assert abs(result.params.f - truth.f) < truth.f * 0.002


def test_fit_recovers_drift_rates_that_asp_cannot_represent():
    """The whole point of 2OC's 14-param model is the 6 in-scan rate
    parameters that ASP OpticalBar's 7-DoF model can't represent. Check
    fit_panoramic recovers them from noise-free synth data."""
    truth = PanoramicParams(
        Xs0=5_620_000.0, Ys0=3_035_000.0, Zs0=170_000.0,
        omega0=math.radians(-10.0),  # forward tilt for Aft camera
        phi0=math.radians(2.0),
        kappa0=math.radians(-5.0),
        Xs1=200.0,        # 200 m linear velocity over the scan
        Ys1=-150.0,
        Zs1=50.0,
        omega1=math.radians(0.1),  # 0.1° per normalised t — tiny drift
        phi1=math.radians(-0.05),
        kappa1=math.radians(0.2),
        P=0.02,
        f=NOMINAL_F,
    )
    ground_xyz = _dense_ground_grid(truth.Xs0, truth.Ys0, n_x=20, n_y=15)
    pixels = _project_to_pixels(truth, ground_xyz)
    gcps = np.column_stack([pixels, ground_xyz])

    # Initial = nadir-pointing camera at the scene centre
    initial = PanoramicParams.from_corner_init(
        corners_xy_local=[(truth.Xs0 - 25_000, truth.Ys0 - 20_000),
                          (truth.Xs0 + 25_000, truth.Ys0 - 20_000),
                          (truth.Xs0 + 25_000, truth.Ys0 + 20_000),
                          (truth.Xs0 - 25_000, truth.Ys0 + 20_000)],
        nominal_f=NOMINAL_F,
        forward_tilt_rad=math.radians(-10.0),
    )
    result = fit_panoramic(
        sub_frame_gcps=gcps,
        initial=initial,
        pixel_pitch=PIXEL_PITCH,
        image_width_px=IMG_W,
        image_height_px=IMG_H,
        nominal_f=NOMINAL_F,
        max_iter=200,
        loss="linear",
    )
    assert result.success, f"fit failed: {result.message}"
    # Sub-pixel reprojection
    assert result.reprojection_rms_px < 0.5, (
        f"RMS too high: {result.reprojection_rms_px:.3f} px"
    )
    # All 14 parameters within reasonable tolerance
    assert abs(result.params.Xs0 - truth.Xs0) < 50.0
    assert abs(result.params.Ys0 - truth.Ys0) < 50.0
    # Rate parameters recovered (the whole point of this test)
    assert abs(result.params.Xs1 - truth.Xs1) < 50.0, \
        f"Xs1: got {result.params.Xs1:.1f}, want {truth.Xs1:.1f}"
    assert abs(result.params.Ys1 - truth.Ys1) < 50.0
    assert abs(result.params.omega1 - truth.omega1) < math.radians(0.05)
    assert abs(result.params.phi1 - truth.phi1) < math.radians(0.05)
    assert abs(result.params.kappa1 - truth.kappa1) < math.radians(0.05)
    assert abs(result.params.P - truth.P) < 0.01


def test_run_preprocess_matcher_nift_finds_sparse_translation():
    img_a = np.zeros((256, 256), dtype=np.float32)
    for idx, x in enumerate(range(32, 224, 32)):
        for idy, y in enumerate(range(32, 224, 32)):
            radius = 4 + ((idx + idy) % 3)
            intensity = 80 + idx * 15 + idy * 7
            cv2.circle(img_a, (x, y), radius, float(intensity), -1)
            cv2.line(img_a, (x - 8, y), (x + 8, y), float(intensity + 20), 1)
            cv2.line(img_a, (x, y - 8), (x, y + 8), float(intensity + 10), 1)

    shift_x = 6.0
    shift_y = 4.0
    M = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
    img_b = cv2.warpAffine(img_a, M, (img_a.shape[1], img_a.shape[0]))

    pts_a, pts_b, conf = run_preprocess_matcher(
        img_a,
        img_b,
        matcher_name="nift",
        max_matches=200,
    )

    assert pts_a is not None
    assert pts_b is not None
    assert conf is not None
    assert len(pts_a) >= 12

    median_shift = np.median(pts_b - pts_a, axis=0)
    assert np.allclose(median_shift, np.array([shift_x, shift_y]), atol=2.0)


def test_gcp_distribution_accepts_full_width_narrow_band():
    cols = np.linspace(200.0, IMG_W - 200.0, 30)
    rows = np.full_like(cols, IMG_H * 0.52)
    gcps = np.column_stack([
        cols,
        rows,
        np.zeros_like(cols),
        np.zeros_like(cols),
        np.zeros_like(cols),
    ])

    ok, summary = _gcp_distribution_ok(gcps, IMG_W, IMG_H)

    assert ok is True
    assert summary["occupied_cells"] == 6
    assert summary["occupied_cols"] == 6
    assert summary["col_span_frac"] > 0.9


def test_gcp_distribution_accepts_five_column_narrow_band():
    cols = np.linspace(IMG_W * 0.10, IMG_W * 0.80, 30)
    rows = np.full_like(cols, IMG_H * 0.48)
    gcps = np.column_stack([
        cols,
        rows,
        np.zeros_like(cols),
        np.zeros_like(cols),
        np.zeros_like(cols),
    ])

    ok, summary = _gcp_distribution_ok(gcps, IMG_W, IMG_H)

    assert ok is True
    assert summary["occupied_cols"] == 5
    assert summary["occupied_cells"] == 5
    assert 0.69 <= summary["col_span_frac"] <= 0.71


def test_gcp_distribution_rejects_one_sided_cluster():
    cols = np.linspace(100.0, IMG_W * 0.22, 30)
    rows = np.linspace(IMG_H * 0.35, IMG_H * 0.65, 30)
    gcps = np.column_stack([
        cols,
        rows,
        np.zeros_like(cols),
        np.zeros_like(cols),
        np.zeros_like(cols),
    ])

    ok, summary = _gcp_distribution_ok(gcps, IMG_W, IMG_H)

    assert ok is False
    assert summary["occupied_cols"] < 5 or summary["col_span_frac"] < 0.55


def test_mapproject_runs_on_synthetic_subframe(tmp_path):
    """Synthesise a tiny raw sub-frame, run mapproject (Algorithm 1),
    and verify (a) it produces a non-empty GeoTIFF in the right CRS,
    (b) the centre pixel maps approximately to the camera centre."""
    import rasterio
    from osgeo import gdal

    # Tiny synthetic sub-frame: 200x150 pixels with everywhere-valid content.
    # Use a smooth gradient (not nodata=background) so bilinear sampling
    # returns finite values.
    sub_w, sub_h = 200, 150
    yy, xx = np.mgrid[0:sub_h, 0:sub_w]
    arr = (50 + (xx + yy) % 200).astype(np.uint8)  # values 50..249, all valid

    sub_path = str(tmp_path / "synthetic_subframe.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(sub_path, sub_w, sub_h, 1, gdal.GDT_Byte)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(255)  # nodata = a value not in arr
    ds.FlushCache()
    ds = None

    # Camera over 5_620_000, 3_035_000 in EPSG:3857 (Bahrain area)
    pitch = 7e-6
    f = 1.524
    cx = 5_620_000.0
    cy = 3_035_000.0
    truth = PanoramicParams(
        Xs0=cx, Ys0=cy, Zs0=170_000.0,
        omega0=0.0, phi0=0.0, kappa0=0.0,
        Xs1=0.0, Ys1=0.0, Zs1=0.0,
        omega1=0.0, phi1=0.0, kappa1=0.0,
        P=0.0, f=f,
    )

    out_path = str(tmp_path / "synth_ortho.tif")
    # Pick a bbox WELL WITHIN the camera footprint:
    #   sub-frame half-width on film = sub_w/2 * pitch
    #   angular half-width = (sub_w/2 * pitch) / f
    #   ground half-width at altitude = angular * Zs0
    #   = (200/2 * 7e-6 / 1.524) * 170_000 ≈ 78 m
    # Use 50 m half-extent to stay well within the imaged area.
    half_extent = 50.0
    bbox = (cx - half_extent, cy - half_extent, cx + half_extent, cy + half_extent)

    result = mapproject(
        params=truth,
        sub_frame_path=sub_path,
        dem_path=None,
        out_path=out_path,
        pixel_pitch=pitch,
        image_width_px=sub_w,
        image_height_px=sub_h,
        resolution_m=20.0,
        bbox_xy=bbox,
        local_crs="EPSG:3857",
        t_srs="EPSG:3857",
        device="cpu",
        chunk_px=512,
    )
    assert result == out_path
    assert os.path.isfile(out_path)

    with rasterio.open(out_path) as ds:
        assert ds.width >= 5 and ds.height >= 5
        assert str(ds.crs) == "EPSG:3857"
        out = ds.read(1)
        nodata = ds.nodata if ds.nodata is not None else 0
    valid = out != nodata
    assert valid.sum() > 0, "Output ortho has no valid pixels"
    # Nadir camera at a flat-ground scene — the sampled intensity should
    # span a non-trivial range (synthetic is a gradient).
    assert out[valid].max() > out[valid].min() + 10.0


def test_extract_ortho_ties_transforms_overlap_points_to_local_crs(tmp_path, monkeypatch):
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    import preprocess.experimental.match_ip as match_ip
    import preprocess.kh_panoramic as kp

    width = height = 128
    transform = from_origin(5_618_000.0, 3_037_000.0, 4.0, 4.0)
    crs = "EPSG:3857"

    prev_path = tmp_path / "prev_ortho.tif"
    curr_path = tmp_path / "curr_ortho.tif"
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": crs,
        "transform": transform,
        "nodata": 0,
    }
    arr = np.full((height, width), 150, dtype=np.uint8)
    with rasterio.open(prev_path, "w", **profile) as ds:
        ds.write(arr, 1)
    with rasterio.open(curr_path, "w", **profile) as ds:
        ds.write(arr, 1)

    xs = np.linspace(16.0, 112.0, 25)
    ys = np.linspace(20.0, 108.0, 25)
    pts = np.column_stack([xs, ys]).astype(np.float32)

    def fake_run_roma_tiled(*a, **kw):
        conf = np.ones(len(pts), dtype=np.float32)
        return pts.copy(), pts.copy(), conf

    def fake_dedup(pts_a, pts_b, conf, cell_px=40):
        return pts_a, pts_b, conf

    def fake_sample_dem_local_xy(x, y, dem_path, local_crs):
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    tr = Transformer.from_crs(crs, "EPSG:32639", always_xy=True)
    expected_x, expected_y = tr.transform(
        transform.c + pts[:, 0] * transform.a,
        transform.f + pts[:, 1] * transform.e,
    )

    def fake_world_to_raw_pixel(
        params,
        X_world,
        Y_world,
        Z_world,
        pixel_pitch,
        image_width_px,
        image_height_px,
    ):
        X_world = np.asarray(X_world, dtype=np.float64)
        Y_world = np.asarray(Y_world, dtype=np.float64)
        assert np.allclose(X_world, expected_x, atol=5.0)
        assert np.allclose(Y_world, expected_y, atol=5.0)
        cols = np.linspace(10.0, image_width_px - 10.0, len(X_world))
        rows = np.linspace(12.0, image_height_px - 12.0, len(Y_world))
        return cols, rows

    monkeypatch.setattr(match_ip, "_run_roma_tiled", fake_run_roma_tiled)
    monkeypatch.setattr(match_ip, "_dedup_spatial", fake_dedup)
    monkeypatch.setattr(kp, "_sample_dem_local_xy", fake_sample_dem_local_xy)
    monkeypatch.setattr(kp, "world_to_raw_pixel", fake_world_to_raw_pixel)

    runtime = PreprocessMatcherRuntime(
        matcher_name="roma",
        device="cpu",
        model_cache=type("DummyCache", (), {"roma": object(), "device": "cpu"})(),
        owns_cache=False,
    )

    gcps = extract_ortho_tie_point_gcps(
        prev_ortho_path=str(prev_path),
        curr_ortho_path=str(curr_path),
        curr_params=_nadir_params(),
        curr_pixel_pitch=PIXEL_PITCH,
        curr_image_width_px=IMG_W,
        curr_image_height_px=IMG_H,
        local_crs="EPSG:32639",
        dem_path=None,
        matcher_runtime=runtime,
        max_matches=200,
        max_tiles=16,
    )

    assert gcps is not None
    assert gcps.shape == (25, 5)
    assert np.all(gcps[:, 2] < 1_000_000.0)


import os  # for the test above


# ---------------------------------------------------------------------------
# cam_gen altitude integration
# ---------------------------------------------------------------------------


def test_ecef_to_local_xyz_recovers_altitude_over_bahrain():
    """A camera at ~170 km altitude over Bahrain, converted from ECEF to
    local UTM 39N, should produce ``Zs0 ≈ 170 km`` and UTM coordinates
    roughly matching Bahrain's UTM band (X ~400 km, Y ~2,900 km in the
    UTM zone 39N projection)."""
    # Bahrain centre approx 26°N 50.5°E at 170 km altitude.
    lat_deg, lon_deg, alt_m = 26.0, 50.5, 170_000.0
    # Compose ECEF position for that geodetic point (WGS84).
    import math as _m
    a = 6378137.0
    f_inv = 298.257223563
    b = a * (1.0 - 1.0 / f_inv)
    e2 = 1.0 - (b / a) ** 2
    lat = _m.radians(lat_deg)
    lon = _m.radians(lon_deg)
    N = a / _m.sqrt(1.0 - e2 * _m.sin(lat) ** 2)
    X = (N + alt_m) * _m.cos(lat) * _m.cos(lon)
    Y = (N + alt_m) * _m.cos(lat) * _m.sin(lon)
    Z = ((1 - e2) * N + alt_m) * _m.sin(lat)

    x_utm, y_utm, zs0 = ecef_to_local_xyz(
        np.array([X, Y, Z]), local_crs="EPSG:32639"
    )
    assert abs(zs0 - alt_m) < 1.0, f"altitude round-trip off: {zs0} vs {alt_m}"
    # UTM 39N centre of Bahrain is roughly (400-500 km easting, 2800-2900 km
    # northing). Just sanity-check magnitudes.
    assert 200_000 < x_utm < 900_000, f"x_utm outside Bahrain UTM band: {x_utm}"
    assert 2_500_000 < y_utm < 3_200_000, f"y_utm outside Bahrain UTM band: {y_utm}"


def test_fit_panoramic_respects_initial_zs0_when_fixed():
    """When ``fix_zs0=True`` and the caller passes a non-default Zs0 in
    ``initial``, the fit must keep Zs0 at the caller's value (not reset to
    the module constant 170 km). This is the hook that cam_gen-derived
    altitudes use to override the default."""
    # Build a tight synthetic GCP set around nadir at altitude 180 km.
    custom_zs0 = 180_000.0
    params = _nadir_params()
    params.Zs0 = custom_zs0
    gcps = _synth_gcps(params, n=60)

    # Start the fit from a perturbed initial with Zs0 set to the desired
    # altitude. With fix_zs0=True, the returned params.Zs0 must equal
    # custom_zs0 (not NOMINAL_Z_S0 = 170_000).
    perturbed = PanoramicParams(**{**vars(params),
                                   "Xs0": params.Xs0 + 3000.0,
                                   "Ys0": params.Ys0 - 2000.0,
                                   "kappa0": 0.01,
                                   "Zs0": custom_zs0})
    result = fit_panoramic(
        sub_frame_gcps=gcps,
        initial=perturbed,
        pixel_pitch=PIXEL_PITCH,
        image_width_px=IMG_W,
        image_height_px=IMG_H,
        nominal_f=NOMINAL_F,
        max_iter=60,
        fix_zs0=True,
        fix_f=True,
        fix_velocities=True,
        fix_rates=True,
        fix_p=True,
    )
    assert result.success or result.reprojection_rms_px < 1.0, (
        f"fit failed: {result.message}"
    )
    assert abs(result.params.Zs0 - custom_zs0) < 1e-6, (
        f"Zs0 drifted: {result.params.Zs0} vs {custom_zs0}"
    )


def _synth_gcps(params: PanoramicParams, n: int = 60) -> np.ndarray:
    """Build a noise-free (col, row, X, Y, Z) table by forward-projecting a
    dense ground grid through ``params``. Used by the test above."""
    import torch
    cx, cy = params.Xs0, params.Ys0
    xs = np.linspace(cx - 30_000, cx + 30_000, 10)
    ys = np.linspace(cy - 18_000, cy + 18_000, 6)
    Xv, Yv = np.meshgrid(xs, ys)
    X = Xv.ravel()
    Y = Yv.ravel()
    Z = np.zeros_like(X)
    # Use forward_project to derive observed film coords, then convert to
    # raw pixels (col, row).
    params_t = params.to_tensor("cpu")
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    Z_t = torch.from_numpy(Z)
    # Iterate to find x_p — need x_p = f * atan(-N_x / N_z) where N depends
    # on t = x_p / L. Two Picard iterations are enough for noise-free data.
    L_ = IMG_W * PIXEL_PITCH
    x_p = torch.zeros_like(X_t, dtype=torch.float64)
    for _ in range(4):
        xp_m, yp_m = forward_project(params_t, X_t, Y_t, Z_t, x_p, L_)
        x_p = xp_m
    # Convert film coords back to raw pixels.
    col = (x_p.numpy() / PIXEL_PITCH) + IMG_W / 2.0
    row = (-yp_m.numpy() / PIXEL_PITCH) + IMG_H / 2.0
    # Keep only points inside the raster.
    in_frame = (col >= 0) & (col < IMG_W) & (row >= 0) & (row < IMG_H)
    gcps = np.column_stack([col, row, X, Y, Z])[in_frame][:n]
    return gcps.astype(np.float64)
