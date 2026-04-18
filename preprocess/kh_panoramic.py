"""14-parameter panoramic camera model for KH-4/9 PC, per 2OC (Hou et al. 2023).

Implements the full per-sub-frame flow:
    1. forward_project() — 2OC §3.2 eqs 1-21, differentiable in PyTorch
    2. fit_panoramic()   — Levenberg–Marquardt LSQ fit of the 14 params to
                           GCPs via scipy.optimize.least_squares, using
                           torch.autograd for the Jacobian
    3. extract_reference_gcps() — RoMa-based dense matching of a raw
                                  sub-frame against a georeferenced
                                  reference ortho
    4. mapproject()      — 2OC §3.4 Algorithm 1 time-iterative ortho,
                           vectorised on GPU in PyTorch

Why this module exists: ASP OpticalBar fits only 7 DoF (6 extrinsics + 1
focal length), forcing constant attitude and constant-direction velocity
through the ~0.4 s panoramic scan. 2OC Table 6 shows actual KH-4B satellite
attitude drifts by 0.02–0.4° per normalised scan time. With only 4 corner
constraints to fit, adjacent sub-frames' 7-DoF fits each absorb that drift
differently in their extrinsics, and the interior projections disagree —
visible as 1000+ m doubling at sub-frame seams in blended mosaics.

2OC's solution: 14 parameters per sub-frame (6 initial pose + 6 linear-in-
time pose rates + IMC + focal length), fit to 100+ reference GCPs.
We use RoMa v2 to extract the GCPs (instead of 2OC's NIFT descriptor).
Everything else — 2OC §3.2 math and Algorithm 1 — is implemented as
written in the paper.

Key equation references (2OC paper, doi:10.3390/rs15215116):
    Eqs 1–6:   linear-in-time exterior orientation at time t
    Eq 7:      t = x_p / L                       (normalised scan time)
    Eq 8:      α = x_p / f                       (cross-track scan angle)
    Eqs 9–10:  collinearity with R_α (cross-track rotation) and R_t (attitude)
    Eq 11:     y_IMC = -P f sin α cos ω_t        (image motion compensation)
    Eqs 15–17: N_x, N_y, N_z = rows of R_t @ (ground - camera_at_t)
    Eq 20:     x_p = f · atan(-N_x / N_z)        (predicted scan pixel)
    Eq 21:     y_p = P f sin α cos ω_t - f cos α · N_y / N_z   (predicted along-track)

There is a typo in 2OC eq 18 ("tan α = -N_x / N_y"); the correct
derivation is `tan α = -N_x / N_z` as confirmed by eq 20.

The paper's Table 6 values for X_s0, Y_s0, Z_s0 look like a LOCAL PLANAR
CRS (e.g., UTM) rather than ECEF — Z_s0 ≈ 170_000 matches 170 km altitude,
not an ECEF Z. We follow the same convention and work entirely in a local
UTM CRS (auto-picked per scene centroid).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict, astuple
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Initial guess for satellite altitude (m). 2OC §3.3 uses 170_000 m.
NOMINAL_Z_S0: float = 170_000.0

# Bounds on the free parameters. These keep the LM fit in the convergence
# basin and prevent unphysical drifts. When the GCP set covers only a
# narrow along-track strip (common on Bahrain — populated region is a
# ~2 km Y-band out of a 20 km sub-frame footprint), the LM fit has a
# gauge freedom in Zs0/f and position/velocity. The L2 regularization
# (see _PRIOR_* below) together with these bounds keeps the solution
# physically plausible.
# KH-4B orbit ranges perigee 154 km → apogee 276 km; KH-9 PC is ~170 km.
# Widened to 140/280 km so per-sub-frame altitudes derived from ASP cam_gen
# are admitted across both missions. Narrow enough that a truly degenerate
# pose (free-Zs0 caller + empty GCPs) still hits a bound instead of
# drifting into orbital free-fall.
_Z_S0_MIN: float = 140_000.0
_Z_S0_MAX: float = 280_000.0
_OMEGA_RANGE_DEG: float = 45.0     # ±45° on the forward/aft tilt
_PHI_KAPPA_RANGE_DEG: float = 60.0  # ±60° on the other two Euler angles
# Rates and velocities are bounded tight to keep the fit's extrapolation
# (outside the row-range where GCPs live) physically reasonable. 2OC
# Table 6 reports rates of 0.02–0.4° per normalised t for KH-4B, so
# ±1° is a generous upper limit. Velocities are bounded to ±1 km per
# normalised t (the actual satellite velocity × scan time is ≈ 2.7 km,
# but the LINEAR rate coefficient in 2OC's eq 1 is typically a small
# perturbation to the nominal trajectory — Table 6 values are ~1–4 km).
_RATE_RANGE_DEG: float = 1.0       # ±1° per normalised scan time on the rates
_VEL_RANGE_M: float = 3_000.0      # ±3 km per normalised scan time on velocity
_P_RANGE: float = 0.1              # ±0.1 on the IMC coefficient
# f bounds kept at ±30 %. An earlier attempt at ±10 % drove every seg00
# fit above the 25-px hard gate (whole-strip fallback) — the data on
# Bahrain KH-9 D3C1213 genuinely needs f ≈ 1.1 m because the sub-frame's
# effective altitude is off-nominal; ±10 % excludes that basin. The price
# of ±30 % is a f=1.07 minimum with a visibly-misaligned seg00 seam that
# Phase 3 (when debugged) is meant to polish. See
# memory/per_segment_phase1_2_3_findings.md for the investigation.
_F_FRAC_RANGE: float = 0.30        # ±30 % on focal length
_XY_S0_RANGE_M: float = 50_000.0   # ±50 km around the initial centroid

# Prior "noise" scales for the L2 regularization (soft priors pulling
# each parameter toward its nominal value). Expressed in metres for
# positions, radians for angles, and appended to the 2N GCP residual
# vector. Each prior residual = (param - nominal) / sigma; a large sigma
# means a weak pull, small sigma means strong pull.
_PRIOR_Z_S0_SIGMA_M: float = 5_000.0    # pull Zs0 toward 170 km with 5 km noise
_PRIOR_RATE_SIGMA_RAD: float = 0.002    # rates weakly penalised (~0.11°)
_PRIOR_VEL_SIGMA_M: float = 200.0       # velocities weakly penalised
_PRIOR_F_FRAC_SIGMA: float = 0.005      # f penalised at 0.5 % deviation
_PRIOR_P_SIGMA: float = 0.05            # P penalised at 0.05 magnitude
_PRIOR_XY_S0_SIGMA_M: float = 5_000.0   # pull Xs0/Ys0 toward centroid with 5 km noise


# ---------------------------------------------------------------------------
# PanoramicParams dataclass
# ---------------------------------------------------------------------------

@dataclass
class PanoramicParams:
    """The 14 parameters of the 2OC panoramic camera model.

    All positions in a local planar CRS (typically UTM) in metres.
    All angles in radians. Linear-in-time coefficients are per unit of
    normalised time t ∈ [0, 1].

    Use :meth:`to_tensor` / :meth:`from_tensor` to convert to/from the
    flat torch tensor that the LM solver optimises.
    """

    # Initial exterior orientation at t=0
    Xs0: float
    Ys0: float
    Zs0: float
    omega0: float   # roll  (rad)
    phi0: float     # pitch (rad)
    kappa0: float   # yaw   (rad)

    # Linear-in-time velocity coefficients (metres per normalised t)
    Xs1: float
    Ys1: float
    Zs1: float

    # Linear-in-time attitude rate coefficients (rad per normalised t)
    omega1: float
    phi1: float
    kappa1: float

    # IMC (image motion compensation) coefficient (unitless)
    P: float

    # Focal length (m)
    f: float

    def to_tensor(self, device: str = "cpu"):
        """Return a flat torch tensor [Xs0, Ys0, Zs0, ω0, φ0, κ0,
        Xs1, Ys1, Zs1, ω1, φ1, κ1, P, f] on the given device.

        MPS does not support float64, so on MPS the tensor is float32
        and callers must cast to match. CPU and CUDA get float64 for
        full precision.
        """
        import torch
        dtype = torch.float32 if str(device).startswith("mps") else torch.float64
        return torch.tensor([
            self.Xs0, self.Ys0, self.Zs0,
            self.omega0, self.phi0, self.kappa0,
            self.Xs1, self.Ys1, self.Zs1,
            self.omega1, self.phi1, self.kappa1,
            self.P, self.f,
        ], dtype=dtype, device=device)

    @classmethod
    def from_tensor(cls, tensor) -> "PanoramicParams":
        """Inverse of :meth:`to_tensor`. Accepts torch tensor or numpy array."""
        if hasattr(tensor, "detach"):
            values = tensor.detach().cpu().numpy().tolist()
        else:
            values = list(map(float, np.asarray(tensor).ravel()))
        if len(values) != 14:
            raise ValueError(f"Expected 14 params, got {len(values)}")
        return cls(*values)

    def bounds(
        self,
        nominal_f: Optional[float] = None,
        f_frac_range: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds arrays for scipy.optimize.least_squares.

        Centred on the current values with the per-parameter ranges at
        module top. ``nominal_f`` lets the caller anchor the focal-length
        bounds on the profile's nominal rather than the (possibly drifted)
        current value. ``f_frac_range`` overrides the default
        ``_F_FRAC_RANGE`` so engineers can stage-tighten per Phase 9
        of the recovery plan (±30 % → ±5 % → ±2 %) without editing
        this module.
        """
        f_nom = nominal_f if nominal_f is not None else self.f
        f_range = (
            float(f_frac_range) if f_frac_range is not None else _F_FRAC_RANGE
        )
        omega_range = math.radians(_OMEGA_RANGE_DEG)
        phi_kappa_range = math.radians(_PHI_KAPPA_RANGE_DEG)
        rate_range = math.radians(_RATE_RANGE_DEG)
        lo = np.array([
            self.Xs0 - _XY_S0_RANGE_M,
            self.Ys0 - _XY_S0_RANGE_M,
            _Z_S0_MIN,
            self.omega0 - omega_range,
            self.phi0 - phi_kappa_range,
            self.kappa0 - phi_kappa_range,
            -_VEL_RANGE_M,
            -_VEL_RANGE_M,
            -_VEL_RANGE_M,
            -rate_range,
            -rate_range,
            -rate_range,
            -_P_RANGE,
            f_nom * (1.0 - f_range),
        ], dtype=np.float64)
        hi = np.array([
            self.Xs0 + _XY_S0_RANGE_M,
            self.Ys0 + _XY_S0_RANGE_M,
            _Z_S0_MAX,
            self.omega0 + omega_range,
            self.phi0 + phi_kappa_range,
            self.kappa0 + phi_kappa_range,
            +_VEL_RANGE_M,
            +_VEL_RANGE_M,
            +_VEL_RANGE_M,
            +rate_range,
            +rate_range,
            +rate_range,
            +_P_RANGE,
            f_nom * (1.0 + f_range),
        ], dtype=np.float64)
        return lo, hi

    @classmethod
    def from_corner_init(
        cls,
        corners_xy_local: List[Tuple[float, float]],
        nominal_f: float,
        forward_tilt_rad: float = 0.0,
        z_s0: float = NOMINAL_Z_S0,
    ) -> "PanoramicParams":
        """Construct a sensible initial parameter set from sub-frame corners.

        2OC §3.3 initial values:
            X_s0, Y_s0 = mean of GCPs (we use corner centroid)
            Z_s0       = 170 km
            ω0         = ±15° (forward_tilt, sign per forward/aft)
            φ0, κ0     = 0
            all rates  = 0
            P          = 0
            f          = nominal

        ``corners_xy_local`` is a list of 4 (X, Y) tuples in the local
        planar CRS — typically EPSG:3857 or UTM.
        """
        if len(corners_xy_local) == 0:
            raise ValueError("corners_xy_local must have at least 1 point")
        xs = [c[0] for c in corners_xy_local]
        ys = [c[1] for c in corners_xy_local]
        return cls(
            Xs0=float(sum(xs) / len(xs)),
            Ys0=float(sum(ys) / len(ys)),
            Zs0=float(z_s0),
            omega0=float(forward_tilt_rad),
            phi0=0.0,
            kappa0=0.0,
            Xs1=0.0,
            Ys1=0.0,
            Zs1=0.0,
            omega1=0.0,
            phi1=0.0,
            kappa1=0.0,
            P=0.0,
            f=float(nominal_f),
        )

    @classmethod
    def from_gcps_nadir(
        cls,
        sub_frame_gcps: np.ndarray,
        pixel_pitch: float,
        image_width_px: int,
        image_height_px: int,
        nominal_f: float,
        z_s0: float = NOMINAL_Z_S0,
    ) -> "PanoramicParams":
        """Estimate an initial 14-parameter set from the GCPs themselves.

        Inverts a nadir camera projection GCP-by-GCP:
            Xs_implied = X_gcp - Zs0 * tan(x_p / f)
            Ys_implied = Y_gcp - (y_p / f) * Zs0

        and takes the MEDIAN across all GCPs. This is typically a much
        better initial Xs0, Ys0 than the corner centroid, because the
        corners are often wrong by 5–15 km (USGS-quad-derived corners
        don't match the actual sub-frame footprint) and the correct
        camera position for an Aft-tilted camera is offset from the
        footprint centroid by Zs0 · tan(tilt).

        All 6 rates and P are initialised to 0, ω0/φ0/κ0 to 0 (the
        fit will pick them up from the GCP residuals).
        """
        if sub_frame_gcps.shape[0] < 3:
            raise ValueError(
                f"from_gcps_nadir needs ≥3 GCPs, got {sub_frame_gcps.shape[0]}"
            )
        cols = sub_frame_gcps[:, 0].astype(np.float64)
        rows = sub_frame_gcps[:, 1].astype(np.float64)
        X_g = sub_frame_gcps[:, 2].astype(np.float64)
        Y_g = sub_frame_gcps[:, 3].astype(np.float64)
        x_p = (cols - image_width_px / 2.0) * pixel_pitch
        y_p = -(rows - image_height_px / 2.0) * pixel_pitch
        # Nadir inversion (Z_gcp ≈ 0):
        #   x_p = f · atan(-N_x / N_z) with N_x = dx, N_z = -Zs ⇒ dx = Zs · tan(x_p/f)
        #   y_p = -f · (N_y / N_z)     with N_y = dy, N_z = -Zs ⇒ dy = (y_p / f) · Zs
        dx_implied = z_s0 * np.tan(x_p / float(nominal_f))
        dy_implied = (y_p / float(nominal_f)) * z_s0
        Xs_implied = np.median(X_g - dx_implied)
        Ys_implied = np.median(Y_g - dy_implied)
        return cls(
            Xs0=float(Xs_implied),
            Ys0=float(Ys_implied),
            Zs0=float(z_s0),
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
            f=float(nominal_f),
        )


# ---------------------------------------------------------------------------
# forward_project — 2OC eqs 1-21
# ---------------------------------------------------------------------------

def ecef_to_local_xyz(iC_ecef, local_crs: str) -> tuple:
    """Convert an ECEF camera centre (metres, from an ASP OpticalBar .tsai)
    into a 3-tuple ``(Xs0_local, Ys0_local, Zs0_altitude)`` where the
    first two are the camera's position projected to the given planar
    CRS (typically local UTM) at the ellipsoid surface, and the third
    is the altitude above the WGS84 ellipsoid.

    This is the minimum needed to inject cam_gen's per-sub-frame pose
    into the LM fit. The Euler-angle extraction from iR is not included
    because (a) the LM converges fine from (forward_tilt, 0, 0) as long
    as the position/altitude are right, and (b) extracting ECEF → local
    Euler cleanly requires ENU-from-ECEF rotation that's fiddly to get
    right for local planar CRSes.
    """
    from pyproj import Transformer
    import numpy as np

    iC = np.asarray(iC_ecef, dtype=np.float64).ravel()
    if iC.size != 3:
        raise ValueError(f"iC_ecef must be 3 elements, got {iC.size}")

    # ECEF → geodetic via pyproj (matches the closed-form helper in
    # camera_model.py but reuses the canonical transformer).
    tr_ecef_to_lla = Transformer.from_crs(
        "EPSG:4978", "EPSG:4326", always_xy=True,
    )
    lon_deg, lat_deg, alt_m = tr_ecef_to_lla.transform(
        float(iC[0]), float(iC[1]), float(iC[2]),
    )

    # (lon, lat) → local CRS. Most callers pass a UTM EPSG; this puts the
    # camera's ground-track position into the same planar frame the GCPs
    # are given in.
    tr_lla_to_local = Transformer.from_crs(
        "EPSG:4326", local_crs, always_xy=True,
    )
    x_local, y_local = tr_lla_to_local.transform(float(lon_deg), float(lat_deg))
    return float(x_local), float(y_local), float(alt_m)


def pano_params_to_opticalbar_tsai(
    params,
    pixel_pitch: float,
    image_width_px: int,
    image_height_px: int,
    local_crs: str,
    camera_params: dict,
    out_path: str,
    at_time: float = 0.5,
) -> None:
    """Write an ASP OpticalBar ``.tsai`` seed from 14-param pose at a
    single scan time.

    Phase 4: ASP's ``bundle_adjust`` needs seed cameras in its own
    OpticalBar format; we produce one per sub-frame by snapshotting the
    14-parameter pose at ``at_time`` (default scan-midpoint t=0.5). The
    linear-in-time rate terms collapse to a single
    ``(position, orientation)`` pair, so this is a LOSSY conversion —
    acceptable only because ASP refits the pose from tie-points + GCPs;
    the seed's job is to land in the right convergence basin.

    Parameters
    ----------
    params
        :class:`PanoramicParams` from a converged :func:`fit_panoramic`.
    pixel_pitch, image_width_px, image_height_px
        Sub-frame geometry in metres / pixels.
    local_crs
        pyproj CRS identifying ``params.Xs0``/``Ys0``'s horizontal
        frame (typically an EPSG:326xx UTM zone).
        ``params.Zs0`` is treated as elevation above the WGS84
        ellipsoid.
    camera_params
        Dict supplying ``scan_time``, ``speed``, ``forward_tilt``,
        ``scan_dir``, ``motion_compensation_factor`` — the OpticalBar
        scan kinematics. These are copied through unchanged (BA refines
        pose + focal length; scan mechanics are fixed).
    out_path
        Destination ``.tsai``.
    at_time
        Normalised scan time (0 = start, 1 = end, 0.5 = midpoint).
    """
    import math

    from pyproj import Transformer

    t = float(at_time)
    Xs_t = float(params.Xs0) + float(params.Xs1) * t
    Ys_t = float(params.Ys0) + float(params.Ys1) * t
    Zs_t = float(params.Zs0) + float(params.Zs1) * t
    omega_t = float(params.omega0) + float(params.omega1) * t
    phi_t = float(params.phi0) + float(params.phi1) * t
    kappa_t = float(params.kappa0) + float(params.kappa1) * t

    tr = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = tr.transform(Xs_t, Ys_t)

    # WGS84 (lat, lon, h-above-ellipsoid) → ECEF.
    a_wgs = 6378137.0
    f_flat = 1.0 / 298.257223563
    e_sq = 2.0 * f_flat - f_flat * f_flat
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    N_rad = a_wgs / math.sqrt(1.0 - e_sq * sin_lat * sin_lat)
    iC_x = (N_rad + Zs_t) * cos_lat * cos_lon
    iC_y = (N_rad + Zs_t) * cos_lat * sin_lon
    iC_z = (N_rad * (1.0 - e_sq) + Zs_t) * sin_lat

    # Rotation matrix world(ENU) → camera. Same convention as
    # forward_project — R = R_z(kappa) · R_y(phi) · R_x(omega).
    co = math.cos(omega_t); so = math.sin(omega_t)
    cp = math.cos(phi_t); sp = math.sin(phi_t)
    ck = math.cos(kappa_t); sk = math.sin(kappa_t)
    R_enu_to_cam = [
        [cp * ck,                co * sk + so * sp * ck,   so * sk - co * sp * ck],
        [-cp * sk,               co * ck - so * sp * sk,   so * ck + co * sp * sk],
        [sp,                     -so * cp,                 co * cp],
    ]
    # Rotation ECEF → local ENU at (lat, lon).
    R_ecef_to_enu = [
        [-sin_lon,              cos_lon,                  0.0],
        [-sin_lat * cos_lon,    -sin_lat * sin_lon,       cos_lat],
        [cos_lat * cos_lon,     cos_lat * sin_lon,        sin_lat],
    ]
    # iR = ECEF → camera = ENU→camera · ECEF→ENU.
    iR = [
        [
            sum(R_enu_to_cam[i][k] * R_ecef_to_enu[k][j] for k in range(3))
            for j in range(3)
        ]
        for i in range(3)
    ]
    iR_flat = [iR[i][j] for i in range(3) for j in range(3)]

    cx = image_width_px / 2.0
    cy = image_height_px / 2.0
    scan_time = float(camera_params.get("scan_time", 0.5))
    forward_tilt = float(camera_params.get("forward_tilt", 0.0))
    speed = float(camera_params.get("speed", 7800.0))
    mcf = float(camera_params.get("motion_compensation_factor", 1.0))
    scan_dir = str(camera_params.get("scan_dir", "right"))

    iC_str = f"{iC_x:.4f} {iC_y:.4f} {iC_z:.4f}"
    iR_str = " ".join(f"{v:.12f}" for v in iR_flat)

    with open(out_path, "w") as fh:
        fh.write(
            "VERSION_4\n"
            "OPTICAL_BAR\n"
            f"image_size = {int(image_width_px)} {int(image_height_px)}\n"
            f"image_center = {cx} {cy}\n"
            f"pitch = {pixel_pitch}\n"
            f"f = {float(params.f)}\n"
            f"scan_time = {scan_time}\n"
            f"forward_tilt = {forward_tilt}\n"
            f"iC = {iC_str}\n"
            f"iR = {iR_str}\n"
            f"speed = {speed}\n"
            "mean_earth_radius = 6371000\n"
            "mean_surface_elevation = 0.0\n"
            f"motion_compensation_factor = {mcf}\n"
            f"scan_dir = {scan_dir}\n"
        )


def forward_project(params_t, X, Y, Z, x_p_obs, L: float):
    """Predict panoramic image coordinates (x_p, y_p) from ground points.

    2OC eqs 1-21 in a single differentiable PyTorch function. The observed
    scan coordinate ``x_p_obs`` is used to compute t = x_p/L (eq 7) and
    α = x_p/f (eq 8) — this is how 2OC's LSQ fit parameterises the
    collinearity (each GCP's pixel x supplies t directly so the residual
    can be computed without an outer fixed-point iteration).

    Parameters
    ----------
    params_t : torch.Tensor
        Shape (14,) float64, the 14 parameters (see PanoramicParams).
    X, Y, Z : torch.Tensor
        Shape (N,) float64, ground points in the local planar CRS (m).
    x_p_obs : torch.Tensor
        Shape (N,) float64, observed x_p in metres on film. For the fit,
        this is the GCP's pixel-x × pixel_pitch. For the ortho iteration,
        it's the current estimate.
    L : float
        Film length along the scan axis in metres
        (= raw_sub_frame_width_in_pixels × pixel_pitch).

    Returns
    -------
    (x_p_model, y_p_model) : tuple of torch.Tensor, each shape (N,)
        Modelled panoramic coordinates per eqs 20 and 21.
    """
    import torch

    Xs0 = params_t[0]
    Ys0 = params_t[1]
    Zs0 = params_t[2]
    omega0 = params_t[3]
    phi0 = params_t[4]
    kappa0 = params_t[5]
    Xs1 = params_t[6]
    Ys1 = params_t[7]
    Zs1 = params_t[8]
    omega1 = params_t[9]
    phi1 = params_t[10]
    kappa1 = params_t[11]
    P = params_t[12]
    f = params_t[13]

    # Eq 7: normalised time per GCP (observed pixel x gives t directly).
    t = x_p_obs / L
    # Eq 8: cross-track scan angle.
    alpha = x_p_obs / f

    # Eqs 1-6: exterior orientation at time t.
    Xs_t = Xs0 + Xs1 * t
    Ys_t = Ys0 + Ys1 * t
    Zs_t = Zs0 + Zs1 * t
    omega_t = omega0 + omega1 * t
    phi_t = phi0 + phi1 * t
    kappa_t = kappa0 + kappa1 * t

    # Standard photogrammetric rotation matrix R = R_z(κ) R_y(φ) R_x(ω).
    # rows of R_t per point (N, 3, 3). All tensor ops are broadcast-friendly.
    co = torch.cos(omega_t)
    so = torch.sin(omega_t)
    cp = torch.cos(phi_t)
    sp = torch.sin(phi_t)
    ck = torch.cos(kappa_t)
    sk = torch.sin(kappa_t)

    r11 = cp * ck
    r12 = co * sk + so * sp * ck
    r13 = so * sk - co * sp * ck
    r21 = -cp * sk
    r22 = co * ck - so * sp * sk
    r23 = so * ck + co * sp * sk
    r31 = sp
    r32 = -so * cp
    r33 = co * cp

    # Eqs 15-17: N_x, N_y, N_z = rows of R_t dotted into (ground - camera_at_t).
    # 2OC has typos in eqs 16, 17 (r32 appears twice in eq 17); we use the
    # standard row-wise matrix-vector product, which is what the derivation
    # in the text actually means.
    dx = X - Xs_t
    dy = Y - Ys_t
    dz = Z - Zs_t

    N_x = r11 * dx + r12 * dy + r13 * dz
    N_y = r21 * dx + r22 * dy + r23 * dz
    N_z = r31 * dx + r32 * dy + r33 * dz

    # Eq 20: x_p = f · atan(-N_x / N_z)
    # Note: the paper uses SINGLE-argument atan (inverse tan), range
    # [-π/2, π/2]. This is fine for panoramic scans (< ±35°) and gives
    # the right sign for nadir-looking cameras where N_z < 0.
    # atan2(-N_x, N_z) is NOT equivalent — it flips the result by π
    # when N_z is negative because atan2 lives in [-π, π].
    # (Paper's eq 18 also has a typo saying tan α = -N_x/N_y, but the
    # derivation via dividing eq 14 row 1 by row 3 gives -N_x/N_z,
    # which is consistent with eq 20.)
    x_p_model = f * torch.atan(-N_x / N_z)

    # Eq 21: y_p = P f sin α cos ω_t - f cos α · (N_y / N_z)
    sin_a = torch.sin(alpha)
    cos_a = torch.cos(alpha)
    y_p_model = P * f * sin_a * torch.cos(omega_t) - f * cos_a * (N_y / N_z)

    return x_p_model, y_p_model


# ---------------------------------------------------------------------------
# raw_to_world — closed-form "raw pixel → world" projection
# ---------------------------------------------------------------------------

def raw_to_world(
    params: "PanoramicParams",
    cols: np.ndarray,
    rows: np.ndarray,
    pixel_pitch: float,
    image_width_px: int,
    image_height_px: int,
    z_world: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form raw (col, row) → world (X, Y) at a known ground elevation.

    Solves the 2OC collinearity equations for (X, Y) given a raw sub-frame
    pixel and a flat ground elevation ``z_world``. Unlike ``world_to_raw_pixel``
    (which iterates forward_project), this is a direct 2×2 linear solve per
    point derived from the collinearity row equations:

        N_x / N_z = -tan(α)                   (from eq 20)
        N_y / N_z = (P f sinα cosω - y_p) / (f cosα)   (from eq 21)

    where N = R_t · (X − Xs_t, Y − Ys_t, z_world − Zs_t). With the two
    ratios known, we have two linear equations in (dx, dy) = (X − Xs_t,
    Y − Ys_t); ``dz = z_world − Zs_t`` is known. Solve and recover
    X = Xs_t + dx, Y = Ys_t + dy.
    """
    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)

    x_p = (cols - image_width_px / 2.0) * pixel_pitch
    y_p = -(rows - image_height_px / 2.0) * pixel_pitch
    L = image_width_px * pixel_pitch

    Xs0, Ys0, Zs0 = params.Xs0, params.Ys0, params.Zs0
    omega0, phi0, kappa0 = params.omega0, params.phi0, params.kappa0
    Xs1, Ys1, Zs1 = params.Xs1, params.Ys1, params.Zs1
    omega1, phi1, kappa1 = params.omega1, params.phi1, params.kappa1
    P = params.P
    f = params.f

    t = x_p / L
    alpha = x_p / f

    Xs_t = Xs0 + Xs1 * t
    Ys_t = Ys0 + Ys1 * t
    Zs_t = Zs0 + Zs1 * t
    omega_t = omega0 + omega1 * t
    phi_t = phi0 + phi1 * t
    kappa_t = kappa0 + kappa1 * t

    co = np.cos(omega_t); so = np.sin(omega_t)
    cp = np.cos(phi_t); sp = np.sin(phi_t)
    ck = np.cos(kappa_t); sk = np.sin(kappa_t)

    # Same rotation matrix as forward_project.
    r11 = cp * ck
    r12 = co * sk + so * sp * ck
    r13 = so * sk - co * sp * ck
    r21 = -cp * sk
    r22 = co * ck - so * sp * sk
    r23 = so * ck + co * sp * sk
    r31 = sp
    r32 = -so * cp
    r33 = co * cp

    # Eq 21: y_p = P f sinα cosω_t - f cosα · (N_y / N_z)
    #   => N_y / N_z = (P f sinα cosω_t - y_p) / (f cosα)
    ratio_y = (P * f * np.sin(alpha) * np.cos(omega_t) - y_p) / (f * np.cos(alpha))
    # Eq 20: x_p = f atan(-N_x / N_z)  =>  N_x / N_z = -tan(α)
    tan_a = np.tan(alpha)

    dz = z_world - Zs_t

    # Linear system:
    #   N_x = r11*dx + r12*dy + r13*dz = -tan_a · N_z
    #   N_z = r31*dx + r32*dy + r33*dz
    # ⇒  (r11 + tan_a*r31) dx + (r12 + tan_a*r32) dy = -(r13 + tan_a*r33) dz
    #
    #   N_y = r21*dx + r22*dy + r23*dz = ratio_y · N_z
    # ⇒  (r21 - ratio_y*r31) dx + (r22 - ratio_y*r32) dy = -(r23 - ratio_y*r33) dz
    a00 = r11 + tan_a * r31
    a01 = r12 + tan_a * r32
    b0 = -(r13 + tan_a * r33) * dz

    a10 = r21 - ratio_y * r31
    a11 = r22 - ratio_y * r32
    b1 = -(r23 - ratio_y * r33) * dz

    det = a00 * a11 - a01 * a10
    # Avoid divide-by-zero on degenerate points.
    det_safe = np.where(np.abs(det) < 1e-20, 1e-20, det)
    dx = (a11 * b0 - a01 * b1) / det_safe
    dy = (a00 * b1 - a10 * b0) / det_safe

    X = Xs_t + dx
    Y = Ys_t + dy
    return X, Y


# ---------------------------------------------------------------------------
# world_to_raw_pixel — inverse projection via fixed-point iteration
# ---------------------------------------------------------------------------

def world_to_raw_pixel(
    params: "PanoramicParams",
    X_world: np.ndarray,
    Y_world: np.ndarray,
    Z_world: np.ndarray,
    pixel_pitch: float,
    image_width_px: int,
    image_height_px: int,
    max_fp_iters: int = 30,
    fp_tol_m: float = 1e-5,
):
    """Inverse-project world points to raw sub-frame pixel coordinates.

    Uses the same fixed-point iteration as :func:`mapproject` but for a
    set of scattered world points rather than a dense output grid. For
    each (X, Y, Z), iteratively solve
        x_p = forward_project(params, X, Y, Z, x_p)
    to convergence, then convert to raw pixel (col, row).

    Used by the tie-point refinement pass: given world coordinates from
    RoMa ortho matches, find the raw pixels in each sub-frame so we can
    create tie-point GCPs.

    Returns
    -------
    (cols, rows) : tuple of np.ndarray, each shape (N,)
        Raw sub-frame pixel coordinates (float, may be fractional).
    """
    import torch

    params_t = params.to_tensor("cpu").to(torch.float64)
    X_t = torch.from_numpy(np.asarray(X_world, dtype=np.float64))
    Y_t = torch.from_numpy(np.asarray(Y_world, dtype=np.float64))
    Z_t = torch.from_numpy(np.asarray(Z_world, dtype=np.float64))
    n = X_t.numel()
    L = float(image_width_px) * pixel_pitch

    x_p = torch.zeros(n, dtype=torch.float64)
    for _ in range(max_fp_iters):
        xp_new, yp_new = forward_project(params_t, X_t, Y_t, Z_t, x_p, L)
        if torch.max(torch.abs(xp_new - x_p)) < fp_tol_m:
            x_p = xp_new
            y_p_final = yp_new
            break
        x_p = xp_new
    else:
        _, y_p_final = forward_project(params_t, X_t, Y_t, Z_t, x_p, L)

    cols = (x_p.numpy() / pixel_pitch) + image_width_px / 2.0
    rows = (-y_p_final.numpy() / pixel_pitch) + image_height_px / 2.0
    return cols, rows


# ---------------------------------------------------------------------------
# fit_panoramic — LM fit of 14 params to GCPs
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Result of a :func:`fit_panoramic` call."""
    params: PanoramicParams
    reprojection_rms_px: float
    reprojection_rms_m: float
    n_gcps: int
    n_iterations: int
    success: bool
    message: str


def fit_panoramic(
    sub_frame_gcps: np.ndarray,
    initial: PanoramicParams,
    pixel_pitch: float,
    image_width_px: int,
    image_height_px: int,
    nominal_f: Optional[float] = None,
    max_iter: int = 50,
    loss: str = "linear",
    f_scale_px: float = 2.0,
    verbose: bool = False,
    regularize: bool = True,
    fix_zs0: bool = True,
    fix_f: bool = False,
    fix_velocities: bool = False,
    fix_rates: bool = False,
    fix_p: bool = False,
    zs0_prior_sigma_m: Optional[float] = None,
    f_frac_range: Optional[float] = None,
    f_prior_frac_sigma: Optional[float] = None,
) -> FitResult:
    """Levenberg–Marquardt fit of the 14 parameters to a list of GCPs.

    Parameters
    ----------
    sub_frame_gcps : np.ndarray
        Shape (N, 5), columns [col, row, X_local, Y_local, Z_local] where
        (col, row) are pixel coordinates in the raw sub-frame and
        (X_local, Y_local, Z_local) are the corresponding ground points
        in the same local planar CRS used by ``initial``.
    initial : PanoramicParams
        Starting point for the LM iteration — typically from
        :meth:`PanoramicParams.from_corner_init`.
    pixel_pitch : float
        Size of one raw sub-frame pixel in metres (e.g. 7e-6 m = 7 µm).
    image_width_px : int
        Width of the raw sub-frame in pixels (used to compute L = w * pitch).
    image_height_px : int
        Height of the raw sub-frame in pixels (used to centre y_p on the
        principal point at row image_height_px / 2).
    nominal_f : float, optional
        Nominal focal length for the bounds' centre. Defaults to
        ``initial.f``.
    max_iter : int
        Maximum LM iterations.
    loss : str
        scipy least_squares loss function. Default 'linear' (plain L2,
        relies on RANSAC-pre-filtered GCPs). Use 'cauchy' or 'huber'
        if GCPs may contain outliers.
    f_scale_px : float
        Robust-loss scale in pixels (≈ typical observation noise). Only
        used when ``loss != 'linear'``.
    verbose : bool
        Print per-iteration convergence diagnostics.
    regularize : bool
        If True (default), append L2 prior residuals pulling Zs0 toward
        170 km, rates toward 0, velocities toward 0, f toward nominal,
        P toward 0, and Xs0/Ys0 toward the centroid of the GCPs. This
        is essential when GCPs cover only a narrow strip of the scan
        (gauge freedom in Zs0/f otherwise pushes the fit to bounds).
    fix_zs0 : bool
        If True (default), hard-fix Zs0 = NOMINAL_Z_S0 (170 km) during
        the fit. Zs0 has a gauge freedom with f and Xs0/Ys0/position —
        pinning it breaks the freedom and keeps the fit in the physical
        basin.
    fix_f : bool
        If True, also hard-fix f = nominal_f. When False (default), f is
        left free within its bounds: this is 2OC's adaptive focal-length
        approach (§4.4), which is necessary on KH-9 PC where the
        effective f per sub-frame can differ 20–30 % from nominal (the
        USGS sub-frame dimensions imply along-track ground scale 18 %
        larger than cross-track, which the LM fit compensates via f).
    zs0_prior_sigma_m : float, optional
        Override the default Gaussian prior σ on ``Zs0`` (default
        :data:`_PRIOR_Z_S0_SIGMA_M` = 5 km). Pass 1 000 m when ``initial.
        Zs0`` comes from a high-confidence source (TLE propagated to
        closest pass) so the LM fit is strongly pulled to the prior
        altitude rather than drifting within a 5 km basin.

    Returns
    -------
    FitResult
    """
    import torch
    from scipy.optimize import least_squares

    if sub_frame_gcps.shape[0] < 7:
        return FitResult(
            params=initial,
            reprojection_rms_px=float("inf"),
            reprojection_rms_m=float("inf"),
            n_gcps=int(sub_frame_gcps.shape[0]),
            n_iterations=0,
            success=False,
            message=f"need at least 7 GCPs, got {sub_frame_gcps.shape[0]}",
        )

    cols = sub_frame_gcps[:, 0].astype(np.float64)
    rows = sub_frame_gcps[:, 1].astype(np.float64)
    X = sub_frame_gcps[:, 2].astype(np.float64)
    Y = sub_frame_gcps[:, 3].astype(np.float64)
    Z = sub_frame_gcps[:, 4].astype(np.float64)

    L = float(image_width_px) * pixel_pitch
    # Observed film coordinates relative to the image principal point at
    # (image_width/2, image_height/2). x_p ∈ [-L/2, +L/2] so α = x_p / f
    # is centred at 0 in the scan middle (eq 8). y_p is the along-track
    # offset on the slit at time t.
    #
    # Photogrammetric convention: film y-axis is UP-positive, but raster
    # row indices increase DOWNWARD. So the row→y_p conversion needs a
    # minus sign to match forward_project's y_p_model output.
    x_p_obs = (cols - image_width_px / 2.0) * pixel_pitch
    y_p_obs_np = -(rows - image_height_px / 2.0) * pixel_pitch

    device = "cpu"  # CPU is plenty for 14-param fit over ~2k GCPs
    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    Z_t = torch.from_numpy(Z).to(device)
    xp_obs_t = torch.from_numpy(x_p_obs).to(device)
    yp_obs_t = torch.from_numpy(y_p_obs_np).to(device)

    f_nom_val = float(nominal_f) if nominal_f is not None else float(initial.f)
    cx_init = float(np.mean(X))
    cy_init = float(np.mean(Y))

    # Zs0 prior: caller-supplied σ anchors toward ``initial.Zs0`` (a per-
    # mission TLE or cam_gen altitude); default σ anchors toward the 170 km
    # nominal (the legacy behaviour).
    zs0_prior_mean = (
        float(initial.Zs0) if zs0_prior_sigma_m is not None else NOMINAL_Z_S0
    )

    # Full-14 prior means and inverse sigmas (only used when regularize=True).
    full_prior_mean = np.array([
        cx_init,                 # Xs0
        cy_init,                 # Ys0
        zs0_prior_mean,          # Zs0 → caller prior or 170 km nominal
        initial.omega0,
        initial.phi0,
        initial.kappa0,
        0.0, 0.0, 0.0,           # velocities → 0
        0.0, 0.0, 0.0,           # rates → 0
        0.0,                     # P → 0
        f_nom_val,               # f → nominal
    ], dtype=np.float64)
    two_px_m = 2.0 * pixel_pitch
    zs0_sigma = (
        float(zs0_prior_sigma_m) if zs0_prior_sigma_m is not None
        else _PRIOR_Z_S0_SIGMA_M
    )
    f_prior_frac = (
        float(f_prior_frac_sigma) if f_prior_frac_sigma is not None
        else _PRIOR_F_FRAC_SIGMA
    )
    full_prior_inv_sigma = np.array([
        two_px_m / _PRIOR_XY_S0_SIGMA_M,
        two_px_m / _PRIOR_XY_S0_SIGMA_M,
        two_px_m / zs0_sigma,
        two_px_m / math.radians(5.0),
        two_px_m / math.radians(10.0),
        two_px_m / math.radians(10.0),
        two_px_m / _PRIOR_VEL_SIGMA_M,
        two_px_m / _PRIOR_VEL_SIGMA_M,
        two_px_m / _PRIOR_VEL_SIGMA_M,
        two_px_m / _PRIOR_RATE_SIGMA_RAD,
        two_px_m / _PRIOR_RATE_SIGMA_RAD,
        two_px_m / _PRIOR_RATE_SIGMA_RAD,
        two_px_m / _PRIOR_P_SIGMA,
        two_px_m / (f_prior_frac * f_nom_val),
    ], dtype=np.float64)

    # Build free-index list: drop requested fixed indices.
    # Index layout: [Xs0, Ys0, Zs0, ω0, φ0, κ0, Xs1, Ys1, Zs1, ω1, φ1, κ1, P, f]
    #                 0    1    2    3   4   5    6    7    8    9  10  11  12 13
    fixed_values = {}
    free_indices = list(range(14))
    if fix_zs0:
        # Respect the caller-supplied Zs0 in ``initial`` so ASP cam_gen's
        # per-sub-frame altitude (or any other per-frame altitude source)
        # can replace the hard-coded 170 km nominal. Bahrain KH-9 D3C1213
        # seg00 was observed to collapse to f=1.07 m when Zs0 was pinned at
        # 170 km against an off-nominal actual altitude; threading the
        # cam_gen altitude here breaks the f/Zs0 gauge freedom.
        fixed_values[2] = float(initial.Zs0)
    if fix_velocities:
        fixed_values[6] = 0.0
        fixed_values[7] = 0.0
        fixed_values[8] = 0.0
    if fix_rates:
        fixed_values[9] = 0.0
        fixed_values[10] = 0.0
        fixed_values[11] = 0.0
    if fix_p:
        fixed_values[12] = 0.0
    if fix_f:
        fixed_values[13] = f_nom_val
    free_indices = [i for i in range(14) if i not in fixed_values]

    free_indices_arr = np.array(free_indices, dtype=int)
    prior_mean = full_prior_mean[free_indices_arr]
    prior_inv_sigma = full_prior_inv_sigma[free_indices_arr]
    if not regularize:
        prior_inv_sigma = np.zeros_like(prior_inv_sigma)

    def _expand(params_free: np.ndarray) -> np.ndarray:
        """Scatter 'free' params back into the 14-vector, filling fixed ones."""
        full = np.empty(14, dtype=np.float64)
        for k, idx in enumerate(free_indices):
            full[idx] = params_free[k]
        for idx, val in fixed_values.items():
            full[idx] = val
        return full

    def residual_np(params_free: np.ndarray) -> np.ndarray:
        full = _expand(params_free)
        params_t = torch.from_numpy(full).to(device)
        xp_mod, yp_mod = forward_project(params_t, X_t, Y_t, Z_t, xp_obs_t, L)
        res_x = (xp_mod - xp_obs_t).detach().cpu().numpy()
        res_y = (yp_mod - yp_obs_t).detach().cpu().numpy()
        res_prior = (params_free - prior_mean) * prior_inv_sigma
        return np.concatenate([res_x, res_y, res_prior])

    n_free = len(free_indices)
    _prior_jac_const = np.diag(prior_inv_sigma)

    def jacobian_np(params_free: np.ndarray) -> np.ndarray:
        full = _expand(params_free)
        full_t = torch.from_numpy(full).to(device)

        def _stacked(p_full):
            xp_mod, yp_mod = forward_project(p_full, X_t, Y_t, Z_t, xp_obs_t, L)
            return torch.cat([
                xp_mod - xp_obs_t,
                yp_mod - yp_obs_t,
            ])

        jac_full = torch.autograd.functional.jacobian(_stacked, full_t)
        # Select only columns for free parameters
        jac_free = jac_full[:, free_indices_arr].detach().cpu().numpy()
        return np.vstack([jac_free, _prior_jac_const])

    # Build initial free-param vector, bounds, x_scale
    p0_full = initial.to_tensor(device).detach().cpu().numpy()
    lo_full, hi_full = initial.bounds(
        nominal_f=nominal_f, f_frac_range=f_frac_range,
    )
    p0 = p0_full[free_indices_arr]
    lo = lo_full[free_indices_arr]
    hi = hi_full[free_indices_arr]
    p0_clipped = np.clip(p0, lo + 1e-9, hi - 1e-9)

    f_scale = f_scale_px * pixel_pitch

    full_x_scale = np.array([
        10_000.0, 10_000.0, 10_000.0,
        0.1, 0.1, 0.1,
        1_000.0, 1_000.0, 1_000.0,
        0.01, 0.01, 0.01,
        0.1, 0.01,
    ], dtype=np.float64)
    x_scale = full_x_scale[free_indices_arr]

    result = least_squares(
        residual_np,
        p0_clipped,
        jac=jacobian_np,
        bounds=(lo, hi),
        method="trf",
        loss=loss,
        f_scale=f_scale,
        x_scale=x_scale,
        max_nfev=max_iter * max(n_free, 1),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        verbose=2 if verbose else 0,
    )

    fitted = PanoramicParams.from_tensor(_expand(result.x))
    n = sub_frame_gcps.shape[0]
    # Strip the n_free prior pseudo-residuals so the reported RMS reflects
    # actual reprojection error on the GCPs only.
    gcp_residuals = result.fun[: 2 * n]
    rms_m = float(np.sqrt(np.mean(gcp_residuals ** 2)))
    rms_px = rms_m / pixel_pitch
    return FitResult(
        params=fitted,
        reprojection_rms_px=rms_px,
        reprojection_rms_m=rms_m,
        n_gcps=n,
        n_iterations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


# ---------------------------------------------------------------------------
# mapproject — 2OC §3.4 Algorithm 1 time-iterative ortho
# ---------------------------------------------------------------------------

def mapproject(
    params: PanoramicParams,
    sub_frame_path: str,
    dem_path: Optional[str],
    out_path: str,
    pixel_pitch: float,
    image_width_px: int,
    image_height_px: int,
    resolution_m: float,
    bbox_xy: Tuple[float, float, float, float],
    local_crs: str,
    t_srs: str = "EPSG:3857",
    device: Optional[str] = None,
    chunk_px: int = 2048,
    max_fp_iters: int = 50,
    fp_tol_m: float = 1e-5,
    flat_z: float = 0.0,
) -> Optional[str]:
    """Time-iterative orthorectification of a raw sub-frame (2OC Algorithm 1).

    For each output ortho pixel at geographic position (X_local, Y_local),
    look up the DEM elevation Z, then fixed-point iterate:
        t = x_p / L;  α = x_p / f;
        (x_p_new, y_p_new) = forward_project(params, X, Y, Z, x_p_old, L)
        if ||(x_p_new - x_p_old, y_p_new - y_p_old)|| < tol: break
    Sample the raw sub-frame at the converged (x_p, y_p) bilinearly.

    The output canvas is in ``local_crs`` at ``resolution_m`` m/px,
    bounded by ``bbox_xy`` = (west, south, east, north). If ``t_srs``
    differs from ``local_crs``, the raster is reprojected after writing
    so the blend pipeline can place it on the EPSG:3857 canvas.

    Implementation notes:
    - Tile the output in ``chunk_px`` × ``chunk_px`` blocks for memory.
    - Use PyTorch for vectorised forward_project + bilinear sampling.
    - Use bilinear DEM interpolation (or ``flat_z`` if no DEM given).
    """
    import torch
    import rasterio
    from rasterio.transform import from_bounds as tfm_from_bounds
    from rasterio.warp import reproject, Resampling, calculate_default_transform

    if device is None:
        try:
            from align.models import get_torch_device
            device = get_torch_device()
        except Exception:
            device = "cpu"
    device = str(device)

    # MPS does not support float64. Degrade to float32 on MPS;
    # keep float64 on CPU and CUDA for full precision.
    fp_dtype = torch.float32 if device.startswith("mps") else torch.float64

    # Read the raw sub-frame into memory once (small enough — ~70 MB)
    with rasterio.open(sub_frame_path) as src:
        sf_data = src.read(1)
        sf_nodata = src.nodata
        sf_dtype = sf_data.dtype
    sf_h, sf_w = sf_data.shape
    if sf_w != image_width_px or sf_h != image_height_px:
        raise ValueError(
            f"sub-frame dims {sf_w}x{sf_h} != expected "
            f"{image_width_px}x{image_height_px}")

    sf_t = torch.from_numpy(sf_data.astype(np.float32)).to(device)

    # Output grid
    bl, bb, br, bt = bbox_xy
    out_w = int(round((br - bl) / resolution_m))
    out_h = int(round((bt - bb) / resolution_m))
    if out_w <= 0 or out_h <= 0:
        return None
    out_tfm = tfm_from_bounds(bl, bb, br, bt, out_w, out_h)

    L = float(image_width_px) * pixel_pitch
    params_t = params.to_tensor(device)
    # Ensure params tensor is the right dtype for the device
    params_t = params_t.to(fp_dtype)

    # DEM lookup: load DEM once if given, otherwise use flat_z.
    dem_arr = None
    dem_tfm = None
    dem_crs = None
    dem_nodata = None
    if dem_path and os.path.isfile(dem_path):
        try:
            with rasterio.open(dem_path) as dem:
                dem_arr = dem.read(1).astype(np.float32)
                dem_tfm = dem.transform
                dem_crs = dem.crs
                dem_nodata = dem.nodata
        except Exception as e:
            print(f"  [kh_panoramic.mapproject] DEM read failed: {e}")
            dem_arr = None

    def _sample_dem(x_local: torch.Tensor, y_local: torch.Tensor) -> torch.Tensor:
        if dem_arr is None:
            return torch.full_like(x_local, flat_z, dtype=x_local.dtype)
        # Project local CRS → DEM CRS (typically EPSG:4326)
        from pyproj import Transformer
        x_np = x_local.detach().cpu().numpy()
        y_np = y_local.detach().cpu().numpy()
        try:
            tr = Transformer.from_crs(local_crs, dem_crs, always_xy=True)
            lon_np, lat_np = tr.transform(x_np, y_np)
        except Exception:
            return torch.full_like(x_local, flat_z, dtype=x_local.dtype)
        # Bilinear sample
        cols = (lon_np - dem_tfm.c) / dem_tfm.a
        rows = (lat_np - dem_tfm.f) / dem_tfm.e
        c0 = np.floor(cols).astype(int)
        r0 = np.floor(rows).astype(int)
        cf = cols - c0
        rf = rows - r0
        h, w = dem_arr.shape
        c0 = np.clip(c0, 0, w - 2)
        r0 = np.clip(r0, 0, h - 2)
        z00 = dem_arr[r0, c0]
        z01 = dem_arr[r0, c0 + 1]
        z10 = dem_arr[r0 + 1, c0]
        z11 = dem_arr[r0 + 1, c0 + 1]
        z = (z00 * (1 - cf) * (1 - rf)
             + z01 * cf * (1 - rf)
             + z10 * (1 - cf) * rf
             + z11 * cf * rf)
        if dem_nodata is not None:
            mask = (z00 == dem_nodata) | (z01 == dem_nodata) | (z10 == dem_nodata) | (z11 == dem_nodata)
            z = np.where(mask, flat_z, z)
        return torch.from_numpy(z.astype(np.float32)).to(x_local.device, dtype=x_local.dtype)

    # Allocate output array (rasterio-compatible, in numpy)
    out_arr = np.full((out_h, out_w), np.nan if sf_nodata is None else sf_nodata,
                      dtype=np.float32)

    sf_w_f = float(image_width_px)
    sf_h_f = float(image_height_px)

    # Tile the output
    for y0 in range(0, out_h, chunk_px):
        y1 = min(out_h, y0 + chunk_px)
        for x0 in range(0, out_w, chunk_px):
            x1 = min(out_w, x0 + chunk_px)
            tw = x1 - x0
            th = y1 - y0
            # Local CRS coordinates of each output pixel centre in this tile
            x_grid = bl + (x0 + 0.5 + torch.arange(tw, device=device, dtype=fp_dtype)) * resolution_m
            y_grid = bt - (y0 + 0.5 + torch.arange(th, device=device, dtype=fp_dtype)) * resolution_m
            X_t, Y_t = torch.meshgrid(x_grid, y_grid, indexing="xy")
            X_t = X_t.reshape(-1)
            Y_t = Y_t.reshape(-1)
            n = X_t.numel()
            Z_t = _sample_dem(X_t, Y_t).to(fp_dtype)

            # Initial: x_p = 0 (centre of scan), y_p = 0
            x_p = torch.zeros(n, dtype=fp_dtype, device=device)
            for _ in range(max_fp_iters):
                xp_new, yp_new = forward_project(params_t, X_t, Y_t, Z_t, x_p, L)
                if torch.max(torch.abs(xp_new - x_p)) < fp_tol_m:
                    x_p = xp_new
                    y_p_final = yp_new
                    break
                x_p = xp_new
            else:
                _, y_p_final = forward_project(params_t, X_t, Y_t, Z_t, x_p, L)

            # Convert (x_p, y_p) on film to raw pixel (col, row) in sub-frame.
            # Film y-axis is UP-positive, row index is DOWN-positive — mirror
            # y_p to get row (matches fit_panoramic's row→y_p convention).
            cols = (x_p / pixel_pitch) + sf_w_f / 2.0
            rows = (-y_p_final / pixel_pitch) + sf_h_f / 2.0

            # Bilinear sample sub-frame
            in_bounds = ((cols >= 0) & (cols <= sf_w_f - 1)
                         & (rows >= 0) & (rows <= sf_h_f - 1))
            cols_safe = torch.clamp(cols, 0, sf_w_f - 1)
            rows_safe = torch.clamp(rows, 0, sf_h_f - 1)
            c0_idx = torch.floor(cols_safe).to(torch.long)
            r0_idx = torch.floor(rows_safe).to(torch.long)
            c0_idx = torch.clamp(c0_idx, 0, image_width_px - 2)
            r0_idx = torch.clamp(r0_idx, 0, image_height_px - 2)
            c1_idx = c0_idx + 1
            r1_idx = r0_idx + 1
            cf = torch.clamp(cols_safe - c0_idx.to(fp_dtype), 0.0, 1.0)
            rf = torch.clamp(rows_safe - r0_idx.to(fp_dtype), 0.0, 1.0)
            v00 = sf_t[r0_idx, c0_idx]
            v01 = sf_t[r0_idx, c1_idx]
            v10 = sf_t[r1_idx, c0_idx]
            v11 = sf_t[r1_idx, c1_idx]
            sampled = (v00.to(fp_dtype) * (1 - cf) * (1 - rf)
                       + v01.to(fp_dtype) * cf * (1 - rf)
                       + v10.to(fp_dtype) * (1 - cf) * rf
                       + v11.to(fp_dtype) * cf * rf)

            # Mark out-of-bounds and nodata
            if sf_nodata is not None:
                v_nd = (v00 == sf_nodata) | (v01 == sf_nodata) | (v10 == sf_nodata) | (v11 == sf_nodata)
                sampled = torch.where(v_nd, torch.tensor(float(sf_nodata), dtype=fp_dtype, device=device), sampled)
            sampled = torch.where(in_bounds, sampled, torch.tensor(float(sf_nodata) if sf_nodata is not None else 0.0,
                                                                   dtype=fp_dtype, device=device))

            tile = sampled.reshape(th, tw).detach().cpu().numpy().astype(np.float32)
            out_arr[y0:y1, x0:x1] = tile

    # Write the output in local_crs first
    profile = {
        "driver": "GTiff",
        "width": out_w,
        "height": out_h,
        "count": 1,
        "dtype": "float32",
        "crs": local_crs,
        "transform": out_tfm,
        "nodata": sf_nodata if sf_nodata is not None else -32768,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "bigtiff": "if_safer",
    }

    if local_crs == t_srs:
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_arr, 1)
        return out_path

    # Write to a tmp local-CRS file, then reproject to t_srs
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(tmp_fd)
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(out_arr, 1)
        with rasterio.open(tmp_path) as src:
            dst_tfm, dst_w, dst_h = calculate_default_transform(
                src.crs, t_srs, src.width, src.height, *src.bounds,
                resolution=resolution_m,
            )
            dst_profile = profile.copy()
            dst_profile.update(crs=t_srs, transform=dst_tfm,
                               width=dst_w, height=dst_h)
            with rasterio.open(out_path, "w", **dst_profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_tfm,
                    dst_crs=t_srs,
                    resampling=Resampling.bilinear,
                )
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
    return out_path


def _sample_dem_local_xy(
    X_local: np.ndarray,
    Y_local: np.ndarray,
    dem_path: Optional[str],
    local_crs: str,
) -> np.ndarray:
    """Sample DEM elevations for local-planar XY coordinates."""
    if dem_path is None or not os.path.isfile(dem_path):
        return np.zeros(len(X_local), dtype=np.float64)

    try:
        import rasterio
        from pyproj import Transformer as _T
    except ImportError:
        return np.zeros(len(X_local), dtype=np.float64)

    try:
        with rasterio.open(dem_path) as dem:
            dem_arr = dem.read(1).astype(np.float32)
            dem_tfm = dem.transform
            dem_crs = dem.crs
            dem_nd = dem.nodata
        tr_local_to_dem = _T.from_crs(local_crs, dem_crs, always_xy=True)
        x_dem, y_dem = tr_local_to_dem.transform(X_local, Y_local)
        cols = (x_dem - dem_tfm.c) / dem_tfm.a
        rows = (y_dem - dem_tfm.f) / dem_tfm.e
        c0 = np.floor(cols).astype(int)
        r0 = np.floor(rows).astype(int)
        cf = cols - c0
        rf = rows - r0
        h, w = dem_arr.shape
        c0 = np.clip(c0, 0, w - 2)
        r0 = np.clip(r0, 0, h - 2)
        z00 = dem_arr[r0, c0]
        z01 = dem_arr[r0, c0 + 1]
        z10 = dem_arr[r0 + 1, c0]
        z11 = dem_arr[r0 + 1, c0 + 1]
        z = (
            z00 * (1 - cf) * (1 - rf)
            + z01 * cf * (1 - rf)
            + z10 * (1 - cf) * rf
            + z11 * cf * rf
        ).astype(np.float64)
        if dem_nd is not None:
            invalid = (
                (z00 == dem_nd) | (z01 == dem_nd)
                | (z10 == dem_nd) | (z11 == dem_nd)
            )
            z = np.where(invalid, 0.0, z)
        return z
    except Exception:
        return np.zeros(len(X_local), dtype=np.float64)


def _gcp_distribution_summary(
    gcps: np.ndarray,
    image_width_px: int,
    image_height_px: int,
    grid_cols: int = 6,
    grid_rows: int = 3,
):
    """Summarize how well GCPs cover the raw sub-frame."""
    if gcps is None or gcps.shape[0] == 0:
        return {
            "occupied_cells": 0,
            "occupied_cols": 0,
            "occupied_rows": 0,
            "left_count": 0,
            "center_count": 0,
            "right_count": 0,
            "col_span_frac": 0.0,
            "row_span_frac": 0.0,
        }

    cols = gcps[:, 0].astype(np.float64)
    rows = gcps[:, 1].astype(np.float64)
    valid = (
        np.isfinite(cols) & np.isfinite(rows)
        & (cols >= 0) & (cols < image_width_px)
        & (rows >= 0) & (rows < image_height_px)
    )
    if valid.sum() == 0:
        return {
            "occupied_cells": 0,
            "occupied_cols": 0,
            "occupied_rows": 0,
            "left_count": 0,
            "center_count": 0,
            "right_count": 0,
            "col_span_frac": 0.0,
            "row_span_frac": 0.0,
        }

    cols = cols[valid]
    rows = rows[valid]
    left_count = int((cols < image_width_px / 3.0).sum())
    center_count = int(((cols >= image_width_px / 3.0) & (cols < 2.0 * image_width_px / 3.0)).sum())
    right_count = int((cols >= 2.0 * image_width_px / 3.0).sum())

    cell_w = image_width_px / float(grid_cols)
    cell_h = image_height_px / float(grid_rows)
    occupied = set()
    for col, row in zip(cols, rows):
        gx = min(grid_cols - 1, max(0, int(col / cell_w)))
        gy = min(grid_rows - 1, max(0, int(row / cell_h)))
        occupied.add((gx, gy))
    occupied_cols = {gx for gx, _ in occupied}
    occupied_rows = {gy for _, gy in occupied}

    col_span = (cols.max() - cols.min()) / max(float(image_width_px), 1.0)
    row_span = (rows.max() - rows.min()) / max(float(image_height_px), 1.0)
    return {
        "occupied_cells": len(occupied),
        "occupied_cols": len(occupied_cols),
        "occupied_rows": len(occupied_rows),
        "left_count": left_count,
        "center_count": center_count,
        "right_count": right_count,
        "col_span_frac": float(col_span),
        "row_span_frac": float(row_span),
    }


def _gcp_distribution_ok(
    gcps: np.ndarray,
    image_width_px: int,
    image_height_px: int,
    *,
    min_occupied_cells: int = 8,
    min_occupied_cols: int = 5,
    min_points_per_third: int = 5,
    full_width_span_frac: float = 0.70,
):
    """Return whether the GCP set spans the scan width well enough.

    The coarse raw→reference pass only needs robust cross-scan coverage.
    Bahrain-like scenes often occupy a narrow along-track band in the raw
    sub-frame, so a strict 6x3 occupied-cell threshold rejects otherwise good
    GCP sets. Accept either:
    - broad 2D coverage (enough occupied cells), or
    - a narrow band that still spans most of the scan width.
    """
    summary = _gcp_distribution_summary(gcps, image_width_px, image_height_px)
    thirds_ok = min(
        summary["left_count"],
        summary["center_count"],
        summary["right_count"],
    ) >= min_points_per_third
    width_ok = (
        summary["occupied_cols"] >= min_occupied_cols
        and summary["col_span_frac"] >= 0.55
    )
    spatial_ok = (
        summary["occupied_cells"] >= min_occupied_cells
        or summary["col_span_frac"] >= full_width_span_frac
    )
    ok = thirds_ok and width_ok and spatial_ok
    return ok, summary


# ---------------------------------------------------------------------------
# extract_reference_gcps — RoMa dense matching against reference
# ---------------------------------------------------------------------------

def extract_reference_gcps(
    sub_frame_path: str,
    reference_ortho_path: str,
    initial_corners_ll: dict,
    local_crs: str,
    matcher_name: str = "roma",
    matcher_runtime=None,
    match_res_m: float = 4.0,
    search_pad_m: float = 2000.0,
    dem_path: Optional[str] = None,
    max_matches: int = 5000,
    max_tiles: int = 300,
    ransac_reproj_px: float = 8.0,
    native_gsd_m: Optional[float] = None,
    mte_enabled: bool = False,
    mte_radius_px: float = 500.0,
) -> Optional[np.ndarray]:
    """Use RoMa v2 to dense-match a raw sub-frame against a reference ortho.

    Returns an (N, 5) numpy array with columns [col, row, X_local, Y_local,
    Z_local] suitable for ``fit_panoramic``. col/row are in the raw
    sub-frame's pixel grid; X_local/Y_local are in ``local_crs``; Z_local
    is the DEM elevation at each point (or 0.0 if ``dem_path`` is None).

    Pipeline:
      1. Read the raw sub-frame into a uint8 (CLAHE-normalised) array.
      2. Reproject the reference to ``initial_corners_ll``'s geographic
         bbox at ``match_res_m`` m/px in ``local_crs``.
      3. Normalise both to uint8, run ``_run_roma_tiled`` for dense
         correspondences in the reference-grid pixel coordinates.
      4. Convert reference pixel coords to local_crs (X, Y) via the
         reference grid transform.
      5. RANSAC-affine filter for outliers (reuses
         ``cv2.estimateAffinePartial2D``).
      6. Sample DEM at each (X, Y) for Z.
      7. Return (N, 5) array.

    Reuses helpers from preprocess/experimental/match_ip.py.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.transform import from_bounds as tfm_from_bounds
    from pyproj import Transformer

    from preprocess.experimental.match_ip import (
        _clahe_u8,
        _dedup_spatial,
        apply_geometric_filters,
        normalize_preprocess_matcher,
        run_preprocess_matcher,
    )

    # 1. Read the raw sub-frame at a DOWNSAMPLED resolution matching
    # ``match_res_m``. The raw KH-9 PC sub-frame is ~37k x 25k, which is
    # far too large to feed directly to RoMa (~1000 tiles at 1024px,
    # ~8 minutes of inference). We resample on read via rasterio's
    # ``out_shape`` parameter so RoMa only sees a ~10k x 6k grid.
    #
    # The sub-frame's ground pixel size varies along the scan but is
    # roughly (Z_s0 / f) * pixel_pitch ≈ (170_000 / 1.524) * 7e-6 ≈
    # 0.78 m/px for KH-9 PC. We downsample by a factor match_res_m /
    # 0.78 so the resampled sub-frame is approximately at match_res_m
    # ground resolution, matching the reference.
    import cv2
    with rasterio.open(sub_frame_path) as src:
        sf_w_full = src.width
        sf_h_full = src.height
        sf_nd = src.nodata
        # Phase 7.1: native_gsd is the raw sub-frame's ground sample
        # distance at its delivery resolution. Caller passes the
        # physically-correct value derived from the camera model
        # (altitude × pixel_pitch / focal_length). Fall back to the
        # legacy 0.8 m/px KH-9 PC estimate only when not supplied
        # (old code paths; unit tests).
        native_gsd = (
            float(native_gsd_m) if native_gsd_m and native_gsd_m > 0.0
            else 0.8
        )
        ds_factor = max(1.0, match_res_m / native_gsd)
        sf_w_ds = max(256, int(round(sf_w_full / ds_factor)))
        sf_h_ds = max(256, int(round(sf_h_full / ds_factor)))
        sf_arr = src.read(
            1,
            out_shape=(sf_h_ds, sf_w_ds),
            resampling=rasterio.enums.Resampling.average,
        ).astype(np.float32)
    if sf_nd is not None:
        sf_arr = np.where(sf_arr == sf_nd, 0, sf_arr)
    sf_u8 = _clahe_u8(sf_arr)
    sf_h, sf_w = sf_u8.shape
    # Scale factors to go from downsampled pixel coords → full sub-frame
    # pixel coords:
    sf_scale_x = sf_w_full / sf_w
    sf_scale_y = sf_h_full / sf_h

    # 2. Compute the expected ground bbox from initial_corners_ll and
    # reproject the reference into that bbox in local_crs at match_res_m.
    corners = {str(k).upper(): v for k, v in initial_corners_ll.items()}
    lats = [corners[k][0] for k in ("NW", "NE", "SE", "SW")]
    lons = [corners[k][1] for k in ("NW", "NE", "SE", "SW")]

    tr_ll_to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    local_xy = [tr_ll_to_local.transform(lon, lat) for lat, lon in zip(lats, lons)]
    local_xs = [xy[0] for xy in local_xy]
    local_ys = [xy[1] for xy in local_xy]
    pad = float(search_pad_m)
    bl = min(local_xs) - pad
    br = max(local_xs) + pad
    bb = min(local_ys) - pad
    bt = max(local_ys) + pad

    ref_w = int(round((br - bl) / match_res_m))
    ref_h = int(round((bt - bb) / match_res_m))
    if ref_w < 100 or ref_h < 100:
        print(f"  [kh_panoramic.gcps] reference reproject area too small "
              f"({ref_w}x{ref_h}px)")
        return None
    ref_tfm = tfm_from_bounds(bl, bb, br, bt, ref_w, ref_h)
    ref_arr = np.zeros((ref_h, ref_w), dtype=np.float32)
    with rasterio.open(reference_ortho_path) as ref:
        ref_nd = ref.nodata if ref.nodata is not None else 0
        reproject(
            ref.read(1), ref_arr,
            src_transform=ref.transform, src_crs=ref.crs,
            dst_transform=ref_tfm, dst_crs=local_crs,
            resampling=Resampling.bilinear,
            src_nodata=ref_nd, dst_nodata=0,
        )
    ref_u8 = _clahe_u8(ref_arr)

    matcher = normalize_preprocess_matcher(matcher_name)
    pts_a, pts_b, conf = run_preprocess_matcher(
        sf_u8.astype(np.float32),
        ref_u8.astype(np.float32),
        matcher_name=matcher,
        matcher_runtime=matcher_runtime,
        max_matches=max_matches,
        max_tiles=max_tiles,
    )

    if pts_a is None or len(pts_a) < 10:
        print(f"  [kh_panoramic.gcps] too few {matcher.upper()} matches "
              f"({0 if pts_a is None else len(pts_a)})")
        return None

    # pts_a come in common-grid pixel coordinates. _run_roma_tiled resizes
    # both inputs to min(a_shape, b_shape) before tiling. Scale pts_a
    # from the common grid → downsampled sub-frame grid → full sub-frame
    # grid, and pts_b from common grid → reference reprojected grid.
    common_h = min(sf_h, ref_h)
    common_w = min(sf_w, ref_w)
    pts_a_full = pts_a.copy()
    pts_a_full[:, 0] *= (sf_w / common_w) * sf_scale_x
    pts_a_full[:, 1] *= (sf_h / common_h) * sf_scale_y
    pts_b_full = pts_b.copy()
    pts_b_full[:, 0] *= ref_w / common_w
    pts_b_full[:, 1] *= ref_h / common_h

    # 4. Phase 1 geometric verification. Panoramic raw vs reference ortho
    # is a cross-modal projection (scanning camera → ortho) so Sampson/F
    # doesn't hold; use MAGSAC++ affine + MTE residual consistency. Thresholds
    # are loose here because real panoramic distortion produces systematic
    # position-dependent residuals that MTE would otherwise flag as outliers.
    pts_a_full, pts_b_full, conf_filtered, _M = apply_geometric_filters(
        pts_a_full, pts_b_full, conf,
        affine_reproj_px=ransac_reproj_px,
        sampson_enabled=False,
        mte_enabled=bool(mte_enabled),
        mte_radius_px=float(mte_radius_px),
        # ≈ 0.75 × affine gate — matches near scan edges have residual structure
        # that a 3 px local-dev threshold drops along with real outliers.
        mte_max_dev_px=max(6.0, ransac_reproj_px * 0.75),
        min_inliers=10,
    )
    if pts_a_full is None:
        print(f"  [kh_panoramic.gcps] geometric filters rejected matches")
        return None

    pts_a_full, pts_b_full, conf_filtered = _dedup_spatial(
        pts_a_full, pts_b_full, conf_filtered, cell_px=30)

    # 5. Convert reference pixel → local_crs ground XY.
    ref_col = pts_b_full[:, 0]
    ref_row = pts_b_full[:, 1]
    X_local = bl + ref_col * match_res_m
    Y_local = bt - ref_row * match_res_m

    # 6. Sample DEM for Z.
    Z_local = _sample_dem_local_xy(X_local, Y_local, dem_path, local_crs)

    gcps = np.column_stack([
        pts_a_full[:, 0],   # col in raw sub-frame
        pts_a_full[:, 1],   # row in raw sub-frame
        X_local,
        Y_local,
        Z_local,
    ]).astype(np.float64)

    coverage_ok, coverage = _gcp_distribution_ok(gcps, sf_w_full, sf_h_full)
    print(
        f"  [kh_panoramic.gcps] {gcps.shape[0]} GCPs extracted; "
        f"cells={coverage['occupied_cells']} "
        f"cols={coverage['occupied_cols']} rows={coverage['occupied_rows']} "
        f"thirds=({coverage['left_count']},"
        f"{coverage['center_count']},{coverage['right_count']}) "
        f"span={coverage['col_span_frac']:.2f}"
    )
    if not coverage_ok:
        print("  [kh_panoramic.gcps] rejected: poor scan-width coverage")
        return None
    return gcps


# ---------------------------------------------------------------------------
# extract_model_guided_gcps — ortho/reference rematch for Stage C
# ---------------------------------------------------------------------------

def extract_model_guided_gcps(
    ortho_path: str,
    reference_ortho_path: str,
    params: "PanoramicParams",
    pixel_pitch: float,
    image_width_px: int,
    image_height_px: int,
    local_crs: str,
    matcher_name: str = "roma",
    matcher_runtime=None,
    match_res_m: float = 2.0,
    dem_path: Optional[str] = None,
    max_matches: int = 4000,
    max_tiles: int = 300,
    ransac_reproj_px: float = 6.0,
    max_world_shift_m: float = 500.0,
) -> Optional[np.ndarray]:
    """Extract second-stage GCPs from a coarse ortho and the reference.

    This is the model-guided rematching stage: after the initial 14p fit
    reduces panoramic distortion, we rematch the ortho against the
    reference, locally refine the reference-side coordinates, and convert
    the ortho-side feature locations back to raw-pixel GCPs via the
    current panoramic model.
    """
    import cv2
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds as tfm_from_bounds
    from pyproj import Transformer

    from preprocess.experimental.match_ip import (
        _clahe_u8,
        _dedup_spatial,
        apply_geometric_filters,
        normalize_preprocess_matcher,
        run_preprocess_matcher,
    )

    with rasterio.open(ortho_path) as ortho:
        ortho_crs = ortho.crs
        ortho_bounds = ortho.bounds
        ortho_nd = ortho.nodata if ortho.nodata is not None else 0

    bl, bb, br, bt = ortho_bounds
    out_w = int(round((br - bl) / match_res_m))
    out_h = int(round((bt - bb) / match_res_m))
    if out_w < 100 or out_h < 100:
        print(
            f"  [kh_panoramic.guided] ortho window too small "
            f"({out_w}x{out_h}px)"
        )
        return None

    out_tfm = tfm_from_bounds(bl, bb, br, bt, out_w, out_h)
    ortho_arr = np.zeros((out_h, out_w), dtype=np.float32)
    ref_arr = np.zeros((out_h, out_w), dtype=np.float32)
    with rasterio.open(ortho_path) as ortho:
        reproject(
            ortho.read(1),
            ortho_arr,
            src_transform=ortho.transform,
            src_crs=ortho.crs,
            dst_transform=out_tfm,
            dst_crs=ortho_crs,
            resampling=Resampling.bilinear,
            src_nodata=ortho_nd,
            dst_nodata=0,
        )
    with rasterio.open(reference_ortho_path) as ref:
        ref_nd = ref.nodata if ref.nodata is not None else 0
        reproject(
            ref.read(1),
            ref_arr,
            src_transform=ref.transform,
            src_crs=ref.crs,
            dst_transform=out_tfm,
            dst_crs=ortho_crs,
            resampling=Resampling.bilinear,
            src_nodata=ref_nd,
            dst_nodata=0,
        )

    ortho_u8 = _clahe_u8(ortho_arr)
    ref_u8 = _clahe_u8(ref_arr)

    matcher = normalize_preprocess_matcher(matcher_name)
    pts_a, pts_b, conf = run_preprocess_matcher(
        ortho_u8.astype(np.float32),
        ref_u8.astype(np.float32),
        matcher_name=matcher,
        matcher_runtime=matcher_runtime,
        max_matches=max_matches,
        max_tiles=max_tiles,
    )

    if pts_a is None or len(pts_a) < 20:
        print(
            f"  [kh_panoramic.guided] too few {matcher.upper()} matches "
            f"({0 if pts_a is None else len(pts_a)})"
        )
        return None

    # Phase 1 geometric verification: ortho-vs-ortho is not pinhole so
    # skip Sampson; MAGSAC++ affine + MTE local consistency catch
    # systematically biased match groups. Loosened MTE threshold because
    # cross-modal (ortho vs reference ortho) matches have systematic
    # local drift that a tight 2.5 px dev gate would incorrectly reject.
    pts_a, pts_b, conf, _M = apply_geometric_filters(
        pts_a, pts_b, conf,
        affine_reproj_px=ransac_reproj_px,
        sampson_enabled=False,
        mte_enabled=False,
        mte_radius_px=400.0,
        mte_max_dev_px=max(5.0, ransac_reproj_px * 0.75),
        min_inliers=20,
    )
    if pts_a is None:
        print(f"  [kh_panoramic.guided] geometric filters rejected matches")
        return None

    pts_a, pts_b, conf = _dedup_spatial(pts_a, pts_b, conf, cell_px=20)
    print(
        f"  [kh_panoramic.guided] {matcher.upper()}+filters kept {pts_a.shape[0]} matches "
        f"before local refinement"
    )

    # Local NCC refinement on the reference side.
    order = np.argsort(-conf)
    order = order[: min(len(order), 1200)]
    tpl_r = 14
    search_r = 48
    refined_a = []
    refined_b = []
    refined_conf = []
    for idx in order:
        ax, ay = pts_a[idx]
        bx, by = pts_b[idx]
        ax_i = int(round(ax))
        ay_i = int(round(ay))
        bx_i = int(round(bx))
        by_i = int(round(by))
        if (
            ax_i - tpl_r < 0 or ay_i - tpl_r < 0
            or ax_i + tpl_r >= out_w or ay_i + tpl_r >= out_h
        ):
            continue
        if (
            bx_i - search_r < 0 or by_i - search_r < 0
            or bx_i + search_r >= out_w or by_i + search_r >= out_h
        ):
            continue
        tpl = ortho_u8[ay_i - tpl_r:ay_i + tpl_r + 1, ax_i - tpl_r:ax_i + tpl_r + 1]
        search = ref_u8[by_i - search_r:by_i + search_r + 1, bx_i - search_r:bx_i + search_r + 1]
        if search.shape[0] < tpl.shape[0] or search.shape[1] < tpl.shape[1]:
            continue
        result = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.20:
            continue
        rx = (bx_i - search_r) + max_loc[0] + tpl_r
        ry = (by_i - search_r) + max_loc[1] + tpl_r
        refined_a.append((float(ax), float(ay)))
        refined_b.append((float(rx), float(ry)))
        refined_conf.append(float(max_val))

    if len(refined_a) >= 20:
        print(
            f"  [kh_panoramic.guided] local refinement kept {len(refined_a)} matches"
        )
        pts_a = np.asarray(refined_a, dtype=np.float64)
        pts_b = np.asarray(refined_b, dtype=np.float64)
        conf = np.asarray(refined_conf, dtype=np.float64)

        refined_pts_a, refined_pts_b, refined_conf, _M = apply_geometric_filters(
            pts_a, pts_b, conf,
            affine_reproj_px=ransac_reproj_px,
            sampson_enabled=False,
            mte_enabled=False,
            mte_radius_px=400.0,
            # Post-NCC-refinement matches are tighter than the pre-refine
            # pass, so MTE can use a narrower dev gate here (~0.5 × affine).
            mte_max_dev_px=max(3.0, ransac_reproj_px * 0.5),
            min_inliers=20,
        )
        if refined_pts_a is None:
            print(
                f"  [kh_panoramic.guided] refined geometric filters rejected matches; "
                f"falling back to unrefined {matcher.upper()} matches"
            )
        else:
            pts_a, pts_b, conf = refined_pts_a, refined_pts_b, refined_conf
            pts_a, pts_b, conf = _dedup_spatial(pts_a, pts_b, conf, cell_px=24)
    else:
        print(
            f"  [kh_panoramic.guided] too few locally refined matches "
            f"({len(refined_a)}); falling back to unrefined {matcher.upper()} matches"
        )

    ortho_world_x = bl + pts_a[:, 0] * match_res_m
    ortho_world_y = bt - pts_a[:, 1] * match_res_m
    ref_world_x = bl + pts_b[:, 0] * match_res_m
    ref_world_y = bt - pts_b[:, 1] * match_res_m

    if str(ortho_crs) != str(local_crs):
        tr_to_local = Transformer.from_crs(ortho_crs, local_crs, always_xy=True)
        curr_x_local, curr_y_local = tr_to_local.transform(ortho_world_x, ortho_world_y)
        tgt_x_local, tgt_y_local = tr_to_local.transform(ref_world_x, ref_world_y)
    else:
        curr_x_local = ortho_world_x
        curr_y_local = ortho_world_y
        tgt_x_local = ref_world_x
        tgt_y_local = ref_world_y

    world_shift = np.hypot(tgt_x_local - curr_x_local, tgt_y_local - curr_y_local)
    keep = world_shift <= float(max_world_shift_m)
    if keep.sum() < 20:
        print(
            f"  [kh_panoramic.guided] too few consistent model-guided matches "
            f"({int(keep.sum())})"
        )
        return None

    curr_x_local = curr_x_local[keep]
    curr_y_local = curr_y_local[keep]
    tgt_x_local = tgt_x_local[keep]
    tgt_y_local = tgt_y_local[keep]

    Z_curr = _sample_dem_local_xy(curr_x_local, curr_y_local, dem_path, local_crs)
    Z_tgt = _sample_dem_local_xy(tgt_x_local, tgt_y_local, dem_path, local_crs)
    cols_raw, rows_raw = world_to_raw_pixel(
        params=params,
        X_world=curr_x_local,
        Y_world=curr_y_local,
        Z_world=Z_curr,
        pixel_pitch=pixel_pitch,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
    )

    in_bounds = (
        np.isfinite(cols_raw) & np.isfinite(rows_raw)
        & (cols_raw >= 0) & (cols_raw < image_width_px)
        & (rows_raw >= 0) & (rows_raw < image_height_px)
    )
    if in_bounds.sum() < 20:
        print(
            f"  [kh_panoramic.guided] too few raw inversions in bounds "
            f"({int(in_bounds.sum())})"
        )
        return None

    gcps = np.column_stack([
        cols_raw[in_bounds],
        rows_raw[in_bounds],
        tgt_x_local[in_bounds],
        tgt_y_local[in_bounds],
        Z_tgt[in_bounds],
    ]).astype(np.float64)

    _, coverage = _gcp_distribution_ok(gcps, image_width_px, image_height_px)
    print(
        f"  [kh_panoramic.guided] {gcps.shape[0]} GCPs; "
        f"cells={coverage['occupied_cells']} "
        f"cols={coverage['occupied_cols']} rows={coverage['occupied_rows']} "
        f"thirds=({coverage['left_count']},"
        f"{coverage['center_count']},{coverage['right_count']}) "
        f"span={coverage['col_span_frac']:.2f}"
    )
    # Model-guided re-matches often refine a localized area of the current
    # ortho/reference overlap. That is acceptable here because the coarse GCP
    # set already established scan-width coverage; the caller merges the two
    # sets and re-checks coverage on the combined GCPs before Stage C.
    return gcps


# ---------------------------------------------------------------------------
# extract_raw_subframe_tie_points — direct raw-to-raw RoMa matching
# ---------------------------------------------------------------------------

def extract_raw_subframe_tie_points(
    sub_path_a: str,
    sub_path_b: str,
    pixel_pitch: float,
    focal_m: float,
    nominal_altitude_m: float = NOMINAL_Z_S0,
    matcher_name: str = "roma",
    matcher_runtime=None,
    match_res_m: float = 4.0,
    max_matches: int = 8000,
    max_tiles: int = 600,
    ransac_reproj_px: float = 80.0,
) -> Optional[np.ndarray]:
    """RoMa-match two adjacent raw sub-frames directly in their own pixel space.

    Unlike :func:`extract_ortho_tie_point_gcps`, which matches already-
    rendered orthos (and only captures features where the current fits
    agree), this function matches the raw sub-frames directly. The same
    physical ground feature appears at (generally different) raw pixels
    in each sub-frame — RoMa matches the feature regardless of the fit's
    current projection. This means we can detect "doubling" (same feature
    at different world positions in the two fits) and generate tie points
    that correct it.

    The raw sub-frames are downsampled by ``match_res_m / native_gsd`` so
    RoMa sees a ~9k × 6k grid instead of the full 37k × 25k. Matches are
    scaled back to full raw-pixel coordinates of each sub-frame.

    Returns
    -------
    np.ndarray (N, 4) [col_a_raw, row_a_raw, col_b_raw, row_b_raw]
        Per-tie-point raw pixel coordinates for each sub-frame. Returns
        None on failure.
    """
    import rasterio

    from preprocess.experimental.match_ip import (
        _clahe_u8,
        _dedup_spatial,
        apply_geometric_filters,
        normalize_preprocess_matcher,
        run_preprocess_matcher,
    )

    native_gsd = (nominal_altitude_m * pixel_pitch) / float(focal_m)
    ds_factor = max(1.0, match_res_m / native_gsd)

    def _read_ds(path: str):
        with rasterio.open(path) as src:
            full_w = src.width
            full_h = src.height
            ds_w = max(256, int(round(full_w / ds_factor)))
            ds_h = max(256, int(round(full_h / ds_factor)))
            arr = src.read(
                1,
                out_shape=(ds_h, ds_w),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)
            nd = src.nodata
        if nd is not None:
            arr = np.where(arr == nd, 0.0, arr)
        return arr, full_w, full_h, ds_w, ds_h

    arr_a, w_a_full, h_a_full, w_a_ds, h_a_ds = _read_ds(sub_path_a)
    arr_b, w_b_full, h_b_full, w_b_ds, h_b_ds = _read_ds(sub_path_b)
    a_u8 = _clahe_u8(arr_a)
    b_u8 = _clahe_u8(arr_b)

    matcher = normalize_preprocess_matcher(matcher_name)
    pts_a, pts_b, conf = run_preprocess_matcher(
        a_u8.astype(np.float32),
        b_u8.astype(np.float32),
        matcher_name=matcher,
        matcher_runtime=matcher_runtime,
        max_matches=max_matches,
        max_tiles=max_tiles,
    )

    if pts_a is None or len(pts_a) < 20:
        print(f"  [kh_panoramic.raw_ties] too few {matcher.upper()} matches "
              f"({0 if pts_a is None else len(pts_a)})")
        return None

    # _run_roma_tiled resizes both inputs to min(a_shape, b_shape) before
    # tiling. Scale from that common grid back to each sub-frame's full
    # raw pixel coordinates.
    common_h = min(a_u8.shape[0], b_u8.shape[0])
    common_w = min(a_u8.shape[1], b_u8.shape[1])

    pts_a_full = pts_a.copy()
    pts_a_full[:, 0] *= (w_a_ds / common_w) * (w_a_full / w_a_ds)
    pts_a_full[:, 1] *= (h_a_ds / common_h) * (h_a_full / h_a_ds)
    pts_b_full = pts_b.copy()
    pts_b_full[:, 0] *= (w_b_ds / common_w) * (w_b_full / w_b_ds)
    pts_b_full[:, 1] *= (h_b_ds / common_h) * (h_b_full / h_b_ds)

    # Phase 1 geometric verification: MAGSAC++ affine (loose threshold —
    # raw sub-frames share only a rough affine relationship) + Sampson
    # distance (both images are pinhole-like scans, epipolar geometry holds)
    # + MTE local consistency (catch systematically biased match groups).
    pts_a_full, pts_b_full, conf_f, _M = apply_geometric_filters(
        pts_a_full, pts_b_full, conf,
        affine_reproj_px=ransac_reproj_px,
        sampson_enabled=True,
        sampson_tau_px=max(15.0, ransac_reproj_px / 6.0),
        mte_enabled=False,
        mte_radius_px=800.0,
        # ≈ 0.15 × affine — raw pan-vs-pan matches have large systematic
        # shifts between sub-frames (~120 px reproj gate) and the local
        # consistency needs to tolerate most of that spread.
        mte_max_dev_px=max(15.0, ransac_reproj_px * 0.15),
        min_inliers=20,
    )
    if pts_a_full is None:
        print(f"  [kh_panoramic.raw_ties] geometric filters rejected matches")
        return None

    pts_a_full, pts_b_full, conf_f = _dedup_spatial(
        pts_a_full, pts_b_full, conf_f, cell_px=80)

    tie = np.column_stack([
        pts_a_full[:, 0], pts_a_full[:, 1],
        pts_b_full[:, 0], pts_b_full[:, 1],
    ]).astype(np.float64)
    print(f"  [kh_panoramic.raw_ties] {tie.shape[0]} raw-pixel tie points")
    return tie


def raw_tie_points_to_gcps(
    params_a: "PanoramicParams",
    tie: np.ndarray,
    pixel_pitch: float,
    image_width_a_px: int,
    image_height_a_px: int,
    image_width_b_px: int,
    image_height_b_px: int,
) -> Optional[np.ndarray]:
    """Convert raw-pixel tie points into GCPs for re-fitting sub-frame B.

    For each tie point, project the A-side raw pixel through ``params_a``
    (the already-fit sub-frame A parameters) to get a world (X, Y) target,
    then pair it with the B-side raw pixel. Returns an (N, 5) GCP array
    [col_b, row_b, X_target, Y_target, 0] ready for :func:`fit_panoramic`.
    """
    if tie is None or tie.shape[0] == 0:
        return None
    cols_a = tie[:, 0]
    rows_a = tie[:, 1]
    cols_b = tie[:, 2]
    rows_b = tie[:, 3]

    # Project seg_a raw pixels to world (ground truth from A's fit).
    X_tgt, Y_tgt = raw_to_world(
        params=params_a,
        cols=cols_a, rows=rows_a,
        pixel_pitch=pixel_pitch,
        image_width_px=image_width_a_px,
        image_height_px=image_height_a_px,
        z_world=0.0,
    )

    # Keep only tie points whose seg_b raw pixel is inside the sub-frame
    # and whose seg_a-projected ground point is finite.
    finite = np.isfinite(X_tgt) & np.isfinite(Y_tgt)
    in_b = (
        (cols_b >= 0) & (cols_b < image_width_b_px) &
        (rows_b >= 0) & (rows_b < image_height_b_px) &
        finite
    )
    if in_b.sum() < 10:
        print(f"  [kh_panoramic.raw_ties] only {in_b.sum()} valid tie points "
              f"after bounds filter")
        return None

    Z_tgt = np.zeros(int(in_b.sum()), dtype=np.float64)
    gcps = np.column_stack([
        cols_b[in_b], rows_b[in_b],
        X_tgt[in_b], Y_tgt[in_b], Z_tgt,
    ]).astype(np.float64)
    print(f"  [kh_panoramic.raw_ties] {gcps.shape[0]} tie-point GCPs "
          f"(from raw feature matches)")
    return gcps


# ---------------------------------------------------------------------------
# extract_ortho_tie_point_gcps — adjacent-ortho RoMa matching → GCPs
# ---------------------------------------------------------------------------

def extract_ortho_tie_point_gcps(
    prev_ortho_path: str,
    curr_ortho_path: str,
    curr_params: "PanoramicParams",
    curr_pixel_pitch: float,
    curr_image_width_px: int,
    curr_image_height_px: int,
    local_crs: str | None = None,
    dem_path: Optional[str] = None,
    matcher_name: str = "roma",
    matcher_runtime=None,
    max_matches: int = 5000,
    max_tiles: int = 300,
    ransac_reproj_px: float = 30.0,
) -> Optional[np.ndarray]:
    """RoMa-match two adjacent orthos → tie-point GCPs for re-fitting.

    Given two orthos produced by initial per-sub-frame fits, RoMa-match
    their geographic overlap to find features that are supposedly the
    same physical ground point but may currently project to different
    world coordinates. Converts each tie point into a GCP for re-fitting
    the CURRENT sub-frame: the GCP's raw (col, row) comes from inverting
    the current fit at the current ortho position of the feature, and
    the target (X, Y) is the PREVIOUS ortho's position of the feature.

    The effect: when fed back to :func:`fit_panoramic`, these GCPs pull
    the current sub-frame's projection toward the previous sub-frame's,
    eliminating inter-sub-frame doubling at the seam.

    Returns
    -------
    np.ndarray (N, 5) [col_curr_raw, row_curr_raw, X_target, Y_target, 0]
        Suitable for concatenation with the current sub-frame's reference
        GCPs before calling :func:`fit_panoramic`. Returns None if no
        tie points can be extracted.
    """
    import rasterio
    from rasterio.windows import from_bounds as win_from_bounds

    from preprocess.experimental.match_ip import (
        _clahe_u8,
        _dedup_spatial,
        apply_geometric_filters,
        normalize_preprocess_matcher,
        run_preprocess_matcher,
    )

    with rasterio.open(prev_ortho_path) as prev, rasterio.open(curr_ortho_path) as curr:
        ortho_crs = prev.crs
        ol = max(prev.bounds.left, curr.bounds.left)
        ob = max(prev.bounds.bottom, curr.bounds.bottom)
        or_ = min(prev.bounds.right, curr.bounds.right)
        ot = min(prev.bounds.top, curr.bounds.top)
        if ol >= or_ or ob >= ot:
            print(f"  [kh_panoramic.ties] no overlap between orthos")
            return None
        w_prev = win_from_bounds(ol, ob, or_, ot, prev.transform)
        w_curr = win_from_bounds(ol, ob, or_, ot, curr.transform)
        prev_arr = prev.read(1, window=w_prev).astype(np.float32)
        curr_arr = curr.read(1, window=w_curr).astype(np.float32)
        prev_nd = prev.nodata if prev.nodata is not None else 0
        curr_nd = curr.nodata if curr.nodata is not None else 0
        prev_px = abs(prev.transform[0])
        # The overlap bbox is defined at (ol, ot) in local CRS = top-left
        # world coordinate of the overlap crop. Each pixel steps by prev_px.
        # pts from RoMa come in pixel coordinates of the overlap crop.

    h = min(prev_arr.shape[0], curr_arr.shape[0])
    w = min(prev_arr.shape[1], curr_arr.shape[1])
    prev_arr = prev_arr[:h, :w]
    curr_arr = curr_arr[:h, :w]

    prev_u8 = _clahe_u8(prev_arr)
    curr_u8 = _clahe_u8(curr_arr)

    matcher = normalize_preprocess_matcher(matcher_name)
    pts_a, pts_b, conf = run_preprocess_matcher(
        prev_u8.astype(np.float32),
        curr_u8.astype(np.float32),
        matcher_name=matcher,
        matcher_runtime=matcher_runtime,
        max_matches=max_matches,
        max_tiles=max_tiles,
    )

    if pts_a is None or len(pts_a) < 20:
        print(f"  [kh_panoramic.ties] too few {matcher.upper()} matches "
              f"({0 if pts_a is None else len(pts_a)})")
        return None

    # Phase 1 geometric verification. Ortho-vs-ortho pair: already
    # rectified to the same CRS so the expected relationship is near-
    # identity with a global drift (that's what we want to capture).
    # Sampson/F-matrix doesn't apply (orthos are not pinhole cameras),
    # so we run MAGSAC++ affine + MTE only.
    pts_a, pts_b, conf_f, _M = apply_geometric_filters(
        pts_a, pts_b, conf,
        affine_reproj_px=ransac_reproj_px,
        sampson_enabled=False,
        mte_enabled=False,
        mte_radius_px=300.0,
        # ≈ 0.2 × affine. Previous 3 px was too tight when affine=30 px
        # because most real inter-ortho drift sits in the 4-15 px range.
        mte_max_dev_px=max(4.0, ransac_reproj_px * 0.2),
        min_inliers=20,
    )
    if pts_a is None:
        print(f"  [kh_panoramic.ties] geometric filters rejected matches")
        return None

    pts_a, pts_b, conf_f = _dedup_spatial(pts_a, pts_b, conf_f, cell_px=40)

    # Convert ortho-crop pixel coordinates to world coordinates.
    # The crops are in the overlap bbox (ol, ob, or_, ot). A crop pixel
    # (col, row) corresponds to world X = ol + col * prev_px, Y = ot - row * prev_px.
    prev_world_x = ol + pts_a[:, 0] * prev_px
    prev_world_y = ot - pts_a[:, 1] * prev_px
    curr_world_x = ol + pts_b[:, 0] * prev_px
    curr_world_y = ot - pts_b[:, 1] * prev_px

    if local_crs and str(ortho_crs) != str(local_crs):
        from pyproj import Transformer
        tr_to_local = Transformer.from_crs(ortho_crs, local_crs, always_xy=True)
        prev_world_x, prev_world_y = tr_to_local.transform(prev_world_x, prev_world_y)
        curr_world_x, curr_world_y = tr_to_local.transform(curr_world_x, curr_world_y)

    # For each tie point, find the CURRENT sub-frame's raw pixel that
    # maps to (curr_world_x, curr_world_y) under the CURRENT fit. This
    # is the observed raw pixel for the tie-point GCP.
    crs_for_dem = local_crs or str(ortho_crs)
    Z_curr = _sample_dem_local_xy(curr_world_x, curr_world_y, dem_path, crs_for_dem)
    cols_raw, rows_raw = world_to_raw_pixel(
        params=curr_params,
        X_world=curr_world_x,
        Y_world=curr_world_y,
        Z_world=Z_curr,
        pixel_pitch=curr_pixel_pitch,
        image_width_px=curr_image_width_px,
        image_height_px=curr_image_height_px,
    )

    # Filter: keep only tie points whose inverted raw pixel is inside
    # the current sub-frame's valid pixel range.
    in_bounds = (
        (cols_raw >= 0) & (cols_raw < curr_image_width_px) &
        (rows_raw >= 0) & (rows_raw < curr_image_height_px)
    )
    if in_bounds.sum() < 10:
        print(f"  [kh_panoramic.ties] {in_bounds.sum()} tie points inside "
              f"sub-frame bounds (need ≥10)")
        return None

    cols_raw = cols_raw[in_bounds]
    rows_raw = rows_raw[in_bounds]
    target_x = prev_world_x[in_bounds]  # PREV ortho position = target
    target_y = prev_world_y[in_bounds]
    Z_out = _sample_dem_local_xy(target_x, target_y, dem_path, crs_for_dem)

    tie_gcps = np.column_stack([cols_raw, rows_raw, target_x, target_y, Z_out]).astype(np.float64)
    mean_shift = np.hypot(
        (prev_world_x[in_bounds] - curr_world_x[in_bounds]).mean(),
        (prev_world_y[in_bounds] - curr_world_y[in_bounds]).mean()
    )
    max_shift = np.hypot(
        np.abs(prev_world_x[in_bounds] - curr_world_x[in_bounds]).max(),
        np.abs(prev_world_y[in_bounds] - curr_world_y[in_bounds]).max()
    )
    print(f"  [kh_panoramic.ties] {tie_gcps.shape[0]} tie-point GCPs "
          f"(mean shift {mean_shift:.1f}m, max {max_shift:.1f}m)")
    return tie_gcps
