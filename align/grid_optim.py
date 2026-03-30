"""Grid-based warp optimizer: affine-baseline + residual parameterisation.

Key design decisions
--------------------
* GridWarper stores a *frozen* affine baseline and a *learnable* residual.
  The clamp and all regularisation apply only to the residual, so a good
  affine initialisation can never be destroyed by the clamp.
* All loss terms are scaled to metres² so weights have stable, interpretable
  meaning across different image sizes.
* A fold-detection pass after optimisation warns/rejects flipped regions.
* Hierarchical coarse-to-fine pyramid optimization: 8→24→64 grid sizes.
"""

import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import RBFInterpolator

from . import constants as _C

# ---------------------------------------------------------------------------
# GridWarper: affine baseline + learnable residual
# ---------------------------------------------------------------------------

class GridWarper(nn.Module):
    """Displacement grid parameterised as frozen_affine_baseline + learnable_residual.

    The baseline is set from the GCP-fitted affine transform and is never
    updated by the optimiser.  The residual (initialised to zero) is what
    the optimiser learns.  Regularisation and the clamp act only on the
    residual, so a good affine init is preserved even when max_residual_norm
    is small.

    Displacement convention:
        grid[x, y]  maps a normalised target coordinate to a normalised
        source coordinate.  Both in [-1, 1] (PyTorch grid_sample convention,
        x = horizontal = column direction).
    """

    def __init__(self, grid_size: Tuple[int, int] = (20, 20)):
        super().__init__()
        H, W = grid_size
        # Learnable residual only (starts at zero)
        self.residual = nn.Parameter(torch.zeros(1, 2, H, W))
        # Frozen affine baseline (set after construction via set_affine_baseline)
        self.register_buffer('affine_baseline', torch.zeros(1, 2, H, W))

    # kept for backward compat with caller that reads .displacements
    @property
    def displacements(self) -> torch.Tensor:
        return self.affine_baseline + self.residual

    def set_affine_baseline(self, baseline: torch.Tensor) -> None:
        """Set the frozen affine baseline.  baseline: (1, 2, H, W)."""
        with torch.no_grad():
            self.affine_baseline.copy_(baseline)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_disp(self, target_size: Tuple[int, int]) -> torch.Tensor:
        """Upsample full displacement (baseline + residual) to target_size.
        Returns (1, 2, H, W)."""
        H, W = target_size
        full = self.affine_baseline + self.residual
        return F.interpolate(full, size=(H, W), mode='bicubic', align_corners=True)

    def get_dense_grid(self, target_size: Tuple[int, int]) -> torch.Tensor:
        """Dense deformation grid (1, H, W, 2) in [-1, 1] for grid_sample."""
        H, W = target_size
        device = self.residual.device
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        identity = torch.stack([x, y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        dense_disp = self._dense_disp(target_size).permute(0, 2, 3, 1)  # (1, H, W, 2)
        return identity + dense_disp

    def warp_points(self, pts: torch.Tensor) -> torch.Tensor:
        """Warp (N, 2) normalised target points → source domain.
        pts format: (x, y) in [-1, 1]."""
        if pts.shape[0] == 0:
            return pts
        grid = pts.view(1, 1, -1, 2)
        full = self.affine_baseline + self.residual
        sampled = F.grid_sample(
            full, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )  # (1, 2, 1, N)
        sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1)  # (N, 2)
        return pts + sampled

    def warp_image_or_features(
        self, tensor: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Warp a source (B, C, H_s, W_s) tensor to target_size."""
        dense_grid = self.get_dense_grid(target_size)
        return F.grid_sample(
            tensor, dense_grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )


# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------

def asap_regularization(
    residual: torch.Tensor,
    grid_size: Tuple[int, int],
    target_size: Tuple[int, int],
    output_res_m: float,
    gcp_density_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ARAP (Cauchy-Riemann) + Laplacian smoothing on the *residual* only.

    gcp_density_map: optional (grid_H, grid_W) tensor in [0, 1] where higher
        values indicate more GCP coverage → ARAP weight is reduced there,
        letting the data term dominate.  In sparse-GCP regions the full ARAP
        weight is preserved to enforce smoothness.

    Returns (arap_loss_m2, laplacian_loss_m2) — both in metres².
    """
    grid_H, grid_W = grid_size
    H, W = target_size

    # Convert residual from normalised [-1,1] to metres
    u = residual[:, 0:1, :, :] * (W / 2.0) * output_res_m
    v = residual[:, 1:2, :, :] * (H / 2.0) * output_res_m

    # Physical spacing between grid nodes in metres
    dx_m = (W / max(1, grid_W - 1)) * output_res_m
    dy_m = (H / max(1, grid_H - 1)) * output_res_m

    u_x = (u[:, :, :, 1:] - u[:, :, :, :-1]) / dx_m
    u_y = (u[:, :, 1:, :] - u[:, :, :-1, :]) / dy_m
    v_x = (v[:, :, :, 1:] - v[:, :, :, :-1]) / dx_m
    v_y = (v[:, :, 1:, :] - v[:, :, :-1, :]) / dy_m

    u_x_c = u_x[:, :, :-1, :]
    v_y_c = v_y[:, :, :, :-1]
    u_y_c = u_y[:, :, :, :-1]
    v_x_c = v_x[:, :, :-1, :]

    # Cauchy-Riemann (dimensionless derivatives squared → already m²/m² = 1)
    # multiply by dx_m² to put back in m² scale comparable to data loss
    scale = dx_m * dy_m

    cr1 = (u_x_c - v_y_c).pow(2)
    cr2 = (u_y_c + v_x_c).pow(2)

    if gcp_density_map is not None:
        # Per-cell weight: reduce ARAP where GCPs are dense
        # cr1/cr2 shape is (grid_H-1, grid_W-1), crop density map to match
        dm = gcp_density_map[:grid_H - 1, :grid_W - 1].to(residual.device)
        cell_weight = (1.0 - dm * 0.3).unsqueeze(0).unsqueeze(0)  # (1, 1, H-2, W-1)
        arap_loss = ((cr1 + cr2) * cell_weight).mean() * scale
    else:
        arap_loss = (cr1.mean() + cr2.mean()) * scale

    u_xx = (u_x[:, :, :, 1:] - u_x[:, :, :, :-1]) / dx_m
    u_yy = (u_y[:, :, 1:, :] - u_y[:, :, :-1, :]) / dy_m
    v_xx = (v_x[:, :, :, 1:] - v_x[:, :, :, :-1]) / dx_m
    v_yy = (v_y[:, :, 1:, :] - v_y[:, :, :-1, :]) / dy_m

    laplacian_loss = (
        u_xx.pow(2).mean() + u_yy.pow(2).mean() +
        v_xx.pow(2).mean() + v_yy.pow(2).mean()
    ) * scale

    return arap_loss, laplacian_loss


# ---------------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------------

def memory_efficient_chamfer(
    pts1: torch.Tensor, pts2: torch.Tensor,
    scale_x_m: float, scale_y_m: float,
    chunk_size: int = 1000,
    max_dist_m: float = 500.0,
) -> torch.Tensor:
    """Bidirectional Chamfer distance in metres (mean of min-distances).

    pts are in normalised [-1, 1] space.  scale_x_m / scale_y_m convert to metres.
    Returns mean one-way distance in metres (symmetric average).

    Distances beyond max_dist_m use a soft clamp (Huber-style): linear with
    slope 0.1, so gradients shrink but never fully zero out.  This prevents
    the chamfer loss from going dead when coastlines are far apart after
    coarse alignment.
    """
    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return torch.tensor(0.0, device=pts1.device)

    def directed(p1, p2):
        mins = []
        for i in range(0, p1.shape[0], chunk_size):
            chunk = p1[i:i + chunk_size]
            dx = (chunk[:, 0:1] - p2[:, 0].unsqueeze(0)) * scale_x_m
            dy = (chunk[:, 1:2] - p2[:, 1].unsqueeze(0)) * scale_y_m
            d = (dx ** 2 + dy ** 2).sqrt()           # metres, shape (C, M)
            mins.append(d.min(dim=1).values)          # (C,)
        all_mins = torch.cat(mins)
        # Robust soft clamp (Huber-style): beyond max_dist_m, gradient = 0.1.
        # Points with large chamfer distance are likely reclamation (real
        # coastline change), not misalignment — capping their gradient
        # prevents reclamation zones from distorting the overall warp.
        all_mins = torch.where(all_mins <= max_dist_m, all_mins,
                               max_dist_m + (all_mins - max_dist_m) * 0.3)
        return all_mins.mean()

    return 0.5 * (directed(pts1, pts2) + directed(pts2, pts1))


# ---------------------------------------------------------------------------
# Fold / invertibility check
# ---------------------------------------------------------------------------

def check_folds(warper: 'GridWarper', target_size: Tuple[int, int]) -> float:
    """Return the fraction of the warp field that has folded (det J <= 0).

    Samples the dense grid at a coarse resolution for speed.
    """
    H, W = target_size
    sample_H = min(H, 512)
    sample_W = min(W, 512)

    with torch.no_grad():
        grid = warper.get_dense_grid((sample_H, sample_W))  # (1, H, W, 2)
        g = grid.squeeze(0)  # (H, W, 2)

        # Central differences for Jacobian
        dx = g[:, 2:, :] - g[:, :-2, :]   # (H, W-2, 2)  / 2
        dy = g[2:, :, :] - g[:-2, :, :]   # (H-2, W, 2)  / 2

        # Crop to common interior
        dx = dx[1:-1, :, :]  # (H-2, W-2, 2)
        dy = dy[:, 1:-1, :]  # (H-2, W-2, 2)

        # det J = dx/du * dy/dv - dx/dv * dy/du
        det = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        fold_frac = float((det <= 0).float().mean().item())

    return fold_frac


# ---------------------------------------------------------------------------
# Hierarchical coarse-to-fine grid optimisation
# ---------------------------------------------------------------------------

def _fit_affine(tgt_np, src_np):
    """Fit a 6-param affine from corresponding points. Returns params or None."""
    n = len(tgt_np)
    if n < 3:
        return None
    A = np.zeros((2 * n, 6), dtype=np.float32)
    b = np.zeros(2 * n, dtype=np.float32)
    for i in range(n):
        A[2*i]   = [tgt_np[i, 0], tgt_np[i, 1], 1, 0, 0, 0]
        A[2*i+1] = [0, 0, 0, tgt_np[i, 0], tgt_np[i, 1], 1]
        b[2*i]   = src_np[i, 0]
        b[2*i+1] = src_np[i, 1]
    params, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    return params


def _fit_rbf_residual(
    real_tgt: np.ndarray,
    real_src: np.ndarray,
    affine_params: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    scale_m: float,
) -> Optional[dict]:
    """Fit RBF interpolation of affine residuals on real GCPs.

    Returns dict with 'src_x', 'src_y' grids and diagnostics, or None on failure.
    Multiquadric kernel with linear polynomial augmentation ensures smooth
    extrapolation (decays to affine far from GCPs).
    """
    n = len(real_tgt)
    if n < 6:
        return None

    # Affine predictions on real GCPs
    aff_x, aff_y = _apply_affine(affine_params, real_tgt[:, 0], real_tgt[:, 1])
    res_x = real_src[:, 0] - aff_x
    res_y = real_src[:, 1] - aff_y

    # Adaptive epsilon: mean nearest-neighbor distance among GCPs
    # Broader kernels capture systematic distortion, not per-GCP noise
    from scipy.spatial import cKDTree
    tree = cKDTree(real_tgt)
    nn_dists, _ = tree.query(real_tgt, k=2)
    mean_nn_dist = float(nn_dists[:, 1].mean())
    epsilon = mean_nn_dist * 1.0

    # Smoothing: 50% of median squared residual — high because GCP residuals
    # are dominated by match noise, not systematic camera distortion
    med_res_sq = float(np.median(res_x**2 + res_y**2))
    smoothing = med_res_sq * 0.5

    try:
        rbf_dx = RBFInterpolator(
            real_tgt, res_x, kernel='multiquadric', epsilon=epsilon,
            smoothing=smoothing, degree=1,
        )
        rbf_dy = RBFInterpolator(
            real_tgt, res_y, kernel='multiquadric', epsilon=epsilon,
            smoothing=smoothing, degree=1,
        )
    except Exception as e:
        print(f"  RBF fit failed: {e}", flush=True)
        return None

    # Evaluate on grid
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    dx_grid = rbf_dx(grid_pts).reshape(xx.shape)
    dy_grid = rbf_dy(grid_pts).reshape(yy.shape)

    # Safety clamp: max 500m deviation from affine
    max_dev_norm = 500.0 / scale_m
    dev = np.sqrt(dx_grid**2 + dy_grid**2)
    max_dev = float(dev.max())
    if max_dev > max_dev_norm:
        clamp_factor = np.minimum(1.0, max_dev_norm / np.maximum(dev, 1e-12))
        dx_grid *= clamp_factor
        dy_grid *= clamp_factor
        print(f"  RBF clamped: max_dev {max_dev*scale_m:.1f}m > 500m", flush=True)

    # Compose: affine + RBF residual
    aff_x_grid, aff_y_grid = _apply_affine(affine_params, xx, yy)
    src_x = aff_x_grid + dx_grid
    src_y = aff_y_grid + dy_grid

    # RBF RMS on real GCPs (evaluate RBF at GCP locations for diagnostics)
    rbf_pred_x = aff_x + rbf_dx(real_tgt)
    rbf_pred_y = aff_y + rbf_dy(real_tgt)
    rbf_rms = float(np.sqrt(np.mean((rbf_pred_x - real_src[:, 0])**2 + (rbf_pred_y - real_src[:, 1])**2)))

    return {
        'src_x': src_x,
        'src_y': src_y,
        'rbf_rms': rbf_rms,
        'max_dev_m': float(max_dev * scale_m),
        'epsilon': epsilon,
        'smoothing': smoothing,
        'mean_nn_dist': mean_nn_dist,
    }


def _apply_affine(params, xx, yy):
    """Apply affine params to grid, return (src_x, src_y)."""
    src_x = params[0]*xx + params[1]*yy + params[2]
    src_y = params[3]*xx + params[4]*yy + params[5]
    return src_x, src_y


def _compute_affine_baseline(
    tgt_pts_n: torch.Tensor,
    src_pts_n: torch.Tensor,
    grid_size: Tuple[int, int],
    device: torch.device,
    W_t: int, H_t: int,
    output_res_m: float,
    n_real_gcps: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Fit affine from GCPs and return baseline displacement (1, 2, H, W).

    Tries RBF interpolation of affine residuals on real GCPs to capture
    spatially-varying camera/film distortion. Falls back to global affine
    if RBF doesn't improve enough.
    """
    if tgt_pts_n.shape[0] < 3:
        return None
    tgt_np = tgt_pts_n.cpu().numpy()
    src_np = src_pts_n.cpu().numpy()

    y_g = np.linspace(-1, 1, grid_size[0], dtype=np.float32)
    x_g = np.linspace(-1, 1, grid_size[1], dtype=np.float32)
    yy, xx = np.meshgrid(y_g, x_g, indexing='ij')

    # Global affine (always computed as fallback)
    global_params = _fit_affine(tgt_np, src_np)
    if global_params is None:
        return None
    print(f"  Affine init params={global_params.round(4)}", flush=True)

    scale = (W_t / 2.0) * output_res_m  # normalized → metres

    # Compute global affine residual
    aff_src_x, aff_src_y = _apply_affine(global_params, tgt_np[:, 0], tgt_np[:, 1])
    affine_rms = np.sqrt(np.mean((aff_src_x - src_np[:, 0])**2 + (aff_src_y - src_np[:, 1])**2))
    print(f"  Affine RMS: {affine_rms*scale:.1f}m", flush=True)

    # Try RBF interpolation of affine residuals using ONLY real GCPs.
    # Boundary GCPs are placed at affine-predicted positions (zero residual),
    # so they carry no distortion signal. RBF naturally decays to affine
    # far from GCPs (no extrapolation blowup like polynomials).
    n_real = n_real_gcps if n_real_gcps is not None else len(tgt_np)
    real_tgt = tgt_np[:n_real]
    real_src = src_np[:n_real]

    if n_real >= 6:
        rbf_result = _fit_rbf_residual(real_tgt, real_src, global_params, xx, yy, scale)
        if rbf_result is not None:
            # Compare RBF vs affine RMS on real GCPs
            aff_real_x, aff_real_y = _apply_affine(global_params, real_tgt[:, 0], real_tgt[:, 1])
            affine_rms_real = float(np.sqrt(np.mean((aff_real_x - real_src[:, 0])**2 + (aff_real_y - real_src[:, 1])**2)))
            rbf_rms = rbf_result['rbf_rms']
            print(f"  RBF baseline: RMS {rbf_rms*scale:.1f}m vs affine {affine_rms_real*scale:.1f}m "
                  f"({n_real} GCPs, eps={rbf_result['epsilon']:.4f}, smooth={rbf_result['smoothing']:.6f})", flush=True)
            print(f"  RBF max_dev={rbf_result['max_dev_m']:.1f}m, mean_nn_dist={rbf_result['mean_nn_dist']:.4f}", flush=True)

            # Accept RBF if it improves real-GCP RMS by >5%
            if rbf_rms < affine_rms_real * 0.95:
                improvement_m = (affine_rms_real - rbf_rms) * scale
                print(f"  Using RBF baseline (improvement: {improvement_m:.1f}m, "
                      f"{(1 - rbf_rms/affine_rms_real)*100:.1f}%)", flush=True)
                baseline = torch.from_numpy(
                    np.stack([rbf_result['src_x'] - xx, rbf_result['src_y'] - yy], axis=0)
                ).unsqueeze(0).to(device)
                return baseline
            else:
                print(f"  RBF not used: improvement {(1 - rbf_rms/affine_rms_real)*100:.1f}% < 5%", flush=True)

    # Fall back to global affine
    src_x, src_y = _apply_affine(global_params, xx, yy)
    baseline = torch.from_numpy(
        np.stack([src_x - xx, src_y - yy], axis=0)
    ).unsqueeze(0).to(device)
    return baseline


def _compute_gcp_weights(
    tgt_pts_n: torch.Tensor,
    source_pts: np.ndarray,
    src_shape: Tuple[int, int],
    n_real_gcps: Optional[int] = None,
    match_weights: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Compute per-GCP confidence weights.

    Real GCPs get weight 1.0 (or precision-based weight from RoMa if available).
    Virtual boundary GCPs (indices >= n_real_gcps) get distance-dependent weight:
    0 near real GCPs (where they'd fight real data), ramping up to 0.15 in
    truly data-sparse regions.
    """
    n_total = tgt_pts_n.shape[0]
    weights = torch.ones(n_total, device=tgt_pts_n.device)

    # Apply per-match precision weights from RoMa to real GCPs
    if match_weights is not None and n_real_gcps is not None:
        n_mw = min(len(match_weights), n_real_gcps)
        weights[:n_mw] = torch.from_numpy(
            match_weights[:n_mw].astype(np.float32)
        ).to(tgt_pts_n.device)

    if n_real_gcps is not None and n_real_gcps < n_total:
        real_pts = tgt_pts_n[:n_real_gcps].cpu().numpy()
        virtual_pts = tgt_pts_n[n_real_gcps:].cpu().numpy()
        # Distance from each virtual GCP to nearest real GCP (normalized coords)
        from scipy.spatial import cKDTree
        tree = cKDTree(real_pts)
        dists, _ = tree.query(virtual_pts, k=1)
        # Ramp: 0.05 when near real GCP (dist < 0.1), up to 0.40 when far
        # (dist > 0.3).  Higher ceiling prevents runaway deformation at image
        # edges where real GCPs are absent — a known boundary extrapolation
        # problem (cf. RBF registration literature).
        ramp = np.clip((dists - 0.1) / 0.2, 0.0, 1.0).astype(np.float32)
        weights[n_real_gcps:] = torch.from_numpy(ramp * 0.35 + 0.05).to(tgt_pts_n.device)
    return weights


def _save_iteration_diagnostic(
    warper: 'GridWarper',
    source_img: np.ndarray,
    target_img: np.ndarray,
    tgt_shape: Tuple[int, int],
    level_name: str,
    iteration: int,
    losses: dict,
    diag_dir: str,
):
    """Save a diagnostic image showing the current warp state.

    Produces a side-by-side: warped source vs target, with GCP residuals
    and loss values overlaid.
    """
    import cv2
    H_t, W_t = tgt_shape

    # Render at reduced resolution for speed while preserving aspect ratio.
    diag_max = 512
    diag_scale = min(1.0, diag_max / max(H_t, W_t))
    diag_h = max(1, int(round(H_t * diag_scale)))
    diag_w = max(1, int(round(W_t * diag_scale)))

    with torch.no_grad():
        dense_grid = warper.get_dense_grid((diag_h, diag_w))  # (1, H, W, 2)

    # Warp source image
    src_t = torch.from_numpy(source_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    warped = F.grid_sample(src_t, dense_grid, mode='bilinear',
                           padding_mode='zeros', align_corners=True)
    warped_np = (warped.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    # Resize target to same diagnostic size
    tgt_resized = cv2.resize(target_img, (diag_w, diag_h))

    # Side-by-side canvas
    canvas = np.zeros((diag_h, diag_w * 2, 3), dtype=np.uint8)
    canvas[:, :diag_w] = warped_np
    canvas[:, diag_w:] = tgt_resized

    # Separator line
    cv2.line(canvas, (diag_w, 0), (diag_w, diag_h - 1), (80, 80, 80), 1)

    # Overlay loss text
    y0 = 16
    for key, val in losses.items():
        text = f"{key}={val:.2f}"
        cv2.putText(canvas, text, (4, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        y0 += 14

    # Labels
    cv2.putText(canvas, "warped src", (4, diag_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "target", (diag_w + 4, diag_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    grid_dir = os.path.join(diag_dir, "grid_iterations")
    os.makedirs(grid_dir, exist_ok=True)
    level_tag = level_name.replace(" ", "_").replace("×", "x")
    out_path = os.path.join(grid_dir, f"grid_{level_tag}_iter{iteration:04d}.jpg")
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _run_single_level(
    warper: GridWarper,
    grid_size: Tuple[int, int],
    tgt_pts_n: torch.Tensor,
    src_pts_n: torch.Tensor,
    src_coast_n: torch.Tensor,
    tgt_coast_n: torch.Tensor,
    src_shape: Tuple[int, int],
    tgt_shape: Tuple[int, int],
    output_res_m: float,
    iters: int,
    lr: float,
    w_data: float,
    w_chamfer: float,
    w_arap: float,
    w_laplacian: float,
    w_disp: float,
    max_residual_norm: float,
    level_name: str = "",
    gcp_weights: Optional[torch.Tensor] = None,
    huber_delta: float = 10.0,
    gcp_density_map: Optional[torch.Tensor] = None,
    source_img: Optional[np.ndarray] = None,
    target_img: Optional[np.ndarray] = None,
    diagnostics_dir: Optional[str] = None,
) -> GridWarper:
    """Run a single level of grid optimisation (inner loop extracted from optimize_grid)."""
    device = warper.residual.device
    H_s, W_s = src_shape
    H_t, W_t = tgt_shape

    scale_x_m = (W_s / 2.0) * output_res_m
    scale_y_m = (H_s / 2.0) * output_res_m
    coast_scale_x_m = scale_x_m
    coast_scale_y_m = scale_y_m

    optimizer = torch.optim.Adam(warper.parameters(), lr=lr)
    # Cosine annealing with warm restarts every 100 iters to escape plateaus
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=1, eta_min=lr * 0.1
    )

    WARMUP_ITERS = min(30, iters // 4)

    # Early stopping: track loss history
    loss_history = []
    # Early stopping constants imported from align.constants
    final_iter = iters

    for i in range(iters):
        optimizer.zero_grad()
        warmup = min(1.0, i / max(1, WARMUP_ITERS))

        # Keep full regularization throughout — decaying ARAP/Laplacian after
        # warmup was allowing local deformations that bend piers and roads.
        eff_w_arap = w_arap
        eff_w_laplacian = w_laplacian

        # 1. Data loss (GCPs) with per-GCP confidence weights
        if tgt_pts_n.shape[0] > 0:
            warped = warper.warp_points(tgt_pts_n)
            diff_m = (warped - src_pts_n) * torch.tensor(
                [scale_x_m, scale_y_m], device=device
            )
            # Per-element Huber
            elem_loss = F.huber_loss(diff_m, torch.zeros_like(diff_m),
                                     delta=huber_delta, reduction='none')
            # Apply per-GCP weights (broadcast over x,y dims)
            if gcp_weights is not None:
                elem_loss = elem_loss * gcp_weights.unsqueeze(1)
            data_loss = elem_loss.mean()
        else:
            data_loss = torch.tensor(0.0, device=device)

        # 2. Chamfer loss
        if src_coast_n.shape[0] > 0 and tgt_coast_n.shape[0] > 0:
            warped_coast = warper.warp_points(tgt_coast_n)
            chamfer_loss = memory_efficient_chamfer(
                warped_coast, src_coast_n,
                scale_x_m=coast_scale_x_m, scale_y_m=coast_scale_y_m,
            )
        else:
            chamfer_loss = torch.tensor(0.0, device=device)

        # 3. ARAP + Laplacian on residual
        arap_loss, laplacian_loss = asap_regularization(
            warper.residual, grid_size=grid_size,
            target_size=(H_t, W_t), output_res_m=output_res_m,
            gcp_density_map=gcp_density_map,
        )

        # 4. Residual magnitude penalty
        res_m = ((warper.residual[:, 0].pow(2).mean() * scale_x_m**2 +
                  warper.residual[:, 1].pow(2).mean() * scale_y_m**2 + 1e-8).sqrt())

        total = (w_data          * data_loss                       +
                 w_chamfer       * chamfer_loss     * warmup       +
                 eff_w_arap      * arap_loss                       +
                 eff_w_laplacian * laplacian_loss                   +
                 w_disp          * res_m)

        total.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            warper.residual.clamp_(-max_residual_norm, max_residual_norm)

        # Track loss for early stopping (exclude chamfer — near-constant
        # across iters, inflates denominator and triggers premature early stop)
        convergence_loss = (w_data * data_loss +
                            eff_w_arap * arap_loss + eff_w_laplacian * laplacian_loss +
                            w_disp * res_m).item()
        loss_val = convergence_loss
        loss_history.append(loss_val)

        if i == 0:
            res_abs_max = warper.residual.abs().max().item()
            print(
                f"  [{level_name}] Iter 1: data={w_data*data_loss.item():.2f} "
                f"cham={w_chamfer*chamfer_loss.item():.2f} "
                f"arap={eff_w_arap*arap_loss.item():.4f} "
                f"res|max|={res_abs_max:.6f}",
                flush=True
            )

        if (i + 1) % 50 == 0:
            with torch.no_grad():
                res_abs_max = warper.residual.abs().max().item()
                res_rms_m = res_m.item()
            print(
                f"  [{level_name}] Iter {i+1}/{iters} | total={total.item():.2f}m "
                f"data={w_data*data_loss.item():.2f} "
                f"cham={w_chamfer*chamfer_loss.item():.2f} "
                f"res|max|={res_abs_max:.6f} resRMS={res_rms_m:.2f}m",
                flush=True
            )

        # Save diagnostic image at iter 1, every 50 iters, and final iter
        if diagnostics_dir is not None and source_img is not None and target_img is not None:
            is_diag_iter = (i == 0 or (i + 1) % 50 == 0)
            if is_diag_iter:
                _save_iteration_diagnostic(
                    warper, source_img, target_img, tgt_shape,
                    level_name, i + 1,
                    {"data": w_data * data_loss.item(),
                     "cham": w_chamfer * chamfer_loss.item(),
                     "total": total.item()},
                    diagnostics_dir,
                )

        # Early stopping: check if loss has plateaued
        if i >= WARMUP_ITERS + _C.EARLY_STOP_WINDOW:
            old_loss = loss_history[-(_C.EARLY_STOP_WINDOW + 1)]
            improvement = (old_loss - loss_val) / (abs(old_loss) + 1e-8)
            if improvement < _C.EARLY_STOP_THRESHOLD:
                final_iter = i + 1
                print(
                    f"  [{level_name}] Early stop at iter {final_iter}/{iters} "
                    f"(<{_C.EARLY_STOP_THRESHOLD*100:.1f}% improvement over {_C.EARLY_STOP_WINDOW} iters)",
                    flush=True
                )
                break

    if final_iter == iters:
        print(f"  [{level_name}] Completed all {iters} iters (no early stop)", flush=True)

    # Save final-iteration diagnostic (if not already saved by the 50-iter cadence)
    if diagnostics_dir is not None and source_img is not None and target_img is not None:
        if final_iter % 50 != 0:
            _save_iteration_diagnostic(
                warper, source_img, target_img, tgt_shape,
                level_name, final_iter,
                {"total": loss_val},
                diagnostics_dir,
            )

    return warper


def optimize_grid_hierarchical(
    source_img: np.ndarray,
    target_img: np.ndarray,
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    source_coast: np.ndarray,
    target_coast: np.ndarray,
    src_shape: Tuple[int, int],
    tgt_shape: Tuple[int, int],
    output_res_m: float = 1.0,
    levels: Optional[List[Tuple[int, int]]] = None,
    lr: float = 0.002,
    w_data: float = 1.0,
    w_chamfer: float = 0.30,
    reclamation_mask_tgt: Optional[np.ndarray] = None,
    w_arap: float = 0.5,
    w_laplacian: float = 0.2,
    w_disp: float = 0.05,
    max_residual_norm: float = 0.03,
    n_real_gcps: Optional[int] = None,
    match_weights: Optional[np.ndarray] = None,
    diagnostics_dir: Optional[str] = None,
    profiler=None,
) -> 'GridWarper':
    """Hierarchical coarse-to-fine grid optimisation.

    levels: list of (grid_size, iters) tuples, e.g. [(8, 200), (24, 200), (64, 200)].
    At each level the displacement field from the previous level is upsampled
    and used to initialise the next, so each level only learns a small residual
    on top of the coarser solution.  Regularisation weights decrease at finer
    levels since smoothness is already ensured by the coarser levels.
    """
    from .profiler import _NullProfiler
    _p = profiler or _NullProfiler()

    if levels is None:
        levels = [(8, 300), (24, 300), (64, 500)]

    # CPU is faster than MPS for tiny grid tensors and avoids compatibility issues.
    device = torch.device('cpu')
    print(f"  [GridOptim] Using device: {device}", flush=True)
    H_s, W_s = src_shape
    H_t, W_t = tgt_shape

    # -----------------------------------------------------------------------
    # Normalise coordinates to [-1, 1]
    # -----------------------------------------------------------------------
    def norm(pts: np.ndarray, W: int, H: int) -> torch.Tensor:
        if len(pts) == 0:
            return torch.empty((0, 2), dtype=torch.float32, device=device)
        p = pts.copy().astype(np.float32)
        p[:, 0] = (p[:, 0] / (W - 1)) * 2 - 1
        p[:, 1] = (p[:, 1] / (H - 1)) * 2 - 1
        return torch.from_numpy(p).to(device)

    src_pts_n  = norm(source_pts,  W_s, H_s)
    tgt_pts_n  = norm(target_pts,  W_t, H_t)
    src_coast_n = norm(source_coast, W_s, H_s)
    tgt_coast_n = norm(target_coast, W_t, H_t)

    def bbox_filter(pts: torch.Tensor, anchors: torch.Tensor,
                    margin: float = 0.08) -> torch.Tensor:
        if pts.shape[0] == 0 or anchors.shape[0] == 0:
            return pts
        lo = (anchors.min(0).values - margin).clamp(-1.0, 1.0)
        hi = (anchors.max(0).values + margin).clamp(-1.0, 1.0)
        keep = ((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) &
                (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
        return pts[keep]

    if src_pts_n.shape[0] > 0:
        src_coast_n = bbox_filter(src_coast_n, src_pts_n)
    if tgt_pts_n.shape[0] > 0:
        tgt_coast_n = bbox_filter(tgt_coast_n, tgt_pts_n)

    # Reclamation-aware filtering: remove coastline points in changed zones
    if reclamation_mask_tgt is not None and tgt_coast_n.shape[0] > 0:
        mask_h, mask_w = reclamation_mask_tgt.shape
        # Convert normalised [-1,1] coords to mask pixel coords
        tgt_coast_px_x = ((tgt_coast_n[:, 0].cpu().numpy() + 1) / 2 * (mask_w - 1)).astype(int)
        tgt_coast_px_y = ((tgt_coast_n[:, 1].cpu().numpy() + 1) / 2 * (mask_h - 1)).astype(int)
        tgt_coast_px_x = np.clip(tgt_coast_px_x, 0, mask_w - 1)
        tgt_coast_px_y = np.clip(tgt_coast_px_y, 0, mask_h - 1)
        recl_keep = ~reclamation_mask_tgt[tgt_coast_px_y, tgt_coast_px_x]
        n_before = tgt_coast_n.shape[0]
        tgt_coast_n = tgt_coast_n[torch.from_numpy(recl_keep).to(tgt_coast_n.device)]
        print(f"  [Reclamation] Filtered target coast: {n_before} → {tgt_coast_n.shape[0]} pts", flush=True)

    if reclamation_mask_tgt is not None and src_coast_n.shape[0] > 0:
        # For source coast, we approximate using the same mask (source is roughly aligned)
        mask_h, mask_w = reclamation_mask_tgt.shape
        src_coast_px_x = ((src_coast_n[:, 0].cpu().numpy() + 1) / 2 * (mask_w - 1)).astype(int)
        src_coast_px_y = ((src_coast_n[:, 1].cpu().numpy() + 1) / 2 * (mask_h - 1)).astype(int)
        src_coast_px_x = np.clip(src_coast_px_x, 0, mask_w - 1)
        src_coast_px_y = np.clip(src_coast_px_y, 0, mask_h - 1)
        recl_keep = ~reclamation_mask_tgt[src_coast_px_y, src_coast_px_x]
        n_before = src_coast_n.shape[0]
        src_coast_n = src_coast_n[torch.from_numpy(recl_keep).to(src_coast_n.device)]
        print(f"  [Reclamation] Filtered source coast: {n_before} → {src_coast_n.shape[0]} pts", flush=True)

    def downsample_pts(pts: torch.Tensor, n: int = 8000) -> torch.Tensor:
        if pts.shape[0] <= n:
            return pts
        idx = torch.linspace(0, pts.shape[0] - 1, n, device=pts.device).long()
        return pts[idx]

    src_coast_n = downsample_pts(src_coast_n)
    tgt_coast_n = downsample_pts(tgt_coast_n)

    # -----------------------------------------------------------------------
    # GCP confidence weights
    # -----------------------------------------------------------------------
    gcp_weights = _compute_gcp_weights(tgt_pts_n, source_pts, src_shape,
                                       n_real_gcps=n_real_gcps,
                                       match_weights=match_weights)

    # -----------------------------------------------------------------------
    # GCP density map for adaptive ARAP (computed at finest grid)
    # -----------------------------------------------------------------------
    finest_gs = levels[-1][0]

    def _compute_gcp_density_map(gcp_pts_n, grid_size, radius=0.15):
        """Per-grid-node GCP density in [0, 1] for adaptive ARAP."""
        gH = gW = grid_size
        density = torch.zeros(gH, gW, device=device)
        if gcp_pts_n.shape[0] == 0:
            return density
        y_g = torch.linspace(-1, 1, gH, device=device)
        x_g = torch.linspace(-1, 1, gW, device=device)
        yy, xx = torch.meshgrid(y_g, x_g, indexing='ij')
        for i in range(gcp_pts_n.shape[0]):
            dx = xx - gcp_pts_n[i, 0]
            dy = yy - gcp_pts_n[i, 1]
            dist = (dx**2 + dy**2).sqrt()
            density += (dist < radius).float()
        # Normalise to [0, 1]
        if density.max() > 0:
            density = density / density.max()
        return density

    # -----------------------------------------------------------------------
    # Affine baseline (computed once at finest requested grid)
    # -----------------------------------------------------------------------
    first_grid_size = (levels[0][0], levels[0][0])
    affine_baseline = _compute_affine_baseline(
        tgt_pts_n, src_pts_n, first_grid_size, device,
        W_t, H_t, output_res_m, n_real_gcps=n_real_gcps,
    )

    if affine_baseline is not None:
        # Verify affine init quality
        tmp_warper = GridWarper(grid_size=first_grid_size).to(device)
        tmp_warper.set_affine_baseline(affine_baseline)
        with torch.no_grad():
            init_err = F.mse_loss(tmp_warper.warp_points(tgt_pts_n), src_pts_n).item()
            init_err_m = math.sqrt(init_err) * (W_t / 2.0) * output_res_m
        print(f"  Affine init residual: {init_err_m:.2f} m RMS", flush=True)
        del tmp_warper
    else:
        print("  WARNING: fewer than 3 GCPs — skipping affine init", flush=True)

    # -----------------------------------------------------------------------
    # Hierarchical pyramid
    # -----------------------------------------------------------------------
    prev_displacement = None  # (1, 2, prev_H, prev_W) total displacement

    for level_idx, (gs, level_iters) in enumerate(levels):
        grid_sz = (gs, gs)
        level_name = f"L{level_idx} {gs}×{gs}"
        section_name = f"L{level_idx}_{gs}x{gs}"
        with _p.section(section_name):
            print(f"\n  === Hierarchical level {level_idx}: {gs}×{gs} grid, "
                  f"{level_iters} iters ===", flush=True)

            warper = GridWarper(grid_size=grid_sz).to(device)

            if prev_displacement is not None:
                # Upsample previous total displacement to current grid size
                upsampled = F.interpolate(
                    prev_displacement, size=grid_sz, mode='bilinear', align_corners=True
                )
                warper.set_affine_baseline(upsampled)
            elif affine_baseline is not None:
                # First level: use affine baseline (resample to this grid size)
                baseline_at_gs = F.interpolate(
                    affine_baseline, size=grid_sz, mode='bilinear', align_corners=True
                )
                warper.set_affine_baseline(baseline_at_gs)

            # Per-level scaling from profile (falls back to last entry for
            # levels beyond the array length).
            from .params import get_params as _get_params
            _prof = _get_params().grid_optim

            def _lvl_scale(arr, idx, fallback_formula=None):
                if idx < len(arr):
                    return arr[idx]
                return arr[-1] if arr else (fallback_formula or 1.0)

            # Regularisation increases at finer levels — chamfer noise from
            # coastline changes (reclamation) dominates at fine grids.
            reg_scale = _lvl_scale(_prof.level_reg_scale, level_idx)
            level_w_arap = w_arap * reg_scale
            level_w_laplacian = w_laplacian * reg_scale

            # Finer levels: trust GCPs more, penalise displacement less.
            data_scale = _lvl_scale(_prof.level_w_data_scale, level_idx)
            disp_scale = _lvl_scale(_prof.level_w_disp_scale, level_idx)
            level_w_data = w_data * data_scale
            level_w_disp = w_disp * disp_scale

            # Finer levels get tighter Huber delta (GCPs act as harder constraints)
            level_huber = max(3.0, 10.0 / (1.0 + level_idx))

            # Chamfer weight decreases at finer levels.
            chamfer_scale = _lvl_scale(_prof.level_chamfer_scale, level_idx)
            level_w_chamfer = w_chamfer * chamfer_scale

            # Adaptive clamp: finer levels allow slightly larger local corrections
            # (reduced from 0.5 to 0.25 to limit deformations at piers/roads)
            level_max_residual = max_residual_norm * (1.0 + 0.25 * level_idx)

            # Uniform ARAP: don't reduce regularization where GCPs are dense.
            # Adaptive ARAP (0.3-0.8) causes grid instability and jagged features.
            # v17 (uniform) got accepted=true with score 70.1, while v18 (0.3)
            # produced north=-88m and regression.
            level_density = None

            warper = _run_single_level(
                warper=warper,
                grid_size=grid_sz,
                tgt_pts_n=tgt_pts_n,
                src_pts_n=src_pts_n,
                src_coast_n=src_coast_n,
                tgt_coast_n=tgt_coast_n,
                src_shape=src_shape,
                tgt_shape=tgt_shape,
                output_res_m=output_res_m,
                iters=level_iters,
                lr=lr,
                w_data=level_w_data,
                w_chamfer=level_w_chamfer,
                w_arap=level_w_arap,
                w_laplacian=level_w_laplacian,
                w_disp=level_w_disp,
                max_residual_norm=level_max_residual,
                level_name=level_name,
                gcp_weights=gcp_weights,
                huber_delta=level_huber,
                gcp_density_map=level_density,
                source_img=source_img if diagnostics_dir else None,
                target_img=target_img if diagnostics_dir else None,
                diagnostics_dir=diagnostics_dir,
            )

            # Store total displacement for next level's initialisation
            with torch.no_grad():
                prev_displacement = warper.displacements.detach().clone()

            # Fold check at each level
            fold_frac = check_folds(warper, (H_t, W_t))
            if fold_frac > 0.01:
                print(
                    f"  [{level_name}] WARNING: {fold_frac*100:.1f}% folds — "
                    f"reverting to previous level", flush=True
                )
                with torch.no_grad():
                    warper.residual.zero_()
                prev_displacement = warper.displacements.detach().clone()
            elif fold_frac > 0:
                print(f"  [{level_name}] Fold check: {fold_frac*100:.2f}% (ok)", flush=True)
            else:
                print(f"  [{level_name}] Fold check: clean", flush=True)

    # Final fold detection
    fold_frac = check_folds(warper, (H_t, W_t))
    if fold_frac > 0.01:
        print(
            f"  WARNING: Final warp has {fold_frac*100:.1f}% folds. "
            f"Falling back to pure affine.", flush=True
        )
        with torch.no_grad():
            warper.residual.zero_()
            if affine_baseline is not None:
                final_baseline = F.interpolate(
                    affine_baseline, size=(levels[-1][0], levels[-1][0]),
                    mode='bilinear', align_corners=True
                )
                warper.set_affine_baseline(final_baseline)

    return warper
