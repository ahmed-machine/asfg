"""Grid-based warp optimizer: affine-baseline + residual parameterisation.

Key design decisions
--------------------
* GridWarper stores a *frozen* affine baseline and a *learnable* residual.
  The clamp and all regularisation apply only to the residual, so a good
  affine initialisation can never be destroyed by the clamp.
* All loss terms are scaled to metres² so weights have stable, interpretable
  meaning across different image sizes.
* DINOv2 model is a module-level singleton (loaded once per process).
* Feature loss is evaluated every FEAT_CADENCE steps to save compute.
* A fold-detection pass after optimisation warns/rejects flipped regions.
* Hierarchical coarse-to-fine pyramid optimization: 8→24→64 grid sizes.
* Multi-scale DINOv2 features: final + intermediate ViT layers.
"""

import math
import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DINOv2 model singleton
# ---------------------------------------------------------------------------
_DINO_MODEL: Optional[torch.nn.Module] = None
_DINO_DEVICE: Optional[torch.device] = None


def _get_dino_model() -> Tuple[torch.nn.Module, torch.device]:
    global _DINO_MODEL, _DINO_DEVICE
    if _DINO_MODEL is None:
        device = torch.device(
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )
        print("  [DINOv2] Loading dinov2_vits14 model...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model = model.to(device)
        model.eval()
        _DINO_MODEL = model
        _DINO_DEVICE = device
        print(f"  [DINOv2] Model loaded on {device}", flush=True)
    return _DINO_MODEL, _DINO_DEVICE


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
            tensor, dense_grid, mode='bicubic',
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
        cell_weight = (1.0 - dm * 0.8).unsqueeze(0).unsqueeze(0)  # (1, 1, H-2, W-1)
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
        # Soft clamp: beyond max_dist_m, gradient = 0.1 (not zero)
        all_mins = torch.where(all_mins <= max_dist_m, all_mins,
                               max_dist_m + (all_mins - max_dist_m) * 0.3)
        return all_mins.mean()

    return 0.5 * (directed(pts1, pts2) + directed(pts2, pts1))


# ---------------------------------------------------------------------------
# DINOv2 feature extraction
# ---------------------------------------------------------------------------

def extract_dinov2_features_tiled(
    img_arr: np.ndarray,
    patch_size: int = 1008,
    stride: int = 756,
) -> torch.Tensor:
    """Extract DINOv2 (ViT-S/14) features with Hann-blended tiled inference.

    img_arr: (H, W, 3) uint8 RGB
    Returns: (384, H/14, W/14) float16 on CPU
    """
    model, device = _get_dino_model()

    H, W, C = img_arr.shape
    assert C == 3, "Image must be RGB"

    patch_h = 14  # ViT patch stride in pixels
    feat_dim = 384
    feat_patch_size = patch_size // patch_h  # feature tokens per tile side

    out_H = math.ceil(H / patch_h)
    out_W = math.ceil(W / patch_h)

    # Accumulator size (may be slightly larger than out_H/W due to stride)
    max_y = ((H - 1) // stride) * stride
    max_x = ((W - 1) // stride) * stride
    acc_H = (max_y + patch_size) // patch_h
    acc_W = (max_x + patch_size) // patch_h

    stitched = torch.zeros((feat_dim, acc_H, acc_W), dtype=torch.float32, device='cpu')
    weight   = torch.zeros((1, acc_H, acc_W),        dtype=torch.float32, device='cpu')

    win1d  = torch.hann_window(feat_patch_size, periodic=False, device='cpu')
    win2d  = (win1d.unsqueeze(0) * win1d.unsqueeze(1)).unsqueeze(0)  # (1, fp, fp)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    n_tiles_y = math.ceil((H) / stride)
    n_tiles_x = math.ceil((W) / stride)
    total_tiles = n_tiles_y * n_tiles_x
    tile_idx = 0

    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                tile_idx += 1
                if tile_idx % 50 == 0 or tile_idx == total_tiles:
                    print(f"  [DINOv2] tile {tile_idx}/{total_tiles}", flush=True)

                y1, y2 = y, min(y + patch_size, H)
                x1, x2 = x, min(x + patch_size, W)
                crop = img_arr[y1:y2, x1:x2]

                ph = patch_size - (y2 - y1)
                pw = patch_size - (x2 - x1)
                if ph > 0 or pw > 0:
                    crop = np.pad(crop, ((0, ph), (0, pw), (0, 0)), mode='reflect')

                t = torch.from_numpy(crop).float().div_(255.0).permute(2, 0, 1).to(device)
                t = (t - mean) / std
                feats = model.forward_features(t.unsqueeze(0))['x_norm_patchtokens']
                feats = feats.permute(0, 2, 1).view(1, feat_dim, feat_patch_size, feat_patch_size)
                feats = feats.squeeze(0).cpu()

                fy, fx = y // patch_h, x // patch_h
                stitched[:, fy:fy + feat_patch_size, fx:fx + feat_patch_size] += feats * win2d
                weight[:,  fy:fy + feat_patch_size, fx:fx + feat_patch_size] += win2d

    stitched = stitched[:, :out_H, :out_W]
    weight   = weight[:,   :out_H, :out_W]
    stitched = stitched / weight.clamp(min=1e-6)

    return stitched.to(torch.float16)


def extract_dinov2_multiscale(
    img_arr: np.ndarray,
    patch_size: int = 1008,
    stride: int = 756,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract DINOv2 features at two scales: final layer + intermediate layer 8.

    Returns (feat_final, feat_mid) both as (C, H/14, W/14) float16 on CPU.
    feat_mid captures finer spatial detail from an intermediate ViT layer.
    """
    model, device = _get_dino_model()

    H, W, C = img_arr.shape
    assert C == 3, "Image must be RGB"

    patch_h = 14
    feat_dim = 384
    feat_patch_size = patch_size // patch_h

    out_H = math.ceil(H / patch_h)
    out_W = math.ceil(W / patch_h)

    max_y = ((H - 1) // stride) * stride
    max_x = ((W - 1) // stride) * stride
    acc_H = (max_y + patch_size) // patch_h
    acc_W = (max_x + patch_size) // patch_h

    stitched_final = torch.zeros((feat_dim, acc_H, acc_W), dtype=torch.float32, device='cpu')
    stitched_mid = torch.zeros((feat_dim, acc_H, acc_W), dtype=torch.float32, device='cpu')
    weight = torch.zeros((1, acc_H, acc_W), dtype=torch.float32, device='cpu')

    win1d = torch.hann_window(feat_patch_size, periodic=False, device='cpu')
    win2d = (win1d.unsqueeze(0) * win1d.unsqueeze(1)).unsqueeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    # Register hook to capture intermediate layer output
    mid_layer_output = [None]
    mid_layer_idx = 8  # layer 8 out of 12 captures mid-level detail

    def hook_fn(module, input, output):
        mid_layer_output[0] = output

    # Register on the 8th transformer block
    hook = model.blocks[mid_layer_idx].register_forward_hook(hook_fn)

    n_tiles_y = math.ceil(H / stride)
    n_tiles_x = math.ceil(W / stride)
    total_tiles = n_tiles_y * n_tiles_x
    tile_idx = 0

    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                tile_idx += 1
                if tile_idx % 50 == 0 or tile_idx == total_tiles:
                    print(f"  [DINOv2-MS] tile {tile_idx}/{total_tiles}", flush=True)

                y1, y2 = y, min(y + patch_size, H)
                x1, x2 = x, min(x + patch_size, W)
                crop = img_arr[y1:y2, x1:x2]

                ph = patch_size - (y2 - y1)
                pw = patch_size - (x2 - x1)
                if ph > 0 or pw > 0:
                    crop = np.pad(crop, ((0, ph), (0, pw), (0, 0)), mode='reflect')

                t = torch.from_numpy(crop).float().div_(255.0).permute(2, 0, 1).to(device)
                t = (t - mean) / std

                out = model.forward_features(t.unsqueeze(0))
                feats_final = out['x_norm_patchtokens']
                feats_final = feats_final.permute(0, 2, 1).view(1, feat_dim, feat_patch_size, feat_patch_size)
                feats_final = feats_final.squeeze(0).cpu()

                # Mid-layer features (captured by hook)
                feats_mid = mid_layer_output[0]
                if feats_mid.dim() == 3:
                    # (B, N+1, C) - skip CLS token
                    feats_mid = feats_mid[:, 1:, :]
                feats_mid = feats_mid.permute(0, 2, 1).view(1, feat_dim, feat_patch_size, feat_patch_size)
                feats_mid = feats_mid.squeeze(0).cpu()

                fy, fx = y // patch_h, x // patch_h
                stitched_final[:, fy:fy + feat_patch_size, fx:fx + feat_patch_size] += feats_final * win2d
                stitched_mid[:, fy:fy + feat_patch_size, fx:fx + feat_patch_size] += feats_mid * win2d
                weight[:, fy:fy + feat_patch_size, fx:fx + feat_patch_size] += win2d

    hook.remove()

    stitched_final = stitched_final[:, :out_H, :out_W] / weight[:, :out_H, :out_W].clamp(min=1e-6)
    stitched_mid = stitched_mid[:, :out_H, :out_W] / weight[:, :out_H, :out_W].clamp(min=1e-6)

    return stitched_final.to(torch.float16), stitched_mid.to(torch.float16)


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
# Main optimisation entry point
# ---------------------------------------------------------------------------

FEAT_CADENCE = 5  # evaluate feature loss every N steps


def optimize_grid(
    source_img: np.ndarray,
    target_img: np.ndarray,
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    source_coast: np.ndarray,
    target_coast: np.ndarray,
    src_shape: Tuple[int, int],
    tgt_shape: Tuple[int, int],
    output_res_m: float = 1.0,
    grid_size: Tuple[int, int] = (20, 20),
    iters: int = 300,
    lr: float = 0.002,
    w_data: float = 1.0,
    w_chamfer: float = 0.3,
    w_feat: float = 0.0,
    w_arap: float = 0.5,
    w_laplacian: float = 0.2,
    w_disp: float = 0.05,
    max_residual_norm: float = 0.03,
) -> 'GridWarper':
    """Optimise a thin-plate displacement grid.

    Parameters
    ----------
    source_img / target_img:
        RGB uint8 arrays (already downsampled by caller) for DINOv2.
    source_pts / target_pts:
        (N, 2) float32, full-resolution image coordinates (px).
    source_coast / target_coast:
        (M, 2) float32, full-resolution image coordinates (px).
    src_shape / tgt_shape:
        (H, W) full-resolution pixel sizes.
    output_res_m:
        metres per pixel of the target grid — used to convert losses to m².
    max_residual_norm:
        Hard clamp on the *residual* in normalised [-1,1] space.
        Default 0.03 ≈ 1155 px ≈ 1005 m at the canonical image size.
        The fold-detection check after optimisation catches bad deformations.
    """
    # CPU is faster than MPS for tiny grid tensors (64×64) and avoids
    # MPS compatibility issues (grid_sampler backward, border padding, bicubic).
    device = torch.device('cpu')

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

    # Focus coastline on the overlap bbox implied by GCPs
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

    def downsample(pts: torch.Tensor, n: int = 8000) -> torch.Tensor:
        if pts.shape[0] <= n:
            return pts
        idx = torch.linspace(0, pts.shape[0] - 1, n, device=pts.device).long()
        return pts[idx]

    src_coast_n = downsample(src_coast_n)
    tgt_coast_n = downsample(tgt_coast_n)

    # -----------------------------------------------------------------------
    # DINOv2 features (skipped when w_feat == 0)
    # -----------------------------------------------------------------------
    if w_feat > 0:
        print("  [DINOv2] Extracting features for source image...", flush=True)
        src_feat = extract_dinov2_features_tiled(source_img)
        print("  [DINOv2] Extracting features for target image...", flush=True)
        tgt_feat = extract_dinov2_features_tiled(target_img)

        src_feat = src_feat.to(device, dtype=torch.float32).unsqueeze(0)
        tgt_feat = tgt_feat.to(device, dtype=torch.float32).unsqueeze(0)

        tgt_feat_norm_const = F.normalize(tgt_feat, p=2, dim=1)
        target_feat_size = (tgt_feat.shape[2], tgt_feat.shape[3])
    else:
        print("  [DINOv2] Skipped (w_feat=0)", flush=True)
        src_feat = torch.zeros(1, 1, 1, 1, device=device)
        tgt_feat_norm_const = torch.zeros(1, 1, 1, 1, device=device)
        target_feat_size = (1, 1)

    # -----------------------------------------------------------------------
    # Warp parameters
    # -----------------------------------------------------------------------
    warper = GridWarper(grid_size=grid_size).to(device)

    # -----------------------------------------------------------------------
    # Affine initialisation
    # -----------------------------------------------------------------------
    if tgt_pts_n.shape[0] >= 3:
        tgt_np = tgt_pts_n.cpu().numpy()
        src_np = src_pts_n.cpu().numpy()
        n = len(tgt_np)
        A = np.zeros((2 * n, 6), dtype=np.float32)
        b = np.zeros(2 * n, dtype=np.float32)
        for i in range(n):
            A[2*i]   = [tgt_np[i, 0], tgt_np[i, 1], 1, 0, 0, 0]
            A[2*i+1] = [0, 0, 0, tgt_np[i, 0], tgt_np[i, 1], 1]
            b[2*i]   = src_np[i, 0]
            b[2*i+1] = src_np[i, 1]
        affine_params, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        print(f"  Affine init rank={rank}, params={affine_params.round(4)}", flush=True)

        with torch.no_grad():
            y_g = np.linspace(-1, 1, grid_size[0], dtype=np.float32)
            x_g = np.linspace(-1, 1, grid_size[1], dtype=np.float32)
            yy, xx = np.meshgrid(y_g, x_g, indexing='ij')
            src_x = affine_params[0]*xx + affine_params[1]*yy + affine_params[2]
            src_y = affine_params[3]*xx + affine_params[4]*yy + affine_params[5]
            baseline = torch.from_numpy(
                np.stack([src_x - xx, src_y - yy], axis=0)
            ).unsqueeze(0).to(device)
            warper.set_affine_baseline(baseline)

            # Verify: data loss should be ~0 after affine init
            init_err = F.mse_loss(warper.warp_points(tgt_pts_n), src_pts_n).item()
            init_err_m = math.sqrt(init_err) * (W_t / 2.0) * output_res_m
            print(f"  Affine init residual: {init_err_m:.2f} m RMS", flush=True)
    else:
        print("  WARNING: fewer than 3 GCPs — skipping affine init", flush=True)

    # -----------------------------------------------------------------------
    # Loss scales: normalised coord 1.0 = (W/2) pixels = (W/2 * res) metres
    # -----------------------------------------------------------------------
    scale_x_m = (W_s / 2.0) * output_res_m  # m per normalised unit in X (source)
    scale_y_m = (H_s / 2.0) * output_res_m  # m per normalised unit in Y (source)
    # Chamfer scale: after warp_points(), both point sets are in source
    # normalised space, so use source dimensions for metre conversion.
    coast_scale_x_m = scale_x_m
    coast_scale_y_m = scale_y_m

    # Feature loss: L2-normalised feature MSE is dimensionless [0,4].
    # Scale it so that 1 unit ≈ 1 metre of average displacement cost.
    # We use a fixed reference: the mean feature-token spacing in metres.
    feat_px_per_token = 14.0
    feat_scale_m = feat_px_per_token * output_res_m   # m per feature token

    # -----------------------------------------------------------------------
    # Optimiser
    # -----------------------------------------------------------------------
    optimizer = torch.optim.Adam(warper.parameters(), lr=lr)

    cached_feat_val = torch.tensor(0.0, device=device)

    # Warmup: ramp chamfer and feature weights from 0 to full over WARMUP iters
    # so the affine baseline isn't disrupted before the residual has started moving.
    WARMUP_ITERS = 50

    print("  Starting grid optimisation...", flush=True)
    for i in range(iters):
        optimizer.zero_grad()

        # Warmup multiplier: 0→1 linearly over WARMUP_ITERS
        warmup = min(1.0, i / max(1, WARMUP_ITERS))

        # 1. Data loss (GCPs) — Huber in metres so outliers don't dominate
        if tgt_pts_n.shape[0] > 0:
            warped = warper.warp_points(tgt_pts_n)
            diff_m = (warped - src_pts_n) * torch.tensor(
                [scale_x_m, scale_y_m], device=device
            )
            # Huber with delta=10 m so GCPs act as hard constraints for
            # errors < 10m (quadratic) and degrade gracefully beyond (linear).
            data_loss = F.huber_loss(diff_m, torch.zeros_like(diff_m), delta=10.0)
        else:
            data_loss = torch.tensor(0.0, device=device)

        # 2. Chamfer loss — mean nearest-neighbour distance in metres
        #    warped_coast is in source normalised space; src_coast_n is also source.
        if src_coast_n.shape[0] > 0 and tgt_coast_n.shape[0] > 0:
            warped_coast = warper.warp_points(tgt_coast_n)
            chamfer_loss = memory_efficient_chamfer(
                warped_coast, src_coast_n,
                scale_x_m=coast_scale_x_m, scale_y_m=coast_scale_y_m,
            )
        else:
            chamfer_loss = torch.tensor(0.0, device=device)

        # 3. Feature loss (DINOv2) — evaluated every FEAT_CADENCE steps
        #    On cadence iterations: compute fresh value (with gradient).
        #    Off-cadence: reuse detached scalar (no gradient, avoids
        #    "backward through graph a second time" error).
        if (i % FEAT_CADENCE) == 0:
            warped_src_feat = warper.warp_image_or_features(
                src_feat, target_size=target_feat_size
            )
            wsfn = F.normalize(warped_src_feat, p=2, dim=1)
            # MSE of unit-norm features is in [0,4]; scale by feat_scale_m
            # so it's roughly comparable to data_loss in metres
            feat_loss = F.mse_loss(wsfn, tgt_feat_norm_const) * feat_scale_m
            cached_feat_val = feat_loss.detach()
        else:
            feat_loss = cached_feat_val

        # 4. ARAP + Laplacian on residual only
        arap_loss, laplacian_loss = asap_regularization(
            warper.residual, grid_size=grid_size,
            target_size=(H_t, W_t), output_res_m=output_res_m,
        )

        # 5. Residual magnitude penalty in metres — pulls toward zero (= affine)
        #    Add epsilon inside sqrt to avoid inf gradient when residual is zero.
        res_m = ((warper.residual[:, 0].pow(2).mean() * scale_x_m**2 +
                  warper.residual[:, 1].pow(2).mean() * scale_y_m**2 + 1e-8).sqrt())

        # Apply warmup to chamfer and feature (data loss is always full strength)
        total = (w_data      * data_loss                  +
                 w_chamfer   * chamfer_loss   * warmup    +
                 w_feat      * feat_loss      * warmup    +
                 w_arap      * arap_loss                  +
                 w_laplacian * laplacian_loss             +
                 w_disp      * res_m)

        total.backward()
        optimizer.step()

        # Hard clamp on residual only (affine baseline is untouched)
        with torch.no_grad():
            warper.residual.clamp_(-max_residual_norm, max_residual_norm)

        # Compute residual stats for logging
        with torch.no_grad():
            res_abs_max = warper.residual.abs().max().item()
            res_rms_m = res_m.item()

        if i == 0:
            print(
                f"  Iter 1 weighted contributions (m):\n"
                f"    data={w_data*data_loss.item():.2f}  "
                f"chamfer={w_chamfer*chamfer_loss.item():.2f}  "
                f"feat={w_feat*feat_loss.item():.4f}  "
                f"arap={w_arap*arap_loss.item():.4f}  "
                f"lap={w_laplacian*laplacian_loss.item():.4f}  "
                f"res={w_disp*res_m.item():.4f}\n"
                f"    residual |max|={res_abs_max:.6f}  RMS={res_rms_m:.2f}m  warmup={warmup:.2f}\n"
                f"    raw_chamfer={chamfer_loss.item():.1f}m  raw_feat_mse={feat_loss.item()/max(feat_scale_m,1e-9):.6f}",
                flush=True
            )

        if (i + 1) % 50 == 0:
            print(
                f"  Iter {i+1}/{iters} | total={total.item():.2f}m "
                f"data={w_data*data_loss.item():.2f} "
                f"cham={w_chamfer*chamfer_loss.item():.2f} "
                f"feat={w_feat*feat_loss.item():.4f} "
                f"arap={w_arap*arap_loss.item():.4f} "
                f"res_pen={w_disp*res_m.item():.4f} "
                f"res|max|={res_abs_max:.6f} resRMS={res_rms_m:.2f}m",
                flush=True
            )

    # -----------------------------------------------------------------------
    # Fold detection
    # -----------------------------------------------------------------------
    fold_frac = check_folds(warper, (H_t, W_t))
    if fold_frac > 0.01:
        print(
            f"  WARNING: {fold_frac*100:.1f}% of warp field has folds "
            f"(det J <= 0).  Falling back to pure affine warp.", flush=True
        )
        # Zero out the residual → output equals affine baseline
        with torch.no_grad():
            warper.residual.zero_()
    elif fold_frac > 0.0:
        print(f"  Fold check: {fold_frac*100:.2f}% folded (acceptable)", flush=True)
    else:
        print("  Fold check: clean (no folds)", flush=True)

    return warper


# ---------------------------------------------------------------------------
# Hierarchical coarse-to-fine grid optimisation (Improvement 1 + 3 + 5)
# ---------------------------------------------------------------------------

def _compute_affine_baseline(
    tgt_pts_n: torch.Tensor,
    src_pts_n: torch.Tensor,
    grid_size: Tuple[int, int],
    device: torch.device,
    W_t: int, H_t: int,
    output_res_m: float,
) -> Optional[torch.Tensor]:
    """Fit affine from GCPs and return baseline displacement (1, 2, H, W)."""
    if tgt_pts_n.shape[0] < 3:
        return None
    tgt_np = tgt_pts_n.cpu().numpy()
    src_np = src_pts_n.cpu().numpy()
    n = len(tgt_np)
    A = np.zeros((2 * n, 6), dtype=np.float32)
    b = np.zeros(2 * n, dtype=np.float32)
    for i in range(n):
        A[2*i]   = [tgt_np[i, 0], tgt_np[i, 1], 1, 0, 0, 0]
        A[2*i+1] = [0, 0, 0, tgt_np[i, 0], tgt_np[i, 1], 1]
        b[2*i]   = src_np[i, 0]
        b[2*i+1] = src_np[i, 1]
    affine_params, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    print(f"  Affine init rank={rank}, params={affine_params.round(4)}", flush=True)

    y_g = np.linspace(-1, 1, grid_size[0], dtype=np.float32)
    x_g = np.linspace(-1, 1, grid_size[1], dtype=np.float32)
    yy, xx = np.meshgrid(y_g, x_g, indexing='ij')
    src_x = affine_params[0]*xx + affine_params[1]*yy + affine_params[2]
    src_y = affine_params[3]*xx + affine_params[4]*yy + affine_params[5]
    baseline = torch.from_numpy(
        np.stack([src_x - xx, src_y - yy], axis=0)
    ).unsqueeze(0).to(device)
    return baseline


def _compute_gcp_weights(
    tgt_pts_n: torch.Tensor,
    source_pts: np.ndarray,
    src_shape: Tuple[int, int],
) -> torch.Tensor:
    """Compute per-GCP confidence weights.

    With good spatial coverage (41+ GCPs across west/east/islands/reefs),
    uniform weighting lets the optimizer use all constraints equally.
    Returns (N,) tensor of uniform 1.0 weights.
    """
    return torch.ones(tgt_pts_n.shape[0], device=tgt_pts_n.device)


def _run_single_level(
    warper: GridWarper,
    grid_size: Tuple[int, int],
    tgt_pts_n: torch.Tensor,
    src_pts_n: torch.Tensor,
    src_coast_n: torch.Tensor,
    tgt_coast_n: torch.Tensor,
    src_feat: torch.Tensor,
    tgt_feat_norm_const: torch.Tensor,
    target_feat_size: Tuple[int, int],
    src_shape: Tuple[int, int],
    tgt_shape: Tuple[int, int],
    output_res_m: float,
    iters: int,
    lr: float,
    w_data: float,
    w_chamfer: float,
    w_feat: float,
    w_arap: float,
    w_laplacian: float,
    w_disp: float,
    max_residual_norm: float,
    level_name: str = "",
    gcp_weights: Optional[torch.Tensor] = None,
    feat_mid: Optional[torch.Tensor] = None,
    tgt_feat_mid_norm: Optional[torch.Tensor] = None,
    w_feat_mid: float = 0.0,
    huber_delta: float = 10.0,
    gcp_density_map: Optional[torch.Tensor] = None,
) -> GridWarper:
    """Run a single level of grid optimisation (inner loop extracted from optimize_grid)."""
    device = warper.residual.device
    H_s, W_s = src_shape
    H_t, W_t = tgt_shape

    scale_x_m = (W_s / 2.0) * output_res_m
    scale_y_m = (H_s / 2.0) * output_res_m
    coast_scale_x_m = scale_x_m
    coast_scale_y_m = scale_y_m
    feat_px_per_token = 14.0
    feat_scale_m = feat_px_per_token * output_res_m

    optimizer = torch.optim.Adam(warper.parameters(), lr=lr)
    cached_feat_val = torch.tensor(0.0, device=device)
    cached_feat_mid_val = torch.tensor(0.0, device=device)

    WARMUP_ITERS = min(30, iters // 4)
    FEAT_CAD = 5

    for i in range(iters):
        optimizer.zero_grad()
        warmup = min(1.0, i / max(1, WARMUP_ITERS))

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

        # 3. Feature loss (DINOv2 final layer)
        if (i % FEAT_CAD) == 0:
            warped_src_feat = warper.warp_image_or_features(
                src_feat, target_size=target_feat_size
            )
            wsfn = F.normalize(warped_src_feat, p=2, dim=1)
            feat_loss = F.mse_loss(wsfn, tgt_feat_norm_const) * feat_scale_m
            cached_feat_val = feat_loss.detach()

            # Mid-layer feature loss (multi-scale DINOv2)
            if feat_mid is not None and tgt_feat_mid_norm is not None and w_feat_mid > 0:
                warped_mid = warper.warp_image_or_features(
                    feat_mid, target_size=target_feat_size
                )
                wmn = F.normalize(warped_mid, p=2, dim=1)
                feat_mid_loss = F.mse_loss(wmn, tgt_feat_mid_norm) * feat_scale_m
                cached_feat_mid_val = feat_mid_loss.detach()
            else:
                feat_mid_loss = torch.tensor(0.0, device=device)
        else:
            feat_loss = cached_feat_val
            feat_mid_loss = cached_feat_mid_val

        # 4. ARAP + Laplacian on residual
        arap_loss, laplacian_loss = asap_regularization(
            warper.residual, grid_size=grid_size,
            target_size=(H_t, W_t), output_res_m=output_res_m,
            gcp_density_map=gcp_density_map,
        )

        # 5. Residual magnitude penalty
        res_m = ((warper.residual[:, 0].pow(2).mean() * scale_x_m**2 +
                  warper.residual[:, 1].pow(2).mean() * scale_y_m**2 + 1e-8).sqrt())

        total = (w_data      * data_loss                       +
                 w_chamfer   * chamfer_loss     * warmup       +
                 w_feat      * feat_loss        * warmup       +
                 w_feat_mid  * feat_mid_loss    * warmup       +
                 w_arap      * arap_loss                       +
                 w_laplacian * laplacian_loss                   +
                 w_disp      * res_m)

        total.backward()
        optimizer.step()

        with torch.no_grad():
            warper.residual.clamp_(-max_residual_norm, max_residual_norm)

        if i == 0:
            res_abs_max = warper.residual.abs().max().item()
            print(
                f"  [{level_name}] Iter 1: data={w_data*data_loss.item():.2f} "
                f"cham={w_chamfer*chamfer_loss.item():.2f} "
                f"feat={w_feat*feat_loss.item():.4f} "
                f"arap={w_arap*arap_loss.item():.4f} "
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
                f"feat={w_feat*feat_loss.item():.4f} "
                f"res|max|={res_abs_max:.6f} resRMS={res_rms_m:.2f}m",
                flush=True
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
    w_chamfer: float = 0.3,
    w_feat: float = 0.0,
    reclamation_mask_tgt: Optional[np.ndarray] = None,
    w_arap: float = 0.5,
    w_laplacian: float = 0.2,
    w_disp: float = 0.05,
    max_residual_norm: float = 0.03,
) -> 'GridWarper':
    """Hierarchical coarse-to-fine grid optimisation.

    levels: list of (grid_size, iters) tuples, e.g. [(8, 200), (24, 200), (64, 200)].
    At each level the displacement field from the previous level is upsampled
    and used to initialise the next, so each level only learns a small residual
    on top of the coarser solution.  Regularisation weights decrease at finer
    levels since smoothness is already ensured by the coarser levels.
    """
    if levels is None:
        levels = [(8, 200), (24, 200), (64, 200)]

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
    # Multi-scale DINOv2 features (skipped when w_feat == 0)
    # -----------------------------------------------------------------------
    if w_feat > 0:
        print("  [DINOv2-MS] Extracting multi-scale features for source image...", flush=True)
        src_feat_final, src_feat_mid = extract_dinov2_multiscale(source_img)
        print("  [DINOv2-MS] Extracting multi-scale features for target image...", flush=True)
        tgt_feat_final, tgt_feat_mid = extract_dinov2_multiscale(target_img)

        src_feat = src_feat_final.to(device, dtype=torch.float32).unsqueeze(0)
        tgt_feat = tgt_feat_final.to(device, dtype=torch.float32).unsqueeze(0)
        src_feat_mid_t = src_feat_mid.to(device, dtype=torch.float32).unsqueeze(0)
        tgt_feat_mid_t = tgt_feat_mid.to(device, dtype=torch.float32).unsqueeze(0)

        tgt_feat_norm_const = F.normalize(tgt_feat, p=2, dim=1)
        tgt_feat_mid_norm = F.normalize(tgt_feat_mid_t, p=2, dim=1)
        target_feat_size = (tgt_feat.shape[2], tgt_feat.shape[3])

        del src_feat_final, src_feat_mid, tgt_feat_final, tgt_feat_mid
    else:
        print("  [DINOv2-MS] Skipped (w_feat=0)", flush=True)
        src_feat = torch.zeros(1, 1, 1, 1, device=device)
        tgt_feat = torch.zeros(1, 1, 1, 1, device=device)
        src_feat_mid_t = torch.zeros(1, 1, 1, 1, device=device)
        tgt_feat_mid_t = torch.zeros(1, 1, 1, 1, device=device)
        tgt_feat_norm_const = torch.zeros(1, 1, 1, 1, device=device)
        tgt_feat_mid_norm = torch.zeros(1, 1, 1, 1, device=device)
        target_feat_size = (1, 1)

    # -----------------------------------------------------------------------
    # GCP confidence weights (Improvement 5)
    # -----------------------------------------------------------------------
    gcp_weights = _compute_gcp_weights(tgt_pts_n, source_pts, src_shape)

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
        W_t, H_t, output_res_m,
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

        # Increase regularisation at finer levels — chamfer noise from coastline
        # changes (reclamation) dominates at fine grids, so stronger ARAP keeps
        # the solution smooth between GCPs (similar to TPS behavior).
        reg_scale = 1.0 + 0.15 * level_idx
        level_w_arap = w_arap * reg_scale
        level_w_laplacian = w_laplacian * reg_scale
        level_w_disp = w_disp * reg_scale

        # Finer levels get tighter Huber delta (GCPs act as harder constraints)
        level_huber = max(3.0, 10.0 / (1.0 + level_idx))

        # Multi-scale feature weight: higher at finer levels
        level_w_feat_mid = 0.3 * w_feat * min(1.0, level_idx / max(1, len(levels) - 1))

        # Chamfer weight decreases at finer levels — more coast points naturally
        # increase chamfer magnitude, and coastline errors (reclamation) dominate
        # at fine resolution. Keep chamfer strongest at coarse (global alignment).
        level_w_chamfer = w_chamfer / (1.0 + 0.5 * level_idx)

        # Compute GCP density map at this grid size for adaptive ARAP
        level_density = _compute_gcp_density_map(tgt_pts_n, gs)

        warper = _run_single_level(
            warper=warper,
            grid_size=grid_sz,
            tgt_pts_n=tgt_pts_n,
            src_pts_n=src_pts_n,
            src_coast_n=src_coast_n,
            tgt_coast_n=tgt_coast_n,
            src_feat=src_feat,
            tgt_feat_norm_const=tgt_feat_norm_const,
            target_feat_size=target_feat_size,
            src_shape=src_shape,
            tgt_shape=tgt_shape,
            output_res_m=output_res_m,
            iters=level_iters,
            lr=lr,
            w_data=w_data,
            w_chamfer=level_w_chamfer,
            w_feat=w_feat,
            w_arap=level_w_arap,
            w_laplacian=level_w_laplacian,
            w_disp=level_w_disp,
            max_residual_norm=max_residual_norm,
            level_name=level_name,
            gcp_weights=gcp_weights,
            feat_mid=src_feat_mid_t,
            tgt_feat_mid_norm=tgt_feat_mid_norm,
            w_feat_mid=level_w_feat_mid,
            huber_delta=level_huber,
            gcp_density_map=level_density,
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
