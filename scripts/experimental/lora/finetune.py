"""Phase 4 — LoRA fine-tune of RoMa v2 with curriculum + per-source weighting.

Trains LoRA adapters on the DINOv3 ViT-L/16 backbone of RoMa v2 against
two label sources:

  1. CORONA SSD pairs (``data/ssd_pairs/``) — supervision from the recorded
     synthetic affine ``M`` on a uniform grid. Geometric augmentation is
     applied per the curriculum schedule (Phase 2.3 of the plan).
  2. mapbh pairs (``data/lora_pairs/``) — teacher pseudo-labels precomputed
     by ``extract_mapbh_pseudo_labels.py``. No geometric augmentation; only
     photometric.

Critical setup detail: after constructing ``RoMaV2(setting="fast")`` the
trainer calls ``model.apply_setting("satast")`` so resolution + bidirectional
flow + balanced sampling match what the model was originally trained at.

The forward path bypasses the wrapped ``model.f`` (which gates gradients
when frozen) by calling ``model.f.get_intermediate_layers`` directly with
manual ImageNet normalisation — this is the only way LoRA gradients flow
through the DINOv3 attention modules.

After training completes the LoRA delta is merged into the base RoMa weights
and saved to ``align/weights/roma_ssd.pth`` so the production ``DECLASS_ROMA_WEIGHTS``
gate can ``load_state_dict(strict=True)`` it directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Curriculum schedule (Phase 2.3)
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStage:
    scale_lo: float
    scale_hi: float
    rot_max_deg: float
    trans_max_frac: float
    tps_strength: float


CURRICULUM = [
    CurriculumStage(0.6, 1.4, 25.0, 0.20, 0.04),   # epoch 0 — wide
    CurriculumStage(0.75, 1.25, 15.0, 0.12, 0.025),  # epoch 1 — medium
    CurriculumStage(0.85, 1.15, 9.0, 0.08, 0.015),  # epoch 2+ — tight
]


def _stage_for_epoch(epoch: int) -> CurriculumStage:
    return CURRICULUM[min(epoch, len(CURRICULUM) - 1)]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _photometric_augment(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    img = img.astype(np.float32)
    img = img * rng.uniform(0.8, 1.2)
    mean = img.mean()
    img = (img - mean) * rng.uniform(0.8, 1.2) + mean
    img = np.clip(img, 0, 255)
    img = 255.0 * (img / 255.0) ** rng.uniform(0.7, 1.4)
    img = img + rng.normal(0, rng.uniform(3, 12), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        # JPEG cycle
        q = int(rng.randint(35, 90))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE)
    return img


def _build_curriculum_affine(
    rng: np.random.RandomState, stage: CurriculumStage, size: int
) -> np.ndarray:
    """Return a 2x3 affine in pixel coords for the current stage."""
    s = rng.uniform(stage.scale_lo, stage.scale_hi)
    rot = np.radians(rng.uniform(-stage.rot_max_deg, stage.rot_max_deg))
    tx = rng.uniform(-stage.trans_max_frac, stage.trans_max_frac) * size
    ty = rng.uniform(-stage.trans_max_frac, stage.trans_max_frac) * size

    cx, cy = size / 2, size / 2
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    a = s * cos_r
    b = -s * sin_r
    c = s * sin_r
    d = s * cos_r
    # Rotate around image centre
    e = (1 - a) * cx - b * cy + tx
    f = -c * cx + (1 - d) * cy + ty
    M = np.array([[a, b, e], [c, d, f]], dtype=np.float32)
    return M


def _apply_affine(img: np.ndarray, M: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpAffine(
        img, M, (size, size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def _build_tps_warp(rng: np.random.RandomState, strength: float, size: int, n_pts: int = 5):
    """Generate a TPS warp from a regular grid + random per-point perturbation.

    Returns (tps, src_pts, dst_pts). The warp maps a pixel at ``src_pts[i]`` in
    the original image to ``dst_pts[i]`` in the warped image.

    Use ``tps.warpImage(img)`` to warp pixels and ``tps.applyTransformation(pts)``
    to map a set of (x, y) image-pixel coordinates from unwarped → warped.
    """
    xs, ys = np.meshgrid(
        np.linspace(0.0, size - 1, n_pts),
        np.linspace(0.0, size - 1, n_pts),
    )
    src_pts = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32)
    dst_pts = src_pts + rng.uniform(-strength, strength, src_pts.shape).astype(np.float32) * size
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    tps = cv2.createThinPlateSplineShapeTransformer()
    # estimateTransformation(transformingShape, targetShape, matches) maps
    # transformingShape onto targetShape. We want the warp that moves
    # ``src_pts`` to ``dst_pts``, so transformingShape = src_pts.
    tps.estimateTransformation(src_pts.reshape(1, -1, 2), dst_pts.reshape(1, -1, 2), matches)
    return tps, src_pts, dst_pts


def _apply_tps_image(img: np.ndarray, tps, size: int) -> np.ndarray:
    return tps.warpImage(img)


def _apply_tps_points_norm(pts_norm: np.ndarray, tps, size: int) -> np.ndarray:
    """Map (N, 2) normalised [-1, 1] points through ``tps``, returning (N, 2) normalised."""
    pts_pix = (pts_norm + 1.0) * (size - 1) / 2.0  # (N, 2) in pixels
    pts_pix_2d = pts_pix.reshape(1, -1, 2).astype(np.float32)
    _, transformed = tps.applyTransformation(pts_pix_2d)
    transformed = transformed.reshape(-1, 2)
    return transformed / ((size - 1) / 2.0) - 1.0


def _random_hflip(
    ref: np.ndarray,
    src: np.ndarray,
    m: torch.Tensor,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Correspondence-aware horizontal flip with p=0.5.

    Flips both images left-right and negates the x-component of the normalised
    correspondence coordinates (cols 0 and 2 of m).
    """
    if rng.random() >= 0.5:
        return ref, src, m
    ref = np.ascontiguousarray(ref[:, ::-1])
    src = np.ascontiguousarray(src[:, ::-1])
    m = m.clone()
    m[:, 0] = -m[:, 0]
    m[:, 2] = -m[:, 2]
    return ref, src, m


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class SourceSpec:
    root: Path
    kind: str  # "corona_synth" or "mapbh"
    weight: float
    label_dir: Path | None = None  # for mapbh: directory of roma_raw_*.pt files
    # Optional per-pair_kind weight overrides (mapbh only). Keys are
    # "kh-modern", "kh-kh", "modern-modern". Falls back to ``weight`` for
    # unknown keys. The plan defaults: kh-kh boosted to 0.8 since the user
    # specifically cares about cross-mission KH↔KH matching.
    pair_kind_weights: dict | None = None


class LoRADataset(Dataset):
    """Combined dataset: yields (ref, src, m_target, c_target, source_weight, kind).

    For CORONA pairs, m_target is built on-the-fly from the recorded affine M
    (per-batch, also accounting for the curriculum perturbation).
    For mapbh pairs, m_target is loaded from the precomputed teacher labels.
    """

    def __init__(
        self,
        sources: list[SourceSpec],
        size: int = 800,
        augment: bool = True,
        n_grid_corona: int = 1500,
        augment_mapbh: bool = False,
        use_tps: bool = True,
        use_hflip: bool = True,
    ) -> None:
        self.size = size
        self.augment = augment
        # Mutated by the trainer between epochs; safe with multiprocessing
        # because each iter(loader) re-pickles the dataset state.
        self.current_epoch = 0
        self.n_grid_corona = n_grid_corona
        # Whether to apply geometric augmentation to mapbh pairs (default off:
        # the cross-temporal change between layers IS the supervision signal).
        # When on, mapbh pairs receive a tight envelope (CORONA epoch 2-3 stage)
        # so the synthetic noise stays modest relative to the natural change.
        self.augment_mapbh = augment_mapbh
        self.use_tps = use_tps
        self.use_hflip = use_hflip
        # Each entry: (spec, meta_path, pair_kind, per-entry weight). pair_kind
        # is "corona_synth" for CORONA and one of {"kh-modern","kh-kh","modern-modern"}
        # for mapbh (read from each meta.json).
        self.entries: list[tuple[SourceSpec, Path, str, float]] = []
        for spec in sources:
            metas = sorted(spec.root.glob("pair_*_meta.json"))
            for meta_path in metas:
                stem = meta_path.name[: -len("_meta.json")]
                ref_path = spec.root / f"{stem}_ref.png"
                src_path_clean = spec.root / f"{stem}_src_clean.png"
                src_path_perturbed = spec.root / f"{stem}_src_perturbed.png"
                # CORONA needs ref + src_perturbed + a valid affine_matrix
                if spec.kind == "corona_synth":
                    if ref_path.exists() and src_path_perturbed.exists():
                        try:
                            meta = json.loads(meta_path.read_text())
                            if "affine_matrix" in meta and meta["affine_matrix"] is not None:
                                self.entries.append((spec, meta_path, "corona_synth", spec.weight))
                        except Exception:
                            continue
                else:  # mapbh: ref + src_clean + label file
                    label_path = (spec.label_dir or spec.root / "labels") / f"roma_raw_{stem}.pt"
                    if not (ref_path.exists() and src_path_clean.exists() and label_path.exists()):
                        continue
                    try:
                        meta = json.loads(meta_path.read_text())
                    except Exception:
                        continue
                    pk = meta.get("pair_kind", "kh-modern")
                    if spec.pair_kind_weights:
                        w = spec.pair_kind_weights.get(pk, spec.weight)
                    else:
                        w = spec.weight
                    self.entries.append((spec, meta_path, pk, float(w)))

    def __len__(self) -> int:
        return len(self.entries)

    def source_weights(self) -> list[float]:
        """Per-sample sampling weight for WeightedRandomSampler.

        Each entry's weight is normalised by the count of pairs of the same
        pair_kind so all kinds get a probabilistic mix in expectation.
        """
        per_kind = {}
        for _, _, pk, _ in self.entries:
            per_kind[pk] = per_kind.get(pk, 0) + 1
        return [
            w / max(per_kind[pk], 1)
            for _, _, pk, w in self.entries
        ]

    def kind_counts(self) -> dict[str, int]:
        out = {}
        for _, _, pk, _ in self.entries:
            out[pk] = out.get(pk, 0) + 1
        return out

    def __getitem__(self, idx: int):
        spec, meta_path, entry_pair_kind, entry_weight = self.entries[idx]
        meta = json.loads(meta_path.read_text())
        stem = meta_path.name[: -len("_meta.json")]
        ref_bgr = cv2.imread(str(spec.root / f"{stem}_ref.png"), cv2.IMREAD_COLOR)
        if spec.kind == "corona_synth":
            src_bgr = cv2.imread(str(spec.root / f"{stem}_src_perturbed.png"), cv2.IMREAD_COLOR)
        else:
            src_bgr = cv2.imread(str(spec.root / f"{stem}_src_clean.png"), cv2.IMREAD_COLOR)
        if ref_bgr is None or src_bgr is None:
            raise RuntimeError(f"failed to load images for {stem}")

        rng = np.random.RandomState()
        if self.augment:
            ref_bgr = _photometric_augment(ref_bgr, rng)
            src_bgr = _photometric_augment(src_bgr, rng)

        # Resize to target size
        if ref_bgr.shape[0] != self.size or ref_bgr.shape[1] != self.size:
            ref_bgr = cv2.resize(ref_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if src_bgr.shape[0] != self.size or src_bgr.shape[1] != self.size:
            src_bgr = cv2.resize(src_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)

        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1)

        # Build labels per source kind. Geometric augmentation order: affine, then TPS.
        # Hflip (correspondence-aware) is applied last so it operates on the
        # already-warped image and label.
        if spec.kind == "corona_synth":
            stage = _stage_for_epoch(self.current_epoch)
            do_aug = self.augment

            # 1) Curriculum-driven extra affine on top of the recorded M
            M_extra = _build_curriculum_affine(rng, stage, self.size) if do_aug else np.eye(2, 3, dtype=np.float32)
            src_warped = _apply_affine(src_bgr, M_extra, self.size)

            # 2) Optional TPS layered on top
            if do_aug and self.use_tps and stage.tps_strength > 0:
                tps, _, _ = _build_tps_warp(rng, stage.tps_strength, self.size)
                src_warped = _apply_tps_image(src_warped, tps, self.size)
            else:
                tps = None

            src_rgb = cv2.cvtColor(src_warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Build labels: ref_pix → M_full → src_pix → tps → final_src_pix
            M_recorded = np.asarray(meta["affine_matrix"], dtype=np.float32)
            orig_size = ref_bgr.shape[0] if ref_bgr.shape[0] != self.size else 1024
            k = self.size / orig_size
            M_record_size = M_recorded.copy()
            M_record_size[:, 2] *= k
            M_full = _compose_affine(M_extra, M_record_size)
            m_target = _build_grid_correspondences(self.n_grid_corona, self.size, M_full, rng)
            if tps is not None:
                # Pass the target side through TPS in normalised coords
                tgt_norm = m_target[:, 2:].numpy()
                tgt_norm_warped = _apply_tps_points_norm(tgt_norm, tps, self.size)
                in_bounds = np.all(np.abs(tgt_norm_warped) < 1.0, axis=1)
                m_target = m_target[in_bounds].clone()
                m_target[:, 2:] = torch.from_numpy(tgt_norm_warped[in_bounds].astype(np.float32))
            c_target = torch.ones(m_target.shape[0], dtype=torch.float32)
        else:  # mapbh
            label_path = (spec.label_dir or spec.root / "labels") / f"roma_raw_{stem}.pt"
            label = torch.load(label_path, map_location="cpu", weights_only=False)
            m_target = label["m"].float()
            c_target = label["c"].float()

            do_aug = self.augment and self.augment_mapbh
            if do_aug:
                # Tight envelope (CORONA epoch 2-3 stage) to keep synthetic noise
                # modest relative to the natural cross-temporal change.
                tight = CURRICULUM[-1]
                M_aug = _build_curriculum_affine(rng, tight, self.size)
                src_bgr = _apply_affine(src_bgr, M_aug, self.size)
                if self.use_tps and tight.tps_strength > 0:
                    tps, _, _ = _build_tps_warp(rng, tight.tps_strength, self.size)
                    src_bgr = _apply_tps_image(src_bgr, tps, self.size)
                else:
                    tps = None
                # Transform target side through M_aug then TPS
                tgt_norm = m_target[:, 2:].numpy()
                tgt_pix = (tgt_norm + 1.0) * (self.size - 1) / 2.0
                ones = np.ones((tgt_pix.shape[0], 1), dtype=np.float32)
                tgt_aug_pix = (np.concatenate([tgt_pix, ones], axis=1) @ M_aug.T)
                tgt_aug_norm = tgt_aug_pix / ((self.size - 1) / 2.0) - 1.0
                if tps is not None:
                    tgt_aug_norm = _apply_tps_points_norm(tgt_aug_norm.astype(np.float32), tps, self.size)
                in_bounds = np.all(np.abs(tgt_aug_norm) < 1.0, axis=1)
                m_target = m_target[in_bounds].clone()
                c_target = c_target[in_bounds].clone()
                m_target[:, 2:] = torch.from_numpy(tgt_aug_norm[in_bounds].astype(np.float32))
            src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Optional correspondence-aware hflip applies to both kinds
        if self.augment and self.use_hflip and m_target.shape[0] > 0:
            ref_rgb_arr = (ref_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            src_rgb_arr = (src_rgb * 255).astype(np.uint8)
            ref_rgb_arr, src_rgb_arr, m_target = _random_hflip(ref_rgb_arr, src_rgb_arr, m_target, rng)
            ref_t = torch.from_numpy(ref_rgb_arr.astype(np.float32) / 255.0).permute(2, 0, 1)
            src_rgb = src_rgb_arr.astype(np.float32) / 255.0

        src_t = torch.from_numpy(src_rgb).permute(2, 0, 1)
        return ref_t, src_t, m_target, c_target, float(entry_weight), spec.kind, entry_pair_kind


def _compose_affine(M_outer: np.ndarray, M_inner: np.ndarray) -> np.ndarray:
    """Return ``M_outer @ M_inner`` for 2x3 affines."""
    A_out = M_outer[:, :2]
    t_out = M_outer[:, 2]
    A_in = M_inner[:, :2]
    t_in = M_inner[:, 2]
    A = A_out @ A_in
    t = A_out @ t_in + t_out
    return np.concatenate([A, t[:, None]], axis=1)


def _build_grid_correspondences(
    n: int, size: int, M_pix: np.ndarray, rng: np.random.RandomState
) -> torch.Tensor:
    """Sample N (x_A, y_A, x_B, y_B) correspondences in normalised [-1, 1] using affine M_pix."""
    # Random query points in normalised coords inside [-0.95, 0.95]
    queries_norm = rng.uniform(-0.95, 0.95, (n, 2)).astype(np.float32)
    # Convert to pixel coords
    queries_pix = (queries_norm + 1.0) * (size - 1) / 2.0  # (n, 2)
    ones = np.ones((n, 1), dtype=np.float32)
    queries_h = np.concatenate([queries_pix, ones], axis=1)
    target_pix = queries_h @ M_pix.T  # (n, 2)
    # Convert back to normalised
    target_norm = target_pix / ((size - 1) / 2.0) - 1.0
    # Filter out targets that fall outside the image
    in_bounds = np.all(np.abs(target_norm) < 1.0, axis=1)
    queries_norm = queries_norm[in_bounds]
    target_norm = target_norm[in_bounds]
    m = np.concatenate([queries_norm, target_norm], axis=1).astype(np.float32)
    return torch.from_numpy(m)


def lora_collate(batch: list) -> dict:
    """Custom collate that keeps per-sample variable-N labels as lists."""
    refs = torch.stack([b[0] for b in batch])
    srcs = torch.stack([b[1] for b in batch])
    ms = [b[2] for b in batch]
    cs = [b[3] for b in batch]
    weights = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    kinds = [b[5] for b in batch]
    pair_kinds = [b[6] for b in batch]
    return {
        "refs": refs, "srcs": srcs, "ms": ms, "cs": cs,
        "weights": weights, "kinds": kinds, "pair_kinds": pair_kinds,
    }


# ---------------------------------------------------------------------------
# Forward path with LoRA gradients (bypass)
# ---------------------------------------------------------------------------


def _bypass_features_with_grad(model_f, img: torch.Tensor, layers: list[int], device: str) -> list[torch.Tensor]:
    """Bypass the wrapped forward to flow gradients into DINOv3 attn modules."""
    from align.romav2.normalizers import imagenet
    img_norm = imagenet(img)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
        raw = model_f.get_intermediate_layers(img_norm, n=layers)
    H = img.shape[2] // 16
    W = img.shape[3] // 16
    return [rearrange(x, "B (H W) D -> B H W D", H=H, W=W) for x in raw]


def _compute_mv_features_no_grad(model, f_A, f_B, device: str):
    """mv_vit forward — no grad, bf16 autocast, mirrors deleted finetune.py."""
    from align.romav2.matcher import _compute_match_embeddings
    f_A_cat = torch.cat(f_A, dim=-1)
    f_B_cat = torch.cat(f_B, dim=-1)
    with torch.autocast(device, torch.bfloat16, enabled=True):
        f_mv = model.matcher.mv_vit(torch.stack((f_A_cat, f_B_cat), dim=1))["x_norm_patchtokens"]
        B_sz = f_A_cat.shape[0]
        H_A, W_A = f_A_cat.shape[1], f_A_cat.shape[2]
        f_mv = f_mv.reshape(B_sz, 2, H_A, W_A, model.matcher.cfg.dim)
    return f_mv[:, 0].float().detach(), f_mv[:, 1].float().detach()


def _forward_train(model, refs: torch.Tensor, srcs: torch.Tensor, device: str):
    """Forward through DINOv3 (with grad on LoRA) → mv_vit (no grad) → matcher head.

    Returns the dict ``{"warp_AB", "confidence_AB", "warp_BA", "confidence_BA"}``.
    """
    from align.romav2.matcher import _compute_match_embeddings, _compute_head_preds
    from align.romav2.geometry import get_normalized_grid

    # Resize to lr (satast: 800)
    img_A_lr = F.interpolate(refs, size=(model.H_lr, model.W_lr), mode="bicubic", align_corners=False, antialias=True)
    img_B_lr = F.interpolate(srcs, size=(model.H_lr, model.W_lr), mode="bicubic", align_corners=False, antialias=True)

    f_A = _bypass_features_with_grad(model.f, img_A_lr, model.f.cfg.layer_idx, device) \
        if hasattr(model.f, "cfg") else _bypass_features_with_grad(model.f, img_A_lr, [11, 17], device)
    f_B = _bypass_features_with_grad(model.f, img_B_lr, [11, 17], device)

    f_mv_A, f_mv_B = _compute_mv_features_no_grad(model, f_A, f_B, device)

    B_sz = f_mv_A.shape[0]
    H_A, W_A = f_mv_A.shape[1], f_mv_A.shape[2]
    x = get_normalized_grid(B_sz, H_A, W_A).to(device)
    x_emb = F.linear(
        x.reshape(B_sz, H_A * W_A, 2),
        model.matcher.scale * model.matcher.omega,
    ).reshape(B_sz, H_A, W_A, -1)
    pos_emb_grid = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)

    _, _, match_emb_AB = _compute_match_embeddings(
        f_A=f_mv_A, f_B=f_mv_B, pos_emb_grid=pos_emb_grid,
        temp=model.matcher.temp, B=B_sz,
        H_A=H_A, W_A=W_A, H_B=H_A, W_B=W_A,
    )
    warp_AB, confidence_AB = _compute_head_preds(
        f_list_A=f_A, match_emb_AB=match_emb_AB, f_mv_A=f_mv_A,
        img_A=img_A_lr, img_B=img_B_lr, head=model.matcher.head,
    )

    if model.bidirectional:
        _, _, match_emb_BA = _compute_match_embeddings(
            f_A=f_mv_B, f_B=f_mv_A, pos_emb_grid=pos_emb_grid,
            temp=model.matcher.temp, B=B_sz,
            H_A=H_A, W_A=W_A, H_B=H_A, W_B=W_A,
        )
        warp_BA, confidence_BA = _compute_head_preds(
            f_list_A=f_B, match_emb_AB=match_emb_BA, f_mv_A=f_mv_B,
            img_A=img_B_lr, img_B=img_A_lr, head=model.matcher.head,
        )
    else:
        warp_BA = None
        confidence_BA = None

    return {
        "warp_AB": warp_AB,
        "confidence_AB": confidence_AB,
        "warp_BA": warp_BA,
        "confidence_BA": confidence_BA,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _compute_loss(preds: dict, batch: dict, conf_weight: float, device: str) -> tuple[torch.Tensor, dict]:
    refs = batch["refs"]
    B_sz = refs.shape[0]
    weights = batch["weights"].to(device)
    ms = batch["ms"]
    cs = batch["cs"]
    kinds = batch["kinds"]
    pair_kinds = batch.get("pair_kinds", kinds)

    total_loss = torch.tensor(0.0, device=device)
    per_kind_loss: dict[str, list[float]] = {
        "corona_synth": [], "kh-modern": [], "kh-kh": [], "modern-modern": [],
    }
    n_used = 0

    warp_AB = preds["warp_AB"]
    conf_AB = preds["confidence_AB"]
    warp_BA = preds["warp_BA"]
    conf_BA = preds["confidence_BA"]

    for i in range(B_sz):
        m = ms[i].to(device)
        c = cs[i].to(device)
        if m.shape[0] == 0:
            continue
        warp_i_AB = warp_AB[i]  # (H_w, W_w, 2)
        H_w, W_w = warp_i_AB.shape[:2]

        # Sample predictions at query points (m[:, :2] in normalised [-1,1])
        px_A = ((m[:, 0] + 1) * 0.5 * (W_w - 1)).clamp(0, W_w - 1).long()
        py_A = ((m[:, 1] + 1) * 0.5 * (H_w - 1)).clamp(0, H_w - 1).long()
        pred_warp_AB = warp_i_AB[py_A, px_A].contiguous()  # (N, 2)
        target_warp_AB = m[:, 2:].contiguous()
        warp_loss_AB = F.smooth_l1_loss(pred_warp_AB, target_warp_AB, beta=2.0)

        pred_conf_AB = conf_AB[i][py_A, px_A, 0].contiguous()
        conf_loss_AB = F.mse_loss(pred_conf_AB, c)

        if warp_BA is not None:
            warp_i_BA = warp_BA[i]
            # For BA we sample at the *target* (B-side) normalised coords
            px_B = ((m[:, 2] + 1) * 0.5 * (W_w - 1)).clamp(0, W_w - 1).long()
            py_B = ((m[:, 3] + 1) * 0.5 * (H_w - 1)).clamp(0, H_w - 1).long()
            pred_warp_BA = warp_i_BA[py_B, px_B].contiguous()
            target_warp_BA = m[:, :2].contiguous()  # query in A is the BA target
            warp_loss_BA = F.smooth_l1_loss(pred_warp_BA, target_warp_BA, beta=2.0)
            pred_conf_BA = conf_BA[i][py_B, px_B, 0].contiguous()
            conf_loss_BA = F.mse_loss(pred_conf_BA, c)
        else:
            warp_loss_BA = torch.tensor(0.0, device=device)
            conf_loss_BA = torch.tensor(0.0, device=device)

        w_loss = warp_loss_AB + warp_loss_BA + conf_weight * (conf_loss_AB + conf_loss_BA)
        bucket = pair_kinds[i] if pair_kinds[i] in per_kind_loss else kinds[i]
        per_kind_loss.setdefault(bucket, []).append(float(w_loss.detach()))
        total_loss = total_loss + weights[i] * w_loss
        n_used += 1

    if n_used == 0:
        return total_loss, per_kind_loss
    return total_loss / max(n_used, 1), per_kind_loss


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------


def _eval_loss(model, loader, device: str, conf_weight: float) -> dict:
    model.eval()
    sum_loss = 0.0
    n = 0
    per_kind: dict[str, list[float]] = {
        "corona_synth": [], "kh-modern": [], "kh-kh": [], "modern-modern": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch["refs"] = batch["refs"].to(device)
            batch["srcs"] = batch["srcs"].to(device)
            preds = _forward_train(model, batch["refs"], batch["srcs"], device)
            loss, kind_losses = _compute_loss(preds, batch, conf_weight, device)
            sum_loss += float(loss)
            for k, v in kind_losses.items():
                per_kind.setdefault(k, []).extend(v)
            n += 1
    model.train()
    out = {"loss": sum_loss / max(n, 1)}
    for k, vs in per_kind.items():
        out[f"{k}_loss"] = float(np.mean(vs)) if vs else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 LoRA fine-tune")
    p.add_argument("--pairs-dir", type=Path, default=REPO_ROOT / "data" / "ssd_pairs")
    p.add_argument("--mapbh-dir", type=Path, default=REPO_ROOT / "data" / "lora_pairs")
    p.add_argument("--mapbh-labels-dir", type=Path, default=None)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--conf-weight", type=float, default=0.5)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--n-grid-corona", type=int, default=1500)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment-mapbh", action="store_true",
                   help="Apply tight-envelope geometric augmentation to mapbh pairs (default off)")
    p.add_argument("--no-tps", action="store_true", help="Disable TPS warp augmentation")
    p.add_argument("--no-hflip", action="store_true", help="Disable correspondence-aware horizontal flip")
    p.add_argument("--mapbh-kh-modern-weight", type=float, default=0.6,
                   help="Per-sample loss weight for mapbh KH↔modern pairs")
    p.add_argument("--mapbh-kh-kh-weight", type=float, default=0.8,
                   help="Per-sample loss weight for mapbh KH↔KH cross-mission/cross-era pairs")
    p.add_argument("--mapbh-mm-weight", type=float, default=0.3,
                   help="Per-sample loss weight for mapbh modern↔modern sanity slice")
    p.add_argument("--run-id", type=str, default=f"lora_r8_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--output-weights", type=Path, default=REPO_ROOT / "align" / "weights" / "roma_ssd.pth")
    args = p.parse_args()

    run_dir = REPO_ROOT / "diagnostics" / "lora_train" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output_weights.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from align.models import get_torch_device
    device = get_torch_device()
    print(f"[finetune] device: {device}")
    print(f"[finetune] run_dir: {run_dir}")

    # ----- Build sources -----
    sources: list[SourceSpec] = []
    if args.pairs_dir.exists():
        sources.append(SourceSpec(args.pairs_dir, "corona_synth", weight=1.0))
        print(f"[finetune] CORONA SSD pairs: {args.pairs_dir}")
    if args.mapbh_dir.exists():
        labels_dir = args.mapbh_labels_dir or (args.mapbh_dir / "labels")
        if labels_dir.exists():
            # User specifically cares about KH↔KH cross-mission matches → boost
            # kh-kh weight relative to kh-modern. modern-modern stays low (sanity).
            pair_kind_weights = {
                "kh-modern": args.mapbh_kh_modern_weight,
                "kh-kh": args.mapbh_kh_kh_weight,
                "modern-modern": args.mapbh_mm_weight,
            }
            sources.append(SourceSpec(
                args.mapbh_dir, "mapbh",
                weight=args.mapbh_kh_modern_weight,  # default fallback
                label_dir=labels_dir,
                pair_kind_weights=pair_kind_weights,
            ))
            print(f"[finetune] mapbh pairs: {args.mapbh_dir}, labels: {labels_dir}")
            print(f"[finetune] mapbh weights: {pair_kind_weights}")
        else:
            print(f"[finetune] mapbh dir exists but {labels_dir} does not — run extract_mapbh_pseudo_labels.py first")
    if not sources:
        raise RuntimeError("no data sources found; populate at least one of pairs-dir or mapbh-dir")

    train_full = LoRADataset(
        sources, size=args.size, augment=True,
        n_grid_corona=args.n_grid_corona,
        augment_mapbh=args.augment_mapbh,
        use_tps=not args.no_tps,
        use_hflip=not args.no_hflip,
    )
    eval_full = LoRADataset(
        sources, size=args.size, augment=False,
        n_grid_corona=args.n_grid_corona,
        augment_mapbh=False,
        use_tps=False,
        use_hflip=False,
    )
    if len(train_full) == 0:
        raise RuntimeError("LoRADataset is empty after filtering")
    print(f"[finetune] dataset size: {len(train_full)}")
    print(f"[finetune] pair-kind counts: {train_full.kind_counts()}")

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(train_full))
    n_train = int(0.8 * len(train_full))
    n_val = int(0.1 * len(train_full))
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(eval_full, val_idx)
    test_ds = Subset(eval_full, test_idx)
    print(f"[finetune] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    sample_weights = train_full.source_weights()
    sampler_weights = [sample_weights[i] for i in train_idx]
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_idx), replacement=True)
    train_loader = DataLoader(
        train_ds, batch_size=1, sampler=sampler,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        collate_fn=lora_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lora_collate
    )

    # ----- Build model -----
    print(f"[finetune] loading RoMa v2 ...")
    from align.romav2 import RoMaV2
    from align.romav2.device import set_device
    set_device(device)
    cfg = RoMaV2.Cfg(setting="fast", compile=False)
    model = RoMaV2(cfg)
    model.apply_setting("satast")
    print(f"[finetune] model satast: H_lr={model.H_lr}")

    # Inject LoRA into DINOv3 blocks 0-17
    from align.romav2.lora import inject_lora, lora_parameters
    from align.romav2.lora_io import save_merged_weights
    n_lora = inject_lora(
        model.f,
        rank=args.rank,
        alpha=args.alpha,
        target_layer_indices=list(range(18)),
        target_param_names=["attn.qkv", "attn.proj"],
        dropout=args.lora_dropout,
    )
    print(f"[finetune] injected {n_lora:,} LoRA parameters at r={args.rank}")

    # Freeze everything except LoRA params
    for p_ in model.parameters():
        p_.requires_grad = False
    lora_p = lora_parameters(model)
    for p_ in lora_p:
        p_.requires_grad = True
    n_train_p = sum(p.numel() for p in lora_p)
    print(f"[finetune] trainable params: {n_train_p:,}")

    optimiser = torch.optim.AdamW(lora_p, lr=args.lr, weight_decay=0.0)
    total_steps = len(train_loader) * args.epochs // max(args.grad_accum, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr,
        total_steps=max(total_steps, 1),
        pct_start=0.05, anneal_strategy="cos",
        final_div_factor=args.lr / 1e-6,
    )

    # ----- Training loop -----
    metrics_path = run_dir / "metrics.jsonl"
    metrics_fp = open(metrics_path, "a")
    metrics_fp.write(json.dumps({
        "type": "config",
        "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
        "epochs": args.epochs, "grad_accum": args.grad_accum,
        "size": args.size, "device": device, "n_train": len(train_ds),
    }) + "\n")
    metrics_fp.flush()

    step = 0
    best_val = float("inf")
    best_lora_state = None
    patience = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        train_full.current_epoch = epoch
        eval_full.current_epoch = 0
        stage = _stage_for_epoch(epoch)
        print(f"[finetune] === epoch {epoch} === curriculum: scale=[{stage.scale_lo:.2f},{stage.scale_hi:.2f}] rot=±{stage.rot_max_deg:.0f} trans=±{stage.trans_max_frac:.2f}")
        model.train()
        accum_count = 0
        optimiser.zero_grad()
        for batch in train_loader:
            batch["refs"] = batch["refs"].to(device)
            batch["srcs"] = batch["srcs"].to(device)
            preds = _forward_train(model, batch["refs"], batch["srcs"], device)
            loss, kind_losses = _compute_loss(preds, batch, args.conf_weight, device)
            loss = loss / max(args.grad_accum, 1)
            loss.backward()
            accum_count += 1

            if accum_count >= args.grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(lora_p, args.grad_clip)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()
                accum_count = 0
                step += 1

                step_record = {
                    "type": "step",
                    "epoch": epoch, "step": step,
                    "loss": float(loss * args.grad_accum),
                    "grad_norm": float(grad_norm),
                    "lr": optimiser.param_groups[0]["lr"],
                }
                for k, vs in kind_losses.items():
                    step_record[f"{k}_loss"] = float(np.mean(vs)) if vs else None
                metrics_fp.write(json.dumps(step_record) + "\n")
                metrics_fp.flush()

                if step % args.val_every == 0:
                    val = _eval_loss(model, val_loader, device, args.conf_weight)
                    elapsed = time.time() - t0
                    parts = [f"val={val['loss']:.4f}"]
                    for k in ("corona_synth", "kh-modern", "kh-kh", "modern-modern"):
                        v = val.get(f"{k}_loss")
                        if v is not None and not (isinstance(v, float) and v != v):
                            parts.append(f"{k}={v:.4f}")
                    print(f"[finetune] step {step:5d} | epoch {epoch} | {' | '.join(parts)} | elapsed={elapsed:.0f}s")
                    metrics_fp.write(json.dumps({"type": "val", "step": step, **val}) + "\n")
                    metrics_fp.flush()
                    if val["loss"] < best_val:
                        best_val = val["loss"]
                        from align.romav2.lora import extract_lora_state_dict
                        best_lora_state = extract_lora_state_dict(model)
                        patience = 0
                    else:
                        patience += 1
                        if patience >= args.early_stop_patience:
                            print(f"[finetune] early stop at step {step} (best val={best_val:.4f})")
                            break
        if patience >= args.early_stop_patience:
            break

    # Final val if we didn't already
    if best_lora_state is None:
        from align.romav2.lora import extract_lora_state_dict
        best_lora_state = extract_lora_state_dict(model)

    # Restore best LoRA state and merge
    from align.romav2.lora import load_lora_state_dict
    load_lora_state_dict(model, best_lora_state)
    save_merged_weights(model, args.output_weights)
    print(f"[finetune] saved merged weights to {args.output_weights}")

    metrics_fp.write(json.dumps({"type": "done", "best_val": best_val, "wall_clock_s": time.time() - t0}) + "\n")
    metrics_fp.close()
    print(f"[finetune] done in {time.time() - t0:.0f}s; best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
