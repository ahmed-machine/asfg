"""Torch device detection and shared model cache."""

from __future__ import annotations

import os

import torch


def get_torch_device(override: str | None = None) -> str:
    """Auto-detect best available torch device."""

    if override and override != "auto":
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ELoFTRWrapper:
    """Wraps HuggingFace EfficientLoFTR to match the dict-based API.

    Accepts: {"image0": [B, 1, H, W], "image1": [B, 1, H, W]}
    Returns: {"mkpts0_f": [M, 2], "mkpts1_f": [M, 2], "mconf": [M], "m_bids": [M]}
    """

    def __init__(self, hf_model, device: str):
        self.model = hf_model
        self.device = device
        self.training = False

    def _forward_single(self, img0_single, img1_single):
        """Run inference on a single pair [1, 1, H, W] each."""
        _, _, H, W = img0_single.shape
        pixel_values = torch.stack([img0_single, img1_single], dim=1)  # [1, 2, 1, H, W]

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        kpts = outputs.keypoints[0]            # [2, N, 2]
        matches = outputs.matches[0]           # [2, N]
        scores = outputs.matching_scores[0]    # [2, N]

        kpts_px = kpts.clone()
        kpts_px[:, :, 0] *= W
        kpts_px[:, :, 1] *= H

        valid = torch.logical_and(scores[0] > 0, matches[0] > -1)
        if valid.any():
            mkpts0 = kpts_px[0][valid]
            matched_idx = matches[0][valid].long()
            mkpts1 = kpts_px[1][matched_idx]
            mconf = scores[0][valid]
            return mkpts0, mkpts1, mconf
        return None

    def __call__(self, batch: dict) -> dict:
        img0 = batch["image0"]  # [B, 1, H, W]
        img1 = batch["image1"]  # [B, 1, H, W]
        B = img0.shape[0]

        # Process pairs one-by-one (HF EfficientLoFTR has a batch bug
        # where fine-matching keypoints get corrupted at B>1)
        all_mkpts0 = []
        all_mkpts1 = []
        all_mconf = []
        all_bids = []

        for b in range(B):
            result = self._forward_single(img0[b:b+1], img1[b:b+1])
            if result is not None:
                mkpts0, mkpts1, mconf = result
                all_mkpts0.append(mkpts0)
                all_mkpts1.append(mkpts1)
                all_mconf.append(mconf)
                all_bids.append(torch.full((mkpts0.shape[0],), b,
                                           dtype=torch.long, device=mkpts0.device))

        if all_mkpts0:
            return {
                "mkpts0_f": torch.cat(all_mkpts0, dim=0),
                "mkpts1_f": torch.cat(all_mkpts1, dim=0),
                "mconf": torch.cat(all_mconf, dim=0),
                "m_bids": torch.cat(all_bids, dim=0),
            }
        else:
            dev = img0.device
            return {
                "mkpts0_f": torch.zeros((0, 2), device=dev),
                "mkpts1_f": torch.zeros((0, 2), device=dev),
                "mconf": torch.zeros((0,), device=dev),
                "m_bids": torch.zeros((0,), dtype=torch.long, device=dev),
            }

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        self.device = str(device)
        return self


class ModelCache:
    """Lazy-loaded cache for neural models."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._eloftr = None
        self._roma = None

    @property
    def roma(self):
        if self._roma is None:
            torch.set_float32_matmul_precision("highest")
            from .romav2 import RoMaV2
            from .romav2.device import set_device
            set_device(self.device)
            print(f"  [ModelCache] Loading RoMa v2 on {self.device}")
            cfg = RoMaV2.Cfg(setting="fast", compile=False)
            self._roma = RoMaV2(cfg)
            print(f"  [ModelCache] RoMa v2 ready")
        return self._roma

    @property
    def eloftr(self):
        if self._eloftr is None:
            from transformers import AutoModelForKeypointMatching

            print(f"  [ModelCache] Loading MatchAnything-ELoFTR on {self.device}")
            hf_model = AutoModelForKeypointMatching.from_pretrained(
                "zju-community/matchanything_eloftr"
            )
            hf_model.eval().to(self.device)
            self._eloftr = ELoFTRWrapper(hf_model, self.device)
            print(f"  [ModelCache] MatchAnything-ELoFTR ready")
        return self._eloftr

    def close(self) -> None:
        self._eloftr = None
        self._roma = None
        if self.device != "cpu":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
