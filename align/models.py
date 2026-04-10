"""Torch device detection and shared model cache."""

from __future__ import annotations

import gc
import os

import torch


def use_ssd_weights() -> bool:
    """Return True when finetuned SSD weights should override base models.

    SSD (self-supervised distillation) is an experimental research thread
    (see hypothesis.md Section 7 and scripts/experimental/ssd/). Downstream
    alignment improvement has not been validated, so weight loading is
    default-off and requires an explicit opt-in via the
    DECLASS_EXPERIMENTAL_SSD environment variable.
    """
    value = os.environ.get("DECLASS_EXPERIMENTAL_SSD", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_torch_device(override: str | None = None) -> str:
    """Auto-detect best available torch device."""

    if override and override != "auto":
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_torch_cache(device: str | None = None) -> None:
    """Release cached allocations for the active torch device."""

    target = device or get_torch_device()
    if target == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif target == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


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

    def extract_backbone_features(self, img_gray: torch.Tensor) -> torch.Tensor:
        """Extract stride-8 backbone features from a grayscale image.

        Args:
            img_gray: (B, 1, H, W) tensor, float32, range [0, 1].

        Returns:
            (B, 256, H//8, W//8) tensor of coarse backbone features.
        """
        with torch.no_grad():
            features = self.model.efficientloftr.backbone(img_gray)
            # features[2] is the coarse feature map: 256ch @ stride 8
            return features[2]

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
            
            # Experimental SSD-finetuned override (see scripts/experimental/ssd/).
            # Opt-in via DECLASS_EXPERIMENTAL_SSD=1. Default off — downstream
            # alignment improvement has not been validated.
            if use_ssd_weights():
                ssd_weights_path = 'align/weights/roma_ssd.pth'
                if os.path.exists(ssd_weights_path):
                    print(f"  [ModelCache] OVERRIDE: Loading SSD-finetuned RoMa weights from {ssd_weights_path}")
                    self._roma.load_state_dict(torch.load(ssd_weights_path, map_location=self.device))
                else:
                    print(f"  [ModelCache] DECLASS_EXPERIMENTAL_SSD set but {ssd_weights_path} not found; using base RoMa weights")
                
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
            # Satellite-tune: lower coarse matching threshold for cross-temporal imagery
            # Default 0.2 is too strict — many valid matches land in 0.1-0.2 range
            hf_model.config.coarse_matching_threshold = 0.1
            self._eloftr = ELoFTRWrapper(hf_model, self.device)
            print(f"  [ModelCache] MatchAnything-ELoFTR ready")
        return self._eloftr

    def close(self) -> None:
        eloftr = self._eloftr
        roma = self._roma
        self._eloftr = None
        self._roma = None
        if eloftr is not None:
            del eloftr
        if roma is not None:
            del roma
        gc.collect()
        clear_torch_cache(self.device)
