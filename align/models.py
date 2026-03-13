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


class ModelCache:
    """Lazy-loaded cache for neural models."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._loftr = None
        self._roma = None
        self._eloftr = None

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
            from .eloftr.config.default import get_cfg_defaults
            from .eloftr.loftr import LoFTR
            from .eloftr.utils.misc import lower_config

            print(f"  [ModelCache] Loading EfficientLoFTR on {self.device}")
            cfg = get_cfg_defaults()
            cfg.LOFTR.COARSE.NPE = [832, 832, 1024, 1024]
            model = LoFTR(config=lower_config(cfg.LOFTR))

            ckpt_name = "eloftr_outdoor_modified.ckpt"
            ckpt_url = (
                "https://huggingface.co/kornia/Efficient_LOFTR/resolve/main/"
                f"{ckpt_name}"
            )
            hub_dir = torch.hub.get_dir()
            ckpt_path = os.path.join(hub_dir, "checkpoints", ckpt_name)

            if not os.path.exists(ckpt_path):
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                print("  [ModelCache] Downloading EfficientLoFTR weights...")
                torch.hub.download_url_to_file(ckpt_url, ckpt_path)

            try:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except Exception:
                print("  [ModelCache] Standard load failed, attempting safe weights-only extract...")
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            clean_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key.replace("matcher.", "") if key.startswith("matcher.") else key
                clean_state_dict[clean_key] = value

            model.load_state_dict(clean_state_dict, strict=False)
            self._eloftr = model.eval().to(self.device)
        return self._eloftr

    @property
    def loftr(self):
        if self._loftr is None:
            import kornia.feature as KF

            print(f"  [ModelCache] Loading LoFTR on {self.device}")
            model = KF.LoFTR(pretrained="outdoor").eval().to(self.device)
            try:
                if hasattr(torch, "compile") and self.device == "cuda":
                    self._loftr = torch.compile(model)
                else:
                    self._loftr = model
            except Exception:
                self._loftr = model
        return self._loftr

    def close(self) -> None:
        self._loftr = None
        self._roma = None
        self._eloftr = None
        if self.device != "cpu":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
