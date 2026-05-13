"""I/O helpers around LoRA-merged RoMa v2 weights.

These wrap ``merge_lora`` + ``torch.save`` and a load-side equivalent so the
training script and eval scripts can persist the merged checkpoint without
duplicating the boilerplate. The on-disk format is a plain ``torch.save`` of
the full RoMa v2 state dict — same shapes as base RoMa v2 — so the production
``DECLASS_ROMA_WEIGHTS`` env-var gate can ``load_state_dict(strict=True)``
the file directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from .lora import merge_lora


def save_merged_weights(model: nn.Module, path: str | Path) -> int:
    """Merge LoRA into base in-place, then save the full state dict.

    Returns the number of LoRA layers merged. Creates parent directories.
    The merge mutates ``model`` — callers that need the unmerged model for
    further training should ``deepcopy`` before calling this.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_merged = merge_lora(model)
    torch.save(model.state_dict(), path)
    return n_merged


def model_with_merged_weights(weights_path: str | Path, device: str | None = None) -> nn.Module:
    """Construct a fresh RoMa v2 (fast setting) and load merged weights.

    Useful in eval scripts that want to instantiate a frozen merged model
    without going through ``ModelCache``.
    """
    from .romav2 import RoMaV2  # local import to avoid circular at module load
    from .device import set_device

    if device is not None:
        set_device(device)

    cfg = RoMaV2.Cfg(setting="fast", compile=False)
    model = RoMaV2(cfg)
    state = torch.load(weights_path, map_location=device or "cpu")
    model.load_state_dict(state, strict=True)
    return model
