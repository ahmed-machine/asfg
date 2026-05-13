"""LoRA adapters for the DINOv3 backbone inside RoMa v2.

Targets the ``attn.qkv`` and ``attn.proj`` linears in DINOv3 ViT-L/16 blocks.
The wrapper composes the base linear (preserving subclass behaviour such as
``LinearKMaskedBias``) and adds a rank-r residual ``B @ A`` scaled by ``alpha/r``.
At init B is zero so the wrapped module is functionally identical to the base.

After training, ``merge_lora(model)`` folds the residual back into the base
weights in place and removes the wrapper so the resulting state dict is the
same shape as the unmodified RoMa v2.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA delta wrapping an existing ``nn.Linear``.

    Forward: ``base(x) + dropout(x) @ A.T @ B.T * (alpha/r)``.

    LoRA parameters are stored in fp32 even when the base linear's weight is
    bf16/fp16; the addition is dtype-cast to match ``base_out``.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear base, got {type(base)}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)

        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_dr = self.lora_dropout(x)
        # Keep LoRA mathematics in fp32 for stability
        x32 = x_dr.to(self.lora_A.dtype)
        delta = F.linear(F.linear(x32, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta.to(base_out.dtype)

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        """Fold the LoRA delta into ``self.base.weight`` in place and return base."""
        delta_w = (self.lora_B.to(torch.float32) @ self.lora_A.to(torch.float32)) * self.scaling
        self.base.weight.data.add_(delta_w.to(self.base.weight.dtype))
        return self.base


def _resolve_blocks(model: nn.Module) -> nn.ModuleList:
    """Find the transformer ``blocks`` attribute on a backbone or wrapped model."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        f = getattr(model, "f", None)
        blocks = getattr(f, "blocks", None) if f is not None else None
    if blocks is None:
        raise ValueError("Could not find .blocks attribute on model or model.f")
    return blocks


def _set_path(parent: nn.Module, dotted: str, value: nn.Module) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], value)


def _get_path(parent: nn.Module, dotted: str) -> nn.Module:
    for p in dotted.split("."):
        parent = getattr(parent, p)
    return parent


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    target_layer_indices: Iterable[int] | None = None,
    target_param_names: Iterable[str] = ("attn.qkv", "attn.proj"),
    dropout: float = 0.05,
) -> int:
    """Walk transformer blocks and wrap each named linear with ``LoRALinear``.

    Returns the number of LoRA parameters added (sum of A + B sizes across all
    wrapped linears). Idempotent only if no wrapping has been done previously;
    re-injection over already-wrapped linears raises.
    """
    blocks = _resolve_blocks(model)
    if target_layer_indices is None:
        target_layer_indices = range(len(blocks))
    target_layer_indices = list(target_layer_indices)
    target_param_names = list(target_param_names)

    n_added = 0
    for idx in target_layer_indices:
        if idx < 0 or idx >= len(blocks):
            continue
        block = blocks[idx]
        for name in target_param_names:
            base = _get_path(block, name)
            if isinstance(base, LoRALinear):
                raise RuntimeError(
                    f"Block {idx}.{name} is already wrapped by LoRALinear; "
                    "call merge_lora before re-injecting"
                )
            if not isinstance(base, nn.Linear):
                raise TypeError(
                    f"Block {idx}.{name} is {type(base).__name__}, expected nn.Linear"
                )
            wrapped = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
            wrapped.to(base.weight.device)
            _set_path(block, name, wrapped)
            n_added += wrapped.lora_A.numel() + wrapped.lora_B.numel()
    return n_added


def merge_lora(model: nn.Module) -> int:
    """In place: fold every ``LoRALinear`` delta into its base, then unwrap.

    Returns the number of wrappers that were merged.
    """
    n_merged = 0
    # Find every LoRALinear and the (parent_module, attr_name) needed to swap it.
    swaps: list[tuple[nn.Module, str, LoRALinear]] = []
    for parent in model.modules():
        for attr_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                swaps.append((parent, attr_name, child))

    for parent, attr_name, wrapped in swaps:
        merged = wrapped.merge_into_base()
        setattr(parent, attr_name, merged)
        n_merged += 1
    return n_merged


def extract_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only the LoRA parameter tensors keyed by their fully-qualified name."""
    out: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[f"{name}.lora_A"] = module.lora_A.detach().clone()
            out[f"{name}.lora_B"] = module.lora_B.detach().clone()
    return out


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> int:
    """Copy LoRA tensors from ``state`` into matching ``LoRALinear`` modules.

    Returns the number of tensors loaded. Raises if the model doesn't have a
    matching module for a key.
    """
    name_to_module = dict(model.named_modules())
    n_loaded = 0
    for k, v in state.items():
        if not (k.endswith(".lora_A") or k.endswith(".lora_B")):
            raise KeyError(f"Unexpected key in LoRA state dict: {k}")
        mod_name, param_name = k.rsplit(".", 1)
        module = name_to_module.get(mod_name)
        if not isinstance(module, LoRALinear):
            raise KeyError(
                f"No LoRALinear at {mod_name!r} (got {type(module).__name__}); "
                "did you forget to inject_lora before loading?"
            )
        getattr(module, param_name).data.copy_(v.to(getattr(module, param_name).dtype))
        n_loaded += 1
    return n_loaded


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the list of trainable LoRA parameters across the model."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params
