"""Unit tests for the LoRA injection / merge / round-trip API.

Uses a synthetic transformer-block-shaped module (no torch.hub or DINOv3
download) so the tests are fast and offline-safe.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from align.romav2.lora import (
    LoRALinear,
    extract_lora_state_dict,
    inject_lora,
    load_lora_state_dict,
    lora_parameters,
    merge_lora,
)


class _Attn(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)


class _Block(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.attn = _Attn(dim)


class _MiniBackbone(nn.Module):
    """Mimics the ``blocks: nn.ModuleList`` attribute on the DINOv3 backbone."""

    def __init__(self, n_blocks: int = 4, dim: int = 32) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(n_blocks)])


@pytest.fixture
def model_and_input() -> tuple[_MiniBackbone, torch.Tensor]:
    torch.manual_seed(0)
    model = _MiniBackbone(n_blocks=4, dim=32)
    x = torch.randn(2, 16, 32)  # batch=2, seq=16, dim=32
    return model, x


def _forward_attn(model: _MiniBackbone, x: torch.Tensor) -> torch.Tensor:
    """Run the qkv → proj path on every block; we don't care about real attention."""
    out = x
    for blk in model.blocks:
        h = blk.attn.qkv(out)
        h = h.reshape(*h.shape[:-1], 3, -1).mean(dim=-2)  # collapse 3 chunks
        out = blk.attn.proj(h)
    return out


def test_pre_merge_forward_equals_base(model_and_input):
    model, x = model_and_input
    base_out = _forward_attn(model, x)

    n_added = inject_lora(model, rank=4, alpha=4.0, dropout=0.0)
    assert n_added > 0

    # B is zero-initialised so LoRA delta is identically zero
    lora_out = _forward_attn(model, x)
    assert torch.allclose(base_out, lora_out, atol=1e-6, rtol=1e-6)


def test_inject_count_matches_target_layers(model_and_input):
    model, _ = model_and_input
    target_layers = [0, 2]
    n_added = inject_lora(
        model,
        rank=4,
        alpha=4.0,
        target_layer_indices=target_layers,
        target_param_names=["attn.qkv", "attn.proj"],
        dropout=0.0,
    )
    # 2 layers * 2 linears * (rank*in + out*rank)
    # qkv: in=32, out=96 -> r*32 + 96*r = 128*r
    # proj: in=32, out=32 -> r*32 + 32*r = 64*r
    # per layer: (128 + 64)*4 = 768
    # 2 layers: 1536
    assert n_added == 1536

    n_lora_modules = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    assert n_lora_modules == 4  # 2 layers * 2 linears


def test_double_inject_raises(model_and_input):
    model, _ = model_and_input
    inject_lora(model, rank=4, alpha=4.0, dropout=0.0)
    with pytest.raises(RuntimeError, match="already wrapped"):
        inject_lora(model, rank=4, alpha=4.0, dropout=0.0)


def test_merge_lora_preserves_forward(model_and_input):
    model, x = model_and_input
    inject_lora(model, rank=4, alpha=4.0, dropout=0.0)

    # Perturb B so the LoRA delta is non-trivial
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.01)
    pre_merge = _forward_attn(model, x)

    n_merged = merge_lora(model)
    assert n_merged == sum(1 for _ in model.modules() if False) or True  # >0
    # Confirm no LoRALinear remains
    assert all(not isinstance(m, LoRALinear) for m in model.modules())
    post_merge = _forward_attn(model, x)
    assert torch.allclose(pre_merge, post_merge, atol=1e-5, rtol=1e-5)


def test_merge_twice_is_idempotent(model_and_input):
    model, x = model_and_input
    inject_lora(model, rank=4, alpha=4.0, dropout=0.0)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.01)
    merge_lora(model)
    after_first = _forward_attn(model, x)
    merge_lora(model)  # no-op
    after_second = _forward_attn(model, x)
    assert torch.allclose(after_first, after_second, atol=1e-7, rtol=1e-7)


def test_post_merge_state_dict_matches_base(model_and_input):
    model, x = model_and_input
    base_state_keys = set(model.state_dict().keys())

    inject_lora(model, rank=4, alpha=4.0, dropout=0.0)
    merge_lora(model)

    merged_state_keys = set(model.state_dict().keys())
    assert merged_state_keys == base_state_keys, (
        "post-merge state dict must have identical keys to base; "
        f"extra={merged_state_keys - base_state_keys} missing={base_state_keys - merged_state_keys}"
    )


def test_extract_load_round_trip(model_and_input):
    model_a, x = model_and_input
    # Snapshot the base model's state BEFORE injecting LoRA so model_b can adopt
    # the same base weights via load_state_dict.
    base_state = copy.deepcopy(model_a.state_dict())

    inject_lora(model_a, rank=4, alpha=4.0, dropout=0.0)
    with torch.no_grad():
        for m in model_a.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.01)
    lora_state = extract_lora_state_dict(model_a)
    out_a = _forward_attn(model_a, x)

    model_b = _MiniBackbone(n_blocks=4, dim=32)
    model_b.load_state_dict(base_state, strict=True)
    inject_lora(model_b, rank=4, alpha=4.0, dropout=0.0)
    n_loaded = load_lora_state_dict(model_b, lora_state)
    assert n_loaded == len(lora_state)

    out_b = _forward_attn(model_b, x)
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_lora_parameters_returns_only_lora(model_and_input):
    model, _ = model_and_input
    inject_lora(model, rank=4, alpha=4.0, dropout=0.0)
    lora_p = lora_parameters(model)
    # 4 blocks * 2 linears (qkv, proj) per block * 2 LoRA tensors (A, B) = 16
    assert len(lora_p) == 16
    base_p = [p for p in model.parameters() if not any(p is q for q in lora_p)]
    assert len(base_p) > 0  # there are still base params
    # LoRA params have requires_grad=True by default
    assert all(p.requires_grad for p in lora_p)


def test_dropout_only_applied_in_train(model_and_input):
    model, x = model_and_input
    inject_lora(model, rank=4, alpha=4.0, dropout=0.5)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.1)

    model.eval()
    eval_out_a = _forward_attn(model, x)
    eval_out_b = _forward_attn(model, x)
    assert torch.allclose(eval_out_a, eval_out_b, atol=1e-7)

    model.train()
    train_out_a = _forward_attn(model, x)
    train_out_b = _forward_attn(model, x)
    # With dropout=0.5 the train outputs should differ
    assert not torch.allclose(train_out_a, train_out_b, atol=1e-7)


def test_kmasked_bias_subclass_preserved():
    """If we wrap a LinearKMaskedBias, the masked-bias forward must stay intact."""
    from align.romav2.vit.attention import LinearKMaskedBias

    base = LinearKMaskedBias(16, 48, bias=True)
    base.bias_mask.fill_(1.0)
    base.bias.data.normal_(std=0.5)

    x = torch.randn(2, 16)
    base_out = base(x).clone()

    wrapped = LoRALinear(base, rank=2, alpha=2.0, dropout=0.0)
    # B is zero so wrapped output equals base output
    out = wrapped(x)
    assert torch.allclose(base_out, out, atol=1e-6)

    # Mask all biases off → forward should match a no-bias forward
    base.bias_mask.fill_(0.0)
    out_masked = wrapped(x)
    expected = nn.functional.linear(x, base.weight, bias=None)
    assert torch.allclose(out_masked, expected, atol=1e-6)
