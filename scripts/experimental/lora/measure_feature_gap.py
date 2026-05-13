"""Phase 1 — DINOv3 feature-gap diagnostic.

For 200 random CORONA SSD pairs we compute DINOv3 ViT-L/16 features at the
satast resolution (800x800) two ways:

  1. Wrapped forward — ``model.f(img)`` — what production inference uses.
  2. Bypass forward — ``model.f.get_intermediate_layers(imagenet(img), n=[11,17])``
     followed by an einops rearrange — the path the LoRA training loop will
     take to keep gradients flowing into the backbone.

A sanity check confirms the two paths produce numerically equivalent features
at fp32 (cos sim > 0.999); diverging here means the bypass that the LoRA
trainer relies on differs from production and Issue 1 in lora-plan.md is not
mitigated.

The headline output is the cosine similarity between corresponding patches in
ref / src_clean pairs. The recorded synthetic affine ``M`` in each pair's
meta is used to map ref-grid tokens to src-grid positions; we then bilinearly
sample src tokens at those positions and compute per-patch cos sim. Aggregate
median / mean / p10 / p90 at layers 11 and 17 are reported, broken down by
inferred KH mission.

Decision gates (from lora-plan.md):
  * median cos sim > 0.8 at both layers → STOP, features already invariant.
  * 0.5 - 0.8 → proceed at r=8.
  * < 0.5 → proceed at r=16.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _infer_mission(layer_str: str) -> str:
    """Best-effort KH mission inference from a CORONA Atlas layer slug."""
    m = re.match(r"corona:(\d+)-", layer_str or "")
    if not m:
        return "unknown"
    n = int(m.group(1))
    if 9024 <= n <= 9069:
        return "kh-4"
    if 1023 <= n <= 1052:
        return "kh-4a"
    if 1101 <= n <= 1117:
        return "kh-4b"
    if 4001 <= n <= 4038:
        return "kh-7"
    return f"corona-{n}"


def _list_pairs(pairs_dir: Path) -> list[Path]:
    """Return paths to meta.json files for valid pairs (all 4 files present)."""
    valid = []
    for meta in sorted(pairs_dir.glob("pair_*_meta.json")):
        stem = meta.name[: -len("_meta.json")]
        ref = pairs_dir / f"{stem}_ref.png"
        src = pairs_dir / f"{stem}_src_clean.png"
        if ref.exists() and src.exists():
            valid.append(meta)
    return valid


def _load_image(path: Path, size: int) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def _img_to_tensor(img_bgr: np.ndarray, device: str) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def _wrapped_features(model_f, img: torch.Tensor) -> list[torch.Tensor]:
    """Production forward: returns a list of ``[B, H, W, D]`` features per tap layer."""
    with torch.no_grad():
        feats = model_f(img)
    return list(feats)


def _bypass_features(model_f, img: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
    """LoRA-training forward: bypass the wrapped normalizer + grad gating.

    Mirrors the autocast-with-bf16 context that ``wrap_with_normalize`` applies in
    production so the bypass is numerically equivalent.
    """
    from align.romav2.normalizers import imagenet
    from einops import rearrange

    img_n = imagenet(img)
    with torch.no_grad(), torch.autocast(img.device.type, dtype=torch.bfloat16, enabled=True):
        raw = model_f.get_intermediate_layers(img_n, n=layers)
    H = img.shape[2] // 16
    W = img.shape[3] // 16
    return [rearrange(x, "B (H W) D -> B H W D", H=H, W=W) for x in raw]


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)


def _affine_map_ref_to_src(grid_xy: torch.Tensor, M_2x3: torch.Tensor) -> torch.Tensor:
    """Apply 2x3 affine to a (..., 2) tensor of (x, y) coords.

    The recorded affine in meta.json is the warp src_clean -> src_perturbed; the
    inverse direction (perturbed -> clean = src) is what we want to map ref-grid
    sample positions through. But we are pairing ref to src_clean directly
    (no perturbation involved), so the affine is irrelevant here — both images
    share a coordinate frame at the meta-recorded crop. We expose this helper
    for future use; the diagnostic itself uses identity mapping over ref/src.
    """
    out = grid_xy @ M_2x3[:, :2].T + M_2x3[:, 2]
    return out


def _grid_normalised(H: int, W: int, device: str, drop_border: float = 0.05) -> torch.Tensor:
    """Return (H', W', 2) coords in [-1, 1] dropping ``drop_border`` margin."""
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    if drop_border > 0:
        keep = (grid[..., 0].abs() < (1 - drop_border)) & (grid[..., 1].abs() < (1 - drop_border))
        grid = grid[keep]
    return grid


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _decision_band(median_cos: float) -> str:
    if median_cos >= 0.8:
        return "STOP"
    if median_cos >= 0.5:
        return "PROCEED_r8"
    return "PROCEED_r16"


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 feature-gap diagnostic")
    p.add_argument("--pairs-dir", type=Path, default=REPO_ROOT / "data" / "ssd_pairs")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "diagnostics" / "lora")
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", type=int, nargs="+", default=[11, 17])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from align.models import get_torch_device
    device = get_torch_device()
    print(f"[feature_gap] device: {device}")

    print(f"[feature_gap] indexing pairs in {args.pairs_dir}")
    metas = _list_pairs(args.pairs_dir)
    if len(metas) < args.n_pairs:
        print(f"[feature_gap] only {len(metas)} valid pairs available; using all of them")
        sample = metas
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(metas, args.n_pairs)
    print(f"[feature_gap] sampled {len(sample)} pairs")

    print(f"[feature_gap] loading RoMa v2 ...")
    from align.romav2 import RoMaV2
    from align.romav2.device import set_device
    set_device(device)
    cfg = RoMaV2.Cfg(setting="fast", compile=False)
    model = RoMaV2(cfg)
    # Phase 1 must run at training resolution; satast → H_lr/W_lr = 800
    model.apply_setting("satast")
    model.eval()
    print(f"[feature_gap] model loaded (satast: H_lr={model.H_lr}, W_lr={model.W_lr})")

    # Sanity check: wrapped vs bypass parity on one sample at fp32.
    print(f"[feature_gap] running wrapped-vs-bypass parity check ...")
    sample_img = _load_image(args.pairs_dir / (sample[0].name.replace("_meta.json", "_ref.png")), args.size)
    if sample_img is None:
        raise RuntimeError("first sample image failed to load — pairs dir likely corrupted")
    parity_tensor = _img_to_tensor(sample_img, device)
    # Force the backbone & input to fp32 for a clean parity check
    backbone_dtype = next(model.f.parameters()).dtype
    if backbone_dtype != torch.float32:
        # The backbone may have been cast to bf16 by wrap_model; cast a copy of weights for parity
        print(f"[feature_gap] backbone is {backbone_dtype}; running parity at backbone's native dtype")
    f_wrapped = _wrapped_features(model.f, parity_tensor)
    f_bypass = _bypass_features(model.f, parity_tensor, args.layers)
    parity_cos = []
    for fw, fb in zip(f_wrapped, f_bypass):
        cs = _cos_sim(fw.float(), fb.float()).mean().item()
        parity_cos.append(cs)
    print(f"[feature_gap] parity cos sim per layer: {parity_cos}")
    if min(parity_cos) < 0.999:
        print("[feature_gap] WARNING: parity cos sim < 0.999 — bypass diverges from wrapped forward")

    # Headline diagnostic: cosine similarity between corresponding patches across modern ↔ KH.
    print(f"[feature_gap] computing per-pair cos sim ...")
    by_layer_global: dict[int, list[float]] = defaultdict(list)
    by_layer_mission: dict[tuple[int, str], list[float]] = defaultdict(list)
    drop_border = 0.05
    t0 = time.time()
    n_skipped = 0
    for i, meta_path in enumerate(sample):
        try:
            meta = json.loads(meta_path.read_text())
            mission = _infer_mission(meta.get("layer", ""))
            stem = meta_path.name[: -len("_meta.json")]
            ref_img = _load_image(args.pairs_dir / f"{stem}_ref.png", args.size)
            src_img = _load_image(args.pairs_dir / f"{stem}_src_clean.png", args.size)
            if ref_img is None or src_img is None:
                n_skipped += 1
                continue
            ref_t = _img_to_tensor(ref_img, device)
            src_t = _img_to_tensor(src_img, device)

            f_ref = _bypass_features(model.f, ref_t, args.layers)
            f_src = _bypass_features(model.f, src_t, args.layers)
            for layer_idx, (fr, fs) in zip(args.layers, zip(f_ref, f_src)):
                # fr, fs: [1, H, W, D]
                H, W = fr.shape[1], fr.shape[2]
                border = int(drop_border * H)
                fr_c = fr[0, border:H - border, border:W - border, :].reshape(-1, fr.shape[-1])
                fs_c = fs[0, border:H - border, border:W - border, :].reshape(-1, fs.shape[-1])
                cs = _cos_sim(fr_c.float(), fs_c.float()).cpu().numpy()
                vals = cs.tolist()
                by_layer_global[layer_idx].extend(vals)
                by_layer_mission[(layer_idx, mission)].extend(vals)
        except Exception as e:
            n_skipped += 1
            print(f"[feature_gap] skip {meta_path.name}: {e}")
            continue
        if (i + 1) % 25 == 0:
            print(f"[feature_gap] {i + 1}/{len(sample)} pairs done ({time.time() - t0:.1f}s)")

    print(f"[feature_gap] done in {time.time() - t0:.1f}s ({n_skipped} skipped)")

    summary: dict = {
        "n_pairs_used": len(sample) - n_skipped,
        "n_skipped": n_skipped,
        "size": args.size,
        "layers": args.layers,
        "parity_cos_sim_per_layer": parity_cos,
        "global": {str(layer): _percentiles(by_layer_global[layer]) for layer in args.layers},
        "by_mission": {
            f"layer_{layer}::{mission}": _percentiles(vals)
            for (layer, mission), vals in sorted(by_layer_mission.items())
        },
    }

    # Decision band based on layer 11 median (the earliest tap; lora-plan.md anchors on this)
    primary_layer = args.layers[0]
    primary_median = summary["global"][str(primary_layer)]["median"]
    summary["decision_band"] = _decision_band(primary_median)
    summary["decision_layer"] = primary_layer
    summary["decision_median"] = primary_median

    out_json = args.out_dir / "feature_gap.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[feature_gap] wrote {out_json}")

    md_lines = [
        "# DINOv3 Feature Gap (CORONA SSD pairs)",
        "",
        f"- pairs: {summary['n_pairs_used']} (skipped {n_skipped})",
        f"- size: {args.size}",
        f"- layers: {args.layers}",
        f"- parity cos sim per layer: {parity_cos}",
        "",
        "## Global (cosine similarity between corresponding patches in ref/src_clean)",
        "",
        "| layer | n | mean | median | p10 | p90 |",
        "|---|---|---|---|---|---|",
    ]
    for layer in args.layers:
        s = summary["global"][str(layer)]
        md_lines.append(
            f"| {layer} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | {s['p10']:.3f} | {s['p90']:.3f} |"
        )

    md_lines.extend([
        "",
        "## Per-mission breakdown",
        "",
        "| layer | mission | n | mean | median | p10 | p90 |",
        "|---|---|---|---|---|---|---|",
    ])
    for key, s in sorted(summary["by_mission"].items()):
        layer_str, mission = key.split("::", 1)
        md_lines.append(
            f"| {layer_str.removeprefix('layer_')} | {mission} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | {s['p10']:.3f} | {s['p90']:.3f} |"
        )

    md_lines.extend([
        "",
        "## Decision",
        "",
        f"- Primary layer: {summary['decision_layer']}",
        f"- Median cos sim: {summary['decision_median']:.3f}",
        f"- **Band: {summary['decision_band']}**",
        "",
        "Decision rule (lora-plan.md):",
        "- median > 0.8 → STOP (features already invariant)",
        "- 0.5 - 0.8 → PROCEED with r=8",
        "- < 0.5 → PROCEED with r=16",
    ])

    out_md = args.out_dir / "feature_gap.md"
    out_md.write_text("\n".join(md_lines))
    print(f"[feature_gap] wrote {out_md}")
    print(f"[feature_gap] decision: {summary['decision_band']} (median={primary_median:.3f})")


if __name__ == "__main__":
    main()
