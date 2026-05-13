"""Phase 5.1 — Fast synthetic-pair eval for A/B-ing RoMa weights.

For 300 held-out CORONA SSD pairs we run ``RoMaV2.match()`` end-to-end and
compare predicted correspondences against the known synthetic affine ``M``
recorded in each pair's meta. The pixel error at 800×800 (satast resolution)
is the headline metric.

Two model variants are evaluated side by side: ``--weights-a`` and
``--weights-b``. Either may be the literal string ``"base"`` (default RoMa
v2 pretrained), or a path to a merged checkpoint produced by
``align/romav2/lora_io.py::save_merged_weights`` (i.e. an
``align/weights/roma_*.pth`` file).

Acceptance gate (lora-plan.md §5.1):
  * ``median_err(b) <= 1.05 * median_err(a)`` AND
  * ``p90_err(b) <= p90_err(a)``
where (a) is the base and (b) is the LoRA-merged variant.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _list_pairs(pairs_dir: Path) -> list[Path]:
    valid = []
    for meta in sorted(pairs_dir.glob("pair_*_meta.json")):
        stem = meta.name[: -len("_meta.json")]
        ref = pairs_dir / f"{stem}_ref.png"
        src = pairs_dir / f"{stem}_src_perturbed.png"
        if ref.exists() and src.exists():
            valid.append(meta)
    return valid


def _load_pair(meta_path: Path, size: int, device: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    meta = json.loads(meta_path.read_text())
    stem = meta_path.name[: -len("_meta.json")]
    ref = cv2.imread(str(meta_path.parent / f"{stem}_ref.png"), cv2.IMREAD_COLOR)
    src = cv2.imread(str(meta_path.parent / f"{stem}_src_perturbed.png"), cv2.IMREAD_COLOR)
    if ref is None or src is None:
        raise RuntimeError(f"failed to load images for {stem}")
    orig_size = ref.shape[0]
    if ref.shape[0] != size or ref.shape[1] != size:
        ref = cv2.resize(ref, (size, size), interpolation=cv2.INTER_AREA)
    if src.shape[0] != size or src.shape[1] != size:
        src = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    src_t = torch.from_numpy(src_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    if "affine_matrix" not in meta:
        raise KeyError(f"meta missing 'affine_matrix': {meta_path.name}")
    M = np.asarray(meta["affine_matrix"], dtype=np.float64)
    if M.shape != (2, 3):
        raise ValueError(f"affine_matrix shape {M.shape} (expected (2, 3)) in {meta_path.name}")
    return ref_t, src_t, M, orig_size


def _build_model(weights_arg: str, device: str):
    """Construct a fresh RoMa v2 (fast then satast) and optionally load merged weights."""
    from align.romav2 import RoMaV2
    from align.romav2.device import set_device
    set_device(device)
    cfg = RoMaV2.Cfg(setting="fast", compile=False)
    model = RoMaV2(cfg)
    model.apply_setting("satast")
    if weights_arg != "base":
        state = torch.load(weights_arg, map_location=device)
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _scale_affine_to_size(M_orig: np.ndarray, orig_size: int, target_size: int) -> np.ndarray:
    """Return the 2x3 affine in target-pixel coords (linear part unchanged, translation scales)."""
    k = target_size / orig_size
    M = M_orig.copy()
    M[:, 2] = M_orig[:, 2] * k
    return M


def _evaluate(model, metas: list[Path], size: int, device: str, num_corresp: int) -> dict:
    """Run model on the test set, return per-pair errors + aggregates."""
    median_per_pair: list[float] = []
    p90_per_pair: list[float] = []
    all_errors: list[float] = []
    n_skipped = 0
    t0 = time.time()
    for i, meta_path in enumerate(metas):
        try:
            ref_t, src_t, M_orig, orig_size = _load_pair(meta_path, size, device)
            with torch.no_grad():
                preds = model.match(ref_t, src_t)
                sampled = model.sample(preds, num_corresp=num_corresp)
            m, _c = sampled[0], sampled[1]
        except Exception as e:
            print(f"[eval_synthetic] skip {meta_path.name}: {e}")
            n_skipped += 1
            continue

        # m: (N, 4) in normalised [-1, 1]; cols 0..1 = ref, cols 2..3 = src
        m_np = m.detach().cpu().numpy()
        # Convert to size×size pixel coords
        # x_pix = (x_norm + 1) * (size - 1) / 2
        ref_pix = (m_np[:, :2] + 1.0) * (size - 1) / 2.0
        src_pix = (m_np[:, 2:] + 1.0) * (size - 1) / 2.0

        # Apply the recorded affine (scaled to 800px coords) to ref_pix to get expected src_pix
        M_target = _scale_affine_to_size(M_orig, orig_size, size)
        ones = np.ones((ref_pix.shape[0], 1))
        ref_h = np.concatenate([ref_pix, ones], axis=1)  # (N, 3)
        expected_src_pix = ref_h @ M_target.T  # (N, 2) = M @ ref_pix

        err = np.linalg.norm(src_pix - expected_src_pix, axis=1)  # (N,)
        # Filter out boundary points
        in_bounds = (
            (ref_pix[:, 0] > 8) & (ref_pix[:, 0] < size - 8)
            & (ref_pix[:, 1] > 8) & (ref_pix[:, 1] < size - 8)
        )
        err = err[in_bounds]
        if err.size == 0:
            n_skipped += 1
            continue
        median_per_pair.append(float(np.median(err)))
        p90_per_pair.append(float(np.percentile(err, 90)))
        all_errors.extend(err.tolist())

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[eval_synthetic] {i + 1}/{len(metas)} pairs ({elapsed:.0f}s)")

    return {
        "n_pairs_used": len(metas) - n_skipped,
        "n_skipped": n_skipped,
        "wall_clock_s": time.time() - t0,
        "median_err_pair_p50": float(np.median(median_per_pair)) if median_per_pair else float("nan"),
        "median_err_pair_p90": float(np.percentile(median_per_pair, 90)) if median_per_pair else float("nan"),
        "p90_err_pair_p50": float(np.median(p90_per_pair)) if p90_per_pair else float("nan"),
        "p90_err_pair_p90": float(np.percentile(p90_per_pair, 90)) if p90_per_pair else float("nan"),
        "global_median_err": float(np.median(all_errors)) if all_errors else float("nan"),
        "global_p90_err": float(np.percentile(all_errors, 90)) if all_errors else float("nan"),
        "n_correspondences_total": len(all_errors),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 5.1 synthetic-pair A/B eval")
    p.add_argument("--pairs-dir", type=Path, default=REPO_ROOT / "data" / "ssd_pairs")
    p.add_argument("--out-path", type=Path, default=REPO_ROOT / "diagnostics" / "lora" / "eval_synthetic.json")
    p.add_argument("--n-pairs", type=int, default=300)
    p.add_argument("--size", type=int, default=800, help="Image side length (must match satast H_lr)")
    p.add_argument("--num-corresp", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights-a", type=str, default="base", help="Path to merged weights or 'base'")
    p.add_argument("--weights-b", type=str, default=str(REPO_ROOT / "align" / "weights" / "roma_ssd.pth"))
    args = p.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    metas = _list_pairs(args.pairs_dir)
    rng = random.Random(args.seed)
    if len(metas) > args.n_pairs:
        sample = rng.sample(metas, args.n_pairs)
    else:
        sample = metas

    from align.models import get_torch_device
    device = get_torch_device()
    print(f"[eval_synthetic] device: {device}")
    print(f"[eval_synthetic] {len(sample)} pairs sampled")

    print(f"[eval_synthetic] === A: {args.weights_a} ===")
    model_a = _build_model(args.weights_a, device)
    res_a = _evaluate(model_a, sample, args.size, device, args.num_corresp)
    print(f"[eval_synthetic] A: median={res_a['median_err_pair_p50']:.2f}px  p90={res_a['p90_err_pair_p50']:.2f}px")
    del model_a

    if args.weights_b == "base" or not Path(args.weights_b).exists():
        print(f"[eval_synthetic] B weights {args.weights_b!r} not present; skipping B run")
        res_b = None
    else:
        print(f"[eval_synthetic] === B: {args.weights_b} ===")
        model_b = _build_model(args.weights_b, device)
        res_b = _evaluate(model_b, sample, args.size, device, args.num_corresp)
        print(f"[eval_synthetic] B: median={res_b['median_err_pair_p50']:.2f}px  p90={res_b['p90_err_pair_p50']:.2f}px")
        del model_b

    output: dict = {
        "a_label": args.weights_a,
        "b_label": args.weights_b,
        "size": args.size,
        "num_corresp": args.num_corresp,
        "n_pairs": len(sample),
        "a": res_a,
        "b": res_b,
    }
    if res_b is not None:
        delta_median = res_b["median_err_pair_p50"] - res_a["median_err_pair_p50"]
        delta_p90 = res_b["p90_err_pair_p50"] - res_a["p90_err_pair_p50"]
        gate_median = res_b["median_err_pair_p50"] <= 1.05 * res_a["median_err_pair_p50"]
        gate_p90 = res_b["p90_err_pair_p50"] <= res_a["p90_err_pair_p50"]
        output["delta_median_px"] = delta_median
        output["delta_p90_px"] = delta_p90
        output["gate_median_pass"] = bool(gate_median)
        output["gate_p90_pass"] = bool(gate_p90)
        output["promotion_candidate"] = bool(gate_median and gate_p90)
        print(f"[eval_synthetic] Δmedian={delta_median:+.2f}px  Δp90={delta_p90:+.2f}px  promote={output['promotion_candidate']}")

    args.out_path.write_text(json.dumps(output, indent=2))
    print(f"[eval_synthetic] wrote {args.out_path}")


if __name__ == "__main__":
    main()
