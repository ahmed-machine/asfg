"""Phase 2.4 — Teacher pseudo-label extraction for mapbh pairs.

Runs frozen base RoMa v2 with ``apply_setting("satast")`` on each pair in
``data/lora_pairs/`` and saves the sampled correspondences as
``data/lora_pairs/labels/roma_raw_<pair_id>.pt``.

Each saved object is a dict::

    {
        "m": torch.Tensor (N, 4) in normalised [-1, 1] coords,
             cols 0..1 = (x_A, y_A), cols 2..3 = (x_B, y_B)
        "c": torch.Tensor (N,) confidence in [0, 1]
        "pair_id": str
        "layer_src": str, "layer_ref": str,
        "size": int  # 800 (satast H_lr)
    }

Use ``num_corresp=1500`` by default; the sampling is multinomial-weighted
so high-confidence patches dominate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _list_pairs(pairs_dir: Path) -> list[Path]:
    return sorted(pairs_dir.glob("pair_*_meta.json"))


def _load_pair(meta_path: Path, size: int, device: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    meta = json.loads(meta_path.read_text())
    stem = meta_path.name[: -len("_meta.json")]
    ref_path = meta_path.parent / f"{stem}_ref.png"
    src_path = meta_path.parent / f"{stem}_src_clean.png"
    ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    src = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if ref is None or src is None:
        raise RuntimeError(f"failed to load images for {stem}")
    if ref.shape[0] != size or ref.shape[1] != size:
        ref = cv2.resize(ref, (size, size), interpolation=cv2.INTER_AREA)
    if src.shape[0] != size or src.shape[1] != size:
        src = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    src_t = torch.from_numpy(src_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return ref_t, src_t, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Extract teacher pseudo-labels for mapbh pairs")
    p.add_argument("--pairs-dir", type=Path, default=REPO_ROOT / "data" / "lora_pairs")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <pairs-dir>/labels")
    p.add_argument("--num-corresp", type=int, default=1500)
    p.add_argument("--size", type=int, default=800,
                   help="Image side length in pixels (must match satast H_lr)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only this many pairs (smoke test)")
    p.add_argument("--skip-existing", action="store_true", default=True)
    args = p.parse_args()

    out_dir = args.out_dir or (args.pairs_dir / "labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = _list_pairs(args.pairs_dir)
    if args.limit is not None:
        metas = metas[: args.limit]
    print(f"[extract_pseudo_labels] {len(metas)} pairs to process")

    from align.models import get_torch_device
    device = get_torch_device()
    print(f"[extract_pseudo_labels] device: {device}")

    from align.romav2 import RoMaV2
    from align.romav2.device import set_device
    set_device(device)
    cfg = RoMaV2.Cfg(setting="fast", compile=False)
    model = RoMaV2(cfg)
    model.apply_setting("satast")
    model.eval()
    print(f"[extract_pseudo_labels] model loaded (satast: H_lr={model.H_lr})")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    n_corresp_total = 0
    conf_medians: list[float] = []
    t0 = time.time()

    for i, meta_path in enumerate(metas):
        stem = meta_path.name[: -len("_meta.json")]
        out_path = out_dir / f"roma_raw_{stem}.pt"
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue
        try:
            ref_t, src_t, meta = _load_pair(meta_path, args.size, device)
            with torch.no_grad():
                preds = model.match(ref_t, src_t)
                sampled = model.sample(preds, num_corresp=args.num_corresp)
            m, c = sampled[0], sampled[1]
        except Exception as e:
            print(f"[extract_pseudo_labels] FAIL {stem}: {e}")
            n_failed += 1
            continue

        torch.save(
            {
                "m": m.detach().cpu(),
                "c": c.detach().cpu(),
                "pair_id": stem,
                "layer_src": meta.get("layer_src"),
                "layer_ref": meta.get("layer_ref"),
                "mission_src": meta.get("mission_src"),
                "mission_ref": meta.get("mission_ref"),
                "size": args.size,
            },
            out_path,
        )
        n_done += 1
        n_corresp_total += int(m.shape[0])
        conf_medians.append(float(c.median().item()))

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(
                f"[extract_pseudo_labels] {n_done} done, {n_skipped} skipped, {n_failed} failed; "
                f"avg N={n_corresp_total / max(n_done, 1):.0f}, "
                f"median conf p50={np.median(conf_medians):.3f}, "
                f"rate={rate:.1f}/s"
            )

    summary = {
        "n_done": n_done,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "wall_clock_s": time.time() - t0,
        "avg_correspondences": (n_corresp_total / max(n_done, 1)),
        "confidence_median_overall": float(np.median(conf_medians)) if conf_medians else None,
        "confidence_p10": float(np.percentile(conf_medians, 10)) if conf_medians else None,
        "confidence_p90": float(np.percentile(conf_medians, 90)) if conf_medians else None,
        "size": args.size,
        "num_corresp": args.num_corresp,
    }
    summary_path = out_dir / "extract_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[extract_pseudo_labels] wrote {summary_path}")
    print(f"[extract_pseudo_labels] done: {n_done} pseudo-labels written")


if __name__ == "__main__":
    main()
