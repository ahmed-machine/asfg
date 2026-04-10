# ASP Bundle Adjustment with RoMa Tie Points — Experimental

**Status: in-progress research. Downstream camera-model improvement has not been quantified on test data. Not part of the default production path.**

See `hypothesis.md` Contributions 6 (neural tie points for BA) and the "2OC-Inspired Implementations" subsection for the full research framing.

## What this is

Replace ASP's CPU-bound `ipfind` interest-point detection with GPU-accelerated RoMa v2 dense correspondences, written to ASP's binary `.match` format and fed directly into `bundle_adjust` via `--match-files-prefix`. Integrated with the 2OC P2 adaptive focal length fit (`--solve-intrinsics --intrinsics-to-float focal_length`).

## Enabling at runtime

Two gates must be satisfied:

1. **CLI gate** — both `--experimental` and `--bundle-adjust` must be passed:
   ```bash
   python3 process.py --csv catalog.csv --reference ref.tif --experimental --bundle-adjust
   ```
   The `--experimental` flag is a deliberate speed-bump to prevent unintentional use during production runs.

2. **Profile gate** — adaptive focal length is further gated by the profile flag `camera.bundle_adjust_solve_intrinsics: true`. `data/profiles/kh4.yaml` and `kh9.yaml` set this to `true`; other profiles default to `false`.

When both gates pass, the pipeline will:

- Run RoMa dense matching between strip frames, writing `.match` files under `output/match_files/`.
- Invoke ASP `bundle_adjust` with `-t opticalbar --inline-adjustments --match-files-prefix <path>`, optionally appending `--solve-intrinsics --intrinsics-to-float focal_length` when the profile flag is set.
- Log fitted focal lengths per frame for direct comparison with 2OC Table 6.
- Mapproject each frame through its adjusted camera model and use those orthos for downstream alignment.

## Components

- `align/experimental/bundle_adjust.py::run_strip_bundle_adjustment` — ASP bundle_adjust wrapper, ISISROOT defensive env injection, `--solve-intrinsics` support, fitted focal logging.
- `preprocess/experimental/match_ip.py::generate_strip_matches` — RoMa → ASP `.match` writer.
- `scripts/experimental/bundle_adjust/run_stereo_roma.py` — standalone research prototype that runs stereo reconstruction + RoMa matching + ASP bundle adjust on a pair of images. Not wired into `process.py`.

## Production-path callees that BA uses

These stay in the production tree; BA imports them across the experimental/production boundary:

- `preprocess/camera_model.py::{generate_camera, mapproject_image}` — production, also used by the non-BA ortho path.
- `preprocess/asp.py::find_asp_tool` — production, ASP install discovery.

## Known gaps

- **Camera-model improvement not quantified.** The BA path runs and produces adjusted cameras, but we have not benchmarked the downstream alignment score improvement vs the non-BA path on a fair test.
- **No joint bundle adjust across per-segment cameras.** When `camera.per_segment_ortho: true` is used (2OC P1), the per-segment cameras are fit independently and mosaiced via `gdalbuildvrt`. Joint BA across segments is the logical next step (2OC §3.2) but is not implemented.
- **`run_stereo_roma.py` is a research prototype.** It has not been validated end-to-end on a full test case.

## Deleting the thread if it turns out to be unviable

1. `rm -rf align/experimental/bundle_adjust.py preprocess/experimental/match_ip.py scripts/experimental/bundle_adjust/`
2. Remove the `--bundle-adjust` and `--experimental` CLI flags from `process.py`.
3. Remove the `stage_bundle_adjust_strips` function from `process.py`.
4. Remove `CameraParams.bundle_adjust_solve_intrinsics` from `align/params.py` and the field from `data/profiles/kh4.yaml` + `kh9.yaml`.
5. Remove Contribution 6 from `hypothesis.md` (or rewrite without the BA integration angle).
