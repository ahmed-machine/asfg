# declass-process

Automated georectification pipeline for declassified Cold War-era reconnaissance satellite imagery acquired by the United States between 1959 and 1984. Given a scanned film frame (or multi-frame strip) and a modern georeferenced basemap, the pipeline estimates and corrects the spatial mapping between the historical image and the reference through a sequence of increasingly fine-grained registration stages. The output is a geometrically corrected GeoTIFF suitable for land-use analysis, climate, and historical research.

The imagery comes from three declassification releases under [Executive Order 12951](https://www.govinfo.gov/content/pkg/FR-1995-02-28/pdf/95-5050.pdf): [Declass-1](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-1) (1995, CORONA/ARGON/LANYARD 1960-1972), [Declass-2](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-2) (2002, KH-7/KH-9 1963-1980), and [Declass-3](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-3) (2011, remaining KH-7/KH-9 missions). About 5% of the 1.5M film frames in these datasets have been digitised and are available online via [USGS EarthExplorer](https://earthexplorer.usgs.gov/).

![Feature matching between reference (left) and target KH-9 frame (right)](assets/feature-matching-example.jpg)

## Supported camera systems

| Designation | Program | Period | Entity prefix | Notes |
|---|---|---|---|---|
| KH-4 | CORONA | 1962-1972 | `DS1` | Panoramic stereo pairs; sub-frame segments stitched into strips |
| KH-7 | GAMBIT | 1963-1967 | `DZB` | High-resolution spot collection |
| KH-9 | HEXAGON | 1971-1984 | `D3C` | Mapping camera; multi-frame `.tgz` archives |

Camera-specific tuning parameters (coarse detection thresholds, matching settings, grid optimizer weights) are managed through YAML profiles in `data/profiles/` with inheritance. Auto-detection from entity ID selects the appropriate profile at runtime. Each profile's parameters tuned with [optuna](https://github.com/optuna/optuna).

Available profiles: `_base.yaml` (defaults), `kh4.yaml`, `kh7.yaml`, `kh9.yaml`, `kh9_tuned.yaml`, `best_kh4.yaml`, `best_kh9.yaml`.

## Usage

**`process.py`** -- End-to-end pipeline: USGS catalog parsing, scene download via the M2M API, archive extraction, frame stitching, rough georeferencing from corner coordinates, alignment against a reference image, and mosaic assembly. All stages are idempotent.

```bash
python process.py --csv catalog.csv --reference reference.tif --output-dir output/
python process.py --csv catalog.csv --auto-reference --entities D3C1213-200346A003
```
Good reference images vary by location, but ideally a high-resolution, older stitched set from the declassified data that was manually georeferenced in QGIS (TPS warp). Auto-reference defaults to Sentinel-2 current imagery if not provided. Additionally, user-defined anchor GCPs are optional (see `data/bahrain_anchor_gcps.json`); good anchors are distinctive features that have remained constant since 1965 to 1984 (or modern day). Automated feature matching has converged to the same score as user-defined anchor GCPs, which are used for validation gut checks.

Note: catalog.csv represents a list of entity IDs downloaded from USGS (see `data/available/`). Entities refers to USGS's formatted entity IDs.

**`auto-align.py`** -- Alignment-only entry point. Takes a roughly georeferenced input GeoTIFF and a reference GeoTIFF and produces an aligned output. Also accepts strip and block manifests for batch processing.

```bash
python auto-align.py input.tif --reference reference.tif -y
python auto-align.py input.tif --reference reference.tif --anchors gcps.json --qa-json qa.json -y
python auto-align.py --strip-manifest manifest.json
python auto-align.py --block-manifest manifest.json
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--profile` | auto | Camera profile (`kh4`, `kh7`, `kh9`, or auto-detected from filename) |
| `--best` | off | Maximum quality mode |
| `--device` | auto | Torch device (`auto`, `cpu`, `mps`, `cuda`) |
| `--match-res` | 5.0 | Resolution in m/px for feature matching |
| `--mask-provider` | coastal_obia | Semantic mask provider (`heuristic`, `coastal_obia`) |
| `--matcher-dense` | roma | Dense matching model |
| `--global-search` / `--no-global-search` | on | Full-reference global localization |
| `--global-search-res` | 40.0 | Global localization resolution in m/px |
| `--force-global` | off | Force global localization even with rough overlap |
| `--anchors` | none | Path to anchor GCPs JSON file |
| `--qa-json` | none | Write QA report JSON to this path |
| `--diagnostics-dir` | none | Directory for diagnostic outputs |
| `--allow-abstain` | off | Allow the pipeline to withhold low-confidence outputs |
| `--tps-fallback` | off | Run TPS fallback warp for comparison |
| `--tin-tarr-thresh` | 1.5 | TIN-TARR area ratio threshold for topological filtering |
| `--skip-fpp` | off | Skip FPP image-based accuracy optimization |

**`process.py --frames-dir`** -- Process a local folder of pre-downloaded sub-frame TIFs. Fetches corner coordinates from the USGS M2M API, stitches sub-frames, detects orientation, georeferences, and mosaics all strips into a seamless output. Supports geographic cropping via `--crop-bbox`.

```bash
python process.py --frames-dir /path/to/frames/ -o output/
python process.py --frames-dir /path/to/frames/ -o output/ --crop-bbox 50.3,25.9,50.7,26.35
python process.py --frames-dir /path/to/frames/ -o output/ -r reference.tif
```

### Single-stage entry points

These run individual pipeline stages in isolation, useful for manual preprocessing or debugging. They delegate to the `preprocess/` modules.

**`stitch_frames.py`** -- Stitch consecutive satellite frames into a single panoramic strip. Auto-detects frame ordering and handles 180-degree rotation correction.

```bash
python stitch_frames.py frame1.tif frame2.tif frame3.tif -o stitched.tif
```

**`georef.py`** -- Rough-georeference imagery using USGS XML bounding boxes. Assigns WGS84 coordinates and reprojects to EPSG:3857.

```bash
python georef.py --input image.tif --xml metadata1.xml
```

## Alignment pipeline

The core registration pipeline (`align/pipeline.py`) proceeds through the following stages:

1. **Global localization** (optional) -- When the input lacks usable geolocation or falls outside the reference footprint, a coarse search localizes the image against the full reference at ~40 m/px using land/water mask cross-correlation. Generates multiple ranked hypotheses and selects the best via NCC verification.

2. **Coarse offset detection** -- Land/water mask template matching at 15 m/px, refined to 5 m/px with NCC, CLAHE NCC, and NMI methods to estimate the initial translation. Supports restricted search regions for strip imagery with partial coverage.

3. **Scale and rotation correction** -- Patch-based local scale detection using MatchAnything-ELoFTR dense matching with RANSAC affine estimation (primary), with multi-scale NCC on land masks and gradient images as fallback. Detects and pre-corrects geometric distortions before fine matching. A post-correction coarse re-detection verifies the correction improved alignment.

4. **Feature matching** -- Dense correspondence estimation using RoMa v2 with a satellite-pretrained DINOv3 backbone on tiled image patches, producing candidate ground control points. Optional anchor GCPs from known landmarks supplement the neural matches. Anchor pre-search uses NCC-based rigid affine fitting to estimate residual offset before fine matching.

5. **Filtering and validation** -- TIN-TARR topological filtering (Delaunay triangulation area ratios), RANSAC-based geometric verification, FPP accuracy difference optimization (image-similarity-based GCP refinement), iterative outlier removal, and holdout splitting (20% withheld) for independent cross-validation.

6. **Grid optimization** -- A PyTorch-based optimizer fits an affine baseline plus a learnable per-cell residual displacement field on a hierarchical multi-resolution grid `[(8, 300), (24, 300), (64, 500)]`. The composite loss includes GCP fidelity (with per-GCP confidence weights), displacement smoothness (ARAP + Laplacian regularization), and land-mask chamfer distance with reclamation-aware masking.

7. **Flow refinement** -- Coarse-to-fine DIS (OpenCV) optical flow with census transform at native resolution corrects residual local distortions below the grid cell size. SEA-RAFT provides supplementary per-pixel uncertainty estimates to reject self-consistent but unreliable flow in textureless regions. Cross-temporal change suppression prevents flow from chasing real coastline changes (reclamation, piers).

8. **QA scoring** -- Grid-based regional shoreline drift (4x6 grid), boundary distance metrics for stable features and shorelines, patch-level phase correlation residuals, and holdout cross-validation on withheld GCPs. Quality grades (A/B/C/D) assigned based on configurable thresholds. The pipeline can optionally abstain from producing output when confidence is low (`--allow-abstain`).

### Checkpointing

The pipeline saves JSON checkpoints at phase boundaries (`post_setup`, `post_coarse`, `post_scale_rotation`, `post_match`, `post_validate`), enabling inspection of intermediate state and potential future resumable execution.

## Project structure

```
auto-align.py           Alignment CLI entry point
process.py              End-to-end ingestion pipeline
georef.py               Standalone rough georeferencing CLI
stitch_frames.py        Standalone frame stitching CLI

align/                  Alignment pipeline modules
  pipeline.py             Main orchestrator and step functions
  types.py                Typed records: MatchPair, GCP, BBox, QaReport, and result dataclasses
  params.py               Camera profile loader, auto-detection, and parameter dataclasses
                            (AlignParams, CoarseParams, ScaleRotationParams, MatchingParams,
                             ValidationParams, GridOptimParams, FlowParams, QaParams)
  constants.py            Named constants (synced from profiles at runtime)
  geo.py                  CRS helpers, overlap I/O, affine fitting, boundary GCPs
  grid_optim.py           PyTorch grid optimizer (affine + residual displacement)
  matching.py             RoMa v2 tiled dense matching (DINOv3 backbone)
  filtering.py            RANSAC, outlier removal, GCP selection
  tin_filter.py           TIN-TARR topological filtering and FPP accuracy optimization
  flow_refine.py          DIS + SEA-RAFT optical flow refinement
  warp.py                 Grid-based warping, DINO feature extraction, and output writing
  qa.py                   Quality metrics, diagnostics, GPU-batched phase correlation
  qa_runner.py            QA orchestration, holdout splitting, quality grading (A/B/C/D)
  scale.py                Scale/rotation detection (ELoFTR + NCC)
  anchors.py              Anchor GCP matching with pre-search rigid correction
  coarse.py               Coarse offset detection (land mask, CLAHE, NMI)
  global_localization.py  Full-reference search at ~40 m/px with hypothesis ranking
  semantic_masking.py     Land/water/shoreline masking (heuristic and coastal OBIA providers)
  image.py                Image utilities: CLAHE, Wallis normalization, Sobel gradients, land masks
  models.py               Torch device detection and shared model cache (RoMa, ELoFTR)
  metadata_priors.py      Metadata prior loading (USGS XML, JSON)
  errors.py               Custom exception classes
  checkpoint.py           Phase checkpoint save/load (JSON + NPZ serialization)
  manifest.py             Strip and block manifest processing
  profiler.py             Hierarchical pipeline profiler with resource tracking
  romav2/                 Vendored RoMa v2 model
  sea_raft/               Vendored SEA-RAFT model

preprocess/             Preprocessing modules
  catalog.py              Camera system definitions, CSV parsing, scene/strip grouping
  usgs.py                 USGS M2M API client, download, archive extraction
  asp.py                  ASP tool discovery (image_mosaic)
  georef.py               Rough georeferencing (corner GCPs), coarse-align-crop, Sentinel-2 fetch
  orientation.py          Multi-method rotation detection and reference-based verification
  stitch.py               Multi-frame strip stitching (ASP image_mosaic + VRT fallback)
  reseau.py               KH-9 reseau grid detection and film distortion correction
  mosaic.py               Strip assembly, seam blending, radiometric normalization

data/profiles/          Camera-specific parameter profiles (YAML with inheritance)
scripts/test/           Test harnesses (run_test.py, compare.py, eval_ground_truth.py, rescore.py)
scripts/tune/           Optuna-based parameter tuning
```

## Dependencies

```
pip install -r requirements.txt
```

Core: numpy, opencv-python, rasterio, scipy, scikit-image, Pillow, torch, torchvision, kornia, safetensors

Also required (imported but not in requirements.txt): transformers, pyproj, pyyaml, affine, GDAL (`brew install gdal` or system package manager)

`align/romav2/` (RoMa v2) and `align/sea_raft/` (SEA-RAFT) are vendored for reproducibility. A CUDA or Apple MPS GPU is recommended; CPU inference works but is substantially slower.

### Optional: Ames Stereo Pipeline (ASP)

[NASA Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/) `image_mosaic` is used for sub-frame stitching of KH-4/7/9 scanner tiles when available. ASP is optional -- when not installed, stitching falls back to the built-in VRT-based stitcher. Download from [ASP releases](https://github.com/NeoGeographyToolkit/StereoPipeline/releases).

## References

1. Edstedt, J., Nordstrom, D., Zhang, Y., et al. (2025). RoMa v2: Harder Better Faster Denser Feature Matching. *arXiv:2511.15706*.
2. He, X., Yu, H., Peng, S., et al. (2025). MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training. *arXiv:2501.07556*. (EfficientLoFTR variant, used for scale/rotation detection)
3. Teed, Z. & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. *ECCV 2020*.
4. Wan, Z., et al. (2024). SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow. *ECCV 2024*. (Uncertainty-aware flow supplement)
5. Simeoni, O., Vo, H. V., Seitzer, M., et al. (2025). DINOv3. *arXiv:2508.10104*. (Satellite-pretrained sat493m weights via timm, used as RoMa v2 backbone)
6. Guo, H., Liu, J., Yang, B., et al. (2022). Outlier removal and feature point pairs optimization for piecewise linear transformation in the co-registration of very high-resolution optical remote sensing imagery. *ISPRS J. Photogrammetry and Remote Sensing*. (TIN-TARR filtering and FPP accuracy optimization)
7. Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *IJCV*, 60(2), 91-110.
8. Donovan, M. et al. sPyMicMac: TPS-based reseau correction for KH-9 Hexagon imagery.
9. Beyer, R. A., Alexandrov, O., & McMichael, S. (2018). The Ames Stereo Pipeline: NASA's Open Source Software for Deriving and Processing Terrain Data. *Earth and Space Science*, 5(9), 537-548. https://doi.org/10.1029/2018EA000409. (image_mosaic sub-frame stitching)
