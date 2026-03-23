# declass-process

Automated georectification pipeline for declassified Cold War-era reconnaissance satellite imagery acquired by the United States between 1959 and 1984. Given a scanned film frame (or multi-frame strip) and a modern georeferenced basemap, the pipeline estimates and corrects the spatial mapping between the historical image and the reference through a sequence of increasingly fine-grained registration stages. The output is a geometrically corrected GeoTIFF suitable for land-use analysis, climate, and historical research.

The imagery comes from three declassification releases under [Executive Order 12951](https://en.wikipedia.org/wiki/Executive_Order_12951): [Declass-1](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-1) (1995, CORONA/ARGON/LANYARD 1960–1972), [Declass-2](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-2) (2002, KH-7/KH-9 1963–1980), and [Declass-3](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-3) (2011, remaining KH-7/KH-9 missions). About 5% of the 1.5M film frames in these datasets have been digitised and are available online via [USGS EarthExplorer](https://earthexplorer.usgs.gov/).

![Feature matching between reference (left) and target KH-9 frame (right)](assets/feature-matching-example.jpg)

## Supported camera systems

| Designation | Program | Period | Entity prefix | Notes |
|---|---|---|---|---|
| KH-4 | CORONA | 1962–1972 | `DS1` | Panoramic stereo pairs; sub-frame segments stitched into strips |
| KH-7 | GAMBIT | 1963–1967 | `DZB` | High-resolution spot collection |
| KH-9 | HEXAGON | 1971–1984 | `D3C` | Mapping camera; multi-frame `.tgz` archives |

Camera-specific tuning parameters (coarse detection thresholds, matching settings, grid optimizer weights) are managed through YAML profiles in `data/profiles/` with inheritance. Auto-detection from entity ID selects the appropriate profile at runtime. Each profile's parameters tuned with [optuna](https://github.com/optuna/optuna)

## Usage

**`process.py`** — End-to-end pipeline: USGS catalog parsing, scene download via the M2M API, archive extraction, frame stitching, rough georeferencing from corner coordinates, alignment against a reference image, and mosaic assembly. All stages are idempotent.

```bash
python process.py --csv catalog.csv --reference reference.tif --output-dir output/
python process.py --csv catalog.csv --auto-reference --entities D3C1213-200346A003
```
Good reference images vary by location, but ideally a high-resolution, older stitched set from the declassified data that was manually georeferenced in QGIS (TPS warp). Auto-reference defaults to Sentinel-2 current imagery if not provided. Additionally, user-defined anchors GCPs are optional (see `data/bahrain_anchor_gcps.json`); good anchors are distinctive features that have remained constant since 1965 to 1984 (or modern day). Automated feature matching has converged to the same score as user-defined anchor GCPs, which are used for validation gut checks.

Note: catalog.csv represents a list of entity IDs downloaded from USGS (see `data/available/`). Entities refers to USGS's formatted entity IDs.

**`auto-align.py`** — Alignment-only entry point. Takes a roughly georeferenced input GeoTIFF and a reference GeoTIFF and produces an aligned output. Also accepts strip and block manifests for batch processing.

```bash
python auto-align.py input.tif --reference reference.tif -y
python auto-align.py input.tif --reference reference.tif --anchors gcps.json --qa-json qa.json -y
python auto-align.py --strip-manifest manifest.json
```

**`process.py --frames-dir`** — Process a local folder of pre-downloaded sub-frame TIFs. Fetches corner coordinates from the USGS M2M API, stitches sub-frames, detects orientation, georeferences, and mosaics all strips into a seamless output. Supports geographic cropping via `--crop-bbox`.

```bash
python process.py --frames-dir /path/to/frames/ -o output/
python process.py --frames-dir /path/to/frames/ -o output/ --crop-bbox 50.3,25.9,50.7,26.35
python process.py --frames-dir /path/to/frames/ -o output/ -r reference.tif
```

### Single-stage entry points

These run individual pipeline stages in isolation, useful for manual preprocessing or debugging. They delegate to the `preprocess/` modules.

**`stitch_frames.py`** — Stitch consecutive satellite frames into a single panoramic strip. Auto-detects frame ordering and handles 180° rotation correction.

```bash
python stitch_frames.py frame1.tif frame2.tif frame3.tif -o stitched.tif
```

**`georef.py`** — Rough-georeference imagery using USGS XML bounding boxes. Assigns WGS84 coordinates and reprojects to EPSG:3857.

```bash
python georef.py --input image.tif --xml metadata1.xml
```

## Alignment pipeline

The core registration pipeline (`align/pipeline.py`) proceeds through the following stages:

1. **Global localization** — When the input lacks usable geolocation or falls outside the reference footprint, a coarse search localizes the image against the full reference at ~40 m/px using land/water mask cross-correlation.

2. **Coarse offset detection** — Land/water mask template matching at 15 m/px, refined to 5 m/px, to estimate the initial translation.

3. **Scale and rotation correction** — MatchAnything-ELoFTR dense matching with RANSAC affine estimation (primary), with multi-scale NCC on land masks and gradient images as fallback, to detect and pre-correct geometric distortions before fine matching. Note: Dense matching with DINOv3 doesn't improve results.

4. **Feature matching** — Dense correspondence estimation using RoMa v2 with a satellite-pretrained ('satast') DINOv3 backbone on tiled image patches, producing candidate ground control points. Optional anchor GCPs from known landmarks supplement the neural matches.

5. **Filtering and validation** — Iterative outlier removal, RANSAC-based geometric verification, local consistency filtering, and holdout splitting for independent QA.

6. **Grid optimization** — A PyTorch-based optimizer fits an affine baseline plus a learnable per-cell residual displacement field on a hierarchical multi-resolution grid. The composite loss includes GCP fidelity, displacement smoothness, and land-mask chamfer distance.

7. **Flow refinement** — DIS (OpenCV) optical flow at native resolution corrects residual local distortions below the grid cell size. SEA-RAFT provides supplementary per-pixel uncertainty estimates to reject self-consistent but unreliable flow in textureless regions.

8. **QA scoring** — Grid-based regional shoreline drift, shoreline and stable-feature IoU, patch-level phase correlation residuals, and holdout cross-validation on withheld GCPs. The pipeline can optionally abstain from producing output when confidence is low (`--allow-abstain`).

## Project structure

```
auto-align.py           Alignment CLI entry point
process.py              End-to-end ingestion pipeline
georef.py               Standalone rough georeferencing CLI
stitch_frames.py        Standalone frame stitching CLI

align/                  Alignment pipeline modules
  pipeline.py             Main orchestrator and step functions
  types.py                Typed records: MatchPair, GCP, and result dataclasses
  params.py               Camera profile loader, detection, and AlignParams
  geo.py                  CRS helpers, overlap I/O, affine fitting, boundary GCPs
  grid_optim.py           PyTorch grid optimizer (affine + residual displacement)
  matching.py             RoMa v2 tiled dense matching (DINOv3)
  filtering.py            RANSAC, outlier removal, GCP selection
  flow_refine.py          DIS + SEA-RAFT optical flow refinement
  warp.py                 Grid-based warping and output writing
  qa.py                   Quality metrics, diagnostics, GPU-batched phase correlation
  scale.py                Scale/rotation detection (ELoFTR + NCC)
  anchors.py              Anchor GCP matching
  coarse.py               Coarse offset detection
  manifest.py             Strip and block manifest processing
  profiler.py             Hierarchical pipeline profiler with resource tracking
  romav2/                 Vendored RoMa v2 model
  sea_raft/               Vendored SEA-RAFT model

preprocess/             Preprocessing modules
  catalog.py              Camera system definitions, CSV parsing, scene/strip grouping
  usgs.py                 USGS M2M API client, download, archive extraction
  georef.py               Rough georeferencing (corner GCPs) and Sentinel-2 reference fetch
  orientation.py          4-way rotation detection and verification
  stitch.py               Multi-frame strip stitching (VRT-based)
  reseau.py               KH-9 réseau grid detection and film distortion correction
  mosaic.py               Strip assembly, seam blending, radiometric normalization
data/profiles/          Camera-specific parameter profiles (YAML with inheritance)
scripts/                Test harnesses, benchmarks, and tuning tools
```

## Dependencies

```
pip install -r requirements.txt
```

numpy, opencv-python, rasterio, scipy, scikit-image, Pillow, torch, torchvision, kornia, transformers, pyproj, affine, pyyaml, GDAL (`brew install gdal` or system package manager)

`align/romav2/` (RoMa v2) and `align/sea_raft/` (SEA-RAFT) are vendored for reproducibility. A CUDA or Apple MPS GPU is recommended; CPU inference works but is substantially slower.

## References

1. Edstedt, J., Nordström, D., Zhang, Y., et al. (2025). RoMa v2: Harder Better Faster Denser Feature Matching. *arXiv:2511.15706*.
2. He, X., Yu, H., Peng, S., et al. (2025). MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training. *arXiv:2501.07556*. (EfficientLoFTR variant, used for scale/rotation detection via HuggingFace)
3. Teed, Z. & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. *ECCV 2020*.
4. Wan, Z., et al. (2024). SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow. *ECCV 2024*. (Uncertainty-aware flow supplement)
5. Siméoni, O., Vo, H. V., Seitzer, M., et al. (2025). DINOv3. *arXiv:2508.10104*. (Satellite-pretrained sat493m weights via timm, used for both the RoMa v2 backbone and the grid optimizer feature loss)
6. Guo, H., Liu, J., Yang, B., et al. (2022). Outlier removal and feature point pairs optimization for piecewise linear transformation in the co-registration of very high-resolution optical remote sensing imagery. *ISPRS J. Photogrammetry and Remote Sensing*.
7. Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *IJCV*, 60(2), 91–110.
8. Donovan, M. et al. sPyMicMac: TPS-based réseau correction for KH-9 Hexagon imagery.
