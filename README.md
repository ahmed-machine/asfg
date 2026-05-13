# ASFG (VERY WIP, STILL SHIFTING)

Automated georectification pipeline for declassified Cold War-era reconnaissance satellite imagery (1959-1984). Given a scanned film frame and a modern georeferenced basemap, the pipeline registers the historical image to the reference through coarse localization, neural feature matching, grid optimization, and optical flow refinement. The output is a geometrically corrected GeoTIFF.

The imagery comes from three declassification releases under [Executive Order 12951](https://www.govinfo.gov/content/pkg/FR-1995-02-28/pdf/95-5050.pdf): [Declass-1](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-1) (1995, CORONA/ARGON/LANYARD 1960-1972), [Declass-2](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-2) (2002, KH-7/KH-9 1963-1980), and [Declass-3](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-3) (2011, remaining KH-7/KH-9 missions). Over 1.5 million of these images exist but the vast majority hasn't been georeferenced or orthorectified due to the manual labor required.

![Feature matching between reference (left) and target KH-9 frame (right)](assets/feature-matching-example.jpg)

## Hypothesis
Historical satellite film georectification as a process is still dependent on manual identification of geographic features (GCPs). Significant research has been done on vision transformers identifying features in modern satellite imagery. Rapid development of urban land cover and shoreline between modern satellite imagery and imagery from 50-60 years makes it very unstable and difficult to reliably match features across this temporal distance, even for humans.

To resolve this, we attempt to reduce the temporal distance by referencing historic satellite imagery. For a given region, we manually georeference one of the earliest film images captured with full coverage (i.e. 1965). The manually aligned image is used as a feature matching reference for automatically identified features in successive film captures (<1-2 years). We progress chronologically through the dataset (1960 to 1980), reference matching each successive image against each prior aligned image in the same coarse area.

Additionally, SoTA vision transformer models require fine-tuning for use with historic imagery (e.g LoRA with matched pairs across camera systems). To ensure high fidelity, we build safe guards around sample distribution (i.e. to avoid overfitting on urban areas), confidence thresholds, and outlier filtering prior to warping.

## Supported camera systems

| Designation | Program | Period | Entity prefix | Notes |
|---|---|---|---|---|
| KH-4 | CORONA | 1962-1972 | `DS1` | Panoramic stereo pairs; sub-frame segments stitched into strips |
| KH-7 | GAMBIT | 1963-1967 | `DZB` | High-resolution spot collection |
| KH-9 | HEXAGON | 1971-1984 | `D3C` | Mapping camera; multi-frame `.tgz` archives |

Camera-specific parameters are managed through YAML profiles in `data/profiles/` with inheritance. Auto-detected from entity ID at runtime.

## Usage

Raw imagery is available via [USGS's M2M API](https://m2m.cr.usgs.gov/). Here's a [sample](https://github.com/ahmed-machine/asfg/blob/master/data/available/corona2_69b17d89ee62ff28.csv) catalog csv. The raw imagery comes in subframes that need to be stitched and orthorectified. Each camera system has a separate profile and requires different parameters. We make extensive use of [NASA Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/) for sub-frame stitching (and fall back to VRT-based stitcher).

**End-to-end pipeline** (`process.py`): catalog parsing, download, extraction, stitching, georeferencing, alignment, and mosaic assembly. All stages are idempotent.

```bash
python process.py --csv catalog.csv --reference reference.tif --output-dir output/
python process.py --csv catalog.csv --auto-reference --entities D3C1213-200346A003
python process.py --frames-dir /path/to/frames/ -o output/ --crop-bbox 50.3,25.9,50.7,26.35
```

**Alignment only** (`auto-align.py`): takes a roughly georeferenced input and a reference GeoTIFF.

```bash
python auto-align.py input.tif --reference reference.tif -y
python auto-align.py input.tif --reference reference.tif --anchors gcps.json --qa-json qa.json -y
python auto-align.py --strip-manifest manifest.json
```

Run `python auto-align.py --help` for all options.

## Alignment pipeline

The core pipeline (`align/pipeline.py`) proceeds through:

1. **Global localization** -- Coarse template matching at ~40 m/px against the full reference using land/water masks.
2. **Coarse offset** -- Land mask NCC at 15 m/px, refined to 5 m/px. CLAHE and NMI fallbacks.
3. **Scale/rotation correction** -- Patch-based detection via ELoFTR + RANSAC affine, with multi-scale NCC fallback.
4. **Feature matching** -- RoMa v2 (DINOv3 backbone) tiled dense matching. Optional anchor GCPs.
5. **Filtering** -- TIN-TARR topological filtering, RANSAC verification, FPP accuracy optimization, holdout splitting.
6. **Grid optimization** -- PyTorch affine + learnable residual on hierarchical grid [(8,300), (24,300), (64,500)]. GCP fidelity + ARAP smoothness + chamfer loss with reclamation-aware masking.
7. **Flow refinement** -- DIS optical flow (census transform) at native resolution. SEA-RAFT uncertainty supplement.
8. **QA scoring** -- Grid-based shoreline drift, boundary distance, patch phase correlation, holdout cross-validation. Quality grades A-D.

## Project structure

```
auto-align.py           Alignment CLI
process.py              End-to-end ingestion pipeline (stage_* functions)
paths.py                On-disk layout helpers (georef_path, stitched_path, ...)

align/                  Alignment pipeline
  pipeline.py             Orchestrator and step functions
  state.py                Pipeline state container (AlignState)
  types.py                Typed records: MatchPair, GCP, BBox, QaReport
  params.py               Profile loader and parameter dataclasses
  constants.py            Named constants
  geo.py                  CRS helpers, overlap I/O, affine fitting
  grid_optim.py           PyTorch grid optimizer (affine + residual)
  matching.py             RoMa v2 tiled dense matching
  filtering.py            RANSAC, outlier removal, TIN-TARR, FPP optimization
  flow_refine.py          DIS + SEA-RAFT optical flow refinement
  warp.py                 Grid warping and output writing
  qa.py                   Quality metrics, holdout validation, QA reports
  scale.py                Scale/rotation detection (ELoFTR + NCC)
  anchors.py              Anchor GCP matching
  coarse.py               Coarse offset detection and global localization
  semantic_masking.py     Land/water/shoreline masking
  image.py                Image utilities (CLAHE, Wallis, Sobel)
  models.py               Torch device detection and model cache
  romav2/                 Vendored RoMa v2
  sea_raft/               Vendored SEA-RAFT

preprocess/             Preprocessing modules
  catalog.py              CSV parsing, camera ID, strip grouping
  usgs.py                 USGS M2M API client
  georef.py               Rough georeferencing, Sentinel-2 fetch
  orientation.py          Rotation detection and verification
  mosaic.py               Strip assembly and blending
  camera_model.py         ASP OpticalBar cam_gen + mapproject
  auto_anchors.py         Automatic anchor GCP generation via coarse RoMa
  dem.py                  DEM tile fetching (Copernicus GLO-30)

data/profiles/          Camera-specific YAML profiles (_base, kh4a, kh4b, kh7, kh8, kh9, kh9_mc)
scripts/test/           Test harnesses (run_test.py, run_piecewise.py, compare.py)
scripts/tune/           Optuna-based parameter tuning
```

## Dependencies

```
pip install -r requirements.txt
```

Core: numpy, opencv-python, rasterio, scipy, torch, torchvision, kornia, safetensors, pyproj, pyyaml, affine, GDAL

A CUDA or Apple MPS GPU is recommended; CPU inference works but is slower.

Optional (but not really): [NASA Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/) for sub-frame stitching (falls back to VRT-based stitcher).

## References

Load-bearing prior art for each pipeline stage:

- **[CORONA Atlas](https://corona.cast.uark.edu/)** (Casana & Cothren, CAST). Rigorous orthorectification of 2,200+ KH-4B images using a panoramic sensor model and DEM ortho. Informs the photogrammetric orthorectification stage and the grid optimizer priors.
- **Sohn, H. G., Kim, G. H., & Yom, J. H. (2004).** Mathematical Modelling of Historical Reconnaissance Corona KH-4B Imagery. *Photogrammetric Record*, 19(105), 51-66. Foundation sensor model for KH-4B panoramic cameras.
- **Hou, Z., Liu, Y., Zhang, L., Ai, H., Sun, Y., Han, X., & Zhu, C. (2023).** 2OC: A General Automated Orientation and Orthorectification Method for Corona KH-4B Panoramic Imagery. *Remote Sensing*, 15(21), 5116. doi:10.3390/rs15215116. 14-parameter panoramic model + model-guided re-matching loop.
- **Dehecq, A., et al. (2020).** Automated processing of KH-9 Hexagon imagery. *Frontiers in Earth Science*, 8, 566802. UW declass_stereo — GCP-free ASP bundle adjust using TanDEM-X DEM as ground control; inspired the ASP integration approach.
- **Ghuffar, S., Bolch, T., Dehecq, A., et al. (2022).** CoSP: Corona Stereo Pipeline. *arXiv:2201.07756*. SuperGlue auto-GCPs + rigorous camera model; prior art for auto-GCP generation on declassified imagery.
- **Edstedt, J., et al. (2025).** RoMa v2: Harder Better Faster Denser Feature Matching. *arXiv:2511.15706*. The dense feature matcher at the heart of the neural matching stage.
- **Simeoni, O., et al. (2025).** DINOv3. *arXiv:2508.10104*. The ViT backbone frozen inside RoMa v2.
- **He, X., et al. (2025).** MatchAnything: Universal Cross-Modality Image Matching. *arXiv:2501.07556*. ELoFTR variant used for patch-based scale/rotation pre-correction.
- **Wan, Z., et al. (2024).** SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow. *ECCV 2024*. The uncertainty-gated dense flow supplement in the flow refinement stage.
- **Beyer, R. A., Alexandrov, O., & McMichael, S. (2018).** The Ames Stereo Pipeline. *Earth and Space Science*, 5(9), 537-548. The ASP toolkit used throughout the orthorectification and bundle-adjust stages.
