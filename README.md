# declass-process

Tools for processing declassified satellite imagery — from rough georeferencing to globally-seeded, QA-scored alignment outputs ready for web map display.

These scripts were developed for the [mapBH](https://www.mapbh.org) project, but the alignment pipeline now supports metadata priors, full-reference global localization, independent QA, and manifest-driven strip/block runs.

## Scripts

### `auto-align.py`

Multi-stage alignment pipeline that takes a roughly-georeferenced GeoTIFF and aligns it against a trusted reference image.

**Pipeline stages:**

1. **Setup + priors** — loads optional metadata priors from JSON/XML sidecars and chooses a local metric CRS even when rough overlap is absent
2. **Global localization** — searches the full reference image at coarse resolution to seed translation before any overlap-dependent step
3. **Coarse offset detection** — semantic-weighted mask matching at 15m/px -> 5m/px for local translation refinement
4. **Scale/rotation detection** — LoFTR/local-patch scale estimation plus fallback NCC-based correction when needed
5. **Neural feature matching cascade** — anchors, SuperPoint+LightGlue, and tiled LoFTR/ELoFTR/RoMa matching with semantic tile prioritization
6. **Filtering + GCP selection** — spatial filtering, holdout split for QA, topology checks, and GCP distribution control
7. **Warp + fallback** — grid-optimized warp plus TPS fallback, both scored with independent QA
8. **Independent QA** — writes JSON QA reports, confidence scores, and can abstain on low-confidence results with `--allow-abstain`

**Key features:**
- Full-reference global localization before overlap-dependent matching
- Metadata prior ingestion from JSON/XML sidecars
- Semantic mask provider abstraction for coarse search, matching, and QA
- Holdout-based independent QA report output (`_qa.json`)
- Manifest-driven strip/block execution through `--strip-manifest` and `--block-manifest`
- Diagnostic visualization output (`_debug.jpg`)
- Apple Silicon MPS GPU acceleration for neural matchers

```bash
# Basic alignment with anchor GCPs
python3 auto-align.py target.tif \
  --reference reference.tif \
  --anchors data/bahrain_anchor_gcps.json \
  --output aligned.tif \
  -y

# Without anchors (pure neural matching)
python3 auto-align.py target.tif \
  --reference reference.tif \
  --output aligned.tif \
  -y

# Force a full-reference global search and keep a QA report
python3 auto-align.py target.tif \
  --reference reference.tif \
  --global-search-res 30 \
  --qa-json aligned_qa.json \
  -y

# Maximum quality mode (slower, more memory)
python3 auto-align.py target.tif \
  --reference reference.tif \
  --anchors data/bahrain_anchor_gcps.json \
  --best -y

# Allow low-confidence runs to abstain instead of forcing an output
python3 auto-align.py target.tif \
  --reference reference.tif \
  --allow-abstain \
  -y

# Run a strip manifest
python3 auto-align.py --strip-manifest manifests/example_strip.json

# Run a block manifest
python3 auto-align.py --block-manifest manifests/example_block.json
```

Common optional flags:

- `--metadata-priors ...` or `--metadata-priors-dir PATH` — load sidecar rough-location priors
- `--global-search` / `--no-global-search` — enable or disable the pre-overlap search stage
- `--reference-window left,bottom,right,top` — restrict the global search window in the work CRS
- `--mask-provider coastal_obia` — choose the semantic mask provider used for localization and QA
- `--diagnostics-dir DIR` — redirect debug images and QA artifacts
- `--qa-json PATH` — persist independent QA metrics
- `--allow-abstain` — drop low-confidence outputs instead of forcing a result

### `process_d3c.py`

End-to-end KH-9 (Declass 3) satellite imagery pipeline:

1. Download EarthExplorer "full" metadata XML (4 corner coordinates + acquisition date)
2. Extract `.tgz` archives
3. Stitch frames into a panoramic strip (GDAL VRT-based, with automatic frame ordering)
4. Parse metadata XML for corner coordinates
5. Georectify using corner GCPs (handles rotated panoramic strips via center-pivot rotation)
6. Crop to Bahrain maritime boundary
7. Trim nodata borders
8. Rename to final format: `YYYY-MM-DD - Bahrain - ENTITY.tif`

Handles 9 KH-9 entities covering Bahrain, with per-entity frame ordering configuration (some strips run east-to-west).

```bash
# Process all entities
python3 process_d3c.py

# Process specific entities
python3 process_d3c.py --entities D3C1213-200346A003 D3C1214-200421F003

# Re-stitch and reprocess from step 3
python3 process_d3c.py --restitch
```

### `stitch_frames.py`

SIFT-based panoramic frame stitcher for consecutive satellite scan frames.

- Uses SIFT feature detection on downscaled images for speed
- Computes homography at reduced resolution, then applies at full resolution
- Feathered distance-transform blending in overlap regions
- Validates homography determinant to catch bad matches

```bash
python3 stitch_frames.py frame1.tif frame2.tif frame3.tif -o mosaic.tif

# Custom downscale factor for feature detection
python3 stitch_frames.py frame1.tif frame2.tif -o mosaic.tif --scale 0.5
```

### `georef.py`

Rough georeferencing of CORONA/LANYARD (DS1) imagery from USGS XML bounding boxes.

For each image:
1. Parses associated USGS metadata XML for bounding box coordinates
2. For stitched images (multiple XMLs), computes the union bounding box
3. Assigns WGS84 coordinates via `gdal_translate -a_ullr`
4. Reprojects to Web Mercator (EPSG:3857) for tile serving

Note: USGS documents ~10-mile positional error for CORONA/LANYARD corner points. This gives a rough georeference suitable for initial placement; use `auto-align.py` for precise alignment.

```bash
python3 georef.py
```

## Data Files

### `data/bahrain_anchor_gcps.json`

10 stable anchor Ground Control Points — manually identified landmarks that existed throughout the 1960-2000 period:

- **Forts**: Bahrain Fort NW corner, Arad Fort NW corner
- **Water features**: Adhari Pool, Ayn Al Hakim
- **Boundaries**: Hoora Graveyard wall
- **Islands**: Jazirat Ash Shaykh, Ya'sub Island, Muhammediyya Island, Al Sayah Island
- **Urban features**: Casino Fountain (Muharraq)

Each GCP includes coordinates, feature type, confidence level, and optional `patch_size_m` for custom matching window sizes.

### `data/bahrain_boundary.geojson`

GeoJSON polygon covering Bahrain's maritime extent (50.27-50.90E, 25.55-26.35N). Used by `process_d3c.py` to crop georectified imagery to the area of interest.

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**Required:**
- `numpy` — array operations
- `opencv-python` — SIFT/FLANN feature matching, image processing
- `rasterio` — GeoTIFF I/O with coordinate system support
- `scipy` — spatial algorithms, interpolation
- `scikit-image` — image processing utilities
- `Pillow` — image I/O

**Optional (for neural matching in `auto-align.py`):**
- `torch` — PyTorch for neural network inference
- `torchvision` — RAFT optical-flow model
- `kornia` — geometric vision utilities
- `lightglue` — LightGlue + SuperPoint feature matching

**Vendored model code:**
- `align/roma/` and `align/eloftr/` are intentionally vendored in-repo for reproducible production runs.

**System dependencies:**
- `gdal` / `gdalwarp` / `gdal_translate` — geospatial processing (install via `brew install gdal` or system package manager)
