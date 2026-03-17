"""Archive extraction for declassified satellite imagery."""

import gzip
import os
import shutil
import subprocess
import sys
import tarfile

# tarfile "data" filter was added in Python 3.11.4 to strip dangerous paths
_TAR_FILTER = {"filter": "data"} if sys.version_info >= (3, 11, 4) else {}


def _is_gzip(file_path: str) -> bool:
    """Check if a file is gzip-compressed by reading the magic bytes."""
    try:
        with open(file_path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def _validate_tiff(file_path: str) -> bool:
    """Check if a file is a valid TIFF using gdalinfo."""
    try:
        result = subprocess.run(
            ["gdalinfo", "-json", file_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return False
        import json
        info = json.loads(result.stdout)
        w, h = info.get("size", [0, 0])
        return w > 0 and h > 0
    except Exception:
        return False


def extract_archive(file_path: str, output_dir: str, entity_id: str) -> str:
    """Extract a .tgz archive or handle a single .tif file.

    Handles gzip-compressed TIFs (common for KH-4/KH-7 USGS downloads).

    Returns the directory containing the extracted/linked frames.
    """
    entity_dir = os.path.join(output_dir, "extracted", entity_id)

    # Check if already extracted
    if os.path.exists(entity_dir):
        tifs = [f for f in os.listdir(entity_dir) if f.lower().endswith(".tif")]
        if tifs:
            print(f"  [skip] Already extracted {len(tifs)} frames in {entity_dir}")
            return entity_dir

    os.makedirs(entity_dir, exist_ok=True)

    if file_path.endswith(".tgz") or file_path.endswith(".tar.gz"):
        # Extract .tgz archive (KH-9 multi-frame strips)
        print(f"  Extracting {os.path.basename(file_path)} ...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=entity_dir, **_TAR_FILTER)
        tifs = sorted(f for f in os.listdir(entity_dir) if f.lower().endswith(".tif"))
        print(f"  Extracted {len(tifs)} frames")

    elif file_path.endswith(".tif"):
        # Check if the .tif is actually gzip-compressed (USGS ships some this way)
        if _is_gzip(file_path):
            out_name = os.path.basename(file_path)
            out_path = os.path.join(entity_dir, out_name)
            if not os.path.exists(out_path):
                print(f"  Decompressing gzip-compressed TIF: {os.path.basename(file_path)}")
                # Check if it's a tar.gz containing TIFs
                try:
                    with tarfile.open(file_path, "r:gz") as tar:
                        members = tar.getnames()
                        tif_members = [m for m in members if m.lower().endswith(".tif")]
                        if tif_members:
                            tar.extractall(path=entity_dir, **_TAR_FILTER)
                            tifs = sorted(f for f in os.listdir(entity_dir)
                                          if f.lower().endswith(".tif"))
                            print(f"  Extracted {len(tifs)} frames from compressed archive")
                            return entity_dir
                except tarfile.TarError:
                    pass  # Not a tar archive, just a gzip-compressed single file

                # Plain gzip-compressed TIF
                with gzip.open(file_path, "rb") as f_in:
                    with open(out_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  Decompressed: {os.path.basename(out_path)}")
        else:
            # Genuine uncompressed TIF — symlink into entity dir
            link_name = os.path.join(entity_dir, os.path.basename(file_path))
            if not os.path.exists(link_name):
                abs_src = os.path.abspath(file_path)
                os.symlink(abs_src, link_name)
            print(f"  Linked single TIF: {os.path.basename(file_path)}")

    else:
        raise ValueError(f"Unknown archive format: {file_path}")

    # Validate extracted TIFFs
    tifs = [os.path.join(entity_dir, f) for f in os.listdir(entity_dir)
            if f.lower().endswith(".tif")]
    invalid = [t for t in tifs if not _validate_tiff(t)]
    if invalid:
        for t in invalid:
            print(f"  WARNING: Removing invalid TIFF: {os.path.basename(t)}")
            os.remove(t)
        remaining = [t for t in tifs if t not in invalid]
        if not remaining:
            raise RuntimeError(f"All extracted TIFFs are invalid for {entity_id}")

    return entity_dir


def list_frames(entity_dir: str) -> list:
    """List TIF frames in an extracted entity directory, sorted alphabetically.

    Excludes generated intermediate files (rotated copies, sub-frame splits).
    """
    if not os.path.exists(entity_dir):
        return []
    exclude_patterns = ("_rot180", "_rot90", "_rot270", "_sub", "_verified_rot")
    return sorted(
        os.path.join(entity_dir, f)
        for f in os.listdir(entity_dir)
        if f.lower().endswith(".tif") and not any(p in f for p in exclude_patterns)
    )
