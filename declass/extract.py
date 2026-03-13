"""Archive extraction for declassified satellite imagery."""

import os
import shutil
import tarfile


def extract_archive(file_path: str, output_dir: str, entity_id: str) -> str:
    """Extract a .tgz archive or handle a single .tif file.

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
            tar.extractall(path=entity_dir, filter="data")
        tifs = sorted(f for f in os.listdir(entity_dir) if f.lower().endswith(".tif"))
        print(f"  Extracted {len(tifs)} frames")

    elif file_path.endswith(".tif"):
        # Single TIF file (KH-4, KH-7) — symlink into entity dir
        link_name = os.path.join(entity_dir, os.path.basename(file_path))
        if not os.path.exists(link_name):
            # Use absolute path for symlink target
            abs_src = os.path.abspath(file_path)
            os.symlink(abs_src, link_name)
        print(f"  Linked single TIF: {os.path.basename(file_path)}")

    else:
        raise ValueError(f"Unknown archive format: {file_path}")

    return entity_dir


def list_frames(entity_dir: str) -> list:
    """List TIF frames in an extracted entity directory, sorted alphabetically."""
    if not os.path.exists(entity_dir):
        return []
    return sorted(
        os.path.join(entity_dir, f)
        for f in os.listdir(entity_dir)
        if f.lower().endswith(".tif") and "_rot180" not in f
    )
