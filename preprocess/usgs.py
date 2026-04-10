"""USGS M2M API client, downloading, and archive extraction for declassified satellite imagery."""

import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import urllib.error

M2M_BASE = "https://m2m.cr.usgs.gov/api/api/json/stable/"
CREDENTIALS_PATH = os.path.expanduser("~/.usgs/credentials.json")

# EarthExplorer dataset aliases
DATASET_ALIASES = {
    "corona2": "corona2",
    "declassii": "declassii",
    "declassiii": "declassiii",
}


class USGSClient:
    """Client for the USGS M2M API."""

    def __init__(self):
        self.api_key = None

    def _request(self, endpoint: str, payload: dict = None) -> dict:
        """Make an authenticated M2M API request."""
        url = M2M_BASE + endpoint
        data = json.dumps(payload or {}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Auth-Token"] = self.api_key

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Handle USGS maintenance redirects
            if e.code in (307, 308):
                redirect_url = e.headers.get("Location", "")
                if "maintenance" in redirect_url.lower():
                    raise RuntimeError(
                        f"USGS M2M API is down for maintenance (redirecting to {redirect_url}). "
                        f"Try again later or use --skip-download with cached files.")
            body_text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"M2M API error {e.code} on {endpoint}: {body_text}")

        if body.get("errorCode"):
            raise RuntimeError(f"M2M API error: {body['errorCode']} — {body.get('errorMessage', '')}")
        return body.get("data")

    def login(self, username: str = None, token: str = None):
        """Authenticate with USGS M2M API.

        Credentials can be passed directly or loaded from ~/.usgs/credentials.json:
        {"username": "...", "token": "..."}
        """
        if not username or not token:
            username, token = _load_credentials()

        result = self._request("login-token", {
            "username": username,
            "token": token,
        })
        self.api_key = result
        print(f"  USGS M2M: Authenticated as {username}")

    def logout(self):
        """End the API session."""
        if self.api_key:
            try:
                self._request("logout")
            except Exception:
                pass
            self.api_key = None

    def request_downloads(self, dataset_alias: str, entity_ids: list) -> list:
        """Request download URLs for a list of entity IDs.

        Returns list of dicts with 'entityId', 'url', 'filesize' keys.
        """
        # Look up the correct product ID for each entity via download-options
        product_ids = {}
        options = self._request("download-options", {
            "datasetName": dataset_alias,
            "entityIds": entity_ids,
        })
        for opt in (options or []):
            eid = opt.get("entityId", "")
            pid = opt.get("id", "")
            if eid and pid:
                product_ids[eid] = pid

        # Build download list
        downloads = []
        for eid in entity_ids:
            pid = product_ids.get(eid)
            if not pid:
                print(f"  WARNING: No product ID found for {eid}, skipping download")
                continue
            downloads.append({
                "entityId": eid,
                "productId": pid,
            })

        # Request downloads
        result = self._request("download-request", {
            "downloads": downloads,
            "datasetName": dataset_alias,
        })

        available = result.get("availableDownloads", [])
        preparing = result.get("preparingDownloads", [])

        if preparing:
            # Check if preparing downloads already have usable URLs
            # (the staging URL is often immediately usable)
            usable = [p for p in preparing if p.get("url")]
            remaining = [p for p in preparing if not p.get("url")]

            if usable:
                print(f"  {len(usable)} downloads have staging URLs")
                available.extend(usable)

            if remaining:
                print(f"  {len(remaining)} downloads preparing, polling...")
                # Extract label from newRecords in the response
                new_records = result.get("newRecords", {})
                label = ""
                if new_records:
                    # Label is the value (all records share the same label)
                    label = next(iter(new_records.values()), "")
                available.extend(self._poll_downloads(remaining, label=label))

        return available

    def _poll_downloads(self, preparing: list, label: str = "",
                        max_wait: int = 600) -> list:
        """Poll download-retrieve until files are ready."""
        ready = []
        waited = 0
        poll_interval = 10

        while preparing and waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval

            result = self._request("download-retrieve", {
                "label": label,
            })

            if result and result.get("available"):
                for item in result["available"]:
                    ready.append(item)
                # Remove from preparing
                ready_ids = {r.get("entityId") for r in ready}
                preparing = [p for p in preparing if p.get("entityId") not in ready_ids]

            if preparing:
                print(f"  Still preparing {len(preparing)} downloads... ({waited}s)")

        if preparing:
            print(f"  WARNING: {len(preparing)} downloads timed out after {max_wait}s")

        return ready

    def download_file(self, url: str, output_path: str, filesize: int = 0):
        """Download a file with progress reporting."""
        if os.path.exists(output_path):
            existing_size = os.path.getsize(output_path)
            if filesize and existing_size >= filesize:
                print(f"  [skip] Already downloaded: {os.path.basename(output_path)}")
                return
            elif not filesize and existing_size > 0:
                print(f"  [skip] Already downloaded: {os.path.basename(output_path)}")
                return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_path = output_path + ".tmp"

        print(f"  Downloading: {os.path.basename(output_path)}", end="", flush=True)
        if filesize:
            print(f" ({filesize / 1024 / 1024:.0f} MB)", end="", flush=True)

        req = urllib.request.Request(url)
        if self.api_key:
            req.add_header("X-Auth-Token", self.api_key)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = urllib.request.urlopen(req, timeout=300)
                break
            except urllib.error.HTTPError as e:
                if e.code == 500 and attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"\n  Server error (500), retrying in {wait}s "
                          f"(attempt {attempt + 2}/{max_retries})...",
                          end="", flush=True)
                    import time as _time
                    _time.sleep(wait)
                    req = urllib.request.Request(url)
                    if self.api_key:
                        req.add_header("X-Auth-Token", self.api_key)
                    continue
                raise

        try:
            with resp:
                total = int(resp.headers.get("Content-Length", 0)) or filesize
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks

                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  Downloading: {os.path.basename(output_path)} "
                                  f"{downloaded / 1024 / 1024:.0f}/{total / 1024 / 1024:.0f} MB "
                                  f"({pct:.0f}%)", end="", flush=True)

            os.rename(tmp_path, output_path)
            print(f"\n  Downloaded: {output_path}")

        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def _load_credentials() -> tuple:
    """Load USGS credentials from ~/.usgs/credentials.json."""
    if not os.path.exists(CREDENTIALS_PATH):
        print(f"\nUSGS credentials not found at: {CREDENTIALS_PATH}")
        print(f"\nTo set up USGS M2M API access:")
        print(f"  1. Create an account at https://ers.cr.usgs.gov/register")
        print(f"  2. Generate an API token at https://m2m.cr.usgs.gov/")
        print(f"  3. Create {CREDENTIALS_PATH} with:")
        print(f'     {{"username": "your_username", "token": "your_api_token"}}')
        print(f"\nAlternatively, use --skip-download and place files manually.")
        sys.exit(1)

    with open(CREDENTIALS_PATH) as f:
        creds = json.load(f)

    username = creds.get("username")
    token = creds.get("token")
    if not username or not token:
        print(f"ERROR: {CREDENTIALS_PATH} must contain 'username' and 'token' fields")
        sys.exit(1)

    return username, token


def fetch_scene_corners(client: USGSClient, dataset: str, entity_id: str) -> dict:
    """Fetch corner coordinates for a scene from the M2M API.

    Args:
        client: Authenticated USGSClient.
        dataset: Dataset alias (e.g. "declassiii").
        entity_id: Entity ID (e.g. "D3C1213-200346F002").

    Returns:
        Dict with NW, NE, SE, SW keys, each (lat, lon) tuple.
    """
    result = client._request("scene-metadata", {
        "datasetName": dataset,
        "entityId": entity_id,
    })

    # Extract corner coordinates from metadata fields
    fields = {}
    for item in result.get("metadata", []):
        fields[item["fieldName"]] = item["value"]

    # Field names match the CSV column conventions
    corner_field_map = {
        "NW": ("NW Corner Lat dec", "NW Corner Long dec"),
        "NE": ("NE Corner Lat dec", "NE Corner Long dec"),
        "SE": ("SE Corner Lat dec", "SE Corner Long dec"),
        "SW": ("SW Corner Lat dec", "SW Corner Long dec"),
    }

    corners = {}
    for corner, (lat_field, lon_field) in corner_field_map.items():
        lat_val = fields.get(lat_field)
        lon_val = fields.get(lon_field)
        if lat_val is None or lon_val is None:
            raise RuntimeError(
                f"Missing corner field for {entity_id}: {lat_field} or {lon_field}. "
                f"Available fields: {sorted(fields.keys())}"
            )
        corners[corner] = (float(lat_val), float(lon_val))

    return corners


def fetch_corners_batch(dataset: str, entity_ids: list) -> dict:
    """Fetch corner coordinates for multiple entities.

    Args:
        dataset: Dataset alias (e.g. "declassiii").
        entity_ids: List of entity IDs.

    Returns:
        Dict mapping entity_id -> corners dict.
    """
    client = USGSClient()
    client.login()
    try:
        results = {}
        for eid in entity_ids:
            print(f"  Fetching metadata: {eid}")
            results[eid] = fetch_scene_corners(client, dataset, eid)
            corners = results[eid]
            print(f"    NW=({corners['NW'][0]:.3f}, {corners['NW'][1]:.3f}) "
                  f"SE=({corners['SE'][0]:.3f}, {corners['SE'][1]:.3f})")
        return results
    finally:
        client.logout()


def download_scenes(scenes: list, output_dir: str, skip_download: bool = False) -> dict:
    """Download imagery for a list of Scene objects.

    Returns dict mapping entity_id -> local file path.
    """
    downloads_dir = os.path.join(output_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    file_map = {}

    if skip_download:
        # Check for existing files
        for scene in scenes:
            eid = scene.entity_id
            for ext in (".tgz", ".tif"):
                path = os.path.join(downloads_dir, f"{eid}{ext}")
                if os.path.exists(path):
                    file_map[eid] = path
                    break
            if eid not in file_map:
                print(f"  WARNING: No downloaded file found for {eid}")
        return file_map

    # Group scenes by dataset for batch download
    from collections import defaultdict
    by_dataset = defaultdict(list)
    for scene in scenes:
        by_dataset[scene.camera_system.ee_dataset].append(scene)

    client = USGSClient()
    client.login()

    try:
        for dataset_alias, dataset_scenes in by_dataset.items():
            # Check which ones we already have
            need_download = []
            for scene in dataset_scenes:
                eid = scene.entity_id
                ext = ".tgz" if scene.camera_system.archive_format == "tgz" else ".tif"
                path = os.path.join(downloads_dir, f"{eid}{ext}")
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    file_map[eid] = path
                    print(f"  [skip] Already downloaded: {eid}")
                else:
                    need_download.append(scene)

            if not need_download:
                continue

            entity_ids = [s.entity_id for s in need_download]
            print(f"  Requesting {len(entity_ids)} downloads from {dataset_alias}...")

            available = client.request_downloads(dataset_alias, entity_ids)

            for item in available:
                eid = item.get("entityId", "")
                url = item.get("url", "")
                filesize = item.get("filesize", 0)

                if not url:
                    print(f"  WARNING: No download URL for {eid}")
                    continue

                # Determine extension from URL or camera system
                scene = next((s for s in need_download if s.entity_id == eid), None)
                if scene:
                    ext = ".tgz" if scene.camera_system.archive_format == "tgz" else ".tif"
                else:
                    ext = ".tgz" if url.endswith(".tgz") else ".tif"

                path = os.path.join(downloads_dir, f"{eid}{ext}")
                client.download_file(url, path, filesize)
                file_map[eid] = path

    finally:
        client.logout()

    return file_map


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

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
