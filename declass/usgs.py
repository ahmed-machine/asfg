"""USGS M2M API client for downloading declassified satellite imagery."""

import json
import os
import sys
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
        # Build download list
        downloads = []
        for eid in entity_ids:
            downloads.append({
                "entityId": eid,
                "productId": "5e83d0b84df8d8c2",  # Standard product
            })

        # Request downloads
        result = self._request("download-request", {
            "downloads": downloads,
            "datasetName": dataset_alias,
        })

        available = result.get("availableDownloads", [])
        preparing = result.get("preparingDownloads", [])

        if preparing:
            # Need to poll for preparation
            print(f"  {len(preparing)} downloads preparing, polling...")
            label = result.get("downloadOptions", result.get("label", ""))
            available.extend(self._poll_downloads(preparing))

        return available

    def _poll_downloads(self, preparing: list, max_wait: int = 600) -> list:
        """Poll download-retrieve until files are ready."""
        ready = []
        waited = 0
        poll_interval = 10

        while preparing and waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval

            result = self._request("download-retrieve", {
                "label": preparing[0].get("label", ""),
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

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
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
