"""One-shot bootstrap: fetch Space-Track TLE history for KH missions.

Populates ``data/kh_tle/<mission_id>.tle`` so
``preprocess.mission_altitude.altitude_m_at`` can propagate a per-frame
altitude prior instead of falling back to the catalog nominal.

Credentials
-----------
Reads ``SPACETRACK_USER`` / ``SPACETRACK_PASS`` from the environment, or from
a ``~/.space-track.env`` file shaped like::

    SPACETRACK_USER=...
    SPACETRACK_PASS=...

A free account at https://www.space-track.org is required.

NORAD ID backfill
-----------------
The catalog ships with ``norad_cat_id`` populated for known KH-9 missions.
Historical NORAD catalog numbers for additional missions can be cross-checked
via Jonathan McDowell's satcat index at https://planet4589.org/space/gcat/
and added to ``data/kh_missions.yaml`` before re-running this script.

Rate limits
-----------
Space-Track caps at 300 requests/hour and 30 requests/minute; this script
makes one request per mission (bulk TLE history for that NORAD ID) with a
3 s sleep between requests, well under both limits.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "kh_missions.yaml"
TLE_DIR = REPO_ROOT / "data" / "kh_tle"

SPACETRACK_BASE = "https://www.space-track.org"
LOGIN_URL = f"{SPACETRACK_BASE}/ajaxauth/login"


def _load_env_file(path: Path) -> dict:
    env = {}
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _load_credentials() -> tuple[str, str]:
    user = os.environ.get("SPACETRACK_USER")
    pw = os.environ.get("SPACETRACK_PASS")
    if not (user and pw):
        env = _load_env_file(Path.home() / ".space-track.env")
        user = user or env.get("SPACETRACK_USER")
        pw = pw or env.get("SPACETRACK_PASS")
    if not (user and pw):
        print(
            "ERROR: Space-Track credentials not found.\n"
            "Set SPACETRACK_USER / SPACETRACK_PASS, or create ~/.space-track.env "
            "with those two lines. Register at https://www.space-track.org.",
            file=sys.stderr,
        )
        sys.exit(2)
    return user, pw


def _load_catalog() -> dict:
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml not installed. Install with `pip install pyyaml`.",
              file=sys.stderr)
        sys.exit(2)
    if not CATALOG_PATH.is_file():
        print(f"ERROR: mission catalog not found at {CATALOG_PATH}", file=sys.stderr)
        sys.exit(2)
    with open(CATALOG_PATH, "r") as fh:
        return yaml.safe_load(fh) or {}


def _mission_filter(catalog: dict, selected: Iterable[str] | None) -> list[tuple[str, dict]]:
    missions = catalog.get("missions") or {}
    if selected:
        return [(mid, missions[mid]) for mid in selected if mid in missions]
    return sorted(missions.items())


def _build_opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def _login(opener: urllib.request.OpenerDirector, user: str, pw: str) -> None:
    data = urllib.parse.urlencode({"identity": user, "password": pw}).encode("utf-8")
    req = urllib.request.Request(LOGIN_URL, data=data, method="POST")
    try:
        with opener.open(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"ERROR: Space-Track login failed ({e.code}): {e.read()[:200]}",
              file=sys.stderr)
        sys.exit(3)
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach Space-Track ({e}): check connectivity.",
              file=sys.stderr)
        sys.exit(3)
    if "Failed" in body or "invalid" in body.lower():
        print(f"ERROR: Space-Track login body indicates failure: {body[:200]}",
              file=sys.stderr)
        sys.exit(3)


def _lookup_satcat(opener: urllib.request.OpenerDirector,
                    cospar_id: str) -> dict | None:
    """Fetch the satcat entry for a COSPAR / INTLDES designator."""
    url = (
        f"{SPACETRACK_BASE}/basicspacedata/query/class/satcat/"
        f"INTLDES/{cospar_id}/format/json"
    )
    try:
        with opener.open(url, timeout=30) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as e:
        print(f"  [satcat] {cospar_id}: {e}", file=sys.stderr)
        return None
    if not isinstance(data, list) or not data:
        return None
    return data[0]


def _fetch_tle_history(opener: urllib.request.OpenerDirector,
                       norad_cat_id: int) -> str | None:
    """Fetch the full TLE archive (two-line format) from gp_history.

    Space-Track retired the `tle` class; `gp_history` is the canonical
    replacement and supports ``format/tle`` for the classic two-line
    rendering. All KH-4/4A/4B/7/9 satellites have long since decayed and
    are fully archived here.
    """
    url = (
        f"{SPACETRACK_BASE}/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{int(norad_cat_id)}/"
        f"orderby/EPOCH%20asc/format/tle"
    )
    try:
        with opener.open(url, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"  [fetch] HTTP {e.code} for NORAD {norad_cat_id}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"  [fetch] URL error for NORAD {norad_cat_id}: {e}", file=sys.stderr)
        return None
    if not text.strip():
        return None
    return text


def _strip_banner(text: str) -> str:
    """Drop any leading lines that aren't the start of a TLE pair."""
    lines = text.splitlines()
    out: list[str] = []
    started = False
    for ln in lines:
        if not started and not ln.startswith("1 "):
            continue
        started = True
        out.append(ln)
    return "\n".join(out) + ("\n" if out else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--missions", nargs="*", default=None,
        help="Mission IDs to fetch (e.g. 1212 1213 1022). Omit to fetch all "
             "catalog entries with a norad_cat_id.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if the .tle file already exists.",
    )
    ap.add_argument(
        "--sleep-s", type=float, default=3.0,
        help="Sleep between requests (s). Default 3.0 to stay under Space-Track limits.",
    )
    args = ap.parse_args()

    user, pw = _load_credentials()
    catalog = _load_catalog()

    TLE_DIR.mkdir(parents=True, exist_ok=True)

    entries = _mission_filter(catalog, args.missions)
    if not entries:
        print("Nothing to fetch (no matching catalog entries).")
        return 0

    opener = _build_opener()
    _login(opener, user, pw)
    if True:
        ok = 0
        skipped = 0
        no_id = 0
        failed = 0
        for mid, entry in entries:
            if (entry.get("tle_quality") == "sparse") and not args.force:
                skipped += 1
                continue
            norad = entry.get("norad_cat_id")
            cospar = entry.get("cospar_id")
            # Skip failed launches and pre-1963 Greek-letter COSPAR IDs that
            # Space-Track doesn't accept as OBJECT_ID.
            if cospar and (str(cospar).upper().startswith("F") or "-F" in str(cospar)):
                print(f"  [skip] {mid}: failed launch ({cospar})")
                skipped += 1
                continue
            out_path = TLE_DIR / (entry.get("tle_file") or f"{mid}.tle")
            if out_path.exists() and not args.force:
                print(f"  [skip] {mid} → {out_path.name} already present (use --force to refetch)")
                skipped += 1
                continue
            if norad is None and cospar is None:
                print(f"  [skip] {mid}: no norad_cat_id or cospar_id in catalog")
                no_id += 1
                continue
            # Always grab the satcat entry first — it also supplies the
            # NORAD ID when the catalog doesn't have one, and its perigee /
            # apogee / decay data is a useful sidecar for missions whose
            # gp_history returns empty (Space-Track coverage is sparse for
            # some 1960s launches).
            import json
            satcat_path = TLE_DIR / f"{mid}.satcat.json"
            if not satcat_path.exists() or args.force:
                sat = _lookup_satcat(opener, str(cospar)) if cospar else None
                if sat is not None:
                    satcat_path.write_text(json.dumps(sat, indent=2) + "\n")
                time.sleep(1.0)  # extra courtesy pause between the two calls

            if norad is None and satcat_path.exists():
                try:
                    sat = json.loads(satcat_path.read_text())
                    norad = int(sat.get("NORAD_CAT_ID"))
                    print(f"  [resolve] {mid}: {cospar} → NORAD {norad} (from satcat)")
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass

            if norad is None:
                print(f"  [skip] {mid}: no NORAD resolvable for {cospar}")
                no_id += 1
                continue

            print(f"  [fetch] {mid} NORAD={norad} → {out_path.name}")
            text = _fetch_tle_history(opener, int(norad))
            if not text:
                print(f"  [fetch] {mid}: gp_history empty — satcat sidecar only")
                failed += 1
                time.sleep(args.sleep_s)
                continue
            cleaned = _strip_banner(text)
            if not cleaned.strip():
                print(f"  [fetch] {mid}: Space-Track returned no TLEs")
                failed += 1
                time.sleep(args.sleep_s)
                continue
            out_path.write_text(cleaned)
            n_pairs = sum(1 for ln in cleaned.splitlines() if ln.startswith("1 "))
            print(f"  [fetch] {mid}: {n_pairs} TLE pairs written ({out_path.stat().st_size:,} bytes)")
            ok += 1
            time.sleep(args.sleep_s)

    print(f"\nDone. fetched={ok} skipped={skipped} no_id={no_id} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
