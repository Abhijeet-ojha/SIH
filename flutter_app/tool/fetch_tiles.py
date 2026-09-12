"""Download an offline raster basemap for ONE demo area and bundle it into the app.

Why this exists
---------------
The map used to be a bare graticule, on the argument that shipping a tile server would
undercut a claim of working without infrastructure. That argument was wrong: needing a
tile server at RUN time would undercut it, needing one ONCE at build time does not. Tiles
baked into the APK are exactly as offline as a drawn grid, and a judge can finally see
which street the track is on.

Deliberately small. This fetches a couple of hundred tiles for one named area, not a
country. OpenStreetMap's tile usage policy forbids bulk downloading, and this script is
rate-limited and capped so it stays a legitimate small-area fetch. For anything larger,
use a commercial provider or run your own renderer.

Usage
-----
    python tool/fetch_tiles.py --lat 12.9716 --lon 77.5946 --radius-km 1.2 --name "MG Road"

Then rebuild: the tiles land in assets/tiles/ and are picked up by pubspec.yaml.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# OSM asks for a real identifying User-Agent. An anonymous or spoofed one gets blocked,
# correctly.
USER_AGENT = (
    "NavPulse-SIH-Prototype/1.0 (offline demo basemap; "
    "https://github.com/SIH-dead-reckoning)"
)

# A hard ceiling, so a mistyped radius cannot turn this into a bulk scrape.
MAX_TILES = 400
REQUEST_INTERVAL_S = 0.25


def deg2tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2**z
    lat_rad = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile2deg(x: int, y: int, z: int) -> tuple[float, float]:
    """North-west corner of tile (x, y)."""
    n = 2**z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def plan(lat: float, lon: float, radius_km: float, zooms: list[int]):
    """Tile ranges covering a square around the centre, at each zoom."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    jobs = []
    for z in zooms:
        x0, y0 = deg2tile(lat + dlat, lon - dlon, z)  # north-west
        x1, y1 = deg2tile(lat - dlat, lon + dlon, z)  # south-east
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                jobs.append((z, x, y))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius-km", type=float, default=1.0)
    ap.add_argument("--zoom", type=int, nargs="+", default=[15, 16, 17])
    ap.add_argument("--name", default="Demo area")
    ap.add_argument("--out", default="assets/tiles")
    args = ap.parse_args()

    jobs = plan(args.lat, args.lon, args.radius_km, args.zoom)
    if len(jobs) > MAX_TILES:
        print(
            f"{len(jobs)} tiles exceeds the {MAX_TILES}-tile cap. Reduce --radius-km or "
            f"drop the highest zoom. This cap exists so the script stays within OSM's "
            f"tile usage policy."
        )
        return 1

    print(f"{len(jobs)} tiles at zoom {args.zoom} around {args.lat}, {args.lon}")
    entries = []
    fetched = skipped = failed = 0

    for z, x, y in jobs:
        rel = f"{z}/{x}/{y}.png"
        path = os.path.join(args.out, rel)
        entries.append({"z": z, "x": x, "y": y})
        if os.path.exists(path) and os.path.getsize(path) > 0:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(
            TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            fetched += 1
        except (urllib.error.URLError, OSError) as e:
            # A missing tile is survivable - the app draws its grid where coverage is
            # absent - so one failure must not abort the whole fetch.
            print(f"  failed {rel}: {e}")
            failed += 1
        time.sleep(REQUEST_INTERVAL_S)

    manifest = {
        "name": args.name,
        "attribution": "(c) OpenStreetMap contributors",
        "center": {"lat": args.lat, "lon": args.lon},
        "radius_km": args.radius_km,
        "tiles": entries,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_mb = sum(
        os.path.getsize(os.path.join(args.out, f"{e['z']}/{e['x']}/{e['y']}.png"))
        for e in entries
        if os.path.exists(os.path.join(args.out, f"{e['z']}/{e['x']}/{e['y']}.png"))
    ) / 1e6
    print(f"fetched {fetched}, cached {skipped}, failed {failed}  ({total_mb:.1f} MB)")
    print("Attribution is mandatory: the app renders '(c) OpenStreetMap contributors'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
