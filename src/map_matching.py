"""
src/map_matching.py
Hidden Markov map matching (Newson & Krumm, "Hidden Markov Map Matching Through Noise and
Sparseness", ACM SIGSPATIAL 2009), applied to the open-loop blackout trajectory.

Why this is worth more than any model-architecture change: during a GPS outage the filter
drifts freely in two dimensions, but the vehicle is not free - it is on a road. Constraining
the trajectory to the road graph collapses the cross-road component of the error entirely
and leaves only along-road error. A 500 m free drift becomes a metres-level position
uncertainty along a known road.

Deliberately not a routing engine: an Overpass extract is fetched once, cached to disk, and
matched against. No new runtime dependency.
"""

import os
import json
import math
import numpy as np
from typing import List, Tuple, Optional, Dict

# ponytail: transitions are scored on straight-line candidate distance rather than
# shortest-path distance through the graph. Full Newson-Krumm runs a Dijkstra per candidate
# pair; the approximation is standard and holds while GPS steps are short relative to block
# size. Upgrade to on-graph routing if matching degrades at junctions.
DEFAULT_SIGMA_M = 8.0      # GPS/DR cross-track noise; sets how hard the emission pulls
DEFAULT_BETA_M = 12.0      # transition tolerance; larger = more willing to jump roads
MAX_CANDIDATES = 8
WAY_SWITCH_PENALTY = 4.0   # log-likelihood cost of changing road; see match_trajectory
MIN_MATCH_STEP_M = 25.0    # decimation spacing; see match_trajectory for why this matters
MAX_MATCH_RADIUS_M = 80.0

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DRIVABLE = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
            "residential", "service", "motorway_link", "trunk_link", "primary_link",
            "secondary_link", "tertiary_link")


class RoadGraph:
    """Road centrelines as polylines in a local ENU frame (metres)."""

    def __init__(self, ways: List[np.ndarray]):
        # Each way is an (n, 2) array of ENU points. Explode into segments once so
        # matching is a flat nearest-segment search rather than nested loops.
        self.ways = [np.asarray(w, dtype=float) for w in ways if len(w) >= 2]
        segs_a, segs_b, way_id = [], [], []
        for wi, w in enumerate(self.ways):
            segs_a.append(w[:-1])
            segs_b.append(w[1:])
            way_id.append(np.full(len(w) - 1, wi))
        if segs_a:
            self.seg_a = np.vstack(segs_a)
            self.seg_b = np.vstack(segs_b)
            self.seg_way = np.concatenate(way_id)
        else:
            self.seg_a = np.zeros((0, 2))
            self.seg_b = np.zeros((0, 2))
            self.seg_way = np.zeros(0, dtype=int)

    def __len__(self):
        return len(self.seg_a)

    def project(self, pt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perpendicular projection of pt onto every segment.
        Returns (projected points (n,2), distances (n,)).
        """
        ab = self.seg_b - self.seg_a
        denom = np.einsum("ij,ij->i", ab, ab)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        t = np.einsum("ij,ij->i", pt - self.seg_a, ab) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = self.seg_a + t[:, None] * ab
        return proj, np.linalg.norm(proj - pt, axis=1)

    def candidates(self, pt: np.ndarray, k: int = MAX_CANDIDATES,
                   radius: float = MAX_MATCH_RADIUS_M):
        """The k nearest road points within radius: [(point, distance, segment index)]."""
        if len(self) == 0:
            return []
        proj, dist = self.project(np.asarray(pt, dtype=float))
        order = np.argsort(dist)[:k]
        return [(proj[i], float(dist[i]), int(i)) for i in order if dist[i] <= radius]

    @classmethod
    def from_geojson(cls, path: str, lat0: float, lon0: float) -> "RoadGraph":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        ways = []
        for el in raw.get("elements", []):
            geom = el.get("geometry")
            if not geom:
                continue
            lat = np.array([g["lat"] for g in geom])
            lon = np.array([g["lon"] for g in geom])
            ways.append(np.column_stack(_enu(lat, lon, lat0, lon0)))
        return cls(ways)


def _enu(lat, lon, lat0, lon0, r_earth=6371000.0):
    """Matches geodetic_to_enu in data_loader so both frames agree."""
    x = r_earth * (np.deg2rad(lon) - np.deg2rad(lon0)) * np.cos(np.deg2rad(lat0))
    y = r_earth * (np.deg2rad(lat) - np.deg2rad(lat0))
    return x, y


def fetch_osm_roads(min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                    cache_path: str, timeout: int = 90) -> str:
    """
    Fetch drivable roads in the bbox from Overpass, once, and cache to disk.

    Network access happens here and nowhere else - everything downstream reads the cache,
    so a demo never depends on Overpass being up. Commit the cached extract for the
    routes you demo on.
    """
    if os.path.exists(cache_path):
        return cache_path

    import urllib.request
    import urllib.parse

    query = (
        f"[out:json][timeout:{timeout}];"
        f"(way[\"highway\"~\"^({'|'.join(DRIVABLE)})$\"]"
        f"({min_lat},{min_lon},{max_lat},{max_lon}););"
        f"out geom;"
    )
    req = urllib.request.Request(
        OVERPASS_URL,
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "SIH-PS168-dead-reckoning/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    return cache_path


def match_trajectory(
    xs: np.ndarray,
    ys: np.ndarray,
    graph: RoadGraph,
    sigma_m: float = DEFAULT_SIGMA_M,
    beta_m: float = DEFAULT_BETA_M,
    way_switch_penalty: float = WAY_SWITCH_PENALTY,
    min_step_m: float = MIN_MATCH_STEP_M,
) -> Dict[str, np.ndarray]:
    """
    Public entry point. Decimates to >= min_step_m spacing, matches, then interpolates the
    matched path back onto the original timestamps.

    The decimation is not an optimisation, it is what makes the matcher work. The
    transition term compares how far the vehicle appears to have moved against how far it
    could have moved along each candidate road. At 4 m spacing that comparison carries
    almost no information - every candidate is consistent with a 4 m step, including one on
    a perpendicular road the vehicle is merely crossing - and the matcher happily walks a
    northbound drive along successive east-west streets. Newson & Krumm assume sparse
    fixes (tens of seconds apart) for exactly this reason.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    if n == 0:
        return {"x": xs.copy(), "y": ys.copy(), "road_distance_m": np.zeros(0),
                "matched": np.zeros(0, dtype=bool)}

    # Pick indices at least min_step_m apart along the path.
    keep = [0]
    for i in range(1, n):
        if math.hypot(xs[i] - xs[keep[-1]], ys[i] - ys[keep[-1]]) >= min_step_m:
            keep.append(i)
    if keep[-1] != n - 1:
        keep.append(n - 1)
    keep = np.array(keep)

    if len(keep) < 3:
        return _match_dense(xs, ys, graph, sigma_m, beta_m, way_switch_penalty)

    sub = _match_dense(xs[keep], ys[keep], graph, sigma_m, beta_m, way_switch_penalty)

    # Interpolate the matched path back to every original sample, by path arc length so
    # the reconstruction follows the road rather than cutting corners.
    s_full = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    s_sub = s_full[keep]
    out_x = np.interp(s_full, s_sub, sub["x"])
    out_y = np.interp(s_full, s_sub, sub["y"])
    road_d = np.interp(s_full, s_sub, np.nan_to_num(sub["road_distance_m"], nan=0.0))
    matched = np.interp(s_full, s_sub, sub["matched"].astype(float)) > 0.5
    return {"x": out_x, "y": out_y, "road_distance_m": road_d, "matched": matched}


def _match_dense(
    xs: np.ndarray,
    ys: np.ndarray,
    graph: RoadGraph,
    sigma_m: float = DEFAULT_SIGMA_M,
    beta_m: float = DEFAULT_BETA_M,
    way_switch_penalty: float = WAY_SWITCH_PENALTY,
) -> Dict[str, np.ndarray]:
    """
    Viterbi over per-observation road candidates.

    Emission:   log N(distance to road; 0, sigma)  - roads near the estimate are likelier.
    Transition: -|candidate step - observed step| / beta - the vehicle's movement along the
                road network should match how far the trajectory says it moved. This is what
                stops the matcher teleporting between parallel roads.

    Returns matched x/y plus the per-step road distance. Falls back to the input point
    wherever no road lies within MAX_MATCH_RADIUS_M, so the output is always the same
    length as the input.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    out_x, out_y = xs.copy(), ys.copy()
    if len(graph) == 0 or n == 0:
        return {"x": out_x, "y": out_y, "road_distance_m": np.full(n, np.nan),
                "matched": np.zeros(n, dtype=bool)}

    cands = [graph.candidates(np.array([xs[i], ys[i]])) for i in range(n)]
    matched = np.zeros(n, dtype=bool)
    road_dist = np.full(n, np.nan)

    # Points with no road within the radius break the chain: the vehicle has left the
    # mapped network, and forcing a match there would be worse than leaving the estimate
    # alone. Solve each contiguous run of matchable points as its own Viterbi.
    runs, start = [], None
    for i in range(n + 1):
        has = i < n and bool(cands[i])
        if has and start is None:
            start = i
        elif not has and start is not None:
            runs.append((start, i))
            start = None

    for lo, hi in runs:
        m = hi - lo
        log_emit = [np.array([-0.5 * (d / sigma_m) ** 2 for _, d, _ in cands[i]])
                    for i in range(lo, hi)]
        back = [np.full(len(cands[i]), -1, dtype=int) for i in range(lo, hi)]
        score = log_emit[0].copy()

        for j in range(1, m):
            i = lo + j
            prev_pts = np.array([p for p, _, _ in cands[i - 1]])
            cur_pts = np.array([p for p, _, _ in cands[i]])
            observed = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            step = np.linalg.norm(cur_pts[None, :, :] - prev_pts[:, None, :], axis=2)
            trans = -np.abs(step - observed) / beta_m

            # Staying on the same road is strongly preferred. Without this the matcher
            # hops between whichever road happens to be nearest, which on a grid means a
            # northbound drive gets matched onto successive east-west streets - each one
            # locally closer than the road it is actually on. True Newson-Krumm rules that
            # out via shortest-path route distance (you cannot reach the next street
            # without driving to a junction first); this penalty buys most of that for
            # four lines. A junction switch still happens when the emission clearly
            # supports it.
            prev_way = np.array([graph.seg_way[si] for _, _, si in cands[i - 1]])
            cur_way = np.array([graph.seg_way[si] for _, _, si in cands[i]])
            trans = trans + np.where(prev_way[:, None] == cur_way[None, :],
                                     0.0, -way_switch_penalty)

            total = score[:, None] + trans          # (prev, cur)
            best_prev = np.argmax(total, axis=0)
            score = total[best_prev, np.arange(len(cur_pts))] + log_emit[j]
            back[j] = best_prev

        k = int(np.argmax(score))
        for j in range(m - 1, -1, -1):
            pt, d, _ = cands[lo + j][k]
            out_x[lo + j], out_y[lo + j] = pt[0], pt[1]
            road_dist[lo + j] = d
            matched[lo + j] = True
            if j > 0:
                k = int(back[j][k])

    return {"x": out_x, "y": out_y, "road_distance_m": road_dist, "matched": matched}


def grid_graph(spacing: float = 100.0, extent: float = 1200.0) -> RoadGraph:
    """A synthetic city grid, for tests and for demos without an Overpass fetch."""
    ways = []
    coords = np.arange(0.0, extent + spacing, spacing)
    for c in coords:
        ways.append(np.column_stack([np.full_like(coords, c), coords]))  # N-S
        ways.append(np.column_stack([coords, np.full_like(coords, c)]))  # E-W
    return RoadGraph(ways)


def demo():
    """
    Self-check on the realistic failure mode.

    Open-loop DR error is dominated by accumulated *heading* error, which displaces the
    trajectory sideways off the road. That cross-track component is exactly what a road
    constraint removes. Along-track error - being 40 m further down the correct road than
    you think - is NOT recoverable by map matching, and claiming otherwise would be the
    same kind of overstatement this repo is being cleaned of.
    """
    rng = np.random.default_rng(5)
    graph = grid_graph()

    # Drive east along y=300 for 600 m, then north along x=700 for 600 m.
    leg1 = np.column_stack([np.linspace(100, 700, 150), np.full(150, 300.0)])
    leg2 = np.column_stack([np.full(150, 700.0), np.linspace(300, 900, 150)])
    truth = np.vstack([leg1, leg2])

    # Heading drift: rotate each step by a slowly growing yaw error, then re-integrate.
    steps = np.diff(truth, axis=0, prepend=truth[:1])
    yaw_err = np.linspace(0.0, 0.09, len(truth))  # ~5 deg by the end, a mild gyro bias
    c, s = np.cos(yaw_err), np.sin(yaw_err)
    rot = np.column_stack([c * steps[:, 0] - s * steps[:, 1],
                           s * steps[:, 0] + c * steps[:, 1]])
    drifted = truth[0] + np.cumsum(rot, axis=0) + rng.normal(0, 2.0, truth.shape)

    res = match_trajectory(drifted[:, 0], drifted[:, 1], graph)
    matched = np.column_stack([res["x"], res["y"]])

    # Cross-track error = distance to the true road, which here is axis-aligned.
    def cross_track(p):
        return np.where(np.arange(len(p)) < 150, np.abs(p[:, 1] - 300.0), np.abs(p[:, 0] - 700.0))

    ct_before = float(np.mean(cross_track(drifted)))
    ct_after = float(np.mean(cross_track(matched)))
    err_before = float(np.mean(np.linalg.norm(drifted - truth, axis=1)))
    err_after = float(np.mean(np.linalg.norm(matched - truth, axis=1)))

    assert res["matched"].all(), "every point should have matched on a dense grid"
    assert ct_after < ct_before * 0.25, \
        f"cross-track not corrected: {ct_before:.1f} m -> {ct_after:.1f} m"
    print(f"map_matching demo OK: cross-track {ct_before:.1f} m -> {ct_after:.1f} m, "
          f"total {err_before:.1f} m -> {err_after:.1f} m "
          f"(along-track error is not recoverable by map matching)")


if __name__ == "__main__":
    demo()
