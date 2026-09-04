"""
scripts/inspect_iovnbd_schema.py

Prints the REAL IO-VNBD schema from disk: headers, dtypes, sample rate, units and value
ranges, for one S- (smartphone) and one V- (vehicle CAN) file.

This exists because the previous loader was written against assumed column names and
assumed a 10 Hz uniform clock. Nothing about the real files should be guessed - run this,
read the output, then write loader code.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BASE = os.path.join(PROJECT_ROOT, "data", "IO-VNBD-repo",
                    "Synchronised V abd S datasets", "Categorised IOVNB Dataset")


def describe(path: str, label: str, max_cols: int = 40):
    print("=" * 78)
    print(f"{label}: {os.path.relpath(path, PROJECT_ROOT)}")
    print("=" * 78)
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"rows={len(df)}  cols={len(df.columns)}  file={os.path.getsize(path)/1e6:.1f} MB")
    print()
    print(f"{'#':>3} {'column':<44} {'dtype':<10} {'min':>12} {'max':>12} {'nan%':>6}")
    print("-" * 92)
    for i, c in enumerate(df.columns[:max_cols]):
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        nan_pct = 100.0 * num.isna().mean()
        lo = f"{num.min():.4g}" if num.notna().any() else "-"
        hi = f"{num.max():.4g}" if num.notna().any() else "-"
        print(f"{i:>3} {repr(c)[1:-1][:44]:<44} {str(s.dtype):<10} {lo:>12} {hi:>12} {nan_pct:>5.1f}%")
    return df


def timing(df: pd.DataFrame, col: str, scale: float, name: str):
    if col not in df.columns:
        print(f"  [{name}] column {col!r} absent")
        return None
    t = pd.to_numeric(df[col], errors="coerce").values.astype(float) * scale
    dt = np.diff(t)
    good = dt[np.isfinite(dt) & (dt > 0)]
    if len(good) == 0:
        print(f"  [{name}] no positive dt")
        return None
    print(f"  [{name}] span={t[-1]-t[0]:.1f}s  median dt={np.median(good):.4f}s "
          f"({1/np.median(good):.2f} Hz)  min={good.min():.4f}  max={good.max():.4f}  "
          f"n_gaps(>3x median)={int(np.sum(good > 3*np.median(good)))}")
    print(f"        uniform? duration == n*median_dt : "
          f"{abs((t[-1]-t[0]) - (len(t)-1)*np.median(good)) < 1e-6}")
    return t


def main():
    if not os.path.isdir(BASE):
        raise SystemExit(f"IO-VNBD not found at {BASE}. Run scripts/fetch_iovnbd.py first.")

    s_path = os.path.join(BASE, "S (Driver A)", "S3a", "S-S3a.csv")
    v_path = os.path.join(BASE, "S (Driver A)", "S3a", "V-S3a.csv")

    s = describe(s_path, "SMARTPHONE (S-)")
    print("\n-- timing --")
    timing(s, " TIME SINCE START (ms)", 1e-3, "S- TIME SINCE START (ms)")

    print()
    v = describe(v_path, "VEHICLE CAN (V-)")
    print("\n-- timing --")
    timing(v, " Time Since Start of Day (seconds)", 1.0, "V- Time Since Start of Day (s)")
    if " Sample period (seconds)" in v.columns:
        sp = pd.to_numeric(v[" Sample period (seconds)"], errors="coerce")
        print(f"  [V- Sample period column] median={sp.median():.4f}s  "
              f"min={sp.min():.4f}  max={sp.max():.4f}")

    print("\n" + "=" * 78)
    print("SYNCHRONISATION CHECK")
    print("=" * 78)
    print(f"S- rows={len(s)}   V- rows={len(v)}   equal={len(s) == len(v)}")

    # Speed sources, all converted to m/s, to see which agree.
    print("\n" + "=" * 78)
    print("SPEED SOURCES (m/s)")
    print("=" * 78)
    gps_kmh = pd.to_numeric(s[" GPS SPEED (Kmh)"], errors="coerce").values
    print(f"  S- 'GPS SPEED (Kmh)' raw : min={np.nanmin(gps_kmh):.2f} max={np.nanmax(gps_kmh):.2f}")
    print(f"      as km/h -> m/s       : max={np.nanmax(gps_kmh)/3.6:.2f} m/s "
          f"({np.nanmax(gps_kmh):.1f} km/h)")
    print(f"      as already m/s       : max={np.nanmax(gps_kmh):.2f} m/s "
          f"({np.nanmax(gps_kmh)*3.6:.1f} km/h)")

    ivs = pd.to_numeric(v[" Indicated Vehicle Speed (km/hr)"], errors="coerce").values
    print(f"  V- 'Indicated Vehicle Speed (km/hr)': max={np.nanmax(ivs):.2f} km/h "
          f"= {np.nanmax(ivs)/3.6:.2f} m/s")

    wheels = [c for c in v.columns if "Wheel Speed" in c]
    print(f"  V- wheel speed columns: {wheels}")
    if wheels:
        w = np.nanmean(np.column_stack(
            [pd.to_numeric(v[c], errors="coerce").values for c in wheels]), axis=1)
        print(f"      mean wheel rad/s: max={np.nanmax(w):.2f}")
        # Effective rolling radius from CAN speed: v = omega * r
        ivs_ms = ivs / 3.6
        m = np.isfinite(w) & np.isfinite(ivs_ms) & (w > 1.0)
        if m.sum() > 100:
            r = np.median(ivs_ms[m] / w[m])
            print(f"      implied effective rolling radius = {r:.4f} m "
                  f"(a passenger car tyre is ~0.30-0.34 m)")

    print("\n" + "=" * 78)
    print("GRAVITY COLUMNS (S-) - no low-pass needed if these are real")
    print("=" * 78)
    gcols = [c for c in s.columns if "GRAVITY" in c.upper()]
    print(f"  {gcols}")
    if len(gcols) == 3:
        g = np.column_stack([pd.to_numeric(s[c], errors="coerce").values for c in gcols])
        mag = np.linalg.norm(g, axis=1)
        print(f"  |gravity| : mean={np.nanmean(mag):.4f}  std={np.nanstd(mag):.4f}  "
              f"(should sit at 9.81 with tiny std)")

    print("\n" + "=" * 78)
    print("IMPLIED LONGITUDINAL ACCELERATION, from each candidate speed source")
    print("=" * 78)
    t_s = pd.to_numeric(s[" TIME SINCE START (ms)"], errors="coerce").values / 1000.0
    for name, arr in [("GPS SPEED as km/h", gps_kmh / 3.6),
                      ("GPS SPEED as m/s", gps_kmh)]:
        dv = np.diff(arr) / np.maximum(np.diff(t_s), 1e-6)
        dv = dv[np.isfinite(dv)]
        print(f"  {name:<22}: p99={np.percentile(np.abs(dv), 99):7.2f}  "
              f"max={np.nanmax(np.abs(dv)):8.2f} m/s^2")


if __name__ == "__main__":
    main()
