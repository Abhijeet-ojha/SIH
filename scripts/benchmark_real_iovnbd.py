"""
scripts/benchmark_real_iovnbd.py

PHASE 6: the final real-data benchmark, in the production configuration.

Reports blackout drift = exit error / blackout distance, per drive, for:
  naive DR (aligned)        double integration with strapdown initial alignment
  fused (hold_last)         production default - see fusion_ekf.DEFAULT_SPEED_SOURCE
  fused (ml)                the ML speed source, kept so its deficit stays visible

Every drive where fused is worse than naive is counted and named, not averaged away.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import joblib

from src.feature_engineering import extract_causal_window_features
from src.fusion_ekf import run_fusion_pipeline
from src.iovnbd_loader import load_benchmark_suite
from src.naive_dr import NaiveDeadReckoning

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics", "real_iovnbd")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "speed_regressor_real.joblib")
BLACKOUT_SEC = 90.0


def blackout_window(df):
    t = df["timestamp"].values
    mid = 0.5 * (t[0] + t[-1])
    start = max(t[0] + 30.0, mid - BLACKOUT_SEC / 2.0)
    return start, start + BLACKOUT_SEC


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    suite = load_benchmark_suite(max_samples_per_drive=6000)
    drives = suite["clean"]
    print(f"{suite['provenance']}\n{len(drives)} drives | {BLACKOUT_SEC:.0f} s blackout\n")

    rows = []
    for i, d in enumerate(drives):
        df = d.get_data()
        t = df["timestamp"].values
        t0, t1 = blackout_window(df)
        idx = np.flatnonzero((t >= t0) & (t < t1))
        if len(idx) < 2:
            continue
        gx, gy = df["pos_x"].values, df["pos_y"].values
        bo_dist = float(np.sum(np.hypot(np.diff(gx[idx]), np.diff(gy[idx]))))
        if bo_dist < 50.0:
            continue

        # Naive DR, restarted at the blackout boundary so it is judged on the same task:
        # survive 90 s open loop from a known state.
        seg = df.iloc[idx].reset_index(drop=True)
        naive = NaiveDeadReckoning(
            initial_heading=float(df["heading"].values[idx[0]]),
            initial_speed=float(df["speed"].values[idx[0]]),
            initial_pos=(float(gx[idx[0]]), float(gy[idx[0]])),
        ).compute(seg)
        naive_exit = float(naive["pos_error_m"].iloc[-1])

        row = {"drive": d.name, "driver": d.driver_id,
               "blackout_distance_m": round(bo_dist, 1),
               "naive_drift_pct": round(100.0 * naive_exit / bo_dist, 1)}

        v_ml = None
        if bundle is not None:
            X, _, t_end, _ = extract_causal_window_features(df, window_sec=1.5, step_sec=0.2)
            v_ml = np.maximum(np.interp(t, t_end, bundle["model"].predict(X.values)), 0.0)

        for label, src, speed in [("fused_hold", "hold_last", df["speed"].values),
                                  ("fused_ml", "ml", v_ml)]:
            if speed is None:
                continue
            res = run_fusion_pipeline(df, speed, driver_style=d.driver_id,
                                      blackout_start_sec=t0, blackout_end_sec=t1,
                                      speed_source=src)
            exit_err = float(res["open_loop_error_m"].values[idx[-1]])
            row[f"{label}_drift_pct"] = round(100.0 * exit_err / bo_dist, 1)
            row[f"{label}_exit_m"] = round(exit_err, 1)
            if label == "fused_hold":
                row["false_stationary_blackout_m"] = round(
                    res.attrs.get("false_stationary_blackout_m", 0.0), 1)

        row["fused_worse_than_naive"] = bool(
            row.get("fused_hold_drift_pct", np.inf) > row["naive_drift_pct"])
        rows.append(row)
        print(f"  [{i+1}/{len(drives)}] {d.name:<10} naive={row['naive_drift_pct']:6.1f}%  "
              f"fused(hold)={row.get('fused_hold_drift_pct', float('nan')):6.1f}%  "
              f"fused(ml)={row.get('fused_ml_drift_pct', float('nan')):6.1f}%"
              + ("   <-- FUSED WORSE" if row["fused_worse_than_naive"] else ""))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "benchmark_real.csv"), index=False)

    worse = out[out["fused_worse_than_naive"]]
    summary = {
        "n_drives": int(len(out)),
        "naive_drift_pct_median": float(out["naive_drift_pct"].median()),
        "fused_hold_drift_pct_median": float(out["fused_hold_drift_pct"].median()),
        "fused_ml_drift_pct_median": float(out["fused_ml_drift_pct"].median())
        if "fused_ml_drift_pct" in out else None,
        "n_fused_worse_than_naive": int(len(worse)),
        "drives_fused_worse": worse["drive"].tolist(),
        "n_meeting_10pct_target": int((out["fused_hold_drift_pct"] < 10.0).sum()),
        "provenance": suite["provenance"],
    }
    with open(os.path.join(OUT_DIR, "benchmark_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 72)
    print("BLACKOUT DRIFT = exit error / blackout distance   (median over drives)")
    print("=" * 72)
    print(f"  naive DR (aligned)      {summary['naive_drift_pct_median']:6.1f}%")
    print(f"  fused, hold_last        {summary['fused_hold_drift_pct_median']:6.1f}%   <- production")
    if summary["fused_ml_drift_pct_median"] is not None:
        print(f"  fused, ML speed         {summary['fused_ml_drift_pct_median']:6.1f}%")
    print(f"\n  meets the 10% target:   {summary['n_meeting_10pct_target']} of {summary['n_drives']} drives")
    print(f"  fused WORSE than naive: {summary['n_fused_worse_than_naive']} of {summary['n_drives']} drives")
    if summary["drives_fused_worse"]:
        print(f"    {', '.join(summary['drives_fused_worse'])}")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
