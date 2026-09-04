"""
scripts/train_real_iovnbd.py

Trains the speed regressor on REAL IO-VNBD with CAN-bus speed labels, and evaluates it
leave-one-driver-out.

Every previous number in this repo came from a model trained on synthetic stand-in data
with a GPS-derived label that was (a) divided by 3.6 when it was already m/s and
(b) step-quantised with glitches up to 134 m/s^2. Neither problem is present here.

Features are the frame-invariant set from src/feature_engineering.py. Nothing is tuned
against a test split: the only knobs are the model defaults, and the held-out driver is
never seen during fitting.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.feature_engineering import extract_causal_window_features
from src.iovnbd_loader import load_benchmark_suite, split_by_driver

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics", "real_iovnbd")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
CACHE = os.path.join(PROJECT_ROOT, "outputs", "cache_real_features.npz")


def build_features(drives, window_sec=1.5, step_sec=0.5):
    """Extract once, reuse everywhere. Keyed by drive so splits stay honest."""
    X_all, y_all, drv_all, name_all = [], [], [], []
    for i, d in enumerate(drives):
        t0 = time.perf_counter()
        X, y, _, _ = extract_causal_window_features(
            d.get_data(), window_sec=window_sec, step_sec=step_sec)
        X_all.append(X.values.astype(np.float32))
        y_all.append(y.astype(np.float32))
        drv_all.append(np.full(len(y), d.driver_id))
        name_all.append(np.full(len(y), d.name))
        print(f"  [{i+1}/{len(drives)}] {d.name:<12} driver {d.driver_id}  "
              f"{len(y):>6} windows  {time.perf_counter()-t0:5.1f}s")
        cols = list(X.columns)
    return (np.vstack(X_all), np.concatenate(y_all), np.concatenate(drv_all),
            np.concatenate(name_all), cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=6000,
                    help="samples per drive (10 Hz, so 6000 = 10 min)")
    ap.add_argument("--step-sec", type=float, default=0.5)
    ap.add_argument("--rebuild", action="store_true", help="ignore the feature cache")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    suite = load_benchmark_suite(max_samples_per_drive=args.max_samples)
    drives = suite["clean"]
    print(f"\n{suite['provenance']}")
    print(f"clean={len(drives)}  flagged={len(suite['flagged'])}  "
          f"unusable={len(suite['unusable'])}")
    print(f"total {sum(d.duration_sec for d in drives)/60:.1f} min of real driving\n")

    if os.path.exists(CACHE) and not args.rebuild:
        z = np.load(CACHE, allow_pickle=True)
        X, y, drv, names, cols = z["X"], z["y"], z["drv"], z["names"], list(z["cols"])
        print(f"[cache] loaded {X.shape[0]} windows x {X.shape[1]} features")
    else:
        print("[features] extracting (frame-invariant set)")
        X, y, drv, names, cols = build_features(drives, step_sec=args.step_sec)
        np.savez_compressed(CACHE, X=X, y=y, drv=drv, names=names, cols=np.array(cols))
        print(f"[features] {X.shape[0]} windows x {X.shape[1]} features -> {CACHE}")

    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if (~finite).any():
        print(f"[clean] dropping {int((~finite).sum())} windows with non-finite values")
        X, y, drv, names = X[finite], y[finite], drv[finite], names[finite]

    print(f"\nlabel: CAN indicated vehicle speed | "
          f"mean {y.mean():.2f} m/s, max {y.max():.2f} m/s, "
          f"{100*np.mean(y < 0.5):.1f}% stationary")

    # ── Leave-one-driver-out ────────────────────────────────────────────────
    rows = []
    print("\n" + "=" * 72)
    print("LEAVE-ONE-DRIVER-OUT (the only split that tests generalisation)")
    print("=" * 72)
    for holdout in sorted(set(drv.tolist())):
        tr, te = drv != holdout, drv == holdout
        if te.sum() < 200 or tr.sum() < 200:
            print(f"  driver {holdout}: skipped ({int(te.sum())} test windows)")
            continue
        m = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42,
                                  n_jobs=-1, min_samples_leaf=4)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        mae = float(mean_absolute_error(y[te], pred))
        r2 = float(r2_score(y[te], pred))
        # A model that cannot beat "always predict the training mean" is worse than a
        # constant, which is what a negative R2 says in plain terms.
        base_mae = float(np.mean(np.abs(y[te] - y[tr].mean())))
        rows.append({"holdout_driver": holdout, "train_windows": int(tr.sum()),
                     "test_windows": int(te.sum()), "mae": round(mae, 4),
                     "r2": round(r2, 4), "constant_baseline_mae": round(base_mae, 4),
                     "beats_constant": bool(mae < base_mae)})
        verdict = "beats constant" if mae < base_mae else "WORSE THAN A CONSTANT"
        print(f"  driver {holdout}: MAE {mae:5.2f} m/s | R2 {r2:6.3f} | "
              f"constant-baseline MAE {base_mae:5.2f} -> {verdict}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "lodo_real.csv"), index=False)

    # ── Production model: all drivers, for the ablation and the EKF ─────────
    print("\n[final] fitting on all clean drives")
    final = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42,
                                  n_jobs=-1, min_samples_leaf=4)
    final.fit(X, y)
    import joblib
    path = os.path.join(MODEL_DIR, "speed_regressor_real.joblib")
    joblib.dump({"model": final, "features": cols,
                 "label": "CAN indicated vehicle speed (m/s)",
                 "provenance": suite["provenance"]}, path, compress=3)
    print(f"[final] saved {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    imp = sorted(zip(cols, final.feature_importances_), key=lambda kv: -kv[1])[:12]
    print("\ntop features:")
    for n, v in imp:
        print(f"   {v:.4f}  {n}")

    with open(os.path.join(OUT_DIR, "training_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"provenance": suite["provenance"], "n_windows": int(X.shape[0]),
                   "n_features": int(X.shape[1]), "lodo": rows,
                   "top_features": [{"feature": n, "importance": float(v)} for n, v in imp]},
                  fh, indent=2)
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
