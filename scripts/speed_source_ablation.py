"""
scripts/speed_source_ablation.py

PHASE 2: find out why the fused system is worse than naive dead reckoning.

Identical EKF, identical blackout protocol, identical everything - the ONLY thing that
changes between arms is where forward speed comes from during the outage:

  A  ml          the trained speed regressor
  B  hold_last   the last GNSS speed before the blackout, held constant
  C  train_mean  the training-set mean speed, constant
  D  oracle      true CAN wheel speed (an upper bound, not a system)

If B or C beats A, the model is a net liability and B becomes the production default until
A can beat it. That is not a rhetorical threat; the script prints the verdict and writes it
to disk.

Every number reported here is blackout drift = exit error / blackout distance, measured
strictly open-loop before the first post-outage GNSS fix. Never final error over total
distance - with GNSS restored the filter is being corrected directly and the number stops
describing dead reckoning at all.

Also logged, because it is the likely mechanism: the speed variance the EKF *assumes*
against the model's actual empirical error. A filter told to trust a source that is wrong
will follow it off the road.
"""

import argparse
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

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics", "real_iovnbd")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "speed_regressor_real.joblib")

BLACKOUT_SEC = 90.0
ARMS = ["ml", "hold_last", "train_mean", "oracle", "oracle_speed_and_heading"]


def blackout_window(df: pd.DataFrame, duration: float = BLACKOUT_SEC):
    """Centre the outage in the drive so there is settled GNSS either side."""
    t = df["timestamp"].values
    mid = 0.5 * (t[0] + t[-1])
    start = max(t[0] + 30.0, mid - duration / 2.0)
    return start, start + duration


def ml_speed(df: pd.DataFrame, bundle) -> np.ndarray:
    """Per-sample model speed, causal windows mapped back onto the full time base."""
    X, _, t_end, _ = extract_causal_window_features(df, window_sec=1.5, step_sec=0.2)
    pred = bundle["model"].predict(X.values)
    return np.interp(df["timestamp"].values, t_end, pred)


def drift_pct(exit_err: float, blackout_dist: float) -> float:
    return float("nan") if blackout_dist < 1.0 else 100.0 * exit_err / blackout_dist


def evaluate(df: pd.DataFrame, speed: np.ndarray, sigma: np.ndarray,
             t0: float, t1: float, driver: str, oracle_yaw: bool = False) -> dict:
    """
    oracle_yaw swaps the phone gyro for the vehicle's own CAN yaw rate. Combined with
    oracle speed it answers the question the other four arms cannot: if BOTH inputs were
    perfect, how much drift would remain? Whatever is left is the filter and the geometry,
    not the sensing.
    """
    work = df
    if oracle_yaw and "can_yaw_rate" in df.columns:
        work = df.copy()
        work["gyro_x"] = 0.0
        work["gyro_y"] = 0.0
        # align_frame projects gyro onto gravity, and gravity here is ~ +Z, so putting the
        # CAN yaw rate on the z axis reproduces it exactly after projection.
        work["gyro_z"] = df["can_yaw_rate"].values
    res = run_fusion_pipeline(work, speed, sigma, driver_style=driver,
                              blackout_start_sec=t0, blackout_end_sec=t1)
    t = df["timestamp"].values
    mask = (t >= t0) & (t < t1)
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return {}
    gx, gy = df["pos_x"].values, df["pos_y"].values
    bo_dist = float(np.sum(np.hypot(np.diff(gx[idx]), np.diff(gy[idx]))))
    ol = res["open_loop_error_m"].values
    return {"blackout_distance_m": bo_dist,
            "exit_error_m": float(ol[idx[-1]]),
            "max_error_m": float(np.max(ol[idx])),
            "blackout_drift_pct": drift_pct(float(ol[idx[-1]]), bo_dist)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=0, help="only the first N drives")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"{MODEL_PATH} missing - run scripts/train_real_iovnbd.py first")
    bundle = joblib.load(MODEL_PATH)

    suite = load_benchmark_suite(max_samples_per_drive=args.max_samples)
    drives = suite["clean"]
    if args.limit:
        drives = drives[:args.limit]
    print(f"{suite['provenance']}\n{len(drives)} drives, {BLACKOUT_SEC:.0f} s blackout\n")

    train_mean = float(np.mean([d.get_data()["speed"].mean() for d in drives]))
    print(f"training-set mean speed (arm C) = {train_mean:.2f} m/s\n")

    rows, sigma_rows = [], []
    for i, d in enumerate(drives):
        df = d.get_data()
        t = df["timestamp"].values
        t0, t1 = blackout_window(df)
        truth = df["speed"].values

        try:
            v_ml = np.maximum(ml_speed(df, bundle), 0.0)
        except Exception as exc:
            print(f"  [{i+1}/{len(drives)}] {d.name}: ML failed ({exc}); skipped")
            continue

        pre = np.flatnonzero(t < t0)
        v_hold = np.full(len(df), truth[pre[-1]] if len(pre) else 0.0)
        v_mean = np.full(len(df), train_mean)

        # Empirical error of the model DURING the outage, against the EKF's assumption.
        bo = (t >= t0) & (t < t1)
        emp_rmse = float(np.sqrt(np.mean((v_ml[bo] - truth[bo]) ** 2)))
        emp_bias = float(np.mean(v_ml[bo] - truth[bo]))

        sources = {"ml": v_ml, "hold_last": v_hold, "train_mean": v_mean,
                   "oracle": truth, "oracle_speed_and_heading": truth}
        row = {"drive": d.name, "driver": d.driver_id,
               "ml_rmse_during_blackout": round(emp_rmse, 3),
               "ml_bias_during_blackout": round(emp_bias, 3)}

        for arm, v in sources.items():
            # Constant sigma across arms so the comparison isolates the speed source.
            sigma = np.full(len(df), 0.2)
            m = evaluate(df, v, sigma, t0, t1, d.driver_id,
                         oracle_yaw=(arm == "oracle_speed_and_heading"))
            if not m:
                continue
            row[f"{arm}_drift_pct"] = round(m["blackout_drift_pct"], 2)
            row[f"{arm}_exit_m"] = round(m["exit_error_m"], 1)
            if arm == "ml":
                row["blackout_distance_m"] = round(m["blackout_distance_m"], 1)

        rows.append(row)
        sigma_rows.append({"drive": d.name, "assumed_sigma_mps": 0.2,
                           "actual_rmse_mps": round(emp_rmse, 3),
                           "overconfidence_ratio": round(emp_rmse / 0.2, 1)})
        print(f"  [{i+1}/{len(drives)}] {d.name:<10} "
              f"ml={row.get('ml_drift_pct', float('nan')):7.1f}%  "
              f"hold={row.get('hold_last_drift_pct', float('nan')):7.1f}%  "
              f"mean={row.get('train_mean_drift_pct', float('nan')):7.1f}%  "
              f"oracle={row.get('oracle_drift_pct', float('nan')):6.1f}%  "
              f"orc+hdg={row.get('oracle_speed_and_heading_drift_pct', float('nan')):6.1f}%  "
              f"| ml RMSE {emp_rmse:5.2f} m/s")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, "speed_source_ablation.csv"), index=False)
    pd.DataFrame(sigma_rows).to_csv(os.path.join(OUT_DIR, "sigma_calibration.csv"), index=False)

    print("\n" + "=" * 72)
    print("BLACKOUT DRIFT % = exit error / blackout distance   (lower is better)")
    print("=" * 72)
    summary = {}
    for arm in ARMS:
        col = f"{arm}_drift_pct"
        if col not in df_out:
            continue
        v = df_out[col].dropna().values
        summary[arm] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                        "n": int(len(v))}
        print(f"  {arm:<11} median {np.median(v):7.1f}%   mean {np.mean(v):7.1f}%   "
              f"n={len(v)}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    ml_med = summary.get("ml", {}).get("median", float("inf"))
    best_dumb = min((summary[a]["median"] for a in ("hold_last", "train_mean")
                     if a in summary), default=float("inf"))
    best_name = min((a for a in ("hold_last", "train_mean") if a in summary),
                    key=lambda a: summary[a]["median"], default="n/a")
    if best_dumb < ml_med:
        verdict = (f"The ML speed source is a NET LIABILITY. '{best_name}' achieves "
                   f"{best_dumb:.1f}% median blackout drift against the model's "
                   f"{ml_med:.1f}%. Make '{best_name}' the production default until the "
                   f"model can beat it.")
    else:
        verdict = (f"The ML speed source is carrying its weight: {ml_med:.1f}% median vs "
                   f"{best_dumb:.1f}% for the best constant-speed fallback "
                   f"('{best_name}').")
    print("  " + verdict)

    sig = pd.DataFrame(sigma_rows)
    if len(sig):
        ratio = float(sig["overconfidence_ratio"].median())
        print(f"\n  The EKF is told to expect sigma_v = 0.2 m/s. The model's actual RMSE "
              f"during\n  blackouts is {sig['actual_rmse_mps'].median():.2f} m/s - it is "
              f"{ratio:.0f}x more wrong than the filter\n  assumes. A filter that trusts a "
              f"bad source follows it off the road; this is the\n  mechanism behind fused "
              f"being worse than naive.")

    with open(os.path.join(OUT_DIR, "speed_source_verdict.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "verdict": verdict,
                   "assumed_sigma_mps": 0.2,
                   "median_actual_rmse_mps": float(sig["actual_rmse_mps"].median()) if len(sig) else None},
                  fh, indent=2)
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
