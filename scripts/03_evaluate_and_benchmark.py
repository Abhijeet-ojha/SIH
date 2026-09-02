"""
scripts/03_evaluate_and_benchmark.py
Multi-Driver Benchmark Suite for SIH 2026 PS-168 (Real IO-VNBD Dataset).
Evaluates quantitative benchmarks across multiple real test drives:
  - Driver A (Normal Urban Multi-run: S3a, S3b)
  - Driver B (Spot-check: S-M)
  - Driver D (Spot-check: S-Y1)
  - Driver E (Aggressive Multi-run: Vfa02, Vta1b)
Uses Confidence-Aware 6-State EKF fusion and Driver-Adaptive physical constraints.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import importlib
from scipy.interpolate import interp1d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from core.features.extractor import CausalFeatureExtractor
from core.models.tabular_models import TabularSpeedModel
from core.fusion.ekf_6state import run_6state_ekf_fusion
from src.naive_dr import NaiveDeadReckoning
from src.metrics import calculate_benchmark_metrics, compute_multi_drive_statistics, format_multi_drive_markdown


def main():
    print("=" * 80)
    print("DAY 3 — MULTI-DRIVER BENCHMARK SUITE (REAL IO-VNBD DATASET)")
    print("=" * 80)

    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "speed_regressor.joblib")
    if not os.path.exists(model_path):
        print("[Notice] Trained model not found. Running training pipeline first...")
        run_day2 = importlib.import_module("scripts.02_train_and_fuse").main
        run_day2()

    model = TabularSpeedModel()
    model.load(model_path)
    print(f"\n[Step 1] Loaded trained speed model from {model_path}")

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    test_drives = suite["test_drives"]

    extractor = CausalFeatureExtractor(window_sec=1.5, step_sec=0.2, sample_rate_hz=10.0, feature_group="all")

    print(f"\n[Step 2] Evaluating {len(test_drives)} Real Test Drives across Drivers A, B, D, E...")
    all_metrics = {}

    for drive in test_drives:
        drive_name = drive.name
        df = drive.get_data()
        d_id = drive.driver_id
        t_orig = df["timestamp"].values
        n = len(df)
        print(f"\n  Evaluating: {drive_name:20s} | Driver {d_id} | Hash: {drive.integrity_hash} | {len(df)} samples | {drive.duration_sec:.1f}s duration")

        # Sliding window features & prediction with uncertainty
        X_test, y_test, t_test, _ = extractor.extract_features(df)
        y_pred, y_std = model.predict_with_uncertainty(X_test)

        init_heading = df["heading"].iloc[0] if "heading" in df.columns else 0.0
        init_speed = df["speed"].iloc[0] if "speed" in df.columns else 0.0
        init_pos = (df["pos_x"].iloc[0], df["pos_y"].iloc[0]) if "pos_x" in df.columns else (0.0, 0.0)

        # Naive DR
        naive_dr = NaiveDeadReckoning(initial_heading=init_heading, initial_speed=init_speed, initial_pos=init_pos)
        naive_res = naive_dr.compute(df)

        # AI-DR Pure Trajectory
        interp_func = interp1d(t_test, y_pred, kind="linear", bounds_error=False, fill_value=(y_pred[0], y_pred[-1]))
        v_dense = np.maximum(0.0, interp_func(t_orig))
        interp_std = interp1d(t_test, y_std, kind="linear", bounds_error=False, fill_value=(y_std[0], y_std[-1]))
        std_dense = np.maximum(0.05, interp_std(t_orig))

        dt_arr = np.diff(t_orig, prepend=t_orig[0])
        dt_arr[0] = dt_arr[1] if n > 1 else 0.1
        gyro_z = df["gyro_z"].values
        heading_ai, px_ai, py_ai = np.zeros(n), np.zeros(n), np.zeros(n)
        heading_ai[0], px_ai[0], py_ai[0] = init_heading, init_pos[0], init_pos[1]
        for i in range(1, n):
            dt = dt_arr[i]
            heading_ai[i] = heading_ai[i-1] + gyro_z[i] * dt
            v_m = 0.5 * (v_dense[i-1] + v_dense[i])
            h_m = 0.5 * (heading_ai[i-1] + heading_ai[i])
            px_ai[i] = px_ai[i-1] + v_m * np.sin(h_m) * dt
            py_ai[i] = py_ai[i-1] + v_m * np.cos(h_m) * dt

        ai_dr_res = pd.DataFrame({
            "timestamp": t_orig,
            "ai_speed": v_dense,
            "ai_speed_std": std_dense,
            "ai_heading": heading_ai,
            "ai_pos_x": px_ai,
            "ai_pos_y": py_ai,
            "ai_pos_error_m": np.sqrt((px_ai - df["pos_x"].values)**2 + (py_ai - df["pos_y"].values)**2)
        })

        # 90s GNSS Blackout with Confidence-Aware 6-State EKF + Driver-Adaptive NHC + ZUPT
        blackout_start = 60.0
        blackout_end = min(drive.duration_sec - 10.0, 150.0)
        fused_res = run_6state_ekf_fusion(
            df=df,
            ai_speed=v_dense,
            ai_speed_std=std_dense,
            driver_style="aggressive" if d_id == "E" else "normal",
            blackout_start_sec=blackout_start,
            blackout_end_sec=blackout_end
        )

        m = calculate_benchmark_metrics(
            gt_df=df,
            naive_df=naive_res,
            ai_dr_df=ai_dr_res,
            fused_df=fused_res,
            blackout_start_sec=blackout_start,
            blackout_end_sec=blackout_end
        )
        all_metrics[drive_name] = m

        print(f"    - Naive Drift:       {m['naive_dead_reckoning']['final_drift_m']:.1f} m ({m['naive_dead_reckoning']['drift_pct_distance']}%)")
        print(f"    - 90s Blackout Exit: {m['ai_dr_gnss_ekf_fusion']['blackout_terminal_exit_error_m']:.2f} m (Peak: {m['ai_dr_gnss_ekf_fusion']['blackout_max_error_m']:.2f} m)")
        print(f"    - Post-GPS Settled:  {m['ai_dr_gnss_ekf_fusion']['post_reacquisition_settled_error_m']:.2f} m")

    # Summary Statistics
    stats = compute_multi_drive_statistics(all_metrics)
    print("\n" + "=" * 80)
    print("DAY 3 STATISTICAL BENCHMARK SUMMARY (MEAN ± STD ACROSS REAL DRIVES)")
    print("=" * 80)
    print(f"  * 90s Blackout Exit Error (AI-EKF): {stats['ai_ekf_terminal_exit_error_m']['mean']:.2f} ± {stats['ai_ekf_terminal_exit_error_m']['std']:.2f} m")
    print(f"  * 90s Blackout Peak Drift (AI-EKF): {stats['ai_ekf_max_outage_error_m']['mean']:.2f} ± {stats['ai_ekf_max_outage_error_m']['std']:.2f} m")
    print(f"  * Naive Final Drift:                {stats['naive_final_drift_m']['mean']:.1f} ± {stats['naive_final_drift_m']['std']:.1f} m")
    print(f"  * Trajectory RMSE (AI-EKF):         {stats['ai_ekf_rmse_m']['mean']:.2f} ± {stats['ai_ekf_rmse_m']['std']:.2f} m")

    metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    json_path = os.path.join(metrics_dir, "benchmark_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    summary_json_path = os.path.join(metrics_dir, "summary_statistics.json")
    with open(summary_json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[PASS] Saved Multi-Driver Benchmark metrics to: {metrics_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
