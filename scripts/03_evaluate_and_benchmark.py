"""
scripts/03_evaluate_and_benchmark.py
Day 3 Execution Script (Real IO-VNBD Dataset):
Evaluates quantitative benchmarks across multiple real IO-VNBD test drives:
  - Driver A (Normal Urban Multi-run: S3a, S3b)
  - Driver B (Spot-check: S-M, n=1)
  - Driver D (Spot-check: S-Y1, n=1)
  - Driver E (Aggressive Multi-run: Vfa02, Vta1b)
Uses Confidence-Aware EKF fusion and Driver-Adaptive physical constraints.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import importlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from src.naive_dr import NaiveDeadReckoning
from src.feature_engineering import extract_window_features
from src.speed_model import SpeedRegressorModel, reconstruct_ai_dr_trajectory
from src.fusion_ekf import run_fusion_pipeline
from src.metrics import calculate_benchmark_metrics, compute_multi_drive_statistics, format_multi_drive_markdown

def main():
    print("=" * 75)
    print("DAY 3 — MULTI-DRIVER BENCHMARK SUITE (REAL IO-VNBD DATASET)")
    print("=" * 75)

    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "speed_regressor.joblib")
    if not os.path.exists(model_path):
        print("[Notice] Trained model not found. Running Day 2 pipeline first...")
        run_day2 = importlib.import_module("scripts.02_train_and_fuse").main
        run_day2()

    model = SpeedRegressorModel()
    model.load(model_path)
    print(f"\n[Step 1] Loaded trained speed model from {model_path}")

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    test_drives = suite["test_drives"]

    print(f"\n[Step 2] Evaluating {len(test_drives)} Real Test Drives across Drivers A, B, D, E...")
    all_metrics = {}

    for drive in test_drives:
        drive_name = drive.name
        df = drive.get_data()
        d_id = drive.driver_id
        print(f"\n  Evaluating: {drive_name:20s} | Driver {d_id} | Hash: {drive.integrity_hash} | {len(df)} samples | {drive.duration_sec:.1f}s duration")

        # Sliding window features & prediction with uncertainty
        X_test, y_test, t_test = extract_window_features(df, window_sec=1.5, step_sec=0.2)
        y_pred, y_std = model.predict_with_uncertainty(X_test)

        init_heading = df["heading"].iloc[0] if "heading" in df.columns else 0.0
        init_speed = df["speed"].iloc[0] if "speed" in df.columns else 0.0
        init_pos = (df["pos_x"].iloc[0], df["pos_y"].iloc[0]) if "pos_x" in df.columns else (0.0, 0.0)

        # Naive DR
        naive_dr = NaiveDeadReckoning(initial_heading=init_heading, initial_speed=init_speed, initial_pos=init_pos)
        naive_res = naive_dr.compute(df)

        # AI-DR Pure
        ai_dr_res = reconstruct_ai_dr_trajectory(
            df, t_test, y_pred, v_std=y_std,
            initial_heading=init_heading, initial_pos=init_pos
        )

        # 90s GNSS Blackout with Confidence-Aware EKF + Driver-Adaptive NHC + ZUPT
        blackout_start = 60.0
        blackout_end = min(drive.duration_sec - 10.0, 150.0)
        fused_res = run_fusion_pipeline(
            df=df,
            ai_speed=ai_dr_res["ai_speed"].values,
            ai_speed_std=ai_dr_res["ai_speed_std"].values,
            driver_style="aggressive" if d_id == "E" else "normal",
            blackout_start_sec=blackout_start,
            blackout_end_sec=blackout_end
        )

        # Calculate metrics
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

    # Compute Multi-Drive Statistics
    stats = compute_multi_drive_statistics(all_metrics)

    # Save outputs
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    json_path = os.path.join(metrics_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({"multi_drive_stats": stats, "per_drive_metrics": all_metrics}, f, indent=2)
    print(f"\n[Step 3] Saved benchmark JSON to: {json_path}")

    md_content = format_multi_drive_markdown(all_metrics, stats)
    md_path = os.path.join(metrics_dir, "benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"[Step 4] Saved benchmark Markdown report to: {md_path}")

    print("\n" + "=" * 75)
    print(md_content)
    print("=" * 75)
    print("DAY 3 COMPLETE: Multi-driver IO-VNBD benchmark deliverables generated.")

if __name__ == "__main__":
    main()
