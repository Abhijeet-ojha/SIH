"""
scripts/02_train_and_fuse.py
Day 2 Execution Script (Real IO-VNBD Dataset):
1. Ingests real multi-driver IO-VNBD training drives (Driver A + Driver E).
2. Extracts sliding-window IMU features.
3. Trains Speed Regressor with heteroscedastic uncertainty estimation.
4. Evaluates on unseen real test drive.
5. Simulates 90-second GNSS blackout with Confidence-Aware EKF, Driver-Adaptive NHC, and ZUPT.
6. Renders deliverables.
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from src.naive_dr import NaiveDeadReckoning
from src.feature_engineering import extract_window_features
from src.speed_model import SpeedRegressorModel, reconstruct_ai_dr_trajectory
from src.fusion_ekf import run_fusion_pipeline
from src.visualizer import plot_speed_model_performance, plot_trajectory_comparison_3way

def main():
    print("=" * 75)
    print("DAY 2 — CONFIDENCE-AWARE AI SPEED REGRESSOR & EKF FUSION (REAL IO-VNBD)")
    print("=" * 75)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    print(f"\n[Step 1] Ingesting {len(train_drives)} Real Training Drives from IO-VNBD...")
    X_train_list, y_train_list = [], []
    for d in train_drives:
        df = d.get_data()
        X_df, y_spd, _ = extract_window_features(df, window_sec=1.5, step_sec=0.2)
        X_train_list.append(X_df)
        y_train_list.append(y_spd)
        print(f"         - {d.name} (Driver {d.driver_id}): {len(X_df)} feature windows | Hash: {d.integrity_hash}")

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    # Train Model
    print(f"\n[Step 2] Training Random Forest Speed Regressor on {len(X_train)} real IMU windows...")
    model = SpeedRegressorModel(model_type="random_forest", n_estimators=100, max_depth=12, random_state=42)
    train_metrics = model.train(X_train, y_train)
    print(f"         Train RMSE: {train_metrics['train_rmse']:.3f} m/s | R2: {train_metrics['train_r2']:.3f}")

    model_save_path = os.path.join(PROJECT_ROOT, "outputs", "models", "speed_regressor.joblib")
    model.save(model_save_path)
    embedded_rules_path = os.path.join(PROJECT_ROOT, "outputs", "models", "embedded_rules.json")
    model.export_embedded_rules(embedded_rules_path)
    print(f"         Saved trained model to {model_save_path}")

    # Evaluate on Unseen Test Drive (Driver E - Aggressive)
    test_drive = [d for d in test_drives if d.driver_id == "E"][0] if any(d.driver_id == "E" for d in test_drives) else test_drives[0]
    print(f"\n[Step 3] Evaluating on Unseen Real Test Drive: {test_drive.name} (Driver {test_drive.driver_id}) | Hash: {test_drive.integrity_hash}")
    test_df = test_drive.get_data()

    X_test, y_test, t_test = extract_window_features(test_df, window_sec=1.5, step_sec=0.2)
    y_pred, y_std, test_metrics = model.evaluate(X_test, y_test)
    print(f"         Test RMSE: {test_metrics['test_rmse']:.3f} m/s | Test MAE: {test_metrics['test_mae']:.3f} m/s | R2: {test_metrics['test_r2']:.3f}")
    print(f"         Mean Uncertainty sigma: {test_metrics['mean_uncertainty_sigma']:.3f} m/s")

    speed_plot_path = os.path.join(PROJECT_ROOT, "outputs", "figures", "02_speed_prediction_vs_gt.png")
    top_feats = model.get_feature_importances(top_n=10)
    plot_speed_model_performance(t_test, y_test, y_pred, y_std, top_feats, output_path=speed_plot_path)

    # Reconstruct AI-DR Trajectory with uncertainty
    print("\n[Step 4] Reconstructing AI-DR Trajectory with Confidence Bounds...")
    init_heading = test_df["heading"].iloc[0] if "heading" in test_df.columns else 0.0
    init_pos = (test_df["pos_x"].iloc[0], test_df["pos_y"].iloc[0]) if "pos_x" in test_df.columns else (0.0, 0.0)

    ai_dr_res = reconstruct_ai_dr_trajectory(
        test_df, t_test, y_pred, v_std=y_std,
        initial_heading=init_heading, initial_pos=init_pos
    )

    # Naive DR
    init_speed = test_df["speed"].iloc[0] if "speed" in test_df.columns else 0.0
    naive_dr = NaiveDeadReckoning(initial_heading=init_heading, initial_speed=init_speed, initial_pos=init_pos)
    naive_res = naive_dr.compute(test_df)

    # Simulate 90-second GNSS Blackout with Confidence-Aware EKF
    blackout_start = 60.0
    blackout_end = min(test_drive.duration_sec - 10.0, 150.0)
    print(f"\n[Step 5] Simulating GNSS Blackout from t={blackout_start:.1f}s to t={blackout_end:.1f}s ({blackout_end - blackout_start:.1f}s outage)...")
    print(f"         Executing Confidence-Aware EKF with Driver-Adaptive NHC (Driver {test_drive.driver_id})...")

    fused_res = run_fusion_pipeline(
        df=test_df,
        ai_speed=ai_dr_res["ai_speed"].values,
        ai_speed_std=ai_dr_res["ai_speed_std"].values,
        driver_style="aggressive" if test_drive.driver_id == "E" else "normal",
        blackout_start_sec=blackout_start,
        blackout_end_sec=blackout_end
    )

    traj_plot_path = os.path.join(PROJECT_ROOT, "outputs", "figures", "03_full_trajectory_comparison.png")
    print(f"\n[Step 6] Rendering 3-Way Trajectory Comparison to {traj_plot_path}...")
    plot_trajectory_comparison_3way(
        gt_df=test_df,
        naive_df=naive_res,
        fused_df=fused_res,
        blackout_start_sec=blackout_start,
        blackout_end_sec=blackout_end,
        output_path=traj_plot_path
    )

    print("\n" + "=" * 75)
    print("DAY 2 COMPLETE: Confidence-Aware AI-EKF Fusion generated on real IO-VNBD data.")
    print("=" * 75)

if __name__ == "__main__":
    main()
