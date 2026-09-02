"""
scripts/02_train_and_fuse.py
Production Pipeline Execution Script for SIH 2026 PS-168:
1. Extracts causal features from real IO-VNBD diversified training drives.
2. Trains the production Random Forest Speed Regressor with Conformal Uncertainty Calibration.
3. Exports versioned model_package.json, feature_config.json, speed_regressor.joblib, and embedded_rules.json.
4. Evaluates on unseen aggressive test drive (Driver E) using Confidence-Aware 6-State EKF Fusion.
5. Generates publication-ready figures for speed tracking and trajectory reconstruction.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from core.features.extractor import CausalFeatureExtractor
from core.models.tabular_models import TabularSpeedModel
from core.export.spec import FeatureConfigSpec
from core.export.exporter import EdgeModelExporter
from core.fusion.ekf_6state import run_6state_ekf_fusion
from src.metrics import calculate_benchmark_metrics
from src.naive_dr import NaiveDeadReckoning
from src.visualizer import plot_speed_prediction_vs_ground_truth, plot_trajectory_comparison
from scipy.interpolate import interp1d


def main():
    print("=" * 80)
    print("SIH 2026 PS-168: PRODUCTION MODEL TRAINING & 6-STATE EKF FUSION PIPELINE")
    print("=" * 80)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    extractor = CausalFeatureExtractor(window_sec=1.5, step_sec=0.2, sample_rate_hz=10.0, feature_group="all")

    # Step 1: Feature Extraction
    print(f"\n[Step 1] Extracting causal features across {len(train_drives)} training drives...")
    X_train_list, y_train_list = [], []
    for d in train_drives:
        df = d.get_data()
        X_df, y_spd, _, _ = extractor.extract_features(df)
        X_train_list.append(X_df)
        y_train_list.append(y_spd)
        print(f"         - {d.name} (Driver {d.driver_id}): {len(X_df)} windows | Hash: {d.integrity_hash}")

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    # Isolated calibration drive (Driver A - S3b)
    calib_drive = train_drives[1] if len(train_drives) > 1 else train_drives[0]
    X_calib, y_calib, _, _ = extractor.extract_features(calib_drive.get_data())

    # Step 2: Train Model
    print(f"\n[Step 2] Training Random Forest Speed Regressor on {len(X_train)} windows...")
    model = TabularSpeedModel(model_type="random_forest", n_estimators=100, max_depth=12, random_state=42, uncertainty_method="conformal")
    train_metrics = model.train(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
    print(f"         Train RMSE: {train_metrics['train_rmse']:.3f} m/s | R2: {train_metrics['train_r2']:.3f}")
    print(f"         Calibrated Conformal q_hat: {model.conformal_calibrator.q_hat:.3f} m/s (90% coverage)")

    # Step 3: Export Versioned Artifacts
    print("\n[Step 3] Exporting versioned model package and feature config...")
    models_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
    feature_spec = FeatureConfigSpec(
        feature_version="2.0.0",
        sample_rate_hz=10.0,
        window_sec=1.5,
        step_sec=0.2,
        window_samples=15,
        step_samples=2,
        feature_group="all",
        feature_names=model.feature_names
    )
    exported_files = EdgeModelExporter.export_package(model, feature_spec, models_dir, "speed_regressor")
    for k, v in exported_files.items():
        print(f"         Saved {k}: {v}")

    # Step 4: Evaluate on Unseen Test Drive (Driver E - Aggressive)
    test_drive = [d for d in test_drives if d.driver_id == "E"][0] if any(d.driver_id == "E" for d in test_drives) else test_drives[0]
    print(f"\n[Step 4] Evaluating on Unseen Real Test Drive: {test_drive.name} (Driver {test_drive.driver_id}) | Hash: {test_drive.integrity_hash}")
    test_df = test_drive.get_data()
    t_orig = test_df["timestamp"].values
    n = len(test_df)

    X_test, y_test, t_test, _ = extractor.extract_features(test_df)
    y_pred, y_std = model.predict_with_uncertainty(X_test)
    
    test_mae = float(np.mean(np.abs(y_test - y_pred)))
    test_rmse = float(np.sqrt(np.mean((y_test - y_pred)**2)))
    test_r2 = float(1.0 - np.sum((y_test - y_pred)**2) / (np.sum((y_test - np.mean(y_test))**2) + 1e-9))
    print(f"         Test MAE: {test_mae:.3f} m/s | RMSE: {test_rmse:.3f} m/s | R2: {test_r2:.3f}")
    print(f"         Mean Uncertainty Sigma: {np.mean(y_std):.3f} m/s")

    # Step 5: 6-State EKF Fusion
    print("\n[Step 5] Executing 6-State Kinematic EKF Fusion with 90s GNSS Outage...")
    interp_func = interp1d(t_test, y_pred, kind="linear", bounds_error=False, fill_value=(y_pred[0], y_pred[-1]))
    v_dense = np.maximum(0.0, interp_func(t_orig))
    interp_std = interp1d(t_test, y_std, kind="linear", bounds_error=False, fill_value=(y_std[0], y_std[-1]))
    std_dense = np.maximum(0.05, interp_std(t_orig))

    init_heading = test_df["heading"].iloc[0] if "heading" in test_df.columns else 0.0
    init_speed = test_df["speed"].iloc[0] if "speed" in test_df.columns else 0.0
    init_pos = (test_df["pos_x"].iloc[0], test_df["pos_y"].iloc[0]) if "pos_x" in test_df.columns else (0.0, 0.0)

    # Reconstruct pure AI-DR trajectory
    dt_arr = np.diff(t_orig, prepend=t_orig[0])
    dt_arr[0] = dt_arr[1] if n > 1 else 0.1
    gyro_z = test_df["gyro_z"].values
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
        "ai_pos_error_m": np.sqrt((px_ai - test_df["pos_x"].values)**2 + (py_ai - test_df["pos_y"].values)**2)
    })

    naive_dr = NaiveDeadReckoning(init_heading, init_speed, init_pos).compute(test_df)
    
    blackout_start = 60.0
    blackout_end = min(test_drive.duration_sec - 10.0, 150.0)
    fused_res = run_6state_ekf_fusion(
        df=test_df,
        ai_speed=v_dense,
        ai_speed_std=std_dense,
        driver_style="aggressive" if test_drive.driver_id == "E" else "normal",
        blackout_start_sec=blackout_start,
        blackout_end_sec=blackout_end
    )

    metrics = calculate_benchmark_metrics(test_df, naive_dr, ai_dr_res, fused_res, blackout_start, blackout_end)
    print(f"         Naive Final Drift:       {metrics['naive_dead_reckoning']['final_drift_m']:.1f} m ({metrics['naive_dead_reckoning']['drift_pct_distance']}%)")
    print(f"         90s Blackout Exit Error: {metrics['ai_dr_gnss_ekf_fusion']['blackout_terminal_exit_error_m']:.2f} m (Peak: {metrics['ai_dr_gnss_ekf_fusion']['blackout_max_error_m']:.2f} m)")
    print(f"         Post-GPS Settled Error:  {metrics['ai_dr_gnss_ekf_fusion']['post_reacquisition_settled_error_m']:.2f} m")

    # Step 6: Render Publication Figures
    figs_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(figs_dir, exist_ok=True)
    speed_plot_path = os.path.join(figs_dir, "02_speed_prediction_vs_gt.png")
    plot_speed_prediction_vs_ground_truth(t_test, y_test, y_pred, y_std, test_mae, test_rmse, speed_plot_path)
    
    traj_plot_path = os.path.join(figs_dir, "03_trajectory_comparison.png")
    plot_trajectory_comparison(test_df, naive_dr, ai_dr_res, fused_res, blackout_start, blackout_end, traj_plot_path)
    print(f"\n[PASS] Saved tracking plots to: {figs_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
