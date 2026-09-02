"""
scripts/ml_regime_and_robustness.py
Phases 12 & 13: Motion Regime Segmentation & Sensor Degradation Robustness.
Evaluates model across:
  - Motion Regimes: Stationary, Low Speed, Cruising, Speed Bins (0-5, 5-20, 20-40 km/h), Turning
  - Robustness Perturbations: Noise injection, Accel bias offset, Gyro bias offset, Packet dropout
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from src.feature_engineering import extract_causal_window_features
from src.speed_model import SpeedRegressorModel
from src.evaluation_framework import evaluate_downstream_navigation

def run_regime_and_robustness_evaluation():
    print("=" * 95)
    print("PHASES 12 & 13: MOTION REGIMES & SENSOR DEGRADATION ROBUSTNESS")
    print("=" * 95)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    print("\n[Step 1] Training winning XGBoost model on full training set...")
    X_train_list, y_train_list = [], []
    for d in train_drives:
        df = d.get_data()
        X_df, y_spd, _, _ = extract_causal_window_features(df, window_sec=1.5, step_sec=0.2)
        X_train_list.append(X_df)
        y_train_list.append(y_spd)

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    model = SpeedRegressorModel(model_type="xgboost", n_estimators=100, max_depth=6, uncertainty_method="conformal")
    
    # Calibrate conformal on first drive
    df_val = test_drives[0].get_data()
    val_X, val_y, _, _ = extract_causal_window_features(df_val, window_sec=1.5, step_sec=0.2)
    model.train(X_train, y_train, X_val=val_X, y_val=val_y)

    # ── Phase 12: Motion Regime Segmentation ─────────────────────────────────
    print("\n--- PHASE 12: PERFORMANCE ACROSS MOTION REGIMES & SPEED BINS ---")
    all_y_true = []
    all_y_pred = []
    all_y_std = []
    all_gz = []

    for d in test_drives:
        df = d.get_data()
        X_test, y_test, t_test, _ = extract_causal_window_features(df, window_sec=1.5, step_sec=0.2)
        preds, stds = model.predict_with_uncertainty(X_test)
        
        all_y_true.append(y_test)
        all_y_pred.append(preds)
        all_y_std.append(stds)
        all_gz.append(X_test["gz_rms"].values)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    y_std_all = np.concatenate(all_y_std)
    gz_all = np.concatenate(all_gz)

    speed_kmh = y_true_all * 3.6

    regimes = {
        "Stationary (v < 0.5 m/s)": y_true_all < 0.5,
        "Low Speed (0.5 <= v < 2.5 m/s)": (y_true_all >= 0.5) & (y_true_all < 2.5),
        "Cruising (v >= 2.5 m/s)": y_true_all >= 2.5,
        "Bin: 0 - 5 km/h": speed_kmh < 5.0,
        "Bin: 5 - 20 km/h": (speed_kmh >= 5.0) & (speed_kmh < 20.0),
        "Bin: 20+ km/h": speed_kmh >= 20.0,
        "Turning Maneuver (gz_rms > 0.08 rad/s)": gz_all > 0.08
    }

    regime_results = []
    print(f"{'Motion Regime / Speed Bin':<40} | {'Samples':<10} | {'MAE (m/s)':<12} | {'RMSE (m/s)':<12} | {'Mean Sigma (m/s)'}")
    print("-" * 95)
    for reg_name, mask in regimes.items():
        if np.sum(mask) == 0:
            continue
        yt = y_true_all[mask]
        yp = y_pred_all[mask]
        ys = y_std_all[mask]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mean_sig = float(np.mean(ys))

        regime_results.append({
            "regime": reg_name,
            "samples": int(np.sum(mask)),
            "mae_mps": mae,
            "rmse_mps": rmse,
            "mean_sigma_mps": mean_sig
        })
        print(f"{reg_name:<40} | {int(np.sum(mask)):<10d} | {mae:<12.3f} | {rmse:<12.3f} | {mean_sig:.3f}")
    print("=" * 95)

    # ── Phase 13: Sensor Degradation & Robustness Benchmark ──────────────────
    print("\n--- PHASE 13: SENSOR DEGRADATION & OUTAGE ROBUSTNESS ---")
    perturbations = [
        ("Clean Baseline (No Noise)", "clean", 0.0, 0.0, 0.0),
        ("+ Gaussian Sensor Noise (acc=0.05m/s^2, gyro=0.01rad/s)", "noise", 0.05, 0.01, 0.0),
        ("+ Accel Bias Drift (+0.15 m/s^2)", "acc_bias", 0.0, 0.0, 0.15),
        ("+ Gyro Bias Drift (+0.02 rad/s)", "gyro_bias", 0.0, 0.02, 0.0),
        ("+ Packet Dropout (5% random window loss)", "dropout", 0.0, 0.0, 0.0)
    ]

    robustness_results = []
    print(f"{'Degradation Condition':<55} | {'Test MAE (m/s)':<16} | {'90s Exit Error (m)'}")
    print("-" * 95)

    for p_name, p_type, n_acc, n_gyr, bias_val in perturbations:
        pert_maes = []
        pert_exit_errs = []

        for d in test_drives:
            df_pert = d.get_data().copy()
            
            if p_type == "noise":
                df_pert["acc_x"] += np.random.normal(0, n_acc, len(df_pert))
                df_pert["acc_y"] += np.random.normal(0, n_acc, len(df_pert))
                df_pert["gyro_z"] += np.random.normal(0, n_gyr, len(df_pert))
            elif p_type == "acc_bias":
                df_pert["acc_x"] += bias_val
                df_pert["acc_y"] += bias_val
            elif p_type == "gyro_bias":
                df_pert["gyro_z"] += bias_val

            X_te, y_te, t_te, _ = extract_causal_window_features(df_pert, window_sec=1.5, step_sec=0.2)
            
            if p_type == "dropout":
                keep_mask = np.random.rand(len(X_te)) > 0.05
                X_te = X_te[keep_mask]
                y_te = y_te[keep_mask]
                t_te = t_te[keep_mask]

            preds, stds = model.predict_with_uncertainty(X_te)
            pert_maes.append(mean_absolute_error(y_te, preds))
            
            nav_m = evaluate_downstream_navigation(d, t_te, preds, stds)
            pert_exit_errs.append(nav_m["blackout_exit_error_m"])

        m_mae = float(np.mean(pert_maes))
        m_exit = float(np.mean(pert_exit_errs))
        s_exit = float(np.std(pert_exit_errs))

        robustness_results.append({
            "condition": p_name,
            "test_mae_mps": m_mae,
            "blackout_exit_error_m": m_exit,
            "blackout_exit_std_m": s_exit
        })
        print(f"{p_name:<55} | {m_mae:<16.3f} | {m_exit:.2f} +/- {s_exit:.1f} m")
    print("=" * 95)

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(out_dir, exist_ok=True)
    
    pd.DataFrame(regime_results).to_csv(os.path.join(out_dir, "phase12_motion_regimes.csv"), index=False)
    pd.DataFrame(robustness_results).to_csv(os.path.join(out_dir, "phase13_sensor_robustness.csv"), index=False)

    print("[PASS] Saved motion regimes & robustness analysis.")
    return pd.DataFrame(regime_results), pd.DataFrame(robustness_results)

if __name__ == "__main__":
    run_regime_and_robustness_evaluation()
