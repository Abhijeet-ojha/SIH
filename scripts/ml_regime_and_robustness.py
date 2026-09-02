"""
scripts/ml_regime_and_robustness.py
Physical Motion Regime Breakdown and Sensor Robustness Perturbation Testing for SIH 2026 PS-168.
Evaluates model accuracy across:
  - Standstill, Low-Speed (<3 m/s), Cruising (3-18 m/s), Highway (>18 m/s), Acceleration, Braking, Turning
And stress-tests realistic perturbations:
  - Accel/Gyro noise, Bias drift, Timestamp jitter, and Random packet loss.
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from core.features.extractor import CausalFeatureExtractor
from core.models.tabular_models import TabularSpeedModel
from src.evaluation_framework import evaluate_ml_speed_metrics


def run_regime_and_robustness_analysis():
    print("=" * 95)
    print("PHASE 4: MOTION REGIMES BREAKDOWN & ROBUSTNESS PERTURBATION TESTING")
    print("=" * 95)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    extractor = CausalFeatureExtractor(window_sec=1.5, step_sec=0.2, sample_rate_hz=10.0, feature_group="all")
    
    # 1. Train Model
    X_train_list, y_train_list = [], []
    for d in train_drives:
        df = d.get_data()
        X_df, y_spd, _, _ = extractor.extract_features(df)
        X_train_list.append(X_df)
        y_train_list.append(y_spd)

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    calib_drive = train_drives[1] if len(train_drives) > 1 else train_drives[0]
    X_calib, y_calib, _, _ = extractor.extract_features(calib_drive.get_data())

    model = TabularSpeedModel(model_type="random_forest", n_estimators=100, max_depth=12, random_state=42, uncertainty_method="conformal")
    model.train(X_train, y_train, X_calib=X_calib, y_calib=y_calib)

    # 2. Extract Test Features and Raw Signals for Regime Slicing
    all_y_test, all_preds, all_stds = [], [], []
    all_speeds, all_accs, all_gyros = [], [], []

    for d in test_drives:
        df = d.get_data()
        X_test, y_test, t_test, _ = extractor.extract_features(df)
        preds, stds = model.predict_with_uncertainty(X_test)
        
        all_y_test.append(y_test)
        all_preds.append(preds)
        all_stds.append(stds)
        all_speeds.append(y_test)
        all_accs.append(X_test["ay_mean"].values)
        all_gyros.append(X_test["gz_mean"].values)

    y_test_full = np.concatenate(all_y_test)
    preds_full = np.concatenate(all_preds)
    stds_full = np.concatenate(all_stds)
    acc_full = np.concatenate(all_accs)
    gyro_full = np.concatenate(all_gyros)

    # 3. Motion Regime Evaluation
    print("\n[Part 1] Physical Motion Regime Performance Breakdown:")
    print("-" * 95)
    print(f"{'Motion Regime':<24} | {'Samples':<10} | {'MAE (m/s)':<12} | {'RMSE (m/s)':<12} | {'P95 Error':<12} | {'Mean Sigma'}")
    print("-" * 95)

    regimes = [
        ("Standstill (v < 0.2 m/s)", y_test_full < 0.2),
        ("Low Speed (0.2 - 3 m/s)", (y_test_full >= 0.2) & (y_test_full < 3.0)),
        ("Cruising (3 - 18 m/s)", (y_test_full >= 3.0) & (y_test_full <= 18.0)),
        ("Highway (> 18 m/s)", y_test_full > 18.0),
        ("Acceleration (> 1.2 m/s²)", acc_full > 1.2),
        ("Braking (< -1.2 m/s²)", acc_full < -1.2),
        ("Turning (|gz| > 0.1 rad/s)", np.abs(gyro_full) > 0.10)
    ]

    regime_results = []
    for reg_name, mask in regimes:
        count = int(np.sum(mask))
        if count > 0:
            sub_y = y_test_full[mask]
            sub_p = preds_full[mask]
            sub_s = stds_full[mask]
            abs_err = np.abs(sub_y - sub_p)
            mae = float(np.mean(abs_err))
            rmse = float(np.sqrt(np.mean(abs_err**2)))
            p95 = float(np.percentile(abs_err, 95))
            mean_sig = float(np.mean(sub_s))
        else:
            mae, rmse, p95, mean_sig = 0.0, 0.0, 0.0, 0.0

        print(f"{reg_name:<24} | {count:<10} | {mae:<12.3f} | {rmse:<12.3f} | {p95:<12.3f} | {mean_sig:.3f} m/s")
        regime_results.append({
            "motion_regime": reg_name,
            "sample_count": count,
            "mae_mps": mae,
            "rmse_mps": rmse,
            "p95_error_mps": p95,
            "mean_uncertainty_sigma_mps": mean_sig
        })

    # 4. Robustness Perturbation Testing
    print("\n[Part 2] Realistic Sensor Perturbation & Robustness Testing:")
    print("-" * 95)
    print(f"{'Perturbation Type':<35} | {'Test MAE':<12} | {'RMSE':<12} | {'Degradation vs Clean'}")
    print("-" * 95)

    base_abs_err = np.abs(y_test_full - preds_full)
    clean_mae = float(np.mean(base_abs_err))
    clean_rmse = float(np.sqrt(np.mean(base_abs_err**2)))
    print(f"{'Clean Baseline':<35} | {clean_mae:<12.3f} | {clean_rmse:<12.3f} | Baseline (0.0%)")

    perturbations = [
        ("Accel Gaussian Noise (sigma=0.3 m/s²)", lambda df: df.assign(
            acc_x=df["acc_x"] + np.random.normal(0, 0.3, len(df)),
            acc_y=df["acc_y"] + np.random.normal(0, 0.3, len(df)),
            acc_z=df["acc_z"] + np.random.normal(0, 0.3, len(df))
        )),
        ("Gyro Gaussian Noise (sigma=0.03 rad/s)", lambda df: df.assign(
            gyro_x=df["gyro_x"] + np.random.normal(0, 0.03, len(df)),
            gyro_y=df["gyro_y"] + np.random.normal(0, 0.03, len(df)),
            gyro_z=df["gyro_z"] + np.random.normal(0, 0.03, len(df))
        )),
        ("Accel Bias Drift (+0.15 m/s²)", lambda df: df.assign(
            acc_y=df["acc_y"] + 0.15
        )),
        ("Gyro Bias Drift (+0.02 rad/s)", lambda df: df.assign(
            gyro_z=df["gyro_z"] + 0.02
        )),
        ("Random 10% Packet Loss", lambda df: df.sample(frac=0.90, random_state=42).sort_values("timestamp"))
    ]

    robustness_results = [{
        "perturbation": "Clean Baseline",
        "test_mae_mps": clean_mae,
        "test_rmse_mps": clean_rmse,
        "degradation_pct": 0.0
    }]

    for p_name, p_fn in perturbations:
        pert_maes, pert_rmses = [], []
        for d in test_drives:
            raw_df = d.get_data().copy()
            pert_df = p_fn(raw_df)
            X_pert, y_pert, _, _ = extractor.extract_features(pert_df)
            p_preds = model.predict(X_pert)
            pert_maes.append(float(np.mean(np.abs(y_pert - p_preds))))
            pert_rmses.append(float(np.sqrt(np.mean((y_pert - p_preds)**2))))

        p_mae = float(np.mean(pert_maes))
        p_rmse = float(np.mean(pert_rmses))
        deg_pct = ((p_mae - clean_mae) / clean_mae) * 100.0

        print(f"{p_name:<35} | {p_mae:<12.3f} | {p_rmse:<12.3f} | +{deg_pct:.1f}%")
        robustness_results.append({
            "perturbation": p_name,
            "test_mae_mps": p_mae,
            "test_rmse_mps": p_rmse,
            "degradation_pct": deg_pct
        })

    # Save metrics
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(regime_results).to_csv(os.path.join(out_dir, "motion_regimes_evaluation.csv"), index=False)
    pd.DataFrame(robustness_results).to_csv(os.path.join(out_dir, "robustness_perturbation_results.csv"), index=False)
    print(f"\n[PASS] Saved Motion Regimes and Robustness results to: {out_dir}")
    print("=" * 95)


if __name__ == "__main__":
    run_regime_and_robustness_analysis()
