"""
scripts/ml_feature_ablation.py
Feature Group Ablation Study for SIH 2026 PS-168:
Evaluates the incremental contribution of each physical feature family:
  1. BASE_STATISTICS (Mean, Std, Min, Max, Range, RMS)
  2. + DYNAMICS_AND_JERK (Jerk da/dt, Angular Accel domega/dt, Jerk RMS)
  3. + ADVANCED_MOMENTS (Median, IQR, MAD, Skewness, Kurtosis, Zero-Crossings)
  4. + CROSS_SIGNAL (Vibration Power, Jerk-Gyro Interaction, Curvature Ratio)
  5. + SPECTRAL_FEATURES (Spectral Centroid, Dominant Frequency, Sub-Band Powers)
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
from src.evaluation_framework import evaluate_ml_speed_metrics, evaluate_downstream_navigation


def run_feature_ablation_study():
    print("=" * 95)
    print("PHASE 2: FEATURE GROUP ABLATION STUDY")
    print("=" * 95)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    feature_groups = [
        ("Base Statistics", "base_stats"),
        ("Base + Dynamics/Jerk", "dynamics"),
        ("Base + Dynamics + Moments", "enhanced"),
        ("Full Feature Suite (+ Spectral)", "all")
    ]

    ablation_results = []

    print("\n" + "=" * 95)
    print(f"{'Feature Configuration':<32} | {'Num Feats':<10} | {'Test MAE':<10} | {'RMSE':<10} | {'R2':<8} | {'90s Exit Error'}")
    print("-" * 95)

    for label, group_key in feature_groups:
        extractor = CausalFeatureExtractor(window_sec=1.5, step_sec=0.2, sample_rate_hz=10.0, feature_group=group_key)
        
        # 1. Extract Training Features
        X_train_list, y_train_list = [], []
        for d in train_drives:
            df = d.get_data()
            X_df, y_spd, _, _ = extractor.extract_features(df)
            X_train_list.append(X_df)
            y_train_list.append(y_spd)

        X_train = pd.concat(X_train_list, ignore_index=True)
        y_train = np.concatenate(y_train_list)

        # 2. Train Random Forest Baseline
        model = TabularSpeedModel(model_type="random_forest", n_estimators=100, max_depth=12, random_state=42)
        model.train(X_train, y_train)

        # 3. Evaluate on Unseen Test Drives
        test_maes, test_rmses, test_r2s, exit_errs = [], [], [], []
        for d in test_drives:
            df = d.get_data()
            X_test, y_test, t_test, _ = extractor.extract_features(df)
            preds, stds = model.predict_with_uncertainty(X_test)
            
            l1_m = evaluate_ml_speed_metrics(y_test, preds)
            l4_m = evaluate_downstream_navigation(d, t_test, preds, stds, blackout_start_sec=60.0)
            
            test_maes.append(l1_m["mae"])
            test_rmses.append(l1_m["rmse"])
            test_r2s.append(l1_m["r2"])
            exit_errs.append(l4_m["blackout_exit_error_m"])

        mean_mae = float(np.mean(test_maes))
        mean_rmse = float(np.mean(test_rmses))
        mean_r2 = float(np.mean(test_r2s))
        mean_exit = float(np.mean(exit_errs))

        print(f"{label:<32} | {X_train.shape[1]:<10} | {mean_mae:<10.3f} | {mean_rmse:<10.3f} | {mean_r2:<8.3f} | {mean_exit:.2f} m")

        ablation_results.append({
            "feature_group": label,
            "feature_key": group_key,
            "num_features": X_train.shape[1],
            "test_mae_mps": mean_mae,
            "test_rmse_mps": mean_rmse,
            "test_r2": mean_r2,
            "blackout_90s_exit_error_m": mean_exit
        })

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "feature_ablation_scorecard.csv")
    pd.DataFrame(ablation_results).to_csv(out_path, index=False)
    print(f"\n[PASS] Saved Feature Ablation Scorecard to: {out_path}")
    print("=" * 95)
    return pd.DataFrame(ablation_results)


if __name__ == "__main__":
    run_feature_ablation_study()
