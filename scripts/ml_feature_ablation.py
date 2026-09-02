"""
scripts/ml_feature_ablation.py
Phase 9: Feature Ablation Study & Permutation Importance Analysis.
Evaluates incremental contributions of feature subsets:
  1. Base Statistics (Mean, Std, Min, Max, P2P, RMS)
  2. + Enhanced Statistical Moments (Median, IQR, MAD, Skew, Kurtosis, Percentiles, ZCR)
  3. + Kinematic Dynamics (Jerk da/dt, Angular Accel domega/dt)
  4. + Orientation-Invariant & Interaction Magnitudes (Acc/Gyro Norms, Vibration Power)
  5. + Data-Driven Spectral Features (Centroid, Dominant Freq, Sub-band Powers)
Computes Permutation Feature Importance to identify top physical drivers of speed.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from src.feature_engineering import extract_causal_window_features

def run_feature_ablation_study():
    print("=" * 95)
    print("PHASE 9: FEATURE ABLATION STUDY & PERMUTATION IMPORTANCE ANALYSIS")
    print("=" * 95)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    print("\n[Step 1] Extracting full causal feature sets across training & test drives...")
    X_train_list, y_train_list = [], []
    for d in train_drives:
        df = d.get_data()
        X_df, y_spd, _, _ = extract_causal_window_features(df, window_sec=1.5, step_sec=0.2, feature_group="all")
        X_train_list.append(X_df)
        y_train_list.append(y_spd)

    X_train_full = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    test_dfs = []
    for d in test_drives:
        df = d.get_data()
        X_df, y_spd, _, _ = extract_causal_window_features(df, window_sec=1.5, step_sec=0.2, feature_group="all")
        test_dfs.append((X_df, y_spd))

    # Define Feature Subsets
    all_cols = list(X_train_full.columns)
    
    # 1. Base Stats
    base_stat_cols = [c for c in all_cols if any(k in c for k in ["_mean", "_std", "_min", "_max", "_p2p", "_rms"]) 
                      and not any(k in c for k in ["spec", "dom_freq", "power", "median", "iqr", "mad", "skew", "kurt", "p10", "p90", "zcr"])]
    
    # 2. Base + Enhanced Stats
    enhanced_stat_cols = [c for c in all_cols if not any(k in c for k in ["spec", "dom_freq", "power", "jerk", "alpha", "vibration", "curv"])]
    
    # 3. + Kinematics & Jerk
    kinematics_cols = [c for c in all_cols if not any(k in c for k in ["spec", "dom_freq", "power"])]
    
    # 4. All Features (+ Spectral)
    full_cols = all_cols

    feature_subsets = [
        ("1. Base Statistical Moments", base_stat_cols),
        ("2. + Enhanced Stats (Median, IQR, Skew, ZCR)", enhanced_stat_cols),
        ("3. + Kinematic Dynamics (Jerk, Ang Accel)", kinematics_cols),
        ("4. + All Features (+ Spectral Centroid & Bands)", full_cols)
    ]

    ablation_results = []
    print("\n" + "=" * 95)
    print(f"{'Feature Configuration':<45} | {'Num Features':<14} | {'Test MAE':<12} | {'RMSE':<12} | {'R2'}")
    print("-" * 95)

    for subset_name, cols in feature_subsets:
        X_tr_sub = X_train_full[cols]

        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
        model.fit(X_tr_sub.values, y_train)

        maes, rmses, r2s = [], [], []
        for X_te_df, y_te in test_dfs:
            preds = np.maximum(0.0, model.predict(X_te_df[cols].values))
            maes.append(mean_absolute_error(y_te, preds))
            rmses.append(np.sqrt(mean_squared_error(y_te, preds)))
            r2s.append(r2_score(y_te, preds))

        m_mae = float(np.mean(maes))
        m_rmse = float(np.mean(rmses))
        m_r2 = float(np.mean(r2s))

        row = {
            "feature_subset": subset_name,
            "num_features": len(cols),
            "test_mae_mps": m_mae,
            "test_rmse_mps": m_rmse,
            "test_r2": m_r2
        }
        ablation_results.append(row)
        print(f"{subset_name:<45} | {len(cols):<14} | {m_mae:<12.3f} | {m_rmse:<12.3f} | {m_r2:<8.3f}")

    print("=" * 95)

    # Permutation Feature Importance Analysis on Top Model
    print("\n[Step 2] Computing Permutation Feature Importance on held-out test data...")
    final_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    final_model.fit(X_train_full.values, y_train)

    X_test_concat = pd.concat([x[0] for x in test_dfs], ignore_index=True)
    y_test_concat = np.concatenate([x[1] for x in test_dfs])

    perm_res = permutation_importance(
        final_model, X_test_concat.values, y_test_concat,
        n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error"
    )

    perm_importances = []
    for i, col in enumerate(all_cols):
        perm_importances.append({
            "feature": col,
            "importance_mean": float(perm_res.importances_mean[i]),
            "importance_std": float(perm_res.importances_std[i])
        })

    perm_df = pd.DataFrame(perm_importances).sort_values("importance_mean", ascending=False)

    print("\nTop 15 Most Influential Physical Features (Permutation Importance on Unseen Test Drives):")
    for rank, (_, r) in enumerate(perm_df.head(15).iterrows(), 1):
        print(f"  {rank:2d}. {r['feature']:<30s} | dMAE: +{r['importance_mean']:.4f} m/s (+/-{r['importance_std']:.4f})")

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(out_dir, exist_ok=True)
    
    ablation_df = pd.DataFrame(ablation_results)
    ablation_path = os.path.join(out_dir, "feature_ablation_study.csv")
    ablation_df.to_csv(ablation_path, index=False)

    perm_path = os.path.join(out_dir, "permutation_feature_importance.csv")
    perm_df.to_csv(perm_path, index=False)

    print(f"\n[PASS] Saved feature ablation study: {ablation_path}")
    print(f"[PASS] Saved permutation importances: {perm_path}")
    print("=" * 95)
    return ablation_df, perm_df

if __name__ == "__main__":
    run_feature_ablation_study()
