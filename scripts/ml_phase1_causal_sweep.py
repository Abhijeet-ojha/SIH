"""
scripts/ml_phase1_causal_sweep.py
Phase 1: Window Size Sweep & Latency Benchmark.
Evaluates context windows W in [0.5s, 1.0s, 1.5s, 2.0s, 3.0s] on real IO-VNBD data.
Measures context window length, feature computation latency, model inference latency,
end-to-end prediction latency, and generalization metrics.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from src.feature_engineering import extract_causal_window_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_causal_window_sweep():
    print("=" * 80)
    print("PHASE 1: CAUSAL WINDOW LENGTH SWEEP & LATENCY PROFILING")
    print("=" * 80)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    window_sizes = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = []

    for W in window_sizes:
        print(f"\n--- Evaluating Causal Context Window W = {W:.1f}s ---")
        
        # 1. Feature extraction on training drives
        X_train_list, y_train_list = [], []
        t_feat_start = time.perf_counter()
        total_windows_extracted = 0

        for d in train_drives:
            df = d.get_data()
            X_df, y_spd, _, lat = extract_causal_window_features(df, window_sec=W, step_sec=0.2, feature_group="all")
            X_train_list.append(X_df)
            y_train_list.append(y_spd)
            total_windows_extracted += len(X_df)

        feat_calc_total_ms = (time.perf_counter() - t_feat_start) * 1000.0
        feat_calc_ms_per_win = feat_calc_total_ms / max(1, total_windows_extracted)

        X_train = pd.concat(X_train_list, ignore_index=True)
        y_train = np.concatenate(y_train_list)

        # 2. Train baseline Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
        t_train_start = time.perf_counter()
        rf.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t_train_start

        # 3. Evaluate on test drives & profile inference latency
        test_maes, test_rmses, test_r2s = [], [], []
        infer_times_ms = []

        for d in test_drives:
            df = d.get_data()
            X_test, y_test, _, _ = extract_causal_window_features(df, window_sec=W, step_sec=0.2, feature_group="all")
            
            # Profile per-window inference latency
            t0 = time.perf_counter()
            preds = np.maximum(0.0, rf.predict(X_test))
            infer_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(X_test))
            infer_times_ms.append(infer_ms)

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            test_maes.append(mae)
            test_rmses.append(rmse)
            test_r2s.append(r2)

        mean_infer_ms = float(np.mean(infer_times_ms))
        e2e_latency_ms = feat_calc_ms_per_win + mean_infer_ms

        row = {
            "context_window_W_sec": W,
            "num_features": X_train.shape[1],
            "train_windows": len(X_train),
            "train_time_s": float(train_time_s),
            "feat_calc_ms_per_window": float(feat_calc_ms_per_win),
            "model_infer_ms_per_window": float(mean_infer_ms),
            "end_to_end_latency_ms": float(e2e_latency_ms),
            "mean_test_mae": float(np.mean(test_maes)),
            "mean_test_rmse": float(np.mean(test_rmses)),
            "mean_test_r2": float(np.mean(test_r2s))
        }
        results.append(row)

        print(f"  Features: {row['num_features']} | "
              f"Feat Calc: {row['feat_calc_ms_per_window']:.3f} ms | "
              f"Infer: {row['model_infer_ms_per_window']:.3f} ms | "
              f"E2E: {row['end_to_end_latency_ms']:.3f} ms")
        print(f"  Test MAE: {row['mean_test_mae']:.3f} m/s | "
              f"Test RMSE: {row['mean_test_rmse']:.3f} m/s | "
              f"Test R²: {row['mean_test_r2']:.3f}")

    res_df = pd.DataFrame(results)
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase1_causal_window_sweep.csv")
    res_df.to_csv(out_path, index=False)

    print("\n" + "=" * 95)
    print(f"{'Context Window (s)':<20} | {'Feat Calc (ms)':<16} | {'Infer (ms)':<14} | {'E2E Latency (ms)':<18} | {'Test MAE (m/s)':<16} | {'Test R²':<10}")
    print("-" * 95)
    for r in results:
        print(f"{r['context_window_W_sec']:<20.1f} | {r['feat_calc_ms_per_window']:<16.3f} | {r['model_infer_ms_per_window']:<14.3f} | {r['end_to_end_latency_ms']:<18.3f} | {r['mean_test_mae']:<16.3f} | {r['mean_test_r2']:<10.3f}")
    print("=" * 95)
    print(f"[PASS] Causal window sweep saved to: {out_path}")
    return res_df

if __name__ == "__main__":
    run_causal_window_sweep()
