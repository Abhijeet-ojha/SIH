"""
scripts/ml_phase1_causal_sweep.py
Phase 1: Causal Window Size Sweep & Latency Benchmark for SIH 2026 PS-168.
Evaluates context windows W in [0.5s, 1.0s, 1.5s, 2.0s, 3.0s] on real IO-VNBD data.
Measures context window length, feature computation latency, model inference latency,
end-to-end latency, MAE, RMSE, R2, P95 absolute error, and downstream 90s blackout navigation error.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from core.features.extractor import CausalFeatureExtractor
from core.models.tabular_models import TabularSpeedModel
from core.fusion.ekf_6state import run_6state_ekf_fusion
from src.metrics import calculate_benchmark_metrics
from src.naive_dr import NaiveDeadReckoning
from scipy.interpolate import interp1d


def run_causal_window_sweep():
    print("=" * 95)
    print("PHASE 1: CAUSAL WINDOW LENGTH SWEEP & LATENCY PROFILING (0.5s - 3.0s)")
    print("=" * 95)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    window_sizes = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = []

    for W in window_sizes:
        print(f"\n--- Evaluating Causal Context Window W = {W:.1f}s ---")
        extractor = CausalFeatureExtractor(window_sec=W, step_sec=0.2, sample_rate_hz=10.0, feature_group="all")
        
        # 1. Feature extraction on training drives
        X_train_list, y_train_list = [], []
        t_feat_start = time.perf_counter()
        total_windows_extracted = 0

        for d in train_drives:
            df = d.get_data()
            X_df, y_spd, _, lat = extractor.extract_features(df)
            X_train_list.append(X_df)
            y_train_list.append(y_spd)
            total_windows_extracted += len(X_df)

        feat_calc_total_ms = (time.perf_counter() - t_feat_start) * 1000.0
        feat_calc_ms_per_win = feat_calc_total_ms / max(1, total_windows_extracted)

        X_train = pd.concat(X_train_list, ignore_index=True)
        y_train = np.concatenate(y_train_list)

        # 2. Train Random Forest model
        model = TabularSpeedModel(model_type="random_forest", n_estimators=100, max_depth=12, random_state=42)
        t_train_start = time.perf_counter()
        model.train(X_train, y_train)
        train_time_s = time.perf_counter() - t_train_start

        # 3. Evaluate on test drives & profile inference & navigation
        test_maes, test_rmses, test_r2s, test_p95s = [], [], [], []
        infer_times_ms = []
        nav_blackout_exit_errors = []

        for d in test_drives:
            df = d.get_data()
            X_test, y_test, t_test, _ = extractor.extract_features(df)
            
            # Profile per-window inference latency
            t0 = time.perf_counter()
            preds, sigmas = model.predict_with_uncertainty(X_test)
            infer_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(X_test))
            infer_times_ms.append(infer_ms)

            abs_err = np.abs(y_test - preds)
            mae = float(np.mean(abs_err))
            rmse = float(np.sqrt(np.mean(abs_err**2)))
            r2 = float(1.0 - (np.sum((y_test - preds)**2) / (np.sum((y_test - np.mean(y_test))**2) + 1e-9)))
            p95 = float(np.percentile(abs_err, 95))

            test_maes.append(mae)
            test_rmses.append(rmse)
            test_r2s.append(r2)
            test_p95s.append(p95)

            # Downstream navigation blackout
            t_orig = df["timestamp"].values
            interp_func = interp1d(t_test, preds, kind="linear", bounds_error=False, fill_value=(preds[0], preds[-1]))
            v_dense = np.maximum(0.0, interp_func(t_orig))
            interp_std = interp1d(t_test, sigmas, kind="linear", bounds_error=False, fill_value=(sigmas[0], sigmas[-1]))
            std_dense = np.maximum(0.05, interp_std(t_orig))

            fused_res = run_6state_ekf_fusion(
                df=df,
                ai_speed=v_dense,
                ai_speed_std=std_dense,
                driver_style="aggressive" if d.driver_id == "E" else "normal",
                blackout_start_sec=60.0,
                blackout_end_sec=min(d.duration_sec - 10.0, 150.0)
            )
            naive_res = NaiveDeadReckoning(
                initial_heading=df["heading"].iloc[0] if "heading" in df.columns else 0.0,
                initial_speed=df["speed"].iloc[0] if "speed" in df.columns else 0.0,
                initial_pos=(df["pos_x"].iloc[0], df["pos_y"].iloc[0]) if "pos_x" in df.columns else (0.0, 0.0)
            ).compute(df)
            
            ai_dr_res = pd.DataFrame({"timestamp": t_orig, "ai_pos_error_m": np.zeros(len(df))})
            m = calculate_benchmark_metrics(df, naive_res, ai_dr_res, fused_res, 60.0, min(d.duration_sec - 10.0, 150.0))
            nav_blackout_exit_errors.append(float(m["ai_dr_gnss_ekf_fusion"]["blackout_terminal_exit_error_m"]))

        mean_infer_ms = float(np.mean(infer_times_ms))
        e2e_latency_ms = feat_calc_ms_per_win + mean_infer_ms

        row = {
            "context_window_W_sec": W,
            "window_samples": int(round(W * 10.0)),
            "num_features": X_train.shape[1],
            "train_windows": len(X_train),
            "feat_calc_ms_per_window": float(feat_calc_ms_per_win),
            "model_infer_ms_per_window": float(mean_infer_ms),
            "end_to_end_latency_ms": float(e2e_latency_ms),
            "mean_test_mae": float(np.mean(test_maes)),
            "mean_test_rmse": float(np.mean(test_rmses)),
            "mean_test_r2": float(np.mean(test_r2s)),
            "p95_error_mps": float(np.mean(test_p95s)),
            "nav_blackout_exit_error_m": float(np.mean(nav_blackout_exit_errors))
        }
        results.append(row)

        print(f"  Features: {row['num_features']} | "
              f"Feat Lat: {row['feat_calc_ms_per_window']:.3f} ms | "
              f"Infer Lat: {row['model_infer_ms_per_window']:.3f} ms | "
              f"E2E Lat: {row['end_to_end_latency_ms']:.3f} ms | "
              f"MAE: {row['mean_test_mae']:.3f} m/s | "
              f"RMSE: {row['mean_test_rmse']:.3f} m/s | "
              f"P95: {row['p95_error_mps']:.3f} m/s | "
              f"90s Exit Error: {row['nav_blackout_exit_error_m']:.2f} m")

    res_df = pd.DataFrame(results)
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase1_causal_window_sweep.csv")
    res_df.to_csv(out_path, index=False)
    res_df.to_csv(os.path.join(out_dir, "window_sweep.csv"), index=False)
    print(f"\n[PASS] Saved Causal Window Sweep results to: {out_path} and window_sweep.csv")
    print("=" * 95)
    return res_df


if __name__ == "__main__":
    run_causal_window_sweep()
