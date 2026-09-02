"""
scripts/ml_system_ablation.py
System Ablation Study across 7 Incremental Architecture Stages (A -> G) for SIH 2026 PS-168:
  Stage A — Raw IMU Dead Reckoning (double integration)
  Stage B — ML Speed Dead Reckoning (ML speed + gyro integration)
  Stage C — ML + 6-State EKF
  Stage D — ML + Calibrated Uncertainty + 6-State EKF
  Stage E — ML + EKF + Non-Holonomic Constraints (NHC)
  Stage F — ML + EKF + NHC + Zero-Velocity Updates (ZUPT)
  Stage G — Full Pipeline (ML + Calibrated Uncertainty + 6-State EKF + NHC + ZUPT + Context Layer)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from core.features.extractor import CausalFeatureExtractor
from core.models.tabular_models import TabularSpeedModel
from core.fusion.ekf_6state import KinematicFusionEKF6State, MultiSensorContextEngine, wrap_angle
from src.naive_dr import NaiveDeadReckoning
from src.metrics import calculate_benchmark_metrics


def run_system_ablation():
    print("=" * 105)
    print("PHASE 3: COMPLETE 7-STAGE SYSTEM ABLATION BENCHMARK (A -> G)")
    print("=" * 105)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    extractor = CausalFeatureExtractor(window_sec=1.5, step_sec=0.2, sample_rate_hz=10.0, feature_group="all")
    
    # 1. Train Production Model
    X_train_list, y_train_list = [], []
    for d in train_drives:
        df = d.get_data()
        X_df, y_spd, _, _ = extractor.extract_features(df)
        X_train_list.append(X_df)
        y_train_list.append(y_spd)

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    calib_drive = train_drives[1] if len(train_drives) > 1 else train_drives[0]
    calib_df = calib_drive.get_data()
    X_calib, y_calib, _, _ = extractor.extract_features(calib_df)

    model = TabularSpeedModel(model_type="random_forest", n_estimators=100, max_depth=12, random_state=42, uncertainty_method="conformal")
    model.train(X_train, y_train, X_calib=X_calib, y_calib=y_calib)

    # Stages configuration
    stages = [
        ("Stage A: Raw IMU Dead Reckoning", "RAW_IMU"),
        ("Stage B: ML Speed Only (Pure AI-DR)", "ML_SPEED_ONLY"),
        ("Stage C: ML + 6-State EKF", "ML_EKF"),
        ("Stage D: ML + Calibrated Uncertainty + EKF", "ML_UNCERTAINTY_EKF"),
        ("Stage E: ML + EKF + NHC", "ML_EKF_NHC"),
        ("Stage F: ML + EKF + NHC + ZUPT", "ML_EKF_NHC_ZUPT"),
        ("Stage G: Full System Pipeline (+ Context Layer)", "FULL_SYSTEM")
    ]

    ablation_summary = []

    print("\n" + "=" * 105)
    print(f"{'System Architecture Stage':<48} | {'90s Blackout Exit Error':<24} | {'Peak Drift (m)':<15} | {'Traj RMSE'}")
    print("-" * 105)

    for stage_name, stage_key in stages:
        stage_exit_errs, stage_peak_errs, stage_traj_rmses = [], [], []

        for d in test_drives:
            df = d.get_data()
            t = df["timestamp"].values
            n = len(df)
            dt_arr = np.diff(t, prepend=t[0])
            dt_arr[0] = dt_arr[1] if n > 1 else 0.10

            init_heading = df["heading"].iloc[0] if "heading" in df.columns else 0.0
            init_speed = df["speed"].iloc[0] if "speed" in df.columns else 0.0
            init_pos = (df["pos_x"].iloc[0], df["pos_y"].iloc[0]) if "pos_x" in df.columns else (0.0, 0.0)

            X_test, y_test, t_test, _ = extractor.extract_features(df)
            preds, sigmas = model.predict_with_uncertainty(X_test)

            interp_func = interp1d(t_test, preds, kind="linear", bounds_error=False, fill_value=(preds[0], preds[-1]))
            v_dense = np.maximum(0.0, interp_func(t))
            interp_std = interp1d(t_test, sigmas, kind="linear", bounds_error=False, fill_value=(sigmas[0], sigmas[-1]))
            std_dense = np.maximum(0.05, interp_std(t))

            # Stage A: Raw IMU
            if stage_key == "RAW_IMU":
                naive_res = NaiveDeadReckoning(init_heading, init_speed, init_pos).compute(df)
                err = naive_res["pos_error_m"].values
                stage_exit_errs.append(float(err[-1]))
                stage_peak_errs.append(float(np.max(err)))
                stage_traj_rmses.append(float(np.sqrt(np.mean(err**2))))
                continue

            # Stage B: ML Speed Only
            if stage_key == "ML_SPEED_ONLY":
                h_ai = np.zeros(n)
                px_ai = np.zeros(n)
                py_ai = np.zeros(n)
                h_ai[0] = init_heading
                px_ai[0] = init_pos[0]
                py_ai[0] = init_pos[1]
                gyro_z = df["gyro_z"].values
                for i in range(1, n):
                    dt = dt_arr[i]
                    h_ai[i] = h_ai[i-1] + gyro_z[i] * dt
                    v_m = 0.5 * (v_dense[i-1] + v_dense[i])
                    h_m = 0.5 * (h_ai[i-1] + h_ai[i])
                    px_ai[i] = px_ai[i-1] + v_m * np.sin(h_m) * dt
                    py_ai[i] = py_ai[i-1] + v_m * np.cos(h_m) * dt
                dx = px_ai - df["pos_x"].values
                dy = py_ai - df["pos_y"].values
                err = np.sqrt(dx**2 + dy**2)
                stage_exit_errs.append(float(err[int(min(len(err)-1, 1500))]))
                stage_peak_errs.append(float(np.max(err)))
                stage_traj_rmses.append(float(np.sqrt(np.mean(err**2))))
                continue

            # Stages C through G: EKF Variants
            use_uncertainty = (stage_key in ["ML_UNCERTAINTY_EKF", "FULL_SYSTEM"])
            use_nhc = (stage_key in ["ML_EKF_NHC", "ML_EKF_NHC_ZUPT", "FULL_SYSTEM"])
            use_zupt = (stage_key in ["ML_EKF_NHC_ZUPT", "FULL_SYSTEM"])
            use_context = (stage_key == "FULL_SYSTEM")

            ekf = KinematicFusionEKF6State(
                init_x=init_pos[0], init_y=init_pos[1], init_v=init_speed, init_heading=init_heading,
                driver_style="aggressive" if d.driver_id == "E" else "normal"
            )
            context_engine = MultiSensorContextEngine()
            
            fused_px, fused_py = np.zeros(n), np.zeros(n)
            acc_x, acc_y = df["acc_x"].values, df["acc_y"].values
            gyro_z = df["gyro_z"].values
            gps_x, gps_y, gps_v = df["pos_x"].values, df["pos_y"].values, df["speed"].values
            ambient_lux = df["ambient_lux"].values if "ambient_lux" in df.columns else np.ones(n) * 1500.0

            win = 5
            for i in range(n):
                dt = dt_arr[i]
                curr_t = t[i]
                s_idx = max(0, i - win)
                e_idx = min(n, i + win)
                acc_var = float(np.var(acc_x[s_idx:e_idx]) + np.var(acc_y[s_idx:e_idx]))
                gyro_abs = float(np.abs(gyro_z[i]))

                mode = context_engine.update_context(ambient_lux[i], min(float(v_dense[i]), float(gps_v[i])), acc_var, gyro_abs) if use_context else "NORMAL_URBAN"
                is_stopped = (acc_var < 0.018 and gyro_abs < 0.01 and v_dense[i] < 0.5) if use_zupt else False

                v_sigma_feed = std_dense[i] if use_uncertainty else 0.20
                ekf.predict(dt=dt, v_ai=v_dense[i], v_ai_std=v_sigma_feed, gyro_z=gyro_z[i], is_stationary=is_stopped, is_tunnel_alert=(mode == "PREDICTIVE_TUNNEL_BLACKOUT"))

                if use_nhc:
                    ekf.update_nhc()
                if is_stopped and use_zupt:
                    ekf.update_zupt()

                in_outage = (60.0 <= curr_t < min(d.duration_sec - 10.0, 150.0))
                if not in_outage:
                    ekf.update_gps(gps_x=gps_x[i], gps_y=gps_y[i], gps_speed=gps_v[i], gps_heading=None)

                fused_px[i] = ekf.x[0]
                fused_py[i] = ekf.x[1]

            # Calculate error during outage
            outage_mask = (t >= 60.0) & (t < min(d.duration_sec - 10.0, 150.0))
            dx = fused_px - gps_x
            dy = fused_py - gps_y
            pos_err = np.sqrt(dx**2 + dy**2)

            outage_err = pos_err[outage_mask] if np.any(outage_mask) else pos_err
            stage_exit_errs.append(float(outage_err[-1]))
            stage_peak_errs.append(float(np.max(outage_err)))
            stage_traj_rmses.append(float(np.sqrt(np.mean(pos_err**2))))

        mean_exit = float(np.mean(stage_exit_errs))
        mean_peak = float(np.mean(stage_peak_errs))
        mean_rmse = float(np.mean(stage_traj_rmses))

        print(f"{stage_name:<48} | {mean_exit:<24.2f} | {mean_peak:<15.2f} | {mean_rmse:.2f} m")

        ablation_summary.append({
            "stage_name": stage_name,
            "stage_key": stage_key,
            "blackout_90s_exit_error_m": mean_exit,
            "blackout_90s_peak_drift_m": mean_peak,
            "trajectory_rmse_m": mean_rmse
        })

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "system_ablation_scorecard.csv")
    pd.DataFrame(ablation_summary).to_csv(out_path, index=False)
    print(f"\n[PASS] Saved System Ablation Scorecard to: {out_path}")
    print("=" * 105)
    return pd.DataFrame(ablation_summary)


if __name__ == "__main__":
    run_system_ablation()
