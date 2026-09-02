"""
scripts/ml_system_ablation.py
Phase 15: Full System Navigation Ablation Table.
Explicitly isolates the individual contributions of:
  1. Raw IMU Dead Reckoning (Double-integration baseline)
  2. AI-DR Pure (ML speed + gyro integration without EKF)
  3. ML Speed + EKF Fusion (Constant measurement variance)
  4. ML Speed + Calibrated Uncertainty + EKF
  5. ML Speed + Calibrated Uncertainty + EKF + Non-Holonomic Constraints (NHC)
  6. Full System (ML Speed + Calibrated Uncertainty + EKF + NHC + Dynamic ZUPT)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from src.feature_engineering import extract_causal_window_features
from src.speed_model import SpeedRegressorModel, reconstruct_ai_dr_trajectory
from src.naive_dr import NaiveDeadReckoning
from src.fusion_ekf import KinematicFusionEKF
from src.metrics import calculate_benchmark_metrics

def run_system_ablation():
    print("=" * 95)
    print("PHASE 15: FULL SYSTEM NAVIGATION ABLATION STUDY (90-SECOND GNSS OUTAGE)")
    print("=" * 95)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    print("\n[Step 1] Training speed regressor with calibrated uncertainty...")
    X_train_list, y_train_list = [], []
    for d in train_drives:
        df = d.get_data()
        X_df, y_spd, _, _ = extract_causal_window_features(df, window_sec=1.5, step_sec=0.2)
        X_train_list.append(X_df)
        y_train_list.append(y_spd)

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    model = SpeedRegressorModel(model_type="xgboost", n_estimators=100, max_depth=6, uncertainty_method="conformal")
    df_val = test_drives[0].get_data()
    val_X, val_y, _, _ = extract_causal_window_features(df_val, window_sec=1.5, step_sec=0.2)
    model.train(X_train, y_train, X_val=val_X, y_val=val_y)

    configurations = [
        ("1. Raw IMU Dead Reckoning (Naive Baseline)", "naive", False, False, False),
        ("2. ML Speed Pure (No EKF)", "ai_pure", False, False, False),
        ("3. ML Speed + EKF (Constant Sigma=0.5m/s)", "ekf_const", False, False, False),
        ("4. ML Speed + Calibrated Uncertainty + EKF", "ekf_calib", True, False, False),
        ("5. ML Speed + Uncertainty + EKF + NHC", "ekf_nhc", True, True, False),
        ("6. Full System (ML + Uncertainty + EKF + NHC + ZUPT)", "full", True, True, True)
    ]

    ablation_summary = []
    print("\n" + "=" * 95)
    print(f"{'System Configuration':<52} | {'90s Exit Error (m)':<20} | {'Peak Drift (m)':<16} | {'Settled (m)'}")
    print("-" * 95)

    for sys_label, sys_mode, use_unc, use_nhc, use_zupt in configurations:
        exit_errors = []
        peak_errors = []
        settled_errors = []

        for d in test_drives:
            df = d.get_data()
            d_id = d.driver_id
            b_start = 60.0
            b_end = min(d.duration_sec - 10.0, 150.0)

            init_heading = df["heading"].iloc[0] if "heading" in df.columns else 0.0
            init_speed = df["speed"].iloc[0] if "speed" in df.columns else 0.0
            init_pos = (df["pos_x"].iloc[0], df["pos_y"].iloc[0]) if "pos_x" in df.columns else (0.0, 0.0)

            X_te, y_te, t_te, _ = extract_causal_window_features(df, window_sec=1.5, step_sec=0.2)
            preds, stds = model.predict_with_uncertainty(X_te)

            t_arr = df["timestamp"].values
            b_mask = np.where((t_arr >= b_start) & (t_arr < b_end))[0]

            if sys_mode == "naive":
                naive_res = NaiveDeadReckoning(init_heading, init_speed, init_pos).compute(df)
                err_arr = naive_res["pos_error_m"].values
                exit_errors.append(float(err_arr[b_mask[-1]]) if len(b_mask) > 0 else float(err_arr[-1]))
                peak_errors.append(float(np.max(err_arr[b_mask])) if len(b_mask) > 0 else float(np.max(err_arr)))
                settled_errors.append(float(err_arr[-1]))

            elif sys_mode == "ai_pure":
                ai_res = reconstruct_ai_dr_trajectory(df, t_te, preds, v_std=stds, initial_heading=init_heading, initial_pos=init_pos)
                err_arr = ai_res["ai_pos_error_m"].values
                exit_errors.append(float(err_arr[b_mask[-1]]) if len(b_mask) > 0 else float(err_arr[-1]))
                peak_errors.append(float(np.max(err_arr[b_mask])) if len(b_mask) > 0 else float(np.max(err_arr)))
                settled_errors.append(float(err_arr[-1]))

            else:
                # Custom EKF run with ablation switches
                ai_res = reconstruct_ai_dr_trajectory(df, t_te, preds, v_std=stds if use_unc else None, initial_heading=init_heading, initial_pos=init_pos)
                
                # Run custom EKF
                n = len(df)
                t_arr = df["timestamp"].values
                dt_arr = np.diff(t_arr, prepend=t_arr[0])
                dt_arr[0] = 0.1

                ekf = KinematicFusionEKF(
                    init_x=init_pos[0],
                    init_y=init_pos[1],
                    init_v=init_speed,
                    init_heading=init_heading,
                    driver_style="aggressive" if d_id == "E" else "normal"
                )

                fused_pos_x = np.zeros(n)
                fused_pos_y = np.zeros(n)
                fused_speed = np.zeros(n)
                open_loop_x = np.zeros(n)
                open_loop_y = np.zeros(n)

                has_lat_lon = "pos_x" in df.columns and "pos_y" in df.columns
                gt_px = df["pos_x"].values if has_lat_lon else np.zeros(n)
                gt_py = df["pos_y"].values if has_lat_lon else np.zeros(n)

                v_ai = ai_res["ai_speed"].values
                v_ai_std = ai_res["ai_speed_std"].values if use_unc else np.ones(n) * 0.5
                acc_x = df["acc_x"].values
                acc_y = df["acc_y"].values
                acc_z = df["acc_z"].values
                gyro_z = df["gyro_z"].values

                for i in range(n):
                    dt = dt_arr[i]
                    cur_t = t_arr[i]
                    is_blackout = (cur_t >= b_start) and (cur_t < b_end)

                    is_stopped = False
                    if use_zupt:
                        a_mag = np.sqrt(acc_x[i]**2 + acc_y[i]**2 + acc_z[i]**2)
                        g_mag = abs(gyro_z[i])
                        if abs(a_mag - 9.81) < 0.25 and g_mag < 0.05 and v_ai[i] < 0.4:
                            is_stopped = True

                    sigma = v_ai_std[i] if use_unc else 0.5
                    ekf.predict(
                        dt=dt,
                        v_ai=v_ai[i],
                        v_ai_std=sigma,
                        gyro_z=gyro_z[i],
                        is_stationary=is_stopped
                    )

                    # NHC constraint
                    if use_nhc:
                        ekf.update_nhc()

                    # ZUPT constraint
                    if is_stopped and use_zupt:
                        ekf.update_zupt()

                    # Open-loop position before GPS correction
                    open_loop_x[i] = ekf.x[0]
                    open_loop_y[i] = ekf.x[1]

                    if not is_blackout and has_lat_lon:
                        h_meas = df["heading"].values[i] if "heading" in df.columns else None
                        ekf.update_gps(gps_x=gt_px[i], gps_y=gt_py[i], gps_speed=df["speed"].values[i], gps_heading=h_meas)

                    fused_pos_x[i] = ekf.x[0]
                    fused_pos_y[i] = ekf.x[1]

                # Metrics
                err = np.sqrt((fused_pos_x - gt_px)**2 + (fused_pos_y - gt_py)**2)
                ol_err = np.sqrt((open_loop_x - gt_px)**2 + (open_loop_y - gt_py)**2)

                b_mask = np.where((t_arr >= b_start) & (t_arr < b_end))[0]
                if len(b_mask) > 0:
                    exit_errors.append(float(ol_err[b_mask[-1]]))
                    peak_errors.append(float(np.max(ol_err[b_mask])))
                else:
                    exit_errors.append(float(err[-1]))
                    peak_errors.append(float(np.max(err)))

                settle_mask = np.where(t_arr >= (b_end + 8.0))[0]
                settled_errors.append(float(err[settle_mask[0]]) if len(settle_mask) > 0 else float(err[-1]))

        m_exit = float(np.mean(exit_errors))
        s_exit = float(np.std(exit_errors))
        m_peak = float(np.mean(peak_errors))
        s_peak = float(np.std(peak_errors))
        m_settle = float(np.mean(settled_errors))

        ablation_summary.append({
            "configuration": sys_label,
            "blackout_exit_error_m": m_exit,
            "blackout_exit_std_m": s_exit,
            "peak_drift_m": m_peak,
            "peak_drift_std_m": s_peak,
            "post_reacquisition_settled_m": m_settle
        })
        print(f"{sys_label:<52} | {f'{m_exit:.2f} +/- {s_exit:.1f} m':<20} | {f'{m_peak:.2f} +/- {s_peak:.1f} m':<16} | {f'{m_settle:.2f} m'}")

    print("=" * 95)
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase15_system_ablation.csv")
    pd.DataFrame(ablation_summary).to_csv(out_path, index=False)
    print(f"[PASS] Saved full system ablation: {out_path}")
    return pd.DataFrame(ablation_summary)

if __name__ == "__main__":
    run_system_ablation()
