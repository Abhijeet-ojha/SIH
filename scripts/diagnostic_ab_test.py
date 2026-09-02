"""
scripts/diagnostic_ab_test.py
Quantitative A/B Diagnostic Script:
Evaluates the 6 IO-VNBD benchmark drives across 3 stages:
  - Stage 0: Baseline (5-State, No NHC, Derived Heading)
  - Stage 1: + Real 6-State NHC (v_lat ≈ 0 pseudo-measurement with driver-adaptive variance)
  - Stage 2: + Native GPS Receiver Orientation & Continuous Speed-Scaled R_h(v)
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from src.feature_engineering import extract_window_features
from src.speed_model import SpeedRegressorModel, reconstruct_ai_dr_trajectory
from src.fusion_ekf import KinematicFusionEKF, run_fusion_pipeline, VehicleContextEngine
from src.naive_dr import NaiveDeadReckoning
from src.metrics import calculate_benchmark_metrics

def run_legacy_5state_fusion(
    df: pd.DataFrame,
    ai_speed: np.ndarray,
    ai_speed_std: np.ndarray,
    driver_style: str = "normal",
    blackout_start_sec: float = 60.0,
    blackout_end_sec: float = 150.0,
    use_derived_heading: bool = True
) -> pd.DataFrame:
    """Simulates Stage 0 (Legacy 5-State EKF without NHC)."""
    n = len(df)
    t = df["timestamp"].values
    gyro_z = df["gyro_z"].values
    gps_x = df["pos_x"].values
    gps_y = df["pos_y"].values
    gps_v = df["speed"].values
    
    if use_derived_heading:
        # Recompute derived heading from dx/dy differences as in old loader
        dx = np.diff(gps_x, prepend=gps_x[0])
        dy = np.diff(gps_y, prepend=gps_y[0])
        dx_smooth = pd.Series(dx).rolling(7, min_periods=1, center=True).mean().values
        dy_smooth = pd.Series(dy).rolling(7, min_periods=1, center=True).mean().values
        course = np.arctan2(dx_smooth, dy_smooth)
        valid_heading = np.zeros(n)
        last_h = course[0] if len(course) > 0 else 0.0
        for i in range(n):
            if gps_v[i] > 0.4:
                last_h = course[i]
            valid_heading[i] = last_h
        gps_h = valid_heading
    else:
        gps_h = df["heading"].values if "heading" in df.columns else None

    acc_x = df["acc_x"].values
    acc_y = df["acc_y"].values
    ambient_lux = df["ambient_lux"].values if "ambient_lux" in df.columns else np.ones(n) * 1500.0

    dt_arr = np.diff(t, prepend=t[0])
    dt_arr[0] = dt_arr[1] if n > 1 else 0.1

    context_engine = VehicleContextEngine()
    init_x = gps_x[0]
    init_y = gps_y[0]
    init_v = gps_v[0]
    init_heading = gps_h[0] if gps_h is not None else 0.0

    # 5-State legacy filter
    x = np.array([init_x, init_y, init_v, init_heading, 0.0], dtype=float)
    P = np.diag([4.0, 4.0, 1.0, 0.05, 0.0001])
    q_pos = 0.05
    q_vel_base = 0.20 if driver_style.lower() in ["aggressive", "e"] else 0.12
    q_heading = 0.003
    q_bias = 1e-6
    R_gps = np.diag([4.0, 4.0, 0.16, 0.12**2])
    R_zupt = np.diag([0.04**2])

    fused_px = np.zeros(n)
    fused_py = np.zeros(n)
    fused_v = np.zeros(n)
    fused_theta = np.zeros(n)
    fused_bg = np.zeros(n)
    is_blackout = np.zeros(n, dtype=bool)
    context_modes = []
    pre_update_px = np.zeros(n)
    pre_update_py = np.zeros(n)

    win = 5
    for i in range(n):
        curr_t = t[i]
        dt = dt_arr[i]

        s_idx = max(0, i - win)
        e_idx = min(n, i + win)
        acc_var = np.var(acc_x[s_idx:e_idx]) + np.var(acc_y[s_idx:e_idx])
        gyro_abs = np.abs(gyro_z[i])

        speed_for_context = min(float(ai_speed[i]), float(gps_v[i]))
        mode = context_engine.update_context(ambient_lux[i], speed_for_context, acc_var, gyro_abs)
        context_modes.append(mode)
        is_stopped = (mode == "STANDSTILL")

        # 5-state predict
        px, py, v, theta, bg = x
        if is_stopped:
            v_eff = 0.0
            theta_new = theta
            alpha_v = 0.0
        else:
            alpha_v = 0.25 / (1.0 + 1.5 * max(0.0, ai_speed_std[i]))
            v_eff = (1.0 - alpha_v) * v + alpha_v * ai_speed[i]
            omega_corr = gyro_z[i] - bg
            theta_new = (theta + omega_corr * dt + np.pi) % (2 * np.pi) - np.pi

        px_new = px + v_eff * np.sin(theta_new) * dt
        py_new = py + v_eff * np.cos(theta_new) * dt
        x = np.array([px_new, py_new, v_eff, theta_new, bg])

        beta_u = 1.5
        q_v_dyn = q_vel_base + beta_u * (ai_speed_std[i]**2)
        q_b_dyn = q_bias * 0.1 if context_engine.tunnel_alert else q_bias
        Q = np.diag([q_pos**2, q_pos**2, q_v_dyn, q_heading**2, q_b_dyn**2])

        F = np.eye(5)
        if not is_stopped:
            dec = 1.0 - alpha_v
            F[0, 2] = dec * np.sin(theta_new) * dt
            F[0, 3] = v_eff * np.cos(theta_new) * dt
            F[0, 4] = -v_eff * np.cos(theta_new) * dt**2
            F[1, 2] = dec * np.cos(theta_new) * dt
            F[1, 3] = -v_eff * np.sin(theta_new) * dt
            F[1, 4] = v_eff * np.sin(theta_new) * dt**2
            F[2, 2] = dec
            F[3, 4] = -dt
        else:
            F[2, 2] = 0.0
        P = F @ P @ F.T + Q

        if is_stopped:
            # 5-state ZUPT
            I5 = np.eye(5)
            Hz = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
            yz = np.array([0.0 - x[2]])
            Sz = Hz @ P @ Hz.T + R_zupt
            Kz = P @ Hz.T @ np.linalg.inv(Sz)
            Kz[0, :] = 0; Kz[1, :] = 0; Kz[3, :] = 0; Kz[4, :] = 0
            x = x + Kz @ yz
            P = (I5 - Kz @ Hz) @ P @ (I5 - Kz @ Hz).T + Kz @ R_zupt @ Kz.T

        in_outage = (blackout_start_sec <= curr_t < blackout_end_sec)
        is_blackout[i] = in_outage
        pre_update_px[i] = x[0]
        pre_update_py[i] = x[1]

        if not in_outage:
            I5 = np.eye(5)
            z_pv = np.array([gps_x[i], gps_y[i], gps_v[i]])
            H_pv = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]], dtype=float)
            R_pv = np.diag([R_gps[0,0], R_gps[1,1], R_gps[2,2]])
            y_pv = z_pv - H_pv @ x
            S_pv = H_pv @ P @ H_pv.T + R_pv
            K_pv = P @ H_pv.T @ np.linalg.inv(S_pv)
            K_pv[3, :] = 0; K_pv[4, :] = 0
            x = x + K_pv @ y_pv
            P = (I5 - K_pv @ H_pv) @ P @ (I5 - K_pv @ H_pv).T + K_pv @ R_pv @ K_pv.T

            if gps_h is not None and gps_v[i] > 1.0:
                z_h = np.array([gps_h[i]])
                H_h = np.array([[0,0,0,1,0]], dtype=float)
                R_h = np.array([[R_gps[3,3]]])
                y_h = np.array([(z_h[0] - x[3] + np.pi) % (2*np.pi) - np.pi])
                S_h = H_h @ P @ H_h.T + R_h
                K_h = P @ H_h.T @ np.linalg.inv(S_h)
                K_h[0,:] = 0; K_h[1,:] = 0; K_h[2,:] = 0
                x = x + K_h @ y_h
                x[3] = (x[3] + np.pi) % (2*np.pi) - np.pi
                P = (I5 - K_h @ H_h) @ P @ (I5 - K_h @ H_h).T + K_h @ R_h @ K_h.T

        fused_px[i] = x[0]
        fused_py[i] = x[1]
        fused_v[i] = x[2]
        fused_theta[i] = x[3]
        fused_bg[i] = x[4]

    res = pd.DataFrame({
        "timestamp": t,
        "fused_pos_x": fused_px,
        "fused_pos_y": fused_py,
        "fused_velocity": fused_v,
        "fused_heading": fused_theta,
        "fused_gyro_bias": fused_bg,
        "is_gnss_blackout": is_blackout,
        "context_mode": context_modes,
        "open_loop_pos_x": pre_update_px,
        "open_loop_pos_y": pre_update_py
    })
    if "pos_x" in df.columns and "pos_y" in df.columns:
        dx = fused_px - df["pos_x"].values
        dy = fused_py - df["pos_y"].values
        res["fused_pos_error_m"] = np.sqrt(dx**2 + dy**2)
        dx_ol = pre_update_px - df["pos_x"].values
        dy_ol = pre_update_py - df["pos_y"].values
        res["open_loop_error_m"] = np.sqrt(dx_ol**2 + dy_ol**2)
    return res

def run_diagnostic():
    print("=" * 80)
    print("DIAGNOSTIC A/B PROGRESSION: STAGE 0 (BASELINE) vs STAGE 1 (6-STATE NHC) vs STAGE 2 (+HEADING)")
    print("=" * 80)

    # Load Speed Model
    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "speed_regressor.joblib")
    model = SpeedRegressorModel()
    model.load(model_path)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    test_drives = suite["test_drives"]

    stages = ["Stage 0 (5-State Baseline)", "Stage 1 (6-State Real NHC)", "Stage 2 (6-State NHC + Native GPS Heading)"]
    stage_results = {s: {} for s in stages}

    for drive in test_drives:
        df = drive.get_data()
        d_id = drive.driver_id
        d_name = drive.name
        blackout_start = 60.0
        blackout_end = min(drive.duration_sec - 10.0, 150.0)

        X_test, y_test, t_test = extract_window_features(df, window_sec=1.5, step_sec=0.2)
        y_pred, y_std = model.predict_with_uncertainty(X_test)
        init_heading = df["heading"].iloc[0] if "heading" in df.columns else 0.0
        init_speed = df["speed"].iloc[0] if "speed" in df.columns else 0.0
        init_pos = (df["pos_x"].iloc[0], df["pos_y"].iloc[0]) if "pos_x" in df.columns else (0.0, 0.0)

        naive_res = NaiveDeadReckoning(init_heading, init_speed, init_pos).compute(df)
        ai_dr_res = reconstruct_ai_dr_trajectory(df, t_test, y_pred, v_std=y_std, initial_heading=init_heading, initial_pos=init_pos)

        # Stage 0: Legacy 5-State, Derived Heading
        res_s0 = run_legacy_5state_fusion(
            df=df, ai_speed=ai_dr_res["ai_speed"].values, ai_speed_std=ai_dr_res["ai_speed_std"].values,
            driver_style="aggressive" if d_id == "E" else "normal",
            blackout_start_sec=blackout_start, blackout_end_sec=blackout_end, use_derived_heading=True
        )
        m_s0 = calculate_benchmark_metrics(df, naive_res, ai_dr_res, res_s0, blackout_start, blackout_end)
        stage_results[stages[0]][d_name] = {
            "exit_m": m_s0["ai_dr_gnss_ekf_fusion"]["blackout_terminal_exit_error_m"],
            "peak_m": m_s0["ai_dr_gnss_ekf_fusion"]["blackout_max_error_m"],
            "bias_end": float(res_s0["fused_gyro_bias"].iloc[-1])
        }

        # Stage 1: 6-State NHC with derived heading
        df_derived = df.copy()
        dx = np.diff(df["pos_x"].values, prepend=df["pos_x"].values[0])
        dy = np.diff(df["pos_y"].values, prepend=df["pos_y"].values[0])
        dx_s = pd.Series(dx).rolling(7, min_periods=1, center=True).mean().values
        dy_s = pd.Series(dy).rolling(7, min_periods=1, center=True).mean().values
        c_arr = np.arctan2(dx_s, dy_s)
        vh = np.zeros(len(df))
        lh = c_arr[0] if len(c_arr) > 0 else 0.0
        for i in range(len(df)):
            if df["speed"].iloc[i] > 0.4: lh = c_arr[i]
            vh[i] = lh
        df_derived["heading"] = vh

        res_s1 = run_fusion_pipeline(
            df=df_derived, ai_speed=ai_dr_res["ai_speed"].values, ai_speed_std=ai_dr_res["ai_speed_std"].values,
            driver_style="aggressive" if d_id == "E" else "normal",
            blackout_start_sec=blackout_start, blackout_end_sec=blackout_end
        )
        m_s1 = calculate_benchmark_metrics(df_derived, naive_res, ai_dr_res, res_s1, blackout_start, blackout_end)
        stage_results[stages[1]][d_name] = {
            "exit_m": m_s1["ai_dr_gnss_ekf_fusion"]["blackout_terminal_exit_error_m"],
            "peak_m": m_s1["ai_dr_gnss_ekf_fusion"]["blackout_max_error_m"],
            "bias_end": float(res_s1["fused_gyro_bias"].iloc[-1])
        }

        # Stage 2: 6-State NHC + Native GPS Heading
        res_s2 = run_fusion_pipeline(
            df=df, ai_speed=ai_dr_res["ai_speed"].values, ai_speed_std=ai_dr_res["ai_speed_std"].values,
            driver_style="aggressive" if d_id == "E" else "normal",
            blackout_start_sec=blackout_start, blackout_end_sec=blackout_end
        )
        m_s2 = calculate_benchmark_metrics(df, naive_res, ai_dr_res, res_s2, blackout_start, blackout_end)
        stage_results[stages[2]][d_name] = {
            "exit_m": m_s2["ai_dr_gnss_ekf_fusion"]["blackout_terminal_exit_error_m"],
            "peak_m": m_s2["ai_dr_gnss_ekf_fusion"]["blackout_max_error_m"],
            "bias_end": float(res_s2["fused_gyro_bias"].iloc[-1])
        }

    print("\n" + "=" * 95)
    print(f"{'Drive Name':<18} | {'Stage 0 Exit (m)':<18} | {'Stage 1 (NHC) Exit':<18} | {'Stage 2 (+Heading) Exit':<22}")
    print("-" * 95)
    for d in test_drives:
        dn = d.name
        e0 = stage_results[stages[0]][dn]["exit_m"]
        e1 = stage_results[stages[1]][dn]["exit_m"]
        e2 = stage_results[stages[2]][dn]["exit_m"]
        print(f"{dn:<18} | {e0:<18.2f} | {e1:<18.2f} | {e2:<22.2f}")
    
    print("-" * 95)
    m0 = np.mean([stage_results[stages[0]][d.name]["exit_m"] for d in test_drives])
    m1 = np.mean([stage_results[stages[1]][d.name]["exit_m"] for d in test_drives])
    m2 = np.mean([stage_results[stages[2]][d.name]["exit_m"] for d in test_drives])
    s0 = np.std([stage_results[stages[0]][d.name]["exit_m"] for d in test_drives])
    s1 = np.std([stage_results[stages[1]][d.name]["exit_m"] for d in test_drives])
    s2 = np.std([stage_results[stages[2]][d.name]["exit_m"] for d in test_drives])
    print(f"{'MEAN ± STD':<18} | {f'{m0:.2f} ± {s0:.2f} m':<18} | {f'{m1:.2f} ± {s1:.2f} m':<18} | {f'{m2:.2f} ± {s2:.2f} m':<22}")
    print("=" * 95)

    print("\nGYRO BIAS TRACKING STABILITY (rad/s at end of drive):")
    for d in test_drives:
        dn = d.name
        b0 = stage_results[stages[0]][dn]["bias_end"]
        b1 = stage_results[stages[1]][dn]["bias_end"]
        b2 = stage_results[stages[2]][dn]["bias_end"]
        print(f"  - {dn:15s} (Driver {d.driver_id}): Stage 0 = {b0:+.5f} | Stage 1 = {b1:+.5f} | Stage 2 = {b2:+.5f}")

if __name__ == "__main__":
    run_diagnostic()
