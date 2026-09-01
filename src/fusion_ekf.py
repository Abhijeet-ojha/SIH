"""
src/fusion_ekf.py
Confidence-Aware 5-State Kinematic Extended Kalman Filter (EKF) with Driver-Style Adaptive Constraints
and Multi-Sensor Predictive Context Layer.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List

def wrap_angle(rad: float) -> float:
    return (rad + np.pi) % (2 * np.pi) - np.pi

class VehicleContextEngine:
    """
    Multi-sensor predictive context detection engine.
    Detects environmental conditions and pre-empts GPS blackouts.
    """
    def __init__(self):
        self.current_mode = "NORMAL_URBAN"
        self.tunnel_alert = False

    def update_context(self, ambient_lux: float, speed_mps: float, acc_var: float, gyro_abs: float) -> str:
        # 1. Standstill / Red Light Stop
        if acc_var < 0.018 and gyro_abs < 0.01 and speed_mps < 0.5:
            self.current_mode = "STANDSTILL"
            self.tunnel_alert = False
            return self.current_mode

        # 2. Predictive Tunnel / Underground Parking Detection
        # Sudden drop in ambient light (< 100 lux) while travelling at speed (> 4 m/s)
        if ambient_lux < 100.0 and speed_mps > 4.0:
            self.current_mode = "PREDICTIVE_TUNNEL_BLACKOUT"
            self.tunnel_alert = True
            return self.current_mode
        else:
            self.tunnel_alert = False

        # 3. Highway Cruising vs Urban
        if speed_mps > 18.0:
            self.current_mode = "HIGHWAY_CRUISING"
        else:
            self.current_mode = "NORMAL_URBAN"

        return self.current_mode

class KinematicFusionEKF:
    """
    Confidence-Aware 5-State Navigation Extended Kalman Filter.
    State Vector: x = [pos_x (East), pos_y (North), forward_velocity (m/s), heading (rad), gyro_bias (rad/s)]^T
    """

    def __init__(
        self,
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_v: float = 0.0,
        init_heading: float = 0.0,
        init_gyro_bias: float = 0.0,
        driver_style: str = "normal",
        q_pos: float = 0.05,
        q_vel_base: float = 0.12,
        q_heading: float = 0.003,
        q_bias: float = 1e-6,
        r_gps_pos: float = 2.0,
        r_gps_vel: float = 0.4,
        r_gps_heading: float = 0.12
    ):
        self.driver_style = driver_style
        self.x = np.array([init_x, init_y, init_v, init_heading, init_gyro_bias], dtype=float)
        self.P = np.diag([4.0, 4.0, 1.0, 0.05, 0.0001])
        
        self.q_pos = q_pos
        self.q_vel_base = q_vel_base
        self.q_heading = q_heading
        self.q_bias = q_bias

        # Driver-Adaptive Physical Constraint Parameters
        if driver_style.lower() in ["aggressive", "e"]:
            self.nhc_lateral_variance = 0.25**2
            self.zupt_acc_thresh = 0.035
            self.q_vel_base = 0.20
        else:
            self.nhc_lateral_variance = 0.05**2
            self.zupt_acc_thresh = 0.018

        self.R_gps = np.diag([r_gps_pos**2, r_gps_pos**2, r_gps_vel**2, r_gps_heading**2])
        self.R_gps_nohdg = np.diag([r_gps_pos**2, r_gps_pos**2, r_gps_vel**2])
        self.R_zupt = np.diag([0.04**2])

    def predict(self, dt: float, v_ai: float, v_ai_std: float, gyro_z: float, is_stationary: bool = False, is_tunnel_alert: bool = False):
        """
        Confidence-Aware Prediction Step:
        Dynamically scales the velocity blending weight alpha_v and process noise Q(t)
        based on AI model uncertainty (v_ai_std).
        """
        px, py, v, theta, bg = self.x

        if is_stationary:
            v_eff = 0.0
            theta_new = theta
            alpha_v = 0.0
        else:
            # Dynamically modulate alpha based on AI model uncertainty sigma_v
            # Confident (sigma -> 0) => alpha ~ 0.25; Uncertain (sigma -> 2.0) => alpha ~ 0.06
            alpha_base = 0.25
            alpha_v = alpha_base / (1.0 + 1.5 * max(0.0, v_ai_std))
            v_eff = (1.0 - alpha_v) * v + alpha_v * v_ai
            omega_corr = gyro_z - bg
            theta_new = wrap_angle(theta + omega_corr * dt)

        px_new = px + v_eff * np.sin(theta_new) * dt
        py_new = py + v_eff * np.cos(theta_new) * dt
        v_new = v_eff
        bg_new = bg

        self.x = np.array([px_new, py_new, v_new, theta_new, bg_new])

        # Dynamic Confidence-Aware Process Noise Q(t)
        beta_uncertainty = 1.5
        q_vel_dynamic = self.q_vel_base + beta_uncertainty * (v_ai_std**2)
        
        # In predictive tunnel alert mode, lock gyro bias random walk
        q_bias_dynamic = self.q_bias * 0.1 if is_tunnel_alert else self.q_bias
        
        Q = np.diag([self.q_pos**2, self.q_pos**2, q_vel_dynamic, self.q_heading**2, q_bias_dynamic**2])

        # Jacobian F (5x5)
        F = np.eye(5)
        if not is_stationary:
            decay = (1.0 - alpha_v)
            F[0, 2] = decay * np.sin(theta_new) * dt
            F[0, 3] = v_eff * np.cos(theta_new) * dt
            F[0, 4] = -v_eff * np.cos(theta_new) * dt**2
            F[1, 2] = decay * np.cos(theta_new) * dt
            F[1, 3] = -v_eff * np.sin(theta_new) * dt
            F[1, 4] = v_eff * np.sin(theta_new) * dt**2
            F[2, 2] = decay
            F[3, 4] = -dt
        else:
            F[2, 2] = 0.0

        self.P = F @ self.P @ F.T + Q

    def update_gps(self, gps_x: float, gps_y: float, gps_speed: float, gps_heading: Optional[float] = None):
        """
        Partitioned Sequential GPS Update with Joseph-form Covariance Projection.
        Step 1: Pos & Vel Update (3-state) -> updates [pos_x, pos_y, forward_v]
        Step 2: Course Heading Update (1-state) -> updates [heading, gyro_bias]
        """
        I = np.eye(5)

        # ── Step 1: Position and Velocity Measurement Update ─────────────────
        z_pv = np.array([gps_x, gps_y, gps_speed])
        H_pv = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0]
        ])
        R_pv = np.diag([self.R_gps[0, 0], self.R_gps[1, 1], self.R_gps[2, 2]])
        y_pv = z_pv - H_pv @ self.x
        S_pv = H_pv @ self.P @ H_pv.T + R_pv
        K_pv = self.P @ H_pv.T @ np.linalg.inv(S_pv)

        # Decouple: Position & speed carry no physical observability of gyro bias or heading
        K_pv[3, :] = 0.0
        K_pv[4, :] = 0.0

        self.x = self.x + K_pv @ y_pv
        # Joseph Form Covariance Update: P = (I - KH) P (I - KH)^T + K R K^T
        IKH_pv = I - K_pv @ H_pv
        self.P = IKH_pv @ self.P @ IKH_pv.T + K_pv @ R_pv @ K_pv.T

        # ── Step 2: Course Heading Measurement Update (when moving) ──────────
        if gps_heading is not None and gps_speed > 1.0:
            z_h = np.array([gps_heading])
            H_h = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
            R_h = np.array([[self.R_gps[3, 3]]])
            y_h = np.array([wrap_angle(z_h[0] - self.x[3])])
            S_h = H_h @ self.P @ H_h.T + R_h
            K_h = self.P @ H_h.T @ np.linalg.inv(S_h)

            # Heading only updates heading and bias, not position/velocity directly
            K_h[0, :] = 0.0
            K_h[1, :] = 0.0
            K_h[2, :] = 0.0

            self.x = self.x + K_h @ y_h
            self.x[3] = wrap_angle(self.x[3])
            IKH_h = I - K_h @ H_h
            self.P = IKH_h @ self.P @ IKH_h.T + K_h @ R_h @ K_h.T

    def update_zupt(self):
        """Zero-Velocity Update (ZUPT) using Joseph-form projection."""
        I = np.eye(5)
        H = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
        y = np.array([0.0 - self.x[2]])
        S = H @ self.P @ H.T + self.R_zupt
        K = self.P @ H.T @ np.linalg.inv(S)

        # Velocity standstill observation only updates velocity
        K[0, :] = 0.0
        K[1, :] = 0.0
        K[3, :] = 0.0
        K[4, :] = 0.0

        self.x = self.x + K @ y
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_zupt @ K.T

def run_fusion_pipeline(
    df: pd.DataFrame,
    ai_speed: np.ndarray,
    ai_speed_std: Optional[np.ndarray] = None,
    driver_style: str = "normal",
    blackout_start_sec: float = 60.0,
    blackout_end_sec: float = 150.0
) -> pd.DataFrame:
    """
    Executes the Confidence-Aware 5-State EKF with Multi-Sensor Context Engine and Adaptive NHC.
    """
    n = len(df)
    t = df["timestamp"].values
    gyro_z = df["gyro_z"].values
    gps_x = df["pos_x"].values
    gps_y = df["pos_y"].values
    gps_v = df["speed"].values
    gps_h = df["heading"].values if "heading" in df.columns else None
    acc_x = df["acc_x"].values
    acc_y = df["acc_y"].values
    ambient_lux = df["ambient_lux"].values if "ambient_lux" in df.columns else np.ones(n) * 1500.0

    if ai_speed_std is None:
        ai_speed_std = np.ones(n) * 0.2

    dt_arr = np.diff(t, prepend=t[0])
    dt_arr[0] = dt_arr[1] if n > 1 else 0.1

    context_engine = VehicleContextEngine()

    init_x = gps_x[0]
    init_y = gps_y[0]
    init_v = gps_v[0]
    init_heading = gps_h[0] if gps_h is not None else 0.0

    ekf = KinematicFusionEKF(
        init_x=init_x,
        init_y=init_y,
        init_v=init_v,
        init_heading=init_heading,
        driver_style=driver_style
    )

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

        # Multi-sensor context detection.
        # IMPORTANT: Use min(ai_speed, gps_speed) for STANDSTILL detection.
        # The AI speed model returns ~0.1-0.2 m/s at genuine stops due to model noise,
        # which would suppress the STANDSTILL flag and allow the kinematic predict()
        # to accumulate positional residual against the GPS fix — that residual then
        # backpropagates through K[4,0] into the gyro bias state, driving it to
        # physically implausible values (up to -0.15 rad/s) over a multi-second stop.
        # GPS speed is reliable enough at update time for stop/go detection.
        speed_for_context = min(float(ai_speed[i]), float(gps_v[i]))
        mode = context_engine.update_context(
            ambient_lux=ambient_lux[i],
            speed_mps=speed_for_context,
            acc_var=acc_var,
            gyro_abs=gyro_abs
        )
        context_modes.append(mode)

        is_stopped = (mode == "STANDSTILL")

        # 1. State prediction with Confidence-Aware dynamic alpha and Q(t)
        ekf.predict(
            dt=dt,
            v_ai=ai_speed[i],
            v_ai_std=ai_speed_std[i],
            gyro_z=gyro_z[i],
            is_stationary=is_stopped,
            is_tunnel_alert=context_engine.tunnel_alert
        )

        if is_stopped:
            ekf.update_zupt()

        # Outage check: strictly open loop in [start, end)
        in_outage = (blackout_start_sec <= curr_t < blackout_end_sec)
        is_blackout[i] = in_outage

        # Capture open loop state BEFORE measurement update
        pre_update_px[i] = ekf.x[0]
        pre_update_py[i] = ekf.x[1]

        # 2. Measurement update during healthy GNSS window
        if not in_outage:
            hdg_meas = gps_h[i] if gps_h is not None and gps_v[i] > 1.0 else None
            ekf.update_gps(gps_x=gps_x[i], gps_y=gps_y[i], gps_speed=gps_v[i], gps_heading=hdg_meas)

        fused_px[i] = ekf.x[0]
        fused_py[i] = ekf.x[1]
        fused_v[i] = ekf.x[2]
        fused_theta[i] = ekf.x[3]
        fused_bg[i] = ekf.x[4]

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
