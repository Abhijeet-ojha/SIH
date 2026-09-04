"""
src/fusion_ekf.py
Confidence-Aware 5-State Kinematic Extended Kalman Filter (EKF) with Driver-Style Adaptive Constraints
and Multi-Sensor Predictive Context Layer.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List

try:
    from .frame_alignment import align_frame
    from .motion_gate import MotionGate
except ImportError:  # direct script execution
    from frame_alignment import align_frame
    from motion_gate import MotionGate

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

    def update_context(self, ambient_lux: float, speed_mps: float, acc_var: float,
                       gyro_abs: float, gate_state: Optional[str] = None) -> str:
        # 0. Phone being handled. This outranks everything: if the body frame is no longer
        #    the vehicle frame, no inertial quantity we derive from it means anything.
        if gate_state == "PHONE_HANDLED":
            self.current_mode = "PHONE_HANDLED"
            self.tunnel_alert = False
            return self.current_mode

        # 1. Standstill / Red Light Stop
        if gate_state == "STATIONARY" or (acc_var < 0.018 and gyro_abs < 0.01 and speed_mps < 0.5):
            self.current_mode = "STANDSTILL"
            self.tunnel_alert = False
            return self.current_mode

        # 2. Predictive Tunnel / Underground Parking Detection
        # Sudden drop in ambient light (< 100 lux) while travelling at speed (> 4 m/s).
        # NaN when the phone has no light sensor, and NaN < 100 is False, so an absent
        # sensor correctly disables this rather than silently reporting daylight.
        if np.isfinite(ambient_lux) and ambient_lux < 100.0 and speed_mps > 4.0:
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
    Confidence-Aware 6-State Navigation Extended Kalman Filter with Real Non-Holonomic Constraints (NHC).
    State Vector: x = [pos_x (East), pos_y (North), v_fwd (m/s), v_lat (m/s), heading (rad), gyro_bias (rad/s)]^T
    """

    def __init__(
        self,
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_v: float = 0.0,
        init_v_lat: float = 0.0,
        init_heading: float = 0.0,
        init_gyro_bias: float = 0.0,
        driver_style: str = "normal",
        q_pos: float = 0.05,
        q_vel_base: float = 0.12,
        q_vel_lat: float = 0.05,
        q_heading: float = 0.003,
        q_bias: float = 1e-6,
        r_gps_pos: float = 2.0,
        r_gps_vel: float = 0.4,
        r_gps_heading: float = 0.12
    ):
        self.driver_style = driver_style
        self.x = np.array([init_x, init_y, init_v, init_v_lat, init_heading, init_gyro_bias], dtype=float)
        self.P = np.diag([4.0, 4.0, 1.0, 0.25, 0.05, 0.0001])
        
        self.q_pos = q_pos
        self.q_vel_base = q_vel_base
        self.q_vel_lat = q_vel_lat
        self.q_heading = q_heading
        self.q_bias = q_bias
        self.r_gps_heading_base = r_gps_heading

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
        self.R_zupt = np.diag([0.04**2, 0.04**2])

    def predict(self, dt: float, v_ai: float, v_ai_std: float, gyro_z: float, is_stationary: bool = False, is_tunnel_alert: bool = False):
        """
        Confidence-Aware 6-State Prediction Step:
        Propagates 2D body velocities [v_fwd, v_lat], yaw, and positions.
        Blends forward velocity with AI model uncertainty (v_ai_std).
        """
        px, py, v_fwd, v_lat, theta, bg = self.x

        if is_stationary:
            v_fwd_eff = 0.0
            v_lat_eff = 0.0
            theta_new = theta
            alpha_v = 0.0
            alpha_lat = 0.0
        else:
            # Dynamically modulate alpha based on AI model uncertainty sigma_v
            alpha_base = 0.25
            alpha_v = alpha_base / (1.0 + 1.5 * max(0.0, v_ai_std))
            v_fwd_eff = (1.0 - alpha_v) * v_fwd + alpha_v * v_ai
            
            # Lateral velocity natural decay in body frame
            alpha_lat = 0.10
            v_lat_eff = (1.0 - alpha_lat) * v_lat
            
            omega_corr = gyro_z - bg
            theta_new = wrap_angle(theta + omega_corr * dt)

        # Kinematics in ENU coordinate system:
        # px (East)  dot = v_fwd * sin(theta) + v_lat * cos(theta)
        # py (North) dot = v_fwd * cos(theta) - v_lat * sin(theta)
        px_new = px + (v_fwd_eff * np.sin(theta_new) + v_lat_eff * np.cos(theta_new)) * dt
        py_new = py + (v_fwd_eff * np.cos(theta_new) - v_lat_eff * np.sin(theta_new)) * dt
        v_fwd_new = v_fwd_eff
        v_lat_new = v_lat_eff
        bg_new = bg

        self.x = np.array([px_new, py_new, v_fwd_new, v_lat_new, theta_new, bg_new])

        # Dynamic Confidence-Aware Process Noise Q(t)
        beta_uncertainty = 1.5
        q_vel_dynamic = self.q_vel_base + beta_uncertainty * (v_ai_std**2)
        q_bias_dynamic = self.q_bias * 0.1 if is_tunnel_alert else self.q_bias
        
        Q = np.diag([
            self.q_pos**2,
            self.q_pos**2,
            q_vel_dynamic,
            self.q_vel_lat**2,
            self.q_heading**2,
            q_bias_dynamic**2
        ])

        # Jacobian F (6x6)
        F = np.eye(6)
        if not is_stationary:
            decay_fwd = (1.0 - alpha_v)
            decay_lat = (1.0 - alpha_lat)
            sin_t = np.sin(theta_new)
            cos_t = np.cos(theta_new)

            # d(px_new)/d(v_fwd), d(px_new)/d(v_lat), d(px_new)/d(theta), d(px_new)/d(bg)
            F[0, 2] = decay_fwd * sin_t * dt
            F[0, 3] = decay_lat * cos_t * dt
            F[0, 4] = (v_fwd_eff * cos_t - v_lat_eff * sin_t) * dt
            F[0, 5] = -(v_fwd_eff * cos_t - v_lat_eff * sin_t) * (dt**2)

            # d(py_new)/d(v_fwd), d(py_new)/d(v_lat), d(py_new)/d(theta), d(py_new)/d(bg)
            F[1, 2] = decay_fwd * cos_t * dt
            F[1, 3] = -decay_lat * sin_t * dt
            F[1, 4] = -(v_fwd_eff * sin_t + v_lat_eff * cos_t) * dt
            F[1, 5] = (v_fwd_eff * sin_t + v_lat_eff * cos_t) * (dt**2)

            F[2, 2] = decay_fwd
            F[3, 3] = decay_lat
            F[4, 5] = -dt
        else:
            F[2, 2] = 0.0
            F[3, 3] = 0.0

        self.P = F @ self.P @ F.T + Q

    def update_nhc(self, lateral_variance: Optional[float] = None):
        """
        Non-Holonomic Constraint (NHC) Pseudo-Measurement Update:
        Ground vehicle physical constraint: lateral velocity v_lat ≈ 0 in vehicle body frame.
        Applies Joseph-form covariance projection with driver-adaptive noise variance.
        """
        I = np.eye(6)
        r_nhc = lateral_variance if lateral_variance is not None else self.nhc_lateral_variance
        
        # Observation matrix H: measures v_lat (state index 3)
        H_nhc = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        R_nhc = np.array([[r_nhc]])
        
        # Virtual observation z = 0.0
        y = np.array([0.0 - self.x[3]])
        S = H_nhc @ self.P @ H_nhc.T + R_nhc
        K = self.P @ H_nhc.T @ np.linalg.inv(S)

        # NHC pseudo-measurement updates lateral velocity & position, decouple from gyro bias
        K[5, :] = 0.0

        self.x = self.x + K @ y
        IKH = I - K @ H_nhc
        self.P = IKH @ self.P @ IKH.T + K @ R_nhc @ K.T

    def update_gps(self, gps_x: float, gps_y: float, gps_speed: float, gps_heading: Optional[float] = None):
        """
        Partitioned Sequential GPS Update with Joseph-form Covariance Projection.
        Step 1: Pos & Forward Vel Update (3-state) -> updates [pos_x, pos_y, forward_v]
        Step 2: Course Heading Update (1-state) with continuous inverse-speed variance scaling.
        """
        I = np.eye(6)

        # ── Step 1: Position and Velocity Measurement Update ─────────────────
        z_pv = np.array([gps_x, gps_y, gps_speed])
        H_pv = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ])
        R_pv = np.diag([self.R_gps[0, 0], self.R_gps[1, 1], self.R_gps[2, 2]])
        y_pv = z_pv - H_pv @ self.x
        S_pv = H_pv @ self.P @ H_pv.T + R_pv
        K_pv = self.P @ H_pv.T @ np.linalg.inv(S_pv)

        # Decouple: Position & forward speed carry no physical observability of gyro bias or heading
        K_pv[4, :] = 0.0
        K_pv[5, :] = 0.0

        self.x = self.x + K_pv @ y_pv
        # Joseph Form Covariance Update: P = (I - KH) P (I - KH)^T + K R K^T
        IKH_pv = I - K_pv @ H_pv
        self.P = IKH_pv @ self.P @ IKH_pv.T + K_pv @ R_pv @ K_pv.T

        # ── Step 2: Course Heading Measurement Update with Speed-Weighted Variance ──
        if gps_heading is not None:
            # Continuous inverse-speed variance scaling:
            # When speed is high (> 5 m/s), R_h ~ R_h_base
            # When speed is low (< 1 m/s), R_h scales up quadratically, naturally distrusting jitter
            v_ref = max(float(gps_speed), 0.2)
            v_scale = 1.0 + (1.5 / v_ref)**2
            r_h_dynamic = (self.r_gps_heading_base**2) * v_scale

            z_h = np.array([gps_heading])
            H_h = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
            R_h = np.array([[r_h_dynamic]])
            y_h = np.array([wrap_angle(z_h[0] - self.x[4])])
            S_h = H_h @ self.P @ H_h.T + R_h
            K_h = self.P @ H_h.T @ np.linalg.inv(S_h)

            # Heading only updates heading and bias, not position/velocity directly
            K_h[0, :] = 0.0
            K_h[1, :] = 0.0
            K_h[2, :] = 0.0
            K_h[3, :] = 0.0

            self.x = self.x + K_h @ y_h
            self.x[4] = wrap_angle(self.x[4])
            IKH_h = I - K_h @ H_h
            self.P = IKH_h @ self.P @ IKH_h.T + K_h @ R_h @ K_h.T

    def update_zupt(self):
        """Zero-Velocity Update (ZUPT) constraining both forward and lateral velocity."""
        I = np.eye(6)
        H = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        ])
        y = np.array([0.0 - self.x[2], 0.0 - self.x[3]])
        S = H @ self.P @ H.T + self.R_zupt
        K = self.P @ H.T @ np.linalg.inv(S)

        # Standstill observation only updates velocity states
        K[0, :] = 0.0
        K[1, :] = 0.0
        K[4, :] = 0.0
        K[5, :] = 0.0

        self.x = self.x + K @ y
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_zupt @ K.T

# Production default for the blackout speed source, set by measurement, not preference.
# scripts/speed_source_ablation.py over 23 real IO-VNBD drives, 90 s outages:
#
#     ml                          62.5% median blackout drift
#     hold_last                   58.5%   <- best practical source
#     train_mean                  60.8%
#     oracle (true CAN speed)     64.1%
#     oracle speed AND heading    39.9%
#
# Two things follow. First, the ML speed source is a net liability: holding the last GNSS
# speed beats it, so hold_last is the default until the model can win the ablation.
# Second, and more important: perfect speed knowledge (oracle) does NOT help - it is no
# better than the model - while adding perfect heading nearly halves the error. The
# bottleneck is yaw, not speed. Effort spent on the speed regressor is effort spent on the
# wrong axis.
DEFAULT_SPEED_SOURCE = "hold_last"


def run_fusion_pipeline(
    df: pd.DataFrame,
    ai_speed: np.ndarray,
    ai_speed_std: Optional[np.ndarray] = None,
    driver_style: str = "normal",
    blackout_start_sec: float = 60.0,
    blackout_end_sec: float = 150.0,
    speed_source: str = DEFAULT_SPEED_SOURCE,
) -> pd.DataFrame:
    """
    Executes the Confidence-Aware 6-State EKF with Multi-Sensor Context Engine and NHC.

    speed_source controls what feeds forward velocity DURING the outage:
      "hold_last"  freeze the last GNSS speed (default - see DEFAULT_SPEED_SOURCE)
      "ml"         use ai_speed as supplied
    Outside the outage both are irrelevant, because GNSS is updating the filter directly.
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
        # The old default of a flat 0.2 m/s was wildly overconfident: the model's measured
        # RMSE during real blackouts is 5.70 m/s, i.e. 28x larger. Telling the filter a
        # source is 28x better than it is makes the filter follow it off the road, which is
        # the mechanism behind fused-worse-than-naive. Absent a calibrated sigma, assume
        # the source is poor rather than perfect.
        ai_speed_std = np.ones(n) * 5.7

    if speed_source not in ("hold_last", "ml"):
        raise ValueError(f"unknown speed_source {speed_source!r}")

    dt_arr = df["dt"].values if "dt" in df.columns else np.diff(t, prepend=t[0])
    if n > 1:
        dt_arr = np.asarray(dt_arr, dtype=float).copy()
        dt_arr[0] = dt_arr[1]

    # Frame alignment + motion gate. gyro_z is no longer trusted as "yaw" - yaw is the
    # gyro projected onto gravity, which is what it always should have been.
    acc_mat = np.column_stack([df["acc_x"].values, df["acc_y"].values, df["acc_z"].values]).astype(float)
    gyro_mat = np.column_stack([df["gyro_x"].values, df["gyro_y"].values, df["gyro_z"].values]).astype(float)
    frame = align_frame(acc_mat, gyro_mat, dt_arr, speed=gps_v)
    yaw_rate = frame["yaw_rate"]

    step_events = df["step_events"].values if "step_events" in df.columns else None
    # The gate needs to know when it is flying blind. During a blackout a false "stopped"
    # freezes position and the error is never observed until GNSS returns, so the gate
    # switches to thresholds that bias toward MOVING (see motion_gate.GateThresholds).
    gnss_available = ~((t >= blackout_start_sec) & (t < blackout_end_sec))
    gate = MotionGate().classify_frame(frame, speed_hint=ai_speed, step_events=step_events,
                                       gnss_available=gnss_available)
    gate_state = gate["state"]

    if speed_source == "hold_last":
        # Freeze the last speed seen before GNSS was lost, and keep it for the outage.
        pre = np.flatnonzero(t < blackout_start_sec)
        held = float(gps_v[pre[-1]]) if len(pre) else float(gps_v[0])
        in_bo = (t >= blackout_start_sec) & (t < blackout_end_sec)
        ai_speed = np.where(in_bo, held, ai_speed)

    context_engine = VehicleContextEngine()

    init_x = gps_x[0]
    init_y = gps_y[0]
    init_v = gps_v[0]
    init_v_lat = 0.0
    init_heading = gps_h[0] if gps_h is not None else 0.0

    ekf = KinematicFusionEKF(
        init_x=init_x,
        init_y=init_y,
        init_v=init_v,
        init_v_lat=init_v_lat,
        init_heading=init_heading,
        driver_style=driver_style
    )

    fused_px = np.zeros(n)
    fused_py = np.zeros(n)
    fused_v = np.zeros(n)
    fused_v_lat = np.zeros(n)
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

        # Causal window only. This used to be [i-win, i+win], which let the filter see
        # future accelerometer samples when deciding whether the vehicle was stopped -
        # future data leaking into a supposedly real-time filter.
        s_idx = max(0, i - win)
        # Variance of HORIZONTAL specific force about gravity, not of the raw body x/y
        # axes. The raw version was the last frame-dependent decision left in the pipeline:
        # var(acc_x)+var(acc_y) changes when the phone is rotated, so the standstill branch
        # below could reach a different verdict for the same drive at a different mounting.
        # tests/test_rotation_invariance.py caught it as a 7.45% spread in end-to-end
        # blackout drift while every alignment channel still matched to 1e-14.
        acc_var = float(np.var(frame["a_horiz_mag"][s_idx:i + 1]))
        gyro_abs = np.abs(yaw_rate[i])

        # Multi-sensor context detection
        speed_for_context = min(float(ai_speed[i]), float(gps_v[i]))
        mode = context_engine.update_context(
            ambient_lux=ambient_lux[i],
            speed_mps=speed_for_context,
            acc_var=acc_var,
            gyro_abs=gyro_abs,
            gate_state=gate_state[i]
        )
        context_modes.append(mode)

        # Freeze on both standstill and phone-handling. The predict() stationary branch
        # holds position and heading, so this is what stops a shaken phone from
        # accumulating forward travel.
        is_stopped = mode in ("STANDSTILL", "PHONE_HANDLED")

        # 1. State prediction with Confidence-Aware dynamic alpha and Q(t)
        ekf.predict(
            dt=dt,
            v_ai=ai_speed[i],
            v_ai_std=ai_speed_std[i],
            gyro_z=yaw_rate[i],
            is_stationary=is_stopped,
            is_tunnel_alert=context_engine.tunnel_alert
        )

        # 2. Continuous Non-Holonomic Constraint (NHC) Pseudo-Measurement Update
        # Actively enforces v_lat ≈ 0 in vehicle body frame at all times (including blackouts)
        ekf.update_nhc()

        # 3. ZUPT when vehicle is at standstill
        if is_stopped:
            ekf.update_zupt()

        # Outage check: strictly open loop in [start, end)
        in_outage = (blackout_start_sec <= curr_t < blackout_end_sec)
        is_blackout[i] = in_outage

        # Capture open loop state BEFORE measurement update
        pre_update_px[i] = ekf.x[0]
        pre_update_py[i] = ekf.x[1]

        # 4. Measurement update during healthy GNSS window
        if not in_outage:
            hdg_meas = gps_h[i] if gps_h is not None else None
            ekf.update_gps(gps_x=gps_x[i], gps_y=gps_y[i], gps_speed=gps_v[i], gps_heading=hdg_meas)

        fused_px[i] = ekf.x[0]
        fused_py[i] = ekf.x[1]
        fused_v[i] = ekf.x[2]
        fused_v_lat[i] = ekf.x[3]
        fused_theta[i] = ekf.x[4]
        fused_bg[i] = ekf.x[5]

    res = pd.DataFrame({
        "timestamp": t,
        "fused_pos_x": fused_px,
        "fused_pos_y": fused_py,
        "fused_velocity": fused_v,
        "fused_lat_velocity": fused_v_lat,
        "fused_heading": fused_theta,
        "fused_gyro_bias": fused_bg,
        "is_gnss_blackout": is_blackout,
        "context_mode": context_modes,
        "open_loop_pos_x": pre_update_px,
        "open_loop_pos_y": pre_update_py,
        "gate_state": gate_state,
    })

    # Phase 5 metric: time spent frozen while the vehicle was actually moving, during a
    # blackout. This is the failure the asymmetric thresholds exist to prevent, and it is
    # invisible in position error alone until it has already cost metres.
    if "speed" in df.columns:
        truly_moving = df["speed"].values > 1.0
        frozen = np.array([m in ("STANDSTILL", "PHONE_HANDLED") for m in context_modes])
        false_stop = frozen & truly_moving & is_blackout
        res["false_stationary"] = false_stop
        res.attrs["false_stationary_blackout_sec"] = float(np.sum(false_stop * dt_arr))
        # Distance the vehicle covered while the filter believed it was parked - the
        # along-track error this directly creates.
        res.attrs["false_stationary_blackout_m"] = float(
            np.sum(df["speed"].values[false_stop] * dt_arr[false_stop]))

    if "pos_x" in df.columns and "pos_y" in df.columns:
        dx = fused_px - df["pos_x"].values
        dy = fused_py - df["pos_y"].values
        res["fused_pos_error_m"] = np.sqrt(dx**2 + dy**2)

        dx_ol = pre_update_px - df["pos_x"].values
        dy_ol = pre_update_py - df["pos_y"].values
        res["open_loop_error_m"] = np.sqrt(dx_ol**2 + dy_ol**2)

    return res
