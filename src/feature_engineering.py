"""
src/feature_engineering.py
Causal sliding-window feature extraction module for IMU signals.
Strictly causal: Window [t - W, t] -> Prediction at t (no future data leakage).
Extracts statistical moments, kinematic magnitudes, jerk, angular acceleration,
orientation-invariant dynamics, and data-driven spectral features.
"""

import time
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from scipy import stats

try:
    from .frame_alignment import align_frame
except ImportError:  # direct script execution
    from frame_alignment import align_frame

def extract_causal_window_features(
    df: pd.DataFrame,
    window_sec: float = 1.5,
    step_sec: float = 0.2,
    sample_rate: float = 10.0,
    feature_group: str = "all"  # 'all', 'base_stats', 'no_spectral', 'minimal'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Extracts strictly CAUSAL sliding window features from IMU signals.
    Prediction timestamp is strictly at the trailing edge of the window (t_current).
    Zero future information is accessed.
    
    Returns:
      (X_df, y_speed, t_current, latency_dict)
    """
    t_start_bench = time.perf_counter()

    # Sample rate is a property of the drive, not a constant. The loader now preserves
    # real timestamps, so read it from the data instead of asserting 10 Hz.
    if "dt" in df.columns:
        med_dt = float(np.median(df["dt"].values[df["dt"].values > 0]))
        if med_dt > 0:
            sample_rate = 1.0 / med_dt

    window_size = max(4, int(round(window_sec * sample_rate)))
    step_size = max(1, int(round(step_sec * sample_rate)))

    t = df["timestamp"].values
    n = len(df)
    dt_scalar = 1.0 / sample_rate
    dt_arr = df["dt"].values if "dt" in df.columns else np.full(n, dt_scalar)

    # ── Frame-invariant channels ──────────────────────────────────────────────
    # Raw ax/ay/az/gx/gy/gz are deliberately NOT features any more. They encode how the
    # phone happens to be sitting, so a model trained on them learns one mounting and
    # fails on the next - which is exactly what the LODRO negative R2 was reporting.
    # Everything below is unchanged by an arbitrary fixed rotation of the phone; see
    # tests/test_frame_invariance.py.
    acc = np.column_stack([df["acc_x"].values, df["acc_y"].values, df["acc_z"].values]).astype(float)
    gyro = np.column_stack([df["gyro_x"].values, df["gyro_y"].values, df["gyro_z"].values]).astype(float)
    speed_for_axis = df["speed"].values if "speed" in df.columns else None
    fr = align_frame(acc, gyro, dt_arr, speed=speed_for_axis)

    a_fwd = fr["a_fwd"]
    a_lat = fr["a_lat"]
    a_vert = fr["a_vert"]
    a_horiz_mag = fr["a_horiz_mag"]
    yaw_rate = fr["yaw_rate"]
    gyro_mag = fr["gyro_mag"]
    tilt_rate = fr["tilt_rate"]
    grav_stab = fr["grav_stability"]

    # Temporal Jerk (da/dt) and Angular Acceleration (domega/dt), on invariant channels.
    jerk_fwd = np.gradient(a_fwd, dt_scalar)
    jerk_vert = np.gradient(a_vert, dt_scalar)
    alpha_yaw = np.gradient(yaw_rate, dt_scalar)

    has_speed = "speed" in df.columns
    speed_arr = df["speed"].values if has_speed else np.zeros(n)

    # Dictionary of all processed signals (NO target or spatial labels)
    signals = {
        "a_fwd": a_fwd, "a_lat": a_lat, "a_vert": a_vert, "a_horiz_mag": a_horiz_mag,
        "yaw_rate": yaw_rate, "gyro_mag": gyro_mag, "tilt_rate": tilt_rate,
        "grav_stab": grav_stab,
        "jerk_fwd": jerk_fwd, "jerk_vert": jerk_vert, "alpha_yaw": alpha_yaw,
    }

    feature_rows = []
    y_targets = []
    t_ends = []

    # Causal sliding loop: start to end, prediction at (end - 1)
    for end in range(window_size, n + 1, step_size):
        start = end - window_size
        current_idx = end - 1
        current_t = t[current_idx]

        # STRICT CAUSAL LEAKAGE CHECK
        assert current_t <= t[end - 1] + 1e-7, "Causal violation: future timestamp accessed!"
        
        t_ends.append(current_t)
        if has_speed:
            y_targets.append(speed_arr[current_idx])

        row = {}
        for sig_name, sig_data in signals.items():
            win = sig_data[start:end]
            mean_val = float(np.mean(win))
            std_val = float(np.std(win))
            min_val = float(np.min(win))
            max_val = float(np.max(win))
            p2p_val = max_val - min_val
            rms_val = float(np.sqrt(np.mean(win**2)))

            # Base Statistical Moments
            row[f"{sig_name}_mean"] = mean_val
            row[f"{sig_name}_std"] = std_val
            row[f"{sig_name}_min"] = min_val
            row[f"{sig_name}_max"] = max_val
            row[f"{sig_name}_p2p"] = p2p_val
            row[f"{sig_name}_rms"] = rms_val

            if feature_group in ["all", "enhanced"]:
                # Advanced Statistical Moments
                med_val = float(np.median(win))
                q25, q75 = np.percentile(win, [25, 75])
                q10, q90 = np.percentile(win, [10, 90])
                iqr_val = float(q75 - q25)
                mad_val = float(np.median(np.abs(win - med_val)))
                skew_val = float(stats.skew(win)) if std_val > 1e-6 else 0.0
                kurt_val = float(stats.kurtosis(win)) if std_val > 1e-6 else 0.0
                
                # Zero crossing rate relative to window mean
                zero_cross = float(np.mean(np.diff(np.sign(win - mean_val) != 0)))

                row[f"{sig_name}_median"] = med_val
                row[f"{sig_name}_iqr"] = iqr_val
                row[f"{sig_name}_mad"] = mad_val
                row[f"{sig_name}_skew"] = skew_val
                row[f"{sig_name}_kurt"] = kurt_val
                row[f"{sig_name}_p10"] = float(q10)
                row[f"{sig_name}_p90"] = float(q90)
                row[f"{sig_name}_zcr"] = zero_cross

            if feature_group in ["all", "spectral"]:
                # Data-Driven Spectral Features (Nyquist = 5 Hz for 10 Hz sampling)
                # Centered FFT
                win_centered = win - mean_val
                fft_mag = np.abs(np.fft.rfft(win_centered))
                freqs = np.fft.rfftfreq(len(win), d=1.0/sample_rate)
                total_power = np.sum(fft_mag**2) + 1e-9

                # Spectral Centroid
                spec_centroid = float(np.sum(freqs * fft_mag**2) / total_power)
                # Dominant Frequency
                dom_freq = float(freqs[np.argmax(fft_mag)])
                
                # Sub-band powers: Low (0 - 1.5 Hz), Mid-High (1.5 - 5.0 Hz)
                low_mask = freqs <= 1.5
                high_mask = (freqs > 1.5) & (freqs <= 5.0)
                p_low = float(np.sum(fft_mag[low_mask]**2) / total_power)
                p_high = float(np.sum(fft_mag[high_mask]**2) / total_power)

                row[f"{sig_name}_spec_centroid"] = spec_centroid
                row[f"{sig_name}_dom_freq"] = dom_freq
                row[f"{sig_name}_power_low"] = p_low
                row[f"{sig_name}_power_high"] = p_high

        # Physical Interaction Features.
        # The old vibration_power = acc_mag_std * acc_horiz_rms was the shortcut the model
        # actually learned: pure vibration energy, which correlates with speed only because
        # the training phone was bolted into one car. It is gone. Road-induced *vertical*
        # vibration genuinely does scale with speed, so that is kept - but it is now safe
        # to use because motion_gate.MotionGate vetoes the whole estimate when the phone is
        # being handled, which is the negative case the training set never contained.
        row["road_vibration"] = row["a_vert_std"] * row["a_horiz_mag_rms"]
        # Turn tightness: lateral acceleration against yaw rate. For a vehicle these obey
        # a_lat ~= v * omega, so their ratio is an orientation-free speed proxy that
        # vibration cannot fake.
        row["turn_speed_proxy"] = float(abs(row["a_lat_mean"]) / (abs(row["yaw_rate_mean"]) + 0.05))
        # How trustworthy the frame is right now. Near zero for a cradled phone.
        row["frame_instability"] = row["grav_stab_mean"] * (row["tilt_rate_rms"] + 0.01)

        feature_rows.append(row)

    X_df = pd.DataFrame(feature_rows)
    y_speed = np.array(y_targets) if has_speed else np.array([])
    t_end = np.array(t_ends)

    # Sanity Gate: Verify no target column in X_df
    for banned in ["speed", "pos_x", "pos_y", "heading", "lat", "lon"]:
        assert banned not in X_df.columns, f"LEAKAGE DETECTED: Column '{banned}' found in features!"

    total_calc_time_ms = (time.perf_counter() - t_start_bench) * 1000.0
    ms_per_window = total_calc_time_ms / max(1, len(X_df))

    latency_info = {
        "context_window_sec": window_sec,
        "step_sec": step_sec,
        "num_windows": len(X_df),
        "total_calc_time_ms": total_calc_time_ms,
        "feature_calc_ms_per_window": ms_per_window
    }

    return X_df, y_speed, t_end, latency_info

# Backward-compatible alias for existing scripts
def extract_window_features(
    df: pd.DataFrame,
    window_sec: float = 1.5,
    step_sec: float = 0.2,
    sample_rate: float = 10.0
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Causal window feature extraction wrapper."""
    X_df, y_speed, t_end, _ = extract_causal_window_features(
        df, window_sec=window_sec, step_sec=step_sec, sample_rate=sample_rate, feature_group="all"
    )
    return X_df, y_speed, t_end

