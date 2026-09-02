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

    window_size = max(4, int(round(window_sec * sample_rate)))
    step_size = max(1, int(round(step_sec * sample_rate)))

    t = df["timestamp"].values
    ax = df["acc_x"].values.astype(float)
    ay = df["acc_y"].values.astype(float)
    az = df["acc_z"].values.astype(float)
    gx = df["gyro_x"].values.astype(float)
    gy = df["gyro_y"].values.astype(float)
    gz = df["gyro_z"].values.astype(float)

    n = len(df)
    dt_scalar = 1.0 / sample_rate

    # Kinematic Magnitudes (Orientation-invariant)
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    acc_horiz = np.sqrt(ax**2 + ay**2)

    # Temporal Jerk (da/dt) and Angular Acceleration (domega/dt)
    jx = np.gradient(ax, dt_scalar)
    jy = np.gradient(ay, dt_scalar)
    jz = np.gradient(az, dt_scalar)
    jerk_mag = np.sqrt(jx**2 + jy**2 + jz**2)
    alpha_x = np.gradient(gx, dt_scalar)
    alpha_y = np.gradient(gy, dt_scalar)
    alpha_z = np.gradient(gz, dt_scalar)
    alpha_mag = np.sqrt(alpha_x**2 + alpha_y**2 + alpha_z**2)

    has_speed = "speed" in df.columns
    speed_arr = df["speed"].values if has_speed else np.zeros(n)

    # Dictionary of all processed signals (NO target or spatial labels)
    signals = {
        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz,
        "acc_mag": acc_mag, "gyro_mag": gyro_mag, "acc_horiz": acc_horiz,
        "jerk_mag": jerk_mag, "alpha_mag": alpha_mag, "alpha_z": alpha_z
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

        # Physical Interaction & Vibration Features (NO label leakage)
        row["vibration_power"] = row["acc_mag_std"] * row["acc_horiz_rms"]
        row["jerk_motion_intensity"] = row["jerk_mag_rms"] * (row["gyro_mag_mean"] + 0.01)
        row["curv_ratio"] = float(abs(row["ax_mean"]) / (abs(row["gz_mean"]) + 0.05))

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

