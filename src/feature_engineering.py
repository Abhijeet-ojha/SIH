"""
src/feature_engineering.py
Sliding-window feature extraction module for IMU signals.
Extracts time-domain statistical features, kinematic magnitudes,
and frequency-domain energy bands to predict vehicle forward velocity.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def extract_window_features(
    df: pd.DataFrame,
    window_sec: float = 1.5,
    step_sec: float = 0.2,
    sample_rate: float = 10.0
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extracts multi-domain sliding window features from IMU signals.
    """
    window_size = max(4, int(round(window_sec * sample_rate)))
    step_size = max(1, int(round(step_sec * sample_rate)))

    t = df["timestamp"].values
    ax = df["acc_x"].values
    ay = df["acc_y"].values
    az = df["acc_z"].values
    gx = df["gyro_x"].values
    gy = df["gyro_y"].values
    gz = df["gyro_z"].values

    # Computed kinematic magnitudes
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    acc_horiz = np.sqrt(ax**2 + ay**2)

    has_speed = "speed" in df.columns
    speed_arr = df["speed"].values if has_speed else np.zeros(len(df))

    signals = {
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "acc_mag": acc_mag,
        "gyro_mag": gyro_mag,
        "acc_horiz": acc_horiz
    }

    feature_rows = []
    y_targets = []
    t_mids = []

    n = len(df)
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        mid = (start + end) // 2
        t_mids.append(t[mid])
        if has_speed:
            y_targets.append(speed_arr[mid])

        row = {}
        for sig_name, sig_data in signals.items():
            win = sig_data[start:end]
            mean_val = np.mean(win)
            std_val = np.std(win)
            min_val = np.min(win)
            max_val = np.max(win)
            p2p_val = max_val - min_val
            rms_val = np.sqrt(np.mean(win**2))

            row[f"{sig_name}_mean"] = mean_val
            row[f"{sig_name}_std"] = std_val
            row[f"{sig_name}_min"] = min_val
            row[f"{sig_name}_max"] = max_val
            row[f"{sig_name}_p2p"] = p2p_val
            row[f"{sig_name}_rms"] = rms_val

            # Frequency / Spectral Energy
            fft_vals = np.abs(np.fft.rfft(win - mean_val))
            row[f"{sig_name}_fft_energy"] = float(np.sum(fft_vals**2) / (len(fft_vals) + 1e-5))

        # Cross-signal features (Centripetal / Kinematic turn cue)
        mean_ax = row["ax_mean"]
        mean_gz = row["gz_mean"]
        row["turn_curv_cue"] = float(abs(mean_ax) / (abs(mean_gz) + 0.05))
        row["vibration_power"] = row["acc_mag_std"] * row["acc_horiz_rms"]

        feature_rows.append(row)

    X_df = pd.DataFrame(feature_rows)
    y_speed = np.array(y_targets) if has_speed else np.array([])
    t_mid = np.array(t_mids)

    return X_df, y_speed, t_mid
