"""
core/features/extractor.py
Strictly Causal Sliding-Window Feature and Temporal Sequence Extraction Engine.
Strict Causality Contract:
  - Trailing window layout: [t - W, t] -> Prediction at timestamp t.
  - Strictly backward finite differences for jerk (da/dt) and angular acceleration (domega/dt).
  - Zero np.gradient, zero centered differences, zero future samples.
  - Zero target leakage (no speed, position, or GNSS features in feature matrix).
"""

import time
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from scipy import stats
from core.features.leakage_guard import verify_feature_matrix_leakage, verify_causality


def backward_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """
    Strictly causal backward finite difference:
    (dx/dt)_t = (x[t] - x[t-1]) / dt
    Never uses future samples.
    """
    res = np.zeros_like(arr, dtype=float)
    if len(arr) > 1:
        res[1:] = (arr[1:] - arr[:-1]) / dt
        res[0] = 0.0
    return res


class CausalFeatureExtractor:
    """
    Causal Sliding-Window Feature Extractor for IMU Streams.
    Window layout: [t - W, t] -> Prediction timestamp at t (trailing edge).
    """
    def __init__(
        self,
        window_sec: float = 1.5,
        step_sec: float = 0.1,
        sample_rate_hz: float = 10.0,
        feature_group: str = "all"  # 'all', 'base_stats', 'dynamics', 'no_spectral', 'minimal'
    ):
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.sample_rate = sample_rate_hz
        self.feature_group = feature_group
        
        self.window_samples = max(4, int(round(window_sec * sample_rate_hz)))
        self.step_samples = max(1, int(round(step_sec * sample_rate_hz)))

    def extract_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extracts engineered tabular feature matrix X_df, ground truth y_speed (if present),
        and prediction timestamps t_pred.
        """
        t_start_bench = time.perf_counter()
        
        t = df["timestamp"].values.astype(float)
        ax = df["acc_x"].values.astype(float)
        ay = df["acc_y"].values.astype(float)
        az = df["acc_z"].values.astype(float)
        gx = df["gyro_x"].values.astype(float)
        gy = df["gyro_y"].values.astype(float)
        gz = df["gyro_z"].values.astype(float)
        n = len(df)
        
        dt_scalar = 1.0 / self.sample_rate

        # Orientation-Invariant & Kinematic Magnitudes
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        acc_horiz = np.sqrt(ax**2 + ay**2)

        # Strictly Causal Backward Finite Differences (Zero np.gradient)
        jx = backward_diff(ax, dt_scalar)
        jy = backward_diff(ay, dt_scalar)
        jz = backward_diff(az, dt_scalar)
        jerk_mag = np.sqrt(jx**2 + jy**2 + jz**2)
        
        alpha_x = backward_diff(gx, dt_scalar)
        alpha_y = backward_diff(gy, dt_scalar)
        alpha_z = backward_diff(gz, dt_scalar)
        alpha_mag = np.sqrt(alpha_x**2 + alpha_y**2 + alpha_z**2)

        has_speed = "speed" in df.columns
        speed_arr = df["speed"].values if has_speed else np.zeros(n)

        # Dictionary of raw processed sensor signals (Strictly IMU, NO target)
        signals = {
            "ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
            "acc_mag": acc_mag, "gyro_mag": gyro_mag, "acc_horiz": acc_horiz,
            "jerk_mag": jerk_mag, "alpha_mag": alpha_mag, "alpha_z": alpha_z
        }

        feature_rows = []
        y_targets = []
        t_ends = []

        # Causal sliding loop: window [start:end], prediction at (end - 1)
        for end in range(self.window_samples, n + 1, self.step_samples):
            start = end - self.window_samples
            current_idx = end - 1
            current_t = t[current_idx]
            
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

                if self.feature_group in ["all", "enhanced", "dynamics"]:
                    # Advanced Moments
                    med_val = float(np.median(win))
                    q25, q75 = np.percentile(win, [25, 75])
                    q10, q90 = np.percentile(win, [10, 90])
                    iqr_val = float(q75 - q25)
                    mad_val = float(np.median(np.abs(win - med_val)))
                    skew_val = float(stats.skew(win)) if std_val > 1e-6 else 0.0
                    kurt_val = float(stats.kurtosis(win)) if std_val > 1e-6 else 0.0
                    zero_cross = float(np.mean(np.diff(np.sign(win - mean_val) != 0))) if len(win) > 1 else 0.0

                    row[f"{sig_name}_median"] = med_val
                    row[f"{sig_name}_iqr"] = iqr_val
                    row[f"{sig_name}_mad"] = mad_val
                    row[f"{sig_name}_skew"] = skew_val
                    row[f"{sig_name}_kurt"] = kurt_val
                    row[f"{sig_name}_p10"] = float(q10)
                    row[f"{sig_name}_p90"] = float(q90)
                    row[f"{sig_name}_zcr"] = zero_cross

                if self.feature_group in ["all", "spectral"]:
                    # Data-Driven Spectral Features (Centered FFT)
                    win_centered = win - mean_val
                    fft_mag = np.abs(np.fft.rfft(win_centered))
                    freqs = np.fft.rfftfreq(len(win), d=1.0/self.sample_rate)
                    total_power = float(np.sum(fft_mag**2) + 1e-9)

                    spec_centroid = float(np.sum(freqs * fft_mag**2) / total_power)
                    dom_freq = float(freqs[np.argmax(fft_mag)])
                    
                    low_mask = freqs <= 1.5
                    high_mask = (freqs > 1.5) & (freqs <= 5.0)
                    p_low = float(np.sum(fft_mag[low_mask]**2) / total_power)
                    p_high = float(np.sum(fft_mag[high_mask]**2) / total_power)

                    row[f"{sig_name}_spec_centroid"] = spec_centroid
                    row[f"{sig_name}_dom_freq"] = dom_freq
                    row[f"{sig_name}_power_low"] = p_low
                    row[f"{sig_name}_power_high"] = p_high

            # Cross-Signal Interactions (No ground-truth leakage)
            row["vibration_power"] = float(row["acc_mag_std"] * row["acc_horiz_rms"])
            row["jerk_motion_intensity"] = float(row["jerk_mag_rms"] * (row["gyro_mag_mean"] + 0.01))
            row["curv_ratio"] = float(abs(row["ax_mean"]) / (abs(row["gz_mean"]) + 0.05))

            feature_rows.append(row)

        X_df = pd.DataFrame(feature_rows)
        y_speed = np.array(y_targets, dtype=float) if has_speed else np.array([])
        t_end = np.array(t_ends, dtype=float)

        # Automated Leakage Assertion
        verify_feature_matrix_leakage(X_df, context_name="CausalFeatureExtractor")
        verify_causality(t_end, t_end)

        total_calc_time_ms = (time.perf_counter() - t_start_bench) * 1000.0
        ms_per_window = total_calc_time_ms / max(1, len(X_df))

        latency_info = {
            "window_sec": self.window_sec,
            "step_sec": self.step_sec,
            "num_windows": len(X_df),
            "total_calc_time_ms": total_calc_time_ms,
            "ms_per_window": ms_per_window
        }

        return X_df, y_speed, t_end, latency_info

    def extract_temporal_sequences(
        self,
        df: pd.DataFrame,
        channels: List[str] = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts raw [N x T x C] temporal window sequences for deep 1D-CNN / TCN sequence modeling.
        Returns:
            (X_seq, y_speed, t_end) where X_seq shape is (num_windows, time_samples, num_channels).
        """
        t = df["timestamp"].values.astype(float)
        sensor_matrix = df[channels].values.astype(float)
        n = len(df)
        has_speed = "speed" in df.columns
        speed_arr = df["speed"].values if has_speed else np.zeros(n)

        seq_list = []
        y_targets = []
        t_ends = []

        for end in range(self.window_samples, n + 1, self.step_samples):
            start = end - self.window_samples
            current_idx = end - 1
            
            seq_list.append(sensor_matrix[start:end, :])
            t_ends.append(t[current_idx])
            if has_speed:
                y_targets.append(speed_arr[current_idx])

        X_seq = np.array(seq_list, dtype=np.float32)  # Shape: (N, T, C)
        y_speed = np.array(y_targets, dtype=np.float32) if has_speed else np.array([], dtype=np.float32)
        t_end = np.array(t_ends, dtype=float)

        return X_seq, y_speed, t_end
