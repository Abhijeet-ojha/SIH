"""
core/preprocessing/resampler.py
Sampling-Rate Agnostic, Timestamp-Aware Synchronization and Resampling Engine.
Resamples heterogeneous sensor streams to canonical sampling grids (10Hz, 20Hz, 50Hz, 100Hz)
strictly using causal past/present data points without future look-ahead.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from core.interfaces.canonical import SensorFrame, CoordinateFrame, SensorQualityReport
from core.preprocessing.quality import SensorQualityMonitor


class CausalSensorResampler:
    """
    Causal Sensor Synchronizer & Resampler.
    Maps irregular raw timestamps [t_0, t_1, ...] to a uniform target frequency grid.
    Strictly causal: Interpolation at grid point t_grid uses only data points where t <= t_grid.
    """
    def __init__(
        self,
        target_rate_hz: float = 10.0,
        max_interpolation_gap_s: float = 0.50
    ):
        self.target_rate_hz = target_rate_hz
        self.target_dt = 1.0 / target_rate_hz
        self.max_gap_s = max_interpolation_gap_s
        self.quality_monitor = SensorQualityMonitor(nominal_sample_rate_hz=target_rate_hz)

    def resample_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        accel_cols: Tuple[str, str, str] = ("acc_x", "acc_y", "acc_z"),
        gyro_cols: Tuple[str, str, str] = ("gyro_x", "gyro_y", "gyro_z"),
        speed_col: Optional[str] = "speed",
        lux_col: Optional[str] = "ambient_lux"
    ) -> Tuple[List[SensorFrame], Optional[np.ndarray], np.ndarray]:
        """
        Resamples a DataFrame to the uniform target rate grid.
        Returns:
            (canonical_sensor_frames, target_speeds_if_available, resampled_timestamps)
        """
        raw_t = df[timestamp_col].values.astype(float)
        ax = df[accel_cols[0]].values.astype(float)
        ay = df[accel_cols[1]].values.astype(float)
        az = df[accel_cols[2]].values.astype(float)
        gx = df[gyro_cols[0]].values.astype(float)
        gy = df[gyro_cols[1]].values.astype(float)
        gz = df[gyro_cols[2]].values.astype(float)
        
        has_speed = speed_col is not None and speed_col in df.columns
        raw_spd = df[speed_col].values.astype(float) if has_speed else None
        
        has_lux = lux_col is not None and lux_col in df.columns
        raw_lux = df[lux_col].values.astype(float) if has_lux else None

        # Build regular target timestamp grid starting from raw_t[0] to raw_t[-1]
        t_start = raw_t[0]
        t_end = raw_t[-1]
        grid_t = np.arange(t_start, t_end, self.target_dt)

        # Causal interpolation: For each grid point t_k, find index in raw_t where raw_t[i] <= t_k
        frames: List[SensorFrame] = []
        resampled_speeds: List[float] = []
        
        # Vectorized searchsorted for causal trailing indices
        indices = np.searchsorted(raw_t, grid_t, side="right") - 1
        indices = np.clip(indices, 0, len(raw_t) - 2)

        for k, idx in enumerate(indices):
            tk = grid_t[k]
            t0, t1 = raw_t[idx], raw_t[idx + 1]
            
            # Gap detection: if gap between raw samples > max_gap, clamp interpolation
            if (t1 - t0) > self.max_gap_s:
                ratio = 0.0
            else:
                ratio = (tk - t0) / max(1e-6, (t1 - t0))
                ratio = np.clip(ratio, 0.0, 1.0)

            interp_acc = np.array([
                ax[idx] + ratio * (ax[idx + 1] - ax[idx]),
                ay[idx] + ratio * (ay[idx + 1] - ay[idx]),
                az[idx] + ratio * (az[idx + 1] - az[idx])
            ], dtype=float)

            interp_gyro = np.array([
                gx[idx] + ratio * (gx[idx + 1] - gx[idx]),
                gy[idx] + ratio * (gy[idx + 1] - gy[idx]),
                gz[idx] + ratio * (gz[idx + 1] - gz[idx])
            ], dtype=float)

            interp_lux = float(raw_lux[idx] + ratio * (raw_lux[idx + 1] - raw_lux[idx])) if raw_lux is not None else 1000.0

            frame = SensorFrame(
                timestamp_s=float(tk),
                timestamp_ns=int(tk * 1e9),
                accel_xyz=interp_acc,
                gyro_xyz=interp_gyro,
                ambient_lux=interp_lux,
                frame_type=CoordinateFrame.DEVICE_FRAME
            )
            # Evaluate frame quality
            frame.quality = self.quality_monitor.evaluate_frame(frame, recent_frames=frames[-10:])
            frames.append(frame)

            if has_speed and raw_spd is not None:
                interp_spd = float(raw_spd[idx] + ratio * (raw_spd[idx + 1] - raw_spd[idx]))
                resampled_speeds.append(max(0.0, interp_spd))

        speeds_arr = np.array(resampled_speeds) if has_speed else None
        return frames, speeds_arr, grid_t
