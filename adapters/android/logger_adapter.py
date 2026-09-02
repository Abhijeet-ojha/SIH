"""
adapters/android/logger_adapter.py
Android Logger CSV File and Sensor Stream Adapter.
Parses multi-sensor Android CSV logs (accel, gyro, light, pressure, GNSS) and yields canonical SensorFrames.
"""

import os
import numpy as np
import pandas as pd
from typing import Iterator, Dict, Any, Optional

from adapters.base import BaseSensorAdapter
from core.interfaces.canonical import SensorFrame, CoordinateFrame
from core.preprocessing.quality import SensorQualityMonitor


class AndroidLoggerAdapter(BaseSensorAdapter):
    """
    Adapter converting Android logger raw output files into canonical SensorFrames.
    """
    def __init__(self, csv_filepath: str, sample_rate_hz: float = 10.0):
        self.filepath = csv_filepath
        self.sample_rate_hz = sample_rate_hz
        self.quality_monitor = SensorQualityMonitor(nominal_sample_rate_hz=sample_rate_hz)
        
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"Android log file not found: {csv_filepath}")
            
        self.df = pd.read_csv(csv_filepath)

    def stream_sensor_frames(self) -> Iterator[SensorFrame]:
        # Harmonize column names
        col_map = {
            "Time": "timestamp", "Timestamp": "timestamp",
            "ACC_X": "acc_x", "ACC_Y": "acc_y", "ACC_Z": "acc_z",
            "GYRO_X": "gyro_x", "GYRO_Y": "gyro_y", "GYRO_Z": "gyro_z",
            "LUX": "ambient_lux", "LIGHT": "ambient_lux"
        }
        df = self.df.rename(columns=col_map)
        
        t = df["timestamp"].values.astype(float)
        # If timestamp in milliseconds or nanoseconds, normalize to seconds
        if t[0] > 1e11:  # ns
            t = t / 1e9
        elif t[0] > 1e8:  # ms
            t = t / 1e3

        ax = df["acc_x"].values.astype(float) if "acc_x" in df.columns else np.zeros(len(df))
        ay = df["acc_y"].values.astype(float) if "acc_y" in df.columns else np.zeros(len(df))
        az = df["acc_z"].values.astype(float) if "acc_z" in df.columns else np.zeros(len(df))
        gx = df["gyro_x"].values.astype(float) if "gyro_x" in df.columns else np.zeros(len(df))
        gy = df["gyro_y"].values.astype(float) if "gyro_y" in df.columns else np.zeros(len(df))
        gz = df["gyro_z"].values.astype(float) if "gyro_z" in df.columns else np.zeros(len(df))
        lux = df["ambient_lux"].values.astype(float) if "ambient_lux" in df.columns else None

        recent_frames = []
        for i in range(len(df)):
            frame = SensorFrame(
                timestamp_s=t[i],
                timestamp_ns=int(t[i] * 1e9),
                accel_xyz=np.array([ax[i], ay[i], az[i]], dtype=float),
                gyro_xyz=np.array([gx[i], gy[i], gz[i]], dtype=float),
                ambient_lux=float(lux[i]) if lux is not None else None,
                frame_type=CoordinateFrame.DEVICE_FRAME
            )
            frame.quality = self.quality_monitor.evaluate_frame(frame, recent_frames=recent_frames[-10:])
            recent_frames.append(frame)
            yield frame

    def get_adapter_metadata(self) -> Dict[str, Any]:
        return {
            "adapter_type": "AndroidLoggerAdapter",
            "filepath": self.filepath,
            "sample_rate_hz": self.sample_rate_hz,
            "total_samples": len(self.df)
        }
