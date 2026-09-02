"""
adapters/generic/dataframe_adapter.py
Generic In-Memory / CSV DataFrame Adapter.
Transforms IO-VNBD and standard tabular sensor datasets into canonical SensorFrame streams.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Dict, Any, Optional

from adapters.base import BaseSensorAdapter
from core.interfaces.canonical import SensorFrame, CoordinateFrame
from core.preprocessing.quality import SensorQualityMonitor


class DataFrameSensorAdapter(BaseSensorAdapter):
    """
    Adapter converting in-memory Pandas DataFrame or CSV files into canonical SensorFrames.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        source_name: str = "IO_VNBD_GENERIC",
        sample_rate_hz: float = 10.0
    ):
        self.df = df.copy()
        self.source_name = source_name
        self.sample_rate_hz = sample_rate_hz
        self.quality_monitor = SensorQualityMonitor(nominal_sample_rate_hz=sample_rate_hz)

    def stream_sensor_frames(self) -> Iterator[SensorFrame]:
        t = self.df["timestamp"].values.astype(float)
        ax = self.df["acc_x"].values.astype(float)
        ay = self.df["acc_y"].values.astype(float)
        az = self.df["acc_z"].values.astype(float)
        gx = self.df["gyro_x"].values.astype(float)
        gy = self.df["gyro_y"].values.astype(float)
        gz = self.df["gyro_z"].values.astype(float)
        lux = self.df["ambient_lux"].values.astype(float) if "ambient_lux" in self.df.columns else None

        recent_frames = []
        for i in range(len(self.df)):
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
            "adapter_type": "DataFrameSensorAdapter",
            "source_name": self.source_name,
            "sample_rate_hz": self.sample_rate_hz,
            "total_samples": len(self.df)
        }
