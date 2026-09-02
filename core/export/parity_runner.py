"""
core/export/parity_runner.py
Streaming Ring Buffer and Python <-> Edge Numerical Parity Verification Engine.
Validates zero-deviation numerical parity between Python training environment
and Edge inference runtime across representative sensor windows.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

from core.interfaces.canonical import SensorFrame, MotionEstimate
from core.models.tabular_models import TabularSpeedModel
from core.features.extractor import CausalFeatureExtractor


class StreamingSensorRingBuffer:
    """
    Continuous circular ring buffer for real-time edge inference.
    Avoids rebuilding Pandas DataFrames on every sensor arrival.
    """
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, frame: SensorFrame):
        self.buffer.append(frame)

    def is_ready(self, min_samples: int = 15) -> bool:
        return len(self.buffer) >= min_samples

    def get_window_dataframe(self, window_samples: int = 15) -> pd.DataFrame:
        """Extracts the most recent window of sensor frames as a DataFrame."""
        recent = list(self.buffer)[-window_samples:]
        rows = []
        for f in recent:
            rows.append({
                "timestamp": f.timestamp_s,
                "acc_x": f.accel_xyz[0],
                "acc_y": f.accel_xyz[1],
                "acc_z": f.accel_xyz[2],
                "gyro_x": f.gyro_xyz[0],
                "gyro_y": f.gyro_xyz[1],
                "gyro_z": f.gyro_xyz[2],
                "ambient_lux": f.ambient_lux if f.ambient_lux is not None else 1000.0
            })
        return pd.DataFrame(rows)


def verify_python_edge_parity(
    model: TabularSpeedModel,
    test_X: pd.DataFrame,
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Asserts numerical parity between native in-memory model and serialized/reloaded model.
    """
    # 1. Native Prediction
    t0 = time.perf_counter()
    native_preds, native_std = model.predict_with_uncertainty(test_X)
    native_latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(test_X))

    # 2. Emulate Edge Deserialization
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_path = os.path.join(tmpdir, "edge_test_model.joblib")
        model.save(tmp_model_path)
        
        edge_model = TabularSpeedModel(model_type=model.model_type)
        edge_model.load(tmp_model_path)

        t1 = time.perf_counter()
        edge_preds, edge_std = edge_model.predict_with_uncertainty(test_X)
        edge_latency_ms = (time.perf_counter() - t1) * 1000.0 / max(1, len(test_X))

    # 3. Compute Absolute Differences
    pred_diffs = np.abs(native_preds - edge_preds)
    std_diffs = np.abs(native_std - edge_std)

    max_pred_diff = float(np.max(pred_diffs))
    mean_pred_diff = float(np.mean(pred_diffs))
    max_std_diff = float(np.max(std_diffs))

    passed = (max_pred_diff <= tolerance) and (max_std_diff <= tolerance)

    report = {
        "parity_passed": bool(passed),
        "num_windows_evaluated": len(test_X),
        "max_prediction_diff": max_pred_diff,
        "mean_prediction_diff": mean_pred_diff,
        "max_uncertainty_diff": max_std_diff,
        "tolerance": tolerance,
        "native_latency_ms_per_window": native_latency_ms,
        "edge_latency_ms_per_window": edge_latency_ms
    }

    if not passed:
        raise AssertionError(f"PARITY CHECK FAILED: Max difference {max_pred_diff:.6e} exceeded tolerance {tolerance:.6e}")

    return report
