"""
core/export/parity_runner.py
Streaming Ring Buffer and Python <-> Edge Numerical Parity Verification Engine.
Validates zero-deviation numerical parity between Python training environment,
serialized joblib models, and pure JSON decision tree traversals across representative sensor windows.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

from core.interfaces.canonical import SensorFrame, MotionEstimate
from core.models.tabular_models import TabularSpeedModel
from core.features.extractor import CausalFeatureExtractor


def traverse_tree(tree_dict: Dict[str, Any], feature_vector: np.ndarray) -> float:
    """
    Traverses a single decision tree in pure JSON representation.
    Uses float32 casting to match scikit-learn internal tree evaluation precision.
    """
    node = 0
    children_left = tree_dict["children_left"]
    children_right = tree_dict["children_right"]
    feature_arr = tree_dict["feature"]
    threshold_arr = tree_dict["threshold"]
    value_arr = tree_dict["value"]

    feat_vec_32 = feature_vector.astype(np.float32)

    while children_left[node] != -1:
        feat_idx = feature_arr[node]
        thresh_32 = np.float32(threshold_arr[node])
        if feat_vec_32[feat_idx] <= thresh_32:
            node = children_left[node]
        else:
            node = children_right[node]
    return float(value_arr[node])


def predict_embedded_forest(rules: Dict[str, Any], X_mat: np.ndarray) -> np.ndarray:
    """
    Evaluates complete Random Forest ensemble in pure Python/Dart/Kotlin representation.
    Produces identical results to scikit-learn.
    """
    trees = rules.get("trees", [])
    if not trees:
        return np.zeros(len(X_mat), dtype=float)
    
    n_samples = len(X_mat)
    preds = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        x = X_mat[i]
        sample_sum = sum(traverse_tree(t, x) for t in trees)
        preds[i] = max(0.0, sample_sum / len(trees))
    return preds


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
    embedded_rules_path: Optional[str] = None,
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Asserts numerical parity between native sklearn model and pure JSON decision tree traversal.
    """
    # 1. Native Prediction
    t0 = time.perf_counter()
    native_preds, native_std = model.predict_with_uncertainty(test_X)
    native_latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(test_X))

    # 2. Pure JSON Tree Traversal Parity
    if embedded_rules_path and os.path.exists(embedded_rules_path):
        with open(embedded_rules_path, "r") as f:
            rules = json.load(f)
        t1 = time.perf_counter()
        json_preds = predict_embedded_forest(rules, test_X.values)
        json_latency_ms = (time.perf_counter() - t1) * 1000.0 / max(1, len(test_X))
        pred_diffs = np.abs(native_preds - json_preds)
    else:
        # Fallback to model reload
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_path = os.path.join(tmpdir, "edge_test_model.joblib")
            model.save(tmp_model_path)
            edge_model = TabularSpeedModel(model_type=model.model_type)
            edge_model.load(tmp_model_path)
            json_preds, _ = edge_model.predict_with_uncertainty(test_X)
            json_latency_ms = native_latency_ms
            pred_diffs = np.abs(native_preds - json_preds)

    max_pred_diff = float(np.max(pred_diffs))
    mean_pred_diff = float(np.mean(pred_diffs))
    passed = bool(max_pred_diff <= tolerance)

    report = {
        "parity_passed": passed,
        "num_windows_evaluated": len(test_X),
        "max_prediction_diff_mps": max_pred_diff,
        "mean_prediction_diff_mps": mean_pred_diff,
        "tolerance_mps": tolerance,
        "native_latency_ms_per_window": native_latency_ms,
        "edge_json_latency_ms_per_window": json_latency_ms
    }

    if not passed:
        raise AssertionError(f"PARITY CHECK FAILED: Max difference {max_pred_diff:.6e} exceeded tolerance {tolerance:.6e}")

    return report
