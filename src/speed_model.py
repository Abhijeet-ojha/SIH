"""
src/speed_model.py
Compatibility Bridge delegating to core/models/ and core/uncertainty/.
Unified Speed Regression & Calibrated Uncertainty Estimation Suite.
"""

import os
import io
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from scipy.interpolate import interp1d

from core.models.tabular_models import TabularSpeedModel
from core.models.temporal_models import CausalTemporalSpeedNet, TemporalSequenceSpeedModel
from core.uncertainty.calibrator import ConformalUncertaintyCalibrator
from core.export.exporter import EdgeModelExporter
from core.export.spec import FeatureConfigSpec


# Conv1DSpeedNet alias for backward compatibility
Conv1DSpeedNet = CausalTemporalSpeedNet


class SpeedRegressorModel:
    """
    Unified multi-model speed estimator wrapper delegating to core/models/TabularSpeedModel.
    """
    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
        uncertainty_method: str = "ensemble"
    ):
        self.model_type = model_type.lower()
        self.uncertainty_method = uncertainty_method.lower()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        self.inner_model = TabularSpeedModel(
            model_type=self.model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            uncertainty_method=self.uncertainty_method
        )
        self.is_trained = False
        self.feature_names = []

    @property
    def conformal_q_hat(self) -> float:
        return self.inner_model.conformal_calibrator.q_hat

    @conformal_q_hat.setter
    def conformal_q_hat(self, val: float):
        self.inner_model.conformal_calibrator.q_hat = float(val)

    @property
    def model(self):
        return self.inner_model.model

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        metrics = self.inner_model.train(X, y, X_calib=X_val, y_calib=y_val)
        self.is_trained = self.inner_model.is_trained
        self.feature_names = self.inner_model.feature_names
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.inner_model.predict(X)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.inner_model.predict_with_uncertainty(X)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        return self.inner_model.evaluate(X, y)

    def get_model_size_kb(self) -> float:
        return self.inner_model.get_model_size_kb()

    def save(self, filepath: str):
        self.inner_model.save(filepath)

    def load(self, filepath: str):
        self.inner_model.load(filepath)
        self.is_trained = self.inner_model.is_trained
        self.feature_names = self.inner_model.feature_names

    def export_embedded_rules(self, filepath: str):
        EdgeModelExporter._export_embedded_rules(self.inner_model, filepath)


def reconstruct_ai_dr_trajectory(
    df: pd.DataFrame,
    t_pred: np.ndarray,
    v_pred: np.ndarray,
    v_std: Optional[np.ndarray] = None,
    initial_heading: float = 0.0,
    initial_pos: Tuple[float, float] = (0.0, 0.0)
) -> pd.DataFrame:
    """Reconstructs pure AI-DR trajectory with interpolated uncertainty."""
    t_orig = df["timestamp"].values
    gyro_z = df["gyro_z"].values
    n = len(df)

    interp_func = interp1d(t_pred, v_pred, kind="linear", bounds_error=False, fill_value=(v_pred[0], v_pred[-1]))
    v_dense = np.maximum(0.0, interp_func(t_orig))

    if v_std is not None:
        interp_std = interp1d(t_pred, v_std, kind="linear", bounds_error=False, fill_value=(v_std[0], v_std[-1]))
        std_dense = np.maximum(0.05, interp_std(t_orig))
    else:
        std_dense = np.ones(n) * 0.2

    dt_arr = np.diff(t_orig, prepend=t_orig[0])
    dt_arr[0] = dt_arr[1] if n > 1 else 0.1

    heading_ai = np.zeros(n)
    pos_x_ai = np.zeros(n)
    pos_y_ai = np.zeros(n)

    heading_ai[0] = initial_heading
    pos_x_ai[0] = initial_pos[0]
    pos_y_ai[0] = initial_pos[1]

    for i in range(1, n):
        dt = dt_arr[i]
        heading_ai[i] = heading_ai[i-1] + gyro_z[i] * dt
        v_mid = 0.5 * (v_dense[i-1] + v_dense[i])
        h_mid = 0.5 * (heading_ai[i-1] + heading_ai[i])
        pos_x_ai[i] = pos_x_ai[i-1] + v_mid * np.sin(h_mid) * dt
        pos_y_ai[i] = pos_y_ai[i-1] + v_mid * np.cos(h_mid) * dt

    res = pd.DataFrame({
        "timestamp": t_orig,
        "ai_speed": v_dense,
        "ai_speed_std": std_dense,
        "ai_heading": heading_ai,
        "ai_pos_x": pos_x_ai,
        "ai_pos_y": pos_y_ai
    })

    if "pos_x" in df.columns and "pos_y" in df.columns:
        dx = pos_x_ai - df["pos_x"].values
        dy = pos_y_ai - df["pos_y"].values
        res["ai_pos_error_m"] = np.sqrt(dx**2 + dy**2)

    return res
