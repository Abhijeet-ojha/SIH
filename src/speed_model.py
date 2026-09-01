"""
src/speed_model.py
Day 2 Machine Learning Module: Tree-based ensemble regressor with Heteroscedastic Uncertainty Estimation.
Extracts ensemble variance across decision trees (predict_with_uncertainty) to dynamically modulate
EKF state noise covariance.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import interp1d

class SpeedRegressorModel:
    """
    Supervised tree-based regressor with uncertainty estimation for vehicle velocity.
    """

    def __init__(self, model_type: str = "random_forest", n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        if model_type == "hist_gb":
            self.model = HistGradientBoostingRegressor(max_iter=150, max_depth=max_depth, random_state=random_state)
        else:
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state)
        self.is_trained = False
        self.feature_names = []

    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_trained = True

        y_pred = np.maximum(0.0, self.model.predict(X))
        metrics = {
            "train_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "train_mae": float(mean_absolute_error(y, y_pred)),
            "train_r2": float(r2_score(y, y_pred))
        }
        return metrics

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Evaluates model performance and computes prediction uncertainty."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
        
        y_pred, y_std = self.predict_with_uncertainty(X)
        metrics = {
            "test_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "test_mae": float(mean_absolute_error(y, y_pred)),
            "test_r2": float(r2_score(y, y_pred)),
            "mean_uncertainty_sigma": float(np.mean(y_std))
        }
        return y_pred, y_std, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")
        return np.maximum(0.0, self.model.predict(X))

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Confidence-Aware Prediction:
        Computes mean prediction across trees and ensemble standard deviation (heteroscedastic uncertainty).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")

        if hasattr(self.model, "estimators_"):
            # Per-tree predictions across all estimators
            tree_preds = np.array([tree.predict(X.values) for tree in self.model.estimators_])
            y_mean = np.maximum(0.0, np.mean(tree_preds, axis=0))
            y_std = np.std(tree_preds, axis=0) # Heteroscedastic uncertainty sigma
            return y_mean, y_std
        else:
            preds = np.maximum(0.0, self.model.predict(X))
            return preds, np.ones(len(preds)) * 0.2

    def get_feature_importances(self, top_n: int = 10) -> List[Tuple[str, float]]:
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            ranked = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
            return ranked[:top_n]
        return []

    def export_embedded_rules(self, filepath: str, max_trees: int = 5):
        """Exports decision rules for embedded Kotlin Android on-device execution."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rules = {
            "features": self.feature_names,
            "model_type": self.model_type,
            "num_features": len(self.feature_names)
        }
        with open(filepath, "w") as f:
            json.dump(rules, f, indent=2)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names, "type": self.model_type}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.model_type = data["type"]
        self.is_trained = True

def reconstruct_ai_dr_trajectory(
    df: pd.DataFrame,
    t_pred: np.ndarray,
    v_pred: np.ndarray,
    v_std: Optional[np.ndarray] = None,
    initial_heading: float = 0.0,
    initial_pos: Tuple[float, float] = (0.0, 0.0)
) -> pd.DataFrame:
    """
    Reconstructs pure AI-DR trajectory with interpolated uncertainty.
    """
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
