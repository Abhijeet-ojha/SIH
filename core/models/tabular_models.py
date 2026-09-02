"""
core/models/tabular_models.py
Tabular Machine Learning Speed Estimators:
  - Random Forest Regressor (Tree Ensemble Uncertainty)
  - HistGradientBoosting Regressor (Fast Mobile Histogram Trees & Quantile Loss)
  - XGBoost Regressor (Gradient Boosted Decision Trees + Conformal Calibration)
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from core.models.base import BaseMotionEstimator
from core.uncertainty.calibrator import ConformalUncertaintyCalibrator


class TabularSpeedModel(BaseMotionEstimator):
    """
    Unified Tabular Speed Estimator for engineered IMU feature vectors.
    """
    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
        uncertainty_method: str = "ensemble"  # 'ensemble', 'conformal', 'quantile'
    ):
        self.model_type = model_type.lower()
        self.uncertainty_method = uncertainty_method.lower()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        self.is_trained = False
        self.feature_names: List[str] = []
        self.conformal_calibrator = ConformalUncertaintyCalibrator(target_coverage=0.90)

        if self.model_type in ["random_forest", "rf"]:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                n_jobs=-1, random_state=random_state
            )
        elif self.model_type in ["hist_gb", "histgb"]:
            self.model = HistGradientBoostingRegressor(
                max_iter=150, max_depth=max_depth, random_state=random_state
            )
            # Auxiliary models for Quantile regression
            self.q_lower_model = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.05, max_iter=100, max_depth=max_depth, random_state=random_state
            )
            self.q_upper_model = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.95, max_iter=100, max_depth=max_depth, random_state=random_state
            )
        elif self.model_type in ["xgboost", "xgb"]:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=min(max_depth, 8),
                learning_rate=0.08, random_state=random_state, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown tabular model_type: {model_type}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_calib: Optional[pd.DataFrame] = None,
        y_calib: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        self.feature_names = list(X_train.columns)
        X_mat = X_train.values

        if self.model_type in ["hist_gb", "histgb"]:
            self.model.fit(X_mat, y_train)
            if self.uncertainty_method == "quantile":
                self.q_lower_model.fit(X_mat, y_train)
                self.q_upper_model.fit(X_mat, y_train)
        else:
            self.model.fit(X_mat, y_train)

        self.is_trained = True

        # Calibrate Conformal Prediction on isolated calibration set if provided
        if X_calib is not None and y_calib is not None and len(X_calib) > 0:
            calib_preds = self.predict(X_calib)
            self.conformal_calibrator.calibrate(y_calib, calib_preds)

        train_preds = self.predict(X_train)
        return {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
            "train_mae": float(mean_absolute_error(y_train, train_preds)),
            "train_r2": float(r2_score(y_train, train_preds))
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        preds = self.model.predict(X_mat)
        return np.maximum(0.0, preds)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (velocity_predictions, sigma_v).
        Supports:
          - 'ensemble': Tree ensemble standard deviation across RF decision trees
          - 'quantile': Transformed quantile interval sigma = (q95 - q05) / (2 * 1.645)
          - 'conformal': Calibrated empirical conformal score sigma = q_hat / 1.645
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")

        y_mean = self.predict(X)
        X_mat = X.values if isinstance(X, pd.DataFrame) else X

        if self.uncertainty_method == "conformal":
            sigma_v = self.conformal_calibrator.get_ekf_sigma(y_mean)
            return y_mean, sigma_v

        elif self.uncertainty_method == "quantile" and self.model_type in ["hist_gb", "histgb"]:
            q_low = np.maximum(0.0, self.q_lower_model.predict(X_mat))
            q_high = np.maximum(0.0, self.q_upper_model.predict(X_mat))
            interval_width = np.maximum(0.10, q_high - q_low)
            sigma_v = interval_width / (2.0 * 1.645)
            return y_mean, sigma_v

        elif hasattr(self.model, "estimators_"):  # Random Forest Ensemble
            tree_preds = np.array([tree.predict(X_mat) for tree in self.model.estimators_])
            y_std = np.std(tree_preds, axis=0)
            return y_mean, np.maximum(0.05, y_std)

        else:
            # Fallback conformal sigma
            sigma_v = self.conformal_calibrator.get_ekf_sigma(y_mean)
            return y_mean, sigma_v

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        y_pred, y_std = self.predict_with_uncertainty(X)
        metrics = {
            "test_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "test_mae": float(mean_absolute_error(y, y_pred)),
            "test_r2": float(r2_score(y, y_pred)),
            "mean_uncertainty_sigma": float(np.mean(y_std))
        }
        return y_pred, y_std, metrics

    def get_model_size_kb(self) -> float:
        buf = io.BytesIO()
        joblib.dump(self.model, buf)
        return float(len(buf.getvalue()) / 1024.0)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "features": self.feature_names,
            "type": self.model_type,
            "uncertainty_method": self.uncertainty_method,
            "conformal_q_hat": self.conformal_calibrator.q_hat,
            "q_lower_model": getattr(self, "q_lower_model", None),
            "q_upper_model": getattr(self, "q_upper_model", None)
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.model_type = data["type"]
        self.uncertainty_method = data.get("uncertainty_method", "ensemble")
        self.conformal_calibrator.q_hat = data.get("conformal_q_hat", 0.50)
        self.conformal_calibrator.is_calibrated = True
        if "q_lower_model" in data and data["q_lower_model"] is not None:
            self.q_lower_model = data["q_lower_model"]
        if "q_upper_model" in data and data["q_upper_model"] is not None:
            self.q_upper_model = data["q_upper_model"]
        self.is_trained = True
