"""
src/speed_model.py
Machine Learning Speed Regression & Calibrated Uncertainty Estimation Suite:
Supports:
  - Random Forest (Tree Ensemble Uncertainty)
  - HistGradientBoosting (Fast Mobile Histogram Trees & Quantile Loss)
  - XGBoost Regressor (Gradient Boosted Trees)
  - 1D-CNN / Temporal Convolutional Network (PyTorch Lightweight Neural Architecture)
  - Split Conformal Prediction for distribution-free calibrated prediction intervals

XGBoost and PyTorch are optional. Annotations are deferred (PEP 563) so that
`forward(self, x: torch.Tensor)` does not evaluate torch.Tensor at class-definition time
when torch is absent - otherwise merely importing this module crashes a fresh clone that
only wants the RandomForest path.
"""
from __future__ import annotations

import os
import io
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from scipy.interpolate import interp1d

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Optional backends. The module supports RandomForest / HistGradientBoosting without
# either of these, and a hard import made a fresh clone crash on `import speed_model`
# before it could run anything at all.
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    import types
    torch = optim = None
    HAS_TORCH = False
    # Conv1DSpeedNet subclasses nn.Module at import time, so the stub has to supply a real
    # base class. Training raises instead - see _train_torch.
    nn = types.SimpleNamespace(Module=object)

TORCH_MISSING_MSG = (
    "model_type='cnn' requires PyTorch; pip install torch, or use 'rf' / 'hgb'."
)

# ── 1. Lightweight 1D-CNN Architecture (PyTorch) ─────────────────────────────
class Conv1DSpeedNet(nn.Module):
    """
    Lightweight 1D Temporal Convolutional Neural Network for on-device speed estimation.
    Designed for low memory and sub-millisecond CPU inference.
    """
    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.fc_in = nn.Linear(in_features, hidden_dim)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.head = nn.Sequential(
            nn.Linear(32 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_features)
        h = self.relu(self.fc_in(x))
        h = h.unsqueeze(1) # (batch, 1, hidden_dim)
        h = self.relu(self.bn1(self.conv1(h)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        out = self.head(h)
        return torch.relu(out) # Speed >= 0

# ── 2. Unified Speed Model Interface ─────────────────────────────────────────
class SpeedRegressorModel:
    """
    Unified multi-model speed estimator with calibrated uncertainty quantification.
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
        self.feature_names = []
        self.conformal_q_hat = 0.50 # Conformal non-conformity threshold
        
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
            self.q_lower_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.05, max_iter=100, max_depth=max_depth, random_state=random_state)
            self.q_upper_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, max_iter=100, max_depth=max_depth, random_state=random_state)
        elif self.model_type in ["xgboost", "xgb"]:
            if not HAS_XGBOOST:
                raise ImportError(
                    "model_type='xgboost' requires xgboost; pip install xgboost, "
                    "or use 'rf' / 'hgb'."
                )
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=min(max_depth, 8),
                learning_rate=0.08, random_state=random_state, n_jobs=-1
            )
        elif self.model_type in ["1d_cnn", "cnn", "tcn"]:
            self.model = None # Initialized upon knowing feature dimension
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        self.feature_names = list(X.columns)
        n_feats = X.shape[1]

        if self.model_type in ["1d_cnn", "cnn", "tcn"]:
            if not HAS_TORCH:
                raise ImportError(TORCH_MISSING_MSG)
            self.model = Conv1DSpeedNet(in_features=n_feats)
            self._train_torch(X.values, y, epochs=40, batch_size=64)
        elif self.model_type in ["hist_gb", "histgb"]:
            self.model.fit(X.values, y)
            if self.uncertainty_method == "quantile":
                self.q_lower_model.fit(X.values, y)
                self.q_upper_model.fit(X.values, y)
        else:
            self.model.fit(X.values, y)

        self.is_trained = True

        # Calibrate Conformal Prediction on validation data if provided
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_preds = self.predict(X_val)
            residuals = np.abs(y_val - val_preds)
            alpha = 0.10 # 90% confidence level
            n_val = len(residuals)
            k = int(np.ceil((n_val + 1) * (1.0 - alpha)))
            k = min(n_val - 1, max(0, k))
            self.conformal_q_hat = float(np.sort(residuals)[k])

        train_preds = self.predict(X)
        return {
            "train_rmse": float(np.sqrt(mean_squared_error(y, train_preds))),
            "train_mae": float(mean_absolute_error(y, train_preds)),
            "train_r2": float(r2_score(y, train_preds))
        }

    def _train_torch(self, X_arr: np.ndarray, y_arr: np.ndarray, epochs: int = 40, batch_size: int = 64):
        X_t = torch.tensor(X_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(self.model.parameters(), lr=0.003, weight_decay=1e-4)

        self.model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")
        
        if self.model_type in ["1d_cnn", "cnn", "tcn"]:
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X.values, dtype=torch.float32)
                preds = self.model(X_t).squeeze().numpy()
                return np.maximum(0.0, preds)
        else:
            return np.maximum(0.0, self.model.predict(X.values))

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes velocity prediction and uncertainty sigma_v.
        Methods:
          - 'ensemble': Tree ensemble standard deviation across decision trees
          - 'quantile': Transformed interval sigma = (q95 - q05) / (2 * 1.645)
          - 'conformal': Calibrated residual sigma = conformal_q_hat / 1.645
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")

        y_mean = self.predict(X)

        if self.uncertainty_method == "conformal":
            # Conformal prediction uncertainty
            sigma_v = np.ones(len(y_mean)) * max(0.10, self.conformal_q_hat / 1.645)
            return y_mean, sigma_v

        elif self.uncertainty_method == "quantile" and self.model_type in ["hist_gb", "histgb"]:
            q_low = np.maximum(0.0, self.q_lower_model.predict(X.values))
            q_high = np.maximum(0.0, self.q_upper_model.predict(X.values))
            interval_width = np.maximum(0.10, q_high - q_low)
            sigma_v = interval_width / (2.0 * 1.645)
            return y_mean, sigma_v

        elif hasattr(self.model, "estimators_"): # Random Forest
            tree_preds = np.array([tree.predict(X.values) for tree in self.model.estimators_])
            y_std = np.std(tree_preds, axis=0)
            return y_mean, np.maximum(0.05, y_std)

        else: # Default residual fallback
            return y_mean, np.ones(len(y_mean)) * max(0.20, self.conformal_q_hat / 1.645)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        y_pred, y_std = self.predict_with_uncertainty(X)
        metrics = {
            "test_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "test_mae": float(mean_absolute_error(y, y_pred)),
            "test_r2": float(r2_score(y, y_pred)),
            "mean_uncertainty_sigma": float(np.mean(y_std))
        }
        return y_pred, y_std, metrics

    def get_feature_importances(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Top-n (feature, importance) pairs, descending.

        Called by scripts/02_train_and_fuse.py, which had been crashing here because the
        method did not exist. Worth having rather than deleting the call: with the feature
        set now frame-invariant, the ranking is the quickest check that the model is
        keyed on motion rather than on how the phone happens to be mounted.

        Returns [] for models without a native importance measure (the CNN).
        """
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return []
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        pairs = sorted(zip(names, (float(v) for v in importances)),
                       key=lambda kv: kv[1], reverse=True)
        return pairs[:top_n]

    def get_model_size_kb(self) -> float:
        """Computes serialized model size in Kilobytes."""
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
            "conformal_q_hat": self.conformal_q_hat
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.model_type = data["type"]
        self.uncertainty_method = data.get("uncertainty_method", "ensemble")
        self.conformal_q_hat = data.get("conformal_q_hat", 0.50)
        self.is_trained = True

# ── 3. AI Dead Reckoning Trajectory Reconstruction ───────────────────────────
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
