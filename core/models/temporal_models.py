"""
core/models/temporal_models.py
True Temporal Sequence 1D-CNN / Temporal Convolutional Network (TCN).
Operates directly on raw sensor sequences [N x C x T] where:
  - C = 6 IMU Channels [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
  - T = Time samples (e.g. 15 samples for 1.5s window at 10 Hz)
Features causal dilated 1D convolutions with receptive field matching window duration W.
"""

import os
import io
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from core.models.base import BaseMotionEstimator
from core.uncertainty.calibrator import ConformalUncertaintyCalibrator


class CausalConv1dBlock(nn.Module):
    """Causal Dilated 1D Convolutional Residual Block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.pad_len
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        h = self.conv(x)
        # Trim future padding to preserve strict causality
        if self.pad_len > 0:
            h = h[:, :, :-self.pad_len]
        h = self.relu(self.bn(h))
        return h + self.shortcut(x)


class CausalTemporalSpeedNet(nn.Module):
    """
    Lightweight Causal Temporal Convolutional Network for raw IMU sequence speed regression.
    Input shape: (batch_size, num_channels, time_steps) -> [B, 6, T]
    """
    def __init__(self, in_channels: int = 6, hidden_dim: int = 24):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.block1 = CausalConv1dBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
        self.block2 = CausalConv1dBlock(hidden_dim, hidden_dim * 2, kernel_size=3, dilation=2)
        self.block3 = CausalConv1dBlock(hidden_dim * 2, hidden_dim * 2, kernel_size=3, dilation=4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels=6, time)
        h = self.in_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.pool(h).squeeze(-1) # (batch, hidden_dim * 2)
        out = self.head(h)
        return torch.relu(out) # Physical constraint: forward speed >= 0


class TemporalSequenceSpeedModel(BaseMotionEstimator):
    """
    High-level wrapper for training and evaluating CausalTemporalSpeedNet on [T x C] IMU tensors.
    """
    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 24,
        epochs: int = 45,
        batch_size: int = 64,
        lr: float = 0.003,
        random_state: int = 42,
        uncertainty_method: str = "conformal"
    ):
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.uncertainty_method = uncertainty_method
        
        torch.manual_seed(random_state)
        self.net = CausalTemporalSpeedNet(in_channels=in_channels, hidden_dim=hidden_dim)
        self.is_trained = False
        self.conformal_calibrator = ConformalUncertaintyCalibrator(target_coverage=0.90)

    def _prepare_tensor(self, X: np.ndarray) -> torch.Tensor:
        # Expected input shape: (N, T, C) -> Transpose to PyTorch (N, C, T)
        if isinstance(X, pd.DataFrame):
            # If tabular dataframe passed by mistake, reshape
            val = X.values.astype(np.float32)
            val = val.reshape((len(val), self.in_channels, -1))
            return torch.tensor(val, dtype=torch.float32)
        elif len(X.shape) == 3 and X.shape[2] == self.in_channels:
            # (N, T, C) -> transpose to (N, C, T)
            return torch.tensor(np.transpose(X, (0, 2, 1)), dtype=torch.float32)
        elif len(X.shape) == 3 and X.shape[1] == self.in_channels:
            return torch.tensor(X, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid input shape for TemporalSequenceSpeedModel: {X.shape}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: Optional[np.ndarray] = None,
        y_calib: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        X_t = self._prepare_tensor(X_train)
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        self.net.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.net(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        self.is_trained = True

        # Conformal calibration on isolated calibration split
        if X_calib is not None and y_calib is not None and len(X_calib) > 0:
            calib_preds = self.predict(X_calib)
            self.conformal_calibrator.calibrate(y_calib, calib_preds)

        train_preds = self.predict(X_train)
        return {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
            "train_mae": float(mean_absolute_error(y_train, train_preds)),
            "train_r2": float(r2_score(y_train, train_preds))
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Temporal model must be trained before predicting.")
        self.net.eval()
        with torch.no_grad():
            X_t = self._prepare_tensor(X)
            preds = self.net(X_t).squeeze().cpu().numpy()
            if preds.ndim == 0:
                preds = np.array([float(preds)])
            return np.maximum(0.0, preds)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = self.predict(X)
        sigma_v = self.conformal_calibrator.get_ekf_sigma(preds)
        return preds, sigma_v

    def get_model_size_kb(self) -> float:
        buf = io.BytesIO()
        torch.save(self.net.state_dict(), buf)
        return float(len(buf.getvalue()) / 1024.0)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim,
            "conformal_q_hat": self.conformal_calibrator.q_hat
        }, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath, map_location="cpu", weights_only=True)
        self.in_channels = data["in_channels"]
        self.hidden_dim = data["hidden_dim"]
        self.net = CausalTemporalSpeedNet(in_channels=self.in_channels, hidden_dim=self.hidden_dim)
        self.net.load_state_dict(data["state_dict"])
        self.conformal_calibrator.q_hat = data.get("conformal_q_hat", 0.50)
        self.conformal_calibrator.is_calibrated = True
        self.is_trained = True
