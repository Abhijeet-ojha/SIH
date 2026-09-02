"""
core/uncertainty/calibrator.py
Distribution-Free Split Conformal Prediction and Uncertainty Calibration Suite.

Mathematical Formulation:
  1. Let D_calib = {(x_i, y_i)}_{i=1}^n be an ISOLATED calibration set (disjoint from train and test).
  2. Non-conformity score: R_i = |y_i - f(x_i)|
  3. For a target coverage level 1 - alpha (e.g. 0.90 for 90%), empirical quantile:
     q_hat = Quantile_{1 - alpha}(R_1, ..., R_n; (n+1)/n)
  4. Guaranteed finite-sample marginal coverage:
     P(Y_{test} in [f(X_{test}) - q_hat, f(X_{test}) + q_hat]) >= 1 - alpha
  
Engineering Mapping to EKF Covariance:
  Under a nominal normal model with significance alpha = 0.10 (z_0.95 = 1.645),
  the half-width q_hat maps to process standard deviation:
     sigma_v = max(sigma_floor, q_hat / 1.645)
  This engineering conversion is explicitly documented as a distribution-free interval projection.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class ConformalUncertaintyCalibrator:
    """
    Split Conformal Calibrator for Motion Estimators.
    """
    def __init__(self, target_coverage: float = 0.90, min_sigma_mps: float = 0.10):
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage
        self.min_sigma = min_sigma_mps
        self.q_hat: float = 0.50
        self.is_calibrated: bool = False
        self.n_calib_samples: int = 0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calibrates non-conformity threshold q_hat on an isolated calibration split."""
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        if n == 0:
            self.q_hat = 0.50
            return
            
        # Finite sample adjusted quantile index: ceil((n + 1) * (1 - alpha)) / n
        k = int(np.ceil((n + 1) * self.target_coverage))
        k = min(n - 1, max(0, k - 1))
        
        sorted_res = np.sort(residuals)
        self.q_hat = float(sorted_res[k])
        self.is_calibrated = True
        self.n_calib_samples = n

    def get_prediction_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns [lower_bound, upper_bound] calibrated prediction intervals."""
        lower = np.maximum(0.0, y_pred - self.q_hat)
        upper = y_pred + self.q_hat
        return lower, upper

    def get_ekf_sigma(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Maps conformal quantile q_hat to EKF process noise standard deviation.
        sigma_v = max(min_sigma, q_hat / z_{1 - alpha/2})
        """
        # z-score for two-sided (1 - alpha): for 90% -> 1.645, for 95% -> 1.960
        z_score = 1.645 if abs(self.target_coverage - 0.90) < 0.02 else 1.960
        sigma_scalar = max(self.min_sigma, self.q_hat / z_score)
        return np.ones(len(y_pred), dtype=float) * sigma_scalar

    def evaluate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluates empirical coverage, width, and calibration error on held-out test data."""
        lower, upper = self.get_prediction_interval(y_pred)
        covered = (y_true >= lower) & (y_true <= upper)
        empirical_cov = float(np.mean(covered))
        mean_width = float(np.mean(upper - lower))
        cal_error = abs(empirical_cov - self.target_coverage)

        return {
            "target_coverage": self.target_coverage,
            "empirical_coverage_pct": empirical_cov * 100.0,
            "mean_interval_width_mps": mean_width,
            "calibration_error": cal_error,
            "q_hat_mps": self.q_hat,
            "n_calib_samples": self.n_calib_samples
        }


def compute_uncertainty_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> Dict[str, float]:
    """Evaluates 90% and 95% nominal Gaussian coverage for models outputting sigma_v."""
    # 90% interval (z = 1.645)
    lower_90 = np.maximum(0.0, y_pred - 1.645 * y_std)
    upper_90 = y_pred + 1.645 * y_std
    cov_90 = float(np.mean((y_true >= lower_90) & (y_true <= upper_90)))
    width_90 = float(np.mean(upper_90 - lower_90))

    # 95% interval (z = 1.960)
    lower_95 = np.maximum(0.0, y_pred - 1.960 * y_std)
    upper_95 = y_pred + 1.960 * y_std
    cov_95 = float(np.mean((y_true >= lower_95) & (y_true <= upper_95)))
    width_95 = float(np.mean(upper_95 - lower_95))

    return {
        "coverage_90_pct": cov_90 * 100.0,
        "mean_width_90_mps": width_90,
        "calibration_error_90": abs(cov_90 - 0.90),
        "coverage_95_pct": cov_95 * 100.0,
        "mean_width_95_mps": width_95,
        "calibration_error_95": abs(cov_95 - 0.95),
        "mean_sigma_mps": float(np.mean(y_std))
    }
