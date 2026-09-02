"""
core/models/base.py
Abstract Base Motion Estimator Contract.
Platform-independent interface for velocity and kinematic motion prediction.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from core.interfaces.canonical import MotionEstimate, SensorFrame


class BaseMotionEstimator(ABC):
    """Abstract Base Class for all ML Motion and Speed Estimators."""
    
    @abstractmethod
    def train(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_calib: Optional[Any] = None,
        y_calib: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Trains the model and optionally calibrates uncertainty on an isolated calibration set."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Predicts scalar or vector velocity."""
        pass

    @abstractmethod
    def predict_with_uncertainty(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (velocity_predictions, uncertainty_sigma_v)."""
        pass

    @abstractmethod
    def get_model_size_kb(self) -> float:
        """Returns serialized model size in Kilobytes."""
        pass
