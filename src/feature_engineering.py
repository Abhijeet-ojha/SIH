"""
src/feature_engineering.py
Compatibility Bridge delegating to core/features/extractor.py and core/features/leakage_guard.py.
Extracts strictly causal sliding window features with zero future data leakage.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

from core.features.extractor import CausalFeatureExtractor
from core.features.leakage_guard import verify_feature_matrix_leakage, verify_causality


def extract_causal_window_features(
    df: pd.DataFrame,
    window_sec: float = 1.5,
    step_sec: float = 0.2,
    sample_rate: float = 10.0,
    feature_group: str = "all"
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Extracts strictly CAUSAL sliding window features from IMU signals.
    Prediction timestamp is strictly at the trailing edge of the window (t_current).
    """
    extractor = CausalFeatureExtractor(
        window_sec=window_sec,
        step_sec=step_sec,
        sample_rate_hz=sample_rate,
        feature_group=feature_group
    )
    return extractor.extract_features(df)


def extract_window_features(
    df: pd.DataFrame,
    window_sec: float = 1.5,
    step_sec: float = 0.2,
    sample_rate: float = 10.0
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for legacy scripts."""
    X_df, y_spd, t_end, _ = extract_causal_window_features(
        df, window_sec=window_sec, step_sec=step_sec, sample_rate=sample_rate, feature_group="all"
    )
    return X_df, y_spd, t_end
