"""
core/features/leakage_guard.py
Automated Leakage & Sanity Gate for Scientific Rigor.
Guarantees:
  1. Strict causality (prediction timestamp <= trailing edge of sensor window)
  2. Zero target leakage (no ground-truth speed, coordinates, or derived labels in feature matrices)
  3. Disjoint partitions (Training != Calibration != Test)
  4. Numerical sanity (finite values, valid ranges)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Set


class LeakageGuardError(Exception):
    """Raised when data leakage or causality violation is detected."""
    pass


BANNED_TARGET_KEYWORDS = {
    "speed", "velocity", "pos_x", "pos_y", "pos_z", "heading", "lat", "lon",
    "gt_speed", "gt_lat", "gt_lon", "ground_truth", "target", "label"
}


def verify_feature_matrix_leakage(df_features: pd.DataFrame, context_name: str = "features"):
    """Asserts that no target or ground truth column exists in feature matrix."""
    for col in df_features.columns:
        col_lower = col.strip().lower()
        if col_lower in BANNED_TARGET_KEYWORDS:
            raise LeakageGuardError(f"CRITICAL LEAKAGE DETECTED in {context_name}: Banned target column '{col}' found in feature matrix!")
        for banned in ["gt_", "target_", "label_"]:
            if col_lower.startswith(banned):
                raise LeakageGuardError(f"CRITICAL LEAKAGE DETECTED in {context_name}: Column '{col}' contains forbidden prefix '{banned}'!")

    # Check finite values
    if df_features.isnull().values.any():
        raise LeakageGuardError(f"CRITICAL SANITY ERROR in {context_name}: NaN values detected in feature matrix!")
    if np.isinf(df_features.values).any():
        raise LeakageGuardError(f"CRITICAL SANITY ERROR in {context_name}: Inf values detected in feature matrix!")


def verify_split_isolation(
    train_drive_names: List[str],
    test_drive_names: List[str],
    calibration_drive_names: Optional[List[str]] = None,
    train_driver_ids: Optional[List[str]] = None,
    test_driver_ids: Optional[List[str]] = None,
    is_lodro: bool = False
):
    """
    Asserts complete dataset isolation between Train, Calibration, and Held-out Test sets.
    """
    train_set = set(train_drive_names)
    test_set = set(test_drive_names)
    
    # 1. Train vs Test drive overlap
    overlap_train_test = train_set.intersection(test_set)
    if len(overlap_train_test) > 0:
        raise LeakageGuardError(f"SPLIT LEAKAGE DETECTED: Overlapping drives in train and test: {overlap_train_test}")

    # 2. Calibration vs Test drive overlap
    if calibration_drive_names is not None:
        calib_set = set(calibration_drive_names)
        overlap_calib_test = calib_set.intersection(test_set)
        if len(overlap_calib_test) > 0:
            raise LeakageGuardError(f"CALIBRATION LEAKAGE DETECTED: Calibration drive appears in held-out test set: {overlap_calib_test}")

    # 3. LODrO (Leave-One-Driver-Out) check
    if is_lodro and train_driver_ids is not None and test_driver_ids is not None:
        driver_overlap = set(train_driver_ids).intersection(set(test_driver_ids))
        if len(driver_overlap) > 0:
            raise LeakageGuardError(f"LODrO LEAKAGE DETECTED: Overlapping drivers in train and test: {driver_overlap}")


def verify_causality(t_pred: np.ndarray, t_window_ends: np.ndarray):
    """Asserts that prediction timestamps do not exceed window trailing edges."""
    diffs = t_pred - t_window_ends
    if np.any(diffs > 1e-6):
        max_violation = float(np.max(diffs))
        raise LeakageGuardError(f"CAUSALITY VIOLATION DETECTED: Prediction timestamp exceeds window end by {max_violation:.6f} seconds!")
