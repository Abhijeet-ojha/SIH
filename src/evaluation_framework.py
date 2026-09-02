"""
src/evaluation_framework.py
Comprehensive 5-Level Evaluation Framework for SIH 2026 PS-168:
- Level 1: ML Speed Regression Metrics (MAE, RMSE, R2, MedAE, P95)
- Level 2: Uncertainty Calibration (90%/95% Coverage, Mean Width, Calibration Error)
- Level 3: Drive-Level & Driver-Level Generalization (LODO & LODrO)
- Level 4: Downstream Navigation (90s Blackout Exit Error, Peak Drift, Trajectory RMSE)
- Level 5: Mobile Feasibility (Latency ms/window, RAM, Model Size KB)
Strict Sanity & Leakage Gate enforced before every experiment.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import extract_causal_window_features
from src.speed_model import reconstruct_ai_dr_trajectory
from src.naive_dr import NaiveDeadReckoning
from src.fusion_ekf import run_fusion_pipeline
from src.metrics import calculate_benchmark_metrics

# ── 1. Strict Sanity & Leakage Gate ──────────────────────────────────────────
def verify_leakage_gate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    train_drive_names: List[str],
    test_drive_names: List[str],
    train_driver_ids: Optional[List[str]] = None,
    test_driver_ids: Optional[List[str]] = None,
    is_lodro: bool = False
):
    """
    Automated assertion gate: Fails immediately if any form of data or label leakage is detected.
    """
    # 1. No target or spatial coordinate columns in feature matrices
    banned_cols = ["speed", "pos_x", "pos_y", "heading", "lat", "lon", "gt_speed", "gt_lat", "gt_lon"]
    for c in X_train.columns:
        c_lower = c.lower()
        assert c_lower not in banned_cols, f"LEAKAGE GATE FAILED: Target column '{c}' present in X_train!"
    for c in X_test.columns:
        c_lower = c.lower()
        assert c_lower not in banned_cols, f"LEAKAGE GATE FAILED: Target column '{c}' present in X_test!"

    # 2. No test drive in training drives
    overlap_drives = set(train_drive_names).intersection(set(test_drive_names))
    assert len(overlap_drives) == 0, f"LEAKAGE GATE FAILED: Overlapping drives found in train/test: {overlap_drives}"

    # 3. If LODrO (Leave-One-Driver-Out), assert strictly disjoint driver sets
    if is_lodro and train_driver_ids is not None and test_driver_ids is not None:
        overlap_drivers = set(train_driver_ids).intersection(set(test_driver_ids))
        assert len(overlap_drivers) == 0, f"LODrO LEAKAGE GATE FAILED: Overlapping drivers in train/test: {overlap_drivers}"

    # 4. Check shape and finite values
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train rows!"
    assert len(X_test) == len(y_test), "Mismatch between X_test and y_test rows!"
    assert not np.isnan(y_train).any(), "NaN detected in y_train!"
    assert not np.isnan(y_test).any(), "NaN detected in y_test!"

# ── 2. Level 1 & Level 2 Metric Calculations ─────────────────────────────────
def evaluate_ml_speed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Level 1: Speed Regression Metrics."""
    abs_err = np.abs(y_true - y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "medae": float(np.median(abs_err)),
        "p95_error": float(np.percentile(abs_err, 95))
    }

def evaluate_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray
) -> Dict[str, float]:
    """Level 2: Prediction Interval Calibration & Coverage."""
    # 90% Nominal interval (z = 1.645)
    lower_90 = np.maximum(0.0, y_pred - 1.645 * y_std)
    upper_90 = y_pred + 1.645 * y_std
    in_90 = (y_true >= lower_90) & (y_true <= upper_90)
    cov_90 = float(np.mean(in_90))
    width_90 = float(np.mean(upper_90 - lower_90))

    # 95% Nominal interval (z = 1.960)
    lower_95 = np.maximum(0.0, y_pred - 1.960 * y_std)
    upper_95 = y_pred + 1.960 * y_std
    in_95 = (y_true >= lower_95) & (y_true <= upper_95)
    cov_95 = float(np.mean(in_95))
    width_95 = float(np.mean(upper_95 - lower_95))

    cal_err_90 = abs(cov_90 - 0.90)
    cal_err_95 = abs(cov_95 - 0.95)

    return {
        "coverage_90_pct": cov_90 * 100.0,
        "mean_width_90_mps": width_90,
        "cal_error_90": cal_err_90,
        "coverage_95_pct": cov_95 * 100.0,
        "mean_width_95_mps": width_95,
        "cal_error_95": cal_err_95,
        "mean_sigma_mps": float(np.mean(y_std))
    }

# ── 3. Level 4: Downstream Navigation Evaluator ──────────────────────────────
def evaluate_downstream_navigation(
    drive_dataset,
    t_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    blackout_start_sec: float = 60.0,
    blackout_end_sec: Optional[float] = None
) -> Dict[str, float]:
    """Level 4: EKF Downstream GNSS Blackout Navigation."""
    df = drive_dataset.get_data()
    d_id = drive_dataset.driver_id
    if blackout_end_sec is None:
        blackout_end_sec = min(drive_dataset.duration_sec - 10.0, 150.0)

    init_heading = df["heading"].iloc[0] if "heading" in df.columns else 0.0
    init_speed = df["speed"].iloc[0] if "speed" in df.columns else 0.0
    init_pos = (df["pos_x"].iloc[0], df["pos_y"].iloc[0]) if "pos_x" in df.columns else (0.0, 0.0)

    naive_res = NaiveDeadReckoning(init_heading, init_speed, init_pos).compute(df)
    ai_dr_res = reconstruct_ai_dr_trajectory(df, t_test, y_pred, v_std=y_std, initial_heading=init_heading, initial_pos=init_pos)

    fused_res = run_fusion_pipeline(
        df=df,
        ai_speed=ai_dr_res["ai_speed"].values,
        ai_speed_std=ai_dr_res["ai_speed_std"].values,
        driver_style="aggressive" if d_id == "E" else "normal",
        blackout_start_sec=blackout_start_sec,
        blackout_end_sec=blackout_end_sec
    )

    m = calculate_benchmark_metrics(df, naive_res, ai_dr_res, fused_res, blackout_start_sec, blackout_end_sec)
    return {
        "blackout_exit_error_m": float(m["ai_dr_gnss_ekf_fusion"]["blackout_terminal_exit_error_m"]),
        "blackout_peak_error_m": float(m["ai_dr_gnss_ekf_fusion"]["blackout_max_error_m"]),
        "post_gps_settled_error_m": float(m["ai_dr_gnss_ekf_fusion"]["post_reacquisition_settled_error_m"]),
        "trajectory_rmse_m": float(m["ai_dr_gnss_ekf_fusion"]["rmse_m"])
    }
