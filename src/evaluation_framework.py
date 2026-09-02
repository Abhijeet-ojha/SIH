"""
src/evaluation_framework.py
Comprehensive 5-Level Evaluation Framework for SIH 2026 PS-168:
  - Level 1: ML Speed Regression Metrics (MAE, RMSE, R2, MedAE, P95)
  - Level 2: Uncertainty Calibration (90%/95% Coverage, Mean Width, Calibration Error)
  - Level 3: Drive-Level & Driver-Level Generalization (LODO & LODrO)
  - Level 4: Downstream Navigation (90s Blackout Exit Error, Peak Drift, Trajectory RMSE)
  - Level 5: Mobile Feasibility (Latency ms/window, RAM, Model Size KB)

Enforces strict 3-way data splits (Train, Isolated Calibration, Held-out Test) and automated leakage gates.
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

from core.features.extractor import CausalFeatureExtractor
from core.features.leakage_guard import verify_feature_matrix_leakage, verify_split_isolation, verify_causality
from core.models.tabular_models import TabularSpeedModel
from core.fusion.ekf_6state import run_6state_ekf_fusion
from core.uncertainty.calibrator import compute_uncertainty_metrics
from src.naive_dr import NaiveDeadReckoning
from src.metrics import calculate_benchmark_metrics
from scipy.interpolate import interp1d


# ── 1. Strict Sanity & Leakage Gate ──────────────────────────────────────────
def verify_leakage_gate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    train_drive_names: List[str],
    test_drive_names: List[str],
    calibration_drive_names: Optional[List[str]] = None,
    train_driver_ids: Optional[List[str]] = None,
    test_driver_ids: Optional[List[str]] = None,
    is_lodro: bool = False
):
    """Automated assertion gate: Fails immediately if any data, target, or split leakage is detected."""
    verify_feature_matrix_leakage(X_train, context_name="X_train")
    verify_feature_matrix_leakage(X_test, context_name="X_test")
    verify_split_isolation(
        train_drive_names=train_drive_names,
        test_drive_names=test_drive_names,
        calibration_drive_names=calibration_drive_names,
        train_driver_ids=train_driver_ids,
        test_driver_ids=test_driver_ids,
        is_lodro=is_lodro
    )


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
    return compute_uncertainty_metrics(y_true, y_pred, y_std)


# ── 3. Level 4: Downstream Navigation Evaluator ──────────────────────────────
def evaluate_downstream_navigation(
    drive_dataset,
    t_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    blackout_start_sec: float = 60.0,
    blackout_end_sec: Optional[float] = None
) -> Dict[str, float]:
    """Level 4: 6-State EKF Downstream GNSS Blackout Navigation."""
    df = drive_dataset.get_data()
    d_id = drive_dataset.driver_id
    if blackout_end_sec is None:
        blackout_end_sec = min(drive_dataset.duration_sec - 10.0, 150.0)

    t_orig = df["timestamp"].values
    n = len(df)

    # Dense interpolation of AI speed and uncertainty
    interp_func = interp1d(t_test, y_pred, kind="linear", bounds_error=False, fill_value=(y_pred[0], y_pred[-1]))
    v_dense = np.maximum(0.0, interp_func(t_orig))

    interp_std = interp1d(t_test, y_std, kind="linear", bounds_error=False, fill_value=(y_std[0], y_std[-1]))
    std_dense = np.maximum(0.05, interp_std(t_orig))

    init_heading = df["heading"].iloc[0] if "heading" in df.columns else 0.0
    init_speed = df["speed"].iloc[0] if "speed" in df.columns else 0.0
    init_pos = (df["pos_x"].iloc[0], df["pos_y"].iloc[0]) if "pos_x" in df.columns else (0.0, 0.0)

    # Reconstruct pure AI-DR trajectory
    dt_arr = np.diff(t_orig, prepend=t_orig[0])
    dt_arr[0] = dt_arr[1] if n > 1 else 0.1
    gyro_z = df["gyro_z"].values

    heading_ai = np.zeros(n)
    pos_x_ai = np.zeros(n)
    pos_y_ai = np.zeros(n)
    heading_ai[0] = init_heading
    pos_x_ai[0] = init_pos[0]
    pos_y_ai[0] = init_pos[1]

    for i in range(1, n):
        dt = dt_arr[i]
        heading_ai[i] = heading_ai[i-1] + gyro_z[i] * dt
        v_mid = 0.5 * (v_dense[i-1] + v_dense[i])
        h_mid = 0.5 * (heading_ai[i-1] + heading_ai[i])
        pos_x_ai[i] = pos_x_ai[i-1] + v_mid * np.sin(h_mid) * dt
        pos_y_ai[i] = pos_y_ai[i-1] + v_mid * np.cos(h_mid) * dt

    ai_dr_res = pd.DataFrame({
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
        ai_dr_res["ai_pos_error_m"] = np.sqrt(dx**2 + dy**2)

    naive_res = NaiveDeadReckoning(init_heading, init_speed, init_pos).compute(df)

    fused_res = run_6state_ekf_fusion(
        df=df,
        ai_speed=v_dense,
        ai_speed_std=std_dense,
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
