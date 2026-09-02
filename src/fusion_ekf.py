"""
src/fusion_ekf.py
Compatibility Bridge delegating to core/fusion/ekf_6state.py.
Confidence-Aware 6-State Kinematic Extended Kalman Filter (EKF) with Driver-Style Adaptive Constraints
and Multi-Sensor Predictive Context Layer.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List

from core.fusion.ekf_6state import (
    KinematicFusionEKF6State,
    MultiSensorContextEngine,
    run_6state_ekf_fusion,
    wrap_angle
)

# Aliases for backward compatibility
VehicleContextEngine = MultiSensorContextEngine
KinematicFusionEKF = KinematicFusionEKF6State


def run_fusion_pipeline(
    df: pd.DataFrame,
    ai_speed: np.ndarray,
    ai_speed_std: Optional[np.ndarray] = None,
    driver_style: str = "normal",
    blackout_start_sec: float = 60.0,
    blackout_end_sec: float = 150.0
) -> pd.DataFrame:
    """Executes the Confidence-Aware 6-State EKF with Multi-Sensor Context Engine and Adaptive NHC."""
    return run_6state_ekf_fusion(
        df=df,
        ai_speed=ai_speed,
        ai_speed_std=ai_speed_std,
        driver_style=driver_style,
        blackout_start_sec=blackout_start_sec,
        blackout_end_sec=blackout_end_sec
    )
