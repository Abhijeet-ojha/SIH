"""
core/preprocessing/quality.py
Deterministic Sensor Quality and Signal Reliability Estimator.
Analyzes timestamp jitter, packet drops, signal clipping/saturation, and IMU stationary noise floors.
Produces an observable SensorQualityReport with an explainable quality_score in [0.0, 1.0].
"""

import numpy as np
from typing import List, Dict, Optional
from core.interfaces.canonical import SensorQualityReport, SensorFrame


class SensorQualityMonitor:
    """
    Observable Sensor Quality Monitor.
    Evaluates streaming and batch sensor frames for physical plausibility and temporal integrity.
    """
    def __init__(
        self,
        nominal_sample_rate_hz: float = 10.0,
        accel_saturation_thresh_mps2: float = 38.0,  # ~4g standard mobile limit
        gyro_saturation_thresh_rads: float = 34.0,   # ~2000 deg/s standard mobile limit
        max_acceptable_jitter_ms: float = 50.0
    ):
        self.nominal_rate = nominal_sample_rate_hz
        self.nominal_dt = 1.0 / nominal_sample_rate_hz
        self.accel_sat = accel_saturation_thresh_mps2
        self.gyro_sat = gyro_saturation_thresh_rads
        self.max_jitter_ms = max_acceptable_jitter_ms
        
        # State tracking for streaming checks
        self.last_timestamp_s: Optional[float] = None
        self.dt_history: List[float] = []

    def evaluate_frame(
        self,
        frame: SensorFrame,
        recent_frames: Optional[List[SensorFrame]] = None
    ) -> SensorQualityReport:
        """Evaluates a single frame and returns its SensorQualityReport."""
        fault_flags = []
        t = frame.timestamp_s
        
        # 1. Temporal Jitter and Drop Assessment
        jitter_ms = 0.0
        dropped_count = 0
        if self.last_timestamp_s is not None:
            actual_dt = t - self.last_timestamp_s
            if actual_dt <= 0.0:
                fault_flags.append("NON_MONOTONIC_OR_DUPLICATE_TIMESTAMP")
                jitter_ms = 999.0
            else:
                jitter_ms = abs(actual_dt - self.nominal_dt) * 1000.0
                if actual_dt > 2.5 * self.nominal_dt:
                    dropped_count = int(round(actual_dt / self.nominal_dt)) - 1
                    fault_flags.append(f"PACKET_LOSS_DETECTED_{dropped_count}_SAMPLES")

        self.last_timestamp_s = t

        # 2. Saturation / Clipping Check
        is_saturated = False
        accel_norm = float(np.linalg.norm(frame.accel_xyz))
        gyro_norm = float(np.linalg.norm(frame.gyro_xyz))
        if accel_norm >= self.accel_sat or np.any(np.abs(frame.accel_xyz) >= self.accel_sat):
            is_saturated = True
            fault_flags.append("ACCEL_SENSOR_SATURATION")
        if gyro_norm >= self.gyro_sat or np.any(np.abs(frame.gyro_xyz) >= self.gyro_sat):
            is_saturated = True
            fault_flags.append("GYRO_SENSOR_SATURATION")

        # 3. Noise Variance and Standstill Check across recent window
        noise_var = 0.0
        is_stationary = False
        if recent_frames and len(recent_frames) >= 5:
            accels = np.array([f.accel_xyz for f in recent_frames[-10:]])
            gyros = np.array([f.gyro_xyz for f in recent_frames[-10:]])
            acc_var = float(np.var(accels, axis=0).sum())
            gyro_var = float(np.var(gyros, axis=0).sum())
            noise_var = acc_var + gyro_var
            if acc_var < 0.025 and gyro_var < 0.005:
                is_stationary = True

        # 4. Deterministic Explainable Quality Score Calculation
        # Quality penalizations:
        # - Jitter penalty: up to 0.30
        # - Packet drop penalty: up to 0.40
        # - Saturation penalty: 0.50
        # - Non-finite penalty: 1.0 (score 0.0)
        if np.isnan(frame.accel_xyz).any() or np.isnan(frame.gyro_xyz).any():
            return SensorQualityReport(
                timestamp_s=t,
                jitter_ms=999.0,
                is_saturated=True,
                quality_score=0.0,
                fault_flags=["NON_FINITE_SENSOR_VALUES"]
            )

        jitter_penalty = min(0.30, (jitter_ms / max(1.0, self.max_jitter_ms)) * 0.30)
        drop_penalty = min(0.40, dropped_count * 0.15)
        sat_penalty = 0.50 if is_saturated else 0.0
        
        quality_score = max(0.0, 1.0 - (jitter_penalty + drop_penalty + sat_penalty))

        return SensorQualityReport(
            timestamp_s=t,
            jitter_ms=jitter_ms,
            packet_loss_ratio=min(1.0, dropped_count / 10.0),
            dropped_samples_count=dropped_count,
            is_saturated=is_saturated,
            noise_variance=noise_var,
            stationary_detected=is_stationary,
            quality_score=float(quality_score),
            fault_flags=fault_flags
        )
