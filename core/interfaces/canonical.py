"""
core/interfaces/canonical.py
Platform-Independent Canonical Sensor, Motion, and Navigation Data Contracts.

Authoritative Unit Conventions:
  - Timestamps: Nanoseconds (int) or Seconds (float) - canonical representations provide both
  - Accelerations: m/s^2 (SI standard)
  - Angular Velocities: rad/s (SI standard)
  - Magnetic Field: microTesla (uT)
  - Pressure: hectoPascals (hPa)
  - Angles / Heading: Radians (rad) in [-pi, +pi], 0 = East, +pi/2 = North (ENU) or 0 = North, +pi/2 = East
  - Velocities: m/s
  - Positions: meters (ENU local tangent plane: x=East, y=North, z=Up)
  - Covariances: SI units squared (m^2, (m/s)^2, rad^2, (rad/s)^2)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


class CoordinateFrame(str, Enum):
    """Authoritative Coordinate Conventions."""
    DEVICE_FRAME = "DEVICE_FRAME"       # Raw sensor axes (phone/wearable/drone body)
    VEHICLE_BODY = "VEHICLE_BODY"       # x=forward, y=lateral (left/right), z=vertical (up)
    ENU = "ENU"                         # Local tangent plane: x=East, y=North, z=Up
    NED = "NED"                         # Local tangent plane: x=North, y=East, z=Down


class MotionRegime(str, Enum):
    """Physical Motion Regimes."""
    STANDSTILL = "STANDSTILL"
    LOW_SPEED = "LOW_SPEED"             # < 3 m/s
    CRUISING = "CRUISING"               # 3 - 18 m/s
    HIGHWAY = "HIGHWAY"                 # > 18 m/s
    ACCELERATING = "ACCELERATING"
    BRAKING = "BRAKING"
    TURNING = "TURNING"
    AGGRESSIVE = "AGGRESSIVE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SensorQualityReport:
    """
    Diagnostic report on sensor stream integrity and physical plausibility.
    Scores range from 0.0 (unusable / severe faults) to 1.0 (nominal, high-fidelity).
    """
    timestamp_s: float
    jitter_ms: float = 0.0
    packet_loss_ratio: float = 0.0
    dropped_samples_count: int = 0
    is_saturated: bool = False
    noise_variance: float = 0.0
    stationary_detected: bool = False
    quality_score: float = 1.0          # in [0.0, 1.0]
    fault_flags: List[str] = field(default_factory=list)

    def is_reliable(self, threshold: float = 0.6) -> bool:
        return self.quality_score >= threshold


@dataclass
class SensorFrame:
    """
    Platform-independent canonical multi-modal sensor measurement frame.
    All device adapters normalize their raw hardware inputs into this structure.
    """
    timestamp_s: float                  # Epoch or relative seconds
    timestamp_ns: int = 0               # High-precision nanoseconds
    
    # 3-Axis IMU (Required for inertial navigation)
    accel_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))      # [ax, ay, az] in m/s^2
    gyro_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))       # [gx, gy, gz] in rad/s
    
    # Optional Exteroceptive & Environmental Sensors
    mag_xyz: Optional[np.ndarray] = None                                                 # [mx, my, mz] in uT
    gravity_xyz: Optional[np.ndarray] = None                                             # Estimated or hardware gravity [gx, gy, gz] in m/s^2
    orientation_quat: Optional[np.ndarray] = None                                        # [qw, qx, qy, qz] quaternion
    ambient_lux: Optional[float] = None                                                  # Illuminance in Lux
    pressure_hpa: Optional[float] = None                                                 # Atmospheric pressure in hPa
    
    # Metadata & Quality
    frame_type: CoordinateFrame = CoordinateFrame.DEVICE_FRAME
    quality: SensorQualityReport = field(default_factory=lambda: SensorQualityReport(timestamp_s=0.0))

    def __post_init__(self):
        if self.timestamp_ns == 0 and self.timestamp_s > 0:
            self.timestamp_ns = int(self.timestamp_s * 1e9)
        if isinstance(self.accel_xyz, list):
            self.accel_xyz = np.array(self.accel_xyz, dtype=float)
        if isinstance(self.gyro_xyz, list):
            self.gyro_xyz = np.array(self.gyro_xyz, dtype=float)


@dataclass
class MotionEstimate:
    """
    Platform-independent motion representation produced by ML motion estimators.
    Supports 1D forward speed (automotive), 2D planar velocity, and full 3D body velocity (drones/robotics).
    """
    timestamp_s: float
    timestamp_ns: int = 0
    
    # Velocity representations
    velocity_fwd_mps: float = 0.0                                                        # Forward scalar speed (m/s) >= 0
    velocity_3d_mps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float)) # [vx, vy, vz] in body or ENU frame
    
    # Calibrated Uncertainty Quantification
    uncertainty_sigma_mps: float = 0.20                                                  # Standard deviation (1-sigma) for EKF process noise
    conformal_interval_mps: Tuple[float, float] = (0.0, 0.0)                             # [y_low, y_high] coverage interval
    conformal_coverage_level: float = 0.90                                               # e.g., 0.90 (90%)
    
    # Contextual Semantics
    motion_regime: MotionRegime = MotionRegime.UNKNOWN
    confidence: float = 1.0                                                              # in [0.0, 1.0]
    feature_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0


@dataclass
class NavigationState:
    """
    Full 6-DOF / 6-State Local Tangent Plane Navigation State produced by the Fusion Engine.
    State Vector convention: x = [pos_East, pos_North, v_fwd, v_lat, heading_yaw, gyro_bias]^T
    """
    timestamp_s: float
    timestamp_ns: int = 0
    
    # Kinematic States
    position_enu: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))   # [p_East, p_North] in meters
    velocity_body: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))  # [v_fwd, v_lat] in m/s
    heading_rad: float = 0.0                                                             # Yaw angle in radians [-pi, pi]
    gyro_bias_rad_s: float = 0.0                                                         # Estimated z-gyro bias in rad/s
    
    # Estimation Covariance (6x6)
    covariance_matrix: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=float))
    
    # System Status
    is_gnss_denied: bool = False
    context_mode: str = "NORMAL_URBAN"
    sensor_quality_score: float = 1.0
    
    # Diagnostics & Extensible Attributes
    position_std_m: Tuple[float, float] = (0.0, 0.0)                                     # (std_East, std_North)
    fused_speed_mps: float = 0.0
