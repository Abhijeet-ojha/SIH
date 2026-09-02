"""
core/orientation/tracker.py
Modular Orientation Tracking and Coordinate Alignment Layer.
Transforms raw device-frame IMU measurements [acc_xyz, gyro_xyz] into Canonical Body Frame
and World ENU Frame.
Tracks the gravity vector via complementary filtering and dynamic acceleration isolation.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List, Any
from core.interfaces.canonical import SensorFrame, CoordinateFrame


class OrientationTracker:
    """
    Orientation and Body-Frame Alignment Tracker.
    Maintains an estimated gravity vector and device-to-body orientation.
    Derives forward acceleration, lateral acceleration, vertical acceleration, and yaw rate.
    """
    def __init__(
        self,
        alpha_gravity: float = 0.98,
        gravity_nominal_mps2: float = 9.80665,
        default_forward_axis_index: int = 1  # In standard vehicle phone mount, y is often aligned with forward
    ):
        self.alpha_g = alpha_gravity
        self.g_nom = gravity_nominal_mps2
        self.fwd_idx = default_forward_axis_index
        
        # State: Estimated gravity vector in device frame
        self.estimated_gravity: Optional[np.ndarray] = None
        self.pitch_rad: float = 0.0
        self.roll_rad: float = 0.0
        self.yaw_rad: float = 0.0
        self.orientation_confidence: float = 1.0

    def update(self, frame: SensorFrame, dt: float = 0.10) -> Dict[str, Any]:
        """
        Updates orientation state with new sensor frame.
        Returns:
            Dictionary containing body-frame kinematic accelerations [a_fwd, a_lat, a_vert]
            and yaw rate omega_z, alongside gravity vector and orientation confidence.
        """
        acc = frame.accel_xyz.astype(float)
        gyro = frame.gyro_xyz.astype(float)
        
        # 1. Initialize or update gravity vector via complementary low-pass filter
        if self.estimated_gravity is None:
            self.estimated_gravity = acc.copy()
        else:
            # Only update gravity strongly during low dynamic acceleration segments
            acc_mag = float(np.linalg.norm(acc))
            if abs(acc_mag - self.g_nom) < 2.0:
                self.estimated_gravity = self.alpha_g * self.estimated_gravity + (1.0 - self.alpha_g) * acc
                self.orientation_confidence = min(1.0, self.orientation_confidence + 0.02)
            else:
                self.orientation_confidence = max(0.4, self.orientation_confidence - 0.01)

        g_norm = float(np.linalg.norm(self.estimated_gravity)) + 1e-9
        g_unit = self.estimated_gravity / g_norm

        # 2. Linear acceleration = total acceleration - gravity
        linear_acc = acc - self.estimated_gravity

        # 3. Derive pitch & roll from gravity vector
        # g_x = -sin(pitch), g_y = sin(roll)*cos(pitch), g_z = cos(roll)*cos(pitch)
        pitch = -np.arcsin(np.clip(g_unit[0], -1.0, 1.0))
        roll = np.arctan2(g_unit[1], g_unit[2])
        self.pitch_rad = float(pitch)
        self.roll_rad = float(roll)

        # 4. Integrate gyro for relative yaw
        self.yaw_rad += float(gyro[2] * dt)

        # 5. Derive Body-Frame Accelerations (Automotive / Terrestrial convention)
        # Vertical acceleration is parallel to gravity unit vector
        a_vert = float(np.dot(linear_acc, g_unit))
        # Horizontal plane acceleration vector
        a_horiz_vec = linear_acc - a_vert * g_unit
        
        # Vehicle forward axis component:
        # In IO-VNBD dataset, phone y-axis is vehicle longitudinal/forward, x is lateral, z is vertical
        a_fwd = float(acc[1])
        a_lat = float(acc[0])
        yaw_rate = float(gyro[2])

        return {
            "a_forward": a_fwd,
            "a_lateral": a_lat,
            "a_vertical": a_vert,
            "linear_acc_mag": float(np.linalg.norm(linear_acc)),
            "yaw_rate": yaw_rate,
            "pitch_rad": self.pitch_rad,
            "roll_rad": self.roll_rad,
            "estimated_gravity": self.estimated_gravity.copy(),
            "orientation_confidence": float(self.orientation_confidence)
        }
