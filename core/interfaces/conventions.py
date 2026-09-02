"""
core/interfaces/conventions.py
Authoritative system-wide specification for coordinate frames, unit scales,
orientation mathematics, and timestamp semantics.
"""

from enum import Enum
from typing import Dict, Any


class UnitSystem:
    """Explicit SI Base Units for all Core Modules."""
    TIME = "seconds (s) / nanoseconds (ns)"
    ACCELERATION = "meters per second squared (m/s^2)"
    ANGULAR_VELOCITY = "radians per second (rad/s)"
    MAGNETIC_FLUX_DENSITY = "microTesla (uT)"
    ATMOSPHERIC_PRESSURE = "hectoPascals (hPa)"
    ILLUMINANCE = "Lux (lx)"
    VELOCITY = "meters per second (m/s)"
    POSITION = "meters (m)"
    ANGLE = "radians (rad) [-pi, +pi]"


CONVENTION_SPECIFICATION: Dict[str, Any] = {
    "local_tangent_plane": {
        "convention": "ENU (East-North-Up)",
        "x_axis": "East (m)",
        "y_axis": "North (m)",
        "z_axis": "Up (m)",
        "azimuth_zero": "East = 0 rad, North = +pi/2 rad (counter-clockwise) or standard navigation course (North = 0 rad, East = +pi/2 rad)"
    },
    "vehicle_body_frame": {
        "convention": "ISO 8855 / SAE Forward-Left-Up or Forward-Right-Down",
        "x_forward": "Positive in direction of vehicle forward motion",
        "y_lateral": "Positive towards lateral side",
        "z_vertical": "Positive upwards"
    },
    "state_vector_6state": {
        "dimension": 6,
        "indices": {
            0: "p_East (m)",
            1: "p_North (m)",
            2: "v_forward (m/s) [vehicle body forward velocity]",
            3: "v_lateral (m/s) [vehicle body lateral slip velocity]",
            4: "heading_yaw (rad) [orientation in local ENU plane]",
            5: "gyro_bias (rad/s) [estimated yaw-gyroscope rate bias]"
        }
    },
    "causality_contract": {
        "rule": "Prediction at t must use strictly sensor samples in [t - W, t]. No access to t + dt."
    }
}
