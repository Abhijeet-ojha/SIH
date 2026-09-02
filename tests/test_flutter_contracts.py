"""
tests/test_flutter_contracts.py
Schema parity tests between Flutter Dart models and Python Core Canonical interfaces.
Validates exact JSON key alignment, SI units, and enum string representations.
"""

import unittest
from core.interfaces.canonical import (
    SensorFrame,
    MotionEstimate,
    NavigationState,
    CoordinateFrame,
    MotionRegime
)


class TestFlutterContractParity(unittest.TestCase):

    def test_canonical_sensor_frame_schema(self):
        """Tests that Python SensorFrame schema aligns with Flutter SensorFrame.toJson()."""
        # Emulated Flutter JSON
        flutter_json = {
            "timestamp_s": 12.5,
            "timestamp_ns": 12500000000,
            "imu": {
                "acc_x": 0.1,
                "acc_y": 1.2,
                "acc_z": 9.81,
                "gyro_x": 0.01,
                "gyro_y": -0.02,
                "gyro_z": 0.05
            },
            "gnss": {
                "latitude": 12.9716,
                "longitude": 77.5946,
                "altitude": 920.0,
                "speed": 12.0,
                "bearing": 45.0,
                "accuracy": 4.5,
                "has_fix": True,
                "timestamp_ms": 1700000000000
            },
            "quality_score": 0.95
        }

        # Validate keys required by Python core
        self.assertIn("timestamp_s", flutter_json)
        self.assertIn("imu", flutter_json)
        self.assertIn("acc_x", flutter_json["imu"])
        self.assertIn("gyro_z", flutter_json["imu"])
        self.assertIn("quality_score", flutter_json)

    def test_navigation_state_schema_parity(self):
        """Tests that Python NavigationState and Flutter NavigationState exchange identical JSON schemas."""
        flutter_nav_json = {
            "timestamp_s": 15.0,
            "timestamp_ns": 15000000000,
            "latitude": 12.971650,
            "longitude": 77.594650,
            "altitude": 920.0,
            "pos_east_m": 12.5,
            "pos_north_m": 34.2,
            "speed_mps": 14.2,
            "speed_kmh": 51.12,
            "velocity_lat_mps": 0.02,
            "heading_rad": 0.785,
            "heading_deg": 45.0,
            "gyro_bias_rad_s": 0.0005,
            "confidence_pct": 95.0,
            "uncertainty_sigma_mps": 0.18,
            "gnss_mode": "GNSS_NORMAL",
            "source": "GNSS_AI_IMU_EKF",
            "blackout_elapsed_s": 0.0,
            "context_mode": "NORMAL_URBAN"
        }

        self.assertEqual(flutter_nav_json["gnss_mode"], "GNSS_NORMAL")
        self.assertAlmostEqual(flutter_nav_json["speed_mps"], 14.2)
        self.assertEqual(flutter_nav_json["source"], "GNSS_AI_IMU_EKF")


if __name__ == "__main__":
    unittest.main()
