"""
tests/test_canonical_interfaces.py
Unit tests for platform-independent Canonical Sensor, Motion, and Navigation interfaces.
"""

import unittest
import numpy as np
from core.interfaces.canonical import (
    SensorFrame,
    MotionEstimate,
    NavigationState,
    SensorQualityReport,
    CoordinateFrame,
    MotionRegime
)


class TestCanonicalInterfaces(unittest.TestCase):

    def test_sensor_frame_instantiation_and_units(self):
        """Tests that SensorFrame correctly stores 3-axis IMU and converts seconds to nanoseconds."""
        acc = np.array([0.1, 9.81, 0.05], dtype=float)
        gyro = np.array([0.01, -0.02, 0.005], dtype=float)
        frame = SensorFrame(
            timestamp_s=123.456,
            accel_xyz=acc,
            gyro_xyz=gyro,
            ambient_lux=1500.0,
            frame_type=CoordinateFrame.DEVICE_FRAME
        )
        self.assertEqual(frame.timestamp_ns, 123456000000)
        self.assertEqual(frame.frame_type, CoordinateFrame.DEVICE_FRAME)
        self.assertEqual(frame.ambient_lux, 1500.0)
        np.testing.assert_array_equal(frame.accel_xyz, acc)
        np.testing.assert_array_equal(frame.gyro_xyz, gyro)

    def test_motion_estimate_contract(self):
        """Tests MotionEstimate data contracts with forward speed and 3D velocity vector."""
        est = MotionEstimate(
            timestamp_s=10.0,
            velocity_fwd_mps=14.5,
            velocity_3d_mps=np.array([14.5, 0.2, 0.0]),
            uncertainty_sigma_mps=0.18,
            conformal_interval_mps=(13.8, 15.2),
            motion_regime=MotionRegime.CRUISING
        )
        self.assertEqual(est.velocity_fwd_mps, 14.5)
        self.assertEqual(est.motion_regime, MotionRegime.CRUISING)
        self.assertAlmostEqual(est.uncertainty_sigma_mps, 0.18)
        self.assertEqual(est.conformal_interval_mps, (13.8, 15.2))

    def test_navigation_state_6state(self):
        """Tests NavigationState 6-state vector and covariance matrix dimensions."""
        nav = NavigationState(
            timestamp_s=15.0,
            position_enu=np.array([120.5, 340.2]),
            velocity_body=np.array([12.0, 0.01]),
            heading_rad=1.57,
            gyro_bias_rad_s=0.001,
            covariance_matrix=np.eye(6)
        )
        self.assertEqual(len(nav.position_enu), 2)
        self.assertEqual(len(nav.velocity_body), 2)
        self.assertEqual(nav.covariance_matrix.shape, (6, 6))

    def test_sensor_quality_report(self):
        """Tests SensorQualityReport reliability criteria."""
        report_ok = SensorQualityReport(timestamp_s=1.0, jitter_ms=2.0, quality_score=0.95)
        self.assertTrue(report_ok.is_reliable(threshold=0.60))

        report_bad = SensorQualityReport(timestamp_s=1.0, is_saturated=True, quality_score=0.40)
        self.assertFalse(report_bad.is_reliable(threshold=0.60))


if __name__ == "__main__":
    unittest.main()
