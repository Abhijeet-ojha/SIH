"""
tests/test_6state_ekf_fusion.py
Unit tests for 6-State EKF Fusion, Covariance Properties, NHC, ZUPT, and GPS Reacquisition.
"""

import unittest
import numpy as np
from core.fusion.ekf_6state import KinematicFusionEKF6State, wrap_angle


class Test6StateEKFFusion(unittest.TestCase):

    def setUp(self):
        self.ekf = KinematicFusionEKF6State(
            init_x=0.0,
            init_y=0.0,
            init_v=10.0,
            init_v_lat=0.0,
            init_heading=0.0,
            init_gyro_bias=0.0,
            driver_style="normal"
        )

    def test_state_dimensions_and_symmetry(self):
        """Tests that state is 6x1 and covariance P is strictly symmetric and positive semi-definite."""
        self.assertEqual(len(self.ekf.x), 6)
        self.assertEqual(self.ekf.P.shape, (6, 6))

        # Propagate forward 10 steps
        for _ in range(10):
            self.ekf.predict(dt=0.10, v_ai=10.0, v_ai_std=0.20, gyro_z=0.02)
            self.ekf.update_nhc()

        # Check Symmetry: P == P^T
        np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-8)
        # Check Positive Semi-Definiteness: Eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(self.ekf.P)
        self.assertTrue(np.all(eigenvalues >= 0.0), f"Negative eigenvalue found: {eigenvalues}")

    def test_nhc_constrains_lateral_velocity(self):
        """Tests that NHC update strongly damps lateral velocity slip."""
        # Inject lateral slip
        self.ekf.x[3] = 3.0  # 3 m/s lateral velocity
        self.ekf.update_nhc()
        self.assertLess(abs(self.ekf.x[3]), 0.6)

    def test_zupt_at_standstill(self):
        """Tests that Zero-Velocity Update (ZUPT) clamps velocity states during red light / stop."""
        self.ekf.x[2] = 2.0  # forward vel residual
        self.ekf.x[3] = 1.0  # lateral vel residual
        self.ekf.update_zupt()
        self.assertLess(abs(self.ekf.x[2]), 0.5)
        self.assertLess(abs(self.ekf.x[3]), 0.5)

    def test_gps_reacquisition_resets_drift(self):
        """Tests that healthy GPS measurement corrects accumulated drift."""
        # Introduce fictitious position drift
        self.ekf.x[0] = 50.0
        self.ekf.x[1] = 80.0
        self.ekf.P[0, 0] = 100.0
        self.ekf.P[1, 1] = 100.0

        # High-accuracy GPS measurement at (5.0, 5.0)
        self.ekf.update_gps(gps_x=5.0, gps_y=5.0, gps_speed=10.0, gps_heading=0.0)
        self.assertLess(abs(self.ekf.x[0] - 5.0), 5.0)
        self.assertLess(abs(self.ekf.x[1] - 5.0), 5.0)


if __name__ == "__main__":
    unittest.main()
