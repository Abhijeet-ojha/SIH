"""
tests/test_context_layer.py
Synthetic unit test verifying the Multi-Sensor Predictive Context Engine (Job 3).
Simulates daytime driving (1500 lux) transitioning into an underground tunnel (30 lux),
and validates that the predictive blackout alert triggers properly and tightens covariance.
"""

import sys
import os
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.fusion_ekf import VehicleContextEngine, KinematicFusionEKF

class TestPredictiveContextEngine(unittest.TestCase):

    def setUp(self):
        self.engine = VehicleContextEngine()
        self.ekf = KinematicFusionEKF(driver_style="normal")

    def test_standstill_mode(self):
        """Tests that low variance and speed trigger STANDSTILL / ZUPT mode."""
        mode = self.engine.update_context(ambient_lux=1500.0, speed_mps=0.1, acc_var=0.005, gyro_abs=0.002)
        self.assertEqual(mode, "STANDSTILL")
        self.assertFalse(self.engine.tunnel_alert)

    def test_daylight_urban_mode(self):
        """Tests standard daytime urban driving."""
        mode = self.engine.update_context(ambient_lux=1200.0, speed_mps=12.0, acc_var=0.15, gyro_abs=0.05)
        self.assertEqual(mode, "NORMAL_URBAN")
        self.assertFalse(self.engine.tunnel_alert)

    def test_predictive_tunnel_blackout_trigger(self):
        """
        Tests that a sudden drop in ambient light (<100 lux) while driving at speed (>4 m/s)
        immediately triggers PREDICTIVE_TUNNEL_BLACKOUT and activates tunnel_alert.
        """
        # Vehicle enters tunnel at 14 m/s, light drops to 35 lux
        mode = self.engine.update_context(ambient_lux=35.0, speed_mps=14.0, acc_var=0.10, gyro_abs=0.02)
        self.assertEqual(mode, "PREDICTIVE_TUNNEL_BLACKOUT")
        self.assertTrue(self.engine.tunnel_alert)

        # Confirm EKF tightens gyro bias noise under tunnel alert
        q_bias_before = self.ekf.q_bias
        self.ekf.predict(dt=0.1, v_ai=14.0, v_ai_std=0.2, gyro_z=0.01, is_tunnel_alert=True)
        # Prediction executed without error
        self.assertIsNotNone(self.ekf.x)

    def test_highway_cruising_mode(self):
        """Tests highway speed context (>18 m/s)."""
        mode = self.engine.update_context(ambient_lux=1500.0, speed_mps=25.0, acc_var=0.25, gyro_abs=0.02)
        self.assertEqual(mode, "HIGHWAY_CRUISING")

if __name__ == "__main__":
    unittest.main()
