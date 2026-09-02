"""
tests/test_causal_pipeline.py
Unit tests verifying strict causality and automated target leakage guards.
"""

import unittest
import numpy as np
import pandas as pd
from core.features.extractor import CausalFeatureExtractor
from core.features.leakage_guard import (
    verify_feature_matrix_leakage,
    verify_split_isolation,
    verify_causality,
    LeakageGuardError
)


class TestCausalPipelineAndLeakage(unittest.TestCase):

    def setUp(self):
        # Create synthetic IMU dataframe (10 seconds at 10 Hz = 100 samples)
        t = np.linspace(0.0, 9.9, 100)
        self.df = pd.DataFrame({
            "timestamp": t,
            "acc_x": np.sin(t) * 0.5,
            "acc_y": 1.2 + 0.1 * np.cos(t),
            "acc_z": 9.81 + 0.05 * np.random.randn(100),
            "gyro_x": 0.01 * np.random.randn(100),
            "gyro_y": 0.01 * np.random.randn(100),
            "gyro_z": 0.05 * np.sin(0.5 * t),
            "speed": 10.0 + 2.0 * np.sin(0.2 * t)
        })

    def test_causality_trailing_edge(self):
        """Tests that every extracted feature window uses only past and current samples."""
        extractor = CausalFeatureExtractor(window_sec=1.5, step_sec=0.2, sample_rate_hz=10.0)
        X_df, y_speed, t_pred, latency_info = extractor.extract_features(self.df)

        self.assertGreater(len(X_df), 0)
        self.assertEqual(len(X_df), len(y_speed))
        self.assertEqual(len(X_df), len(t_pred))

        # First prediction timestamp must be at the end of the first window (1.5s -> index 14 -> t=1.4s)
        self.assertAlmostEqual(t_pred[0], self.df["timestamp"].iloc[14], places=4)
        # All timestamps must be strictly monotonic
        self.assertTrue(np.all(np.diff(t_pred) > 0))

    def test_target_leakage_rejection(self):
        """Tests that presence of ground-truth columns raises LeakageGuardError."""
        bad_df = pd.DataFrame({
            "acc_x_mean": [0.1, 0.2],
            "speed": [12.0, 14.0]  # LEAKAGE!
        })
        with self.assertRaises(LeakageGuardError):
            verify_feature_matrix_leakage(bad_df, context_name="bad_test_df")

    def test_split_isolation_rejection(self):
        """Tests that overlapping drives or drivers in train/test trigger immediate failure."""
        train_drives = ["drive_01", "drive_02", "drive_03"]
        test_drives = ["drive_03", "drive_04"]  # Overlap drive_03!
        with self.assertRaises(LeakageGuardError):
            verify_split_isolation(train_drives, test_drives)

        # LODrO driver overlap
        with self.assertRaises(LeakageGuardError):
            verify_split_isolation(
                ["d1", "d2"], ["d3"],
                train_driver_ids=["A", "B"],
                test_driver_ids=["B", "C"],
                is_lodro=True
            )


if __name__ == "__main__":
    unittest.main()
