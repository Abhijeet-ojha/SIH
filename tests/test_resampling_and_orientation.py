"""
tests/test_resampling_and_orientation.py
Unit tests for sampling-rate agnostic resampling and orientation tracking.
"""

import unittest
import numpy as np
import pandas as pd
from core.preprocessing.resampler import CausalSensorResampler
from core.preprocessing.quality import SensorQualityMonitor
from core.orientation.tracker import OrientationTracker
from core.interfaces.canonical import SensorFrame, CoordinateFrame


class TestResamplingAndOrientation(unittest.TestCase):

    def test_multi_rate_resampling(self):
        """Tests resampling raw irregular timestamps to 10Hz, 20Hz, 50Hz grids."""
        # 5 seconds of raw data with slight jitter ~ 15 Hz
        raw_t = np.sort(np.random.uniform(0.0, 5.0, 75))
        raw_df = pd.DataFrame({
            "timestamp": raw_t,
            "acc_x": np.sin(raw_t),
            "acc_y": np.cos(raw_t),
            "acc_z": 9.81 * np.ones(len(raw_t)),
            "gyro_x": np.zeros(len(raw_t)),
            "gyro_y": np.zeros(len(raw_t)),
            "gyro_z": 0.1 * np.ones(len(raw_t)),
            "speed": 5.0 * raw_t
        })

        for target_hz in [10.0, 20.0, 50.0]:
            resampler = CausalSensorResampler(target_rate_hz=target_hz)
            frames, speeds, grid_t = resampler.resample_dataframe(raw_df)
            
            self.assertGreater(len(frames), 10)
            self.assertEqual(len(frames), len(grid_t))
            self.assertIsNotNone(speeds)
            # Grid spacing should match 1 / target_hz
            dt_grid = np.diff(grid_t)
            np.testing.assert_allclose(dt_grid, 1.0 / target_hz, rtol=1e-5)

    def test_orientation_gravity_tracking(self):
        """Tests that OrientationTracker isolates gravity and body accelerations."""
        tracker = OrientationTracker()
        # Feed nominal stationary upright phone: az = 9.81 m/s^2
        for _ in range(15):
            frame = SensorFrame(
                timestamp_s=0.1,
                accel_xyz=np.array([0.0, 0.0, 9.81]),
                gyro_xyz=np.zeros(3)
            )
            out = tracker.update(frame, dt=0.10)

        # In steady state, linear acceleration magnitude should be near 0
        self.assertLess(out["linear_acc_mag"], 0.2)
        self.assertGreaterEqual(out["orientation_confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
