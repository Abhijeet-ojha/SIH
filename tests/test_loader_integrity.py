"""
tests/test_loader_integrity.py
The gate for Phase 0. Everything downstream is meaningless if these fail.

The loader used to clamp every dt <= 0 or > 1.0 s to 0.1 s and cumsum the result, which
manufactured a uniform 10 Hz timeline. The tell is in the repo's own audit output:
duration_s came out as exactly samples * 0.1 for every single drive. Any real interval of
1 s or more was compressed 10x, which is where the 17-33 m/s^2 implied accelerations came
from - 2.5 g in a passenger car.
"""

import os
import sys
import glob
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_real_iovnbd_drive, DriveDataset, _path_length

SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
SPEED_INTEGRAL_TOL = 0.05     # integral(v dt) vs GPS path length
# A passenger car lives inside ~8 m/s^2. But this is measured on GPS-derived speed, which
# carries ~0.1 m/s of noise; differenced at 10 Hz that alone contributes several m/s^2 of
# spikes. So: a loose cap on the peak, and a tight cap on the 99th percentile, which noise
# tails cannot reach. Both together still catch the original defect by a wide margin - the
# fabricated 10 Hz timeline produced 17-33 m/s^2 *sustained*, not as isolated spikes.
MAX_IMPLIED_ACCEL = 12.0      # peak, noise included
MAX_IMPLIED_ACCEL_P99 = 8.0   # 99th percentile: the actual vehicle limit


def _sample_drives():
    return sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.csv")))


class TestLoaderIntegrity(unittest.TestCase):

    def test_timeline_is_not_fabricated(self):
        """
        A real log has jitter. If duration == samples * median_dt to machine precision,
        the timestamps were synthesised rather than read.
        """
        t = np.array([0.0, 0.1, 0.25, 0.31, 1.9, 2.0, 2.11, 2.2])  # irregular, one big gap
        df = pd.DataFrame({
            "timestamp": t,
            "acc_x": np.zeros(8), "acc_y": np.zeros(8), "acc_z": np.full(8, 9.80665),
            "gyro_x": np.zeros(8), "gyro_y": np.zeros(8), "gyro_z": np.zeros(8),
            "gt_lat": np.full(8, 28.6), "gt_lon": np.full(8, 77.2), "speed": np.zeros(8),
        })
        d = DriveDataset(df, name="irregular")
        out = d.get_data()

        np.testing.assert_allclose(out["timestamp"].values, t, atol=1e-12,
                                   err_msg="loader did not preserve the real timestamps")
        self.assertAlmostEqual(d.duration_sec, 2.2, places=9)
        # The 1.59 s gap must be recorded, not rewritten to 0.1 s.
        self.assertEqual(d.n_gaps, 1, "logging gap was silently absorbed")
        self.assertGreater(out["dt"].max(), 1.0, "the large interval was clamped away")

    def test_non_monotonic_rows_are_dropped_not_invented(self):
        t = np.array([0.0, 0.1, 0.05, 0.2, 0.3])  # one backwards sample
        df = pd.DataFrame({
            "timestamp": t,
            "acc_x": np.arange(5.0), "acc_y": np.zeros(5), "acc_z": np.full(5, 9.80665),
            "gyro_x": np.zeros(5), "gyro_y": np.zeros(5), "gyro_z": np.zeros(5),
            "gt_lat": np.full(5, 28.6), "gt_lon": np.full(5, 77.2), "speed": np.zeros(5),
        })
        d = DriveDataset(df, name="nonmono")
        self.assertEqual(d.n_dropped_nonmonotonic, 1)
        self.assertEqual(d.num_samples, 4)
        self.assertTrue(np.all(np.diff(d.get_data()["timestamp"].values) > 0))
        # Column alignment must survive the drop.
        self.assertEqual(list(d.get_data()["acc_x"].values), [0.0, 1.0, 3.0, 4.0])

    def test_speed_units_agree_with_distance_travelled(self):
        """
        integral(v dt) must equal the GPS path length. This settles the /3.6 question with
        physics rather than trusting a column header, and the two candidates differ by
        3.6x so they are trivially separable.
        """
        for path in _sample_drives():
            with self.subTest(drive=os.path.basename(path)):
                d = load_real_iovnbd_drive(path, driver_id="A")
                df = d.get_data()
                integ = float(np.sum(df["speed"].values * df["dt"].values))
                path_len = _path_length(df["pos_x"].values, df["pos_y"].values)
                rel = abs(integ - path_len) / max(path_len, 1e-6)
                self.assertLess(
                    rel, SPEED_INTEGRAL_TOL,
                    f"speed integral {integ:.1f} m vs path {path_len:.1f} m "
                    f"({rel:.1%}); chosen unit {d.speed_unit}, evidence {d.speed_unit_evidence}"
                )

    def test_implied_acceleration_is_physical(self):
        """
        A compressed time axis shows up as sustained impossible acceleration. Difference a
        lightly median-filtered speed so that GPS speed noise - which is real, and produces
        isolated spikes of several m/s^2 on its own - does not masquerade as one.
        """
        for path in _sample_drives():
            with self.subTest(drive=os.path.basename(path)):
                d = load_real_iovnbd_drive(path, driver_id="A")
                df = d.get_data()
                v = pd.Series(df["speed"].values).rolling(3, center=True, min_periods=1).median().values
                dt = np.maximum(df["dt"].values, 1e-6)
                accel = np.abs(np.diff(v) / dt[1:])
                self.assertLess(float(np.max(accel)), MAX_IMPLIED_ACCEL,
                                f"peak implied accel {np.max(accel):.1f} m/s^2 is not a car")
                p99 = float(np.percentile(accel, 99))
                self.assertLess(p99, MAX_IMPLIED_ACCEL_P99,
                                f"p99 implied accel {p99:.1f} m/s^2 exceeds the vehicle limit")

    def test_acc_y_is_not_derived_from_the_label(self):
        """
        The old fallback set acc_y = np.gradient(speed), deriving a feature from the
        target. If that ever fired, every metric in the repo was the model reading its own
        answer key.
        """
        for path in _sample_drives():
            with self.subTest(drive=os.path.basename(path)):
                df = load_real_iovnbd_drive(path, driver_id="A").get_data()
                dspeed = np.gradient(df["speed"].values)
                if np.std(dspeed) < 1e-9 or np.std(df["acc_y"].values) < 1e-9:
                    continue
                r = abs(float(np.corrcoef(df["acc_y"].values, dspeed)[0, 1]))
                self.assertLess(r, 0.99, f"acc_y correlates {r:.4f} with d(speed) - it is the label")

    def test_missing_imu_raises_instead_of_fabricating(self):
        df = pd.DataFrame({
            "timestamp": np.arange(5) * 0.1,
            "gt_lat": np.full(5, 28.6), "gt_lon": np.full(5, 77.2), "speed": np.zeros(5),
        })
        with self.assertRaises(ValueError):
            DriveDataset(df, name="no_imu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
