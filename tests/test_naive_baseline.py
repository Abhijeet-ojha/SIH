"""
tests/test_naive_baseline.py
The gate for Phase 1. The baseline has to be honest or the comparison means nothing.

The old baseline integrated raw acc_y with no gravity or bias removal. A constant offset
integrates into a velocity ramp and then a quadratic position error, which is how a
1,221 m drive produced 19,338 m of "drift". That is not double-integration drift, it is an
uncorrected bias - and a 99.9% improvement quoted against it is a strawman.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.naive_dr import NaiveDeadReckoning

DT = 0.1
DURATION = 60.0
N = int(DURATION / DT)
GRAVITY_PROJECTION = 1.6  # m/s^2 of gravity leaking onto the "forward" axis via phone tilt


def _constant_velocity_drive(speed=12.0, gravity_leak=GRAVITY_PROJECTION):
    """
    Straight line at constant speed. True forward acceleration is zero, so anything the
    baseline reports is bias. acc_y carries a gravity projection because the phone is not
    perfectly level - which is the normal case, not an edge case.
    """
    t = np.arange(N) * DT
    return pd.DataFrame({
        "timestamp": t,
        "acc_x": np.zeros(N),
        "acc_y": np.full(N, gravity_leak),
        "acc_z": np.full(N, 9.80665),
        "gyro_x": np.zeros(N),
        "gyro_y": np.zeros(N),
        "gyro_z": np.zeros(N),
        "pos_x": np.zeros(N),
        "pos_y": speed * t,
        "speed": np.full(N, speed),
    })


class TestNaiveBaseline(unittest.TestCase):

    def setUp(self):
        self.df = _constant_velocity_drive()
        self.path_len = float(self.df["pos_y"].iloc[-1])

    def test_corrected_baseline_drifts_less_than_5pct(self):
        dr = NaiveDeadReckoning(initial_speed=12.0, remove_gravity=True)
        res = dr.compute(self.df)
        drift = float(res["pos_error_m"].iloc[-1])
        self.assertLess(drift / self.path_len, 0.05,
                        f"aligned baseline drifted {drift:.1f} m over {self.path_len:.0f} m")
        self.assertAlmostEqual(dr.acc_bias, GRAVITY_PROJECTION, places=6,
                               msg="initial alignment did not recover the bias")

    def test_uncorrected_baseline_is_the_strawman(self):
        """The old behaviour must still be reproducible, and must still be catastrophic."""
        dr = NaiveDeadReckoning(initial_speed=12.0, remove_gravity=False)
        res = dr.compute(self.df)
        drift = float(res["pos_error_m"].iloc[-1])
        self.assertGreater(drift / self.path_len, 1.0,
                           "uncorrected integration should diverge; if it does not, "
                           "the test data no longer contains a bias to expose it")

    def test_correction_is_worth_orders_of_magnitude(self):
        good = NaiveDeadReckoning(initial_speed=12.0, remove_gravity=True).compute(self.df)
        bad = NaiveDeadReckoning(initial_speed=12.0, remove_gravity=False).compute(self.df)
        ratio = float(bad["pos_error_m"].iloc[-1]) / max(float(good["pos_error_m"].iloc[-1]), 1e-6)
        self.assertGreater(ratio, 20.0,
                           "gravity removal should dominate; any 'improvement' headline "
                           "measured against the uncorrected version is measuring this")

    def test_gyro_bias_is_removed_too(self):
        """A gyro offset rotates the whole trajectory; alignment must catch it."""
        df = _constant_velocity_drive()
        df["gyro_z"] = 0.01  # ~0.6 deg/s, a normal MEMS offset
        dr = NaiveDeadReckoning(initial_speed=12.0, remove_gravity=True)
        res = dr.compute(df)
        self.assertAlmostEqual(dr.gyro_bias, 0.01, places=6)
        self.assertLess(abs(float(res["naive_heading"].iloc[-1])), 0.05,
                        "heading ramped despite alignment")


if __name__ == "__main__":
    unittest.main(verbosity=2)
