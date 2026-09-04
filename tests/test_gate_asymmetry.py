"""
tests/test_gate_asymmetry.py

PHASE 5: the motion gate must fail toward "moving" while GNSS is down.

The two failure directions cost very different amounts:

  false "stopped" during a blackout   position freezes while the car keeps going. Every
                                      metre becomes along-track error, nothing observes it
                                      until GNSS returns, and map matching cannot recover
                                      it (map matching only fixes cross-track).
  false "stopped" with GNSS up        corrected on the next fix. Nearly free.

So the gate uses one set of thresholds per regime, and the metrics count the failure
explicitly rather than letting it hide inside a position error.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.frame_alignment import G0, align_frame
from src.fusion_ekf import run_fusion_pipeline
from src.motion_gate import MOVING, STATIONARY, GateThresholds, MotionGate

DT = 0.1
N = 900        # 90 s


def _marginal_drive(rng, n=N):
    """
    A vehicle moving steadily but quietly - the ambiguous case the two regimes must treat
    differently. Horizontal specific force sits between the blackout threshold (0.06) and
    the GNSS-available threshold (0.12), so the regime decides the answer.
    """
    t = np.arange(n) * DT
    acc = np.column_stack([
        rng.normal(0, 0.09, n),
        rng.normal(0, 0.09, n),
        np.full(n, G0) + rng.normal(0, 0.09, n),
    ])
    gyro = np.column_stack([rng.normal(0, 0.002, n)] * 2 + [rng.normal(0, 0.002, n)])
    return acc, gyro, t


class TestGateAsymmetry(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(23)
        self.acc, self.gyro, self.t = _marginal_drive(self.rng)
        self.dt = np.full(len(self.t), DT)
        self.fr = align_frame(self.acc, self.gyro, self.dt)

    def test_thresholds_are_actually_asymmetric(self):
        th = GateThresholds()
        self.assertLess(th.blackout_still_acc_rms, th.still_acc_rms)
        self.assertLess(th.blackout_still_yaw_rate, th.still_yaw_rate)
        self.assertGreater(th.blackout_grav_stability_max, th.grav_stability_max)
        self.assertGreater(th.blackout_debounce_frames, th.debounce_frames)

    def test_blackout_regime_is_more_reluctant_to_declare_a_stop(self):
        gate = MotionGate()
        with_gnss = gate.classify_frame(self.fr, gnss_available=np.ones(len(self.t), bool))
        in_blackout = gate.classify_frame(self.fr, gnss_available=np.zeros(len(self.t), bool))

        stops_gnss = int(np.sum(with_gnss["state"] == STATIONARY))
        stops_blackout = int(np.sum(in_blackout["state"] == STATIONARY))
        print(f"\n  marginal drive: STATIONARY samples with GNSS = {stops_gnss}, "
              f"during blackout = {stops_blackout}")
        self.assertLess(stops_blackout, stops_gnss,
                        "the blackout regime must be strictly harder to stop in; "
                        f"got {stops_blackout} vs {stops_gnss}")
        self.assertGreater(
            int(np.sum(in_blackout["state"] == MOVING)),
            int(np.sum(with_gnss["state"] == MOVING)),
            "during a blackout the gate should resolve ambiguity toward MOVING")

    def test_omitting_gnss_mask_defaults_to_the_permissive_regime(self):
        """Callers that do not pass the mask must not silently get blackout behaviour."""
        gate = MotionGate()
        default = gate.classify_frame(self.fr)
        explicit = gate.classify_frame(self.fr, gnss_available=np.ones(len(self.t), bool))
        self.assertTrue(np.array_equal(default["state"], explicit["state"]))

    def test_false_stationary_during_blackout_is_measured(self):
        """
        The metric exists and is non-zero exactly when the filter freezes on a car that is
        genuinely moving. Without this, the failure only shows up as position error, well
        after it has cost metres.
        """
        n = 600
        t = np.arange(n) * DT
        # Genuinely moving at 12 m/s throughout, but the IMU is almost silent - a smooth
        # motorway cruise, which is precisely when a stillness test is most tempted to fire.
        df = pd.DataFrame({
            "timestamp": t, "dt": np.full(n, DT), "gap_mask": False,
            "acc_x": self.rng.normal(0, 0.01, n), "acc_y": self.rng.normal(0, 0.01, n),
            "acc_z": np.full(n, G0) + self.rng.normal(0, 0.01, n),
            "gyro_x": np.zeros(n), "gyro_y": np.zeros(n), "gyro_z": np.zeros(n),
            "speed": np.full(n, 12.0),
            "pos_x": np.zeros(n), "pos_y": 12.0 * t,
            "heading": np.zeros(n), "ambient_lux": np.nan, "step_events": 0.0,
            "driver_id": "A",
        })
        res = run_fusion_pipeline(df, df["speed"].values, np.full(n, 0.2),
                                  driver_style="A",
                                  blackout_start_sec=20.0, blackout_end_sec=50.0)
        self.assertIn("false_stationary", res.columns)
        self.assertIn("false_stationary_blackout_sec", res.attrs)
        self.assertIn("false_stationary_blackout_m", res.attrs)

        secs = res.attrs["false_stationary_blackout_sec"]
        metres = res.attrs["false_stationary_blackout_m"]
        print(f"  silent 12 m/s cruise through a 30 s blackout: "
              f"false-stationary {secs:.1f} s / {metres:.1f} m")
        # The metric must be self-consistent: distance is speed times the frozen time.
        self.assertAlmostEqual(metres, secs * 12.0, delta=1.0)
        # And the asymmetric thresholds should keep it small on this case.
        self.assertLess(secs, 10.0,
                        f"gate froze for {secs:.1f} s of a 30 s blackout on a moving car")


if __name__ == "__main__":
    unittest.main(verbosity=2)
