"""
tests/test_shake_rejection.py
The gate for Phase 3. Shaking a stationary phone must not produce travel.

Before the frame alignment and motion gate landed, the on-device speed estimate was
max(0, (mean|a| - g)*1.85 + std(ay)*4.20) - two terms that both rise monotonically with
vibration - so agitating a stationary phone integrated into hundreds of metres of
imaginary forward motion. These cases are the regression guard.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.frame_alignment import align_frame, G0
from src.motion_gate import MotionGate, MOVING

DT = 0.1
N = 600  # 60 s
MAX_DISPLACEMENT_M = 2.0


def _integrate(fr, gate_result, dt=DT):
    """
    Integrate the gated speed estimate the way the EKF does: when the gate says we are not
    in a moving vehicle, velocity is held at zero and position does not advance.
    """
    moving = gate_result["in_vehicle_moving"]
    # Deliberately the OLD, broken speed law. If the gate works, it never gets to run.
    acc_mag_mean = np.abs(fr["a_vert"]) + fr["a_horiz_mag"] + G0
    naive_speed = np.maximum(0.0, (acc_mag_mean - G0) * 1.85)
    v = np.where(moving, naive_speed, 0.0)
    return float(np.sum(v * dt))


class TestShakeRejection(unittest.TestCase):

    def setUp(self):
        self.t = np.arange(N) * DT
        self.dt = np.full(N, DT)
        self.gate = MotionGate()

    def _run(self, acc, gyro):
        fr = align_frame(acc, gyro, self.dt)
        res = self.gate.classify_frame(fr)
        return fr, res

    def test_sinusoidal_shake(self):
        """1-10 Hz hand shake at 5 m/s^2 on a stationary phone."""
        for f_hz in (1.0, 3.0, 5.0, 8.0, 10.0):
            s = 5.0 * np.sin(2 * np.pi * f_hz * self.t)
            acc = np.column_stack([s, 0.6 * s, np.full(N, G0) + 0.4 * s])
            # A hand cannot translate without also rotating.
            gyro = np.column_stack([1.5 * np.sin(2 * np.pi * f_hz * self.t),
                                    1.1 * np.cos(2 * np.pi * f_hz * self.t),
                                    0.4 * np.sin(2 * np.pi * f_hz * self.t)])
            fr, res = self._run(acc, gyro)
            disp = _integrate(fr, res)
            self.assertLess(disp, MAX_DISPLACEMENT_M,
                            f"{f_hz} Hz shake produced {disp:.1f} m of travel")

    def test_random_walk_hand_motion(self):
        """Non-periodic handling: someone fidgeting with the phone."""
        rng = np.random.default_rng(7)
        walk = np.cumsum(rng.normal(0, 0.8, (N, 3)), axis=0)
        walk -= walk.mean(axis=0)
        acc = np.column_stack([walk[:, 0], walk[:, 1], np.full(N, G0) + walk[:, 2]])
        gyro = np.cumsum(rng.normal(0, 0.15, (N, 3)), axis=0)
        gyro -= gyro.mean(axis=0)
        fr, res = self._run(acc, gyro)
        disp = _integrate(fr, res)
        self.assertLess(disp, MAX_DISPLACEMENT_M, f"random handling produced {disp:.1f} m")

    def test_pickup_and_rotate_180(self):
        """Phone lifted out of the cradle and turned over."""
        acc = np.tile([0.0, 0.0, G0], (N, 1)).astype(float)
        gyro = np.zeros((N, 3))
        # Flip about body x over 2 s, starting at t = 20 s.
        s, e = 200, 220
        ang = np.linspace(0, np.pi, e - s)
        rate = np.pi / ((e - s) * DT)
        gyro[s:e, 0] = rate
        acc[s:e] = np.column_stack([np.zeros(e - s), G0 * np.sin(ang), G0 * np.cos(ang)])
        acc[e:] = np.array([0.0, 0.0, -G0])
        fr, res = self._run(acc, gyro)
        disp = _integrate(fr, res)
        self.assertLess(disp, MAX_DISPLACEMENT_M, f"pickup+rotate produced {disp:.1f} m")
        # The manoeuvre itself must be detected, not merely survived.
        during = res["state"][s:e + 30]
        self.assertTrue(np.any(during != MOVING), "180 deg flip was not detected at all")

    def test_genuine_driving_is_not_rejected(self):
        """False-positive budget: the gate must not veto real motion. Target < 2%."""
        rng = np.random.default_rng(3)
        acc = np.column_stack([
            rng.normal(0, 0.15, N),
            0.8 * np.sin(2 * np.pi * 0.05 * self.t) + rng.normal(0, 0.15, N),
            np.full(N, G0) + rng.normal(0, 0.2, N),
        ])
        gyro = np.column_stack([rng.normal(0, 0.01, N), rng.normal(0, 0.01, N),
                                np.full(N, 0.12) + rng.normal(0, 0.01, N)])
        fr, res = self._run(acc, gyro)
        fp = 1.0 - float(np.mean(res["in_vehicle_moving"]))
        self.assertLess(fp, 0.02, f"gate rejected {fp:.1%} of genuine driving")


if __name__ == "__main__":
    unittest.main(verbosity=2)
