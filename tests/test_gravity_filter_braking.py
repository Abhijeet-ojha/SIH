"""
tests/test_gravity_filter_braking.py

PHASE 4: the gravity estimator must not eat sustained braking.

The original 0.2 Hz low-pass had a ~0.8 s time constant. A first-order filter has one time
constant serving two conflicting jobs - reject linear acceleration (wants slow) and track
real tilt (wants fast) - and at 0.2 Hz it resolved that by absorbing anything sustained
beyond about a second into its own "gravity" estimate. A 4 s brake therefore came out as
near-zero horizontal specific force, which is how naive_dr's alignment search originally
mistook a steady -1.0 m/s^2 decel for a standstill.

This test finds real hard-brake segments in IO-VNBD by the vehicle's own CAN longitudinal
accelerometer, then requires the phone-derived horizontal acceleration to still see them.
"""

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.frame_alignment import align_frame
from src.iovnbd_loader import dataset_root, discover_pairs, load_pair

HAVE_DATA = os.path.isdir(dataset_root())

BRAKE_THRESHOLD = 2.0     # m/s^2 of CAN-measured deceleration to count as "braking"
MIN_BRAKE_SAMPLES = 15    # 1.5 s - long enough for the low-pass to have absorbed it
# Measured across 32 brake events on 5 real drives: the 0.2 Hz low-pass retains a median
# 15% of the vehicle-measured deceleration, the Mahony filter 51% - a 3.5x improvement.
# Per-drive spread is wide (0.05 to 1.70), because retention also depends on how well the
# forward axis is estimated on that drive, so the floor is set below the measured median
# and the test aggregates across drives rather than trusting any single one.
RETENTION_FLOOR = 0.35
DRIVES = ["S3a", "S3c", "M", "Vfa02", "Vw14c"]


def brake_segments(can_long_accel, min_len=MIN_BRAKE_SAMPLES, thresh=BRAKE_THRESHOLD):
    """Contiguous runs where the vehicle reports sustained deceleration."""
    mask = can_long_accel < -thresh
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


@unittest.skipUnless(HAVE_DATA, "IO-VNBD not present; run scripts/fetch_iovnbd.py")
class TestGravityFilterUnderBraking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        by_stem = {p["stem"]: p for p in discover_pairs()}
        cls.frames, cls.total_segs = [], 0
        for stem in DRIVES:
            if stem not in by_stem:
                continue
            d = load_pair(by_stem[stem], max_samples=24000, strict=False)
            df = d.get_data()
            segs = brake_segments(df["can_long_accel"].values)
            if not segs:
                continue
            cls.total_segs += len(segs)
            cls.frames.append({
                "stem": stem, "df": df, "segs": segs,
                "acc": np.column_stack([df.acc_x, df.acc_y, df.acc_z]),
                "gyro": np.column_stack([df.gyro_x, df.gyro_y, df.gyro_z]),
                "can": df["can_long_accel"].values,
            })

    def test_real_brake_segments_exist(self):
        self.assertGreater(self.total_segs, 20,
                           f"only {self.total_segs} sustained brake events across "
                           f"{len(self.frames)} drives - too few to conclude anything")

    def _retention(self, mode):
        """|a_fwd| during braking as a fraction of the vehicle's own measured decel."""
        ratios = []
        for f in self.frames:
            fr = align_frame(f["acc"], f["gyro"], f["df"]["dt"].values,
                             speed=f["df"]["speed"].values, gravity_mode=mode)
            for a, b in f["segs"]:
                can_mag = float(np.median(np.abs(f["can"][a:b])))
                if can_mag > 1e-6:
                    ratios.append(float(np.median(np.abs(fr["a_fwd"][a:b]))) / can_mag)
        return np.array(ratios), None

    def test_mahony_retains_sustained_deceleration(self):
        ratios, _ = self._retention("mahony")
        med = float(np.median(ratios))
        print(f"\n  mahony : retains {med:.0%} of CAN-measured decel over "
              f"{self.total_segs} brake events on {len(self.frames)} drives "
              f"(p25 {np.percentile(ratios, 25):.0%})")
        self.assertGreater(med, RETENTION_FLOOR,
                           f"gyro-propagated gravity still swallowed braking "
                           f"(retained {med:.0%})")

    def test_mahony_beats_the_lowpass_it_replaces(self):
        """The A/B the flag exists for. Old implementation stays reachable."""
        r_mah, _ = self._retention("mahony")
        r_lp, _ = self._retention("lowpass")
        m_mah, m_lp = float(np.median(r_mah)), float(np.median(r_lp))
        print(f"  lowpass: retains {m_lp:.0%}   ->   mahony {m_mah:.0%} "
              f"({m_mah / max(m_lp, 1e-9):.1f}x better)")
        self.assertGreater(m_mah, m_lp,
                           "the replacement must actually beat the filter it replaces")

    def test_gravity_magnitude_stays_physical(self):
        """A propagated estimate that drifts off 9.81 would corrupt every projection."""
        f = self.frames[0]
        fr = align_frame(f["acc"], f["gyro"], f["df"]["dt"].values, gravity_mode="mahony")
        mag = np.linalg.norm(fr["gravity"], axis=1)
        self.assertLess(float(np.max(np.abs(mag - 9.80665))), 1e-6,
                        "Mahony gravity magnitude drifted off g")


if __name__ == "__main__":
    unittest.main(verbosity=2)
