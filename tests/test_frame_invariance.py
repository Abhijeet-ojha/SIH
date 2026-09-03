"""
tests/test_frame_invariance.py
The gate for Phase 2. Rotating the phone must not change what the model sees.

The pipeline used to feed raw ax/ay/az/gx/gy/gz statistics to the regressor, which encodes
how the phone happens to be sitting rather than how the vehicle is moving. That is why
held-out-driver R2 went negative: a different car and a different mount produce a different
mapping, and the model had learned the mount.

This test applies 20 random fixed rotations to the raw IMU and requires the extracted
features to be unchanged. Against the old feature set it fails catastrophically.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_real_iovnbd_drive
from src.feature_engineering import extract_causal_window_features
from src.frame_alignment import align_frame

SAMPLE = os.path.join(PROJECT_ROOT, "data", "samples", "drive_03_test_urban.csv")
N_ROTATIONS = 20
FEATURE_TOL = 1e-6
MAE_TOL_PCT = 0.02


def random_rotation(rng):
    """Uniform random rotation matrix via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class TestFrameInvariance(unittest.TestCase):

    # Feature extraction runs ~12 ms/window, so the full 3000-sample drive costs ~25 s per
    # rotation. Invariance is a per-window algebraic property - it does not need the whole
    # drive to show up - so the feature-level tests use a slice and the cheap
    # channel-level test uses everything.
    FEATURE_SLICE = 700

    @classmethod
    def setUpClass(cls):
        cls.drive = load_real_iovnbd_drive(SAMPLE, driver_id="A")
        cls.df = cls.drive.get_data()
        cls.df_small = cls.df.iloc[:cls.FEATURE_SLICE].reset_index(drop=True)
        cls.rng = np.random.default_rng(11)

    def _rotate_df(self, R, src=None):
        src = self.df if src is None else src
        df = src.copy()
        acc = np.column_stack([src.acc_x, src.acc_y, src.acc_z]) @ R.T
        gyro = np.column_stack([src.gyro_x, src.gyro_y, src.gyro_z]) @ R.T
        df["acc_x"], df["acc_y"], df["acc_z"] = acc[:, 0], acc[:, 1], acc[:, 2]
        df["gyro_x"], df["gyro_y"], df["gyro_z"] = gyro[:, 0], gyro[:, 1], gyro[:, 2]
        return df

    def test_no_raw_body_axis_features_survive(self):
        """Any feature named after a phone axis is by definition mount-dependent."""
        X, _, _, _ = extract_causal_window_features(self.df_small)
        offenders = [c for c in X.columns
                     if c.split("_")[0] in {"ax", "ay", "az", "gx", "gy", "gz"}]
        self.assertEqual(offenders, [], f"frame-dependent features still present: {offenders}")

    def test_alignment_channels_are_invariant(self):
        dt = self.df["dt"].values
        acc = np.column_stack([self.df.acc_x, self.df.acc_y, self.df.acc_z])
        gyro = np.column_stack([self.df.gyro_x, self.df.gyro_y, self.df.gyro_z])
        base = align_frame(acc, gyro, dt, speed=self.df.speed.values)

        for i in range(N_ROTATIONS):
            R = random_rotation(self.rng)
            rot = align_frame(acc @ R.T, gyro @ R.T, dt, speed=self.df.speed.values)
            for key in ["a_vert", "a_horiz_mag", "yaw_rate", "gyro_mag", "tilt_rate",
                        "grav_stability", "a_fwd", "a_lat"]:
                err = float(np.max(np.abs(base[key] - rot[key])))
                self.assertLess(err, FEATURE_TOL,
                                f"rotation {i}: channel {key} moved by {err:.3e}")

    def test_extracted_features_are_invariant(self):
        X0, y0, _, _ = extract_causal_window_features(self.df_small)
        for i in range(N_ROTATIONS):
            R = random_rotation(self.rng)
            Xr, yr, _, _ = extract_causal_window_features(self._rotate_df(R, self.df_small))
            self.assertEqual(list(X0.columns), list(Xr.columns))
            delta = np.max(np.abs(X0.values - Xr.values), axis=0)
            worst = int(np.argmax(delta))
            self.assertLess(float(delta[worst]), FEATURE_TOL,
                            f"rotation {i}: feature '{X0.columns[worst]}' moved by {delta[worst]:.3e}")
            np.testing.assert_allclose(y0, yr, atol=1e-12)

    def test_end_to_end_speed_mae_is_stable_under_rotation(self):
        """
        A ridge fit on the unrotated features, scored on rotated ones. If the features are
        genuinely invariant the MAE cannot move; this catches invariance that holds
        per-channel but breaks once the model is in the loop.
        """
        from sklearn.linear_model import Ridge

        X0, y0, _, _ = extract_causal_window_features(self.df_small)
        model = Ridge(alpha=1.0).fit(X0.values, y0)
        mae0 = float(np.mean(np.abs(model.predict(X0.values) - y0)))

        for i in range(5):
            R = random_rotation(self.rng)
            Xr, yr, _, _ = extract_causal_window_features(self._rotate_df(R, self.df_small))
            mae = float(np.mean(np.abs(model.predict(Xr.values) - yr)))
            rel = abs(mae - mae0) / max(mae0, 1e-9)
            self.assertLess(rel, MAE_TOL_PCT,
                            f"rotation {i}: MAE moved {rel:.2%} ({mae0:.4f} -> {mae:.4f})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
