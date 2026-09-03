"""
tests/test_kotlin_parity.py
The gate for Phase 5. The phone must run what was benchmarked.

Honest scope statement, because this matters more than the test passing:

  * WHAT THIS VERIFIES: the exported model evaluates identically to sklearn; the golden
    vectors are self-consistent; and the Kotlin engine's feature vector is the same 16
    quantities in the same order as the Python exporter, checked by parsing the Kotlin
    source. Feature reordering is by far the most likely way parity breaks silently, and
    this catches it.

  * WHAT THIS DOES NOT VERIFY: that compiled Kotlin produces the same floating-point
    numbers. There is no Kotlin compiler in this environment. Running the engine against
    outputs/models/golden_vectors.json on a device or a JVM with kotlinc is still
    outstanding. Until someone does that, the README may not claim verified on-device
    parity - only that the model and the pipeline have been ported.
"""

import os
import re
import sys
import json
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.export_ondevice_model import ONDEVICE_FEATURES, eval_trees, compact_features
from src.data_loader import load_real_iovnbd_drive

MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "ondevice_model.json")
GOLDEN_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "golden_vectors.json")
KOTLIN_PATH = os.path.join(
    PROJECT_ROOT, "android_logger", "app", "src", "main", "java", "com", "sih",
    "sensorlogger", "OnDeviceInferenceEngine.kt"
)

# Kotlin ring-buffer name -> Python channel name
BUFFERS = {
    "aFwd": "a_fwd", "aLat": "a_lat", "aVert": "a_vert", "aHorizMag": "a_horiz_mag",
    "yawRate": "yaw_rate", "gyroMag": "gyro_mag", "tiltRate": "tilt_rate",
    "gravStab": "grav_stab",
}
REDUCERS = {"mean": "mean", "std": "std", "rms": "rms", "absMean": "absmean"}


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_kotlin_features():
    """Extract the feature vector Kotlin builds, in order, from buildFeatures()."""
    src = _read(KOTLIN_PATH)
    m = re.search(r"buildFeatures\(\)\s*:\s*DoubleArray\s*=\s*doubleArrayOf\((.*?)\n\s*\)",
                  src, re.S)
    assert m, "buildFeatures() not found in the Kotlin engine"
    body = re.sub(r"//.*", "", m.group(1))

    names, depth, cur = [], 0, ""
    for chunk in body:
        if chunk == "(":
            depth += 1
        elif chunk == ")":
            depth -= 1
        if chunk == "," and depth == 0:
            names.append(cur.strip())
            cur = ""
        else:
            cur += chunk
    if cur.strip():
        names.append(cur.strip())

    out = []
    for expr in names:
        expr = expr.strip()
        if "*" in expr:
            # The one composite: std(aVert) * rms(aHorizMag) == road_vibration.
            out.append("road_vibration" if expr == "std(aVert) * rms(aHorizMag)" else f"?{expr}")
            continue
        mm = re.fullmatch(r"(\w+)\((\w+)\)", expr)
        if not mm or mm.group(1) not in REDUCERS or mm.group(2) not in BUFFERS:
            out.append(f"?{expr}")
            continue
        out.append(f"{BUFFERS[mm.group(2)]}_{REDUCERS[mm.group(1)]}")
    return out


class TestOnDeviceParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for p in (MODEL_PATH, GOLDEN_PATH):
            if not os.path.exists(p):
                raise unittest.SkipTest(f"{p} missing - run scripts/export_ondevice_model.py")
        cls.model = _read_json(MODEL_PATH)
        cls.golden = _read_json(GOLDEN_PATH)

    def test_exported_model_is_deployable_size(self):
        kb = os.path.getsize(MODEL_PATH) / 1024.0
        self.assertLess(kb, 500.0,
                        f"{kb:.0f} KB; the 13.4 MB Random Forest was never shippable")

    def test_kotlin_feature_vector_matches_exporter(self):
        kotlin = parse_kotlin_features()
        unresolved = [f for f in kotlin if f.startswith("?")]
        self.assertEqual(unresolved, [], f"unrecognised Kotlin feature expressions: {unresolved}")
        self.assertEqual(
            kotlin, ONDEVICE_FEATURES,
            "Kotlin buildFeatures() and ONDEVICE_FEATURES disagree.\n"
            f"  kotlin: {kotlin}\n  python: {ONDEVICE_FEATURES}"
        )

    def test_kotlin_gate_thresholds_match_python(self):
        """Drift between the two gate implementations would be invisible at runtime."""
        from src.motion_gate import GateThresholds
        src = _read(KOTLIN_PATH)
        th = GateThresholds()
        for kt_name, py_val in [
            ("GRAV_STABILITY_MAX", th.grav_stability_max),
            ("TILT_RATE_MAX", th.tilt_rate_max),
            ("STILL_ACC_RMS", th.still_acc_rms),
            ("STILL_YAW_RATE", th.still_yaw_rate),
            ("DEBOUNCE_FRAMES", float(th.debounce_frames)),
        ]:
            m = re.search(rf"const val {kt_name}\s*=\s*([0-9.]+)", src)
            self.assertIsNotNone(m, f"{kt_name} not found in Kotlin")
            self.assertAlmostEqual(float(m.group(1)), py_val, places=6,
                                   msg=f"{kt_name}: Kotlin {m.group(1)} vs Python {py_val}")

    def test_model_feature_order_is_recorded(self):
        self.assertEqual(self.model["features"], ONDEVICE_FEATURES)

    def test_golden_vectors_reproduce(self):
        """Re-derive the golden features from the raw IMU and confirm they still match."""
        drive = load_real_iovnbd_drive(
            os.path.join(PROJECT_ROOT, "data", "samples", self.golden["source_drive"]),
            driver_id="A")
        df = drive.get_data().iloc[:self.golden["n_samples"]].reset_index(drop=True)
        X, _, idx = compact_features(df)

        np.testing.assert_array_equal(idx, np.array(self.golden["window_index"]))
        np.testing.assert_allclose(X, np.array(self.golden["features"]), atol=1e-12,
                                   err_msg="feature extraction drifted from the golden vectors")

        preds = np.array([eval_trees(self.model, r) for r in X])
        np.testing.assert_allclose(preds, np.array(self.golden["predicted_speed"]), atol=1e-12)

    def test_no_hardcoded_speed_constants_remain(self):
        """The original two-constant speed law must be gone, not merely bypassed."""
        src = _read(KOTLIN_PATH)
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("*"))
        self.assertNotIn("* 1.85", code)
        self.assertNotIn("* 4.20", code)
        self.assertNotIn("0.85 * statePosX", code, "fixed-gain GPS blend still present")


if __name__ == "__main__":
    unittest.main(verbosity=2)
