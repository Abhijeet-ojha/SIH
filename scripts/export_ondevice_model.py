"""
scripts/export_ondevice_model.py

Trains the model the phone will actually run, and exports it plus golden vectors so the
Kotlin port can be checked for parity.

Background: outputs/models/embedded_rules.json contained feature *names* and nothing else -
no weights - while OnDeviceInferenceEngine.kt computed speed as
    max(0, (mean|a| - 9.80665) * 1.85 + std(ay) * 4.20)
i.e. two hardcoded constants. The README claimed on-device ML. This closes that gap.

Two deliberate scope choices:
  * A compact 16-feature set, not the 201 the offline pipeline uses. The phone recomputes
    these from a ring buffer every window; FFT sub-band powers over twelve channels are not
    worth the battery, and a feature the Kotlin cannot reproduce exactly is a feature that
    breaks parity.
  * Gradient boosting with shallow trees, exported as plain JSON arrays. Tree evaluation is
    ~25 lines of Kotlin and reproduces Python bit-for-bit, which a quantised blob would not.
    The 13.4 MB Random Forest was never deployable anyway.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import GradientBoostingRegressor
from src.data_loader import load_real_iovnbd_drive
from src.frame_alignment import align_frame

MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")

WINDOW_SEC = 1.5
ONDEVICE_FEATURES = [
    "a_fwd_mean", "a_fwd_std", "a_fwd_rms",
    "a_lat_std", "a_lat_rms",
    "a_vert_std", "a_vert_rms",
    "a_horiz_mag_mean", "a_horiz_mag_std", "a_horiz_mag_rms",
    "yaw_rate_absmean", "yaw_rate_std",
    "gyro_mag_mean", "tilt_rate_rms", "grav_stab_mean",
    "road_vibration",
]


def compact_features(df: pd.DataFrame, window_sec: float = WINDOW_SEC):
    """
    The exact 16 features the Kotlin engine computes, over the same trailing window.
    Causal: window is [i-w+1, i] and the target is the speed at i.
    """
    dt = df["dt"].values if "dt" in df.columns else np.gradient(df["timestamp"].values)
    med_dt = float(np.median(dt[dt > 0]))
    w = max(4, int(round(window_sec / med_dt)))

    acc = np.column_stack([df.acc_x, df.acc_y, df.acc_z]).astype(float)
    gyro = np.column_stack([df.gyro_x, df.gyro_y, df.gyro_z]).astype(float)
    speed = df["speed"].values if "speed" in df.columns else None
    fr = align_frame(acc, gyro, dt, speed=speed)

    ch = {
        "a_fwd": fr["a_fwd"], "a_lat": fr["a_lat"], "a_vert": fr["a_vert"],
        "a_horiz_mag": fr["a_horiz_mag"], "yaw_rate": fr["yaw_rate"],
        "gyro_mag": fr["gyro_mag"], "tilt_rate": fr["tilt_rate"],
        "grav_stab": fr["grav_stability"],
    }

    rows, idx = [], []
    for i in range(w - 1, len(df)):
        s = i - w + 1
        r = {}
        for name in ("a_fwd", "a_lat", "a_vert", "a_horiz_mag"):
            win = ch[name][s:i + 1]
            r[f"{name}_mean"] = float(np.mean(win))
            r[f"{name}_std"] = float(np.std(win))
            r[f"{name}_rms"] = float(np.sqrt(np.mean(win**2)))
        yw = ch["yaw_rate"][s:i + 1]
        r["yaw_rate_absmean"] = float(np.mean(np.abs(yw)))
        r["yaw_rate_std"] = float(np.std(yw))
        r["gyro_mag_mean"] = float(np.mean(ch["gyro_mag"][s:i + 1]))
        r["tilt_rate_rms"] = float(np.sqrt(np.mean(ch["tilt_rate"][s:i + 1]**2)))
        r["grav_stab_mean"] = float(np.mean(ch["grav_stab"][s:i + 1]))
        r["road_vibration"] = r["a_vert_std"] * r["a_horiz_mag_rms"]
        rows.append([r[k] for k in ONDEVICE_FEATURES])
        idx.append(i)

    X = np.array(rows, dtype=float)
    y = df["speed"].values[idx] if "speed" in df.columns else np.zeros(len(idx))
    return X, y, np.array(idx)


def export_tree(t) -> dict:
    """sklearn tree -> plain arrays. -1 in `feature` marks a leaf."""
    tr = t.tree_
    return {
        "feature": tr.feature.tolist(),
        "threshold": tr.threshold.tolist(),
        "left": tr.children_left.tolist(),
        "right": tr.children_right.tolist(),
        "value": tr.value[:, 0, 0].tolist(),
    }


def eval_trees(model_json: dict, x: np.ndarray) -> float:
    """Reference evaluator. The Kotlin must reproduce this exactly."""
    out = model_json["init"]
    for tree in model_json["trees"]:
        node = 0
        while tree["feature"][node] != -2:
            f = tree["feature"][node]
            node = tree["left"][node] if x[f] <= tree["threshold"][node] else tree["right"][node]
        out += model_json["learning_rate"] * tree["value"][node]
    return out


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_files = sorted(f for f in os.listdir(SAMPLES_DIR) if "train" in f)
    test_files = sorted(f for f in os.listdir(SAMPLES_DIR) if "test" in f)
    if not train_files:
        raise SystemExit(f"no training drives in {SAMPLES_DIR}")

    Xs, ys = [], []
    for f in train_files:
        d = load_real_iovnbd_drive(os.path.join(SAMPLES_DIR, f), driver_id="A")
        X, y, _ = compact_features(d.get_data())
        Xs.append(X)
        ys.append(y)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    print(f"[train] {X.shape[0]} windows x {X.shape[1]} features from {len(train_files)} drives")

    model = GradientBoostingRegressor(
        n_estimators=60, max_depth=3, learning_rate=0.1, random_state=42
    ).fit(X, y)

    payload = {
        "model_type": "gradient_boosting_regressor",
        "features": ONDEVICE_FEATURES,
        "window_sec": WINDOW_SEC,
        "init": float(model.init_.constant_[0][0]),
        "learning_rate": float(model.learning_rate),
        "trees": [export_tree(e[0]) for e in model.estimators_],
        "provenance": "trained on data/samples (SYNTHETIC). Retrain on real drives before "
                      "quoting any accuracy number from this model.",
    }
    out_path = os.path.join(MODEL_DIR, "ondevice_model.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"[export] {out_path} ({size_kb:.1f} KB)")

    # The exported JSON must evaluate identically to the sklearn object, or the Kotlin
    # port is chasing a moving target.
    ref = np.array([eval_trees(payload, row) for row in X[:500]])
    sk = model.predict(X[:500])
    max_dev = float(np.max(np.abs(ref - sk)))
    assert max_dev < 1e-9, f"exported trees deviate from sklearn by {max_dev}"
    print(f"[verify] exported trees match sklearn to {max_dev:.2e}")

    # Golden vectors: raw IMU in, features and speed out. tests/test_kotlin_parity.py
    # checks Python against these; the Kotlin engine is checked against the same file.
    gold_drive = load_real_iovnbd_drive(os.path.join(SAMPLES_DIR, test_files[0]), driver_id="A")
    gdf = gold_drive.get_data().iloc[:500].reset_index(drop=True)
    gX, gy, gidx = compact_features(gdf)
    golden = {
        "source_drive": test_files[0],
        "n_samples": int(len(gdf)),
        "input": {
            "dt": gdf["dt"].tolist(),
            "acc": np.column_stack([gdf.acc_x, gdf.acc_y, gdf.acc_z]).tolist(),
            "gyro": np.column_stack([gdf.gyro_x, gdf.gyro_y, gdf.gyro_z]).tolist(),
        },
        "window_index": gidx.tolist(),
        "features": gX.tolist(),
        "predicted_speed": [eval_trees(payload, r) for r in gX],
        "gt_speed": gy.tolist(),
    }
    gold_path = os.path.join(MODEL_DIR, "golden_vectors.json")
    with open(gold_path, "w", encoding="utf-8") as fh:
        json.dump(golden, fh)
    print(f"[export] {gold_path} ({os.path.getsize(gold_path)/1024.0:.1f} KB, "
          f"{len(gX)} windows)")

    mae = float(np.mean(np.abs(np.array(golden['predicted_speed']) - gy)))
    print(f"[holdout] MAE on {test_files[0]}: {mae:.3f} m/s  (synthetic data - not a benchmark)")


if __name__ == "__main__":
    main()
