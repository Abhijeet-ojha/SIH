"""
src/frame_alignment.py
Phone-frame -> vehicle-frame alignment.

The rest of the pipeline used to assume acc_y == "forward" and gyro_z == "yaw".
Both are false for a phone at an arbitrary angle, which is the hardest requirement in
the problem statement. Everything here is built from quantities that do not change when
the whole phone is rotated:

    a_vert   = a . g_hat                 (scalar)
    a_horiz  = a - (a . g_hat) g_hat     (3-vector, magnitude is invariant)
    yaw_rate = omega . g_hat             (scalar)

plus a forward axis f_hat estimated from data. f_hat lives in the phone body frame and
rotates with it, so a . f_hat is invariant too: rotating the phone by R sends
a -> Ra and f_hat -> R f_hat, and (Ra).(R f_hat) == a . f_hat.

Causal throughout: the gravity filter is a forward-only single pole, matching what the
Android engine can do in real time.
"""

import numpy as np
from typing import Dict, Optional, Tuple

G0 = 9.80665

# ponytail: single-pole low-pass instead of a Madgwick/Mahony filter. Gravity is the only
# attitude we need and a car does not tumble. Upgrade to TYPE_ROTATION_VECTOR on Android
# (already registered) or a full AHRS if pitch/roll rate ever matters.
#
# KNOWN LIMITATION, measured, not theoretical: at 0.2 Hz the time constant is ~0.8 s, so any
# acceleration sustained longer than a few seconds leaks into the gravity estimate and
# disappears from a_horiz. A long steady brake therefore reads as near-zero horizontal
# specific force. Every complementary attitude filter has this failure; the standard fix is
# to subtract a velocity-derived acceleration prediction before updating gravity, which
# needs a speed estimate we do not trust yet. naive_dr._find_alignment_window works around
# it by using a non-tracking median gravity. Revisit once the speed model is trained on
# real drives.
GRAVITY_LP_HZ = 0.2

# Calibration knobs. These are physical thresholds, not magic numbers - retune per mount.
FWD_MIN_SPEED_MPS = 1.5      # below this, GPS course and accel direction are both noise
FWD_MIN_ACCEL_MPS2 = 0.35    # a real accel/brake event, not idle jitter
FWD_MIN_SAMPLES = 25         # too few events -> refuse to estimate, report low confidence


def lowpass_gravity(acc: np.ndarray, dt: np.ndarray, fc_hz: float = GRAVITY_LP_HZ) -> np.ndarray:
    """
    Causal single-pole low-pass of the raw accelerometer -> gravity estimate, shape (n, 3).

    On Android prefer Sensor.TYPE_GRAVITY, which the sensor hub computes for free; this
    exists so the offline pipeline sees the same signal as the phone.
    """
    acc = np.asarray(acc, dtype=float)
    n = len(acc)
    g = np.zeros((n, 3))
    if n == 0:
        return g

    # Initialise from the first second so the filter does not start at the origin and
    # spend 10 seconds converging.
    warm = max(1, min(n, int(round(1.0 / max(float(np.median(dt)), 1e-3)))))
    g[0] = np.mean(acc[:warm], axis=0)

    tau = 1.0 / (2.0 * np.pi * fc_hz)
    for i in range(1, n):
        alpha = dt[i] / (dt[i] + tau) if dt[i] > 0 else 0.0
        g[i] = (1.0 - alpha) * g[i - 1] + alpha * acc[i]
    return g


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalise, leaving zero-length rows as zero rather than NaN."""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)


def gravity_stability(g_hat: np.ndarray, window: int) -> np.ndarray:
    """
    Dispersion of the gravity direction over a trailing window.

    This is the single most useful signal in the whole repo for telling "the vehicle is
    moving" from "a human is holding the phone". A cradled phone holds its gravity
    direction to ~0.01; a shaken one swings past 0.3.

    Causal: window [i-w+1, i].
    """
    n = len(g_hat)
    out = np.zeros(n)
    window = max(2, int(window))
    for i in range(n):
        s = max(0, i - window + 1)
        w = g_hat[s:i + 1]
        if len(w) < 2:
            continue
        # Norm of the per-axis std: 0 for a perfectly steady direction.
        out[i] = float(np.linalg.norm(np.std(w, axis=0)))
    return out


def estimate_forward_axis(
    acc_horiz: np.ndarray,
    speed: np.ndarray,
    dt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Estimate the vehicle forward direction as a unit 3-vector in the phone body frame.

    During GPS-healthy acceleration and braking, horizontal specific force lies along the
    vehicle's longitudinal axis. Take the principal axis of horizontal acceleration over
    those samples, then fix the sign by correlating with the observed speed change.

    Returns (f_hat, confidence in [0, 1]). Confidence is the fraction of horizontal
    acceleration variance explained by the principal axis - near 1.0 when the events are
    genuinely longitudinal, near 0.5 when the data is isotropic noise.
    """
    acc_horiz = np.asarray(acc_horiz, dtype=float)
    speed = np.asarray(speed, dtype=float)

    dv = np.gradient(speed, edge_order=1) / np.maximum(dt, 1e-6)

    mask = (np.abs(dv) > FWD_MIN_ACCEL_MPS2) & (speed > FWD_MIN_SPEED_MPS)
    if valid_mask is not None:
        mask &= valid_mask

    if int(np.sum(mask)) < FWD_MIN_SAMPLES:
        # Not enough evidence. Refuse rather than return a confident wrong axis.
        return np.array([0.0, 1.0, 0.0]), 0.0

    A = acc_horiz[mask]
    A = A - np.mean(A, axis=0)
    cov = A.T @ A / max(1, len(A) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    f_hat = evecs[:, order[0]]

    # Sign: forward acceleration must correlate positively with speed increase.
    proj = acc_horiz[mask] @ f_hat
    if float(np.sum(proj * dv[mask])) < 0:
        f_hat = -f_hat

    total = float(np.sum(evals))
    confidence = float(evals[0] / total) if total > 1e-12 else 0.0
    return unit(f_hat), confidence


def align_frame(
    acc: np.ndarray,
    gyro: np.ndarray,
    dt: np.ndarray,
    speed: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    stability_window: int = 20
) -> Dict[str, np.ndarray]:
    """
    Full alignment. acc/gyro are (n, 3) in the phone body frame; dt is (n,) seconds.

    Returns frame-invariant channels. Every value here is unchanged if you pre-multiply
    acc and gyro by any fixed rotation - that is what tests/test_frame_invariance.py checks.
    """
    acc = np.asarray(acc, dtype=float)
    gyro = np.asarray(gyro, dtype=float)
    dt = np.asarray(dt, dtype=float)

    grav = lowpass_gravity(acc, dt)
    g_hat = unit(grav)

    # Linear (gravity-removed) acceleration.
    lin = acc - grav

    a_vert = np.sum(lin * g_hat, axis=1)
    a_horiz = lin - a_vert[:, None] * g_hat
    a_horiz_mag = np.linalg.norm(a_horiz, axis=1)

    yaw_rate = np.sum(gyro * g_hat, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    # Tilt rate: angular velocity not about the vertical. Large when the phone is being
    # handled, near zero for a mounted phone on a flat road.
    tilt_rate = np.sqrt(np.maximum(gyro_mag**2 - yaw_rate**2, 0.0))

    grav_stab = gravity_stability(g_hat, stability_window)

    if speed is not None:
        f_hat, f_conf = estimate_forward_axis(a_horiz, speed, dt, valid_mask)
    else:
        f_hat, f_conf = np.array([0.0, 1.0, 0.0]), 0.0

    # Lateral axis completes the right-handed frame: l_hat = g_hat x f_hat.
    g_ref = unit(np.mean(g_hat, axis=0))
    l_hat = unit(np.cross(g_ref, f_hat))

    a_fwd = a_horiz @ f_hat
    a_lat = a_horiz @ l_hat

    return {
        "gravity": grav,
        "g_hat": g_hat,
        "a_vert": a_vert,
        "a_horiz": a_horiz,
        "a_horiz_mag": a_horiz_mag,
        "a_fwd": a_fwd,
        "a_lat": a_lat,
        "yaw_rate": yaw_rate,
        "gyro_mag": gyro_mag,
        "tilt_rate": tilt_rate,
        "grav_stability": grav_stab,
        "forward_axis": f_hat,
        "forward_confidence": np.array([f_conf]),
    }


def add_frame_columns(df, stability_window: int = 20):
    """
    Convenience wrapper: takes a standardized drive dataframe, returns it with the
    frame-invariant channels appended. Used by feature_engineering and the EKF pipeline
    so they never touch raw acc_y / gyro_z again.
    """
    dt = df["dt"].values if "dt" in df.columns else np.gradient(df["timestamp"].values)
    acc = np.column_stack([df["acc_x"].values, df["acc_y"].values, df["acc_z"].values])
    gyro = np.column_stack([df["gyro_x"].values, df["gyro_y"].values, df["gyro_z"].values])
    speed = df["speed"].values if "speed" in df.columns else None

    fr = align_frame(acc, gyro, dt, speed=speed, stability_window=stability_window)

    out = df.copy()
    for k in ["a_vert", "a_horiz_mag", "a_fwd", "a_lat", "yaw_rate", "gyro_mag",
              "tilt_rate", "grav_stability"]:
        out[k] = fr[k]
    out.attrs["forward_axis"] = fr["forward_axis"]
    out.attrs["forward_confidence"] = float(fr["forward_confidence"][0])
    return out


def demo():
    """Self-check: rotating the whole rig must not change any invariant channel."""
    rng = np.random.default_rng(0)
    n = 600
    dt = np.full(n, 0.1)
    t = np.arange(n) * 0.1

    # A phone lying flat, vehicle accelerating forward along body +y, turning at 0.2 rad/s.
    acc = np.column_stack([
        rng.normal(0, 0.02, n),
        1.2 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.02, n),
        np.full(n, G0) + rng.normal(0, 0.02, n),
    ])
    gyro = np.column_stack([
        rng.normal(0, 0.001, n),
        rng.normal(0, 0.001, n),
        np.full(n, 0.2) + rng.normal(0, 0.001, n),
    ])
    speed = 8.0 + np.cumsum(acc[:, 1] * dt)

    base = align_frame(acc, gyro, dt, speed=speed)

    # Arbitrary fixed rotation of the entire phone.
    ang = 0.9
    axis = unit(np.array([0.3, -0.7, 0.65]))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

    rot = align_frame(acc @ R.T, gyro @ R.T, dt, speed=speed)

    for key in ["a_vert", "a_horiz_mag", "yaw_rate", "gyro_mag", "tilt_rate", "grav_stability"]:
        err = float(np.max(np.abs(base[key] - rot[key])))
        assert err < 1e-9, f"{key} is not frame-invariant: max abs diff {err}"

    # The forward axis must rotate with the phone, keeping a_fwd invariant.
    err_fwd = float(np.max(np.abs(base["a_fwd"] - rot["a_fwd"])))
    assert err_fwd < 1e-6, f"a_fwd not invariant: {err_fwd}"
    assert base["forward_confidence"][0] > 0.9, "forward axis should be confident here"

    print("frame_alignment demo OK: invariance holds under arbitrary rotation")


if __name__ == "__main__":
    demo()
