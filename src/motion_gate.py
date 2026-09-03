"""
src/motion_gate.py
Binary IN_VEHICLE_MOVING gate. Runs before the speed regressor and can veto it.

Why this exists: the old speed estimate was a monotone function of vibration energy, so
shaking a stationary phone produced forward travel. No amount of regressor tuning fixes
that, because the training set (a phone bolted into one car) contains zero examples of
vibration without motion. The gate supplies the missing distinction from physics instead.

Discriminators, cheapest first:
  1. gravity-direction stability  - a cradled phone holds it to ~0.01 over 2 s; a shaken
     one swings past 0.3. This one feature does most of the work.
  2. tilt rate                    - angular velocity not about vertical. A mounted phone
     on a road has almost none; a handled phone has lots.
  3. yaw/lateral coherence        - a turning vehicle satisfies a_lat ~= v * yaw_rate.
     A shaken phone satisfies neither the magnitude nor the sign.
  4. step detector                - if the OS step counter increments, the user is
     walking. Free, runs on the sensor hub, and is decisive.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# Gate decisions
MOVING = "IN_VEHICLE_MOVING"
STATIONARY = "STATIONARY"
HANDLING = "PHONE_HANDLED"


@dataclass
class GateThresholds:
    """
    Physical thresholds, not magic numbers. Retune per mount and per phone - a phone in a
    soft pocket vibrates differently from one in a rigid vent cradle. Use
    GateThresholds.calibrate() once SHL Car/Still/Walk data is available.
    """
    grav_stability_max: float = 0.08     # unit-vector dispersion over the window
    tilt_rate_max: float = 0.25          # rad/s, non-yaw angular rate
    still_acc_rms: float = 0.12          # m/s^2 horizontal, below this nothing is moving
    still_yaw_rate: float = 0.02         # rad/s
    coherence_min: float = 0.15          # corr(a_lat, v * yaw_rate) when genuinely driving
    debounce_frames: int = 5             # consecutive frames required to flip state
    # Coherence is computed and reported but does NOT veto by default. On the only data
    # available here it fires on ~11% of genuine driving (median correlation -0.01),
    # blowing the <2% false-positive budget on its own, while gravity stability and tilt
    # rate together already reject shaking completely. The physics is sound - a turning
    # vehicle really does feel v*omega laterally - but a veto that has never been
    # validated against real centripetal acceleration is a veto that ships a bug.
    # Turn it on once real drive data confirms the correlation holds.
    use_coherence_veto: bool = False

    @classmethod
    def calibrate(cls, feats: Dict[str, np.ndarray], is_vehicle: np.ndarray,
                  false_positive_target: float = 0.02) -> "GateThresholds":
        """
        Pick thresholds from labelled data: set each cut where it rejects the requested
        fraction of genuine vehicle motion. Feed this SHL Car (positive) vs Still/Walk in
        hand (negative). Returns defaults for any feature not supplied.
        """
        t = cls()
        veh = np.asarray(is_vehicle, dtype=bool)
        q = 100.0 * (1.0 - false_positive_target)
        if "grav_stability" in feats:
            t.grav_stability_max = float(np.percentile(np.asarray(feats["grav_stability"])[veh], q))
        if "tilt_rate" in feats:
            t.tilt_rate_max = float(np.percentile(np.abs(np.asarray(feats["tilt_rate"])[veh]), q))
        return t


def _rolling(x: np.ndarray, window: int, fn) -> np.ndarray:
    """Causal trailing-window reduction. Window [i-w+1, i]."""
    n = len(x)
    out = np.zeros(n)
    window = max(2, int(window))
    for i in range(n):
        out[i] = fn(x[max(0, i - window + 1):i + 1])
    return out


class MotionGate:
    """
    Stateless-per-call gate over a whole drive (offline), with a streaming counterpart in
    the Kotlin engine. Hysteresis is applied so the EKF does not chatter between states.
    """

    def __init__(self, thresholds: Optional[GateThresholds] = None, window: int = 20):
        self.th = thresholds or GateThresholds()
        self.window = window

    def classify(
        self,
        grav_stability: np.ndarray,
        tilt_rate: np.ndarray,
        a_horiz_mag: np.ndarray,
        yaw_rate: np.ndarray,
        a_lat: Optional[np.ndarray] = None,
        speed_hint: Optional[np.ndarray] = None,
        step_events: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns dict with 'state' (object array of MOVING/STATIONARY/HANDLING) and
        'in_vehicle_moving' (bool array). All inputs come from frame_alignment.align_frame.
        """
        n = len(grav_stability)
        w = self.window

        tilt_rms = _rolling(np.abs(tilt_rate), w, lambda v: float(np.sqrt(np.mean(v**2))))
        acc_rms = _rolling(a_horiz_mag, w, lambda v: float(np.sqrt(np.mean(v**2))))
        yaw_abs = _rolling(np.abs(yaw_rate), w, lambda v: float(np.mean(v)))

        # 1. Phone is being handled: gravity direction wandering, or the phone is being
        #    rotated about a non-vertical axis. Either means the body frame is no longer
        #    the vehicle frame, so any speed we infer is meaningless.
        handled = (grav_stability > self.th.grav_stability_max) | (tilt_rms > self.th.tilt_rate_max)

        # 2. Walking beats every inertial heuristic: if the OS says steps, it is steps.
        if step_events is not None:
            steps = _rolling(np.asarray(step_events, dtype=float), w, lambda v: float(np.sum(v)))
            handled |= steps > 0

        # 3. Genuinely still: nothing horizontal, no yaw.
        still = (acc_rms < self.th.still_acc_rms) & (yaw_abs < self.th.still_yaw_rate)

        # 4. Coherence check, when we have enough to compute it. A vehicle turning at
        #    yaw rate w while moving at v feels lateral acceleration v*w. Shaking does not
        #    reproduce that relationship, so low coherence over a window that otherwise
        #    looks like motion is evidence of handling.
        if a_lat is not None and speed_hint is not None:
            expected = np.asarray(speed_hint) * np.asarray(yaw_rate)
            coh = np.zeros(n)
            for i in range(n):
                s = max(0, i - w + 1)
                x, y = a_lat[s:i + 1], expected[s:i + 1]
                if len(x) >= 4 and np.std(x) > 1e-6 and np.std(y) > 1e-6:
                    coh[i] = float(np.corrcoef(x, y)[0, 1])
            # Only demote when the vehicle is actually turning - straight-line driving has
            # nothing to correlate and would otherwise be flagged as handling.
            if self.th.use_coherence_veto:
                turning = yaw_abs > 0.05
                handled |= turning & (coh < self.th.coherence_min)
        else:
            coh = np.zeros(n)

        raw = np.where(handled, HANDLING, np.where(still, STATIONARY, MOVING))

        # 5. Debounce: require k consecutive identical frames before committing a change.
        state = np.empty(n, dtype=object)
        current = STATIONARY
        run_val, run_len = raw[0] if n else STATIONARY, 0
        for i in range(n):
            if raw[i] == run_val:
                run_len += 1
            else:
                run_val, run_len = raw[i], 1
            if run_len >= self.th.debounce_frames:
                current = run_val
            state[i] = current

        return {
            "state": state,
            "in_vehicle_moving": state == MOVING,
            "grav_stability": grav_stability,
            "tilt_rms": tilt_rms,
            "acc_rms": acc_rms,
            "coherence": coh,
        }

    def classify_frame(self, fr: Dict[str, np.ndarray], speed_hint: Optional[np.ndarray] = None,
                       step_events: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Convenience: take frame_alignment.align_frame() output directly."""
        return self.classify(
            grav_stability=fr["grav_stability"],
            tilt_rate=fr["tilt_rate"],
            a_horiz_mag=fr["a_horiz_mag"],
            yaw_rate=fr["yaw_rate"],
            a_lat=fr.get("a_lat"),
            speed_hint=speed_hint,
            step_events=step_events,
        )


def demo():
    """Self-check: shaking a stationary phone must not read as vehicle motion."""
    from frame_alignment import align_frame, G0

    rng = np.random.default_rng(1)
    n, dt_s = 600, 0.1
    dt = np.full(n, dt_s)
    t = np.arange(n) * dt_s

    # Case A: stationary phone, hand shake at 3 Hz, 5 m/s^2, with the wrist rotation that
    # necessarily accompanies it.
    shake = 5.0 * np.sin(2 * np.pi * 3.0 * t)
    acc_shake = np.column_stack([shake, 0.6 * shake, np.full(n, G0) + 0.4 * shake])
    gyro_shake = np.column_stack([1.5 * np.sin(2 * np.pi * 3.0 * t),
                                  1.1 * np.cos(2 * np.pi * 3.0 * t),
                                  0.4 * np.sin(2 * np.pi * 3.0 * t)])
    fr = align_frame(acc_shake, gyro_shake, dt)
    res = MotionGate().classify_frame(fr)
    frac = float(np.mean(res["in_vehicle_moving"]))
    assert frac < 0.05, f"shake classified as vehicle motion {frac:.1%} of the time"

    # Case B: real driving - steady gravity, modest road vibration, a sustained turn.
    acc_drive = np.column_stack([
        rng.normal(0, 0.15, n),
        0.8 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.15, n),
        np.full(n, G0) + rng.normal(0, 0.2, n),
    ])
    gyro_drive = np.column_stack([rng.normal(0, 0.01, n), rng.normal(0, 0.01, n),
                                  np.full(n, 0.12) + rng.normal(0, 0.01, n)])
    fr2 = align_frame(acc_drive, gyro_drive, dt)
    res2 = MotionGate().classify_frame(fr2)
    frac2 = float(np.mean(res2["in_vehicle_moving"]))
    assert frac2 > 0.9, f"genuine driving rejected, only {frac2:.1%} accepted"

    print(f"motion_gate demo OK: shake accepted {frac:.1%}, driving accepted {frac2:.1%}")


if __name__ == "__main__":
    demo()
