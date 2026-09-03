# Claude Code Rebuild Prompt — SIH Dead Reckoning

Paste everything below the line into Claude Code, in this repo, **one phase at a time**.
Do not run all phases in one go. Each phase has a gate; if the gate fails, stop and fix.

---

## Context you must read before writing any code

Read these files fully first: `src/data_loader.py`, `src/naive_dr.py`,
`src/feature_engineering.py`, `src/fusion_ekf.py`, `src/metrics.py`,
`src/speed_model.py`, and
`android_logger/app/src/main/java/com/sih/sensorlogger/OnDeviceInferenceEngine.kt`.

There is **one root-cause bug** wearing three costumes, and every phase below exists
to kill it:

**The speed estimate is a function of vibration energy, and nothing else.**

- Android, `OnDeviceInferenceEngine.pushSample()`:
  `dynamicSpeed = max(0.0, (meanMagAcc - 9.80665) * 1.85 + stdAy * 4.20)`.
  Both terms increase monotonically with vibration. By Jensen's inequality
  `E[|g + noise|] > |g|`, so *any* agitation of a stationary phone raises term one;
  term two is literally a vibration statistic. Shaking the phone therefore produces
  forward speed, and since `v >= 0` and position integrates `v*sin(heading)`,
  shaking produces forward *travel*. This is the reported bug.
- Python, `feature_engineering.extract_causal_window_features()`: the feature set is
  `acc_mag_std`, `vibration_power = acc_mag_std * acc_horiz_rms`, `jerk_mag_rms`,
  spectral band powers, etc. — again, vibration energy. In IO-VNBD the phone is rigidly
  fixed in one car, so vibration correlates with speed and the model happily learns that
  shortcut. It has **zero examples of vibration without motion**, so it cannot learn the
  difference.
- This is why the evidence looks the way it does. Verify each claim against the CSVs in
  `outputs/metrics/ml_experiments/` before proceeding: LODRO R² goes *negative* on
  held-out drivers B (-0.61) and E (-0.57) — a different car and mount breaks the spurious
  vibration↔speed mapping; adding Gaussian sensor noise *improves* MAE (1.0006 → 0.9670) —
  noise dilutes a spurious feature; and RF, XGBoost and a 1D-CNN all land within 4 m of
  each other (414.7 / 415.2 / 418.7) — every architecture finds the same single shortcut.

**The shake bug and the negative R² are the same bug.** Fixing the frame and adding
negative examples fixes both. Do not treat them as separate work items.

Ground rules for every phase:

- Do not add a dependency where numpy/scipy/sklearn or a native Android sensor will do.
- Every phase leaves one runnable check behind (`assert`-based, in `tests/`). No fixtures,
  no frameworks.
- Never report a metric without also reporting the gate that would have falsified it.
- If a phase's gate fails, say so plainly and stop. Do not tune until it passes.

---

## Phase 0 — Fix the time axis and the labels. Nothing else matters until this passes.

`src/data_loader.py::_standardize()` fabricates the timeline:

```python
dt_raw[dt_raw <= 0.0] = 0.1
dt_raw[dt_raw > 1.0] = 0.1
out["timestamp"] = np.cumsum(dt_raw)
```

Proof it is fabricated: in `outputs/metrics/ml_experiments/phase4_target_audit.csv`,
`duration_s` is exactly `samples * 0.1` for **every** drive (3000 → 299.9, 954 → 95.3).
A real log does not do that. Any true sample interval at or above 1.0 s was silently
rewritten to 0.1 s, compressing elapsed time and inflating every derived acceleration.
That is the source of the 17–33 m/s² implied accelerations in the same file.

Do this:

1. Delete the clamping. Preserve the real timestamps. Handle genuine gaps by *recording*
   them (`out["dt"]`, and a `gap_mask` for `dt > 3 * median_dt`), never by overwriting them.
2. Determine the true sample rate per drive empirically from the median dt and stop
   passing a hardcoded `sample_rate=10.0` into feature extraction. Resample to a uniform
   grid explicitly if you need one, and record that in the drive metadata.
3. Resolve the speed-unit question with data, not with the column header. The header says
   `GPS Speed (Kmh)` and the loader divides by 3.6, but the audit shows a "highway" drive
   topping out at 6.7 m/s (24 km/h), which is not a highway. Write the cross-check:

   **`integral(v * dt)` must equal the GPS path length from the ENU positions to within 5%.**

   Run it both with and without the `/3.6`. Exactly one will pass. Take that one. Report
   both numbers so the choice is auditable.
4. Delete the `out["acc_y"] = np.gradient(out["speed"], dt)` fallback. If the accelerometer
   columns are missing, raise. That fallback derives a *feature* from the *label*; if it
   ever fired, every metric in the repo is meaningless and you need to know that.

**Gate** (`tests/test_loader_integrity.py`), for every drive:
`abs(integral(v*dt) - gps_path_length) / gps_path_length < 0.05`;
`duration_s != samples * median_dt` for at least one drive (the timeline is no longer
synthetic); `max_implied_accel < 8.0` m/s²; and `acc_y` is not a transform of `speed`
(correlation with `np.gradient(speed)` is not ~1.0).

Then re-run `scripts/ml_phase4_target_audit.py` and paste the before/after table.
**Do not proceed to Phase 1 until this gate is green.** Expect ML R² to move on its own.

---

## Phase 1 — Fix the naive baseline so the comparison is honest.

`src/naive_dr.py::compute()` integrates raw `acc_y` with no gravity or bias removal:
`velocity_est[i] = velocity_est[i-1] + acc_y[i] * dt`. If `acc_y` carries any gravity
projection or bias, that integrates into a ramp — hence 19,338 m of "drift" on a 1,221 m
drive, and hence the "99.9% improvement" headline that a reviewer will dismantle in ten
seconds.

Implement the honest textbook baseline: estimate the gravity vector from the first
stationary window, subtract it, remove accelerometer bias over that window, then
double-integrate. Keep the broken one available as
`NaiveDeadReckoning(remove_gravity=False)` and label it in outputs as
*"unbiased-integration strawman, shown for reference"* — never as the comparison baseline.

**Gate** (`tests/test_naive_baseline.py`): on a synthetic constant-velocity trajectory with
gravity added to the accelerometer, corrected naive DR drifts < 5% of path length over 60 s;
the uncorrected version drifts > 100%. Re-run `scripts/01_run_naive_baseline.py` and report
the new baseline numbers.

---

## Phase 2 — Make the phone frame irrelevant. This is the actual shake fix.

Right now `acc_y` is assumed to be "forward" and `gyro_z` is assumed to be "yaw". Both
assumptions are false for a phone at an arbitrary angle — the hardest requirement in the
problem statement, and the one the current system sidesteps entirely because IO-VNBD is
vehicle-frame data that hides the problem.

Add `src/frame_alignment.py`:

1. **Gravity estimate** — low-pass the accelerometer (a 0.2 Hz single-pole filter is
   enough; do not write a Madgwick filter). On Android use `Sensor.TYPE_GRAVITY` and
   `TYPE_LINEAR_ACCELERATION` directly — the platform already computes these on the
   sensor hub, for free.
2. **Projection** — decompose linear acceleration into the component along gravity
   (vertical) and the 2D component orthogonal to it (horizontal). Yaw rate becomes
   `omega · ĝ`, the gyro projected onto the gravity unit vector. These three quantities
   are invariant to how the phone is rotated. `gyro_z` is not.
3. **Forward-axis estimation** — within the horizontal plane, the forward direction is the
   principal axis of horizontal acceleration during GPS-healthy acceleration/braking
   events, sign-disambiguated by correlating with GPS speed change. Estimate it while GPS
   is available, hold it through the blackout, expose a confidence, and re-estimate on any
   large gravity-direction change (the phone was picked up and re-seated).

Rewrite the feature extractor to consume **only** frame-invariant channels: horizontal
accel magnitude and its projection on the estimated forward axis, vertical accel, yaw rate
about gravity, gravity-direction stability, and the existing statistics computed over
*those*. Delete raw `ax/ay/az/gx/gy/gz` moments from the feature set.

**Gate** (`tests/test_frame_invariance.py`): take one drive, apply 20 random fixed 3D
rotations to the raw IMU, extract features under each. Every frame-invariant feature must
match the unrotated case within 1e-6, and end-to-end predicted speed MAE must not change
by more than 2%. Today this test fails catastrophically; that is the point.

---

## Phase 3 — Motion gating: teach it that vibration is not motion.

Add `src/motion_gate.py` — a binary `IN_VEHICLE_MOVING` classifier that runs *before* the
speed regressor and can veto it.

The cheap, robust discriminator is **gravity-vector stability**. A phone in a cradle holds
its gravity direction to within ~0.01 of a unit vector over a 2 s window; a handheld,
shaken phone swings 0.3+. Compute the `std` of the gravity unit vector over a sliding
window and make it the primary feature. Then add, in order of cost:

- **yaw-rate/lateral-accel coherence** — a turning vehicle satisfies `a_lat ≈ v * omega`;
  a shaken phone produces neither the correlation nor a consistent sign;
- **spectral separation** — human handling concentrates energy at 2–8 Hz and is
  non-stationary; road/engine coupling is lower-amplitude, higher-frequency, sustained;
- **magnetometer heading change vs integrated gyro yaw** — these agree in a vehicle and
  diverge when the phone is rotated relative to the vehicle;
- **`Sensor.TYPE_STEP_DETECTOR`** — if the step counter increments, the user is walking,
  not driving. Runs on the sensor hub, costs no battery, roughly five lines. Highest
  value per line in this document.

Wire it into `fusion_ekf.KinematicFusionEKF`: when the gate says not-moving, apply a ZUPT
with tight R and **do not propagate position**. This replaces the current `is_stationary`
path, which only triggers on near-perfect stillness (`varAy < 0.02 && abs(gz) < 0.012`) and
is therefore trivially bypassed by shaking.

**Gate** (`tests/test_shake_rejection.py`): synthesize 60 s of stationary phone with
(a) 1–10 Hz sinusoidal shake at 5 m/s² amplitude, (b) random-walk hand motion, (c) the
phone being picked up and rotated 180°. Accumulated position displacement must be **< 2 m**
in all three; today this produces hundreds of metres. Also assert that on a real IO-VNBD
drive the gate does not fire during genuine motion (false-positive rate < 2%).

---

## Phase 4 — Report metrics honestly.

`src/metrics.py` computes `fused_drift_pct = fused_final_err / total_dist` — error at the
*end of the drive*, after GPS has returned and is directly correcting the filter. That
measures "does the EKF track GPS while GPS is on." It is not a dead-reckoning number, and
it is the source of the "< 0.05% drift" headline.

Make `blackout_terminal_exit_error_m / blackout_distance_m` the **primary, first-reported**
metric everywhere: `README.md`, `docs/PROPOSAL_NARRATIVE.md`,
`outputs/metrics/benchmark_summary.md`, and the dashboard. Keep the closed-loop number,
clearly labelled as closed-loop.

Report the honest per-drive table even though it currently reads 40–79% against a ~10%
target. Report `improvements.blackout_error_reduction_pct` including the three negative
values (-6.54, -134.0, -32.23) — the fused system is currently worse than naive on half the
test set, that field is already in the repo, and a reviewer who opens it before you mention
it has found your rebuttal for you.

Demote README "Key Innovations" #2 and #3 to an *"evaluated, measured contribution"*
section with the real numbers from `phase15_system_ablation.csv`: calibrated
heteroscedastic uncertainty and NHC together move 414.41 m → 413.90 m, i.e. 0.5 m out of
414. A team that reports its own negative results reads as scientists; a team caught
inflating them does not. Re-measure both after Phase 2 — NHC contributes nothing today
largely because it is applied in a body frame that is itself wrong.

**Gate:** `grep -r "0.05%" README.md docs/` returns nothing, and every headline accuracy
claim in the README traces to a named field in a file under `outputs/metrics/`.

---

## Phase 5 — Put the real model on the phone.

`OnDeviceInferenceEngine` contains two hardcoded constants, not the trained model, and
`updateGpsFix` is a fixed-gain complementary blend (`0.85/0.15`), not the 6-state
Joseph-form EKF that was benchmarked. The README describes an algorithm that is not
running.

Export the trained regressor and port the Phase 2/3 pipeline to Kotlin so the phone runs
what was benchmarked. Use XGBoost (454 KB) or the CNN (75 KB), not the 13.4 MB Random
Forest. Register the sensors the app currently ignores — `MainActivity` registers only
`TYPE_ACCELEROMETER` and `TYPE_GYROSCOPE`, while `pushSample()` takes an `ambientLux`
argument that is fed a constant. Add `TYPE_GRAVITY`, `TYPE_LINEAR_ACCELERATION`,
`TYPE_MAGNETIC_FIELD`, `TYPE_PRESSURE`, `TYPE_LIGHT`, `TYPE_STEP_DETECTOR`,
`TYPE_ROTATION_VECTOR`.

**Gate** (`tests/test_kotlin_parity.py`): feed the same 500-sample CSV window through the
Python pipeline and the Kotlin engine (via a small JVM harness or a golden-vector file);
predicted speeds must agree within 1e-3. Until this passes, the README may not claim
on-device ML.

---

## Phase 6 — Barometer and map matching. Only after 0–5 are green.

- **Barometer** (`TYPE_PRESSURE`) gives ~0.1 m relative vertical resolution. It is the
  answer to the multi-level-parking requirement currently missing entirely, and it
  separates a flyover from the service road beneath it. Add altitude as a 7th EKF state
  with pressure as the measurement; calibrate the offset against GPS altitude while GPS is
  healthy.
- **Map matching** is where the remaining accuracy is. Snap the blackout trajectory to the
  OSM road graph (fetch once via Overpass, cache to disk, ship the cached extract; do not
  add a routing engine). A Hidden Markov Model over candidate road segments — Newson &
  Krumm, *Hidden Markov Map Matching Through Noise and Sparseness*, is the standard
  reference — turns a 500 m free-drift error into a metres-level along-road error, because
  the vehicle is constrained to roads. Worth more than every model-architecture change in
  `model_benchmark_summary.csv` combined.

**Gate:** map-matched blackout exit error over blackout distance < 15% on at least four of
six test drives, reported per drive, with the unmatched number alongside.

---

## Repo hygiene, do alongside

- `download_dataset.py` does not download IO-VNBD; it *generates synthetic* Delhi-coordinate
  data. Rename it `generate_synthetic_drives.py` and write a real fetch script, or document
  the manual acquisition step. Right now `README.md` Quick Start tells a fresh clone to run
  `run_all.py`, which needs the gitignored `data/IO-VNBD-repo/` and will crash.
- Make `run_all.py` fall back to `data/samples/` with a loud banner
  (`SYNTHETIC DATA — NOT A BENCHMARK RESULT`) so a fresh clone runs end to end.
- `Driver_B_M_train` (offset 10000) and `Driver_B_M_n1` (offset 0) are different segments of
  the same drive, split across train and test. Same for `S-Y1`. Disclose it or drop it.
