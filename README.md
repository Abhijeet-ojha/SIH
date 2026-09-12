# SIH Problem Statement 168: AI-Assisted Vehicle Dead Reckoning & GNSS Fusion Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Research and demonstration prototype for **Smart India Hackathon (SIH) Problem Statement 168**: *AI-Assisted Dead Reckoning for Terrestrial Navigation in Extended GNSS-Denied Environments*.

Developed and evaluated on real multi-driver data from the **IO-VNBD Benchmark Suite** (Coventry, UK).

## NavPulse indoor walking prototype - version 2.1.0

NavPulse now has a dedicated **Indoor walking / GPS off** mode for demonstrating
relative pedestrian motion in a room. It starts at local `(0, 0)` without waiting
for a satellite fix or requesting location permission. The vehicle estimator and
its phone-handling gate are bypassed in this mode.

**Publication status:** the Flutter implementation and signed demo APK have been
built in the development workspace but are not yet published on `main` or as a
GitHub Release. The instructions below apply to that local 2.1.0 build; cloning
`main` alone does not yet provide the Flutter app. This README update does not
publish an APK download.

### Two navigation modes

| Mode | Position input | Intended demonstration |
| --- | --- | --- |
| Indoor walking (default in 2.1.0) | Accelerometer step cycles, relative phone heading, configured step length | Walking from a local start point with GPS and networking off |
| Vehicle navigation | GNSS, IMU, motion gate, bundled speed model and six-state EKF | Vehicle tracking and controlled GNSS withholding |

Indoor tracking requests accelerometer and gyroscope samples at approximately
50 Hz. Android's game rotation vector supplies relative heading when available,
with rotation-vector or gyroscope fallback. Each detected step advances the
estimated position; the map redraws the trail at room scale with a START marker.
The default step length is **0.65 m**, and the step threshold is **0.6**.

### Install and demonstrate the local APK

Local artifact: `outputs/apk/navpulse-indoor-walk.apk` (approximately 19.2 MB).
Package: `com.sih2026.navpulse`, version `2.1.0`, version code `2`.
It is a release-mode APK signed with the local debug key for sideloaded demos.

1. Install the new APK as an update to NavPulse Localizer.
2. Enable airplane mode. Also turn off Wi-Fi and Bluetooth if they remain enabled;
   phone Location can be off for indoor mode.
3. Open the app and confirm **INDOOR WALK / READY** appears.
4. Hold the phone screen-up, top edge pointing in your walking direction. Tap
   **Play** and hold still for two seconds while the detector warms up.
5. Walk 8-10 normal steps. Watch the step count, estimated distance and amber trail.
6. Turn your body and phone together through 90 degrees, then continue walking.
7. Stop walking: speed returns to zero after about 1.4 seconds without a detected
   step. Tap **Stop**, then **Sessions**, to inspect the session summary.

The presentation handset is a **OnePlus Nord CE 3**. Physical performance on that
handset has not yet been verified by the developer; test a short route before
presenting. The indoor code path uses local motion sensors and does not request
GNSS or network access. An airplane-mode walkthrough on the actual handset is
still a required demonstration check, not a completed validation claim.

### Troubleshooting the room demo

- **Still says SEARCHING:** check that the new 2.1.0 APK is installed and Indoor
  walking is enabled. The previous app used the vehicle pipeline and waited for GPS.
- **No steps:** open **Pipeline** and inspect accelerometer rate and step signal.
  A zero rate or sensor error indicates a sensor-stream problem. If samples are
  live but gentle steps are missed, stop and lower **Navigate > Step threshold**
  from `0.6` to `0.4`, then restart.
- **Distance scale is wrong:** while stopped, adjust step length using a measured
  distance divided by the detected step count.
- **Trail direction is wrong:** keep the phone facing your walking direction and
  turn it together with your body. Restart to establish a new starting direction.
- **More panels:** swipe the horizontal panel selector to reach Sessions or Settings.

Keep the app foregrounded during the demo. Session records are currently in memory
and disappear when the app process closes.

### Validation and limits

For the local 2.1.0 build:

- **22 Flutter tests passed**, covering pedestrian signal detection, gentle gait,
  turns, stopping, stationary noise, GPS-free startup, phone portrait layout, and
  existing vehicle/filter functionality.
- **9 Python model parity checks passed** for feature order, thresholds and the
  exported model. These do not validate pedestrian accuracy or compiled
  cross-language floating-point equivalence.
- Android release build completed; APK signature and package/version checks passed.
- Pedestrian movement tests use controlled sensor signals. No measured indoor
  position-accuracy result or completed physical-phone walk test is claimed.

This is **relative pedestrian dead reckoning**: it estimates movement from the
starting point, not indoor latitude/longitude or a floor plan. Step length,
missed/false steps, independent phone rotation and heading drift affect the path.
Repeated hand shaking can cause false steps. The displayed drift allowance is a
heuristic, not a calibrated confidence interval. Vehicle benchmark results below
do not measure this pedestrian mode.

### Build the Flutter source when available

The local app is under `flutter_app/`. The verified toolchain was Flutter 3.24.5 /
Dart 3.5.4, Java 17 and Android SDK 34. Configure `JAVA_HOME` and `ANDROID_HOME` for
your machine, then run from that directory:

```bash
flutter pub get
flutter test
flutter build apk --release --no-shrink
```

Build output: `build/app/outputs/flutter-apk/app-release.apk`. On Windows, a local
build directory outside OneDrive can avoid synchronization-related file locks.

---

## 🚀 Key Innovations

> **Read this first.** Items 1–5 below describe what the system *does*. Items with a
> measured contribution have it stated inline; where the measurement says a component
> contributes ~nothing, that is stated too. See
> [Measured contributions](#-measured-contributions-including-the-negative-results).

0. **Frame-invariant motion estimation with a phone-handling gate** *(the load-bearing one)*
   - Nothing downstream sees raw `acc_x/y/z` or `gyro_z` — including the EKF's own
     standstill test, which was the last frame-dependent decision and is now computed from
     horizontal specific force about gravity. Gravity is estimated per sample,
     and every feature is built from quantities invariant to how the phone is held:
     vertical and horizontal specific force, yaw rate about gravity, and a data-estimated
     vehicle forward axis. `tests/test_frame_invariance.py` applies 20 random rotations to
     the raw IMU and requires every feature to be bit-identical.
   - A motion gate (`src/motion_gate.py`) vetoes the speed estimate when the phone is being
     handled rather than carried by a moving vehicle. Primary discriminator is
     gravity-direction stability — a cradled phone holds it to ~0.01 over 2 s, a shaken one
     swings past 0.3 — plus non-yaw tilt rate and the OS step detector.
   - This replaced a speed law that was a monotone function of vibration energy, under which
     shaking a stationary phone produced forward travel. `tests/test_shake_rejection.py`
     holds accumulated displacement under 2 m for 1–10 Hz shake, random handling, and a
     180° flip, at a <2% false-positive rate on genuine motion.

1. **Partitioned Sequential 6-State EKF with Joseph-Form Covariance Projection**
   - State Vector: $\mathbf{x} = [p_{\text{East}}, p_{\text{North}}, v_{\text{fwd}}, v_{\text{lat}}, \psi, b_g]^T$.
   - Decoupled Position/Forward Velocity updates from Course Heading/Bias updates.
   - Numerically stable Joseph-form covariance projection guarantees $\mathbf{P}_{k|k} > 0$ and eliminates gyro bias divergence during stops.

2. **Confidence-Aware AI-EKF Fusion (Heteroscedastic Uncertainty)**
   - Random Forest Regressor trained across diversified multi-driver profiles (A, B, D, E) predicts forward velocity while extracting tree ensemble variance $\sigma_v^2(t)$.
   - Dynamically scales EKF process noise covariance $\mathbf{Q}_v(t)$ and blending weight $\alpha_v(t)$.

3. **Driver-Style-Adaptive Physical Constraints (Real NHC & Dynamic ZUPT)**
   - Actively executes Joseph-form Non-Holonomic Constraint (NHC) pseudo-measurement updates ($v_{\text{lat}} \approx 0$) with driver-adaptive measurement variance ($R_{\text{lat}} = 0.05^2$ for normal, $0.25^2$ for aggressive) to suppress lateral cross-track divergence during GNSS blackouts.
   - Dual-axis Zero-Velocity Updates (ZUPT) constrain both forward and lateral velocity when stopped.

4. **Continuous Speed-Weighted GPS Heading Ingestion**
   - Ingests native receiver `GPS ORIENTATION (°)` with continuous inverse-speed variance scaling $R_h(v)$, smoothly distrusting orientation jitter at low speeds.

5. **Multi-Sensor Predictive Context Layer & On-Device Android Engine**
   - Monitors ambient light and kinematic state to trigger pre-emptive blackout alerts before satellite fix loss.
   - Kotlin engine (`android_logger/`) runs the exported gradient-boosted model
     (`outputs/models/ondevice_model.json`, 43 KB, trained on 99,714 real windows with CAN
     speed labels), the same frame alignment and motion
     gate, and the same 6-state Joseph-form EKF. Feature-order and threshold parity with
     the Python pipeline is enforced by `tests/test_kotlin_parity.py`.
   - **Not yet verified:** floating-point parity of *compiled* Kotlin against
     `outputs/models/golden_vectors.json`. No Kotlin toolchain was available; the test
     checks feature order, thresholds and the exported model, not compiled output.

6. **Map matching to the OSM road graph** (`src/map_matching.py`)
   - HMM/Viterbi matching (Newson & Krumm 2009) of the open-loop blackout trajectory onto
     cached OSM road centrelines. Removes the cross-track component of DR drift, which is
     the component heading error produces. On the synthetic grid self-check: cross-track
     11.7 m → 2.0 m. Along-track error is *not* recoverable this way, and is not claimed.

---

## 📊 Results on real IO-VNBD

All numbers below come from **real IO-VNBD** (3.7 GB, 564 CSVs, `scripts/fetch_iovnbd.py`),
never from synthetic data. The synthetic generator is quarantined under
`tests/fixtures/` behind an import guard, and `get_real_iovnbd_benchmark_suite()` now
raises rather than silently falling back to it.

**Primary metric: blackout drift = exit error ÷ distance travelled with GNSS off**,
measured strictly open-loop before the first post-outage fix. The older headline divided
end-of-drive error by total distance, with GNSS restored and correcting the filter, so it
was not a dead-reckoning number at all.

### What the dataset actually contains

32 synchronised S-/V- pairs. 23 pass the quality gates (~169 min at 6000 samples/drive),
6 are flagged, 3 are too short to hold a 90 s outage. Flagged and dropped drives are named
in `outputs/metrics/real_iovnbd/`, because which data you discarded is part of the result:

| Drive | Why flagged |
| :--- | :--- |
| `Y1` | speed label and reference GNSS track disagree by 10.6% |
| `Vta1b`, `Vw16b` | exceed 6 m/s² longitudinal (Vw16b confirmed at 7.4 m/s² by the vehicle's own accelerometer) |
| `Vw1`, `Vw15` | parked recordings — 27 m and 9 m of path |
| `Vw9`, `Vw13`, `Vw17` | shorter than 60 s |

### Three unit mislabels in the published headers

Each was verified numerically against an independent column before being acted on
(`scripts/inspect_iovnbd_schema.py` reproduces all of it):

| Column | Header says | Actually is | Evidence |
| :--- | :--- | :--- | :--- |
| S- `GPS SPEED (Kmh)` | km/h | **m/s** | peaks at 26.83 while CAN reads 98.18 km/h on the same rows; ratio 1.018 |
| V- `Wheel Speed (rad/sec)` | rad/s | **km/h** | corr 0.999987 with CAN speed; implied "tyre radius" 0.2781 m = exactly 1/3.6 |
| S- `GPS SPEED` as a label | usable | **unusable** | step-quantised, glitches to 134 m/s² implied accel vs CAN's 6.6 |

The previous loader divided the first column by 3.6, making every speed label 3.6× too
small — which is why a motorway drive appeared to top out at 10.6 km/h. **Speed labels now
come from CAN**, and the loader cross-checks ∫v·dt against GPS path length (real drives
land at 0.05–1.2%).

### The dataset is not synchronised, and the gyro axes are not what the headers say

Two defects found this run, both in the published data, both verified numerically. Together
they were the largest single cause of poor results:

**1. The S- (phone) and V- (vehicle) files are misaligned in time**, despite living in a
directory called "Synchronised V abd S datasets". Cross-correlating the phone's vertical
gyro against the CAN bus yaw rate — the same physical quantity, two instruments — gives:

| Drive | Lag | Correlation at 0 lag | Correlation at best lag |
| :--- | ---: | ---: | ---: |
| S3a | +6.7 s | 0.200 | **0.974** |
| S2 | −8.7 s | — | 0.918 |
| S3c | −0.5 s | — | 0.996 |
| Vfa02 | −8.8 s | — | 0.174 |

Offsets run from −39 s to +14 s and differ per drive. Every model trained before this was
pairing IMU windows with labels from several seconds later. `estimate_sv_lag()` now
measures and corrects the offset on load.

**2. The gyroscope axis order does not match the accelerometer's.** Accelerometer and
GRAVITY both put vertical on Z (GRAVITY Z ≈ 9.806). The gyroscope does not: on every drive
tested, the channel labelled **"Pitch"** carries vehicle yaw. Magnitudes confirm it — CAN
yaw p99 = 0.5428 rad/s, "Pitch" p99 = 0.5504, "Yaw" p99 = 0.1072 — and the channel labelled
"Yaw" correlates **0.000** with the vehicle's actual turning.

Before the fix the phone's yaw rate was 1–14% of the vehicle's true yaw rate. After it,
the regression slope is **0.92–1.01 on every drive**, and integrated heading tracks the CAN
yaw rate to **1.0°–7.2° over 90 seconds** on the good drives. Heading was the dominant
error term identified by the ablation; this is what fixing it looks like.

A consequence worth stating: only **6 of 23** drives have a phone that actually tracks the
vehicle in yaw (correlation ≥ 0.55 after lag correction). The rest — mostly Driver E's `Vw`
series — do not, so their *heading* is unusable even though their *speed* data is fine.
Those two uses are now separated rather than conflated.

### Headline benchmark — 6 heading-reliable drives, 90 s blackout

| Configuration | Median blackout drift |
| :--- | :--- |
| Naive DR (with strapdown alignment) | 126.3% |
| **Fused, production** | **51.3%** |

Per drive:

| Drive | Naive | Fused | |
| :--- | ---: | ---: | :--- |
| S1 | 157.2% | 65.7% | |
| S2 | 241.1% | 27.8% | |
| S3a | 215.0% | 36.9% | 4.9% with the ML speed source |
| S3c | 44.8% | 132.6% | **fused worse than naive** |
| M | 92.0% | 32.1% | |
| Vw5 | 95.5% | 73.0% | |

**0 of 6 drives meet the ~10% target. Fused is worse than naive on 1 of 6** (`S3c`, whose
GNSS heading reference disagrees with both the gyro and the CAN bus by 267° over the
window — unresolved).

Across all 23 clean drives (including the 17 whose heading is unreliable): naive 86.9%,
fused 57.0%, worse than naive on 7.

### The speed-source ablation, and what it found

`scripts/speed_source_ablation.py` runs an identical EKF and blackout protocol, changing
only where forward speed comes from:

| Arm | Median blackout drift (σ = 0.2, as shipped) |
| :--- | :--- |
| A — ML regressor | 62.5% |
| B — last GNSS speed, held | **58.5%** |
| C — training-set mean | 60.8% |
| D — oracle (true CAN speed) | 64.1% |
| E — **oracle speed *and* oracle heading** | **39.9%** |

Two findings, and the second matters far more than the first.

**1. The filter was 28× overconfident.** The EKF was told to expect σ_v = 0.2 m/s. The
model's measured RMSE during real blackouts is 5.70 m/s. A filter that trusts a bad source
follows it off the road — that is the mechanism behind fused-worse-than-naive. With σ
calibrated to the measured error, the ML arm improves from 62.5% to 55.1% and stops being
a liability — in the final benchmark it edges ahead at 55.8% vs 57.0%. `hold_last` remains
the default because it is simpler and cannot fail catastrophically, and a ~1 pp median gap
is well inside the spread across 23 drives.

**2. Speed is not the bottleneck — heading is.** Perfect speed knowledge (arm D, 64.1%)
is *no better* than the ML model. Adding perfect heading (arm E) nearly halves the error to
39.9%. Effort spent on the speed regressor is effort spent on the wrong axis. And since
even perfect speed *and* heading leaves 39.9%, a third error source remains beyond both.

### Speed model, leave-one-driver-out, real CAN labels

| Held-out driver | MAE | R² | Constant-baseline MAE | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| A | 4.37 m/s | −0.156 | 6.94 | beats constant |
| B | 4.32 m/s | **0.507** | 6.55 | beats constant |
| E | 6.97 m/s | −0.098 | 8.45 | beats constant |

It now beats a constant predictor on every held-out driver — it did not before — but R² is
still negative on two of three. Top feature is `a_vert_rms` (importance 0.269): road
vibration, which genuinely encodes speed but transfers poorly across vehicles.

### Frame invariance — the headline evidence

IO-VNBD cannot demonstrate arbitrary phone orientation on its own: gravity per-axis std is
[0.010, 0.009, 0.00005] m/s², i.e. the phone was rigidly pre-aligned and effectively never
tilted. So `tests/test_rotation_invariance.py` manufactures the variation — 12 random SO(3)
rotations of the raw accelerometer, gyroscope and gravity streams, on a real drive:

| Measure | Result |
| :--- | :--- |
| Max deviation of any alignment channel | **1e-14 to 1e-16** (machine precision) |
| Blackout drift across 12 mountings | median 42.02%, min 42.02%, max 42.02% |
| Spread | **0.000 pp = 0.00% relative** (requirement: < 5%) |

That spread was 0.93% until this test caught a real defect: `run_fusion_pipeline` still
computed its standstill test from the variance of raw `acc_x`/`acc_y`, which changes when
the phone is rotated. Every alignment channel matched to 1e-14 while end-to-end drift
still moved by 2.6 pp — the discrete standstill decision was amplifying a frame-dependent
input. Replacing it with the variance of horizontal specific force about gravity made
end-to-end drift *exactly* invariant.

### Gravity estimator, validated on real braking

The 0.2 Hz low-pass absorbed sustained deceleration into its own gravity estimate. Measured
over **32 real brake events across 5 drives**, scored against the vehicle's own longitudinal
accelerometer:

| Gravity estimator | Deceleration retained |
| :--- | :--- |
| 0.2 Hz low-pass (old, still reachable via `gravity_mode="lowpass"`) | 19% |
| Gyro-propagated Mahony (new default) | **61%** — 3.3× better |

### Gate asymmetry

A false "stopped" during a blackout is unrecoverable; the same error with GNSS up is
corrected on the next fix. The gate therefore runs different thresholds per regime.
On a marginal drive: 306 STATIONARY samples with GNSS available, **4** during a blackout.
On a silent 12 m/s cruise through a 30 s outage, false-stationary fell from **360 m to 4.8 m**
once a speed-corroboration precondition was added — an IMU-only stillness test cannot
separate "parked" from "gliding smoothly", and in a blackout nothing corrects it.

---

## 📱 The Android app

`android_logger/` is now a live dead-reckoning HUD, not just a CSV logger. It runs
`OnDeviceInferenceEngine` on the same samples it records, so the algorithm's belief is on
screen in real time — previously the engine existed but nothing ever called it.

Built for a phone in a windscreen cradle, at speed, at night:

- **One dominant number** — current position uncertainty in metres, colour-graded
  green/amber/red, readable at a glance without focusing.
- **Live track view** (`TrackView.kt`) — the estimated path drawn top-down and auto-scaled.
  The dead-reckoned section is drawn in amber and the GNSS-tracked section in blue, so
  divergence is visible as it happens. Plain Canvas, no map SDK: no API key, no network,
  works in a basement — which is where the system is supposed to earn its keep.
- **SIMULATE GNSS BLACKOUT** — withholds fixes from the filter while the phone keeps
  receiving them. This is the demo: press it and watch the track come off the road, release
  it and see the exit error. No tunnel required.
- **Motion state in plain words** — "IN VEHICLE — MOVING" / "STATIONARY" / "PHONE HANDLED",
  each with the reason underneath ("Gravity direction is moving — speed estimate vetoed").
  Status is carried by text *and* colour, never colour alone.
- **Sensor chips** — which sensors this specific handset actually has. Absence is a real
  result on any given phone, so it is shown rather than assumed.
- Screen kept awake, portrait-locked, dark ground throughout.

### Safety sharing and offline navigation (Flutter app)

The Flutter app (`flutter_app/`) adds the part that makes a dead-reckoned position useful
to somebody other than the person holding the phone.

- **Absolute position from a relative track.** The pipeline works in local metres from
  wherever the session started, which cannot be given to anyone. `lib/safety/position_report.dart`
  pins that frame to the Earth using the last real GNSS fix and reports latitude, longitude,
  **an uncertainty circle** (fix accuracy and accumulated drift, combined in quadrature)
  and **how old the fix is**.
- **It refuses to invent a coordinate.** With no fix ever obtained, `isShareable` is false
  and the message says "position unknown" plus the distance travelled. Sending the session
  origin would send a responder somewhere wrong, which is worse than sending nothing.
- **SMS, because it survives what this system exists for.** Data is the first thing to go
  in a basement, a tunnel or a crowd. Direct send via `SmsManager`, falling back to a
  prefilled messaging app when `SEND_SMS` is refused, so it never ends with nothing.
  Anything that cannot go out now is queued and flushed when a fix or signal returns.
- **Offline basemap, optional.** `flutter_app/tool/fetch_tiles.py` bakes OpenStreetMap
  tiles for ONE demo area into the APK; nothing is fetched at run time. With no tiles the
  map draws a metre grid, so no build depends on the fetch having been run.
- **Destination guidance, honestly scoped.** Long-press the map to set a destination and
  get distance plus relative bearing. Not turn-by-turn: there is no road graph on the
  device, so a route down streets would be a drawing rather than a route.

---

## ⚠️ Limitations

Stated plainly, because each one bounds what the numbers above mean.

- **Training data is pre-aligned.** Every IO-VNBD drive has the phone fixed to the vehicle
  frame. Rotation invariance is demonstrated by synthetic rotation of real data, not by
  real varied mounting. No public dataset we found has vehicle IMU with varied phone
  placement and usable ground truth; SHL has the placements but is pedestrian/transit.
- **No Indian road data.** Validation is UK (Coventry). Indian traffic, road surface and
  driving style are unrepresented.
- **No real tunnel or basement recordings.** Blackouts are simulated by withholding GNSS
  from a drive recorded in the open. Real multipath and reacquisition behaviour are absent.
- **Barometer is not an EKF state.** It is logged by the Android app and converted to
  relative altitude, but multi-level parking is not yet solved.
- **Driver coverage is skewed.** Of 23 clean drives, 17 are Driver E and 1 is Driver B;
  Driver D is entirely flagged out. Leave-one-driver-out therefore rests on three groups.
- **Only 6 drives have usable heading.** The other 17 have a phone that does not track the
  vehicle in yaw even after lag correction, so the blackout benchmark rests on six drives —
  five of them from two drivers.
- **S3c is unexplained.** Its GNSS heading reference disagrees with both the phone gyro and
  the CAN yaw rate by 267° over the test window, and it is the one drive where fused is
  worse than naive. Not diagnosed.
- **The Android UI has never been compiled.** No Android toolchain in this environment. The
  layout, theme, drawables and Kotlin are written and internally consistent, but the first
  `./gradlew assembleDebug` may still surface errors. Build it before relying on it
  for a demo.
- **Compiled-Kotlin parity is unverified.** No Kotlin toolchain was available. The parity
  test checks feature order, thresholds and the exported model against golden vectors —
  not compiled floating-point output.
- **2 of 23 drives meet the 10% target.** The system does not yet do what the problem
  statement asks.
- **The pedestrian drift figure is an allowance, not a measurement.** The safety
  uncertainty circle grows at 20% of distance walked. Published step-length methods land
  at 5-15%; this is set deliberately above them because an over-large circle costs a
  searcher time while an over-small one sends them to the wrong place. It has not been
  measured on this system and must not be quoted as an accuracy.
- **Emergency contacts do not persist.** They live in memory for the session, like the
  online calibration. Both need the same one-line swap to a real store.

---

## 📁 Repository Structure

```
SIH/
├── android_logger/              # Android sensor logging & on-device Kotlin inference app
│   ├── app/src/main/java/       # Native MainActivity and OnDeviceInferenceEngine
│   └── build.gradle.kts
├── data/
│   └── samples/                 # Sample drives for immediate execution
├── docs/
│   └── PROPOSAL_NARRATIVE.md    # Comprehensive technical proposal narrative for jury
├── outputs/
│   ├── figures/                 # High-resolution benchmark plots (PNG)
│   ├── metrics/                 # Benchmark JSON and Markdown summary reports
│   └── models/                  # Trained speed regressor and embedded rule schemas
├── scripts/
│   ├── 01_run_naive_baseline.py # Day 1: Naive double-integration baseline
│   ├── 02_train_and_fuse.py     # Day 2: AI speed training & 6-state EKF fusion
│   ├── 03_evaluate_and_benchmark.py # Day 3: Multi-driver benchmark evaluation
│   ├── diagnostic_ab_test.py    # A/B diagnostic progression tool (NHC & Heading evaluation)
│   └── run_all.py               # Master pipeline runner with 2-stage verification
├── src/                         # Core Python library
│   ├── data_loader.py           # IO-VNBD data loader with SHA256 integrity checks
│   ├── feature_engineering.py   # 65-dimensional sliding-window feature extractor
│   ├── speed_model.py           # Random Forest ensemble with uncertainty quantification
│   ├── fusion_ekf.py            # 6-State Confidence-Aware Kinematic EKF with NHC
│   ├── metrics.py               # Standardized navigation benchmark metrics
│   └── visualizer.py            # Trajectory & error plotting utilities
├── tests/
│   └── test_context_layer.py    # Unit tests for predictive context engine & NHC
├── scripts/fetch_iovnbd.py      # Fetches the REAL IO-VNBD dataset (3.7 GB via Git LFS)
├── src/iovnbd_loader.py         # Loader written against the real schema; CAN speed labels
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/Abhijeet-ojha/SIH.git
cd SIH
pip install -r requirements.txt
```

### 2. Run the Full Pipeline & 2-Stage Verification
```bash
python scripts/run_all.py
```

**Fetch the real dataset first — there is no synthetic fallback any more:**

```bash
python scripts/fetch_iovnbd.py            # ~3.7 GB, 564 CSVs, Git LFS
python scripts/fetch_iovnbd.py --verify   # confirm no unresolved LFS pointers
```

The benchmark loader raises if IO-VNBD is absent rather than quietly substituting
synthetic drives. The generator that produced those drives now lives in
`tests/fixtures/synthetic_drives.py` behind an import guard that refuses to load outside
`tests/`. Synthetic vibration comes from a hand-written noise model and cannot exercise the
vibration-to-speed coupling this method depends on, so any metric derived from it was
measuring a simulator.

Then, in order:

```bash
python scripts/inspect_iovnbd_schema.py    # print the real schema before trusting anything
python scripts/train_real_iovnbd.py        # train on CAN speed labels, leave-one-driver-out
python scripts/speed_source_ablation.py    # which speed source actually helps
python scripts/benchmark_real_iovnbd.py    # final per-drive blackout drift
```

### 3. Run the gates
```bash
python -m unittest discover -s tests -t .   # 44 tests, ~3 min
```
Each phase of the rebuild has a falsifiable gate: loader integrity (timeline not
fabricated, speed units cross-checked against GPS path length), naive-baseline honesty,
frame invariance under rotation, shake rejection, and Kotlin parity. See
[docs/REBUILD_PROMPT.md](docs/REBUILD_PROMPT.md) for what each one is defending against.

This executes:
- **Day 1 Pipeline**: Naive Dead Reckoning Baseline ($O(t^3)$ drift proof)
- **Day 2 Pipeline**: Confidence-Aware Random Forest training & EKF GNSS Fusion
- **Day 3 Pipeline**: Full 6-Drive Multi-Driver Evaluation
- **Stage A Verification**: Proves confidence-scaling code path is actively executing
- **Stage B Verification**: Asserts 100% bitwise determinism via back-to-back SHA256 hash matching

---

## 📜 Citation & Dataset
This project uses the **IO-VNBD (Input-Output Vehicle Navigation Benchmark Dataset)**:
- *Onyekpe, U. et al.* (Coventry University, UK).
- Dataset repository: [github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)

---

## 📄 License
MIT License. Created for Smart India Hackathon (SIH) 2026.
