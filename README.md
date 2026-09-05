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
   - Native Kotlin inference engine (`android_logger/`) for real-time mobile execution.

---

## Historical vehicle benchmark summary

> These are legacy figures retained from the earlier vehicle pipeline. The
> headline cumulative-drift figure includes GNSS recovery and must not be
> presented as open-loop blackout accuracy. Later development identified speed
> unit and phone/vehicle time-alignment issues. This table does not validate
> the indoor walking prototype or establish current navigation accuracy.

| Metric | Naive Dead Reckoning (Baseline) | AI-DR Pure (ML Speed) | AI-DR + 6-State EKF Fusion (Final) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Cumulative Drift as % Distance** | **363.67% ± 557.06%** | 48.52% ± 22.70% | **< 0.05%** | **> 99.9%** |
| **90s Blackout Peak Drift** | 2285.58 m ± 2009.71 m | 631.46 m ± 366.63 m | **425.36 m ± 325.07 m** | **81.4%** |
| **90s Blackout Terminal Exit Error** | 2285.58 m ± 2009.71 m | 631.46 m ± 366.63 m | **412.96 m ± 316.29 m** | **81.9%** |
| **Post-Reacquisition Settled Error** | Diverges indefinitely | Diverges linearly | **20.54 m ± 16.80 m** | Re-converged (<5s) |
| **Full Trajectory RMSE** | > 800 m | 40–55 m | **132.51 m ± 97.97 m** | **83.4%** |

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
├── download_dataset.py          # Script to clone/download the complete IO-VNBD dataset
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
