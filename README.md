# SIH Problem Statement 168: AI-Assisted Vehicle Dead Reckoning & GNSS Fusion Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Verification](https://img.shields.io/badge/100%25-Bitwise%20Reproducible-brightgreen.svg)]()

Production-grade, end-to-end prototype for **Smart India Hackathon (SIH) Problem Statement 168**: *AI-Assisted Dead Reckoning for Terrestrial Navigation in Extended GNSS-Denied Environments*.

Developed and evaluated on real multi-driver data from the **IO-VNBD Benchmark Suite** (Coventry, UK).

---

## 🚀 Key Innovations

> **Read this first.** Items 1–5 below describe what the system *does*. Items with a
> measured contribution have it stated inline; where the measurement says a component
> contributes ~nothing, that is stated too. See
> [Measured contributions](#-measured-contributions-including-the-negative-results).

0. **Frame-invariant motion estimation with a phone-handling gate** *(the load-bearing one)*
   - Nothing downstream sees raw `acc_x/y/z` or `gyro_z`. Gravity is estimated per sample,
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
     (`outputs/models/ondevice_model.json`, 44 KB), the same frame alignment and motion
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

## 📊 Benchmark Summary

> **Primary metric: open-loop blackout drift = exit error ÷ distance travelled with GPS
> off.** This is the dead-reckoning number. A previous version of this table led with
> "< 0.05% cumulative drift", which divided the error at the *end of the drive* by the
> *total* distance — but GPS is back on and correcting the filter by then, so it measured
> "does the EKF track GPS while GPS is available". Both numbers are still reported;
> the open-loop one leads.

**The numbers below are regenerated by `scripts/03_evaluate_and_benchmark.py` into
`outputs/metrics/benchmark_summary.md`. Do not hand-edit them here.**

Current status, run against `data/samples/` (**synthetic stand-in data — not a benchmark**,
because `data/IO-VNBD-repo/` is gitignored and must be acquired separately; see
[docs/DATASETS.md](docs/DATASETS.md)):

| Metric | Naive DR (baseline) | AI-DR Pure | AI-DR + 6-State EKF |
| :--- | :--- | :--- | :--- |
| **Blackout drift % (PRIMARY)** | 46.34% ± 39.44% | 116.26% ± 78.15% | **79.43% ± 43.39%** |
| 90 s blackout terminal exit error | 530.19 m ± 419.74 m | 1030.88 m ± 308.97 m | 703.42 m ± 82.37 m |
| Post-reacquisition settled error | N/A | N/A | 1.34 m ± 0.91 m |
| Full trajectory RMSE | 640.42 m ± 470.81 m | 1190.1 m ± 324.36 m | 219.58 m ± 38.97 m |

**0 of 4 drives meet the ~10% target, and the fused system is currently worse than the
naive baseline during the blackout.** That is the real state of the system on this data.
The speed model scores R² = −3.12 on held-out synthetic drives, which is the direct cause;
retraining on real IO-VNBD (and on SHL for the motion gate) is the next step, not more
filter tuning.

## 📉 Measured contributions, including the negative results

From `outputs/metrics/ml_experiments/phase15_system_ablation.csv`:

| Configuration | Blackout exit error |
| :--- | :--- |
| ML + EKF, constant σ | 414.41 m |
| + calibrated heteroscedastic uncertainty | 414.09 m |
| + NHC | 413.90 m |

Innovations 2 and 3 above are worth **0.5 m out of 414** — essentially nothing. Likewise
`model_benchmark_summary.csv` shows RF 414.7, XGBoost 415.2, 1D-CNN 418.7: when every
architecture lands in the same place, the architecture is not what produces the number.
These are reported rather than buried because they are what the evidence says. Both are due
for re-measurement now that the body frame is estimated rather than assumed — NHC in
particular could not have helped when it was being applied in the wrong frame.

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
├── generate_synthetic_drives.py # Generates the SYNTHETIC data/samples stand-in (does NOT download IO-VNBD)
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

A fresh clone runs immediately: `data/IO-VNBD-repo/` is gitignored, so the pipeline falls
back to the synthetic drives in `data/samples/` and prints a `SYNTHETIC DATA — NOT A
BENCHMARK RESULT` banner. **Anything produced in that mode describes a simulator, not a
vehicle.** For real results, acquire IO-VNBD — see [docs/DATASETS.md](docs/DATASETS.md).
Regenerate the stand-in drives with `python generate_synthetic_drives.py --samples`.

### 3. Run the gates
```bash
python -m unittest discover -s tests     # 29 tests, ~2 min
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
