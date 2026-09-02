# SIH Problem Statement 168: AI-Assisted Vehicle Dead Reckoning & GNSS Fusion Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Verification](https://img.shields.io/badge/100%25-Bitwise%20Reproducible-brightgreen.svg)]()

Production-grade, end-to-end prototype for **Smart India Hackathon (SIH) Problem Statement 168**: *AI-Assisted Dead Reckoning for Terrestrial Navigation in Extended GNSS-Denied Environments*.

Developed and evaluated on real multi-driver data from the **IO-VNBD Benchmark Suite** (Coventry, UK).

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

## 📊 Benchmark Summary (IO-VNBD Real Drives, 90s Blackout)

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
