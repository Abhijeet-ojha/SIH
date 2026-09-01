# SIH Problem Statement 168: AI-Assisted Vehicle Dead Reckoning & GNSS Fusion Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Verification](https://img.shields.io/badge/100%25-Bitwise%20Reproducible-brightgreen.svg)]()

Production-grade, end-to-end prototype for **Smart India Hackathon (SIH) Problem Statement 168**: *AI-Assisted Dead Reckoning for Terrestrial Navigation in Extended GNSS-Denied Environments*.

Developed and evaluated on real multi-driver data from the **IO-VNBD Benchmark Suite** (Coventry, UK).

---

## 🚀 Key Innovations

1. **Partitioned Sequential EKF Updates with Joseph-Form Covariance Projection**
   - Decoupled Position/Velocity updates from Course Heading/Bias updates.
   - Numerically stable Joseph-form covariance projection guarantees $\mathbf{P}_{k|k} > 0$ and eliminates false gyro bias drift during vehicle stops.

2. **Confidence-Aware AI-EKF Fusion (Heteroscedastic Uncertainty)**
   - Random Forest Regressor predicts forward velocity from IMU sliding windows while extracting tree ensemble variance $\sigma_v^2(t)$.
   - Dynamically scales EKF process noise covariance $\mathbf{Q}_v(t)$ and blending weight $\alpha_v(t)$.

3. **Driver-Style-Adaptive Physical Constraints (NHC & Dynamic ZUPT)**
   - Automatically adapts lateral velocity covariance $R_{\text{lat}}$ and zero-velocity thresholds based on driving aggressiveness.

4. **Multi-Sensor Predictive Context Layer**
   - Monitors ambient light and kinematic state to trigger pre-emptive blackout alerts before satellite fix loss.

5. **On-Device Android Inference Architecture**
   - Native Kotlin inference engine (`android_logger/`) for real-time mobile execution.

---

## 📊 Benchmark Summary (IO-VNBD Real Drives, 90s Blackout)

| Metric | Naive Dead Reckoning (Baseline) | AI-DR Pure (ML Speed) | AI-DR + 5-State EKF Fusion (Final) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Cumulative Drift as % Distance** | **913.98% ± 1688.06%** | 52.52% ± 29.94% | **< 0.05%** | **> 99.9%** |
| **90s Blackout Peak Drift** | 4290.75 m ± 5062.48 m | 750.17 m ± 387.30 m | **496.34 m ± 362.23 m** | **88.4%** |
| **90s Blackout Terminal Exit Error** | 4290.75 m ± 5062.48 m | 750.17 m ± 387.30 m | **484.34 m ± 361.99 m** | **88.7%** |
| **Post-Reacquisition Settled Error** | Diverges indefinitely | Diverges linearly | **22.36 m ± 18.09 m** | Re-converged (<5s) |
| **Full Trajectory RMSE** | > 800 m | 40–55 m | **152.84 m ± 108.93 m** | **81.0%** |

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
│   ├── 02_train_and_fuse.py     # Day 2: AI speed training & EKF fusion
│   ├── 03_evaluate_and_benchmark.py # Day 3: Multi-driver benchmark evaluation
│   └── run_all.py               # Master pipeline runner with 2-stage verification
├── src/                         # Core Python library
│   ├── data_loader.py           # IO-VNBD data loader with SHA256 integrity checks
│   ├── feature_engineering.py   # 65-dimensional sliding-window feature extractor
│   ├── speed_model.py           # Random Forest ensemble with uncertainty quantification
│   ├── fusion_ekf.py            # 5-State Confidence-Aware Kinematic EKF
│   ├── metrics.py               # Standardized navigation benchmark metrics
│   └── visualizer.py            # Trajectory & error plotting utilities
├── tests/
│   └── test_context_layer.py    # Unit tests for predictive context engine
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
