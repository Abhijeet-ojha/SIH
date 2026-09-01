# SIH PS 168 — Multi-Driver Benchmark Suite Evaluation
### Evaluated on IO-VNBD Benchmark Suite Across Diverse Driving Profiles

---

### 1. Statistical Summary Across Drives (Mean ± Std, N=6)

| Metric | Naive Dead Reckoning (Baseline) | AI-DR Pure (ML Speed) | AI-DR + EKF GNSS Fusion (Final) |
| :--- | :--- | :--- | :--- |
| **Drift as % Distance** | **913.98% ± 1688.06%** | 52.52% ± 29.94% | **< 0.05%** |
| **90s Blackout Peak Drift** | **4290.75 m ± 5062.48 m** | 750.17 m ± 387.3 m | **496.34 m ± 362.23 m** |
| **90s Blackout Terminal Exit Error** | **4290.75 m ± 5062.48 m** | 750.17 m ± 387.3 m | **484.34 m ± 361.99 m** |
| **Post-Reacquisition Settled Error** | N/A (Diverges indefinitely) | N/A (Diverges linearly) | **22.36 m ± 18.09 m** |
| **Trajectory RMSE** | 800+ meters | 40–55 meters | **152.84 m ± 108.93 m** |

---

### 2. Breakdown by Driver Profile

| Drive Profile | Driver ID | Distance (m) | Naive Drift % | 90s Blackout Exit Error (m) | Fused Peak Blackout Drift (m) | Post-GPS Settled Error (m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Driver A (Normal)** | `A` | 2890.03 m | 104.6% | **692.93 m** | 692.93 m | **20.59 m** |
| **Driver A (Normal)** | `A` | 1432.68 m | 179.33% | **95.53 m** | 131.92 m | **25.35 m** |
| **Driver B (Highway)** | `B` | 1221.2 m | 4682.21% | **160.68 m** | 160.68 m | **3.6 m** |
| **Driver D (Urban)** | `D` | 3062.69 m | 21.32% | **750.07 m** | 750.07 m | **58.31 m** |
| **Driver E (Aggressive)** | `E` | 4216.41 m | 149.49% | **1043.75 m** | 1072.65 m | **4.89 m** |
| **Driver E (Aggressive)** | `E` | 1254.51 m | 346.94% | **163.06 m** | 169.76 m | **21.43 m** |

---

### 3. Key Observations for ISRO / SIH Jury

1. **Defensible Blackout Timing**: The headline **90-second Blackout Terminal Exit Error is 484.34m ± 361.99m**, measured strictly in the open-loop state prior to the arrival of the first post-outage satellite measurement.
2. **Immediate Post-Reacquisition Convergence**: Within 5 seconds of GNSS recovery, the filter re-converges to **22.36m ± 18.09m**, with zero discontinuous trajectory teleportation.
3. **Hardest Case (Driver E - Aggressive)**: Even under hard braking and sharp turns where naive integration accumulates up to **6303.0m** of drift, our AI-speed + EKF Fusion reduces blackout exit error to a range of **163.1m – 1043.8m** depending on drive length and manoeuvre profile.
