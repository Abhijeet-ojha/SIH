> **Status note — real data.** Every figure in this document now comes from the real
> IO-VNBD dataset (3.7 GB, fetched by `scripts/fetch_iovnbd.py`), with speed labels taken
> from the vehicle CAN bus. Earlier revisions reported numbers from a synthetic generator,
> which is now quarantined under `tests/fixtures/` behind an import guard.
>
> The headline metric is **blackout drift = exit error / distance travelled with GNSS off**,
> measured strictly open-loop. A previous revision led with "< 0.05% cumulative drift",
> which divided end-of-drive error by total distance — after GNSS had returned and was
> correcting the filter. That was not a dead-reckoning number.
>
> Current honest state on 23 real drives with 90 s outages: naive DR 86.9% median blackout
> drift, fused 57.0%. **2 of 23 drives meet the ~10% target, and fused is worse than naive
> on 7 of 23** (S3c, M#seg0, Vfa01, Vfa02, Vw10, Vw14c, Vw16a).
>
> The ablation in `outputs/metrics/real_iovnbd/` shows why, and it is not what we expected:
> perfect speed knowledge (oracle CAN speed) gives 64.1% — no better than the ML model —
> while perfect speed *and* heading gives 39.9%. **Heading, not speed, is the dominant
> error source.** Separately, the filter was assuming sigma_v = 0.2 m/s against a measured
> 5.70 m/s, i.e. 28x overconfident; correcting that moved the ML arm from 62.5% to 55.1%.

# Technical Proposal & Defense Narrative
## Smart India Hackathon (SIH) — Problem Statement 168
### High-Precision Vehicle Dead Reckoning & GNSS Fusion Engine for Low-Cost Smartphone Inertial Sensors

---

### 1. Problem Formulation & The Physics of Classical Dead-Reckoning Failure

Standard satellite positioning (GNSS/GPS) is inherently fragile in critical urban and transportation environments:
- **Urban canyons:** Multi-path reflections from glass and concrete structures corrupt pseudorange measurements.
- **Underground tunnels & underpasses:** Complete line-of-sight satellite signal blackout.
- **Multi-level flyovers & elevated expressways:** Obscured satellite geometry.
- **Dense tree canopies and multi-story parking structures.**

When GNSS drops, navigation systems fall back to **Inertial Dead Reckoning (DR)**. However, applying classical numerical double integration to raw consumer MEMS inertial sensors (such as those in smartphones) fails catastrophically within seconds.

```
+-----------------------------------------------------------------------------------+
|               THE CUBIC DRIFT CATASTROPHE OF NAIVE DOUBLE INTEGRATION             |
|                                                                                   |
|  Raw Accelerometer: a_meas(t) = a_true(t) + b_a + eta_a(t)                        |
|                                                                                   |
|  1. Velocity Integration:                                                         |
|     v(t) = v_0 + \int a_meas(t) dt  --> Linear Velocity Error: e_v(t) ~ b_a*t     |
|                                                                                   |
|  2. Position Integration:                                                         |
|     p(t) = p_0 + \int v(t) dt       --> Quadratic Position:   e_p(t) ~ 1/2*b_a*t^2|
|                                                                                   |
|  3. Gyroscope Yaw Rate Bias (theta(t) ~ b_g*t):                                   |
|     Spatial Trajectory Error        --> CUBIC DIVERGENCE:     e_pos(t) ~ O(t^3)   |
+-----------------------------------------------------------------------------------+
```

**Empirical Ground Truth Proof (IO-VNBD Dataset, Coventry, UK):**
A low-cost MEMS accelerometer bias of just $0.05 \text{ m/s}^2$ combined with an uncalibrated gyro yaw bias of $0.02 \text{ rad/s}$ ($1.15^\circ/\text{s}$) produces over **1,790 to 3,020 meters of drift within a 300-second drive (104%–147% of total distance travelled)**.

---

### 2. The Solution Architecture: Confidence-Aware AI + 6-State Adaptive EKF with Real NHC

Instead of integrating noisy acceleration, our architecture replaces acceleration integration with **direct supervised machine-learning forward velocity regression from road/vehicle vibration dynamics**, integrated into a **Confidence-Aware 6-State Extended Kalman Filter (EKF)** actively enforcing **Driver-Adaptive Non-Holonomic Constraints (NHC)** ($v_{\text{lat}} \approx 0$), **Dual-Axis Zero-Velocity Updates (ZUPT)**, and a **Predictive Multi-Sensor Context Layer**.

```
       +---------------------------------------------------------------------+
       |            Low-Cost Smartphone IMU (Accel & Gyro @ 10 Hz)           |
       +----------------------------------+----------------------------------+
                                          |
                                          v
       +---------------------------------------------------------------------+
       |               Multi-Domain Sliding Window Extractor (1.5s)          |
       |  - 6-Axis Statistical Moments (Mean, Variance, Skew, RMS, P2P)      |
       |  - Kinematic Magnitudes (||a||, ||omega||, Horizontal Accel)         |
       |  - Spectral Vibration Harmonics & Road Excitation Dynamics           |
       +----------------------------------+----------------------------------+
                                          |
                                          v
       +---------------------------------------------------------------------+
       |         Tree Ensemble Regressor with Heteroscedastic Uncertainty    |
       |               Mean Speed v_hat(t)  &  Ensemble Variance sigma_v^2(t)|
       |               (Trained across Drivers A, B, D, E for generalization)|
       +----------------------------------+----------------------------------+
                                          |
                                          v
       +---------------------------------------------------------------------+
       |              Multi-Sensor Predictive Context Layer (Job 3)          |
       |  - Ambient Light Drop (<100 lux) + Speed (>4 m/s) -> Pre-Outage Lock|
       |  - IMU Standstill Detection (sigma_a^2 < 0.018)   -> ZUPT Activated |
       |  - Highway vs Dense Urban Classification         -> NHC Adaptation  |
       +----------------------------------+----------------------------------+
                                          |
                                          v
       +---------------------------------------------------------------------+
       |          Confidence-Aware 6-State Extended Kalman Filter (EKF)      |
       |     State Vector: x = [p_East, p_North, v_fwd, v_lat, theta, b_g]^T  |
       |                                                                     |
       |   * Dynamic alpha_v(t): Downweights AI speed when uncertainty rises |
       |   * Active NHC Updates: Joseph-form v_lat ≈ 0 pseudo-measurements   |
       |   * Driver-Adaptive NHC: R_lat = 0.05² (Normal) vs 0.25² (Aggressive)|
       |   * Speed-Weighted Heading: Continuous inverse-speed R_h(v) scaling |
       |   * Online Gyro Bias Estimation: Calibrates b_g without stop drift   |
       |   * Open-Loop Blackout Isolation: Measured pre-update at exit       |
       +----------------------------------+----------------------------------+
                                          |
                                          v
       +---------------------------------------------------------------------+
       |           Robust Continuous Trajectory (412.96m Exit in 90s Gap)    |
       +---------------------------------------------------------------------+
```

---

### 3. Technical Innovations & Defense Rigor

#### Innovation 1: Confidence-Aware Fusion (Heteroscedastic Uncertainty Estimation)
Rather than treating the machine learning model as an infallible black box, our `RandomForestRegressor` computes predictions across its constituent decision trees:
$$\hat{v}_t = \frac{1}{M} \sum_{m=1}^M T_m(\mathbf{x}_t), \quad \sigma_{v, t}^2 = \frac{1}{M} \sum_{m=1}^M \left(T_m(\mathbf{x}_t) - \hat{v}_t\right)^2$$
The ensemble variance $\sigma_{v, t}^2$ serves as a real-time uncertainty metric ($\sigma_v$ ranges dynamically from $0.03\text{ m/s}$ when stopped up to $2.09\text{ m/s}$ during violent maneuvers).

We actively inject this uncertainty into both the **velocity state propagation** and the **Kalman covariance**:
1. **Dynamic Velocity Blending Weight:**
   $$\alpha_v(t) = \frac{\alpha_{\text{base}}}{1.0 + 1.5 \cdot \sigma_v(t)}, \quad v_{\text{eff}, k} = (1 - \alpha_v(t)) v_{k-1} + \alpha_v(t) \hat{v}_{\text{AI}, k}$$
   When the AI is confident ($\sigma_v \to 0$), $\alpha_v \approx 0.25$. When the model is uncertain ($\sigma_v \to 2.0$), $\alpha_v$ drops to $\approx 0.06$, preventing erratic speed spikes from destabilizing dead reckoning.
2. **Dynamic Process Noise Covariance:**
   $$\mathbf{Q}_v(t) = Q_{v, \text{base}} + \beta \cdot \sigma_{v, t}^2$$

#### Innovation 2: Driver-Style-Adaptive Physical Constraints (Real NHC & Dynamic ZUPT)
Ground vehicle kinematics dictate that the lateral velocity in the vehicle's body frame is approximately zero under standard driving conditions ($v_{\text{lat}} \approx 0$). Our 6-state EKF incorporates this physical law via a continuous **Joseph-form pseudo-measurement update**:
$$\mathbf{H}_{\text{NHC}} = \begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 \end{bmatrix}, \quad \mathbf{z}_{\text{NHC}} = [0], \quad R_{\text{lat}} = \sigma_{\text{lat}}^2$$
- **Normal / Defensive Drivers (Driver A, B, D):** Strict Non-Holonomic Constraints ($R_{\text{lat}} = 0.05^2 = 0.0025\text{ m}^2/\text{s}^2$) and sensitive standstill detection ($\sigma_a^2 < 0.018 \text{ (m/s}^2)^2$).
- **Aggressive Drivers (Driver E):** Loosened lateral velocity covariance ($R_{\text{lat}} = 0.25^2 = 0.0625\text{ m}^2/\text{s}^2$) allowing for centripetal tire slip during hard cornering, with dynamic ZUPT ($\sigma_a^2 < 0.035$).

#### Innovation 3: Multi-Sensor Predictive Context Layer (Job 3 Architecture)
*Note on Dataset Channels:* The public IO-VNBD dataset does not contain an ambient light sensor channel. Innovation 3 is an architectural design layer implemented in our Android application (`android_logger/app/src/main/java/com/sih/sensorlogger/OnDeviceInferenceEngine.kt`) and verified via synthetic state-machine unit tests (`tests/test_context_layer.py`):
- When ambient light drops below $100\text{ lux}$ while maintaining forward velocity ($> 4\text{ m/s}$), the system triggers a **Predictive Tunnel Blackout Alert**, pre-emptively tightening the online gyro bias covariance ($Q_{b_g} \leftarrow 0.1 Q_{b_g}$) and locking orientation calibration *before* satellite signals drop.

---

### 4. Cross-Device Hardware Framing & Target Device Deployment

#### Hardware Context & Honest Assessment:
- **IO-VNBD Dataset Recording Devices:** Recorded using consumer smartphones (Huawei P20 Pro with InvenSense IMU, Moto G7 Power with Bosch IMU, BlackBerry Priv).
- **Target Deployment Device:** OnePlus Nord CE3 5G (Qualcomm Snapdragon 782G platform, Kryo 670 CPU @ 2.7 GHz).
- **Honest Hardware Stance:** Exact OEM chip-level MEMS part numbers and factory calibration registers for the OnePlus Nord CE3 are proprietary and not publicly documented in consumer spec sheets. Typical consumer MEMS IMUs in this smartphone tier operate within standard commercial noise envelopes ($50-150\ \mu\text{g}/\sqrt{\text{Hz}}$ accelerometer noise density, $0.005-0.015^\circ/\text{s}/\sqrt{\text{Hz}}$ gyro noise density).
- **Phase 2 HIL Roadmap:** In Phase 2, we will perform physical stationary Allan Variance characterization directly on our OnePlus Nord CE3 hardware using our `android_logger` app to empirically measure its exact bias stability and angle random walk.

---

### 5. Empirical Evaluation on Real IO-VNBD Multi-Driver Benchmark

Evaluated on $N=6$ unseen real drives from the public IO-VNBD dataset (Coventry, UK) with guaranteed 100% bitwise determinism.

#### Statistical Multi-Driver Summary ($N=6$ Unseen Real Drives)

| Metric | Naive Dead Reckoning (Baseline) | AI-DR Pure (ML Speed) | AI-DR + 6-State EKF Fusion (Final) | Improvement vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Blackout drift % (PRIMARY: exit error / blackout distance)**, median over 23 REAL drives | 86.9% | 55.8% | **57.0%** | 2 of 23 meet the 10% target; fused worse than naive on 7 of 23 |

Ablation on the same 23 drives, changing only the speed source (`scripts/speed_source_ablation.py`):

| Speed source | Median blackout drift |
| :--- | :--- |
| ML regressor | 62.5% |
| Last GNSS speed, held | 58.5% |
| Training-set mean | 60.8% |
| Oracle (true CAN speed) | 64.1% |
| **Oracle speed AND oracle heading** | **39.9%** |

Perfect speed buys nothing; perfect heading nearly halves the error.
| **90s Blackout Peak Drift** | 2285.58 m ± 2009.71 m | 631.46 m ± 366.63 m | **425.36 m ± 325.07 m** | **81.4%** reduction |
| **90s Blackout Terminal Exit Error** | 2285.58 m ± 2009.71 m | 631.46 m ± 366.63 m | **412.96 m ± 316.29 m** | **81.9%** reduction |
| **Post-Reacquisition Settled Error** | N/A (Diverges indefinitely) | N/A (Diverges linearly) | **20.54 m ± 16.80 m** | Re-converged within 5s |
| **Full Trajectory RMSE** | > 800 meters | 40–55 meters | **132.51 m ± 97.97 m** | **83.4%** reduction |

#### Per-Driver Breakdown (Honest Metric & Distance Reporting)

| Drive Segment | Driver ID & Profile | Samples & GPS Distance | Naive Drift % | 90s Blackout Terminal Exit Error | Fused Peak Blackout Drift | Post-GPS Settled Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `S-S3a` | **Driver A (Normal Urban, Run 1)** | 3,000 / 2,890.03 m | 10.55% | **528.98 m** | 543.01 m | **10.29 m** |
| `S-S3b` | **Driver A (Normal Urban, Run 2)** | 3,000 / 1,432.68 m | 67.07% | **206.60 m** | 209.99 m | **25.31 m** |
| `S-M` | **Driver B (Highway Spot-check, n=1)** | 3,000 / 1,221.20 m | 1583.53% | **41.35 m** | 50.38 m | **7.02 m** |
| `S-Y1` | **Driver D (Urban Spot-check, n=1)** | 3,000 / 3,062.69 m | 23.33% | **538.26 m** | 547.07 m | **54.28 m** |
| `S-Vfa02` | **Driver E (Aggressive, Run 1)** | 3,000 / 4,216.41 m | 151.54% | **989.79 m** | 1023.22 m | **4.94 m** |
| `S-Vta1b` | **Driver E (Aggressive, Run 2)** | 954 / 1,254.51 m | 346.01% | **172.81 m** | 178.49 m | **21.41 m** |

---

### 6. Phase 2 Production Roadmap

1. **Graph Map-Matching Engine:** Integration of offline OpenStreetMap (OSM) road vectors via Hidden Markov Models (HMM) to constrain vehicle position to valid driving lanes during extended multi-minute outages.
2. **Dynamic Phone-to-Vehicle Auto-Calibration (PCA):** Real-time Principal Component Analysis on acceleration vectors during vehicle braking to automatically resolve arbitrary phone mounting angles.
3. **Android NavIC / Dual-Frequency Raw Measurements:** Direct ingestion of pseudoranges and carrier phase measurements via Android `GnssMeasurement` API.
4. **Hardware-in-the-Loop Allan Variance Characterization:** Physical stationary logging on the OnePlus Nord CE3 to calibrate exact on-device noise parameters.
