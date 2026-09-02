# NavPulse — Flutter Mobile Localization Client

Offline-First, GNSS-Aware, AI/IMU Real-Time Mobile Localization System for **SIH 2026 Problem Statement 168**.

## Core Architecture
- **Onboard Sensors**: Ingests high-rate Accelerometer, Gyroscope, Magnetometer, and GNSS location streams.
- **Canonical Normalization**: Standardizes readings into `SensorFrame` with explicit SI units ($m/s^2$, $rad/s$, $uT$).
- **On-Device Motion Estimator**: Causal statistical sliding-window velocity estimator with calibrated split-conformal uncertainty.
- **On-Device 6-State EKF**: Propagates $[p_E, p_N, v_{fwd}, v_{lat}, \theta_{yaw}, b_{gyro}]^T$ with Non-Holonomic Constraints (NHC) and Zero-Velocity Updates (ZUPT).
- **Autonomous GNSS State Transitions**: Detects and switches across `NORMAL`, `DEGRADED`, `DENIED` (Tunnel/Blackout), and `REACQUIRED`.
- **Zero Internet Requirement**: Runs 100% offline. Disconnecting the laptop/Wi-Fi never crashes or halts mobile localization.
- **Local WebSocket Telemetry**: Streams real-time `NavigationState` packets over local Wi-Fi / Hotspot to the laptop dashboard.

## Running the App
1. Install dependencies:
   ```bash
   flutter pub get
   ```
2. Run on connected Android / iOS device:
   ```bash
   flutter run --release
   ```
3. Open **Settings** within the app to configure your laptop's local IP address (e.g. `192.168.1.100:8765`).
