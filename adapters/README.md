# Sensor Adapter Layer

The `adapters/` layer serves as the platform-to-core translation boundary for the **Universal Sensor-Agnostic GNSS-Denied Localization Engine**.

## Architecture & Responsibilities
- **Hardware Isolation**: Core ML, uncertainty, and 6-state EKF fusion algorithms have zero dependencies on device APIs (Android, iOS, WearOS, Drone PX4, ROS).
- **Canonical Normalization**: Adapters consume heterogeneous physical sensor events and yield standard `SensorFrame` streams with authoritative SI units ($m/s^2$, $rad/s$, $hPa$, $lux$).
- **Deterministic Quality Scoring**: Observes timestamp regularity, packet drops, sensor clipping, and IMU noise floor.

```text
┌──────────────────────────────────────────────────────────────┐
│                      PLATFORM ADAPTERS                       │
├───────────────┬────────────────┬──────────────┬──────────────┤
│    Android    │  Generic CSV   │  WearOS /    │  Drone / ROS │
│    Logger     │   Dataframe    │  Smartwatch  │  (Future)    │
│  (Implemented)│  (Implemented) │  (Interface) │  (Interface) │
└───────┬───────┴────────┬───────┴──────┬───────┴──────┬───────┘
        │                │              │              │
        └────────────────┴──────┬───────┴──────────────┘
                                ↓
                      Canonical SensorFrame
                                ↓
                    Universal Core Engine
```

## Implemented Adapters
1. **[AndroidLoggerAdapter](file:///c:/Users/Abhijeet%20ojha/Desktop/SIH/adapters/android/logger_adapter.py)**: Ingests raw Android CSV logs and formats multi-modal IMU, Light, Pressure data.
2. **[DataFrameSensorAdapter](file:///c:/Users/Abhijeet%20ojha/Desktop/SIH/adapters/generic/dataframe_adapter.py)**: Ingests in-memory Pandas DataFrames and benchmark suites (e.g. IO-VNBD).

## Future Platform Interfaces
- `adapters/wearable/`: Wrist/Smartwatch PPG/IMU adapter with human gait / biomechanical motion modeling.
- `adapters/drone/`: PX4 / ArduPilot MAVLink / ROS IMU + Barometer adapter with 3D velocity vector estimation.
- `adapters/ros/`: ROS 2 `sensor_msgs/Imu` node adapter.
