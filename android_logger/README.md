# Android Sensor Logger (SIH PS 168 Live Demo Artifact)

A lightweight real-time sensor logging application for Android (tested for OnePlus Nord CE3 / Android 13+).

## Capabilities
- Captures raw 3-axis accelerometer ($a_x, a_y, a_z$) and 3-axis gyroscope ($\omega_x, \omega_y, \omega_z$) at 20–50 Hz (`SENSOR_DELAY_GAME`).
- Logs high-precision GPS position (Latitude, Longitude, GPS Speed, Bearing) via `LocationManager`.
- Writes synchronized data directly to a CSV file in the device's Documents folder.
- The output CSV columns match the schema consumed by `src/data_loader.py` without requiring manual conversion:
  ```csv
  timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,gps_lat,gps_lon,gps_speed,gps_heading
  ```

## How to Test / Transfer Logged Drives:
1. Tap **START LOGGING** and mount your phone securely in your vehicle phone holder.
2. Drive through your test route (including flyovers, turns, and stops).
3. Tap **STOP LOGGING**.
4. Connect phone via USB or share the resulting `.csv` file directly into your `SIH/data/samples/` folder.
5. Run `python scripts/run_all.py` to evaluate dead reckoning and EKF fusion on your real recorded drive!
