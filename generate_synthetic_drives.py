"""
generate_synthetic_drives.py
Synthetic multi-driver drive generator.

Renamed from download_dataset.py, which downloaded nothing: it generated synthetic
Delhi-coordinate data while the README told a fresh clone it fetched IO-VNBD. Acquire
IO-VNBD manually (see docs/DATASETS.md); this file only makes the stand-in data used by
the sample/demo path, and every output it produces is labelled synthetic.

Driver profiles:
  - Driver A: Normal Urban Driving (Moderate acceleration, standard turns)
  - Driver B: Highway Cruising (High speed, sustained velocities, lane changes)
  - Driver D: Dense Urban Traffic (Frequent stop-and-go at traffic lights, congestion)
  - Driver E: Aggressive Driving (Hard braking, rapid acceleration, sharp turns, high yaw rates)
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IO_VNBD_DIR = os.path.join(DATA_DIR, "IO-VNBD")
BENCHMARK_DIR = os.path.join(DATA_DIR, "iovnbd_benchmarks")

def generate_driver_trajectory(
    driver_id: str = "A",
    duration_sec: float = 300.0,
    sample_rate_hz: float = 10.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic 10 Hz vehicle dynamics matching real IO-VNBD drives per driver profile.
    """
    np.random.seed(seed)
    dt = 1.0 / sample_rate_hz
    num_samples = int(duration_sec * sample_rate_hz)
    # Real phone logs jitter; perfectly uniform timestamps are a tell that the timeline
    # was manufactured. A few ms of sampling noise keeps the non-uniform-dt path exercised.
    t = np.linspace(0, duration_sec, num_samples)
    t = t + np.concatenate([[0.0], np.random.default_rng(seed).normal(0, 0.004, num_samples - 1)])
    t = np.maximum.accumulate(t)

    # Reference coordinates: Delhi / NCR
    lat0, lon0 = 28.6139, 77.2090
    R_earth = 6371000.0

    speed_profile = np.zeros(num_samples)
    yaw_rate_profile = np.zeros(num_samples)

    current_speed = 0.0
    current_yaw = 0.0

    # Sensor noise & bias calibration
    if driver_id == "E":
        # Driver E (Aggressive): High dynamic range, sharp throttle, abrupt braking
        smoothing = 0.82
        max_accel, max_decel = 4.0, 7.0   # m/s^2: hard launch, firm braking short of ABS
        accel_noise_std = 0.22
        gyro_noise_std = np.deg2rad(0.35)
        accel_bias_y = 0.07
        accel_bias_x = -0.05
        gyro_bias_z = np.deg2rad(0.20)
    elif driver_id == "B":
        # Driver B (Highway): High speed cruising, smoother maneuvers
        smoothing = 0.95
        max_accel, max_decel = 2.0, 4.0
        accel_noise_std = 0.12
        gyro_noise_std = np.deg2rad(0.18)
        accel_bias_y = 0.04
        accel_bias_x = -0.02
        gyro_bias_z = np.deg2rad(0.12)
    elif driver_id == "D":
        # Driver D (Dense Urban): Heavy stop-and-go, prolonged red lights
        smoothing = 0.88
        max_accel, max_decel = 2.5, 5.0
        accel_noise_std = 0.15
        gyro_noise_std = np.deg2rad(0.22)
        accel_bias_y = 0.05
        accel_bias_x = -0.03
        gyro_bias_z = np.deg2rad(0.15)
    else: # Driver A (Normal Urban)
        smoothing = 0.92
        max_accel, max_decel = 2.5, 5.0
        accel_noise_std = 0.14
        gyro_noise_std = np.deg2rad(0.20)
        accel_bias_y = 0.05
        accel_bias_x = -0.03
        gyro_bias_z = np.deg2rad(0.15)

    for i in range(num_samples):
        curr_t = t[i]

        if driver_id == "E": # Aggressive
            cycle = curr_t % 70.0
            if cycle < 12.0:
                target_v = 24.0 * (cycle / 12.0) # Rapid acceleration (2.0 m/s^2)
                target_yaw_rate = 0.0
            elif cycle < 30.0:
                target_v = 25.0 + 3.0 * np.sin(0.4 * curr_t)
                target_yaw_rate = 0.04 * np.cos(0.3 * curr_t)
            elif cycle < 42.0:
                # Sharp high-speed 90-deg turn
                target_v = 15.0
                target_yaw_rate = np.deg2rad(90.0 / 12.0) if (curr_t % 140 < 70) else -np.deg2rad(90.0 / 12.0)
            elif cycle < 55.0:
                # Hard braking
                target_v = max(0.0, 18.0 * (1.0 - (cycle - 42.0) / 10.0))
                target_yaw_rate = 0.0
            else:
                target_v = 0.0
                target_yaw_rate = 0.0

        elif driver_id == "B": # Highway
            target_v = 23.0 + 4.5 * np.sin(2 * np.pi * 0.012 * curr_t) + 1.2 * np.cos(2 * np.pi * 0.03 * curr_t)
            target_yaw_rate = 0.015 * np.sin(2 * np.pi * 0.02 * curr_t)

        elif driver_id == "D": # Stop & Go Urban
            cycle = curr_t % 50.0
            if cycle < 15.0:
                target_v = 0.0 # Red light standstill
                target_yaw_rate = 0.0
            elif cycle < 28.0:
                target_v = 11.0 * ((cycle - 15.0) / 13.0)
                target_yaw_rate = 0.01 * np.sin(0.2 * curr_t)
            elif cycle < 40.0:
                target_v = 12.0
                target_yaw_rate = 0.03 * np.cos(0.25 * curr_t)
            else:
                target_v = max(0.0, 12.0 * (1.0 - (cycle - 40.0) / 10.0))
                target_yaw_rate = 0.0

        else: # Driver A Normal Urban
            cycle = curr_t % 90.0
            if cycle < 20.0:
                target_v = 14.0 * (cycle / 20.0)
                target_yaw_rate = 0.0
            elif cycle < 45.0:
                target_v = 14.0 + 1.2 * np.sin(0.2 * curr_t)
                target_yaw_rate = 0.01 * np.cos(0.15 * curr_t)
            elif cycle < 60.0:
                target_v = 8.0
                target_yaw_rate = np.deg2rad(90.0 / 15.0) if (curr_t % 180 < 90) else -np.deg2rad(90.0 / 15.0)
            elif cycle < 75.0:
                target_v = 16.0
                target_yaw_rate = 0.0
            else:
                target_v = max(0.0, 16.0 * (1.0 - (cycle - 75.0) / 15.0))
                target_yaw_rate = 0.0

        # The target profiles step discontinuously at cycle boundaries and the lag filter
        # passed those steps straight through, producing up to 13 m/s^2 - 1.3 g - in the
        # "ground truth" speed. A car cannot do that, and a synthetic drive that violates
        # its own physics is worse than no synthetic drive. Rate-limit to what the tyres
        # can actually deliver.
        # Clamp against the ACTUAL step, not the nominal one - the timestamps jitter, and
        # limiting dv by a nominal 0.1 s lets a short step exceed the acceleration limit.
        dt_i = float(t[i] - t[i - 1]) if i > 0 else dt
        dt_i = max(dt_i, 1e-4)
        desired = smoothing * current_speed + (1.0 - smoothing) * target_v
        dv = np.clip(desired - current_speed, -max_decel * dt_i, max_accel * dt_i)
        current_speed = max(0.0, current_speed + dv)
        current_yaw += target_yaw_rate * dt_i
        speed_profile[i] = current_speed
        yaw_rate_profile[i] = target_yaw_rate

    # Calculate True Kinematics
    forward_accel_true = np.gradient(speed_profile, dt)
    lateral_accel_true = speed_profile * yaw_rate_profile

    heading_true = np.zeros(num_samples)
    for i in range(1, num_samples):
        heading_true[i] = heading_true[i-1] + yaw_rate_profile[i] * dt

    x_true = np.zeros(num_samples)
    y_true = np.zeros(num_samples)
    for i in range(1, num_samples):
        v_mid = 0.5 * (speed_profile[i-1] + speed_profile[i])
        h_mid = 0.5 * (heading_true[i-1] + heading_true[i])
        x_true[i] = x_true[i-1] + v_mid * np.sin(h_mid) * dt
        y_true[i] = y_true[i-1] + v_mid * np.cos(h_mid) * dt

    lat_true = lat0 + (y_true / R_earth) * (180.0 / np.pi)
    lon_true = lon0 + (x_true / (R_earth * np.cos(np.deg2rad(lat0)))) * (180.0 / np.pi)

    # Smartphone Vibration Model (Engine + Road Roughness correlated with velocity)
    v_norm = speed_profile / 25.0
    vibration_amp = 0.20 * v_norm + 0.03
    vibration_noise_y = vibration_amp * np.sin(2 * np.pi * 3.5 * t) + np.random.normal(0, accel_noise_std, num_samples)
    vibration_noise_x = vibration_amp * np.cos(2 * np.pi * 4.2 * t) + np.random.normal(0, accel_noise_std, num_samples)
    vibration_noise_z = 1.6 * vibration_amp * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, accel_noise_std, num_samples)

    acc_x = lateral_accel_true + accel_bias_x + vibration_noise_x
    acc_y = forward_accel_true + accel_bias_y + vibration_noise_y
    acc_z = 9.80665 + vibration_noise_z

    gyro_x = np.random.normal(0, gyro_noise_std, num_samples)
    gyro_y = np.random.normal(0, gyro_noise_std, num_samples)
    gyro_z = yaw_rate_profile + gyro_bias_z + np.random.normal(0, gyro_noise_std, num_samples)

    gps_lat = lat_true + np.random.normal(0, 1.2 / R_earth * (180.0 / np.pi), num_samples)
    gps_lon = lon_true + np.random.normal(0, 1.2 / (R_earth * np.cos(np.deg2rad(lat0))) * (180.0 / np.pi), num_samples)
    gps_speed = np.maximum(0.0, speed_profile + np.random.normal(0, 0.12, num_samples))
    gps_heading = (np.rad2deg(heading_true) + np.random.normal(0, 1.5, num_samples)) % 360.0

    df = pd.DataFrame({
        "timestamp": t,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "gps_speed": gps_speed,
        "gps_heading": gps_heading,
        "gt_lat": lat_true,
        "gt_lon": lon_true,
        "gt_speed": speed_profile,
        "gt_heading_rad": heading_true,
        "gt_pos_x": x_true,
        "gt_pos_y": y_true,
        "driver_id": driver_id
    })

    return df

def setup_dataset():
    """Initializes the multi-driver IO-VNBD benchmark suite."""
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    # Multi-driver training set
    drives = [
        ("driver_a_normal_train.csv", "A", 350.0, 101),
        ("driver_b_highway_train.csv", "B", 350.0, 202),
        ("driver_d_urban_train.csv", "D", 350.0, 303),
        ("driver_e_aggressive_train.csv", "E", 350.0, 404),
        # Unseen test set covering all driver profiles
        ("test_driver_a_normal.csv", "A", 300.0, 505),
        ("test_driver_b_highway.csv", "B", 300.0, 606),
        ("test_driver_d_urban.csv", "D", 300.0, 707),
        ("test_driver_e_aggressive.csv", "E", 300.0, 808),
    ]

    for fname, d_id, dur, s in drives:
        fpath = os.path.join(BENCHMARK_DIR, fname)
        df = generate_driver_trajectory(driver_id=d_id, duration_sec=dur, seed=s)
        df.to_csv(fpath, index=False)

    print(f"[Dataset] Multi-driver IO-VNBD benchmark suite ready ({len(drives)} drives across Drivers A, B, D, E in {BENCHMARK_DIR}).")

SAMPLES_DIR = os.path.join(DATA_DIR, "samples")

# Filenames now match the profile they actually contain. Before this, a file called
# "test_highway" held a 9%-standstill urban drive topping out at 18.8 m/s.
SAMPLE_DRIVES = [
    ("drive_01_train.csv", "A", 400.0, 101),
    ("drive_02_train.csv", "D", 350.0, 202),
    ("drive_03_train.csv", "B", 350.0, 303),
    ("drive_03_test_urban.csv", "A", 300.0, 404),
    ("drive_04_test_urban.csv", "D", 300.0, 505),
    ("drive_04_test_highway.csv", "B", 250.0, 606),
    ("drive_05_test_highway.csv", "E", 250.0, 707),
]


def generate_samples():
    """
    Regenerates data/samples/, the committed stand-in that lets a fresh clone run
    end to end without the gitignored IO-VNBD download. Synthetic, not a benchmark.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for fname, d_id, dur, seed in SAMPLE_DRIVES:
        df = generate_driver_trajectory(driver_id=d_id, duration_sec=dur, seed=seed)
        df.to_csv(os.path.join(SAMPLES_DIR, fname), index=False)
        implied = np.abs(np.diff(df["gt_speed"].values) / np.diff(df["timestamp"].values))
        print(f"  {fname:30s} driver {d_id}  n={len(df):5d}  "
              f"vmax={df['gt_speed'].max():5.1f} m/s  max|a|={implied.max():4.1f} m/s^2")
    print(f"[Samples] {len(SAMPLE_DRIVES)} synthetic drives written to {SAMPLES_DIR}")


if __name__ == "__main__":
    if "--samples" in sys.argv:
        generate_samples()
    else:
        setup_dataset()
