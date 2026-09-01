"""
scripts/01_run_naive_baseline.py
Day 1 Execution Script (Real IO-VNBD Dataset):
Loads real vehicle drive data from IO-VNBD (Coventry UK), runs the naive
dead-reckoning baseline (double integration of raw IMU), and proves exponential cubic drift.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from src.naive_dr import NaiveDeadReckoning
from src.visualizer import plot_naive_baseline

def main():
    print("=" * 75)
    print("DAY 1 — NAIVE DEAD-RECKONING BASELINE ON REAL IO-VNBD DATASET")
    print("=" * 75)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    test_drive = suite["test_drives"][0] # Driver A - S3a
    df = test_drive.get_data()

    print(f"\n[Step 1] Loaded Real IO-VNBD Drive: {test_drive.name} (Driver {test_drive.driver_id})")
    print(f"         Provenance: {suite['provenance']}")
    print(f"         Samples: {len(df)} | Duration: {test_drive.duration_sec:.1f}s | Sample Rate: {test_drive.sample_rate:.1f} Hz")
    print(f"         Location (Coventry UK): Lat={test_drive.lat0:.5f}°, Lon={test_drive.lon0:.5f}°")
    print(f"         Mean Speed: {df['speed'].mean() * 3.6:.1f} km/h | Max Speed: {df['speed'].max() * 3.6:.1f} km/h")

    # Execute Naive Dead Reckoning
    print("\n[Step 2] Executing Naive Dead Reckoning (double-integrating raw accelerometer + gyro)...")
    init_heading = df["heading"].iloc[0] if "heading" in df.columns else 0.0
    init_speed = df["speed"].iloc[0] if "speed" in df.columns else 0.0
    init_pos = (df["pos_x"].iloc[0], df["pos_y"].iloc[0]) if "pos_x" in df.columns else (0.0, 0.0)

    dr_engine = NaiveDeadReckoning(
        initial_heading=init_heading,
        initial_speed=init_speed,
        initial_pos=init_pos
    )
    naive_res = dr_engine.compute(df)

    final_err = float(naive_res["pos_error_m"].iloc[-1])
    max_err = float(naive_res["pos_error_m"].max())
    print(f"\n[Day 1 Results - Naive Dead Reckoning]")
    print(f"  - Final Position Drift:   {final_err:.2f} meters")
    print(f"  - Maximum Position Error:  {max_err:.2f} meters")
    print(f"  - Max Velocity Error:      {float(naive_res['speed_error_mps'].max()):.2f} m/s")

    output_fig = os.path.join(PROJECT_ROOT, "outputs", "figures", "01_naive_dr_drift.png")
    print(f"\n[Step 3] Rendering Day 1 failure proof plot to {output_fig}...")
    plot_naive_baseline(df, naive_res, output_path=output_fig)
    print("=" * 75)
    print("DAY 1 COMPLETE: Real IO-VNBD baseline failure evidence generated.")
    print("=" * 75)

if __name__ == "__main__":
    main()
