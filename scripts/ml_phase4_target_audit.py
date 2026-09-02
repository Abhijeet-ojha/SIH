"""
scripts/ml_phase4_target_audit.py
Phase 4: Target Signal Quality Audit on Real IO-VNBD Dataset.
Analyzes speed distributions, stationary noise floors, GPS speed spikes,
discontinuities, and implied acceleration (dv/dt) across all drives.
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite

def audit_target_signals():
    print("=" * 80)
    print("PHASE 4: TARGET SIGNAL QUALITY AUDIT (REAL IO-VNBD DATASET)")
    print("=" * 80)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    all_drives = suite["train_drives"] + suite["test_drives"]

    report_rows = []
    print(f"\nAuditing {len(all_drives)} real drives...")

    for d in all_drives:
        df = d.get_data()
        speed = df["speed"].values
        t = df["timestamp"].values
        dt = np.diff(t, prepend=t[0])
        dt[dt <= 0] = 0.1

        # Implied acceleration from target speed: dv/dt
        dv = np.diff(speed, prepend=speed[0])
        implied_accel = dv / dt

        # Stationary detection: samples where vehicle speed < 0.1 m/s
        stationary_mask = speed < 0.1
        stationary_pct = 100.0 * np.sum(stationary_mask) / len(speed)

        # Discontinuity check: |dv/dt| > 8 m/s^2 (unrealistic for normal passenger car)
        spike_mask = np.abs(implied_accel) > 8.0
        spike_count = int(np.sum(spike_mask))

        row = {
            "drive": d.name,
            "driver": d.driver_id,
            "samples": len(speed),
            "duration_s": float(d.duration_sec),
            "min_speed": float(np.min(speed)),
            "max_speed": float(np.max(speed)),
            "mean_speed": float(np.mean(speed)),
            "median_speed": float(np.median(speed)),
            "std_speed": float(np.std(speed)),
            "stationary_pct": float(stationary_pct),
            "max_implied_accel": float(np.max(np.abs(implied_accel))),
            "spike_count_gt8ms2": spike_count
        }
        report_rows.append(row)

        print(f"  - {d.name:22s} (Driver {d.driver_id}): "
              f"Speed [{row['min_speed']:.2f}, {row['max_speed']:.2f}] m/s | "
              f"Mean: {row['mean_speed']:.2f} m/s | "
              f"Stop %: {row['stationary_pct']:.1f}% | "
              f"Spikes (>8m/s²): {spike_count}")

    report_df = pd.DataFrame(report_rows)
    
    # Save target audit report
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase4_target_audit.csv")
    report_df.to_csv(out_path, index=False)
    print(f"\n[PASS] Target audit saved to: {out_path}")
    print("=" * 80)
    return report_df

if __name__ == "__main__":
    audit_target_signals()
