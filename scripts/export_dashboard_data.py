"""
scripts/export_dashboard_data.py
Pre-computes and exports full multi-drive trajectories (Ground Truth, Naive DR,
AI-DR Pure, and Fused EKF) with telemetry to dashboard/data/drives.json.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite, enu_to_geodetic
from src.feature_engineering import extract_window_features
from src.speed_model import SpeedRegressorModel, reconstruct_ai_dr_trajectory
from src.naive_dr import NaiveDeadReckoning
from src.fusion_ekf import run_fusion_pipeline, wrap_angle

def export_all():
    print("Loading benchmark suite & model...")
    suite = get_real_iovnbd_benchmark_suite()
    model = SpeedRegressorModel()
    model.load(os.path.join(PROJECT_ROOT, "outputs", "models", "speed_regressor.joblib"))

    out_dir = os.path.join(PROJECT_ROOT, "dashboard", "data")
    os.makedirs(out_dir, exist_ok=True)

    drives_data = {}

    for drive in suite["test_drives"]:
        drive_name = drive.name
        df = drive.get_data()
        d_id = drive.driver_id
        duration = float(df["timestamp"].iloc[-1])
        print(f"Exporting drive: {drive_name} (Driver {d_id}, {len(df)} samples, {duration:.1f}s)...")

        # Extract features and AI speed
        X, y, t_win = extract_window_features(df, window_sec=1.5, step_sec=0.2)
        y_pred, y_std = model.predict_with_uncertainty(X)
        ai_df = reconstruct_ai_dr_trajectory(
            df, t_win, y_pred, v_std=y_std,
            initial_heading=df["heading"].iloc[0],
            initial_pos=(df["pos_x"].iloc[0], df["pos_y"].iloc[0])
        )

        # Naive DR
        naive_dr = NaiveDeadReckoning(
            initial_heading=df["heading"].iloc[0],
            initial_speed=df["speed"].iloc[0],
            initial_pos=(df["pos_x"].iloc[0], df["pos_y"].iloc[0])
        )
        naive_df = naive_dr.compute(df)

        # Fused EKF (90s blackout between 60s and 150s)
        blackout_start = 60.0
        blackout_end = min(duration - 10.0, 150.0)
        fused_df = run_fusion_pipeline(
            df=df,
            ai_speed=ai_df["ai_speed"].values,
            ai_speed_std=ai_df["ai_speed_std"].values,
            driver_style="aggressive" if d_id == "E" else "normal",
            blackout_start_sec=blackout_start,
            blackout_end_sec=blackout_end
        )

        # Get origin lat/lon for ENU -> Geodetic conversion
        lat0 = getattr(drive, "origin_lat", 52.4068) # Coventry, UK default
        lon0 = getattr(drive, "origin_lon", -1.5197)

        # Convert trajectories to lat/lon for Leaflet mapping
        n = len(df)
        # Subsample for smooth, responsive 60fps web playback (every 2nd sample = 50Hz, or every 4th = 25Hz)
        step = 2
        indices = np.arange(0, n, step)

        t_sub = df["timestamp"].values[indices]
        gt_x = df["pos_x"].values[indices]
        gt_y = df["pos_y"].values[indices]
        gt_v = df["speed"].values[indices]
        gt_h = df["heading"].values[indices]

        naive_x = naive_df["naive_pos_x"].values[indices]
        naive_y = naive_df["naive_pos_y"].values[indices]

        fused_x = fused_df["fused_pos_x"].values[indices]
        fused_y = fused_df["fused_pos_y"].values[indices]
        fused_v = fused_df["fused_velocity"].values[indices]
        fused_h = fused_df["fused_heading"].values[indices]
        fused_bg = fused_df["fused_gyro_bias"].values[indices]
        fused_err = fused_df["fused_pos_error_m"].values[indices] if "fused_pos_error_m" in fused_df.columns else np.zeros(len(indices))
        is_bl = fused_df["is_gnss_blackout"].values[indices]
        modes = fused_df["context_mode"].values[indices]

        ai_v = ai_df["ai_speed"].values[indices]
        ai_s = ai_df["ai_speed_std"].values[indices]

        # Convert positions to Lat/Lon
        gt_lats, gt_lons = enu_to_geodetic(gt_x, gt_y, lat0, lon0)
        naive_lats, naive_lons = enu_to_geodetic(naive_x, naive_y, lat0, lon0)
        fused_lats, fused_lons = enu_to_geodetic(fused_x, fused_y, lat0, lon0)

        # Calculate naive error array
        naive_err = np.sqrt((naive_x - gt_x)**2 + (naive_y - gt_y)**2)

        # Calculate cumulative distance
        dx = np.diff(gt_x, prepend=gt_x[0])
        dy = np.diff(gt_y, prepend=gt_y[0])
        cum_dist = np.cumsum(np.sqrt(dx**2 + dy**2))

        # Build clean JSON record
        drive_record = {
            "name": drive_name,
            "driver_id": d_id,
            "driver_profile": "Aggressive Driving" if d_id == "E" else ("Highway Cruising" if d_id == "B" else ("Dense Urban" if d_id == "D" else "Normal Urban")),
            "total_duration_sec": round(duration, 2),
            "total_distance_m": round(float(cum_dist[-1]), 2),
            "blackout_start_sec": blackout_start,
            "blackout_end_sec": blackout_end,
            "origin": {"lat": lat0, "lon": lon0},
            "num_frames": len(indices),
            "fps": round(1.0 / (t_sub[1] - t_sub[0]), 1) if len(t_sub) > 1 else 50.0,
            "frames": []
        }

        for idx in range(len(indices)):
            drive_record["frames"].append({
                "t": round(float(t_sub[idx]), 3),
                "gt_lat": round(float(gt_lats[idx]), 7),
                "gt_lon": round(float(gt_lons[idx]), 7),
                "gt_v": round(float(gt_v[idx]), 2),
                "gt_h": round(float(gt_h[idx]), 3),
                "naive_lat": round(float(naive_lats[idx]), 7),
                "naive_lon": round(float(naive_lons[idx]), 7),
                "naive_err": round(float(naive_err[idx]), 2),
                "fused_lat": round(float(fused_lats[idx]), 7),
                "fused_lon": round(float(fused_lons[idx]), 7),
                "fused_v": round(float(fused_v[idx]), 2),
                "fused_h": round(float(fused_h[idx]), 3),
                "fused_bg_mrad": round(float(fused_bg[idx] * 1000.0), 3),
                "fused_err": round(float(fused_err[idx]), 2),
                "ai_v": round(float(ai_v[idx]), 2),
                "ai_sigma_v": round(float(ai_s[idx]), 3),
                "is_blackout": bool(is_bl[idx]),
                "context_mode": str(modes[idx]),
                "cum_dist_m": round(float(cum_dist[idx]), 1)
            })

        drives_data[drive_name] = drive_record

    out_file = os.path.join(out_dir, "drives.json")
    with open(out_file, "w") as f:
        json.dump(drives_data, f, indent=2)

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n[SUCCESS] Exported {len(drives_data)} drives to {out_file} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    export_all()
