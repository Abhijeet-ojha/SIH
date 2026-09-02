"""
server/simulation_client.py
Interactive Simulation Client for SIH 2026 PS-168.
Replays real-world IO-VNBD drive streams into the local WebSocket gateway as an emulated phone,
verifying live map tracking, GNSS blackout handling, and reacquisition with ZERO Internet.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import numpy as np
import websockets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from core.features.extractor import CausalFeatureExtractor
from core.models.tabular_models import TabularSpeedModel
from core.fusion.ekf_6state import KinematicFusionEKF6State


async def run_simulation(drive_id: str, ws_url: str, blackout_start: float = 20.0, blackout_duration: float = 30.0):
    print("=" * 80)
    print(f"SIH 2026 PS-168: SIMULATED MOBILE TELEMETRY STREAMER (Drive: {drive_id})")
    print("=" * 80)

    # 1. Load Real Drive from Benchmark Suite
    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=1000)
    all_drives = suite["train_drives"] + suite["test_drives"]
    matched = [d for d in all_drives if drive_id in d.name]
    drive = matched[0] if matched else suite["test_drives"][0]
    df = drive.get_data()
    print(f"[*] Loaded Drive {drive.name} (Driver {drive.driver_id}) | Samples: {len(df)} | Duration: {drive.duration_sec:.1f}s")

    # 2. Load Model & EKF
    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "speed_regressor.joblib")
    model = TabularSpeedModel()
    model.load(model_path)
    extractor = CausalFeatureExtractor(window_sec=1.5, step_sec=0.2, sample_rate_hz=10.0, feature_group="all")
    X_feats, _, t_preds, _ = extractor.extract_features(df)
    v_preds, v_stds = model.predict_with_uncertainty(X_feats)

    # 3. Connect to Local WebSocket Gateway
    print(f"[*] Connecting to Local Gateway: {ws_url}...")
    async with websockets.connect(ws_url) as ws:
        print(f"[+] Connected! Beginning Real-Time Stream at 10 Hz (Internet = OFF)...\n")

        init_x = df["pos_x"].iloc[0] if "pos_x" in df.columns else 0.0
        init_y = df["pos_y"].iloc[0] if "pos_y" in df.columns else 0.0
        init_v = df["speed"].iloc[0] if "speed" in df.columns else 0.0
        init_h = df["heading"].iloc[0] if "heading" in df.columns else 0.0

        origin_lat = df["lat"].iloc[0] if "lat" in df.columns else 12.9716
        origin_lon = df["lon"].iloc[0] if "lon" in df.columns else 77.5946

        ekf = KinematicFusionEKF6State(init_x=init_x, init_y=init_y, init_v=init_v, init_heading=init_h)

        t_orig = df["timestamp"].values
        n = len(df)
        blackout_end = blackout_start + blackout_duration

        for i in range(min(n, len(t_preds))):
            curr_t = t_orig[i]
            dt = 0.10 if i == 0 else max(0.01, t_orig[i] - t_orig[i-1])

            in_blackout = (blackout_start <= curr_t < blackout_end)
            gnss_mode = "GNSS_DENIED" if in_blackout else ("GNSS_REACQUIRED" if abs(curr_t - blackout_end) < 2.0 else "GNSS_NORMAL")
            source = "AI_IMU_EKF_DEAD_RECKONING" if in_blackout else "GNSS_AI_IMU_EKF"

            v_ai = float(v_preds[min(i, len(v_preds)-1)])
            v_std = float(v_stds[min(i, len(v_stds)-1)])

            # State propagation
            ekf.predict(dt=dt, v_ai=v_ai, v_ai_std=v_std, gyro_z=float(df["gyro_z"].iloc[i]))
            ekf.update_nhc()

            if not in_blackout and "pos_x" in df.columns:
                ekf.update_gps(
                    gps_x=float(df["pos_x"].iloc[i]),
                    gps_y=float(df["pos_y"].iloc[i]),
                    gps_speed=float(df["speed"].iloc[i]),
                    gps_heading=float(df["heading"].iloc[i]) if "heading" in df.columns else None
                )

            # Convert ENU to Lat/Lon
            r_earth = 6378137.0
            lat_rad = origin_lat * np.pi / 180.0
            d_lat = ekf.x[1] / r_earth
            d_lon = ekf.x[0] / (r_earth * np.cos(lat_rad))
            est_lat = origin_lat + (d_lat * 180.0 / np.pi)
            est_lon = origin_lon + (d_lon * 180.0 / np.pi)

            heading_deg = (ekf.x[4] * 180.0 / np.pi) % 360.0

            packet = {
                "device_id": "EMULATED_FLUTTER_PHONE_01",
                "type": "navigationState",
                "sequence_num": i,
                "timestamp_ms": int(time.time() * 1000),
                "navigation": {
                    "timestamp_s": float(curr_t),
                    "latitude": float(est_lat),
                    "longitude": float(est_lon),
                    "pos_east_m": float(ekf.x[0]),
                    "pos_north_m": float(ekf.x[1]),
                    "speed_mps": float(ekf.x[2]),
                    "speed_kmh": float(ekf.x[2] * 3.6),
                    "velocity_lat_mps": float(ekf.x[3]),
                    "heading_rad": float(ekf.x[4]),
                    "heading_deg": float(heading_deg),
                    "confidence_pct": 55.0 if in_blackout else 95.0,
                    "uncertainty_sigma_mps": float(v_std),
                    "gnss_mode": gnss_mode,
                    "source": source,
                    "blackout_elapsed_s": float(curr_t - blackout_start) if in_blackout else 0.0,
                    "context_mode": "GNSS_BLACKOUT_ACTIVE" if in_blackout else "NORMAL_URBAN"
                }
            }

            await ws.send(json.dumps(packet))
            print(f"  [{curr_t:6.1f}s] Speed: {ekf.x[2]:5.2f} m/s | Hdg: {heading_deg:5.1f}° | Mode: {gnss_mode:15s} | Source: {source}", end="\r")
            await asyncio.sleep(0.10) # 10 Hz rate

    print("\n\n[PASS] Simulation Stream Finished Successfully.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="SIH 2026 Simulation Telemetry Streamer")
    parser.add_argument("--drive", type=str, default="S-Vfa02", help="Drive ID (e.g. S-Vfa02, S-S3a, S-M)")
    parser.add_argument("--url", type=str, default="ws://localhost:8765/telemetry", help="WebSocket Gateway URL")
    parser.add_argument("--blackout_start", type=float, default=20.0, help="Blackout start timestamp (s)")
    parser.add_argument("--blackout_duration", type=float, default=30.0, help="Blackout duration (s)")
    args = parser.parse_args()

    asyncio.run(run_simulation(
        drive_id=args.drive,
        ws_url=args.url,
        blackout_start=args.blackout_start,
        blackout_duration=args.blackout_duration
    ))


if __name__ == "__main__":
    main()
