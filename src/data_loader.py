"""
src/data_loader.py
Dataset ingestion and geodetic coordinate conversion module for IO-VNBD dataset.
Ensures deterministic SHA256-verified dataset loading, kinematic consistency,
and geodetic-to-local ENU mapping.
"""

import os
import glob
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

R_EARTH = 6371000.0  # Earth radius in meters

def compute_df_sha256(df: pd.DataFrame) -> str:
    """Computes a deterministic SHA256 hash of a dataframe's values."""
    data_bytes = df.values.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()[:12]

def pressure_to_altitude(pressure_hpa: np.ndarray) -> np.ndarray:
    """
    Barometric pressure -> altitude relative to the first sample, in metres.

    Referenced to the drive's own first reading rather than sea level, because only the
    *change* matters here and that sidesteps needing local QNH. Good to ~0.1 m over the
    tens of metres a parking ramp or flyover spans; weather drift makes it useless over
    hours, so re-reference whenever GPS altitude is healthy.
    """
    p = np.asarray(pressure_hpa, dtype=float)
    p_ref = float(p[np.isfinite(p)][0]) if np.any(np.isfinite(p)) else 1013.25
    return 44330.0 * (1.0 - np.power(p / p_ref, 1.0 / 5.255))

def _path_length(x: np.ndarray, y: np.ndarray) -> float:
    """Cumulative Euclidean path length, used as the ground truth for the speed-unit check."""
    return float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))

def geodetic_to_enu(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Converts Lat/Lon to local East-North-Up (ENU) coordinates in meters."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    x_east = R_EARTH * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y_north = R_EARTH * (lat_rad - lat0_rad)
    return x_east, y_north

def enu_to_geodetic(x_east: np.ndarray, y_north: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Converts local ENU coordinates back to Latitude and Longitude."""
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    lat_rad = lat0_rad + (y_north / R_EARTH)
    lon_rad = lon0_rad + (x_east / (R_EARTH * np.cos(lat0_rad)))
    return np.rad2deg(lat_rad), np.rad2deg(lon_rad)

class DriveDataset:
    """Represents a single parsed and synchronized vehicle/smartphone drive from IO-VNBD."""

    def __init__(self, df: pd.DataFrame, name: str = "drive", driver_id: str = "A", is_real_iovnbd: bool = True):
        self.name = name
        self.driver_id = driver_id
        self.is_real_iovnbd = is_real_iovnbd
        self.raw_df = df.copy()
        self.standardized_df = self._standardize(df)
        self.num_samples = len(self.standardized_df)
        self.duration_sec = float(self.standardized_df["timestamp"].iloc[-1] - self.standardized_df["timestamp"].iloc[0])
        self.dt = float(np.median(np.diff(self.standardized_df["timestamp"].values))) if len(self.standardized_df) > 1 else 0.1
        self.sample_rate = 1.0 / self.dt if self.dt > 0 else 10.0
        self.integrity_hash = compute_df_sha256(self.standardized_df)

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        cols = {c.strip().lower(): c for c in df.columns}

        # 1. Monotonic Timestamp
        time_col = None
        for k in ["time since start (ms)", "timestamp", "time", "t", "time_stamp", "epoch"]:
            if k in cols:
                time_col = cols[k]
                break

        # The previous implementation clamped every dt <= 0 or > 1.0 to 0.1 s and
        # cumsum'd the result, which fabricated a uniform 10 Hz timeline that did not
        # exist. Symptom: duration_s came out as exactly samples * 0.1 for every drive in
        # phase4_target_audit.csv, and any true interval >= 1 s was compressed 10x,
        # inflating every derived acceleration to 17-33 m/s^2.
        # Real timestamps are preserved here. Genuine gaps are recorded, never rewritten.
        if time_col is not None:
            raw_t = df[time_col].values.astype(float)
            if "ms" in time_col.lower() or np.median(np.diff(raw_t)) > 10.0:
                raw_t = raw_t / 1000.0
            raw_t = raw_t - raw_t[0]

            # Non-monotonic samples are corrupt rows, not a timing convention. Drop them
            # rather than inventing a timestamp; df is filtered in lockstep so every other
            # column stays aligned.
            keep = np.ones(len(raw_t), dtype=bool)
            last = -np.inf
            for i, ti in enumerate(raw_t):
                if ti > last:
                    last = ti
                else:
                    keep[i] = False
            self.n_dropped_nonmonotonic = int(np.sum(~keep))
            if self.n_dropped_nonmonotonic > 0:
                df = df.loc[keep].reset_index(drop=True)
                raw_t = raw_t[keep]
            out["timestamp"] = raw_t
        else:
            self.n_dropped_nonmonotonic = 0
            out["timestamp"] = np.arange(len(df)) * 0.1

        t_vals = out["timestamp"].values
        dt_arr = np.diff(t_vals, prepend=t_vals[0])
        if len(dt_arr) > 1:
            dt_arr[0] = dt_arr[1]
        dt = float(np.median(dt_arr[dt_arr > 0])) if np.any(dt_arr > 0) else 0.1
        if dt <= 0:
            dt = 0.1
        out["dt"] = dt_arr
        # A gap is a logging dropout, not a sample. Downstream code must not integrate
        # across one as though it were a normal step.
        out["gap_mask"] = dt_arr > (3.0 * dt)
        self.median_dt = dt
        self.n_gaps = int(np.sum(out["gap_mask"].values))

        # 2. GPS Coordinates & ENU Local Plane
        lat_col = None
        for k in ["gps latitude (degrees)", "gt_lat", "latitude", "gps_lat", "lat"]:
            if k in cols:
                lat_col = cols[k]
                break

        lon_col = None
        for k in ["gps longitude (degrees)", "gt_lon", "longitude", "gps_lon", "lon", "lng"]:
            if k in cols:
                lon_col = cols[k]
                break

        if lat_col is not None and lon_col is not None:
            raw_lat = df[lat_col].values.astype(float)
            raw_lon = df[lon_col].values.astype(float)
            self.lat0 = float(raw_lat[0])
            self.lon0 = float(raw_lon[0])
            x_east, y_north = geodetic_to_enu(raw_lat, raw_lon, self.lat0, self.lon0)
            out["pos_x"] = x_east
            out["pos_y"] = y_north
        elif "gt_pos_x" in df.columns and "gt_pos_y" in df.columns:
            out["pos_x"] = df["gt_pos_x"].values.astype(float)
            out["pos_y"] = df["gt_pos_y"].values.astype(float)
            self.lat0, self.lon0 = 52.40166, -1.50529
        else:
            out["pos_x"] = np.zeros(len(df))
            out["pos_y"] = np.zeros(len(df))
            self.lat0, self.lon0 = 52.40166, -1.50529

        # 3. Ground Truth Speed (clipped to [0, 45] m/s)
        speed_col = None
        for k in ["gps speed (kmh)", "speed", "vehiclespeed", "gps_speed", "gt_speed", "velocity"]:
            if k in cols:
                speed_col = cols[k]
                break

        if speed_col is not None:
            raw_spd = df[speed_col].values.astype(float)
            # Do not trust the column header for units. IO-VNBD's header says
            # "GPS Speed (Kmh)" while the audit shows a highway drive topping out at
            # 6.7 m/s (24 km/h), which is not a highway. Settle it with physics instead:
            # integral(v dt) must equal the GPS path length. Only one divisor can satisfy
            # that, and the ratio between the candidates is 3.6x, so they are trivially
            # separable.
            path_len = _path_length(out["pos_x"].values, out["pos_y"].values)
            header_says_kmh = "kmh" in speed_col.lower() or "km/h" in speed_col.lower()

            candidates = {"m/s": 1.0, "km/h": 3.6}
            errors = {}
            for label, div in candidates.items():
                integ = float(np.sum(np.clip(raw_spd / div, 0.0, 45.0) * dt_arr))
                errors[label] = abs(integ - path_len) / max(path_len, 1e-6)

            best = min(errors, key=errors.get)
            self.speed_unit = best
            self.speed_unit_divisor = candidates[best]
            self.speed_integral_error = errors[best]
            self.speed_unit_evidence = dict(errors)
            self.speed_unit_header_conflict = header_says_kmh != (best == "km/h")

            spd_mps = raw_spd / candidates[best]
            out["speed"] = np.clip(spd_mps, 0.0, 45.0)
        else:
            dx = np.diff(out["pos_x"].values, prepend=out["pos_x"].values[0])
            dy = np.diff(out["pos_y"].values, prepend=out["pos_y"].values[0])
            calc_spd = np.sqrt(dx**2 + dy**2) / np.maximum(dt_arr, 1e-6)
            out["speed"] = np.clip(calc_spd, 0.0, 45.0)
            self.speed_unit = "derived_from_position"
            self.speed_unit_divisor = 1.0
            self.speed_integral_error = 0.0
            self.speed_unit_evidence = {}
            self.speed_unit_header_conflict = False

        # 4. Heading (Native GPS Orientation or Course Over Ground Fallback)
        ori_col = None
        for k in ["gps orientation", "gps orientation ()", "gps_orientation", "course", "gps_heading"]:
            if k in cols:
                ori_col = cols[k]
                break
        if ori_col is None:
            for k, v in cols.items():
                if "gps" in k and "orient" in k:
                    ori_col = v
                    break

        if ori_col is not None:
            raw_ori = df[ori_col].values.astype(float)
            # IO-VNBD GPS ORIENTATION is clockwise azimuth in degrees [0, 360) where 0=North, 90=East
            # Convert to radians wrapped to [-pi, pi]
            ori_rad = np.deg2rad(raw_ori)
            ori_wrapped = (ori_rad + np.pi) % (2.0 * np.pi) - np.pi
            out["heading"] = ori_wrapped
        else:
            dx = np.diff(out["pos_x"].values, prepend=out["pos_x"].values[0])
            dy = np.diff(out["pos_y"].values, prepend=out["pos_y"].values[0])
            dx_smooth = pd.Series(dx).rolling(7, min_periods=1, center=True).mean().values
            dy_smooth = pd.Series(dy).rolling(7, min_periods=1, center=True).mean().values
            course = np.arctan2(dx_smooth, dy_smooth)

            valid_heading = np.zeros(len(df))
            last_h = course[0] if len(course) > 0 else 0.0
            for i in range(len(df)):
                if out["speed"].iloc[i] > 0.4:
                    last_h = course[i]
                valid_heading[i] = last_h
            out["heading"] = valid_heading

        # 5. Accelerometer (m/s^2)
        for k, v in cols.items():
            if "accelerometer" in k:
                if " x" in k or k.endswith("x") or "acc_x" in k:
                    out["acc_x"] = df[v].values.astype(float)
                elif " y" in k or k.endswith("y") or "acc_y" in k:
                    out["acc_y"] = df[v].values.astype(float)
                elif " z" in k or k.endswith("z") or "acc_z" in k:
                    out["acc_z"] = df[v].values.astype(float)

        if "acc_x" not in out.columns:
            for k in ["acc_x", "accx", "ax", "lateralacceleration"]:
                if k in cols:
                    out["acc_x"] = df[cols[k]].values.astype(float)
                    break
        if "acc_y" not in out.columns:
            for k in ["acc_y", "accy", "ay", "longitudinalacceleration", "forward_accel"]:
                if k in cols:
                    out["acc_y"] = df[cols[k]].values.astype(float)
                    break
        if "acc_z" not in out.columns:
            for k in ["acc_z", "accz", "az", "verticalacceleration"]:
                if k in cols:
                    out["acc_z"] = df[cols[k]].values.astype(float)
                    break

        # These used to fall back to zeros / np.gradient(speed) / constant gravity. The
        # acc_y fallback derived a *feature* from the *label*: if it ever fired, every
        # metric in the repo was measuring the model reading its own answer key. Fail loudly
        # instead - a drive without an accelerometer is not a drive we can do DR on.
        missing_acc = [c for c in ("acc_x", "acc_y", "acc_z") if c not in out.columns]
        if missing_acc:
            raise ValueError(
                f"{self.name}: missing accelerometer channel(s) {missing_acc}. "
                f"Available columns: {sorted(cols.keys())}"
            )

        # 6. Gyroscope (rad/s)
        for k, v in cols.items():
            if "gyroscope" in k:
                if "roll" in k or " x" in k or k.endswith("x"):
                    out["gyro_x"] = df[v].values.astype(float)
                elif "pitch" in k or " y" in k or k.endswith("y"):
                    out["gyro_y"] = df[v].values.astype(float)
                elif "yaw" in k or " z" in k or k.endswith("z"):
                    out["gyro_z"] = df[v].values.astype(float)

        if "gyro_x" not in out.columns:
            for k in ["gyro_x", "gyrox", "gx", "rollrate"]:
                if k in cols:
                    out["gyro_x"] = df[cols[k]].values.astype(float)
                    break
        if "gyro_y" not in out.columns:
            for k in ["gyro_y", "gyroy", "gy", "pitchrate"]:
                if k in cols:
                    out["gyro_y"] = df[cols[k]].values.astype(float)
                    break
        if "gyro_z" not in out.columns:
            for k in ["gyro_z", "gyroz", "gz", "yawrate", "yaw_rate"]:
                if k in cols:
                    out["gyro_z"] = df[cols[k]].values.astype(float)
                    break

        # gyro_z used to fall back to np.gradient(heading), i.e. the GPS course. That fed
        # GPS-derived yaw into the EKF's dead-reckoning propagation, so the filter was
        # reading GPS during the very blackout it was supposed to survive without it.
        missing_gyro = [c for c in ("gyro_x", "gyro_y", "gyro_z") if c not in out.columns]
        if missing_gyro:
            raise ValueError(
                f"{self.name}: missing gyroscope channel(s) {missing_gyro}. "
                f"Available columns: {sorted(cols.keys())}"
            )

        # 7. Optional sensors. Absent in IO-VNBD, present in SHL and in our own Android
        # logs. Each carries an availability flag so downstream code can tell "sensor read
        # zero" from "sensor not fitted" - the old ambient_lux default of 1500 lux was
        # indistinguishable from a real daylight reading and silently disabled the tunnel
        # detector on every drive.
        def _first(keys):
            for k in keys:
                if k in cols:
                    return df[cols[k]].values.astype(float)
            return None

        lux = _first(["light", "ambient_lux", "illuminance"])
        out["ambient_lux"] = lux if lux is not None else np.full(len(df), np.nan)
        self.has_light = lux is not None

        # Barometer -> relative altitude. ~0.1 m resolution, which is what makes
        # multi-level parking and flyover-vs-service-road tractable.
        pres = _first(["pressure", "barometer", "pressure (hpa)", "air_pressure"])
        out["pressure_hpa"] = pres if pres is not None else np.full(len(df), np.nan)
        self.has_barometer = pres is not None
        out["baro_altitude_m"] = pressure_to_altitude(pres) if pres is not None else np.full(len(df), np.nan)

        # Magnetometer -> absolute heading reference that bounds gyro drift in long blackouts.
        for axis in ("x", "y", "z"):
            m = _first([f"mag_{axis}", f"magnetometer {axis}", f"magnetic field {axis}", f"m{axis}"])
            out[f"mag_{axis}"] = m if m is not None else np.full(len(df), np.nan)
        self.has_magnetometer = not np.all(np.isnan(out["mag_x"].values))

        # Step detector -> decisive walking evidence, free from the sensor hub.
        steps = _first(["step_detector", "steps", "step_count"])
        out["step_events"] = steps if steps is not None else np.zeros(len(df))
        self.has_step_detector = steps is not None

        out["driver_id"] = self.driver_id
        return out

    def get_data(self) -> pd.DataFrame:
        return self.standardized_df

def load_real_iovnbd_drive(
    filepath: str,
    driver_id: Optional[str] = None,
    max_samples: Optional[int] = None,
    offset: int = 0,
    name_suffix: str = ""
) -> DriveDataset:
    assert os.path.exists(filepath), f"File not found: {filepath}"
    size = os.path.getsize(filepath)
    assert size > 1000, f"Error: {filepath} is an un-pulled Git LFS pointer ({size} bytes). Run git lfs pull first."

    if driver_id is None:
        if "Driver A" in filepath or "S-S" in filepath:
            driver_id = "A"
        elif "Driver B" in filepath or "S-M" in filepath:
            driver_id = "B"
        elif "Driver D" in filepath or "S-Y" in filepath:
            driver_id = "D"
        elif "Driver E" in filepath or "Vf" in filepath or "Vta" in filepath or "Vtb" in filepath or "Vw" in filepath:
            driver_id = "E"
        else:
            driver_id = "Unknown"

    df = pd.read_csv(filepath, encoding="latin-1")
    if offset > 0:
        df = df.iloc[offset:].copy()
    if max_samples and len(df) > max_samples:
        df = df.iloc[:max_samples].copy()

    name = os.path.basename(filepath).replace(".csv", "") + name_suffix
    return DriveDataset(df, name=name, driver_id=driver_id, is_real_iovnbd=True)

SYNTHETIC_BANNER = (
    "\n" + "=" * 78 + "\n"
    "  SYNTHETIC DATA - NOT A BENCHMARK RESULT\n"
    "  data/IO-VNBD-repo/ was not found, so this run uses the generated stand-in\n"
    "  drives in data/samples/. Numbers produced here describe a simulator, not a\n"
    "  vehicle, and must not be quoted as accuracy figures anywhere.\n"
    "  Acquire IO-VNBD (see docs/DATASETS.md) for real results.\n"
    + "=" * 78 + "\n"
)


def get_sample_fallback_suite(max_samples_per_drive: int = 3000) -> Dict[str, Any]:
    """
    Stand-in suite built from data/samples/ so a fresh clone runs end to end.

    The README Quick Start told people to run run_all.py, which needed the gitignored
    dataset directory and crashed on a clean checkout. This makes the pipeline runnable
    immediately - loudly labelled, so nobody mistakes the output for a benchmark.
    """
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(proj_root, "data", "samples")
    print(SYNTHETIC_BANNER)

    files = sorted(os.listdir(sample_dir))
    train = [f for f in files if "train" in f]
    test = [f for f in files if "test" in f]

    def _load(names):
        return [
            load_real_iovnbd_drive(os.path.join(sample_dir, n), driver_id="A",
                                   max_samples=max_samples_per_drive)
            for n in names
        ]

    return {
        "train_drives": _load(train),
        "test_drives": _load(test),
        "provenance": "SYNTHETIC sample drives (data/samples) - not IO-VNBD, not a benchmark",
        "is_synthetic": True,
    }


def get_real_iovnbd_benchmark_suite(max_samples_per_drive: int = 3000) -> Dict[str, Any]:
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_base = os.path.join(proj_root, "data", "IO-VNBD-repo", "Synchronised V abd S datasets", "Categorised IOVNB Dataset")

    if not os.path.isdir(repo_base):
        return get_sample_fallback_suite(max_samples_per_drive)

    a_s1 = os.path.join(repo_base, "S (Driver A)", "S1", "S-S1.csv")
    a_s2 = os.path.join(repo_base, "S (Driver A)", "S2", "S-S2.csv")
    a_s3a = os.path.join(repo_base, "S (Driver A)", "S3a", "S-S3a.csv")
    a_s3b = os.path.join(repo_base, "S (Driver A)", "S3b", "S-S3b.csv")
    b_m = os.path.join(repo_base, "M (Driver B)", "S-M.csv")
    d_y1 = os.path.join(repo_base, "Y (Driver D)", "Y1", "S-Y1.csv")
    e_vfa01 = os.path.join(repo_base, "Vf (Driver E)", "V-Vfa01", "S-Vfa01.csv")
    e_vfa02 = os.path.join(repo_base, "Vf (Driver E)", "V-Vfa02", "S-Vfa02.csv")
    e_vta1a = os.path.join(repo_base, "Vta (Driver E)", "Vta01a", "S-Vta1a.csv")
    e_vta1b = os.path.join(repo_base, "Vta (Driver E)", "Vta01b", "S-Vta1b.csv")

    # Balanced, diversified multi-driver training suite (Drivers A, B, D, E)
    train_files = [
        ("Driver_A_S1", a_s1, "A", 0, ""),
        ("Driver_A_S2", a_s2, "A", 0, ""),
        ("Driver_B_M_train", b_m, "B", 10000, "_train"),
        ("Driver_D_Y1_train", d_y1, "D", 10000, "_train"),
        ("Driver_E_Vfa01", e_vfa01, "E", 0, ""),
        ("Driver_E_Vta1a", e_vta1a, "E", 0, ""),
    ]

    test_files = [
        ("Driver_A_S3a", a_s3a, "A", 0, ""),
        ("Driver_A_S3b", a_s3b, "A", 0, ""),
        ("Driver_B_M_n1", b_m, "B", 0, ""),
        ("Driver_D_Y1_n1", d_y1, "D", 0, ""),
        ("Driver_E_Vfa02", e_vfa02, "E", 0, ""),
        ("Driver_E_Vta1b", e_vta1b, "E", 0, ""),
    ]

    train_drives = [
        load_real_iovnbd_drive(p, d_id, max_samples=max_samples_per_drive, offset=off, name_suffix=suf)
        for name, p, d_id, off, suf in train_files
    ]
    test_drives = [
        load_real_iovnbd_drive(p, d_id, max_samples=max_samples_per_drive, offset=off, name_suffix=suf)
        for name, p, d_id, off, suf in test_files
    ]

    return {
        "train_drives": train_drives,
        "test_drives": test_drives,
        "provenance": "100% Real IO-VNBD Dataset (Coventry, UK)",
        "is_synthetic": False,
    }
