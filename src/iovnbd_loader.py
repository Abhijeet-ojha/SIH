"""
src/iovnbd_loader.py
Loader for the REAL IO-VNBD dataset, written against the schema as it exists on disk
(run scripts/inspect_iovnbd_schema.py to reproduce every claim below).

Each drive is a row-synchronised pair:
  S-<stem>.csv  smartphone: accelerometer, GRAVITY (a real channel, not a filter output),
                gyroscope (Yaw/Pitch/Roll, not X/Y/Z), magnetometer, phone GPS
  V-<stem>.csv  vehicle CAN + reference GNSS: wheel speeds, indicated vehicle speed,
                steering, yaw rate, brake, gear, engine speed

Both files have identical row counts and a hard 10.000 Hz clock (V- carries an explicit
"Sample period (seconds)" column that reads exactly 0.1 for every row).

THREE UNIT MISLABELS IN THE PUBLISHED HEADERS. Every one of them was verified numerically
against a second, independent column before being acted on:

  1. S- " GPS SPEED (Kmh)" is in METRES PER SECOND, not km/h.
     Evidence: on S3a it peaks at 26.83 while the CAN bus reads 98.18 km/h for the same
     rows; 26.83 * 3.6 = 96.6. median(CAN_mps / GPS_raw) = 1.018.
     The previous loader divided this column by 3.6, making every speed label 3.6x too
     small - which is why a motorway drive appeared to top out at 10.6 km/h.

  2. V- " Wheel Speed * (rad/sec)" columns are in KM/H, not rad/s.
     Evidence: corr(mean wheel, indicated vehicle speed) = 0.999987 and the ratio is
     0.9988. Treating them as rad/s implies an effective rolling radius of 0.2781 m,
     which is exactly 1/3.6 - the signature of a missing unit conversion, not a tyre.

  3. S- " GPS SPEED" is also unusable as a training label regardless of units: it is
     step-quantised (median sample-to-sample change is exactly 0) and carries glitches up
     to 134 m/s^2 of implied acceleration. CAN indicated vehicle speed on the same drive
     peaks at 6.6 m/s^2 with p99 = 2.27. Speed labels come from CAN.
"""

import os
import re
import glob
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

R_EARTH = 6371000.0
G0 = 9.80665

# Physical plausibility gate. A passenger car lives inside ~6 m/s^2 longitudinally; the
# CAN label satisfies this on every drive. Anything beyond it is a data fault, not driving.
MAX_LONGITUDINAL_ACCEL = 6.0
ACCEL_VIOLATION_FRACTION = 0.001   # tolerated fraction of samples above the limit

# A drive that barely moves cannot exercise dead reckoning: Vw1 covers 27 m in 20 minutes
# and Vw15 covers 9 m. Blackout drift is undefined when the blackout distance is ~0, so
# these are excluded rather than allowed to produce a divide-by-noise percentage.
MIN_PATH_LENGTH_M = 200.0
MIN_MEAN_SPEED_MPS = 1.0

# integral(v dt) vs GPS path length. Real drives land at 0.05-1.2%; Y1 sits at 10.6%,
# which means its CAN label and its reference GNSS track disagree about how far the car
# went. Flagged, not silently accepted.
MAX_SPEED_INTEGRAL_ERROR = 0.03

# GNSS course over ground is meaningless below walking pace; below this it is held.
HEADING_VALID_MIN_SPEED = 2.0   # m/s


def _canon(col: str) -> str:
    """
    Canonical column key: drop the unit parenthetical, collapse whitespace, uppercase.
    The published headers carry leading spaces, a trailing space on one column, and
    mojibake superscripts ('m/s\\xb2') from a latin-1 round-trip, so exact matching on the
    raw string is not viable.
    """
    return re.sub(r"\s+", " ", col.split("(")[0]).strip().upper()


def _resolve(df: pd.DataFrame, wanted: str, where: str) -> np.ndarray:
    """Fetch a column by canonical name, or raise. Never returns a fabricated default."""
    lookup = {_canon(c): c for c in df.columns}
    if wanted not in lookup:
        raise KeyError(
            f"{where}: required column {wanted!r} not present. "
            f"Available: {sorted(lookup.keys())}"
        )
    return pd.to_numeric(df[lookup[wanted]], errors="coerce").values.astype(float)


def _has(df: pd.DataFrame, wanted: str) -> bool:
    return wanted in {_canon(c) for c in df.columns}


def geodetic_to_enu(lat, lon, lat0, lon0):
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    lat0_r, lon0_r = np.deg2rad(lat0), np.deg2rad(lon0)
    return (R_EARTH * (lon_r - lon0_r) * np.cos(lat0_r), R_EARTH * (lat_r - lat0_r))


class IOVNBDDrive:
    """
    One synchronised S-/V- pair, standardised into the column names the pipeline expects.

    Sensor inputs come from the phone (S-). Ground truth - position, heading, speed - comes
    from the vehicle (V-). That split is the whole experiment: phone sensors in,
    vehicle-grade reference out.
    """

    def __init__(self, s_df: pd.DataFrame, v_df: pd.DataFrame, name: str, driver_id: str,
                 speed_source: str = "can_indicated", strict: bool = True):
        if len(s_df) != len(v_df):
            raise ValueError(
                f"{name}: S- has {len(s_df)} rows, V- has {len(v_df)}. These files are "
                "documented as row-synchronised; a mismatch means the pair is not aligned "
                "and must not be used for supervised training."
            )
        self.name = name
        self.driver_id = driver_id
        self.speed_source = speed_source
        self.strict = strict
        # Populated by _assert_physical. A flagged drive is usable but must be reported
        # separately, never folded silently into a headline average.
        self.quality_flags: List[str] = []
        self.standardized_df = self._build(s_df, v_df, speed_source)
        self.num_samples = len(self.standardized_df)
        t = self.standardized_df["timestamp"].values
        self.duration_sec = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        self.dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
        self.median_dt = self.dt
        self.sample_rate = 1.0 / self.dt if self.dt > 0 else 10.0
        self.integrity_hash = hashlib.sha256(
            self.standardized_df.select_dtypes(include=[np.number]).values.tobytes()
        ).hexdigest()[:12]

    # ── construction ─────────────────────────────────────────────────────────
    def _build(self, s: pd.DataFrame, v: pd.DataFrame, speed_source: str) -> pd.DataFrame:
        out = pd.DataFrame()
        n = len(s)

        # Time: the S- clock, in seconds from the drive's own start. Preserved, never
        # regenerated - dt is recorded so downstream code can see gaps instead of
        # integrating across them as if they were normal steps.
        t_ms = _resolve(s, "TIME SINCE START", f"{self.name} S-")
        t = (t_ms - t_ms[0]) / 1000.0
        if not np.all(np.diff(t) > 0):
            raise ValueError(f"{self.name}: S- timestamps are not strictly increasing")
        out["timestamp"] = t
        dt_arr = np.diff(t, prepend=t[0])
        dt_arr[0] = dt_arr[1] if n > 1 else 0.1
        out["dt"] = dt_arr
        med = float(np.median(dt_arr[dt_arr > 0]))
        out["gap_mask"] = dt_arr > 3.0 * med
        self.n_gaps = int(out["gap_mask"].sum())

        # Ground-truth position and heading from the vehicle GNSS, not the phone's.
        lat = _resolve(v, "LATITUDE", f"{self.name} V-")
        lon = _resolve(v, "LONGITUDE", f"{self.name} V-")
        self.lat0, self.lon0 = float(lat[0]), float(lon[0])
        out["pos_x"], out["pos_y"] = geodetic_to_enu(lat, lon, self.lat0, self.lon0)
        # ── Speed label ──────────────────────────────────────────────────────
        can_kmh = _resolve(v, "INDICATED VEHICLE SPEED", f"{self.name} V-")
        wheel_cols = [c for c in v.columns if "WHEEL SPEED" in _canon(c)]
        if len(wheel_cols) != 4:
            raise KeyError(f"{self.name}: expected 4 wheel-speed columns, found {wheel_cols}")
        # Mislabel #2: these read km/h despite the "(rad/sec)" header.
        wheel_kmh = np.nanmean(
            np.column_stack([pd.to_numeric(v[c], errors="coerce").values for c in wheel_cols]),
            axis=1)

        out["speed_can"] = can_kmh / 3.6
        out["speed_wheel"] = wheel_kmh / 3.6
        # Mislabel #1: the phone's "(Kmh)" column is metres per second already.
        out["speed_gps_phone"] = _resolve(s, "GPS SPEED", f"{self.name} S-")

        sources = {"can_indicated": "speed_can", "wheel": "speed_wheel",
                   "gps_phone": "speed_gps_phone"}
        if speed_source not in sources:
            raise ValueError(f"unknown speed_source {speed_source!r}, pick from {list(sources)}")
        out["speed"] = out[sources[speed_source]].values

        # V- heading is a clockwise-from-north azimuth in degrees, matching the ENU
        # convention used by the EKF (x = East = v*sin(psi), y = North = v*cos(psi)).
        #
        # But GNSS heading is course-over-ground, and course over ground is undefined when
        # you are not going anywhere: at a standstill it is pure noise that swings through
        # the full circle. Taken raw it made S3c appear to rotate 267 degrees during a 90 s
        # window in which the phone gyro and the CAN yaw rate agreed to within 4.2 degrees.
        # Using it as a heading reference was injecting that noise straight into the EKF.
        # Hold the last heading measured while genuinely moving instead.
        raw_hdg = (np.deg2rad(_resolve(v, "HEADING", f"{self.name} V-")) + np.pi) \
            % (2 * np.pi) - np.pi
        spd_for_hdg = can_kmh / 3.6
        moving = spd_for_hdg > HEADING_VALID_MIN_SPEED
        hdg = raw_hdg.copy()
        if moving.any():
            last = raw_hdg[np.argmax(moving)]
            for i in range(len(hdg)):
                if moving[i]:
                    last = raw_hdg[i]
                hdg[i] = last
        out["heading"] = hdg
        out["heading_valid"] = moving
        self.heading_valid_fraction = float(np.mean(moving))

        # ── Phone IMU ────────────────────────────────────────────────────────
        out["acc_x"] = _resolve(s, "ACCELEROMETER X", f"{self.name} S-")
        out["acc_y"] = _resolve(s, "ACCELEROMETER Y", f"{self.name} S-")
        out["acc_z"] = _resolve(s, "ACCELEROMETER Z", f"{self.name} S-")

        # Gyroscope axis order does NOT match the accelerometer/gravity axis order.
        #
        # Accelerometer and GRAVITY both put vertical on Z (GRAVITY Z ~ 9.806, X and Y
        # within +/-0.2). The gyroscope does not: on every drive tested, the channel
        # labelled "Pitch" is the one carrying vehicle yaw. Correlations against the CAN
        # bus yaw rate, after per-drive lag correction:
        #
        #   S3a 0.974   S3c 0.996   S1 0.948   M 0.778   (channel: Pitch, every time)
        #
        # while the channel labelled "Yaw" correlates 0.000. Magnitudes agree too: CAN yaw
        # p99 = 0.5428 rad/s, "Pitch" p99 = 0.5504, "Yaw" p99 = 0.1072.
        #
        # So vertical is mapped by measurement, not by column name. Only two gyro-derived
        # quantities are used downstream - the component along gravity (yaw rate) and the
        # vector magnitude - and the magnitude is permutation-invariant, so getting the
        # vertical axis right is sufficient; the remaining two axes keep their labels.
        out["gyro_x"] = _resolve(s, "GYROSCOPE ROLL", f"{self.name} S-")
        out["gyro_y"] = _resolve(s, "GYROSCOPE YAW", f"{self.name} S-")
        out["gyro_z"] = _resolve(s, "GYROSCOPE PITCH", f"{self.name} S-")

        # A real gravity channel, so frame alignment need not low-pass for it. Note its
        # per-axis std here is ~0.01 m/s^2: the phone was rigidly pre-aligned to the
        # vehicle and effectively never tilted, which is exactly why arbitrary-orientation
        # performance cannot be measured on this dataset without synthetic rotation.
        for ax in "XYZ":
            out[f"grav_{ax.lower()}"] = _resolve(s, f"GRAVITY {ax}", f"{self.name} S-")
        gmag = np.linalg.norm(out[["grav_x", "grav_y", "grav_z"]].values, axis=1)
        self.gravity_magnitude_mean = float(np.mean(gmag))
        self.gravity_axis_std = out[["grav_x", "grav_y", "grav_z"]].values.std(axis=0).tolist()

        for ax in "XYZ":
            out[f"mag_{ax.lower()}"] = _resolve(s, f"MAGNETIC FIELD {ax}", f"{self.name} S-")

        # ── CAN extras, kept because they are free ground truth ──────────────
        out["can_yaw_rate"] = np.deg2rad(_resolve(v, "YAW RATE", f"{self.name} V-"))
        out["can_long_accel"] = _resolve(v, "INDICATED LONGITUDINAL ACCELERATION",
                                         f"{self.name} V-") * G0
        out["can_lat_accel"] = _resolve(v, "INDICATED LATERAL ACCELERATION",
                                        f"{self.name} V-") * G0
        if _has(v, "BRAKE POSITION"):
            out["can_brake"] = _resolve(v, "BRAKE POSITION", f"{self.name} V-")
        if _has(v, "STEERING ANGLE"):
            out["can_steering_deg"] = _resolve(v, "STEERING ANGLE", f"{self.name} V-")

        # Sensors the phone logs in SHL and in our own Android app, absent here. NaN rather
        # than a plausible-looking constant, so downstream code can tell "not fitted" from
        # "read zero" - the old loader defaulted ambient light to 1500 lux, silently
        # disabling the tunnel detector on every drive while looking like daylight.
        out["ambient_lux"] = np.nan
        out["pressure_hpa"] = np.nan
        out["baro_altitude_m"] = np.nan
        out["step_events"] = 0.0
        self.has_light = self.has_barometer = self.has_step_detector = False
        self.has_magnetometer = True

        out["driver_id"] = self.driver_id
        self._assert_physical(out)
        return out

    # ── validation ───────────────────────────────────────────────────────────
    def _assert_physical(self, out: pd.DataFrame):
        """
        Reject implausible dynamics loudly. No clipping: a drive whose label implies 2.5 g
        of longitudinal acceleration is telling you the label is wrong, and silently
        clamping it to 6 m/s^2 would hide exactly the defect worth knowing about.
        """
        v = out["speed"].values
        dt = np.maximum(out["dt"].values, 1e-6)

        # CAN speed is quantised at roughly 0.1 km/h. Differencing a quantised signal at
        # 10 Hz amplifies that quantisation into spurious spikes: on Vta1b a genuine
        # ~4.8 m/s^2 brake (as measured by the vehicle's own accelerometer) produced
        # 10 m/s^2 of *apparent* acceleration purely from step noise. A 3-sample median
        # removes the quantisation without touching real dynamics, which last ~1 s.
        v_s = pd.Series(v).rolling(3, center=True, min_periods=1).median().values
        accel = np.abs(np.diff(v_s) / dt[1:])
        self.max_implied_accel = float(np.nanmax(accel))
        self.p99_implied_accel = float(np.nanpercentile(accel, 99))
        frac = float(np.mean(accel > MAX_LONGITUDINAL_ACCEL))
        self.accel_violation_fraction = frac

        # Independent cross-check: the vehicle's own longitudinal accelerometer. This is a
        # direct measurement rather than a difference of a quantised label, so it is the
        # authority on whether the CAR exceeded the limit. Checking both means a real
        # label fault still fails even if smoothing hid it from the differenced signal.
        if "can_long_accel" in out.columns:
            a_can = np.abs(out["can_long_accel"].values)
            self.max_can_long_accel = float(np.nanmax(a_can))
            can_frac = float(np.nanmean(a_can > MAX_LONGITUDINAL_ACCEL))
            if can_frac > ACCEL_VIOLATION_FRACTION:
                self._fail(
                    f"vehicle's own longitudinal accelerometer reports {can_frac:.3%} of "
                    f"samples above {MAX_LONGITUDINAL_ACCEL} m/s^2 "
                    f"(max {self.max_can_long_accel:.1f} m/s^2)."
                )
        else:
            self.max_can_long_accel = float("nan")

        # Cross-check the label against the distance actually travelled.
        path = float(np.sum(np.hypot(np.diff(out["pos_x"].values), np.diff(out["pos_y"].values))))
        integ = float(np.sum(v * out["dt"].values))
        self.path_length_m = path
        self.speed_integral_m = integ
        self.speed_integral_error = abs(integ - path) / max(path, 1e-6)

        if path < MIN_PATH_LENGTH_M or float(np.mean(v)) < MIN_MEAN_SPEED_MPS:
            self._fail(
                f"negligible motion: {path:.0f} m of path, mean speed "
                f"{float(np.mean(v)):.2f} m/s. Dead reckoning is undefined here."
            )
        if self.speed_integral_error > MAX_SPEED_INTEGRAL_ERROR:
            self._fail(
                f"speed label and reference track disagree by "
                f"{self.speed_integral_error:.1%} (integral(v dt)={integ:.0f} m vs GPS path "
                f"{path:.0f} m). One of the two is wrong; do not train on this drive."
            )
        if frac > ACCEL_VIOLATION_FRACTION:
            self._fail(
                f"{frac:.3%} of samples imply longitudinal acceleration above "
                f"{MAX_LONGITUDINAL_ACCEL} m/s^2 (max {self.max_implied_accel:.1f}, "
                f"p99 {self.p99_implied_accel:.1f}) using speed_source={self.speed_source!r}. "
                "Not clipping - inspect the source."
            )

    def _fail(self, msg: str):
        """Raise in strict mode; otherwise record the flag and carry on."""
        if self.strict:
            raise ValueError(f"{self.name}: {msg}")
        self.quality_flags.append(msg)

    def get_data(self) -> pd.DataFrame:
        return self.standardized_df

    def summary(self) -> Dict[str, object]:
        return {
            "drive": self.name,
            "driver": self.driver_id,
            "samples": self.num_samples,
            "duration_s": round(self.duration_sec, 1),
            "sample_rate_hz": round(self.sample_rate, 3),
            "speed_source": self.speed_source,
            "max_speed_mps": round(float(np.max(self.standardized_df["speed"])), 2),
            "mean_speed_mps": round(float(np.mean(self.standardized_df["speed"])), 2),
            "max_implied_accel": round(self.max_implied_accel, 2),
            "p99_implied_accel": round(self.p99_implied_accel, 2),
            "accel_violation_frac": round(self.accel_violation_fraction, 6),
            "path_length_m": round(self.path_length_m, 1),
            "speed_integral_err": round(self.speed_integral_error, 4),
            "n_gaps": self.n_gaps,
            "sv_lag_s": round(getattr(self, "sv_lag_samples", 0) * 0.1, 1),
            "sync_corr": round(getattr(self, "sv_sync_correlation", float("nan")), 3),
            "heading_reliable": getattr(self, "heading_reliable", None),
            "gravity_axis_std": [round(x, 5) for x in self.gravity_axis_std],
            "flags": len(self.quality_flags),
            "hash": self.integrity_hash,
        }


# ── discovery / loading ──────────────────────────────────────────────────────

def dataset_root(project_root: Optional[str] = None) -> str:
    project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "data", "IO-VNBD-repo",
                        "Synchronised V abd S datasets", "Categorised IOVNB Dataset")


def _driver_of(group: str) -> str:
    """Driver id is encoded in the group directory name, e.g. 'Vf (Driver E)'."""
    m = re.search(r"Driver\s+([A-Z])", group)
    return m.group(1) if m else "Unknown"


def discover_pairs(root: Optional[str] = None) -> List[Dict[str, str]]:
    """Every synchronised S-/V- pair on disk. S- files with no V- partner are skipped:
    without CAN there is no trustworthy speed label."""
    root = root or dataset_root()
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} not found. Run scripts/fetch_iovnbd.py")
    found = []
    for cur, _, files in os.walk(root):
        for f in files:
            if not (f.startswith("S-") and f.endswith(".csv")):
                continue
            stem = f[2:-4]
            v = f"V-{stem}.csv"
            if v not in files:
                continue
            group = os.path.relpath(cur, root).split(os.sep)[0]
            found.append({"stem": stem, "group": group, "driver": _driver_of(group),
                          "s_path": os.path.join(cur, f), "v_path": os.path.join(cur, v)})
    return sorted(found, key=lambda d: (d["driver"], d["stem"]))


MAX_LAG_SAMPLES = 400          # +/- 40 s at 10 Hz
MIN_SYNC_CORRELATION = 0.55    # below this the phone is not rigidly tracking the vehicle


def estimate_sv_lag(gyro_vertical: np.ndarray, can_yaw_rate: np.ndarray,
                    max_lag: int = MAX_LAG_SAMPLES) -> Tuple[int, float]:
    """
    Estimate the sample offset between the S- (phone) and V- (vehicle) files.

    THE FILES ARE NOT ACTUALLY SYNCHRONISED, despite the directory being named
    "Synchronised V abd S datasets". Cross-correlating the phone's vertical gyro against
    the CAN bus yaw rate - the same physical quantity measured by two instruments - shows
    offsets from -8.8 s to +6.7 s depending on the drive. On S3a the correlation is 0.200
    at zero lag and 0.974 at 67 samples.

    This is the defect underneath everything else. Training paired IMU windows with speed
    labels from several seconds later, which is why held-out R2 was negative; and it
    corrupted heading, which the speed-source ablation identified as the dominant error
    term. Nothing downstream can be fixed while the inputs and labels disagree about when.

    Returns (lag_samples, correlation_at_that_lag). Positive lag means the phone stream
    trails the vehicle stream and must be advanced.
    """
    a = np.asarray(gyro_vertical, dtype=float)
    b = np.asarray(can_yaw_rate, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    max_lag = int(min(max_lag, n // 4))

    def corr(lag: int) -> float:
        if lag > 0:
            x, y = a[lag:], b[:-lag]
        elif lag < 0:
            x, y = a[:lag], b[-lag:]
        else:
            x, y = a, b
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
            return 0.0
        return float(np.corrcoef(x[m], y[m])[0, 1])

    lags = range(-max_lag, max_lag + 1)
    best = max(lags, key=lambda L: abs(corr(L)))
    return best, corr(best)


def contiguous_segments(t_sec: np.ndarray, min_len: int = 600,
                        max_gap_sec: float = 3.0) -> List[Tuple[int, int]]:
    """
    Split a timestamp series at recording discontinuities.

    Some drives concatenate more than one logging session: S3b jumps backwards by 2707 s
    at row 2042, Y1 does it three times (-30 s, -3.5 s, -304 s). Those are logger restarts,
    not jitter. Integrating across one would fabricate a teleport, and rewriting the
    timestamps - as the previous loader did by clamping every dt to 0.1 s - would silently
    glue two unrelated sessions into one drive. Splitting keeps every real sample and
    invents nothing.
    """
    dt = np.diff(t_sec)
    breaks = np.flatnonzero((dt <= 0) | (dt > max_gap_sec)) + 1
    bounds = [0, *breaks.tolist(), len(t_sec)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_len]


def load_pair(pair: Dict[str, str], max_samples: Optional[int] = None, offset: int = 0,
              speed_source: str = "can_indicated", segment: int = 0,
              min_segment_len: int = 600, strict: bool = True,
              correct_lag: bool = True) -> IOVNBDDrive:
    """
    Load one pair. `segment` selects among contiguous recording sessions (0 = longest).

    correct_lag re-aligns the phone stream against the vehicle stream by cross-correlating
    the vertical gyro with the CAN yaw rate. Leave it on: the published files are NOT
    synchronised (see estimate_sv_lag), and training on misaligned pairs is what produced
    negative held-out R2.
    """
    s = pd.read_csv(pair["s_path"], encoding="latin-1", low_memory=False)
    v = pd.read_csv(pair["v_path"], encoding="latin-1", low_memory=False)
    n = min(len(s), len(v))
    s, v = s.iloc[:n].reset_index(drop=True), v.iloc[:n].reset_index(drop=True)

    lag, sync_corr = 0, float("nan")
    if correct_lag:
        gv = _resolve(s, "GYROSCOPE PITCH", f"{pair['stem']} S-")   # vertical axis
        cy = np.deg2rad(_resolve(v, "YAW RATE", f"{pair['stem']} V-"))
        lag, sync_corr = estimate_sv_lag(gv, cy)
        if lag > 0:
            s, v = s.iloc[lag:].reset_index(drop=True), v.iloc[:-lag].reset_index(drop=True)
        elif lag < 0:
            s, v = s.iloc[:lag].reset_index(drop=True), v.iloc[-lag:].reset_index(drop=True)

    t = _resolve(s, "TIME SINCE START", f"{pair['stem']} S-") / 1000.0
    segs = contiguous_segments(t, min_len=min_segment_len)
    if not segs:
        raise ValueError(
            f"{pair['stem']}: no contiguous run of >= {min_segment_len} samples "
            f"({len(t)} rows, {int(np.sum(np.diff(t) <= 0))} backward jumps)"
        )
    segs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
    if segment >= len(segs):
        raise IndexError(f"{pair['stem']}: segment {segment} of {len(segs)} available")
    a, b = segs[segment]

    s, v = s.iloc[a:b].reset_index(drop=True), v.iloc[a:b].reset_index(drop=True)
    if offset:
        s, v = s.iloc[offset:].reset_index(drop=True), v.iloc[offset:].reset_index(drop=True)
    if max_samples and len(s) > max_samples:
        s, v = s.iloc[:max_samples], v.iloc[:max_samples]

    name = pair["stem"] if len(segs) == 1 else f"{pair['stem']}#seg{segment}"
    drive = IOVNBDDrive(s.reset_index(drop=True), v.reset_index(drop=True),
                        name=name, driver_id=pair["driver"], speed_source=speed_source,
                        strict=strict)
    drive.n_segments_in_file = len(segs)
    drive.sv_lag_samples = int(lag)
    drive.sv_sync_correlation = float(sync_corr)
    # A phone that does not track the vehicle even after lag correction was not rigidly
    # mounted for that drive. That makes its HEADING unusable, but not its speed: the
    # vibration-to-speed relationship does not depend on the yaw axis lining up. So this is
    # recorded as a separate property rather than a blanket quality flag - excluding these
    # drives from speed training would throw away most of the dataset for the wrong reason.
    drive.heading_reliable = bool(np.isfinite(sync_corr)
                                  and abs(sync_corr) >= MIN_SYNC_CORRELATION)
    return drive


def segments_available(pair: Dict[str, str], min_segment_len: int = 600) -> int:
    s = pd.read_csv(pair["s_path"], encoding="latin-1", low_memory=False,
                    usecols=lambda c: "TIME SINCE START" in _canon(c))
    t = _resolve(s, "TIME SINCE START", f"{pair['stem']} S-") / 1000.0
    return len(contiguous_segments(t, min_len=min_segment_len))


def load_benchmark_suite(max_samples_per_drive: int = 12000,
                         speed_source: str = "can_indicated",
                         include_flagged: bool = False,
                         heading_reliable_only: bool = False) -> Dict[str, object]:
    """
    Every usable real drive, with the quality gates applied.

    Of the 32 synchronised pairs on disk: 24 pass clean (~308 min), 5 are flagged
    (Y1 label/track disagreement 10.6%; Vta1b and Vw16b exceed 6 m/s^2; Vw1 and Vw15 are
    parked recordings covering 27 m and 9 m), and 3 are too short to hold a 90 s blackout
    (Vw9, Vw13, Vw17). Flagged drives are excluded by default and reported separately -
    they are not deleted, because "which drives we dropped and why" is part of the result.
    """
    pairs = discover_pairs()
    clean, flagged, unusable = [], [], []
    for p in pairs:
        try:
            d = load_pair(p, max_samples=max_samples_per_drive,
                          speed_source=speed_source, strict=False)
        except Exception as exc:                     # too short / unparseable
            unusable.append({"stem": p["stem"], "reason": str(exc)})
            continue
        (flagged if d.quality_flags else clean).append(d)

    drives = clean + (flagged if include_flagged else [])
    heading_ok = [d for d in drives if getattr(d, "heading_reliable", False)]
    if heading_reliable_only:
        drives = heading_ok
    return {
        "drives": drives,
        "clean": clean,
        "flagged": flagged,
        "unusable": unusable,
        "heading_reliable": heading_ok,
        "speed_source": speed_source,
        "provenance": (f"REAL IO-VNBD, {len(clean)} clean drives "
                       f"({len(heading_ok)} with reliable phone-to-vehicle yaw), "
                       f"speed label = {speed_source}, S-/V- lag corrected"),
    }


def split_by_driver(drives: List[IOVNBDDrive], holdout_driver: str
                    ) -> Tuple[List[IOVNBDDrive], List[IOVNBDDrive]]:
    """Leave-one-driver-out split. Generalisation across drivers and vehicles is the
    property that matters; a random window split would leak the same drive into both."""
    train = [d for d in drives if d.driver_id != holdout_driver]
    test = [d for d in drives if d.driver_id == holdout_driver]
    return train, test
