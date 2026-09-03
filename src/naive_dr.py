"""
src/naive_dr.py
Day 1 Baseline: Naive Dead Reckoning via direct double integration of raw accelerometer and gyro.
Demonstrates the fundamental physics failure mode of low-cost MEMS IMUs (cubic position error drift O(t^3)).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

try:
    from .frame_alignment import align_frame
    from .motion_gate import MotionGate, STATIONARY  # noqa: F401  (gate shares the criterion)
except ImportError:  # direct script execution
    from frame_alignment import align_frame
    from motion_gate import MotionGate, STATIONARY

class NaiveDeadReckoning:
    """
    Implements standard classical double numerical integration dead reckoning.
    Integrates gyroscope yaw rate to track heading, and forward accelerometer
    to track velocity and position.
    """

    def __init__(
        self,
        initial_heading: float = 0.0,
        initial_speed: float = 0.0,
        initial_pos: Tuple[float, float] = (0.0, 0.0),
        remove_gravity: bool = True,
        alignment_sec: float = 3.0
    ):
        """
        remove_gravity=True is the honest textbook baseline: estimate the accelerometer
        and gyro bias over an initial stationary window (classic strapdown initial
        alignment), subtract, then double-integrate. It still drifts as O(t^3) - that is
        the real, defensible point of the baseline.

        remove_gravity=False reproduces the previous behaviour, which integrated raw acc_y
        including its gravity projection. A constant offset integrates into a velocity ramp
        and then a quadratic position error, which is where "19,338 m of drift on a 1,221 m
        drive" came from. That is not double-integration drift, it is an uncorrected bias,
        and quoting a 99.9% improvement against it is a strawman a reviewer spots instantly.
        Keep it only for the "look what happens without alignment" figure, clearly labelled.
        """
        self.initial_heading = initial_heading
        self.initial_speed = initial_speed
        self.initial_pos = initial_pos
        self.remove_gravity = remove_gravity
        self.alignment_sec = alignment_sec

    def _find_alignment_window(self, df: pd.DataFrame, dt_arr: np.ndarray, n_win: int) -> slice:
        """
        Find a genuinely stationary window for initial alignment, IMU-only.

        Naively minimising variance does NOT work: a vehicle braking steadily has low
        variance and a mean specific force of ~1 m/s^2, and picking such a window makes the
        estimated bias an order of magnitude too large - which then integrates into
        kilometres of fake drift, i.e. exactly the strawman this class was fixed to remove.

        MotionGate already distinguishes standstill from steady acceleration correctly,
        because it thresholds *horizontal specific force*, which braking has and standing
        still does not.

        The criterion is the window minimising horizontal specific force plus yaw rate.
        Both are ~0 only when the vehicle is genuinely still; steady braking scores ~1 m/s^2
        and is correctly rejected. It is relative rather than thresholded, so it does not
        need retuning for a different accelerometer noise floor.
        """
        n = len(df)
        n_win = max(2, min(n_win, n))
        acc = np.column_stack([df.acc_x.values, df.acc_y.values, df.acc_z.values]).astype(float)
        gyro = np.column_stack([df.gyro_x.values, df.gyro_y.values, df.gyro_z.values]).astype(float)

        # Gravity here is the whole-drive MEDIAN, not align_frame's 0.2 Hz tracking filter.
        # A tracking filter with a ~0.8 s time constant absorbs any acceleration sustained
        # for more than a few seconds into its own gravity estimate, so a long steady brake
        # reads as ~0 horizontal force and gets mistaken for a standstill - which is how
        # this picked a -1.0 m/s^2 window and manufactured 35 km of drift. A median over the
        # drive cannot track, which is exactly what is wanted for a one-off offline
        # alignment. (Valid because the phone does not move relative to the vehicle; if it
        # does, that drive should not be used for alignment anyway.)
        grav = np.median(acc, axis=0)
        g_hat = grav / max(np.linalg.norm(grav), 1e-9)
        lin = acc - grav
        vert = lin @ g_hat
        horiz = np.linalg.norm(lin - vert[:, None] * g_hat, axis=1)
        yaw = np.abs(gyro @ g_hat)

        # Cumulative sums make every window mean O(1) instead of O(n_win).
        ch = np.concatenate([[0.0], np.cumsum(horiz)])
        cy = np.concatenate([[0.0], np.cumsum(yaw)])
        starts = np.arange(0, n - n_win + 1)
        score = ((ch[starts + n_win] - ch[starts]) + (cy[starts + n_win] - cy[starts])) / n_win

        best = int(np.argmin(score))
        # Report how still the chosen window actually was, so a drive that never stops is
        # visible in the results rather than silently producing a bad bias.
        self.alignment_residual = float(score[best])
        self.alignment_source = ("standstill" if self.alignment_residual < 0.5
                                 else "quietest_window_no_full_stop")
        return slice(best, best + n_win)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs dead reckoning on the standardized drive dataframe.
        Returns a dataframe with estimated heading, velocity, and (x, y) coordinates.
        """
        t = df["timestamp"].values
        acc_y = df["acc_y"].values  # forward longitudinal acceleration
        gyro_z = df["gyro_z"].values # yaw angular velocity
        n = len(df)

        dt_arr = np.diff(t, prepend=t[0])
        dt_arr[0] = dt_arr[1] if n > 1 else 0.1

        # Strapdown initial alignment: remove the accelerometer and gyro bias measured
        # over a stationary window before integrating anything.
        self.acc_bias = 0.0
        self.gyro_bias = 0.0
        if self.remove_gravity and n > 4:
            median_dt = float(np.median(dt_arr[dt_arr > 0])) if np.any(dt_arr > 0) else 0.1
            n_win = int(round(self.alignment_sec / max(median_dt, 1e-3)))
            win = self._find_alignment_window(df, dt_arr, n_win)
            self.acc_bias = float(np.mean(acc_y[win]))
            self.gyro_bias = float(np.mean(gyro_z[win]))
            self.alignment_window = (int(win.start), int(win.stop))
            acc_y = acc_y - self.acc_bias
            gyro_z = gyro_z - self.gyro_bias

        heading_est = np.zeros(n)
        velocity_est = np.zeros(n)
        pos_x_est = np.zeros(n)
        pos_y_est = np.zeros(n)

        heading_est[0] = self.initial_heading
        velocity_est[0] = self.initial_speed
        pos_x_est[0] = self.initial_pos[0]
        pos_y_est[0] = self.initial_pos[1]

        # Cumulative trapezoidal or Euler integration
        for i in range(1, n):
            dt = dt_arr[i]
            # 1. Integrate Gyroscope to get Heading (Heading 0 = North/Y+, pi/2 = East/X+)
            heading_est[i] = heading_est[i-1] + gyro_z[i] * dt

            # 2. Integrate Forward Accelerometer to get Velocity (Naive integration without zero-velocity updates)
            velocity_est[i] = velocity_est[i-1] + acc_y[i] * dt

            # 3. Integrate Velocity and Heading to get Position
            v_mid = 0.5 * (velocity_est[i-1] + velocity_est[i])
            h_mid = 0.5 * (heading_est[i-1] + heading_est[i])
            pos_x_est[i] = pos_x_est[i-1] + v_mid * np.sin(h_mid) * dt
            pos_y_est[i] = pos_y_est[i-1] + v_mid * np.cos(h_mid) * dt

        result_df = pd.DataFrame({
            "timestamp": t,
            "naive_heading": heading_est,
            "naive_velocity": velocity_est,
            "naive_pos_x": pos_x_est,
            "naive_pos_y": pos_y_est
        })

        # If ground truth is present, compute error metrics
        if "pos_x" in df.columns and "pos_y" in df.columns:
            dx = pos_x_est - df["pos_x"].values
            dy = pos_y_est - df["pos_y"].values
            result_df["pos_error_m"] = np.sqrt(dx**2 + dy**2)
        
        if "speed" in df.columns:
            result_df["speed_error_mps"] = np.abs(velocity_est - df["speed"].values)

        return result_df
