"""
src/naive_dr.py
Day 1 Baseline: Naive Dead Reckoning via direct double integration of raw accelerometer and gyro.
Demonstrates the fundamental physics failure mode of low-cost MEMS IMUs (cubic position error drift O(t^3)).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class NaiveDeadReckoning:
    """
    Implements standard classical double numerical integration dead reckoning.
    Integrates gyroscope yaw rate to track heading, and forward accelerometer
    to track velocity and position.
    """

    def __init__(self, initial_heading: float = 0.0, initial_speed: float = 0.0, initial_pos: Tuple[float, float] = (0.0, 0.0)):
        self.initial_heading = initial_heading
        self.initial_speed = initial_speed
        self.initial_pos = initial_pos

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
