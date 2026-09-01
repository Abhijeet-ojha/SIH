"""
src/visualizer.py
High-resolution plotting module for SIH deliverables.
Generates publication-ready figures showing confidence-aware uncertainty bands,
multi-sensor context layers, and 3-way trajectory overlays.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_naive_baseline(
    gt_df: pd.DataFrame,
    naive_df: pd.DataFrame,
    output_path: str = "outputs/figures/01_naive_dr_drift.png"
):
    """
    Day 1 Deliverable: Visualizes the severe failure mode of naive double-integration dead reckoning.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = plt.figure(figsize=(15, 6), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(gt_df["pos_x"], gt_df["pos_y"], label="Ground Truth (Real IO-VNBD GPS)", color="#1b9e77", linewidth=2.5, zorder=4)
    ax1.plot(naive_df["naive_pos_x"], naive_df["naive_pos_y"], label="Naive Dead Reckoning", color="#d95f02", linestyle="--", linewidth=2.0, zorder=3)
    
    ax1.scatter([gt_df["pos_x"].iloc[0]], [gt_df["pos_y"].iloc[0]], color="#2b83ba", s=100, label="Start Point", zorder=5)
    ax1.scatter([gt_df["pos_x"].iloc[-1]], [gt_df["pos_y"].iloc[-1]], color="#1b9e77", marker="X", s=120, label="True End", zorder=5)
    ax1.scatter([naive_df["naive_pos_x"].iloc[-1]], [naive_df["naive_pos_y"].iloc[-1]], color="#d95f02", marker="X", s=120, label="Naive Drifted End", zorder=5)

    ax1.set_title("Day 1 Baseline: Naive Dead Reckoning vs Ground Truth (IO-VNBD)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("East Position (meters)", fontsize=11)
    ax1.set_ylabel("North Position (meters)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", framealpha=0.9, fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gt_df["timestamp"], gt_df["speed"], label="Ground Truth Speed", color="#1b9e77", linewidth=2.0)
    ax2.plot(naive_df["timestamp"], naive_df["naive_velocity"], label="Naive Integrated Speed (v0 + ∫a dt)", color="#d95f02", linestyle="--", linewidth=1.8)
    ax2.set_title("Velocity Drift (Accelerometer Integration)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Speed (m/s)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(naive_df["timestamp"], naive_df["pos_error_m"], color="#e41a1c", linewidth=2.0)
    ax3.set_title("Cubic Position Error Drift O(t³)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_ylabel("Position Error (meters)", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved Day 1 Baseline plot: {output_path}")

def plot_speed_model_performance(
    t_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    top_features: list,
    output_path: str = "outputs/figures/02_speed_prediction_vs_gt.png"
):
    """
    Day 2 Deliverable: Visualizes ML Speed Regressor with Confidence / Uncertainty Bands.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300, gridspec_kw={"width_ratios": [1.4, 1.0]})

    # 1. Predicted vs Ground Truth with ±2σ Confidence Band
    ax1.plot(t_test, y_test, label="Ground Truth Velocity (GPS)", color="#1b9e77", linewidth=2.2, zorder=4)
    ax1.plot(t_test, y_pred, label="AI Predicted Speed (Tree Ensemble)", color="#7570b3", linestyle="--", linewidth=1.8, zorder=5)
    
    # 95% Confidence / Uncertainty Interval
    upper_bound = y_pred + 2.0 * y_std
    lower_bound = np.maximum(0.0, y_pred - 2.0 * y_std)
    ax1.fill_between(t_test, lower_bound, upper_bound, color="#7570b3", alpha=0.25, label="Heteroscedastic Uncertainty (±2σ)", zorder=3)

    ax1.set_title("Confidence-Aware Velocity Estimation with ±2σ Uncertainty", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylabel("Speed (m/s)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=9)

    # 2. Feature Importances
    if top_features:
        names = [f[0].replace("_", " ").title() for f in top_features][::-1]
        scores = [f[1] for f in top_features][::-1]
        ax2.barh(range(len(names)), scores, color="#7570b3", alpha=0.85, edgecolor="#333333", height=0.65)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_title("Top 10 Influential IMU Vibration Features", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Feature Importance Score", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.5, axis="x")
    else:
        ax2.text(0.5, 0.5, "Feature Importances Not Available", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved Confidence-Aware ML Speed Model plot: {output_path}")

def plot_trajectory_comparison_3way(
    gt_df: pd.DataFrame,
    naive_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    blackout_start_sec: float,
    blackout_end_sec: float,
    output_path: str = "outputs/figures/03_full_trajectory_comparison.png"
):
    """
    Day 2 & Day 3 Key Deliverable: 3-Way Trajectory Comparison with Multi-Sensor Context & Blackout.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax1.plot(gt_df["pos_x"], gt_df["pos_y"], label="Ground Truth (Real IO-VNBD)", color="#2ca02c", linewidth=2.8, zorder=5)
    ax1.plot(fused_df["fused_pos_x"], fused_df["fused_pos_y"], label="Confidence-Aware EKF Fusion (Ours)", color="#1f77b4", linestyle="-", linewidth=2.2, zorder=4)
    ax1.plot(naive_df["naive_pos_x"], naive_df["naive_pos_y"], label="Naive DR (Double Integration)", color="#d62728", linestyle=":", linewidth=2.0, alpha=0.8, zorder=3)

    t = gt_df["timestamp"].values
    blackout_mask = (t >= blackout_start_sec) & (t <= blackout_end_sec)
    if np.any(blackout_mask):
        ax1.plot(gt_df.loc[blackout_mask, "pos_x"], gt_df.loc[blackout_mask, "pos_y"], 
                 color="#ff7f0e", linewidth=4.5, alpha=0.85, label=f"90s GNSS Blackout ({blackout_start_sec:.0f}s - {blackout_end_sec:.0f}s)", zorder=6)

    ax1.scatter([gt_df["pos_x"].iloc[0]], [gt_df["pos_y"].iloc[0]], color="#000000", s=90, marker="o", label="Start Point", zorder=7)
    ax1.scatter([gt_df["pos_x"].iloc[-1]], [gt_df["pos_y"].iloc[-1]], color="#2ca02c", s=110, marker="X", label="True Destination", zorder=7)

    ax1.set_title("Full Trajectory Comparison (Real IO-VNBD Dataset)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("East Position (meters)", fontsize=11)
    ax1.set_ylabel("North Position (meters)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", framealpha=0.92, fontsize=9)

    ax2.plot(naive_df["timestamp"], naive_df["pos_error_m"], label="Naive DR Error", color="#d62728", linestyle=":", linewidth=2.0)
    ax2.plot(fused_df["timestamp"], fused_df["fused_pos_error_m"], label="Confidence-Aware EKF Error", color="#1f77b4", linewidth=2.2)

    ax2.axvspan(blackout_start_sec, blackout_end_sec, color="#ff7f0e", alpha=0.2, label="90s GNSS Outage Window")

    ax2.set_title("Position Drift Over Time with Open-Loop Isolation", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Drive Time (seconds)", fontsize=11)
    ax2.set_ylabel("Error vs Ground Truth (meters)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.92, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved 3-Way Trajectory Comparison plot: {output_path}")
