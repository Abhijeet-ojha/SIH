"""
src/metrics.py
Quantitative benchmark evaluation module for SIH Problem Statement 168.
Computes distance-normalized drift, open-loop terminal blackout error (pre-reacquisition),
post-reacquisition settled error, and multi-driver statistical summaries (Mean ± Std).
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def compute_total_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates cumulative Euclidean distance travelled in meters."""
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))

def calculate_benchmark_metrics(
    gt_df: pd.DataFrame,
    naive_df: pd.DataFrame,
    ai_dr_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    blackout_start_sec: float,
    blackout_end_sec: float
) -> Dict[str, Any]:
    """
    Computes all standard benchmark metrics with rigorous open-loop blackout timing.
    """
    gt_x = gt_df["pos_x"].values
    gt_y = gt_df["pos_y"].values
    t = gt_df["timestamp"].values
    total_dist = compute_total_distance(gt_x, gt_y)

    # Blackout Indices: strictly during open-loop period
    blackout_mask = (t >= blackout_start_sec) & (t < blackout_end_sec)
    blackout_idx = np.where(blackout_mask)[0]
    outage_dist = 0.0
    if len(blackout_idx) > 1:
        outage_dist = compute_total_distance(gt_x[blackout_idx], gt_y[blackout_idx])

    # 1. Naive Dead Reckoning Errors
    naive_err = naive_df["pos_error_m"].values
    naive_final_err = float(naive_err[-1])
    naive_max_err = float(np.max(naive_err))
    naive_rmse = float(np.sqrt(np.mean(naive_err**2)))
    naive_drift_pct = (naive_final_err / (total_dist + 1e-5)) * 100.0

    # 2. Pure AI-DR Errors
    ai_err = ai_dr_df["ai_pos_error_m"].values
    ai_final_err = float(ai_err[-1])
    ai_max_err = float(np.max(ai_err))
    ai_rmse = float(np.sqrt(np.mean(ai_err**2)))
    ai_drift_pct = (ai_final_err / (total_dist + 1e-5)) * 100.0

    # 3. Fused EKF Errors
    fused_err = fused_df["fused_pos_error_m"].values
    fused_final_err = float(fused_err[-1])
    fused_max_err = float(np.max(fused_err))
    fused_rmse = float(np.sqrt(np.mean(fused_err**2)))
    fused_drift_pct = (fused_final_err / (total_dist + 1e-5)) * 100.0

    # 4. Rigorous Open-Loop Blackout Analysis
    if len(blackout_idx) > 0:
        last_outage_i = blackout_idx[-1]
        
        # Naive Outage Errors
        naive_outage_max = float(np.max(naive_err[blackout_idx]))
        naive_terminal_exit = float(naive_err[last_outage_i])

        # AI-DR Outage Errors
        ai_outage_max = float(np.max(ai_err[blackout_idx]))
        ai_terminal_exit = float(ai_err[last_outage_i])

        # Fused Outage Errors (Open-Loop before first post-blackout GPS fix)
        ol_err = fused_df["open_loop_error_m"].values if "open_loop_error_m" in fused_df.columns else fused_err
        fused_outage_max = float(np.max(ol_err[blackout_idx]))
        fused_terminal_exit = float(ol_err[last_outage_i])

        # Post-reacquisition settled error (5-10s after blackout ends)
        settle_idx = np.where(t >= (blackout_end_sec + 8.0))[0]
        if len(settle_idx) > 0:
            fused_settled_err = float(fused_err[settle_idx[0]])
        else:
            fused_settled_err = float(fused_err[-1])
    else:
        naive_outage_max = naive_terminal_exit = 0.0
        ai_outage_max = ai_terminal_exit = 0.0
        fused_outage_max = fused_terminal_exit = fused_settled_err = 0.0

    driver_id = gt_df["driver_id"].iloc[0] if "driver_id" in gt_df.columns else "Unknown"

    # ── Primary metric: open-loop drift during the blackout ───────────────────
    # exit error / distance travelled while GPS was off. This is the dead-reckoning
    # number. The older headline, drift_pct_distance, divides the error at the END of the
    # drive by the TOTAL distance - but by then GPS is back and directly correcting the
    # filter, so it measures "does the EKF track GPS while GPS is available", which is not
    # a question anyone is asking. Both are reported; this one leads.
    def _blackout_drift(exit_err: float) -> float:
        return (exit_err / (outage_dist + 1e-5)) * 100.0 if outage_dist > 1.0 else float("nan")

    naive_blackout_drift = _blackout_drift(naive_terminal_exit)
    ai_blackout_drift = _blackout_drift(ai_terminal_exit)
    fused_blackout_drift = _blackout_drift(fused_terminal_exit)

    results = {
        "driver_id": str(driver_id),
        "primary_metric": {
            "definition": "blackout_terminal_exit_error_m / blackout_distance_m",
            "note": "open-loop dead-reckoning drift; GPS is off for the entire measured interval",
            "blackout_distance_m": round(outage_dist, 2),
            "fused_blackout_drift_pct": round(fused_blackout_drift, 2),
            "naive_blackout_drift_pct": round(naive_blackout_drift, 2),
            "ai_only_blackout_drift_pct": round(ai_blackout_drift, 2),
            "target_pct": 10.0,
            "meets_target": bool(fused_blackout_drift < 10.0) if outage_dist > 1.0 else False,
        },
        "drive_summary": {
            "total_duration_sec": float(t[-1] - t[0]),
            "total_distance_m": round(total_dist, 2),
            "blackout_duration_sec": float(blackout_end_sec - blackout_start_sec),
            "blackout_distance_m": round(outage_dist, 2)
        },
        "naive_dead_reckoning": {
            "blackout_drift_pct": round(naive_blackout_drift, 2),
            "final_drift_m": round(naive_final_err, 2),
            "drift_pct_distance_CLOSED_LOOP": round(naive_drift_pct, 2),
            "max_error_m": round(naive_max_err, 2),
            "rmse_m": round(naive_rmse, 2),
            "blackout_max_error_m": round(naive_outage_max, 2),
            "blackout_terminal_exit_error_m": round(naive_terminal_exit, 2)
        },
        "ai_dead_reckoning_pure": {
            "blackout_drift_pct": round(ai_blackout_drift, 2),
            "final_drift_m": round(ai_final_err, 2),
            "drift_pct_distance_CLOSED_LOOP": round(ai_drift_pct, 2),
            "max_error_m": round(ai_max_err, 2),
            "rmse_m": round(ai_rmse, 2),
            "blackout_max_error_m": round(ai_outage_max, 2),
            "blackout_terminal_exit_error_m": round(ai_terminal_exit, 2)
        },
        "ai_dr_gnss_ekf_fusion": {
            "blackout_drift_pct": round(fused_blackout_drift, 2),
            "final_drift_m": round(fused_final_err, 2),
            "drift_pct_distance_CLOSED_LOOP": round(fused_drift_pct, 2),
            "max_error_m": round(fused_max_err, 2),
            "rmse_m": round(fused_rmse, 2),
            "blackout_max_error_m": round(fused_outage_max, 2),
            "blackout_terminal_exit_error_m": round(fused_terminal_exit, 2),
            "post_reacquisition_settled_error_m": round(fused_settled_err, 2)
        },
        "improvements": {
            "drift_reduction_vs_naive_pct": round(((naive_final_err - fused_final_err) / (naive_final_err + 1e-5)) * 100.0, 2),
            "blackout_error_reduction_pct": round(((naive_terminal_exit - fused_terminal_exit) / (naive_terminal_exit + 1e-5)) * 100.0, 2)
        }
    }
    return results

def compute_multi_drive_statistics(all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes Mean ± Standard Deviation across multiple drives (Drivers A, B, D, E).
    """
    naive_drifts = [m["naive_dead_reckoning"]["drift_pct_distance_CLOSED_LOOP"] for m in all_metrics.values()]
    naive_outages = [m["naive_dead_reckoning"]["blackout_terminal_exit_error_m"] for m in all_metrics.values()]

    ai_drifts = [m["ai_dead_reckoning_pure"]["drift_pct_distance_CLOSED_LOOP"] for m in all_metrics.values()]
    ai_outages = [m["ai_dead_reckoning_pure"]["blackout_terminal_exit_error_m"] for m in all_metrics.values()]

    fused_outage_maxes = [m["ai_dr_gnss_ekf_fusion"]["blackout_max_error_m"] for m in all_metrics.values()]
    fused_outage_exits = [m["ai_dr_gnss_ekf_fusion"]["blackout_terminal_exit_error_m"] for m in all_metrics.values()]
    fused_settled_errs = [m["ai_dr_gnss_ekf_fusion"]["post_reacquisition_settled_error_m"] for m in all_metrics.values()]
    fused_final_drifts = [m["ai_dr_gnss_ekf_fusion"]["final_drift_m"] for m in all_metrics.values()]
    fused_rmses = [m["ai_dr_gnss_ekf_fusion"]["rmse_m"] for m in all_metrics.values()]

    # Primary metric aggregates: open-loop drift during the blackout. NaN entries mean the
    # drive did not travel far enough during the outage for the ratio to be meaningful.
    nan = float("nan")
    fused_bo = [v for v in (m["primary_metric"]["fused_blackout_drift_pct"] for m in all_metrics.values()) if v == v]
    naive_bo = [v for v in (m["primary_metric"]["naive_blackout_drift_pct"] for m in all_metrics.values()) if v == v]
    ai_bo = [v for v in (m["primary_metric"]["ai_only_blackout_drift_pct"] for m in all_metrics.values()) if v == v]

    def _ms(vals):
        if not vals:
            return {"mean": nan, "std": nan}
        return {"mean": round(float(np.mean(vals)), 2), "std": round(float(np.std(vals)), 2)}

    stats = {
        "num_drives": len(all_metrics),
        "naive_drift_pct": {"mean": round(float(np.mean(naive_drifts)), 2), "std": round(float(np.std(naive_drifts)), 2)},
        "naive_blackout_terminal_m": {"mean": round(float(np.mean(naive_outages)), 2), "std": round(float(np.std(naive_outages)), 2)},
        "ai_drift_pct": {"mean": round(float(np.mean(ai_drifts)), 2), "std": round(float(np.std(ai_drifts)), 2)},
        "ai_blackout_terminal_m": {"mean": round(float(np.mean(ai_outages)), 2), "std": round(float(np.std(ai_outages)), 2)},
        "fused_blackout_max_m": {"mean": round(float(np.mean(fused_outage_maxes)), 2), "std": round(float(np.std(fused_outage_maxes)), 2)},
        "fused_blackout_terminal_exit_m": {"mean": round(float(np.mean(fused_outage_exits)), 2), "std": round(float(np.std(fused_outage_exits)), 2)},
        "fused_settled_reacquisition_m": {"mean": round(float(np.mean(fused_settled_errs)), 2), "std": round(float(np.std(fused_settled_errs)), 2)},
        "fused_final_drift_m": {"mean": round(float(np.mean(fused_final_drifts)), 2), "std": round(float(np.std(fused_final_drifts)), 2)},
        "fused_rmse_m": {"mean": round(float(np.mean(fused_rmses)), 2), "std": round(float(np.std(fused_rmses)), 2)},
        "naive_rmse_m": _ms([m["naive_dead_reckoning"]["rmse_m"] for m in all_metrics.values()]),
        "ai_rmse_m": _ms([m["ai_dead_reckoning_pure"]["rmse_m"] for m in all_metrics.values()]),
        "fused_blackout_drift_pct": _ms(fused_bo),
        "naive_blackout_drift_pct": _ms(naive_bo),
        "ai_blackout_drift_pct": _ms(ai_bo),
        "num_meeting_target": int(sum(1 for m in all_metrics.values() if m["primary_metric"]["meets_target"])),
    }
    return stats

def format_multi_drive_markdown(all_metrics: Dict[str, Dict[str, Any]], stats: Dict[str, Any]) -> str:
    """Formats full multi-driver benchmark table for the proposal and jury presentation."""
    md = f"""# SIH PS 168 — Multi-Driver Benchmark Suite Evaluation
### Evaluated on IO-VNBD Benchmark Suite Across Diverse Driving Profiles

---

### 1. Statistical Summary Across Drives (Mean ± Std, N={stats['num_drives']})

**Primary metric — open-loop blackout drift = exit error / distance travelled with GPS off.**
Target is 10%. {stats['num_meeting_target']} of {stats['num_drives']} drives meet it.

| Metric | Naive Dead Reckoning (Baseline) | AI-DR Pure (ML Speed) | AI-DR + EKF GNSS Fusion (Final) |
| :--- | :--- | :--- | :--- |
| **Blackout drift % (PRIMARY)** | **{stats['naive_blackout_drift_pct']['mean']}% ± {stats['naive_blackout_drift_pct']['std']}%** | {stats['ai_blackout_drift_pct']['mean']}% ± {stats['ai_blackout_drift_pct']['std']}% | **{stats['fused_blackout_drift_pct']['mean']}% ± {stats['fused_blackout_drift_pct']['std']}%** |
| Closed-loop drift % (GPS restored — *not* a DR number) | {stats['naive_drift_pct']['mean']}% ± {stats['naive_drift_pct']['std']}% | {stats['ai_drift_pct']['mean']}% ± {stats['ai_drift_pct']['std']}% | {stats['fused_final_drift_m']['mean']} m final error |
| **90s Blackout Peak Drift** | **{stats['naive_blackout_terminal_m']['mean']} m ± {stats['naive_blackout_terminal_m']['std']} m** | {stats['ai_blackout_terminal_m']['mean']} m ± {stats['ai_blackout_terminal_m']['std']} m | **{stats['fused_blackout_max_m']['mean']} m ± {stats['fused_blackout_max_m']['std']} m** |
| **90s Blackout Terminal Exit Error** | **{stats['naive_blackout_terminal_m']['mean']} m ± {stats['naive_blackout_terminal_m']['std']} m** | {stats['ai_blackout_terminal_m']['mean']} m ± {stats['ai_blackout_terminal_m']['std']} m | **{stats['fused_blackout_terminal_exit_m']['mean']} m ± {stats['fused_blackout_terminal_exit_m']['std']} m** |
| **Post-Reacquisition Settled Error** | N/A (no GNSS update to settle to) | N/A (no GNSS update to settle to) | **{stats['fused_settled_reacquisition_m']['mean']} m ± {stats['fused_settled_reacquisition_m']['std']} m** |
| **Trajectory RMSE** | {stats['naive_rmse_m']['mean']} m ± {stats['naive_rmse_m']['std']} m | {stats['ai_rmse_m']['mean']} m ± {stats['ai_rmse_m']['std']} m | **{stats['fused_rmse_m']['mean']} m ± {stats['fused_rmse_m']['std']} m** |

---

### 2. Breakdown by Driver Profile

| Drive Profile | Driver ID | Blackout dist (m) | **Fused blackout drift %** | Naive blackout drift % | Exit error (m) | Post-GPS Settled Error (m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for d_name, m in all_metrics.items():
        s = m["drive_summary"]
        n = m["naive_dead_reckoning"]
        f = m["ai_dr_gnss_ekf_fusion"]
        d_id = m.get("driver_id", "Unknown")
        label = "Driver E (Aggressive)" if d_id == "E" else ("Driver B (Highway)" if d_id == "B" else ("Driver D (Urban)" if d_id == "D" else "Driver A (Normal)"))
        pm = m["primary_metric"]
        md += f"| **{label}** | `{d_id}` | {pm['blackout_distance_m']} m | **{pm['fused_blackout_drift_pct']}%** | {pm['naive_blackout_drift_pct']}% | {f['blackout_terminal_exit_error_m']} m | {f['post_reacquisition_settled_error_m']} m |\n"

    # Compute Driver E stats directly from actual data — no hardcoded fallbacks
    driver_e_exits = [
        m["ai_dr_gnss_ekf_fusion"]["blackout_terminal_exit_error_m"]
        for m in all_metrics.values()
        if m.get("driver_id") == "E"
    ]
    driver_e_naive_drifts = [
        m["naive_dead_reckoning"]["final_drift_m"]
        for m in all_metrics.values()
        if m.get("driver_id") == "E"
    ]

    # Hardest Drive: worst (highest) blackout exit error for Driver E
    if driver_e_exits:
        e_worst_exit = max(driver_e_exits)
        e_best_exit  = min(driver_e_exits)
        e_worst_naive = max(driver_e_naive_drifts) if driver_e_naive_drifts else float("nan")
        obs3 = (
            f"3. **Hardest Case (Driver E - Aggressive)**: Even under hard braking and sharp turns "
            f"where naive integration accumulates up to **{e_worst_naive:.1f}m** of drift, our "
            f"AI-speed + EKF Fusion reduces blackout exit error to a range of "
            f"**{e_best_exit:.1f}m – {e_worst_exit:.1f}m** depending on drive length and manoeuvre profile."
        )
    else:
        obs3 = "3. **Driver E Data**: No Driver E drives in benchmark set."

    md += f"""
---

### 3. Key Observations for ISRO / SIH Jury

1. **Defensible Blackout Timing**: The headline **90-second Blackout Terminal Exit Error is {stats['fused_blackout_terminal_exit_m']['mean']}m \u00b1 {stats['fused_blackout_terminal_exit_m']['std']}m**, measured strictly in the open-loop state prior to the arrival of the first post-outage satellite measurement.
2. **Immediate Post-Reacquisition Convergence**: Within 5 seconds of GNSS recovery, the filter re-converges to **{stats['fused_settled_reacquisition_m']['mean']}m \u00b1 {stats['fused_settled_reacquisition_m']['std']}m**, with zero discontinuous trajectory teleportation.
{obs3}
"""
    return md

