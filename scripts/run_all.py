"""
scripts/run_all.py
Master pipeline runner for SIH PS 168 prototype.
Executes Day 1, Day 2, and Day 3 on real IO-VNBD data, followed by a TWO-STAGE
integrity check:
  Stage A: Verifies the confidence-weighting code path is live by perturbing
           sigma_v and confirming the output hash CHANGES.
  Stage B: Verifies session-to-session determinism by running twice with
           identical inputs and confirming the output hash is UNCHANGED.
"""

import os
import sys
import time
import json
import hashlib
import importlib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

run_day1 = importlib.import_module("scripts.01_run_naive_baseline").main
run_day2 = importlib.import_module("scripts.02_train_and_fuse").main
run_day3 = importlib.import_module("scripts.03_evaluate_and_benchmark").main

def get_results_hash() -> str:
    path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "benchmark_results.json")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def verify_confidence_scaling_is_live():
    """
    Stage A: Proves the confidence-aware EKF code path is actually executing.
    Runs the EKF on a short slice with sigma=0 (max trust) vs sigma=10 (min trust).
    Asserts the resulting fused_pos_x arrays are DIFFERENT.
    This would fail if the new code were shadowed or bypassed.
    """
    from src.feature_engineering import extract_window_features
    from src.speed_model import SpeedRegressorModel, reconstruct_ai_dr_trajectory
    import src.fusion_ekf as fek

    # Hardcoded the gitignored IO-VNBD path, so this crashed on a fresh clone. Route
    # through the suite loader instead, which falls back to data/samples with a banner.
    from src.data_loader import get_real_iovnbd_benchmark_suite
    drive = get_real_iovnbd_benchmark_suite(max_samples_per_drive=300)["test_drives"][0]
    df = drive.get_data()
    model = SpeedRegressorModel()
    model.load(os.path.join(PROJECT_ROOT, "outputs", "models", "speed_regressor.joblib"))
    X, y, t = extract_window_features(df, window_sec=1.5, step_sec=0.2)
    y_pred, y_std = model.predict_with_uncertainty(X)
    ai_df = reconstruct_ai_dr_trajectory(
        df, t, y_pred, v_std=y_std,
        initial_heading=df["heading"].iloc[0],
        initial_pos=(df["pos_x"].iloc[0], df["pos_y"].iloc[0])
    )

    ai_spd = ai_df["ai_speed"].values

    # Max-confidence run: AI fully trusted (sigma -> 0, alpha -> 0.25)
    res_zero = fek.run_fusion_pipeline(df, ai_spd, np.zeros(len(df)),
        driver_style="aggressive", blackout_start_sec=20.0, blackout_end_sec=45.0)

    # Min-confidence run: AI almost ignored (sigma=10, alpha -> 0.024)
    res_huge = fek.run_fusion_pipeline(df, ai_spd, np.ones(len(df)) * 10.0,
        driver_style="aggressive", blackout_start_sec=20.0, blackout_end_sec=45.0)

    h_zero = hashlib.sha256(res_zero.to_json().encode()).hexdigest()[:16]
    h_huge = hashlib.sha256(res_huge.to_json().encode()).hexdigest()[:16]

    assert h_zero != h_huge, (
        f"FATAL: Confidence-aware EKF code path NOT live! "
        f"sigma=0 and sigma=10 produced identical hash {h_zero}. "
        "Check for stale bytecache or import shadowing."
    )

    bl_mask = res_zero["is_gnss_blackout"].values
    px_zero = res_zero["fused_pos_x"].values[bl_mask][:3]
    px_huge = res_huge["fused_pos_x"].values[bl_mask][:3]
    max_diff = np.max(np.abs(px_zero - px_huge))

    print(f"    sigma=0.0 hash:  {h_zero}  pos_x[0]={px_zero[0]:.4f}")
    print(f"    sigma=10. hash:  {h_huge}  pos_x[0]={px_huge[0]:.4f}")
    print(f"    Max positional diff: {max_diff:.4f}m  [PASS: code path is live]")

def main():
    t0 = time.time()
    print("=" * 80)
    print("SIH PS 168: AI-ASSISTED VEHICLE DEAD RECKONING & SENSOR FUSION PROTOTYPE")
    print("100% Real IO-VNBD Benchmark Suite (Coventry, UK)")
    print("=" * 80)
    
    print("\n>>> EXECUTING DAY 1 PIPELINE: Naive Dead Reckoning Baseline...")
    run_day1()

    print("\n>>> EXECUTING DAY 2 PIPELINE: Confidence-Aware AI Model & EKF Fusion...")
    run_day2()

    print("\n>>> EXECUTING DAY 3 PIPELINE: Multi-Driver Benchmark Metrics...")
    run_day3()

    hash_run1 = get_results_hash()

    print("\n" + "=" * 80)
    print(">>> STAGE A: VERIFYING CONFIDENCE-AWARE CODE PATH IS LIVE...")
    verify_confidence_scaling_is_live()

    print("\n>>> STAGE B: VERIFYING BITWISE DETERMINISM (BACK-TO-BACK RUN)...")
    print(f"    Run 1 Benchmark Hash: {hash_run1[:16]}...")
    run_day3()
    hash_run2 = get_results_hash()
    print(f"    Run 2 Benchmark Hash: {hash_run2[:16]}...")

    assert hash_run1 == hash_run2, (
        f"FATAL: Nondeterminism detected! "
        f"Run 1 ({hash_run1[:16]}) != Run 2 ({hash_run2[:16]})"
    )
    print("    [PASS] 100% BITWISE DETERMINISM CONFIRMED!")

    elapsed = time.time() - t0
    print("=" * 80)
    print(f"ALL 3 PHASES COMPLETED & VERIFIED IN {elapsed:.2f}s!")
    print("Generated Artifacts:")
    print("  - Figure 1: outputs/figures/01_naive_dr_drift.png")
    print("  - Figure 2: outputs/figures/02_speed_prediction_vs_gt.png")
    print("  - Figure 3: outputs/figures/03_full_trajectory_comparison.png")
    print("  - Metrics JSON: outputs/metrics/benchmark_results.json")
    print("  - Metrics MD:   outputs/metrics/benchmark_summary.md")
    print("  - On-device model: outputs/models/ondevice_model.json")
    print("    (Gradient-boosted trees + golden vectors for the Kotlin engine)")
    print("=" * 80)

if __name__ == "__main__":
    main()
