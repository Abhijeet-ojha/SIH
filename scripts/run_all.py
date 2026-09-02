"""
scripts/run_all.py
Master End-to-End Execution Pipeline and Scientific Validation Suite for SIH 2026 PS-168.
Orchestrates:
  Stage 1:  Repository Unit Tests & Interface Validation (tests/)
  Stage 2:  Target Signal Quality Audit (scripts/ml_phase4_target_audit.py)
  Stage 3:  Causal Window Length Sweep & Latency Benchmark (scripts/ml_phase1_causal_sweep.py)
  Stage 4:  Feature Group Ablation Study (scripts/ml_feature_ablation.py)
  Stage 5:  Master ML Speed Model Benchmark (RF, HistGB, XGB, Temporal CNN) (scripts/ml_benchmark_suite.py)
  Stage 6:  Motion Regimes & Sensor Robustness Analysis (scripts/ml_regime_and_robustness.py)
  Stage 7:  7-Stage System Ablation Benchmark (A -> G) (scripts/ml_system_ablation.py)
  Stage 8:  Production Training & 6-State EKF Fusion (scripts/02_train_and_fuse.py)
  Stage 9:  Edge Model Export & Python <-> Edge Parity Verification (scripts/verify_edge_parity.py)
  Stage 10: Multi-Driver Cross-Validation Benchmark (scripts/03_evaluate_and_benchmark.py)
  Stage 11: Export Interactive Visualizer Data (scripts/export_dashboard_data.py)
  Stage 12: Final Engineering Scorecard Generation
"""

import os
import sys
import time
import subprocess
import importlib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_stage(title: str, module_path: str, function_name: str = "main"):
    print("\n" + "=" * 85)
    print(f"STAGE: {title}")
    print("=" * 85)
    t0 = time.perf_counter()
    mod = importlib.import_module(module_path)
    fn = getattr(mod, function_name)
    fn()
    elapsed = time.perf_counter() - t0
    print(f"[COMPLETED] {title} in {elapsed:.2f}s")


def run_unit_tests():
    print("\n" + "=" * 85)
    print("STAGE 1: RUNNING UNIT & INTEGRATION TEST SUITE (tests/)")
    print("=" * 85)
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(PROJECT_ROOT, "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Unit test suite failed! Stopping pipeline.")
    print("[PASS] All unit and integration tests passed cleanly.")


def generate_final_scorecard():
    print("\n" + "=" * 105)
    print("FINAL MASTER ENGINEERING SCORECARD — SIH 2026 PS-168")
    print("=" * 105)
    
    scorecard_path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments", "ml_models_benchmark_scorecard.csv")
    if os.path.exists(scorecard_path):
        df_models = pd.read_csv(scorecard_path)
        print("\n1. Model Architecture & Deployment Scorecard:")
        print(df_models.to_string(index=False))

    ablation_path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments", "system_ablation_scorecard.csv")
    if os.path.exists(ablation_path):
        df_abl = pd.read_csv(ablation_path)
        print("\n2. System Ablation Attribution (A -> G):")
        print(df_abl.to_string(index=False))

    parity_path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments", "edge_parity_report.json")
    if os.path.exists(parity_path):
        import json
        with open(parity_path) as f:
            p_data = json.load(f)
        print("\n3. Python <-> Edge Deployment Parity:")
        print(f"   - Parity Verified: {p_data.get('parity_passed')} (Max diff: {p_data.get('max_prediction_diff', 0):.8f} m/s <= {p_data.get('tolerance')})")
        print(f"   - Edge Latency:    {p_data.get('edge_latency_ms_per_window', 0):.3f} ms / window")

    print("\n" + "=" * 105)
    print("PIPELINE COMPLETE: ALL EXPERIMENTS, AUDITS, PARITY TESTS, AND ARTIFACTS PRODUCED.")
    print("=" * 105)


def main():
    start_total = time.perf_counter()
    print("=" * 85)
    print("SIH 2026 PS-168 — UNIVERSAL GNSS-DENIED LOCALIZATION ENGINE MASTER PIPELINE")
    print("=" * 85)

    # 1. Tests
    run_unit_tests()

    # 2. Target Signal Quality Audit
    run_stage("Phase 4: Target Signal Quality Audit", "scripts.ml_phase4_target_audit", "audit_target_signals")

    # 3. Causal Window Sweep
    run_stage("Phase 1: Causal Window Length Sweep", "scripts.ml_phase1_causal_sweep", "run_causal_window_sweep")

    # 4. Feature Ablation Study
    run_stage("Phase 2: Feature Group Ablation", "scripts.ml_feature_ablation", "run_feature_ablation_study")

    # 5. Master ML Benchmark Suite (RF, HistGB, XGB, Temporal CNN)
    run_stage("Master ML Benchmark Suite & LODrO", "scripts.ml_benchmark_suite", "run_ml_benchmark_suite")

    # 6. Motion Regimes & Robustness Testing
    run_stage("Motion Regimes & Sensor Robustness", "scripts.ml_regime_and_robustness", "run_regime_and_robustness_analysis")

    # 7. System Ablation Benchmark (A -> G)
    run_stage("7-Stage System Ablation", "scripts.ml_system_ablation", "run_system_ablation")

    # 8. Production Training & 6-State EKF Fusion
    run_stage("Production Training & 6-State EKF Fusion", "scripts.02_train_and_fuse", "main")

    # 9. Edge Export & Parity Verification
    run_stage("Edge Model Export & Numerical Parity", "scripts.verify_edge_parity", "main")

    # 10. Multi-Driver Benchmark
    run_stage("Multi-Driver Benchmark Suite", "scripts.03_evaluate_and_benchmark", "main")

    # 11. Dashboard Data Export
    run_stage("Export Dashboard Visualizer Data", "scripts.export_dashboard_data", "main")

    # 12. Final Engineering Scorecard
    generate_final_scorecard()

    total_time = time.perf_counter() - start_total
    print(f"\n>> Complete Scientific Master Pipeline Executed Successfully in {total_time:.2f}s <<\n")


if __name__ == "__main__":
    main()
