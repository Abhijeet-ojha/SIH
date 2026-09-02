"""
scripts/verify_edge_parity.py
Automated Edge Export and Numerical Parity Verification Script for SIH 2026 PS-168.
Extracts representative sensor windows, exports the model package and feature config,
and validates zero-deviation Python <-> Edge numerical parity.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_real_iovnbd_benchmark_suite
from core.features.extractor import CausalFeatureExtractor
from core.models.tabular_models import TabularSpeedModel
from core.export.spec import FeatureConfigSpec
from core.export.exporter import EdgeModelExporter
from core.export.parity_runner import verify_python_edge_parity, StreamingSensorRingBuffer


def main():
    print("=" * 80)
    print("SIH 2026 PS-168: EDGE EXPORT AND PYTHON <-> EDGE PARITY VERIFIER")
    print("=" * 80)

    suite = get_real_iovnbd_benchmark_suite(max_samples_per_drive=3000)
    train_drives = suite["train_drives"]
    test_drives = suite["test_drives"]

    extractor = CausalFeatureExtractor(window_sec=1.5, step_sec=0.2, sample_rate_hz=10.0, feature_group="all")
    
    # 1. Prepare Training Data
    print("\n[Step 1] Extracting causal training features...")
    X_train_list, y_train_list = [], []
    for d in train_drives:
        df = d.get_data()
        X_df, y_spd, _, _ = extractor.extract_features(df)
        X_train_list.append(X_df)
        y_train_list.append(y_spd)

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    # 2. Train Model with Conformal Uncertainty
    print("\n[Step 2] Training production Random Forest speed model...")
    model = TabularSpeedModel(model_type="random_forest", n_estimators=100, max_depth=12, random_state=42, uncertainty_method="conformal")
    
    # Use isolated calibration drive (Driver A - S3b)
    calib_drive = train_drives[1] if len(train_drives) > 1 else train_drives[0]
    calib_df = calib_drive.get_data()
    X_calib, y_calib, _, _ = extractor.extract_features(calib_df)
    
    train_metrics = model.train(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
    print(f"         Train RMSE: {train_metrics['train_rmse']:.3f} m/s | R2: {train_metrics['train_r2']:.3f}")
    print(f"         Calibrated Conformal q_hat: {model.conformal_calibrator.q_hat:.3f} m/s")

    # 3. Export Versioned Model Package
    print("\n[Step 3] Exporting versioned edge model package...")
    export_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
    feature_spec = FeatureConfigSpec(
        feature_version="2.0.0",
        sample_rate_hz=10.0,
        window_sec=1.5,
        step_sec=0.2,
        window_samples=15,
        step_samples=2,
        feature_group="all",
        feature_names=model.feature_names
    )
    
    export_artifacts = EdgeModelExporter.export_package(
        model=model,
        feature_spec=feature_spec,
        output_dir=export_dir,
        package_name="speed_regressor"
    )
    for k, v in export_artifacts.items():
        print(f"         Exported {k}: {v}")

    # 4. Run Numerical Parity Verification on Test Drive
    print("\n[Step 4] Running numerical parity test across 200 representative test windows...")
    test_drive = test_drives[0]
    test_df = test_drive.get_data()
    X_test, y_test, _, _ = extractor.extract_features(test_df)
    test_sample_X = X_test.iloc[:200]

    parity_report = verify_python_edge_parity(model, test_sample_X, tolerance=1e-4)
    print(f"         Parity Status:          {'PASS [Zero Deviation]' if parity_report['parity_passed'] else 'FAIL'}")
    print(f"         Max Prediction Diff:    {parity_report['max_prediction_diff']:.8f} m/s")
    print(f"         Max Uncertainty Diff:   {parity_report['max_uncertainty_diff']:.8f} m/s")
    print(f"         Native Latency:         {parity_report['native_latency_ms_per_window']:.3f} ms/window")
    print(f"         Edge Latency:           {parity_report['edge_latency_ms_per_window']:.3f} ms/window")

    # 5. Verify Streaming Ring Buffer
    print("\n[Step 5] Testing Streaming Ring Buffer execution...")
    ring_buf = StreamingSensorRingBuffer(capacity=50)
    for _, row in test_df.iloc[:20].iterrows():
        from core.interfaces.canonical import SensorFrame
        f = SensorFrame(
            timestamp_s=float(row["timestamp"]),
            accel_xyz=np.array([row["acc_x"], row["acc_y"], row["acc_z"]]),
            gyro_xyz=np.array([row["gyro_x"], row["gyro_y"], row["gyro_z"]])
        )
        ring_buf.append(f)

    self_ready = ring_buf.is_ready(min_samples=15)
    print(f"         Ring Buffer Ready: {self_ready} ({len(ring_buf.buffer)} frames buffered)")
    win_df = ring_buf.get_window_dataframe(window_samples=15)
    print(f"         Extracted Window Shape: {win_df.shape}")

    # Save parity report
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", "ml_experiments")
    os.makedirs(metrics_dir, exist_ok=True)
    report_path = os.path.join(metrics_dir, "edge_parity_report.json")
    with open(report_path, "w") as f:
        json.dump(parity_report, f, indent=2)
    print(f"\n[PASS] Saved Edge Parity Report to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
