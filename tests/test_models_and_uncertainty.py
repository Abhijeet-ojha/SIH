"""
tests/test_models_and_uncertainty.py
Unit tests for Tabular Speed Models, Temporal (T x C) Models, and Split Conformal Calibration.
"""

import unittest
import numpy as np
import pandas as pd
from core.models.tabular_models import TabularSpeedModel
from core.models.temporal_models import TemporalSequenceSpeedModel
from core.uncertainty.calibrator import ConformalUncertaintyCalibrator, compute_uncertainty_metrics


class TestModelsAndUncertainty(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 120
        # Tabular synthetic data
        self.X_train = pd.DataFrame({
            "acc_mag_mean": np.random.uniform(9.0, 11.0, n),
            "gyro_mag_mean": np.random.uniform(0.01, 0.2, n),
            "jerk_mag_rms": np.random.uniform(0.1, 1.0, n),
            "acc_horiz_rms": np.random.uniform(0.05, 2.0, n)
        })
        self.y_train = 10.0 * self.X_train["acc_horiz_rms"].values + np.random.normal(0, 0.5, n)

        self.X_calib = pd.DataFrame({
            "acc_mag_mean": np.random.uniform(9.0, 11.0, 30),
            "gyro_mag_mean": np.random.uniform(0.01, 0.2, 30),
            "jerk_mag_rms": np.random.uniform(0.1, 1.0, 30),
            "acc_horiz_rms": np.random.uniform(0.05, 2.0, 30)
        })
        self.y_calib = 10.0 * self.X_calib["acc_horiz_rms"].values + np.random.normal(0, 0.5, 30)

        # Temporal synthetic sequences: (N, T=15, C=6)
        self.X_seq_train = np.random.randn(n, 15, 6).astype(np.float32)
        self.X_seq_calib = np.random.randn(30, 15, 6).astype(np.float32)

    def test_tabular_random_forest_training_and_uncertainty(self):
        """Tests Random Forest speed prediction and tree ensemble uncertainty."""
        model = TabularSpeedModel(model_type="random_forest", n_estimators=20, max_depth=6)
        metrics = model.train(self.X_train, self.y_train, X_calib=self.X_calib, y_calib=self.y_calib)
        self.assertIn("train_rmse", metrics)
        
        preds, sigma_v = model.predict_with_uncertainty(self.X_calib)
        self.assertEqual(len(preds), len(self.X_calib))
        self.assertEqual(len(sigma_v), len(self.X_calib))
        self.assertTrue(np.all(preds >= 0.0))
        self.assertTrue(np.all(sigma_v > 0.0))

    def test_conformal_calibrator_guarantee(self):
        """Tests that ConformalUncertaintyCalibrator achieves empirical coverage on validation set."""
        calibrator = ConformalUncertaintyCalibrator(target_coverage=0.90)
        y_true = np.array([10.0, 12.0, 15.0, 8.0, 20.0, 11.0, 14.0, 16.0, 9.0, 13.0])
        y_pred = np.array([9.8, 12.2, 14.7, 8.3, 19.5, 11.1, 14.2, 15.8, 9.2, 12.9])
        
        calibrator.calibrate(y_true, y_pred)
        cov_res = calibrator.evaluate_coverage(y_true, y_pred)
        self.assertGreaterEqual(cov_res["empirical_coverage_pct"], 80.0)

    def test_temporal_sequence_cnn_training(self):
        """Tests training and inference on the Causal Temporal Sequence CNN."""
        model = TemporalSequenceSpeedModel(in_channels=6, hidden_dim=16, epochs=5, batch_size=32)
        metrics = model.train(self.X_seq_train, self.y_train, X_calib=self.X_seq_calib, y_calib=self.y_calib)
        self.assertIn("train_rmse", metrics)

        preds, sigma = model.predict_with_uncertainty(self.X_seq_calib)
        self.assertEqual(len(preds), len(self.X_seq_calib))
        self.assertTrue(np.all(preds >= 0.0))


if __name__ == "__main__":
    unittest.main()
