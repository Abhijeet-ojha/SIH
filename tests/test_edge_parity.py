"""
tests/test_edge_parity.py
Unit tests verifying numerical parity between Python native inference and edge serialization.
"""

import unittest
import numpy as np
import pandas as pd
from core.models.tabular_models import TabularSpeedModel
from core.export.parity_runner import verify_python_edge_parity


class TestEdgeParity(unittest.TestCase):

    def test_python_edge_numerical_parity(self):
        """Tests that serialized and reloaded model produces identical predictions within 1e-4 tolerance."""
        np.random.seed(42)
        n = 80
        X_train = pd.DataFrame({
            "acc_mag_mean": np.random.uniform(9.0, 11.0, n),
            "gyro_mag_mean": np.random.uniform(0.01, 0.2, n),
            "jerk_mag_rms": np.random.uniform(0.1, 1.0, n)
        })
        y_train = 5.0 * X_train["acc_mag_mean"].values + np.random.normal(0, 0.2, n)

        model = TabularSpeedModel(model_type="random_forest", n_estimators=10, max_depth=5)
        model.train(X_train, y_train)

        test_X = pd.DataFrame({
            "acc_mag_mean": np.random.uniform(9.0, 11.0, 20),
            "gyro_mag_mean": np.random.uniform(0.01, 0.2, 20),
            "jerk_mag_rms": np.random.uniform(0.1, 1.0, 20)
        })

        parity_report = verify_python_edge_parity(model, test_X, tolerance=1e-4)
        self.assertTrue(parity_report["parity_passed"])
        self.assertLessEqual(parity_report["max_prediction_diff"], 1e-4)


if __name__ == "__main__":
    unittest.main()
