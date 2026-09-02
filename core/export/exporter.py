"""
core/export/exporter.py
Edge Model and Rule Exporter.
Serializes trained models, metadata specifications, and embedded rule representations
for zero-overhead edge deployment.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from core.export.spec import FeatureConfigSpec, ModelPackageSpec
from core.models.tabular_models import TabularSpeedModel


class EdgeModelExporter:
    """Exports trained models and feature specs into self-contained deployment packages."""
    
    @staticmethod
    def export_package(
        model: TabularSpeedModel,
        feature_spec: FeatureConfigSpec,
        output_dir: str,
        package_name: str = "speed_regressor"
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save core model artifact
        model_filename = f"{package_name}.joblib"
        model_path = os.path.join(output_dir, model_filename)
        model.save(model_path)

        # 2. Save feature configuration spec
        feature_config_path = os.path.join(output_dir, "feature_config.json")
        feature_spec.save_json(feature_config_path)

        # 3. Save comprehensive model package manifest
        package_spec = ModelPackageSpec(
            model_version="2.0.0",
            model_type=model.model_type,
            uncertainty_method=model.uncertainty_method,
            conformal_q_hat=float(model.conformal_calibrator.q_hat),
            model_artifact_filename=model_filename,
            feature_config=feature_spec
        )
        package_manifest_path = os.path.join(output_dir, "model_package.json")
        package_spec.save_json(package_manifest_path)

        # 4. Export embedded decision tree rules for mobile/embedded C++/Java runtimes
        embedded_rules_path = os.path.join(output_dir, "embedded_rules.json")
        EdgeModelExporter._export_embedded_rules(model, embedded_rules_path)

        return {
            "model_path": model_path,
            "feature_config_path": feature_config_path,
            "package_manifest_path": package_manifest_path,
            "embedded_rules_path": embedded_rules_path
        }

    @staticmethod
    def _export_embedded_rules(model: TabularSpeedModel, filepath: str):
        """Exports decision rules for on-device inference without heavy runtime dependencies."""
        rules_payload = {
            "model_type": model.model_type,
            "num_features": len(model.feature_names),
            "features": model.feature_names,
            "conformal_q_hat": float(model.conformal_calibrator.q_hat),
            "uncertainty_method": model.uncertainty_method
        }

        # If Random Forest, extract lightweight tree structures
        if hasattr(model.model, "estimators_"):
            trees_data = []
            for tree in model.model.estimators_[:5]:  # Export top 5 representative trees
                t = tree.tree_
                trees_data.append({
                    "node_count": int(t.node_count),
                    "feature": t.feature.tolist()[:30],
                    "threshold": [float(x) for x in t.threshold.tolist()[:30]],
                    "value": [float(x[0][0]) for x in t.value.tolist()[:30]]
                })
            rules_payload["sample_trees"] = trees_data

        with open(filepath, "w") as f:
            json.dump(rules_payload, f, indent=2)
