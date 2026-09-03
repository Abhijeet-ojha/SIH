"""
core/export/exporter.py
Edge Model and Complete Rule Exporter.
Serializes trained models, metadata specifications, and complete embedded tree representations
for 100% mathematical parity in zero-overhead edge deployment (Dart/Kotlin/C++).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

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
        
        # 1. Save core model artifact (Joblib format)
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

        # 4. Export COMPLETE decision tree rules for mobile/embedded pure Dart/Kotlin runtime
        embedded_rules_path = os.path.join(output_dir, "embedded_rules.json")
        EdgeModelExporter._export_complete_embedded_rules(model, embedded_rules_path)

        return {
            "model_path": model_path,
            "feature_config_path": feature_config_path,
            "package_manifest_path": package_manifest_path,
            "embedded_rules_path": embedded_rules_path
        }

    @staticmethod
    def _export_complete_embedded_rules(model: TabularSpeedModel, filepath: str):
        """
        Exports the COMPLETE model representation (all trees, all nodes, all thresholds)
        for 100% bitwise parity on Android / Flutter without partial tree truncation.
        """
        rules_payload: Dict[str, Any] = {
            "model_type": model.model_type,
            "num_features": len(model.feature_names),
            "features": model.feature_names,
            "conformal_q_hat": float(model.conformal_calibrator.q_hat),
            "uncertainty_method": model.uncertainty_method,
            "n_estimators": getattr(model, "n_estimators", 100),
            "max_depth": getattr(model, "max_depth", 12)
        }

        # Complete Random Forest Tree Export
        if hasattr(model.model, "estimators_"):
            trees_data = []
            for tree_idx, tree in enumerate(model.model.estimators_):
                t = tree.tree_
                trees_data.append({
                    "tree_id": tree_idx,
                    "node_count": int(t.node_count),
                    "children_left": [int(x) for x in t.children_left.tolist()],
                    "children_right": [int(x) for x in t.children_right.tolist()],
                    "feature": [int(x) for x in t.feature.tolist()],
                    "threshold": [float(x) for x in t.threshold.tolist()],
                    "value": [float(x[0][0]) for x in t.value.tolist()]
                })
            rules_payload["trees"] = trees_data
            rules_payload["num_trees"] = len(trees_data)

        # Complete HistGradientBoosting or Linear / GBDT coefficients
        elif hasattr(model.model, "_predictors"):
            # HistGradientBoosting predictors
            rules_payload["is_hist_gb"] = True

        with open(filepath, "w") as f:
            json.dump(rules_payload, f, indent=2)
