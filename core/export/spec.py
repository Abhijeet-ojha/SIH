"""
core/export/spec.py
Machine-Readable Versioned Specifications for Feature Configurations and Exported Model Packages.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional


@dataclass
class FeatureConfigSpec:
    """Versioned Feature Specification for Edge Preprocessing."""
    feature_version: str = "2.0.0"
    sample_rate_hz: float = 10.0
    window_sec: float = 1.5
    step_sec: float = 0.2
    window_samples: int = 15
    step_samples: int = 2
    feature_group: str = "all"
    feature_names: List[str] = field(default_factory=list)
    num_features: int = 0
    coordinate_convention: str = "ISO_8855_VEHICLE_BODY"
    target_unit: str = "meters_per_second"
    raw_signals: List[str] = field(default_factory=list)
    stat_moments: List[str] = field(default_factory=list)
    spectral_features: List[str] = field(default_factory=list)
    cross_interactions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "FeatureConfigSpec":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ModelPackageSpec:
    """Versioned Deployment Package Metadata."""
    model_version: str = "2.0.0"
    model_type: str = "random_forest"
    uncertainty_method: str = "conformal"
    target_coverage: float = 0.90
    conformal_q_hat: float = 0.50
    training_dataset_hash: str = "IO_VNBD_BENCHMARK_2026"
    calib_split_info: str = "ISOLATED_CALIB_SPLIT_NO_TEST_LEAKAGE"
    model_artifact_filename: str = "speed_regressor.joblib"
    feature_config: FeatureConfigSpec = field(default_factory=FeatureConfigSpec)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save_json(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "ModelPackageSpec":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
