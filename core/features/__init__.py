"""
core/features package
"""
from core.features.leakage_guard import (
    verify_feature_matrix_leakage,
    verify_split_isolation,
    verify_causality,
    LeakageGuardError
)
from core.features.extractor import CausalFeatureExtractor

__all__ = [
    "verify_feature_matrix_leakage",
    "verify_split_isolation",
    "verify_causality",
    "LeakageGuardError",
    "CausalFeatureExtractor"
]
