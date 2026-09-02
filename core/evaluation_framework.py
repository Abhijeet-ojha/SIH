"""
core/evaluation_framework.py
Re-exports the 5-Level Evaluation Framework from src.evaluation_framework for backward and modular compatibility.
"""

from src.evaluation_framework import (
    verify_leakage_gate,
    evaluate_ml_speed_metrics,
    evaluate_uncertainty_calibration,
    evaluate_downstream_navigation,
    benchmark_model_inference_latency
)

__all__ = [
    "verify_leakage_gate",
    "evaluate_ml_speed_metrics",
    "evaluate_uncertainty_calibration",
    "evaluate_downstream_navigation",
    "benchmark_model_inference_latency"
]
