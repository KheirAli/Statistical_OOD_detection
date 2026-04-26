"""Metrics package.

Re-exports keep the legacy `from ood.metrics import ...` import surface
intact while letting new metrics (SNR, future contrast scores, …) live
in their own files.
"""
from .curves import (
    evaluate_delta_map,
    manual_auc,
    manual_average_precision,
    manual_precision_recall_curve,
    manual_roc_curve,
)
from .snr import compute_snr

__all__ = [
    "evaluate_delta_map",
    "manual_auc",
    "manual_average_precision",
    "manual_precision_recall_curve",
    "manual_roc_curve",
    "compute_snr",
]
