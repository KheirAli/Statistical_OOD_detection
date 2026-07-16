"""L2 distance between [0,1]-normalized anomaly map and binary GT mask.

For an anomaly map A and binary GT mask M (both shape HxW):
    A_norm = (A - A.min()) / (A.max() - A.min())
    L2_sq  = sum_i (A_norm_i - M_i)^2
    MSE    = L2_sq / N

Threshold-free, monotone in heatmap quality:
- Sharp localizer (A=1 on tumor, A=0 elsewhere) → near 0
- Diffuse blob (A~0.5 over body, M sparse 1s) → ~0.25 * N
- Inverted (A=1 on healthy, A=0 on tumor) → max
"""
from __future__ import annotations

import numpy as np


def compute_mask_l2(
    anomaly_map: np.ndarray,
    gt_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    normalize: str = "minmax",
) -> dict:
    """L2 distance between normalized anomaly map and GT mask.

    Args:
        anomaly_map: (H, W) real-valued anomaly score map.
        gt_mask: (H, W) binary mask (nonzero = OOD pixel).
        valid_mask: optional (H, W) mask. Only valid pixels are scored.
        normalize: "minmax" → A = (A-min)/(max-min) over valid pixels.
                   "none"   → use raw scores (assumes already in [0,1]).

    Returns:
        dict with l2_sq (sum over valid), mse (l2_sq/n_valid),
        n_valid (count), and a_minmax_used.
    """
    if anomaly_map.shape != gt_mask.shape:
        raise ValueError(
            f"shape mismatch: anomaly_map {anomaly_map.shape} vs gt_mask {gt_mask.shape}"
        )
    if valid_mask is None:
        valid = np.ones_like(gt_mask, dtype=bool)
    else:
        if valid_mask.shape != gt_mask.shape:
            raise ValueError(
                f"shape mismatch: valid_mask {valid_mask.shape} vs gt_mask {gt_mask.shape}"
            )
        valid = valid_mask.astype(bool)

    a = anomaly_map.astype(np.float64)
    m = (gt_mask.astype(bool)).astype(np.float64)

    if normalize == "minmax":
        a_v = a[valid]
        if a_v.size == 0:
            return {"l2_sq": float("nan"), "mse": float("nan"),
                    "n_valid": 0, "a_min": float("nan"), "a_max": float("nan")}
        a_min, a_max = float(a_v.min()), float(a_v.max())
        denom = max(a_max - a_min, 1e-12)
        a_norm = (a - a_min) / denom
    elif normalize == "none":
        a_norm = a
        a_min, a_max = float("nan"), float("nan")
    else:
        raise ValueError(f"unknown normalize={normalize!r}")

    diff_sq = (a_norm - m) ** 2
    l2_sq = float(diff_sq[valid].sum())
    n_valid = int(valid.sum())
    mse = l2_sq / max(n_valid, 1)
    return {"l2_sq": l2_sq, "mse": mse, "n_valid": n_valid,
            "a_min": a_min, "a_max": a_max}
