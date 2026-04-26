"""Signal-to-noise ratio for anomaly maps (Eq. 9 in our paper).

    SNR(F̂_σ(x*), M*) =
        ( Σ_{i : M*_i = 1}  F̂_σ(x*)_i^2  /  #(M* = 1) )
        ----------------------------------------------------
        ( Σ_{i : M*_i = 0}  F̂_σ(x*)_i^2  /  #(M* = 0) )

Mean-squared anomaly response on OOD pixels divided by mean-squared
response on ID pixels. AUROC measures pixel *ordering*; SNR measures how
much the OOD pixels actually *stand out* against the background. A heatmap
that's flat-but-with-tiny-bumps on OOD pixels can hit ~1.0 AUROC but have
near-1.0 SNR; both numbers together describe a method honestly.

Higher SNR = better contrast. Treat as ratio (>1 = OOD louder than ID).
"""
from __future__ import annotations

import numpy as np


def compute_snr(anomaly_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """Eq. 9 SNR for a single image.

    Args:
        anomaly_map: `(H, W)` real-valued anomaly score map.
        gt_mask: `(H, W)` binary mask. Nonzero = OOD pixel.

    Returns:
        Float SNR. NaN if either side is empty (all-OOD or no-OOD image), or
        if the ID-pixel denominator is exactly zero.

    Notes:
        Squares are taken on the raw scores — no centering, no normalization.
        That matches the paper's definition. If you want a normalized SNR
        rescale `anomaly_map` upstream.
    """
    if anomaly_map.shape != gt_mask.shape:
        raise ValueError(
            f"shape mismatch: anomaly_map {anomaly_map.shape} vs gt_mask {gt_mask.shape}"
        )
    sq = anomaly_map.astype(np.float64) ** 2
    pos = gt_mask.astype(bool)
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    den = sq[neg].sum() / n_neg
    if den == 0.0:
        return float("nan")
    num = sq[pos].sum() / n_pos
    return float(num / den)
