"""Evaluation metrics: ROC, PR, AUC — pure numpy, no sklearn."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def manual_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FPR, TPR for all thresholds."""
    desc_idx = np.argsort(-y_score)
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    distinct_idx = np.where(np.diff(y_score))[0]
    threshold_idx = np.concatenate([distinct_idx, [len(y_true) - 1]])

    tps = np.cumsum(y_true)[threshold_idx]
    fps = (threshold_idx + 1) - tps

    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])

    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    tpr = tps / tps[-1] if tps[-1] > 0 else tps

    thresholds = y_score[threshold_idx]
    return fpr, tpr, thresholds


def manual_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    return float(np.trapz(y, x))


def manual_precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision, recall for all thresholds."""
    desc_idx = np.argsort(-y_score)
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    total_pos = y_true.sum()

    precision = tps / (tps + fps)
    recall = tps / total_pos if total_pos > 0 else tps

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    thresholds = y_score

    return precision, recall, thresholds


def manual_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under PR curve)."""
    precision, recall, _ = manual_precision_recall_curve(y_true, y_score)
    return float(np.sum(np.diff(recall) * precision[:-1]))


def evaluate_delta_map(
    delta_map: np.ndarray,
    labels_fine: np.ndarray,
    gt_mask_binary: np.ndarray,
    anomaly_threshold: float = 0.5,
    smooth_sigma: Optional[float] = None,
) -> Dict:
    """Run both superpixel-level and pixel-level evaluation.

    Args:
        delta_map: (H, W) float array of anomaly scores (may contain NaN).
        labels_fine: (H, W) int array of refined superpixel labels.
        gt_mask_binary: (H, W) uint8 binary ground-truth mask.
        anomaly_threshold: fraction of anomalous pixels for SP majority vote.
        smooth_sigma: if not None, apply Gaussian smoothing to delta_map first.

    Returns:
        dict with sp_roc_auc, sp_ap, px_roc_auc, px_ap, and curve data.
    """
    # Superpixel-level uses raw (NaN-safe) scores
    score_map_sp = np.nan_to_num(delta_map, nan=0.0)

    # Pixel-level: smooth raw delta_map (NaN propagates through kernel,
    # then excluded by valid_mask — matches original full_run_orig.py)
    if smooth_sigma is not None:
        score_map_px = gaussian_filter(delta_map, sigma=smooth_sigma, mode="nearest")
    else:
        score_map_px = delta_map.copy()

    # --- Superpixel-level ---
    all_sp_ids = sorted(np.unique(labels_fine).astype(int).tolist())
    sp_scores = []
    sp_labels = []

    for sp_id in all_sp_ids:
        sp_mask = labels_fine == sp_id
        n_pix = int(sp_mask.sum())
        if n_pix < 1:
            continue

        score = float(np.nanmean(score_map_sp[sp_mask]))
        if np.isnan(score):
            continue

        frac_anomalous = gt_mask_binary[sp_mask].mean()
        label = int(frac_anomalous > anomaly_threshold)

        sp_scores.append(score)
        sp_labels.append(label)

    sp_scores = np.array(sp_scores)
    sp_labels = np.array(sp_labels)

    fpr_sp, tpr_sp, thresh_sp = manual_roc_curve(sp_labels, sp_scores)
    sp_roc_auc = manual_auc(fpr_sp, tpr_sp)
    sp_ap = manual_average_precision(sp_labels, sp_scores)

    # --- Pixel-level ---
    valid = ~np.isnan(score_map_px)
    pixel_scores = score_map_px[valid].ravel()
    pixel_labels = gt_mask_binary[valid].ravel()

    fpr_px, tpr_px, thresh_px = manual_roc_curve(pixel_labels, pixel_scores)
    px_roc_auc = manual_auc(fpr_px, tpr_px)
    px_ap = manual_average_precision(pixel_labels, pixel_scores)

    return {
        "sp_roc_auc": sp_roc_auc,
        "sp_ap": sp_ap,
        "px_roc_auc": px_roc_auc,
        "px_ap": px_ap,
        "num_superpixels": len(sp_scores),
        "num_anomalous_sp": int(sp_labels.sum()),
        "curves": {
            "sp_fpr": fpr_sp.tolist(),
            "sp_tpr": tpr_sp.tolist(),
            "px_fpr": fpr_px.tolist(),
            "px_tpr": tpr_px.tolist(),
        },
    }
