"""Delta map scoring via typical set analysis with RGB + PCA features.

This is the algorithmic core — the file agents will edit most when testing
new scoring functions. Extracted from New_dataset_clean.ipynb Cells 13-14.

The key idea: for each refined superpixel, build a factorized PMF from
DPS reconstructions (P_rgb * P_pca), compute its entropy H, then measure
how far the label image's pixels deviate: delta = |avg_nlogp - H|.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def quantize_u8_to_bins(x_u8: np.ndarray, bins: int) -> np.ndarray:
    """Map [0, 255] uint8 values to [0, bins-1] bin indices."""
    x = x_u8.astype(np.uint16)
    return ((x * bins) // 256).astype(np.uint8)


def quantize_pca_to_bins(x_pca: np.ndarray, bins: int) -> np.ndarray:
    """Map PCA projection values in [-1, 1] to [0, bins-1] bin indices."""
    x = np.clip(x_pca, -1.0, 1.0)
    x_shifted = (x + 1.0) / 2.0
    return np.clip((x_shifted * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)


def joint_nd_pmf(
    rgb_q: np.ndarray,
    pca_q: np.ndarray,
    bins_rgb: int,
    bins_pca: int,
    smooth_sigma: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build factorized PMF: P(r,g,b) * P(p1,...,pk).

    Returns (pmf_rgb, pmf_pca) separately to avoid memory explosion.
    """
    # RGB joint PMF (bins_rgb^3)
    r = rgb_q[:, 0].astype(np.int64)
    g = rgb_q[:, 1].astype(np.int64)
    b = rgb_q[:, 2].astype(np.int64)
    idx_rgb = np.ravel_multi_index((r, g, b), dims=(bins_rgb,) * 3)
    hist_rgb = np.bincount(idx_rgb, minlength=bins_rgb**3).astype(np.float64)
    hist_rgb = hist_rgb.reshape((bins_rgb,) * 3)
    if smooth_sigma > 0:
        hist_rgb = gaussian_filter(hist_rgb, sigma=smooth_sigma, mode="nearest")
    pmf_rgb = hist_rgb / hist_rgb.sum()
    pmf_rgb = pmf_rgb + eps
    pmf_rgb = pmf_rgb / pmf_rgb.sum()

    # PCA joint PMF (bins_pca^k)
    k = pca_q.shape[1]
    pca_tensors = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
    idx_pca = np.ravel_multi_index(pca_tensors, dims=(bins_pca,) * k)
    hist_pca = np.bincount(idx_pca, minlength=bins_pca**k).astype(np.float64)
    hist_pca = hist_pca.reshape((bins_pca,) * k)
    if smooth_sigma > 0:
        hist_pca = gaussian_filter(hist_pca, sigma=smooth_sigma, mode="nearest")
    pmf_pca = hist_pca / hist_pca.sum()
    pmf_pca = pmf_pca + eps
    pmf_pca = pmf_pca / pmf_pca.sum()

    return pmf_rgb, pmf_pca


def entropy_bits_combined(
    pmf_rgb: np.ndarray, pmf_pca: np.ndarray
) -> float:
    """H(P_rgb) + H(P_pca) in bits (factorized assumption).

    PMFs must already have eps mixed in (no zero entries).
    """
    def _h(pmf: np.ndarray) -> float:
        p = pmf.ravel()
        return float(-np.sum(p * np.log2(p)))
    return _h(pmf_rgb) + _h(pmf_pca)


def avg_neg_logp_bits_combined(
    rgb_q: np.ndarray,
    pca_q: np.ndarray,
    pmf_rgb: np.ndarray,
    pmf_pca: np.ndarray,
) -> float:
    """-(1/n) sum log2 P(x_rgb) - (1/n) sum log2 P(x_pca).

    PMFs must already have eps mixed in (no zero entries).
    """
    r = rgb_q[:, 0].astype(np.int64)
    g = rgb_q[:, 1].astype(np.int64)
    b = rgb_q[:, 2].astype(np.int64)
    p_rgb = pmf_rgb[r, g, b]
    nll_rgb = float(-np.mean(np.log2(p_rgb)))

    k = pca_q.shape[1]
    pca_tensors = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
    p_pca = pmf_pca[pca_tensors]
    nll_pca = float(-np.mean(np.log2(p_pca)))

    return nll_rgb + nll_pca


def compute_delta_map(
    labels_fine: np.ndarray,
    parent_map: Dict[int, List[int]],
    images_recon_all: np.ndarray,
    pca_feats_recon: np.ndarray,
    label_image: np.ndarray,
    label_pca_map: np.ndarray,
    bins_rgb: int = 32,
    bins_pca: int = 8,
    smooth_sigma: float = 0.1,
    min_pixels: int = 2,
    use_label_as_target: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict, List[int]]:
    """Compute per-superpixel delta scores using typical set analysis.

    Delta = |avg_neg_logp - H| where the PMF is built from DPS reconstructions
    and evaluated on the label (test) image.

    Args:
        labels_fine: (H, W) refined superpixel labels.
        parent_map: original_id -> [child_ids] from recursive_subdivide.
        images_recon_all: (B, H, W, 3) uint8 reconstruction images.
        pca_feats_recon: (B, H, W, k) float PCA projections in [-1, 1].
        label_image: (H, W, 3) uint8 label/test image.
        label_pca_map: (H, W, k) float PCA projections of label image.
        bins_rgb: quantization bins for RGB [0, 255].
        bins_pca: quantization bins for PCA [-1, 1].
        smooth_sigma: Gaussian smoothing on PMF histograms.
        min_pixels: minimum superpixel size to score.
        use_label_as_target: if True, evaluate on label image (OOD test).
        eps: numerical stability constant.

    Returns:
        delta_map: (H, W) float array (NaN where not scored).
        info: dict per refined_id with H_bits, avg_neg_logp, delta.
        labels_used: sorted list of scored superpixel IDs.
    """
    H_img, W_img = labels_fine.shape
    delta_map = np.full((H_img, W_img), np.nan, dtype=np.float32)
    info: Dict = {}
    labels_used: List[int] = []

    # Build reverse map: child -> original parent
    child_to_parent: Dict[int, int] = {}
    for orig_id, children in parent_map.items():
        for child_id in children:
            child_to_parent[child_id] = orig_id

    all_refined_ids = sorted(np.unique(labels_fine).astype(int).tolist())

    for refined_id in all_refined_ids:
        sp_mask = labels_fine == refined_id
        n_pix = int(sp_mask.sum())

        if n_pix < min_pixels:
            continue

        # 1) Build PMFs from reconstructions
        rows, cols = np.where(sp_mask)
        recon_rgb = images_recon_all[:, rows, cols, :3].reshape(-1, 3).astype(np.uint8)
        if recon_rgb.shape[0] == 0:
            continue
        recon_rgb_q = quantize_u8_to_bins(recon_rgb, bins=bins_rgb)

        N = images_recon_all.shape[0]
        recon_pca = pca_feats_recon[:N, rows, cols, :]
        recon_pca_flat = recon_pca.reshape(-1, recon_pca.shape[-1])
        recon_pca_q = quantize_pca_to_bins(recon_pca_flat, bins=bins_pca)

        pmf_rgb, pmf_pca = joint_nd_pmf(
            recon_rgb_q, recon_pca_q,
            bins_rgb=bins_rgb, bins_pca=bins_pca,
            smooth_sigma=smooth_sigma, eps=eps,
        )
        H_bits = entropy_bits_combined(pmf_rgb, pmf_pca)

        # 2) Get target sequence
        if use_label_as_target and label_image is not None and label_pca_map is not None:
            target_rgb = label_image[sp_mask, :3].astype(np.uint8)
            target_pca = label_pca_map[sp_mask]
        else:
            continue

        if target_rgb.shape[0] == 0:
            continue

        target_rgb_q = quantize_u8_to_bins(target_rgb, bins=bins_rgb)
        target_pca_q = quantize_pca_to_bins(target_pca, bins=bins_pca)

        # 3) Delta
        avg_nlogp = avg_neg_logp_bits_combined(
            target_rgb_q, target_pca_q, pmf_rgb, pmf_pca
        )
        delta_sp = float(np.abs(avg_nlogp - H_bits))
        delta_map[sp_mask] = delta_sp

        info[int(refined_id)] = {
            "orig_parent": child_to_parent.get(refined_id),
            "num_pixels": n_pix,
            "H_bits": float(H_bits),
            "avg_neg_logp_bits": float(avg_nlogp),
            "delta_sp": float(delta_sp),
        }
        labels_used.append(int(refined_id))

    return delta_map, info, sorted(labels_used)
