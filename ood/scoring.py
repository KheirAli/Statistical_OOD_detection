# """Delta map scoring via typical set analysis with RGB + PCA features.

# This is the algorithmic core — the file agents will edit most when testing
# new scoring functions. Extracted from New_dataset_clean.ipynb Cells 13-14.

# The key idea: for each refined superpixel, build a factorized PMF from
# DPS reconstructions (P_rgb * P_pca), compute its entropy H, then measure
# how far the label image's pixels deviate: delta = |avg_nlogp - H|.
# """

# from typing import Dict, List, Optional, Tuple

# import numpy as np
# from scipy.ndimage import gaussian_filter

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# # def quantize_u8_to_bins(x_u8: np.ndarray, bins: int) -> np.ndarray:
# #     """Map [0, 255] uint8 values to [0, bins-1] bin indices."""
# #     x = x_u8.astype(np.uint16)
# #     return ((x * bins) // 256).astype(np.uint8)


# # def quantize_pca_to_bins(x_pca: np.ndarray, bins: int) -> np.ndarray:
# #     """Map PCA projection values in [-1, 1] to [0, bins-1] bin indices."""
# #     x = np.clip(x_pca, -1.0, 1.0)
# #     x_shifted = (x + 1.0) / 2.0
# #     return np.clip((x_shifted * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)

# def quantize_u8_to_bins(x_u8: np.ndarray, bins: int) -> np.ndarray:
#     return ((x_u8.astype(np.uint16) * bins) // 256).astype(np.uint8)


# def quantize_pca_to_bins(x_pca: np.ndarray, bins: int) -> np.ndarray:
#     x = np.clip(x_pca, -1.0, 1.0)
#     return np.clip(((x + 1.0) / 2.0 * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)


# def _hist_dims(bins_rgb, bins_pca, n_pca, gray_scale):
#     if gray_scale:
#         return (bins_rgb,) + (bins_pca,) * n_pca
#     return (bins_rgb, bins_rgb, bins_rgb) + (bins_pca,) * n_pca


# def _flat_bin_index(rgb_q, pca_q, dims, gray_scale):
#     """Flatten multi-dimensional quantised indices to a single integer per observation."""
#     k = pca_q.shape[1]
#     pca_idx = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
#     if gray_scale:
#         return np.ravel_multi_index((rgb_q[:, 0].astype(np.int64),) + pca_idx, dims=dims)
#     return np.ravel_multi_index(
#         (rgb_q[:, 0].astype(np.int64), rgb_q[:, 1].astype(np.int64),
#          rgb_q[:, 2].astype(np.int64)) + pca_idx, dims=dims)

# def joint_nd_pmf(
#     rgb_q: np.ndarray,
#     pca_q: np.ndarray,
#     bins_rgb: int,
#     bins_pca: int,
#     smooth_sigma: float = 1.0,
#     eps: float = 1e-12,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Build factorized PMF: P(r,g,b) * P(p1,...,pk).

#     Returns (pmf_rgb, pmf_pca) separately to avoid memory explosion.
#     """
#     # RGB joint PMF (bins_rgb^3)
#     r = rgb_q[:, 0].astype(np.int64)
#     g = rgb_q[:, 1].astype(np.int64)
#     b = rgb_q[:, 2].astype(np.int64)
#     idx_rgb = np.ravel_multi_index((r, g, b), dims=(bins_rgb,) * 3)
#     hist_rgb = np.bincount(idx_rgb, minlength=bins_rgb**3).astype(np.float64)
#     hist_rgb = hist_rgb.reshape((bins_rgb,) * 3)
#     if smooth_sigma > 0:
#         hist_rgb = gaussian_filter(hist_rgb, sigma=smooth_sigma, mode="nearest")
#     pmf_rgb = hist_rgb / hist_rgb.sum()
#     pmf_rgb = pmf_rgb + eps
#     pmf_rgb = pmf_rgb / pmf_rgb.sum()

#     # PCA joint PMF (bins_pca^k)
#     k = pca_q.shape[1]
#     pca_tensors = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
#     idx_pca = np.ravel_multi_index(pca_tensors, dims=(bins_pca,) * k)
#     hist_pca = np.bincount(idx_pca, minlength=bins_pca**k).astype(np.float64)
#     hist_pca = hist_pca.reshape((bins_pca,) * k)
#     if smooth_sigma > 0:
#         hist_pca = gaussian_filter(hist_pca, sigma=smooth_sigma, mode="nearest")
#     pmf_pca = hist_pca / hist_pca.sum()
#     pmf_pca = pmf_pca + eps
#     pmf_pca = pmf_pca / pmf_pca.sum()

#     return pmf_rgb, pmf_pca


# def entropy_bits_combined(
#     pmf_rgb: np.ndarray, pmf_pca: np.ndarray
# ) -> float:
#     """H(P_rgb) + H(P_pca) in bits (factorized assumption).

#     PMFs must already have eps mixed in (no zero entries).
#     """
#     def _h(pmf: np.ndarray) -> float:
#         p = pmf.ravel()
#         return float(-np.sum(p * np.log2(p)))
#     return _h(pmf_rgb) + _h(pmf_pca)


# def avg_neg_logp_bits_combined(
#     rgb_q: np.ndarray,
#     pca_q: np.ndarray,
#     pmf_rgb: np.ndarray,
#     pmf_pca: np.ndarray,
# ) -> float:
#     """-(1/n) sum log2 P(x_rgb) - (1/n) sum log2 P(x_pca).

#     PMFs must already have eps mixed in (no zero entries).
#     """
#     r = rgb_q[:, 0].astype(np.int64)
#     g = rgb_q[:, 1].astype(np.int64)
#     b = rgb_q[:, 2].astype(np.int64)
#     p_rgb = pmf_rgb[r, g, b]
#     nll_rgb = float(-np.mean(np.log2(p_rgb)))

#     k = pca_q.shape[1]
#     pca_tensors = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
#     p_pca = pmf_pca[pca_tensors]
#     nll_pca = float(-np.mean(np.log2(p_pca)))

#     return nll_rgb + nll_pca


# # def compute_delta_map(
# #     labels_fine: np.ndarray,
# #     parent_map: Dict[int, List[int]],
# #     images_recon_all: np.ndarray,
# #     pca_feats_recon: np.ndarray,
# #     label_image: np.ndarray,
# #     label_pca_map: np.ndarray,
# #     bins_rgb: int = 32,
# #     bins_pca: int = 8,
# #     smooth_sigma: float = 0.1,
# #     min_pixels: int = 2,
# #     use_label_as_target: bool = True,
# #     eps: float = 1e-12,
# # ) -> Tuple[np.ndarray, Dict, List[int]]:
# #     """Compute per-superpixel delta scores using typical set analysis.

# #     Delta = |avg_neg_logp - H| where the PMF is built from DPS reconstructions
# #     and evaluated on the label (test) image.

# #     Args:
# #         labels_fine: (H, W) refined superpixel labels.
# #         parent_map: original_id -> [child_ids] from recursive_subdivide.
# #         images_recon_all: (B, H, W, 3) uint8 reconstruction images.
# #         pca_feats_recon: (B, H, W, k) float PCA projections in [-1, 1].
# #         label_image: (H, W, 3) uint8 label/test image.
# #         label_pca_map: (H, W, k) float PCA projections of label image.
# #         bins_rgb: quantization bins for RGB [0, 255].
# #         bins_pca: quantization bins for PCA [-1, 1].
# #         smooth_sigma: Gaussian smoothing on PMF histograms.
# #         min_pixels: minimum superpixel size to score.
# #         use_label_as_target: if True, evaluate on label image (OOD test).
# #         eps: numerical stability constant.

# #     Returns:
# #         delta_map: (H, W) float array (NaN where not scored).
# #         info: dict per refined_id with H_bits, avg_neg_logp, delta.
# #         labels_used: sorted list of scored superpixel IDs.
# #     """
# #     H_img, W_img = labels_fine.shape
# #     delta_map = np.full((H_img, W_img), np.nan, dtype=np.float32)
# #     info: Dict = {}
# #     labels_used: List[int] = []

# #     # Build reverse map: child -> original parent
# #     child_to_parent: Dict[int, int] = {}
# #     for orig_id, children in parent_map.items():
# #         for child_id in children:
# #             child_to_parent[child_id] = orig_id

# #     all_refined_ids = sorted(np.unique(labels_fine).astype(int).tolist())

# #     for refined_id in all_refined_ids:
# #         sp_mask = labels_fine == refined_id
# #         n_pix = int(sp_mask.sum())

# #         if n_pix < min_pixels:
# #             continue

# #         # 1) Build PMFs from reconstructions
# #         rows, cols = np.where(sp_mask)
# #         recon_rgb = images_recon_all[:, rows, cols, :3].reshape(-1, 3).astype(np.uint8)
# #         if recon_rgb.shape[0] == 0:
# #             continue
# #         recon_rgb_q = quantize_u8_to_bins(recon_rgb, bins=bins_rgb)

# #         N = images_recon_all.shape[0]
# #         recon_pca = pca_feats_recon[:N, rows, cols, :]
# #         recon_pca_flat = recon_pca.reshape(-1, recon_pca.shape[-1])
# #         recon_pca_q = quantize_pca_to_bins(recon_pca_flat, bins=bins_pca)

# #         pmf_rgb, pmf_pca = joint_nd_pmf(
# #             recon_rgb_q, recon_pca_q,
# #             bins_rgb=bins_rgb, bins_pca=bins_pca,
# #             smooth_sigma=smooth_sigma, eps=eps,
# #         )
# #         H_bits = entropy_bits_combined(pmf_rgb, pmf_pca)

# #         # 2) Get target sequence
# #         if use_label_as_target and label_image is not None and label_pca_map is not None:
# #             target_rgb = label_image[sp_mask, :3].astype(np.uint8)
# #             target_pca = label_pca_map[sp_mask]
# #         else:
# #             continue

# #         if target_rgb.shape[0] == 0:
# #             continue

# #         target_rgb_q = quantize_u8_to_bins(target_rgb, bins=bins_rgb)
# #         target_pca_q = quantize_pca_to_bins(target_pca, bins=bins_pca)

# #         # 3) Delta
# #         avg_nlogp = avg_neg_logp_bits_combined(
# #             target_rgb_q, target_pca_q, pmf_rgb, pmf_pca
# #         )
# #         delta_sp = float(np.abs(avg_nlogp - H_bits))
# #         delta_map[sp_mask] = delta_sp

# #         info[int(refined_id)] = {
# #             "orig_parent": child_to_parent.get(refined_id),
# #             "num_pixels": n_pix,
# #             "H_bits": float(H_bits),
# #             "avg_neg_logp_bits": float(avg_nlogp),
# #             "delta_sp": float(delta_sp),
# #         }
# #         labels_used.append(int(refined_id))

# #     return delta_map, info, sorted(labels_used)
# def compute_delta_map(
#     labels_fine:     np.ndarray,    # (H, W)           int32
#     recon_images:    np.ndarray,    # (N_recon, H, W, 3) uint8
#     pca_feats_recon: np.ndarray,    # (N_recon, H, W, k) float32
#     label_image:     np.ndarray,    # (H, W, 3)         uint8
#     label_pca_map:   np.ndarray,    # (H, W, k)         float32
#     bins_rgb: int, bins_pca: int,
#     smooth_sigma: float,            # kept for interface compat; negligible at 0.1
#     min_pixels: int,
#     device: torch.device,
#     eps: float = 1e-12,
#     gray_scale: bool = False,
# ) -> Tuple[np.ndarray, Dict, List]:
#     """
#     Fully vectorised GPU implementation using SPARSE histogram counting.

#     Why sparse?
#     -----------
#     The joint histogram has n_bins = bins_rgb^3 * bins_pca^n_pca bins.
#     For cable (colour, bins_rgb=32, bins_pca=8, n_pca=3): n_bins = 16.7M.
#     A dense (n_sp × n_bins) matrix would require gigabytes.

#     But each superpixel only has N_recon × |pixels_sp| observations (typically
#     1k–5k), so >99.9% of histogram bins are empty.  We only ever touch bins
#     that were actually observed.

#     Memory cost: O(N_recon × H × W) — about 12–50 MB for typical inputs,
#     completely independent of n_bins.

#     Algorithm
#     ---------
#     1. Compute flat bin index b_i for every recon observation i.
#     2. Encode each observation as a single int64:  key_i = sp_id_i * n_bins + b_i
#     3. Sort keys → unique_consecutive → (unique_key, count) pairs.
#     4. Entropy per SP: H_s = -Σ (c/n_s) log2(c/n_s)  over unique keys in s.
#     5. Target NLL: for each target pixel, searchsorted into sorted unique keys
#        to find its recon count, then NLL = -log2(count / n_s).
#     6. delta_s = avg_NLL_s - H_s  (> 0 → OOD)
#     """
#     H, W    = labels_fine.shape
#     N_recon = recon_images.shape[0]
#     n_pca   = pca_feats_recon.shape[-1]

#     # ── 1. Remap SP labels to contiguous 0..n_sp-1 ───────────────────────
#     sp_ids_orig = np.unique(labels_fine)
#     n_sp        = len(sp_ids_orig)
#     remap       = np.zeros(int(sp_ids_orig.max()) + 1, dtype=np.int64)
#     for new_id, old_id in enumerate(sp_ids_orig):
#         remap[old_id] = new_id
#     labels_c  = remap[labels_fine]                       # (H, W)

#     sp_counts = np.bincount(labels_c.ravel(), minlength=n_sp)
#     valid_sp  = sp_counts >= max(min_pixels, 1)

#     # ── 2. Quantise (vectorised CPU, very cheap) ──────────────────────────
#     if gray_scale:
#         recon_rgb_flat = recon_images[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
#         label_rgb_flat = label_image[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
#     else:
#         recon_rgb_flat = recon_images[..., :3].reshape(-1, 3).astype(np.uint8)
#         label_rgb_flat = label_image[..., :3].reshape(-1, 3).astype(np.uint8)

#     recon_pca_flat = pca_feats_recon.reshape(-1, n_pca)   # (N*H*W, k)
#     label_pca_flat = label_pca_map.reshape(-1, n_pca)     # (H*W, k)

#     recon_rgb_q = quantize_u8_to_bins(recon_rgb_flat, bins_rgb)
#     recon_pca_q = quantize_pca_to_bins(recon_pca_flat, bins_pca)
#     label_rgb_q = quantize_u8_to_bins(label_rgb_flat, bins_rgb)
#     label_pca_q = quantize_pca_to_bins(label_pca_flat, bins_pca)

#     dims   = _hist_dims(bins_rgb, bins_pca, n_pca, gray_scale)
#     n_bins = int(np.prod(dims))

#     # Flat bin index per observation  (CPU, vectorised numpy)
#     recon_flat_bins = _flat_bin_index(recon_rgb_q, recon_pca_q, dims, gray_scale)  # (N*H*W,)
#     label_flat_bins = _flat_bin_index(label_rgb_q, label_pca_q, dims, gray_scale)  # (H*W,)

#     # ── 3. Send compact data to GPU ───────────────────────────────────────
#     # SP labels for recon: tile labels N_recon times  →  (N*H*W,)
#     sp_r = torch.from_numpy(labels_c.ravel()).long().to(device).repeat(N_recon)
#     sp_l = torch.from_numpy(labels_c.ravel()).long().to(device)                  # (H*W,)

#     bins_r = torch.from_numpy(recon_flat_bins).long().to(device)  # (N*H*W,)
#     bins_l = torch.from_numpy(label_flat_bins).long().to(device)  # (H*W,)

#     # ── 4. Encode (sp_id, bin_id) as a single int64 key ──────────────────
#     #   key = sp_id * n_bins + bin_id   (unique per (sp, bin) pair)
#     #   int64 max ≈ 9.2e18; n_sp * n_bins ≤ ~1e10 for all practical settings
#     keys_recon = sp_r * n_bins + bins_r    # (N*H*W,)
#     keys_label = sp_l * n_bins + bins_l    # (H*W,)

#     # ── 5. Sort recon keys → unique pairs + counts ────────────────────────
#     keys_sorted = keys_recon.sort().values                                  # ascending sort
#     unique_keys, counts = torch.unique_consecutive(keys_sorted,             # sorted unique keys
#                                                    return_counts=True)      # + how many times seen
#     # unique_keys: (U,)  counts: (U,)  where U << N*H*W

#     # Recover SP id from each unique key
#     # unique_sp = (unique_keys // n_bins).long()   # (U,)
#     unique_sp = torch.div(unique_keys, n_bins, rounding_mode='trunc').long()   # (U,)

#     # ── 6. Total recon observations per SP ───────────────────────────────
#     sp_totals = torch.zeros(n_sp, dtype=torch.float32, device=device)
#     sp_totals.scatter_add_(0, unique_sp, counts.float())       # (n_sp,)

#     # ── 7. Entropy H_s = -Σ (c/n_s) log2(c/n_s) ─────────────────────────
#     probs   = counts.float() / sp_totals[unique_sp]            # p(k | s) for each unique key
#     entropy_contributions = -probs * torch.log2(probs + eps)   # (U,)
#     entropy_sp = torch.zeros(n_sp, dtype=torch.float32, device=device)
#     entropy_sp.scatter_add_(0, unique_sp, entropy_contributions)  # (n_sp,)

#     # ── 8. Target NLL via binary search into sorted unique_keys ──────────
#     #   For each target pixel, find whether its (sp, bin) key appears in the
#     #   recon's sparse set of observed keys.
#     pos         = torch.searchsorted(unique_keys.contiguous(),
#                                      keys_label.contiguous())          # (H*W,)
#     pos_clamped = pos.clamp(0, len(unique_keys) - 1)

#     # Did the target key actually appear in recon?
#     found = (unique_keys[pos_clamped] == keys_label)                   # (H*W,) bool

#     # Recon count for this (sp, bin) — valid only where found=True
#     recon_count = torch.where(found,
#                               counts[pos_clamped].float(),
#                               torch.zeros(H * W, dtype=torch.float32, device=device))

#     # Denominator: total recon obs for each target pixel's SP
#     sp_totals_for_label = sp_totals[sp_l]    # (H*W,)

#     # Probability: count/total if observed, eps/total if not
#     eps_t = torch.tensor(eps, dtype=torch.float32, device=device)
#     prob_target = torch.where(
#         found,
#         recon_count / (sp_totals_for_label + eps_t),
#         eps_t / (sp_totals_for_label + eps_t),
#     )                                                                    # (H*W,)

#     nll_per_pixel = -torch.log2(prob_target + eps_t)                    # (H*W,)

#     # Average NLL per SP
#     nll_sum = torch.zeros(n_sp, dtype=torch.float32, device=device)
#     nll_sum.scatter_add_(0, sp_l, nll_per_pixel)
#     sp_counts_t = torch.from_numpy(sp_counts).float().to(device).clamp(min=1.0)
#     avg_nll_sp  = nll_sum / sp_counts_t                                  # (n_sp,)

#     # ── 9. OOD score Δ = avg_NLL − H  (> 0 → anomalous) ─────────────────
#     delta_sp = avg_nll_sp - entropy_sp                                   # (n_sp,)

#     # Zero-out invalid (below min_pixels) SPs
#     valid_t  = torch.from_numpy(valid_sp).to(device)
#     delta_sp = torch.where(valid_t, delta_sp,
#                            torch.full_like(delta_sp, float("nan")))

#     delta_sp_np  = delta_sp.cpu().numpy()
#     entropy_np   = entropy_sp.cpu().numpy()
#     avg_nll_np   = avg_nll_sp.cpu().numpy()

#     # ── 10. Build (H, W) output map ───────────────────────────────────────
#     delta_map = delta_sp_np[labels_c].astype(np.float32)
#     delta_map[~valid_sp[labels_c]] = np.nan

#     # ── 11. Info dict ─────────────────────────────────────────────────────
#     info:        Dict[int, Dict[str, float]] = {}
#     labels_used: List[int] = []
#     for new_id, old_id in enumerate(sp_ids_orig):
#         if valid_sp[new_id]:
#             info[int(old_id)] = {
#                 "num_pixels":        float(sp_counts[new_id]),
#                 "H_bits":            float(entropy_np[new_id]),
#                 "avg_neg_logp_bits": float(avg_nll_np[new_id]),
#                 "delta_sp":          float(delta_sp_np[new_id]),
#             }
#             labels_used.append(int(old_id))

#     return delta_map, info, labels_used



# """Delta map scoring via typical set analysis with RGB + PCA features.

# This is the algorithmic core. Extracted from New_dataset_clean.ipynb Cells 13-14.

# The key idea: for each refined superpixel, build a factorized PMF from
# DPS reconstructions (P_rgb * P_pca), compute its entropy H, then measure
# how far the label image's pixels deviate: delta = |avg_nlogp - H|.

# Changes vs. original (aligned with full_run_hist.py):
#   - Added GPU-accelerated path: compute_delta_map_gpu.
#     Uses two small DENSE histograms (RGB + PCA separately) so Gaussian
#     smoothing can be applied on the GPU without any memory explosion.
#     Memory: (n_sp × bins_rgb³) + (n_sp × bins_pca^k) — typically < 15 MB.
#   - _gaussian_kernel_1d, _smooth_hist_nd: separable GPU bin-dimension smoothing.
#   - Original compute_delta_map (CPU per-superpixel loop) is unchanged and
#     kept as the reference / fallback implementation.
# """

# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn.functional as F
# from scipy.ndimage import gaussian_filter


# # ─────────────────────────────────────────────────────────────────────────────
# # Quantisation helpers (shared by CPU and GPU paths)
# # ─────────────────────────────────────────────────────────────────────────────

# def quantize_u8_to_bins(x_u8: np.ndarray, bins: int) -> np.ndarray:
#     """Map [0, 255] uint8 values to [0, bins-1] bin indices."""
#     x = x_u8.astype(np.uint16)
#     return ((x * bins) // 256).astype(np.uint8)


# # def quantize_pca_to_bins(x_pca: np.ndarray, bins: int) -> np.ndarray:
# #     """Map PCA projection values in [-1, 1] to [0, bins-1] bin indices."""
# #     x = np.clip(x_pca, -1.0, 1.0)
# #     x_shifted = (x + 1.0) / 2.0
# #     return np.clip((x_shifted * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)


# def quantize_pca_to_bins(x_pca: np.ndarray, bins: int) -> np.ndarray:
#     """Legacy: clips to [-1, 1]. Only kept for the CPU reference path."""
#     x = np.clip(x_pca, -1.0, 1.0)
#     x_shifted = (x + 1.0) / 2.0
#     return np.clip((x_shifted * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)


# def quantize_pca_adaptive(
#     x_pca: np.ndarray,
#     bins: int,
#     pca_min: np.ndarray,   # (k,) per-component minimum  — computed from recon
#     pca_max: np.ndarray,   # (k,) per-component maximum  — computed from recon
# ) -> np.ndarray:
#     """
#     Data-driven quantization: maps the observed recon range → [0, bins-1].
#     No clipping, no information loss.

#     pca_min / pca_max must be computed from the RECON features so the
#     histogram bins cover the in-distribution range.  Label features that
#     fall outside this range (OOD signal) are clamped to the boundary bins
#     rather than silently lost.

#     Args:
#         x_pca  : (N, k) float array of PCA projections.
#         bins   : number of histogram bins per component.
#         pca_min: (k,) minimum per component (from recon data).
#         pca_max: (k,) maximum per component (from recon data).

#     Returns:
#         (N, k) uint8 bin indices in [0, bins-1].
#     """
#     rng = pca_max - pca_min                     # (k,) — width of each component
#     rng = np.where(rng < 1e-8, 1e-8, rng)       # avoid division by zero

#     # Map to [0, 1] using recon range, then scale to bins
#     x_norm = (x_pca - pca_min) / rng            # outside recon range → <0 or >1
#     return np.clip((x_norm * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)
# # ─────────────────────────────────────────────────────────────────────────────
# # CPU reference implementation (original, unchanged)
# # ─────────────────────────────────────────────────────────────────────────────

# def joint_nd_pmf(
#     rgb_q: np.ndarray,
#     pca_q: np.ndarray,
#     bins_rgb: int,
#     bins_pca: int,
#     smooth_sigma: float = 1.0,
#     eps: float = 1e-12,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Build factorized PMF: P(r,g,b) * P(p1,...,pk).

#     Returns (pmf_rgb, pmf_pca) separately to avoid memory explosion.
#     """
#     # RGB joint PMF (bins_rgb^3)
#     r = rgb_q[:, 0].astype(np.int64)
#     g = rgb_q[:, 1].astype(np.int64)
#     b = rgb_q[:, 2].astype(np.int64)
#     idx_rgb = np.ravel_multi_index((r, g, b), dims=(bins_rgb,) * 3)
#     hist_rgb = np.bincount(idx_rgb, minlength=bins_rgb**3).astype(np.float64)
#     hist_rgb = hist_rgb.reshape((bins_rgb,) * 3)
#     if smooth_sigma > 0:
#         hist_rgb = gaussian_filter(hist_rgb, sigma=smooth_sigma, mode="nearest")
#     pmf_rgb = hist_rgb / hist_rgb.sum()
#     pmf_rgb = pmf_rgb + eps
#     pmf_rgb = pmf_rgb / pmf_rgb.sum()

#     # PCA joint PMF (bins_pca^k)
#     k = pca_q.shape[1]
#     pca_tensors = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
#     idx_pca = np.ravel_multi_index(pca_tensors, dims=(bins_pca,) * k)
#     hist_pca = np.bincount(idx_pca, minlength=bins_pca**k).astype(np.float64)
#     hist_pca = hist_pca.reshape((bins_pca,) * k)
#     if smooth_sigma > 0:
#         hist_pca = gaussian_filter(hist_pca, sigma=smooth_sigma, mode="nearest")
#     pmf_pca = hist_pca / hist_pca.sum()
#     pmf_pca = pmf_pca + eps
#     pmf_pca = pmf_pca / pmf_pca.sum()

#     return pmf_rgb, pmf_pca


# def entropy_bits_combined(
#     pmf_rgb: np.ndarray, pmf_pca: np.ndarray
# ) -> float:
#     """H(P_rgb) + H(P_pca) in bits (factorized assumption)."""
#     def _h(pmf: np.ndarray) -> float:
#         p = pmf.ravel()
#         return float(-np.sum(p * np.log2(p)))
#     return _h(pmf_rgb) + _h(pmf_pca)


# def avg_neg_logp_bits_combined(
#     rgb_q: np.ndarray,
#     pca_q: np.ndarray,
#     pmf_rgb: np.ndarray,
#     pmf_pca: np.ndarray,
# ) -> float:
#     """-(1/n) sum log2 P(x_rgb) - (1/n) sum log2 P(x_pca)."""
#     r = rgb_q[:, 0].astype(np.int64)
#     g = rgb_q[:, 1].astype(np.int64)
#     b = rgb_q[:, 2].astype(np.int64)
#     p_rgb = pmf_rgb[r, g, b]
#     nll_rgb = float(-np.mean(np.log2(p_rgb)))

#     k = pca_q.shape[1]
#     pca_tensors = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
#     p_pca = pmf_pca[pca_tensors]
#     nll_pca = float(-np.mean(np.log2(p_pca)))

#     return nll_rgb + nll_pca


# def compute_delta_map(
#     labels_fine: np.ndarray,
#     parent_map: Dict[int, List[int]],
#     images_recon_all: np.ndarray,
#     pca_feats_recon: np.ndarray,
#     label_image: np.ndarray,
#     label_pca_map: np.ndarray,
#     bins_rgb: int = 32,
#     bins_pca: int = 8,
#     smooth_sigma: float = 0.1,
#     min_pixels: int = 2,
#     use_label_as_target: bool = True,
#     eps: float = 1e-12,
# ) -> Tuple[np.ndarray, Dict, List[int]]:
#     """Compute per-superpixel delta scores using typical set analysis (CPU).

#     Delta = |avg_neg_logp - H| where the PMF is built from DPS reconstructions
#     and evaluated on the label (test) image.

#     This is the reference CPU implementation. For large datasets prefer
#     compute_delta_map_gpu which vectorises all superpixels on GPU.
#     """
#     H_img, W_img = labels_fine.shape
#     delta_map = np.full((H_img, W_img), np.nan, dtype=np.float32)
#     info: Dict = {}
#     labels_used: List[int] = []

#     child_to_parent: Dict[int, int] = {}
#     for orig_id, children in parent_map.items():
#         for child_id in children:
#             child_to_parent[child_id] = orig_id

#     all_refined_ids = sorted(np.unique(labels_fine).astype(int).tolist())

#     for refined_id in all_refined_ids:
#         sp_mask = labels_fine == refined_id
#         n_pix = int(sp_mask.sum())

#         if n_pix < min_pixels:
#             continue

#         rows, cols = np.where(sp_mask)
#         recon_rgb = images_recon_all[:, rows, cols, :3].reshape(-1, 3).astype(np.uint8)
#         if recon_rgb.shape[0] == 0:
#             continue
#         recon_rgb_q = quantize_u8_to_bins(recon_rgb, bins=bins_rgb)

#         N = images_recon_all.shape[0]
#         recon_pca = pca_feats_recon[:N, rows, cols, :]
#         recon_pca_flat = recon_pca.reshape(-1, recon_pca.shape[-1])
#         recon_pca_q = quantize_pca_to_bins(recon_pca_flat, bins=bins_pca)

#         pmf_rgb, pmf_pca = joint_nd_pmf(
#             recon_rgb_q, recon_pca_q,
#             bins_rgb=bins_rgb, bins_pca=bins_pca,
#             smooth_sigma=smooth_sigma, eps=eps,
#         )
#         H_bits = entropy_bits_combined(pmf_rgb, pmf_pca)

#         if use_label_as_target and label_image is not None and label_pca_map is not None:
#             target_rgb = label_image[sp_mask, :3].astype(np.uint8)
#             target_pca = label_pca_map[sp_mask]
#         else:
#             continue

#         if target_rgb.shape[0] == 0:
#             continue

#         target_rgb_q = quantize_u8_to_bins(target_rgb, bins=bins_rgb)
#         target_pca_q = quantize_pca_to_bins(target_pca, bins=bins_pca)

#         avg_nlogp = avg_neg_logp_bits_combined(
#             target_rgb_q, target_pca_q, pmf_rgb, pmf_pca
#         )
#         delta_sp = float(np.abs(avg_nlogp - H_bits))
#         delta_map[sp_mask] = delta_sp

#         info[int(refined_id)] = {
#             "orig_parent": child_to_parent.get(refined_id),
#             "num_pixels": n_pix,
#             "H_bits": float(H_bits),
#             "avg_neg_logp_bits": float(avg_nlogp),
#             "delta_sp": float(delta_sp),
#         }
#         labels_used.append(int(refined_id))

#     return delta_map, info, sorted(labels_used)


# # ─────────────────────────────────────────────────────────────────────────────
# # GPU-accelerated implementation
# # ─────────────────────────────────────────────────────────────────────────────

# def _gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
#     """1-D Gaussian kernel, radius = ceil(3σ), normalised to sum = 1."""
#     radius = max(1, int(np.ceil(3.0 * sigma)))
#     ax = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
#     g  = torch.exp(-ax ** 2 / (2.0 * sigma ** 2))
#     return g / g.sum()


# def _smooth_hist_nd(
#     hist:   torch.Tensor,
#     sigma:  float,
#     device: torch.device,
# ) -> torch.Tensor:
#     """Separable 1-D Gaussian smoothing along every bin dimension.

#     hist shape: (n_sp, b1, b2, ..., bn) — SP axis is dim 0.
#     Applies one F.conv1d pass per bin dimension.
#     Memory cost = size of the histogram only — no expansion.
#     """
#     if sigma <= 0:
#         return hist
#     k   = _gaussian_kernel_1d(sigma, device)
#     out = hist
#     for dim in range(1, out.ndim):
#         shape    = list(out.shape)
#         n_bins_d = shape[dim]
#         perm               = list(range(len(shape)))
#         perm[dim], perm[-1] = perm[-1], perm[dim]
#         x = out.permute(perm).reshape(-1, n_bins_d)
#         x = F.conv1d(x.unsqueeze(1), k.view(1, 1, -1), padding=len(k) // 2).squeeze(1)
#         new_shape      = [shape[i] for i in perm]
#         new_shape[-1]  = n_bins_d
#         out = x.reshape(new_shape).permute(perm)
#     return out


# def compute_delta_map_gpu(
#     labels_fine:     np.ndarray,
#     images_recon_all: np.ndarray,
#     pca_feats_recon: np.ndarray,
#     label_image:     np.ndarray,
#     label_pca_map:   np.ndarray,
#     parent_map: Dict[int, List[int]] = None,
#     bins_rgb:     int   = 32,
#     bins_pca:     int   = 8,
#     smooth_sigma: float = 0.1,
#     min_pixels:   int   = 2,
#     device: torch.device = None,
#     eps:    float = 1e-12,
#     gray_scale: bool = False,
#     **unused_kwargs,
# ) -> Tuple[np.ndarray, Dict, List[int]]:
#     """GPU-accelerated factorized PMF delta scoring.

#     Drop-in replacement for compute_delta_map with identical outputs.

#     Algorithm
#     ---------
#     PMF model  : P(r,g,b) × P(p1,…,pk)   [factorized — two small dense histograms]
#     Smoothing  : separable Gaussian on bin dimensions (GPU, via _smooth_hist_nd)
#     Delta      : |avg_NLL − H|             [absolute, matching CPU reference]

#     Memory
#     ------
#     RGB histogram : (n_sp, bins_rgb, bins_rgb, bins_rgb) ≈ 85 × 32³ ≈ 11 MB
#     PCA histogram : (n_sp, bins_pca, …, bins_pca)        ≈ 85 × 8³  ≈ 0.2 MB
#     Both small regardless of bin count — smoothing is free.

#     Args:
#         labels_fine: (H, W) refined superpixel labels.
#         parent_map: original_id -> [child_ids] (kept for API parity, not used).
#         images_recon_all: (B, H, W, 3) uint8 reconstruction images.
#         pca_feats_recon: (B, H, W, k) float PCA projections in [-1, 1].
#         label_image: (H, W, 3) uint8 label/test image.
#         label_pca_map: (H, W, k) float PCA projections of label image.
#         bins_rgb: quantization bins for RGB.
#         bins_pca: quantization bins for PCA.
#         smooth_sigma: Gaussian sigma for histogram bin smoothing.
#         min_pixels: skip superpixels smaller than this.
#         device: torch.device (defaults to cuda if available, else cpu).
#         eps: numerical stability constant.
#         gray_scale: if True, use grayscale (1-channel) RGB histogram.

#     Returns:
#         delta_map: (H, W) float array (NaN where not scored).
#         info: dict per refined_id.
#         labels_used: sorted list of scored superpixel IDs.
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     H, W    = labels_fine.shape
#     N_recon = images_recon_all.shape[0]
#     n_pca   = pca_feats_recon.shape[-1]

#     # ── Remap SP labels → contiguous 0..n_sp-1 ───────────────────────────
#     sp_ids_orig = np.unique(labels_fine)
#     n_sp        = len(sp_ids_orig)
#     remap       = np.zeros(int(sp_ids_orig.max()) + 1, dtype=np.int64)
#     for ni, oi in enumerate(sp_ids_orig):
#         remap[oi] = ni
#     labels_c  = remap[labels_fine]
#     sp_counts = np.bincount(labels_c.ravel(), minlength=n_sp)
#     valid_sp  = sp_counts >= max(min_pixels, 1)

#     # ── Quantise ─────────────────────────────────────────────────────────
#     # if gray_scale:
#     #     r_rgb    = images_recon_all[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
#     #     l_rgb    = label_image[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
#     #     rgb_dims = (bins_rgb,)
#     # else:
#     #     r_rgb    = images_recon_all[..., :3].reshape(-1, 3).astype(np.uint8)
#     #     l_rgb    = label_image[..., :3].reshape(-1, 3).astype(np.uint8)
#     #     rgb_dims = (bins_rgb, bins_rgb, bins_rgb)

#     # rq_rgb = quantize_u8_to_bins(r_rgb, bins_rgb)
#     # rq_pca = quantize_pca_to_bins(pca_feats_recon.reshape(-1, n_pca), bins_pca)
#     # lq_rgb = quantize_u8_to_bins(l_rgb, bins_rgb)
#     # lq_pca = quantize_pca_to_bins(label_pca_map.reshape(-1, n_pca), bins_pca)
#     if gray_scale:
#         r_rgb    = images_recon_all[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
#         l_rgb    = label_image[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
#         rgb_dims = (bins_rgb,)
#     else:
#         r_rgb    = images_recon_all[..., :3].reshape(-1, 3).astype(np.uint8)
#         l_rgb    = label_image[..., :3].reshape(-1, 3).astype(np.uint8)
#         rgb_dims = (bins_rgb, bins_rgb, bins_rgb)

#     recon_pca_flat = pca_feats_recon.reshape(-1, n_pca)   # (N*H*W, k)
#     label_pca_flat = label_pca_map.reshape(-1, n_pca)     # (H*W,   k)

#     # DATA-DRIVEN PCA bounds — computed from recon, applied to both recon and label
#     # Use percentiles instead of min/max to be robust to extreme outliers
#     pca_min = np.percentile(recon_pca_flat, 1,  axis=0)   # (k,)
#     pca_max = np.percentile(recon_pca_flat, 99, axis=0)   # (k,)

#     rq_rgb = quantize_u8_to_bins(r_rgb, bins_rgb)
#     rq_pca = quantize_pca_adaptive(recon_pca_flat, bins_pca, pca_min, pca_max)
#     lq_rgb = quantize_u8_to_bins(l_rgb, bins_rgb)
#     lq_pca = quantize_pca_adaptive(label_pca_flat, bins_pca, pca_min, pca_max)
#     pca_dims   = (bins_pca,) * n_pca
#     n_rgb_bins = int(np.prod(rgb_dims))
#     n_pca_bins = int(np.prod(pca_dims))

#     if gray_scale:
#         rb = rq_rgb[:, 0].astype(np.int64)
#         lb = lq_rgb[:, 0].astype(np.int64)
#     else:
#         rb = np.ravel_multi_index((rq_rgb[:,0].astype(np.int64),
#                                    rq_rgb[:,1].astype(np.int64),
#                                    rq_rgb[:,2].astype(np.int64)), dims=rgb_dims)
#         lb = np.ravel_multi_index((lq_rgb[:,0].astype(np.int64),
#                                    lq_rgb[:,1].astype(np.int64),
#                                    lq_rgb[:,2].astype(np.int64)), dims=rgb_dims)

#     rp = np.ravel_multi_index(tuple(rq_pca[:,i].astype(np.int64) for i in range(n_pca)), dims=pca_dims)
#     lp = np.ravel_multi_index(tuple(lq_pca[:,i].astype(np.int64) for i in range(n_pca)), dims=pca_dims)

#     # ── GPU tensors ───────────────────────────────────────────────────────
#     sp_r = torch.from_numpy(labels_c.ravel()).long().to(device).repeat(N_recon)
#     sp_l = torch.from_numpy(labels_c.ravel()).long().to(device)

#     def _build_hist(bins_np, n_bins, bin_shape):
#         b = torch.from_numpy(bins_np).long().to(device)
#         return torch.bincount(sp_r * n_bins + b, minlength=n_sp * n_bins)\
#                     .float().reshape(n_sp, *bin_shape)

#     h_rgb = _build_hist(rb, n_rgb_bins, rgb_dims)
#     h_pca = _build_hist(rp, n_pca_bins, pca_dims)

#     # ── Gaussian smoothing on bin dimensions ──────────────────────────────
#     if smooth_sigma > 0:
#         h_rgb = _smooth_hist_nd(h_rgb, smooth_sigma, device)
#         h_pca = _smooth_hist_nd(h_pca, smooth_sigma, device)

#     h_rgb += eps;  h_pca += eps
#     p_rgb = h_rgb / h_rgb.reshape(n_sp,-1).sum(1).view(n_sp,*([1]*len(rgb_dims)))
#     p_pca = h_pca / h_pca.reshape(n_sp,-1).sum(1).view(n_sp,*([1]*len(pca_dims)))

#     def _entropy(pmf):
#         q = pmf.reshape(n_sp, -1)
#         return -(q * torch.log2(q + eps)).sum(1)

#     entropy_sp = _entropy(p_rgb) + _entropy(p_pca)

#     def _avg_nll(lbins, pmf, n_bins):
#         b   = torch.from_numpy(lbins).long().to(device)
#         nll = -torch.log2(pmf.reshape(n_sp, n_bins)[sp_l, b] + eps)
#         s   = torch.zeros(n_sp, dtype=torch.float32, device=device)
#         s.scatter_add_(0, sp_l, nll)
#         cnt = torch.from_numpy(sp_counts).float().to(device).clamp(min=1.0)
#         return s / cnt

#     avg_nll    = _avg_nll(lb, p_rgb, n_rgb_bins) + _avg_nll(lp, p_pca, n_pca_bins)
#     delta_sp   = torch.abs(avg_nll - entropy_sp)
#     valid_t    = torch.from_numpy(valid_sp).to(device)
#     delta_sp   = torch.where(valid_t, delta_sp, torch.full_like(delta_sp, float("nan")))

#     d_np = delta_sp.cpu().numpy()
#     e_np = entropy_sp.cpu().numpy()
#     n_np = avg_nll.cpu().numpy()

#     delta_map = d_np[labels_c].astype(np.float32)
#     delta_map[~valid_sp[labels_c]] = np.nan

#     # Build parent reverse-map for info dict (mirrors CPU implementation)
#     # child_to_parent: Dict[int, int] = {}
#     # for orig_id, children in parent_map.items():
#     #     for child_id in children:
#     #         child_to_parent[child_id] = orig_id
#     child_to_parent: Dict[int, int] = {}
#     if parent_map is not None:
#         for orig_id, children in parent_map.items():
#             for child_id in children:
#                 child_to_parent[child_id] = orig_id
#     info:        Dict = {}
#     labels_used: List[int] = []
#     for ni, oi in enumerate(sp_ids_orig):
#         if valid_sp[ni]:
#             info[int(oi)] = {
#                 "orig_parent":       child_to_parent.get(int(oi)),
#                 "num_pixels":        float(sp_counts[ni]),
#                 "H_bits":            float(e_np[ni]),
#                 "avg_neg_logp_bits": float(n_np[ni]),
#                 "delta_sp":          float(d_np[ni]),
#             }
#             labels_used.append(int(oi))

#     return delta_map, info, sorted(labels_used)


"""Delta map scoring via typical set analysis with RGB + PCA features.

Changes vs. previous version
─────────────────────────────
CRASH FIX   _try_dense_hist checks n_sp × n_bins before allocating.
            If it would exceed MAX_HIST_ELEMENTS, returns None and the
            caller automatically uses the sparse sort+unique path instead.
            This fixes the CUDA illegal-memory-access crash with
            bins_pca=16, n_pca=5  (5635 SPs × 16^5 = 5.9 billion entries,
            ~23 GB at float32 — physically impossible).

MEMORY OPT  Three dtype savings vs. previous float32-everywhere approach:
            1. Histogram counts: torch.int16  (2 bytes, saves 2×)
            2. PMF:              torch.float16 (2 bytes, saves 2×)
            3. SP label tensors: torch.int32  (4 bytes, saves 2× vs int64)
            Combined: 3–4× lower GPU RAM for the histogram tensors.

DATA RANGE  quantize_pca_adaptive replaces the hard [-1,1] clip.
            Bounds come from the 1st–99th percentile of the recon features,
            so no information is lost for PCA components with a larger
            dynamic range.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


# Maximum elements for a dense (n_sp × n_bins) histogram tensor.
# At int16 (2 bytes): 200 M elements ≈ 400 MB.  Tune to your GPU RAM.
MAX_HIST_ELEMENTS = 200_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Quantisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def quantize_u8_to_bins(x_u8: np.ndarray, bins: int) -> np.ndarray:
    """Map [0,255] uint8 → [0, bins-1].  Returns uint8."""
    return ((x_u8.astype(np.uint16) * bins) // 256).astype(np.uint8)


def quantize_pca_to_bins(x_pca: np.ndarray, bins: int) -> np.ndarray:
    """Legacy hard clip to [-1,1].  Kept for CPU reference path."""
    x = np.clip(x_pca, -1.0, 1.0)
    return np.clip(((x + 1.0) / 2.0 * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)


def quantize_pca_adaptive(
    x_pca:   np.ndarray,    # (N, k)
    bins:    int,
    pca_min: np.ndarray,    # (k,)  per-component min from recon
    pca_max: np.ndarray,    # (k,)  per-component max from recon
) -> np.ndarray:
    """
    Data-driven PCA quantization — no information lost.

    Maps the observed reconstruction range → [0, bins-1].
    Label pixels outside the recon range are clamped to the boundary bins
    (they carry the OOD signal) rather than silently aliased.

    pca_min / pca_max must be computed from the RECON features (e.g. via
    np.percentile at 1 / 99) to cover the in-distribution range.
    """
    rng = np.where((pca_max - pca_min) < 1e-8, 1e-8, pca_max - pca_min)
    x_norm = (x_pca - pca_min) / rng
    return np.clip((x_norm * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# GPU smoothing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g  = torch.exp(-ax ** 2 / (2.0 * sigma ** 2))
    return g / g.sum()


def _smooth_hist_nd(hist: torch.Tensor, sigma: float, device: torch.device) -> torch.Tensor:
    """Separable 1-D Gaussian along every bin dimension (dim ≥ 1)."""
    if sigma <= 0:
        return hist
    k   = _gaussian_kernel_1d(sigma, device)
    out = hist.float()
    for dim in range(1, out.ndim):
        shape = list(out.shape)
        nb    = shape[dim]
        perm  = list(range(len(shape))); perm[dim], perm[-1] = perm[-1], perm[dim]
        x     = out.permute(perm).reshape(-1, nb)
        x     = F.conv1d(x.unsqueeze(1), k.view(1,1,-1), padding=len(k)//2).squeeze(1)
        ns    = [shape[i] for i in perm]; ns[-1] = nb
        out   = x.reshape(ns).permute(perm)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dense histogram builder  (int16 counts → float16 PMF)
# ─────────────────────────────────────────────────────────────────────────────

def _try_dense_hist(
    sp_r:         torch.Tensor,   # (N*H*W,)  int32
    bins_np:      np.ndarray,     # (N*H*W,)  flat bin index
    n_sp:         int,
    n_bins:       int,
    bin_shape:    tuple,
    device:       torch.device,
    smooth_sigma: float,
    eps:          float,
) -> Optional[torch.Tensor]:
    """
    Build a dense (n_sp, *bin_shape) PMF on GPU using int16 counts.

    Memory per histogram: n_sp × n_bins × 2 bytes (int16 counts)
                        + n_sp × n_bins × 2 bytes (float16 PMF)
    vs. previous float32: n_sp × n_bins × 4 bytes → 2× saving.

    Returns float16 PMF tensor, or None if n_sp × n_bins > MAX_HIST_ELEMENTS.
    int16 max = 32 767.  With N_recon=24 and avg SP ≈ 11px, max count ≈ 264 ✓
    """
    if n_sp * n_bins > MAX_HIST_ELEMENTS:
        return None

    # Use int64 for the combined key to avoid overflow
    b        = torch.from_numpy(bins_np).to(device, dtype=torch.int64)
    combined = sp_r.long() * n_bins + b

    counts   = torch.bincount(combined, minlength=n_sp * n_bins)\
                    .to(torch.int16)\
                    .reshape(n_sp, *bin_shape)

    hist = counts.float()
    if smooth_sigma > 0:
        hist = _smooth_hist_nd(hist, smooth_sigma, device)

    hist += eps
    norm  = hist.reshape(n_sp, -1).sum(1).view(n_sp, *([1]*len(bin_shape)))
    return (hist / norm).to(torch.float16)   # (n_sp, *bin_shape)  fp16


def _dense_entropy(pmf: torch.Tensor, n_sp: int, eps: float) -> torch.Tensor:
    q = pmf.float().reshape(n_sp, -1)
    return -(q * torch.log2(q + eps)).sum(1)          # (n_sp,)  fp32


def _dense_avg_nll(
    lbins_np: np.ndarray,
    pmf:      torch.Tensor,
    n_bins:   int,
    n_sp:     int,
    sp_l:     torch.Tensor,     # (H*W,)  int32
    sp_counts: np.ndarray,
    device:   torch.device,
    eps:      float,
) -> torch.Tensor:
    b    = torch.from_numpy(lbins_np).to(device, dtype=torch.int64)
    p    = pmf.float().reshape(n_sp, n_bins)[sp_l.long(), b]
    nll  = -torch.log2(p + eps)
    s    = torch.zeros(n_sp, dtype=torch.float32, device=device)
    s.scatter_add_(0, sp_l.long(), nll)
    return s / torch.from_numpy(sp_counts).float().to(device).clamp(min=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sparse histogram path  (sort + unique_consecutive)
#   Memory: O(N_recon × H × W) — independent of n_bins.
#   Activated automatically when the dense path would exceed MAX_HIST_ELEMENTS.
#   Note: histogram smoothing is not applied (eps-smoothing only).
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_entropy_and_nll(
    sp_r:      torch.Tensor,   # (N*H*W,)  int32
    sp_l:      torch.Tensor,   # (H*W,)    int32
    bins_r_np: np.ndarray,     # (N*H*W,)  flat recon bin indices  int64
    bins_l_np: np.ndarray,     # (H*W,)    flat label bin indices  int64
    n_sp:      int,
    n_bins:    int,
    sp_counts: np.ndarray,
    device:    torch.device,
    eps:       float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (entropy_sp, avg_nll_sp), both (n_sp,) float32."""
    bins_r = torch.from_numpy(bins_r_np).to(device, dtype=torch.int64)
    bins_l = torch.from_numpy(bins_l_np).to(device, dtype=torch.int64)

    keys_r              = sp_r.long() * n_bins + bins_r
    keys_l              = sp_l.long() * n_bins + bins_l
    keys_sorted         = keys_r.sort().values
    unique_keys, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    unique_sp           = torch.div(unique_keys, n_bins, rounding_mode='trunc').long()

    sp_totals = torch.zeros(n_sp, dtype=torch.float32, device=device)
    sp_totals.scatter_add_(0, unique_sp, counts.float())

    probs       = counts.float() / sp_totals[unique_sp]
    ent_contrib = -probs * torch.log2(probs + eps)
    entropy_sp  = torch.zeros(n_sp, dtype=torch.float32, device=device)
    entropy_sp.scatter_add_(0, unique_sp, ent_contrib)

    pos         = torch.searchsorted(unique_keys.contiguous(), keys_l.contiguous())
    pos_c       = pos.clamp(0, len(unique_keys) - 1)
    found       = unique_keys[pos_c] == keys_l
    recon_cnt   = torch.where(found, counts[pos_c].float(),
                              torch.zeros(len(sp_l), dtype=torch.float32, device=device))
    sp_tot_l    = sp_totals[sp_l.long()]
    eps_t       = torch.tensor(eps, dtype=torch.float32, device=device)
    prob_tgt    = torch.where(found,
                              recon_cnt / (sp_tot_l + eps_t),
                              eps_t     / (sp_tot_l + eps_t))
    nll_px      = -torch.log2(prob_tgt + eps_t)
    nll_sum     = torch.zeros(n_sp, dtype=torch.float32, device=device)
    nll_sum.scatter_add_(0, sp_l.long(), nll_px)
    avg_nll_sp  = nll_sum / torch.from_numpy(sp_counts).float().to(device).clamp(min=1.0)
    return entropy_sp, avg_nll_sp


# ─────────────────────────────────────────────────────────────────────────────
# CPU reference (unchanged algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def joint_nd_pmf(rgb_q, pca_q, bins_rgb, bins_pca, smooth_sigma=1.0, eps=1e-12):
    r, g, b  = rgb_q[:,0].astype(np.int64), rgb_q[:,1].astype(np.int64), rgb_q[:,2].astype(np.int64)
    hist_rgb = np.bincount(np.ravel_multi_index((r,g,b), dims=(bins_rgb,)*3),
                           minlength=bins_rgb**3).astype(np.float64).reshape((bins_rgb,)*3)
    if smooth_sigma > 0: hist_rgb = gaussian_filter(hist_rgb, sigma=smooth_sigma, mode="nearest")
    pmf_rgb  = (hist_rgb/hist_rgb.sum()) + eps; pmf_rgb /= pmf_rgb.sum()

    k        = pca_q.shape[1]
    hist_pca = np.bincount(np.ravel_multi_index(tuple(pca_q[:,i].astype(np.int64) for i in range(k)),
                                                dims=(bins_pca,)*k),
                           minlength=bins_pca**k).astype(np.float64).reshape((bins_pca,)*k)
    if smooth_sigma > 0: hist_pca = gaussian_filter(hist_pca, sigma=smooth_sigma, mode="nearest")
    pmf_pca  = (hist_pca/hist_pca.sum()) + eps; pmf_pca /= pmf_pca.sum()
    return pmf_rgb, pmf_pca


def entropy_bits_combined(pmf_rgb, pmf_pca):
    def _h(p): return float(-np.sum(p.ravel() * np.log2(p.ravel())))
    return _h(pmf_rgb) + _h(pmf_pca)


def avg_neg_logp_bits_combined(rgb_q, pca_q, pmf_rgb, pmf_pca):
    r, g, b = rgb_q[:,0].astype(np.int64), rgb_q[:,1].astype(np.int64), rgb_q[:,2].astype(np.int64)
    k       = pca_q.shape[1]
    return (float(-np.mean(np.log2(pmf_rgb[r,g,b]))) +
            float(-np.mean(np.log2(pmf_pca[tuple(pca_q[:,i].astype(np.int64) for i in range(k))]))))


def compute_delta_map(
    labels_fine:      np.ndarray,
    parent_map:       Optional[Dict[int, List[int]]] = None,
    images_recon_all: np.ndarray = None,
    pca_feats_recon:  np.ndarray = None,
    label_image:      np.ndarray = None,
    label_pca_map:    np.ndarray = None,
    bins_rgb: int = 32, bins_pca: int = 8,
    smooth_sigma: float = 0.1, min_pixels: int = 2,
    use_label_as_target: bool = True, eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict, List[int]]:
    """CPU per-superpixel reference implementation."""
    H_img, W_img = labels_fine.shape
    delta_map = np.full((H_img, W_img), np.nan, dtype=np.float32)
    info: Dict = {}; labels_used: List[int] = []
    child_to_parent: Dict[int,int] = {}
    if parent_map:
        for oid, ch in parent_map.items():
            for c in ch: child_to_parent[c] = oid

    for rid in sorted(np.unique(labels_fine).astype(int).tolist()):
        sp_mask = labels_fine == rid
        n_pix   = int(sp_mask.sum())
        if n_pix < min_pixels: continue
        rows, cols     = np.where(sp_mask)
        recon_rgb      = images_recon_all[:, rows, cols, :3].reshape(-1, 3).astype(np.uint8)
        if recon_rgb.shape[0] == 0: continue
        recon_pca_flat = pca_feats_recon[:, rows, cols, :].reshape(-1, pca_feats_recon.shape[-1])
        pca_min = np.percentile(recon_pca_flat, 1,  axis=0)
        pca_max = np.percentile(recon_pca_flat, 99, axis=0)
        rq_rgb  = quantize_u8_to_bins(recon_rgb, bins_rgb)
        rq_pca  = quantize_pca_adaptive(recon_pca_flat, bins_pca, pca_min, pca_max)
        pmf_rgb, pmf_pca = joint_nd_pmf(rq_rgb, rq_pca, bins_rgb, bins_pca, smooth_sigma, eps)
        H_bits = entropy_bits_combined(pmf_rgb, pmf_pca)
        if not (use_label_as_target and label_image is not None): continue
        tgt_rgb = label_image[sp_mask, :3].astype(np.uint8)
        tgt_pca = label_pca_map[sp_mask]
        if tgt_rgb.shape[0] == 0: continue
        avg_nlogp = avg_neg_logp_bits_combined(
            quantize_u8_to_bins(tgt_rgb, bins_rgb),
            quantize_pca_adaptive(tgt_pca, bins_pca, pca_min, pca_max),
            pmf_rgb, pmf_pca)
        delta_sp = float(np.abs(avg_nlogp - H_bits))
        delta_map[sp_mask] = delta_sp
        info[int(rid)] = {"orig_parent": child_to_parent.get(rid), "num_pixels": n_pix,
                          "H_bits": float(H_bits), "avg_neg_logp_bits": float(avg_nlogp),
                          "delta_sp": float(delta_sp)}
        labels_used.append(int(rid))
    return delta_map, info, sorted(labels_used)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-accelerated implementation
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_map_gpu(
    labels_fine:      np.ndarray,
    images_recon_all: np.ndarray,
    pca_feats_recon:  np.ndarray,
    label_image:      np.ndarray,
    label_pca_map:    np.ndarray,
    parent_map:       Optional[Dict[int, List[int]]] = None,
    bins_rgb:         int   = 32,
    bins_pca:         int   = 8,
    smooth_sigma:     float = 0.1,
    min_pixels:       int   = 2,
    device:           torch.device = None,
    eps:              float = 1e-12,
    gray_scale:       bool  = False,
    **unused_kwargs,
) -> Tuple[np.ndarray, Dict, List[int]]:
    """
    GPU-accelerated factorized PMF scoring with automatic memory management.

    For each feature space (RGB, PCA) independently:
      • If n_sp × n_bins ≤ MAX_HIST_ELEMENTS  → dense int16 histogram (fast, smoothable)
      • Otherwise                              → sparse sort+unique (O(N×H×W), any bin count)

    This means bins_pca=16 / n_pca=5 / 5635 SPs works fine — the PCA histogram
    (5.9 B entries) automatically uses the sparse path while RGB stays dense.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W    = labels_fine.shape
    N_recon = images_recon_all.shape[0]
    n_pca   = pca_feats_recon.shape[-1]

    # ── Remap SP labels (int32 saves vs int64) ────────────────────────────
    sp_ids_orig = np.unique(labels_fine)
    n_sp        = len(sp_ids_orig)
    remap       = np.zeros(int(sp_ids_orig.max()) + 1, dtype=np.int32)
    for ni, oi in enumerate(sp_ids_orig):
        remap[oi] = ni
    labels_c  = remap[labels_fine]
    sp_counts = np.bincount(labels_c.ravel(), minlength=n_sp)
    valid_sp  = sp_counts >= max(min_pixels, 1)

    sp_r = torch.from_numpy(labels_c.ravel()).to(device, dtype=torch.int32).repeat(N_recon)
    sp_l = torch.from_numpy(labels_c.ravel()).to(device, dtype=torch.int32)

    # ── Quantise ─────────────────────────────────────────────────────────
    if gray_scale:
        r_rgb    = images_recon_all[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
        l_rgb    = label_image[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
        rgb_dims = (bins_rgb,)
    else:
        r_rgb    = images_recon_all[..., :3].reshape(-1, 3).astype(np.uint8)
        l_rgb    = label_image[..., :3].reshape(-1, 3).astype(np.uint8)
        rgb_dims = (bins_rgb, bins_rgb, bins_rgb)

    recon_pca_flat = pca_feats_recon.reshape(-1, n_pca)
    label_pca_flat = label_pca_map.reshape(-1, n_pca)

    # Data-driven PCA range (robust percentiles from recon)
    pca_min = np.percentile(recon_pca_flat, 1,  axis=0)
    pca_max = np.percentile(recon_pca_flat, 99, axis=0)

    rq_rgb = quantize_u8_to_bins(r_rgb, bins_rgb)
    rq_pca = quantize_pca_adaptive(recon_pca_flat, bins_pca, pca_min, pca_max)
    lq_rgb = quantize_u8_to_bins(l_rgb, bins_rgb)
    lq_pca = quantize_pca_adaptive(label_pca_flat, bins_pca, pca_min, pca_max)

    pca_dims   = (bins_pca,) * n_pca
    n_rgb_bins = int(np.prod(rgb_dims))
    n_pca_bins = int(np.prod(pca_dims))

    # Flat bin indices: int32 safe for RGB (max 32^3=32768), int64 for PCA
    if gray_scale:
        rb = rq_rgb[:, 0].astype(np.int32); lb = lq_rgb[:, 0].astype(np.int32)
    else:
        rb = np.ravel_multi_index((rq_rgb[:,0].astype(np.int64),
                                   rq_rgb[:,1].astype(np.int64),
                                   rq_rgb[:,2].astype(np.int64)), dims=rgb_dims).astype(np.int32)
        lb = np.ravel_multi_index((lq_rgb[:,0].astype(np.int64),
                                   lq_rgb[:,1].astype(np.int64),
                                   lq_rgb[:,2].astype(np.int64)), dims=rgb_dims).astype(np.int32)

    rp = np.ravel_multi_index(
        tuple(rq_pca[:,i].astype(np.int64) for i in range(n_pca)), dims=pca_dims)
    lp = np.ravel_multi_index(
        tuple(lq_pca[:,i].astype(np.int64) for i in range(n_pca)), dims=pca_dims)

    # ── RGB ───────────────────────────────────────────────────────────────
    pmf_rgb = _try_dense_hist(sp_r, rb, n_sp, n_rgb_bins, rgb_dims,
                              device, smooth_sigma, eps)
    if pmf_rgb is not None:
        entropy_rgb = _dense_entropy(pmf_rgb, n_sp, eps)
        avg_nll_rgb = _dense_avg_nll(lb.astype(np.int64), pmf_rgb, n_rgb_bins,
                                     n_sp, sp_l, sp_counts, device, eps)
    else:
        print(f"[INFO] RGB histogram using sparse path "
              f"(n_sp={n_sp}, n_rgb_bins={n_rgb_bins:,})")
        entropy_rgb, avg_nll_rgb = _sparse_entropy_and_nll(
            sp_r, sp_l, rb.astype(np.int64), lb.astype(np.int64),
            n_sp, n_rgb_bins, sp_counts, device, eps)

    # ── PCA: dense if fits, sparse otherwise (CRASH FIX) ─────────────────
    pmf_pca = _try_dense_hist(sp_r, rp.astype(np.int32) if rp.max() < 2**31-1 else rp,
                              n_sp, n_pca_bins, pca_dims,
                              device, smooth_sigma, eps)
    if pmf_pca is not None:
        entropy_pca = _dense_entropy(pmf_pca, n_sp, eps)
        avg_nll_pca = _dense_avg_nll(lp, pmf_pca, n_pca_bins,
                                     n_sp, sp_l, sp_counts, device, eps)
    else:
        mem_mb = n_sp * n_pca_bins * 2 / 1e6
        print(f"[INFO] PCA histogram too large for dense "
              f"({mem_mb:.0f} MB at int16, n_sp={n_sp}, n_pca_bins={n_pca_bins:,}); "
              f"using sparse path — no smoothing applied to PCA.")
        entropy_pca, avg_nll_pca = _sparse_entropy_and_nll(
            sp_r, sp_l, rp, lp, n_sp, n_pca_bins, sp_counts, device, eps)

    # ── Delta = |avg_NLL − H| ─────────────────────────────────────────────
    delta_sp = torch.abs((avg_nll_rgb + avg_nll_pca) - (entropy_rgb + entropy_pca))
    valid_t  = torch.from_numpy(valid_sp).to(device)
    delta_sp = torch.where(valid_t, delta_sp, torch.full_like(delta_sp, float("nan")))

    d_np = delta_sp.cpu().numpy()
    e_np = (entropy_rgb + entropy_pca).cpu().numpy()
    n_np = (avg_nll_rgb + avg_nll_pca).cpu().numpy()

    delta_map_out = d_np[labels_c].astype(np.float32)
    delta_map_out[~valid_sp[labels_c]] = np.nan

    child_to_parent: Dict[int,int] = {}
    if parent_map:
        for oid, ch in parent_map.items():
            for c in ch: child_to_parent[c] = oid

    info: Dict = {}; labels_used: List[int] = []
    for ni, oi in enumerate(sp_ids_orig):
        if valid_sp[ni]:
            info[int(oi)] = {"orig_parent": child_to_parent.get(int(oi)),
                             "num_pixels": float(sp_counts[ni]),
                             "H_bits": float(e_np[ni]),
                             "avg_neg_logp_bits": float(n_np[ni]),
                             "delta_sp": float(d_np[ni])}
            labels_used.append(int(oi))

    return delta_map_out, info, sorted(labels_used)