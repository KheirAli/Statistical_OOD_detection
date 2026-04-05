"""Visualization: delta maps, ROC curves, evaluation overlays.

Extracted from New_dataset_clean.ipynb Cells 8, 11, 14, 16.
"""

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # headless by default
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float


def show_boundaries(
    img: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Overlay superpixel boundaries on image."""
    b = find_boundaries(labels, mode="thick")
    vis = img_as_float(img).copy()
    if vis.ndim == 2:
        vis[b] = 1.0
        cmap = "gray"
    else:
        vis[b, :] = [1.0, 1.0, 0.0]
        cmap = None
    plt.figure(figsize=(7, 7))
    plt.imshow(vis, cmap=cmap)
    plt.axis("off")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_recon_vs_label(
    images_recon_all: np.ndarray,
    label_image: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side: mean reconstruction vs label image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    mean_recon = images_recon_all.mean(axis=0).astype(np.uint8)
    axes[0].imshow(mean_recon)
    axes[0].set_title("Mean reconstruction")
    axes[0].axis("off")

    axes[1].imshow(label_image)
    axes[1].set_title("Label image")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_delta_map(
    delta_map: np.ndarray,
    label_image: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Delta heatmap + overlay on label image."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(np.nan_to_num(delta_map), cmap="hot")
    axes[0].set_title("Delta map (RGB + PCA)")
    axes[0].axis("off")

    base = label_image.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    axes[1].imshow(base)
    axes[1].set_title("Label image")
    axes[1].axis("off")

    axes[2].imshow(base)
    axes[2].imshow(np.nan_to_num(delta_map), cmap="hot", alpha=0.5)
    axes[2].set_title("Delta overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_evaluation(
    eval_results: Dict,
    gt_mask_binary: np.ndarray,
    delta_map: np.ndarray,
    smooth_sigma: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """4-panel plot: SP ROC, pixel ROC, GT overlay, delta heatmap."""
    from scipy.ndimage import gaussian_filter

    curves = eval_results["curves"]
    fpr_sp = np.array(curves["sp_fpr"])
    tpr_sp = np.array(curves["sp_tpr"])
    fpr_px = np.array(curves["px_fpr"])
    tpr_px = np.array(curves["px_tpr"])

    sp_auc = eval_results["sp_roc_auc"]
    px_auc = eval_results["px_roc_auc"]

    score_map = np.nan_to_num(delta_map, nan=0.0)
    if smooth_sigma is not None:
        score_map = gaussian_filter(score_map, sigma=smooth_sigma)

    vmax_d = np.percentile(score_map[score_map > 0], 97) if (score_map > 0).any() else 1.0

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), gridspec_kw={"width_ratios": [1, 1, 1, 1]})

    # SP ROC
    ax = axes[0]
    ax.plot(fpr_sp, tpr_sp, color="darkorange", lw=2, label=f"SP AUC = {sp_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (Superpixel)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_aspect("equal")

    # Pixel ROC
    ax = axes[1]
    ax.plot(fpr_px, tpr_px, color="crimson", lw=2, label=f"Pixel AUC = {px_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (Pixel)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_aspect("equal")

    # GT overlay
    ax = axes[2]
    vis = np.zeros((*gt_mask_binary.shape, 3), dtype=np.float32)
    sum_norm = np.clip(score_map / (vmax_d + 1e-8), 0, 1)
    vis[..., 0] = sum_norm
    vis[..., 1] = gt_mask_binary.astype(np.float32)
    ax.imshow(np.clip(vis, 0, 1))
    ax.set_title("Red=score, Green=GT")
    ax.axis("off")

    # Heatmap
    ax = axes[3]
    im = ax.imshow(score_map, cmap="hot", vmin=0, vmax=vmax_d)
    ax.set_title("Delta heatmap")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Delta (bits)")

    sigma_str = f"sigma={smooth_sigma}" if smooth_sigma else "raw"
    plt.suptitle(
        f"SP AUC: {sp_auc:.4f} | Pixel AUC: {px_auc:.4f} ({sigma_str})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(
    all_results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing methods by AUROC."""
    methods = list(all_results.keys())
    sp_aucs = [all_results[m].get("sp_roc_auc", 0) for m in methods]
    px_aucs = [all_results[m].get("px_roc_auc", 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, sp_aucs, width, label="SP AUROC", color="darkorange")
    ax.bar(x + width / 2, px_aucs, width, label="Pixel AUROC", color="crimson")

    ax.set_ylabel("AUROC")
    ax.set_title("Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.05])

    for i, (s, p) in enumerate(zip(sp_aucs, px_aucs)):
        ax.text(i - width / 2, s + 0.01, f"{s:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
