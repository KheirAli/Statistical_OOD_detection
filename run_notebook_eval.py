#!/usr/bin/env python3
"""Run the original notebook evaluation logic as a script.

Faithfully reproduces New_dataset_clean.ipynb cells 0-5, 6-7, 9, 13-14, 15.
Uses per-sample superpixel masks and correct label images.

Usage:
    python run_notebook_eval.py --sample samples_010 --device cuda:2
    python run_notebook_eval.py --all --device cuda:2
"""
import argparse
import gc
import os
import sys
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.segmentation import slic, find_boundaries
from skimage.util import img_as_float
from torchvision.models.feature_extraction import create_feature_extractor


# ── Default paths (matching notebook) ────────────────────────────────
DEFAULT_RESULTS_DIR = "./results_patches"
DEFAULT_FIGURES_DIR = "./figures"
DEFAULT_TEST_ORIGIN = "Combined_half_sigma_batched"
DEFAULT_GT_ROOT = "/data/akheirandish3/mvtec_ad/cable/ground_truth/combined"


# ── Superpixel functions (Cell 5) ────────────────────────────────────
def rgb_variance(img, mask):
    if mask.sum() == 0:
        return 0.0
    pix = img[mask]
    if pix.ndim == 1:
        return float(np.var(pix))
    return float(np.mean([np.var(pix[:, c]) for c in range(pix.shape[1])]))


def slic_features_lab(img, alpha_grad=10.0):
    img_f = img_as_float(img).astype(np.float32)
    if img_f.ndim == 2:
        g = sobel(img_f)
        return np.dstack([img_f, alpha_grad * g])
    lab = rgb2lab(img_f).astype(np.float32)
    gray = rgb2gray(img_f)
    g = sobel(gray).astype(np.float32)
    return np.dstack([lab, alpha_grad * g[..., None]])


def split_region(img, full_mask, region_id, n_sub=4, compactness=8.0, alpha_grad=10.0):
    m = full_mask == region_id
    if m.sum() < n_sub:
        return full_mask
    rows, cols = np.where(m)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    img_crop = img[r0:r1, c0:c1]
    m_crop = m[r0:r1, c0:c1]
    feats = slic_features_lab(img_crop, alpha_grad=alpha_grad)
    sub = slic(feats, n_segments=n_sub, compactness=compactness, sigma=0.5,
               start_label=0, mask=m_crop, channel_axis=-1)
    new_labels = full_mask.copy()
    base = int(full_mask.max()) + 1
    rr, cc = np.where(m_crop)
    used = np.unique(sub[rr, cc])
    used = used[used >= 0]
    for i, u in enumerate(used):
        sel = (sub == u) & m_crop
        r_sel, c_sel = np.where(sel)
        new_labels[r0 + r_sel, c0 + c_sel] = base + i
    return new_labels


def recursive_subdivide(img, labels, var_threshold=0.0, min_pixels=20,
                        max_sub=6, max_depth=4, compactness=12.0,
                        alpha_grad=10.0, target_size=10):
    from collections import deque
    labels_out = labels.copy().astype(np.int32)
    orig_ids = list(np.unique(labels_out))
    parent_map = {int(oid): [int(oid)] for oid in orig_ids}

    def _update_parent(orig_parent, old_child, new_children):
        children = parent_map[int(orig_parent)]
        children = [c for c in children if c != old_child]
        children.extend(new_children)
        parent_map[int(orig_parent)] = children

    queue = deque()
    for oid in orig_ids:
        queue.append((int(oid), 0, int(oid)))
    while queue:
        rid, depth, orig_parent = queue.popleft()
        m = labels_out == rid
        area = int(m.sum())
        if area < min_pixels or depth >= max_depth:
            continue
        v = rgb_variance(img, m)
        if v < var_threshold:
            continue
        n_sub = max(2, min(max_sub, area // target_size))
        if n_sub < 2:
            continue
        old_max = int(labels_out.max())
        labels_out = split_region(img, labels_out, rid, n_sub=n_sub,
                                  compactness=compactness, alpha_grad=alpha_grad)
        new_max = int(labels_out.max())
        if new_max == old_max:
            continue
        new_children = list(range(old_max + 1, new_max + 1))
        if (labels_out == rid).sum() > 0:
            new_children.append(rid)
        _update_parent(orig_parent, rid, new_children)
        for child in new_children:
            if child != rid:
                queue.append((child, depth + 1, orig_parent))
    final_ids = sorted(list(np.unique(labels_out).astype(int)))
    return labels_out, final_ids, parent_map


# ── Embedder (Cell 6) ────────────────────────────────────────────────
def patchify_context(features, patchsize=3, stride=1):
    padding = (patchsize - 1) // 2
    unfolder = torch.nn.Unfold(kernel_size=patchsize, stride=stride, padding=padding)
    B, C, H, W = features.shape
    unfolded = unfolder(features)
    unfolded = unfolded.view(B, C, patchsize * patchsize, H * W)
    pooled = unfolded.mean(dim=2)
    return pooled.view(B, C, H, W)


class ResNetPixelEmbedder(nn.Module):
    def __init__(self, resnet_name="resnet18", layers=("layer1", "layer2", "layer3"),
                 out_size=None, use_imagenet_norm=True, use_patch_context=True,
                 proj_dim_per_layer=None):
        super().__init__()
        if resnet_name == "resnet18":
            try:
                net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except AttributeError:
                net = models.resnet18(pretrained=True)
        elif resnet_name == "resnet50":
            try:
                net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            except AttributeError:
                net = models.resnet50(pretrained=True)
        else:
            raise ValueError("resnet_name must be resnet18 or resnet50")
        net.eval()
        return_nodes = {ln: ln for ln in layers}
        self.extractor = create_feature_extractor(net, return_nodes=return_nodes)
        self.layers = layers
        self.out_size = out_size
        self.use_patch_context = use_patch_context
        self.use_imagenet_norm = use_imagenet_norm
        if use_imagenet_norm:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @torch.no_grad()
    def forward(self, x):
        B, _, H, W = x.shape
        if x.min() < 0:
            x = (x + 1) / 2.0
        if self.use_imagenet_norm:
            x = (x - self.mean) / self.std
        feats = self.extractor(x)
        outH, outW = (self.out_size, self.out_size) if self.out_size is not None else (H, W)
        ups = []
        for ln in self.layers:
            f = feats[ln]
            if self.use_patch_context:
                f = patchify_context(f, patchsize=3, stride=1)
            f = F.interpolate(f, size=(outH, outW), mode="bilinear", align_corners=False)
            f = F.normalize(f, dim=1)
            ups.append(f)
        return torch.cat(ups, dim=1)


def to_tensor01(img_np, device="cuda"):
    t = torch.from_numpy(img_np).float()
    if t.ndim == 2:
        t = t[..., None].repeat(1, 1, 3)
    if t.max() > 1.0:
        t = t / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


# ── Scoring (Cell 13) ────────────────────────────────────────────────
def quantize_u8_to_bins(x_u8, bins):
    x = x_u8.astype(np.uint16)
    return ((x * bins) // 256).astype(np.uint8)


def quantize_pca_to_bins(x_pca, bins):
    x = np.clip(x_pca, -1.0, 1.0)
    x_shifted = (x + 1.0) / 2.0
    return np.clip((x_shifted * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)


def joint_nd_pmf(rgb_q, pca_q, bins_rgb, bins_pca, smooth_sigma=1.0, eps=1e-12):
    r, g, b = rgb_q[:, 0].astype(np.int64), rgb_q[:, 1].astype(np.int64), rgb_q[:, 2].astype(np.int64)
    idx_rgb = np.ravel_multi_index((r, g, b), dims=(bins_rgb,) * 3)
    hist_rgb = np.bincount(idx_rgb, minlength=bins_rgb**3).astype(np.float64).reshape((bins_rgb,) * 3)
    if smooth_sigma > 0:
        hist_rgb = gaussian_filter(hist_rgb, sigma=smooth_sigma, mode="nearest")
    hist_rgb += eps
    pmf_rgb = hist_rgb / hist_rgb.sum()

    k = pca_q.shape[1]
    pca_tensors = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
    idx_pca = np.ravel_multi_index(pca_tensors, dims=(bins_pca,) * k)
    hist_pca = np.bincount(idx_pca, minlength=bins_pca**k).astype(np.float64).reshape((bins_pca,) * k)
    if smooth_sigma > 0:
        hist_pca = gaussian_filter(hist_pca, sigma=smooth_sigma, mode="nearest")
    hist_pca += eps
    pmf_pca = hist_pca / hist_pca.sum()
    return pmf_rgb, pmf_pca


def entropy_bits_combined(pmf_rgb, pmf_pca, eps=1e-12):
    def H(pmf):
        p = pmf.ravel()
        return float(-np.sum(p * np.log2(p + eps)))
    return H(pmf_rgb) + H(pmf_pca)


def avg_neg_logp_bits_combined(rgb_q, pca_q, pmf_rgb, pmf_pca, eps=1e-12):
    r, g, b = rgb_q[:, 0].astype(np.int64), rgb_q[:, 1].astype(np.int64), rgb_q[:, 2].astype(np.int64)
    p_rgb = pmf_rgb[r, g, b]
    nll_rgb = float(-np.mean(np.log2(p_rgb + eps)))
    k = pca_q.shape[1]
    pca_tensors = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
    p_pca = pmf_pca[pca_tensors]
    nll_pca = float(-np.mean(np.log2(p_pca + eps)))
    return nll_rgb + nll_pca


def compute_pca_basis(label_feat_cpu, n_components=9):
    C, H, W = label_feat_cpu.shape
    X = label_feat_cpu.permute(1, 2, 0).reshape(-1, C).numpy()
    mu = X.mean(axis=0, keepdims=True)
    X_centered = X - mu
    _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    explained = (S[:n_components]**2) / (S**2).sum()
    print(f"  PCA explained variance ({n_components} components): {explained.sum() * 100:.1f}%")
    return mu, components


def project_to_pca(feat_hwc, mu, components):
    proj = (feat_hwc - mu) @ components.T
    return np.clip(proj, -1.0, 1.0)


# ── Metrics (Cell 15) ────────────────────────────────────────────────
def manual_roc_curve(y_true, y_score):
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


def manual_auc(x, y):
    order = np.argsort(x)
    x, y = x[order], y[order]
    return float(np.trapz(y, x))


def manual_precision_recall_curve(y_true, y_score):
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
    return precision, recall, y_score


def manual_average_precision(y_true, y_score):
    precision, recall, _ = manual_precision_recall_curve(y_true, y_score)
    # NOTE: original notebook has -np.sum here (sign bug), keeping it for exact reproduction
    return float(-np.sum(np.diff(recall) * precision[:-1]))


# ── Main evaluation for one sample ───────────────────────────────────
def evaluate_one_sample(sample_name: str, embedder, device: str,
                        results_dir: str, figures_dir: str,
                        test_origin: str, gt_root: str,
                        patches: int, bins_rgb: int = 32,
                        bins_pca: int = 8) -> dict:
    sample = "_".join(sample_name.split("_")[1:])
    print(f"\n{'='*60}")
    print(f"  Sample: {sample_name} (id={sample})")
    print(f"{'='*60}")

    # ── Load label image (Cell 5 in notebook) ─────────────────────
    label_image_path = f"{results_dir}/{sample_name}/{test_origin}_8_4/inpainting/label/0_00000.png"
    if not os.path.exists(label_image_path):
        raise FileNotFoundError(f"Label image not found: {label_image_path}")
    label_image = io.imread(label_image_path)[:, :, :3]
    print(f"  Label image: {label_image.shape}")

    # ── Load superpixel mask (per-sample) ─────────────────────────
    per_sample_mask_dir = os.path.join(figures_dir, sample_name)
    mask_path = os.path.join(per_sample_mask_dir, "mask.png")
    if not os.path.exists(mask_path):
        # Fall back to global mask
        mask_path = os.path.join(figures_dir, "mask.png")
    mask = io.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.int32)
    print(f"  Mask: {mask_path} ({len(np.unique(mask))} regions)")

    # ── Load reconstructions (Cell 4) ─────────────────────────────
    images_all = []
    for item in range(patches):
        directory = f"{results_dir}/{sample_name}/{test_origin}_{item}_4/inpainting/recon"
        image_paths = sorted(glob(os.path.join(directory, "*.png")))
        for path in image_paths:
            img = io.imread(path)
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            images_all.append(img)
    images_recon_all = np.array(images_all)
    print(f"  Reconstructions: {images_recon_all.shape}")

    # ── Recursive subdivide (Cell 5) ──────────────────────────────
    labels_fine, final_ids, parent_map = recursive_subdivide(
        img=label_image, labels=mask,
        var_threshold=0.0, min_pixels=20, max_sub=6, max_depth=4,
        compactness=12.0, alpha_grad=10.0, target_size=10,
    )
    print(f"  Superpixels: {len(np.unique(mask))} -> {len(final_ids)}")

    # ── Embed reconstructions (Cell 7) ────────────────────────────
    x_all = torch.from_numpy(images_recon_all).float().permute(0, 3, 1, 2)
    if x_all.max() > 1.0:
        x_all = x_all / 255.0
    feat_list = []
    with torch.no_grad():
        for i in range(x_all.shape[0]):
            xi = x_all[i:i+1].to(device)
            fi = embedder(xi)
            feat_list.append(fi.cpu())
            del xi, fi
            if i % 10 == 0:
                torch.cuda.empty_cache()
    feat_map = torch.cat(feat_list, dim=0)
    del feat_list
    gc.collect()
    torch.cuda.empty_cache()

    # ── PCA (Cell 14) ─────────────────────────────────────────────
    n_pca = 5  # notebook uses 9 but 8^9=134M bins is too slow; 8^5=32K is tractable
    model_device = next(embedder.parameters()).device
    with torch.no_grad():
        label_x = to_tensor01(label_image, device=str(model_device))
        label_feat = embedder(label_x).squeeze(0).cpu()

    mu_pca, components_pca = compute_pca_basis(label_feat, n_components=n_pca)

    C, H, W = label_feat.shape
    X_label = label_feat.permute(1, 2, 0).reshape(-1, C).numpy()
    label_pca_map = project_to_pca(X_label, mu_pca, components_pca).reshape(H, W, n_pca)

    B = feat_map.shape[0]
    feat_np = feat_map.permute(0, 2, 3, 1).numpy()
    feat_flat = feat_np.reshape(B, -1, C)
    feat_centered = feat_flat - mu_pca[None]
    proj_flat = feat_centered @ components_pca.T
    pca_feats_recon = np.clip(proj_flat, -1.0, 1.0).reshape(B, H, W, n_pca)

    # ── Delta map (Cell 14 scoring) ───────────────────────────────
    delta_map = np.full((H, W), np.nan, dtype=np.float32)
    labels_used = []
    all_refined_ids = sorted(np.unique(labels_fine).astype(int).tolist())

    for refined_id in all_refined_ids:
        sp_mask = labels_fine == refined_id
        n_pix = int(sp_mask.sum())
        if n_pix < 2:
            continue

        recon_rgb = images_recon_all[:, sp_mask, :3].reshape(-1, 3).astype(np.uint8)
        if recon_rgb.shape[0] == 0:
            continue
        recon_rgb_q = quantize_u8_to_bins(recon_rgb, bins=bins_rgb)

        N = images_recon_all.shape[0]
        recon_pca = pca_feats_recon[:N, sp_mask, :]
        recon_pca_flat = recon_pca.reshape(-1, recon_pca.shape[-1])
        recon_pca_q = quantize_pca_to_bins(recon_pca_flat, bins=bins_pca)

        pmf_rgb, pmf_pca = joint_nd_pmf(recon_rgb_q, recon_pca_q,
                                         bins_rgb=bins_rgb, bins_pca=bins_pca,
                                         smooth_sigma=0.1, eps=1e-12)
        H_bits = entropy_bits_combined(pmf_rgb, pmf_pca, eps=1e-12)

        target_rgb = label_image[sp_mask, :3].astype(np.uint8)
        target_pca = label_pca_map[sp_mask]
        if target_rgb.shape[0] == 0:
            continue
        target_rgb_q = quantize_u8_to_bins(target_rgb, bins=bins_rgb)
        target_pca_q = quantize_pca_to_bins(target_pca, bins=bins_pca)

        avg_nlogp = avg_neg_logp_bits_combined(target_rgb_q, target_pca_q,
                                                pmf_rgb, pmf_pca, eps=1e-12)
        delta_sp = float(np.abs(avg_nlogp - H_bits))
        delta_map[sp_mask] = delta_sp
        labels_used.append(refined_id)

    print(f"  Scored {len(labels_used)} superpixels")

    # ── GT mask (Cell 15) ─────────────────────────────────────────
    gt_mask_path = f"{gt_root}/{sample}_mask.png"
    if not os.path.exists(gt_mask_path):
        raise FileNotFoundError(f"GT mask not found: {gt_mask_path}")
    gt_raw = np.array(Image.open(gt_mask_path).convert("L"))
    gt_mask_binary = (gt_raw[::4, ::4] > 0).astype(np.uint8)

    # ── Evaluation (Cell 15/16: raw delta, Cell 17: smoothed) ─────
    results = {}
    for label_str, score_map in [("raw", delta_map),
                                  ("smoothed_sigma5", gaussian_filter(
                                      np.nan_to_num(delta_map, nan=0.0), sigma=5))]:
        all_sp_ids = sorted(np.unique(labels_fine).astype(int).tolist())
        sp_scores, sp_labels = [], []
        for sp_id in all_sp_ids:
            sp_m = labels_fine == sp_id
            if int(sp_m.sum()) < 1:
                continue
            score = float(np.nanmean(score_map[sp_m]))
            if np.isnan(score):
                continue
            frac = gt_mask_binary[sp_m].mean()
            sp_scores.append(score)
            sp_labels.append(int(frac > 0.5))
        sp_scores = np.array(sp_scores)
        sp_labels = np.array(sp_labels)

        fpr, tpr, _ = manual_roc_curve(sp_labels, sp_scores)
        sp_auc = manual_auc(fpr, tpr)
        sp_ap = manual_average_precision(sp_labels, sp_scores)

        valid = ~np.isnan(score_map)
        px_scores = score_map[valid].ravel()
        px_labels = gt_mask_binary[valid].ravel()
        fpr_px, tpr_px, _ = manual_roc_curve(px_labels, px_scores)
        px_auc = manual_auc(fpr_px, tpr_px)
        px_ap = manual_average_precision(px_labels, px_scores)

        results[label_str] = {
            "sp_auc": sp_auc, "sp_ap": sp_ap,
            "px_auc": px_auc, "px_ap": px_ap,
        }
        print(f"  [{label_str}] SP AUC={sp_auc:.4f} | Pixel AUC={px_auc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="samples_010")
    parser.add_argument("--all", action="store_true", help="Run all cable samples 000-010")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=str, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--test-origin", type=str, default=DEFAULT_TEST_ORIGIN)
    parser.add_argument("--gt-root", type=str, default=DEFAULT_GT_ROOT)
    parser.add_argument("--patches", type=int, default=24)
    parser.add_argument("--bins-rgb", type=int, default=32)
    parser.add_argument("--bins-pca", type=int, default=8)
    args = parser.parse_args()

    device = args.device

    embedder = ResNetPixelEmbedder(
        resnet_name="resnet18",
        layers=("layer1", "layer2", "layer3"),
        out_size=None,
        use_patch_context=True,
        proj_dim_per_layer=None,
    ).to(device).eval()

    if args.all:
        sample_names = [f"samples_{i:03d}" for i in range(11)]
    else:
        sample_names = [args.sample]

    all_results = {}
    for sname in sample_names:
        try:
            all_results[sname] = evaluate_one_sample(
                sname, embedder, device,
                results_dir=args.results_dir,
                figures_dir=args.figures_dir,
                test_origin=args.test_origin,
                gt_root=args.gt_root,
                patches=args.patches,
                bins_rgb=args.bins_rgb,
                bins_pca=args.bins_pca,
            )
        except Exception as e:
            print(f"  FAILED: {e}")

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  AVERAGES ({len(all_results)} samples)")
        print(f"{'='*60}")
        for key in ["raw", "smoothed_sigma5"]:
            sp_aucs = [v[key]["sp_auc"] for v in all_results.values()]
            px_aucs = [v[key]["px_auc"] for v in all_results.values()]
            print(f"  [{key}] Mean SP AUC={np.mean(sp_aucs):.4f} | Mean Pixel AUC={np.mean(px_aucs):.4f}")


if __name__ == "__main__":
    main()
