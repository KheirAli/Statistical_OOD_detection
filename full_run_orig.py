#!/usr/bin/env python3
"""Batch superpixel/pixel anomaly evaluation with configurable PCA + ResNet embedding.

Example:
    python full_run.py \
        --sample-start 0 \
        --sample-end 10 \
        --n-pca 6 \
        --resnet-model resnet18
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

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
from skimage.segmentation import find_boundaries, slic
from skimage.transform import resize
from skimage.util import img_as_float
from torchvision.models.feature_extraction import create_feature_extractor


# FACE_SAMPLE_NAMES = [
#     "samples_00107",
#     "samples_00129_1",
#     "samples_00040_1",
#     "samples_00044_1",
#     "samples_00049_1",
#     "samples_00061_1",
#     "samples_00065_1",
#     "samples_00080_1",
#     "samples_00092_1",
#     "samples_00098_1",
# ]

FACE_SAMPLE_NAMES = [
    "samples_107",
    "samples_129",
    "samples_40",
    "samples_44",
    "samples_49",
    "samples_61",
    "samples_65",
    "samples_80",
    "samples_92",
    "samples_98",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch anomaly evaluation over samples.")
    parser.add_argument(
        "--category",
        choices=["cable", "faces"],
        default="cable",
        help="Dataset category preset. cable uses sequential samples; faces uses fixed sample names.",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--results-dir", default="./results_patches")
    parser.add_argument("--figures-dir", default="./figures")
    parser.add_argument("--test-origin", default="Combined_half_sigma_batched")
    parser.add_argument("--defect-type", default="combined", help="MVTec defect folder (e.g., combined, bent_wire)")
    parser.add_argument("--gt-root", default="/data/akheirandish3/mvtec_ad/cable/ground_truth")

    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=10)
    parser.add_argument("--patches", type=int, default=24)
    parser.add_argument("--label-patch", type=int, default=8)

    parser.add_argument("--resnet-model", choices=["resnet18", "resnet50", "resnet101", "resnet152"], default="resnet18")
    parser.add_argument("--layers", default="layer1,layer2,layer3", help="Comma-separated ResNet layers")
    parser.add_argument("--use-patch-context", action="store_true", default=True)
    parser.add_argument("--no-patch-context", dest="use_patch_context", action="store_false")
    parser.add_argument("--proj-dim", type=int, default=0, help="Set >0 to use 1x1 projection per chosen layer")

    parser.add_argument("--n-pca", type=int, default=5)
    parser.add_argument("--bins-rgb", type=int, default=32)
    parser.add_argument("--bins-pca", type=int, default=8)
    parser.add_argument("--smooth-sigma", type=float, default=0.1)
    parser.add_argument("--min-pixels", type=int, default=2)

    parser.add_argument("--var-threshold", type=float, default=0.0)
    parser.add_argument("--split-min-pixels", type=int, default=20)
    parser.add_argument("--max-sub", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--compactness", type=float, default=12.0)
    parser.add_argument("--alpha-grad", type=float, default=10.0)
    parser.add_argument("--target-size", type=int, default=10)

    parser.add_argument("--gt-downsample", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional output text report path. If empty, saved under results_dir with timestamp.",
    )
    parser.add_argument("--strict", action="store_true", help="Stop on first sample failure")
    return parser.parse_args()


def resolve_report_path(args: argparse.Namespace) -> str:
    if args.report_path.strip():
        return args.report_path.strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.category == "faces":
        name = f"evaluation_report_faces_{ts}.txt"
    else:
        name = f"evaluation_report_cable_{args.sample_start:03d}_{args.sample_end:03d}_{ts}.txt"
    return os.path.join(args.results_dir, name)


def resolve_sample_names(args: argparse.Namespace) -> List[str]:
    if args.category == "faces":
        return FACE_SAMPLE_NAMES
    return [f"samples_{idx:03d}" for idx in range(args.sample_start, args.sample_end + 1)]


# def apply_category_presets(args: argparse.Namespace) -> None:
#     if args.category == "faces":
#         args.test_origin = "New_masking_half_sigma_batched"
#         args.gt_root = "/data/akheirandish3/mvtec_ad/faces/ground_truth/random"

def apply_category_presets(args):
    if args.category == "faces":
        if args.test_origin == "Combined_half_sigma_batched":
            args.test_origin = "New_masking_half_sigma_batched"
        if args.gt_root == "/data/akheirandish3/mvtec_ad/cable/ground_truth":
            args.gt_root = "/data/akheirandish3/mvtec_ad/faces/ground_truth/random"
def extract_sample_id(sample_name: str) -> str:
    """Extract numeric ID used by GT mask naming, e.g. samples_00129_1 -> 00129."""
    m = re.search(r"samples_(\d+)", sample_name)
    if not m:
        raise ValueError(f"Unable to extract numeric ID from sample name: {sample_name}")
    return m.group(1)


def save_text_report(
    report_path: str,
    args: argparse.Namespace,
    device: torch.device,
    layers: Tuple[str, ...],
    results: List["SampleMetrics"],
    failed_samples: List[str],
    sp_roc_avg: float,
    sp_ap_avg: float,
    px_roc_avg: float,
    px_ap_avg: float,
) -> None:
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)

    lines: List[str] = []
    lines.append("Batch Evaluation Report")
    lines.append("=" * 72)
    lines.append(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Inputs Used")
    lines.append("-" * 72)
    arg_dict = vars(args).copy()
    arg_dict["resolved_device"] = str(device)
    arg_dict["resolved_layers"] = ",".join(layers)
    for key in sorted(arg_dict.keys()):
        lines.append(f"{key}: {arg_dict[key]}")

    lines.append("")
    lines.append("Per-Sample Results")
    lines.append("-" * 72)
    lines.append("sample\tsp_roc_auc\tsp_ap\tpixel_roc_auc\tpixel_ap")
    for item in results:
        lines.append(
            f"{item.sample_name}\t{item.sp_roc_auc:.6f}\t{item.sp_ap:.6f}\t"
            f"{item.pixel_roc_auc:.6f}\t{item.pixel_ap:.6f}"
        )

    lines.append("")
    lines.append("Averages")
    lines.append("-" * 72)
    lines.append(f"avg_sp_roc_auc: {sp_roc_avg:.6f}")
    lines.append(f"avg_sp_ap: {sp_ap_avg:.6f}")
    lines.append(f"avg_pixel_roc_auc: {px_roc_avg:.6f}")
    lines.append(f"avg_pixel_ap: {px_ap_avg:.6f}")
    lines.append(f"successful_samples: {len(results)}")
    lines.append(f"failed_samples: {len(failed_samples)}")
    if failed_samples:
        lines.append(f"failed_list: {', '.join(failed_samples)}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3 and img.shape[-1] == 4:
        return img[..., :3]
    if img.ndim == 3 and img.shape[-1] == 3:
        return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def to_tensor01(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img_np).float()
    if t.ndim == 2:
        t = t[..., None].repeat(1, 1, 3)
    if t.max() > 1.0:
        t = t / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def load_mask(figures_dir: str) -> np.ndarray:
    mask_path = os.path.join(figures_dir, "mask.png")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = io.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32)


def load_label_image(results_dir: str, sample_name: str, test_origin: str, label_patch: int) -> np.ndarray:
    path = os.path.join(
        results_dir,
        sample_name,
        f"{test_origin}_{label_patch}_4",
        "inpainting",
        "label",
        "0_00000.png",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label image not found: {path}")
    return ensure_rgb(io.imread(path))


def load_recon_images(results_dir: str, sample_name: str, test_origin: str, patches: int) -> np.ndarray:
    images_all: List[np.ndarray] = []
    for item in range(patches):
        directory = os.path.join(results_dir, sample_name, f"{test_origin}_{item}_4", "inpainting", "recon")
        image_paths = sorted(glob(os.path.join(directory, "*.png")))
        for path in image_paths:
            images_all.append(ensure_rgb(io.imread(path)))

    if not images_all:
        raise FileNotFoundError(f"No recon images found for sample {sample_name}")

    out = np.array(images_all)
    if out.ndim != 4 or out.shape[-1] != 3:
        raise ValueError(f"Unexpected recon shape: {out.shape}")
    return out


def patchify_context(features: torch.Tensor, patchsize: int = 3, stride: int = 1) -> torch.Tensor:
    padding = (patchsize - 1) // 2
    unfolder = torch.nn.Unfold(kernel_size=patchsize, stride=stride, padding=padding)
    bsz, ch, h, w = features.shape
    unfolded = unfolder(features)
    unfolded = unfolded.view(bsz, ch, patchsize * patchsize, h * w)
    pooled = unfolded.mean(dim=2)
    return pooled.view(bsz, ch, h, w)


class ResNetPixelEmbedder(nn.Module):
    def __init__(
        self,
        resnet_name: str = "resnet18",
        layers: Tuple[str, ...] = ("layer1", "layer2", "layer3"),
        out_size: Optional[int] = None,
        use_imagenet_norm: bool = True,
        use_patch_context: bool = True,
        proj_dim_per_layer: Optional[int] = None,
    ):
        super().__init__()
        if resnet_name == "resnet18":
            try:
                net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except AttributeError:
                net = models.resnet18(pretrained=True)
            ch_map = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
        elif resnet_name == "resnet50":
            try:
                net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            except AttributeError:
                net = models.resnet50(pretrained=True)
            ch_map = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        elif resnet_name == "resnet101":
            try:
                net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            except AttributeError:
                net = models.resnet101(pretrained=True)
            ch_map = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        elif resnet_name == "resnet152":
            try:
                net = models.resnet152(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            except AttributeError:
                net = models.resnet152(pretrained=True)
            ch_map = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        else:
            raise ValueError("resnet_name must be resnet18 or resnet50")

        return_nodes = {ln: ln for ln in layers}
        self.extractor = create_feature_extractor(net.eval(), return_nodes=return_nodes)
        self.layers = layers
        self.out_size = out_size
        self.use_patch_context = use_patch_context
        self.proj_dim_per_layer = proj_dim_per_layer

        self.proj = nn.ModuleDict()
        if proj_dim_per_layer is not None:
            for ln in layers:
                self.proj[ln] = nn.Conv2d(ch_map[ln], proj_dim_per_layer, kernel_size=1, bias=False)

        self.use_imagenet_norm = use_imagenet_norm
        if use_imagenet_norm:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {x.shape}")

        bsz, _, h, w = x.shape
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        if self.use_imagenet_norm:
            x = (x - self.mean) / self.std

        feats = self.extractor(x)
        out_h, out_w = (self.out_size, self.out_size) if self.out_size is not None else (h, w)

        ups = []
        for ln in self.layers:
            f = feats[ln]
            if self.use_patch_context:
                f = patchify_context(f, patchsize=3, stride=1)
            if self.proj_dim_per_layer is not None:
                f = self.proj[ln](f)
            f = F.interpolate(f, size=(out_h, out_w), mode="bilinear", align_corners=False)
            f = F.normalize(f, dim=1)
            ups.append(f)
        return torch.cat(ups, dim=1)


def compute_recon_features(
    embedder: ResNetPixelEmbedder,
    images_recon_all: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    x_all = torch.from_numpy(images_recon_all).float().permute(0, 3, 1, 2)
    if x_all.max() > 1.0:
        x_all = x_all / 255.0

    feats = []
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            xb = x_all[i : i + batch_size].to(device)
            feats.append(embedder(xb).cpu())
    return torch.cat(feats, dim=0)


def rgb_variance(img: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    pix = img[mask]
    if pix.ndim == 1:
        return float(np.var(pix))
    return float(np.mean([np.var(pix[:, c]) for c in range(pix.shape[1])]))


def slic_features_lab(img: np.ndarray, alpha_grad: float = 10.0) -> np.ndarray:
    img_f = img_as_float(img).astype(np.float32)
    if img_f.ndim == 2:
        g = sobel(img_f)
        return np.dstack([img_f, alpha_grad * g])
    lab = rgb2lab(img_f).astype(np.float32)
    gray = rgb2gray(img_f)
    g = sobel(gray).astype(np.float32)
    return np.dstack([lab, alpha_grad * g[..., None]])


def split_region(
    img: np.ndarray,
    full_mask: np.ndarray,
    region_id: int,
    n_sub: int = 4,
    compactness: float = 8.0,
    alpha_grad: float = 10.0,
) -> np.ndarray:
    m = full_mask == region_id
    if m.sum() < n_sub:
        return full_mask

    rows, cols = np.where(m)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    img_crop = img[r0:r1, c0:c1]
    m_crop = m[r0:r1, c0:c1]
    feats = slic_features_lab(img_crop, alpha_grad=alpha_grad)

    sub = slic(
        feats,
        n_segments=n_sub,
        compactness=compactness,
        sigma=0.5,
        start_label=0,
        mask=m_crop,
        channel_axis=-1,
    )

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


def recursive_subdivide(
    img: np.ndarray,
    labels: np.ndarray,
    var_threshold: float = 150.0,
    min_pixels: int = 16,
    max_sub: int = 6,
    max_depth: int = 4,
    compactness: float = 12.0,
    alpha_grad: float = 10.0,
    target_size: int = 200,
) -> Tuple[np.ndarray, List[int], Dict[int, List[int]]]:
    labels_out = labels.copy().astype(np.int32)
    orig_ids = list(np.unique(labels_out))
    parent_map = {int(oid): [int(oid)] for oid in orig_ids}

    def _update_parent(orig_parent: int, old_child: int, new_children: List[int]) -> None:
        children = parent_map[int(orig_parent)]
        children = [c for c in children if c != old_child]
        children.extend(new_children)
        parent_map[int(orig_parent)] = children

    from collections import deque

    queue = deque((int(oid), 0, int(oid)) for oid in orig_ids)

    while queue:
        rid, depth, orig_parent = queue.popleft()
        m = labels_out == rid
        area = int(m.sum())

        if area < min_pixels or depth >= max_depth:
            continue

        if rgb_variance(img, m) < var_threshold:
            continue

        n_sub = max(2, min(max_sub, area // target_size))
        if n_sub < 2:
            continue

        old_max = int(labels_out.max())
        labels_out = split_region(
            img,
            labels_out,
            rid,
            n_sub=n_sub,
            compactness=compactness,
            alpha_grad=alpha_grad,
        )
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


def quantize_u8_to_bins(x_u8: np.ndarray, bins: int) -> np.ndarray:
    x = x_u8.astype(np.uint16)
    return ((x * bins) // 256).astype(np.uint8)


def quantize_pca_to_bins(x_pca: np.ndarray, bins: int) -> np.ndarray:
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
    r = rgb_q[:, 0].astype(np.int64)
    g = rgb_q[:, 1].astype(np.int64)
    b = rgb_q[:, 2].astype(np.int64)
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


def entropy_bits_combined(pmf_rgb: np.ndarray, pmf_pca: np.ndarray, eps: float = 1e-12) -> float:
    def _entropy(pmf: np.ndarray) -> float:
        p = pmf.ravel()
        return float(-np.sum(p * np.log2(p + eps)))

    return _entropy(pmf_rgb) + _entropy(pmf_pca)


def avg_neg_logp_bits_combined(
    rgb_q: np.ndarray,
    pca_q: np.ndarray,
    pmf_rgb: np.ndarray,
    pmf_pca: np.ndarray,
    eps: float = 1e-12,
) -> float:
    r = rgb_q[:, 0].astype(np.int64)
    g = rgb_q[:, 1].astype(np.int64)
    b = rgb_q[:, 2].astype(np.int64)
    p_rgb = pmf_rgb[r, g, b]
    nll_rgb = float(-np.mean(np.log2(p_rgb + eps)))

    k = pca_q.shape[1]
    pca_tensors = tuple(pca_q[:, i].astype(np.int64) for i in range(k))
    p_pca = pmf_pca[pca_tensors]
    nll_pca = float(-np.mean(np.log2(p_pca + eps)))
    return nll_rgb + nll_pca


def compute_pca_basis(label_feat_cpu: torch.Tensor, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    ch, h, w = label_feat_cpu.shape
    x = label_feat_cpu.permute(1, 2, 0).reshape(-1, ch).numpy()
    mu = x.mean(axis=0, keepdims=True)
    x_centered = x - mu
    _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:n_components]
    explained = (s[:n_components] ** 2) / (s**2).sum()
    print(f"PCA explained variance ({n_components}): {explained.sum() * 100:.2f}%")
    return mu, components


def project_to_pca(feat_hwc: np.ndarray, mu: np.ndarray, components: np.ndarray) -> np.ndarray:
    proj = (feat_hwc - mu) @ components.T
    return np.clip(proj, -1.0, 1.0)


def compute_typical_set_with_pca(
    labels_fine: np.ndarray,
    recon_images: np.ndarray,
    pca_feats_recon: np.ndarray,
    bins_rgb: int,
    bins_pca: int,
    smooth_sigma: float,
    min_pixels: int,
    label_image: np.ndarray,
    label_pca_map: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]], List[int]]:
    h_img, w_img = labels_fine.shape
    delta_map = np.full((h_img, w_img), np.nan, dtype=np.float32)
    info: Dict[int, Dict[str, float]] = {}
    labels_used: List[int] = []

    for refined_id in sorted(np.unique(labels_fine).astype(int).tolist()):
        sp_mask = labels_fine == refined_id
        n_pix = int(sp_mask.sum())
        if n_pix < min_pixels:
            continue

        recon_rgb = recon_images[:, sp_mask, :3].reshape(-1, 3).astype(np.uint8)
        if recon_rgb.shape[0] == 0:
            continue
        recon_rgb_q = quantize_u8_to_bins(recon_rgb, bins=bins_rgb)

        recon_pca_flat = pca_feats_recon[:, sp_mask, :].reshape(-1, pca_feats_recon.shape[-1])
        recon_pca_q = quantize_pca_to_bins(recon_pca_flat, bins=bins_pca)

        pmf_rgb, pmf_pca = joint_nd_pmf(
            recon_rgb_q,
            recon_pca_q,
            bins_rgb=bins_rgb,
            bins_pca=bins_pca,
            smooth_sigma=smooth_sigma,
            eps=eps,
        )
        h_bits = entropy_bits_combined(pmf_rgb, pmf_pca, eps=eps)

        target_rgb = label_image[sp_mask, :3].astype(np.uint8)
        target_pca = label_pca_map[sp_mask]
        if target_rgb.shape[0] == 0:
            continue

        target_rgb_q = quantize_u8_to_bins(target_rgb, bins=bins_rgb)
        target_pca_q = quantize_pca_to_bins(target_pca, bins=bins_pca)
        avg_nlogp = avg_neg_logp_bits_combined(target_rgb_q, target_pca_q, pmf_rgb, pmf_pca, eps=eps)

        delta_sp = float(np.abs(avg_nlogp - h_bits))
        delta_map[sp_mask] = delta_sp
        info[refined_id] = {
            "num_pixels": float(n_pix),
            "H_bits": float(h_bits),
            "avg_neg_logp_bits": float(avg_nlogp),
            "delta_sp": float(delta_sp),
        }
        labels_used.append(refined_id)

    return delta_map, info, labels_used


def manual_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    desc_idx = np.argsort(-y_score)
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    distinct_idx = np.where(np.diff(y_score))[0]
    threshold_idx = np.concatenate([distinct_idx, [len(y_true) - 1]])

    tps = np.cumsum(y_true)[threshold_idx]
    fps = (threshold_idx + 1) - tps

    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])

    fpr = fps / fps[-1] if fps[-1] > 0 else fps.astype(np.float64)
    tpr = tps / tps[-1] if tps[-1] > 0 else tps.astype(np.float64)
    thresholds = y_score[threshold_idx]
    return fpr, tpr, thresholds


def manual_auc(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    x_ord, y_ord = x[order], y[order]
    return float(np.trapz(y_ord, x_ord))


def manual_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    desc_idx = np.argsort(-y_score)
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    total_pos = y_true.sum()

    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / total_pos if total_pos > 0 else tps.astype(np.float64)

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall, y_score


def manual_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, _ = manual_precision_recall_curve(y_true, y_score)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def load_gt_mask(gt_root: str, defect_type: str, sample_str: str, target_shape: Tuple[int, int], downsample: int) -> np.ndarray:
    if len(sample_str) > 4:
        sample_str = str(int(sample_str))
    direct_path = os.path.join(gt_root, f"{sample_str}_mask.png")
    defect_path = os.path.join(gt_root, defect_type, f"{sample_str}_mask.png")

    if os.path.exists(direct_path):
        gt_path = direct_path
    elif os.path.exists(defect_path):
        gt_path = defect_path
    else:
        raise FileNotFoundError(f"GT mask not found at either {direct_path} or {defect_path}")

    mask = np.array(Image.open(gt_path).convert("L"))
    if downsample > 1:
        mask = mask[::downsample, ::downsample]
    gt_mask = (mask > 0).astype(np.uint8)

    if gt_mask.shape != target_shape:
        gt_mask = resize(
            gt_mask,
            output_shape=target_shape,
            order=0,
            mode="edge",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)

    return gt_mask


@dataclass
class SampleMetrics:
    sample_name: str
    sp_roc_auc: float
    sp_ap: float
    pixel_roc_auc: float
    pixel_ap: float


def evaluate_sample(
    sample_name: str,
    args: argparse.Namespace,
    embedder: ResNetPixelEmbedder,
    base_mask: np.ndarray,
    device: torch.device,
) -> SampleMetrics:
    sample_str = extract_sample_id(sample_name)
    print(f"\n[INFO] Running sample: {sample_name}")

    label_image = load_label_image(args.results_dir, sample_name, args.test_origin, args.label_patch)
    recon_images = load_recon_images(args.results_dir, sample_name, args.test_origin, args.patches)
    if recon_images.shape[1:3] != base_mask.shape:
        raise ValueError(
            f"Mask shape {base_mask.shape} and recon shape {recon_images.shape[1:3]} differ for {sample_name}"
        )

    labels_fine, _, _ = recursive_subdivide(
        img=label_image,
        labels=base_mask,
        var_threshold=args.var_threshold,
        min_pixels=args.split_min_pixels,
        max_sub=args.max_sub,
        max_depth=args.max_depth,
        compactness=args.compactness,
        alpha_grad=args.alpha_grad,
        target_size=args.target_size,
    )

    feat_map = compute_recon_features(embedder, recon_images, batch_size=args.batch_size, device=device)

    with torch.no_grad():
        label_x = to_tensor01(label_image, device=device)
        label_feat = embedder(label_x).squeeze(0).cpu()

    mu_pca, components_pca = compute_pca_basis(label_feat, n_components=args.n_pca)
    ch, h, w = label_feat.shape
    x_label = label_feat.permute(1, 2, 0).reshape(-1, ch).numpy()
    label_pca_map = project_to_pca(x_label, mu_pca, components_pca).reshape(h, w, args.n_pca)

    bsz = feat_map.shape[0]
    feat_np = feat_map.permute(0, 2, 3, 1).numpy()
    feat_flat = feat_np.reshape(bsz, -1, ch)
    feat_centered = feat_flat - mu_pca[None]
    proj_flat = feat_centered @ components_pca.T
    pca_feats_recon = np.clip(proj_flat, -1.0, 1.0).reshape(bsz, h, w, args.n_pca)

    delta_map, _, _ = compute_typical_set_with_pca(
        labels_fine=labels_fine,
        recon_images=recon_images,
        pca_feats_recon=pca_feats_recon,
        bins_rgb=args.bins_rgb,
        bins_pca=args.bins_pca,
        smooth_sigma=args.smooth_sigma,
        min_pixels=args.min_pixels,
        label_image=label_image,
        label_pca_map=label_pca_map,
    )

    gt_mask_binary = load_gt_mask(
        gt_root=args.gt_root,
        defect_type=args.defect_type,
        sample_str=sample_str,
        target_shape=delta_map.shape,
        downsample=args.gt_downsample,
    )

    all_sp_ids = sorted(np.unique(labels_fine).astype(int).tolist())
    sp_scores = []
    sp_labels = []
    for sp_id in all_sp_ids:
        sp_mask = labels_fine == sp_id
        if int(sp_mask.sum()) < 1:
            continue
        score = float(np.nanmean(delta_map[sp_mask]))
        if np.isnan(score):
            continue
        frac_anomalous = gt_mask_binary[sp_mask].mean()
        label = int(frac_anomalous > 0.5)
        sp_scores.append(score)
        sp_labels.append(label)

    sp_scores = np.array(sp_scores, dtype=np.float64)
    sp_labels = np.array(sp_labels, dtype=np.int32)
    if sp_scores.size == 0:
        raise RuntimeError(f"No superpixel scores computed for {sample_name}")

    fpr, tpr, _ = manual_roc_curve(sp_labels, sp_scores)
    sp_roc_auc = manual_auc(fpr, tpr)
    sp_ap = manual_average_precision(sp_labels, sp_scores)

    valid_mask = ~np.isnan(delta_map) if np.isnan(delta_map).any() else np.ones_like(delta_map, dtype=bool)
    delta_map_smooth = gaussian_filter(delta_map, sigma=5, mode="nearest")
    pixel_scores = delta_map_smooth[valid_mask].ravel()
    pixel_labels = gt_mask_binary[valid_mask].ravel()

    fpr_px, tpr_px, _ = manual_roc_curve(pixel_labels, pixel_scores)
    pixel_roc_auc = manual_auc(fpr_px, tpr_px)
    pixel_ap = manual_average_precision(pixel_labels, pixel_scores)

    print(
        f"[RESULT] {sample_name} | "
        f"SP ROC-AUC={sp_roc_auc:.4f}, SP AP={sp_ap:.4f}, "
        f"Pixel ROC-AUC={pixel_roc_auc:.4f}, Pixel AP={pixel_ap:.4f}"
    )

    return SampleMetrics(
        sample_name=sample_name,
        sp_roc_auc=sp_roc_auc,
        sp_ap=sp_ap,
        pixel_roc_auc=pixel_roc_auc,
        pixel_ap=pixel_ap,
    )


def main() -> None:
    args = parse_args()
    apply_category_presets(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    layers = tuple([x.strip() for x in args.layers.split(",") if x.strip()])
    proj_dim = args.proj_dim if args.proj_dim > 0 else None
    sample_names = resolve_sample_names(args)

    pca_hist_size = args.bins_pca ** args.n_pca
    if pca_hist_size > 50_000_000:
        raise ValueError(
            f"bins_pca**n_pca={pca_hist_size} is too large and will be very slow. "
            "Lower --n-pca or --bins-pca."
        )

    print(f"[CONFIG] device={device}")
    print(f"[CONFIG] category={args.category}")
    print(f"[CONFIG] resnet_model={args.resnet_model}, layers={layers}, n_pca={args.n_pca}")
    if args.category == "faces":
        print(f"[CONFIG] sample_count={len(sample_names)}")
    else:
        print(f"[CONFIG] sample_range={args.sample_start:03d}..{args.sample_end:03d}")

    mask = load_mask(args.figures_dir)
    embedder = ResNetPixelEmbedder(
        resnet_name=args.resnet_model,
        layers=layers,
        out_size=None,
        use_imagenet_norm=True,
        use_patch_context=args.use_patch_context,
        proj_dim_per_layer=proj_dim,
    ).to(device).eval()

    results: List[SampleMetrics] = []
    failed_samples: List[str] = []

    for sample_name in sample_names:
        try:
            out = evaluate_sample(sample_name, args, embedder, mask, device)
            results.append(out)
        except Exception as exc:
            failed_samples.append(sample_name)
            print(f"[WARN] Failed {sample_name}: {exc}")
            if args.strict:
                raise

    if not results:
        raise RuntimeError("No samples were evaluated successfully.")

    sp_roc_avg = float(np.mean([m.sp_roc_auc for m in results]))
    sp_ap_avg = float(np.mean([m.sp_ap for m in results]))
    px_roc_avg = float(np.mean([m.pixel_roc_auc for m in results]))
    px_ap_avg = float(np.mean([m.pixel_ap for m in results]))

    print("\n" + "=" * 60)
    print(f"Successful samples: {len(results)}")
    print(f"Failed samples: {len(failed_samples)}")
    if failed_samples:
        print("Failed list:", ", ".join(failed_samples))
    print("=" * 60)
    print(f"AVG SP ROC-AUC:    {sp_roc_avg:.6f}")
    print(f"AVG SP AP:         {sp_ap_avg:.6f}")
    print(f"AVG Pixel ROC-AUC: {px_roc_avg:.6f}")
    print(f"AVG Pixel AP:      {px_ap_avg:.6f}")
    print("=" * 60)

    report_path = resolve_report_path(args)
    save_text_report(
        report_path=report_path,
        args=args,
        device=device,
        layers=layers,
        results=results,
        failed_samples=failed_samples,
        sp_roc_avg=sp_roc_avg,
        sp_ap_avg=sp_ap_avg,
        px_roc_avg=px_roc_avg,
        px_ap_avg=px_ap_avg,
    )
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
