
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Literal

import numpy as np
from skimage.segmentation import slic, find_boundaries
from skimage.util import img_as_float
from skimage.transform import resize

Mode = Literal["ids", "random", "topk_from_score", "threshold_from_score"]


@dataclass
class SuperpixelMaskResult:
    labels: np.ndarray          # (H,W) int superpixel id per pixel
    mask: np.ndarray            # (H,W) uint8 {0,1} where 1 = masked/selected region
    masked_image: np.ndarray    # same shape as input (H,W,3) or (H,W)
    chosen_ids: np.ndarray      # (K,) chosen superpixel ids


def kmeans_simple(X: np.ndarray, n_clusters: int = 2, max_iter: int = 100, random_state: int = 0) -> np.ndarray:
    """
    Simple K-Means clustering implementation (no sklearn needed).
    
    Args:
        X: (N, D) array of features
        n_clusters: number of clusters
        max_iter: maximum iterations
        random_state: random seed
    
    Returns:
        labels: (N,) cluster assignments
    """
    np.random.seed(random_state)
    N, D = X.shape
    
    # Initialize centroids randomly from data points
    indices = np.random.choice(N, size=n_clusters, replace=False)
    centroids = X[indices].copy()
    
    labels = np.zeros(N, dtype=np.int32)
    
    for _ in range(max_iter):
        # Assign each point to nearest centroid
        distances = np.zeros((N, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        
        # Check convergence
        if np.all(new_labels == labels):
            break
        
        labels = new_labels
        
        # Update centroids
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(axis=0)
    
    return labels


def compute_superpixels_slic(
    img: np.ndarray,
    n_segments: int = 300,
    compactness: float = 10.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    SLIC superpixels. Returns labels (H,W) in {0..K-1}.
    """
    if img.ndim not in (2, 3):
        raise ValueError("img must be (H,W) or (H,W,3)")

    img_f = img_as_float(img)  # float in [0,1] if possible
    labels = slic(
        img_f,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1 if img.ndim == 3 else None,
    )
    return labels.astype(np.int32)

def subdivide_superpixels_into_two(
    img: np.ndarray,
    labels: np.ndarray,
    compactness: float = 10.0,
    sigma: float = 1.0,
    min_pixels_to_subdivide: int = 30
) -> np.ndarray:
    """
    Subdivide each superpixel into exactly 2 sub-superpixels using SLIC.
    
    Args:
        img: (H, W, C) or (H, W) image array
        labels: (H, W) superpixel labels
        compactness: SLIC compactness parameter
        sigma: SLIC sigma parameter
        min_pixels_to_subdivide: Minimum pixels required to subdivide
    
    Returns:
        new_labels: (H, W) array with subdivided labels (spatially continuous)
    """
    H, W = labels.shape
    K = int(labels.max()) + 1
    
    # New labels array
    new_labels = np.zeros((H, W), dtype=np.int32)
    next_label_id = 0
    
    # Process each superpixel
    for sp_id in range(K):
        # Create mask for this superpixel
        mask_2d = (labels == sp_id)
        n_pixels = mask_2d.sum()
        
        if n_pixels == 0:
            continue
        
        # Skip subdivision if too small
        if n_pixels < min_pixels_to_subdivide:
            new_labels[mask_2d] = next_label_id
            next_label_id += 1
            continue
        
        # Extract bounding box of this superpixel
        rows, cols = np.where(mask_2d)
        r_min, r_max = rows.min(), rows.max() + 1
        c_min, c_max = cols.min(), cols.max() + 1
        
        # Extract the region
        img_region = img[r_min:r_max, c_min:c_max]
        mask_region = mask_2d[r_min:r_max, c_min:c_max]
        
        # Apply SLIC on this region with n_segments=2
        img_f = img_as_float(img_region)
        try:
            sub_labels = slic(
                img_f,
                n_segments=2,
                compactness=compactness,
                sigma=sigma,
                start_label=0,
                mask=mask_region,  # Only segment within superpixel
                channel_axis=-1 if img.ndim == 3 else None,
            )
        except:
            # Fallback if SLIC fails (e.g., region too small)
            sub_labels = np.zeros_like(mask_region, dtype=np.int32)
            # Simple split: left half = 0, right half = 1
            mid_col = mask_region.shape[1] // 2
            sub_labels[:, mid_col:] = 1
        
        # Only keep labels within the original superpixel mask
        sub_labels[~mask_region] = -1
        
        # Assign new global labels
        for sub_id in range(2):
            sub_mask_region = (sub_labels == sub_id) & mask_region
            if sub_mask_region.sum() > 0:
                new_labels[r_min:r_max, c_min:c_max][sub_mask_region] = next_label_id + sub_id
        
        next_label_id += 2
    
    return new_labels

def superpixel_scores_from_map(labels: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    """
    Aggregate per-pixel score_map into per-superpixel mean score.
    labels: (H,W), score_map: (H,W)
    returns: (K,)
    """
    if score_map.shape != labels.shape:
        raise ValueError(f"score_map shape {score_map.shape} must match labels {labels.shape}")

    K = int(labels.max()) + 1
    flat_lab = labels.reshape(-1)
    flat_score = score_map.reshape(-1).astype(np.float64)

    sums = np.bincount(flat_lab, weights=flat_score, minlength=K)
    counts = np.bincount(flat_lab, minlength=K).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    return sums / counts


def mask_by_superpixels(
    img: np.ndarray,
    n_segments: int = 300,
    compactness: float = 10.0,
    sigma: float = 1.0,
    mode: Mode = "random",
    ids: Optional[Iterable[int]] = None,
    random_frac: float = 0.10,
    seed: int = 0,
    score_map: Optional[np.ndarray] = None,
    topk: int = 20,
    threshold: Optional[float] = None,
    invert: bool = False,
    mask_value: float | int = 0,
) -> SuperpixelMaskResult:
    """
    Build a binary mask from superpixels.

    Modes:
      - "ids": choose superpixels in `ids`
      - "random": choose random_frac of superpixels
      - "topk_from_score": choose topk superpixels by mean(score_map)
      - "threshold_from_score": choose superpixels where mean(score_map) >= threshold

    invert=True flips mask (1<->0). Useful if you want to KEEP chosen regions
    and mask everything else.

    mask_value: what to write in masked pixels in the output image.
               For RGB images you can pass scalar (broadcast) or a length-3 array.
    """
    labels = compute_superpixels_slic(img, n_segments=n_segments,
                                     compactness=compactness, sigma=sigma)
    K = int(labels.max()) + 1

    if mode == "ids":
        if ids is None:
            raise ValueError("mode='ids' requires ids")
        chosen = np.array(list(ids), dtype=np.int32)

    elif mode == "random":
        rng = np.random.default_rng(seed)
        m = max(1, int(round(random_frac * K)))
        chosen = rng.choice(K, size=m, replace=False).astype(np.int32)

    elif mode in ("topk_from_score", "threshold_from_score"):
        if score_map is None:
            raise ValueError(f"mode='{mode}' requires score_map (H,W)")
        sp_scores = superpixel_scores_from_map(labels, score_map)

        if mode == "topk_from_score":
            k = int(min(max(topk, 1), K))
            chosen = np.argsort(-sp_scores)[:k].astype(np.int32)
        else:
            if threshold is None:
                raise ValueError("mode='threshold_from_score' requires threshold")
            chosen = np.where(sp_scores >= float(threshold))[0].astype(np.int32)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    mask = np.isin(labels, chosen).astype(np.uint8)  # 1 on chosen superpixels
    if invert:
        mask = (1 - mask).astype(np.uint8)

    masked = np.array(img, copy=True)
    if masked.ndim == 2:
        masked[mask.astype(bool)] = mask_value
    else:
        # RGB: mask pixels across channel axis
        masked[mask.astype(bool), :] = mask_value

    return SuperpixelMaskResult(
        labels=labels,
        mask=mask,
        masked_image=masked,
        chosen_ids=chosen,
    )


def boundaries_overlay(img: np.ndarray, labels: np.ndarray, boundary_value: float | int = 1.0) -> np.ndarray:
    """
    Optional: return an image with superpixel boundaries highlighted.
    No cv2; uses find_boundaries. For RGB, boundary pixels set to boundary_value.
    """
    b = find_boundaries(labels, mode="thick")
    out = img_as_float(img).copy()

    if out.ndim == 2:
        out[b] = boundary_value
    else:
        out[b, :] = boundary_value
    return out


# if __name__ == "__main__":
#     # Demo with a local image using skimage.io (no cv2)
#     from skimage import io, color, filters

#     img = io.imread("/home/akheirandish3/diffusion-posterior-sampling/data/samples/00243.png")  # RGB or RGBA
#     if img.ndim == 3 and img.shape[-1] == 4:
#         img = img[..., :3]  # drop alpha

#     # Example score map: Sobel magnitude (as a placeholder for your OOD map)
#     gray = color.rgb2gray(img) if img.ndim == 3 else img.astype(np.float32)
#     score = np.abs(filters.sobel(gray))  # (H,W)

#     # Step 1: Create initial superpixels
#     res = mask_by_superpixels(
#         img,
#         n_segments=50,
#         compactness=10,
#         mode="topk_from_score",
#         score_map=score,
#         topk=20,
#         invert=False,     # set True to keep only top regions and mask others
#         mask_value=0,
#     )
    
#     # Step 2: Subdivide each superpixel into 2 sub-superpixels
#     # subdivided_labels = subdivide_superpixels_into_two(img, res.labels, use_spatial=True, spatial_weight=0.3)
#     # Same as before, but now with spatial continuity!
#     subdivided_labels = subdivide_superpixels_into_two(
#         img, 
#         res.labels, 
#         compactness=10,        # Controls shape regularity
#         sigma=1.0,            # Gaussian smoothing
#         min_pixels_to_subdivide=30
#     )
    
#     print(f"Original superpixels: {res.labels.max() + 1}")
#     print(f"Subdivided superpixels: {subdivided_labels.max() + 1}")

#     io.imsave("/home/akheirandish3/diffusion-posterior-sampling/data/mask.png", (res.labels).astype(np.uint8))
#     io.imsave("/home/akheirandish3/diffusion-posterior-sampling/data/mask_subdivided.png", (subdivided_labels % 256).astype(np.uint8))

#     vis = boundaries_overlay(img, res.labels, boundary_value=255 if img.dtype == np.uint8 else 1.0)
#     io.imsave("/home/akheirandish3/diffusion-posterior-sampling/data/superpixels.png", (vis * 255).astype(np.uint8) if vis.dtype != np.uint8 else vis)
    
#     vis_sub = boundaries_overlay(img, subdivided_labels, boundary_value=255 if img.dtype == np.uint8 else 1.0)
#     io.imsave("/home/akheirandish3/diffusion-posterior-sampling/data/superpixels_subdivided.png", (vis_sub * 255).astype(np.uint8) if vis_sub.dtype != np.uint8 else vis_sub)

#     print(f"Saved: mask.png, superpixels.png, mask_subdivided.png, superpixels_subdivided.png")

if __name__ == "__main__":
    import argparse
    from skimage import io, color, filters

    parser = argparse.ArgumentParser(description="Generate superpixel masks from image")
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--n_segments', type=int, default=50,
                        help='Number of superpixels')
    parser.add_argument('--compactness', type=float, default=10.0,
                        help='SLIC compactness parameter')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Gaussian smoothing sigma')
    parser.add_argument('--topk', type=int, default=20,
                        help='Top-k superpixels to select')
    parser.add_argument('--output_dir', type=str, default="/home/akheirandish3/diffusion-posterior-sampling/data",
                        help='Output directory for masks')
    
    args = parser.parse_args()

    # Load image
    img = io.imread(args.input_image)
    print(f"Original shape: {img.shape}")

    # Resize to 256x256
    img_resized = resize(img, (256, 256), anti_aliasing=True)

    # Convert back to uint8 if needed
    img_resized = (img_resized * 255).astype(np.uint8)

    # Save
    io.imsave(args.input_image, img_resized)
    img = io.imread(args.input_image)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # drop alpha

    # Example score map: Sobel magnitude (as a placeholder for your OOD map)
    gray = color.rgb2gray(img) if img.ndim == 3 else img.astype(np.float32)
    score = np.abs(filters.sobel(gray))  # (H,W)

    # Step 1: Create initial superpixels
    res = mask_by_superpixels(
        img,
        n_segments=args.n_segments,
        compactness=args.compactness,
        mode="topk_from_score",
        score_map=score,
        topk=args.topk,
        invert=False,
        mask_value=0,
    )
    
    # Step 2: Subdivide each superpixel into 2 sub-superpixels
    # subdivided_labels = subdivide_superpixels_into_two(
    #     img, 
    #     res.labels, 
    #     compactness=args.compactness,
    #     sigma=args.sigma,
    #     min_pixels_to_subdivide=30
    # )
    
    print(f"Original superpixels: {res.labels.max() + 1}")
    # print(f"Subdivided superpixels: {subdivided_labels.max() + 1}")

    io.imsave(f"{args.output_dir}/mask.png", (res.labels).astype(np.uint8))
    # io.imsave(f"{args.output_dir}/mask_subdivided.png", (subdivided_labels % 256).astype(np.uint8))

    vis = boundaries_overlay(img, res.labels, boundary_value=255 if img.dtype == np.uint8 else 1.0)
    io.imsave(f"{args.output_dir}/superpixels.png", (vis * 255).astype(np.uint8) if vis.dtype != np.uint8 else vis)
    
    # vis_sub = boundaries_overlay(img, subdivided_labels, boundary_value=255 if img.dtype == np.uint8 else 1.0)
    # io.imsave(f"{args.output_dir}/superpixels_subdivided.png", (vis_sub * 255).astype(np.uint8) if vis_sub.dtype != np.uint8 else vis_sub)

    print(f"Saved to: {args.output_dir}")
