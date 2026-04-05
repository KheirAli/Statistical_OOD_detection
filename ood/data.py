"""Data loading: DPS reconstructions, label images, masks.

Extracted from New_dataset_clean.ipynb Cells 0-4.
"""

import os
from glob import glob
from typing import Optional

import numpy as np
from skimage import io
import PIL.Image as Image


def load_reconstructions(
    results_dir: str,
    sample_name: str,
    test_origin: str,
    num_patches: int,
    bottom_suffix: str,
) -> np.ndarray:
    """Load all DPS reconstruction images across patches.

    Scans ``{results_dir}/{sample_name}/{test_origin}_{patch}_{bottom_suffix}/inpainting/recon/*.png``
    for each patch index 0..num_patches-1, stacks into a single array.

    Returns:
        (B, H, W, 3) uint8 array where B = total images across all patches.
    """
    all_images = []

    for patch_idx in range(num_patches):
        directory = os.path.join(
            results_dir, sample_name,
            f"{test_origin}_{patch_idx}_{bottom_suffix}",
            "inpainting", "recon",
        )
        image_paths = sorted(glob(os.path.join(directory, "*.png")))

        for path in image_paths:
            img = io.imread(path)
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            all_images.append(img)

        if not image_paths:
            print(f"Warning: no images in {directory}")

    if not all_images:
        raise FileNotFoundError(
            f"No reconstruction images found in {results_dir}/{sample_name}/"
        )

    # Verify consistent shape
    shapes = {img.shape for img in all_images}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent image shapes: {shapes}")

    return np.array(all_images)


def load_label_image(
    results_dir: str,
    sample_name: str,
    test_origin: str,
    bottom_suffix: str,
    label_patch_idx: int = 8,
) -> np.ndarray:
    """Load the label (original) image from DPS output.

    Tries the preferred patch index first, then falls back to any available patch.

    Returns:
        (H, W, 3) uint8 array.
    """
    # Try preferred patch, then fall back to any patch that has label output
    # Label files may be named 0_00000.png (batched) or 00000.png (single)
    label_names = ["0_00000.png", "00000.png"]
    path = None
    for idx in [label_patch_idx] + list(range(50)):
        for lname in label_names:
            candidate = os.path.join(
                results_dir, sample_name,
                f"{test_origin}_{idx}_{bottom_suffix}",
                "inpainting", "label", lname,
            )
            if os.path.exists(candidate):
                path = candidate
                break
        if path:
            break
    if path is None:
        raise FileNotFoundError(
            f"No label image found in {results_dir}/{sample_name}/ for any patch index"
        )

    img = io.imread(path)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img[:, :, :3]


def load_superpixel_mask(figures_dir: str) -> np.ndarray:
    """Load the superpixel label mask from figures/mask.png.

    Returns:
        (H, W) int array of superpixel IDs.
    """
    path = os.path.join(figures_dir, "mask.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Superpixel mask not found: {path}")
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    # Use 16-bit read if available, to support superpixel IDs > 255
    if mask.dtype == np.uint8 and mask.max() == 255:
        mask_16 = np.array(Image.open(path).convert("I"))
        if mask_16.max() > 255:
            mask = mask_16
    return mask.astype(np.int32)


def load_gt_mask(
    path_template: str,
    sample: str,
    downsample_factor: int = 4,
) -> np.ndarray:
    """Load ground-truth anomaly mask and binarize.

    Args:
        path_template: path with ``{sample}`` placeholder.
        sample: sample identifier to substitute.
        downsample_factor: spatial downsampling factor.

    Returns:
        (H, W) uint8 binary mask (1 = anomaly).
    """
    path = path_template.format(sample=sample)
    if not os.path.exists(path):
        raise FileNotFoundError(f"GT mask not found: {path}")

    mask = np.array(Image.open(path).convert("L"))
    if downsample_factor > 1:
        mask = mask[::downsample_factor, ::downsample_factor]

    return (mask > 0).astype(np.uint8)


def load_gt_mask_flexible(
    gt_root: str,
    sample_str: str,
    target_shape: tuple,
    downsample_factor: int = 4,
    defect_type: str = "combined",
) -> np.ndarray:
    """Load GT mask with flexible path resolution.

    Searches for ``{sample_str}_mask.png`` directly in gt_root and under
    ``{gt_root}/{defect_type}/``.  Strips leading zeros from sample_str
    when the raw ID is longer than 4 digits.  Resizes to target_shape if
    the loaded mask doesn't match after downsampling.

    Returns:
        (H, W) uint8 binary mask (1 = anomaly).
    """
    from skimage.transform import resize as sk_resize

    if len(sample_str) > 4:
        sample_str = str(int(sample_str))

    candidates = [
        os.path.join(gt_root, f"{sample_str}_mask.png"),
        os.path.join(gt_root, defect_type, f"{sample_str}_mask.png"),
    ]
    gt_path = None
    for c in candidates:
        if os.path.exists(c):
            gt_path = c
            break
    if gt_path is None:
        raise FileNotFoundError(
            f"GT mask not found at any of: {candidates}"
        )

    mask = np.array(Image.open(gt_path).convert("L"))
    if downsample_factor > 1:
        mask = mask[::downsample_factor, ::downsample_factor]
    gt_mask = (mask > 0).astype(np.uint8)

    if gt_mask.shape != target_shape:
        gt_mask = sk_resize(
            gt_mask,
            output_shape=target_shape,
            order=0,
            mode="edge",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)

    return gt_mask


def parse_sample_id(sample_name: str) -> str:
    """Extract sample ID from sample_name like 'samples_010' -> '010'."""
    parts = sample_name.split("_")[1:]
    return "_".join(parts)
