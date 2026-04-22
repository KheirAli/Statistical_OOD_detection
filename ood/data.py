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
) -> np.ndarray:
    """Load the label (original) image from DPS output.

    Searches patch indices 0..49 for the first available label file. Label
    files may be named `0_00000.png` (batched) or `00000.png` (single).

    Returns:
        (H, W, 3) uint8 array.
    """
    label_names = ["0_00000.png", "00000.png"]
    for idx in range(50):
        for lname in label_names:
            candidate = os.path.join(
                results_dir, sample_name,
                f"{test_origin}_{idx}_{bottom_suffix}",
                "inpainting", "label", lname,
            )
            if os.path.exists(candidate):
                img = io.imread(candidate)
                if img.ndim == 3 and img.shape[-1] == 4:
                    img = img[..., :3]
                return img[:, :, :3]
    raise FileNotFoundError(
        f"No label image found in {results_dir}/{sample_name}/ for any patch index"
    )


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


def parse_sample_id(sample_name: str) -> str:
    """Extract sample ID from sample_name like 'samples_010' -> '010'."""
    parts = sample_name.split("_")[1:]
    return "_".join(parts)
