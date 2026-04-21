"""Tests for ood/superpixels.py — recursive SLIC refinement."""
import numpy as np

from ood.superpixels import recursive_subdivide


def test_recursive_subdivide_returns_expected_outputs():
    # Small colorful synthetic image
    rng = np.random.default_rng(0)
    H, W = 64, 64
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    # Initial labels: 4 quadrants
    labels = np.zeros((H, W), dtype=np.int32)
    labels[:H//2, :W//2] = 0
    labels[:H//2, W//2:] = 1
    labels[H//2:, :W//2] = 2
    labels[H//2:, W//2:] = 3

    labels_fine, final_ids, parent_map = recursive_subdivide(
        img=img, labels=labels,
        var_threshold=0.0, min_pixels=20,
        max_sub=4, max_depth=2,
        compactness=10.0, alpha_grad=1.0,
        target_size=10,
    )
    # Outputs
    assert labels_fine.shape == (H, W)
    assert labels_fine.dtype in (np.int32, np.int64)
    # Refined has at least as many SPs as input
    assert len(final_ids) >= 4
    # parent_map: for every final_id, there's an entry in some child list
    all_children = set()
    for orig, children in parent_map.items():
        all_children.update(children)
    for sid in final_ids:
        assert sid in all_children


def test_recursive_subdivide_no_split_with_large_min_pixels():
    """With min_pixels much larger than region size, splitting should not occur."""
    H, W = 32, 32
    img = np.full((H, W, 3), 100, dtype=np.uint8)  # constant image
    labels = np.zeros((H, W), dtype=np.int32)
    labels[:, :W//2] = 0
    labels[:, W//2:] = 1

    labels_fine, final_ids, parent_map = recursive_subdivide(
        img=img, labels=labels,
        var_threshold=0.0, min_pixels=10_000,  # impossible to split
        max_sub=4, max_depth=2,
        compactness=10.0, alpha_grad=1.0,
        target_size=10,
    )
    # Should return the 2 original SPs unchanged
    assert len(final_ids) == 2
