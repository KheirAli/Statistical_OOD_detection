"""Tests for tools/prepare_xray_dataset.py — SIXray → MVTec ingestion."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
INGEST = ROOT / "tools" / "prepare_xray_dataset.py"


def _make_fake_sixray(root: Path, n_neg=5, n_pos=5):
    """Build a minimal SIXray-layout fixture."""
    (root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (root / "annotation").mkdir(parents=True, exist_ok=True)
    (root / "ImageSets").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    # Fake images at non-square size to exercise resizing
    W, H = 512, 384
    neg_stems = [f"N{i}" for i in range(n_neg)]
    pos_stems = [f"P{i}" for i in range(n_pos)]
    for stem in neg_stems + pos_stems:
        arr = (rng.random((H, W)) * 200).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(root / "JPEGImages" / f"{stem}.jpg")

    # VOC annotations for positives only (one bbox each, at (100,100)-(200,200))
    for stem in pos_stems:
        (root / "annotation" / f"{stem}.xml").write_text(
            "<annotation>"
            f"<size><width>{W}</width><height>{H}</height></size>"
            "<object><bndbox>"
            "<xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>200</ymax>"
            "</bndbox></object>"
            "</annotation>"
        )

    (root / "ImageSets" / "negative.txt").write_text("\n".join(neg_stems) + "\n")
    (root / "ImageSets" / "positive.txt").write_text("\n".join(pos_stems) + "\n")


def test_ingestion_produces_mvtec_layout(tmp_path):
    sixray = tmp_path / "SIXray_fake"
    _make_fake_sixray(sixray, n_neg=5, n_pos=4)
    out = tmp_path / "xray_mvtec"

    subprocess.run(
        [sys.executable, str(INGEST),
         "--sixray_root", str(sixray),
         "--output_root", str(out),
         "--n_train", "3",
         "--n_test_clean", "1",
         "--n_test", "3",
         "--image_size", "256"],
        check=True,
    )

    # Directory layout
    assert (out / "train" / "good").is_dir()
    assert (out / "test" / "good").is_dir()
    assert (out / "test" / "prohibited").is_dir()
    assert (out / "ground_truth" / "prohibited").is_dir()
    # Counts
    assert len(list((out / "train" / "good").glob("*.png"))) == 3
    assert len(list((out / "test" / "good").glob("*.png"))) == 1
    assert len(list((out / "test" / "prohibited").glob("*.png"))) == 3
    assert len(list((out / "ground_truth" / "prohibited").glob("*_mask.png"))) == 3
    # Manifest exists
    assert (out / "manifest.txt").is_file()


def test_ingestion_output_image_format(tmp_path):
    sixray = tmp_path / "SIXray_fake"
    _make_fake_sixray(sixray, n_neg=2, n_pos=2)
    out = tmp_path / "xray_mvtec"
    subprocess.run(
        [sys.executable, str(INGEST),
         "--sixray_root", str(sixray),
         "--output_root", str(out),
         "--n_train", "1", "--n_test_clean", "0", "--n_test", "1",
         "--image_size", "256"],
        check=True,
    )
    # Images: 256x256 RGB with replicated channels (grayscale stored as 3-channel)
    img_path = next((out / "train" / "good").glob("*.png"))
    img = Image.open(img_path)
    assert img.size == (256, 256)
    assert img.mode == "RGB"
    arr = np.array(img)
    assert arr.shape == (256, 256, 3)
    # All 3 channels should be identical (replicated grayscale)
    assert np.array_equal(arr[:, :, 0], arr[:, :, 1])
    assert np.array_equal(arr[:, :, 1], arr[:, :, 2])


def test_ingestion_bbox_to_mask(tmp_path):
    sixray = tmp_path / "SIXray_fake"
    _make_fake_sixray(sixray, n_neg=1, n_pos=1)
    out = tmp_path / "xray_mvtec"
    subprocess.run(
        [sys.executable, str(INGEST),
         "--sixray_root", str(sixray),
         "--output_root", str(out),
         "--n_train", "1", "--n_test_clean", "0", "--n_test", "1",
         "--image_size", "256"],
        check=True,
    )
    mask_path = next((out / "ground_truth" / "prohibited").glob("*_mask.png"))
    mask = np.array(Image.open(mask_path))
    assert mask.shape == (256, 256)
    # Binary mask
    assert set(np.unique(mask).tolist()).issubset({0, 255})
    # Original bbox (100,100)-(200,200) in a 512x384 image → scaled to ~(50,66)-(100,133) in 256x256
    # Simplest assertion: at least SOME pixels are on and SOME are off.
    assert mask.sum() > 0, "mask should have at least some anomaly pixels"
    assert (mask == 0).sum() > 0, "mask should have at least some background pixels"


def test_ingestion_handles_missing_annotation(tmp_path):
    """If a positive image has no VOC annotation, the script should still succeed
    (writing a blank mask) rather than crashing."""
    sixray = tmp_path / "SIXray_fake"
    _make_fake_sixray(sixray, n_neg=1, n_pos=2)
    # Remove one annotation
    (sixray / "annotation" / "P0.xml").unlink()
    out = tmp_path / "xray_mvtec"
    result = subprocess.run(
        [sys.executable, str(INGEST),
         "--sixray_root", str(sixray),
         "--output_root", str(out),
         "--n_train", "1", "--n_test_clean", "0", "--n_test", "2",
         "--image_size", "128"],
        check=True, capture_output=True, text=True,
    )
    # Warning is printed, but the run succeeds
    assert "missing" in result.stdout.lower() or "had no VOC" in result.stdout
    # Both mask files exist (one blank)
    assert len(list((out / "ground_truth" / "prohibited").glob("*_mask.png"))) == 2
