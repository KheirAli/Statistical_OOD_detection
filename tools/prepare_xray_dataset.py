"""Convert SIXray (or a similarly-structured X-ray dataset) into our MVTec-compatible layout.

Input layout (SIXray-ish — adjust assumptions via CLI):
  <sixray_root>/
    annotation/        *.xml  (Pascal-VOC-style, one per positive image)
    JPEGImages/        <stem>.jpg
    ImageSets/
      positive.txt     list of positive image stems (one per line)
      negative.txt     list of negative image stems

Output layout (MVTec-compatible, ingested by our pipeline):
  <output_root>/
    train/good/                      N clean X-rays at target size, 3-ch
    test/
      good/                          held-out clean X-rays
      prohibited/                    positive X-rays
    ground_truth/prohibited/         bbox-derived binary masks, {0, 255}

All images resized to --image_size × --image_size. Grayscale is replicated to 3
channels so the DDAD UNet (in_channels=3) works unchanged.

Usage:
  python tools/prepare_xray_dataset.py \\
      --sixray_root /data/akheirandish3/SIXray_raw \\
      --output_root /data/akheirandish3/xray_mvtec \\
      --n_train 10000 --n_test 500 --image_size 256
"""
import argparse
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def _find_file(stem: str, candidates: List[Path]) -> Path:
    """Find <stem>.{png,jpg,jpeg,bmp,tif} under any of candidates."""
    for root in candidates:
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
                    ".PNG", ".JPG", ".JPEG"):
            p = root / f"{stem}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"image for stem {stem} not under {candidates}")


def _load_stems(list_path: Path) -> List[str]:
    """Read one image stem per line; strip extension if present."""
    with open(list_path) as f:
        stems = []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Strip a trailing extension if the list file contains full filenames
            s = Path(s).stem
            stems.append(s)
    return stems


def _parse_voc_bboxes(xml_path: Path) -> Tuple[Tuple[int, int], List[Tuple[int, int, int, int]]]:
    """Parse a Pascal-VOC annotation. Returns (image_size_wh, [bbox (xmin,ymin,xmax,ymax), ...])."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        boxes.append((xmin, ymin, xmax, ymax))
    return (w, h), boxes


def _bboxes_to_mask(img_size_wh: Tuple[int, int],
                    bboxes: List[Tuple[int, int, int, int]],
                    out_size: int) -> np.ndarray:
    """Render bboxes into a binary mask at out_size × out_size."""
    orig_w, orig_h = img_size_wh
    m = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for xmin, ymin, xmax, ymax in bboxes:
        m[ymin:ymax, xmin:xmax] = 255
    # Nearest-neighbor resize to preserve binary-ness
    return np.array(
        Image.fromarray(m).resize((out_size, out_size), Image.NEAREST),
        dtype=np.uint8,
    )


def _resize_to_3ch(src: Path, out: Path, size: int):
    """Load image, resize, force 3-channel (replicate grayscale), write PNG."""
    img = Image.open(src).convert("L")  # force grayscale first — X-rays may be saved as RGB but are really gray
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img)
    rgb = np.stack([arr, arr, arr], axis=-1)   # replicate → (H, W, 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out)


def _save_mask(mask: np.ndarray, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sixray_root", required=True,
                   help="Root of SIXray download (contains annotation/, JPEGImages/, ImageSets/ or similar)")
    p.add_argument("--output_root", required=True,
                   help="Where to write the MVTec-layout dataset")
    p.add_argument("--n_train", type=int, default=10_000,
                   help="# of negative images for training")
    p.add_argument("--n_test_clean", type=int, default=50,
                   help="# of held-out clean images for image-AUROC")
    p.add_argument("--n_test", type=int, default=500,
                   help="# of positive images for anomaly eval")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--annotation_dir", default="annotation",
                   help="Subdir of sixray_root with VOC XMLs (per-positive bboxes)")
    p.add_argument("--images_dir", default="JPEGImages",
                   help="Subdir of sixray_root holding the image files")
    p.add_argument("--positive_list", default="ImageSets/positive.txt",
                   help="Relative path to list of positive image stems")
    p.add_argument("--negative_list", default="ImageSets/negative.txt",
                   help="Relative path to list of negative image stems")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)

    root = Path(args.sixray_root)
    out = Path(args.output_root)
    img_candidates = [root / args.images_dir, root / "Image", root]

    # ── 1. Training set: sample n_train from negatives ──
    neg_list = root / args.negative_list
    if not neg_list.exists():
        print(f"ERROR: negative list not found at {neg_list}", file=sys.stderr)
        print("Adjust --negative_list, or pre-stage a list of negative stems, one per line.", file=sys.stderr)
        sys.exit(1)
    neg_stems = _load_stems(neg_list)
    rng.shuffle(neg_stems)
    train_stems = neg_stems[: args.n_train]
    test_clean_stems = neg_stems[args.n_train : args.n_train + args.n_test_clean]
    print(f"[train] {len(train_stems)} negative stems → {out}/train/good/")
    for stem in train_stems:
        src = _find_file(stem, img_candidates)
        _resize_to_3ch(src, out / "train" / "good" / f"{stem}.png", args.image_size)

    print(f"[test/good] {len(test_clean_stems)} held-out clean stems → {out}/test/good/")
    for stem in test_clean_stems:
        src = _find_file(stem, img_candidates)
        _resize_to_3ch(src, out / "test" / "good" / f"{stem}.png", args.image_size)

    # ── 2. Test anomalies: positives + bbox-derived masks ──
    pos_list = root / args.positive_list
    if not pos_list.exists():
        print(f"ERROR: positive list not found at {pos_list}", file=sys.stderr)
        sys.exit(1)
    pos_stems = _load_stems(pos_list)
    rng.shuffle(pos_stems)
    test_pos_stems = pos_stems[: args.n_test]
    print(f"[test/prohibited] {len(test_pos_stems)} positive stems → {out}/test/prohibited/")

    ann_root = root / args.annotation_dir
    missing_ann = 0
    for stem in test_pos_stems:
        src = _find_file(stem, img_candidates)
        _resize_to_3ch(src, out / "test" / "prohibited" / f"{stem}.png", args.image_size)

        ann = ann_root / f"{stem}.xml"
        if not ann.exists():
            missing_ann += 1
            # fall back: write an empty mask so downstream eval doesn't crash
            blank = np.zeros((args.image_size, args.image_size), dtype=np.uint8)
            _save_mask(blank, out / "ground_truth" / "prohibited" / f"{stem}_mask.png")
            continue
        img_size_wh, bboxes = _parse_voc_bboxes(ann)
        mask = _bboxes_to_mask(img_size_wh, bboxes, args.image_size)
        _save_mask(mask, out / "ground_truth" / "prohibited" / f"{stem}_mask.png")

    if missing_ann:
        print(f"  ⚠ {missing_ann}/{len(test_pos_stems)} positives had no VOC annotation (blank mask written)")

    # ── 3. Sanity manifest ──
    manifest = out / "manifest.txt"
    with open(manifest, "w") as f:
        f.write(f"sixray_root={root}\n")
        f.write(f"image_size={args.image_size}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"n_train={len(train_stems)}\n")
        f.write(f"n_test_clean={len(test_clean_stems)}\n")
        f.write(f"n_test_positive={len(test_pos_stems)}\n")
        f.write(f"n_test_positive_missing_ann={missing_ann}\n")
    print(f"\nDone. Manifest: {manifest}")


if __name__ == "__main__":
    main()
