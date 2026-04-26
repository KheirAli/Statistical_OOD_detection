"""Run a baseline over every defective image in every MVTec category.

Usage:
    python tools/eval_baseline_mvtec.py \
        --baseline simplenet \
        --output_root ./results_eval/mvtec_full \
        [--categories bottle cable ...]      # default: all 15

What it does, per (baseline × category):
  1. Walks `<mvtec_root>/<class>/test/<defect>/` for every defect subdir.
     Skips `test/good/` (no positives → per-image AUC undefined).
  2. For each (sample, defect) pair, loads the test image + matching mask
     under `ground_truth/<defect>/<sample>_mask.png`.
  3. Calls `baseline.score(image, gt_mask)` and computes pixel AUROC + AP
     + SNR (raw and gaussian-σ=5 smoothed) per image.
  4. Writes a sweep JSON at `<output_root>/<baseline>_<class>/sweep.json`
     with per-image numbers + bootstrap 95% CIs over per-image AUCs.
  5. Aggregates per-category means into one master CSV at
     `<output_root>/<baseline>_per_category.csv` and an overall MVTec
     average (mean across 15 categories).

CKPT path resolution per baseline (override via `--ckpt_template`):
  simplenet:       <ckpt_root>/simplenet_mvtec_full/simplenet_mvtec/run/models/0/mvtec_<class>/ckpt.pth
  supersimplenet:  <ckpt_root>/supersimplenet_hf/mvtec/<class>/1/weights.pt
  cutpaste:        <ckpt_root>/cutpaste_mvtec_full/model-<class>-*.tch  (latest by name)

Single-pass methods only (SimpleNet, SuperSimpleNet, CutPaste). Multi-seed
diffusion baselines (MDPS, DOOD) need a different runner that handles the
20-seed reconstruction loop — out of scope here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ood.baselines import build_baseline                          # noqa: E402
from ood.metrics import (                                         # noqa: E402
    compute_snr, manual_auc, manual_average_precision, manual_roc_curve,
)

sys.path.insert(0, str(REPO_ROOT / "tools"))
from ci_stats import bootstrap_mean_ci                            # noqa: E402


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


# ─────────────── ckpt + train-dir resolution ──────────────────────

def _resolve_ckpt(baseline: str, category: str, ckpt_root: Path) -> str:
    if baseline == "simplenet":
        # Prefer the queue-trained ckpt (`simplenet_mvtec_full`); fall back to
        # the original cluster ckpt for cable (which we never re-trained
        # since it was already on disk before the queue ran).
        primary = (
            ckpt_root / "simplenet_mvtec_full" / "simplenet_mvtec" / "run"
            / "models" / "0" / f"mvtec_{category}" / "ckpt.pth"
        )
        if primary.exists():
            return str(primary)
        fallback = Path(
            f"/data/akheirandish3/SimpleNet/results/MVTecAD_Results/"
            f"simplenet_mvtec/run/models/0/mvtec_{category}/ckpt.pth"
        )
        if fallback.exists():
            return str(fallback)
        raise FileNotFoundError(
            f"No SimpleNet ckpt for {category}; tried {primary} and {fallback}"
        )
    if baseline == "supersimplenet":
        return str(
            ckpt_root / "supersimplenet_hf" / "mvtec" / category / "1" / "weights.pt"
        )
    if baseline == "cutpaste":
        # Filename has a date stamp; pick the most recent.
        pattern = str(ckpt_root / "cutpaste_mvtec_full" / f"model-{category}-*.tch")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No CutPaste ckpt for {category} matching {pattern}")
        return matches[-1]
    raise ValueError(f"unknown baseline {baseline!r}")


def _baseline_params(baseline: str, ckpt: str, category: str,
                     mvtec_root: Path, device: str) -> Dict[str, Any]:
    """Build the params dict consumed by `build_baseline`."""
    if baseline == "simplenet":
        return {"ckpt": ckpt, "device": device}
    if baseline == "supersimplenet":
        return {"ckpt": ckpt, "device": device}
    if baseline == "cutpaste":
        return {
            "ckpt": ckpt,
            "device": device,
            "train_dir": str(mvtec_root / category / "train" / "good"),
            "embed_cache": str(
                Path(ckpt).parent / f"_embed_cache_{category}.pt"
            ),
        }
    raise ValueError(baseline)


# ─────────────── per-sample scoring loop ─────────────────────────

def _list_defective_samples(category_root: Path) -> List[Tuple[str, str]]:
    """Return [(defect_subdir, sample_id), ...] for every defective image.

    `test/good/` is skipped — no positives → AUROC undefined per-image.
    """
    out = []
    for d in sorted((category_root / "test").iterdir()):
        if d.name == "good" or not d.is_dir():
            continue
        for img in sorted(d.glob("*.png")):
            out.append((d.name, img.stem))   # ('combined', '000')
    return out


def _load_image_uint8(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _load_gt_mask(path: Path, size: int = 256) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"))
    if mask.shape[0] != size:
        # Resize via PIL nearest to preserve binary structure.
        mask = np.asarray(
            Image.fromarray(mask).resize((size, size), Image.NEAREST),
        )
    return (mask > 0).astype(np.uint8)


def _score_one_sample(
    baseline, image: np.ndarray, gt: np.ndarray,
    smooth_sigmas: List[Any],
) -> Dict[str, Dict[str, float]]:
    """Returns {"raw": {...}, "sigma_5.0": {...}} for one (image, gt) pair.

    Mirrors evaluate.py's _baseline_metrics shape so downstream
    aggregation works without changes.
    """
    from scipy.ndimage import gaussian_filter as _gf

    amap = baseline.score(image, gt)
    n_pos = int(gt.sum())
    n_neg = int(gt.size - n_pos)
    out: Dict[str, Dict[str, float]] = {}
    for sigma in smooth_sigmas:
        key = f"sigma_{sigma}" if sigma else "raw"
        m = amap if not sigma else _gf(amap, sigma=sigma, mode="nearest")
        if n_pos == 0 or n_neg == 0:
            out[key] = {k: float("nan") for k in
                        ("px_roc_auc", "px_ap", "px_snr")}
            continue
        scores = m.ravel()
        labels = gt.ravel()
        fpr, tpr, _ = manual_roc_curve(labels, scores)
        out[key] = {
            "px_roc_auc": float(manual_auc(fpr, tpr)),
            "px_ap": float(manual_average_precision(labels, scores)),
            "px_snr": float(compute_snr(m, gt)),
        }
    return out


# ─────────────── per-category aggregation ────────────────────────

def _aggregate_category(per_image: Dict[str, Dict[str, Dict[str, float]]],
                        smooth_sigmas: List[Any]) -> Dict[str, Any]:
    """Bootstrap-CI aggregate per (smooth-sigma, metric) over per-image scores."""
    METRICS = ("px_roc_auc", "px_ap", "px_snr")
    averaged: Dict[str, Dict[str, Any]] = {}
    samples = list(per_image.keys())
    for sigma in smooth_sigmas:
        key = f"sigma_{sigma}" if sigma else "raw"
        block: Dict[str, Any] = {}
        for stat in METRICS:
            vals = [per_image[s][key][stat] for s in samples]
            m, lo, hi, n = bootstrap_mean_ci(vals)
            block[stat] = m
            block[f"{stat}_ci95"] = [lo, hi]
            block[f"{stat}_n"] = n
        averaged[key] = block
    return averaged


# ─────────────── main loop ───────────────────────────────────────

def run_category(
    baseline_name: str, category: str,
    ckpt_root: Path, mvtec_root: Path,
    device: str, output_dir: Path,
    smooth_sigmas: List[Any],
) -> Dict[str, Any]:
    print(f"\n=== {baseline_name} / {category} ===", flush=True)
    cat_root = mvtec_root / category
    samples = _list_defective_samples(cat_root)
    print(f"  {len(samples)} defective samples across "
          f"{len({d for d, _ in samples})} subdirs", flush=True)

    ckpt = _resolve_ckpt(baseline_name, category, ckpt_root)
    params = _baseline_params(baseline_name, ckpt, category, mvtec_root, device)
    baseline = build_baseline(baseline_name, params)

    per_image: Dict[str, Dict[str, Dict[str, float]]] = {}
    for defect, sid in samples:
        img_path = cat_root / "test" / defect / f"{sid}.png"
        gt_path = cat_root / "ground_truth" / defect / f"{sid}_mask.png"
        if not gt_path.exists():
            print(f"  [skip] missing GT for {defect}/{sid}", flush=True)
            continue
        img = _load_image_uint8(img_path)
        gt = _load_gt_mask(gt_path)
        per_image[f"{defect}/{sid}"] = _score_one_sample(
            baseline, img, gt, smooth_sigmas,
        )

    averaged = _aggregate_category(per_image, smooth_sigmas)
    sweep = {
        "baseline": baseline_name,
        "category": category,
        "ckpt": str(ckpt),
        "n_samples": len(per_image),
        "per_image": per_image,
        "averaged": averaged,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sweep.json"
    out_path.write_text(json.dumps(sweep, indent=2, default=str))
    # Print headline
    px = averaged["sigma_5.0"]
    print(f"  Px AUC: {px['px_roc_auc']:.4f} "
          f"[{px['px_roc_auc_ci95'][0]:.4f}, {px['px_roc_auc_ci95'][1]:.4f}] | "
          f"SNR: {px['px_snr']:.4f}  "
          f"(n={px['px_roc_auc_n']})", flush=True)
    return sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                   choices=("simplenet", "supersimplenet", "cutpaste"))
    p.add_argument("--mvtec_root", default="/data/akheirandish3/mvtec_ad")
    p.add_argument("--ckpt_root", default="/data2/rohan/baseline_ckpts")
    p.add_argument("--output_root", default="./results_eval/mvtec_full")
    p.add_argument("--categories", nargs="+", default=None,
                   help="Subset of MVTec categories. Default: all 15.")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    cats = args.categories or MVTEC_CATEGORIES
    smooth_sigmas: List[Any] = [None, 5.0]

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    for cat in cats:
        try:
            sweep = run_category(
                baseline_name=args.baseline,
                category=cat,
                ckpt_root=Path(args.ckpt_root),
                mvtec_root=Path(args.mvtec_root),
                device=args.device,
                output_dir=out_root / f"{args.baseline}_{cat}",
                smooth_sigmas=smooth_sigmas,
            )
            row = {
                "category": cat,
                "n_samples": sweep["n_samples"],
                "px_auc_raw": sweep["averaged"]["raw"]["px_roc_auc"],
                "px_auc_smooth5": sweep["averaged"]["sigma_5.0"]["px_roc_auc"],
                "px_ap_smooth5": sweep["averaged"]["sigma_5.0"]["px_ap"],
                "px_snr_smooth5": sweep["averaged"]["sigma_5.0"]["px_snr"],
            }
            summary.append(row)
        except Exception as e:                                       # noqa: BLE001
            print(f"  [error] {cat}: {e}", flush=True)
            summary.append({"category": cat, "error": str(e)})
        # Free GPU between categories
        torch.cuda.empty_cache()

    # CSV with per-category numbers + overall mean across categories
    csv_path = out_root / f"{args.baseline}_per_category.csv"
    fieldnames = ["category", "n_samples", "px_auc_raw",
                  "px_auc_smooth5", "px_ap_smooth5", "px_snr_smooth5"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + ["error"])
        w.writeheader()
        for row in summary:
            w.writerow({k: row.get(k, "") for k in fieldnames + ["error"]})
        # Mean-across-categories row (paper convention)
        valid = [r for r in summary if "px_auc_smooth5" in r]
        if valid:
            mean_row = {"category": "MEAN_ACROSS_CATEGORIES",
                        "n_samples": sum(r["n_samples"] for r in valid)}
            for k in ("px_auc_raw", "px_auc_smooth5", "px_ap_smooth5", "px_snr_smooth5"):
                mean_row[k] = float(np.mean([r[k] for r in valid]))
            w.writerow({k: mean_row.get(k, "") for k in fieldnames + ["error"]})
    print(f"\nWrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
