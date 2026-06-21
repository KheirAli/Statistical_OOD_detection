#!/usr/bin/env python
"""Evaluate DDAD feature-extractor checkpoints on DVXRay test folders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO / "tools"
DDAD_DIR = REPO / "DDAD"
for path in (str(TOOLS_DIR), str(DDAD_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ddad_da_finetune import (  # noqa: E402
    build_feature_extractor,
    build_unet,
    load_config,
    strip_module_prefix,
)
from anomaly_map import heat_map  # noqa: E402
from dataset import Dataset_maker  # noqa: E402
from metrics import Metric  # noqa: E402
from reconstruction import Reconstruction  # noqa: E402


def load_feature_extractor(checkpoint_path, feature_extractor, device):
    model = build_feature_extractor(feature_extractor, device=device, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def filter_dataset(dataset, subdirs, limit_per_subdir=None):
    if not subdirs and limit_per_subdir is None:
        return dataset

    subdir_set = set(subdirs or [])
    counts = {}
    kept = []
    for path in dataset.image_files:
        subdir = Path(path).parent.name
        if subdir_set and subdir not in subdir_set:
            continue
        if limit_per_subdir is not None:
            count = counts.get(subdir, 0)
            if count >= limit_per_subdir:
                continue
            counts[subdir] = count + 1
        kept.append(path)
    dataset.image_files = kept
    return dataset


def labels_from_paths(paths):
    return [0 if Path(path).parent.name == "good" else 1 for path in paths]


def make_recon_cache(cfg, unet_ckpt, cache_path, test_subdirs, limit_per_subdir):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    cfg.data.test_subfolder = None
    dataset = Dataset_maker(
        root=cfg.data.data_dir,
        category=cfg.data.category,
        config=cfg,
        is_train=False,
    )
    dataset = filter_dataset(dataset, test_subdirs, limit_per_subdir=limit_per_subdir)
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty after filtering")

    paths = list(dataset.image_files)
    subdir_counts = {}
    for path in paths:
        subdir = Path(path).parent.name
        subdir_counts[subdir] = subdir_counts.get(subdir, 0) + 1

    print(f"eval images: {len(dataset)} {subdir_counts}")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.test_batch_size,
        shuffle=False,
        num_workers=cfg.model.num_workers,
        drop_last=False,
    )
    unet = build_unet(cfg, unet_ckpt)
    recon = Reconstruction(unet, cfg)

    inputs, recons, gts, labels = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for idx, (inp, gt, lab) in enumerate(loader, start=1):
            inp = inp.to(cfg.model.device)
            x0 = recon(inp, inp, cfg.model.w)[-1].detach().cpu()
            inputs.append(inp.cpu())
            recons.append(x0)
            if gt.shape[-2:] != (cfg.data.image_size, cfg.data.image_size):
                gt = F.interpolate(gt.float(), size=(cfg.data.image_size, cfg.data.image_size), mode="nearest")
            gts.append(gt.cpu())
            labels.extend([0 if item == "good" else 1 for item in lab])
            if idx == 1 or idx % 25 == 0:
                print(f"reconstructed {idx}/{len(loader)}", flush=True)

    cache = {
        "inputs": inputs,
        "recons": recons,
        "gts": gts,
        "labels": labels,
        "paths": paths,
        "subdir_counts": subdir_counts,
        "recon_seconds": round(time.time() - t0, 2),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def segmentation_at_best_f1(maps, gts):
    maps = maps.astype(np.float32)
    gts = (gts > 0.5).astype(np.uint8)
    if gts.sum() == 0:
        return {"f1": float("nan"), "dice": float("nan"), "iou": float("nan"), "threshold": float("nan")}

    mn, mx = float(maps.min()), float(maps.max())
    norm = (maps - mn) / (mx - mn + 1e-8)
    flat_scores = norm.reshape(-1)
    flat_gt = gts.reshape(-1)
    thresholds = np.quantile(flat_scores, np.linspace(0.5, 0.999, 60))
    best = {"f1": -1.0}
    positives = flat_gt.sum()
    for threshold in thresholds:
        pred = flat_scores >= threshold
        tp = np.logical_and(pred, flat_gt == 1).sum()
        fp = np.logical_and(pred, flat_gt == 0).sum()
        fn = positives - tp
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        iou = tp / (tp + fp + fn + 1e-12)
        if f1 > best["f1"]:
            best = {
                "f1": float(f1),
                "dice": float(f1),
                "iou": float(iou),
                "threshold": float(threshold),
            }
    return best


def evaluate_checkpoint(cfg, cache, checkpoint_path, feature_extractor, compute_pro):
    fe = load_feature_extractor(checkpoint_path, feature_extractor, cfg.model.device)
    anomaly_maps, gt_list, predictions = [], [], []
    labels = list(cache["labels"])
    t0 = time.time()
    with torch.no_grad():
        for inp, x0, gt in zip(cache["inputs"], cache["recons"], cache["gts"]):
            inp = inp.to(cfg.model.device)
            x0 = x0.to(cfg.model.device)
            amap = heat_map(x0, inp, fe, cfg).detach().cpu()
            anomaly_maps.append(amap)
            gt_list.append(gt.cpu())
            predictions.append(float(amap.max()))

    metric = Metric(labels, predictions, anomaly_maps, gt_list, cfg)
    row = {
        "checkpoint": Path(checkpoint_path).name,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "runtime_sec": round(time.time() - t0, 2),
    }
    for name, func in (
        ("image_auroc", metric.image_auroc),
        ("pixel_auroc", metric.pixel_auroc),
    ):
        try:
            row[name] = float(func())
        except Exception as exc:
            row[name] = float("nan")
            row[f"{name}_error"] = str(exc)

    if compute_pro:
        try:
            row["pro"] = float(metric.pixel_pro())
        except Exception as exc:
            row["pro"] = float("nan")
            row["pro_error"] = str(exc)
    else:
        row["pro"] = float("nan")

    maps = torch.cat(anomaly_maps, dim=0).squeeze(1).numpy()
    gts = torch.cat(gt_list, dim=0).squeeze(1).numpy()
    try:
        maps_norm = (maps - maps.min()) / (maps.max() - maps.min() + 1e-8)
        row["pixel_ap"] = float(average_precision_score((gts > 0.5).reshape(-1), maps_norm.reshape(-1)))
    except Exception as exc:
        row["pixel_ap"] = float("nan")
        row["pixel_ap_error"] = str(exc)
    row.update(segmentation_at_best_f1(maps, gts))
    try:
        row["image_ap"] = float(average_precision_score(labels, predictions))
    except Exception as exc:
        row["image_ap"] = float("nan")
        row["image_ap_error"] = str(exc)
    try:
        row["image_auroc_sklearn"] = float(roc_auc_score(labels, predictions))
    except Exception:
        row["image_auroc_sklearn"] = row.get("image_auroc", float("nan"))

    if torch.cuda.is_available():
        row["gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DDAD_DIR / "config_dvxray.yaml"))
    parser.add_argument("--unet_ckpt", default="/data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000")
    parser.add_argument("--feat_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--feature_extractor", default="resnet101")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test_subdirs", nargs="*", default=["good", "anomaly", "scissors"])
    parser.add_argument("--limit_per_subdir", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_pro", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config, device=args.device, feature_extractor=args.feature_extractor)
    cfg.model.num_workers = args.num_workers
    cfg.data.test_batch_size = 1
    cfg.data.test_subfolder = None

    cache_path = args.cache_path
    if cache_path is None:
        subdir_tag = "_".join(args.test_subdirs) if args.test_subdirs else "all"
        limit_tag = f"_limit{args.limit_per_subdir}" if args.limit_per_subdir else ""
        cache_path = out_dir / f"recon_cache_{subdir_tag}{limit_tag}.pt"

    cache = make_recon_cache(
        cfg,
        args.unet_ckpt,
        cache_path,
        args.test_subdirs,
        args.limit_per_subdir,
    )

    checkpoints = sorted(
        Path(args.feat_dir).glob("feat*.pth"),
        key=lambda path: int(path.stem.replace("feat", "")),
    )
    if not checkpoints:
        raise RuntimeError(f"No feat*.pth checkpoints found in {args.feat_dir}")

    rows = []
    for checkpoint in checkpoints:
        print(f"evaluating {checkpoint.name}", flush=True)
        row = evaluate_checkpoint(
            cfg,
            cache,
            checkpoint,
            args.feature_extractor,
            compute_pro=not args.no_pro,
        )
        rows.append(row)
        print(
            f"{checkpoint.name}: image_auc={row['image_auroc']:.4f} "
            f"pixel_auc={row['pixel_auroc']:.4f} pixel_ap={row['pixel_ap']:.4f}",
            flush=True,
        )

    csv_path = out_dir / "checkpoint_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(out_dir / "checkpoint_metrics.json", "w") as handle:
        json.dump({"cache": cache_path.as_posix() if isinstance(cache_path, Path) else cache_path, "rows": rows}, handle, indent=2)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
