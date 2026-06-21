#!/usr/bin/env python
"""Per-image pixel metrics for DDAD feature-extractor checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO / "tools"), str(REPO / "DDAD")):
    if path not in sys.path:
        sys.path.insert(0, path)

from anomaly_map import heat_map  # noqa: E402
from ddad_da_finetune import load_config  # noqa: E402
from ddad_eval_fe_checkpoints import load_feature_extractor, make_recon_cache  # noqa: E402


def per_image_metrics(scores, gt):
    scores = scores.astype(np.float32).reshape(-1)
    gt = (gt > 0.5).astype(np.uint8).reshape(-1)
    out = {}
    if gt.min() == gt.max():
        out["pixel_auroc"] = float("nan")
    else:
        out["pixel_auroc"] = float(roc_auc_score(gt, scores))
    try:
        out["pixel_ap"] = float(average_precision_score(gt, scores))
    except Exception:
        out["pixel_ap"] = float("nan")
    out["gt_pixels"] = int(gt.sum())
    out["gt_fraction"] = float(gt.mean())
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "DDAD" / "config_dvxray.yaml"))
    parser.add_argument("--unet_ckpt", default="/data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000")
    parser.add_argument("--feat_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--feature_extractor", default="resnet101")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test_subdirs", nargs="*", default=["scissors"])
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, device=args.device, feature_extractor=args.feature_extractor)
    cfg.model.num_workers = args.num_workers
    cfg.data.test_batch_size = 1
    cfg.data.test_subfolder = None

    cache_path = args.cache_path
    if cache_path is None:
        cache_path = Path(args.out_csv).with_suffix(".recon_cache.pt")
    cache = make_recon_cache(
        cfg,
        args.unet_ckpt,
        cache_path,
        args.test_subdirs,
        limit_per_subdir=None,
    )

    checkpoints = sorted(
        Path(args.feat_dir).glob("feat*.pth"),
        key=lambda path: int(path.stem.replace("feat", "")),
    )
    rows = []
    for checkpoint in checkpoints:
        print(f"per-image scoring {checkpoint.name}", flush=True)
        fe = load_feature_extractor(checkpoint, args.feature_extractor, cfg.model.device)
        with torch.no_grad():
            for idx, (inp, x0, gt, path) in enumerate(
                zip(cache["inputs"], cache["recons"], cache["gts"], cache["paths"])
            ):
                amap = heat_map(x0.to(cfg.model.device), inp.to(cfg.model.device), fe, cfg)
                scores = amap.detach().cpu().squeeze().numpy()
                mask = gt.squeeze().numpy()
                metrics = per_image_metrics(scores, mask)
                image_id = Path(path).stem.replace("_OL", "")
                rows.append(
                    {
                        "checkpoint": checkpoint.name,
                        "image_id": image_id,
                        "path": path,
                        **metrics,
                    }
                )
                if idx == 0 or (idx + 1) % 16 == 0:
                    print(f"  {checkpoint.name}: {idx + 1}/{len(cache['paths'])}", flush=True)
        del fe
        torch.cuda.empty_cache()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["checkpoint", "image_id", "path", "pixel_auroc", "pixel_ap", "gt_pixels", "gt_fraction"]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with out_csv.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
