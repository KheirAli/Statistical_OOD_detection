#!/usr/bin/env python3
"""Evaluate OOD detection pipeline.

Sampling is handled by tools/run_ddad_reconstruction.py or
tools/run_ddad_dps_sampling.py. evaluate.py only scores existing recons.

Usage:
    python evaluate.py --config configs/experiment_ddad_native.yaml \\
        --skip_sampling --no_plots --scorer typical_set \\
        --n_pca 5 --bins_pca 16 --sample_name samples_000
    python evaluate.py --config configs/experiment_ddad_native.yaml \\
        --skip_sampling --no_plots --scorer local_gaussian --sigma_rohan 0.1 \\
        --sample_names samples_000 samples_001 samples_002 ... samples_010
"""

import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np
import yaml
import subprocess, sys

from ood.data import load_gt_mask, load_label_image, load_reconstructions, load_superpixel_mask, parse_sample_id
from ood.superpixels import recursive_subdivide
from ood.embeddings import ResNetPixelEmbedder, embed_and_project
from ood.scoring import compute_delta_map
from ood.scoring_local_gaussian import compute_delta_map_local_gaussian
from ood.metrics import evaluate_delta_map
from ood.visualize import plot_delta_map, plot_evaluation, plot_comparison, show_boundaries
# Note: ood.sampler and ood.baselines were removed in the push-ready cleanup.
# Reconstructions are now generated via tools/run_ddad_reconstruction.py or
# tools/run_ddad_dps_sampling.py. Always pass --skip_sampling to evaluate.py.


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _run_single_sample(cfg: dict, no_plots: bool = False) -> dict:
    """Run evaluation on a single sample. Returns metrics dict."""
    import torch

    data_cfg = cfg["data"]
    sample_name = data_cfg["sample_name"]
    sample_id = parse_sample_id(sample_name)
    device = cfg["embeddings"].get("device", "cuda")

    # Per-sample figures directory so each image gets its own superpixel mask
    figures_dir = os.path.join(data_cfg["figures_dir"], sample_name)

    print(f"\n=== OOD Evaluation: {sample_name} ===")
    print("Sampling disabled, using existing results.")

    # Ensure per-sample superpixel mask exists (generate if missing)
    mask_path = os.path.join(figures_dir, "mask.png")
    if not os.path.exists(mask_path):
        image_dir = data_cfg["image_dir"]
        input_image = os.path.join(image_dir, f"{sample_id}.png")
        os.makedirs(figures_dir, exist_ok=True)
        if os.path.exists(input_image):
            print(f"  Generating superpixels for {sample_name}...")
            subprocess.run(
                [sys.executable, "super_pixel_generation.py",
                 f"--input_image={input_image}",
                 f"--output_dir={figures_dir}"],
                check=True,
            )

    # Stage 1: Load data
    print("Loading data...")
    recon_all = load_reconstructions(
        results_dir=data_cfg["results_dir"],
        sample_name=sample_name,
        test_origin=data_cfg["test_origin"],
        num_patches=cfg["sampling"].get("num_patches", 24),
        bottom_suffix=data_cfg["bottom_suffix"],
    )
    label_image = load_label_image(
        results_dir=data_cfg["results_dir"],
        sample_name=sample_name,
        test_origin=data_cfg["test_origin"],
        bottom_suffix=data_cfg["bottom_suffix"],
    )
    sp_mask = load_superpixel_mask(figures_dir)
    gt_mask = load_gt_mask(
        path_template=data_cfg["gt_mask"]["path"],
        sample=sample_id,
        downsample_factor=data_cfg["gt_mask"]["downsample_factor"],
    )

    # Stage 2: Superpixel refinement
    sp_cfg = cfg["superpixels"]
    labels_fine, final_ids, parent_map = recursive_subdivide(
        img=label_image, labels=sp_mask,
        var_threshold=sp_cfg["var_threshold"], min_pixels=sp_cfg["min_pixels"],
        max_sub=sp_cfg["max_sub"], max_depth=sp_cfg["max_depth"],
        compactness=sp_cfg["compactness"], alpha_grad=sp_cfg["alpha_grad"],
        target_size=sp_cfg["target_size"],
    )
    print(f"  {len(np.unique(sp_mask))} -> {len(final_ids)} superpixels")

    # Stage 3+4: Embed + PCA
    embed_cfg = cfg["embeddings"]
    embedder = ResNetPixelEmbedder(
        resnet_name=embed_cfg["backbone"],
        layers=tuple(embed_cfg["layers"]),
        use_patch_context=embed_cfg["use_patch_context"],
        patchify_size=embed_cfg.get("patchify_size", 3),
        proj_dim_per_layer=embed_cfg.get("proj_dim_per_layer"),
    ).to(device).eval()

    embed_result = embed_and_project(
        embedder=embedder, label_image=label_image,
        images_recon_all=recon_all, n_pca=cfg["pca"]["n_components"],
        device=device,
    )
    del embedder
    torch.cuda.empty_cache()

    # Stage 5: Scoring
    score_cfg = cfg["scoring"]
    algorithm = score_cfg.get("algorithm", "typical_set")
    if algorithm == "local_gaussian":
        sigma_rohan = score_cfg.get("sigma_rohan")
        if sigma_rohan is None:
            sigma_path = os.path.join(data_cfg["results_dir"], "sigma.txt")
            if os.path.exists(sigma_path):
                with open(sigma_path) as f:
                    sigma_rohan = float(f.read().strip())
                print(f"  sigma_rohan={sigma_rohan} (from {sigma_path})")
            else:
                raise ValueError("local_gaussian scorer needs --sigma_rohan or a sigma.txt under results_dir")
        # Use base SP mask (not refined) for local_gaussian: the scorer needs
        # D = n_sp * C to be tractable relative to n_realizations.  With 5636
        # refined SPs (D=16908) and rank ≤ n_realizations-1, the precision matrix
        # has near-zero coverage → metric collapses to 0.5.  Base mask (~44 SPs,
        # D=132) gives full-rank covariance with n_realizations=1000.
        lg_labels = sp_mask   # base mask, not labels_fine
        lg_parent = {int(sid): [int(sid)] for sid in np.unique(sp_mask)}
        delta_map, _info, labels_used = compute_delta_map_local_gaussian(
            labels_fine=lg_labels, parent_map=lg_parent,
            images_recon_all=recon_all, label_image=label_image,
            sigma=sigma_rohan,
            n_realizations=score_cfg.get("n_realizations", 1000),
            min_pixels=score_cfg["min_pixels"],
            metric=score_cfg.get("lg_metric", "typicality_unsigned"),
        )
    else:
        delta_map, _info, labels_used = compute_delta_map(
            labels_fine=labels_fine, parent_map=parent_map,
            images_recon_all=recon_all, pca_feats_recon=embed_result["pca_feats_recon"],
            label_image=label_image, label_pca_map=embed_result["label_pca_map"],
            bins_rgb=score_cfg["bins_rgb"], bins_pca=score_cfg["bins_pca"],
            smooth_sigma=score_cfg["smooth_sigma"], min_pixels=score_cfg["min_pixels"],
            use_label_as_target=score_cfg["use_label_as_target"], eps=score_cfg["eps"],
        )
    print(f"  Scored {len(labels_used)} superpixels [{algorithm}]")

    # Stage 6: Evaluation
    eval_cfg = cfg["eval"]
    all_metrics = {}
    for sigma in eval_cfg["delta_smooth_sigmas"]:
        key = f"sigma_{sigma}" if sigma else "raw"
        result = evaluate_delta_map(
            delta_map=delta_map, labels_fine=labels_fine,
            gt_mask_binary=gt_mask, anomaly_threshold=eval_cfg["sp_anomaly_threshold"],
            smooth_sigma=sigma,
        )
        all_metrics[key] = result
        print(f"  [{key}] SP AUC: {result['sp_roc_auc']:.4f} | Pixel AUC: {result['px_roc_auc']:.4f}")

    return all_metrics


def _run_sweep(cfg: dict, args) -> None:
    """Run evaluation on multiple samples and report averaged metrics."""
    all_runs = {}
    for sname in args.sample_names:
        cfg_copy = deepcopy(cfg)
        cfg_copy["data"]["sample_name"] = sname
        metrics = _run_single_sample(cfg_copy, no_plots=args.no_plots)
        all_runs[sname] = metrics

    # Average across samples
    first_keys = list(all_runs[args.sample_names[0]].keys())
    print(f"\n{'='*60}")
    print(f"SWEEP RESULTS ({len(args.sample_names)} images)")
    print(f"Config: backbone={cfg['embeddings']['backbone']}, "
          f"n_pca={cfg['pca']['n_components']}, "
          f"bins_rgb={cfg['scoring']['bins_rgb']}, "
          f"bins_pca={cfg['scoring']['bins_pca']}")
    for metric_key in first_keys:
        sp_aucs = [all_runs[s][metric_key]["sp_roc_auc"] for s in args.sample_names]
        px_aucs = [all_runs[s][metric_key]["px_roc_auc"] for s in args.sample_names]
        print(f"  [{metric_key}] Mean SP AUC: {np.mean(sp_aucs):.6f} | Mean Px AUC: {np.mean(px_aucs):.6f}")
    print(f"{'='*60}")

    # Save sweep results JSON
    output_dir = cfg["eval"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    sweep_results = {
        "config": {
            "backbone": cfg["embeddings"]["backbone"],
            "n_pca": cfg["pca"]["n_components"],
            "bins_rgb": cfg["scoring"]["bins_rgb"],
            "bins_pca": cfg["scoring"]["bins_pca"],
        },
        "sample_names": args.sample_names,
        "per_sample": {},
        "averaged": {},
    }
    for sname in args.sample_names:
        sweep_results["per_sample"][sname] = {
            k: {kk: vv for kk, vv in v.items() if kk != "curves"}
            for k, v in all_runs[sname].items()
        }
    for metric_key in first_keys:
        sp_aucs = [all_runs[s][metric_key]["sp_roc_auc"] for s in args.sample_names]
        px_aucs = [all_runs[s][metric_key]["px_roc_auc"] for s in args.sample_names]
        sweep_results["averaged"][metric_key] = {
            "sp_roc_auc": float(np.mean(sp_aucs)),
            "px_roc_auc": float(np.mean(px_aucs)),
        }

    sweep_name = (f"sweep_{cfg['embeddings']['backbone']}_pca{cfg['pca']['n_components']}"
                  f"_rgb{cfg['scoring']['bins_rgb']}_pca{cfg['scoring']['bins_pca']}")
    sweep_path = os.path.join(output_dir, f"{sweep_name}.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"Sweep results saved to {sweep_path}")


def main():
    parser = argparse.ArgumentParser(description="OOD Detection Evaluation Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--sample_name", type=str, default=None, help="Override sample name")
    parser.add_argument("--skip_sampling", action="store_true", help="Skip DPS sampling, use existing results")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/cuda:N)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Override ResNet backbone (resnet18/50/101/152)")
    parser.add_argument("--n_pca", type=int, default=None,
                        help="Override number of PCA components")
    parser.add_argument("--bins_rgb", type=int, default=None,
                        help="Override RGB histogram bins")
    parser.add_argument("--bins_pca", type=int, default=None,
                        help="Override PCA histogram bins")
    parser.add_argument("--sample_names", type=str, nargs="+", default=None,
                        help="Run on multiple samples and average metrics")
    parser.add_argument("--scorer", type=str, default="typical_set",
                        choices=["typical_set", "local_gaussian"],
                        help="Scoring algorithm: ood/scoring.py (typical_set) or ood/scoring_local_gaussian.py (rohan)")
    parser.add_argument("--sigma_rohan", type=float, default=None,
                        help="σ for local_gaussian scorer (in [0,1] image scale). "
                             "If omitted, reads {results_dir}/sigma.txt when scorer=local_gaussian.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.sample_name:
        cfg["data"]["sample_name"] = args.sample_name
    if args.skip_sampling:
        cfg.setdefault("sampling", {})["enabled"] = False
    if args.device:
        cfg["embeddings"]["device"] = args.device
    if args.output_dir:
        cfg["eval"]["output_dir"] = args.output_dir
    if args.backbone:
        cfg["embeddings"]["backbone"] = args.backbone
    if args.n_pca is not None:
        cfg["pca"]["n_components"] = args.n_pca
    if args.bins_rgb is not None:
        cfg["scoring"]["bins_rgb"] = args.bins_rgb
    if args.bins_pca is not None:
        cfg["scoring"]["bins_pca"] = args.bins_pca
    cfg.setdefault("scoring", {})["algorithm"] = args.scorer
    if args.sigma_rohan is not None:
        cfg["scoring"]["sigma_rohan"] = args.sigma_rohan

    # Everything is routed through _run_sweep (works for 1 or many samples).
    if not args.sample_names:
        args.sample_names = [args.sample_name or cfg["data"]["sample_name"]]
    _run_sweep(cfg, args)


if __name__ == "__main__":
    main()
