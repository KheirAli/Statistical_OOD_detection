#!/usr/bin/env python3
"""Evaluate OOD detection pipeline.

Usage:
    python evaluate.py --config configs/experiment.yaml
    python evaluate.py --config configs/experiment.yaml --skip_sampling
    python evaluate.py --config configs/experiment.yaml --skip_sampling --baselines
    python evaluate.py --config configs/experiment.yaml --skip_sampling --no_plots
    python evaluate.py --config configs/experiment.yaml --sample_name samples_005
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
from ood.metrics import evaluate_delta_map
from ood.visualize import plot_delta_map, plot_evaluation, plot_comparison, show_boundaries
from ood.sampler import run_sampling
from ood.baselines import run_baselines


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _run_single_sample(cfg: dict, no_plots: bool = False, sampling_only: bool = False) -> dict:
    """Run evaluation on a single sample. Returns metrics dict."""
    import torch

    data_cfg = cfg["data"]
    sample_name = data_cfg["sample_name"]
    sample_id = parse_sample_id(sample_name)
    device = cfg["embeddings"].get("device", "cuda")

    # Per-sample figures directory so each image gets its own superpixel mask
    figures_dir = os.path.join(data_cfg["figures_dir"], sample_name)
    data_cfg_run = deepcopy(data_cfg)
    data_cfg_run["figures_dir"] = figures_dir

    print(f"\n=== OOD Evaluation: {sample_name} ===")

    run_sampling(cfg["sampling"], data_cfg_run)

    # Ensure per-sample superpixel mask exists (may have been generated
    # during sampling, or needs generating now for eval-only runs)
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

    if sampling_only:
        print(f"  Sampling complete for {sample_name}.")
        return {}

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
    delta_map, _info, labels_used = compute_delta_map(
        labels_fine=labels_fine, parent_map=parent_map,
        images_recon_all=recon_all, pca_feats_recon=embed_result["pca_feats_recon"],
        label_image=label_image, label_pca_map=embed_result["label_pca_map"],
        bins_rgb=score_cfg["bins_rgb"], bins_pca=score_cfg["bins_pca"],
        smooth_sigma=score_cfg["smooth_sigma"], min_pixels=score_cfg["min_pixels"],
        use_label_as_target=score_cfg["use_label_as_target"], eps=score_cfg["eps"],
    )
    print(f"  Scored {len(labels_used)} superpixels")

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
        metrics = _run_single_sample(
            cfg_copy, no_plots=args.no_plots,
            sampling_only=getattr(args, "sampling_only", False),
        )
        all_runs[sname] = metrics

    if getattr(args, "sampling_only", False):
        print("Sampling complete for all samples.")
        return

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
    parser.add_argument("--sampling_only", action="store_true", help="Run DPS sampling only, skip evaluation")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--baselines", action="store_true", help="Run baseline comparisons")
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
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.sample_name:
        cfg["data"]["sample_name"] = args.sample_name
    if args.skip_sampling:
        cfg["sampling"]["enabled"] = False
    if args.baselines:
        cfg["baselines"]["enabled"] = True
        for key in ("simplenet", "ddad"):
            if key in cfg["baselines"]:
                cfg["baselines"][key]["enabled"] = True
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

    # Multi-sample sweep mode
    if args.sample_names:
        _run_sweep(cfg, args)
        return

    data_cfg = cfg["data"]
    sample_name = data_cfg["sample_name"]
    sample_id = parse_sample_id(sample_name)
    device = cfg["embeddings"].get("device", "cuda")

    print(f"=== OOD Evaluation: {sample_name} ===")

    # ── Stage 0: Sampling ─────────────────────────────────────
    run_sampling(cfg["sampling"], data_cfg)

    if args.sampling_only:
        print(f"Sampling complete for {sample_name}.")
        return

    # ── Stage 1: Load data ────────────────────────────────────
    print("Loading data...")
    recon_all = load_reconstructions(
        results_dir=data_cfg["results_dir"],
        sample_name=sample_name,
        test_origin=data_cfg["test_origin"],
        num_patches=cfg["sampling"].get("num_patches", 24),
        bottom_suffix=data_cfg["bottom_suffix"],
    )
    print(f"  Reconstructions: {recon_all.shape}")

    label_image = load_label_image(
        results_dir=data_cfg["results_dir"],
        sample_name=sample_name,
        test_origin=data_cfg["test_origin"],
        bottom_suffix=data_cfg["bottom_suffix"],
    )
    print(f"  Label image: {label_image.shape}")

    sp_mask = load_superpixel_mask(data_cfg["figures_dir"])
    print(f"  Superpixel mask: {sp_mask.shape}, {len(np.unique(sp_mask))} regions")

    gt_mask = load_gt_mask(
        path_template=data_cfg["gt_mask"]["path"],
        sample=sample_id,
        downsample_factor=data_cfg["gt_mask"]["downsample_factor"],
    )
    print(f"  GT mask: {gt_mask.shape}, {gt_mask.sum()} anomalous pixels ({100*gt_mask.mean():.1f}%)")

    # ── Stage 2: Superpixel refinement ────────────────────────
    print("Refining superpixels...")
    sp_cfg = cfg["superpixels"]
    labels_fine, final_ids, parent_map = recursive_subdivide(
        img=label_image,
        labels=sp_mask,
        var_threshold=sp_cfg["var_threshold"],
        min_pixels=sp_cfg["min_pixels"],
        max_sub=sp_cfg["max_sub"],
        max_depth=sp_cfg["max_depth"],
        compactness=sp_cfg["compactness"],
        alpha_grad=sp_cfg["alpha_grad"],
        target_size=sp_cfg["target_size"],
    )
    print(f"  {len(np.unique(sp_mask))} -> {len(final_ids)} superpixels")

    # ── Stage 3+4: Embed + PCA ────────────────────────────────
    print("Computing embeddings + PCA...")
    import torch

    embed_cfg = cfg["embeddings"]
    embedder = ResNetPixelEmbedder(
        resnet_name=embed_cfg["backbone"],
        layers=tuple(embed_cfg["layers"]),
        use_patch_context=embed_cfg["use_patch_context"],
        patchify_size=embed_cfg.get("patchify_size", 3),
        proj_dim_per_layer=embed_cfg.get("proj_dim_per_layer"),
    ).to(device).eval()

    embed_result = embed_and_project(
        embedder=embedder,
        label_image=label_image,
        images_recon_all=recon_all,
        n_pca=cfg["pca"]["n_components"],
        device=device,
    )
    print(f"  PCA feats: label={embed_result['label_pca_map'].shape}, recon={embed_result['pca_feats_recon'].shape}")

    # Free GPU memory
    del embedder
    torch.cuda.empty_cache()

    # ── Stage 5: Scoring ──────────────────────────────────────
    print("Computing delta map...")
    score_cfg = cfg["scoring"]
    delta_map, info, labels_used = compute_delta_map(
        labels_fine=labels_fine,
        parent_map=parent_map,
        images_recon_all=recon_all,
        pca_feats_recon=embed_result["pca_feats_recon"],
        label_image=label_image,
        label_pca_map=embed_result["label_pca_map"],
        bins_rgb=score_cfg["bins_rgb"],
        bins_pca=score_cfg["bins_pca"],
        smooth_sigma=score_cfg["smooth_sigma"],
        min_pixels=score_cfg["min_pixels"],
        use_label_as_target=score_cfg["use_label_as_target"],
        eps=score_cfg["eps"],
    )
    print(f"  Scored {len(labels_used)} superpixels")

    # ── Stage 6: Evaluation ───────────────────────────────────
    print("Evaluating...")
    eval_cfg = cfg["eval"]
    all_metrics = {}

    for sigma in eval_cfg["delta_smooth_sigmas"]:
        key = f"sigma_{sigma}" if sigma else "raw"
        result = evaluate_delta_map(
            delta_map=delta_map,
            labels_fine=labels_fine,
            gt_mask_binary=gt_mask,
            anomaly_threshold=eval_cfg["sp_anomaly_threshold"],
            smooth_sigma=sigma,
        )
        all_metrics[key] = result
        print(f"  [{key}] SP AUC: {result['sp_roc_auc']:.4f} | Pixel AUC: {result['px_roc_auc']:.4f}")

    # ── Baselines ─────────────────────────────────────────────
    baseline_results = {}
    if cfg["baselines"].get("enabled", False):
        print("Running baselines...")
        baseline_results = run_baselines(cfg["baselines"])
        for name, res in baseline_results.items():
            if "error" not in res:
                print(f"  [{name}] Pixel AUC: {res.get('px_roc_auc', 'N/A')}")

    # ── Save results ──────────────────────────────────────────
    output_dir = os.path.join(eval_cfg["output_dir"], sample_name)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Save delta map
    np.save(os.path.join(output_dir, "delta_map.npy"), delta_map)

    # Save JSON results
    # Strip non-serializable curve data for compact JSON
    metrics_clean = {}
    for key, val in all_metrics.items():
        metrics_clean[key] = {k: v for k, v in val.items() if k != "curves"}

    results_json = {
        "config": cfg,
        "metrics": metrics_clean,
        "baselines": {k: {kk: vv for kk, vv in v.items() if kk != "raw_output"} for k, v in baseline_results.items()},
        "scoring_info": {
            "num_scored_superpixels": len(labels_used),
            "total_superpixels": len(final_ids),
        },
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/results.json")

    # ── Plots ─────────────────────────────────────────────────
    if not args.no_plots and eval_cfg.get("save_plots", True):
        print("Generating plots...")

        show_boundaries(
            label_image, labels_fine,
            title=f"Refined: {len(final_ids)} superpixels",
            save_path=os.path.join(plots_dir, "superpixels.png"),
        )

        plot_delta_map(
            delta_map, label_image,
            save_path=os.path.join(plots_dir, "delta_map.png"),
        )

        for key, result in all_metrics.items():
            sigma = None if key == "raw" else float(key.split("_")[1])
            plot_evaluation(
                eval_results=result,
                gt_mask_binary=gt_mask,
                delta_map=delta_map,
                smooth_sigma=sigma,
                save_path=os.path.join(plots_dir, f"evaluation_{key}.png"),
            )

        # Comparison plot if baselines ran
        if baseline_results:
            comparison = {}
            for key, val in all_metrics.items():
                comparison[f"ours_{key}"] = val
            for name, res in baseline_results.items():
                if "error" not in res:
                    comparison[name] = res
            plot_comparison(comparison, save_path=os.path.join(plots_dir, "comparison.png"))

        print(f"Plots saved to {plots_dir}/")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    for key, val in metrics_clean.items():
        print(f"  [{key}] SP AUC: {val['sp_roc_auc']:.4f} | SP AP: {val['sp_ap']:.4f} | Px AUC: {val['px_roc_auc']:.4f} | Px AP: {val['px_ap']:.4f}")
    if baseline_results:
        for name, res in baseline_results.items():
            if "error" not in res:
                print(f"  [{name}] Px AUC: {res.get('px_roc_auc', 'N/A')}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
