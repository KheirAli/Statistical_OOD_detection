#!/usr/bin/env python3
"""
Hyperparameter sweep for the OOD evaluation pipeline.

Tests all combinations of:
  - Feature mode: PCA vs Autoencoder
  - n_pca / AE latent dim
  - bins_rgb
  - bins_pca
  - smooth_sigma

Results are written as individual JSON files and a single combined
summary CSV + ranked table printed to stdout.

Usage:
    python run_hyperparameter_sweep.py \\
        --config configs/experiment_ddad_native.yaml \\
        --sample_names samples_000 samples_001 samples_002 \\
        --output_dir ./sweep_results \\
        --device cuda:0

    # With AE checkpoints
    python run_hyperparameter_sweep.py \\
        --config configs/experiment_ddad_native.yaml \\
        --sample_names samples_000 samples_001 \\
        --autoencoder_paths /models/ae_3.pth /models/ae_5.pth \\
        --output_dir ./sweep_results
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from itertools import product

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Default sweep grid  (override via CLI)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N_PCA        = [3, 5, 9]
DEFAULT_BINS_RGB     = [16, 32, 64]
DEFAULT_BINS_PCA     = [8, 16]
DEFAULT_SMOOTH_SIGMA = [0.0, 0.1, 0.5, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# Import evaluate internals so we don't shell out per run
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_config, _run_single_sample   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Result accumulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _avg_metrics(per_sample: dict, sample_names: list, sigma_key: str) -> dict:
    """Average a specific sigma_key across all samples."""
    sp_aucs = [per_sample[s][sigma_key]["sp_roc_auc"] for s in sample_names]
    px_aucs = [per_sample[s][sigma_key]["px_roc_auc"] for s in sample_names]
    sp_aps  = [per_sample[s][sigma_key]["sp_ap"]       for s in sample_names]
    px_aps  = [per_sample[s][sigma_key]["px_ap"]       for s in sample_names]
    return {
        "sp_roc_auc": float(np.mean(sp_aucs)),
        "px_roc_auc": float(np.mean(px_aucs)),
        "sp_ap":      float(np.mean(sp_aps)),
        "px_ap":      float(np.mean(px_aps)),
    }


def _best_sigma_key(averaged: dict) -> str:
    """Return the smooth-sigma key with the highest mean pixel AUC."""
    return max(averaged, key=lambda k: averaged[k]["px_roc_auc"])


# ─────────────────────────────────────────────────────────────────────────────
# Single sweep run (one hyperparameter combo, all samples)
# ─────────────────────────────────────────────────────────────────────────────

def _run_combo(
    base_cfg:     dict,
    sample_names: list,
    n_pca:        int,
    bins_rgb:     int,
    bins_pca:     int,
    smooth_sigma: float,
    ae_path:      str,            # "" → PCA mode
    output_dir:   str,
) -> dict:
    """Run all samples for one hyperparameter combination.  Returns summary dict."""

    mode_tag = (f"ae_{os.path.splitext(os.path.basename(ae_path))[0]}"
                if ae_path else "pca")
    combo_id = (f"{mode_tag}_n{n_pca}_rgb{bins_rgb}"
                f"_pca{bins_pca}_s{str(smooth_sigma).replace('.','p')}")

    print(f"\n{'─'*60}")
    print(f"COMBO: {combo_id}")
    print(f"{'─'*60}")

    per_sample = {}
    for sname in sample_names:
        cfg = deepcopy(base_cfg)
        cfg["data"]["sample_name"]        = sname
        cfg["pca"]["n_components"]         = n_pca
        cfg["scoring"]["bins_rgb"]         = bins_rgb
        cfg["scoring"]["bins_pca"]         = bins_pca
        cfg["scoring"]["smooth_sigma"]     = smooth_sigma
        cfg["scoring"]["autoencoder_path"] = ae_path

        try:
            per_sample[sname] = _run_single_sample(cfg)
        except Exception as exc:
            print(f"  [WARN] {sname} failed: {exc}")
            per_sample[sname] = {}

    # Collect all sigma keys (from first successful sample)
    first_ok = next((per_sample[s] for s in sample_names if per_sample[s]), {})
    sigma_keys = list(first_ok.keys())

    averaged = {k: _avg_metrics(per_sample, sample_names, k)
                for k in sigma_keys if all(k in per_sample[s] for s in sample_names)}

    best_key = _best_sigma_key(averaged) if averaged else "raw"

    summary = {
        "combo_id":    combo_id,
        "mode":        "AE" if ae_path else "PCA",
        "ae_path":     ae_path,
        "n_pca":       n_pca,
        "bins_rgb":    bins_rgb,
        "bins_pca":    bins_pca,
        "smooth_sigma": smooth_sigma,
        "best_eval_sigma": best_key,
        "per_sample":  per_sample,
        "averaged":    averaged,
        **averaged.get(best_key, {}),    # top-level quick access: sp_roc_auc etc.
    }

    # Save individual JSON
    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, f"{combo_id}.json")
    with open(jpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  → best [{best_key}]  "
          f"SP AUC={averaged.get(best_key,{}).get('sp_roc_auc',float('nan')):.4f}  "
          f"Px AUC={averaged.get(best_key,{}).get('px_roc_auc',float('nan')):.4f}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Print ranked results table
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(results: list, sort_by: str = "px_roc_auc") -> None:
    results_sorted = sorted(results, key=lambda r: r.get(sort_by, 0), reverse=True)
    cols = ["combo_id", "sp_roc_auc", "px_roc_auc", "sp_ap", "px_ap",
            "n_pca", "bins_rgb", "bins_pca", "smooth_sigma", "best_eval_sigma"]
    W = 14

    sep = "─" * (W * len(cols) + len(cols) + 1)
    print(f"\n{'='*60}")
    print(f"RANKED RESULTS  (sorted by {sort_by})")
    print(sep)
    print("│" + "│".join(c[:W-1].ljust(W-1) for c in cols) + "│")
    print(sep)
    for i, r in enumerate(results_sorted):
        marker = " ★" if i == 0 else "  "
        vals = []
        for c in cols:
            v = r.get(c, "—")
            s = (f"{v:.4f}" if isinstance(v, float) else str(v))[:W-1].ljust(W-1)
            vals.append(s)
        print("│" + "│".join(vals) + f"│{marker}")
    print(sep)
    print("  ★ = best")


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV summary
# ─────────────────────────────────────────────────────────────────────────────

def _save_csv(results: list, path: str) -> None:
    cols = ["combo_id", "mode", "n_pca", "bins_rgb", "bins_pca", "smooth_sigma",
            "best_eval_sigma", "sp_roc_auc", "px_roc_auc", "sp_ap", "px_ap", "ae_path"]
    lines = [",".join(cols)]
    for r in results:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nCSV summary saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Hyperparameter sweep for OOD evaluation")
    p.add_argument("--config",          required=True,   help="Path to experiment YAML")
    p.add_argument("--sample_names",    nargs="+", required=True)
    p.add_argument("--output_dir",      default="./sweep_results")
    p.add_argument("--device",          default=None)
    p.add_argument("--sort_by",         default="px_roc_auc",
                   choices=["px_roc_auc", "sp_roc_auc", "px_ap", "sp_ap"])

    # Grid parameters
    p.add_argument("--n_pca",         type=int, nargs="+", default=DEFAULT_N_PCA,
                   help=f"PCA components / AE latent dims to try (default: {DEFAULT_N_PCA})")
    p.add_argument("--bins_rgb",      type=int, nargs="+", default=DEFAULT_BINS_RGB,
                   help=f"RGB histogram bins to try (default: {DEFAULT_BINS_RGB})")
    p.add_argument("--bins_pca",      type=int, nargs="+", default=DEFAULT_BINS_PCA,
                   help=f"PCA histogram bins to try (default: {DEFAULT_BINS_PCA})")
    p.add_argument("--smooth_sigma",  type=float, nargs="+", default=DEFAULT_SMOOTH_SIGMA,
                   help=f"Histogram smooth sigmas to try (default: {DEFAULT_SMOOTH_SIGMA})")

    # Feature modes
    p.add_argument("--pca_only",      action="store_true",
                   help="Only run PCA mode (skip AE)")
    p.add_argument("--ae_only",       action="store_true",
                   help="Only run AE mode (skip PCA)")
    p.add_argument("--autoencoder_paths", nargs="+", default=[],
                   help="AE checkpoint paths to sweep over")

    args = p.parse_args()

    base_cfg = load_config(args.config)
    base_cfg.setdefault("sampling", {})["enabled"] = False
    base_cfg.setdefault("scoring", {}).setdefault("algorithm", "typical_set")
    base_cfg.setdefault("scoring", {}).setdefault("min_pixels", 2)
    if args.device:
        base_cfg["embeddings"]["device"] = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build list of (ae_path,) to sweep — "" means PCA
    ae_paths = []
    if not args.ae_only:
        ae_paths.append("")           # PCA mode
    if not args.pca_only:
        ae_paths.extend(args.autoencoder_paths)

    if not ae_paths:
        print("[WARN] No feature modes selected. Using PCA.")
        ae_paths = [""]

    # Build full grid
    grid = list(product(ae_paths, args.n_pca, args.bins_rgb,
                        args.bins_pca, args.smooth_sigma))
    total = len(grid)
    print(f"\nSweep: {total} combinations × {len(args.sample_names)} samples")
    print(f"Output dir: {args.output_dir}")

    # ── Run sweep ─────────────────────────────────────────────────────────
    all_results = []
    t0 = time.time()
    for idx, (ae_path, n_pca, bins_rgb, bins_pca, smooth_sigma) in enumerate(grid, 1):
        print(f"\n[{idx}/{total}]", end=" ")
        try:
            result = _run_combo(
                base_cfg=base_cfg,
                sample_names=args.sample_names,
                n_pca=n_pca, bins_rgb=bins_rgb,
                bins_pca=bins_pca, smooth_sigma=smooth_sigma,
                ae_path=ae_path,
                output_dir=args.output_dir,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"  [ERROR] combo failed: {exc}")

        elapsed = time.time() - t0
        remaining = elapsed / idx * (total - idx)
        print(f"  Elapsed: {elapsed/60:.1f} min  ETA: {remaining/60:.1f} min")

    # ── Summary ───────────────────────────────────────────────────────────
    _print_table(all_results, sort_by=args.sort_by)

    csv_path = os.path.join(args.output_dir, f"sweep_summary_{ts}.csv")
    _save_csv(all_results, csv_path)

    # Full JSON dump
    full_path = os.path.join(args.output_dir, f"sweep_full_{ts}.json")
    with open(full_path, "w") as f:
        json.dump({
            "timestamp":    ts,
            "sample_names": args.sample_names,
            "grid": {
                "n_pca":        args.n_pca,
                "bins_rgb":     args.bins_rgb,
                "bins_pca":     args.bins_pca,
                "smooth_sigma": args.smooth_sigma,
                "ae_paths":     ae_paths,
            },
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"Full results: {full_path}")

    # Print best config
    if all_results:
        best = max(all_results, key=lambda r: r.get(args.sort_by, 0))
        print(f"\n★ BEST CONFIG (by {args.sort_by}):")
        print(f"  combo_id    : {best['combo_id']}")
        print(f"  mode        : {best['mode']}")
        print(f"  n_pca       : {best['n_pca']}")
        print(f"  bins_rgb    : {best['bins_rgb']}")
        print(f"  bins_pca    : {best['bins_pca']}")
        print(f"  smooth_sigma: {best['smooth_sigma']}")
        print(f"  SP AUC      : {best.get('sp_roc_auc', float('nan')):.4f}")
        print(f"  Px AUC      : {best.get('px_roc_auc', float('nan')):.4f}")
        print(f"  SP AP       : {best.get('sp_ap', float('nan')):.4f}")
        print(f"  Px AP       : {best.get('px_ap', float('nan')):.4f}")
        if best["ae_path"]:
            print(f"  AE path     : {best['ae_path']}")


if __name__ == "__main__":
    main()