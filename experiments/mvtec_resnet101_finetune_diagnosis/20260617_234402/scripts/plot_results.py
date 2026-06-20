"""
Aggregate + plot results from per-category metric CSVs. CSV-only (no GPU),
safe to run repeatedly on partial results.

Produces per-category:
  <cat>/figures/checkpoint_metric_curves.{png,pdf}
  <cat>/figures/interpolation_metric_curves.{png,pdf}
  <cat>/figures/lora_training_curves.{png,pdf}   (if lora_sweep.csv present)
And aggregate (under exp/figures, exp/metrics):
  metrics/all_categories_checkpoint_sweep.{csv,json}
  metrics/all_categories_interpolation_sweep.csv
  metrics/all_categories_lora_sweep.csv
  figures/all_categories_{pixel_auroc,image_auroc,pro}_vs_epoch.{png,pdf}
  figures/overfitting_summary_heatmap.{png,pdf}
  figures/all_categories_best_interpolation_alpha.{png,pdf}
  figures/lora_vs_full_finetune_summary.{png,pdf}
"""
import os, sys, glob, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATS = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable",
        "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush",
        "transistor", "zipper"]
METRICS = ["image_auroc", "pixel_auroc", "pro", "auprc", "dice", "iou", "f1",
           "precision", "recall"]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_csv(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def save(fig, base):
    fig.savefig(base + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def per_category(exp, cat):
    cdir = os.path.join(exp, cat)
    figs = os.path.join(cdir, "figures"); os.makedirs(figs, exist_ok=True)
    cs = read_csv(os.path.join(cdir, "metrics", "checkpoint_sweep.csv"))
    if cs:
        ep = [int(r["epoch"]) for r in cs]
        order = np.argsort(ep); ep = [ep[i] for i in order]
        cs = [cs[i] for i in order]
        best_e = next((int(r["epoch"]) for r in cs if r["is_best"] in ("True", "1", True)), None)
        fin_e = max(ep)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for m in ["image_auroc", "pixel_auroc", "pro", "auprc", "f1"]:
            y = [fl(r[m]) for r in cs]
            ax.plot(ep, y, marker="o", label=m)
        ax.axvline(0, color="gray", ls=":", alpha=0.6)
        if best_e is not None:
            ax.axvline(best_e, color="green", ls="--", alpha=0.7, label=f"best (e{best_e})")
        ax.axvline(fin_e, color="red", ls="--", alpha=0.5, label=f"final (e{fin_e})")
        ax.set_xlabel("DA epoch (0 = pretrained)"); ax.set_ylabel("metric")
        ax.set_title(f"{cat}: checkpoint sweep"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        save(fig, os.path.join(figs, "checkpoint_metric_curves"))

    isweep = read_csv(os.path.join(cdir, "metrics", "interpolation_sweep.csv"))
    if isweep:
        legs = sorted(set(r["leg"] for r in isweep))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for leg in legs:
            rr = sorted([r for r in isweep if r["leg"] == leg], key=lambda r: fl(r["alpha"]))
            a = [fl(r["alpha"]) for r in rr]
            ax.plot(a, [fl(r["pixel_auroc"]) for r in rr], marker="o", label=f"{leg} pxAUROC")
            ax.plot(a, [fl(r["pro"]) for r in rr], marker="s", ls="--", label=f"{leg} PRO")
        ax.set_xlabel("alpha (0=first endpoint, 1=second)"); ax.set_ylabel("metric")
        ax.set_title(f"{cat}: interpolation sweep"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        save(fig, os.path.join(figs, "interpolation_metric_curves"))

    lsweep = read_csv(os.path.join(cdir, "metrics", "lora_sweep.csv"))
    if lsweep:
        runs = sorted(set(r["lora_run"] for r in lsweep))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for run in runs:
            rr = sorted([r for r in lsweep if r["lora_run"] == run], key=lambda r: int(r["epoch"]))
            e = [int(r["epoch"]) for r in rr]
            ax.plot(e, [fl(r["pixel_auroc"]) for r in rr], marker="o", label=f"{run} pxAUROC")
        ax.set_xlabel("LoRA DA epoch"); ax.set_ylabel("pixel AUROC")
        ax.set_title(f"{cat}: LoRA training curves"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        save(fig, os.path.join(figs, "lora_training_curves"))
    return cs, isweep, lsweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    args = ap.parse_args()
    exp = args.exp_dir
    figs = os.path.join(exp, "figures"); os.makedirs(figs, exist_ok=True)
    mdir = os.path.join(exp, "metrics"); os.makedirs(mdir, exist_ok=True)

    all_cs, all_is, all_ls = [], [], []
    cat_curves = {}
    for cat in CATS:
        cs, isweep, lsweep = per_category(exp, cat)
        if cs:
            all_cs += cs; cat_curves[cat] = cs
        all_is += [dict(r, category=cat) for r in isweep]
        all_ls += lsweep

    # aggregate checkpoint csv/json
    if all_cs:
        keys = list(all_cs[0].keys())
        with open(os.path.join(mdir, "all_categories_checkpoint_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_cs)
        with open(os.path.join(mdir, "all_categories_checkpoint_sweep.json"), "w") as f:
            json.dump(all_cs, f, indent=2)
    if all_is:
        keys = list(all_is[0].keys())
        with open(os.path.join(mdir, "all_categories_interpolation_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_is)
    if all_ls:
        keys = list(all_ls[0].keys())
        with open(os.path.join(mdir, "all_categories_lora_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_ls)

    # metric-vs-epoch grids
    for metric, fname in [("pixel_auroc", "all_categories_pixel_auroc_vs_epoch"),
                          ("image_auroc", "all_categories_image_auroc_vs_epoch"),
                          ("pro", "all_categories_pro_vs_epoch")]:
        if not cat_curves:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for cat, cs in cat_curves.items():
            rr = sorted(cs, key=lambda r: int(r["epoch"]))
            ax.plot([int(r["epoch"]) for r in rr], [fl(r[metric]) for r in rr],
                    marker="o", ms=3, label=cat)
        ax.set_xlabel("DA epoch (0=pretrained)"); ax.set_ylabel(metric)
        ax.set_title(f"All categories: {metric} vs epoch")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        save(fig, os.path.join(figs, fname))

    # overfitting heatmap: (best - final) per metric per category
    if cat_curves:
        cats = list(cat_curves.keys())
        mat = np.full((len(cats), len(METRICS)), np.nan)
        for i, cat in enumerate(cats):
            cs = cat_curves[cat]
            ft = [r for r in cs if r["is_pretrained"] not in ("True", "1", True)]
            for j, m in enumerate(METRICS):
                vals = [(int(r["epoch"]), fl(r[m])) for r in ft if not np.isnan(fl(r[m]))]
                if not vals:
                    continue
                final = max(vals, key=lambda x: x[0])[1]
                best = max(v for _, v in vals)
                mat[i, j] = best - final
        fig, ax = plt.subplots(figsize=(10, 0.5 * len(cats) + 2))
        im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0)
        ax.set_xticks(range(len(METRICS))); ax.set_xticklabels(METRICS, rotation=45, ha="right")
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
        for i in range(len(cats)):
            for j in range(len(METRICS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=6)
        ax.set_title("Overfitting gap (best_early - final), higher = more overfitting")
        fig.colorbar(im, ax=ax, fraction=0.025)
        save(fig, os.path.join(figs, "overfitting_summary_heatmap"))

    # best interpolation alpha per category
    if all_is:
        cats, alphas, gains = [], [], []
        for cat in CATS:
            rr = [r for r in all_is if r["category"] == cat and r["leg"] == "pre_to_final"]
            if not rr:
                continue
            rr = sorted(rr, key=lambda r: fl(r["alpha"]))
            best = max(rr, key=lambda r: fl(r["pixel_auroc"]))
            final = max(rr, key=lambda r: fl(r["alpha"]))
            cats.append(cat); alphas.append(fl(best["alpha"]))
            gains.append(fl(best["pixel_auroc"]) - fl(final["pixel_auroc"]))
        if cats:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["green" if g > 0 else "gray" for g in gains]
            ax.bar(cats, alphas, color=colors)
            ax.set_ylabel("best alpha (pixel AUROC)"); ax.set_ylim(0, 1)
            ax.set_title("Best interpolation alpha per category (green=beats final)")
            plt.xticks(rotation=45, ha="right")
            save(fig, os.path.join(figs, "all_categories_best_interpolation_alpha"))

    # lora vs full finetune
    if all_ls and cat_curves:
        cats, full_v, lora_v = [], [], []
        for cat in CATS:
            cs = cat_curves.get(cat)
            ls = [r for r in all_ls if r["category"] == cat]
            if not cs or not ls:
                continue
            full_best = max((fl(r["pixel_auroc"]) for r in cs
                             if r["is_pretrained"] not in ("True", "1", True)), default=np.nan)
            lora_best = max((fl(r["pixel_auroc"]) for r in ls), default=np.nan)
            cats.append(cat); full_v.append(full_best); lora_v.append(lora_best)
        if cats:
            x = np.arange(len(cats)); fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(x - 0.2, full_v, 0.4, label="full FT (best)")
            ax.bar(x + 0.2, lora_v, 0.4, label="LoRA (best)")
            ax.set_xticks(x); ax.set_xticklabels(cats, rotation=45, ha="right")
            ax.set_ylabel("pixel AUROC"); ax.legend(); ax.set_title("LoRA vs full fine-tuning")
            save(fig, os.path.join(figs, "lora_vs_full_finetune_summary"))

    print(f"plots written under {figs}")


if __name__ == "__main__":
    main()
