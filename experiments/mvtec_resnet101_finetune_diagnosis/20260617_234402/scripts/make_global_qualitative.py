"""Global qualitative summary grid: one row per (reliable) category, columns =
input | GT | pretrained map | final-FT map | recommended-model map.

Uses cached reconstructions + the metric CSVs to pick the recommended model.
Writes figures/global_qualitative_summary_grid.{png,pdf}.
"""
import os, sys, csv, glob, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import ddad_diag as D
from anomaly_map import heat_map

# reliable categories (best global pixel-AUROC >= 0.6), cable first
ORDER = ["cable", "leather", "hazelnut", "capsule", "pill", "toothbrush",
         "screw", "metal_nut", "transistor"]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_csv(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def map_for(cfg, cache, src, idx, device):
    fe = D.load_fe_from_state(device, src, parallel=True)
    with torch.no_grad():
        m = heat_map(cache["recons"][idx].to(device), cache["inputs"][idx].to(device),
                     fe, cfg).detach().cpu().squeeze().numpy()
    del fe; torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    exp = args.exp_dir
    cats = [c for c in ORDER if os.path.exists(os.path.join(exp, c, "metrics", "checkpoint_sweep.csv"))]

    ncol = 5
    fig, axes = plt.subplots(len(cats), ncol, figsize=(2.3 * ncol, 2.3 * len(cats)))
    if len(cats) == 1:
        axes = axes[None, :]
    col_titles = ["input", "GT", "pretrained (feat0)", "final FT (feat8)", "recommended"]

    for r, cat in enumerate(cats):
        cfg = D.load_config(cat, device=args.device)
        cfg.data.exclude_test_subdirs = ["combined"] if cat == "cable" else []
        cache = torch.load(os.path.join(exp, "cache", f"recon_{cat}.pt"), map_location="cpu")
        defect = [i for i, l in enumerate(cache["labels"]) if l == 1]
        idx = defect[len(defect) // 2] if defect else 0

        cs = read_csv(os.path.join(exp, cat, "metrics", "checkpoint_sweep.csv"))
        n_final = max(int(x["epoch"]) for x in cs)
        best_e = next((int(x["epoch"]) for x in cs if x["is_best"] in ("True", "1")), n_final)
        feat_dir = os.path.join(exp, cat, "checkpoints")

        # recommended model: replicate report logic lite (best of final/best-early/interp/lora by pixel_auroc)
        isweep = read_csv(os.path.join(exp, cat, "metrics", "interpolation_sweep.csv"))
        rec_src, rec_lbl = os.path.join(feat_dir, f"feat{best_e}.pth"), f"best-early feat{best_e}"
        if isweep:
            pf = [x for x in isweep if x["leg"] == "pre_to_final"]
            bi = max(pf, key=lambda x: fl(x["pixel_auroc"]))
            if fl(bi["pixel_auroc"]) >= fl(max(cs, key=lambda x: fl(x["pixel_auroc"]))["pixel_auroc"]):
                a = fl(bi["alpha"])
                rec_src = D.interpolate_state_dicts(
                    torch.load(os.path.join(feat_dir, "feat0.pth"), map_location="cpu"),
                    torch.load(os.path.join(feat_dir, f"feat{n_final}.pth"), map_location="cpu"), a)
                rec_lbl = f"interp α={a:g}"

        img = ((cache["inputs"][idx].squeeze().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        gt = cache["gts"][idx].squeeze().numpy()
        m_pre = map_for(cfg, cache, os.path.join(feat_dir, "feat0.pth"), idx, args.device)
        m_fin = map_for(cfg, cache, os.path.join(feat_dir, f"feat{n_final}.pth"), idx, args.device)
        m_rec = map_for(cfg, cache, rec_src, idx, args.device)

        vmax = np.percentile(np.concatenate([m_pre.ravel(), m_fin.ravel(), m_rec.ravel()]), 99)
        panels = [(img, None), (gt, "gray"), (m_pre, "jet"), (m_fin, "jet"), (m_rec, "jet")]
        for c, (im, cmap) in enumerate(panels):
            ax = axes[r, c]
            if cmap == "jet":
                ax.imshow(im, cmap="jet", vmin=im.min(), vmax=vmax)
            elif cmap == "gray":
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
        axes[r, 0].set_ylabel(cat, fontsize=10)
        axes[r, 4].set_xlabel(rec_lbl, fontsize=7)

    fig.suptitle("Global qualitative summary — anomaly maps by FE variant (reliable categories)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(exp, "figures", "global_qualitative_summary_grid")
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out + ".png")


if __name__ == "__main__":
    main()
