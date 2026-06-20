"""
Per-category qualitative comparison grid. Uses the cached reconstructions and
the metric CSVs to pick the key models, recomputes anomaly maps for a few
defective test images, and renders an advisor-ready grid:

  rows = sample images; columns =
    input | GT | pretrained | best-early FT | final FT | best interp | best LoRA

Consistent colorbars per row are achieved by normalizing each model's map by a
shared per-image scale. Saves <cat>/figures/qualitative_comparison_grid.{png,pdf}
"""
import os, sys, glob, csv, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import ddad_diag as D
from anomaly_map import heat_map


def read_csv(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def amap_for(cfg, cache, sd_or_path, idxs, device):
    fe = D.load_fe_from_state(device, sd_or_path, parallel=True)
    maps = []
    with torch.no_grad():
        for i in idxs:
            inp = cache["inputs"][i].to(device); x0 = cache["recons"][i].to(device)
            m = heat_map(x0, inp, fe, cfg).detach().cpu().squeeze().numpy()
            maps.append(m)
    del fe; torch.cuda.empty_cache()
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--unet_ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--exclude_test_subdirs", nargs="*", default=[])
    args = ap.parse_args()

    cat = args.category
    cdir = os.path.join(args.exp_dir, cat)
    figs = os.path.join(cdir, "figures"); os.makedirs(figs, exist_ok=True)
    cfg = D.load_config(cat, device=args.device)
    cfg.data.exclude_test_subdirs = args.exclude_test_subdirs

    cache_path = os.path.join(args.cache_dir, f"recon_{cat}.pt")
    if not os.path.exists(cache_path):
        unet = D.build_unet(cfg, args.unet_ckpt)
        cache = D.reconstruct_and_cache(cfg, unet, cache_path)
        del unet; torch.cuda.empty_cache()
    else:
        cache = torch.load(cache_path, map_location="cpu")

    # pick defective samples
    defect = [i for i, l in enumerate(cache["labels"]) if l == 1]
    idxs = defect[:: max(1, len(defect) // args.n_samples)][:args.n_samples] or list(range(min(args.n_samples, cache["n"])))

    # identify key models
    cs = read_csv(os.path.join(cdir, "metrics", "checkpoint_sweep.csv"))
    n_final = max(int(r["epoch"]) for r in cs) if cs else None
    best_e = next((int(r["epoch"]) for r in cs if r["is_best"] in ("True", "1")), n_final)
    models = [("pretrained", os.path.join(args.feat_dir, "feat0.pth")),
              (f"best-early(e{best_e})", os.path.join(args.feat_dir, f"feat{best_e}.pth")),
              (f"final(e{n_final})", os.path.join(args.feat_dir, f"feat{n_final}.pth"))]

    isweep = read_csv(os.path.join(cdir, "metrics", "interpolation_sweep.csv"))
    if isweep:
        pf = [r for r in isweep if r["leg"] == "pre_to_final"]
        if pf:
            bi = max(pf, key=lambda r: fl(r["pixel_auroc"]))
            a = fl(bi["alpha"])
            sd = D.interpolate_state_dicts(
                torch.load(os.path.join(args.feat_dir, "feat0.pth"), map_location="cpu"),
                torch.load(os.path.join(args.feat_dir, f"feat{n_final}.pth"), map_location="cpu"), a)
            models.append((f"interp(a={a:g})", sd))

    lsweep = read_csv(os.path.join(cdir, "metrics", "lora_sweep.csv"))
    if lsweep:
        bl = max(lsweep, key=lambda r: fl(r["pixel_auroc"]))
        # find that checkpoint file
        ldir = os.path.join(cdir, "checkpoints", "lora", bl["lora_run"])
        lp = os.path.join(ldir, f"{bl['checkpoint_name']}.pth")
        if os.path.exists(lp):
            models.append((f"LoRA({bl['lora_run']})", lp))

    # compute maps
    model_maps = {name: amap_for(cfg, cache, src, idxs, args.device) for name, src in models}

    ncol = 2 + len(models)
    fig, axes = plt.subplots(len(idxs), ncol, figsize=(2.4 * ncol, 2.4 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    for r, i in enumerate(idxs):
        img = ((cache["inputs"][i].squeeze().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        gt = cache["gts"][i].squeeze().numpy()
        axes[r, 0].imshow(img); axes[r, 0].set_ylabel(f"img {i}", fontsize=8)
        axes[r, 1].imshow(gt, cmap="gray")
        if r == 0:
            axes[r, 0].set_title("input", fontsize=9); axes[r, 1].set_title("GT", fontsize=9)
        for c, (name, _) in enumerate(models):
            m = model_maps[name][r]
            vmax = np.percentile(np.concatenate([model_maps[name][rr].ravel()
                                                 for rr in range(len(idxs))]), 99)
            axes[r, 2 + c].imshow(m, cmap="jet", vmin=m.min(), vmax=vmax)
            if r == 0:
                axes[r, 2 + c].set_title(name, fontsize=8)
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle(f"{cat}: qualitative anomaly-map comparison", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(figs, "qualitative_comparison_grid.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(figs, "qualitative_comparison_grid.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[{cat}] wrote qualitative_comparison_grid")


if __name__ == "__main__":
    main()
