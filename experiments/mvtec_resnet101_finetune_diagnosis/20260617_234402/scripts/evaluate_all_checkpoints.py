"""
Per-category evaluation harness for the diagnosis.

Reconstructs the test set ONCE (cached), then evaluates many FE variants:
  - checkpoint sweep:    feat0 (pretrained) .. featN (final)
  - interpolation sweep: WiSE-FT alpha between feat0 and featN (+ best-early legs)
  - LoRA runs:           each provided lora dir's checkpoints

Writes per-category:
  metrics/checkpoint_sweep.{csv,json}
  metrics/interpolation_sweep.{csv,json}
  metrics/lora_sweep.csv            (if --lora_dirs given)
"""
import os, sys, json, csv, argparse, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import ddad_diag as D

COLUMNS = ["category", "checkpoint_name", "epoch", "is_pretrained", "is_final",
           "is_best", "image_auroc", "pixel_auroc", "pro", "auprc", "dice", "iou",
           "f1", "precision", "recall", "loss", "runtime_sec", "gpu_memory_mb",
           "checkpoint_path", "eval_command"]
MAIN_METRIC = "pixel_auroc"


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "NA") for k in COLUMNS})


def best_row(rows, exclude_pretrained=True):
    cand = [r for r in rows if not (exclude_pretrained and r["is_pretrained"])]
    cand = [r for r in cand if isinstance(r.get(MAIN_METRIC), float) and not np.isnan(r[MAIN_METRIC])]
    if not cand:
        return None
    return max(cand, key=lambda r: r[MAIN_METRIC])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--unet_ckpt", required=True)
    ap.add_argument("--feat_dir", required=True, help="dir with feat0..featN.pth")
    ap.add_argument("--out_dir", required=True, help="category dir (metrics/ under it)")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pro", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="limit #test images (debug)")
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--no_interp", action="store_true")
    ap.add_argument("--lora_dirs", nargs="*", default=[],
                    help="name:dir entries; each dir has feat0..featK.pth")
    ap.add_argument("--exclude_test_subdirs", nargs="*", default=[])
    ap.add_argument("--lora_only", action="store_true",
                    help="skip checkpoint+interpolation sweeps; only (re)compute lora_sweep.csv")
    args = ap.parse_args()

    cfg = D.load_config(args.category, device=args.device)
    cfg.data.exclude_test_subdirs = args.exclude_test_subdirs
    compute_pro = not args.no_pro
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # ---- 1. reconstruction cache ----
    cache_path = os.path.join(args.cache_dir, f"recon_{args.category}.pt")
    print(f"[{args.category}] building/loading reconstruction cache ...")
    unet = D.build_unet(cfg, args.unet_ckpt)
    cache = D.reconstruct_and_cache(cfg, unet, cache_path, seed=args.seed, limit=args.limit)
    print(f"[{args.category}] cache: {cache['n']} test images "
          f"(recon {cache.get('recon_seconds', 0):.1f}s)")
    del unet
    torch.cuda.empty_cache()

    # da loss per epoch (if log present)
    da_loss = {}
    log_p = os.path.join(args.feat_dir, "da_log.json")
    if os.path.exists(log_p):
        with open(log_p) as f:
            dl = json.load(f)
        for e in dl.get("epochs", []):
            da_loss[e["epoch"]] = e["loss"]

    # ---- 2. checkpoint sweep ----
    feats = sorted(glob.glob(os.path.join(args.feat_dir, "feat*.pth")),
                   key=lambda p: int(os.path.basename(p)[4:-4]))
    n_final = max(int(os.path.basename(p)[4:-4]) for p in feats)
    rows = []
    for p in ([] if args.lora_only else feats):
        epoch = int(os.path.basename(p)[4:-4])
        tag = f"feat{epoch}"
        fe = D.load_fe_from_state(args.device, p, parallel=True)
        m = D.evaluate_fe(cfg, cache, fe, compute_pro=compute_pro, tag=tag)
        del fe; torch.cuda.empty_cache()
        row = {**m, "category": args.category, "checkpoint_name": tag, "epoch": epoch,
               "is_pretrained": epoch == 0, "is_final": epoch == n_final,
               "is_best": False, "loss": da_loss.get(epoch, "NA"),
               "checkpoint_path": os.path.abspath(p),
               "eval_command": f"evaluate_all_checkpoints.py --category {args.category} --feat_dir {args.feat_dir}"}
        rows.append(row)
        print(f"  {tag}: pxAUROC={m['pixel_auroc']:.4f} imgAUROC={m['image_auroc']:.4f} "
              f"PRO={m['pro']:.4f} AUPRC={m['auprc']:.4f} F1={m['f1']:.4f}")
    b = best_row(rows)
    if b:
        b["is_best"] = True
    if not args.lora_only:
        write_csv(os.path.join(metrics_dir, "checkpoint_sweep.csv"), rows)
        with open(os.path.join(metrics_dir, "checkpoint_sweep.json"), "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[{args.category}] best early/full checkpoint: "
              f"{b['checkpoint_name'] if b else 'NA'} (final=feat{n_final})")

    # ---- 3. interpolation sweep ----
    if not args.no_interp and not args.lora_only:
        sd_pre = torch.load(os.path.join(args.feat_dir, "feat0.pth"), map_location="cpu")
        sd_fin = torch.load(os.path.join(args.feat_dir, f"feat{n_final}.pth"), map_location="cpu")
        best_epoch = b["epoch"] if b else n_final
        sd_best = torch.load(os.path.join(args.feat_dir, f"feat{best_epoch}.pth"), map_location="cpu")

        irows = []
        legs = [("pre_to_final", sd_pre, sd_fin)]
        if best_epoch != n_final:
            legs.append(("pre_to_best", sd_pre, sd_best))
            legs.append(("best_to_final", sd_best, sd_fin))
        for leg_name, sa, sb in legs:
            for a in args.alphas:
                sd = D.interpolate_state_dicts(sa, sb, a)
                fe = D.load_fe_from_state(args.device, sd, parallel=True)
                m = D.evaluate_fe(cfg, cache, fe, compute_pro=compute_pro,
                                  tag=f"{leg_name}_a{a}")
                del fe; torch.cuda.empty_cache()
                m.update({"category": args.category, "leg": leg_name, "alpha": a})
                irows.append(m)
                print(f"  interp {leg_name} a={a:.2f}: pxAUROC={m['pixel_auroc']:.4f} "
                      f"PRO={m['pro']:.4f}")
        # csv
        ic = ["category", "leg", "alpha", "image_auroc", "pixel_auroc", "pro",
              "auprc", "dice", "iou", "f1", "precision", "recall", "runtime_sec"]
        with open(os.path.join(metrics_dir, "interpolation_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ic); w.writeheader()
            for r in irows:
                w.writerow({k: r.get(k, "NA") for k in ic})
        with open(os.path.join(metrics_dir, "interpolation_sweep.json"), "w") as f:
            json.dump(irows, f, indent=2)

    # ---- 4. LoRA runs ----
    if args.lora_dirs:
        lrows = []
        lc = ["category", "lora_run", "checkpoint_name", "epoch", "is_final", "is_best",
              "n_trainable", "n_total", "pct_trainable", "image_auroc", "pixel_auroc",
              "pro", "auprc", "dice", "iou", "f1", "precision", "recall", "loss"]
        for entry in args.lora_dirs:
            name, ldir = entry.split(":", 1)
            llog = {}
            lp = os.path.join(ldir, "da_log.json")
            n_tr = n_tot = pct = "NA"
            if os.path.exists(lp):
                with open(lp) as f:
                    ld = json.load(f)
                llog = {e["epoch"]: e["loss"] for e in ld.get("epochs", [])}
                n_tr = ld.get("n_trainable", "NA"); n_tot = ld.get("n_total", "NA")
                pct = ld.get("pct_trainable", "NA")
            lfeats = sorted(glob.glob(os.path.join(ldir, "feat*.pth")),
                            key=lambda p: int(os.path.basename(p)[4:-4]))
            lfinal = max(int(os.path.basename(p)[4:-4]) for p in lfeats)
            sub = []
            for p in lfeats:
                ep = int(os.path.basename(p)[4:-4])
                if ep == 0:
                    continue  # feat0 == pretrained, already in checkpoint sweep
                fe = D.load_fe_from_state(args.device, p, parallel=True)
                m = D.evaluate_fe(cfg, cache, fe, compute_pro=compute_pro,
                                  tag=f"{name}_feat{ep}")
                del fe; torch.cuda.empty_cache()
                m.update({"category": args.category, "lora_run": name,
                          "checkpoint_name": f"feat{ep}", "epoch": ep,
                          "is_final": ep == lfinal, "n_trainable": n_tr,
                          "n_total": n_tot, "pct_trainable": pct,
                          "loss": llog.get(ep, "NA")})
                sub.append(m)
                print(f"  lora {name} feat{ep}: pxAUROC={m['pixel_auroc']:.4f} "
                      f"PRO={m['pro']:.4f}")
            sb = best_row(sub, exclude_pretrained=False)
            for r in sub:
                r["is_best"] = (sb is not None and r is sb)
            lrows += sub
        with open(os.path.join(metrics_dir, "lora_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=lc); w.writeheader()
            for r in lrows:
                w.writerow({k: r.get(k, "NA") for k in lc})
        print(f"[{args.category}] wrote lora_sweep.csv ({len(lrows)} rows)")

    print(f"[{args.category}] evaluation complete -> {metrics_dir}")


if __name__ == "__main__":
    main()
