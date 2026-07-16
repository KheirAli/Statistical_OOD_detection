"""Score one baseline on the 19 dvxray gun/scissors/bat positive images.

Twin of `eval_one_chaos_ct.py`, pointed at the dvxray shadow dir.

Usage:
    python tools/eval_one_dvxray.py <baseline> [--ckpt PATH] [--out JSON] [--samples ...]
"""
import argparse, json, sys, os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ood.baselines import build_baseline                                # noqa: E402
from ood.metrics import compute_snr_zscore, manual_auc, manual_average_precision, manual_roc_curve, compute_mask_l2  # noqa: E402
from ci_stats import bootstrap_mean_ci                                  # noqa: E402

OUT_ROOT_DEFAULT = Path("/data2/rohan/baseline_eval_results_v2/dvxray")

# DvXray test/anomaly + ground_truth/anomaly were populated with the 19 gun/scissors/bat
# positives + their SAM3-derived binary masks. Filenames pair as <id>_OL.png ↔ <id>_OL_mask.png.
TEST_IMGS = Path("/data2/rohan/datasets/mvtec_ad_dvxray/xray/test/anomaly")
GT_MASKS  = Path("/data2/rohan/datasets/mvtec_ad_dvxray/xray/ground_truth/anomaly")
DEFAULT_SAMPLES = [
    "P02330", "P02117", "P01818", "P01927", "P04469", "P04460",            # Bat
    "P01965", "P02052", "P01207", "P00209", "P01733", "P01115",            # Gun
    "P01861", "P02254", "P01950", "P03237", "P03262", "P02244", "P03921",  # Scissors (P03921 is multi-object)
]

CKPTS = {
    "simplenet":      "/data2/rohan/baseline_ckpts/simplenet_dvxray_5k/DvXray/simplenet_dvxray/run/models/0/mvtec_xray/ckpt.pth",
    "patchcore":      "/data2/rohan/baseline_ckpts/patchcore_dvxray_5k/DvXray_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_0/models/mvtec_xray",
    "supersimplenet": "/data2/rohan/baseline_ckpts/supersimplenet_dvxray_5k/superSimpleNet/checkpoints/mvtec/xray/1/weights.pt",
}


def _save_heatmap_outputs(out_dir: Path, sid: str, img: np.ndarray,
                          gt: np.ndarray, amap_raw: np.ndarray,
                          amap_smooth: np.ndarray) -> None:
    np.save(out_dir / f"{sid}_amap_raw.npy",    amap_raw.astype(np.float32))
    np.save(out_dir / f"{sid}_amap_smooth.npy", amap_smooth.astype(np.float32))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_min, a_max = float(amap_smooth.min()), float(amap_smooth.max())
    a_norm = (amap_smooth - a_min) / max(a_max - a_min, 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    axes[0].imshow(img); axes[0].set_title(f"input ({sid})"); axes[0].axis("off")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT mask"); axes[1].axis("off")
    axes[2].imshow(amap_raw, cmap="jet")
    axes[2].set_title(f"raw amap [{amap_raw.min():.2f}, {amap_raw.max():.2f}]")
    axes[2].axis("off")
    axes[3].imshow(img); axes[3].imshow(a_norm, cmap="jet", alpha=0.5)
    axes[3].set_title(f"σ=5 overlay [{a_min:.2f}, {a_max:.2f}]")
    axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sid}_panel.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", choices=list(CKPTS))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--invert_score", action="store_true",
                   help="negate anomaly map (for ckpts trained with reversed polarity)")
    p.add_argument("--save_heatmaps", action="store_true", default=True)
    p.add_argument("--no_save_heatmaps", dest="save_heatmaps", action="store_false")
    p.add_argument("--samples", nargs="+", default=None,
                   help="override DEFAULT_SAMPLES (P-IDs without _OL.png suffix)")
    p.add_argument("--test_dir", default=None,
                   help="override TEST_IMGS dir (e.g. .../test/scissors). "
                        "If --samples not given, auto-discovers from *_OL.png in this dir.")
    p.add_argument("--gt_dir", default=None,
                   help="override GT_MASKS dir (must pair with --test_dir).")
    p.add_argument("--out_root", default=None)
    args = p.parse_args()

    test_dir = Path(args.test_dir) if args.test_dir else TEST_IMGS
    gt_dir = Path(args.gt_dir) if args.gt_dir else GT_MASKS
    if args.samples:
        samples = args.samples
    elif args.test_dir:
        samples = sorted(p.stem.replace("_OL", "") for p in test_dir.glob("*_OL.png"))
    else:
        samples = DEFAULT_SAMPLES
    out_root = Path(args.out_root) if args.out_root else OUT_ROOT_DEFAULT

    cfg = {"ckpt": args.ckpt or CKPTS[args.baseline], "device": "cuda:0"}
    if args.baseline == "simplenet" and args.invert_score:
        cfg["invert_score"] = True

    print(f"=== {args.baseline} (dvxray, n={len(samples)}) ===", flush=True)
    b = build_baseline(args.baseline, cfg)

    baseline_dir = out_root / args.baseline
    heatmap_dir = baseline_dir / "heatmaps"
    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for sid in samples:
        img_path = test_dir / f"{sid}_OL.png"
        gt_path  = gt_dir / f"{sid}_OL_mask.png"
        img = np.array(Image.open(img_path).convert("RGB"))
        gt  = (np.array(Image.open(gt_path).convert("L")) > 0).astype(np.uint8)
        amap = b.score(img, gt)
        # Baselines emit a 256x256 amap regardless of input size. Resize GT (NN)
        # and viz img (bilinear) to amap shape — same convention as eval_baseline_mvtec.
        if gt.shape != amap.shape:
            gt = np.asarray(
                Image.fromarray((gt * 255).astype(np.uint8)).resize(
                    (amap.shape[1], amap.shape[0]), Image.NEAREST
                )
            )
            gt = (gt > 0).astype(np.uint8)
            img = np.asarray(
                Image.fromarray(img).resize(
                    (amap.shape[1], amap.shape[0]), Image.BILINEAR
                )
            )
        if args.invert_score and args.baseline != "simplenet":
            amap = -amap
        amap_s = gaussian_filter(amap, sigma=5.0, mode="nearest")
        fpr, tpr, _ = manual_roc_curve(gt.ravel(), amap_s.ravel())
        l2 = compute_mask_l2(amap_s, gt, normalize="minmax")
        m = {
            "sample": sid,
            "px_auc": float(manual_auc(fpr, tpr)),
            "px_ap":  float(manual_average_precision(gt.ravel(), amap_s.ravel())),
            "px_snr": float(compute_snr_zscore(amap_s, gt)),
            "mask_l2_sq": float(l2["l2_sq"]),
            "mask_mse":   float(l2["mse"]),
        }
        metrics.append(m)
        print(f"  {sid}: AUC={m['px_auc']:.3f}  AP={m['px_ap']:.3f}  SNR={m['px_snr']:.2f}", flush=True)

        if args.save_heatmaps:
            _save_heatmap_outputs(heatmap_dir, sid, img, gt, amap, amap_s)

    aggs = {}
    for k in ("px_auc", "px_ap", "px_snr", "mask_l2_sq", "mask_mse"):
        vals = [m[k] for m in metrics]
        mean, lo, hi, n = bootstrap_mean_ci(vals)
        aggs[k] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi, "n": n}

    out = {
        "baseline": args.baseline, "ckpt": cfg["ckpt"],
        "snr_definition": "zscore: (mean_OOD - mean_ID) / std_ID",
        "smooth_sigma": 5.0,
        "per_image": metrics, "aggregated": aggs,
    }
    out_path = Path(args.out) if args.out else baseline_dir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    print(f"  Px AUC  {aggs['px_auc']['mean']:.4f} [{aggs['px_auc']['ci95_lo']:.4f}, {aggs['px_auc']['ci95_hi']:.4f}]")
    print(f"  Px AP   {aggs['px_ap']['mean']:.4f} [{aggs['px_ap']['ci95_lo']:.4f}, {aggs['px_ap']['ci95_hi']:.4f}]")
    print(f"  Px SNR  {aggs['px_snr']['mean']:.4f} [{aggs['px_snr']['ci95_lo']:.4f}, {aggs['px_snr']['ci95_hi']:.4f}]")
    print(f"  Mask MSE {aggs['mask_mse']['mean']:.4f} [{aggs['mask_mse']['ci95_lo']:.4f}, {aggs['mask_mse']['ci95_hi']:.4f}] (lower=better)")


if __name__ == "__main__":
    main()
