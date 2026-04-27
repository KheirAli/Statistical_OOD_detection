"""Score one baseline on the 10 defective faces test images.

Run in its own Python invocation per baseline to avoid sys.path / module
namespace clashes (SuperSimpleNet ships a `common/` package that conflicts
with SimpleNet's `common.py` in the same process).

Usage:
    python tools/eval_one_faces.py <baseline> [--ckpt PATH] [--out JSON]
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
from ood.metrics import compute_snr, manual_auc, manual_average_precision, manual_roc_curve  # noqa: E402
from ci_stats import bootstrap_mean_ci                                  # noqa: E402

TEST_IMGS = Path("/data2/rohan/datasets/mvtec_ad_faces_5k/faces/test/random")
GT_MASKS  = Path("/data2/rohan/datasets/mvtec_ad_faces_5k/faces/ground_truth/random")
SAMPLES = ["40", "44", "49", "61", "65", "80", "92", "98", "107", "129"]

CKPTS = {
    "simplenet": "/data2/rohan/baseline_ckpts/simplenet_faces_70k/FacesAD_70k/simplenet_faces/run/models/0/mvtec_faces/ckpt.pth",
    "patchcore": "/data2/rohan/baseline_ckpts/patchcore_faces_5k/FacesAD_5k/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models/mvtec_faces",
    "supersimplenet": "/data2/rohan/baseline_ckpts/supersimplenet_faces_5k/superSimpleNet/checkpoints/mvtec/faces/1/weights.pt",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", choices=list(CKPTS))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--invert_score", action="store_true",
                   help="negate anomaly map (for ckpts trained with reversed polarity)")
    args = p.parse_args()

    cfg = {"ckpt": args.ckpt or CKPTS[args.baseline], "device": "cuda:0"}
    if args.baseline == "simplenet" and args.invert_score:
        cfg["invert_score"] = True

    print(f"=== {args.baseline} ===", flush=True)
    b = build_baseline(args.baseline, cfg)

    metrics = []
    for sid in SAMPLES:
        img = np.array(Image.open(TEST_IMGS / f"{sid}.png").convert("RGB"))
        gt = (np.array(Image.open(GT_MASKS / f"{sid}_mask.png").convert("L")) > 0).astype(np.uint8)
        amap = b.score(img, gt)
        if args.invert_score and args.baseline != "simplenet":
            # SimpleNet handles invert_score in the wrapper; for others we flip here.
            amap = -amap
        amap_s = gaussian_filter(amap, sigma=5.0, mode="nearest")
        fpr, tpr, _ = manual_roc_curve(gt.ravel(), amap_s.ravel())
        m = {
            "sample": sid,
            "px_auc": float(manual_auc(fpr, tpr)),
            "px_ap": float(manual_average_precision(gt.ravel(), amap_s.ravel())),
            "px_snr": float(compute_snr(amap_s, gt)),
        }
        metrics.append(m)
        print(f"  {sid}: AUC={m['px_auc']:.3f}  AP={m['px_ap']:.3f}  SNR={m['px_snr']:.2f}", flush=True)

    aggs = {}
    for k in ("px_auc", "px_ap", "px_snr"):
        vals = [m[k] for m in metrics]
        mean, lo, hi, n = bootstrap_mean_ci(vals)
        aggs[k] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi, "n": n}

    out = {"baseline": args.baseline, "ckpt": cfg["ckpt"], "per_image": metrics, "aggregated": aggs}
    out_path = Path(args.out) if args.out else ROOT / f"results_eval/faces_70k/{args.baseline}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    print(f"  Px AUC  {aggs['px_auc']['mean']:.4f} [{aggs['px_auc']['ci95_lo']:.4f}, {aggs['px_auc']['ci95_hi']:.4f}]")
    print(f"  Px AP   {aggs['px_ap']['mean']:.4f} [{aggs['px_ap']['ci95_lo']:.4f}, {aggs['px_ap']['ci95_hi']:.4f}]")
    print(f"  Px SNR  {aggs['px_snr']['mean']:.4f} [{aggs['px_snr']['ci95_lo']:.4f}, {aggs['px_snr']['ci95_hi']:.4f}]")


if __name__ == "__main__":
    main()
