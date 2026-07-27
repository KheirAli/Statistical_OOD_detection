"""Anomalib-style pixel metrics (pooled AUROC/AUPR/F1Max + AUPRO fpr<=0.3)
for the 50-image xray scissors set, per method."""
import numpy as np, glob, os, sys, torch
from PIL import Image
from scipy.ndimage import label as cc_label, gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = "/data/akherandish3/Statistical_OOD_detection"
B1 = "/data2/rohan/baseline_eval_results_v2/dvxray_scissors"
B2 = f"{REPO}/results_eval_xray_baselines_b2"
GTD = "/data/akheirandish3/mvtec_ad/xray/ground_truth/scissors"
SCRATCH = os.path.dirname(os.path.abspath(__file__))

EIGHT = np.ones((3, 3), dtype=int)


def load_gt(patient, shape):
    m = np.array(Image.open(f"{GTD}/{patient}_OL_mask.png").convert("L"))
    if m.shape != shape:
        m = np.array(Image.fromarray(m).resize((shape[1], shape[0]), Image.NEAREST))
    return (m > 127).astype(np.uint8)


def feature_baseline(method):
    maps, gts, names = [], [], []
    for root in [f"{B1}/{method}/heatmaps", f"{B2}/{method}/heatmaps"]:
        for f in sorted(glob.glob(root + "/*_amap_smooth.npy")):
            p = os.path.basename(f).replace("_amap_smooth.npy", "")
            a = np.load(f).astype(np.float32)
            maps.append(a); gts.append(load_gt(p, a.shape)); names.append(p)
    return maps, gts, names


def ddad(which):
    d = torch.load(f"{B2}/ddad_x50_{which}_maps.pt", map_location="cpu", weights_only=False)
    maps = [m.squeeze().numpy().astype(np.float32) for m in d["maps"]]
    gts = [(g.squeeze().numpy() > 0.5).astype(np.uint8) for g in d["gts"]]
    names = [os.path.basename(f).split(".")[0] for f in d["files"]]
    return maps, gts, names


def ours():
    maps, gts, names = [], [], []
    for sub in ["old", "b2"]:
        for f in sorted(glob.glob(f"{SCRATCH}/xray_maps/{sub}/*.npz")):
            d = np.load(f)
            delta = np.nan_to_num(d["delta_map"], nan=0.0)
            sm = gaussian_filter(delta, sigma=5.0, mode="nearest")
            valid = d["valid_mask"].astype(bool)
            sm = np.where(valid, sm, sm[valid].min())  # invalid pixels -> lowest score
            maps.append(sm.astype(np.float32)); gts.append(d["gt_mask"].astype(np.uint8))
            names.append(os.path.basename(f)[:-4])
    return maps, gts, names


def aupro(maps, gts, fpr_limit=0.3, n_thr=600):
    normal = np.concatenate([m[g == 0].ravel() for m, g in zip(maps, gts)])
    normal_sorted = np.sort(normal)
    n_norm = len(normal_sorted)
    # thresholds giving FPR uniformly in (0, fpr_limit]
    fprs_target = np.linspace(0, fpr_limit, n_thr + 1)[1:]
    idx = np.clip((n_norm * (1 - fprs_target)).astype(int), 0, n_norm - 1)
    thr = normal_sorted[idx]
    fpr_actual = 1.0 - np.searchsorted(normal_sorted, thr, side="left") / n_norm
    regions = []
    for m, g in zip(maps, gts):
        lab, k = cc_label(g, structure=EIGHT)
        for r in range(1, k + 1):
            regions.append(np.sort(m[lab == r].ravel()))
    pro = np.zeros(len(thr))
    for rs in regions:
        pro += 1.0 - np.searchsorted(rs, thr, side="left") / len(rs)
    pro /= len(regions)
    order = np.argsort(fpr_actual)
    x, y = fpr_actual[order], pro[order]
    x = np.concatenate([[0.0], x]); y = np.concatenate([[y[0] if len(y) else 0.0], y])
    return float(np.trapezoid(y, x) / fpr_limit), len(regions)


def f1max(scores, labels, bins=8192):
    lo, hi = scores.min(), scores.max()
    h_pos, edges = np.histogram(scores[labels == 1], bins=bins, range=(lo, hi))
    h_neg, _ = np.histogram(scores[labels == 0], bins=bins, range=(lo, hi))
    tp = np.cumsum(h_pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(h_neg[::-1])[::-1].astype(np.float64)
    fn = h_pos.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    return float(f1.max())


def evaluate(name, maps, gts, names):
    scores = np.concatenate([m.ravel() for m in maps])
    labels = np.concatenate([g.ravel() for g in gts])
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    f1 = f1max(scores, labels)
    pro, n_reg = aupro(maps, gts)
    per_img = np.mean([roc_auc_score(g.ravel(), m.ravel()) for m, g in zip(maps, gts)])
    print(f"{name:22s} n={len(maps):2d} regions={n_reg:3d} | pooled AUROC={auroc:.4f} "
          f"AUPR={aupr:.4f} F1Max={f1:.4f} AUPRO={pro:.4f} | per-img AUROC mean={per_img:.4f}")
    return dict(method=name, n=len(maps), auroc=auroc, aupr=aupr, f1max=f1,
                aupro=pro, per_image_auroc=per_img)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "baselines"
    results = []
    if which in ("baselines", "all"):
        for m in ["patchcore", "simplenet", "supersimplenet"]:
            results.append(evaluate(m, *feature_baseline(m)))
        for w in ["reference", "v7"]:
            for smooth in (False, True):
                maps, gts, names = ddad(w)
                if smooth:
                    maps = [gaussian_filter(m, 5.0, mode="nearest") for m in maps]
                results.append(evaluate(f"ddad_{w}{'_s5' if smooth else '_raw'}", maps, gts, names))
    if which in ("ours", "all"):
        results.append(evaluate("ours_e2e_seeded_s5", *ours()))
    import json
    out = f"{SCRATCH}/anomalib_metrics_{which}.json"
    json.dump(results, open(out, "w"), indent=1)
    print("saved", out)
