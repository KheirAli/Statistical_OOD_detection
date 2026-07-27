"""Anomalib-style metrics (AUPRO, pooled px AUROC/AUPR/F1Max, per-image F1Max,
per-image AUROC) for CT (body-masked) and MVTec (15 cats, defect subcats only).

Methods: ours (delta-map dumps), patchcore/simplenet/supersimplenet (saved
sigma-5 heatmaps), DDAD (CT only, saved maps).
Pixels outside the valid/body mask are excluded everywhere.
"""
import numpy as np, glob, os, re, sys, json, torch
from PIL import Image
from scipy.ndimage import label as cc_label, gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score

SCRATCH = os.path.dirname(os.path.abspath(__file__))
BASE = "/data2/rohan/baseline_eval_results_v2"
MVTEC = "/data/akheirandish3/mvtec_ad"
CT_GT = "/data2/akheirandish3/id_new_warped_images/masks_ood_med_med"
EIGHT = np.ones((3, 3), dtype=int)
CATS = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut",
        "pill","screw","tile","toothbrush","transistor","wood","zipper"]
SKIP_SUB = ("good", "combined")


def resize_mask(m, shape):
    if m.shape != shape:
        m = np.array(Image.fromarray(m.astype(np.uint8)).resize((shape[1], shape[0]), Image.NEAREST))
    return m


def f1max_counts(scores, labels, bins=8192, lo=None, hi=None):
    lo = scores.min() if lo is None else lo
    hi = scores.max() if hi is None else hi
    h_pos, _ = np.histogram(scores[labels == 1], bins=bins, range=(lo, hi))
    h_neg, _ = np.histogram(scores[labels == 0], bins=bins, range=(lo, hi))
    tp = np.cumsum(h_pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(h_neg[::-1])[::-1].astype(np.float64)
    fn = h_pos.sum() - tp
    return float((2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).max())


def aupro(maps, gts, valids, fpr_limit=0.3, n_thr=600):
    normal = np.concatenate([m[(g == 0) & v] for m, g, v in zip(maps, gts, valids)])
    ns = np.sort(normal); n = len(ns)
    fprs = np.linspace(0, fpr_limit, n_thr + 1)[1:]
    thr = ns[np.clip((n * (1 - fprs)).astype(int), 0, n - 1)]
    fpr_act = 1.0 - np.searchsorted(ns, thr, side="left") / n
    regions = []
    for m, g, v in zip(maps, gts, valids):
        lab, k = cc_label((g > 0) & v, structure=EIGHT)
        for r in range(1, k + 1):
            regions.append(np.sort(m[lab == r].ravel()))
    if not regions:
        return float("nan"), 0
    pro = np.zeros(len(thr))
    for rs in regions:
        pro += 1.0 - np.searchsorted(rs, thr, side="left") / len(rs)
    pro /= len(regions)
    o = np.argsort(fpr_act)
    x, y = np.concatenate([[0.0], fpr_act[o]]), None
    y = np.concatenate([[pro[o][0]], pro[o]])
    return float(np.trapezoid(y, x) / fpr_limit), len(regions)


def dataset_metrics(maps, gts, valids):
    scores = np.concatenate([m[v] for m, v in zip(maps, valids)])
    labels = np.concatenate([(g > 0).astype(np.uint8)[v] for g, v in zip(gts, valids)])
    res = dict(
        pooled_auroc=roc_auc_score(labels, scores),
        pooled_aupr=average_precision_score(labels, scores),
        pooled_f1max=f1max_counts(scores, labels),
    )
    res["aupro"], res["n_regions"] = aupro(maps, gts, valids)
    per_auc, per_f1 = [], []
    for m, g, v in zip(maps, gts, valids):
        y = (g > 0).astype(np.uint8)[v]; s = m[v]
        if y.max() == 0 or y.min() == 1:
            continue
        per_auc.append(roc_auc_score(y, s))
        per_f1.append(f1max_counts(s, y))
    res["per_image_auroc"] = float(np.mean(per_auc))
    res["per_image_f1max"] = float(np.mean(per_f1))
    res["n_images"] = len(per_auc)
    return res


# ── loaders ──────────────────────────────────────────────────────────────────
def ours_mvtec(cat):
    maps, gts, valids = [], [], []
    for f in sorted(glob.glob(f"{SCRATCH}/mvtec_maps/{cat}/*.npz")):
        sub = os.path.basename(f).split("__")[0]
        if any(s in sub for s in SKIP_SUB):
            continue
        d = np.load(f)
        sm = gaussian_filter(np.nan_to_num(d["delta_map"], nan=0.0), 5.0, mode="nearest")
        maps.append(sm.astype(np.float32)); gts.append(d["gt_mask"])
        valids.append(d["valid_mask"].astype(bool))
    return maps, gts, valids


def baseline_mvtec(method, cat):
    maps, gts, valids = [], [], []
    for f in sorted(glob.glob(f"{BASE}/mvtec_full/{method}_{cat}/heatmaps/*_amap_smooth.npy")):
        stem = os.path.basename(f).replace("_amap_smooth.npy", "")
        sub, idx = stem.rsplit("_", 1)
        if any(s in sub for s in SKIP_SUB):
            continue
        m = np.load(f).astype(np.float32)
        gp = f"{MVTEC}/{cat}/ground_truth/{sub}/{idx}_mask.png"
        if not os.path.exists(gp):
            continue
        g = resize_mask((np.array(Image.open(gp).convert("L")) > 127).astype(np.uint8), m.shape)
        maps.append(m); gts.append(g); valids.append(np.ones(m.shape, bool))
    return maps, gts, valids


def ct_body(pid, shape):
    p = f"/data2/akheirandish3/id_new_warped_images/masks_body_ood_default_large/image_{pid}_mask.png"
    b = np.array(Image.open(p).convert("L")) > 0
    return resize_mask(b, shape).astype(bool)


def ours_ct():
    maps, gts, valids = [], [], []
    for f in sorted(glob.glob(f"{SCRATCH}/ct_maps/*.npz")):
        d = np.load(f)
        sm = gaussian_filter(np.nan_to_num(d["delta_map"], nan=0.0), 5.0, mode="nearest")
        maps.append(sm.astype(np.float32)); gts.append(d["gt_mask"])
        valids.append(d["valid_mask"].astype(bool))
    return maps, gts, valids


def baseline_ct(method):
    maps, gts, valids = [], [], []
    for f in sorted(glob.glob(f"{BASE}/chaos_ct/{method}/heatmaps/*_amap_smooth.npy")):
        pid = re.search(r"image_(\d+)", f).group(1)
        m = np.load(f).astype(np.float32)
        g = resize_mask((np.array(Image.open(f"{CT_GT}/image_{pid}_mask.png").convert("L")) > 127).astype(np.uint8), m.shape)
        b = np.array(Image.open(f.replace("_amap_smooth.npy", "_body_mask.png")).convert("L")) > 0
        maps.append(m); gts.append(g); valids.append(resize_mask(b, m.shape).astype(bool))
    return maps, gts, valids


def ddad_ct(smooth=True):
    d = torch.load("/data/akherandish3/Statistical_OOD_detection/DDAD/hparam_sweep/ct_maps_reference.pt",
                   map_location="cpu", weights_only=False)
    maps, gts, valids = [], [], []
    for m, g, f in zip(d["maps"], d["gts"], d["files"]):
        m = m.squeeze().numpy().astype(np.float32)
        if smooth:
            m = gaussian_filter(m, 5.0, mode="nearest")
        g = (g.squeeze().numpy() > 0.5).astype(np.uint8)
        pid = re.search(r"image_(\d+)", f).group(1)
        maps.append(m); gts.append(g); valids.append(ct_body(pid, m.shape))
    return maps, gts, valids


if __name__ == "__main__":
    which = sys.argv[1]
    out = {}
    if which == "ct":
        for name, loader in [("ours", ours_ct),
                             ("patchcore", lambda: baseline_ct("patchcore")),
                             ("simplenet", lambda: baseline_ct("simplenet")),
                             ("supersimplenet", lambda: baseline_ct("supersimplenet")),
                             ("ddad_reference_s5", lambda: ddad_ct(True)),
                             ("ddad_reference_raw", lambda: ddad_ct(False))]:
            r = dataset_metrics(*loader()); out[name] = r
            print(f"{name:20s} n={r['n_images']:4d} reg={r['n_regions']:4d} AUPRO={r['aupro']:.4f} "
                  f"pAUROC={r['pooled_auroc']:.4f} pAUPR={r['pooled_aupr']:.4f} "
                  f"pF1Max={r['pooled_f1max']:.4f} piF1Max={r['per_image_f1max']:.4f} piAUROC={r['per_image_auroc']:.4f}")
    elif which == "mvtec":
        methods = {"ours": ours_mvtec,
                   "patchcore": lambda c: baseline_mvtec("patchcore", c),
                   "simplenet": lambda c: baseline_mvtec("simplenet", c),
                   "supersimplenet": lambda c: baseline_mvtec("supersimplenet", c)}
        for name, loader in methods.items():
            out[name] = {}
            for cat in CATS:
                maps, gts, valids = loader(cat)
                if not maps:
                    print(f"{name}/{cat}: NO MAPS"); continue
                r = dataset_metrics(maps, gts, valids); out[name][cat] = r
                print(f"{name:15s} {cat:12s} n={r['n_images']:4d} AUPRO={r['aupro']:.4f} "
                      f"pAUROC={r['pooled_auroc']:.4f} pAUPR={r['pooled_aupr']:.4f} "
                      f"pF1Max={r['pooled_f1max']:.4f} piF1Max={r['per_image_f1max']:.4f} piAUROC={r['per_image_auroc']:.4f}")
            done = [c for c in CATS if c in out[name]]
            mean = {k: float(np.mean([out[name][c][k] for c in done]))
                    for k in ("aupro","pooled_auroc","pooled_aupr","pooled_f1max","per_image_f1max","per_image_auroc")}
            out[name]["MEAN"] = mean
            print(f"{name:15s} {'MEAN('+str(len(done))+')':12s}        AUPRO={mean['aupro']:.4f} "
                  f"pAUROC={mean['pooled_auroc']:.4f} pAUPR={mean['pooled_aupr']:.4f} "
                  f"pF1Max={mean['pooled_f1max']:.4f} piF1Max={mean['per_image_f1max']:.4f} piAUROC={mean['per_image_auroc']:.4f}")
    json.dump(out, open(f"{SCRATCH}/anomalib_metrics_{which}_full.json", "w"), indent=1)
    print("saved", f"{SCRATCH}/anomalib_metrics_{which}_full.json")
