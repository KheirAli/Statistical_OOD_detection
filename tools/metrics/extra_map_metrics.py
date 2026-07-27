"""Per-image px AP / SNR / mask-MSE / PSNR from saved anomaly maps.

This script reproduces the "extra metric" cells of the all-method comparison
tables (MVTec / CT / xray-50 / faces) for methods whose per-image maps are on
disk but whose metrics.json does not carry every metric — chiefly DDAD (saved
raw maps) and GLASS (saved smoothed heatmaps), plus our own delta-map dumps.

Metric definitions (identical to the recorded baseline protocol):
  - px AP   : average precision over pixels of one image, computed with
              ood.metrics.manual_average_precision (same impl as the recorded
              baseline metrics.json values — validated bit-exact on xray b1).
  - px SNR  : z-score separation on the smoothed map,
                  (mean over GT-anomalous px - mean over normal px) / std(normal px).
              NOTE: this is a z-score, NOT the ratio mean_OOD/mean_ID.
  - mask-MSE: min-max normalize the smoothed map to [0,1] over the valid
              region, then mean squared error against the binary GT mask over
              valid pixels.  (The eval-harness JSON stores this same quantity
              under its -- misleadingly named -- "psnr" key.)
  - PSNR    : mean over images of 10*log10(1 / mask-MSE_i), in dB.

Smoothing: every map is Gaussian-smoothed with sigma=5 before scoring
(feature-baseline heatmaps on disk are already smoothed; DDAD .pt maps and our
delta-map npz dumps are raw, so they are smoothed here).

Valid region: CT metrics are restricted to the per-image body mask; all other
datasets use every pixel.  MVTec aggregates per-image -> per-category mean
(defect images only, /good/ excluded) -> unweighted mean over the 15
categories; other datasets are plain means over images.

Map sources:
  DDAD      : DDAD/hparam_sweep/mvtec_maps/{reference,v7}_<cat>.pt,
              DDAD/hparam_sweep/ct_maps_reference.pt,
              DDAD/hparam_sweep/faces_maps.pt (official) and
              DDAD/hparam_sweep/mvtec_maps/facesv7_faces.pt (v7),
              results_eval_xray_baselines_b2/ddad_x50_{reference,v7}_maps.pt
              (each: {"maps": [N,1,256,256], "gts": ..., "files": [...]})
  baselines : /data2/rohan/baseline_eval_results_v2/<ds>/<method>/heatmaps/
              *_amap_smooth.npy  (patchcore / simplenet / supersimplenet / glass)
  ours      : results_eval_anomalib_metrics/ours_*_maps*/ *.npz dumps written
              by evaluate.py when DUMP_DELTA_DIR is set
              (keys: delta_map raw, gt_mask, valid_mask).

Usage (env `ood`, repo root):
  python tools/metrics/extra_map_metrics.py mvtec-ddad
  python tools/metrics/extra_map_metrics.py mvtec-baselines-mse
  python tools/metrics/extra_map_metrics.py mvtec-ours
  python tools/metrics/extra_map_metrics.py ct
  python tools/metrics/extra_map_metrics.py faces
  python tools/metrics/extra_map_metrics.py xray-ddad-snr
  python tools/metrics/extra_map_metrics.py xray-glass-mse
"""
import glob
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ood.metrics import manual_average_precision

BASE = "/data2/rohan/baseline_eval_results_v2"
MVTEC = "/data/akheirandish3/mvtec_ad"
AM = "results_eval_anomalib_metrics"
CT_GT = "/data2/akheirandish3/id_new_warped_images/masks_ood_med_med"
CT_BODY = "/data2/akheirandish3/id_new_warped_images/masks_body_ood_default_large"
XRAY_GT = f"{MVTEC}/xray/ground_truth/scissors"
CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
        "wood", "zipper"]
S3_CATS = ("bottle", "leather", "metal_nut", "pill")  # best-of suffix _3 maps


def resize_mask(m, shape):
    if m.shape != shape:
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255)
                     .resize((shape[1], shape[0]), Image.NEAREST)) > 127
    return m.astype(bool)


def px_ap(m, gt, valid):
    return manual_average_precision((gt > 0)[valid].astype(np.uint8).ravel(),
                                    m[valid].ravel())


def px_snr(m, gt, valid):
    pos, neg = m[(gt > 0) & valid], m[(gt == 0) & valid]
    return float((pos.mean() - neg.mean()) / (neg.std() + 1e-12))


def px_snr_paper(m, gt, valid):
    """Paper definition: ratio of mean heatmap value on OOD vs ID pixels,
    on the min-max normalized map (keeps both means non-negative)."""
    mv = m[valid]
    mm = np.empty_like(m)
    mm[valid] = (mv - mv.min()) / (mv.max() - mv.min() + 1e-12)
    pos, neg = mm[(gt > 0) & valid], mm[(gt == 0) & valid]
    return float(pos.mean() / (neg.mean() + 1e-12))


def mask_mse(m, gt, valid):
    mv, gv = m[valid], (gt > 0)[valid]
    mm = (mv - mv.min()) / (mv.max() - mv.min() + 1e-12)
    return float(((mm - gv) ** 2).mean())


def image_row(m, gt, valid=None):
    if valid is None:
        valid = np.ones(m.shape, bool)
    return (px_ap(m, gt, valid), px_snr(m, gt, valid), mask_mse(m, gt, valid),
            px_snr_paper(m, gt, valid))


def report(name, rows):
    """rows: list of (ap, snr, mse, snr_paper) per image."""
    a = np.array(rows)
    psnr = np.mean(10 * np.log10(1 / np.maximum(a[:, 2], 1e-12)))
    print(f"{name:24s} n={len(a):4d}  AP={a[:,0].mean():.4f}  "
          f"SNR={a[:,1].mean():.2f}  SNRpaper={a[:,3].mean():.2f}  "
          f"MSE={a[:,2].mean():.4f}  PSNR={psnr:.2f}")
    return a


def smooth5(m):
    return gaussian_filter(np.asarray(m).squeeze().astype(np.float32), 5.0)


def ddad_pt_rows(path, body_masked=False, skip_good=False):
    d = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for m, g, f in zip(d["maps"], d["gts"], d["files"]):
        if skip_good and "/good/" in f:
            continue
        m = smooth5(m)
        g = (np.asarray(g).squeeze() > 0.5).astype(np.uint8)
        if g.sum() == 0:
            continue
        valid = None
        if body_masked:
            pid = os.path.basename(f).split(".")[0].replace("image_", "")
            valid = resize_mask(
                np.array(Image.open(f"{CT_BODY}/image_{pid}_mask.png").convert("L")) > 0,
                m.shape)
        rows.append(image_row(m, g, valid))
    return rows


def ours_npz_rows(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f)
        sm = gaussian_filter(np.nan_to_num(d["delta_map"], nan=0.0), 5.0,
                             mode="nearest").astype(np.float32)
        if d["gt_mask"].sum() == 0:
            continue
        rows.append(image_row(sm, d["gt_mask"], d["valid_mask"].astype(bool)))
    return rows


def catmean(per_cat_rows, name):
    """per_cat_rows: {cat: [(ap,snr,mse,snr_paper), ...]} -> per-cat mean -> 15-cat mean."""
    stats = []
    for cat, rows in per_cat_rows.items():
        a = np.array(rows)
        stats.append([a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(),
                      np.mean(10 * np.log10(1 / np.maximum(a[:, 2], 1e-12))),
                      a[:, 3].mean()])
    s = np.mean(stats, axis=0)
    print(f"{name:24s} 15-cat mean  AP={s[0]:.4f}  SNR={s[1]:.2f}  "
          f"SNRpaper={s[4]:.2f}  MSE={s[2]:.4f}  PSNR={s[3]:.2f}")


def main():
    which = sys.argv[1]

    if which == "mvtec-ddad":
        for variant in ["reference", "v7"]:
            per_cat = {}
            for cat in CATS:
                per_cat[cat] = ddad_pt_rows(
                    f"DDAD/hparam_sweep/mvtec_maps/{variant}_{cat}.pt",
                    skip_good=True)
            catmean(per_cat, f"ddad_{variant}")

    elif which == "mvtec-baselines-mse":
        # AP/SNR for these methods are already recorded in
        # {method}_per_category.csv; this recomputes everything from the
        # stored heatmaps (they are already sigma-5 smoothed).
        for meth in ["patchcore", "simplenet", "supersimplenet"]:
            per_cat = {}
            for cat in CATS:
                rows = []
                for f in sorted(glob.glob(
                        f"{BASE}/mvtec_full/{meth}_{cat}/heatmaps/*_amap_smooth.npy")):
                    stem = os.path.basename(f)[:-len("_amap_smooth.npy")]
                    sub, idx = stem.rsplit("_", 1)
                    if "good" in sub or "combined" in sub:
                        continue
                    gp = f"{MVTEC}/{cat}/ground_truth/{sub}/{idx}_mask.png"
                    if not os.path.exists(gp):
                        continue
                    m = np.load(f).astype(np.float32)
                    g = resize_mask(
                        np.array(Image.open(gp).convert("L")) > 127, m.shape)
                    rows.append(image_row(m, g.astype(np.uint8)))
                per_cat[cat] = rows
            catmean(per_cat, meth)

    elif which == "mvtec-ours":
        per_cat = {}
        for cat in CATS:
            root = (f"{AM}/ours_mvtec_maps_s3/{cat}" if cat in S3_CATS
                    else f"{AM}/ours_mvtec_maps/{cat}")
            rows = []
            for f in sorted(glob.glob(f"{root}/*.npz")):
                sub = os.path.basename(f).split("__")[0]
                if "good" in sub or "combined" in sub:
                    continue
                d = np.load(f)
                sm = gaussian_filter(np.nan_to_num(d["delta_map"], nan=0.0),
                                     5.0, mode="nearest").astype(np.float32)
                rows.append(image_row(sm, d["gt_mask"],
                                      d["valid_mask"].astype(bool)))
            per_cat[cat] = rows
        catmean(per_cat, "ours_bestof")

    elif which == "ct":
        report("ddad_reference",
               ddad_pt_rows("DDAD/hparam_sweep/ct_maps_reference.pt",
                            body_masked=True))
        for maps_dir, name in [("ours_ct_maps", "ours_seededAE"),
                               ("ours_ct_maps_released", "ours_releasedAE")]:
            if glob.glob(f"{AM}/{maps_dir}/*.npz"):
                report(name, ours_npz_rows(f"{AM}/{maps_dir}/*.npz"))
        for meth in ["patchcore", "simplenet", "supersimplenet", "glass"]:
            rows = []
            for f in sorted(glob.glob(
                    f"{BASE}/chaos_ct/{meth}/heatmaps/*_amap_smooth.npy")):
                pid = re.search(r"(\d+)_amap", os.path.basename(f)).group(1)
                m = np.load(f).astype(np.float32)
                g = resize_mask(
                    np.array(Image.open(
                        f"{CT_GT}/image_{pid}_mask.png").convert("L")) > 127,
                    m.shape).astype(np.uint8)
                b = resize_mask(
                    np.array(Image.open(
                        f.replace("_amap_smooth.npy", "_body_mask.png"))
                        .convert("L")) > 0, m.shape)
                rows.append(image_row(m, g, b))
            report(meth, rows)

    elif which == "faces":
        gtd = f"{MVTEC}/faces/ground_truth/random"
        for meth in ["patchcore", "simplenet", "supersimplenet", "glass"]:
            rows = []
            for f in sorted(glob.glob(
                    f"{BASE}/faces/{meth}/heatmaps/*_amap_smooth.npy")):
                pid = os.path.basename(f)[:-len("_amap_smooth.npy")]
                gp = f"{gtd}/{pid}_mask.png"
                if not os.path.exists(gp):
                    continue  # 2 of 24 faces have no GT mask
                m = np.load(f).astype(np.float32)
                g = resize_mask(
                    np.array(Image.open(gp).convert("L")) > 127,
                    m.shape).astype(np.uint8)
                if g.sum() == 0:
                    continue
                rows.append(image_row(m, g))
            report(meth, rows)
        report("ddad_official",
               ddad_pt_rows("DDAD/hparam_sweep/faces_maps.pt"))
        report("ddad_v7",
               ddad_pt_rows("DDAD/hparam_sweep/mvtec_maps/facesv7_faces.pt"))
        report("ours_seeded", ours_npz_rows(f"{AM}/ours_faces_maps/*.npz"))

    elif which == "xray-ddad-snr":
        # The stored xray50_extra_metrics.json triples have SNR = NaN for
        # DDAD; recompute from the saved x50 maps.  Note: the recorded DDAD
        # x50 mask-MSE (0.137/0.139) was computed on the maps as stored
        # (unsmoothed); the extra sigma-5 here shifts MSE slightly while
        # leaving AP/SNR essentially unchanged.
        for name in ["reference", "v7"]:
            rows = ddad_pt_rows(
                f"results_eval_xray_baselines_b2/ddad_x50_{name}_maps.pt")
            report(f"ddad_{name}", rows)

    elif which == "xray-glass-mse":
        # GLASS has b1 maps only (n=32); heatmaps already sigma-5 smoothed.
        rows = []
        for f in sorted(glob.glob(
                f"{BASE}/dvxray_scissors/glass/heatmaps/*_amap_smooth.npy")):
            p = os.path.basename(f)[:-len("_amap_smooth.npy")]
            m = np.load(f).astype(np.float32)
            g = resize_mask(
                np.array(Image.open(f"{XRAY_GT}/{p}_OL_mask.png").convert("L")) > 127,
                m.shape).astype(np.uint8)
            rows.append(image_row(m, g))
        report("glass_b1", rows)

    else:
        raise SystemExit(f"unknown target: {which} (see module docstring)")


if __name__ == "__main__":
    main()
