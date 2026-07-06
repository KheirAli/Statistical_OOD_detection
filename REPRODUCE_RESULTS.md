# Reproducing the Best-AE Evaluation Results

This document describes how to reproduce our headline results: the typical-set /
superpixel OOD detector with a **per-category PixelAutoEncoder (1792 → 3, learned
feature reduction) on an ImageNet-pretrained ResNet-101 backbone**, evaluated on
all 15 MVTec-AD categories plus CT scans, chest X-ray, and FFHQ faces.

All numbers below are **pixel-level ROC-AUC at Gaussian smoothing sigma = 5**,
averaged over defect subcategories (excluding `good`/`combined`). Every command is
evaluation-only: it consumes precomputed DPS reconstructions and trained AE
checkpoints, so results are deterministic (they reproduce bit-for-bit).

## 1. Environment

```bash
conda activate ood        # Python 3.10, torch, torchvision, scikit-image, scipy, pyyaml
cd <repo root>
```

## 2. Required inputs

### 2.1 DPS reconstructions (evaluation inputs)

One directory per (category, subcategory) at the repo root:
`./results_patches_ddad_native_<category>_<subcategory><suffix>/` — produced by the
DPS inpainting stage (`sample_batch.py` / the sampling configs in `configs/`).
Suffix `_1` is used for all MVTec categories except **pill**, which uses `_3`;
CT uses no suffix; xray and faces use `_1`.

### 2.2 Checkpoint inventory (every AE + backbone)

| Target | AE checkpoint | Backbone |
|---|---|---|
| bottle, cable, capsule, grid, hazelnut, leather, metal_nut, screw, tile, toothbrush, transistor, zipper | `all_categories_embedder/models/pixel_ae_pretrained/<cat>.pth` | `experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/<cat>/checkpoints/feat0.pth` (ImageNet anchor, injected via `DDAD_FE_WEIGHTS`) |
| carpet | `all_categories_embedder/models/pixel_ae_pretrained_fixed/carpet.pth` (corrected seeded recipe) | same as above (`carpet/checkpoints/feat0.pth`) |
| wood | `all_categories_embedder/models/pixel_ae_pretrained_fixed/wood.pth` (corrected seeded recipe) | same as above (`wood/checkpoints/feat0.pth`) |
| pill | `models/mvtec_embedders/pixel_autoencoder_pill.pth` | torchvision ResNet-101 (default in `evaluate.py`, **no** weight injection) |
| CT | `models/pixel_autoencoder_CT.pth` | torchvision ResNet-101 |
| xray (scissors) | `all_categories_embedder/models/xray_mine/last_layer.pth` | `experiments/xray_finetune_mine/last_layer/checkpoints/feat4.pth` (DDAD-DA last-layer fine-tune, 4 epochs) |
| faces (FFHQ) | `models/pixel_autoencoder_faces.pth` | torchvision ResNet-101 |

Notes:
- `.pth` files are not tracked in git — package them alongside a release (or a
  download link) when publishing.
- The carpet/wood AEs come from the corrected AE recipe (seeded,
  `--ae_loss cos` + `--select_loss mse_var`, best-of-3 restarts). They can be
  regenerated bit-exact with
  `CATS="carpet wood" bash all_categories_embedder/rerun_all15_fixed_ae.sh`.
- Ground-truth masks are read from `/data/akheirandish3/mvtec_ad/<category>/ground_truth/`
  (edit `MVTEC` in `reproduce_results.sh` for a different data root).

## 3. Run

```bash
GPU=0 bash reproduce_results.sh              # all 18 targets, sequentially
GPU=0 bash reproduce_results.sh carpet pill  # any subset
```

The script dispatches each target to the right pipeline:
- **14 MVTec categories** → `all_categories_embedder/evaluate_ddad_fe.py` with the
  feat0 backbone injected via `DDAD_FE_WEIGHTS` and the per-category AE
  (`--n_pca 3 --bins_pca 32 --bins_rgb 32 --superpixel_target_size 30
  --smooth_sigma 1 --max_reconstructions 50 --results_suffix _1`).
- **pill** → plain `evaluate.py` (torchvision backbone), original AE, suffix `_3`.
- **CT** → `evaluate.py`, CT AE, `--bins_pca 64 --bins_rgb 64 --sigma_rohan 1.0
  --max_reconstructions 40`, no suffix.
- **xray** → `evaluate_ddad_fe.py`, last-layer fine-tuned backbone + its AE,
  `--superpixel_target_size 10 --max_reconstructions 40`, subcategory `scissors`.
- **faces** → `evaluate.py`, faces AE, `--sigma_rohan 1.0`, suffix `_1`.

Outputs: `./results_eval_reproduction/<target>/` with one JSON per subcategory and
an `*_summary.json` per category.

## 4. Summarize

```bash
python - <<'EOF'
import json, glob, statistics as st
for s in sorted(glob.glob("results_eval_reproduction/*/*summary.json")):
    d = json.load(open(s))
    rows = [r for r in d["rows"] if r["eval_sigma"] == "sigma_5.0"
            and r["subcategory"] not in ("good", "combined")]
    if rows:
        print(f'{d["category"]:12s} px ROC-AUC (s5) = {st.mean(r["px_roc_auc"] for r in rows):.4f}')
# CT/xray/faces single-subcategory runs may store results without a summary file:
for f in sorted(glob.glob("results_eval_reproduction/{CT,xray,faces}/*.json")):
    d = json.load(open(f))
    if "averaged" in d and d["averaged"]:
        print(f'{d["category"]:12s} px ROC-AUC (s5) = {d["averaged"]["sigma_5.0"]["px_roc_auc"]:.4f}')
EOF
```

## 5. Expected results

| Target | px ROC-AUC (sigma 5) |
|---|---:|
| bottle | 0.9771 |
| cable | 0.9569 |
| capsule | 0.9553 |
| carpet | 0.9259 |
| grid | 0.9126 |
| hazelnut | 0.9688 |
| leather | 0.9842 |
| metal_nut | 0.9580 |
| pill | 0.9193 |
| screw | 0.9231 |
| tile | 0.9007 |
| toothbrush | 0.9477 |
| transistor | 0.9432 |
| wood | 0.8912 |
| zipper | 0.9666 |
| **MVTec mean** | **0.9420** |
| CT scan | 0.9378 |
| X-ray (scissors) | 0.9528 |
| FFHQ faces | 0.9105 |

All MVTec, xray, and faces numbers reproduce exactly (deterministic evaluation);
CT may vary in the 4th decimal.
