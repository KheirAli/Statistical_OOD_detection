# Feature-extractor fine-tuning methods for the typical-set OOD detector

This directory documents the **four ways we adapt the ResNet-101 feature
extractor** that feeds our statistical OOD / anomaly detector, plus the
**autoencoder (AE) head** that every variant shares. Each method has its own
sub-directory with a detailed, self-contained README:

| method | sub-dir | what is trained | trainable params (ResNet-101) |
|--------|---------|-----------------|-------------------------------|
| **Full FT**   | [`full_ft/`](full_ft/README.md)     | all backbone weights                                  | 44.5 M (100 %) |
| **Last layer**| [`last_layer/`](last_layer/README.md)| only `layer3`'s last bottleneck block                 | 1.1 M (2.5 %)  |
| **LoRA**      | [`lora/`](lora/README.md)           | low-rank adapters on `layer2`/`layer3` conv2          | 0.5 M (1.1 %)  |
| **Interp**    | [`interp/`](interp/README.md)       | nothing — linear blend of pretrained ↔ full-FT weights| 0 (post-hoc)   |

Plus the **pretrained** baseline = ImageNet ResNet-101, no adaptation (`feat0`).

---

## The big picture

```
                    ┌─────────────────────────── per method ───────────────────────────┐
ImageNet ResNet-101 │  (A) DDAD domain-adaptation fine-tune  ──►  adapted backbone       │
   (feat0 anchor)   │      full / last_layer / lora / interp                            │
                    └───────────────────────────────────────────────────────────────────┘
                                              │
                        (B) train a per-(category,method) pixel-AE on the
                            adapted backbone's features (train/good images)
                                              │
                        (C) evaluate through OUR detector on DPS/DDAD recons
                            (typical-set / superpixel / AE→3, smooth σ=5)
```

The detector itself is **unchanged** across methods; we only change the backbone
(step A) and retrain the AE head on top of it (step B). That makes the four
methods directly comparable.

---

## (A) The fine-tuning objective (DDAD domain adaptation)

All three *trained* variants (full / last_layer / lora) use the **same DDAD
domain-adaptation loss** — only the *set of trainable parameters* differs.

For each batch of clean (anomaly-free) training images, the **frozen diffusion
U-Net** reconstructs them, and the feature extractor `fe` is trained so that
**clean and reconstructed images map to the same features**, while staying
anchored to the original pretrained features `frozen`:

```
loss1 = 1 - cos( fe(recon),  fe(target) )                 # domain adaptation
loss2 = (1 - cos( fe(target), frozen(target) )) * DLlambda # distillation anchor (clean)
loss3 = (1 - cos( fe(recon),  frozen(recon)  )) * DLlambda # distillation anchor (recon)
loss  = loss1 + loss2 + loss3
```
(`DDAD/feature_extractor.py::loss_fucntion`, `DLlambda = 0.1` for MVTec.)

**Trainer:** `experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/scripts/ddad_da_finetune.py`
A checkpoint is saved **every epoch** (`feat0` = ImageNet anchor, `feat1..featN`
= after each DA epoch), so a downstream sweep can pick the best-early checkpoint.

> ⚠️ **Over-adaptation is real.** DDAD's own config uses `DA_epochs = 4`. The
> deeper you push this invariance objective (more epochs, more trainable
> params), the more the backbone forgets the general ImageNet structure the
> typical-set detector relies on. Empirically: **full FT ≤ pretrained on
> already-easy categories** (carpet full-FT even *collapses* to ~0.44 px-AUC),
> while **last_layer / LoRA / interp stay close to pretrained**. Prefer the
> minimal-capacity variants and an early checkpoint.

---

## (B) The AE head — `all_categories_embedder/train_pixel_ae_interp.py`

A per-pixel 1×1-conv autoencoder maps the **~1792-dim** ResNet feature vector at
each pixel to a **3-dim latent** and back (`evaluate.py::PixelAutoEncoder`,
1792→512→256→64→**3**(Tanh)→64→256→512→1792, decoder output L2-normalized). It is
a learned, nonlinear replacement for a linear PCA→3 reduction, trained only on
nominal `train/good` features.

### ✅ Corrected AE recipe (this is the fix)

The earlier AE trainer was **incorrect** in two ways; both are now fixed:

1. **Non-deterministic** (unseeded `random_split` + weight init). A 3-dim
   bottleneck has many equally-good reconstructions with different latent
   orientations, and the detector histograms those axes → AUC swung **0.70–0.83
   across seeds** on the same backbone. **Fix:** `--seed` (fixed split + init +
   cuDNN-deterministic) and `--n_restarts N` (best-of-N by validation loss).
2. **Wrong loss.** A later edit used the variance-penalty loss for *training*,
   which **hurt** (e.g. pill pretrained 0.92 → 0.865). The correct recipe is the
   `train_embedder.py` one: **cosine reconstruction loss for training**, and a
   **MSE + latent-variance penalty for checkpoint selection** (the variance
   penalty `0.1·ReLU(0.33 − var(z))` stops the 3-dim latent from collapsing).
   **Fix:** decoupled `--ae_loss` (training, default `cos`) and `--select_loss`
   (selection, default `mse_var`).

```bash
# canonical AE training (what every method README uses)
DDAD_FE_WEIGHTS=<backbone.pth> python all_categories_embedder/train_pixel_ae_interp.py \
    --weights   <backbone.pth> \
    --data_dir  /data/akheirandish3/mvtec_ad/<category>/train/good \
    --save_model models/<...>/<category>.pth \
    --latent_dim 3 --num_epochs 30 \
    --seed 0 --n_restarts 3 \
    --ae_loss cos --select_loss mse_var      # ← corrected recipe
    # --max_images 500   (optional: cap for very large train sets, e.g. xray 5k)
```

**Every (category, method) pair gets its OWN AE** trained on that backbone's
features — never reuse one fixed AE across backbones (the features differ).

---

## (C) Evaluation — `all_categories_embedder/evaluate_ddad_fe.py`

`evaluate_ddad_fe.py` monkeypatches the embedder so it loads the adapted
backbone from `DDAD_FE_WEIGHTS`, then runs the standard `evaluate.py` detector.
The pretrained baseline uses plain `evaluate.py` (ImageNet backbone).

```bash
DDAD_FE_WEIGHTS=<backbone.pth> python all_categories_embedder/evaluate_ddad_fe.py \
    --config configs/experiment_ddad_native.yaml \
    --category <cat> --all_subcategories \
    --results_root ./results_patches_ddad_native \
    --gt_root /data/akheirandish3/mvtec_ad/<cat>/ground_truth \
    --skip_sampling --autoencoder_path models/<...>/<cat>.pth \
    --n_pca 3 --bins_pca 32 --bins_rgb 32 \
    --superpixel_target_size 30 --smooth_sigma 1 --max_reconstructions 50 \
    --results_suffix _3 --output_dir results_eval/<...>/<cat>
```

> 📌 **Reconstruction set matters.** Use `--results_suffix _3` for MVTec (the
> `run_eval_auto_encoder.sh` config) — it is markedly better than `_1` for
> several categories (e.g. pill pretrained 0.81 on `_1` vs **0.92** on `_3`).

Metrics per (category, method) are the mean over **defective** subcategories
(good/combined excluded) at smoothing σ=5: `px_roc_auc`, `sp_roc_auc`,
`px_ap`, `sp_ap`, `snr`, `psnr`.

---

## End-to-end drivers

- `all_categories_embedder/matched_unet_pipeline.sh CAT UNET_CKPT` — fine-tune
  one MVTec category against its own diffusion U-Net (full + lora + interp),
  train the AEs, evaluate. (Add `--train_scope last_layer` support via the
  underlying trainer.)
- `all_categories_embedder/xray_finetune_mine.sh` — the X-ray study: runs **all
  four scopes** (pretrained / last_layer / full / lora) against rohan's
  `xray/2000` U-Net, with mse_cos AEs, on one GPU set. A worked example.

See each sub-directory README for the exact, copy-pasteable commands and the
expected behaviour of that method.
