# Full fine-tuning (Full FT)

**All 44.5 M weights of the ResNet-101 feature extractor are trained** with the
DDAD domain-adaptation loss. This is the highest-capacity adaptation — and, on
this typical-set detector, usually the **worst** of the four (it over-adapts and
forgets the general ImageNet features the detector relies on).

> Read [`../README.md`](../README.md) first for the shared DDAD loss, AE recipe,
> and evaluation. This file only covers what is specific to Full FT.

---

## What it trains

```python
# ddad_da_finetune.py, --lora_rank 0 (default), --train_scope full (default)
trainable = list(fe.parameters())     # every conv/bn/fc weight is updated
# n_trainable = n_total = 44,549,160   (100 %)
```

Every epoch saves a **plain ResNet-101 state dict**:
```
<out_dir>/feat0.pth        # ImageNet anchor (frozen reference, alpha=0)
<out_dir>/feat1.pth ... featN.pth   # after each DA epoch
<out_dir>/da_log.json      # per-epoch loss, timing, param counts
```
`feat0` is the **pretrained baseline** (no adaptation). The "Full FT" backbone is
a later epoch — usually an **early one** (see "Choosing a checkpoint").

---

## Run it

### 1. Fine-tune
```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ddad_env python \
  experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/scripts/ddad_da_finetune.py \
    --category   pill \
    --unet_ckpt  /data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/combined/1000 \
    --out_dir    experiments/<exp>/pill/checkpoints \
    --da_epochs  4 \
    --da_batch_size 16 --lr 1e-4 --seed 0
    # --train_scope full   (default; do NOT pass --lora_rank)
```

Hyperparameters (all have sensible defaults in the trainer):

| flag | default | notes |
|------|---------|-------|
| `--da_epochs` | 8 | **use 4** (DDAD's recommended value); more = more over-adaptation |
| `--da_batch_size` | 16 | half is `target`, half is `input` (DDAD recon scheme) |
| `--lr` | 1e-4 | AdamW |
| `--w_da` | from config (3.0) | diffusion conditioning strength during reconstruction |
| `--seed` | 42 | RNG seed |

### 2. Train the AE head on the Full-FT backbone
```bash
DDAD_FE_WEIGHTS=experiments/<exp>/pill/checkpoints/feat4.pth \
python all_categories_embedder/train_pixel_ae_interp.py \
    --weights   experiments/<exp>/pill/checkpoints/feat4.pth \
    --data_dir  /data/akheirandish3/mvtec_ad/pill/train/good \
    --save_model models/pixel_ae_fullft/pill.pth \
    --latent_dim 3 --num_epochs 30 --seed 0 --n_restarts 3 \
    --ae_loss cos --select_loss mse_var
```

### 3. Evaluate
```bash
DDAD_FE_WEIGHTS=experiments/<exp>/pill/checkpoints/feat4.pth \
python all_categories_embedder/evaluate_ddad_fe.py \
    --config configs/experiment_ddad_native.yaml --category pill --all_subcategories \
    --results_root ./results_patches_ddad_native \
    --gt_root /data/akheirandish3/mvtec_ad/pill/ground_truth \
    --skip_sampling --autoencoder_path models/pixel_ae_fullft/pill.pth \
    --n_pca 3 --bins_pca 32 --bins_rgb 32 \
    --superpixel_target_size 30 --smooth_sigma 1 --max_reconstructions 50 \
    --results_suffix _3 --output_dir results_eval_fullft/pill
```

---

## Choosing a checkpoint

The DA loss keeps falling every epoch (the backbone keeps collapsing
clean≈recon), so the **last** epoch is the **most over-adapted**. Pick the
checkpoint by the **downstream detector AUC**, not by training loss — in
practice an **early epoch (1–4)** is best, and for easy categories even `feat0`
(pretrained) wins.

---

## Expected behaviour / results

- On **hard / large-domain-gap** categories (grid, hazelnut, screw), Full FT can
  *help* (e.g. screw px-AUC 0.92 → 0.98) — adaptation gain outweighs forgetting.
- On **already-easy** categories it **hurts**: cable, leather, metal_nut,
  transistor, wood, zipper all drop vs pretrained; **carpet full-FT collapses to
  ~0.44 px-AUC** (below chance) because the backbone becomes so invariant to
  texture deviations that defects look *more* typical than normal regions.
- **X-ray:** 0.943 px-AUC vs pretrained 0.947 — slightly below pretrained.
- **Average over 15 MVTec categories: full FT 0.891 px-AUC vs pretrained 0.942.**

**Takeaway:** Full FT is the most aggressive and the least safe. Use it only when
a single category clearly benefits, and always compare against `feat0`.
