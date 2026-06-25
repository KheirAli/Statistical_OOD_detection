# Last-layer fine-tuning

**Freeze the entire backbone and train only `layer3`'s last bottleneck block**
(~1.1 M params, 2.5 %). This is the **lightweight** adaptation: it adjusts only
the deepest features the detector actually consumes, while leaving the general
ImageNet structure intact. On easy categories it is the **safest** way to get a
small gain without the over-adaptation that sinks Full FT.

> Read [`../README.md`](../README.md) first for the shared DDAD loss, AE recipe,
> and evaluation. This file only covers what is specific to last-layer training.

---

## What it trains

The detector embeds `[layer1, layer2, layer3]` features, so the *last* trainable
stage that influences them is **`layer3`**. We train only its **final block**
(`layer3[-1]`), freezing everything else:

```python
# ddad_da_finetune.py, --train_scope last_layer
base = fe.module
for p in base.parameters():        p.requires_grad = False   # freeze all
for p in base.layer3[-1].parameters(): p.requires_grad = True   # train last block
# -> trainable 1,117,184 / 44,549,160  (2.508 %)
```

Same DDAD domain-adaptation loss as Full FT — only the trainable set shrinks.
Saves `feat0..featN` plain ResNet-101 state dicts (the frozen weights are carried
through unchanged), and `da_log.json` records `train_scope = last_layer(layer3[-1])`.

> Want a bigger "last layer"? Editing `base.layer3[-1]` → `base.layer3` trains the
> whole `layer3` stage (~58 % of params) — closer to Full FT, not recommended.

---

## Run it

### 1. Fine-tune (last block only)
```bash
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n ddad_env python \
  experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/scripts/ddad_da_finetune.py \
    --category   xray \
    --unet_ckpt  /data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000 \
    --out_dir    experiments/xray_finetune_mine/last_layer/checkpoints \
    --da_epochs  4 --seed 0 \
    --train_scope last_layer
```

### 2. Train the AE head on the last-layer backbone
```bash
DDAD_FE_WEIGHTS=experiments/xray_finetune_mine/last_layer/checkpoints/feat4.pth \
python all_categories_embedder/train_pixel_ae_interp.py \
    --weights   experiments/xray_finetune_mine/last_layer/checkpoints/feat4.pth \
    --data_dir  /data/akheirandish3/mvtec_ad/xray/train/good \
    --save_model models/pixel_ae_last_layer/xray.pth \
    --latent_dim 3 --num_epochs 30 --seed 0 --n_restarts 3 \
    --ae_loss cos --select_loss mse_var --max_images 500
```

### 3. Evaluate
```bash
DDAD_FE_WEIGHTS=experiments/xray_finetune_mine/last_layer/checkpoints/feat4.pth \
python all_categories_embedder/evaluate_ddad_fe.py \
    --config configs/experiment_ddad_native_xray_real.yaml \
    --category xray --subcategory scissors \
    --results_root ./results_patches_ddad_native \
    --gt_root /data/akheirandish3/mvtec_ad/xray/ground_truth \
    --skip_sampling --autoencoder_path models/pixel_ae_last_layer/xray.pth \
    --n_pca 3 --bins_pca 32 --bins_rgb 32 \
    --superpixel_target_size 10 --smooth_sigma 1 --max_reconstructions 40 \
    --results_suffix _1 --output_dir results_eval_xray_mine/last_layer
```

---

## Expected behaviour / results

- **X-ray/scissors: 0.953 px-AUC — the best of all four scopes**, edging out
  pretrained (0.947), full FT (0.943) and LoRA (0.932). The minimal capacity is
  exactly enough to align the deepest features to the X-ray domain without
  forgetting.
- On MVTec, last-layer stays close to pretrained on easy categories (no
  collapse like Full FT) and gives small gains where adaptation helps.
- Best metric ↔ best SNR agree (X-ray SNR 2.61, highest of the four).

**Takeaway:** the recommended default when you want *some* adaptation. Minimal
trainable capacity = adaptation gain without catastrophic forgetting. Pair it
with an early checkpoint (`feat1..feat4`).
