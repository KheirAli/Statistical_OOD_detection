# DVXRay DDAD Feature-Extractor Fine-Tuning Handoff

This run fine-tuned only the DDAD feature extractor on clean DVXRay negatives.
No scissors/anomaly test images were used for training.

## Shared Inputs

- Clean train split: `/data2/rohan/datasets/mvtec_ad_dvxray/xray/train/good/`
  - 5,000 `*_OL.png` symlinks to clean DVXRay negative samples.
- Scissors eval split: `/data2/rohan/datasets/mvtec_ad_dvxray/xray/test/scissors/`
  - 32 accepted scissors images.
- Scissors masks: `/data2/rohan/datasets/mvtec_ad_dvxray/xray/ground_truth/scissors/`
- DDAD DVXRay U-Net checkpoint used here:
  `/data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000`

## Code To Push

- `tools/ddad_da_finetune.py`
  - Trains `feat0.pth ... featN.pth` from clean `train/good`.
- `tools/ddad_eval_fe_checkpoints.py`
  - Aggregate checkpoint eval over selected test subfolders.
- `tools/ddad_eval_fe_per_image.py`
  - Per-image pixel AUROC/AP for failure-case analysis.
- `DDAD/dataset.py`
  - Test loader now falls back to `test/*/*.png` when no legacy `combined` or
    `random` folder exists. DVXRay needs this.
- `DDAD/config_dvxray.yaml`
  - DVXRay MVTec-style dataset/config paths.
- `.gitignore`
  - Ignores generated DVXRay experiment outputs.

Do not push the generated `experiments/dvxray_*` folders; they include large
checkpoint and reconstruction-cache artifacts.

## Commands Used

Smoke one-batch training:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n ood \
  python tools/ddad_da_finetune.py \
    --config DDAD/config_dvxray.yaml \
    --feature_extractor resnet101 \
    --unet_ckpt /data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000 \
    --out_dir experiments/dvxray_resnet101_ddad_finetune/smoke_one_batch \
    --da_epochs 1 \
    --da_batch_size 4 \
    --num_workers 0 \
    --max_batches 1 \
    --log_every 1
```

Full fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n ood \
  python tools/ddad_da_finetune.py \
    --config DDAD/config_dvxray.yaml \
    --feature_extractor resnet101 \
    --unet_ckpt /data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000 \
    --out_dir experiments/dvxray_resnet101_ddad_finetune/e2000_full \
    --da_epochs 4 \
    --da_batch_size 16 \
    --num_workers 2 \
    --log_every 25
```

Scissors-only aggregate eval:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n ood \
  python tools/ddad_eval_fe_checkpoints.py \
    --config DDAD/config_dvxray.yaml \
    --feature_extractor resnet101 \
    --unet_ckpt /data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000 \
    --feat_dir experiments/dvxray_resnet101_ddad_finetune/e2000_full \
    --out_dir experiments/dvxray_resnet101_ddad_finetune/e2000_eval_scissors32 \
    --test_subdirs scissors \
    --num_workers 2 \
    --no_pro
```

Per-image scissors eval:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n ood \
  python tools/ddad_eval_fe_per_image.py \
    --config DDAD/config_dvxray.yaml \
    --feature_extractor resnet101 \
    --unet_ckpt /data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000 \
    --feat_dir experiments/dvxray_resnet101_ddad_finetune/e2000_full \
    --cache_path experiments/dvxray_resnet101_ddad_finetune/e2000_eval_scissors32/recon_cache_scissors.pt \
    --out_csv experiments/dvxray_resnet101_ddad_finetune/e2000_eval_scissors32/per_image_metrics.csv \
    --test_subdirs scissors \
    --num_workers 0
```

## Results

Training loss:

| checkpoint | epoch loss |
| --- | ---: |
| feat1 | 0.1355 |
| feat2 | 0.0608 |
| feat3 | 0.0477 |
| feat4 | 0.0425 |

32-image scissors-only eval:

| checkpoint | pixel AUROC | pixel AP |
| --- | ---: | ---: |
| feat0 | 0.7397 | 0.0086 |
| feat1 | 0.8181 | 0.0128 |
| feat2 | 0.8532 | 0.0185 |
| feat3 | 0.8548 | 0.0178 |
| feat4 | 0.8555 | 0.0183 |

Per-image mean pixel AUROC:

| checkpoint | mean | median |
| --- | ---: | ---: |
| feat0 | 0.7227 | 0.7497 |
| feat1 | 0.8306 | 0.8504 |
| feat2 | 0.8626 | 0.9080 |
| feat3 | 0.8589 | 0.9096 |
| feat4 | 0.8615 | 0.9121 |

Known hard cases:

| image | feat0 | best after FT | note |
| --- | ---: | ---: | --- |
| P04649 | 0.6339 | 0.7376 at feat4 | improved |
| P04666 | 0.6067 | 0.8356 at feat3 | improved substantially |
| P04803 | 0.4251 | 0.4726 at feat1 | still a failure case |

Recommendation: use `feat2.pth` for the next stage if we care about robust
per-image behavior and pixel AP/F1; use `feat4.pth` only if selecting strictly
by aggregate pixel AUROC.

## Shared Artifacts

The final checkpoints and small metric CSV/JSON files should be mirrored to:

`/data2/rohan/ckpts/DDAD/DvXray/feature_extractors/resnet101_e2000_clean5k/`

That directory intentionally excludes large reconstruction caches.
