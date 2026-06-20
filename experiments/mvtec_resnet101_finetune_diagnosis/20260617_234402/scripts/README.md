# ResNet-101 DDAD Domain-Adaptation Fine-Tuning

Per-category fine-tuning of a ResNet-101 feature extractor using the DDAD
domain-adaptation loss, producing the four backbone variants that the OOD
detector is later evaluated on:

| Variant | What it is | Produced by |
|---------|-----------|-------------|
| `pretrained` | ImageNet ResNet-101, no adaptation (`feat0.pth`) | saved as the alpha=0 anchor |
| `full FT`    | all weights adapted (`feat1..featN.pth`)        | `ddad_da_finetune.py` (`--lora_rank 0`) |
| `LoRA`       | low-rank adapters on `layer2`/`layer3` conv2    | `ddad_da_finetune.py` (`--lora_rank r`) |
| `interp`     | WiSE-FT blend `(1-α)·feat0 + α·featN`            | `interpolate_resnet101_weights.py` |

## What the fine-tuning optimizes

For each batch of clean (anomaly-free) training images, the frozen diffusion
U-Net reconstructs them, and the feature extractor is trained so that **clean
and reconstructed images map to the same features**, while staying anchored to
the original pretrained features. The DDAD loss
(`DDAD/feature_extractor.py::loss_fucntion`) has three cosine-similarity terms:

```
loss1 = 1 - cos( fe(recon),  fe(target) )                # domain adaptation
loss2 = 1 - cos( fe(target), frozen(target) ) * DLlambda # distillation anchor (clean)
loss3 = 1 - cos( fe(recon),  frozen(recon)  ) * DLlambda # distillation anchor (recon)
```

A checkpoint is saved **every epoch** (`feat{epoch}.pth`) so a downstream sweep
can pick the best-early checkpoint per category. LoRA checkpoints are folded
back into a plain ResNet-101 state dict on save, so the eval harness loads them
identically to a fully fine-tuned model.

## Files

| File | Role |
|------|------|
| `ddad_da_finetune.py` | The trainer. Full FT **and** LoRA (`--lora_rank`). Saves `feat0..featN` + `da_log.json`. |
| `ddad_diag.py` | Helpers: `build_resnet101`, `build_unet`, `inject_lora`, `merge_lora_to_plain_resnet`, `interpolate_state_dicts`. Also makes the vendored `DDAD/` package importable. |
| `interpolate_resnet101_weights.py` | WiSE-FT linear interpolation between two checkpoints. |
| `evaluate_all_checkpoints.py` | Cached-reconstruction eval sweep over feat0..featN + interpolation + LoRA; writes metric CSVs. |
| `run_mvtec_resnet101_finetune_all_categories.sh` | End-to-end launcher: UNet symlink → full FT → LoRA → eval sweep → qualitative, per category. |
| `redo_lora.sh` | Re-run only the LoRA stage for selected categories. |
| `plot_results.py`, `make_report.py`, `make_qualitative.py`, `make_global_qualitative.py` | Reporting / figures. |

## Requirements

- **Conda env `ddad_env`** (DDAD dependencies: torch, torchvision, omegaconf, …).
- **Vendored DDAD package** at repo-root `DDAD/` (tracked). `ddad_diag.py` adds it
  to `sys.path`; it provides `reconstruction`, `feature_extractor`, `dataset`,
  `unet`, `resnet`, `anomaly_map`, `metrics`.
- **Diffusion U-Net checkpoints** under
  `/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/` (per-category where
  available, e.g. `cable/3000`; otherwise the shared `combined/1000`).
- **MVTec-AD data** at `/data/akheirandish3/mvtec_ad/<category>/`.

> Note: the scripts assume this exact directory depth — `ddad_diag.py` resolves
> the repo root as `__file__/../../../..`, and the launcher hardcodes
> `experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402`. Keep the
> files in this location.

## How to run

### Full pipeline, all 15 categories
```bash
cd /data/akherandish3/Statistical_OOD_detection
GPU=5 DA_EPOCHS=8 \
  bash experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/scripts/run_mvtec_resnet101_finetune_all_categories.sh
```

Useful env knobs (see header of the launcher):
- `CATS="bottle grid zipper"` — subset (default: all 15)
- `DA_EPOCHS=8` — full-FT epochs (LoRA training-time fractions derived from this)
- `DO_LORA=1 DO_INTERP=1 DO_QUAL=1` — toggle stages
- `NO_PRO=1` — skip the slow PRO metric during eval
- `LIMIT=N` — cap #test images (debug)

LoRA configs run by default (`name:rank:epochs`): `r4_50:4:4`, `r8_50:8:4`, `r8_100:8:8`.

### Single category, full fine-tune only
```bash
CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n ddad_env \
  python experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/scripts/ddad_da_finetune.py \
    --category cable \
    --unet_ckpt /data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/combined/1000 \
    --out_dir   experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/cable/checkpoints \
    --da_epochs 8
```

### Single category, LoRA (rank 8)
Add `--lora_rank 8 [--lora_alpha 1.0]` and point `--out_dir` at a `lora/<name>` subdir.

### Interpolated (WiSE-FT) backbone
```bash
python .../scripts/interpolate_resnet101_weights.py \
  --a .../cable/checkpoints/feat0.pth \
  --b .../cable/checkpoints/feat8.pth \
  --alpha 0.3 \
  --out  .../cable/checkpoints/interp_a0.3.pth
```

## Outputs (per category, under `<exp>/<category>/`)
```
checkpoints/feat0.pth ... feat8.pth      # full-FT, one per epoch (feat0 = pretrained anchor)
checkpoints/da_log.json                  # per-epoch loss / timing / param counts
checkpoints/lora/<name>/feat*.pth        # LoRA checkpoints (merged to plain resnet101)
metrics/, figures/, qualitative/         # eval sweep results + plots
```

## Downstream

These backbones feed the OOD-detector evaluation in
`all_categories_embedder/` (`run_eval_ae_ours.sh` / `evaluate_ddad_fe.py`),
which selects a backbone per `KIND` (pretrained / fullft / lora / interp) and
trains a 3-dim pixel-AE head on top before scoring. The
`Fine-tuning methods judged by OUR detector` figure is produced by
`all_categories_embedder/plot_methods_ours.py`.
