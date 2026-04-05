# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Statistical Out-of-Distribution (OOD) detection using Diffusion Posterior Sampling (DPS). The approach reconstructs images through inpainting with diffusion models, then uses reconstruction error (delta maps) analyzed via PCA and superpixel aggregation to detect anomalies.

**This repo is designed to be overlaid onto the [Diffusion Posterior Sampling](https://github.com/DPS2022/diffusion-posterior-sampling) repository.** It depends on `guided_diffusion` and `data.dataloader` modules from that base repo at `../dps/`.

## Environment

Use the `ood` conda environment: `conda activate ood` (Python 3.10, torch, torchvision, scikit-image, scipy, matplotlib, pyyaml, pillow).

DPS must be on PYTHONPATH: `export PYTHONPATH=/home/rohan/ood/dps:$PYTHONPATH`

## Running Evaluations

```bash
# Evaluate with existing DPS results (most common)
python evaluate.py --config configs/experiment.yaml --skip_sampling

# Evaluate with baselines for comparison
python evaluate.py --config configs/experiment.yaml --skip_sampling --baselines

# Headless sweep (no plots, just metrics JSON)
python evaluate.py --config configs/experiment.yaml --skip_sampling --no_plots

# Override sample name for quick iteration
python evaluate.py --config configs/experiment.yaml --skip_sampling --sample_name samples_005

# Override experiment parameters from CLI
python evaluate.py --config configs/experiment.yaml --skip_sampling --no_plots \
    --backbone resnet50 --n_pca 5 --bins_rgb 64 --bins_pca 16

# Multi-image sweep (runs each sample, averages AUC)
python evaluate.py --config configs/experiment.yaml --skip_sampling --no_plots \
    --backbone resnet18 --n_pca 5 --bins_rgb 64 --bins_pca 16 \
    --sample_names samples_000 samples_001 samples_002

# DPS sampling only (no evaluation)
python evaluate.py --config configs/experiment.yaml --sampling_only \
    --sample_names samples_000 samples_001 samples_002

# Full cable experiment (sampling + all eval sweeps)
screen -S cable bash run_cable_experiments.sh
```

Results are saved to `results_eval/{sample_name}/results.json`.
Sweep results are saved to `results_eval/sweep_{backbone}_pca{n}_{bins}.json`.

To test a new algorithm: edit `ood/scoring.py`, then re-run eval.
To test new embeddings: edit `ood/embeddings.py`, then re-run eval.
To change sampling params (noise sigma, sampler, etc.): edit `configs/experiment.yaml`.

## Eval Harness Architecture (`ood/` package)

```
ood/
  sampler.py      # Stage 0: DPS sampling orchestration (wraps sample_batch.py)
  data.py         # Stage 1: Load reconstructions, labels, masks
  superpixels.py  # Stage 2: Recursive color-homogeneous refinement
  embeddings.py   # Stage 3: ResNetPixelEmbedder + PCA projection
  scoring.py      # Stage 4: Delta map (typical set with RGB+PCA PMFs) — algorithmic core
  metrics.py      # Stage 5: ROC/PR/AUC computation
  visualize.py    # Stage 6: Plotting
  baselines.py    # Stage 7: SimpleNet / DDAD comparison
```

Config: `configs/experiment.yaml` — single YAML controlling all pipeline parameters.

### CLI Override Flags

| Flag | Config key | Example |
|------|-----------|---------|
| `--backbone` | `embeddings.backbone` | `resnet18`, `resnet50`, `resnet101`, `resnet152` |
| `--n_pca` | `pca.n_components` | `5`, `6`, `7` |
| `--bins_rgb` | `scoring.bins_rgb` | `32`, `64` |
| `--bins_pca` | `scoring.bins_pca` | `8`, `16` |
| `--sample_names` | (multi-sample sweep) | `samples_000 samples_001 ...` |
| `--sampling_only` | (run DPS only) | flag |

## Legacy Commands

### Superpixel generation
```bash
python3 super_pixel_generation.py --input_image=<path_to_image> --n_segments=150
```
Outputs: `{output_dir}/mask.png`, `{output_dir}/superpixels.png`

### Run DPS sampling (single image)
```bash
CUDA_VISIBLE_DEVICES=<gpu> python3 sample_batch.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/inpainting_config.yaml \
    --save_dir=./results_patches/<name> \
    --data_root=<image_directory> \
    --seed 0 \
    --box_coords <top> <bottom> <left> <right> \
    --mask_prob 0.9 \
    --mask_path data/mask.png \
    --num_measurements 4
```
Note: `sample_batch.py` line 159 has a `break` that limits processing to 1 image per run.

### Batch processing
```bash
bash run_with_mask.sh
```
Parallelizes patch-based sampling across multiple GPUs. Edit the `patches`, `SAMPLE_FILES`, and directory variables at the top before running.

## Architecture

### Pipeline
1. **Superpixel generation** (`super_pixel_generation.py`) — SLIC segmentation to create region masks
2. **DPS sampling** (`sample_batch.py`) — inpainting-based reconstruction using diffusion models, produces `input/`, `recon/`, `label/` images
3. **Analysis** (`evaluate.py` or `New_dataset_clean.ipynb`) — loads reconstructions, computes delta maps, runs PCA embedding, aggregates per-superpixel scores, evaluates AUROC/TPR/FPR

### Configuration (all in `configs/`)
- `model_config.yaml` — UNet architecture (256×256 images, 96 channels) and **model checkpoint path**
- `diffusion_config.yaml` — DDPM sampler settings (1000 steps, linear noise schedule)
- `inpainting_config.yaml` — task config: conditioning method (posterior sampling), dataset root, mask type (`refined_box`), noise sigma
- `experiment.yaml` — unified config controlling the full eval pipeline
- Other task configs exist (deblur, super-resolution, phase retrieval) but inpainting is the primary one

### Data (all in `/data/akheirandish3/`)
- mvtec_ad, mvtec_ad_mini_combined, mvtec_cable_256, SimpleNet, and PaDIS_results contain datasets
- Cable test images: `/data/akheirandish3/mvtec_ad/cable/test/combined/` (11 images, 000-010)
- Cable GT masks: `/data/akheirandish3/mvtec_ad/cable/ground_truth/combined/`
- guided-diffusion and ckpts contain model information and ckpts from the trained diffusion models.

### Utilities (`util/`)
- `img_utils.py` — mask generation (`mask_generator` class: box, random, refined_box types), image normalization, FFT ops
- `tools.py` — YAML loading, image I/O, patch extraction, tensor-numpy conversions
- `resizer.py` — image resizing with cubic/lanczos/box interpolation
- `fastmri_utils.py` — FFT-shift operations for complex tensors
- `compute_metric.py` — PSNR and LPIPS evaluation

### Notebooks
- `New_dataset_clean.ipynb` — **primary notebook** (RGB images, ready to use)
- `New_dataset.ipynb` — grayscale variant (needs verification)

## Important Details

- All images are 256×256. The patch grid divides this into 64×64 patches.
- `mask_type: refined_box` uses ring-based distance transforms with spatial probability maps; `mask_prob` controls emphasis.
- Model checkpoints are at `/data/akheirandish3/ckpts/guided_diffusion/` (CT_scan, mvtec_cable_256, ffhq).
- Dataset paths reference `/data/akheirandish3/` and `/data2/akheirandish3/` — these are cluster-specific.
- The `guided_diffusion` package (UNet, DDPM sampler, conditioning methods, operators) must be importable for `sample_batch.py` to work.
- ResNet backbone options: resnet18, resnet50, resnet101, resnet152 (all with ImageNet pretrained weights).

## Bug Fixes Applied (2026-04-04)

The following bugs were found during code review and fixed:

1. **`ood/scoring.py`** — Boolean 2D mask indexing on 4D array (`images_recon_all[:, sp_mask, :]`) crashed at runtime. Fixed to use `np.where(sp_mask)` for explicit row/col indexing.
2. **`ood/metrics.py`** — `manual_average_precision` had a sign error (`-np.sum(...)`) making AP always negative. Removed the erroneous negation.
3. **`ood/scoring.py`** — Double-epsilon in PMF computation: eps was added to histograms before normalization AND again in entropy/NLL functions, causing inconsistent H vs NLL values. Fixed: eps added once after normalization; entropy/NLL functions operate directly on the valid PMF.
4. **`ood/data.py`** — Superpixel mask loaded as uint8 channel 0, truncating IDs > 255 after recursive subdivision. Fixed to attempt 16-bit read via PIL `convert("I")`.
5. **`ood/superpixels.py`** — Residual pixels after SLIC splitting kept stale label ID, confusing the parent_map. Fixed by assigning residual pixels a fresh ID.
6. **`ood/embeddings.py`** — ResNet-152 weight variant (`IMAGENET1K_V2`) doesn't exist in all torchvision versions. Fixed with explicit per-model weight variant registry.
7. **`dps/guided_diffusion/gaussian_diffusion.py`** — `if t != 0` fails with batch size > 1 (num_measurements > 1). Fixed to `if t[0] != 0` in both DDPM and DDIM samplers.
8. **`super_pixel_generation.py`** — Script overwrote input image with resized version via `io.imsave(args.input_image, ...)`. Fixed to resize in memory only when dimensions don't match 256×256.
